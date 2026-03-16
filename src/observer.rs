use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::config::MnemonicConfig;
use crate::embeddings::Embedder;
use crate::models::*;
use crate::store::MemoryStore;

/// Processes PostToolUse hook events automatically.
/// Uses Gemini Flash to extract facts from tool outputs.
/// Captures tool outcomes, detects error→fix pairs, tracks file touches.
pub struct Observer {
    store: Arc<MemoryStore>,
    embedder: Arc<Embedder>,
    session_id: String,
    extract_facts: bool,
    extract_tools: Vec<String>,
    write_tools: Vec<String>,
    snippet_length: usize,
    fact_expiry_days: i64,
}

impl Observer {
    pub fn new(store: Arc<MemoryStore>, embedder: Arc<Embedder>, config: &MnemonicConfig) -> Self {
        Self {
            store,
            embedder,
            session_id: format!("session-{}", &uuid::Uuid::new_v4().to_string()[..8]),
            extract_facts: config.extract_facts,
            extract_tools: config.extract_tools.clone(),
            write_tools: config.write_tools.clone(),
            snippet_length: config.snippet_length,
            fact_expiry_days: config.fact_expiry_days,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Process a single PostToolUse hook event.
    pub async fn process_hook_event(&self, event: &HookEvent) {
        let now = chrono::Utc::now().to_rfc3339();
        let tool_name = &event.tool_name;

        // Determine success/failure from tool_response
        let success = self.detect_success(event);
        let outcome = if success { "ok" } else { "error" };

        let snippet_len = self.snippet_length;
        let snippet = event
            .tool_response
            .as_ref()
            .and_then(|r| serde_json::to_string(r).ok())
            .map(|s| s.chars().take(snippet_len).collect::<String>());

        // 1. Record command outcome
        let cmd = CommandRecord {
            id: format!("cmd_{}", &uuid::Uuid::new_v4().to_string()[..8]),
            tool: tool_name.clone(),
            outcome: outcome.to_string(),
            duration_ms: None,
            result_snippet: snippet.clone(),
            session_id: self.session_id.clone(),
            recorded_at: now.clone(),
            embedding: None,
        };
        if let Err(e) = self.store.insert_command(&cmd) {
            warn!("Failed to record command: {}", e);
        }

        // 2. Track file touches
        if success {
            if let Some(path) = extract_file_path(event) {
                let is_write = self.write_tools.iter().any(|t| t == tool_name)
                    || tool_name.contains("write")
                    || tool_name.contains("edit")
                    || tool_name.contains("patch");
                if let Err(e) = self.store.bump_key_file(&path, is_write) {
                    warn!("Failed to bump key file: {}", e);
                }
            }
        }

        // 3. Error→fix pairing
        if !success {
            let error_msg = self.extract_error_message(event);

            let dbg = DebugRecord {
                id: format!("dbg_{}", &uuid::Uuid::new_v4().to_string()[..8]),
                error_message: error_msg,
                root_cause: String::new(),
                fix: String::new(),
                tool: Some(tool_name.clone()),
                session_id: self.session_id.clone(),
                recorded_at: now.clone(),
                embedding: None,
            };
            if let Err(e) = self.store.insert_debug_record(&dbg) {
                warn!("Failed to record debug: {}", e);
            }
        } else if let Some((error_id, error_msg)) = self.store.last_error() {
            // Success after error → pair as fix
            let fix_desc = format!(
                "Retried {} successfully after error: {}",
                tool_name,
                &error_msg[..error_msg.len().min(100)]
            );

            // Use Gemini to infer root cause
            let root_cause = match self.embedder.infer_root_cause(&error_msg, &fix_desc).await {
                Ok(cause) => cause,
                Err(_) => "Unknown".to_string(),
            };

            if let Err(e) = self.store.pair_error_with_fix(&error_id, &fix_desc) {
                warn!("Failed to pair error with fix: {}", e);
            } else {
                info!("Paired error {} → fix (root cause: {})", error_id, root_cause);
            }
        }

        // 4. Auto-extract facts using Gemini Flash
        // FIXED: Use broad matching — match tool name with or without mcp__ prefix
        if success && self.extract_facts && self.should_extract(tool_name) {
            let input_json = event
                .tool_input
                .as_ref()
                .map(|v| serde_json::to_string(v).unwrap_or_default())
                .unwrap_or_default();

            // Truncate output to avoid huge prompts
            let output_json = snippet.clone().unwrap_or_default();

            // Only extract if there's meaningful content
            if !output_json.is_empty() && output_json.len() > 20 {
                match self.embedder.extract_facts(tool_name, &input_json, &output_json).await {
                    Ok(facts) if !facts.is_empty() => {
                        for extracted in &facts {
                            let fact = MemoryFact {
                                id: format!("fact_{}", &uuid::Uuid::new_v4().to_string()[..8]),
                                content: extracted.content.clone(),
                                category: extracted.category.clone(),
                                confidence: extracted.confidence,
                                created_at: now.clone(),
                                source: format!("auto:{}", tool_name),
                                expires_at: if extracted.category == "environment" {
                                    Some((chrono::Utc::now() + chrono::Duration::days(self.fact_expiry_days)).to_rfc3339())
                                } else {
                                    None
                                },
                                pinned: false,
                                importance: 0.6, // auto-extracted facts start at moderate importance
                                access_count: 0,
                                updated_at: now.clone(),
                                valid_at: None,
                                invalid_at: None,
                                superseded_by: None,
                                ttl: None,
                                session_id: Some(self.session_id.clone()),
                                embedding: None,
                            };

                            if let Err(e) = self.store.insert_fact(&fact) {
                                warn!("Failed to save extracted fact: {}", e);
                            } else {
                                // Embed immediately
                                if let Ok(emb) = self.embedder.embed_one(
                                    &format!("[{}] {}", fact.category, fact.content)
                                ) {
                                    let _ = self.store.update_embedding("facts", &fact.id, &emb);
                                }

                                // Extract entities for the graph
                                if let Ok(entities) = self.embedder.extract_entities(&fact.content).await {
                                    self.link_entities_to_fact(&fact.id, &entities);
                                }
                            }
                        }
                        info!(
                            "Auto-extracted {} facts from {} call",
                            facts.len(),
                            tool_name
                        );
                    }
                    Ok(_) => {} // no facts extracted, that's fine
                    Err(e) => {
                        debug!("Fact extraction failed for {}: {}", tool_name, e);
                    }
                }
            }
        }

        // 5. Embed new un-embedded items periodically
        let (_, cmds, _, _) = self.store.stats();
        if cmds % 10 == 0 {
            if let Err(e) = self.embedder.embed_all(&self.store) {
                warn!("Background embedding failed: {}", e);
            }
        }

        // 6. Clean up TTL-expired facts periodically
        if cmds % 50 == 0 {
            if let Err(e) = self.store.cleanup_ttl_facts() {
                warn!("TTL cleanup failed: {}", e);
            }
        }
    }

    /// Link extracted entities to a fact in the graph
    pub fn link_entities_to_fact(&self, fact_id: &str, entities: &[ExtractedEntity]) {
        for entity in entities {
            match self.store.find_or_create_entity(&entity.name, &entity.entity_type) {
                Ok(entity_id) => {
                    let edge = EntityEdge {
                        entity_id,
                        fact_id: fact_id.to_string(),
                        relation: entity.relation.clone(),
                    };
                    if let Err(e) = self.store.insert_entity_edge(&edge) {
                        warn!("Failed to create entity edge: {}", e);
                    }
                }
                Err(e) => {
                    warn!("Failed to find/create entity '{}': {}", entity.name, e);
                }
            }
        }
    }

    /// Broader tool matching for auto-extraction
    fn should_extract(&self, tool_name: &str) -> bool {
        // Direct match against configured tools
        if self.extract_tools.iter().any(|t| t == tool_name) {
            return true;
        }

        // Strip mcp__ prefix for matching (e.g. mcp__server__vm_exec → vm_exec)
        let short_name = tool_name
            .rsplit("__")
            .next()
            .unwrap_or(tool_name);

        // Match common tool patterns that produce valuable facts
        let valuable_patterns = [
            "bash", "edit", "write", "grep", "glob", "read",
            "vm_exec", "run_script", "run_command", "run_tests",
            "run_javascript", "get_console_logs", "run_ui_test",
            "run_auto_test", "cloud_exec", "gpu_exec",
            "send_terminal_input", "run_scenario",
            "scp", "cmake", "make", "build",
        ];

        // Case-insensitive match on short name or full name
        let lower_name = tool_name.to_lowercase();
        let lower_short = short_name.to_lowercase();

        valuable_patterns.iter().any(|p| {
            lower_name.contains(p) || lower_short == *p
        })
    }

    /// Better success detection — handles various response formats
    fn detect_success(&self, event: &HookEvent) -> bool {
        event
            .tool_response
            .as_ref()
            .map(|r| {
                // Explicit success field
                if let Some(v) = r.get("success") {
                    return v.as_bool().unwrap_or(true);
                }
                // Error field present = failure
                if r.get("error").is_some() {
                    return false;
                }
                // is_error field
                if let Some(v) = r.get("is_error") {
                    return !v.as_bool().unwrap_or(false);
                }
                // Exit code check (for Bash tool)
                if let Some(code) = r.get("exitCode").or(r.get("exit_code")) {
                    return code.as_i64().unwrap_or(0) == 0;
                }
                // Default: assume success
                true
            })
            .unwrap_or(true)
    }

    /// Extract error message from various response formats
    fn extract_error_message(&self, event: &HookEvent) -> String {
        event
            .tool_response
            .as_ref()
            .and_then(|r| {
                // Try common error fields
                r.get("error")
                    .or(r.get("message"))
                    .or(r.get("reason"))
                    .or(r.get("stderr"))
                    .and_then(|v| v.as_str())
            })
            .unwrap_or("Unknown error")
            .chars()
            .take(500)
            .collect()
    }

    /// Process a hook event from raw JSON bytes.
    pub async fn process_raw(&self, data: &[u8]) {
        match serde_json::from_slice::<HookEvent>(data) {
            Ok(event) => {
                info!("Hook event: {} → processing", event.tool_name);
                self.process_hook_event(&event).await;
            }
            Err(e) => {
                debug!("Failed to parse hook event: {}", e);
            }
        }
    }
}

/// Extract file path from tool input (works for Write, Edit, Read, etc.)
fn extract_file_path(event: &HookEvent) -> Option<String> {
    event
        .tool_input
        .as_ref()
        .and_then(|input| {
            input
                .get("file_path")
                .or(input.get("path"))
                .or(input.get("filePath"))
                .or(input.get("file"))
                .and_then(|v| v.as_str())
        })
        .map(|s| s.to_string())
}
