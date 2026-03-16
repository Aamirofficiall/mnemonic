use rmcp::{
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, Implementation, ProtocolVersion,
            ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router, ServerHandler,
    ErrorData as McpError,
};
use std::sync::Arc;
use tracing::{info, warn};

use crate::config::MnemonicConfig;
use crate::embeddings::Embedder;
use crate::models::*;
use crate::observer::Observer;
use crate::search::SearchEngine;
use crate::store::MemoryStore;

#[derive(Clone)]
pub struct MemoryServer {
    store: Arc<MemoryStore>,
    embedder: Arc<Embedder>,
    observer: Arc<Observer>,
    config: Arc<MnemonicConfig>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl MemoryServer {
    pub fn new(store: Arc<MemoryStore>, embedder: Arc<Embedder>, config: Arc<MnemonicConfig>) -> Self {
        let observer = Arc::new(Observer::new(store.clone(), embedder.clone(), &config));
        Self {
            store,
            embedder,
            observer,
            config,
            tool_router: Self::tool_router(),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TOOL 1: search_memory — 3-tier search with combined filters
    // ═══════════════════════════════════════════════════════════════════════

    #[tool(description = "Search project memory. Finds past errors and their fixes, learned facts, command patterns, workflows. Uses multi-stage search: FTS5 keyword + vector similarity + temporal + strength weighting. Supports temporal queries like 'errors last 3 days'.")]
    async fn search_memory(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let limit = params.limit.unwrap_or(self.config.default_limit);
        let categories = params.categories.as_deref();
        let pinned_only = params.pinned_only.unwrap_or(false);
        let since = params.since.as_deref();

        let engine = SearchEngine::new(&self.store, &self.embedder);
        let results = engine.search(
            &params.query, limit, categories, self.config.min_similarity,
            pinned_only, since,
        ).await;

        if results.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No matching memories found.",
            )]));
        }

        let json = serde_json::to_string_pretty(&results).map_err(|e| {
            McpError::internal_error(format!("Failed to serialize results: {}", e), None)
        })?;

        info!(
            "search_memory('{}') → {} results",
            params.query,
            results.len()
        );
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TOOL 2: save_memory — with TTL, supersede, entity extraction
    // ═══════════════════════════════════════════════════════════════════════

    #[tool(description = "Save a fact to project memory. Categories: gotcha, pattern, preference, convention, environment, failure, insight, debug. Use pinned=true for critical knowledge. Use supersedes to replace old facts. Use ttl for ephemeral facts ('session', '1h', '1d', '7d').")]
    async fn save_memory(
        &self,
        Parameters(params): Parameters<SaveParams>,
    ) -> Result<CallToolResult, McpError> {
        let confidence = params.confidence.unwrap_or(0.9);
        let pinned = params.pinned.unwrap_or(false);
        let now = chrono::Utc::now().to_rfc3339();

        // Compute expiry from TTL
        let expires_at = match params.ttl.as_deref() {
            Some("session") => None, // handled by session cleanup
            Some("permanent") => None,
            Some(ttl) => {
                // Parse duration and set expiry
                let duration = match ttl {
                    "1h" => Some(chrono::Duration::hours(1)),
                    "1d" => Some(chrono::Duration::days(1)),
                    "7d" => Some(chrono::Duration::days(7)),
                    "30d" => Some(chrono::Duration::days(30)),
                    _ => None,
                };
                duration.map(|d| (chrono::Utc::now() + d).to_rfc3339())
            }
            None => {
                // Default: environment facts expire
                if params.category == "environment" {
                    Some((chrono::Utc::now() + chrono::Duration::days(self.config.fact_expiry_days)).to_rfc3339())
                } else {
                    None
                }
            }
        };

        let fact = MemoryFact {
            id: format!("fact_{}", &uuid::Uuid::new_v4().to_string()[..8]),
            content: params.content.clone(),
            category: params.category.clone(),
            confidence,
            created_at: now.clone(),
            source: "explicit".to_string(),
            expires_at,
            pinned,
            importance: params.importance.unwrap_or(0.5),
            access_count: 0,
            updated_at: now.clone(),
            valid_at: None,
            invalid_at: None,
            superseded_by: None,
            ttl: params.ttl.clone(),
            session_id: Some(self.observer.session_id().to_string()),
            embedding: None,
        };

        self.store.insert_fact(&fact).map_err(|e| {
            McpError::internal_error(format!("Failed to save: {}", e), None)
        })?;

        // Handle supersede
        if let Some(ref old_id) = params.supersedes {
            match self.store.supersede_fact(old_id, &fact.id) {
                Ok(true) => info!("Superseded fact {} with {}", old_id, fact.id),
                Ok(false) => warn!("Fact {} not found for supersede", old_id),
                Err(e) => warn!("Failed to supersede {}: {}", old_id, e),
            }
        }

        // Embed the new fact immediately
        match self.embedder.embed_one_async(&format!("[{}] {}", params.category, params.content)).await {
            Ok(emb) => {
                let _ = self.store.update_embedding("facts", &fact.id, &emb);
                info!("Embedded fact {} ({} dims)", fact.id, emb.len());
            }
            Err(e) => {
                warn!("Failed to embed fact {}: {}", fact.id, e);
            }
        }

        // Auto-extract entities and build graph
        match self.embedder.extract_entities(&params.content).await {
            Ok(entities) if !entities.is_empty() => {
                self.observer.link_entities_to_fact(&fact.id, &entities);
                info!("Linked {} entities to fact {}", entities.len(), fact.id);
            }
            Ok(_) => {}
            Err(e) => {
                warn!("Entity extraction failed for fact {}: {}", fact.id, e);
            }
        }

        let mut result_msg = format!("Saved: [{}] {}", params.category, params.content);
        if pinned {
            result_msg.push_str(" (pinned)");
        }
        if params.supersedes.is_some() {
            result_msg.push_str(&format!(" (supersedes {})", params.supersedes.as_ref().unwrap()));
        }
        if params.ttl.is_some() {
            result_msg.push_str(&format!(" (ttl: {})", params.ttl.as_ref().unwrap()));
        }
        result_msg.push_str(&format!("\nFact ID: {}", fact.id));

        info!("save_memory: {}", &result_msg[..result_msg.len().min(80)]);
        Ok(CallToolResult::success(vec![Content::text(result_msg)]))
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TOOL 3: get_context — full project memory context
    // ═══════════════════════════════════════════════════════════════════════

    #[tool(description = "Get full project memory context. Returns pinned facts, key files, git handoff, recent errors and fixes, patterns. Respects token budget. Call at session start for continuity.")]
    async fn get_context(
        &self,
        Parameters(params): Parameters<GetContextParams>,
    ) -> Result<CallToolResult, McpError> {
        let context = self.store.load_context();
        let (facts, cmds, debug, wf) = self.store.stats();
        let (entities, edges) = self.store.entity_stats();
        let token_budget = params.token_budget.unwrap_or(self.config.max_context_tokens);

        let mut output = String::new();

        output.push_str(&format!(
            "# Project Memory ({} facts, {} commands, {} debug records, {} workflows)\n",
            facts, cmds, debug, wf
        ));
        if entities > 0 {
            output.push_str(&format!("# Entity Graph: {} entities, {} edges\n", entities, edges));
        }
        output.push('\n');

        if !context.handoff.last_commit.is_empty() || !context.handoff.next_task.is_empty() {
            output.push_str("## Handoff\n");
            if !context.handoff.last_commit.is_empty() {
                output.push_str(&format!("- Last commit: {}\n", context.handoff.last_commit));
            }
            if !context.handoff.uncommitted_files.is_empty() {
                output.push_str(&format!(
                    "- Uncommitted: {}\n",
                    context.handoff.uncommitted_files.join(", ")
                ));
            }
            if !context.handoff.next_task.is_empty() {
                output.push_str(&format!("- Next task: {}\n", context.handoff.next_task));
            }
            if !context.handoff.blocked_on.is_empty() {
                output.push_str(&format!("- Blocked on: {}\n", context.handoff.blocked_on));
            }
            output.push('\n');
        }

        if !context.key_files.is_empty() {
            output.push_str("## Key Files\n");
            for kf in &context.key_files {
                output.push_str(&format!(
                    "- `{}` — {} ({}x)\n",
                    kf.path, kf.description, kf.touch_count
                ));
            }
            output.push('\n');
        }

        // Show pinned facts first (most important)
        let pinned_facts: Vec<_> = context.facts.iter().filter(|f| f.pinned).collect();
        if !pinned_facts.is_empty() {
            output.push_str("## Pinned (Critical Knowledge)\n");
            for fact in &pinned_facts {
                output.push_str(&format!("- {} (confidence: {:.1}, importance: {:.2})\n", fact.content, fact.confidence, fact.importance));
            }
            output.push('\n');
        }

        let categories = ["gotcha", "pattern", "convention", "environment", "failure", "preference", "insight", "debug"];
        for cat in &categories {
            let cat_facts: Vec<_> = context.facts.iter()
                .filter(|f| f.category == *cat && !f.pinned) // skip pinned (shown above)
                .collect();
            if !cat_facts.is_empty() {
                let title = format!("{}{}", &cat[..1].to_uppercase(), &cat[1..]);
                output.push_str(&format!("## {}\n", title));
                for fact in cat_facts {
                    output.push_str(&format!("- {} (confidence: {:.1}, importance: {:.2})\n", fact.content, fact.confidence, fact.importance));
                }
                output.push('\n');
            }

            // Respect token budget (rough estimate: 4 chars per token)
            if output.len() > token_budget * 4 {
                output.push_str("\n... (truncated to fit token budget)\n");
                break;
            }
        }

        let recent_fixes: Vec<_> = context
            .recent_debug
            .iter()
            .filter(|d| !d.fix.is_empty())
            .take(self.config.debug_fixes_shown)
            .collect();
        if !recent_fixes.is_empty() {
            output.push_str("## Debug Patterns\n");
            for dbg in recent_fixes {
                output.push_str(&format!("- {} → {}\n", dbg.error_message, dbg.fix));
            }
            output.push('\n');
        }

        if !context.patterns.work_context.is_empty()
            || !context.patterns.conventions.is_empty()
            || !context.patterns.toolchain.is_empty()
        {
            output.push_str("## Project Context\n");
            if !context.patterns.work_context.is_empty() {
                output.push_str(&format!("- Context: {}\n", context.patterns.work_context));
            }
            if !context.patterns.conventions.is_empty() {
                output.push_str(&format!("- Conventions: {}\n", context.patterns.conventions));
            }
            if !context.patterns.toolchain.is_empty() {
                output.push_str(&format!("- Toolchain: {}\n", context.patterns.toolchain));
            }
            output.push('\n');
        }

        info!("get_context → {} chars", output.len());
        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TOOL 4: observe_tool_call — auto-learning from tool usage
    // ═══════════════════════════════════════════════════════════════════════

    #[tool(description = "Observe a tool call outcome. Records commands, tracks file touches, detects error→fix pairs, and auto-extracts reusable facts via Gemini. Call this after every tool use for auto-learning.")]
    async fn observe_tool_call(
        &self,
        Parameters(params): Parameters<ObserveParams>,
    ) -> Result<CallToolResult, McpError> {
        let event = HookEvent {
            session_id: None,
            tool_name: params.tool_name.clone(),
            tool_input: params.tool_input,
            tool_response: params.tool_response,
            tool_use_id: None,
            cwd: None,
        };

        self.observer.process_hook_event(&event).await;

        let preview: String = params.tool_name.chars().take(40).collect();
        info!("observe_tool_call: {}", preview);
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Observed: {}",
            params.tool_name
        ))]))
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TOOL 5: pin_memory — mark fact as permanent
    // ═══════════════════════════════════════════════════════════════════════

    #[tool(description = "Pin a memory fact so it never expires or decays. Use for critical knowledge that must persist across all sessions.")]
    async fn pin_memory(
        &self,
        Parameters(params): Parameters<PinParams>,
    ) -> Result<CallToolResult, McpError> {
        match self.store.pin_fact(&params.fact_id) {
            Ok(true) => {
                info!("Pinned fact: {}", params.fact_id);
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Pinned: {}", params.fact_id
                ))]))
            }
            Ok(false) => {
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Fact not found: {}", params.fact_id
                ))]))
            }
            Err(e) => Err(McpError::internal_error(format!("Pin failed: {}", e), None)),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TOOL 6: unpin_memory — remove permanent marker
    // ═══════════════════════════════════════════════════════════════════════

    #[tool(description = "Unpin a memory fact, allowing it to decay normally over time.")]
    async fn unpin_memory(
        &self,
        Parameters(params): Parameters<UnpinParams>,
    ) -> Result<CallToolResult, McpError> {
        match self.store.unpin_fact(&params.fact_id) {
            Ok(true) => {
                info!("Unpinned fact: {}", params.fact_id);
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Unpinned: {}", params.fact_id
                ))]))
            }
            Ok(false) => {
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Fact not found: {}", params.fact_id
                ))]))
            }
            Err(e) => Err(McpError::internal_error(format!("Unpin failed: {}", e), None)),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TOOL 7: search_entity — find all memories about an entity
    // ═══════════════════════════════════════════════════════════════════════

    #[tool(description = "Search for all memories about a specific entity (file, class, concept, bug). Uses entity graph for relationship traversal.")]
    async fn search_entity(
        &self,
        Parameters(params): Parameters<SearchEntityParams>,
    ) -> Result<CallToolResult, McpError> {
        let limit = params.limit.unwrap_or(20);
        let results = self.store.search_entity(&params.name, limit);

        if results.is_empty() {
            // Fall back to FTS search if entity graph has no results
            let fts_results = self.store.fts_search(&params.name, limit);
            if fts_results.is_empty() {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "No memories found about '{}'. Try a broader search with search_memory.",
                    params.name
                ))]));
            }
            let json = serde_json::to_string_pretty(&fts_results).map_err(|e| {
                McpError::internal_error(format!("Serialize error: {}", e), None)
            })?;
            return Ok(CallToolResult::success(vec![Content::text(json)]));
        }

        let json = serde_json::to_string_pretty(&results).map_err(|e| {
            McpError::internal_error(format!("Serialize error: {}", e), None)
        })?;

        info!("search_entity('{}') → {} results", params.name, results.len());
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TOOL 8: consolidate_memory — prune, merge, boost
    // ═══════════════════════════════════════════════════════════════════════

    #[tool(description = "Consolidate memory: prune weak/expired facts, merge near-duplicates via Gemini, boost frequently-accessed facts. Run periodically for memory hygiene.")]
    async fn consolidate_memory(
        &self,
        Parameters(params): Parameters<ConsolidateParams>,
    ) -> Result<CallToolResult, McpError> {
        let threshold = params.merge_threshold.unwrap_or(self.store.merge_similarity);
        let dry_run = params.dry_run.unwrap_or(false);
        let mut report = String::new();

        // 1. Apply decay to all non-pinned facts
        let archived = if !dry_run {
            self.store.apply_decay().unwrap_or(0)
        } else {
            0
        };
        report.push_str(&format!("Decay applied: {} facts archived (below threshold)\n", archived));

        // 2. Clean up TTL-expired facts
        let ttl_cleaned = if !dry_run {
            self.store.cleanup_ttl_facts().unwrap_or(0)
        } else {
            0
        };
        report.push_str(&format!("TTL cleanup: {} expired facts archived\n", ttl_cleaned));

        // 3. Find and merge similar pairs
        let pairs = self.store.find_similar_pairs(threshold);
        report.push_str(&format!("Similar pairs found: {} (threshold: {:.2})\n", pairs.len(), threshold));

        let mut merged_count = 0;
        for (fact_a, fact_b, sim) in &pairs {
            if dry_run {
                report.push_str(&format!(
                    "  Would merge (sim={:.3}): '{}...' + '{}...'\n",
                    sim,
                    &fact_a.content[..fact_a.content.len().min(50)],
                    &fact_b.content[..fact_b.content.len().min(50)],
                ));
                continue;
            }

            // Use Gemini to merge the content
            match self.embedder.merge_facts(&fact_a.content, &fact_b.content).await {
                Ok(merged_content) => {
                    // Create merged fact
                    let merged_fact = MemoryFact {
                        id: format!("fact_{}", &uuid::Uuid::new_v4().to_string()[..8]),
                        content: merged_content.clone(),
                        category: fact_a.category.clone(),
                        confidence: fact_a.confidence.max(fact_b.confidence),
                        created_at: chrono::Utc::now().to_rfc3339(),
                        source: format!("merged:{}+{}", fact_a.id, fact_b.id),
                        expires_at: None,
                        pinned: fact_a.pinned || fact_b.pinned,
                        importance: fact_a.importance.max(fact_b.importance),
                        access_count: fact_a.access_count + fact_b.access_count,
                        updated_at: chrono::Utc::now().to_rfc3339(),
                        valid_at: None,
                        invalid_at: None,
                        superseded_by: None,
                        ttl: None,
                        session_id: Some(self.observer.session_id().to_string()),
                        embedding: None,
                    };

                    if let Err(e) = self.store.insert_fact(&merged_fact) {
                        warn!("Failed to insert merged fact: {}", e);
                        continue;
                    }

                    // Embed the merged fact
                    if let Ok(emb) = self.embedder.embed_one_async(
                        &format!("[{}] {}", merged_fact.category, merged_content)
                    ).await {
                        let _ = self.store.update_embedding("facts", &merged_fact.id, &emb);
                    }

                    // Mark old facts as merged
                    let _ = self.store.mark_merged(&fact_a.id, &merged_fact.id);
                    let _ = self.store.mark_merged(&fact_b.id, &merged_fact.id);

                    // Transfer entity edges to merged fact
                    // (entity edges from old facts remain pointing to old facts,
                    //  but the merged fact will get its own entity extraction)
                    if let Ok(entities) = self.embedder.extract_entities(&merged_content).await {
                        self.observer.link_entities_to_fact(&merged_fact.id, &entities);
                    }

                    merged_count += 1;
                    report.push_str(&format!(
                        "  Merged (sim={:.3}): {} + {} → {}\n",
                        sim, fact_a.id, fact_b.id, merged_fact.id,
                    ));
                }
                Err(e) => {
                    warn!("Failed to merge facts: {}", e);
                }
            }
        }

        report.push_str(&format!("\nTotal merged: {}\n", merged_count));

        // 4. Stats
        let (facts, cmds, debug, wf) = self.store.stats();
        report.push_str(&format!("Final stats: {} facts, {} commands, {} debug, {} workflows\n", facts, cmds, debug, wf));

        info!("consolidate_memory: archived={}, ttl_cleaned={}, merged={}", archived, ttl_cleaned, merged_count);
        Ok(CallToolResult::success(vec![Content::text(report)]))
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TOOL 9: reflect — synthesize insights from memories
    // ═══════════════════════════════════════════════════════════════════════

    #[tool(description = "Synthesize insights from related memories. Uses Gemini to find patterns across 50+ memories and produce actionable insights.")]
    async fn reflect(
        &self,
        Parameters(params): Parameters<ReflectParams>,
    ) -> Result<CallToolResult, McpError> {
        let limit = params.limit.unwrap_or(50);
        let facts = self.store.load_active_facts();

        let selected: Vec<String> = if let Some(ref topic) = params.topic {
            // Search for topic-related facts
            let engine = SearchEngine::new(&self.store, &self.embedder);
            let results = engine.search(
                topic, limit, None, 0.2, false, None,
            ).await;
            results.iter().map(|r| r.content.clone()).collect()
        } else {
            // Use all active facts up to limit
            facts.iter()
                .take(limit)
                .map(|f| format!("[{}] {}", f.category, f.content))
                .collect()
        };

        if selected.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No memories to reflect on. Save some facts first.",
            )]));
        }

        match self.embedder.reflect(&selected, params.topic.as_deref()).await {
            Ok(insights) => {
                info!("reflect({}) → {} chars from {} memories",
                    params.topic.as_deref().unwrap_or("all"),
                    insights.len(),
                    selected.len()
                );
                Ok(CallToolResult::success(vec![Content::text(insights)]))
            }
            Err(e) => {
                Err(McpError::internal_error(format!("Reflect failed: {}", e), None))
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TOOL 10: graph_query — traverse entity relationships
    // ═══════════════════════════════════════════════════════════════════════

    #[tool(description = "Query the memory graph for an entity. Returns all connected facts, related entities, and relationship types. Use for navigating relationships between files, bugs, concepts.")]
    async fn graph_query(
        &self,
        Parameters(params): Parameters<GraphQueryParams>,
    ) -> Result<CallToolResult, McpError> {
        let nodes = self.store.graph_query(&params.entity);

        if nodes.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "No entity found matching '{}'. Entities are auto-extracted when saving facts. Try save_memory with content mentioning files or concepts first.",
                params.entity
            ))]));
        }

        let json = serde_json::to_string_pretty(&nodes).map_err(|e| {
            McpError::internal_error(format!("Serialize error: {}", e), None)
        })?;

        let total_facts: usize = nodes.iter().map(|n| n.facts.len()).sum();
        let total_connected: usize = nodes.iter().map(|n| n.connected_entities.len()).sum();
        info!("graph_query('{}') → {} nodes, {} facts, {} connected entities",
            params.entity, nodes.len(), total_facts, total_connected);

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TOOL 11: list_pinned — audit all pinned facts
    // ═══════════════════════════════════════════════════════════════════════

    #[tool(description = "List all pinned memory facts. Returns facts that are marked as permanent/critical knowledge.")]
    async fn list_pinned(
        &self,
        Parameters(params): Parameters<ListPinnedParams>,
    ) -> Result<CallToolResult, McpError> {
        let limit = params.limit.unwrap_or(100);
        let facts = self.store.load_pinned_facts(params.category.as_deref(), limit);

        if facts.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No pinned facts found. Use pin_memory(fact_id) or save_memory(pinned=true) to pin facts.",
            )]));
        }

        let output: Vec<serde_json::Value> = facts.iter().map(|f| {
            serde_json::json!({
                "id": f.id,
                "content": f.content,
                "category": f.category,
                "confidence": f.confidence,
                "importance": format!("{:.2}", f.importance),
                "created_at": f.created_at,
            })
        }).collect();

        let json = serde_json::to_string_pretty(&output).map_err(|e| {
            McpError::internal_error(format!("Serialize error: {}", e), None)
        })?;

        info!("list_pinned → {} facts", facts.len());
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TOOL 12: bulk_manage — archive, delete, pin/unpin multiple facts
    // ═══════════════════════════════════════════════════════════════════════

    #[tool(description = "Bulk memory management: archive, delete, or pin/unpin multiple facts at once. Filter by query, category, age, or specific IDs.")]
    async fn bulk_manage(
        &self,
        Parameters(params): Parameters<BulkManageParams>,
    ) -> Result<CallToolResult, McpError> {
        let action = params.action.as_str();
        let unpinned_only = params.unpinned_only.unwrap_or(true);

        let older_than_days = params.older_than.as_deref().and_then(|s| {
            if let Some(num) = s.strip_suffix('d') {
                num.parse::<i64>().ok()
            } else {
                None
            }
        });

        let result = match action {
            "archive" => {
                let count = self.store.bulk_archive(
                    params.category.as_deref(),
                    older_than_days,
                    unpinned_only,
                ).map_err(|e| McpError::internal_error(format!("Archive failed: {}", e), None))?;
                format!("Archived {} facts", count)
            }
            "delete" => {
                if let Some(ref ids) = params.fact_ids {
                    let count = self.store.bulk_delete(ids)
                        .map_err(|e| McpError::internal_error(format!("Delete failed: {}", e), None))?;
                    format!("Deleted {} facts", count)
                } else {
                    "Error: delete requires fact_ids list".to_string()
                }
            }
            "unpin" => {
                if let Some(ref query) = params.query {
                    let count = self.store.bulk_unpin_by_query(query)
                        .map_err(|e| McpError::internal_error(format!("Unpin failed: {}", e), None))?;
                    format!("Unpinned {} facts matching '{}'", count, query)
                } else if let Some(ref ids) = params.fact_ids {
                    let mut count = 0;
                    for id in ids {
                        if self.store.unpin_fact(id).unwrap_or(false) {
                            count += 1;
                        }
                    }
                    format!("Unpinned {} facts", count)
                } else {
                    "Error: unpin requires query or fact_ids".to_string()
                }
            }
            "pin" => {
                if let Some(ref ids) = params.fact_ids {
                    let count = self.store.bulk_pin(ids)
                        .map_err(|e| McpError::internal_error(format!("Pin failed: {}", e), None))?;
                    format!("Pinned {} facts", count)
                } else {
                    "Error: pin requires fact_ids list".to_string()
                }
            }
            _ => format!("Unknown action: {}. Use: archive, delete, unpin, pin", action),
        };

        info!("bulk_manage({}): {}", action, result);
        Ok(CallToolResult::success(vec![Content::text(result)]))
    }
}

#[tool_handler]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2025_06_18,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: Some(
                "Project-scoped persistent memory for AI agents with decay, pinning, entity resolution, and consolidation. \
                 Use search_memory to find past errors/fixes, facts, and patterns (supports temporal queries like 'last 3 days'). \
                 Use save_memory to remember important facts (use pinned=true for critical knowledge). \
                 Use get_context at session start for continuity. \
                 Use pin_memory/unpin_memory to manage permanent vs decaying facts. \
                 Use search_entity to find all memories about a specific entity. \
                 Use consolidate_memory to prune weak memories and merge duplicates. \
                 Use reflect to synthesize insights from related memories. \
                 Use graph_query to traverse entity relationships. \
                 Use list_pinned to audit permanent knowledge. \
                 Use bulk_manage for batch operations (archive, delete, pin/unpin)."
                    .into(),
            ),
        }
    }
}
