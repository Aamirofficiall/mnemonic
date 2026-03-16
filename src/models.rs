use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ─── Memory Fact (bi-temporal, importance-scored) ────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFact {
    pub id: String,
    pub content: String,
    pub category: String,
    pub confidence: f64,
    // Bi-temporal timestamps (Zep/Graphiti pattern)
    pub created_at: String,  // when system learned this fact
    pub updated_at: String,  // last modification time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_at: Option<String>,   // when fact became true in reality
    #[serde(skip_serializing_if = "Option::is_none")]
    pub invalid_at: Option<String>, // when fact stopped being true (replaces superseded_by)
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    #[serde(default)]
    pub pinned: bool,
    #[serde(default = "default_importance")]
    pub importance: f64,        // 0.0-1.0 LLM-inferred (CrewAI pattern)
    #[serde(default)]
    pub access_count: u32,      // for decay reinforcement
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<String>, // backward compat — prefer invalid_at
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
}

fn default_importance() -> f64 { 0.5 }

impl MemoryFact {
    pub fn is_expired(&self) -> bool {
        if self.pinned { return false; }
        // Bi-temporal: invalid_at means fact is no longer true
        if self.invalid_at.is_some() { return true; }
        // Legacy: superseded_by
        if self.superseded_by.is_some() { return true; }
        // TTL-based expiry
        if let Some(ref exp) = self.expires_at {
            if let Ok(date) = DateTime::parse_from_rfc3339(exp) {
                if date < Utc::now() { return true; }
            }
        }
        false
    }

    pub fn is_active(&self) -> bool { !self.is_expired() }

    /// Compute decay score using ideal formula from research:
    /// importance × exp(-λ × age_days / (1 + α × ln(1 + access_count)))
    /// with floor ρ = 0.1
    pub fn decay_score(&self) -> f64 {
        if self.pinned { return 1.0; }
        let age_days = self.age_days();
        let lambda = 0.16;  // base decay rate
        let alpha = 0.3;    // reinforcement strength
        let rho = 0.1;      // minimum retention floor
        let reinforced = 1.0 + alpha * (1.0 + self.access_count as f64).ln();
        let raw = (-lambda * age_days / reinforced).exp();
        self.importance * raw.max(rho)
    }

    pub fn age_days(&self) -> f64 {
        if let Ok(created) = DateTime::parse_from_rfc3339(&self.updated_at) {
            let age = Utc::now() - created.with_timezone(&Utc);
            age.num_hours() as f64 / 24.0
        } else if let Ok(created) = DateTime::parse_from_rfc3339(&self.created_at) {
            let age = Utc::now() - created.with_timezone(&Utc);
            age.num_hours() as f64 / 24.0
        } else {
            0.0
        }
    }
}

// ─── AUDN Action (mem0 pattern) ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AudnAction {
    Add,
    Update(String),  // ID of fact to update
    Delete(String),  // ID of fact to invalidate
    None,            // skip — duplicate
}

// ─── Entity (for memory graph) ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityEdge {
    pub entity_id: String,
    pub fact_id: String,
    pub relation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityAlias {
    pub entity_id: String,
    pub alias: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub entity: Entity,
    pub facts: Vec<MemoryFact>,
    pub connected_entities: Vec<String>,
    pub relations: std::collections::HashMap<String, Vec<String>>,
}

// ─── Command Record ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandRecord {
    pub id: String,
    pub tool: String,
    pub outcome: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_snippet: Option<String>,
    pub session_id: String,
    pub recorded_at: String,
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
}

// ─── Debug Record ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugRecord {
    pub id: String,
    pub error_message: String,
    pub root_cause: String,
    pub fix: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    pub session_id: String,
    pub recorded_at: String,
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
}

// ─── Workflow Pattern ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowPattern {
    pub id: String,
    pub name: String,
    pub steps: Vec<String>,
    pub success_count: i32,
    pub failure_count: i32,
    pub last_used: String,
    pub session_id: String,
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
}

// ─── Key File ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyFile {
    pub path: String,
    pub description: String,
    pub touch_count: i32,
}

// ─── Handoff ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Handoff {
    #[serde(default)] pub last_commit: String,
    #[serde(default)] pub uncommitted_files: Vec<String>,
    #[serde(default)] pub next_task: String,
    #[serde(default)] pub blocked_on: String,
    #[serde(default)] pub updated_at: String,
}

// ─── Patterns ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Patterns {
    #[serde(default)] pub work_context: String,
    #[serde(default)] pub conventions: String,
    #[serde(default)] pub toolchain: String,
}

// ─── Search Result ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub result_type: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    pub similarity: f32,
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

// ─── Full Memory Context ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryContext {
    pub facts: Vec<MemoryFact>,
    pub patterns: Patterns,
    pub key_files: Vec<KeyFile>,
    pub handoff: Handoff,
    pub recent_commands: Vec<CommandRecord>,
    pub recent_debug: Vec<DebugRecord>,
    pub workflows: Vec<WorkflowPattern>,
}

// ─── Hook Event ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct HookEvent {
    pub session_id: Option<String>,
    pub tool_name: String,
    pub tool_input: Option<serde_json::Value>,
    pub tool_response: Option<serde_json::Value>,
    pub tool_use_id: Option<String>,
    pub cwd: Option<String>,
}

// ─── MCP Tool Parameters ────────────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(description = "Search project memory. Multi-stage: FTS5 + vector + RRF fusion + optional cross-encoder rerank.")]
pub struct SearchParams {
    #[schemars(description = "Natural language query. Supports temporal: 'last 3 days'.")]
    pub query: String,
    #[schemars(description = "Max results (default 10)")]
    pub limit: Option<usize>,
    #[schemars(description = "Filter by categories: fact, command, debug, workflow")]
    pub categories: Option<Vec<String>>,
    #[schemars(description = "Max tokens in response")]
    pub token_budget: Option<usize>,
    #[schemars(description = "Only pinned facts")]
    pub pinned_only: Option<bool>,
    #[schemars(description = "Facts created after this ISO date")]
    pub since: Option<String>,
    #[schemars(description = "Deep mode: adds cross-encoder reranking (~200ms)")]
    pub deep: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(description = "Get full project memory context with memory pressure warnings.")]
pub struct GetContextParams {
    #[schemars(description = "Max tokens (default 25000)")]
    pub token_budget: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(description = "Save fact with AUDN cycle: auto-dedup, importance scoring, entity extraction.")]
pub struct SaveParams {
    #[schemars(description = "The fact to remember")]
    pub content: String,
    #[schemars(description = "Category: gotcha, pattern, preference, convention, environment, failure, insight, debug")]
    pub category: String,
    #[schemars(description = "Confidence 0.0-1.0 (default 0.9)")]
    pub confidence: Option<f64>,
    #[schemars(description = "Pin permanently")]
    pub pinned: Option<bool>,
    #[schemars(description = "ID of fact this supersedes")]
    pub supersedes: Option<String>,
    #[schemars(description = "TTL: 'session', '1h', '1d', '7d', '30d', 'permanent'")]
    pub ttl: Option<String>,
    #[schemars(description = "Override importance 0.0-1.0 (default: LLM-inferred)")]
    pub importance: Option<f64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(description = "Observe tool call for auto-learning.")]
pub struct ObserveParams {
    #[schemars(description = "Tool name")]
    pub tool_name: String,
    #[schemars(description = "Tool input JSON")]
    pub tool_input: Option<serde_json::Value>,
    #[schemars(description = "Tool response JSON")]
    pub tool_response: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(description = "Pin a fact permanently.")]
pub struct PinParams { pub fact_id: String }

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(description = "Unpin a fact.")]
pub struct UnpinParams { pub fact_id: String }

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(description = "Search entity graph. Includes alias resolution.")]
pub struct SearchEntityParams {
    pub name: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(description = "Consolidate: decay + TTL cleanup + dual-layer dedup (0.98 hash + 0.85 LLM merge).")]
pub struct ConsolidateParams {
    pub merge_threshold: Option<f32>,
    pub dry_run: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(description = "Reflect: synthesize insights from memories via Gemini.")]
pub struct ReflectParams {
    pub topic: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(description = "Query entity graph with bidirectional traversal.")]
pub struct GraphQueryParams {
    pub entity: String,
    pub depth: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(description = "List all pinned facts.")]
pub struct ListPinnedParams {
    pub category: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(description = "Bulk operations: archive, delete, pin, unpin.")]
pub struct BulkManageParams {
    pub action: String,
    pub query: Option<String>,
    pub category: Option<String>,
    pub unpinned_only: Option<bool>,
    pub older_than: Option<String>,
    pub fact_ids: Option<Vec<String>>,
}

// ─── Embedding Item ─────────────────────────────────────────────────────────

pub struct EmbeddingItem {
    pub id: String,
    pub item_type: String,
    pub text: String,
}

// ─── Extracted Entity ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub name: String,
    pub entity_type: String,
    pub relation: String,
}

// ─── AUDN + Importance combined response from Gemini ────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct AudnResponse {
    pub action: String,          // "add", "update", "delete", "none"
    pub importance: f64,         // 0.0-1.0
    #[serde(default)]
    pub update_target_id: Option<String>,
    #[serde(default)]
    pub merged_content: Option<String>,
    #[serde(default)]
    pub entities: Vec<ExtractedEntity>,
    #[serde(default)]
    pub aliases: Vec<String>,    // alternative names for entities
}
