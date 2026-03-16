use serde::Deserialize;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

// ─── Public Config ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MnemonicConfig {
    // API
    pub gemini_key: Option<String>,

    // Embeddings
    pub embed_model: String,
    pub embed_dims: usize,
    pub embed_batch_size: usize,

    // LLM (fact extraction + root cause)
    pub llm_model: String,
    pub llm_temperature: f32,
    pub llm_max_tokens: u32,

    // Search
    pub min_similarity: f32,
    pub default_limit: usize,

    // Categories
    pub custom_categories: Vec<String>,

    // Observe
    pub extract_facts: bool,
    pub extract_tools: Vec<String>,
    pub write_tools: Vec<String>,

    // Limits
    pub max_commands: usize,
    pub max_debug: usize,
    pub max_key_files: usize,
    pub fact_expiry_days: i64,
    pub snippet_length: usize,

    // Context
    pub recent_commands: usize,
    pub recent_debug: usize,
    pub debug_fixes_shown: usize,

    // Decay (ideal formula: importance × exp(-λ × age / (1 + α × ln(1 + access))))
    pub decay_rate: f64,           // legacy — kept for backward compat
    pub half_life_days: f64,       // CrewAI pattern: 7 for sprints, 30 default, 180 for KB
    pub decay_lambda: f64,         // base decay rate λ (default 0.16)
    pub decay_alpha: f64,          // reinforcement strength α (default 0.3)
    pub decay_floor: f64,          // minimum retention floor ρ (default 0.1)

    // Composite scoring (CrewAI pattern: w_sim × similarity + w_decay × decay + w_imp × importance)
    pub weight_similarity: f64,    // default 0.5
    pub weight_decay: f64,         // default 0.3
    pub weight_importance: f64,    // default 0.2

    // RRF fusion (Zep/Graphiti pattern)
    pub rrf_k: f64,                // default 60.0
    pub rrf_bm25_weight: f64,      // default 1.0
    pub rrf_vector_weight: f64,    // default 0.7

    // Dedup thresholds (CrewAI dual-layer pattern)
    pub dedup_hash_threshold: f32,  // 0.98 — instant dedup, no LLM
    pub dedup_llm_threshold: f32,   // 0.85 — LLM merge
    pub merge_similarity: f32,      // consolidation threshold (legacy)

    // Entity resolution
    pub entity_match_threshold: f32, // 0.7 (mem0 pattern)

    // Memory pressure (Letta pattern)
    pub memory_pressure_warn: f64,   // 0.7 — warn agent
    pub memory_pressure_compact: f64, // 1.0 — auto-compact

    // Context
    pub max_context_tokens: usize,

    // Sync (optional cloud backup)
    pub sync_enabled: bool,
    pub sync_backend: String,
    pub sync_bucket: String,
    pub sync_region: String,
    pub sync_prefix: String,
    pub sync_interval_secs: u64,
}

impl Default for MnemonicConfig {
    fn default() -> Self {
        Self {
            gemini_key: None,
            embed_model: "gemini-embedding-001".into(),
            embed_dims: 768,
            embed_batch_size: 100,
            llm_model: "gemini-2.0-flash".into(),
            llm_temperature: 0.1,
            llm_max_tokens: 1024,
            min_similarity: 0.3,
            default_limit: 10,
            custom_categories: vec![],
            extract_facts: true,
            extract_tools: vec![
                "Bash".into(), "Edit".into(), "Write".into(),
                "Grep".into(), "Glob".into(),
            ],
            write_tools: vec![
                "Write".into(), "Edit".into(), "NotebookEdit".into(),
            ],
            max_commands: 200,
            max_debug: 100,
            max_key_files: 15,
            fact_expiry_days: 30,
            snippet_length: 500,
            recent_commands: 20,
            recent_debug: 20,
            debug_fixes_shown: 5,
            decay_rate: 0.95,
            half_life_days: 30.0,
            decay_lambda: 0.16,
            decay_alpha: 0.3,
            decay_floor: 0.1,
            weight_similarity: 0.5,
            weight_decay: 0.3,
            weight_importance: 0.2,
            rrf_k: 60.0,
            rrf_bm25_weight: 1.0,
            rrf_vector_weight: 0.7,
            dedup_hash_threshold: 0.98,
            dedup_llm_threshold: 0.85,
            merge_similarity: 0.92,
            entity_match_threshold: 0.7,
            memory_pressure_warn: 0.7,
            memory_pressure_compact: 1.0,
            max_context_tokens: 25000,
            sync_enabled: false,
            sync_backend: "s3".into(),
            sync_bucket: "your-bucket".into(),
            sync_region: "us-east-1".into(),
            sync_prefix: "projects".into(),
            sync_interval_secs: 300,
        }
    }
}

impl MnemonicConfig {
    /// Load config: defaults → global config → project config → env vars
    pub fn load(project_root: Option<&Path>) -> Self {
        let mut config = Self::default();

        // Layer 1: Global config
        let global_path = Self::global_config_path();
        if global_path.exists() {
            if let Some(toml) = Self::load_toml(&global_path) {
                config.merge(&toml);
                info!("Loaded global config: {}", global_path.display());
            }
        }

        // Layer 2: Project config
        if let Some(root) = project_root {
            let project_path = root.join(".mnemonic.toml");
            if project_path.exists() {
                if let Some(toml) = Self::load_toml(&project_path) {
                    config.merge(&toml);
                    info!("Loaded project config: {}", project_path.display());
                }
            }
        }

        // Layer 3: Env vars override config file
        if let Ok(key) = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
        {
            config.gemini_key = Some(key);
        }

        config
    }

    /// Path to global config: ~/.mnemonic/config.toml
    pub fn global_config_path() -> PathBuf {
        let home = std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."));
        home.join(".mnemonic").join("config.toml")
    }

    /// Build Gemini embed URL from config model name
    pub fn embed_url(&self) -> String {
        format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:embedContent",
            self.embed_model
        )
    }

    /// Build Gemini batch embed URL from config model name
    pub fn batch_embed_url(&self) -> String {
        format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:batchEmbedContents",
            self.embed_model
        )
    }

    /// Build Gemini generate URL from config LLM model name
    pub fn generate_url(&self) -> String {
        format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            self.llm_model
        )
    }

    /// Write a default global config file
    pub fn write_global_config(api_key: Option<&str>) -> std::io::Result<PathBuf> {
        let path = Self::global_config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let key_line = match api_key {
            Some(k) => format!("gemini_key = \"{}\"", k),
            None => "# gemini_key = \"your-key-here\"  # Get one at https://aistudio.google.com/apikey".into(),
        };

        let content = format!(
r#"# mnemonic global configuration
# Docs: https://github.com/Aamirofficiall/mnemonic/blob/main/docs/INTEGRATION.md

[api]
{key_line}

[embeddings]
model = "gemini-embedding-001"
dimensions = 768
batch_size = 100

[llm]
model = "gemini-2.0-flash"
temperature = 0.1
max_tokens = 1024

[search]
min_similarity = 0.3
default_limit = 10
"#);

        std::fs::write(&path, content)?;
        Ok(path)
    }

    /// Write a default project config file
    pub fn write_project_config(project_root: &Path, project_name: &str) -> std::io::Result<PathBuf> {
        let path = project_root.join(".mnemonic.toml");

        let content = format!(
r#"# mnemonic project configuration
# Overrides ~/.mnemonic/config.toml for this project

[project]
name = "{project_name}"

# [categories]
# custom = ["infra", "security", "performance"]

# [search]
# min_similarity = 0.4
# default_limit = 5

[observe]
extract_facts = true
extract_tools = ["Bash", "Edit", "Write", "Grep", "Glob"]
write_tools = ["Write", "Edit", "NotebookEdit"]

[limits]
max_commands = 200
max_debug = 100
max_key_files = 15
fact_expiry_days = 30
snippet_length = 500

[context]
recent_commands = 20
recent_debug = 20
debug_fixes_shown = 5
"#);

        std::fs::write(&path, content)?;
        Ok(path)
    }

    // ─── Private ─────────────────────────────────────────────────────────────

    fn load_toml(path: &Path) -> Option<TomlConfig> {
        let content = std::fs::read_to_string(path).ok()?;
        match toml::from_str::<TomlConfig>(&content) {
            Ok(config) => Some(config),
            Err(e) => {
                warn!("Failed to parse {}: {}", path.display(), e);
                None
            }
        }
    }

    fn merge(&mut self, t: &TomlConfig) {
        // API
        if let Some(ref api) = t.api {
            if let Some(ref key) = api.gemini_key {
                self.gemini_key = Some(key.clone());
            }
        }

        // Embeddings
        if let Some(ref emb) = t.embeddings {
            if let Some(ref model) = emb.model { self.embed_model = model.clone(); }
            if let Some(dims) = emb.dimensions { self.embed_dims = dims; }
            if let Some(bs) = emb.batch_size { self.embed_batch_size = bs; }
        }

        // LLM
        if let Some(ref llm) = t.llm {
            if let Some(ref model) = llm.model { self.llm_model = model.clone(); }
            if let Some(temp) = llm.temperature { self.llm_temperature = temp; }
            if let Some(mt) = llm.max_tokens { self.llm_max_tokens = mt; }
        }

        // Search
        if let Some(ref search) = t.search {
            if let Some(ms) = search.min_similarity { self.min_similarity = ms; }
            if let Some(dl) = search.default_limit { self.default_limit = dl; }
        }

        // Categories
        if let Some(ref cats) = t.categories {
            if let Some(ref custom) = cats.custom {
                self.custom_categories = custom.clone();
            }
        }

        // Observe
        if let Some(ref obs) = t.observe {
            if let Some(ef) = obs.extract_facts { self.extract_facts = ef; }
            if let Some(ref et) = obs.extract_tools { self.extract_tools = et.clone(); }
            if let Some(ref wt) = obs.write_tools { self.write_tools = wt.clone(); }
        }

        // Limits
        if let Some(ref lim) = t.limits {
            if let Some(mc) = lim.max_commands { self.max_commands = mc; }
            if let Some(md) = lim.max_debug { self.max_debug = md; }
            if let Some(mkf) = lim.max_key_files { self.max_key_files = mkf; }
            if let Some(fed) = lim.fact_expiry_days { self.fact_expiry_days = fed; }
            if let Some(sl) = lim.snippet_length { self.snippet_length = sl; }
        }

        // Context
        if let Some(ref ctx) = t.context {
            if let Some(rc) = ctx.recent_commands { self.recent_commands = rc; }
            if let Some(rd) = ctx.recent_debug { self.recent_debug = rd; }
            if let Some(dfs) = ctx.debug_fixes_shown { self.debug_fixes_shown = dfs; }
        }

        // Decay
        if let Some(ref decay) = t.decay {
            if let Some(r) = decay.rate { self.decay_rate = r; }
            if let Some(h) = decay.half_life_days { self.half_life_days = h; }
            if let Some(l) = decay.lambda { self.decay_lambda = l; }
            if let Some(a) = decay.alpha { self.decay_alpha = a; }
            if let Some(f) = decay.floor { self.decay_floor = f; }
            if let Some(ms) = decay.merge_similarity_threshold { self.merge_similarity = ms; }
            if let Some(mt) = decay.max_context_tokens { self.max_context_tokens = mt; }
        }

        // Scoring
        if let Some(ref sc) = t.scoring {
            if let Some(v) = sc.weight_similarity { self.weight_similarity = v; }
            if let Some(v) = sc.weight_decay { self.weight_decay = v; }
            if let Some(v) = sc.weight_importance { self.weight_importance = v; }
            if let Some(v) = sc.rrf_k { self.rrf_k = v; }
            if let Some(v) = sc.rrf_bm25_weight { self.rrf_bm25_weight = v; }
            if let Some(v) = sc.rrf_vector_weight { self.rrf_vector_weight = v; }
            if let Some(v) = sc.dedup_hash_threshold { self.dedup_hash_threshold = v; }
            if let Some(v) = sc.dedup_llm_threshold { self.dedup_llm_threshold = v; }
            if let Some(v) = sc.entity_match_threshold { self.entity_match_threshold = v; }
            if let Some(v) = sc.memory_pressure_warn { self.memory_pressure_warn = v; }
            if let Some(v) = sc.memory_pressure_compact { self.memory_pressure_compact = v; }
        }

        // Sync
        if let Some(ref sync) = t.sync {
            if let Some(en) = sync.enabled { self.sync_enabled = en; }
            if let Some(ref be) = sync.backend { self.sync_backend = be.clone(); }
            if let Some(ref bu) = sync.bucket { self.sync_bucket = bu.clone(); }
            if let Some(ref re) = sync.region { self.sync_region = re.clone(); }
            if let Some(ref pr) = sync.prefix { self.sync_prefix = pr.clone(); }
            if let Some(si) = sync.sync_interval_secs { self.sync_interval_secs = si; }
        }
    }
}

// ─── TOML Deserialization Structs (all fields optional for partial configs) ──

#[derive(Deserialize, Default)]
struct TomlConfig {
    api: Option<TomlApi>,
    embeddings: Option<TomlEmbeddings>,
    llm: Option<TomlLlm>,
    search: Option<TomlSearch>,
    project: Option<TomlProject>,
    categories: Option<TomlCategories>,
    observe: Option<TomlObserve>,
    limits: Option<TomlLimits>,
    context: Option<TomlContext>,
    decay: Option<TomlDecay>,
    scoring: Option<TomlScoring>,
    sync: Option<TomlSync>,
}

#[derive(Deserialize, Default)]
struct TomlDecay {
    rate: Option<f64>,
    half_life_days: Option<f64>,
    lambda: Option<f64>,
    alpha: Option<f64>,
    floor: Option<f64>,
    merge_similarity_threshold: Option<f32>,
    max_context_tokens: Option<usize>,
}

#[derive(Deserialize, Default)]
struct TomlScoring {
    weight_similarity: Option<f64>,
    weight_decay: Option<f64>,
    weight_importance: Option<f64>,
    rrf_k: Option<f64>,
    rrf_bm25_weight: Option<f64>,
    rrf_vector_weight: Option<f64>,
    dedup_hash_threshold: Option<f32>,
    dedup_llm_threshold: Option<f32>,
    entity_match_threshold: Option<f32>,
    memory_pressure_warn: Option<f64>,
    memory_pressure_compact: Option<f64>,
}

#[derive(Deserialize, Default)]
struct TomlApi {
    gemini_key: Option<String>,
}

#[derive(Deserialize, Default)]
struct TomlEmbeddings {
    model: Option<String>,
    dimensions: Option<usize>,
    batch_size: Option<usize>,
}

#[derive(Deserialize, Default)]
struct TomlLlm {
    model: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

#[derive(Deserialize, Default)]
struct TomlSearch {
    min_similarity: Option<f32>,
    default_limit: Option<usize>,
}

#[derive(Deserialize, Default)]
struct TomlProject {
    name: Option<String>,
}

#[derive(Deserialize, Default)]
struct TomlCategories {
    custom: Option<Vec<String>>,
}

#[derive(Deserialize, Default)]
struct TomlObserve {
    extract_facts: Option<bool>,
    extract_tools: Option<Vec<String>>,
    write_tools: Option<Vec<String>>,
}

#[derive(Deserialize, Default)]
struct TomlLimits {
    max_commands: Option<usize>,
    max_debug: Option<usize>,
    max_key_files: Option<usize>,
    fact_expiry_days: Option<i64>,
    snippet_length: Option<usize>,
}

#[derive(Deserialize, Default)]
struct TomlContext {
    recent_commands: Option<usize>,
    recent_debug: Option<usize>,
    debug_fixes_shown: Option<usize>,
}

#[derive(Deserialize, Default)]
struct TomlSync {
    enabled: Option<bool>,
    backend: Option<String>,
    bucket: Option<String>,
    region: Option<String>,
    prefix: Option<String>,
    sync_interval_secs: Option<u64>,
}
