mod config;
mod embeddings;
mod models;
mod observer;
mod project;
mod search;
mod server;
mod store;
#[cfg(feature = "sync")]
mod sync;

use anyhow::Result;
use clap::{Parser, Subcommand};
use rmcp::{transport::stdio, ServiceExt};
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "mnemonic",
    about = "State-of-the-art memory engine for AI coding agents — RRF search, bi-temporal, decay, entity graph",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize memory for current project
    Init,
    /// Show memory stats
    Stats,
    /// Start MCP server (default — backwards compatible)
    Serve,

    // ─── Direct CLI commands (0 token overhead) ─────────────────────────
    /// Save a fact to memory
    Save {
        /// The fact content
        content: String,
        /// Category: gotcha, pattern, convention, failure, insight, debug, environment, preference
        #[arg(long, short = 'c', default_value = "insight")]
        cat: String,
        /// Pin permanently
        #[arg(long)]
        pin: bool,
        /// TTL: session, 1h, 1d, 7d, 30d, permanent
        #[arg(long)]
        ttl: Option<String>,
        /// Importance 0.0-1.0 (default: auto)
        #[arg(long)]
        importance: Option<f64>,
        /// ID of fact this supersedes
        #[arg(long)]
        supersedes: Option<String>,
    },
    /// Search memory (RRF fusion: FTS5 + vector + composite scoring)
    Search {
        /// Natural language query
        query: String,
        /// Max results
        #[arg(long, short = 'n', default_value = "10")]
        limit: usize,
        /// Only pinned facts
        #[arg(long)]
        pinned: bool,
        /// Only facts after this date (ISO)
        #[arg(long)]
        since: Option<String>,
        /// Deep mode: adds cross-encoder reranking
        #[arg(long)]
        deep: bool,
    },
    /// Search entity graph
    Entity {
        /// Entity name (file, class, concept)
        name: String,
        #[arg(long, short = 'n', default_value = "20")]
        limit: usize,
    },
    /// Query entity graph with bidirectional traversal
    Graph {
        /// Entity to query
        entity: String,
    },
    /// Get full context for session start
    Context {
        /// Token budget
        #[arg(long, default_value = "25000")]
        budget: usize,
    },
    /// Pin a fact permanently
    Pin { fact_id: String },
    /// Unpin a fact
    Unpin { fact_id: String },
    /// Supersede an old fact with new content
    Supersede {
        /// Old fact ID to invalidate
        old_id: String,
        /// New content
        content: String,
        /// Category
        #[arg(long, short = 'c', default_value = "insight")]
        cat: String,
    },
    /// Consolidate: decay + dedup + merge
    Consolidate {
        /// Similarity threshold for merging
        #[arg(long, default_value = "0.90")]
        threshold: f32,
        /// Dry run — report only
        #[arg(long)]
        dry_run: bool,
    },
    /// Reflect: synthesize insights
    Reflect {
        /// Topic to focus on
        #[arg(long)]
        topic: Option<String>,
        /// Max memories to analyze
        #[arg(long, default_value = "50")]
        limit: usize,
    },
    /// Bulk operations: archive, delete, pin, unpin
    Bulk {
        /// Action: archive, delete, pin, unpin
        action: String,
        /// Filter by category
        #[arg(long)]
        cat: Option<String>,
        /// Only unpinned
        #[arg(long)]
        unpinned: bool,
        /// Older than (e.g. 7d, 30d)
        #[arg(long)]
        older_than: Option<String>,
    },
    /// List pinned facts
    Pinned {
        /// Filter by category
        #[arg(long)]
        cat: Option<String>,
        #[arg(long, short = 'n', default_value = "100")]
        limit: usize,
    },
    /// Observe tool call (reads JSON from stdin)
    Observe,
}

/// Shared runtime: config + store + embedder
struct Runtime {
    store: Arc<store::MemoryStore>,
    embedder: Arc<embeddings::Embedder>,
    config: Arc<config::MnemonicConfig>,
    observer: Arc<observer::Observer>,
}

impl Runtime {
    fn new() -> Result<Self> {
        let project_root = project::detect_project_root();
        let cfg = config::MnemonicConfig::load(project_root.as_deref());
        let config = Arc::new(cfg);
        let db_path = project::memory_db_path();
        let store = Arc::new(store::MemoryStore::open(&db_path, &config)?);
        let embedder = Arc::new(embeddings::Embedder::new(&config)?);
        let observer = Arc::new(observer::Observer::new(store.clone(), embedder.clone(), &config));
        Ok(Self { store, embedder, config, observer })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    eprintln!("[MNEMONIC] v2 binary started");
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env().add_directive(tracing::Level::WARN.into()),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    let cli = Cli::parse();

    match cli.command.unwrap_or(Commands::Serve) {
        Commands::Init => cmd_init(),
        Commands::Stats => cmd_stats(),
        Commands::Serve => cmd_serve().await,

        // ─── Direct CLI commands ────────────────────────────────────────
        Commands::Save { content, cat, pin, ttl, importance, supersedes } => {
            let rt = Runtime::new()?;
            cmd_save(&rt, &content, &cat, pin, ttl.as_deref(), importance, supersedes.as_deref()).await
        }
        Commands::Search { query, limit, pinned, since, deep } => {
            let rt = Runtime::new()?;
            cmd_search(&rt, &query, limit, pinned, since.as_deref(), deep).await
        }
        Commands::Entity { name, limit } => {
            let rt = Runtime::new()?;
            cmd_entity(&rt, &name, limit)
        }
        Commands::Graph { entity } => {
            let rt = Runtime::new()?;
            cmd_graph(&rt, &entity)
        }
        Commands::Context { budget } => {
            let rt = Runtime::new()?;
            cmd_context(&rt, budget)
        }
        Commands::Pin { fact_id } => {
            let rt = Runtime::new()?;
            cmd_pin(&rt, &fact_id)
        }
        Commands::Unpin { fact_id } => {
            let rt = Runtime::new()?;
            cmd_unpin(&rt, &fact_id)
        }
        Commands::Supersede { old_id, content, cat } => {
            let rt = Runtime::new()?;
            cmd_supersede(&rt, &old_id, &content, &cat).await
        }
        Commands::Consolidate { threshold, dry_run } => {
            let rt = Runtime::new()?;
            cmd_consolidate(&rt, threshold, dry_run).await
        }
        Commands::Reflect { topic, limit } => {
            let rt = Runtime::new()?;
            cmd_reflect(&rt, topic.as_deref(), limit).await
        }
        Commands::Bulk { action, cat, unpinned, older_than } => {
            let rt = Runtime::new()?;
            cmd_bulk(&rt, &action, cat.as_deref(), unpinned, older_than.as_deref())
        }
        Commands::Pinned { cat, limit } => {
            let rt = Runtime::new()?;
            cmd_pinned(&rt, cat.as_deref(), limit)
        }
        Commands::Observe => {
            let rt = Runtime::new()?;
            cmd_observe(&rt).await
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLI command implementations — compact JSON output for minimal tokens
// ═══════════════════════════════════════════════════════════════════════════════

async fn cmd_save(
    rt: &Runtime, content: &str, cat: &str, pin: bool,
    ttl: Option<&str>, importance: Option<f64>, supersedes: Option<&str>,
) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    let imp = importance.unwrap_or(0.5);

    let expires_at = match ttl {
        Some("session") | Some("permanent") | None => None,
        Some("1h") => Some((chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339()),
        Some("1d") => Some((chrono::Utc::now() + chrono::Duration::days(1)).to_rfc3339()),
        Some("7d") => Some((chrono::Utc::now() + chrono::Duration::days(7)).to_rfc3339()),
        Some("30d") => Some((chrono::Utc::now() + chrono::Duration::days(30)).to_rfc3339()),
        Some(_) => None,
    };

    let fact = models::MemoryFact {
        id: format!("fact_{}", &uuid::Uuid::new_v4().to_string()[..8]),
        content: content.to_string(),
        category: cat.to_string(),
        confidence: 0.9,
        created_at: now.clone(),
        updated_at: now.clone(),
        valid_at: None,
        invalid_at: None,
        source: "cli".to_string(),
        expires_at,
        pinned: pin,
        importance: imp,
        access_count: 0,
        superseded_by: None,
        ttl: ttl.map(|s| s.to_string()),
        session_id: Some(rt.observer.session_id().to_string()),
        embedding: None,
    };

    rt.store.insert_fact(&fact)?;

    // Supersede old fact
    if let Some(old_id) = supersedes {
        rt.store.supersede_fact(old_id, &fact.id)?;
    }

    // Embed
    if let Ok(emb) = rt.embedder.embed_one_async(&format!("[{}] {}", cat, content)).await {
        let _ = rt.store.update_embedding("facts", &fact.id, &emb);
    }

    // Extract entities
    if let Ok(entities) = rt.embedder.extract_entities(content).await {
        rt.observer.link_entities_to_fact(&fact.id, &entities);
    }

    // Compact output
    println!("{}", serde_json::json!({"id": fact.id, "status": "saved"}));
    Ok(())
}

async fn cmd_search(
    rt: &Runtime, query: &str, limit: usize, pinned_only: bool,
    since: Option<&str>, deep: bool,
) -> Result<()> {
    let engine = search::SearchEngine::new(&rt.store, &rt.embedder);
    let results = engine.search(query, limit, None, rt.config.min_similarity, pinned_only, since).await;

    // Increment access count for returned facts
    for r in &results {
        if r.result_type == "fact" {
            let _ = rt.store.increment_access_count(&r.id);
        }
    }

    println!("{}", serde_json::to_string(&results)?);
    Ok(())
}

fn cmd_entity(rt: &Runtime, name: &str, limit: usize) -> Result<()> {
    let results = rt.store.search_entity(name, limit);
    println!("{}", serde_json::to_string(&results)?);
    Ok(())
}

fn cmd_graph(rt: &Runtime, entity: &str) -> Result<()> {
    let nodes = rt.store.graph_query(entity);
    println!("{}", serde_json::to_string(&nodes)?);
    Ok(())
}

fn cmd_context(rt: &Runtime, budget: usize) -> Result<()> {
    let context = rt.store.load_context();
    let (facts, cmds, debug, wf) = rt.store.stats();
    let (entities, edges) = rt.store.entity_stats();

    let mut out = String::new();
    out.push_str(&format!("# Memory: {} facts, {} cmds, {} debug | Graph: {} entities, {} edges\n\n", facts, cmds, debug, entities, edges));

    // Pinned first (most critical)
    let pinned: Vec<_> = context.facts.iter().filter(|f| f.pinned).collect();
    if !pinned.is_empty() {
        out.push_str("## Pinned\n");
        for f in &pinned {
            out.push_str(&format!("- [{}] {} (imp:{:.1})\n", f.category, f.content, f.importance));
        }
        out.push('\n');
    }

    // Active facts by category (skip pinned, already shown)
    let categories = ["gotcha","convention","pattern","failure","insight","environment","preference","debug"];
    for cat in &categories {
        let cat_facts: Vec<_> = context.facts.iter().filter(|f| f.category == *cat && !f.pinned).collect();
        if !cat_facts.is_empty() {
            out.push_str(&format!("## {}\n", format!("{}{}", &cat[..1].to_uppercase(), &cat[1..])));
            for f in &cat_facts {
                out.push_str(&format!("- {} (imp:{:.1})\n", f.content, f.importance));
            }
            out.push('\n');
        }
        // Memory pressure check
        if out.len() > budget * 4 {
            out.push_str(&format!("\n⚠ Memory pressure: output truncated at {}K chars (budget: {}K tokens)\n", out.len()/1000, budget/1000));
            break;
        }
    }

    // Debug fixes
    let fixes: Vec<_> = context.recent_debug.iter().filter(|d| !d.fix.is_empty()).take(5).collect();
    if !fixes.is_empty() {
        out.push_str("## Error→Fix\n");
        for d in fixes {
            out.push_str(&format!("- {} → {}\n", d.error_message, d.fix));
        }
    }

    print!("{}", out);
    Ok(())
}

fn cmd_pin(rt: &Runtime, fact_id: &str) -> Result<()> {
    match rt.store.pin_fact(fact_id)? {
        true => println!("{{\"status\":\"pinned\",\"id\":\"{}\"}}", fact_id),
        false => println!("{{\"status\":\"not_found\",\"id\":\"{}\"}}", fact_id),
    }
    Ok(())
}

fn cmd_unpin(rt: &Runtime, fact_id: &str) -> Result<()> {
    match rt.store.unpin_fact(fact_id)? {
        true => println!("{{\"status\":\"unpinned\",\"id\":\"{}\"}}", fact_id),
        false => println!("{{\"status\":\"not_found\",\"id\":\"{}\"}}", fact_id),
    }
    Ok(())
}

async fn cmd_supersede(rt: &Runtime, old_id: &str, content: &str, cat: &str) -> Result<()> {
    // Save new fact and invalidate old
    cmd_save(rt, content, cat, false, None, None, Some(old_id)).await
}

async fn cmd_consolidate(rt: &Runtime, threshold: f32, dry_run: bool) -> Result<()> {
    let archived = if !dry_run { rt.store.apply_decay()? } else { 0 };
    let ttl_cleaned = if !dry_run { rt.store.cleanup_ttl_facts()? } else { 0 };

    let pairs = rt.store.find_similar_pairs(threshold);
    let mut merged = 0;

    if !dry_run {
        for (a, b, sim) in &pairs {
            if let Ok(merged_content) = rt.embedder.merge_facts(&a.content, &b.content).await {
                let now = chrono::Utc::now().to_rfc3339();
                let fact = models::MemoryFact {
                    id: format!("fact_{}", &uuid::Uuid::new_v4().to_string()[..8]),
                    content: merged_content.clone(),
                    category: a.category.clone(),
                    confidence: a.confidence.max(b.confidence),
                    created_at: now.clone(), updated_at: now, valid_at: None, invalid_at: None,
                    source: format!("merged:{}+{}", a.id, b.id),
                    expires_at: None, pinned: a.pinned || b.pinned,
                    importance: a.importance.max(b.importance),
                    access_count: a.access_count + b.access_count,
                    superseded_by: None, ttl: None,
                    session_id: Some(rt.observer.session_id().to_string()),
                    embedding: None,
                };
                if rt.store.insert_fact(&fact).is_ok() {
                    let _ = rt.store.mark_merged(&a.id, &fact.id);
                    let _ = rt.store.mark_merged(&b.id, &fact.id);
                    merged += 1;
                }
            }
        }
    }

    println!("{}", serde_json::json!({
        "decay_archived": archived,
        "ttl_cleaned": ttl_cleaned,
        "pairs_found": pairs.len(),
        "merged": merged,
        "dry_run": dry_run,
    }));
    Ok(())
}

async fn cmd_reflect(rt: &Runtime, topic: Option<&str>, limit: usize) -> Result<()> {
    let facts = rt.store.load_active_facts();
    let selected: Vec<String> = if let Some(t) = topic {
        let engine = search::SearchEngine::new(&rt.store, &rt.embedder);
        let results = engine.search(t, limit, None, 0.2, false, None).await;
        results.iter().map(|r| r.content.clone()).collect()
    } else {
        facts.iter().take(limit).map(|f| format!("[{}] {}", f.category, f.content)).collect()
    };

    if selected.is_empty() {
        println!("No memories to reflect on.");
        return Ok(());
    }

    match rt.embedder.reflect(&selected, topic).await {
        Ok(insights) => print!("{}", insights),
        Err(e) => eprintln!("Reflect failed: {}", e),
    }
    Ok(())
}

fn cmd_bulk(rt: &Runtime, action: &str, cat: Option<&str>, unpinned: bool, older_than: Option<&str>) -> Result<()> {
    let older_days = older_than.and_then(|s| s.strip_suffix('d')?.parse::<i64>().ok());

    let result = match action {
        "archive" => {
            let count = rt.store.bulk_archive(cat, older_days, unpinned)?;
            serde_json::json!({"action":"archive","count":count})
        }
        "pin" | "unpin" => {
            serde_json::json!({"error":"pin/unpin requires --fact-ids (use mnemonic pin/unpin instead)"})
        }
        _ => serde_json::json!({"error": format!("unknown action: {}", action)}),
    };
    println!("{}", result);
    Ok(())
}

fn cmd_pinned(rt: &Runtime, cat: Option<&str>, limit: usize) -> Result<()> {
    let facts = rt.store.load_pinned_facts(cat, limit);
    let out: Vec<serde_json::Value> = facts.iter().map(|f| serde_json::json!({
        "id": f.id, "content": f.content, "category": f.category,
        "importance": f.importance, "access_count": f.access_count,
    })).collect();
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}

async fn cmd_observe(rt: &Runtime) -> Result<()> {
    use std::io::Read;
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    if let Ok(event) = serde_json::from_str::<models::HookEvent>(&input) {
        rt.observer.process_hook_event(&event).await;
        println!("{{\"status\":\"observed\",\"tool\":\"{}\"}}", event.tool_name);
    } else {
        eprintln!("Failed to parse hook event");
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// MCP server (backwards compatible)
// ═══════════════════════════════════════════════════════════════════════════════

async fn cmd_serve() -> Result<()> {
    // Override log level for serve mode
    let project_root = project::detect_project_root();
    let config = config::MnemonicConfig::load(project_root.as_deref());
    let config = Arc::new(config);

    let db_path = project::memory_db_path();
    info!("Memory DB: {}", db_path.display());

    #[cfg(feature = "sync")]
    let sync_manager = {
        let hash = project_root
            .as_ref()
            .map(|r| project::project_hash(r))
            .unwrap_or_else(|| "default".into());

        match sync::SyncManager::new(&config, &db_path, &hash).await {
            Ok(Some(mgr)) => {
                match mgr.pull().await {
                    Ok(action) => info!("S3 sync pull: {:?}", action),
                    Err(e) => tracing::warn!("S3 sync pull failed: {}", e),
                }
                let mgr = Arc::new(mgr);
                let _bg = mgr.clone().spawn_background();
                Some(mgr)
            }
            Ok(None) => None,
            Err(e) => {
                tracing::warn!("S3 sync init failed: {}", e);
                None
            }
        }
    };

    let store = Arc::new(store::MemoryStore::open(&db_path, &config)?);
    let embedder = Arc::new(embeddings::Embedder::new(&config)?);
    let _ = embedder.embed_all(&store);

    let server = server::MemoryServer::new(store.clone(), embedder.clone(), config.clone());

    info!("mnemonic MCP server starting on stdio");
    let service = server
        .serve(stdio())
        .await
        .inspect_err(|e| tracing::error!("Server error: {:?}", e))?;

    service.waiting().await?;

    #[cfg(feature = "sync")]
    if let Some(ref mgr) = sync_manager {
        mgr.signal_shutdown();
        if let Err(e) = mgr.push().await {
            tracing::warn!("S3 sync final push failed: {}", e);
        }
    }

    Ok(())
}

fn cmd_init() -> Result<()> {
    let project_root = project::detect_project_root();
    let db_path = project::memory_db_path();

    println!("mnemonic init");
    println!("=============\n");

    let existing_config = config::MnemonicConfig::load(project_root.as_deref());
    if existing_config.gemini_key.is_none() {
        println!("  [?] No Gemini API key. Get one at https://aistudio.google.com/apikey");
        print!("      Enter key (or Enter to skip): ");
        std::io::Write::flush(&mut std::io::stdout())?;
        let mut key_input = String::new();
        std::io::stdin().read_line(&mut key_input)?;
        let key_input = key_input.trim();
        if !key_input.is_empty() {
            let path = config::MnemonicConfig::write_global_config(Some(key_input))?;
            println!("  [+] Saved to {}", path.display());
        } else {
            if !config::MnemonicConfig::global_config_path().exists() {
                let path = config::MnemonicConfig::write_global_config(None)?;
                println!("  [+] Created {}", path.display());
            }
            println!("  [!] Set GEMINI_API_KEY env var or edit ~/.mnemonic/config.toml");
        }
    } else {
        println!("  [=] Gemini API key found");
    }

    if let Some(root) = &project_root {
        let project_config_path = root.join(".mnemonic.toml");
        if !project_config_path.exists() {
            let name = root.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or("unnamed".into());
            let path = config::MnemonicConfig::write_project_config(root, &name)?;
            println!("  [+] Created {}", path.display());
        }
    }

    let cfg = config::MnemonicConfig::load(project_root.as_deref());
    let store = store::MemoryStore::open(&db_path, &cfg)?;
    let (facts, cmds, debug, wf) = store.stats();
    let (entities, edges) = store.entity_stats();
    println!("  [+] DB: {} ({} facts, {} entities, {} edges)", db_path.display(), facts, entities, edges);
    println!("      {} commands, {} debug, {} workflows", cmds, debug, wf);
    println!("\nCLI ready. Usage:");
    println!("  mnemonic save 'fact' --cat gotcha --pin");
    println!("  mnemonic search 'query' --limit 5");
    println!("  mnemonic entity comp_renderer");
    println!("  mnemonic context --budget 5000");
    println!("  mnemonic serve  # MCP mode (backwards compatible)");

    Ok(())
}

fn cmd_stats() -> Result<()> {
    let db_path = project::memory_db_path();
    if !db_path.exists() {
        println!("No memory DB. Run `mnemonic init`.");
        return Ok(());
    }
    let project_root = project::detect_project_root();
    let cfg = config::MnemonicConfig::load(project_root.as_deref());
    let store = store::MemoryStore::open(&db_path, &cfg)?;
    let (facts, cmds, debug, wf) = store.stats();
    let (entities, edges) = store.entity_stats();
    let context = store.load_context();

    println!("mnemonic stats");
    println!("====================");
    println!("DB: {}", db_path.display());
    println!("Facts:    {} ({} active)", facts, context.facts.len());
    println!("Entities: {} ({} edges)", entities, edges);
    println!("Commands: {}", cmds);
    println!("Debug:    {} ({} with fixes)", debug, context.recent_debug.iter().filter(|d| !d.fix.is_empty()).count());
    println!("Workflows: {}", wf);
    let pinned_count = context.facts.iter().filter(|f| f.pinned).count();
    println!("Pinned:   {}", pinned_count);
    println!("Key files: {}", context.key_files.len());

    Ok(())
}
