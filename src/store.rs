use anyhow::Result;
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::Mutex;
use tracing::{info, warn};

use crate::config::MnemonicConfig;
use crate::models::*;

pub struct MemoryStore {
    conn: Mutex<Connection>,
    pub max_commands: usize,
    pub max_debug: usize,
    pub max_key_files: usize,
    pub recent_commands: usize,
    pub recent_debug: usize,
    pub decay_rate: f64,
    pub merge_similarity: f32,
}

impl MemoryStore {
    pub fn open(path: &Path, config: &MnemonicConfig) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;

        // WAL mode — concurrent reads, crash-safe
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA cache_size=-8000;",
        )?;

        let store = Self {
            conn: Mutex::new(conn),
            max_commands: config.max_commands,
            max_debug: config.max_debug,
            max_key_files: config.max_key_files,
            recent_commands: config.recent_commands,
            recent_debug: config.recent_debug,
            decay_rate: config.decay_rate,
            merge_similarity: config.merge_similarity,
        };
        store.create_tables()?;
        store.migrate()?;

        // Diagnostic: verify DB is readable
        let (f, c, d, w) = store.stats();
        eprintln!("[store] Opened DB at {:?}: {} facts, {} cmds, {} debug, {} workflows", path, f, c, d, w);

        Ok(store)
    }

    pub fn conn_ref(&self) -> std::sync::MutexGuard<'_, rusqlite::Connection> {
        self.conn.lock().unwrap()
    }

    fn create_tables(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL,
                expires_at TEXT,
                embedding BLOB,
                pinned INTEGER NOT NULL DEFAULT 0,
                strength REAL NOT NULL DEFAULT 1.0,
                superseded_by TEXT,
                ttl TEXT,
                session_id TEXT,
                updated_at TEXT,
                valid_at TEXT,
                invalid_at TEXT,
                importance REAL NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                fact_id, content, category
            );

            CREATE TABLE IF NOT EXISTS commands (
                id TEXT PRIMARY KEY,
                tool TEXT NOT NULL,
                outcome TEXT NOT NULL,
                duration_ms INTEGER,
                result_snippet TEXT,
                session_id TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                embedding BLOB
            );

            CREATE TABLE IF NOT EXISTS debug_records (
                id TEXT PRIMARY KEY,
                error_message TEXT NOT NULL,
                root_cause TEXT NOT NULL,
                fix TEXT NOT NULL,
                tool TEXT,
                session_id TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                embedding BLOB
            );

            CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                steps TEXT NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 1,
                failure_count INTEGER NOT NULL DEFAULT 0,
                last_used TEXT NOT NULL,
                session_id TEXT NOT NULL,
                embedding BLOB
            );

            CREATE TABLE IF NOT EXISTS key_files (
                path TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                touch_count INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS handoff (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS patterns (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS query_cache (
                query TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entity_edges (
                entity_id TEXT NOT NULL,
                fact_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                PRIMARY KEY (entity_id, fact_id, relation),
                FOREIGN KEY (entity_id) REFERENCES entities(id),
                FOREIGN KEY (fact_id) REFERENCES facts(id)
            );

            CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
            CREATE INDEX IF NOT EXISTS idx_facts_expires ON facts(expires_at);
            CREATE INDEX IF NOT EXISTS idx_commands_tool ON commands(tool);
            CREATE INDEX IF NOT EXISTS idx_commands_recorded ON commands(recorded_at);
            CREATE INDEX IF NOT EXISTS idx_debug_recorded ON debug_records(recorded_at);
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
            CREATE INDEX IF NOT EXISTS idx_entity_edges_entity ON entity_edges(entity_id);
            CREATE INDEX IF NOT EXISTS idx_entity_edges_fact ON entity_edges(fact_id);

            CREATE TABLE IF NOT EXISTS code_files (
                path TEXT PRIMARY KEY,
                language TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                symbol_count INTEGER DEFAULT 0,
                line_count INTEGER DEFAULT 0,
                last_parsed TEXT,
                pagerank REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS code_symbols (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                signature TEXT,
                line_start INTEGER,
                line_end INTEGER,
                parent_symbol TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS code_deps (
                from_file TEXT NOT NULL,
                to_file TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                PRIMARY KEY (from_file, to_file, symbol_name)
            );

            CREATE INDEX IF NOT EXISTS idx_code_symbols_file ON code_symbols(file_path);
            CREATE INDEX IF NOT EXISTS idx_code_symbols_name ON code_symbols(name);
            CREATE INDEX IF NOT EXISTS idx_code_deps_from ON code_deps(from_file);
            CREATE INDEX IF NOT EXISTS idx_code_deps_to ON code_deps(to_file);",
        )?;
        Ok(())
    }

    /// Backward-compatible migration: add columns if missing
    fn migrate(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        // Check which columns exist on facts
        let columns: Vec<String> = conn
            .prepare("PRAGMA table_info(facts)")?
            .query_map([], |row| row.get::<_, String>(1))?
            .filter_map(|r| r.ok())
            .collect();

        let migrations = vec![
            ("pinned", "ALTER TABLE facts ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"),
            ("strength", "ALTER TABLE facts ADD COLUMN strength REAL NOT NULL DEFAULT 1.0"),
            ("superseded_by", "ALTER TABLE facts ADD COLUMN superseded_by TEXT"),
            ("ttl", "ALTER TABLE facts ADD COLUMN ttl TEXT"),
            ("session_id", "ALTER TABLE facts ADD COLUMN session_id TEXT"),
            // v2: bi-temporal + importance + access_count
            ("updated_at", "ALTER TABLE facts ADD COLUMN updated_at TEXT"),
            ("valid_at", "ALTER TABLE facts ADD COLUMN valid_at TEXT"),
            ("invalid_at", "ALTER TABLE facts ADD COLUMN invalid_at TEXT"),
            ("importance", "ALTER TABLE facts ADD COLUMN importance REAL NOT NULL DEFAULT 0.5"),
            ("access_count", "ALTER TABLE facts ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0"),
        ];

        for (col, sql) in migrations {
            if !columns.contains(&col.to_string()) {
                match conn.execute_batch(sql) {
                    Ok(_) => info!("Migration: added column '{}' to facts", col),
                    Err(e) => warn!("Migration '{}' failed (may already exist): {}", col, e),
                }
            }
        }

        // Migrate superseded_by → invalid_at for existing data
        let _ = conn.execute(
            "UPDATE facts SET invalid_at = COALESCE(updated_at, created_at) WHERE superseded_by IS NOT NULL AND invalid_at IS NULL",
            [],
        );

        // Backfill updated_at from created_at
        let _ = conn.execute(
            "UPDATE facts SET updated_at = created_at WHERE updated_at IS NULL",
            [],
        );

        // Create indexes that depend on migrated columns
        let _ = conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_facts_pinned ON facts(pinned);
            CREATE INDEX IF NOT EXISTS idx_facts_superseded ON facts(superseded_by);"
        );

        // Entity alias table
        let _ = conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS entity_aliases (
                entity_id TEXT NOT NULL,
                alias TEXT NOT NULL,
                PRIMARY KEY (entity_id, alias),
                FOREIGN KEY (entity_id) REFERENCES entities(id)
            );
            CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias ON entity_aliases(alias);
            CREATE INDEX IF NOT EXISTS idx_facts_importance ON facts(importance);
            CREATE INDEX IF NOT EXISTS idx_facts_access ON facts(access_count);
            CREATE INDEX IF NOT EXISTS idx_facts_invalid ON facts(invalid_at);"
        );

        Ok(())
    }

    // ─── Insert Operations ──────────────────────────────────────────────────

    pub fn insert_fact(&self, fact: &MemoryFact) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO facts (id, content, category, confidence, created_at, updated_at, valid_at, invalid_at, source, expires_at, embedding, pinned, strength, superseded_by, ttl, session_id, importance, access_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
            params![
                fact.id,
                fact.content,
                fact.category,
                fact.confidence,
                fact.created_at,
                fact.updated_at,
                fact.valid_at,
                fact.invalid_at,
                fact.source,
                fact.expires_at,
                fact.embedding.as_ref().map(|e| floats_to_blob(e)),
                fact.pinned as i32,
                1.0f64, // strength — kept for backward compat
                fact.superseded_by,
                fact.ttl,
                fact.session_id,
                fact.importance,
                fact.access_count,
            ],
        )?;
        // Insert into FTS index (ignore duplicates)
        let _ = conn.execute(
            "INSERT OR IGNORE INTO facts_fts (fact_id, content, category) VALUES (?1, ?2, ?3)",
            params![fact.id, fact.content, fact.category],
        );
        Ok(())
    }

    pub fn insert_command(&self, cmd: &CommandRecord) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO commands (id, tool, outcome, duration_ms, result_snippet, session_id, recorded_at, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                cmd.id,
                cmd.tool,
                cmd.outcome,
                cmd.duration_ms,
                cmd.result_snippet,
                cmd.session_id,
                cmd.recorded_at,
                cmd.embedding.as_ref().map(|e| floats_to_blob(e)),
            ],
        )?;
        // Rolling cap
        conn.execute(
            &format!("DELETE FROM commands WHERE id NOT IN (SELECT id FROM commands ORDER BY recorded_at DESC LIMIT {})", self.max_commands),
            [],
        )?;
        Ok(())
    }

    pub fn insert_debug_record(&self, dbg: &DebugRecord) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO debug_records (id, error_message, root_cause, fix, tool, session_id, recorded_at, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                dbg.id,
                dbg.error_message,
                dbg.root_cause,
                dbg.fix,
                dbg.tool,
                dbg.session_id,
                dbg.recorded_at,
                dbg.embedding.as_ref().map(|e| floats_to_blob(e)),
            ],
        )?;
        // Rolling cap
        conn.execute(
            &format!("DELETE FROM debug_records WHERE id NOT IN (SELECT id FROM debug_records ORDER BY recorded_at DESC LIMIT {})", self.max_debug),
            [],
        )?;
        Ok(())
    }

    pub fn insert_workflow(&self, wf: &WorkflowPattern) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let steps_json = serde_json::to_string(&wf.steps)?;
        conn.execute(
            "INSERT OR REPLACE INTO workflows (id, name, steps, success_count, failure_count, last_used, session_id, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                wf.id,
                wf.name,
                steps_json,
                wf.success_count,
                wf.failure_count,
                wf.last_used,
                wf.session_id,
                wf.embedding.as_ref().map(|e| floats_to_blob(e)),
            ],
        )?;
        Ok(())
    }

    pub fn bump_key_file(&self, path: &str, is_write: bool) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let increment = if is_write { 2 } else { 1 };

        let existing: Option<i32> = conn
            .query_row(
                "SELECT touch_count FROM key_files WHERE path = ?1",
                params![path],
                |row| row.get(0),
            )
            .ok();

        if let Some(count) = existing {
            conn.execute(
                "UPDATE key_files SET touch_count = ?1 WHERE path = ?2",
                params![count + increment, path],
            )?;
        } else if is_write {
            let desc = Path::new(path)
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_default();
            conn.execute(
                "INSERT INTO key_files (path, description, touch_count) VALUES (?1, ?2, ?3)",
                params![path, desc, 1],
            )?;
            conn.execute(
                &format!("DELETE FROM key_files WHERE path NOT IN (SELECT path FROM key_files ORDER BY touch_count DESC LIMIT {})", self.max_key_files),
                [],
            )?;
        }
        Ok(())
    }

    pub fn update_handoff(&self, last_commit: &str, uncommitted: &[String]) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        let files_json = serde_json::to_string(uncommitted)?;

        conn.execute(
            "INSERT OR REPLACE INTO handoff (key, value) VALUES ('last_commit', ?1)",
            params![last_commit],
        )?;
        conn.execute(
            "INSERT OR REPLACE INTO handoff (key, value) VALUES ('uncommitted_files', ?1)",
            params![files_json],
        )?;
        conn.execute(
            "INSERT OR REPLACE INTO handoff (key, value) VALUES ('updated_at', ?1)",
            params![now],
        )?;
        Ok(())
    }

    pub fn update_embedding(&self, table: &str, id: &str, embedding: &[f32]) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let blob = floats_to_blob(embedding);
        conn.execute(
            &format!("UPDATE {} SET embedding = ?1 WHERE id = ?2", table),
            params![blob, id],
        )?;
        Ok(())
    }

    pub fn cache_query_embedding(&self, query: &str, embedding: &[f32]) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT OR REPLACE INTO query_cache (query, embedding, created_at) VALUES (?1, ?2, ?3)",
            params![query, floats_to_blob(embedding), now],
        )?;
        Ok(())
    }

    // ─── Pin / Unpin ────────────────────────────────────────────────────────

    pub fn pin_fact(&self, fact_id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let rows = conn.execute(
            "UPDATE facts SET pinned = 1, strength = MAX(strength, 1.0) WHERE id = ?1",
            params![fact_id],
        )?;
        Ok(rows > 0)
    }

    pub fn unpin_fact(&self, fact_id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let rows = conn.execute(
            "UPDATE facts SET pinned = 0 WHERE id = ?1",
            params![fact_id],
        )?;
        Ok(rows > 0)
    }

    pub fn load_pinned_facts(&self, category: Option<&str>, limit: usize) -> Vec<MemoryFact> {
        let conn = self.conn.lock().unwrap();

        if let Some(cat) = category {
            let mut stmt = conn.prepare(
                "SELECT id, content, category, confidence, created_at, source, expires_at, embedding, pinned, strength, superseded_by, ttl, session_id, updated_at, valid_at, invalid_at, importance, access_count
                 FROM facts WHERE pinned = 1 AND category = ?1 ORDER BY created_at DESC LIMIT ?2"
            ).unwrap();
            stmt.query_map(params![cat, limit as i64], |row| fact_from_row(row))
                .unwrap()
                .filter_map(|r| r.ok())
                .collect()
        } else {
            let mut stmt = conn.prepare(
                "SELECT id, content, category, confidence, created_at, source, expires_at, embedding, pinned, strength, superseded_by, ttl, session_id, updated_at, valid_at, invalid_at, importance, access_count
                 FROM facts WHERE pinned = 1 ORDER BY created_at DESC LIMIT ?1"
            ).unwrap();
            stmt.query_map(params![limit as i64], |row| fact_from_row(row))
                .unwrap()
                .filter_map(|r| r.ok())
                .collect()
        }
    }

    // ─── Supersede ──────────────────────────────────────────────────────────

    /// Bi-temporal invalidation (Zep pattern): mark old fact as no longer valid
    pub fn supersede_fact(&self, old_id: &str, new_id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        let rows = conn.execute(
            "UPDATE facts SET superseded_by = ?1, invalid_at = ?2 WHERE id = ?3 AND invalid_at IS NULL",
            params![new_id, now, old_id],
        )?;
        Ok(rows > 0)
    }

    /// Bi-temporal invalidation without replacement
    pub fn invalidate_fact(&self, fact_id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        let rows = conn.execute(
            "UPDATE facts SET invalid_at = ?1 WHERE id = ?2 AND invalid_at IS NULL",
            params![now, fact_id],
        )?;
        Ok(rows > 0)
    }

    // ─── Strength / Decay ───────────────────────────────────────────────────

    pub fn boost_strength(&self, fact_id: &str, boost: f64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE facts SET strength = MIN(strength + ?1, 2.0) WHERE id = ?2",
            params![boost, fact_id],
        )?;
        Ok(())
    }

    pub fn apply_decay(&self) -> Result<usize> {
        // Ideal decay: importance × exp(-λ × age / (1 + α × ln(1 + access)))
        // We compute decay scores in Rust and archive facts below floor
        let facts = self.load_active_facts();
        let conn = self.conn.lock().unwrap();
        let mut archived = 0;
        for fact in &facts {
            if fact.pinned { continue; }
            let score = fact.decay_score();
            if score < self.decay_rate * 0.05 { // below effective floor
                conn.execute(
                    "UPDATE facts SET invalid_at = ?1 WHERE id = ?2 AND invalid_at IS NULL",
                    params![chrono::Utc::now().to_rfc3339(), fact.id],
                )?;
                archived += 1;
            }
        }
        Ok(archived)
    }

    pub fn increment_access_count(&self, fact_id: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE facts SET access_count = access_count + 1 WHERE id = ?1",
            params![fact_id],
        )?;
        Ok(())
    }

    // ─── Entity Alias Table ────────────────────────────────────────────────

    pub fn add_entity_alias(&self, entity_id: &str, alias: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO entity_aliases (entity_id, alias) VALUES (?1, lower(?2))",
            params![entity_id, alias],
        )?;
        Ok(())
    }

    pub fn resolve_entity_by_alias(&self, name: &str) -> Option<String> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT entity_id FROM entity_aliases WHERE lower(alias) = lower(?1)",
            params![name],
            |row| row.get(0),
        ).ok()
    }

    // ─── Entity Graph ───────────────────────────────────────────────────────

    pub fn insert_entity(&self, entity: &Entity) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO entities (id, name, entity_type, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![entity.id, entity.name, entity.entity_type, entity.created_at],
        )?;
        Ok(())
    }

    pub fn insert_entity_edge(&self, edge: &EntityEdge) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO entity_edges (entity_id, fact_id, relation) VALUES (?1, ?2, ?3)",
            params![edge.entity_id, edge.fact_id, edge.relation],
        )?;
        Ok(())
    }

    /// Find entity by name (case-insensitive, partial match) + alias resolution
    pub fn find_entities(&self, name: &str) -> Vec<Entity> {
        let conn = self.conn.lock().unwrap();
        let pattern = format!("%{}%", name.to_lowercase());

        // Search by name
        let mut results: Vec<Entity> = Vec::new();
        if let Ok(mut stmt) = conn.prepare(
            "SELECT id, name, entity_type, created_at FROM entities WHERE lower(name) LIKE ?1 ORDER BY name LIMIT 50"
        ) {
            results.extend(
                stmt.query_map(params![pattern], |row| Ok(Entity {
                    id: row.get(0)?, name: row.get(1)?, entity_type: row.get(2)?, created_at: row.get(3)?,
                })).unwrap().filter_map(|r| r.ok())
            );
        }

        // Also search by alias (Zep pattern: alias resolution)
        if let Ok(mut stmt) = conn.prepare(
            "SELECT e.id, e.name, e.entity_type, e.created_at FROM entities e
             JOIN entity_aliases a ON a.entity_id = e.id
             WHERE lower(a.alias) LIKE ?1 LIMIT 20"
        ) {
            let seen: std::collections::HashSet<String> = results.iter().map(|e| e.id.clone()).collect();
            for ent in stmt.query_map(params![pattern], |row| Ok(Entity {
                id: row.get(0)?, name: row.get(1)?, entity_type: row.get(2)?, created_at: row.get(3)?,
            })).unwrap().filter_map(|r| r.ok()) {
                if !seen.contains(&ent.id) {
                    results.push(ent);
                }
            }
        }

        results
    }

    /// Find or create an entity by name and type
    pub fn find_or_create_entity(&self, name: &str, entity_type: &str) -> Result<String> {
        let conn = self.conn.lock().unwrap();

        // Try exact match first
        let existing: Option<String> = conn
            .query_row(
                "SELECT id FROM entities WHERE lower(name) = lower(?1) AND entity_type = ?2",
                params![name, entity_type],
                |row| row.get(0),
            )
            .ok();

        if let Some(id) = existing {
            return Ok(id);
        }

        // Create new entity
        let id = format!("ent_{}", &uuid::Uuid::new_v4().to_string()[..8]);
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO entities (id, name, entity_type, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![id, name, entity_type, now],
        )?;
        Ok(id)
    }

    /// Get all facts connected to an entity
    pub fn get_entity_facts(&self, entity_id: &str) -> Vec<(MemoryFact, String)> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT f.id, f.content, f.category, f.confidence, f.created_at, f.source, f.expires_at, f.embedding, f.pinned, f.strength, f.superseded_by, f.ttl, f.session_id, f.updated_at, f.valid_at, f.invalid_at, f.importance, f.access_count, ee.relation
                 FROM facts f
                 JOIN entity_edges ee ON ee.fact_id = f.id
                 WHERE ee.entity_id = ?1 AND f.superseded_by IS NULL AND f.invalid_at IS NULL
                 ORDER BY f.created_at DESC",
            )
            .unwrap();
        stmt.query_map(params![entity_id], |row| {
            let fact = fact_from_row(row)?;
            let relation: String = row.get(18)?;
            Ok((fact, relation))
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect()
    }

    /// Get all entities connected to a given entity (via shared facts)
    pub fn get_connected_entities(&self, entity_id: &str) -> Vec<(Entity, String)> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT DISTINCT e.id, e.name, e.entity_type, e.created_at, ee2.relation
                 FROM entity_edges ee1
                 JOIN entity_edges ee2 ON ee1.fact_id = ee2.fact_id
                 JOIN entities e ON e.id = ee2.entity_id
                 WHERE ee1.entity_id = ?1 AND ee2.entity_id != ?1
                 LIMIT 50",
            )
            .unwrap();
        stmt.query_map(params![entity_id], |row| {
            Ok((
                Entity {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    entity_type: row.get(2)?,
                    created_at: row.get(3)?,
                },
                row.get::<_, String>(4)?,
            ))
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect()
    }

    /// Full graph query: entity + facts + connected entities
    pub fn graph_query(&self, name: &str) -> Vec<GraphNode> {
        let entities = self.find_entities(name);
        let mut nodes = Vec::new();

        for entity in entities {
            let fact_pairs = self.get_entity_facts(&entity.id);
            let connected = self.get_connected_entities(&entity.id);

            let facts: Vec<MemoryFact> = fact_pairs.iter().map(|(f, _)| f.clone()).collect();
            let connected_names: Vec<String> = connected.iter().map(|(e, _)| e.name.clone()).collect();

            // Group facts by relation type
            let mut relations: std::collections::HashMap<String, Vec<String>> =
                std::collections::HashMap::new();
            for (fact, relation) in &fact_pairs {
                relations
                    .entry(relation.clone())
                    .or_default()
                    .push(fact.content.clone());
            }

            nodes.push(GraphNode {
                entity,
                facts,
                connected_entities: connected_names,
                relations,
            });
        }

        nodes
    }

    // ─── Search Entity (combines entity graph + FTS) ────────────────────────

    pub fn search_entity(&self, name: &str, limit: usize) -> Vec<SearchResult> {
        let mut results = Vec::new();

        // 1. Entity graph search
        let entities = self.find_entities(name);
        for entity in &entities {
            let fact_pairs = self.get_entity_facts(&entity.id);
            for (fact, relation) in fact_pairs {
                if fact.is_active() {
                    results.push(SearchResult {
                        result_type: "fact".into(),
                        id: fact.id.clone(),
                        content: fact.content.clone(),
                        category: Some(fact.category.clone()),
                        similarity: 0.95, // graph match = high relevance
                        metadata: Some(serde_json::json!({
                            "entity": entity.name,
                            "entity_type": entity.entity_type,
                            "relation": relation,
                            "confidence": fact.confidence,
                            "pinned": fact.pinned,
                            "importance": format!("{:.2}", fact.importance),
                        })),
                    });
                }
            }
        }

        // 2. Fall back to FTS for terms not in entity graph
        if results.is_empty() {
            let fts_results = self.fts_search(name, limit);
            results.extend(fts_results);
        }

        // Dedup by fact ID
        let mut seen = std::collections::HashSet::new();
        results.retain(|r| seen.insert(r.id.clone()));

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(limit);
        results
    }

    // ─── Bulk Operations ────────────────────────────────────────────────────

    pub fn bulk_archive(&self, category: Option<&str>, older_than_days: Option<i64>, unpinned_only: bool) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let mut conditions = vec!["superseded_by IS NULL".to_string(), "invalid_at IS NULL".to_string()];

        if unpinned_only {
            conditions.push("pinned = 0".to_string());
        }
        if let Some(cat) = category {
            conditions.push(format!("category = '{}'", cat.replace('\'', "''")));
        }
        if let Some(days) = older_than_days {
            let cutoff = (chrono::Utc::now() - chrono::Duration::days(days)).to_rfc3339();
            conditions.push(format!("created_at < '{}'", cutoff));
        }

        let sql = format!(
            "UPDATE facts SET superseded_by = 'archived:bulk' WHERE {}",
            conditions.join(" AND ")
        );
        let count = conn.execute(&sql, [])?;
        Ok(count)
    }

    pub fn bulk_delete(&self, fact_ids: &[String]) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let mut count = 0;
        for id in fact_ids {
            count += conn.execute("DELETE FROM facts WHERE id = ?1", params![id])?;
            let _ = conn.execute("DELETE FROM facts_fts WHERE fact_id = ?1", params![id]);
            let _ = conn.execute("DELETE FROM entity_edges WHERE fact_id = ?1", params![id]);
        }
        Ok(count)
    }

    pub fn bulk_pin(&self, fact_ids: &[String]) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let mut count = 0;
        for id in fact_ids {
            count += conn.execute(
                "UPDATE facts SET pinned = 1, strength = MAX(strength, 1.0) WHERE id = ?1",
                params![id],
            )?;
        }
        Ok(count)
    }

    pub fn bulk_unpin_by_query(&self, query: &str) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let pattern = format!("%{}%", query.to_lowercase());
        let count = conn.execute(
            "UPDATE facts SET pinned = 0 WHERE lower(content) LIKE ?1 AND pinned = 1",
            params![pattern],
        )?;
        Ok(count)
    }

    /// Clean up session-specific facts
    pub fn cleanup_session_facts(&self, session_id: &str) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let count = conn.execute(
            "UPDATE facts SET superseded_by = 'expired:session' WHERE session_id = ?1 AND ttl = 'session' AND pinned = 0",
            params![session_id],
        )?;
        Ok(count)
    }

    /// Clean up TTL-expired facts
    pub fn cleanup_ttl_facts(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let now = chrono::Utc::now();
        let mut total = 0;

        // Get all facts with TTL set
        let facts: Vec<(String, String, String)> = {
            let mut stmt = conn
                .prepare("SELECT id, ttl, created_at FROM facts WHERE ttl IS NOT NULL AND ttl != 'permanent' AND ttl != 'session' AND pinned = 0 AND superseded_by IS NULL AND invalid_at IS NULL")
                .unwrap();
            stmt.query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .unwrap()
            .filter_map(|r| r.ok())
            .collect()
        };

        for (id, ttl, created_at) in facts {
            if let Ok(created) = chrono::DateTime::parse_from_rfc3339(&created_at) {
                let duration = parse_ttl_duration(&ttl);
                if let Some(dur) = duration {
                    if created + dur < now {
                        conn.execute(
                            "UPDATE facts SET superseded_by = 'expired:ttl' WHERE id = ?1",
                            params![id],
                        )?;
                        total += 1;
                    }
                }
            }
        }
        Ok(total)
    }

    // ─── Consolidation ──────────────────────────────────────────────────────

    /// Find pairs of facts with high embedding similarity for potential merging
    pub fn find_similar_pairs(&self, threshold: f32) -> Vec<(MemoryFact, MemoryFact, f32)> {
        let facts = self.load_active_facts();
        let mut pairs = Vec::new();

        for i in 0..facts.len() {
            for j in (i + 1)..facts.len() {
                if let (Some(ref emb_a), Some(ref emb_b)) = (&facts[i].embedding, &facts[j].embedding) {
                    let sim = cosine_similarity(emb_a, emb_b);
                    if sim >= threshold {
                        pairs.push((facts[i].clone(), facts[j].clone(), sim));
                    }
                }
            }
        }

        pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        pairs
    }

    /// Mark a fact as superseded by another (used during consolidation merge)
    pub fn mark_merged(&self, old_id: &str, merged_into_id: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE facts SET superseded_by = ?1 WHERE id = ?2",
            params![format!("merged:{}", merged_into_id), old_id],
        )?;
        Ok(())
    }

    // ─── Load Operations ────────────────────────────────────────────────────

    pub fn load_facts(&self) -> Vec<MemoryFact> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT id, content, category, confidence, created_at, source, expires_at, embedding, pinned, strength, superseded_by, ttl, session_id, updated_at, valid_at, invalid_at, importance, access_count FROM facts")
            .unwrap();
        stmt.query_map([], |row| fact_from_row(row))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect()
    }

    /// Load only active (non-expired, non-superseded) facts
    pub fn load_active_facts(&self) -> Vec<MemoryFact> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT id, content, category, confidence, created_at, source, expires_at, embedding, pinned, strength, superseded_by, ttl, session_id, updated_at, valid_at, invalid_at, importance, access_count FROM facts WHERE invalid_at IS NULL AND superseded_by IS NULL")
            .unwrap();
        stmt.query_map([], |row| fact_from_row(row))
            .unwrap()
            .filter_map(|r| r.ok())
            .filter(|f| f.is_active())
            .collect()
    }

    /// Load facts with combined filters
    pub fn load_facts_filtered(
        &self,
        pinned_only: bool,
        since: Option<&str>,
        category: Option<&str>,
        limit: usize,
    ) -> Vec<MemoryFact> {
        let conn = self.conn.lock().unwrap();
        let mut conditions = vec!["superseded_by IS NULL".to_string(), "invalid_at IS NULL".to_string()];

        if pinned_only {
            conditions.push("pinned = 1".to_string());
        }
        if let Some(since_date) = since {
            conditions.push(format!("created_at >= '{}'", since_date.replace('\'', "''")));
        }
        if let Some(cat) = category {
            conditions.push(format!("category = '{}'", cat.replace('\'', "''")));
        }

        let sql = format!(
            "SELECT id, content, category, confidence, created_at, source, expires_at, embedding, pinned, strength, superseded_by, ttl, session_id, updated_at, valid_at, invalid_at, importance, access_count FROM facts WHERE {} ORDER BY created_at DESC LIMIT {}",
            conditions.join(" AND "),
            limit
        );

        let mut stmt = conn.prepare(&sql).unwrap();
        stmt.query_map([], |row| fact_from_row(row))
            .unwrap()
            .filter_map(|r| r.ok())
            .filter(|f| f.is_active())
            .collect()
    }

    pub fn load_commands(&self, limit: usize) -> Vec<CommandRecord> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT id, tool, outcome, duration_ms, result_snippet, session_id, recorded_at, embedding FROM commands ORDER BY recorded_at DESC LIMIT ?1")
            .unwrap();
        stmt.query_map(params![limit as i64], |row| {
            Ok(CommandRecord {
                id: row.get(0)?,
                tool: row.get(1)?,
                outcome: row.get(2)?,
                duration_ms: row.get(3)?,
                result_snippet: row.get(4)?,
                session_id: row.get(5)?,
                recorded_at: row.get(6)?,
                embedding: row.get::<_, Option<Vec<u8>>>(7)?.map(|b| blob_to_floats(&b)),
            })
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect()
    }

    pub fn load_debug_records(&self, limit: usize) -> Vec<DebugRecord> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT id, error_message, root_cause, fix, tool, session_id, recorded_at, embedding FROM debug_records ORDER BY recorded_at DESC LIMIT ?1")
            .unwrap();
        stmt.query_map(params![limit as i64], |row| {
            Ok(DebugRecord {
                id: row.get(0)?,
                error_message: row.get(1)?,
                root_cause: row.get(2)?,
                fix: row.get(3)?,
                tool: row.get(4)?,
                session_id: row.get(5)?,
                recorded_at: row.get(6)?,
                embedding: row.get::<_, Option<Vec<u8>>>(7)?.map(|b| blob_to_floats(&b)),
            })
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect()
    }

    pub fn load_key_files(&self) -> Vec<KeyFile> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT path, description, touch_count FROM key_files ORDER BY touch_count DESC")
            .unwrap();
        stmt.query_map([], |row| {
            Ok(KeyFile {
                path: row.get(0)?,
                description: row.get(1)?,
                touch_count: row.get(2)?,
            })
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect()
    }

    pub fn load_handoff(&self) -> Handoff {
        let conn = self.conn.lock().unwrap();
        let mut kv = std::collections::HashMap::new();
        if let Ok(mut stmt) = conn.prepare("SELECT key, value FROM handoff") {
            if let Ok(rows) = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            }) {
                for row in rows.flatten() {
                    kv.insert(row.0, row.1);
                }
            }
        }
        let uncommitted: Vec<String> = kv
            .get("uncommitted_files")
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();

        Handoff {
            last_commit: kv.get("last_commit").cloned().unwrap_or_default(),
            uncommitted_files: uncommitted,
            next_task: kv.get("next_task").cloned().unwrap_or_default(),
            blocked_on: kv.get("blocked_on").cloned().unwrap_or_default(),
            updated_at: kv.get("updated_at").cloned().unwrap_or_default(),
        }
    }

    pub fn load_patterns(&self) -> Patterns {
        let conn = self.conn.lock().unwrap();
        let mut kv = std::collections::HashMap::new();
        if let Ok(mut stmt) = conn.prepare("SELECT key, value FROM patterns") {
            if let Ok(rows) = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            }) {
                for row in rows.flatten() {
                    kv.insert(row.0, row.1);
                }
            }
        }
        Patterns {
            work_context: kv.get("work_context").cloned().unwrap_or_default(),
            conventions: kv.get("conventions").cloned().unwrap_or_default(),
            toolchain: kv.get("toolchain").cloned().unwrap_or_default(),
        }
    }

    pub fn load_workflows(&self) -> Vec<WorkflowPattern> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT id, name, steps, success_count, failure_count, last_used, session_id, embedding FROM workflows ORDER BY last_used DESC")
            .unwrap();
        stmt.query_map([], |row| {
            let steps_json: String = row.get(2)?;
            let steps: Vec<String> = serde_json::from_str(&steps_json).unwrap_or_default();
            Ok(WorkflowPattern {
                id: row.get(0)?,
                name: row.get(1)?,
                steps,
                success_count: row.get(3)?,
                failure_count: row.get(4)?,
                last_used: row.get(5)?,
                session_id: row.get(6)?,
                embedding: row.get::<_, Option<Vec<u8>>>(7)?.map(|b| blob_to_floats(&b)),
            })
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect()
    }

    pub fn load_context(&self) -> MemoryContext {
        let facts: Vec<MemoryFact> = self
            .load_active_facts();
        MemoryContext {
            facts,
            patterns: self.load_patterns(),
            key_files: self.load_key_files(),
            handoff: self.load_handoff(),
            recent_commands: self.load_commands(self.recent_commands),
            recent_debug: self.load_debug_records(self.recent_debug),
            workflows: self.load_workflows(),
        }
    }

    // ─── FTS5 Search (Tier 1: ~0.1ms) ───────────────────────────────────────

    pub fn fts_search(&self, query: &str, limit: usize) -> Vec<SearchResult> {
        let conn = self.conn.lock().unwrap();
        let words: Vec<String> = query
            .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
            .filter(|w| !w.is_empty())
            .map(|w| format!("\"{}\"", w))
            .collect();

        if words.is_empty() {
            return vec![];
        }

        let fts_query = words.join(" OR ");
        let mut results = Vec::new();

        // Search facts via FTS5 — join with facts table for full data + proper similarity
        if let Ok(mut stmt) = conn.prepare(
            "SELECT f.id, f.content, f.category, fts.rank, f.pinned, f.importance, f.confidence, f.access_count, f.updated_at
             FROM facts_fts fts
             JOIN facts f ON f.id = fts.fact_id
             WHERE facts_fts MATCH ?1
             AND f.invalid_at IS NULL AND f.superseded_by IS NULL
             ORDER BY fts.rank
             LIMIT ?2",
        ) {
            if let Ok(rows) = stmt.query_map(params![fts_query, limit as i64], |row| {
                let rank: f64 = row.get(3)?;
                let pinned: bool = row.get::<_, i32>(4)? != 0;
                let importance: f64 = row.get(5)?;
                let confidence: f64 = row.get(6)?;
                let access_count: u32 = row.get::<_, u32>(7).unwrap_or(0);
                let updated_at: String = row.get::<_, Option<String>>(8)?.unwrap_or_default();
                // FTS5 rank is negative BM25 (more negative = better match)
                // Normalize to 0.0-1.0: best match (-15) → 1.0, worst (-0.1) → 0.1
                let base_sim = (-rank / 15.0).clamp(0.05, 1.0);
                // Decay from age + access reinforcement
                let age_days = if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&updated_at) {
                    (chrono::Utc::now() - dt.with_timezone(&chrono::Utc)).num_hours() as f64 / 24.0
                } else { 0.0 };
                let reinforced = 1.0 + 0.3 * (1.0 + access_count as f64).ln();
                let decay = (-0.16 * age_days / reinforced).exp().max(0.1);
                // Composite: 0.5 × relevance + 0.3 × recency + 0.2 × importance
                let sim = (0.5 * base_sim + 0.3 * decay + 0.2 * importance).clamp(0.0, 1.0) as f32;
                Ok(SearchResult {
                    result_type: "fact".into(),
                    id: row.get(0)?,
                    content: row.get(1)?,
                    category: Some(row.get::<_, String>(2)?),
                    similarity: sim,
                    metadata: Some(serde_json::json!({
                        "source": "fts5",
                        "pinned": pinned,
                        "importance": format!("{:.2}", importance),
                        "decay": format!("{:.3}", decay),
                    })),
                })
            }) {
                results.extend(rows.flatten());
            }
        }

        // Search commands via LIKE
        let pattern = format!("%{}%", query.to_lowercase());
        if let Ok(mut stmt) = conn.prepare(
            "SELECT id, tool, outcome, result_snippet FROM commands
             WHERE lower(tool) LIKE ?1 OR lower(outcome) LIKE ?1
             LIMIT ?2",
        ) {
            if let Ok(rows) = stmt.query_map(params![pattern, limit as i64], |row| {
                let tool: String = row.get(1)?;
                let outcome: String = row.get(2)?;
                let snippet: Option<String> = row.get(3)?;
                let content = format!(
                    "{} → {}{}",
                    tool,
                    outcome,
                    snippet.map(|s| format!(" | {}", s)).unwrap_or_default()
                );
                Ok(SearchResult {
                    result_type: "command".into(),
                    id: row.get(0)?,
                    content,
                    category: None,
                    similarity: 0.5,
                    metadata: Some(serde_json::json!({"tool": tool, "outcome": outcome})),
                })
            }) {
                results.extend(rows.flatten());
            }
        }

        // Search debug records via LIKE
        if let Ok(mut stmt) = conn.prepare(
            "SELECT id, error_message, fix FROM debug_records
             WHERE lower(error_message) LIKE ?1 OR lower(fix) LIKE ?1
             LIMIT ?2",
        ) {
            if let Ok(rows) = stmt.query_map(params![pattern, limit as i64], |row| {
                let error: String = row.get(1)?;
                let fix: String = row.get(2)?;
                let content = if fix.is_empty() {
                    format!("Error: {}", error)
                } else {
                    format!("Error: {} → Fix: {}", error, fix)
                };
                Ok(SearchResult {
                    result_type: "debug".into(),
                    id: row.get(0)?,
                    content,
                    category: None,
                    similarity: 0.5,
                    metadata: Some(serde_json::json!({"error": error, "fix": fix})),
                })
            }) {
                results.extend(rows.flatten());
            }
        }

        results
    }

    // ─── Vector Data (for Tier 2/3 search) ──────────────────────────────────

    pub fn load_all_embeddings(&self) -> Vec<(String, String, Vec<f32>)> {
        let conn = self.conn.lock().unwrap();
        let mut results = Vec::new();

        for (table, item_type) in &[
            ("facts", "fact"),
            ("commands", "command"),
            ("debug_records", "debug"),
            ("workflows", "workflow"),
        ] {
            let sql = if *table == "facts" {
                format!(
                    "SELECT id, embedding FROM {} WHERE embedding IS NOT NULL AND superseded_by IS NULL AND invalid_at IS NULL",
                    table
                )
            } else {
                format!(
                    "SELECT id, embedding FROM {} WHERE embedding IS NOT NULL",
                    table
                )
            };
            if let Ok(mut stmt) = conn.prepare(&sql) {
                if let Ok(rows) = stmt.query_map([], |row| {
                    let id: String = row.get(0)?;
                    let blob: Vec<u8> = row.get(1)?;
                    Ok((id, item_type.to_string(), blob_to_floats(&blob)))
                }) {
                    results.extend(rows.flatten());
                }
            }
        }
        results
    }

    pub fn load_cached_query_embedding(&self, query: &str) -> Option<Vec<f32>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT embedding FROM query_cache WHERE query = ?1",
            params![query],
            |row| {
                let blob: Vec<u8> = row.get(0)?;
                Ok(blob_to_floats(&blob))
            },
        )
        .ok()
    }

    /// Get items that need embeddings (no embedding yet)
    pub fn load_unembedded(&self) -> Vec<EmbeddingItem> {
        let conn = self.conn.lock().unwrap();
        let mut items = Vec::new();

        if let Ok(mut stmt) =
            conn.prepare("SELECT id, content, category FROM facts WHERE embedding IS NULL AND superseded_by IS NULL AND invalid_at IS NULL")
        {
            if let Ok(rows) = stmt.query_map([], |row| {
                let id: String = row.get(0)?;
                let content: String = row.get(1)?;
                let category: String = row.get(2)?;
                Ok(EmbeddingItem {
                    id,
                    item_type: "facts".into(),
                    text: format!("[{}] {}", category, content),
                })
            }) {
                items.extend(rows.flatten());
            }
        }

        if let Ok(mut stmt) = conn.prepare(
            "SELECT id, tool, outcome, result_snippet FROM commands WHERE embedding IS NULL",
        ) {
            if let Ok(rows) = stmt.query_map([], |row| {
                let id: String = row.get(0)?;
                let tool: String = row.get(1)?;
                let outcome: String = row.get(2)?;
                let snippet: Option<String> = row.get(3)?;
                Ok(EmbeddingItem {
                    id,
                    item_type: "commands".into(),
                    text: format!(
                        "{} → {}{}",
                        tool,
                        outcome,
                        snippet.map(|s| format!(" | {}", s)).unwrap_or_default()
                    ),
                })
            }) {
                items.extend(rows.flatten());
            }
        }

        if let Ok(mut stmt) = conn.prepare(
            "SELECT id, error_message, fix FROM debug_records WHERE embedding IS NULL",
        ) {
            if let Ok(rows) = stmt.query_map([], |row| {
                let id: String = row.get(0)?;
                let error: String = row.get(1)?;
                let fix: String = row.get(2)?;
                Ok(EmbeddingItem {
                    id,
                    item_type: "debug_records".into(),
                    text: if fix.is_empty() {
                        format!("Error: {}", error)
                    } else {
                        format!("Error: {} Fix: {}", error, fix)
                    },
                })
            }) {
                items.extend(rows.flatten());
            }
        }

        items
    }

    // ─── Recent Error Tracking (for auto error→fix pairing) ─────────────────

    pub fn last_error(&self) -> Option<(String, String)> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT id, error_message FROM debug_records WHERE fix = '' ORDER BY recorded_at DESC LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .ok()
    }

    pub fn pair_error_with_fix(&self, error_id: &str, fix: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE debug_records SET fix = ?1 WHERE id = ?2",
            params![fix, error_id],
        )?;
        Ok(())
    }

    /// Count total items across all tables
    pub fn stats(&self) -> (usize, usize, usize, usize) {
        let conn = self.conn.lock().unwrap();
        let facts: usize = conn
            .query_row("SELECT COUNT(*) FROM facts", [], |r| r.get(0))
            .unwrap_or(0);
        let cmds: usize = conn
            .query_row("SELECT COUNT(*) FROM commands", [], |r| r.get(0))
            .unwrap_or(0);
        let debug: usize = conn
            .query_row("SELECT COUNT(*) FROM debug_records", [], |r| r.get(0))
            .unwrap_or(0);
        let wf: usize = conn
            .query_row("SELECT COUNT(*) FROM workflows", [], |r| r.get(0))
            .unwrap_or(0);
        (facts, cmds, debug, wf)
    }

    /// Count entities and edges
    pub fn entity_stats(&self) -> (usize, usize) {
        let conn = self.conn.lock().unwrap();
        let entities: usize = conn
            .query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0))
            .unwrap_or(0);
        let edges: usize = conn
            .query_row("SELECT COUNT(*) FROM entity_edges", [], |r| r.get(0))
            .unwrap_or(0);
        (entities, edges)
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn fact_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryFact> {
    let created_at: String = row.get(4)?;
    Ok(MemoryFact {
        id: row.get(0)?,
        content: row.get(1)?,
        category: row.get(2)?,
        confidence: row.get(3)?,
        created_at: created_at.clone(),
        updated_at: row.get::<_, Option<String>>(13).ok().flatten().unwrap_or_else(|| created_at.clone()),
        valid_at: row.get(14).unwrap_or(None),
        invalid_at: row.get(15).unwrap_or(None),
        source: row.get(5)?,
        expires_at: row.get(6)?,
        embedding: row.get::<_, Option<Vec<u8>>>(7)?.map(|b| blob_to_floats(&b)),
        pinned: row.get::<_, i32>(8).unwrap_or(0) != 0,
        importance: row.get::<_, f64>(16).unwrap_or(0.5),
        access_count: row.get::<_, u32>(17).unwrap_or(0),
        superseded_by: row.get(10).unwrap_or(None),
        ttl: row.get(11).unwrap_or(None),
        session_id: row.get(12).unwrap_or(None),
    })
}

fn floats_to_blob(floats: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(floats.len() * 4);
    for f in floats {
        blob.extend_from_slice(&f.to_le_bytes());
    }
    blob
}

fn blob_to_floats(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom > 0.0 {
        dot / denom
    } else {
        0.0
    }
}

/// Reciprocal Rank Fusion (Zep/Graphiti pattern)
/// RRF_score(d) = Σ w_i / (k + rank_i(d))
/// Gold standard for combining heterogeneous score distributions
pub fn rrf_fuse(
    result_lists: &[(&[SearchResult], f64)], // (results, weight)
    k: f64,
    limit: usize,
) -> Vec<SearchResult> {
    use std::collections::HashMap;

    let mut scores: HashMap<String, (f64, SearchResult)> = HashMap::new();

    for (results, weight) in result_lists {
        for (rank, result) in results.iter().enumerate() {
            let rrf_score = weight / (k + rank as f64 + 1.0);
            let entry = scores.entry(result.id.clone()).or_insert_with(|| (0.0, result.clone()));
            entry.0 += rrf_score;
        }
    }

    let mut fused: Vec<(f64, SearchResult)> = scores.into_values().collect();
    fused.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    fused.into_iter()
        .take(limit)
        .map(|(score, mut r)| {
            r.similarity = score as f32;
            r
        })
        .collect()
}

/// Compute composite score (CrewAI pattern):
/// w_sim × similarity + w_decay × decay_score + w_imp × importance
pub fn composite_score(
    similarity: f64,
    decay_score: f64,
    importance: f64,
    w_sim: f64,
    w_decay: f64,
    w_imp: f64,
) -> f64 {
    w_sim * similarity + w_decay * decay_score + w_imp * importance
}

/// Parse TTL duration string to chrono::Duration
fn parse_ttl_duration(ttl: &str) -> Option<chrono::Duration> {
    match ttl {
        "1h" => Some(chrono::Duration::hours(1)),
        "1d" => Some(chrono::Duration::days(1)),
        "7d" => Some(chrono::Duration::days(7)),
        "30d" => Some(chrono::Duration::days(30)),
        "permanent" | "session" => None, // handled separately
        s => {
            // Try to parse "Nd" format
            if let Some(num) = s.strip_suffix('d') {
                num.parse::<i64>().ok().map(chrono::Duration::days)
            } else if let Some(num) = s.strip_suffix('h') {
                num.parse::<i64>().ok().map(chrono::Duration::hours)
            } else {
                None
            }
        }
    }
}
