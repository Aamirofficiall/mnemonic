//! Code tree: tree-sitter based code intelligence for AI agents.
//! Extracts definitions, references, and dependencies from source code.
//! Stores everything in the entity graph (same SQLite DB as memory).

use anyhow::{anyhow, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use streaming_iterator::StreamingIterator;
use tracing::{debug, info, warn};

use crate::store::MemoryStore;

// ─── Tag: extracted symbol from source ──────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Tag {
    pub file: String,       // relative path
    pub name: String,       // symbol name
    pub kind: TagKind,      // definition or reference
    pub symbol_type: String, // function, class, method, struct, etc.
    pub line: usize,        // 0-indexed line number
    pub signature: String,  // the source line containing the definition
}

#[derive(Debug, Clone, PartialEq)]
pub enum TagKind {
    Definition,
    Reference,
}

// ─── CodeSymbol: stored in DB ───────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct CodeSymbol {
    pub id: String,
    pub file_path: String,
    pub name: String,
    pub kind: String,       // function, class, struct, method, trait, module, macro
    pub signature: String,
    pub line_start: usize,
    pub line_end: usize,
    pub parent_symbol: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CodeFile {
    pub path: String,
    pub language: String,
    pub content_hash: String,
    pub symbol_count: usize,
    pub line_count: usize,
    pub pagerank: f64,
}

// ─── Language detection ─────────────────────────────────────────────────────

fn detect_language(path: &Path) -> Option<&'static str> {
    match path.extension()?.to_str()? {
        "rs" => Some("rust"),
        "py" => Some("python"),
        "c" | "h" => Some("c"),
        "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" => Some("cpp"),
        "js" | "jsx" | "mjs" => Some("javascript"),
        "ts" | "tsx" => Some("typescript"),
        "swift" => Some("swift"),
        _ => None,
    }
}

// ─── Tree-sitter query files (bundled) ──────────────────────────────────────

fn get_tags_query(language: &str) -> Option<&'static str> {
    match language {
        "rust" => Some(include_str!("queries/rust-tags.scm")),
        "python" => Some(include_str!("queries/python-tags.scm")),
        "c" => Some(include_str!("queries/c-tags.scm")),
        "cpp" => Some(include_str!("queries/cpp-tags.scm")),
        "javascript" => Some(include_str!("queries/javascript-tags.scm")),
        "typescript" => Some(include_str!("queries/typescript-tags.scm")),
        "swift" => Some(include_str!("queries/swift-tags.scm")),
        _ => None,
    }
}

fn get_ts_language(language: &str) -> Option<tree_sitter::Language> {
    match language {
        "rust" => Some(tree_sitter_rust::LANGUAGE.into()),
        "python" => Some(tree_sitter_python::LANGUAGE.into()),
        "c" => Some(tree_sitter_c::LANGUAGE.into()),
        "cpp" => Some(tree_sitter_cpp::LANGUAGE.into()),
        "javascript" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "typescript" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        "swift" => Some(npezza93_tree_sitter_swift::language().into()),
        _ => None,
    }
}

// ─── Parse a single file ────────────────────────────────────────────────────

pub fn parse_file(path: &Path, root: &Path) -> Result<Vec<Tag>> {
    let language = detect_language(path).ok_or_else(|| anyhow!("unsupported language: {:?}", path))?;
    let ts_lang = get_ts_language(language).ok_or_else(|| anyhow!("no tree-sitter grammar for {}", language))?;
    let query_src = get_tags_query(language).ok_or_else(|| anyhow!("no tags query for {}", language))?;

    let source = std::fs::read_to_string(path)?;
    let rel_path = path.strip_prefix(root).unwrap_or(path).to_string_lossy().to_string();

    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&ts_lang)?;

    let tree = parser.parse(&source, None).ok_or_else(|| anyhow!("parse failed: {}", rel_path))?;

    let query = tree_sitter::Query::new(&ts_lang, query_src)
        .map_err(|e| anyhow!("query compile error for {}: {}", language, e))?;

    let mut cursor = tree_sitter::QueryCursor::new();

    let mut tags = Vec::new();
    let lines: Vec<&str> = source.lines().collect();

    let mut matches = cursor.matches(&query, tree.root_node(), source.as_bytes());
    while let Some(m) = { matches.advance(); matches.get() } {
        let mut name = String::new();
        let mut line = 0;
        let mut kind = TagKind::Definition;
        let mut symbol_type = String::new();

        for capture in m.captures {
            let capture_name = &query.capture_names()[capture.index as usize];
            let text = capture.node.utf8_text(source.as_bytes()).unwrap_or("");

            if capture_name.starts_with("name.definition.") {
                name = text.to_string();
                line = capture.node.start_position().row;
                kind = TagKind::Definition;
                symbol_type = capture_name.strip_prefix("name.definition.").unwrap_or("unknown").to_string();
            } else if capture_name.starts_with("name.reference.") {
                name = text.to_string();
                line = capture.node.start_position().row;
                kind = TagKind::Reference;
                symbol_type = capture_name.strip_prefix("name.reference.").unwrap_or("unknown").to_string();
            }
        }

        if !name.is_empty() {
            let signature = lines.get(line).unwrap_or(&"").trim().to_string();
            tags.push(Tag {
                file: rel_path.clone(),
                name,
                kind,
                symbol_type,
                line,
                signature: signature.chars().take(200).collect(),
            });
        }
    }

    Ok(tags)
}

// ─── Scan directory ─────────────────────────────────────────────────────────

pub fn scan_directory(root: &Path) -> Result<Vec<(PathBuf, String)>> {
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            // Skip hidden dirs, build artifacts, node_modules, etc.
            !name.starts_with('.') && name != "target" && name != "node_modules"
                && name != "build" && name != "__pycache__" && name != "venv"
                && name != ".venv" && name != "dist" && name != "vendor"
        })
    {
        let entry = entry?;
        if entry.file_type().is_file() {
            if let Some(lang) = detect_language(entry.path()) {
                files.push((entry.path().to_path_buf(), lang.to_string()));
            }
        }
    }
    Ok(files)
}

// ─── Content hash ───────────────────────────────────────────────────────────

pub fn file_hash(path: &Path) -> Result<String> {
    let content = std::fs::read(path)?;
    let hash = Sha256::digest(&content);
    Ok(format!("{:x}", hash))
}

// ─── Full parse: scan + parse all files ─────────────────────────────────────

pub struct ParseResult {
    pub files: Vec<CodeFile>,
    pub symbols: Vec<CodeSymbol>,
    pub tags: Vec<Tag>,
    pub files_parsed: usize,
    pub files_skipped: usize,
}

pub fn parse_project(root: &Path, store: &MemoryStore) -> Result<ParseResult> {
    let source_files = scan_directory(root)?;
    info!("Found {} source files in {}", source_files.len(), root.display());

    let mut all_tags = Vec::new();
    let mut all_symbols = Vec::new();
    let mut all_files = Vec::new();
    let mut parsed = 0;
    let mut skipped = 0;

    for (path, language) in &source_files {
        let rel_path = path.strip_prefix(root).unwrap_or(path).to_string_lossy().to_string();

        // Check if file changed (content hash)
        let hash = match file_hash(path) {
            Ok(h) => h,
            Err(_) => { skipped += 1; continue; }
        };

        if let Some(existing_hash) = store.get_code_file_hash(&rel_path) {
            if existing_hash == hash {
                skipped += 1;
                debug!("Skipped (unchanged): {}", rel_path);
                continue;
            }
        }

        // Parse file
        match parse_file(path, root) {
            Ok(tags) => {
                let line_count = std::fs::read_to_string(path)
                    .map(|s| s.lines().count())
                    .unwrap_or(0);

                // Extract definitions as symbols
                let mut def_count = 0;
                for tag in tags.iter().filter(|t| t.kind == TagKind::Definition) {
                    let sym_id = format!("sym_{}_{}", &uuid::Uuid::new_v4().to_string()[..8], tag.name);
                    all_symbols.push(CodeSymbol {
                        id: sym_id,
                        file_path: rel_path.clone(),
                        name: tag.name.clone(),
                        kind: tag.symbol_type.clone(),
                        signature: tag.signature.clone(),
                        line_start: tag.line,
                        line_end: tag.line,
                        parent_symbol: None,
                    });
                    def_count += 1;
                }

                all_files.push(CodeFile {
                    path: rel_path.clone(),
                    language: language.clone(),
                    content_hash: hash,
                    symbol_count: def_count,
                    line_count,
                    pagerank: 0.0,
                });

                parsed += 1;
                debug!("Parsed: {} ({} defs)", rel_path, def_count);
                all_tags.extend(tags);
            }
            Err(e) => {
                warn!("Parse error {}: {}", rel_path, e);
                skipped += 1;
            }
        }
    }

    info!("Parsed {} files ({} skipped), {} symbols, {} tags",
        parsed, skipped, all_symbols.len(), all_tags.len());

    Ok(ParseResult {
        files: all_files,
        symbols: all_symbols,
        tags: all_tags,
        files_parsed: parsed,
        files_skipped: skipped,
    })
}

// ─── Store results in DB ────────────────────────────────────────────────────

impl MemoryStore {
    pub fn get_code_file_hash(&self, path: &str) -> Option<String> {
        let conn = self.conn_ref();
        conn.query_row(
            "SELECT content_hash FROM code_files WHERE path = ?1",
            rusqlite::params![path],
            |row| row.get(0),
        ).ok()
    }

    pub fn upsert_code_file(&self, file: &CodeFile) -> Result<()> {
        let conn = self.conn_ref();
        conn.execute(
            "INSERT OR REPLACE INTO code_files (path, language, content_hash, symbol_count, line_count, last_parsed, pagerank)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                file.path, file.language, file.content_hash,
                file.symbol_count, file.line_count,
                chrono::Utc::now().to_rfc3339(), file.pagerank,
            ],
        )?;
        Ok(())
    }

    pub fn upsert_code_symbol(&self, sym: &CodeSymbol) -> Result<()> {
        let conn = self.conn_ref();
        conn.execute(
            "INSERT OR REPLACE INTO code_symbols (id, file_path, name, kind, signature, line_start, line_end, parent_symbol, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                sym.id, sym.file_path, sym.name, sym.kind, sym.signature,
                sym.line_start, sym.line_end, sym.parent_symbol,
                chrono::Utc::now().to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn upsert_code_dep(&self, from: &str, to: &str, symbol: &str, weight: f64) -> Result<()> {
        let conn = self.conn_ref();
        conn.execute(
            "INSERT OR REPLACE INTO code_deps (from_file, to_file, symbol_name, weight)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![from, to, symbol, weight],
        )?;
        Ok(())
    }

    pub fn clear_code_for_file(&self, path: &str) -> Result<()> {
        let conn = self.conn_ref();
        conn.execute("DELETE FROM code_symbols WHERE file_path = ?1", rusqlite::params![path])?;
        conn.execute("DELETE FROM code_deps WHERE from_file = ?1 OR to_file = ?1", rusqlite::params![path])?;
        Ok(())
    }

    pub fn load_code_files(&self) -> Vec<CodeFile> {
        let conn = self.conn_ref();
        let mut stmt = conn.prepare(
            "SELECT path, language, content_hash, symbol_count, line_count, pagerank FROM code_files ORDER BY pagerank DESC"
        ).unwrap();
        stmt.query_map([], |row| {
            Ok(CodeFile {
                path: row.get(0)?,
                language: row.get(1)?,
                content_hash: row.get(2)?,
                symbol_count: row.get(3)?,
                line_count: row.get(4)?,
                pagerank: row.get(5)?,
            })
        }).unwrap().filter_map(|r| r.ok()).collect()
    }

    pub fn load_code_symbols(&self, file_path: Option<&str>) -> Vec<CodeSymbol> {
        let conn = self.conn_ref();
        let (sql, params): (&str, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(fp) = file_path {
            ("SELECT id, file_path, name, kind, signature, line_start, line_end, parent_symbol FROM code_symbols WHERE file_path = ?1 ORDER BY line_start",
             vec![Box::new(fp.to_string())])
        } else {
            ("SELECT id, file_path, name, kind, signature, line_start, line_end, parent_symbol FROM code_symbols ORDER BY file_path, line_start",
             vec![])
        };
        let mut stmt = conn.prepare(sql).unwrap();
        stmt.query_map(rusqlite::params_from_iter(params), |row| {
            Ok(CodeSymbol {
                id: row.get(0)?,
                file_path: row.get(1)?,
                name: row.get(2)?,
                kind: row.get(3)?,
                signature: row.get(4)?,
                line_start: row.get(5)?,
                line_end: row.get(6)?,
                parent_symbol: row.get(7)?,
            })
        }).unwrap().filter_map(|r| r.ok()).collect()
    }

    pub fn code_stats(&self) -> (usize, usize, usize) {
        let conn = self.conn_ref();
        let files: usize = conn.query_row("SELECT COUNT(*) FROM code_files", [], |r| r.get(0)).unwrap_or(0);
        let symbols: usize = conn.query_row("SELECT COUNT(*) FROM code_symbols", [], |r| r.get(0)).unwrap_or(0);
        let deps: usize = conn.query_row("SELECT COUNT(*) FROM code_deps", [], |r| r.get(0)).unwrap_or(0);
        (files, symbols, deps)
    }
}

// ─── Build dependency graph from tags ───────────────────────────────────────

pub fn build_dependencies(tags: &[Tag], store: &MemoryStore) -> Result<usize> {
    // Map: symbol_name → files where it's defined
    let mut defs: HashMap<String, Vec<String>> = HashMap::new();
    // Map: (file, symbol_name) → reference count
    let mut refs: HashMap<(String, String), usize> = HashMap::new();

    for tag in tags {
        match tag.kind {
            TagKind::Definition => {
                defs.entry(tag.name.clone()).or_default().push(tag.file.clone());
            }
            TagKind::Reference => {
                *refs.entry((tag.file.clone(), tag.name.clone())).or_insert(0) += 1;
            }
        }
    }

    let mut dep_count = 0;

    for ((ref_file, name), count) in &refs {
        if let Some(def_files) = defs.get(name) {
            for def_file in def_files {
                if def_file != ref_file {
                    // Edge: ref_file references something defined in def_file
                    let mut weight = 1.0;

                    // Long names are more specific (Aider heuristic)
                    if name.len() >= 8 && (name.contains('_') || name.chars().any(|c| c.is_uppercase())) {
                        weight *= 10.0;
                    }

                    // Private names less important
                    if name.starts_with('_') {
                        weight *= 0.1;
                    }

                    // Too generic (defined in many files)
                    if def_files.len() > 5 {
                        weight *= 0.1;
                    }

                    // Reference frequency
                    weight *= (*count as f64).sqrt();

                    store.upsert_code_dep(ref_file, def_file, name, weight)?;
                    dep_count += 1;
                }
            }
        }
    }

    info!("Built {} cross-file dependencies", dep_count);
    Ok(dep_count)
}

// ─── Render code map with binary search token budget (Aider pattern) ────────

fn render_n_files(files: &[CodeFile], store: &MemoryStore, n: usize) -> String {
    let mut output = String::new();
    for file in files.iter().take(n) {
        let symbols = store.load_code_symbols(Some(&file.path));
        let defs: Vec<String> = symbols.iter()
            .filter(|s| s.kind != "call" && s.kind != "implementation")
            .map(|s| {
                if s.kind == "function" || s.kind == "method" {
                    format!("{}()", s.name)
                } else {
                    s.name.clone()
                }
            })
            .collect();
        // Dedup consecutive identical names
        let mut deduped: Vec<String> = Vec::new();
        for d in &defs {
            if deduped.last().map(|l| l != d).unwrap_or(true) {
                deduped.push(d.clone());
            }
        }
        if !deduped.is_empty() {
            output.push_str(&format!("{}: {}\n", file.path, deduped.join(", ")));
        } else {
            output.push_str(&format!("{}\n", file.path));
        }
    }
    output
}

pub fn render_codemap(store: &MemoryStore, file_filter: Option<&str>, budget_chars: usize) -> String {
    let files = store.load_code_files();

    if let Some(filter) = file_filter {
        // Single file — try exact, then suffix match
        let mut symbols = store.load_code_symbols(Some(filter));
        if symbols.is_empty() {
            let all = store.load_code_symbols(None);
            symbols = all.into_iter().filter(|s| s.file_path.ends_with(filter)).collect();
        }
        if symbols.is_empty() {
            return format!("No symbols found for: {}", filter);
        }
        let mut output = format!("{}:\n", filter);
        for sym in &symbols {
            if sym.kind == "call" || sym.kind == "implementation" { continue; }
            output.push_str(&format!("  {} {} | {}\n", sym.kind, sym.name, sym.signature));
        }
        let (fc, sc, dc) = store.code_stats();
        output.push_str(&format!("\n[{} files, {} symbols, {} deps]\n", fc, sc, dc));
        return output;
    }

    // Binary search: find max N files that fit within budget (Aider pattern)
    let mut lo: usize = 1;
    let mut hi: usize = files.len();
    let mut best_output = render_n_files(&files, store, 1);

    while lo <= hi {
        let mid = (lo + hi) / 2;
        let rendered = render_n_files(&files, store, mid);
        if rendered.len() <= budget_chars {
            best_output = rendered;
            lo = mid + 1;
        } else {
            if mid == 0 { break; }
            hi = mid - 1;
        }
    }

    let shown = best_output.lines().count();
    let (fc, sc, dc) = store.code_stats();
    if shown < files.len() {
        best_output.push_str(&format!("\n... ({}/{} files shown, budget: {} chars)\n", shown, fc, budget_chars));
    }
    best_output.push_str(&format!("[{} files, {} symbols, {} deps]\n", fc, sc, dc));
    best_output
}

// ─── Render as JSON ─────────────────────────────────────────────────────────

pub fn render_codemap_json(store: &MemoryStore, file_filter: Option<&str>) -> serde_json::Value {
    if let Some(filter) = file_filter {
        let symbols = store.load_code_symbols(Some(filter));
        serde_json::json!({
            "file": filter,
            "symbols": symbols,
        })
    } else {
        let files = store.load_code_files();
        let mut file_data = Vec::new();
        for file in &files {
            let symbols = store.load_code_symbols(Some(&file.path));
            file_data.push(serde_json::json!({
                "path": file.path,
                "language": file.language,
                "symbols": symbols.iter().filter(|s| s.kind != "call").collect::<Vec<_>>(),
                "line_count": file.line_count,
            }));
        }
        let (fc, sc, dc) = store.code_stats();
        serde_json::json!({
            "files": file_data,
            "stats": {"files": fc, "symbols": sc, "deps": dc},
        })
    }
}

// ─── PageRank (simple power iteration) ──────────────────────────────────────

pub fn compute_pagerank(store: &MemoryStore, damping: f64, iterations: usize, changed_files: Option<&[String]>) -> Result<usize> {
    let conn = store.conn_ref();

    // Get all files
    let mut stmt = conn.prepare("SELECT path FROM code_files").unwrap();
    let files: Vec<String> = stmt.query_map([], |r| r.get(0))
        .unwrap().filter_map(|r| r.ok()).collect();

    // If only specific files changed, still compute full PageRank but log it
    if let Some(changed) = changed_files {
        debug!("PageRank: {} files changed, recomputing full graph ({} files)", changed.len(), files.len());
    }

    if files.is_empty() { return Ok(0); }

    let n = files.len();
    let file_idx: HashMap<String, usize> = files.iter().enumerate().map(|(i, f)| (f.clone(), i)).collect();

    // Build adjacency from deps
    let mut stmt = conn.prepare("SELECT from_file, to_file, weight FROM code_deps").unwrap();
    let deps: Vec<(String, String, f64)> = stmt.query_map([], |r| {
        Ok((r.get(0)?, r.get(1)?, r.get(2)?))
    }).unwrap().filter_map(|r| r.ok()).collect();

    // Outgoing weight sums per file
    let mut out_weight: Vec<f64> = vec![0.0; n];
    for (from, _to, w) in &deps {
        if let Some(&i) = file_idx.get(from) {
            out_weight[i] += w;
        }
    }

    // Power iteration
    let mut rank = vec![1.0 / n as f64; n];
    for _ in 0..iterations {
        let mut new_rank = vec![(1.0 - damping) / n as f64; n];
        for (from, to, w) in &deps {
            if let (Some(&i), Some(&j)) = (file_idx.get(from), file_idx.get(to)) {
                if out_weight[i] > 0.0 {
                    new_rank[j] += damping * rank[i] * w / out_weight[i];
                }
            }
        }
        rank = new_rank;
    }

    // Normalize to 0-1
    let max_rank = rank.iter().cloned().fold(0.0f64, f64::max).max(0.001);

    // Store ranks
    for (i, file) in files.iter().enumerate() {
        let normalized = rank[i] / max_rank;
        conn.execute(
            "UPDATE code_files SET pagerank = ?1 WHERE path = ?2",
            rusqlite::params![normalized, file],
        )?;
    }

    info!("PageRank computed for {} files ({} iterations)", n, iterations);
    Ok(n)
}

// ─── Symbol drill-down ──────────────────────────────────────────────────────

pub fn render_symbol_detail(store: &MemoryStore, file: &str, symbol_name: &str) -> String {
    // Try exact match first, then suffix match
    let mut symbols = store.load_code_symbols(Some(file));
    if symbols.is_empty() {
        // Try suffix match (user passes "store.rs", DB has "mnemonic-repo/src/store.rs")
        let all = store.load_code_symbols(None);
        symbols = all.into_iter().filter(|s| s.file_path.ends_with(file)).collect();
    }
    let sym = symbols.iter().find(|s| s.name == symbol_name);

    match sym {
        Some(s) => {
            let mut out = String::new();
            let file_path = s.file_path.clone();
            out.push_str(&format!("{} {} ({}:{})\n", s.kind, s.name, s.file_path, s.line_start));
            out.push_str(&format!("  signature: {}\n", s.signature));

            // Find callers + callees in one lock scope, then release
            let (callers, callees) = {
                let conn = store.conn_ref();
                let callers: Vec<(String, f64)> = conn
                    .prepare("SELECT from_file, weight FROM code_deps WHERE symbol_name = ?1 AND to_file LIKE ?2")
                    .unwrap()
                    .query_map(rusqlite::params![symbol_name, format!("%{}", file_path)], |r| Ok((r.get(0)?, r.get(1)?)))
                    .unwrap()
                    .filter_map(|r| r.ok())
                    .collect();
                let callees: Vec<(String, String)> = conn
                    .prepare("SELECT to_file, symbol_name FROM code_deps WHERE from_file = ?1 LIMIT 10")
                    .unwrap()
                    .query_map(rusqlite::params![file_path], |r| Ok((r.get(0)?, r.get(1)?)))
                    .unwrap()
                    .filter_map(|r| r.ok())
                    .collect();
                (callers, callees)
            }; // conn dropped here — no deadlock

            if !callers.is_empty() {
                out.push_str("  called_by:\n");
                for (caller, _w) in &callers {
                    out.push_str(&format!("    - {}\n", caller));
                }
            }

            if !callees.is_empty() {
                out.push_str("  calls:\n");
                for (target, sym_name) in &callees {
                    out.push_str(&format!("    - {}::{}\n", target, sym_name));
                }
            }

            // Find related memory facts (FTS only, no Gemini — conn released above)
            let entity_facts = store.fts_search(symbol_name, 3);
            let relevant: Vec<_> = entity_facts.iter()
                .filter(|f| f.result_type == "fact")
                .collect();
            if !relevant.is_empty() {
                out.push_str("  related_facts:\n");
                for fact in &relevant {
                    let preview: String = fact.content.chars().take(80).collect();
                    out.push_str(&format!("    - {}\n", preview));
                }
            }

            out
        }
        None => format!("Symbol '{}' not found in {}", symbol_name, file),
    }
}

// ─── Render as XML ──────────────────────────────────────────────────────────

pub fn render_codemap_xml(store: &MemoryStore, file_filter: Option<&str>) -> String {
    let (fc, sc, dc) = store.code_stats();
    let mut out = String::new();

    if let Some(filter) = file_filter {
        let symbols = store.load_code_symbols(Some(filter));
        out.push_str(&format!("<codetree file=\"{}\">\n", escape_xml(filter)));
        for sym in &symbols {
            if sym.kind == "call" || sym.kind == "implementation" { continue; }
            out.push_str(&format!("  <{} name=\"{}\" signature=\"{}\" line=\"{}\"/>\n",
                sym.kind, escape_xml(&sym.name), escape_xml(&sym.signature), sym.line_start));
        }
        out.push_str("</codetree>\n");
    } else {
        out.push_str(&format!("<codetree files=\"{}\" symbols=\"{}\" deps=\"{}\">\n", fc, sc, dc));
        let files = store.load_code_files();
        for file in &files {
            let symbols = store.load_code_symbols(Some(&file.path));
            let defs: Vec<_> = symbols.iter()
                .filter(|s| s.kind != "call" && s.kind != "implementation")
                .collect();
            if defs.is_empty() {
                out.push_str(&format!("  <file path=\"{}\" language=\"{}\" rank=\"{:.2}\"/>\n",
                    escape_xml(&file.path), file.language, file.pagerank));
            } else {
                out.push_str(&format!("  <file path=\"{}\" language=\"{}\" rank=\"{:.2}\">\n",
                    escape_xml(&file.path), file.language, file.pagerank));
                for sym in &defs {
                    out.push_str(&format!("    <{} name=\"{}\" line=\"{}\"/>\n",
                        sym.kind, escape_xml(&sym.name), sym.line_start));
                }
                out.push_str("  </file>\n");
            }
        }
        out.push_str("</codetree>\n");
    }
    out
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

// ─── Memory integration: link code symbols to entity graph ──────────────────

pub fn link_symbols_to_entities(store: &MemoryStore) -> Result<usize> {
    let symbols = store.load_code_symbols(None);
    let mut linked = 0;

    for sym in &symbols {
        if sym.kind == "call" || sym.kind == "implementation" { continue; }

        // Create or find entity for this symbol
        match store.find_or_create_entity(&sym.name, &format!("code_{}", sym.kind)) {
            Ok(entity_id) => {
                // Create edge: entity → code symbol (using fact_id field for symbol_id)
                let edge = crate::models::EntityEdge {
                    entity_id: entity_id.clone(),
                    fact_id: sym.id.clone(),
                    relation: "defines".to_string(),
                };
                let _ = store.insert_entity_edge(&edge);

                // Also create entity for the file
                if let Ok(file_entity_id) = store.find_or_create_entity(&sym.file_path, "file") {
                    let file_edge = crate::models::EntityEdge {
                        entity_id: file_entity_id,
                        fact_id: sym.id.clone(),
                        relation: "contains".to_string(),
                    };
                    let _ = store.insert_entity_edge(&file_edge);
                }

                linked += 1;
            }
            Err(_) => {}
        }
    }

    info!("Linked {} code symbols to entity graph", linked);
    Ok(linked)
}

// ─── Full pipeline: parse + store + build deps ──────────────────────────────

pub fn index_project(root: &Path, store: &MemoryStore) -> Result<serde_json::Value> {
    let t0 = std::time::Instant::now();

    let result = parse_project(root, store)?;

    // Store files
    for file in &result.files {
        store.clear_code_for_file(&file.path)?;
        store.upsert_code_file(file)?;
    }

    // Store symbols
    for sym in &result.symbols {
        store.upsert_code_symbol(sym)?;
    }

    // Build dependencies
    let dep_count = build_dependencies(&result.tags, store)?;

    // Compute PageRank (pass changed files for logging)
    let changed: Vec<String> = result.files.iter().map(|f| f.path.clone()).collect();
    let _ranked = compute_pagerank(store, 0.85, 20, if changed.is_empty() { None } else { Some(&changed) })?;

    // Link symbols to entity graph
    let linked = link_symbols_to_entities(store)?;

    let elapsed = t0.elapsed().as_millis();
    let (fc, sc, dc) = store.code_stats();

    Ok(serde_json::json!({
        "files_parsed": result.files_parsed,
        "files_skipped": result.files_skipped,
        "symbols": sc,
        "deps": dc,
        "total_files": fc,
        "time_ms": elapsed,
    }))
}
