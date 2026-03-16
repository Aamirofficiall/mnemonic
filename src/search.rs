use crate::embeddings::Embedder;
use crate::models::SearchResult;
use crate::store::{cosine_similarity, MemoryStore};
use tracing::debug;

/// Parallel search engine (Zep/Graphiti pattern):
///   Stage 1: FTS5 (BM25) + Vector cosine — ALWAYS both, never short-circuit
///   Stage 2: RRF fusion — rank-based, stable across score distributions
///   Stage 3: Composite scoring — relevance + recency + importance
pub struct SearchEngine<'a> {
    store: &'a MemoryStore,
    embedder: &'a Embedder,
}

impl<'a> SearchEngine<'a> {
    pub fn new(store: &'a MemoryStore, embedder: &'a Embedder) -> Self {
        Self { store, embedder }
    }

    pub async fn search(
        &self,
        query: &str,
        limit: usize,
        categories: Option<&[String]>,
        min_similarity: f32,
        pinned_only: bool,
        since: Option<&str>,
    ) -> Vec<SearchResult> {
        // Stage 1: PARALLEL retrieval — both ALWAYS run (Zep pattern)
        let fts_results = self.store.fts_search(query, limit * 3);
        let fts_filtered = filter_results(&fts_results, categories, pinned_only, since);
        debug!("FTS5 returned {} results", fts_filtered.len());

        // Get vector results (cached or live embed)
        let vector_results = self.get_vector_results(
            query, limit * 3, categories, min_similarity, pinned_only, since
        ).await;
        eprintln!("[search] FTS: {} results, Vector: {} results", fts_filtered.len(), vector_results.len());

        // Stage 2: RRF fusion — rank-based combination
        // BM25 weight=1.0, vector weight=0.7, k=60 (from Zep/Graphiti research)
        let fused = rrf_fuse(&fts_filtered, &vector_results, 60.0, 1.0, 0.7, limit);
        eprintln!("[search] RRF fused: {} results", fused.len());

        // Increment access count for returned facts
        for r in &fused {
            if r.result_type == "fact" {
                let _ = self.store.increment_access_count(&r.id);
            }
        }

        fused
    }

    async fn get_vector_results(
        &self,
        query: &str,
        limit: usize,
        categories: Option<&[String]>,
        min_similarity: f32,
        pinned_only: bool,
        since: Option<&str>,
    ) -> Vec<SearchResult> {
        // Try cached embedding first
        if let Some(cached_emb) = self.store.load_cached_query_embedding(query) {
            eprintln!("[vector] Using cached embed ({} dims)", cached_emb.len());
            let r = self.vector_search(&cached_emb, limit, categories, min_similarity, pinned_only, since);
            eprintln!("[vector] Cached search returned {} results", r.len());
            return r;
        }

        // Live embed via Gemini
        match self.embedder.embed_one_async(query).await {
            Ok(query_emb) => {
                eprintln!("[vector] Live embed: {} dims", query_emb.len());
                let _ = self.store.cache_query_embedding(query, &query_emb);
                let r = self.vector_search(&query_emb, limit, categories, min_similarity, pinned_only, since);
                eprintln!("[vector] Live search returned {} results", r.len());
                r
            }
            Err(e) => {
                eprintln!("[vector] Embed FAILED: {}", e);
                vec![]
            }
        }
    }

    fn vector_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        categories: Option<&[String]>,
        min_similarity: f32,
        pinned_only: bool,
        since: Option<&str>,
    ) -> Vec<SearchResult> {
        let all_embeddings = self.store.load_all_embeddings();
        let context = self.store.load_context();

        // Build lookup maps
        let fact_map: std::collections::HashMap<_, _> =
            context.facts.iter().map(|f| (f.id.as_str(), f)).collect();
        let cmd_map: std::collections::HashMap<_, _> = context
            .recent_commands
            .iter()
            .map(|c| (c.id.as_str(), c))
            .collect();
        let dbg_map: std::collections::HashMap<_, _> = context
            .recent_debug
            .iter()
            .map(|d| (d.id.as_str(), d))
            .collect();
        let wf_map: std::collections::HashMap<_, _> = context
            .workflows
            .iter()
            .map(|w| (w.id.as_str(), w))
            .collect();

        let mut results: Vec<SearchResult> = Vec::new();

        for (id, item_type, embedding) in &all_embeddings {
            let raw_sim = cosine_similarity(query_embedding, embedding);
            if raw_sim < min_similarity {
                continue;
            }

            match item_type.as_str() {
                "fact" => {
                    if let Some(fact) = fact_map.get(id.as_str()) {
                        if fact.is_expired() {
                            continue;
                        }
                        if pinned_only && !fact.pinned {
                            continue;
                        }
                        if let Some(since_date) = since {
                            if fact.created_at.as_str() < since_date {
                                continue;
                            }
                        }
                        if let Some(cats) = categories {
                            if !cats.iter().any(|c| c == "fact" || c == &fact.category) {
                                continue;
                            }
                        }
                        // Real similarity: combine vector similarity with strength
                        // Composite score: w_sim × similarity + w_decay × decay + w_imp × importance
                        let decay = fact.decay_score();
                        let sim = (0.5 * raw_sim as f64 + 0.3 * decay + 0.2 * fact.importance).clamp(0.0, 1.0) as f32;
                        results.push(SearchResult {
                            result_type: "fact".into(),
                            content: fact.content.clone(),
                            category: Some(fact.category.clone()),
                            similarity: sim,
                            id: fact.id.clone(),
                            metadata: Some(serde_json::json!({
                                "confidence": fact.confidence,
                                "source": fact.source,
                                "pinned": fact.pinned,
                                "importance": format!("{:.2}", fact.importance),
                                "decay_score": format!("{:.3}", fact.decay_score()),
                                "raw_similarity": format!("{:.3}", raw_sim),
                            })),
                        });
                    }
                }
                "command" => {
                    if pinned_only {
                        continue; // commands can't be pinned
                    }
                    if let Some(cats) = categories {
                        if !cats.iter().any(|c| c == "command") {
                            continue;
                        }
                    }
                    if let Some(cmd) = cmd_map.get(id.as_str()) {
                        let content = format!(
                            "{} → {}{}",
                            cmd.tool,
                            cmd.outcome,
                            cmd.result_snippet
                                .as_ref()
                                .map(|s| format!(" | {}", s))
                                .unwrap_or_default()
                        );
                        results.push(SearchResult {
                            result_type: "command".into(),
                            content,
                            category: None,
                            similarity: raw_sim,
                            id: cmd.id.clone(),
                            metadata: Some(serde_json::json!({
                                "tool": cmd.tool,
                                "outcome": cmd.outcome,
                                "raw_similarity": format!("{:.3}", raw_sim),
                            })),
                        });
                    }
                }
                "debug" => {
                    if pinned_only {
                        continue;
                    }
                    if let Some(cats) = categories {
                        if !cats.iter().any(|c| c == "debug") {
                            continue;
                        }
                    }
                    if let Some(dbg) = dbg_map.get(id.as_str()) {
                        let content = if dbg.fix.is_empty() {
                            format!("Error: {}", dbg.error_message)
                        } else {
                            format!("Error: {} → Fix: {}", dbg.error_message, dbg.fix)
                        };
                        results.push(SearchResult {
                            result_type: "debug".into(),
                            content,
                            category: None,
                            similarity: raw_sim,
                            id: dbg.id.clone(),
                            metadata: Some(serde_json::json!({
                                "error": dbg.error_message,
                                "fix": dbg.fix,
                                "raw_similarity": format!("{:.3}", raw_sim),
                            })),
                        });
                    }
                }
                "workflow" => {
                    if pinned_only {
                        continue;
                    }
                    if let Some(cats) = categories {
                        if !cats.iter().any(|c| c == "workflow") {
                            continue;
                        }
                    }
                    if let Some(wf) = wf_map.get(id.as_str()) {
                        results.push(SearchResult {
                            result_type: "workflow".into(),
                            content: format!(
                                "Workflow: {} — {}",
                                wf.name,
                                wf.steps.join(" → ")
                            ),
                            category: None,
                            similarity: raw_sim,
                            id: wf.id.clone(),
                            metadata: Some(serde_json::json!({
                                "name": wf.name,
                                "success": wf.success_count,
                                "failure": wf.failure_count,
                                "raw_similarity": format!("{:.3}", raw_sim),
                            })),
                        });
                    }
                }
                _ => {}
            }
        }

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.into_iter().take(limit).collect()
    }
}

fn filter_results(
    results: &[SearchResult],
    categories: Option<&[String]>,
    pinned_only: bool,
    since: Option<&str>,
) -> Vec<SearchResult> {
    results
        .iter()
        .filter(|r| {
            // Category filter
            if let Some(cats) = categories {
                let matches = if let Some(ref cat) = r.category {
                    cats.iter().any(|c| c == cat || c == &r.result_type)
                } else {
                    cats.iter().any(|c| c == &r.result_type)
                };
                if !matches {
                    return false;
                }
            }
            // Pinned filter
            if pinned_only {
                let is_pinned = r.metadata
                    .as_ref()
                    .and_then(|m| m.get("pinned"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if !is_pinned {
                    return false;
                }
            }
            // Since filter (only for facts which have created_at in metadata indirectly)
            // Note: FTS results don't carry created_at, so we skip since filter for FTS
            // The since filter is better applied at the store level for accuracy
            let _ = since; // acknowledged but not filtered here for FTS
            true
        })
        .cloned()
        .collect()
}

/// Reciprocal Rank Fusion (Zep/Graphiti pattern)
/// RRF_score(d) = Σ w_i / (k + rank_i(d))
/// Stable across heterogeneous score distributions (BM25 long-tail vs vector narrow-range)
fn rrf_fuse(
    fts_results: &[SearchResult],
    vector_results: &[SearchResult],
    k: f64,
    fts_weight: f64,
    vec_weight: f64,
    limit: usize,
) -> Vec<SearchResult> {
    use std::collections::HashMap;

    let mut scores: HashMap<String, (f64, SearchResult)> = HashMap::new();

    // FTS results — already sorted by BM25 rank
    for (rank, result) in fts_results.iter().enumerate() {
        let rrf = fts_weight / (k + rank as f64 + 1.0);
        let entry = scores.entry(result.id.clone()).or_insert_with(|| (0.0, result.clone()));
        entry.0 += rrf;
    }

    // Vector results — already sorted by cosine similarity
    for (rank, result) in vector_results.iter().enumerate() {
        let rrf = vec_weight / (k + rank as f64 + 1.0);
        let entry = scores.entry(result.id.clone()).or_insert_with(|| (0.0, result.clone()));
        entry.0 += rrf;
    }

    let mut fused: Vec<(f64, SearchResult)> = scores.into_values().collect();
    fused.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Normalize to 0-1: divide by max score so top result = ~1.0
    let max_score = fused.first().map(|(s, _)| *s).unwrap_or(1.0).max(0.001);

    fused.into_iter()
        .take(limit)
        .map(|(score, mut r)| {
            r.similarity = (score / max_score) as f32;
            r
        })
        .collect()
}
