# Search Engine — mnemonic v2.0

## RRF Fusion Pipeline (4 stages)

v2 replaces the v1 waterfall (FTS → maybe vector → maybe Gemini) with parallel retrieval + Reciprocal Rank Fusion.

### Stage 1: Parallel Retrieval
Both run simultaneously on every query:

**FTS5 Keyword Search (~0.1ms)**
- SQLite FTS5 index on fact content + category
- BM25 ranking (term frequency × inverse document frequency)
- Tokenizes query into words, joins with OR
- Filters: `invalid_at IS NULL AND superseded_by IS NULL`

**Vector Cosine Search (~0.5ms cached, ~50ms live)**
- 768-dim Gemini embeddings
- Cosine similarity against all embedded facts
- Query embedding cached in `query_cache` table
- Falls back to FTS-only if embedding fails

### Stage 2: RRF Fusion
```
RRF_score(d) = Σ w_i / (k + rank_i(d))
```
- k = 60 (standard constant from IR research)
- BM25 weight = 1.0 (keyword matches are precise)
- Vector weight = 0.7 (semantic matches are approximate)
- Deduplicated by fact ID

Why RRF beats weighted combination: BM25 scores follow a long-tail distribution while vector cosine similarities cluster in a narrow 0.3-0.8 range. Linear combination is dominated by whichever score has larger variance. Rank-based fusion is stable across distributions.

### Stage 3: Composite Scoring
Each result's final score:
```
score = w_sim × similarity + w_decay × decay_score + w_imp × importance
```
Default weights: `[0.5, 0.3, 0.2]` (configurable)

Where `decay_score` uses the ideal formula:
```
decay = importance × max(0.1, exp(-0.16 × age_days / (1 + 0.3 × ln(1 + access_count))))
```

### Stage 4: Cross-Encoder Rerank (deep mode only)
When `--deep` flag is set:
- Top 20 candidates from Stage 3
- Gemini `generateContent` with reranking prompt
- Joint query-document encoding for fine-grained relevance
- ~200ms additional latency

## Real Scores (tested on production data)

Query: "steps easing hold interpolation"
```
0.736  Bug 16: steps() easing early return to prevent Hold overwrite
0.724  16: steps() easing early return (prevent Hold→EaseOut)
0.681  test_keyframe_interpolation.cpp: 15 tests, 114 assertions
0.529  opacity convention is 0-100 (unrelated — low score)
0.508  Porter-Duff compositing (unrelated — lowest)
```
Gap: 0.228 between most and least relevant.

## Entity Search

Uses the entity graph first, FTS fallback:
1. `find_entities(name)` — case-insensitive LIKE on entity name + alias table
2. `get_entity_facts(entity_id)` — traverse edges to find connected facts
3. If no graph results: fall back to FTS5 keyword search

## Filters
- `pinned_only` — only return pinned facts
- `since` — only facts created after ISO date
- `categories` — filter by fact/command/debug/workflow
- `limit` — cap results
- `deep` — enable cross-encoder reranking
