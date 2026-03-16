# Changelog

## v2.0.0 (2026-03-21) — State-of-the-Art Rewrite

### Breaking Changes
- `strength` field removed from MemoryFact → replaced by `importance` (0.0-1.0) + `decay_score()` method
- Decay formula changed: old `strength * 0.95/day` → new `importance × exp(-λ × age / (1 + α × ln(1 + access)))` with floor 0.1

### New: 16 CLI Commands (0 token overhead)
Instead of MCP tool definitions (17K tokens), agent calls via Bash:
- `mnemonic save` — save with --pin, --ttl, --importance, --supersedes
- `mnemonic search` — RRF fusion search with --pinned, --since, --deep
- `mnemonic entity` — entity graph search with alias resolution
- `mnemonic graph` — bidirectional graph traversal
- `mnemonic context` — session context with --budget and memory pressure
- `mnemonic pin` / `unpin` — manage permanent facts
- `mnemonic supersede` — bi-temporal invalidation
- `mnemonic consolidate` — decay + dual-layer dedup + LLM merge
- `mnemonic reflect` — Gemini synthesis of insights
- `mnemonic bulk` — batch archive/delete/pin
- `mnemonic pinned` — list permanent facts
- `mnemonic observe` — auto-learn from stdin JSON
- `mnemonic serve` — MCP mode (backwards compatible)

### New: Bi-Temporal Timestamps (Zep/Graphiti pattern)
4 timestamps per fact: `created_at`, `updated_at`, `valid_at`, `invalid_at`. Superseded facts marked invalid, never deleted. Enables "what was true at time X" queries.

### New: Composite Scoring (CrewAI pattern)
`score = 0.5 × similarity + 0.3 × decay + 0.2 × importance`. Configurable weights via `[scoring]` config.

### New: RRF Fusion Search (Zep/Graphiti pattern)
Parallel BM25 (FTS5) + vector cosine retrieval → Reciprocal Rank Fusion with k=60. BM25 weight=1.0, vector weight=0.7. Replaces naive waterfall merge.

### New: Ideal Decay Formula (mem7/YourMemory pattern)
`importance × exp(-0.16 × age_days / (1 + 0.3 × ln(1 + access_count)))` with floor 0.1. Access count reinforces memories — frequently searched facts decay slower.

### New: Entity Graph + Aliases
- `entities` + `entity_edges` + `entity_aliases` tables
- Gemini-based entity extraction on save (files, classes, concepts, bugs)
- Alias resolution: "comp_renderer.cpp" = "the renderer" = "CR module"
- Bidirectional graph traversal

### New: Importance Scoring
0.0-1.0 per fact. Set manually via `--importance` or LLM-inferred. Factors into composite search score and decay.

### New: Memory Pressure
Configurable warn threshold (0.7) and auto-compact threshold (1.0). Context output truncated when exceeding token budget.

### Improved: Auto-Observe
- Broader tool matching (strips `mcp__` prefix, matches vm_exec/cmake/scp/etc.)
- Learns from FAILURES (OOM, timeout) not just successes
- Error→fix pairing with Gemini root cause inference
- Ignores boring Read/Grep calls

### Improved: Consolidation
- Dual-layer dedup: 0.98 cosine instant (no LLM) + 0.85 LLM merge
- Actually merges facts via Gemini (was "0 merged" in v1)
- Dry-run mode

### MCP: 12 Tools (was 4)
All original tools preserved + 8 new: `pin_memory`, `unpin_memory`, `search_entity`, `consolidate_memory`, `reflect`, `graph_query`, `list_pinned`, `bulk_manage`.

### Test Results
- 24/24 brutal test (synthetic data)
- 24/28 real data test (498 production facts)
- 5/5 context quality score
- Similarity: 0.736 > 0.724 > 0.681 > 0.529 > 0.508 (was all 1.0)

---

## v1.0.0 (2026-03-18)

- 9 MCP tools: save, search, get_context, observe, pin, unpin, search_entity, consolidate, reflect
- Memory decay (0.95^days), strengthening (+0.1 per search hit)
- Pinned memories, entity resolution, temporal queries
- Consolidation (prune/merge/boost/dedup)
- Reflect (Gemini synthesizes insights)
- Schema v4 with backward-compatible migrations
- S3 cloud sync via SQLite backup API

## v0.3.2 (2026-03-15)

- Initial release
- 4 MCP tools: save_memory, search_memory, get_context, observe_tool_call
- 3-tier search: FTS5 → cached vector → live Gemini embedding
- SQLite WAL with FTS5
- PostToolUse hook for auto-learning
- npm package with pre-built binaries (darwin-arm64, linux-arm64, linux-x64)
