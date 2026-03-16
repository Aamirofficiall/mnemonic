# Architecture — mnemonic v2.0

## Dual-Mode Operation

```
Agent (Claude Code, Cursor, etc.)
  │
  ├─── CLI mode ──→ Bash("mnemonic search 'query'")
  │                    │
  │                    ├─ Opens SQLite DB directly
  │                    ├─ Runs search/save/graph operation
  │                    ├─ Prints compact JSON to stdout
  │                    └─ Exits (no persistent process)
  │
  └─── MCP mode ──→ mnemonic serve (stdio JSON-RPC)
                       │
                       ├─ Long-running process
                       ├─ 12 MCP tools via JSON-RPC
                       └─ Same engine, same DB
```

CLI mode saves **17,000 tokens** per session (no MCP tool definitions). MCP mode provided for backwards compatibility.

## Source Files

| File | Lines | Responsibility |
|---|---|---|
| `main.rs` | ~500 | CLI subcommands (16 commands) + MCP serve entry point |
| `store.rs` | ~1500 | SQLite: CRUD, FTS5, entity graph, RRF fusion, decay |
| `search.rs` | ~260 | 3-tier search engine with composite scoring |
| `server.rs` | ~650 | MCP tool handlers (12 tools) |
| `observer.rs` | ~320 | Auto-learning from tool calls, error→fix pairing |
| `embeddings.rs` | ~480 | Gemini API: embed, extract entities, merge, reflect |
| `models.rs` | ~380 | Data types, MCP params, bi-temporal MemoryFact |
| `config.rs` | ~470 | 3-layer config (defaults → global → project → env) |
| `project.rs` | ~85 | Git root detection, project hash, DB path |
| `sync.rs` | ~200 | Optional S3 cloud sync (feature-gated) |

## Data Flow

### Save Path
```
content + category + options
  ├─ 1. Create MemoryFact with bi-temporal timestamps
  ├─ 2. Insert into SQLite + FTS5 index
  ├─ 3. Handle supersede (set invalid_at on old fact)
  ├─ 4. Embed via Gemini (768-dim vector)
  ├─ 5. Extract entities via Gemini (files, classes, concepts)
  └─ 6. Create/resolve entity nodes + edges in graph
```

### Search Path (RRF Fusion)
```
query + filters
  ├─ Stage 1: Parallel FTS5 (BM25) + vector cosine retrieval
  ├─ Stage 2: RRF fusion — score = Σ w/(k+rank), k=60
  ├─ Stage 3: Composite — 0.5×sim + 0.3×decay + 0.2×importance
  └─ Stage 4 (deep): Cross-encoder rerank via Gemini
```

### Observe Path (Auto-Learning)
```
tool event → detect success → record command → track files
  → error→fix pairing → extract facts via Gemini → embed + store
```

## Key Algorithms

| Algorithm | Source | Formula |
|---|---|---|
| Decay | mem7/YourMemory | `importance × exp(-0.16 × age / (1 + 0.3 × ln(1 + access)))` floor 0.1 |
| RRF | Zep/Graphiti | `Σ w_i/(k + rank_i)` k=60, BM25=1.0, vec=0.7 |
| Composite | CrewAI | `0.5×sim + 0.3×decay + 0.2×importance` |
| Dedup | CrewAI | 0.98 cosine instant + 0.85 LLM merge |
| Bi-temporal | Zep/Graphiti | 4 timestamps: created, updated, valid_at, invalid_at |

## Database Schema

### facts
`id, content, category, confidence, created_at, updated_at, valid_at, invalid_at, source, expires_at, embedding, pinned, importance, access_count, superseded_by, ttl, session_id`

### Entity Graph
`entities(id, name, entity_type, created_at)` + `entity_edges(entity_id, fact_id, relation)` + `entity_aliases(entity_id, alias)`

## Backwards Compatibility
- v1 databases auto-migrate (ALTER TABLE ADD COLUMN)
- `superseded_by` preserved alongside `invalid_at`
- MCP `serve` unchanged — all 12 tools work as before
- New CLI commands are additive
