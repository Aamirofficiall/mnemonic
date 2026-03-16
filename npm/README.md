# mnemonic-ai

> Persistent memory for AI coding agents. Remembers what matters, forgets what doesn't.

mnemonic-ai gives your AI agent long-term memory across sessions. It learns from every interaction, builds a knowledge graph of your codebase, and surfaces the right context when you need it.

```bash
npm install -g mnemonic-ai
mnemonic init
```

Works with **Claude Code**, **Cursor**, **Windsurf**, and any MCP-compatible client.

---

## What Makes It Different

Most memory tools are simple vector stores - append facts, search by similarity, done. That breaks down fast: duplicates pile up, outdated info poisons results, everything scores the same, and after 200+ memories the context is useless noise.

mnemonic-ai solves the problems that matter at scale:

### Multi-Signal Search

Search isn't just "find the closest vector." mnemonic-ai runs **keyword matching and semantic similarity in parallel**, then fuses results using rank-based scoring that's stable across different score distributions. Results are then weighted by three signals - how relevant it is, how recent it is, and how important it is - so you get the fact that actually helps, not just the one with the closest embedding.

```bash
$ mnemonic search "GPU performance optimization" --limit 3

# Returns results ranked by composite relevance:
# 0.82  "GPU Sprint: 1300ms → 40ms per frame (32x faster)"     ← most relevant
# 0.61  "DEVICE_LOCAL memory: 21x speedup for GPU transfers"   ← related
# 0.49  "cmake build takes ~60s with -j4"                      ← tangentially related
```

### Memory That Fades - Intelligently

Memories aren't permanent by default. They decay over time using a reinforcement model: facts you access frequently stay strong, facts nobody searches for gradually fade. The decay accounts for importance - a critical gotcha decays slower than a debug note, even if neither is accessed.

Pin facts that must never fade. Set TTL for notes that should auto-expire. The result: after months of use, your context contains the 30 facts that actually matter, not 3,000 that don't.

### Knowledge Graph

Every saved fact is automatically analyzed for **entities** - file names, class names, concepts, bug IDs. These become nodes in a graph with bidirectional edges. When you query a file, you don't just get facts that mention it - you get connected files, related concepts, and linked bugs.

```bash
$ mnemonic graph comp_renderer

# Entity: comp_renderer.cpp
# Facts: 25 connected
# Connected: scene_compiler.cpp, gpu_transform.cpp, GPU cache, Porter-Duff, Bug 21, Bug 30...
```

This means you can navigate your codebase knowledge the way you think about it - by relationships, not keywords.

### Contradiction Handling

Real projects generate contradictory information: "Bug 14 is deferred" then later "Bug 14 is fixed." mnemonic-ai handles this with **bi-temporal tracking** - every fact records when it was learned and when it stopped being true. When you supersede a fact, the old version is marked invalid and hidden from all searches and context, but the history is preserved. Chain supersedes work: session 1 → session 2 → session 3, only the latest understanding is returned.

### Auto-Learning From Tool Usage

Pipe your tool call outcomes to mnemonic-ai and it learns automatically:
- Build timing patterns ("cmake takes ~60s")
- Error → fix pairs with root cause inference
- Failure patterns ("accuracy tests cause OOM - run per-suite")
- File hotspots (which files get touched most)

It's smart about what to learn: a 58-second build is interesting, a 50ms file read is not. Boring calls are filtered out automatically.

### Dual-Layer Deduplication

Two levels of duplicate detection:
- **Fast path** - near-identical facts are caught instantly by embedding similarity, no AI call needed
- **Smart merge** - semantically similar facts (e.g., three progressive updates about the same bug) are merged into one comprehensive fact using AI, preserving all unique information

### Context That Fits

When your agent starts a new session, `mnemonic context` returns exactly what it needs to be productive - pinned critical knowledge first, then active facts by category, filtered to fit your token budget. Not a 65KB dump of everything ever saved. The context is quality-ranked: a new session reading just 5,000 tokens of mnemonic-ai context knows the key files, critical conventions, project status, how to run tests, and what not to do.

---

## Features at a Glance

| Capability | What It Does |
|---|---|
| **Semantic + keyword search** | Parallel retrieval fused by rank, not just vector similarity |
| **Composite scoring** | Results ranked by relevance + recency + importance |
| **Memory decay** | Reinforcement-based: accessed facts stay, unused ones fade |
| **Knowledge graph** | Auto-extracted entities with bidirectional traversal |
| **Entity aliases** | "comp_renderer.cpp" = "the renderer" = "CR module" |
| **Bi-temporal timestamps** | Tracks when facts were true, not just when they were saved |
| **Supersede chains** | Replace outdated info cleanly, history preserved |
| **TTL / pinning** | Ephemeral notes auto-expire, critical facts persist forever |
| **Auto-learning** | Learns from tool calls - builds, errors, timeouts, file touches |
| **Smart deduplication** | Fast hash dedup + AI-powered semantic merge |
| **Consolidation** | On-demand cleanup: decay, dedup, merge, archive |
| **Reflection** | AI synthesis of patterns across hundreds of memories |
| **Token-budgeted context** | Session context sized to fit, quality over quantity |
| **16 CLI commands** | Direct shell access, zero protocol overhead |
| **12 MCP tools** | Full MCP server for compatible clients |
| **Cloud sync** | Optional S3 backup for durability across containers |

---

## How It Compares

| | mnemonic-ai | mem0 | Zep | Letta/MemGPT | CrewAI Memory |
|---|---|---|---|---|---|
| **Search** | Multi-signal fusion (keyword + vector + rank) | Vector only | Hybrid (best raw retrieval) | Vector per tier | Vector + recency |
| **Scoring** | Composite (relevance + decay + importance) | Similarity only | Community detection | None | Composite (closest to ours) |
| **Memory decay** | Reinforcement-based with floor | None (0% stale precision) | Temporal invalidation | Implicit via summarization | Exponential decay |
| **Contradiction handling** | Bi-temporal supersede chains | LLM-based AUDN cycle | Bi-temporal invalidation (best) | None | LLM consolidation |
| **Knowledge graph** | Auto-extracted with aliases | Optional (Neo4j) | Temporal KG (best) | None | Scope tree only |
| **Deduplication** | Dual-layer (fast hash + AI merge) | LLM-dependent | 2-phase embed + LLM | None | Dual-layer (closest) |
| **Auto-learning** | Observes tool calls, learns from failures | Explicit add only | Continuous graph updates | Self-editing (best) | Post-task extraction |
| **Context management** | Token-budgeted, quality-ranked | Limit param only | Precomputed summaries | Virtual paging (best) | Scope-filtered |
| **CLI** | 16 native commands | None | MCP server only | CLI + web UI | `crewai memory` |
| **Infra required** | SQLite (zero infra) | Qdrant / 24 backends | Neo4j + Postgres | Postgres + pgvector | LanceDB |
| **Install** | `npm install -g mnemonic-ai` | `pip install mem0ai` | Docker compose | `pip install letta` | Part of CrewAI |

**Where mnemonic-ai leads:** Zero-infrastructure setup (single binary, SQLite), CLI-first design for coding agents, combined best features from across the field in one tool.

**Where others lead:** Zep has the most mature temporal knowledge graph. Letta has the most innovative self-editing memory architecture. mem0 has the largest ecosystem (47K stars, 24 vector store backends).

---

## Usage

### CLI (recommended)

```bash
# Save knowledge
mnemonic save "opacity is 0-100, not 0.0-1.0" --cat convention --pin
mnemonic save "debug: checking line 1680" --cat debug --ttl session
mnemonic save "cmake build takes ~60s with -j4" --cat pattern --importance 0.8

# Search
mnemonic search "GPU performance optimization" --limit 5
mnemonic search "opacity" --pinned --deep

# Knowledge graph
mnemonic entity comp_renderer
mnemonic graph "GPU cache"

# Session context
mnemonic context --budget 5000

# Memory management
mnemonic pin <fact_id>
mnemonic unpin <fact_id>
mnemonic pinned --cat gotcha
mnemonic supersede <old_id> "Bug 14 FIXED" --cat insight
mnemonic bulk archive --cat debug --unpinned
mnemonic consolidate --threshold 0.9 --dry-run
mnemonic reflect --topic "rendering pipeline"

# Auto-learning
echo '{"tool_name":"Bash","tool_response":{"stdout":"Built in 58s"}}' | mnemonic observe

# Info
mnemonic stats
```

### MCP Server

For MCP-compatible clients (Claude Code, Cursor, Windsurf):

```bash
mnemonic serve
```

12 tools via stdio JSON-RPC: `search_memory` · `save_memory` · `get_context` · `observe_tool_call` · `pin_memory` · `unpin_memory` · `search_entity` · `consolidate_memory` · `reflect` · `graph_query` · `list_pinned` · `bulk_manage`

**MCP config** (`.mcp.json`):
```json
{
  "mcpServers": {
    "memory": {
      "command": "mnemonic",
      "args": ["serve"],
      "env": { "GEMINI_API_KEY": "${GEMINI_API_KEY}" }
    }
  }
}
```

---

## Install

**npm** (pre-built binaries for macOS and Linux):
```bash
npm install -g mnemonic-ai
```

**From source**:
```bash
git clone https://github.com/Aamirofficiall/mnemonic
cd mnemonic
cargo install --path .
```

**Setup**:
```bash
cd your-project
mnemonic init
```

Prompts for a [Gemini API key](https://aistudio.google.com/apikey) (free), creates project config, and initializes the database.

---

## Configuration

3-layer config: defaults → `~/.mnemonic/config.toml` → `.mnemonic.toml` → env vars.

```toml
# ~/.mnemonic/config.toml
[api]
gemini_key = "your-key"       # or set GEMINI_API_KEY env var

[search]
min_similarity = 0.3
default_limit = 10

[observe]
extract_facts = true
extract_tools = ["Bash", "Edit", "Write", "Grep", "Glob"]

[decay]
half_life_days = 30           # 7 for sprints, 180 for knowledge bases

[scoring]
weight_similarity = 0.5
weight_decay = 0.3
weight_importance = 0.2
```

### Cloud Sync (optional)

Persist memory to S3 for durability across containers and CI. Build with `--features sync`.

```toml
[sync]
enabled = true
bucket = "your-bucket"
region = "us-east-1"
```

---

## Storage

Per-project SQLite database at `~/.mnemonic/<project-hash>/memory.db`. Zero infrastructure - no external database, no server process for CLI mode.

| Table | Purpose |
|---|---|
| `facts` | Knowledge with bi-temporal metadata, importance, access tracking |
| `facts_fts` | Full-text search index |
| `entities` | Knowledge graph nodes (files, classes, concepts, bugs) |
| `entity_edges` | Graph relationships (found_in, fixes, relates_to) |
| `entity_aliases` | Alternative names for entities |
| `commands` | Tool call history with outcomes |
| `debug_records` | Error → fix pairs with root cause |
| `key_files` | Most-touched files by access frequency |
| `query_cache` | Cached search embeddings for fast repeat queries |

---

## Requirements

- **Gemini API key** - free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- **Node.js 16+** (for npm install) or **Rust** (to build from source)
- macOS (arm64), Linux (arm64, x64)

---

## Author

**Aamir Shahzad** - [aamirshahzad.uk](https://aamirshahzad.uk)

## License

Apache-2.0
