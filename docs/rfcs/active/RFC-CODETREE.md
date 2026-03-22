# RFC: mnemonic codetree — Code Intelligence for AI Agents

**Status:** DRAFT
**Author:** Aamir Shahzad
**Date:** 2026-03-22
**Context:** After building the memory engine (v2.0.2), agents still hallucinate function names, guess parameter types, and don't know file dependencies. Code tree solves this by giving the agent a structured, token-efficient map of the entire codebase.

---

## Problem

AI coding agents waste tokens and hallucinate because they lack structural understanding of the codebase:

1. **No function awareness** — agent guesses function names, parameters, return types
2. **No dependency knowledge** — doesn't know which files import which
3. **Token waste** — reads entire 2000-line files to find one function signature
4. **No incremental updates** — re-reads everything every session
5. **No connection to memory** — past bug fixes don't connect to the functions they fixed

**Measured cost:** A 50-file project is 100K-300K tokens raw. Aider's repo map compresses this to ~1,024 tokens (99.3% reduction) while preserving all structural information an agent needs.

---

## Solution

Add `mnemonic codetree` command that:
1. Parses source code using tree-sitter (170+ languages, sub-ms incremental)
2. Extracts definitions (functions, classes, structs, methods) and references (calls, imports)
3. Builds a file dependency graph with PageRank ranking
4. Stores everything in the entity graph (same SQLite DB as memory)
5. Outputs a token-budgeted code map the agent reads instead of source files

---

## Architecture

### Parsing: Tree-sitter

Tree-sitter is the clear winner based on research across 15+ tools:
- 170+ language grammars
- Sub-millisecond incremental parsing
- Handles broken/partial code (error-tolerant)
- Used by: Aider, Cursor, Continue.dev, Cline, Repomix, GitHub, Copilot

NOT using LSP because:
- LSP expects an open editor, fails on cold repos
- Claude Code team tested LSP: "symbol resolution failures, 8.5% higher token consumption without quality improvement"
- Requires running language servers (infrastructure overhead)

### Tag Extraction

Tree-sitter `.scm` query files extract definitions and references. Each language has a small query file (10-60 lines):

**Python:**
```scheme
(function_definition name: (identifier) @name.definition.function) @definition.function
(class_definition name: (identifier) @name.definition.class) @definition.class
(call function: (identifier) @name.reference.call) @reference.call
```

**Rust:**
```scheme
(struct_item name: (type_identifier) @name.definition.class) @definition.class
(function_item name: (identifier) @name.definition.function) @definition.function
(impl_item trait: (type_identifier) @name.reference.implementation) @reference.implementation
```

Supported languages (priority order):
1. **Rust** — structs, enums, functions, methods, traits, modules, macros, impls
2. **C/C++** — structs, classes, functions, methods, typedefs, enums
3. **Python** — classes, functions, calls
4. **Swift** — classes, protocols, methods, properties, functions
5. **TypeScript/JavaScript** — classes, functions, methods, interfaces

### Graph Construction (Aider pattern)

After extracting tags from all files:

1. Build a directed graph where **nodes = files** and **edges = cross-file references**
2. For each identifier defined in file A and referenced in file B, create edge B → A
3. Weight edges by importance:

| Condition | Weight |
|---|---|
| Base | 1.0 |
| Long name (8+ chars, camelCase/snake_case) | x10 |
| Private (starts with `_`) | x0.1 |
| Defined in >5 files (too generic) | x0.1 |
| Recently modified file | x5 |
| Referenced in agent's current context | x50 |

4. Run PageRank on the weighted graph to rank files by importance
5. Distribute file rank across symbols proportionally to edge weights

### Token Budget Fitting (Aider pattern)

Default budget: 1,024 tokens. Binary search to find max symbols that fit:

1. Start at `min(budget / 25, num_symbols)`
2. Render top-N symbols with file context
3. Count tokens (sample 1% of lines for texts >200 chars)
4. Adjust N until output is within 15% of budget

### Storage: Entity Graph

Code tree data stored in the same SQLite DB as memory facts:

```sql
-- Reuse existing tables
entities (id, name, entity_type, created_at)
-- entity_type: "file", "function", "class", "struct", "method", "trait", "module"

entity_edges (entity_id, fact_id_or_symbol_id, relation)
-- relation: "defines", "references", "imports", "calls", "implements"

-- New table for code symbols
code_symbols (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,        -- function, class, struct, method, trait, module, macro
    signature TEXT,            -- full signature line from source
    line_start INTEGER,
    line_end INTEGER,
    parent_symbol TEXT,        -- for methods: the class/struct/impl they belong to
    content_hash TEXT,         -- SHA-256 of the symbol's source (for incremental)
    updated_at TEXT
);

-- File-level metadata
code_files (
    path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,  -- SHA-256 of file content
    language TEXT NOT NULL,
    symbol_count INTEGER,
    line_count INTEGER,
    last_parsed TEXT,
    pagerank REAL DEFAULT 0.0
);

-- Cross-file dependencies
code_deps (
    from_file TEXT NOT NULL,
    to_file TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    PRIMARY KEY (from_file, to_file, symbol_name)
);
```

### Connection to Memory

When an agent saves a memory fact mentioning a function or file:
- Entity extraction links the fact to the code_symbol
- `mnemonic entity build_inverse_matrix` returns BOTH the code signature AND past bug fixes
- `mnemonic graph renderer.cpp` shows BOTH file dependencies AND memory facts

---

## CLI Commands

### `mnemonic codetree`

Parse and index source code:

```bash
# Parse a directory (initial or incremental)
mnemonic codetree src/
# Output: {"files": 50, "symbols": 340, "deps": 120, "time_ms": 450}

# Parse with explicit language
mnemonic codetree src/ --lang rust

# Only re-parse changed files
mnemonic codetree src/ --update
# Output: {"changed": 3, "unchanged": 47, "symbols_updated": 12, "time_ms": 45}
```

### `mnemonic codemap`

Generate a token-budgeted code map (the agent reads this):

```bash
# Default: compact tree, 1024 token budget
mnemonic codemap src/
# Output:
# src/renderer.cpp:
#   class GpuRenderer
#     fn render_frame(scene: &Scene) -> Frame
#     fn flush_cache()
#     fn build_inverse_matrix(pos, anchor, rot) -> Mat3x3
# src/scene_compiler.cpp:
#   class SceneCompiler
#     fn compile(json: Value) -> Scene
#     fn process_keyframes(elem) -> Vec<Keyframe>
# [50 files, 340 symbols, 120 deps]

# Custom token budget
mnemonic codemap src/ --budget 2048

# JSON output
mnemonic codemap src/ --format json

# XML output (some LLMs prefer this)
mnemonic codemap src/ --format xml

# Only show one file in detail
mnemonic codemap src/ --file renderer.cpp

# Drill into one function
mnemonic codemap src/ --file renderer.cpp --symbol build_inverse_matrix
# Output:
# fn build_inverse_matrix(pos: Vec2, anchor: Vec2, rot: f32) -> Mat3x3
#   params:
#     pos: Vec2 — element position offset
#     anchor: Vec2 — transform origin
#     rot: f32 — rotation in radians
#   calls: cos(), sin(), scale_x(), scale_y()
#   called_by: render_shape_layer_transformed(), apply_transform()
#   lines: 1523-1548
#   related_facts: Bug 31 (pos+anchor fix), Bug 32 (cache store position)
```

### `mnemonic codedeps`

Show file dependency graph:

```bash
mnemonic codedeps src/
# Output:
# renderer.cpp -> gpu_context.h, scene_compiler.cpp, transform.comp
# scene_compiler.cpp -> composition.cpp, properties.cpp
# gpu_transform.cpp -> gpu_context.h, transform.comp

# Show what depends on a file (reverse deps)
mnemonic codedeps src/ --reverse renderer.cpp
# Output:
# renderer.cpp is used by: main.cpp, test_renderer.cpp, precomp.cpp
```

---

## Output Formats

### Tree format (default, ~500-1000 tokens for 50 files)

```
src/
  renderer.cpp: GpuRenderer{render_frame(), flush_cache(), build_inverse_matrix()}
  scene_compiler.cpp: SceneCompiler{compile(), process_keyframes(), add_layer()}
  gpu_context.h: GpuContext{shared(), create_buffer(), submit_command()}
  properties.cpp: AnimProperty{evaluate(), get_value()}
  composition.cpp: Composition{add_layer(), precompose(), render()}
[50 files, 340 symbols]
```

### JSON format

```json
{
  "files": [{
    "path": "src/renderer.cpp",
    "language": "cpp",
    "pagerank": 0.85,
    "symbols": [{
      "name": "GpuRenderer",
      "kind": "class",
      "line": 45,
      "children": [{
        "name": "render_frame",
        "kind": "method",
        "signature": "Frame render_frame(const Scene& scene)",
        "line": 120
      }]
    }],
    "imports": ["gpu_context.h", "scene_compiler.cpp"],
    "imported_by": ["main.cpp", "test_renderer.cpp"]
  }]
}
```

### XML format

```xml
<codetree files="50" symbols="340">
  <file path="src/renderer.cpp" language="cpp" rank="0.85">
    <class name="GpuRenderer">
      <method name="render_frame" signature="Frame render_frame(const Scene&amp; scene)" line="120"/>
      <method name="flush_cache" signature="void flush_cache()" line="200"/>
    </class>
  </file>
</codetree>
```

---

## Incremental Updates

### Strategy: Content hash + mtime

1. On `mnemonic codetree src/`:
   - List all source files
   - For each file, check `mtime` against `code_files.last_parsed`
   - If mtime changed, compute SHA-256 hash
   - If hash differs from `code_files.content_hash`, re-parse
   - Otherwise skip (unchanged)

2. On git commit (post-commit hook):
   - `mnemonic codetree src/ --update` runs automatically
   - Only re-parses files in the commit diff
   - Updates PageRank for affected files

3. Performance targets:
   - Initial parse of 50 files: < 2 seconds
   - Incremental update of 3 changed files: < 100ms
   - Token budget rendering: < 50ms

---

## Implementation Plan

### Phase 1: Tree-sitter parsing + symbol extraction

1. Add `tree-sitter` and language grammar crates to Cargo.toml
2. Create `src/codetree.rs` module
3. Implement tag extraction using `.scm` query files
4. Store symbols in `code_symbols` table
5. CLI command: `mnemonic codetree src/`

### Phase 2: Dependency graph + PageRank

1. Build cross-file reference graph from definition/reference tags
2. Implement PageRank (can use simple power iteration, no external dep needed)
3. Store file ranks in `code_files` table
4. Store deps in `code_deps` table

### Phase 3: Token-budgeted output

1. Binary search fitting algorithm (from Aider)
2. Tree/JSON/XML output formats
3. CLI command: `mnemonic codemap src/`
4. Drill-down: `--file` and `--symbol` flags

### Phase 4: Memory integration

1. Connect code symbols to entity graph
2. `mnemonic entity function_name` returns code signature + memory facts
3. `mnemonic graph file.cpp` shows deps + related memories

### Phase 5: Auto-update

1. Post-commit hook: `mnemonic codetree --update`
2. Incremental parsing via content hash
3. PageRank recalculation on changed subgraph

---

## Dependencies

### Rust crates needed

```toml
tree-sitter = "0.24"
tree-sitter-rust = "0.23"
tree-sitter-cpp = "0.23"
tree-sitter-python = "0.23"
tree-sitter-c = "0.23"
# Swift and TypeScript via tree-sitter-language-pack or individual crates
sha2 = "0.10"              # for content hashing
```

### .scm query files

Bundle the 7 query files (python, rust, cpp, c, swift, typescript, javascript) directly in the binary via `include_str!()`. Total: ~6KB. No external files needed.

---

## Verification Checklist

### Phase 1
- [ ] `mnemonic codetree src/` parses Rust files and extracts functions, structs, impls
- [ ] `mnemonic codetree src/` parses C++ files and extracts classes, functions
- [ ] `mnemonic codetree src/` parses Python files and extracts classes, functions
- [ ] Symbols stored in SQLite `code_symbols` table
- [ ] Incremental: only re-parses changed files (by mtime + hash)

### Phase 2
- [ ] Cross-file references detected (function defined in A, called in B)
- [ ] PageRank computed for all files
- [ ] Dependencies stored in `code_deps` table
- [ ] `mnemonic codedeps src/` shows dependency graph

### Phase 3
- [ ] `mnemonic codemap src/` outputs compact tree in ~1024 tokens for 50 files
- [ ] `mnemonic codemap src/ --format json` outputs structured JSON
- [ ] `mnemonic codemap src/ --file renderer.cpp` shows one file detail
- [ ] `mnemonic codemap src/ --file renderer.cpp --symbol build_inverse_matrix` shows function detail
- [ ] Token budget binary search works (within 15% of target)

### Phase 4
- [ ] `mnemonic entity function_name` returns code signature + related memory facts
- [ ] `mnemonic graph file.cpp` shows file deps + memory facts together
- [ ] Saving a memory fact about a function auto-links to code_symbol

### Phase 5
- [ ] Post-commit hook triggers `mnemonic codetree --update`
- [ ] 3 changed files re-parse in < 100ms
- [ ] PageRank recalculated only for affected files

---

## What This Enables

With codetree + memory combined, an agent session starts with:

```bash
mnemonic context --budget 2000     # what happened before (memory)
mnemonic codemap src/ --budget 1000 # what the code IS (structure)
```

3,000 tokens total. The agent knows:
- Every file, function, class in the project
- Every gotcha, convention, bug fix from past sessions
- Every dependency between files
- Which functions had bugs and what the fixes were

No hallucination. No reading 50 files. No guessing function names.

---

## Research Sources

- **Aider repo map** (repomap.py) — PageRank, tree-sitter, token budget binary search
- **Repomix** — tree-sitter compress mode (70% token reduction)
- **Continue.dev** — AST-aware chunking, signature extraction
- **Cursor** — Merkle tree incremental updates
- **GitNexus** — post-commit hooks, knowledge graph integration
- **Greptile** — NL descriptions improve retrieval 12%
- **Cody/Sourcegraph** — SCIP for compiler-accurate indexing (future upgrade path)
- Research doc: 741 sources analyzed across 15+ tools
