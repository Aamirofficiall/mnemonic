# Testing — mnemonic v2.0

## Test Suites

| Suite | File | Tests | What |
|---|---|---|---|
| Integration (v1) | `tests/integration.rs` | 37 | Core MCP functionality |
| RFC Upgrade | `tests/rfc_upgrade_tests.rs` | 13 | All 10 RFC features |
| Brutal | `brutal-test.py` | 24 | User's exact spec scenarios |
| Stress (6 phases) | `stress-test.py` | 21 | Flood, contradictions, multi-file, categories, chaos, scale |
| Real Data CLI | `real-data-cli-test.py` | 28 | 498 production facts + all CLI commands |

## Running

```bash
export GEMINI_API_KEY="your-key"

# Rust integration tests (MCP mode)
cargo test --test rfc_upgrade_tests -- --nocapture --test-threads=1

# Docker tests (CLI mode, real binary)
docker run --rm -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -v ./target/release/mnemonic:/usr/local/bin/mnemonic:ro \
  -v ./brutal-test.py:/test.py:ro \
  debian:bookworm-slim bash -c "apt-get update -qq && apt-get install -y -qq git python3 ca-certificates && python3 -u /test.py"
```

## Results (2026-03-21)

### Brutal Test: 24/24 PASS
- Entity search by filename: comp_renderer→Bug30, scene_compiler→Bug16, properties→Bug19
- Bidirectional graph: GPU cache → comp_renderer confirmed
- Auto-observe: learned cmake timing + OOM + timeout from 10 calls
- Supersede: pinned fact hidden after supersede
- Similarity: 0.736 > 0.724 > 0.681 > 0.529 > 0.508 (not all 1.0)
- Consolidation: merged 3 Bug 21 pairs (sim 0.885, 0.847, 0.843)
- Bulk archive: 5 debug facts archived, pinned survived

### Real Data CLI: 24/28 PASS
- Entity: 9/12 queries return results (3 gaps: partial filename matching)
- Graph: comp_renderer → 25 facts, GPU cache → 16, bidirectional confirmed
- Context quality: 5/5 (key files, conventions, status, tests, gotchas)
- Reflect: 5292 chars of structured insights
- MCP backwards compatible: 12 tools registered

### v2 Docker: 24/25 PASS
- All 16 CLI commands work
- Composite scoring: 5 unique scores
- Bi-temporal supersede: deferred hidden, FIXED visible
- Pin/unpin lifecycle complete
- Bulk archive by category
- Observe via stdin
- MCP serve: 12 tools backwards compatible
