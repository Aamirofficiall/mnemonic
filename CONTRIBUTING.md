# Contributing to mnemonic

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone
git clone https://github.com/Aamirofficiall/mnemonic
cd mnemonic

# Build
cargo build

# Run tests (requires Gemini API key)
export GEMINI_API_KEY="your-key"
cargo test --test rfc_upgrade_tests -- --nocapture --test-threads=1

# Run locally
cargo run -- save "test fact" --cat pattern
cargo run -- search "test"
cargo run -- stats
```

## Project Structure

```
src/
  main.rs          CLI commands + MCP serve entry point
  store.rs         SQLite operations, schema, RRF fusion
  search.rs        Search engine with composite scoring
  server.rs        MCP tool handlers
  observer.rs      Auto-learning from tool calls
  embeddings.rs    Gemini API (embed, extract, merge, reflect)
  models.rs        Data types and MCP parameter schemas
  config.rs        3-layer configuration system
  project.rs       Git root detection, DB path resolution
  sync.rs          Optional S3 cloud sync
tests/
  integration.rs       Core MCP tests
  rfc_upgrade_tests.rs All v2 feature tests
docs/                  Architecture, search, integration guides
npm/                   npm package (launcher + binaries)
```

## Making Changes

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes
4. Run tests: `cargo test --test rfc_upgrade_tests -- --nocapture --test-threads=1`
5. Submit a pull request

## Guidelines

- Keep CLI output compact (JSON for machine consumption)
- New features need tests in `tests/rfc_upgrade_tests.rs`
- All search changes must maintain composite scoring
- Don't break MCP backwards compatibility
- Run `cargo clippy` before submitting

## Reporting Issues

Open an issue at [github.com/Aamirofficiall/mnemonic/issues](https://github.com/Aamirofficiall/mnemonic/issues) with:
- What you expected
- What happened
- Steps to reproduce
- Output of `mnemonic stats`

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
