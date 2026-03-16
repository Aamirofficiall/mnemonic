# Integration Guide — mnemonic v2.0

## Claude Code (CLI mode — recommended)

Add to your project's `CLAUDE.md`:

```markdown
## Memory

This project uses mnemonic for persistent memory. Available commands:

- `Bash("mnemonic search 'query'")` — search memory (JSON output)
- `Bash("mnemonic save 'fact' --cat gotcha --pin")` — save a fact
- `Bash("mnemonic entity filename")` — find facts about a file/class
- `Bash("mnemonic graph entity")` — entity relationship graph
- `Bash("mnemonic context --budget 5000")` — session context
- `Bash("mnemonic reflect --topic 'topic'")` — synthesize insights

At session start, run: `Bash("mnemonic context --budget 5000")`
```

Zero MCP overhead. No `.mcp.json` needed. No tool definitions.

## Claude Code (MCP mode — backwards compatible)

`.mcp.json`:
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

12 tools available: `search_memory`, `save_memory`, `get_context`, `observe_tool_call`, `pin_memory`, `unpin_memory`, `search_entity`, `consolidate_memory`, `reflect`, `graph_query`, `list_pinned`, `bulk_manage`.

## Cursor

Same MCP config as Claude Code. Add to `.cursor/mcp.json`.

## Custom Agents

### CLI integration (any agent that can run shell commands)
```python
import subprocess, json

def memory_search(query, limit=5):
    r = subprocess.run(["mnemonic", "search", query, "--limit", str(limit)],
                       capture_output=True, text=True)
    return json.loads(r.stdout)

def memory_save(content, category="insight", pin=False):
    cmd = ["mnemonic", "save", content, "--cat", category]
    if pin: cmd.append("--pin")
    subprocess.run(cmd, capture_output=True)
```

### PostToolUse hook (auto-learning)
```bash
# ~/.claude/hooks/mnemonic-hook.sh
#!/bin/bash
cat | mnemonic observe
```

## Docker / CI

```dockerfile
COPY --from=builder /mnemonic /usr/local/bin/mnemonic
ENV GEMINI_API_KEY=your-key
RUN mnemonic init
```

Memory persists in `~/.mnemonic/<project-hash>/memory.db`. Mount as volume for persistence across container restarts.

## npm install

```bash
npm install -g mnemonic-ai
```

Ships pre-built binaries for darwin-arm64, linux-arm64, linux-x64.
