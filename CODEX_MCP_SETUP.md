# Codex MCP wiring for `pmll-memory-mcp`

This repository includes a helper script to wire the `pmll-memory-mcp` server into Codex's MCP configuration.

## Prerequisites

- Python 3.11+ with `pmll-memory-mcp` installed.
- Codex client configured on your machine.

## Preferred (CLI) wiring

If you have the Codex CLI, run:

```bash
./scripts/wire_codex_pmll_memory_mcp.sh cli
```

This runs the equivalent of:

```bash
codex mcp remove pmllMemory
codex mcp add pmllMemory /path/to/python -m pmll_memory_mcp.server
```

## Fallback (direct config file)

If CLI is unavailable:

```bash
./scripts/wire_codex_pmll_memory_mcp.sh file
```

`auto` mode (default) will use CLI if available, else file mode:

```bash
./scripts/wire_codex_pmll_memory_mcp.sh
```

The file mode writes/updates this block in `~/.codex/config.toml`:

```toml
[mcp_servers.pmllMemory]
command = "/path/to/python"
args = ["-m", "pmll_memory_mcp.server"]
```

## Verify

```bash
codex mcp list
```

You should see `pmllMemory` listed.
