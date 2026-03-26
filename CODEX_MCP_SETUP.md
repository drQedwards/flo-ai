# Codex MCP wiring for `pmll-memory-mcp`

This repository includes a helper script to wire the `pmll-memory-mcp` server into Codex's MCP configuration.

## Prerequisites

- Python 3.11+.
- Codex client configured on your machine.
- `pmll-memory-mcp` installed in the same Python environment used for the MCP command.
- `uv >= 0.8.6` if you want to run the same validation workflow used in CI.

## Install `pmll-memory-mcp`

Install from PyPI:

```bash
python -m pip install pmll-memory-mcp
```

### About `https://github.com/drqsatoshi/Pmll.git`

That repository currently does **not** contain Python packaging metadata (`pyproject.toml` or `setup.py`), so `pip install git+https://github.com/drqsatoshi/Pmll.git` fails. Use the PyPI package command above unless that repo adds packaging files.

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

## Verify MCP wiring

```bash
codex mcp list
```

You should see `pmllMemory` listed.

## Optional: run package validation tests

```bash
export PATH=$HOME/.local/bin:$PATH
cd flo_ai
uv sync
uv build
uv run pytest -m "not (integration)"
```
