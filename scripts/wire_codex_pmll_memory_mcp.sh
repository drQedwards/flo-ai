#!/usr/bin/env bash
set -euo pipefail

SERVER_NAME="pmllMemory"
PMLL_REPO_URL="https://github.com/drqsatoshi/Pmll.git"
METHOD="${1:-auto}" # auto | cli | file

resolve_python_bin() {
  local candidate="${PYTHON_BIN:-}"
  if [[ -n "$candidate" ]]; then
    echo "$candidate"
    return 0
  fi

  if command -v pyenv >/dev/null 2>&1; then
    candidate="$(pyenv root)/versions/3.11.14/bin/python"
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  return 1
}

ensure_pmll_installed() {
  local python_bin="$1"

  if ! "$python_bin" -c "import pmll_memory_mcp" >/dev/null 2>&1; then
    echo "pmll_memory_mcp is not installed for: $python_bin" >&2
    echo "Install one of the following:" >&2
    echo "  $python_bin -m pip install pmll-memory-mcp" >&2
    exit 1
  fi
}

wire_with_cli() {
  local python_bin="$1"
  codex mcp remove "$SERVER_NAME" >/dev/null 2>&1 || true
  codex mcp add "$SERVER_NAME" "$python_bin" -m pmll_memory_mcp.server
  echo "Configured Codex MCP server '${SERVER_NAME}' using Codex CLI"
  echo "Using command: ${python_bin} -m pmll_memory_mcp.server"
}

wire_with_file() {
  local python_bin="$1"
  local config_dir="${HOME}/.codex"
  local config_path="${config_dir}/config.toml"

  mkdir -p "$config_dir"

  local tmp_file
  tmp_file="$(mktemp)"
  if [[ -f "$config_path" ]]; then
    cp "$config_path" "$tmp_file"
  else
    : > "$tmp_file"
  fi

  python3 - "$tmp_file" "$SERVER_NAME" "$python_bin" <<'PY'
from pathlib import Path
import re
import sys

config_path = Path(sys.argv[1])
server_name = sys.argv[2]
python_bin = sys.argv[3]

text = config_path.read_text(encoding="utf-8")
pattern = re.compile(rf"(?ms)^\[mcp_servers\.{re.escape(server_name)}\]\n.*?(?=^\[|\Z)")
block = (
    f"[mcp_servers.{server_name}]\n"
    f"command = \"{python_bin}\"\n"
    "args = [\"-m\", \"pmll_memory_mcp.server\"]\n"
)

if pattern.search(text):
    text = pattern.sub(block + "\n", text).rstrip() + "\n"
else:
    if text and not text.endswith("\n"):
        text += "\n"
    text += "\n" + block

config_path.write_text(text, encoding="utf-8")
print(f"Wrote MCP server '{server_name}' to {config_path}")
PY

  mv "$tmp_file" "$config_path"

  echo "Configured Codex MCP server '${SERVER_NAME}' in ${config_path}"
  echo "Using command: ${python_bin} -m pmll_memory_mcp.server"
}

main() {
  local python_bin
  python_bin="$(resolve_python_bin)" || {
    echo "Could not determine a runnable Python binary." >&2
    exit 1
  }

  if [[ ! -x "$python_bin" ]]; then
    echo "Resolved Python binary is not executable: $python_bin" >&2
    exit 1
  fi

  ensure_pmll_installed "$python_bin"

  case "$METHOD" in
    auto)
      if command -v codex >/dev/null 2>&1; then
        wire_with_cli "$python_bin"
      else
        wire_with_file "$python_bin"
      fi
      ;;
    cli)
      if ! command -v codex >/dev/null 2>&1; then
        echo "codex CLI not found. Install Codex CLI or run: $0 file" >&2
        exit 1
      fi
      wire_with_cli "$python_bin"
      ;;
    file)
      wire_with_file "$python_bin"
      ;;
    *)
      echo "Usage: $0 [auto|cli|file]" >&2
      exit 1
      ;;
  esac
}

main "$@"
