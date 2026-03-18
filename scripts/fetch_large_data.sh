#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/fetch_large_data.sh [--force]

Bootstrap the raw corpus expected by the large dataset config.
This fills in the missing data/raw/... sources with a deterministic synthetic
corpus so scripts/prepare_large.sh can build end-to-end on a fresh clone.

Options:
  --force    Regenerate the bootstrap corpus even if the raw sources exist.

Environment:
  MONO_LM_PYTHON_BIN  Optional Python interpreter override.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

resolve_python_bin() {
  if [[ -n "${MONO_LM_PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$MONO_LM_PYTHON_BIN"
    return
  fi
  if [[ -x "$repo_root/.venv/bin/python" ]]; then
    printf '%s\n' "$repo_root/.venv/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "error: no Python interpreter found. Set MONO_LM_PYTHON_BIN or create $repo_root/.venv." >&2
  exit 1
}

python_bin="$(resolve_python_bin)"
if [[ ! -x "$python_bin" ]]; then
  echo "error: Python interpreter is not executable: $python_bin" >&2
  exit 1
fi

cd "$repo_root"
exec "$python_bin" "$repo_root/scripts/bootstrap_large_raw_data.py" "$@"
