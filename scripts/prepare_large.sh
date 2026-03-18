#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/prepare_large.sh [extra mono-lm-dataset args]

Bootstrap the raw large corpus if needed, then build the large dataset defined
in configs/dataset/large_local.toml.

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
bootstrap_marker="$repo_root/data/raw/.mono_lm_bootstrap_large.json"

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

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$repo_root"
echo "Fetching large raw data if needed"
"$repo_root/scripts/fetch_large_data.sh"
config_path="$repo_root/configs/dataset/large_local.toml"
if [[ -f "$bootstrap_marker" ]]; then
  config_path="$repo_root/configs/dataset/large_bootstrap.toml"
  echo "Detected bootstrapped raw corpus; using $config_path"
fi
echo "Preparing large dataset with $config_path"
exec "$python_bin" -m mono_lm.dataset_pipeline build --config "$config_path" "$@"
