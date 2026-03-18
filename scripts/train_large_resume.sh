#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train_large_resume.sh [extra mono-lm-train args]

Resume the large baseline training run from the latest checkpoint defined by
configs/training/baseline.toml.

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

config_path="$repo_root/configs/training/baseline.toml"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

mapfile -t config_paths < <(
  "$python_bin" - "$repo_root" "$config_path" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
config_path = Path(sys.argv[2])
sys.path.insert(0, str(repo_root / "src"))

from mono_lm.training.config import load_training_config

config = load_training_config(config_path)
artifact_dir = config.data.artifact_dir
print(artifact_dir if artifact_dir is not None else "")
print(config.run.output_dir)
PY
)

artifact_dir="${config_paths[0]}"
run_dir="${config_paths[1]}"
latest_checkpoint="$run_dir/checkpoints/latest.pt"

if [[ -n "$artifact_dir" ]]; then
  train_text_path="$artifact_dir/final/train.txt"
  validation_text_path="$artifact_dir/final/validation.txt"
  if [[ ! -f "$train_text_path" || ! -f "$validation_text_path" ]]; then
    echo "error: expected prepared large dataset artifacts under $artifact_dir/final." >&2
    echo "Run scripts/prepare_large.sh before resuming training." >&2
    exit 1
  fi
fi

if [[ ! -f "$latest_checkpoint" ]]; then
  echo "error: no latest checkpoint found at $latest_checkpoint" >&2
  echo "Start the large training run once before using this resume script." >&2
  exit 1
fi

cd "$repo_root"
echo "Resuming large baseline training from $latest_checkpoint"
exec "$python_bin" -m mono_lm.training train --config "$config_path" --resume latest "$@"
