#!/usr/bin/env bash
# Create an isolated, exactly pinned ASR/scoring environment. Never touches base.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
wheel_path="${1:-/tmp/xinteraction-provider-wheels/funasrnano-0.2.0a6-py3-none-any.whl}"
venv_path="${2:-$repo_root/workspace/validation/longform_asr_venv}"
bootstrap_python="${PYTHON_BOOTSTRAP:-/home/rime/miniforge3/envs/funasr/bin/python}"
expected_sha256="da6ea124ee062b26336ffbbee0f358b9888588cd9091ebf036f11878916d1227"

if [[ ! -f "$wheel_path" ]]; then
  echo "FunASR wheel not found: $wheel_path" >&2
  exit 1
fi
observed_sha256="$(sha256sum "$wheel_path" | cut -d' ' -f1)"
if [[ "$observed_sha256" != "$expected_sha256" ]]; then
  echo "FunASR wheel hash mismatch: $observed_sha256" >&2
  exit 1
fi
if [[ -e "$venv_path" && ! -x "$venv_path/bin/python" ]]; then
  echo "refusing to replace non-venv path: $venv_path" >&2
  exit 1
fi
if [[ ! -x "$venv_path/bin/python" ]]; then
  "$bootstrap_python" -m venv "$venv_path"
fi

"$venv_path/bin/python" -m pip install --disable-pip-version-check \
  "$wheel_path" \
  'cn2an>=0.5.23,<1' \
  'rapidfuzz>=3,<4' \
  'soundfile>=0.12,<1' \
  'librosa>=0.10,<1'

"$venv_path/bin/python" -c \
  'import importlib.metadata as m; assert m.version("funasrnano") == "0.2.0a6"; print(m.version("funasrnano"))'
mkdir -p "$venv_path/evidence"
printf '%s  %s\n' "$observed_sha256" "$wheel_path" > "$venv_path/evidence/funasrnano-wheel.sha256"
"$venv_path/bin/python" -m pip freeze --all > "$venv_path/evidence/pip-freeze.txt"
sha256sum "$venv_path/evidence/pip-freeze.txt" > "$venv_path/evidence/pip-freeze.txt.sha256"

echo "ASR validation venv ready: $venv_path"
