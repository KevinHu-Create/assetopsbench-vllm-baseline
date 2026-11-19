#!/usr/bin/env bash
# Run only on the selected Linux NVIDIA GPU host; does not create cloud resources.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvidia-smi >/dev/null
nvidia-smi
python3 -c 'import sys; assert sys.version_info >= (3, 11), "Python >= 3.11 is required"'
if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  if ! python3 -m venv .venv; then
    # Colab can omit the stdlib ensurepip wheel bundle. Seed a separate venv
    # with the registry package instead of modifying Colab's torch installation.
    python3 -m pip install virtualenv==21.15.9
    python3 -m virtualenv .venv
  fi
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu129
.venv/bin/vllm download-kernels
.venv/bin/python -m task2.environment --require-gpu --out outputs/environment.json
TASK2_PYTHON=.venv/bin/python bash scripts/prepare_original.sh
