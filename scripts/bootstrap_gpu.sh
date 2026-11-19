#!/usr/bin/env bash
# Run only on the selected Linux NVIDIA GPU host; does not create cloud resources.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvidia-smi >/dev/null
nvidia-smi
python3 -c 'import sys; assert sys.version_info >= (3, 11), "Python >= 3.11 is required"'
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu129
.venv/bin/vllm download-kernels
.venv/bin/python -m task2.environment --require-gpu --out outputs/environment.json
TASK2_PYTHON=.venv/bin/python bash scripts/prepare_original.sh
