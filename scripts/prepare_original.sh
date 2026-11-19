#!/usr/bin/env bash
# Download a fixed IBM source snapshot and generate a separate model-only workload.
set -euo pipefail
cd "$(dirname "$0")/.."
TASK2_PYTHON="${TASK2_PYTHON:-python3}"
TASK2_IBM_COMMIT=0d044061ab3ffd4b213ddf6a0868c360d7e9667c
if [[ ! -d upstream/AssetOpsBench/.git ]]; then
  git clone https://github.com/IBM/AssetOpsBench.git upstream/AssetOpsBench
fi
# Never reset or discard local edits in an existing checkout.
if [[ -n "$(git -C upstream/AssetOpsBench status --porcelain)" ]]; then
  echo 'The upstream checkout has local edits; inspect them before proceeding.' >&2
  exit 2
fi
git -C upstream/AssetOpsBench checkout --detach "$TASK2_IBM_COMMIT"
"$TASK2_PYTHON" -m task2.prepare --upstream upstream/AssetOpsBench --out workloads/original
