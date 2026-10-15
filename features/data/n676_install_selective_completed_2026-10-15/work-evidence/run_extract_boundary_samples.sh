#!/usr/bin/env bash
set -euo pipefail
: "${SOURCE_COMMIT:?Actual frozen K2 required}"
export SOURCE_COMMIT CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
ROOT=/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014
PY=/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/render-py312/bin/python
exec "$ROOT/tools/run_capped.sh" 4G "$PY" "$(dirname "$0")/extract_boundary_samples.py"
