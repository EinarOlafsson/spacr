#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n703-debian-adapter-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
ulimit -c 0
status=0
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock tools/run_capped.sh 8G /home/olafsson/anaconda3/bin/python -m pytest -q tests/test_runtime_i18n_syntax.py::test_french_reviewed_runtime_text_is_source_bound_and_gate_clean tests/test_runtime_i18n_syntax.py::test_swedish_reviewed_runtime_text_is_source_bound_and_gate_clean > /mnt/wd4tb/scratch/ci-current-runtime-review-contracts-20261015/retirement-final.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/ci-current-runtime-review-contracts-20261015/retirement-final.exit
exit "$status"
