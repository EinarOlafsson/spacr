#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n703-debian-adapter-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
status=0
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock tools/run_capped.sh 8G /home/olafsson/anaconda3/bin/python -m pytest -q tests/test_docs_media_budget.py::test_docs_workflow_publishes_each_branch_to_its_own_host > /mnt/wd4tb/scratch/ci-docs-host-routing-test-20261015/final-corrected.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/ci-docs-host-routing-test-20261015/final-corrected.exit
exit "$status"
