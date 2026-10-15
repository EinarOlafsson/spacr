#!/usr/bin/env bash
set -euo pipefail
TASK_DIR=/mnt/wd4tb/scratch/candidate-1514-b51fb194bb-20261015/ci-artifact-verification
TASK_TREE=/mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
exec 8>/mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock
flock 8
exec 9>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 9
cd "$TASK_TREE"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH='' PIP_NO_INDEX=1 HF_HUB_OFFLINE=1
TASK_CPU=$(/usr/bin/python3 -c 'import os;print(min(os.sched_getaffinity(0)))')
set +e
tools/run_capped.sh 2G taskset -c "$TASK_CPU" /usr/bin/python3 "$TASK_DIR/verify.py" >"$TASK_DIR/verify.log" 2>&1
TASK_EXIT=$?
set -e
printf '%s\n' "$TASK_EXIT" >"$TASK_DIR/verify.exit"
exit "$TASK_EXIT"
