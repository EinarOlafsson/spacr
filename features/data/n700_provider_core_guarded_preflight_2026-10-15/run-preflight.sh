#!/bin/bash
set -euo pipefail
ulimit -c 0
out=/mnt/wd4tb/scratch/n700-provider-core-authoring-20261015
exec > "$out/guarded-preflight.log" 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$out/guarded-preflight.exit"' EXIT
exec 9>/mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock
flock 9
exec 8>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 8
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
tools/run_capped.sh 2G /home/olafsson/anaconda3/bin/python "$out/guarded-preflight.py"
