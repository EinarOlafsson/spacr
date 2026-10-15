#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n702-picker-pin-provenance-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec 9>/mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock
flock 9
set +e
tools/run_capped.sh 4G /home/olafsson/anaconda3/bin/python /mnt/wd4tb/scratch/work04-source-implementation-20261015/run-tests.py > /mnt/wd4tb/scratch/work04-source-implementation-20261015/owning-tests.log 2>&1
rc=$?
printf '%s\n' "$rc" > /mnt/wd4tb/scratch/work04-source-implementation-20261015/owning-tests.exit
exit "$rc"
