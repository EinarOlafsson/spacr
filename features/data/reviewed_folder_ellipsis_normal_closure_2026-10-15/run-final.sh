#!/bin/bash
set -euo pipefail
ulimit -c 0
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SPACR_NO_GL=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD:/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages"
exec 9>/mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock
flock 9
exec 8>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 8
set +e
tools/run_capped.sh 8G /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/reviewed-folder-ellipsis-20261015/scoped-six-runtime.py /mnt/wd4tb/scratch/reviewed-folder-ellipsis-20261015/config.FINAL.json > /mnt/wd4tb/scratch/reviewed-folder-ellipsis-20261015/run.log 2>&1
rc=$?
printf '%s\n' "$rc" > /mnt/wd4tb/scratch/reviewed-folder-ellipsis-20261015/run.exit
exit "$rc"
