#!/bin/bash
set -uo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 SPACR_NO_NETWORK=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export XDG_CONFIG_HOME=/mnt/wd4tb/scratch/home-b-f580-closure-20261015/consumer-ui/config
export XDG_DATA_HOME=/mnt/wd4tb/scratch/home-b-f580-closure-20261015/consumer-ui/data
export XDG_CACHE_HOME=/mnt/wd4tb/scratch/home-b-f580-closure-20261015/consumer-ui/cache
export TMPDIR=/mnt/wd4tb/scratch/home-b-f580-closure-20261015/consumer-ui/tmp
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages:.
mkdir -p "$TMPDIR"
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 4G /usr/bin/time -v -o /mnt/wd4tb/scratch/home-b-f580-closure-20261015/consumer-ui.time /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/home-b-f580-closure-20261015/check-consumers.py /mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015 /mnt/wd4tb/scratch/home-b-f580-closure-20261015/consumer-ui d3191c3a99e1ca85ef31c0e5487a6bea9f2f5b57 > /mnt/wd4tb/scratch/home-b-f580-closure-20261015/consumer-ui.log 2>&1
status=$?
echo "$status" > /mnt/wd4tb/scratch/home-b-f580-closure-20261015/consumer-ui.exit
exit "$status"
