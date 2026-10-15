#!/usr/bin/env bash
set -uo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SPACR_NO_NETWORK=1
export XDG_CONFIG_HOME=/mnt/wd4tb/scratch/home-b-n655-hint-value-20261015/config
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages:$PWD
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 2G /usr/bin/time -v -o /mnt/wd4tb/scratch/home-b-n655-hint-value-20261015/baseline.time /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/home-b-n655-hint-value-20261015/baseline.py > /mnt/wd4tb/scratch/home-b-n655-hint-value-20261015/baseline.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/wd4tb/scratch/home-b-n655-hint-value-20261015/baseline.exit
exit "$result"
