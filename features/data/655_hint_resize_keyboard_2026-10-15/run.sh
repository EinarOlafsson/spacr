#!/bin/bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CONFIG_HOME=/mnt/wd4tb/scratch/n655-hint-grip-20261015/config
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages:.
mkdir -p "$XDG_CONFIG_HOME"
phase="$1"
shift
/usr/bin/time -v -o /mnt/wd4tb/scratch/n655-hint-grip-20261015/"$phase".time tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q "$@" > /mnt/wd4tb/scratch/n655-hint-grip-20261015/"$phase".log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/wd4tb/scratch/n655-hint-grip-20261015/"$phase".exit
exit "$result"
