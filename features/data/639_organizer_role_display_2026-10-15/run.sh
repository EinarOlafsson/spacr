#!/bin/bash
set -u
run_dir=/mnt/wd4tb/scratch/home-b-n639-role-display-20261015
run_root=/mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015
stage="$1"
shift
mkdir -p "$run_dir/$stage"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export XDG_CONFIG_HOME="$run_dir/$stage/config" XDG_CACHE_HOME="$run_dir/$stage/cache" XDG_DATA_HOME="$run_dir/$stage/data" XDG_STATE_HOME="$run_dir/$stage/state" SPACR_HOME="$run_dir/$stage/spacr"
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages:"$run_root"
cd "$run_root"
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock /usr/bin/time -v -o "$run_dir/$stage/time.txt" tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q -o addopts= "$@" > "$run_dir/$stage/run.log" 2>&1
result=$?
printf '%s\n' "$result" > "$run_dir/$stage/exit.txt"
cat "$run_dir/$stage/run.log"
exit "$result"
