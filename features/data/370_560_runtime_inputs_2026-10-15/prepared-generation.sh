#!/bin/bash
set -euo pipefail
: "${1:?coordinator worktree required}"
: "${2:?scratch output directory required}"
cd "$1"
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONPATH="/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages:$PWD"
mkdir -p "$2"
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /usr/bin/time -v -o "$2/generation.time" /home/olafsson/anaconda3/envs/spacr/bin/python features/data/370_560_runtime_inputs_2026-10-15/scoped-runtime.py "$PWD" "$2" > "$2/generation.log" 2>&1
