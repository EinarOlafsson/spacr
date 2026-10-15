#!/usr/bin/env bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-f370-generator-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/mnt/wd4tb/scratch/home-b-f534-empty-map-20261015
/usr/bin/time -v -o "$TMPDIR/oversized.time" tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q --no-cov tests/test_spatial_submodules.py -k 'zero_qv_map_keeps_unverified_outputs and oversized' > "$TMPDIR/oversized.log" 2>&1
result=$?
printf '%s\n' "$result" > "$TMPDIR/oversized.exit"
exit "$result"
