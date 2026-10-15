#!/usr/bin/env bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-f370-generator-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/mnt/wd4tb/scratch/home-b-f534-empty-map-20261015
/usr/bin/time -v -o "$TMPDIR/after.time" tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q --no-cov tests/test_spatial_submodules.py -k 'zero_qv_map or xenium_reads_are_counted or visium_hd_bins_are_registered or blank_windows_are_skipped' > "$TMPDIR/after.log" 2>&1
result=$?
printf '%s\n' "$result" > "$TMPDIR/after.exit"
exit "$result"
