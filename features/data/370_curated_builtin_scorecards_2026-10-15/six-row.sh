#!/bin/bash
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/mnt/wd4tb/scratch/n702-cli-pins-20261014
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
/usr/bin/time -v tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_f370_curated_builtin_scorecards.py::test_exact_builtin_preserves_all_six_approved_matched_scores_and_dataset -n 0 --junitxml=/mnt/wd4tb/scratch/f370-builtin-core-20261015/six-row.xml > /mnt/wd4tb/scratch/f370-builtin-core-20261015/six-row.log 2> /mnt/wd4tb/scratch/f370-builtin-core-20261015/six-row.time
status=$?
echo "$status" > /mnt/wd4tb/scratch/f370-builtin-core-20261015/six-row.exit
exit "$status"
