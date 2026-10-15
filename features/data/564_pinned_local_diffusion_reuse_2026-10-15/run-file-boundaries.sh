#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-f564-pretrained-reuse-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
status=0
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_f564_checkpoint_file_boundaries.py tests/test_f564_pinned_diffusion_reuse.py -k 'fifo_refuses or growing_stream or actual_weights_roundtrip or changed_open_file' > /mnt/wd4tb/scratch/f564-pretrained-backbone-plan-20261015/file-boundary-final.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/f564-pretrained-backbone-plan-20261015/file-boundary-final.exit
exit "$status"
