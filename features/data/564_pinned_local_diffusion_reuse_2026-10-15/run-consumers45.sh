#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-f564-pretrained-reuse-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
status=0
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/qt/test_f564_compiled_checkpoint_controls.py tests/test_f564_pinned_diffusion_locales.py --basetemp=/mnt/wd4tb/scratch/f564-pretrained-backbone-plan-20261015/compiled45-tmp --junitxml=/mnt/wd4tb/scratch/f564-pretrained-backbone-plan-20261015/consumers45.xml > /mnt/wd4tb/scratch/f564-pretrained-backbone-plan-20261015/consumers45.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/f564-pretrained-backbone-plan-20261015/consumers45.exit
exit "$status"
