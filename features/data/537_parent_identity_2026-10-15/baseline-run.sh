#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-f537-parent-identity-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
phase_status=0
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_lineage_parent_identity_f537.py -k 'not installed_reviewed_language' --basetemp=/mnt/wd4tb/scratch/f537-parent-identity-20261015/baseline-pytest --junitxml=/mnt/wd4tb/scratch/f537-parent-identity-20261015/baseline.xml > /mnt/wd4tb/scratch/f537-parent-identity-20261015/baseline.log 2>&1 || phase_status=$?
printf '%s\n' "$phase_status" > /mnt/wd4tb/scratch/f537-parent-identity-20261015/baseline.exit
exit "$phase_status"
