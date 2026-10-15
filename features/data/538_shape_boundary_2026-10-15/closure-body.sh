#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-f538-shape-boundary-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen SPACR_LANGUAGE=en OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
phase_status=0
tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/f538-shape-boundary-20261015/scoped-runtime.py --write > /mnt/wd4tb/scratch/f538-shape-boundary-20261015/runtime.log 2>&1 || phase_status=$?
printf '%s\n' "$phase_status" > /mnt/wd4tb/scratch/f538-shape-boundary-20261015/runtime.exit
if [ "$phase_status" -ne 0 ]; then exit "$phase_status"; fi
phase_status=0
tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_unmix_shape_boundary_f538.py -k installed_reviewed_language --basetemp=/mnt/wd4tb/scratch/f538-shape-boundary-20261015/locale-pytest --junitxml=/mnt/wd4tb/scratch/f538-shape-boundary-20261015/locale.xml > /mnt/wd4tb/scratch/f538-shape-boundary-20261015/locale.log 2>&1 || phase_status=$?
printf '%s\n' "$phase_status" > /mnt/wd4tb/scratch/f538-shape-boundary-20261015/locale.exit
exit "$phase_status"
