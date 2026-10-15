#!/bin/bash
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen SPACR_LANGUAGE=en OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd /mnt/wd4tb/spacr-worktrees/codex-f562-finite-features-20261015
set +e
tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/f562-finite-features-20261015/scoped-runtime.py --write > /mnt/wd4tb/scratch/f562-finite-features-20261015/runtime.log 2>&1
phase_status=$?
printf '%s\n' "$phase_status" > /mnt/wd4tb/scratch/f562-finite-features-20261015/runtime.exit
if [ "$phase_status" != 0 ]; then exit "$phase_status"; fi
tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q --basetemp=/mnt/wd4tb/scratch/f562-finite-features-20261015/owning-pytest tests/test_mil_finite_features_f562.py tests/test_multiple_instance_learning.py -k 'finite_features or direct_mil' --junitxml=/mnt/wd4tb/scratch/f562-finite-features-20261015/owning.xml > /mnt/wd4tb/scratch/f562-finite-features-20261015/owning.log 2>&1
phase_status=$?
printf '%s\n' "$phase_status" > /mnt/wd4tb/scratch/f562-finite-features-20261015/owning.exit
if [ "$phase_status" != 0 ]; then exit "$phase_status"; fi
tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/f562-finite-features-20261015/audit-arrivals.py > /mnt/wd4tb/scratch/f562-finite-features-20261015/audit-arrivals.log 2>&1
phase_status=$?
printf '%s\n' "$phase_status" > /mnt/wd4tb/scratch/f562-finite-features-20261015/audit-arrivals.exit
if [ "$phase_status" != 0 ]; then exit "$phase_status"; fi
