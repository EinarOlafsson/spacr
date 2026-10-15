#!/bin/bash
cd /mnt/wd4tb/spacr-worktrees/codex-f562-coordinate-identity-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 SPACR_LANGUAGE=en
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_mil_coordinate_identity_f562.py tests/test_multiple_instance_learning.py -k 'coordinate_identity or direct_mil or refuses_labels' --basetemp=/mnt/wd4tb/scratch/f562-coordinate-identity-20261015/owning-pytest --junitxml=/mnt/wd4tb/scratch/f562-coordinate-identity-20261015/owning.xml > /mnt/wd4tb/scratch/f562-coordinate-identity-20261015/owning.log 2>&1
printf '%s\n' "$?" > /mnt/wd4tb/scratch/f562-coordinate-identity-20261015/owning.exit
