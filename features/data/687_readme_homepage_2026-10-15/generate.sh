#!/bin/bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-f370-generator-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock /usr/bin/time -v tools/run_capped.sh 4G /home/olafsson/anaconda3/bin/python tools/build_readme_images.py > /mnt/wd4tb/scratch/home-b-n687-readme-homepage-20261015/generate.log 2> /mnt/wd4tb/scratch/home-b-n687-readme-homepage-20261015/generate.time
status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/home-b-n687-readme-homepage-20261015/generate.exit
exit "$status"
