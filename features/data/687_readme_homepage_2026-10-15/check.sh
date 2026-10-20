#!/bin/bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-f370-generator-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/usr/bin/time -v tools/run_capped.sh 2G /home/olafsson/anaconda3/bin/python /mnt/wd4tb/scratch/home-b-n687-readme-homepage-20261015/verify.py > /mnt/wd4tb/scratch/home-b-n687-readme-homepage-20261015/verify.log 2> /mnt/wd4tb/scratch/home-b-n687-readme-homepage-20261015/verify.time
status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/home-b-n687-readme-homepage-20261015/verify.exit
[ "$status" = 0 ] || exit "$status"
/usr/bin/time -v tools/run_capped.sh 2G /home/olafsson/anaconda3/bin/python tools/build_readme_images.py --check > /mnt/wd4tb/scratch/home-b-n687-readme-homepage-20261015/currency.log 2> /mnt/wd4tb/scratch/home-b-n687-readme-homepage-20261015/currency.time
status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/home-b-n687-readme-homepage-20261015/currency.exit
[ "$status" = 0 ] || exit "$status"
/usr/bin/time -v tools/run_capped.sh 2G /home/olafsson/anaconda3/bin/python -m pytest -q tests/test_pypi_readme_images.py > /mnt/wd4tb/scratch/home-b-n687-readme-homepage-20261015/pypi.log 2> /mnt/wd4tb/scratch/home-b-n687-readme-homepage-20261015/pypi.time
status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/home-b-n687-readme-homepage-20261015/pypi.exit
exit "$status"
