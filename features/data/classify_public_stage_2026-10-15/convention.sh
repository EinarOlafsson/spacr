#!/bin/bash
cd /mnt/wd4tb/spacr-worktrees/codex-classify-public-stage-20261015 || exit 90
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen SPACR_LANGUAGE=en OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -n 0 tests/qt/test_classify_public_maturity_route.py tests/qt/test_home_v2.py::test_every_app_has_a_stage_and_it_is_written_down_once --tb=short --junitxml=/mnt/wd4tb/scratch/classify-public-stage-20261015/convention.xml > /mnt/wd4tb/scratch/classify-public-stage-20261015/convention.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/wd4tb/scratch/classify-public-stage-20261015/convention.exit
exit "$result"
