#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n703-debian-adapter-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -c 0
status=0
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock tools/run_capped.sh 8G /home/olafsson/anaconda3/bin/python -m pytest -q tests/test_the_settings_flow_artefacts_are_current.py::test_every_section_says_today_what_the_source_says_today > /mnt/wd4tb/scratch/ci-settings-flow-current-20261015/baseline.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/ci-settings-flow-current-20261015/baseline.exit
exit "$status"
