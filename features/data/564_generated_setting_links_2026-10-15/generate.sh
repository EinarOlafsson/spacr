#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-coordinator-20261014
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONPATH="/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages:$PWD"
mkdir -p /mnt/wd4tb/scratch/f564-generated-flows-20261015/tmp
export TMPDIR=/mnt/wd4tb/scratch/f564-generated-flows-20261015/tmp
tools/run_capped.sh 6G /usr/bin/time -v -o /mnt/wd4tb/scratch/f564-generated-flows-20261015/consumers.time /home/olafsson/anaconda3/envs/spacr/bin/python tools/build_setting_consumer_map.py > /mnt/wd4tb/scratch/f564-generated-flows-20261015/consumers.log 2>&1
tools/run_capped.sh 6G /usr/bin/time -v -o /mnt/wd4tb/scratch/f564-generated-flows-20261015/flows.time /home/olafsson/anaconda3/envs/spacr/bin/python tools/settings_flow.py --rst > /mnt/wd4tb/scratch/f564-generated-flows-20261015/flows.log 2>&1
tools/run_capped.sh 6G /usr/bin/time -v -o /mnt/wd4tb/scratch/f564-generated-flows-20261015/help.time /home/olafsson/anaconda3/envs/spacr/bin/python tools/build_help_search_index.py > /mnt/wd4tb/scratch/f564-generated-flows-20261015/help.log 2>&1
