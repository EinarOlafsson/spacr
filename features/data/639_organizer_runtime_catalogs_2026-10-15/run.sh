#!/usr/bin/env bash
set -uo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-n180-graph-api-20261015
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export QT_QPA_PLATFORM=offscreen
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
export XDG_RUNTIME_DIR=/mnt/wd4tb/scratch/home-b-n639-runtime-producer-20261015/xdg-runtime
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
exec 9>/mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock
flock 9
printf '\nHome-B N639 ACTUAL START: normal runtime-only 21-source/189-target producer on merged public dc085efb71, private f84e3edf2c. Final plan f9b2fb5e and driver 1d3962bd independently accepted; 6G/thread1/common heavy.lock. All nine preflights before writes, preserve N702 targets and all foreign catalog/history/API/Help. No models/API/Help generation. Source/input freeze until terminal; whole N639 still awaits actual Work clip receipt.\n' >> /mnt/wd4tb/scratch/home-b-coordination-20261014/TO_HOME.txt
/usr/bin/time -v -o /mnt/wd4tb/scratch/home-b-n639-runtime-producer-20261015/producer.time tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -u /mnt/wd4tb/scratch/home-b-n639-runtime-producer-20261015/generate.py /mnt/wd4tb/scratch/home-b-n639-runtime-producer-20261015/plan.json /mnt/wd4tb/scratch/home-b-n639-runtime-producer-20261015/run-1 > /mnt/wd4tb/scratch/home-b-n639-runtime-producer-20261015/producer.log 2>&1
status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/home-b-n639-runtime-producer-20261015/producer.exit
exit "$status"
