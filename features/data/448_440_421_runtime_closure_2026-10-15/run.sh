#!/usr/bin/env bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-n703-debian-adapter-20261015
TASK_SCRATCH=/mnt/wd4tb/scratch/n448-n440-n421-runtime-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen SPACR_NO_GL=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
export PYTHONPYCACHEPREFIX=$TASK_SCRATCH/pycache
TASK_CPU=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os; print(min(os.sched_getaffinity(0)))')
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G taskset -c "$TASK_CPU" /home/olafsson/anaconda3/envs/spacr/bin/python "$TASK_SCRATCH/generate.py" --root "$PWD" --approved-inputs "$TASK_SCRATCH/approved-inputs.json" --approved-inputs-sha256 61f5a971cfc2d2943dd5cc60135a6b1186684006efc65b7248b0e4ae62227db4 --write > "$TASK_SCRATCH/producer.log" 2>&1
TASK_RC=$?
printf '%s\n' "$TASK_RC" > "$TASK_SCRATCH/producer.exit"
exit "$TASK_RC"
