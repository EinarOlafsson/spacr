#!/bin/bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-n703-debian-adapter-20261015
TASK_SCRATCH=/mnt/wd4tb/scratch/ci-notebook-regeneration-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export XDG_CONFIG_HOME=$TASK_SCRATCH/config XDG_CACHE_HOME=$TASK_SCRATCH/cache PYTHONPYCACHEPREFIX=$TASK_SCRATCH/pycache
mkdir -p "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
exec 9>/mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock
flock 9
tools/run_capped.sh 8G /home/olafsson/anaconda3/envs/spacr/bin/python tools/build_notebook_settings.py --check > "$TASK_SCRATCH/baseline.log" 2>&1
TASK_RC=$?
echo "$TASK_RC" > "$TASK_SCRATCH/baseline.exit"
if [ "$TASK_RC" -eq 0 ]; then exit 0; fi
if [ "$TASK_RC" -ne 1 ]; then exit "$TASK_RC"; fi
tools/run_capped.sh 8G /home/olafsson/anaconda3/envs/spacr/bin/python tools/build_notebook_settings.py > "$TASK_SCRATCH/generation.log" 2>&1
TASK_RC=$?
echo "$TASK_RC" > "$TASK_SCRATCH/generation.exit"
if [ "$TASK_RC" -ne 0 ]; then exit "$TASK_RC"; fi
tools/run_capped.sh 8G /home/olafsson/anaconda3/envs/spacr/bin/python tools/build_notebook_settings.py --check > "$TASK_SCRATCH/verification.log" 2>&1
TASK_RC=$?
echo "$TASK_RC" > "$TASK_SCRATCH/verification.exit"
exit "$TASK_RC"
