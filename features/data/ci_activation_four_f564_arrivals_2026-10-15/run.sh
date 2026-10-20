#!/usr/bin/env bash
set -u
TASK_DIR=/mnt/wd4tb/scratch/ci-activation-four-arrivals-20261015
TASK_TREE=/mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
TASK_MODE=$1
mkdir -p "$TASK_DIR/tmp" "$TASK_DIR/config" "$TASK_DIR/cache"
exec 9>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 9
cd "$TASK_TREE"
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen SPACR_NO_GL=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
export TMPDIR="$TASK_DIR/tmp" XDG_CONFIG_HOME="$TASK_DIR/config" XDG_CACHE_HOME="$TASK_DIR/cache" PYTHONPYCACHEPREFIX="$TASK_DIR/pycache"
tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python "$TASK_DIR/pytest-driver.py" "$TASK_MODE" >"$TASK_DIR/$TASK_MODE.log" 2>&1
TASK_RESULT=$?
printf '%s\n' "$TASK_RESULT" >"$TASK_DIR/$TASK_MODE.exit"
exit "$TASK_RESULT"
