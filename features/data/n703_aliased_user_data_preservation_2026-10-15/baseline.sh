#!/bin/bash
set -u
TASK_SCRATCH=/mnt/wd4tb/scratch/n703-aliased-user-data-20261015
cd /mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPYCACHEPREFIX=$TASK_SCRATCH/pycache TMPDIR=$TASK_SCRATCH/tmp XDG_CONFIG_HOME=$TASK_SCRATCH/config XDG_CACHE_HOME=$TASK_SCRATCH/cache
mkdir -p "$TMPDIR" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
/home/olafsson/anaconda3/envs/spacr/bin/python -m pytest tests/test_aliased_user_data_preservation_703.py -q --tb=short --junitxml="$TASK_SCRATCH/baseline.xml" --basetemp="$TASK_SCRATCH/baseline-tmp" -o cache_dir="$TASK_SCRATCH/pytest-cache" > "$TASK_SCRATCH/baseline.log" 2>&1
TASK_RC=$?
echo "$TASK_RC" > "$TASK_SCRATCH/baseline.exit"
exit "$TASK_RC"
