#!/bin/bash
set -u
TASK_SCRATCH=/mnt/wd4tb/scratch/n703-aliased-user-data-20261015
cd /mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPYCACHEPREFIX=$TASK_SCRATCH/pycache TMPDIR=$TASK_SCRATCH/tmp XDG_CONFIG_HOME=$TASK_SCRATCH/config XDG_CACHE_HOME=$TASK_SCRATCH/cache
mkdir -p "$TMPDIR" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest tests/test_retirement_respects_update_leases_703.py::test_actual_removal_refusal_uses_exact_reviewed_locale -q --tb=short --junitxml="$TASK_SCRATCH/locale-closure.xml" --basetemp="$TASK_SCRATCH/locale-closure-tmp" -o cache_dir="$TASK_SCRATCH/pytest-cache" > "$TASK_SCRATCH/locale-closure.log" 2>&1
TASK_RC=$?
echo "$TASK_RC" > "$TASK_SCRATCH/locale-closure.exit"
exit "$TASK_RC"
