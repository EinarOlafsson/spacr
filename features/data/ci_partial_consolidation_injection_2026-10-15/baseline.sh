#!/bin/bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
TASK_SCRATCH=/mnt/wd4tb/scratch/ci-partial-consolidation-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen SPACR_NO_GL=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
export TMPDIR=$TASK_SCRATCH/tmp XDG_CONFIG_HOME=$TASK_SCRATCH/config XDG_CACHE_HOME=$TASK_SCRATCH/cache PYTHONPYCACHEPREFIX=$TASK_SCRATCH/pycache
mkdir -p "$TMPDIR" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -p pytestqt.plugin 'tests/qt/test_626_partial_consolidation.py::test_failed_copy_preserves_working_state_and_aborts_every_action[sort_button]' -q --tb=short --junitxml="$TASK_SCRATCH/baseline.xml" --basetemp="$TASK_SCRATCH/baseline-tmp" -o cache_dir="$TASK_SCRATCH/pytest-cache" > "$TASK_SCRATCH/baseline.log" 2>&1
TASK_RC=$?
echo "$TASK_RC" > "$TASK_SCRATCH/baseline.exit"
exit "$TASK_RC"
