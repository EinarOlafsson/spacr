#!/bin/bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
TASK_SCRATCH=/mnt/wd4tb/scratch/ci-toggle-focus-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen SPACR_NO_GL=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH=$TASK_SCRATCH:/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
export TMPDIR=$TASK_SCRATCH/tmp XDG_CONFIG_HOME=$TASK_SCRATCH/config XDG_CACHE_HOME=$TASK_SCRATCH/cache PYTHONPYCACHEPREFIX=$TASK_SCRATCH/pycache
mkdir -p "$TMPDIR" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -p pytestqt.plugin -p toggle_focus_diagnostic -p qt_session_theme_context tests/qt/test_655_toggle_accessibility.py::test_keyboard_focus_remains_visible_when_the_field_paints_the_switch -q --tb=short --junitxml="$TASK_SCRATCH/themed.xml" --basetemp="$TASK_SCRATCH/themed-tmp" -o cache_dir="$TASK_SCRATCH/pytest-cache" > "$TASK_SCRATCH/themed.log" 2>&1
TASK_RC=$?
echo "$TASK_RC" > "$TASK_SCRATCH/themed.exit"
exit "$TASK_RC"
