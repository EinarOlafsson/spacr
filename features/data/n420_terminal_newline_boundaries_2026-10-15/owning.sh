#!/bin/bash
set -u
TASK_SCRATCH=/mnt/wd4tb/scratch/n420-stream-crlf-20261015
cd /mnt/wd4tb/spacr-worktrees/codex-n420-stream-crlf-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen SPACR_NO_GL=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPYCACHEPREFIX=$TASK_SCRATCH/pycache TMPDIR=$TASK_SCRATCH/tmp XDG_CONFIG_HOME=$TASK_SCRATCH/config XDG_CACHE_HOME=$TASK_SCRATCH/cache
export COVERAGE_FILE=$TASK_SCRATCH/.coverage
mkdir -p "$TMPDIR" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -p pytestqt.plugin -p xdist.plugin -p pytest_cov.plugin -n 0 tests/qt/test_pty_stream_newlines_420.py tests/qt/test_pty_stream_ansi_420.py tests/qt/test_pty_incremental_unicode_420.py tests/qt/test_sign_in_fragmented_urls_420.py tests/qt/test_ai_sign_in_happens_inside_spacr.py --cov=spacr.qt.ai.pty_sign_in --cov-branch --cov-report=json:/mnt/wd4tb/scratch/n420-stream-crlf-20261015/coverage.json --cov-report=term -q --tb=short --junitxml="$TASK_SCRATCH/owning.xml" --basetemp="$TASK_SCRATCH/owning-tmp" -o cache_dir="$TASK_SCRATCH/pytest-cache" > "$TASK_SCRATCH/owning.log" 2>&1
TASK_RC=$?
echo "$TASK_RC" > "$TASK_SCRATCH/owning.exit"
exit "$TASK_RC"
