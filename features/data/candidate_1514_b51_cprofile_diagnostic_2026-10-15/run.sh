#!/usr/bin/env bash
set -euo pipefail
TASK_DIR=/mnt/wd4tb/scratch/candidate-1514-b51fb194bb-20261015/installed-cprofile-diagnostic
TASK_ROOT=/mnt/wd4tb/scratch/candidate-1514-b51fb194bb-20261015
TASK_TREE=/mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
TASK_PYTHON="$TASK_ROOT/smoke-preflight/venv/bin/python"
exec 8>/mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock
flock 8
exec 9>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 9
mkdir -p "$TASK_DIR/tmp" "$TASK_DIR/home" "$TASK_DIR/config" "$TASK_DIR/cache"
export HOME="$TASK_DIR/home" TMPDIR="$TASK_DIR/tmp" XDG_CONFIG_HOME="$TASK_DIR/config" XDG_CACHE_HOME="$TASK_DIR/cache"
unset SPACR_HOME PYTHONPATH PYTHONHOME
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen SPACR_NO_GL=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SPACR_MODEL_DOWNLOAD=off PIP_NO_INDEX=1
cd "$TASK_DIR"
"$TASK_PYTHON" "$TASK_DIR/origin-snapshot.py" origin-before >"$TASK_DIR/origin-before.log" 2>&1
TASK_CPU=$(/usr/bin/python3 -c 'import os;print(min(os.sched_getaffinity(0)))')
set +e
"$TASK_TREE/tools/run_capped.sh" 8G taskset -c "$TASK_CPU" timeout --signal=TERM --kill-after=30s 3900 "$TASK_PYTHON" "$TASK_DIR/profile-driver.py" >"$TASK_DIR/profiled-navigation.log" 2>&1
TASK_EXIT=$?
printf '%s\n' "$TASK_EXIT" >"$TASK_DIR/profiled-navigation.exit"
"$TASK_PYTHON" "$TASK_DIR/origin-snapshot.py" origin-after >"$TASK_DIR/origin-after.log" 2>&1
TASK_ORIGIN_EXIT=$?
printf '%s\n' "$TASK_ORIGIN_EXIT" >"$TASK_DIR/origin-after.exit"
"$TASK_PYTHON" "$TASK_DIR/verify-profile-outputs.py" >"$TASK_DIR/profiling-completeness.log" 2>&1
TASK_PROFILE_EXIT=$?
printf '%s\n' "$TASK_PROFILE_EXIT" >"$TASK_DIR/profiling-completeness.exit"
set -e
if test "$TASK_ORIGIN_EXIT" -ne 0; then exit "$TASK_ORIGIN_EXIT"; fi
if test "$TASK_PROFILE_EXIT" -ne 0; then exit "$TASK_PROFILE_EXIT"; fi
exit "$TASK_EXIT"
