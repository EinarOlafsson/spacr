#!/usr/bin/env bash
set -euo pipefail
TASK_DIR=/mnt/wd4tb/scratch/candidate-1514-b51fb194bb-20261015/smoke-preflight
TASK_ROOT=/mnt/wd4tb/scratch/candidate-1514-b51fb194bb-20261015
TASK_TREE=/mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
exec 8>/mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock
flock 8
exec 9>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 9
export HOME="$TASK_DIR/home" TMPDIR="$TASK_DIR/tmp" XDG_CONFIG_HOME="$TASK_DIR/config" XDG_CACHE_HOME="$TASK_DIR/cache"
unset SPACR_HOME PYTHONPATH PYTHONHOME PIP_NO_INDEX PIP_EXTRA_INDEX_URL
export PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple PIP_FIND_LINKS="$TASK_ROOT/wheelhouse" PIP_DISABLE_PIP_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$TASK_DIR"
set +e
"$TASK_TREE/tools/run_capped.sh" 2G "$TASK_DIR/venv/bin/python" -m pip --disable-pip-version-check install --dry-run --only-binary=:all: --retries 0 --timeout 20 --index-url https://pypi.org/simple --report "$TASK_DIR/online-resolution.json" "$TASK_ROOT/artifacts/spacr-1.5.1.4-py3-none-any.whl[qt]" >"$TASK_DIR/online-resolution.log" 2>&1
TASK_EXIT=$?
set -e
printf '%s\n' "$TASK_EXIT" >"$TASK_DIR/online-resolution.exit"
exit "$TASK_EXIT"
