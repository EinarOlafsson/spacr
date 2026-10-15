#!/bin/bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
TASK_SCRATCH=/mnt/wd4tb/scratch/ci-lint-38699600639-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen SPACR_NO_GL=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export TMPDIR=$TASK_SCRATCH/tmp XDG_CONFIG_HOME=$TASK_SCRATCH/config XDG_CACHE_HOME=$TASK_SCRATCH/cache PYTHONPYCACHEPREFIX=$TASK_SCRATCH/pycache RUFF_CACHE_DIR=$TASK_SCRATCH/ruff-cache
mkdir -p "$TMPDIR" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
mapfile -t targets < <(grep -Ev '^[[:space:]]*(#|$)' .github/quality-targets.txt)
test "${#targets[@]}" -eq 6 || exit 91
printf '%s
' "${targets[@]}" > "$TASK_SCRATCH/final-targets.txt"
/home/olafsson/anaconda3/envs/spacr/bin/python -m pytest tests/test_ci_suite_classification.py::test_torch_heartbeat_runs_once_serially_and_its_failure_blocks_ci -q --tb=short --junitxml="$TASK_SCRATCH/refactor-owning.xml" --basetemp="$TASK_SCRATCH/refactor-tmp" -o cache_dir="$TASK_SCRATCH/pytest-cache" > "$TASK_SCRATCH/refactor-owning.log" 2>&1
echo "$?" > "$TASK_SCRATCH/refactor-owning.exit"
/home/olafsson/anaconda3/envs/spacr/bin/python -m ruff check "${targets[@]}" > "$TASK_SCRATCH/refactor-six-ruff.log" 2>&1
echo "$?" > "$TASK_SCRATCH/refactor-six-ruff.exit"
/home/olafsson/anaconda3/envs/spacr/bin/python -m mypy --python-executable="$TASK_SCRATCH/tool-runtime/bin/python" --cache-dir="$TASK_SCRATCH/mypy-six-cache" "${targets[@]}" > "$TASK_SCRATCH/refactor-six-mypy.log" 2>&1
echo "$?" > "$TASK_SCRATCH/refactor-six-mypy.exit"
/home/olafsson/anaconda3/envs/spacr/bin/python -m xenon --max-absolute B --max-modules B --max-average A "${targets[@]}" > "$TASK_SCRATCH/refactor-six-xenon.log" 2>&1
echo "$?" > "$TASK_SCRATCH/refactor-six-xenon.exit"
