#!/bin/bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
TASK_SCRATCH=/mnt/wd4tb/scratch/ci-lint-38699600639-20261015
mapfile -t targets < <(grep -Ev '^[[:space:]]*(#|$)' .github/quality-targets.txt)
/home/olafsson/anaconda3/envs/spacr/bin/python -m mypy --python-executable="$TASK_SCRATCH/tool-runtime/bin/python" --cache-dir="$TASK_SCRATCH/mypy-isolated-six-cache" "${targets[@]}" > "$TASK_SCRATCH/refactor-six-mypy-isolated.log" 2>&1
TASK_RC=$?
echo "$TASK_RC" > "$TASK_SCRATCH/refactor-six-mypy-isolated.exit"
exit "$TASK_RC"
