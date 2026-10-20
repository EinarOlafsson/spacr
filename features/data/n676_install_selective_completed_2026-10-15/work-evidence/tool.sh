#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
: "${SOURCE_COMMIT:?Actual frozen combined selective K source required}"
export SOURCE_COMMIT
name=$1;shift
cd "$ROOT"
log=$N/logs/$TAG-$name.log
[[ ! -e "$log" ]] || { echo "Preserve existing log: $log" >&2; exit 2; }
exec "$ROOT/tools/run_capped.sh" "${TOOL_CAP:-8G}" "$PY" "$N/work/run_logged.py" "$log" "$PY" "$WORK/run_tool.py" "$@"
