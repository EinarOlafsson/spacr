#!/usr/bin/env bash
# Root launches after clean-source/integration acceptance. No capture starts on preparation.
set -euo pipefail
ROOT=/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014
TURN=/media/carruthers/mnt3/claude/n676_stages/work/lesson05-selective-20261015/worker-apply-capture-r2
PY=/home/carruthers/anaconda3/envs/spacr/bin/python
: "${SPACR_FINAL_CAPTURE_REVISION:?Root must supply the accepted integrated full clean commit}"
[[ "$SPACR_FINAL_CAPTURE_REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "${DISPLAY:-}" != :0 && "${DISPLAY:-}" != :0.0 ]] || { echo 'Desktop DISPLAY is forbidden; invoke with DISPLAY unset' >&2; exit 2; }
cd -- "$ROOT"
[[ "$(git rev-parse HEAD)" == "$SPACR_FINAL_CAPTURE_REVISION" && -z "$(git status --porcelain)" ]] || { echo 'Source revision mismatch or dirty checkout' >&2; exit 2; }
# Root peers this preparation and invokes only after actual23 push; never auto-run.
[[ ! -e /media/carruthers/mnt3/claude/n676_stages/host/lesson05-worker-apply-r2 ]] || exit 2
"$ROOT/tools/run_capped.sh" 1G "$PY" "$TURN/capture_job.py" --stage /media/carruthers/mnt3/claude/n676_stages/host/lesson05-worker-apply-r2 --revision "$SPACR_FINAL_CAPTURE_REVISION" --check-only
exec 8>"$TURN/run.lock"
flock -n 8 || { echo 'Native Apply diagnostic already has a runner' >&2; exit 2; }
export SPACR_HOME_APPLY_DIAG_LEASE_PID=$$
LEASE_ACQUIRED=0
cleanup() {
  local saved=$?
  if [[ "$LEASE_ACQUIRED" == 1 ]]; then
    "$ROOT/tools/run_capped.sh" 1G "$PY" - <<'CLEAN'
import os,sys
sys.path.insert(0,'/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
import fcntl
from pathlib import Path
prefix='codex-ws-home-worker-apply-capture | '
token='owner_pid='+os.environ['SPACR_HOME_APPLY_DIAG_LEASE_PID']
with Path('/media/carruthers/mnt3/claude/ram_leases.md').open('r+') as f:
    fcntl.flock(f,fcntl.LOCK_EX)
    rows=f.readlines();f.seek(0)
    f.writelines(row for row in rows if not (row.startswith(prefix) and token in row.split(' | ')))
    f.truncate();f.flush();os.fsync(f.fileno())
CLEAN
  fi
  return "$saved"
}
trap cleanup EXIT
"$ROOT/tools/run_capped.sh" 1G "$PY" - <<'LEASE'
import os,sys
sys.path.insert(0,'/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
import fcntl
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
label='codex-ws-home-worker-apply-capture'
with Path('/media/carruthers/mnt3/claude/ram_leases.md').open('r+') as f:
    fcntl.flock(f,fcntl.LOCK_EX)
    rows=f.readlines()
    if any(row.startswith(label+' | ') for row in rows): raise RuntimeError('Own lease label already exists; preserve and inspect')
    f.seek(0,2)
    f.write(f'\n{label} | one complete worker-dispatch Apply/Revert capture comparison;300s execution/shared30s cleanup grace | 12 GB | {datetime.now(ZoneInfo("America/Detroit")).isoformat()} | owner_pid={os.environ["SPACR_HOME_APPLY_DIAG_LEASE_PID"]} | removed on exit; execution300s inclpostprocess; sharedterminationcleanup<=30s; total<=330s\n')
    f.flush();os.fsync(f.fileno())
LEASE
LEASE_ACQUIRED=1
# Explicit maintainer defaults; no idle/gap bypass from inherited environment.
export GPU_IDLE_S=360 GPU_GAP_S=600 GPU_POLL_S=15 TURN_S=600
"$ROOT/tools/gpu_turn.sh" capture-home-worker-apply "$TURN/native_once.sh"
