#!/usr/bin/env bash
set -euo pipefail
ROOT=/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014
phase=$1; shift
export SPACR_02_PHASE_LABEL="codex-ws-lesson02-selective-${phase}-$$" SPACR_02_PHASE_PID=$$
lease() {
 "$ROOT/tools/run_capped.sh" 1G /usr/bin/python3 - "$1" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0,'/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
import fcntl,os,datetime
p=Path('/media/carruthers/mnt3/claude/ram_leases.md');label=os.environ['SPACR_02_PHASE_LABEL'];token='owner_pid='+os.environ['SPACR_02_PHASE_PID']
with p.open('r+') as f:
 fcntl.flock(f,fcntl.LOCK_EX);rows=f.readlines()
 if sys.argv[1]=='add':
  assert not any(r.startswith(label+' | ') for r in rows)
  f.seek(0,2);f.write('\n'+label+' | lesson02 selective downstream coordinator/tools; no synthesis or native capture | 10 GB | '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' | '+token+' | removed on phase exit\n')
 else:
  f.seek(0);f.writelines(r for r in rows if not(r.startswith(label+' | ') and token in r.split(' | ')));f.truncate()
 f.flush();os.fsync(f.fileno())
PY
}
lease add
trap 'lease remove' EXIT
"$@"
