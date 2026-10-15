#!/usr/bin/env bash
# Root launches only after source/helper/card review. No install or audio synthesis.
set -euo pipefail
ROOT=/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014
W=/media/carruthers/mnt3/claude/n676_stages/work/lesson02-selective-20261015
SP=/home/carruthers/anaconda3/envs/spacr/bin/python
: "${SOURCE_COMMIT:?Exact reviewed clean renderer/validator source commit required}"
: "${APPROVED_CARD_EVIDENCE:?Actual root card acceptance required}"
: "${APPROVED_CARD_SHA256:?Actual card acceptance hash required}"
export SOURCE_COMMIT APPROVED_CARD_EVIDENCE APPROVED_CARD_SHA256
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export SELECTIVE_STAGE=${SELECTIVE_STAGE:-/media/carruthers/mnt3/claude/n676_stages/host/stage-02-selective-r2-20261015}
[[ "${1:-}" == stage || "${1:-}" == overlay || "${1:-}" == all ]] || { echo 'Usage: run_stage_and_overlay.sh stage|overlay|all' >&2; exit 2; }
cd "$ROOT"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" && -z "$(git status --porcelain)" ]] || exit 2
export SELECTIVE_LEASE_LABEL="codex-ws-02-selective-overlay-$$" SELECTIVE_LEASE_PID=$$
lease() {
 "$ROOT/tools/run_capped.sh" 1G "$SP" - "$1" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0,'/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
import fcntl,os,datetime
p=Path('/media/carruthers/mnt3/claude/ram_leases.md');label=os.environ['SELECTIVE_LEASE_LABEL'];token='owner_pid='+os.environ['SELECTIVE_LEASE_PID']
with p.open('r+') as f:
 fcntl.flock(f,fcntl.LOCK_EX);rows=f.readlines()
 if sys.argv[1]=='add':
  assert not any(r.startswith(label+' | ') for r in rows)
  f.seek(0,2);f.write('\n'+label+' | approved02scene7-only CPUstage/overlay | 10 GB | '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' | '+token+' | removed on phase exit\n')
 else:
  f.seek(0);f.writelines(r for r in rows if not(r.startswith(label+' | ') and token in r.split(' | ')));f.truncate()
 f.flush();os.fsync(f.fileno())
PY
}
lease add
trap 'lease remove' EXIT
phase() {
 local name=$1 cap=$2
 "$ROOT/tools/run_capped.sh" "$cap" "$SP" /media/carruthers/mnt3/claude/n676_stages/work/run_logged.py "$W/scene7-$name-r2.log" "$SP" "$W/stage_visual_overlay.py" "$name"
}
[[ "$1" == overlay ]] || phase stage 2G
[[ "$1" == stage ]] || phase overlay 8G
