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
unset SPACR_HOME PYTHONPATH PYTHONHOME PIP_EXTRA_INDEX_URL PIP_NO_INDEX
export PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple PIP_FIND_LINKS="$TASK_ROOT/wheelhouse" PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_REQUIRE_VIRTUALENV=1
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SPACR_MODEL_DOWNLOAD=off
cd "$TASK_DIR"
"$TASK_TREE/tools/run_capped.sh" 2G "$TASK_DIR/venv/bin/python" -m pip download --no-deps --only-binary=:all: --require-hashes --retries 0 --timeout 20 --index-url https://pypi.org/simple --dest "$TASK_ROOT/wheelhouse" -r "$TASK_DIR/resolved-small-wheels.txt" >"$TASK_DIR/dependency-download.log" 2>&1
"$TASK_DIR/venv/bin/python" - <<'PY'
from pathlib import Path
import hashlib,json,subprocess,zipfile
from email.parser import BytesParser
r=Path('/mnt/wd4tb/scratch/candidate-1514-b51fb194bb-20261015');p=r/'smoke-preflight';pins=json.loads((p/'resolved-small-wheels.json').read_text());out=[]
for row in pins:
 matches=[]
 for wheel in (r/'wheelhouse').glob('*.whl'):
  with zipfile.ZipFile(wheel) as z:
   md=BytesParser().parsebytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))))
  if md['Name'].lower().replace('_','-')==row['name'].lower().replace('_','-') and md['Version']==row['version']:matches.append(wheel)
 assert len(matches)==1,(row['name'],matches)
 w=matches[0];sha=hashlib.sha256(w.read_bytes()).hexdigest();assert sha==row['sha256'];out.append(dict(row,path=str(w),bytes=w.stat().st_size))
assert sum(row['bytes'] for row in out)<50*1024*1024
(p/'downloaded-small-wheels.json').write_text(json.dumps(out,indent=2)+'\n')
tree=Path('/mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015');source='b51fb194bb183990ab25c981d70e3030063bf9ac';helper=tree/'packaging/smoke_distribution_artifacts.py';normal=subprocess.check_output(['git','-C',str(tree),'show',source+':packaging/smoke_distribution_artifacts.py']);assert helper.read_bytes()==normal
expected={'spacr-1.5.1.4-py3-none-any.whl':'0be5f07cf06de0c5d0273267b9da2036378a8f3ad081596b556f71270fdf84ac','spacr-1.5.1.4.tar.gz':'b552e71a09f5448ec1e7f53a8c603ed45d3c3fdda39e5dc6db58a15f5b1936db'}
actual={n:hashlib.sha256((r/'artifacts'/n).read_bytes()).hexdigest() for n in expected};assert actual==expected
(p/'full-smoke-before.json').write_text(json.dumps({'source':source,'helper_sha256':hashlib.sha256(normal).hexdigest(),'artifacts':actual,'dependency_wheel_bytes':sum(row['bytes'] for row in out)},indent=2)+'\n')
PY
export PIP_NO_INDEX=1
TASK_CPU=$(/usr/bin/python3 -c 'import os;print(min(os.sched_getaffinity(0)))')
set +e
"$TASK_TREE/tools/run_capped.sh" 8G taskset -c "$TASK_CPU" timeout --signal=TERM --kill-after=15s 1100 "$TASK_DIR/venv/bin/python" "$TASK_TREE/packaging/smoke_distribution_artifacts.py" --dist-dir "$TASK_ROOT/artifacts" --extras qt --report "$TASK_DIR/full-smoke.json" >"$TASK_DIR/full-smoke.log" 2>&1
TASK_EXIT=$?
set -e
printf '%s\n' "$TASK_EXIT" >"$TASK_DIR/full-smoke.exit"
/usr/bin/python3 - <<'PY'
from pathlib import Path
import hashlib,json
r=Path('/mnt/wd4tb/scratch/candidate-1514-b51fb194bb-20261015');p=r/'smoke-preflight';before=json.loads((p/'full-smoke-before.json').read_text());actual={n:hashlib.sha256((r/'artifacts'/n).read_bytes()).hexdigest() for n in before['artifacts']};after={'artifacts':actual,'unchanged':actual==before['artifacts']};(p/'full-smoke-after.json').write_text(json.dumps(after,indent=2)+'\n');assert after['unchanged']
PY
exit "$TASK_EXIT"
