#!/bin/bash
set -euo pipefail
ulimit -c 0
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
out=/mnt/wd4tb/scratch/full-nine-api-readme-audit-20261015
exec > "$out/run.log" 2>&1
trap 'rc=$?; date -u +%Y-%m-%dT%H:%M:%SZ > "$out/finished.utc"; printf "%s\n" "$rc" > "$out/exit"' EXIT
exec 9>/mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock
flock 9
exec 8>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 8
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GIT_NO_LAZY_FETCH=1
 date -u +%Y-%m-%dT%H:%M:%SZ > "$out/started.utc"
/home/olafsson/anaconda3/bin/python "$out/pin-inputs.py" "$out/before.json"
printf '%s\n' 'PHASE full-nine-locale-api-readme-audit'
set +e
/usr/bin/time -v env CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen tools/run_capped.sh 8G /home/olafsson/anaconda3/bin/python tools/build_documentation_i18n.py --audit > "$out/audit.log" 2> "$out/resources.log"
audit_rc=$?
set -e
printf '%s\n' "$audit_rc" > "$out/audit.exit"
/home/olafsson/anaconda3/bin/python "$out/pin-inputs.py" "$out/after.json"
cmp "$out/before.json" "$out/after.json"
printf '%s\n' 'Input/source/catalog/README hashes and Git state unchanged.'
exit "$audit_rc"
