#!/bin/bash
set -euo pipefail
ulimit -c 0
base=/mnt/wd4tb/scratch/n700-provider-core-authoring-20261015
out="$base/normal26-run"
mkdir -p "$out"
exec > "$out/run.log" 2>&1
trap 'rc=$?; date -u +%Y-%m-%dT%H:%M:%SZ > "$out/finished.utc"; printf "%s\n" "$rc" > "$out/exit"' EXIT
exec 9>/mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock
flock 9
exec 8>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 8
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 GIT_NO_LAZY_FETCH=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
date -u +%Y-%m-%dT%H:%M:%SZ > "$out/started.utc"
/usr/bin/time -v tools/run_capped.sh 8G /home/olafsson/anaconda3/bin/python "$base/normal26-producer.py" "$base/normal26-config.FINAL.json"
