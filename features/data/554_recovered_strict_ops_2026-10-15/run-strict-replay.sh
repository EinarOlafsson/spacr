#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-f564-pretrained-reuse-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen SPACR_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MPLCONFIGDIR=/mnt/wd4tb/scratch/f554-recovered-inputs-20261015/mpl TMPDIR=/mnt/wd4tb/scratch/f554-recovered-inputs-20261015/tmp
mkdir -p "$MPLCONFIGDIR" "$TMPDIR"
status=0
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 8G /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/f554-recovered-inputs-20261015/readonly-replay.py /mnt/wd4tb/spacr-worktrees/codex-f564-pretrained-reuse-20261015 /mnt/wd4tb/scratch/f554-recovered-inputs-20261015/path-provenance-packet.json > /mnt/wd4tb/scratch/f554-recovered-inputs-20261015/strict-real-ops.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/f554-recovered-inputs-20261015/strict-real-ops.exit
exit "$status"
