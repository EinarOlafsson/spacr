#!/usr/bin/env bash
set -euo pipefail
capture_root=/mnt/wd4tb/scratch/spacr-completion/366-live-repaired-full
mkdir -p "$capture_root/home" "$capture_root/tmp"
cd /mnt/wd4tb/scratch/spacr-completion/worktree
exec tools/run_capped.sh 4G env HOME="$capture_root/home" XDG_CONFIG_HOME="$capture_root/home/config" \
    TMPDIR="$capture_root/tmp" CUDA_VISIBLE_DEVICES= QT_QPA_PLATFORM=offscreen \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    timeout --signal=TERM --kill-after=10s 300s \
    /mnt/firecuda2/Claude/toxoplasma_projects/tutorials/refresh_2026-09-09/.venv/bin/python \
    "$capture_root/measure.py"
