#!/bin/bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/usr/bin/time -v -o /mnt/wd4tb/scratch/readme-picker-audit-contract-20261015/malformed-before.time tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_readme_picker_audit_contract.py -k unknown_html_declaration > /mnt/wd4tb/scratch/readme-picker-audit-contract-20261015/malformed-before.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/wd4tb/scratch/readme-picker-audit-contract-20261015/malformed-before.exit
exit "$result"
