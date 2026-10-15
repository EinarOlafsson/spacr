#!/usr/bin/env bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-f370-generator-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out=/mnt/wd4tb/scratch/home-b-f370-generator-20261015
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 2G /usr/bin/time -v -o "$out/glossary-after.time" /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_f370_scorecard_generator_bindings.py -k archived_metrics --junitxml="$out/glossary-after.xml" > "$out/glossary-after.log" 2>&1
result=$?
printf '%s\n' "$result" > "$out/glossary-after.exit"
exit "$result"
