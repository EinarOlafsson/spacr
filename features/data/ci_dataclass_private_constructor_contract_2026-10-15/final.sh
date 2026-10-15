#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n703-debian-adapter-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
status=0
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock tools/run_capped.sh 8G /home/olafsson/anaconda3/bin/python -m pytest -q tests/test_docstring_correctness.py::test_generated_constructor_models_resist_signature_evasions tests/test_docstring_correctness.py::test_no_new_public_callable_ghost_parameters > /mnt/wd4tb/scratch/ci-dataclass-private-constructor-20261015/final.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/ci-dataclass-private-constructor-20261015/final.exit
exit "$status"
