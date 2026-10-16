#!/bin/bash
set -euo pipefail
ulimit -c 0
out=/mnt/wd4tb/scratch/ci-integration-38816815119-20261015
exec > "$out/final.log" 2>&1
trap 'printf "%s\n" "$?" > "$out/final.exit"' EXIT
exec 9>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 9
cd /mnt/wd4tb/spacr-worktrees/codex-n702-picker-pin-provenance-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD"
tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr/bin/python "$out/pytest-offline.py" -q tests/test_submodules_plaque_well_integration.py::test_an_unknown_well_detector_key_is_refused tests/test_submodules_plaque_well_integration.py::test_a_disabled_well_detector_leaves_the_source_untouched tests/test_submodules_plaque_well_integration.py::test_a_local_well_detector_checkpoint_wins --junitxml="$out/final.xml"
