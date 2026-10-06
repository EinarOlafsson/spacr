#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
archive="${root}/features/data/43_hosted_1f3_complete_failures_2026-10-06"
cd "$root"
python "$archive/VERIFY.py"

target='tests/test_native_tzyx_batch_f548.py::test_real_mask_pipeline_receives_each_native_volume_with_z_axis'
CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen tools/run_capped.sh 4G python -m pytest -q -p no:randomly --tb=short "$target"

scratch="$(mktemp -d /mnt/wd4tb/scratch/cellpose-407-archive.XXXXXX)"
trap 'rm -rf "$scratch"' EXIT
python -m zipfile -e "$archive/proofs/cellpose-4.0.7-py3-none-any.whl" "$scratch/package"
PYTHONPATH="$scratch/package:." python -c 'import cellpose; assert cellpose.__file__.startswith("$scratch/package"")'
PYTHONPATH="$scratch/package:." CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen tools/run_capped.sh 4G python -m pytest -q -p no:randomly --tb=short "$target"

CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen tools/run_capped.sh 4G python -m pytest -q -p no:randomly --tb=short tests/test_native_tzyx_batch_f548.py
