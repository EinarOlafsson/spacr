#!/bin/bash
set -euo pipefail
ulimit -c 0
out=/mnt/wd4tb/scratch/ci-model-resolution-fixtures-20261015
exec > "$out/baseline.log" 2>&1
trap 'printf "%s\n" "$?" > "$out/baseline.exit"' EXIT
exec 9>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 9
cd /mnt/wd4tb/spacr-worktrees/codex-n702-picker-pin-provenance-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD"
tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr/bin/python "$out/pytest-offline.py" -q tests/test_cov_object_cellpose_masks.py::test_a_checkpoint_path_that_is_not_there_stops_the_run tests/test_cellpose_dino_runs_in_its_own_environment.py::test_load_backend_refuses_what_it_cannot_run tests/test_item60_deep_spacr_coverage.py::test_apply_model_to_tar_reports_crop_format tests/test_item60_deep_spacr_coverage.py::test_generate_activation_map_registered_attribution_path tests/test_item60_deep_spacr_coverage.py::test_train_model_rejects_missing_initial_checkpoint tests/test_colony_cfu_counting.py::test_a_colony_detector_that_is_neither_file_nor_zoo_key_is_refused tests/test_cov_deep_spacr_entry_fusion.py::test_deep_spacr_relative_tar_regenerated_and_missing_model_skips tests/test_mask_worker_model_identity.py::test_a_missing_custom_checkpoint_is_not_downloaded_as_stock --junitxml="$out/baseline.xml"
