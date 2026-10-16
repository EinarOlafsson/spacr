#!/bin/bash
set -euo pipefail
ulimit -c 0
out=/mnt/wd4tb/scratch/ci-model-resolution-second-20261015
exec > "$out/final.log" 2>&1
trap 'printf "%s\n" "$?" > "$out/final.exit"' EXIT
exec 9>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 9
cd /mnt/wd4tb/spacr-worktrees/codex-n702-picker-pin-provenance-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD"
tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr/bin/python "$out/pytest-offline.py" -q tests/test_coverage_fill_cellpose_gpu_funcs.py::TestIdentifyMasksFinetune::test_custom_model_not_found_refuses_stock_fallback tests/test_prefixed_backends_run_in_their_own_environment.py::test_load_backend_refuses_what_it_cannot_run tests/test_inference_augmentation.py::test_folder_and_tar_use_same_augmentation_and_save_diagnostics tests/test_plaque_assay_growth_models_and_figure_notes.py::test_a_zoo_base_without_a_local_file_is_fetched_into_the_model_folder tests/test_470_real_object_classifier.py::test_a_missing_channel_skips_rather_than_guesses --junitxml="$out/final.xml"
