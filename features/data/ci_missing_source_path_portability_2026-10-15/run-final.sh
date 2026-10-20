#!/usr/bin/env bash
set -euo pipefail
TASK_OUT=/mnt/wd4tb/scratch/ci-docsource-dpr-hygiene-20261015
exec > "$TASK_OUT/final.log" 2>&1
trap 'printf "%s\n" "$?" > "$TASK_OUT/final.exit"' EXIT
exec 9>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 9
cd /mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD"
tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr/bin/python "$TASK_OUT/pytest-offline.py" -q 'tests/test_docstring_correctness.py::test_repaired_record_documents_every_constructor_parameter[spacr.runctx.RunContext]' tests/test_docstring_inline_markup_renders.py::test_no_fault_outside_the_named_files tests/test_docstring_inline_markup_renders.py::test_the_remaining_count_does_not_rise tests/test_n702_cluster_download_http.py::test_missing_laptop_path_downloads_verified_bytes_and_logs_one_line tests/test_n702_cluster_download_http.py::test_pinned_version_downloads_exactly_the_recorded_identity tests/test_test_suite_hygiene.py::test_no_user_home_paths_in_the_suite --junitxml="$TASK_OUT/final.xml"
