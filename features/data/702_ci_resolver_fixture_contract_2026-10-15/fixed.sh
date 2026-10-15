#!/usr/bin/env bash
set -uo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTEST_QT_API=pyside6 SPACR_NO_GL=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/wd4tb/scratch/ci-n702-resolution-20261015:/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
export TMPDIR=/mnt/wd4tb/scratch/n702-cli-pins-20261014
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 8G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q -p ci_network_guard -p ci_existing_qapp tests/test_missing_model_downloads.py::test_prefixed_model_validation_does_not_download tests/test_missing_model_downloads.py::test_preflight_recognizes_downloadable_model_without_fetching tests/test_submodules_plaque_well_integration.py::test_a_plaque_model_zoo_key_is_fetched tests/test_submodules_plaque_well_integration.py::test_true_fetches_the_default_well_detector tests/test_missing_model_downloads.py::test_gui_thread_never_starts_a_missing_checkpoint_download tests/test_n702_legacy_planning_never_discovers_providers.py > /mnt/wd4tb/scratch/ci-n702-resolution-20261015/fixed.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/wd4tb/scratch/ci-n702-resolution-20261015/fixed.exit
exit "$result"
