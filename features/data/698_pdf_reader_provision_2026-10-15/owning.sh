#!/usr/bin/env bash
set -uo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTEST_QT_API=pyside6 SPACR_NO_GL=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
export TMPDIR=/mnt/wd4tb/scratch/n702-cli-pins-20261014
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 8G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_web_mask_upload_inputs_n698.py::test_real_pdf_pages_render_at_explicit_dpi_without_writing_inputs tests/test_web_mask_upload_inputs_n698.py::test_pdf_selection_and_limits_are_checked_before_render tests/test_web_mask_upload_inputs_n698.py::test_pdf_missing_reader_and_page_limit_are_explained tests/test_dependency_profiles.py::test_minimum_profile_covers_every_lower_bounded_direct_requirement tests/test_dependency_profiles.py::test_minimum_profile_versions_obey_declared_ranges tests/test_optimal_dependency_versions.py::test_every_constraint_pin_is_the_floor_or_explains_itself tests/test_optimal_dependency_versions.py::test_no_pin_sits_below_the_floor_it_is_meant_to_verify > /mnt/wd4tb/scratch/n698-pdf-reader-provision-20261015/owning.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/wd4tb/scratch/n698-pdf-reader-provision-20261015/owning.exit
exit "$result"
