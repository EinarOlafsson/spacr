#!/usr/bin/env bash
set -uo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTEST_QT_API=pyside6 SPACR_NO_GL=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
export TMPDIR=/mnt/wd4tb/scratch/n702-cli-pins-20261014
ci_cpu_index=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os; print(min(os.sched_getaffinity(0)))')
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock tools/run_capped.sh 8G taskset -c "$ci_cpu_index" /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/qt/test_a_pyqtgraph_plot_is_styled_for_the_file_too.py::test_size_is_inherited_but_raster_resolution_stays_editable tests/qt/test_the_save_dialog_changes_the_page_too.py::TestTheGreyedControlsSayWhy::test_the_resolution_reason_says_vector_has_none tests/qt/test_the_save_figure_dialog_four_things.py::TestTheResolutionFollowsTheFormat::test_pdf_and_svg_have_none tests/qt/test_the_save_figure_dialog_four_things.py::TestTheResolutionFollowsTheFormat::test_it_comes_back_when_png_is_chosen_again > /mnt/wd4tb/scratch/ci-save-figure-caption-contracts-20261015/fixed.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/wd4tb/scratch/ci-save-figure-caption-contracts-20261015/fixed.exit
exit "$result"
