#!/usr/bin/env bash
set -uo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTEST_QT_API=pyside6 SPACR_NO_GL=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
export TMPDIR=/mnt/wd4tb/scratch/n702-cli-pins-20261014
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 8G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/qt/test_cov_qt_app.py::test_every_app_carries_the_maturity_it_was_given tests/qt/test_app_maturity.py::test_an_unassessed_module_never_reads_stable tests/qt/test_app_maturity.py::test_no_module_reads_stable_without_somebody_having_said_so tests/qt/test_home_v2.py::test_the_alpha_and_beta_lists_are_the_ones_that_were_asked_for tests/qt/test_home_v2.py::test_declared_stable_registration_survives_maturity_while_undeclared_defaults_alpha tests/qt/test_app_maturity.py::test_a_module_that_declares_nothing_lands_on_alpha tests/qt/test_app_maturity.py::test_the_default_never_overrules_a_module_that_did_declare_one > /mnt/wd4tb/scratch/ci-app-maturity-20261015/fixed.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/wd4tb/scratch/ci-app-maturity-20261015/fixed.exit
exit "$result"
