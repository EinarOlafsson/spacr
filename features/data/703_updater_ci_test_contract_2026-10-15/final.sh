#!/bin/bash
set -uo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export XDG_CONFIG_HOME=/mnt/wd4tb/scratch/n703-updater-tests-20261015/config
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages:.
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 4G /usr/bin/time -v -o /mnt/wd4tb/scratch/n703-updater-tests-20261015/final.time /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q -s tests/qt/test_cov_qt_app.py::test_accepting_an_upgrade_waits_for_authenticated_shutdown_handoff tests/qt/test_cov_qt_app.py::test_an_unavailable_update_plan_reports_its_error_without_closing tests/qt/test_cov_qt_app.py::test_an_unreachable_update_server_is_reported tests/qt/test_cov_qt_app.py::test_the_result_is_delivered_on_the_gui_thread tests/qt/test_cov_qt_app.py::test_being_up_to_date_says_so tests/qt/test_cov_qt_app.py::test_declining_an_upgrade_does_nothing > /mnt/wd4tb/scratch/n703-updater-tests-20261015/final.log 2>&1
status=$?
echo "$status" > /mnt/wd4tb/scratch/n703-updater-tests-20261015/final.exit
exit "$status"
