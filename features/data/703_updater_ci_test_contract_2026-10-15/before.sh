#!/bin/bash
set -uo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export XDG_CONFIG_HOME=/mnt/wd4tb/scratch/n703-updater-tests-20261015/config
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages:.
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 4G /usr/bin/time -v -o /mnt/wd4tb/scratch/n703-updater-tests-20261015/before.time /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q -s tests/qt/test_cov_qt_app.py::test_accepting_an_upgrade_runs_pip tests/qt/test_cov_qt_app.py::test_a_failing_pip_upgrade_shows_its_exit_code > /mnt/wd4tb/scratch/n703-updater-tests-20261015/before.log 2>&1
status=$?
echo "$status" > /mnt/wd4tb/scratch/n703-updater-tests-20261015/before.exit
exit "$status"
