#!/bin/bash
set -euo pipefail
ulimit -c 0
out=/mnt/wd4tb/scratch/ci-second-qt-fixtures-20261015
exec > "$out/plainfield-final.log" 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$out/plainfield-final.exit"' EXIT
exec 9>/mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock
flock 9
exec 8>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 8
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SPACR_NO_NETWORK=1 SPACR_NO_GL=1 HF_HUB_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages"
/usr/bin/time -v tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -p pytestqt.plugin -q \
 tests/qt/test_app_screen_switches_and_run_changes_at_their_edges.py::test_a_chosen_model_is_written_into_a_plain_field \
 --basetemp="$out/pytest-plainfield-final" --junitxml="$out/plainfield-final.xml"
