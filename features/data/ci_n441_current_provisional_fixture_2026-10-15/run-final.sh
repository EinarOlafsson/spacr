#!/bin/bash
set -euo pipefail
ulimit -c 0
out=/mnt/wd4tb/scratch/ci-n441-provisional-fixture-20261015
exec > "$out/final.log" 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$out/final.exit"' EXIT
exec 9>/mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock
flock 9
exec 8>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 8
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SPACR_NO_NETWORK=1 SPACR_NO_GL=1 HF_HUB_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages"
/usr/bin/time -v tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -p pytestqt.plugin -q tests/qt/test_n441_metadata_actual_locales.py::test_actual_metadata_widgets_use_all_ten_installed_translations --basetemp="$out/pytest-final" --junitxml="$out/final.xml"
