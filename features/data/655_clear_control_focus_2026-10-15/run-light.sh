#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export QT_QPA_PLATFORM=offscreen
export PYTEST_QT_API=pyside6
export SPACR_NO_GL=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
export TMPDIR=/mnt/wd4tb/scratch/n655-clear-focus-20261015/tmp
export PYTHONPYCACHEPREFIX=/mnt/wd4tb/scratch/n655-clear-focus-20261015/pycache
export MPLCONFIGDIR=/mnt/wd4tb/scratch/n655-clear-focus-20261015/mpl
export XDG_CACHE_HOME=/mnt/wd4tb/scratch/n655-clear-focus-20261015/xdgcache
export COVERAGE_FILE=/mnt/wd4tb/scratch/n655-clear-focus-20261015/.coverage.owning
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-workflow-keyboard-20261015 || exit 2
tools/run_capped.sh 2G /usr/bin/time -v -o "${SPACR_TEST_TIMING:?}" /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q -p no:randomly --tb=short "$@"
