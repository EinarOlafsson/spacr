#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES=''
export QT_QPA_PLATFORM=offscreen
export PYTEST_QT_API=pyside6
export SPACR_NO_GL=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
export TMPDIR=/mnt/wd4tb/scratch/n506-tooltip-lifetime-20261014/tmp
export PYTHONPYCACHEPREFIX=/mnt/wd4tb/scratch/n506-tooltip-lifetime-20261014/pycache
export MPLCONFIGDIR=/mnt/wd4tb/scratch/n506-tooltip-lifetime-20261014/mpl
export XDG_CACHE_HOME=/mnt/wd4tb/scratch/n506-tooltip-lifetime-20261014/xdgcache
export COVERAGE_FILE=/mnt/wd4tb/scratch/n506-tooltip-lifetime-20261014/.coverage.owning
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-tooltips-20261014 || exit 2
tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q -p no:randomly --tb=short "$@"
