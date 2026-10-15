#!/usr/bin/env bash
set -eu
phase="$1"
shift
proof=/mnt/wd4tb/scratch/n197-shortcut-captions-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTEST_QT_API=pyside6 SPACR_NO_GL=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages:/mnt/wd4tb/spacr-worktrees/codex-home-b-ui-editorial-20261015
export XDG_CACHE_HOME="$proof/cache" MPLCONFIGDIR="$proof/mpl" TMPDIR="$proof/tmp"
exec tools/run_capped.sh 2G /usr/bin/time -v -o "$proof/$phase-time.txt" /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q -p no:randomly "$@" --basetemp="$proof/$phase-temp" --junitxml="$proof/$phase.xml" > "$proof/$phase.log" 2>&1
