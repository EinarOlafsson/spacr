#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTEST_QT_API=pyside6 SPACR_NO_GL=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
export XDG_CONFIG_HOME=/mnt/wd4tb/scratch/n703-coalesced-runtime-20261015/qsettings SPACR_HOME=/mnt/wd4tb/scratch/n703-coalesced-runtime-20261015/app-home
export TMPDIR=/mnt/wd4tb/scratch/n702-cli-pins-20261014
cd /mnt/wd4tb/spacr-worktrees/codex-n703-debian-adapter-20261015
export PYTHONPATH="$PWD:$PYTHONPATH"
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /bin/bash /mnt/wd4tb/scratch/n703-coalesced-runtime-20261015/inside.sh > /mnt/wd4tb/scratch/n703-coalesced-runtime-20261015/final66-run.log 2>&1
