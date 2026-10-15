#!/bin/bash
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTEST_QT_API=pyside6 SPACR_NO_GL=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
export XDG_CONFIG_HOME=/mnt/wd4tb/scratch/n703-cache-cleanup-gui-20261015/qsettings SPACR_HOME=/mnt/wd4tb/scratch/n703-cache-cleanup-gui-20261015/app-home
export TMPDIR=/mnt/wd4tb/scratch/n702-cli-pins-20261014
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export PYTHONPATH="$PWD:$PYTHONPATH"
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_n703_optional_cleanup_preview.py::test_sound_override_strips_whitespace_while_other_writer_paths_remain_raw -n 0 --junitxml=/mnt/wd4tb/scratch/n703-cache-cleanup-gui-20261015/sounds-owning.xml > /mnt/wd4tb/scratch/n703-cache-cleanup-gui-20261015/sounds-owning.log 2>&1
status=$?
echo "$status" > /mnt/wd4tb/scratch/n703-cache-cleanup-gui-20261015/sounds-owning.exit
exit "$status"
