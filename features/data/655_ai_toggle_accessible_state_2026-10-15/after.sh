#!/bin/bash
set -uo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export XDG_CONFIG_HOME=/mnt/wd4tb/scratch/n655-ai-toggle-state-20261015/config
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages:.
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 2G /usr/bin/time -v -o /mnt/wd4tb/scratch/n655-ai-toggle-state-20261015/after.time /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/qt/test_655_ai_toggle_accessible_state.py tests/qt/test_655_ai_toggle_keyboard.py::test_actual_activation_key_toggles_once_when_given_focus tests/qt/test_655_ai_toggle_keyboard.py::test_mouse_click_still_proves_the_toggle_is_actionable > /mnt/wd4tb/scratch/n655-ai-toggle-state-20261015/after.log 2>&1
status=$?
echo "$status" > /mnt/wd4tb/scratch/n655-ai-toggle-state-20261015/after.exit
exit "$status"
