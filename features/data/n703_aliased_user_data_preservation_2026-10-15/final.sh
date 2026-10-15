#!/bin/bash
set -u
TASK_SCRATCH=/mnt/wd4tb/scratch/n703-aliased-user-data-20261015
cd /mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPYCACHEPREFIX=$TASK_SCRATCH/pycache TMPDIR=$TASK_SCRATCH/tmp XDG_CONFIG_HOME=$TASK_SCRATCH/config XDG_CACHE_HOME=$TASK_SCRATCH/cache
mkdir -p "$TMPDIR" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest tests/test_aliased_user_data_preservation_703.py tests/test_install_cleanup.py::test_every_released_layout_is_found_and_removed_down_to_user_data tests/test_install_cleanup.py::test_the_remover_will_not_delete_shared_folders_user_data_or_anything_outside tests/test_retirement_respects_update_leases_703.py -k "not actual_removal_refusal_uses_exact_reviewed_locale" -q --tb=short --junitxml="$TASK_SCRATCH/final.xml" --basetemp="$TASK_SCRATCH/final-tmp" -o cache_dir="$TASK_SCRATCH/pytest-cache" > "$TASK_SCRATCH/final.log" 2>&1
TASK_RC=$?
echo "$TASK_RC" > "$TASK_SCRATCH/final.exit"
exit "$TASK_RC"
