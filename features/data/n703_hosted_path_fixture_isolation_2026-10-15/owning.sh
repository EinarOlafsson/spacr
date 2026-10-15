#!/bin/bash
set -u
TASK_SCRATCH=/mnt/wd4tb/scratch/n703-hosted-path-fixtures-20261015
cd /mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen SPACR_NO_GL=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPYCACHEPREFIX=$TASK_SCRATCH/pycache TMPDIR=$TASK_SCRATCH/tmp XDG_CONFIG_HOME=$TASK_SCRATCH/config XDG_CACHE_HOME=$TASK_SCRATCH/cache
export PYTHONPATH=/mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015:$TASK_SCRATCH/hosted-checkout-metadata
mkdir -p "$TMPDIR" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest tests/test_macos_runtime_transaction_703.py::test_offline_staging_keeps_previous_runtime_and_rolls_back_after_activation 'tests/test_macos_runtime_transaction_703.py::test_unsupported_environment_is_refused_before_any_package_mutation[editable]' tests/test_macos_runtime_transaction_703.py::test_unknown_target_is_refused_without_package_mutation tests/test_macos_runtime_transaction_703.py::test_failed_restoration_keeps_original_bytes_journal_and_lock tests/test_current_runtime_floor_703.py tests/test_runtime_source_timestamp_collision_703.py tests/test_pip_runtime_transaction_703.py::test_real_pip_transaction_keeps_the_original_until_verified -q --tb=short --junitxml="$TASK_SCRATCH/owning.xml" --basetemp="$TASK_SCRATCH/owning-tmp" -o cache_dir="$TASK_SCRATCH/pytest-cache" > "$TASK_SCRATCH/owning.log" 2>&1
TASK_RC=$?
echo "$TASK_RC" > "$TASK_SCRATCH/owning.exit"
exit "$TASK_RC"
