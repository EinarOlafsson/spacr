#!/bin/bash
set -eu
cd /mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADIO_ANALYTICS_ENABLED=False HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR=/mnt/wd4tb/scratch/n698-managed-provisioning-20261015/pytest-channel-tmp
ci_cpu_index=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os;print(min(os.sched_getaffinity(0)))')
exec 9>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 9
set +e
tools/run_capped.sh 8G taskset -c "$ci_cpu_index" /mnt/wd4tb/scratch/n698-managed-gradio-launch-20261015/venv/bin/python -m pytest -q tests/test_web_api_channel_selection_n698.py tests/test_web_mask_gradio_app_n698.py::test_actual_app_builds_eight_class_tabs_without_server_or_model_start tests/test_web_mask_gradio_app_n698.py::test_refused_consent_and_bad_series_never_enter_backend tests/test_web_mask_gradio_app_n698.py::test_real_upload_choice_callbacks_expose_channels_and_handle_pdf tests/test_web_mask_managed_gradio_n698.py --junitxml=/mnt/wd4tb/scratch/n698-managed-provisioning-20261015/channel-api-final.xml > /mnt/wd4tb/scratch/n698-managed-provisioning-20261015/channel-api-final.log 2>&1
status=$?
echo "$status" > /mnt/wd4tb/scratch/n698-managed-provisioning-20261015/channel-api-final.exit
exit "$status"
