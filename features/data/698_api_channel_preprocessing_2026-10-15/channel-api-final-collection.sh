#!/bin/bash
set -eu
cd /mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADIO_ANALYTICS_ENABLED=False HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR=/mnt/wd4tb/scratch/n698-managed-provisioning-20261015/pytest-channel-tmp
ci_cpu_index=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os;print(min(os.sched_getaffinity(0)))')
exec 9>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 9
set +e
tools/run_capped.sh 8G taskset -c "$ci_cpu_index" /mnt/wd4tb/scratch/n698-managed-gradio-launch-20261015/venv/bin/python -m pytest -q tests/test_web_api_channel_selection_n698.py::test_managed_empty_channel_choices_accept_explicit_api_channel --junitxml=/mnt/wd4tb/scratch/n698-managed-provisioning-20261015/channel-api-final-collection.xml > /mnt/wd4tb/scratch/n698-managed-provisioning-20261015/channel-api-final-collection.log 2>&1
status=$?
echo "$status" > /mnt/wd4tb/scratch/n698-managed-provisioning-20261015/channel-api-final-collection.exit
exit "$status"
