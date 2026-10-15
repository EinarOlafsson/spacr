#!/bin/bash
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False YOLO_OFFLINE=1 YOLO_AUTOINSTALL=False
export TMPDIR=/mnt/wd4tb/scratch/n698-public-backend-integration-20261015
cd /mnt/wd4tb/spacr-worktrees/codex-n698-public-backend-integration-20261015
set +e
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /mnt/wd4tb/scratch/n698-web-runtime-20261014/bin/python -m pytest -q tests/test_web_shared_masks_n698.py tests/test_web_model_registry_n698.py tests/test_web_plaque_service_n698.py tests/test_web_mask_gradio_app_n698.py tests/test_web_mask_operator_command_n698.py tests/test_web_approved_examples_n698.py --junitxml=/mnt/wd4tb/scratch/n698-public-backend-integration-20261015/results.xml > /mnt/wd4tb/scratch/n698-public-backend-integration-20261015/run.log 2>&1
status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/n698-public-backend-integration-20261015/run.exit
exit "$status"
