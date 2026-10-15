#!/bin/bash
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR=/mnt/wd4tb/scratch/n698-native-nuclei-cpu-20261015
export HF_HOME=/mnt/wd4tb/scratch/n698-native-nuclei-cpu-20261015/hf-cache
export TORCH_HOME=/mnt/wd4tb/scratch/n698-native-nuclei-cpu-20261015/torch-cache
export CELLPOSE_LOCAL_MODELS_PATH=/mnt/wd4tb/scratch/n698-native-nuclei-cpu-20261015/cellpose-cache
export MPLCONFIGDIR=/mnt/wd4tb/scratch/n698-native-nuclei-cpu-20261015/matplotlib
export TORCHINDUCTOR_CACHE_DIR=/mnt/wd4tb/scratch/n698-native-nuclei-cpu-20261015/inductor
export TORCH_COMPILE_DISABLE=1
cd /mnt/wd4tb/spacr-worktrees/codex-f562-direct-table-identity-20261015
set +e
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G timeout 600 /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/n698-native-nuclei-cpu-20261015/parity.py > /mnt/wd4tb/scratch/n698-native-nuclei-cpu-20261015/native.log 2>&1
status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/n698-native-nuclei-cpu-20261015/native.exit
exit "$status"
