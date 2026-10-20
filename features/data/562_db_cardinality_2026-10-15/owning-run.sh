#!/bin/bash
cd /mnt/wd4tb/spacr-worktrees/codex-f562-db-cardinality-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_mil_db_cardinality_f562.py tests/test_multiple_instance_learning.py -k 'not synthetic and not attention and not table_route and not stored_embeddings_join' --basetemp=/mnt/wd4tb/scratch/f562-db-cardinality-20261015/owning-pytest --junitxml=/mnt/wd4tb/scratch/f562-db-cardinality-20261015/owning.xml > /mnt/wd4tb/scratch/f562-db-cardinality-20261015/owning.log 2>&1
printf '%s\n' "$?" > /mnt/wd4tb/scratch/f562-db-cardinality-20261015/owning.exit
