#!/bin/bash
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd /mnt/wd4tb/spacr-worktrees/codex-f535-torch-grouped-split-20261015
set +e
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_cell_cycle_torch_grouped_holdout.py tests/test_cell_cycle_phase_three_ways.py::test_torch_training_fixture_supports_two_grouped_holdouts tests/test_cell_cycle_phase_three_ways.py::test_tabular_phase_shared_split_keeps_requested_identity_disjoint tests/test_cell_cycle_phase_three_ways.py::test_tabular_phase_inadequate_holdout_has_no_invented_scores --tb=short --junitxml=/mnt/wd4tb/scratch/f535-torch-grouped-split-20261015/final2.xml > /mnt/wd4tb/scratch/f535-torch-grouped-split-20261015/final2.log 2>&1
status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/f535-torch-grouped-split-20261015/final2.exit
exit "$status"
