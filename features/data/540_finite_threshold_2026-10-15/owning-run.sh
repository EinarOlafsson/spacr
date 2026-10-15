#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-f540-finite-threshold-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
phase_status=0
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_viability_finite_threshold_f540.py tests/test_live_dead_viability.py::test_manual_thresholds_parse_and_bad_ones_are_refused tests/test_viability_at_its_edges.py::test_a_run_measured_on_cells_is_called_by_its_stain_only tests/test_viability_at_its_edges.py::test_morphology_reads_the_first_channel_and_draws_its_log_cut -k 'not installed_reviewed_language' --basetemp=/mnt/wd4tb/scratch/f540-finite-threshold-20261015/owning-pytest --junitxml=/mnt/wd4tb/scratch/f540-finite-threshold-20261015/owning.xml > /mnt/wd4tb/scratch/f540-finite-threshold-20261015/owning.log 2>&1 || phase_status=$?
printf '%s\n' "$phase_status" > /mnt/wd4tb/scratch/f540-finite-threshold-20261015/owning.exit
exit "$phase_status"
