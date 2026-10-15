#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-measured-channel-identity-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 SPACR_LANGUAGE=en
phase_status=0
tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_measured_channel_identity.py tests/test_cell_cycle_at_its_edges.py::test_the_channel_helpers_fall_back_to_the_measured_channels tests/test_cell_cycle_at_its_edges.py::test_one_method_with_fucci_writes_no_agreement_table tests/test_cell_cycle_phase_three_ways.py::test_fucci_states_come_from_two_reporters tests/test_viability_at_its_edges.py::test_a_run_measured_on_cells_is_called_by_its_stain_only tests/test_viability_at_its_edges.py::test_morphology_reads_the_first_channel_and_draws_its_log_cut -k 'not installed_reviewed_language' --basetemp=/mnt/wd4tb/scratch/measured-channel-identity-20261015/owning-pytest --junitxml=/mnt/wd4tb/scratch/measured-channel-identity-20261015/owning.xml > /mnt/wd4tb/scratch/measured-channel-identity-20261015/owning.log 2>&1 || phase_status=$?
printf '%s\n' "$phase_status" > /mnt/wd4tb/scratch/measured-channel-identity-20261015/owning.exit
if [ "$phase_status" -ne 0 ]; then exit "$phase_status"; fi
phase_status=0
tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/measured-channel-identity-20261015/api-eligibility.py > /mnt/wd4tb/scratch/measured-channel-identity-20261015/api-eligibility.log 2>&1 || phase_status=$?
printf '%s\n' "$phase_status" > /mnt/wd4tb/scratch/measured-channel-identity-20261015/api-eligibility.exit
exit "$phase_status"
