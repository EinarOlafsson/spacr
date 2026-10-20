#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-f538-channel-boundary-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 SPACR_LANGUAGE=en
phase_status=0
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_unmix_channel_boundary_f538.py::test_zero_dimensional_boolean_arrays_refuse_before_loading tests/test_unmix_channel_boundary_f538.py::test_zero_dimensional_integral_float_and_string_arrays_preserve_order tests/test_unmix_channel_boundary_f538.py::test_valid_physical_order_and_saved_integral_selectors_are_preserved 'tests/test_unmix_channel_boundary_f538.py::test_invalid_physical_selection_refuses_before_control_load_or_output[measure-selection0]' 'tests/test_unmix_channel_boundary_f538.py::test_invalid_physical_selection_refuses_before_control_load_or_output[measure-selection1]' 'tests/test_unmix_channel_boundary_f538.py::test_invalid_physical_selection_refuses_before_control_load_or_output[measure-selection4]' --basetemp=/mnt/wd4tb/scratch/f538-channel-boundary-20261015/bool-array-owning-pytest --junitxml=/mnt/wd4tb/scratch/f538-channel-boundary-20261015/bool-array-owning.xml > /mnt/wd4tb/scratch/f538-channel-boundary-20261015/bool-array-owning.log 2>&1 || phase_status=$?
printf '%s\n' "$phase_status" > /mnt/wd4tb/scratch/f538-channel-boundary-20261015/bool-array-owning.exit
exit "$phase_status"
