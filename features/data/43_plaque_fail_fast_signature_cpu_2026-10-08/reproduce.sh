#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' SPACR_DEVICE=cpu QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg
python_bin=${SPACR_PROOF_PYTHON:-python}
tools/run_capped.sh 4G "$python_bin" -m pytest -q -p no:randomly tests/test_cellpose_api_contract.py::test_every_converted_double_declares_the_installed_signature tests/test_test_suite_hygiene.py::test_the_cellpose_mocks_do_not_swallow_channel_axis tests/test_test_suite_hygiene.py::test_the_cellpose_mock_ratchet_is_empty_and_stays_that_way tests/test_plaque_segmentation_diagnostics.py::test_incompatible_flow_and_metric_returns_refuse_before_model_evaluation
PYTHONPATH=. tools/run_capped.sh 4G "$python_bin" features/data/43_plaque_fail_fast_signature_cpu_2026-10-08/signature-probe.py
