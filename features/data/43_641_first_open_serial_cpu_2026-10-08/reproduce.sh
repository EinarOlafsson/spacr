#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' SPACR_DEVICE=cpu QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg
python_bin=${SPACR_PROOF_PYTHON:-python}
tools/run_capped.sh 4G "$python_bin" -m pytest -q -p no:randomly tests/test_ci_suite_classification.py
tools/run_capped.sh 4G "$python_bin" -m pytest -v --tb=short -p no:randomly tests/qt/test_641_module_first_open_timing.py --durations=0
tools/run_capped.sh 4G "$python_bin" -m pytest -v --tb=short -p no:randomly tests/qt/test_641_module_first_open_budgets.py --cov=spacr.qt.app --cov-branch --cov-report=json:first-open-structural-coverage.json
