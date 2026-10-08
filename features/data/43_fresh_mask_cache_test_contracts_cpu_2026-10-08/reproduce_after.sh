#!/usr/bin/env bash
set -euo pipefail
proof_dir=features/data/43_fresh_mask_cache_test_contracts_cpu_2026-10-08
replay_dir=$(mktemp -d /mnt/wd4tb/scratch/fresh-mask-cache-replay-XXXXXX)
export CUDA_VISIBLE_DEVICES="" QT_QPA_PLATFORM=offscreen SPACR_DEVICE=cpu MPLBACKEND=Agg
export PYTHONPATH=.:/mnt/wd4tb/scratch/ci-55bad-20261008/pytest842-overlay:"$proof_dir"
tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q --tb=short -p no:randomly -p seed_mask_targets --basetemp="$replay_dir/after" \
  tests/qt/test_main_window.py::test_main_window_constructs_and_switches \
  tests/qt/test_cov_wf_qt_app.py::test_a_screen_can_be_rebuilt_before_it_was_ever_built
