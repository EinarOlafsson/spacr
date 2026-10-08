#!/usr/bin/env bash
set -euo pipefail
proof_dir=features/data/43_fresh_home_test_contracts_cpu_2026-10-08
replay_dir=$(mktemp -d /mnt/wd4tb/scratch/fresh-home-replay-XXXXXX)
export CUDA_VISIBLE_DEVICES="" QT_QPA_PLATFORM=offscreen SPACR_DEVICE=cpu MPLBACKEND=Agg
export PYTHONPATH=.:/mnt/wd4tb/scratch/ci-55bad-20261008/pytest842-overlay:"$proof_dir"
tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q --tb=short -p no:randomly -p seed_saved_targets --basetemp="$replay_dir/after" \
  'tests/qt/test_641_screen_prewarm.py::test_a_half_built_screen_finishes_inside_its_open' \
  'tests/qt/test_opening_things_does_not_hang.py::test_a_navigation_that_arrives_mid_open_waits_for_it' \
  'tests/qt/test_a_module_screen_is_sheeted_once_before_it_is_seen.py::test_opening_a_module_repolishes_its_screen_once[measure]' \
  'tests/qt/test_a_module_screen_is_sheeted_once_before_it_is_seen.py::test_classify_controls_exist_before_the_first_screen_sheet' \
  'tests/qt/test_a_tile_that_cannot_open_says_so.py::test_a_missing_package_is_named_rather_than_silent' \
  'tests/qt/test_home_navigation_acceptance.py::test_visible_home_tile_opens_real_screen[measure]' \
  'tests/qt/test_a_closed_category_builds_nothing_until_opened.py::test_the_run_is_given_exactly_what_a_window_that_built_everything_gives'
