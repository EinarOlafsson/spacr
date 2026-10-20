#!/bin/bash
cd /mnt/wd4tb/spacr-worktrees/codex-classify-public-stage-20261015 || exit 90
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen SPACR_LANGUAGE=en OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -n 0 tests/qt/test_classify_public_maturity_route.py tests/qt/test_maturity_visibility_preferences.py::test_visible_app_registry_obeys_each_switch_independently tests/qt/test_maturity_visibility_preferences.py::test_home_contains_only_enabled_maturity_stages tests/qt/test_maturity_visibility_preferences.py::test_sidebar_refreshes_without_rebuilding_or_leaving_empty_headers tests/qt/test_app_maturity.py::test_the_default_never_overrules_a_module_that_did_declare_one tests/qt/test_app_maturity.py::test_an_assessed_module_is_left_alone_by_the_default --tb=short --junitxml=/mnt/wd4tb/scratch/classify-public-stage-20261015/owning.xml > /mnt/wd4tb/scratch/classify-public-stage-20261015/owning.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/wd4tb/scratch/classify-public-stage-20261015/owning.exit
exit "$result"
