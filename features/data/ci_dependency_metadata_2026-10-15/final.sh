#!/usr/bin/env bash
set -uo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
 tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest --noconftest -q tests/test_dependency_profiles.py::test_minimum_profile_covers_every_lower_bounded_direct_requirement tests/test_dependency_profiles.py::test_minimum_profile_versions_obey_declared_ranges tests/test_optimal_dependency_versions.py::test_every_constraint_pin_is_the_floor_or_explains_itself tests/test_optimal_dependency_versions.py::test_no_pin_sits_below_the_floor_it_is_meant_to_verify > /mnt/wd4tb/scratch/ci-dependency-metadata-20261015/final.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/wd4tb/scratch/ci-dependency-metadata-20261015/final.exit
exit "$result"
