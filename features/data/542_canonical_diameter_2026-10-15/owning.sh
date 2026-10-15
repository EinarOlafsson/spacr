#!/bin/bash
cd /mnt/wd4tb/spacr-worktrees/codex-f542-canonical-diameter-20261015 || exit 90
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen SPACR_LANGUAGE=en OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -n 0 tests/test_colony_cfu_counting.py::test_a_large_photo_is_worked_on_small_and_measured_in_its_own_pixels tests/test_colony_cfu_counting.py::test_physical_units_come_from_the_dish_diameter tests/test_colony_cfu_counting.py::test_without_a_known_size_the_areas_stay_in_pixels tests/test_colony_cfu_counting.py::test_a_manual_pixel_size_overrides_the_dish tests/test_colony_cfu_counting.py::test_of_two_round_regions_the_dish_is_the_larger --tb=short --junitxml=/mnt/wd4tb/scratch/f542-canonical-diameter-20261015/owning.xml > /mnt/wd4tb/scratch/f542-canonical-diameter-20261015/owning.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/wd4tb/scratch/f542-canonical-diameter-20261015/owning.exit
exit "$result"
