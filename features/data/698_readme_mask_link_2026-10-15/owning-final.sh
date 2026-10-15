#!/usr/bin/env bash
set -uo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SPACR_NO_NETWORK=1
export XDG_CONFIG_HOME=/mnt/wd4tb/scratch/home-b-n698-readme-link-20261015/config
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages:$PWD
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 4G /usr/bin/time -v -o /mnt/wd4tb/scratch/home-b-n698-readme-link-20261015/owning-final.time /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_readme_presentation.py::test_model_zoo_links_to_the_matching_site_mask_generator tests/test_readme_presentation.py::test_the_readme_lists_the_default_model_of_each_family tests/test_readme_presentation.py::test_every_readme_carries_the_same_model_zoo_block tests/test_readme_is_built_from_images.py::test_the_readmes_and_images_are_current tests/test_readme_is_built_from_images.py::test_no_stray_images_are_left_behind tests/test_readme_is_built_from_images.py::test_panels_open_readable_text_and_keep_every_detail_button > /mnt/wd4tb/scratch/home-b-n698-readme-link-20261015/owning-final.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/wd4tb/scratch/home-b-n698-readme-link-20261015/owning-final.exit
exit "$result"
