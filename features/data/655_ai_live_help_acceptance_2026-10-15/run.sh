#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CONFIG_HOME=/mnt/wd4tb/scratch/n655-ai-toggle-help-20261015/config
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages:.
mkdir -p "$XDG_CONFIG_HOME"
/usr/bin/time -v -o /mnt/wd4tb/scratch/n655-ai-toggle-help-20261015/existing.time /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_the_help_index_is_generated_not_typed.py::test_the_generated_module_says_it_is_generated tests/test_the_help_index_is_generated_not_typed.py::test_every_setting_consumer_is_addressable tests/qt/test_the_help_search_field_lands_where_it_says.py::test_enabled_object_helpers_have_search_rows_local_prose_and_exact_links tests/qt/test_the_help_search_field_lands_where_it_says.py::test_the_page_of_a_symbol_is_addressed_in_the_readers_language tests/qt/test_the_help_search_field_lands_where_it_says.py::test_the_local_docstring_is_read_without_importing_the_module > /mnt/wd4tb/scratch/n655-ai-toggle-help-20261015/existing.log 2>&1
/usr/bin/time -v -o /mnt/wd4tb/scratch/n655-ai-toggle-help-20261015/four-symbols.time /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/n655-ai-toggle-help-20261015/check_four_symbols.py 17e8d8ced5f85d5cc9909fe19bb8a16953101821 fb0ab7eda845610e06591c6fc21cd044ba56288d327d7f1ab1af778440807544 > /mnt/wd4tb/scratch/n655-ai-toggle-help-20261015/four-symbols.json 2> /mnt/wd4tb/scratch/n655-ai-toggle-help-20261015/four-symbols.stderr
