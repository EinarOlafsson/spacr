#!/bin/bash
set -eu
export CUDA_VISIBLE_DEVICES=''
export QT_QPA_PLATFORM=offscreen
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
s=/mnt/wd4tb/scratch/home-b-bridge-spinner-help-20261015
p=/home/olafsson/anaconda3/envs/spacr/bin/python
/usr/bin/time -v -o "$s/four-symbols.time" "$p" "$s/check_four_symbols.py" > "$s/four-symbols.json" 2> "$s/four-symbols.stderr"
/usr/bin/time -v -o "$s/existing.time" "$p" -m pytest -q tests/test_the_help_index_is_generated_not_typed.py::test_the_generated_module_says_it_is_generated tests/test_the_help_index_is_generated_not_typed.py::test_every_setting_consumer_is_addressable tests/qt/test_the_help_search_field_lands_where_it_says.py::test_enabled_object_helpers_have_search_rows_local_prose_and_exact_links tests/qt/test_the_help_search_field_lands_where_it_says.py::test_the_page_of_a_symbol_is_addressed_in_the_readers_language tests/qt/test_the_help_search_field_lands_where_it_says.py::test_the_local_docstring_is_read_without_importing_the_module > "$s/existing.txt" 2>&1
