#!/bin/bash
set -uo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export XDG_CONFIG_HOME=/mnt/wd4tb/scratch/n703-help-lexical-scope-20261015/config
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages:.
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 2G /usr/bin/time -v -o /mnt/wd4tb/scratch/n703-help-lexical-scope-20261015/after.time /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/qt/test_help_local_docstring_lexical_scope.py tests/qt/test_the_help_search_field_lands_where_it_says.py::test_enabled_object_helpers_have_search_rows_local_prose_and_exact_links tests/qt/test_the_help_search_field_lands_where_it_says.py::test_the_page_of_a_symbol_is_addressed_in_the_readers_language tests/qt/test_the_help_search_field_lands_where_it_says.py::test_the_local_docstring_is_read_without_importing_the_module tests/qt/test_the_help_search_field_lands_where_it_says.py::test_the_local_docstring_is_read_off_the_disk_or_not_at_all tests/qt/test_the_help_search_field_lands_where_it_says.py::test_a_source_that_cannot_be_parsed_is_no_docstring > /mnt/wd4tb/scratch/n703-help-lexical-scope-20261015/after.log 2>&1
status=$?
echo "$status" > /mnt/wd4tb/scratch/n703-help-lexical-scope-20261015/after.exit
exit "$status"
