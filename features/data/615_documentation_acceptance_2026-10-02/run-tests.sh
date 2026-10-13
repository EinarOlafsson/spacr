#!/usr/bin/env bash
set -euo pipefail
cd /tmp/spacr-implementation-20261001/suggest-capture
export SPACR_DOCS_BUILT=1
export SPACR_DOCS_API_LANGUAGE=all
export SPACR_DOCS_BUILD_DIR=/tmp/spacr-implementation-20261001/documentation-acceptance-615/full/html
export PYTHONPATH=/tmp/spacr-implementation-20261001/review
/home/carruthers/anaconda3/envs/spacr/bin/python tools/build_help_search_index.py --check
exec /home/carruthers/anaconda3/envs/spacr/bin/python -m pytest -q --junitxml=/tmp/spacr-implementation-20261001/documentation-acceptance-615/final-tests.xml -p docs_output_plugin tests/test_the_help_index_is_generated_not_typed.py tests/test_guide_i18n.py::test_catalog_languages_are_known_and_labelled tests/test_guide_i18n.py::test_published_translations_keep_markup_and_app_ui_names tests/test_guide_i18n.py::test_glossary_matches_the_runtime_catalogs tests/test_api_i18n_frontend.py::test_furo_placement_catalog_version_safe_rst_and_persistence tests/test_rendered_docs_show_no_markup_leak.py::test_no_built_page_shows_rest_markup_as_text
