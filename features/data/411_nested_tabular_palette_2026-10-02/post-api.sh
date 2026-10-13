#!/usr/bin/env bash
set -euo pipefail
cd /tmp/spacr-implementation-20261001/suggest-capture
stage=/tmp/spacr-implementation-20261001/f411-slice-tabular-palette
py=/home/carruthers/anaconda3/envs/spacr/bin/python
export XDG_CONFIG_HOME="$stage/private-config"
export QT_QPA_PLATFORM=offscreen
"$py" -u "$stage/audit-runtime.py" > "$stage/runtime-audit.log" 2>&1
"$py" tools/build_help_search_index.py > "$stage/help-generator.log" 2>&1
"$py" tools/build_help_search_index.py --check > "$stage/help-check.log" 2>&1
"$py" -m pytest -q --junitxml="$stage/focused-tests.xml" tests/test_nested_helper_docs.py tests/test_built_nested_helper_api.py tests/test_api_i18n_extractor.py::test_public_docstrings_matches_reviewed_visible_coverage tests/test_api_i18n_extractor.py::test_public_docstrings_exclude_the_exact_non_rendered_autoapi_boundary tests/test_docstring_correctness.py::test_callable_boundary_is_cross_checked_with_i18n_extractor tests/test_documentation_i18n.py::test_documentation_api_catalog_inventory_and_hashes_are_current tests/test_the_help_index_is_generated_not_typed.py > "$stage/focused-tests.log" 2>&1
"$py" "$stage/snapshot.py" before > "$stage/snapshot-before.log" 2>&1
"$py" "$stage/run-phase.py" dummy > "$stage/dummy.log" 2>&1
"$py" "$stage/run-phase.py" html > "$stage/html.log" 2>&1
"$py" "$stage/snapshot.py" after > "$stage/snapshot-after.log" 2>&1
"$py" "$stage/verify-rendered.py" > "$stage/rendered.log" 2>&1
"$py" -m pytest -q --junitxml="$stage/browser-tests.xml" tests/test_api_i18n_frontend.py::test_every_complete_real_catalog_renders_through_the_browser_selector > "$stage/browser-tests.log" 2>&1
