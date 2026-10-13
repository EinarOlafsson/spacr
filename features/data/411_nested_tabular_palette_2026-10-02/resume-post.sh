#!/usr/bin/env bash
set -euo pipefail
cd /tmp/spacr-implementation-20261001/suggest-capture
stage=/tmp/spacr-implementation-20261001/f411-slice-tabular-palette
py=/home/carruthers/anaconda3/envs/spacr/bin/python
export XDG_CONFIG_HOME="$stage/private-config"
export QT_QPA_PLATFORM=offscreen
"$py" -m pytest -q --junitxml="$stage/fixture-recheck.xml" tests/test_built_nested_helper_api.py::test_tabular_and_palette_helpers_render_without_exposing_private_parents > "$stage/fixture-recheck.log" 2>&1
"$py" "$stage/snapshot.py" before > "$stage/snapshot-before.log" 2>&1
"$py" "$stage/run-phase.py" dummy > "$stage/dummy.log" 2>&1
"$py" "$stage/run-phase.py" html > "$stage/html.log" 2>&1
"$py" "$stage/snapshot.py" after > "$stage/snapshot-after.log" 2>&1
"$py" "$stage/verify-rendered.py" > "$stage/rendered.log" 2>&1
"$py" -m pytest -q --junitxml="$stage/browser-tests.xml" tests/test_api_i18n_frontend.py::test_every_complete_real_catalog_renders_through_the_browser_selector > "$stage/browser-tests.log" 2>&1
export SPACR_DOCS_BUILT=1
export SPACR_DOCS_BUILD_DIR="$stage/full/html"
"$py" -m pytest -q --junitxml="$stage/real-helper-browser-tests.xml" tests/test_built_nested_helper_api.py::test_enabled_helpers_exist_in_the_full_built_site tests/test_built_nested_helper_api.py::test_enabled_helpers_use_their_own_real_catalog_entries_in_the_browser -k 'exist_in or tabular or command_palette' > "$stage/real-helper-browser-tests.log" 2>&1
