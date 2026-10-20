"""Existing table-selection help has reviewed current runtime translations."""
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('language', ['de', 'es', 'fr', 'hi', 'is', 'ko', 'pt', 'sv', 'zh_CN'])
def test_table_selection_runtime_tooltip(language):
    from spacr.qt.i18n_catalogs import setting_tooltip

    root = Path(__file__).resolve().parents[1]
    review = json.loads((root / 'docs/i18n/reviewed/runtime' / language /
                         '2026-10-15-n594-table-selection.json').read_text(encoding="utf-8"))
    record, = review['records']
    assert setting_tooltip('tables', record['source'], language, 'ml_analyze') == record['translation']
    assert record['translation'] != record['source']
