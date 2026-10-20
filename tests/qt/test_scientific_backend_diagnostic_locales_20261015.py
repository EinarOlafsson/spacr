"""Actual locale lookup of the three source-bound scientific backend diagnostics."""
import json
from pathlib import Path

import pytest

from spacr.qt.i18n import tr


ROOT = Path(__file__).resolve().parents[2]
NAMES = (
    '2026-10-14-counterfactual-reference-channels.json',
    '2026-10-14-mil-well-label-identity.json',
    '2026-10-14-cell-cycle-grouped-evaluation.json',
)


def _cases():
    for language in ('de', 'es', 'fr', 'hi', 'is', 'ko', 'pt', 'sv', 'zh_CN'):
        for filename in NAMES:
            payload = json.loads((ROOT / 'docs/i18n/reviewed/runtime' / language / filename).read_text(encoding="utf-8"))
            for record in payload['records']:
                yield language, record['source'], record['translation']


@pytest.mark.parametrize('language,source,target', list(_cases()))
def test_exact_scientific_diagnostic_uses_current_runtime_locale(language, source, target):
    values = {'well': 'plate_A_1', 'key': 'plate_A_1_f1_o1'}
    assert tr(source, language, **values) == target.format(**values)
