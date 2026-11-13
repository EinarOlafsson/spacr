"""Installed translation of the actual foundation comparison input refusal."""
import json
from pathlib import Path

import numpy as np
import pytest

from spacr import embeddings as emb

from tests import locale_cases


def _installed_foundation_comparison_refusal_case(monkeypatch, language):
    """One case of :func:`test_installed_foundation_comparison_refusal`."""
    path = (Path(__file__).parents[1] / 'docs/i18n/reviewed/runtime' / language
            / '2026-10-15-foundation-comparison-input.json')
    expected = json.loads(path.read_text(encoding="utf-8"))['records'][0]['translation']
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    crops = np.ones((4, 2, 2, 1), dtype=np.uint16)
    before = crops.copy()
    calls = []
    monkeypatch.setattr(emb, 'embed_array', lambda *args: calls.append(args))
    with pytest.raises(emb.EmbeddingError) as error:
        emb._backbone_scorecards(crops, ['a', 'b', 'a'], {
            'first': emb.EmbeddingSpec(normalize=False)})
    assert str(error.value) == expected
    assert expected != 'Provide a four-dimensional crop stack and one label per crop.'
    assert calls == []
    np.testing.assert_array_equal(crops, before)


def test_installed_foundation_comparison_refusal(subtests, monkeypatch):
    for label, case in locale_cases.cases(('language', ['sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is'])):
        with subtests.test(**label), locale_cases.monkeypatch_scope() as case_monkeypatch:
            _installed_foundation_comparison_refusal_case(monkeypatch=case_monkeypatch, **case)
