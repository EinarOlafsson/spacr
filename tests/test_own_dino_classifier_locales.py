"""Actual installed runtime translations at own-DINO classifier boundaries."""
import json
from pathlib import Path

import pytest

from spacr import embeddings as emb
from tests.test_own_dino_classifier import checkpoint as checkpoint

from tests import locale_cases


def _installed_own_dino_classifier_refusals_case(checkpoint, monkeypatch, tmp_path, language, boundary):
    """One case of :func:`test_installed_own_dino_classifier_refusals`."""
    import numpy as np

    review_file = ('2026-10-15-f561-classifier-backend-home-b.json'
                   if boundary == 'invalid'
                   else '2026-10-15-f561-classifier-ui-runtime-home-b.json')
    review_path = (Path(__file__).parents[1] / 'docs/i18n/reviewed/runtime'
                   / language / review_file)
    reviews = json.loads(review_path.read_text(encoding='utf-8'))['records']
    source = ('Own-DINO classifier settings, checkpoint or input are incompatible.'
              if boundary == 'invalid' else 'Classifier training stopped.')
    expected = next(r['translation'] for r in reviews if r['source'] == source)
    assert expected != source
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    path, _, _ = checkpoint()
    before = path.read_bytes()
    output = tmp_path / 'not-published.pt'
    if boundary == 'invalid':
        with pytest.raises(emb.EmbeddingError) as error:
            emb._dino_classifier_from_checkpoint(
                path, num_classes=True, source_channels=[0, 1])
    else:
        with pytest.raises(InterruptedError) as error:
            emb._dino_finetune_classifier(
                np.ones((8, 4, 4, 2)), path, output, labels=['a', 'b']*4,
                groups=[f'w{i//2}' for i in range(8)], source_channels=[0, 1],
                cancelled=lambda: True, epochs=1)
    assert str(error.value) == expected
    assert path.read_bytes() == before
    assert not output.exists()


def test_installed_own_dino_classifier_refusals(subtests, checkpoint, monkeypatch, tmp_path):
    for label, case in locale_cases.cases(('language', ['sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is']), ('boundary', ['invalid', 'cancelled'])):
        with subtests.test(**label), locale_cases.monkeypatch_scope() as case_monkeypatch:
            _installed_own_dino_classifier_refusals_case(checkpoint=checkpoint, monkeypatch=case_monkeypatch, tmp_path=locale_cases.case_dir(tmp_path, label), **case)
