"""Actual installed runtime translations at own-DINO classifier boundaries."""
import json
from pathlib import Path

import pytest

from spacr import embeddings as emb
from test_own_dino_classifier import checkpoint


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is'])
@pytest.mark.parametrize('boundary', ['invalid', 'cancelled'])
def test_installed_own_dino_classifier_refusals(checkpoint, monkeypatch, tmp_path,
                                               language, boundary):
    import numpy as np

    review_path = (Path(__file__).parents[1] / 'docs/i18n/reviewed/runtime'
                   / language / '2026-10-15-own-dino-classifier.json')
    reviews = json.loads(review_path.read_text())['records']
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
