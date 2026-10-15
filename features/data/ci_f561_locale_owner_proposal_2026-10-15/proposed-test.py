"""Actual installed runtime translations at own-DINO classifier boundaries."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from spacr import embeddings as emb
from tests.test_own_dino_classifier import checkpoint as checkpoint


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is'])
@pytest.mark.parametrize('boundary', ['invalid', 'cancelled'])
def test_installed_own_dino_classifier_refusals(checkpoint, monkeypatch, tmp_path,
                                               language, boundary):
    import numpy as np

    source = ('Own-DINO classifier settings, checkpoint or input are incompatible.'
              if boundary == 'invalid' else 'Classifier training stopped.')
    review_name = ('2026-10-15-f561-classifier-backend-home-b.json'
                   if boundary == 'invalid' else
                   '2026-10-15-f561-classifier-ui-runtime-home-b.json')
    review_path = (Path(__file__).parents[1] / 'docs/i18n/reviewed/runtime'
                   / language / review_name)
    reviews = json.loads(review_path.read_text())['records']
    matched = [row for row in reviews
               if row['table'] == 'ui' and row['key'] == source
               and row['source'] == source]
    assert len(matched) == 1
    record = matched[0]
    expected = record['translation']
    source_hash = hashlib.sha256(source.encode('utf-8')).hexdigest()
    assert record['source_sha256'] == source_hash
    catalogue = importlib.import_module(f'spacr.qt.i18n_catalogs.{language}')
    assert catalogue.SOURCE_HASHES[('UI', source)] == source_hash
    assert catalogue.UI[source] == expected
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
