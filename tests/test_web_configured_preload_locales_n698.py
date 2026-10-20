"""Actual startup refusal uses the reviewed installed locale without loading models."""
import hashlib
import json
from pathlib import Path
import socket

import pytest

from spacr.qt import i18n
from tools.web_mask_generator import masks, service


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is'])
def test_actual_configured_startup_key_refusal_is_localized(language, monkeypatch, tmp_path):
    source = 'Unknown provisioned model key: {key}'
    review_path = (Path(__file__).parents[1] / 'docs/i18n/reviewed/runtime' / language /
                   '2026-10-15-configured-startup-preload.json')
    review = json.loads(review_path.read_text(encoding="utf-8"))
    record, = [row for row in review['records'] if row['source'] == source]
    assert review['language'] == language
    assert record['table'] == 'ui'
    assert record['source_sha256'] == hashlib.sha256(source.encode()).hexdigest()
    expected = record['translation'].format(key='unlisted_model')
    assert record['translation'] != source
    monkeypatch.setenv(i18n.ENV_LANGUAGE, language)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail('Unknown startup key must not construct or fetch a model')

    monkeypatch.setattr(service, '_load_model', forbidden)
    monkeypatch.setattr(masks, '_load_mask_model', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    path = tmp_path / 'operator-fixture'
    original = b'private fixture; not decoded or changed'
    path.write_bytes(original)
    instance = service.ModelService({'unlisted_model': path})

    with pytest.raises(ValueError) as error:
        instance.preload_default()

    assert str(error.value) == expected
    assert calls == []
    assert path.read_bytes() == original
