"""Run only after ordinary reviewed catalogs install the frozen nine targets."""
import hashlib
import json
from pathlib import Path
import socket

import pytest

from spacr import attribution as cf


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'fr', 'pt', 'is', 'hi', 'ko', 'zh_CN'])
def test_actual_pinned_checkpoint_refusal_is_localized(tmp_path, monkeypatch, language):
    reviews = (Path(__file__).parents[1] / 'docs/i18n/reviewed/runtime' / language
               / '2026-10-15-counterfactual-pinned-reuse.json')
    record = next(row for row in json.loads(reviews.read_text(encoding="utf-8"))['records']
                  if row['table'] == 'ui' and row['key'] ==
                  'The counterfactual checkpoint is missing, unreadable, unsafe, or incompatible with the current inputs.')
    assert hashlib.sha256(record['source'].encode()).hexdigest() == record['source_sha256']
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    monkeypatch.setattr(socket.socket, 'connect', lambda *a: pytest.fail('network requested'))
    path = tmp_path / 'selected-local-checkpoint.pt'
    original = b'private test fixture; invalid expected hash refuses before decoding'
    path.write_bytes(original)
    with pytest.raises(ValueError) as error:
        cf._cf_load_checkpoint(path, '0' * 64, {})
    assert str(error.value) == record['translation']
    assert str(error.value) != record['source']
    assert path.read_bytes() == original
