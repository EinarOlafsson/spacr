"""All complete curation sentences reach the actual status label in each locale."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spacr.qt import i18n
from spacr.qt.screens.make_masks import MakeMasksScreen, _StatusLabel
from tests.qt.test_n448_complete_curation_status import TEMPLATES


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is'])
def test_actual_curation_status_callbacks_use_all_reviewed_sentences(
        qapp, monkeypatch, tmp_path, language):
    review_path = (Path(__file__).parents[2] / 'docs/i18n/reviewed/runtime' / language /
                   '2026-10-15-complete-curation-status.json')
    review = json.loads(review_path.read_text(encoding="utf-8"))
    records = {row['source']: row for row in review['records']}
    assert set(records) == set(TEMPLATES.values())
    monkeypatch.setenv(i18n.ENV_LANGUAGE, language)
    for (state, keep), source in TEMPLATES.items():
        record = records[source]
        assert record['source_sha256'] == hashlib.sha256(source.encode()).hexdigest()
        assert record['translation'] != source
        files = ['judged.tif'] if state == 'last' else ['judged.tif', 'following.tif']
        status = _StatusLabel()
        calls = []
        subject = SimpleNamespace(
            _current_index=0, _folder=str(tmp_path), _image_files=files,
            _loading=False,
            _hide_judged=state == 'empty', _status_label=status,
            _blind_label=lambda path: Path(path).name, _visible_indices=lambda: [],
            _sync_queue_tally=lambda: calls.append('tally'))

        def next_field():
            calls.append('next')
            if state == 'moved':
                subject._current_index = 1

        subject._on_next = next_field
        moved = MakeMasksScreen._advance_after_verdict(subject, keep, 'judged.tif')
        assert status.text() == record['translation'].format(
            judged='judged.tif', now='following.tif')
        assert moved is (state == 'moved')
        assert calls == ['next', 'tally']
        status.close()
        status.deleteLater()
