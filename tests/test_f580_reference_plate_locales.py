"""Runtime lookup and invalid-selection errors use reviewed current captions."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize('language', ['de', 'es', 'fr', 'hi', 'is', 'ko', 'pt', 'sv', 'zh_CN'])
def test_reference_plate_runtime_lookup_and_refusal(language, tmp_path, monkeypatch):
    from spacr import intensity_rescale
    from spacr.qt import i18n
    from spacr.qt.i18n_catalogs import setting_label, setting_tooltip

    root = Path(__file__).resolve().parents[1]
    review_root = root / 'docs/i18n/reviewed/runtime' / language
    reviews = [json.loads((review_root / filename).read_text()) for filename in (
        '2026-10-15-f580-reference-plate.json',
        '2026-10-15-f580-session-controls-home-b.json',
    )]
    records = {(r['table'], r['key']): r
               for review in reviews for r in review['records']}
    key = 'intensity_calibration_reference_plate'
    label, tip = records[('setting_labels', key)], records[('setting_tooltips', key)]
    caption = records[('ui', 'Without session inputs, the selected reference plate is used; leave it blank to use the first plate by name.')]
    toggle = records[('setting_tooltips', 'intensity_calibration')]
    assert setting_tooltip('intensity_calibration', toggle['source'], language, 'measure') == toggle['translation']
    assert setting_label(key, label['source'], language, 'measure') == label['translation']
    assert setting_tooltip(key, tip['source'], language, 'measure') == tip['translation']
    assert i18n.tr(caption['source'], language=language) == caption['translation']
    module = importlib.import_module('spacr.qt.i18n_catalogs.' + language)
    from hashlib import sha256

    for record in (label, tip, toggle, caption):
        assert record['source_sha256'] == sha256(record['source'].encode()).hexdigest()
        assert module.SOURCE_HASHES[(record['table'].upper(), record['key'])] == record['source_sha256']
        assert record['translation'] != record['source']
    assert module.UI[caption['source']] != caption['source']
    monkeypatch.setattr(i18n, 'current_language', lambda: language)
    field = tmp_path / 'plate1_A01_1.npy'
    np.save(field, np.stack([np.full((8, 8), 600, dtype=np.uint16),
                             np.ones((8, 8), dtype=np.uint16)], axis=-1))
    before = field.read_bytes()
    settings = {'intensity_calibration_wells': ['A01'],
                'intensity_calibration_statistic': 'median',
                'intensity_calibration_offset': 100, 'channels': [0], key: 'missing'}
    with pytest.raises(ValueError) as caught:
        intensity_rescale._build_calibration_plan(tmp_path, [field.name], settings)
    assert str(caught.value).startswith(i18n.tr('Bad settings', language=language) + ': ')
    assert key in str(caught.value) and 'missing' in str(caught.value)
    assert list(tmp_path.iterdir()) == [field] and field.read_bytes() == before
