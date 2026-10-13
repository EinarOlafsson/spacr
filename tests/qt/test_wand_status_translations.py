"""Pending and failed applied-image states retain their translated status text."""
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize('language', ['de', 'es', 'fr', 'sv', 'pt', 'is', 'zh_CN', 'ko', 'hi'])
def test_pending_and_failed_wand_status_is_localized(qtbot, monkeypatch, language):
    from spacr.qt import i18n, detect_chain
    from spacr.qt.screens.make_masks import _MaskCanvas

    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / 'docs/i18n/reviewed/runtime' / language /
                          '2026-09-22-wand-enhancement-status.json').read_text())
    targets = {row['source']: row['translation'] for row in payload['records']}
    monkeypatch.setattr(i18n, 'current_language', lambda: language)
    canvas = _MaskCanvas()
    qtbot.addWidget(canvas)
    canvas.image = np.arange(16, dtype=np.uint16).reshape(4, 4)
    canvas.mask = np.zeros((4, 4), dtype=np.uint16)
    original = canvas.image.copy()
    canvas.enhance_display = True
    canvas.enhance_chain = detect_chain.NO_CHAIN._replace(gamma=0.5)
    monkeypatch.setattr(canvas, '_ask_for_enhanced', lambda *args: None)
    messages = []
    canvas.status.connect(messages.append)
    assert canvas.wand_source() is None
    pending = 'Image enhancement is updating. Try the Wand again when it finishes.'
    assert messages[-1] == targets[pending]
    error = 'sample failure /tmp/test.tif'
    canvas._take_enhanced((canvas.detection_base(), canvas.enhance_chain, ValueError(error)))
    failure = targets['Image enhancement failed: {error}'].format(error=error)
    assert messages[-1] == failure
    assert canvas.wand_source() is None
    assert messages[-1] == failure
    assert not canvas.mask.any()
    np.testing.assert_array_equal(canvas.image, original)
    canvas.close_enhancer()
