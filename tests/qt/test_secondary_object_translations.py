"""Localized secondary controls retain source identities and numerical output."""
import hashlib
import json
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import pytest
from PySide6.QtWidgets import QLabel

ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = ('de', 'es', 'fr', 'sv', 'pt', 'is', 'zh_CN', 'ko', 'hi')


@pytest.mark.parametrize('language', LANGUAGES)
def test_secondary_controls_and_relationships_are_localized(
        qtbot, qt_theme_applied, monkeypatch, tmp_path, language):
    from spacr.qt import cpu_modes, i18n
    from spacr.qt.screens.make_masks import MakeMasksScreen

    payload = json.loads((ROOT / 'docs/i18n/reviewed/runtime' / language /
                          '2026-09-23-secondary-objects.json').read_text())
    targets = {row['source']: row['translation'] for row in payload['records']}
    monkeypatch.setattr(i18n, 'current_language', lambda: language)
    images, primary_folder = tmp_path / 'images', tmp_path / 'primary'
    images.mkdir()
    primary_folder.mkdir()
    field = np.zeros((64, 64), np.uint16)
    field[8:56, 8:56] = 200
    primary = np.zeros_like(field)
    primary[28:36, 28:36] = 900
    imageio.imwrite(images / 'field.tif', field)
    primary_path = primary_folder / 'field.tif'
    imageio.imwrite(primary_path, primary)
    original_hash = hashlib.sha256(primary_path.read_bytes()).hexdigest()
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    try:
        selector = screen._primary_selector
        assert selector.path.placeholderText() == targets['Primary-mask file or folder']
        captions = {label.text() for label in selector.findChildren(QLabel)}
        assert {targets['Primary object class'], targets['Secondary object class'],
                targets['Primary masks']} <= captions
        assert selector.status.text() == targets[
            'Choose a primary mask from a different file than the editable output mask.']
        assert screen._open_folder(str(images))
        selector.path.setText(str(primary_folder))
        selector._source_changed()
        qtbot.waitUntil(lambda: selector.snapshot is not None)
        assert selector.status.text() == targets['{n} primary objects ready.'].format(n=1)
        assert selector.snapshot.primary_class == 'nucleus'
        assert selector.snapshot.secondary_class == 'cell'
        index = screen._mag_mode.findData(cpu_modes.SECONDARY)
        assert screen._mag_mode.itemText(index) == targets['Secondary objects from primary masks']
        screen._mag_mode.setCurrentIndex(index)
        assert screen._btn_otsu.text() == targets['{method} detect'].format(
            method=targets['Secondary objects from primary masks'])
        screen._methods_card.set_expanded(True)
        screen.resize(1707, 900)
        screen.show()
        qtbot.wait(50)
        secondary = screen._method_groups['secondary']
        margins = screen._methods_card.body_layout.contentsMargins()
        assert (secondary.minimumSizeHint().width() + margins.left() + margins.right()
                <= screen._settings_scroll.viewport().width())
        growth = screen._secondary_widgets['secondary_growth']
        assert [(growth.itemText(i), growth.itemData(i)) for i in range(growth.count())] == [
            (targets['Intensity watershed'], 'intensity'),
            (targets['Distance growth within threshold'], 'distance')]
        growth.setCurrentIndex(growth.findData('distance'))
        screen._secondary_widgets['propagate_sigma'].setValue(0)
        stop = screen._secondary_widgets['propagate_stop']
        stop.setCurrentIndex(stop.findData('absolute'))
        screen._secondary_widgets['propagate_stop_value'].setValue(100)
        screen._min_area.setValue(1)
        screen._detect_normalized.setChecked(False)
        screen._combine_mode.setCurrentIndex(screen._combine_mode.findData('replace'))
        screen._on_detect_otsu()
        expected = targets['{name}: {count} ({ids})'].format(
            name=targets['Matched'], count=1, ids='900')
        assert expected in screen._secondary_relations.text()
        assert set(np.unique(screen._canvas.mask)) == {0, 900}
        assert hashlib.sha256(primary_path.read_bytes()).hexdigest() == original_hash
    finally:
        screen.close()
