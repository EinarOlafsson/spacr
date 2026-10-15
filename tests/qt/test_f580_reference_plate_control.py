"""The alpha reference choice reaches production preview and invalidates gains."""
import copy
import hashlib

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QLineEdit

from spacr.qt.widgets import calibration_preview as preview

KEY = 'intensity_calibration_reference_plate'


def fixture(root):
    merged = root / 'merged'
    merged.mkdir(parents=True)
    for plate, value in [('plate1', 600), ('plate2', 1100)]:
        np.save(merged / f'{plate}_A01_1.npy', np.stack([
            np.full((16, 16), value, dtype=np.uint16),
            np.ones((16, 16), dtype=np.uint16)], axis=-1))
    return {'src': str(merged), 'intensity_calibration': True,
            'intensity_calibration_wells': ['A01'],
            'intensity_calibration_statistic': 'median',
            'intensity_calibration_offset': 100, KEY: 'plate2',
            'test_mode': False, 'timelapse': False, 'cell_mask_dim': 1,
            'nucleus_mask_dim': None, 'pathogen_mask_dim': None,
            'organelle_mask_dim': None}


def source_bytes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def test_alpha_control_named_snapshot_preview_and_stale_invalidation(qtbot, tmp_path, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    monkeypatch.setattr(preferences, '_settings', lambda: QSettings(
        str(tmp_path / 'alpha.ini'), QSettings.IniFormat))
    screen = AppScreen('measure')
    qtbot.addWidget(screen)
    dialog = None
    try:
        screen._open_the_heading_of(KEY)
        screen._refresh_alpha_visibility()
        assert not screen.setting_row_is_visible(KEY)
        model = screen._settings_model
        editor = model._widgets[KEY]
        assert isinstance(editor, QLineEdit)
        assert editor.objectName() == 'IntensityCalibrationReferencePlate'
        preferences._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert screen.setting_row_is_visible(KEY)
        settings = fixture(tmp_path / 'data')
        for key, value in settings.items():
            if key != 'organelle_mask_dim':
                assert model.set_value_for_key(key, value)
        original = copy.deepcopy(model.collect())
        before = source_bytes(tmp_path / 'data')
        snapshot = preview._preview_settings(model)
        assert snapshot[KEY] == 'plate2'
        dialog = preview._CalibrationPreview(lambda: preview._preview_settings(model), threaded=False)
        qtbot.addWidget(dialog)
        dialog._start()
        qtbot.waitUntil(lambda: bool(dialog._reports), timeout=10000)
        calibration = dialog._reports[0]['calibration']
        assert calibration['reference_plate'] == 'plate2'
        assert calibration['plates']['plate1']['gain'] == {'0': 2.0}
        assert calibration['plates']['plate2']['gain'] == {'0': 1.0}
        assert dialog.table.rowCount() == 2
        assert model.collect() == original
        assert source_bytes(tmp_path / 'data') == before
        assert model.set_value_for_key(KEY, 'plate1')
        dialog._check_current()
        assert dialog.table.rowCount() == 0 and dialog._snapshot is None
        assert not dialog._reports
        assert source_bytes(tmp_path / 'data') == before
    finally:
        if dialog is not None:
            dialog.close()
        retire_pyqtgraph_menus(screen)
        screen.close()
