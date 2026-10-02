"""Production-plan parity, read-only behavior and real threaded preview lifetime."""
import copy
import hashlib
import threading

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import QDialog

from spacr.qt.widgets import calibration_preview as preview

_DIALOG_EXEC = QDialog.exec


def _fixture(root):
    merged = root / 'merged'
    merged.mkdir(parents=True)
    for plate, gain in [('plate1', 1), ('plate2', 2)]:
        for field in (1, 2):
            values = np.full((16, 16), 100 + 500 * gain, dtype=np.uint16)
            labels = np.full_like(values, 42)
            np.save(merged / f'{plate}_A01_{field}.npy', np.stack([values, labels], axis=-1))
    return {'src': str(root), 'intensity_calibration': True,
            'intensity_calibration_wells': ['A01'],
            'intensity_calibration_statistic': 'median',
            'intensity_calibration_offset': 100, 'test_mode': False,
            'timelapse': False, 'cell_mask_dim': 1,
            'nucleus_mask_dim': None, 'pathogen_mask_dim': None,
            'organelle_mask_dim': None}


def _hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def test_preview_uses_production_plan_without_writing(tmp_path):
    from spacr.measure import _prepare_measurement_calibration
    from spacr.settings import get_measure_crop_settings
    settings = _fixture(tmp_path)
    original = copy.deepcopy(settings)
    before = _hashes(tmp_path)
    report = preview._plan_gains(settings)[0]
    production = get_measure_crop_settings({**settings, 'src': str(tmp_path / 'merged')})
    _, expected = _prepare_measurement_calibration(
        production, sorted(p.name for p in (tmp_path / 'merged').glob('*.npy')))
    assert report['calibration'] == expected
    assert expected['plates']['plate1']['gain'] == {'0': 1.0}
    assert expected['plates']['plate2']['gain'] == {'0': .5}
    assert expected['plates']['plate2']['n_reference_fields'] == 2
    assert settings == original and _hashes(tmp_path) == before


def test_multiple_sources_are_independent_and_qc_exclusions_apply(tmp_path):
    import json
    first = _fixture(tmp_path / 'one')
    second = _fixture(tmp_path / 'two')
    qc = tmp_path / 'one' / 'qc'
    qc.mkdir()
    (qc / 'image_quality.json').write_text(json.dumps({
        'version': 1, 'policy': {'image_qc_mode': 'exclude'},
        'excluded_fields': ['plate2_A01_2.npy']}))
    settings = {**first, 'src': [first['src'], second['src']]}
    before = _hashes(tmp_path)
    reports = preview._plan_gains(settings)
    assert len(reports) == 2
    assert reports[0]['calibration']['plates']['plate2']['n_reference_fields'] == 1
    assert reports[1]['calibration']['plates']['plate2']['n_reference_fields'] == 2
    assert _hashes(tmp_path) == before


@pytest.mark.parametrize('change,match', [
    ({'test_mode': True}, 'Test mode'),
    ({'intensity_calibration': False}, 'Enable intensity'),
    ({'src': 's3://bucket/project'}, 'local source'),
    ({'intensity_calibration_wells': ['B02']}, 'reference'),
])
def test_invalid_preview_preserves_source(tmp_path, change, match):
    settings = _fixture(tmp_path)
    before = _hashes(tmp_path)
    with pytest.raises(ValueError, match=match):
        preview._plan_gains({**settings, **change})
    assert _hashes(tmp_path) == before


def test_actual_worker_keeps_event_loop_live_and_displays_exact_gains(qtbot, tmp_path):
    settings = _fixture(tmp_path)
    dialog = preview._CalibrationPreview(lambda: copy.deepcopy(settings))
    qtbot.addWidget(dialog)
    dialog.show()
    beats = []
    timer = QTimer(dialog)
    timer.timeout.connect(lambda: beats.append(1))
    timer.start(1)
    qtbot.mouseClick(dialog.refresh, Qt.LeftButton)
    qtbot.waitUntil(lambda: dialog.table.rowCount() == 2, timeout=30000)
    qtbot.waitUntil(lambda: dialog._runner.active_jobs() == 0)
    assert beats and dialog.table.item(1, 4).text() == '0.5'
    settings['intensity_calibration_offset'] = 101
    qtbot.waitUntil(lambda: dialog.table.rowCount() == 0)
    assert 'changed' in dialog.status.text()
    dialog.close()


@pytest.mark.parametrize('operation', ['edit', 'invalid', 'cancel', 'close'])
def test_blocked_worker_cannot_publish_stale_result(qtbot, monkeypatch, operation):
    entered, release = threading.Event(), threading.Event()
    state = {'offset': 100}
    invalid = [False]
    def getter():
        if invalid[0]:
            raise ValueError('unfinished')
        return dict(state)
    def work(settings):
        entered.set()
        release.wait(10)
        return []
    monkeypatch.setattr(preview, '_plan_gains', work)
    dialog = preview._CalibrationPreview(getter)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog._start()
    qtbot.waitUntil(entered.is_set)
    try:
        if operation == 'edit':
            state['offset'] = 101
            dialog._check_current()
        elif operation == 'invalid':
            invalid[0] = True
            dialog._check_current()
        elif operation == 'cancel':
            dialog._cancel()
        else:
            dialog.close()
        if operation != 'close':
            assert not dialog.refresh.isEnabled()
        assert not dialog._reports
    finally:
        threads = [pair[0] for pair in dialog._runner._jobs.values()]
        from spacr.qt import bridge
        threads.extend(pair[0] for pair in bridge._PARKED_THREADS)
        release.set()
        from shiboken6 import isValid
        qtbot.waitUntil(lambda: all(not isValid(thread) or not thread.isRunning()
                                    for thread in threads), timeout=10000)
        bridge.prune_parked_threads()
        dialog.close()


def test_measure_alpha_row_owns_button_and_preserves_settings(qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QPushButton

    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus
    monkeypatch.setattr(preferences, '_settings', lambda: QSettings(
        str(tmp_path / 'alpha.ini'), QSettings.IniFormat))
    screen = AppScreen('measure')
    try:
        screen._open_the_heading_of('intensity_calibration_wells')
        screen._refresh_alpha_visibility()
        model = screen._settings_model
        control = model._widgets['intensity_calibration_wells']
        button = control.findChild(QPushButton, 'CalibrationPreviewButton')
        assert button is not None, type(control)
        assert not screen.setting_row_is_visible('intensity_calibration_wells')
        preferences._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert screen.setting_row_is_visible('intensity_calibration_wells')
        model.set_value_for_key('intensity_calibration', True)
        model.set_value_for_key('intensity_calibration_wells', ['A01'])
        before = model.collect()
        snapshot = preview._preview_settings(model)
        assert snapshot['intensity_calibration_wells'] == ['A01']
        assert model.collect() == before
        offset = model._widgets['intensity_calibration_offset']
        offset.lineEdit().setText('-')
        with pytest.raises(ValueError, match='Incomplete'):
            preview._preview_settings(model)
        assert offset.lineEdit().text() == '-'
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()


@pytest.mark.parametrize('theme', ['dark', 'light'])
def test_actual_button_opens_read_only_dialog_and_renders(qtbot, tmp_path, monkeypatch, theme):
    import os
    from pathlib import Path

    from PySide6.QtWidgets import QApplication, QPushButton

    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.theme import apply_qpalette, stylesheet
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus
    monkeypatch.setattr(preferences, '_settings', lambda: QSettings(
        str(tmp_path / 'alpha.ini'), QSettings.IniFormat))
    monkeypatch.setattr(preview._CalibrationPreview, 'exec', _DIALOG_EXEC)
    app = QApplication.instance()
    apply_qpalette(app, theme)
    app.setStyleSheet(stylesheet(theme))
    preferences._set_show_alpha_features(True)
    screen = AppScreen('measure')
    settings = _fixture(tmp_path)
    try:
        screen.resize(1100, 850)
        screen.show()
        model = screen._settings_model
        for key, value in settings.items():
            if key != 'organelle_mask_dim':
                assert model.set_value_for_key(key, value)
        screen._open_the_heading_of('intensity_calibration_wells')
        screen._refresh_alpha_visibility()
        button = screen.findChild(QPushButton, 'CalibrationPreviewButton')
        assert button is not None
        parent = button.parentWidget()
        while parent is not None:
            if hasattr(parent, 'set_expanded'):
                parent.set_expanded(True)
            parent = parent.parentWidget()
        qtbot.wait(40)
        screen._settings_scroll.ensureWidgetVisible(button)
        screen._settings_scroll.horizontalScrollBar().setValue(0)
        qtbot.waitUntil(button.isVisible)
        viewport = screen._settings_scroll.viewport()
        from PySide6.QtCore import QPoint, QRect
        position = button.mapTo(viewport, QPoint(0, 0))
        assert viewport.rect().contains(QRect(position, button.size()))
        assert model._widgets['intensity_calibration_wells'].minimumWidth() == 0
        before = model.collect()
        finished = []
        import time
        deadline = time.monotonic() + 20
        def inspect_dialog():
            dialog = screen.findChild(preview._CalibrationPreview)
            if time.monotonic() > deadline:
                if dialog is not None:
                    dialog.reject()
                return
            if dialog is None or dialog.table.rowCount() != 2:
                QTimer.singleShot(20, inspect_dialog)
                return
            assert dialog.table.item(1, 4).text() == '0.5'
            out = os.environ.get('SPACR_F580_CAPTURE_DIR')
            if out:
                folder = Path(out)
                folder.mkdir(parents=True, exist_ok=True)
                dialog.grab().save(str(folder / f'gains-{theme}.png'))
                screen.grab().save(str(folder / f'controls-{theme}.png'))
            finished.append(True)
            dialog.reject()
        QTimer.singleShot(20, inspect_dialog)
        qtbot.mouseClick(button, Qt.LeftButton)
        assert finished and model.collect() == before
        assert not (tmp_path / 'measurements').exists()
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()


def test_close_then_reopen_waits_for_retired_source_scan(qtbot, monkeypatch):
    from PySide6.QtWidgets import QLineEdit, QPushButton, QVBoxLayout, QWidget

    from spacr.qt import bridge
    entered, release = threading.Event(), threading.Event()
    starts = []
    def work(settings):
        starts.append(1)
        entered.set()
        release.wait(10)
        return []
    monkeypatch.setattr(preview, '_plan_gains', work)
    monkeypatch.setattr(preview, '_preview_settings', lambda model: {})
    monkeypatch.setattr(preview._CalibrationPreview, 'exec', _DIALOG_EXEC)
    host = QWidget()
    qtbot.addWidget(host)
    edit = QLineEdit(host)
    QVBoxLayout(host).addWidget(edit)
    preview._attach_preview(edit, object())
    host.show()
    button = edit.findChild(QPushButton, 'CalibrationPreviewButton')
    closed = []
    def close_started_dialog():
        dialog = host.findChild(preview._CalibrationPreview)
        if not entered.is_set() or dialog is None:
            QTimer.singleShot(10, close_started_dialog)
            return
        closed.append(True)
        dialog.reject()
    QTimer.singleShot(10, close_started_dialog)
    try:
        qtbot.mouseClick(button, Qt.LeftButton)
        assert closed and not button.isEnabled()
        button.click()
        qtbot.wait(30)
        assert starts == [1], 'closing must not allow a duplicate source scan'
    finally:
        release.set()
        qtbot.waitUntil(button.isEnabled, timeout=10000)
        bridge.prune_parked_threads()
        host.close()
