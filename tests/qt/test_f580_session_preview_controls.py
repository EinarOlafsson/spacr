"""Session calibration controls hand exact inputs to an inert preview planner."""
import copy
import threading

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog, QFormLayout, QLabel, QLineEdit, QPushButton, QWidget

from spacr.qt.widgets import calibration_preview as preview

KEYS = ('intensity_calibration_sessions', 'intensity_calibration_session_column',
        'intensity_calibration_beads', 'intensity_calibration_reference_session')
NAMES = ('IntensityCalibrationSessions', 'IntensityCalibrationSessionColumn',
         'IntensityCalibrationBeads', 'IntensityCalibrationReferenceSession')
VALUES = ('/unread/acquisitions.csv', 'acquisition_day', '/unread/beads', 'day2')


def _report(version=1):
    rows = {
        'day1' if version == 2 else 'plate1': {
            'gain': {'0': .5}, 'reference_statistic': {'0': 250},
            'n_reference_fields': 3},
        'day2' if version == 2 else 'plate2': {
            'gain': {'0': 1}, 'reference_statistic': {'0': 500},
            'n_reference_fields': 2},
    }
    plan = {'version': version, 'reference_plate': None if version == 2 else 'plate2'}
    plan['sessions' if version == 2 else 'plates'] = rows
    if version == 2:
        plan['reference_session'] = 'day2'
    return [{'source': '/unread/merged', 'calibration': plan}]


@pytest.fixture
def model(qtbot):
    from spacr.qt.screens.settings_model import SettingsWidgets

    owner = QWidget()
    qtbot.addWidget(owner)
    controls = SettingsWidgets('measure', parent=owner)
    controls.build_sections()
    return controls


def test_actual_measure_form_names_and_gates_all_four_controls(qtbot, tmp_path, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    monkeypatch.setattr(preferences, '_settings', lambda: QSettings(
        str(tmp_path / 'alpha.ini'), QSettings.IniFormat))
    screen = AppScreen('measure')
    qtbot.addWidget(screen)
    try:
        for key in KEYS:
            screen._open_the_heading_of(key)
        screen._refresh_alpha_visibility()
        controls = screen._settings_model
        for key, name, default in zip(KEYS, NAMES, ('', 'session', '', '')):
            editor = controls._widgets[key]
            assert isinstance(editor, QLineEdit)
            assert editor.objectName() == name
            assert editor.text() == default
            assert not screen.setting_row_is_visible(key)
        preferences._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert all(screen.setting_row_is_visible(key) for key in KEYS)
        for key, value in zip(KEYS, VALUES):
            controls._widgets[key].setText(value)
        preferences._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(key) for key in KEYS)
        snapshot = preview._preview_settings(controls)
        assert [snapshot[key] for key in KEYS] == list(VALUES)
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()


@pytest.mark.parametrize('key,kind', [(KEYS[0], 'csv'), (KEYS[2], 'folder')])
def test_path_controls_choose_the_right_path_kind_and_preserve_cancel(model, monkeypatch, key, kind):
    editor = model._widgets[key]
    editor.setText('/unread/original')
    actions = editor.actions()
    assert len(actions) == 1, 'path editor needs its actual file/folder chooser'
    calls = []
    answer = ['']

    def file_dialog(parent, title, directory, file_filter):
        calls.append(('csv', parent, directory, file_filter))
        return answer[0], file_filter

    def folder_dialog(parent, title, directory):
        calls.append(('folder', parent, directory))
        return answer[0]

    monkeypatch.setattr(QFileDialog, 'getOpenFileName', file_dialog)
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', folder_dialog)
    changed = []
    editor.editingFinished.connect(lambda: changed.append(editor.text()))
    actions[0].trigger()
    assert editor.text() == '/unread/original'
    assert not changed
    answer[0] = '/unread/chosen.csv' if kind == 'csv' else '/unread/chosen_folder'
    actions[0].trigger()
    assert editor.text() == answer[0]
    assert model._valid_committed_value(key) == answer[0]
    assert changed == [answer[0]]
    assert all(call[0] == kind and call[1] is editor for call in calls)
    if kind == 'csv':
        assert all('*.csv' in call[3] for call in calls)


def test_preview_handoff_and_each_new_input_invalidates_cached_gains(qtbot, model, monkeypatch):
    for key, value in zip(KEYS, VALUES):
        model._widgets[key].setText(value)
    snapshots = []

    def planner(snapshot):
        snapshots.append(copy.deepcopy(snapshot))
        return _report()

    monkeypatch.setattr(preview, '_plan_gains', planner)
    dialog = preview._CalibrationPreview(lambda: preview._preview_settings(model), threaded=False)
    qtbot.addWidget(dialog)
    try:
        for index, key in enumerate(KEYS):
            dialog._start()
            assert dialog.table.rowCount() == 2
            assert snapshots[-1][key] == model._widgets[key].text()
            assert all(name in snapshots[-1] for name in KEYS)
            before = copy.deepcopy(snapshots[-1])
            model._widgets[key].setText(f'edited-{index}')
            dialog._check_current()
            assert dialog.table.rowCount() == 0
            assert not dialog._reports and dialog._snapshot is None
            assert 'Settings changed' in dialog.status.text()
            assert snapshots[-1] == before
    finally:
        dialog.close()


@pytest.mark.parametrize('version', [1, 2], ids=['legacy-plates', 'explicit-sessions'])
def test_preview_renders_versioned_identity_and_exact_gains(qtbot, monkeypatch, version):
    monkeypatch.setattr(preview, '_plan_gains', lambda snapshot: _report(version))
    dialog = preview._CalibrationPreview(lambda: {}, threaded=False)
    qtbot.addWidget(dialog)
    try:
        dialog._start()
        assert dialog.table.rowCount() == 2
        expected = ('Session', 'Reference session') if version == 2 else ('Plate', 'Reference plate')
        assert tuple(dialog.table.horizontalHeaderItem(index).text() for index in (1, 2)) == expected
        assert [dialog.table.item(row, 1).text() for row in range(2)] == (
            ['day1', 'day2'] if version == 2 else ['plate1', 'plate2'])
        assert [dialog.table.item(row, 2).text() for row in range(2)] == [
            'day2' if version == 2 else 'plate2'] * 2
        assert [[dialog.table.item(row, col).text() for col in (3, 4, 5, 6)] for row in range(2)] == [
            ['0', '0.5', '250', '3'], ['0', '1', '500', '2']]
    finally:
        dialog.close()


def test_changed_session_input_discards_late_worker_result(qtbot, model, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    model._widgets[KEYS[0]].setText(VALUES[0])

    def planner(snapshot):
        entered.set()
        assert release.wait(5), 'test must release its own inert planner'
        return _report(2)

    monkeypatch.setattr(preview, '_plan_gains', planner)
    dialog = preview._CalibrationPreview(lambda: preview._preview_settings(model))
    qtbot.addWidget(dialog)
    dialog._start()
    qtbot.waitUntil(entered.is_set, timeout=5000)
    try:
        assert dialog._snapshot[KEYS[0]] == VALUES[0]
        model._widgets[KEYS[0]].setText('/unread/new.csv')
        dialog._check_current()
        assert dialog._snapshot is None
    finally:
        release.set()
        qtbot.waitUntil(lambda: dialog._runner.active_jobs() == 0, timeout=5000)
        dialog.close()
    assert dialog.table.rowCount() == 0
    assert not dialog._reports


def test_legacy_snapshot_still_keeps_optional_inputs_blank(model):
    values = preview._preview_settings(model)
    assert values[KEYS[1]] == 'session'
    assert all(values[key] in ('', None) for key in (KEYS[0], KEYS[2], KEYS[3]))


def test_new_preview_relabels_sessions_back_to_plates_without_stale_headers(qtbot, monkeypatch):
    reports = _report(2)
    monkeypatch.setattr(preview, '_plan_gains', lambda snapshot: copy.deepcopy(reports))
    dialog = preview._CalibrationPreview(lambda: {}, threaded=False)
    qtbot.addWidget(dialog)
    try:
        dialog._start()
        assert dialog.table.horizontalHeaderItem(1).text() == 'Session'
        reports[:] = _report(1)
        dialog._start()
        assert dialog.table.horizontalHeaderItem(1).text() == 'Plate'
        assert dialog.table.horizontalHeaderItem(2).text() == 'Reference plate'
        reports.extend(_report(2))
        dialog._start()
        assert dialog.table.rowCount() == 4
        assert dialog.table.horizontalHeaderItem(1).text() == 'Plate / Session'
        assert dialog.table.horizontalHeaderItem(2).text() == 'Reference plate / Reference session'
    finally:
        dialog.close()


def test_preview_editor_wraps_its_own_form_when_mounted_after_construction(qtbot):
    owner = QWidget()
    qtbot.addWidget(owner)
    form = QFormLayout(owner)
    owner.resize(260, 180)
    edit = QLineEdit()
    preview._attach_preview(edit, None)
    label = QLabel('Calibration reference wells')
    label.setWordWrap(True)
    form.addRow(label, edit)
    owner.show()
    button = edit.findChild(QPushButton, 'CalibrationPreviewButton')
    assert form.rowWrapPolicy() == QFormLayout.WrapLongRows
    assert edit.minimumWidth() == 0
    qtbot.waitUntil(lambda: owner.rect().contains(edit.geometry()))
    assert edit.rect().contains(button.geometry())
