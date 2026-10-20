"""Reference selection reaches preview identity without changing unrelated controls."""
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QLabel, QLineEdit

from spacr.qt.screens.settings_model import SettingsWidgets
from spacr.qt.widgets import calibration_preview as preview


@pytest.mark.parametrize('app,key,expected', [
    ('measure', 'intensity_calibration_reference_plate', 'IntensityCalibrationReferencePlate'),
    ('mask', 'intensity_calibration_reference_plate', ''),
    ('measure', 'intensity_calibration_wells', ''),
    ('measure', 'src', ''),
])
def test_reference_editor_identity_is_scoped(qtbot, app, key, expected):
    widget = QLineEdit()
    qtbot.addWidget(widget)
    observed = []
    model = SimpleNamespace(
        app_key=app, _parent=None,
        _route_control=lambda *args: ('plain', {}),
        _build_plain=lambda plan: widget,
        _add_source_actions=lambda *args: None,
        _add_calibration_preview_action=lambda key, control: observed.append(control.objectName()))
    assert SettingsWidgets._widget_for(model, 'str', None, '', key) is widget
    assert widget.objectName() == expected
    assert observed == [expected]


def _model():
    values = {'intensity_calibration_reference_plate': 'plate2',
              'intensity_calibration_wells': ['A01']}
    return SimpleNamespace(
        _defaults=values, _valid_committed_value=lambda key: values.get(key)), values


def test_preview_copies_reference_selection_and_other_mutable_values():
    model, values = _model()
    snapshot = preview._preview_settings(model)
    assert snapshot['intensity_calibration_reference_plate'] == 'plate2'
    values['intensity_calibration_wells'].append('B02')
    assert snapshot['intensity_calibration_wells'] == ['A01']


def test_reference_change_invalidates_displayed_preview(qtbot):
    model, values = _model()
    dialog = preview._CalibrationPreview(lambda: preview._preview_settings(model))
    qtbot.addWidget(dialog)
    dialog._snapshot = preview._preview_settings(model)
    dialog._reports = [{'previous': 'gains'}]
    dialog.table.setRowCount(1)
    generation = dialog._generation
    values['intensity_calibration_reference_plate'] = 'plate1'
    dialog._check_current()
    assert dialog.table.rowCount() == 0
    assert dialog._reports == []
    assert dialog._snapshot is None
    assert dialog._generation == generation + 1
    dialog.close()


def test_preview_explains_explicit_and_blank_reference(qtbot):
    dialog = preview._CalibrationPreview(dict)
    qtbot.addWidget(dialog)
    texts = ' '.join(label.text() for label in dialog.findChildren(QLabel))
    assert 'Without session inputs, the selected reference plate is used; leave it blank to use the first plate by name.' in texts
    assert 'With session inputs, leave the reference session blank to use the first session by name, or the only session on the selected reference plate.' in texts
    assert 'A selected plate and session must agree; a plate with multiple sessions requires an explicit reference session.' in texts
    dialog.close()
