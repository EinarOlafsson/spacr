"""FEATURES accepts saved setting types and stays usable after partial failures."""
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox, QWidget

from spacr.qt.screens import measure_inputs as mi


@pytest.mark.parametrize('value,expected', [('yes', True), ('1', True), ('false', False), (None, False)])
def test_saved_boolean_values_reach_the_checkbox(qtbot, value, expected):
    widget = QCheckBox()
    qtbot.addWidget(widget)
    assert mi.write_setting_value(widget, value)
    assert widget.isChecked() is expected


@pytest.mark.parametrize('kind,value,expected', [
    (QSpinBox, '12.8', 12), (QDoubleSpinBox, '12.75', 12.75)])
def test_numeric_controls_accept_numbers_and_preserve_them_on_bad_input(qtbot, kind, value, expected):
    widget = kind()
    qtbot.addWidget(widget)
    assert mi.write_setting_value(widget, value)
    assert widget.value() == expected
    for invalid in (None, 'not a number'):
        assert not mi.write_setting_value(widget, invalid)
        assert widget.value() == expected


def test_auto_numeric_control_keeps_auto_distinct_from_a_number(qtbot):
    from spacr.qt.screens.settings_model import AUTO_TEXT
    widget = QDoubleSpinBox()
    qtbot.addWidget(widget)
    widget.setRange(-1, 100)
    widget.setSpecialValueText(AUTO_TEXT)
    assert mi.write_setting_value(widget, 5)
    assert widget.value() == 5
    assert mi.write_setting_value(widget, None)
    assert widget.value() == widget.minimum()


def test_choice_controls_match_data_then_text_without_inventing_options(qtbot):
    widget = QComboBox()
    qtbot.addWidget(widget)
    widget.addItem('Automatic', None)
    widget.addItem('Channel two', '2')
    widget.addItem('Custom', 'custom-key')
    assert mi.write_setting_value(widget, 2)
    assert widget.currentData() == '2'
    assert mi.write_setting_value(widget, 'Custom')
    assert widget.currentData() == 'custom-key'
    assert not mi.write_setting_value(widget, 'unavailable')
    assert widget.currentData() == 'custom-key' and widget.count() == 3
    assert mi.write_setting_value(widget, None)
    assert widget.currentText() == 'Automatic'


def test_text_none_clears_the_control_and_unknown_widgets_are_refused(qtbot):
    widget = QLineEdit('old')
    qtbot.addWidget(widget)
    assert mi.write_setting_value(widget, None)
    assert widget.text() == ''
    assert mi.write_setting_value(widget, 3)
    assert widget.text() == '3'
    assert not mi.write_setting_value(object(), 'value')


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    window = mi.MeasureInputsScreen(threaded=False)
    qtbot.addWidget(window)
    return window


def test_a_failing_settings_reader_keeps_table_derived_answers(screen, monkeypatch, tmp_path):
    screen.set_destination(str(tmp_path/'features'))
    expected = mi.field_table_settings(screen.inputs.table(), {},
                                      dst=mi.field_table_destination(screen.inputs.table(), screen.destination()))
    monkeypatch.setattr(screen.settings, 'collect', Mock(side_effect=RuntimeError('form rebuilding')))
    assert screen.derived_settings() == expected


def test_a_broken_setting_does_not_drop_other_settings_in_the_same_pack(screen, monkeypatch):
    broken = QWidget(screen)
    broken.set_value = Mock(side_effect=RuntimeError('editor closed'))
    usable = QSpinBox(screen)
    monkeypatch.setattr(screen.settings, '_widgets', {'broken': broken, 'usable': usable})
    assert screen.apply_settings_dict({'missing': 1, 'broken': 2, 'usable': 17}) == 1
    assert usable.value() == 17
    broken.set_value.assert_called_once_with(2)


def test_one_broken_derived_control_does_not_prevent_the_others_refreshing(screen, monkeypatch):
    broken = QWidget(screen)
    broken.set_value = Mock(side_effect=RuntimeError('editor closed'))
    src = QLineEdit(screen)
    monkeypatch.setattr(screen, '_decided_widgets', {'unrecognized': broken, 'src': src})
    expected = mi.field_table_settings(screen.inputs.table(), {},
                                      dst=mi.field_table_destination(screen.inputs.table(), screen.destination()))
    screen._show_decided_values()
    assert src.text() == str(expected['src'])
    broken.set_value.assert_called_once_with(None)


@pytest.mark.parametrize('result,phrase', [
    (None, 'produced no result'),
    ({'db_path': 'missing.db', 'stems': ['field-a'], 'db_exists': False}, 'WARNING')])
def test_completion_reports_absent_results_and_missing_database(screen, result, phrase):
    emitted = []
    screen.run_finished.connect(emitted.append)
    screen._set_running(True)
    screen._on_done(result)
    assert screen.inputs.isEnabled()
    assert screen.result() == result and emitted == [result]
    assert phrase in screen._log.toPlainText()
    assert not screen.run_button.isEnabled()  # an empty input table is still invalid


def test_settings_section_failure_still_leaves_a_working_window(qtbot, qt_theme_applied, monkeypatch):
    from spacr.qt.screens.settings_model import SettingsWidgets
    monkeypatch.setattr(SettingsWidgets, 'build_sections', Mock(side_effect=RuntimeError('missing section')))
    window = mi.MeasureInputsScreen(threaded=False)
    qtbot.addWidget(window)
    assert window.settings is not None
    assert window._settings_area.widget() is not None
    assert not window.run() and 'Nothing was measured' in window._status.text()
