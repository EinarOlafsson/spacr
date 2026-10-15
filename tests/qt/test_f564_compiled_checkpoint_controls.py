"""Actual offscreen Qt settings controls consume the four reviewed rows per locale."""
import hashlib
import json
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QTextDocument
from PySide6.QtWidgets import QFormLayout


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'fr', 'pt', 'is', 'hi', 'ko', 'zh_CN'])
def test_actual_checkpoint_labels_and_help_use_compiled_targets(qtbot, tmp_path, monkeypatch, language):
    from spacr.qt import preferences
    from spacr.qt.i18n import retranslate_widget_tree
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens.settings_model import _unwrap_setting_label
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    monkeypatch.setenv('SPACR_LANGUAGE', language)
    private = tmp_path / 'preferences.ini'
    monkeypatch.setattr(preferences, '_settings', lambda: QSettings(str(private), QSettings.IniFormat))
    preferences._set_show_alpha_features(True)
    review = (Path(__file__).parents[2] / 'docs/i18n/reviewed/runtime' / language
              / '2026-10-15-counterfactual-pinned-reuse.json')
    rows = json.loads(review.read_text())['records']
    expected = {(row['table'], row['key']): row for row in rows}
    screen = AppScreen('activation')
    qtbot.addWidget(screen)
    try:
        screen._open_the_heading_of('counterfactuals')
        screen._refresh_alpha_visibility()
        retranslate_widget_tree(screen, language)
        model = screen._settings_model
        heading = next(section for section in screen.rendered_settings_sections()
                       if section.property('settingsCategorySource') == 'Counterfactuals')
        for key in ['counterfactual_checkpoint', 'counterfactual_checkpoint_sha256']:
            editor = model._widgets[key]
            assert screen.setting_row_is_visible(key)
            label = None
            for index in range(heading._form.rowCount()):
                field = heading._form.itemAt(index, QFormLayout.FieldRole).widget()
                if field is editor or getattr(field, '_spacr_field', None) is editor:
                    label = _unwrap_setting_label(heading._form.itemAt(index, QFormLayout.LabelRole).widget())
                    break
            assert label is not None
            label_row = expected[('setting_labels', key)]
            help_row = expected[('setting_tooltips', key)]
            for row in [label_row, help_row]:
                assert hashlib.sha256(row['source'].encode()).hexdigest() == row['source_sha256']
            assert label.text() == label_row['translation']
            document = QTextDocument()
            document.setHtml(str(editor.property('apiTooltipHtml')))
            assert help_row['translation'] in document.toPlainText()
            assert editor.property('settingKey') == key
        assert model.collect()['counterfactual_checkpoint'] == ''
        assert model.collect()['counterfactual_checkpoint_sha256'] == ''
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
