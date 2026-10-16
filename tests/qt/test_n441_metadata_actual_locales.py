"""Actual metadata callbacks consume the ten normal reviewed locale records."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'fr', 'pt', 'hi', 'ko', 'is', 'zh_CN'])
def test_actual_metadata_widgets_use_all_ten_installed_translations(
        qapp, monkeypatch, tmp_path, language):
    from PySide6.QtCore import QSettings
    from spacr.qt import i18n, preferences
    from spacr.qt.screens import settings_model as model
    from spacr.regex_infer import _metadata_convention, _metadata_convention_example

    path = (Path(__file__).resolve().parents[2] / 'docs/i18n/reviewed/runtime'
            / language / '2026-10-15-metadata-convention-status.json')
    records = json.loads(path.read_text())['records']
    targets = {record['source']: record['translation'] for record in records}
    assert len(targets) == 10
    for record in records:
        assert record['source_sha256'] == hashlib.sha256(record['source'].encode()).hexdigest()
    settings = QSettings(str(tmp_path / 'preferences.ini'), QSettings.IniFormat)
    monkeypatch.setattr(preferences, '_settings', lambda: settings)
    monkeypatch.delenv(i18n.ENV_LANGUAGE, raising=False)
    preferences.set_language(language)
    settings.sync()
    assert preferences.get_language() == i18n.current_language() == language
    source = tmp_path / 'raw-field-original.tif'
    pixels = b'private opaque pixels; no decoder is used by these UI callbacks'
    source.write_bytes(pixels)
    assert _metadata_convention('scanr')['status'] == 'provisional'
    field = model._MetadataTypeField(default='scanr', source_folder=lambda: str(tmp_path))
    field.set_threaded(False)
    key = field.get_value()
    keys = [field.combo.itemData(index) for index in range(field.combo.count())]
    def translated(template, **values):
        leading = template[:len(template) - len(template.lstrip())]
        trailing = template[len(template.rstrip()):]
        return leading + targets[template.strip()].format(**values) + trailing
    try:
        i18n.retranslate_widget_tree(field, language)
        assert field.combo.currentText().endswith(translated(model.PROVISIONAL_SUFFIX))
        example = _metadata_convention_example('scanr')
        assert field.example.text() == (translated(model.LOOKS_LIKE, example=example)
                                        + '  ' + translated(model.RECONSTRUCTED_NOT_DOCUMENTED))
        assert _metadata_convention('evos')['status'] == 'confirmed'
        field.set_value('evos')
        assert field.get_value() == 'evos'
        assert not field.combo.currentText().endswith(translated(model.PROVISIONAL_SUFFIX))
        assert field.example.text() == translated(
            model.LOOKS_LIKE, example=_metadata_convention_example('evos'))
        assert translated(model.RECONSTRUCTED_NOT_DOCUMENTED) not in field.example.text()
        field.set_value(key)
        assert field.get_value() == key
        field._say(model.READING_THE_FOLDER)
        assert field.report.text() == translated(model.READING_THE_FOLDER)
        field._on_failed('')
        assert field.report.text() == translated(model.NO_IMAGES_IN_THE_FOLDER)
        field._on_tested(3, 3, '', 'tif')
        assert field.report.text() == translated(model.ALL_FILES_PARSE, total=3, extension='tif')
        bad = 'unchanged_Å01_C2.tif'
        field._on_tested(2, 3, bad, 'tif')
        assert field.report.text() == translated(model.SOME_FILES_PARSE, matched=2, total=3, first=bad)
        field._on_ranked([], 4)
        assert field.report.text() == translated(model.NOTHING_FITS, total=4)
        ranked = [('opera_phenix', 3, 4), ('evos', 1, 4)]
        original = deepcopy(ranked)
        field._on_ranked(ranked, 4)
        expected = translated(model.RANKING_HEADING, total=4) + '\n' + '\n'.join(
            translated(model.ONE_RANKING_ROW, matched=count, total=4,
                       label=_metadata_convention(name)['label'], key=name)
            for name, count, _ in ranked)
        assert field.report.text() == expected
        preferences.set_language('en')
        settings.sync()
        assert preferences.get_language() == i18n.current_language() == 'en'
        i18n.retranslate_widget_tree(field, 'en')
        assert field.report.text().startswith(model.RANKING_HEADING.format(total=4))
        preferences.set_language(language)
        settings.sync()
        assert preferences.get_language() == i18n.current_language() == language
        i18n.retranslate_widget_tree(field, language)
        assert field.report.text() == expected
        assert ranked == original and field.get_value() == key
        assert [field.combo.itemData(index) for index in range(field.combo.count())] == keys
        assert source.read_bytes() == pixels
    finally:
        field.close()
        field.deleteLater()
