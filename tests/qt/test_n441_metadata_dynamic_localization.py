"""Actual metadata widgets translate canonical templates before inserting data."""
from copy import deepcopy

import pytest


@pytest.fixture
def metadata_field(qapp, monkeypatch, tmp_path):
    from spacr.qt import i18n
    from spacr.qt.screens import settings_model as model

    targets = {
        model.PROVISIONAL_SUFFIX: '   [provisoire]',
        model.LOOKS_LIKE: 'Exemple : {example}',
        model.RECONSTRUCTED_NOT_DOCUMENTED: "Cette convention est provisoire. Vérifiez son exemple et testez votre dossier avant de lancer l’analyse.",
        model.READING_THE_FOLDER: 'Lecture du dossier…',
        model.NO_IMAGES_IN_THE_FOLDER: 'Aucune image dans ce dossier.',
        model.ALL_FILES_PARSE: 'Les {total} fichiers .{extension} sont reconnus.',
        model.SOME_FILES_PARSE: '{matched} sur {total} reconnus ; premier échec : {first}',
        model.NOTHING_FITS: 'Aucune convention ne reconnaît ces {total} noms.',
        model.RANKING_HEADING: 'Sur {total} fichiers :',
        model.ONE_RANKING_ROW: '    {matched} sur {total} — {label} ({key})',
    }
    original = i18n._exact_translation
    seen = []
    def reviewed_fixture_lookup(source, language):
        if language == 'fr' and source in targets:
            seen.append(source)
            return targets[source]
        return original(source, language)
    monkeypatch.setattr(i18n, '_exact_translation', reviewed_fixture_lookup)
    monkeypatch.setattr(i18n, 'current_language', lambda: 'fr')
    field = model._MetadataTypeField(default='scanr', source_folder=lambda: str(tmp_path))
    field.set_threaded(False)
    yield field, model, targets, seen
    field.close()
    field.deleteLater()


def test_provisional_menu_translates_marker_and_keeps_raw_keys(metadata_field):
    field, model, targets, seen = metadata_field
    index = field.combo.findData('scanr')
    assert field.combo.itemText(index).endswith(targets[model.PROVISIONAL_SUFFIX])
    assert field.get_value() == 'scanr'
    assert model.PROVISIONAL_SUFFIX in seen


def test_provisional_example_translates_both_pieces_without_changing_filename(metadata_field):
    from spacr.regex_infer import _metadata_convention_example

    field, model, targets, seen = metadata_field
    filename = _metadata_convention_example('scanr')
    assert field.example.text() == targets[model.LOOKS_LIKE].format(example=filename) + '  ' + targets[model.RECONSTRUCTED_NOT_DOCUMENTED]
    assert field.get_value() == 'scanr'


def test_confirmed_evos_keeps_translated_example_without_a_provisional_warning(
        metadata_field):
    from spacr.regex_infer import _metadata_convention, _metadata_convention_example

    field, model, targets, seen = metadata_field
    assert _metadata_convention('scanr')['status'] == 'provisional'
    field.set_value('evos')
    assert _metadata_convention('evos')['status'] == 'confirmed'
    assert field.get_value() == 'evos'
    assert not field.combo.currentText().endswith(targets[model.PROVISIONAL_SUFFIX])
    assert field.example.text() == targets[model.LOOKS_LIKE].format(
        example=_metadata_convention_example('evos'))
    assert targets[model.RECONSTRUCTED_NOT_DOCUMENTED] not in field.example.text()


@pytest.mark.parametrize('matched,total,first,source', [
    (3, 3, '', 'ALL_FILES_PARSE'),
    (2, 3, 'unparsed_Å01_C2.tif', 'SOME_FILES_PARSE'),
])
def test_actual_parse_result_translates_before_inserting_unmodified_data(
        metadata_field, matched, total, first, source):
    field, model, targets, seen = metadata_field
    original = field.get_value()
    field._on_tested(matched, total, first, 'tif')
    template = getattr(model, source)
    assert field.report.text() == targets[template].format(
        matched=matched, total=total, first=first, extension='tif')
    assert field.get_value() == original
    assert template in seen


def test_empty_and_reading_reports_use_exact_template_lookup(metadata_field):
    field, model, targets, seen = metadata_field
    field._say(model.READING_THE_FOLDER)
    assert field.report.text() == targets[model.READING_THE_FOLDER]
    field._on_failed('')
    assert field.report.text() == targets[model.NO_IMAGES_IN_THE_FOLDER]


def test_ranked_results_translate_components_without_assigning_a_convention(metadata_field):
    from spacr.regex_infer import _metadata_convention

    field, model, targets, seen = metadata_field
    ranked = [('opera_phenix', 3, 4), ('evos', 1, 4)]
    before = deepcopy(ranked)
    chosen = field.get_value()
    field._on_ranked(ranked, 4)
    expected = [targets[model.RANKING_HEADING].format(total=4)]
    expected += [targets[model.ONE_RANKING_ROW].format(
        matched=count, total=4, label=_metadata_convention(key)['label'], key=key)
        for key, count, _ in ranked]
    assert field.report.text() == '\n'.join(expected)
    assert ranked == before and field.get_value() == chosen
    field._on_ranked([], 4)
    assert field.report.text() == targets[model.NOTHING_FITS].format(total=4)
    assert field.get_value() == chosen


def test_language_refresh_rebuilds_dynamic_text_from_preserved_values(metadata_field):
    field, model, targets, seen = metadata_field
    field._on_tested(1, 3, 'original_field_C1.tif', 'tif')
    key = field.get_value()
    field.retranslate_dynamic_content('en')
    assert field.report.text() == model.SOME_FILES_PARSE.format(
        matched=1, total=3, first='original_field_C1.tif')
    field.retranslate_dynamic_content('fr')
    assert field.report.text() == targets[model.SOME_FILES_PARSE].format(
        matched=1, total=3, first='original_field_C1.tif')
    assert field.get_value() == key
    assert field.combo.currentText().endswith(targets[model.PROVISIONAL_SUFFIX])


def test_actual_tree_language_changes_preserve_field_owned_provisional_items(metadata_field):
    from spacr.qt.i18n import retranslate_widget_tree

    field, model, targets, seen = metadata_field
    selected = field.get_value()
    for language in ('fr', 'en', 'fr'):
        retranslate_widget_tree(field, language)
        suffix = targets[model.PROVISIONAL_SUFFIX] if language == 'fr' else model.PROVISIONAL_SUFFIX
        assert field.combo.currentText().endswith(suffix)
        assert field.get_value() == selected


@pytest.mark.parametrize('key', ['scanr', 'zeiss_zen_split_tiles'])
def test_provisional_warning_is_truthful_for_manual_and_constructed_examples(
        qapp, monkeypatch, key):
    from spacr.qt import i18n
    from spacr.qt.screens.settings_model import _MetadataTypeField
    from spacr.regex_infer import _metadata_convention_example

    monkeypatch.setattr(i18n, 'current_language', lambda: 'en')
    field = _MetadataTypeField(default=key)
    try:
        assert field.example.text() == (
            'Looks like:  ' + _metadata_convention_example(key)
            + '  This convention is provisional. Check its example and test your folder before running.')
        assert field.get_value() == key
    finally:
        field.close()
        field.deleteLater()


def test_no_matching_convention_uses_natural_language_without_python_none_literal(qapp, monkeypatch):
    from spacr.qt import i18n
    from spacr.qt.screens.settings_model import _MetadataTypeField

    monkeypatch.setattr(i18n, 'current_language', lambda: 'en')
    field = _MetadataTypeField(default='evos')
    try:
        field._on_ranked([], 7)
        assert field.report.text() == (
            'No built-in convention can read any of these 7 names. Write a custom_regex, '
            'or run Import, which reads the folder names too.')
        assert field.get_value() == 'evos'
    finally:
        field.close()
        field.deleteLater()
