"""Protocol choices advance explicitly and preserve safe native input handoffs."""
from copy import deepcopy
import re

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtGui import QTextDocument
from PySide6.QtWidgets import QComboBox, QWidget

from spacr.qt.widgets import sample_project as sample


@pytest.fixture
def protocol_data():
    data = deepcopy(sample.workflow_map())
    data['protocols'] = {
        'count': dict(title='Count objects', summary='Segment and measure images.', pathway='high_content',
                      inputs=['raw_images'], checkpoint='Inspect masks before measuring.',
                      steps=[dict(module='mask', action='Preview and run Mask.', after=[]),
                             dict(module='measure', action='Measure the merged project.', after=['mask'])]),
        'train': dict(title='Train a model', summary='Review masks.', pathway='train_model', inputs=['masks'],
                      steps=[dict(module='make_masks', action='Correct labels.', after=[])]),
        'plot': dict(title='Plot measurements', summary='Explore feature tables.', pathway='high_content',
                     inputs=['measurements'], steps=[dict(module='graph_builder', action='Make a plot.', after=[])])
    }
    data['protocol_tree'] = dict(id='protocols', title='Protocols', description='Choose your data.', children=[
        dict(id='images', title='Images', description='Start with images.', children=[
            dict(id='segment', title='Segment images', description='Choose an image scale.', children=[
                dict(id='high', title='High content', description='Use automated masks.', children=[
                    dict(id='count', title='Generate Masks', description='Preview and measure images.',
                         protocol='count', start_module='measure')]),
                dict(id='low', title='Low content', description='Review individual masks.', children=[
                    dict(id='train', title='Make Masks', description='Correct labels before training.',
                         protocol='train', start_module='make_masks')])])]),
        dict(id='features', title='Measured features', description='Explore existing measurements.', children=[
            dict(id='plot', title='Plot measurements', description='Choose feature columns.',
                 protocol='plot', start_module='graph_builder')])])
    return data


def _choose(dialog, value):
    children = dialog._trail[-1].get('children', ())
    row = next(i for i, node in enumerate(children) if node['id'] == value)
    dialog._choice_list.setCurrentRow(row)


def _advance(dialog, value):
    _choose(dialog, value)
    dialog._next.click()


def _walk(dialog, values):
    for value in values:
        _advance(dialog, value)


def test_single_level_choices_explain_selection_and_advance_only_on_next(qtbot, protocol_data):
    dialog = sample._ProtocolBrowserDialog(data=protocol_data)
    qtbot.addWidget(dialog)
    dialog.show()
    assert not dialog.isModal()
    assert not dialog.findChildren(QComboBox)
    assert dialog._choice_list.count() == 2
    assert not dialog._previous.isEnabled() and not dialog._next.isEnabled()
    _choose(dialog, 'images')
    assert dialog._text.toPlainText() == 'Start with images.'
    assert len(dialog._trail) == 1 and dialog._selected_protocol is None
    assert dialog._choice_list.item(0).text() == 'Images'
    dialog._next.click()
    assert dialog._trail[-1]['id'] == 'images'
    assert dialog._choice_list.count() == 1
    assert not dialog._next.isEnabled()
    assert dialog._selected_protocol is None
    _walk(dialog, ['segment', 'high', 'count'])
    assert dialog._selected_protocol == 'count'
    assert not dialog._choice_list.isVisible() and not dialog._next.isVisible()
    assert dialog._diagram_host.isVisible()
    assert dialog._diagram.nodes['measure'].highlighted
    assert not dialog._diagram.nodes['mask'].highlighted
    text = dialog._text.toPlainText()
    assert all(value in text for value in ('Count objects', 'Preview and run Mask.',
                                          'Measure the merged project.', 'Inputs', 'Outputs', 'Checkpoint'))
    assert 'step-measure' in dialog._text.toHtml()


def test_previous_preserves_choices_and_new_branch_has_its_own_explanation(qtbot, protocol_data):
    dialog = sample._ProtocolBrowserDialog(data=protocol_data)
    qtbot.addWidget(dialog)
    dialog.show()
    _walk(dialog, ['images', 'segment', 'high', 'count'])
    dialog._previous.click()
    assert dialog._choice_list.currentItem().text() == 'Generate Masks'
    assert dialog._selected_protocol is None
    dialog._previous.click()
    assert dialog._choice_list.currentItem().text() == 'High content'
    _choose(dialog, 'low')
    assert dialog._text.toPlainText() == 'Review individual masks.'
    _walk(dialog, ['low', 'train'])
    assert dialog._selected_protocol == 'train'
    assert 'measure' not in dialog._diagram.nodes
    for _ in range(4):
        dialog._previous.click()
    assert dialog._trail[-1]['id'] == 'protocols'
    assert dialog._choice_list.currentItem().text() == 'Images'
    _walk(dialog, ['features', 'plot'])
    assert dialog._selected_protocol == 'plot'


def test_source_and_metadata_snapshot_survive_later_navigator_changes(qtbot, protocol_data, tmp_path):
    class Owner(QWidget):
        def __init__(self):
            super().__init__()
            self.calls = []
        def _open_easy_input(self, key, path, context):
            self.calls.append((key, path, deepcopy(context)))
            return True
    owner = Owner()
    qtbot.addWidget(owner)
    source = tmp_path / 'project'
    context = {'generated_outputs': {'artifacts': [{'kind': 'measurements'}]},
               'analysis_inputs': {'db_browser': {'path': 'original.db'}}}
    dialog = sample._ProtocolBrowserDialog(owner, str(source), context, data=protocol_data)
    qtbot.addWidget(dialog)
    assert dialog._choice_list.currentItem().text() == 'Measured features'
    assert len(dialog._trail) == 1 and dialog._selected_protocol is None
    _walk(dialog, ['features', 'plot'])
    context['analysis_inputs']['db_browser']['path'] = 'changed.db'
    assert not owner.calls
    dialog._activate_link(QUrl('spacr://module/graph_builder'))
    assert owner.calls[0][0:2] == ('graph_builder', str(source))
    assert owner.calls[0][2]['analysis_inputs']['db_browser']['path'] == 'original.db'
    assert not source.exists()
    assert not dialog._activate_link(QUrl('spacr://module/graph_builder?run=1'))
    assert len(owner.calls) == 1


def test_hidden_module_links_and_remote_resources_are_inactive(qtbot, monkeypatch, protocol_data):
    from spacr.qt.ai import prompts

    original = prompts._navigation_modules
    monkeypatch.setattr(prompts, '_navigation_modules', lambda: tuple(row for row in original() if row[0] != 'make_masks'))
    protocol_data['protocols']['train']['summary'] = '<img src="https://example.invalid/private.png">'
    dialog = sample._ProtocolBrowserDialog(data=protocol_data)
    qtbot.addWidget(dialog)
    _walk(dialog, ['images', 'segment', 'low', 'train'])
    assert 'spacr://module/make_masks' not in dialog._text.toHtml()
    assert '<img src=' in dialog._text.toPlainText()
    assert not any(value.isImageFormat() for value in dialog._text.document().allFormats())
    assert dialog._text.document().loadResource(QTextDocument.ImageResource, QUrl('https://example.invalid/image.png')) is None
    assert not dialog._activate_link(QUrl('spacr://module/make_masks'))


def test_popup_reuses_window_and_preserves_progress_until_input_changes(qtbot, monkeypatch, protocol_data):
    monkeypatch.setattr(sample, 'workflow_map', lambda: protocol_data)
    owner = QWidget()
    qtbot.addWidget(owner)
    first = sample._show_protocols(owner)
    qtbot.addWidget(first)
    _walk(first, ['images', 'segment', 'high', 'count'])
    first.close()
    assert sample._show_protocols(owner) is first
    assert first._selected_protocol == 'count'
    assert sample._show_protocols(owner, '/chosen/project', {'protocol_data': 'features'}) is first
    assert first._trail[-1]['id'] == 'protocols'
    assert first._selected_protocol is None
    assert first._choice_list.currentItem().text() == 'Measured features'


@pytest.mark.parametrize('source,context,expected', [
    ('/selected/figure.pdf', {}, 'images'),
    ('/selected/plate/merged/field.npy', {'generated_outputs': {'artifacts': [{'kind': 'merged'}]}}, 'images_masks'),
    ('/selected/measurements.db', {'database_schemas': [{'objects': True, 'numeric': True}]}, 'features'),
    ('/selected/reads.fastq.gz', {}, 'fastq'),
])
def test_input_hints_select_only_top_choice_without_skipping_levels(qtbot, source, context, expected):
    dialog = sample._ProtocolBrowserDialog(source_path=source, context=context)
    qtbot.addWidget(dialog)
    row = dialog._choice_list.currentRow()
    assert dialog._trail[-1]['children'][row]['id'] == expected
    assert len(dialog._trail) == 1 and dialog._selected_protocol is None


def test_every_shipped_protocol_is_reachable_and_native_links_stay_allowlisted(qtbot):
    from spacr.qt.ai.prompts import _navigation_modules
    from spacr.qt.widgets.ai_chat_panel import _ai_module_key

    data = sample.workflow_map()
    dialog = sample._ProtocolBrowserDialog(data=data)
    qtbot.addWidget(dialog)
    visible = {row[0] for row in _navigation_modules()}
    paths = []
    pending = [(data['protocol_tree'], [])]
    while pending:
        node, path = pending.pop()
        if node.get('protocol'):
            paths.append((node, path))
        pending.extend((child, path + [child['id']]) for child in node.get('children', ()))
    assert {node['protocol'] for node, _ in paths} == set(data['protocols'])
    assert set(data['pathways']) <= {guide['pathway'] for guide in data['protocols'].values()}
    for leaf, path in paths:
        dialog._set_input()
        _walk(dialog, path)
        assert dialog._selected_protocol == leaf['protocol']
        start = leaf.get('start_module')
        if start:
            assert dialog._diagram.nodes[start].highlighted
        for href in re.findall(r'href="([^"]+)"', dialog._text.toHtml()):
            if href.startswith('spacr:'):
                assert _ai_module_key(QUrl(href)) in visible


def test_protocol_wizard_current_capabilities_and_registry_contracts(tmp_path):
    """Four input families retain real reader and crop/provenance boundaries."""
    from spacr.qt.organisms import ORGANISMS
    from spacr.qt.ai.prompts import _navigation_modules

    data = sample.workflow_map()
    assert [row['key'] for row in sample._protocol_options()] == [
        'small_images', 'plate_images', 'sequencing', 'features']
    for family in ('small_images', 'plate_images'):
        result = sample._protocol_result(family, 'pdf')
        assert [row[0] for row in result['actions']] == ['analyze_plaques']
        assert result['source_path'] == ''
        assert set(data['modules']) <= result['protocol']['catalog'].keys()
        assert {row[0] for row in _navigation_modules()} <= result['protocol']['catalog'].keys()
        assert set(ORGANISMS) <= result['protocol']['catalog'].keys()
        assert '<details><summary>' in result['protocol']['markdown']
    modes = dict(sequencer='other', read_mode='mixed')
    result = sample._protocol_result('sequencing', 'fastq_gz', modes)
    assert not result['actions'] and any('N740' in text for text in result['requirements'])
    assert modes == dict(sequencer='other', read_mode='mixed')
    with pytest.raises(ValueError, match='required'):
        sample._protocol_result('sequencing', 'fastq_gz', dict(sequencer='other'))
    result = sample._protocol_result('features', 'database', dict(experiment='drug'))
    assert 'dose_response' in {row[0] for row in result['actions']}
    assert 'regression' not in {row[0] for row in result['actions']}
    assert any('N741' in text for text in result['requirements'])
    assert not {'annotate', 'classify_merged', 'umap'}.intersection(row[0] for row in result['actions'])
    generic = sample._protocol_result('features', 'database', dict(experiment='other', organism='other'))
    assert {'graph_builder', 'gate_editor', 'db_browser'} <= {row[0] for row in generic['actions']}
    assert not {'regression', 'dose_response'}.intersection(row[0] for row in generic['actions'])
    assert generic['protocol']['selection']['choices']['organism'] == 'other'
    database = tmp_path / 'measurements.db'
    import sqlite3
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE fov (fieldID INTEGER, intensity REAL)')
        connection.execute('INSERT INTO fov VALUES (1, 7.5)')
    raw = database.read_bytes()
    result = sample._protocol_result('features', 'database', dict(experiment='screen', barcodes='unmapped'), str(database))
    handoff = sample._protocol_handoff(result, 'map_barcodes')
    assert handoff['source_path'] == '' and handoff['settings']['src'] == ''
    assert handoff['clear_source']
    assert result['source_path'] == str(database) and database.read_bytes() == raw
    with pytest.raises(ValueError, match='required'):
        sample._protocol_result('sequencing', 'fastq_gz', dict(sequencer='other', read_mode='single'))


def test_protocol_wizard_handoff_uses_real_metadata_and_copies_source_context(tmp_path):
    """Selections become compatible settings only after an explicit card handoff."""
    from spacr.regex_infer import _metadata_convention_keys

    source = tmp_path / 'images'
    source.mkdir()
    raw = source / 'plate_A01_F001_C1.tif'
    import numpy as np
    import tifffile
    tifffile.imwrite(raw, np.arange(64, dtype=np.uint16).reshape(8, 8))
    original = raw.read_bytes()
    fields = sample._protocol_options('small_images', 'tiff_other')
    metadata = next(field for field in fields if field['name'] == 'metadata_type')
    assert {row['key'] for row in metadata['options']} == set(_metadata_convention_keys())
    choices = dict(metadata_type='custom', goal='mask')
    context = dict(custom_regex=r'(?P<plateID>.+)_(?P<wellID>A01)_F(?P<fieldID>\d+)_C(?P<chanID>\d+)')
    context['submodule_settings'] = {'mask': {'settings': {'cell_channel': 2}, 'reason': 'verified channels'}}
    result = sample._protocol_result('small_images', 'tiff_other', choices, str(source), context)
    handoff = sample._protocol_handoff(result, 'mask')
    assert handoff['settings']['src'] == str(source)
    assert handoff['settings']['metadata_type'] == 'custom'
    assert handoff['settings']['custom_regex'] == context['custom_regex']
    assert 'img_format' not in handoff['settings']
    assert handoff['context']['submodule_settings']['mask']['settings']['cell_channel'] == 2
    assert handoff['context']['submodule_settings']['mask']['reason'] == 'verified channels'
    handoff['context']['protocol_selection']['choices']['metadata_type'] = 'auto'
    assert result['context']['protocol_selection']['choices'] == choices
    with pytest.raises(ValueError):
        sample._protocol_handoff(result, 'regression')
    assert raw.read_bytes() == original
    assert not (source / 'measurements').exists()


def test_protocol_recipe_nested_controls_pdf_and_legacy_reopen(qtbot, tmp_path):
    """Real popup links update check/details; expanded PDF retains their content."""
    result = sample._protocol_result('sequencing', 'fastq_gz', dict(sequencer='other', read_mode='single', single_direction='R2'))
    dialog = sample._ProtocolBrowserDialog(result=result)
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog._save_pdf.isVisible()
    assert 'Read orientation: R2' in dialog._text.toPlainText()
    assert not dialog._choice_list.isVisible()
    dialog._activate_link(QUrl('spacr://protocol/check/map_barcodes'))
    dialog._activate_link(QUrl('spacr://protocol/expand/map_barcodes'))
    assert 'map_barcodes' in dialog._checked
    assert 'Setting explanations' in dialog._text.toPlainText()
    dialog._activate_link(QUrl('spacr://protocol/expand/map_barcodes:settings'))
    assert 'target_sequence' in dialog._text.toPlainText()
    assert "Selected: 'R2'" in dialog._text.toPlainText()
    assert '[x]' in dialog._recipe_result['protocol']['markdown']
    assert '### target_sequence' in dialog._recipe_result['protocol']['markdown']
    path = tmp_path / 'protocol.pdf'
    assert dialog._export_recipe_pdf(path) == str(path)
    assert path.read_bytes().startswith(b'%PDF-')
    exported = dialog._recipe_html(expand_all=True)
    assert 'spacr://' not in exported
    assert 'https://einarolafsson.github.io/spacr/' in exported
    assert "Factory default: &#x27;path&#x27;" in exported
    assert 'All registered routes and prerequisites' not in dialog._recipe_html(expand_all=True)
    dialog._activate_link(QUrl('spacr://protocol/expand/catalog'))
    assert 'All registered routes and prerequisites' in dialog._recipe_html(expand_all=True)
    dialog._set_input()
    assert dialog._recipe_result is None
    assert dialog._choice_list.isVisible() and not dialog._save_pdf.isVisible()


def test_protocol_procedure_preserves_order_optional_training_and_dimensional_limits():
    """Curated steps distinguish required outputs from optional incompatible branches."""
    image = sample._protocol_result('small_images', 'tiff_yokogawa', source_path='/selected/images',
                                    context={'saved_settings': {'module': 'measure', 'settings': {'src': '/unrelated'}}})
    steps = image['protocol']['steps']
    assert [step['module'] for step in steps[:3]] == ['mask', 'measure', 'graph_builder']
    assert steps[1]['after'] == ['mask'] and not steps[1]['optional']
    handoff = sample._protocol_handoff(image, 'measure')
    assert handoff['source_path'] == '' and handoff['settings']['src'] == '' and handoff['clear_source']
    assert 'saved_settings' not in handoff['context'] and 'protocol_native_sources' not in handoff['context']
    manual = sample._protocol_result('small_images', 'tiff_yokogawa', dict(goal='make_masks'))
    training = next(step for step in manual['protocol']['steps'] if step['module'] == 'train_cellpose')
    assert training['optional'] and training['after'] == ['make_masks']
    markdown = manual['protocol']['markdown']
    assert 'Starting alternatives' in markdown and 'Optional downstream step' in markdown
    assert 'After: Make Masks' in markdown and 'Required artifacts:' in markdown
    native = sample._protocol_result('small_images', 'tiff_yokogawa', context={
        'image_dimensions': [{'z': 2, 't': 3, 'axes_unresolved': False}]})
    assert [row[0] for row in native['actions']] == ['mask']
    assert [step['module'] for step in native['protocol']['steps']] == ['mask']
    assert 'does not enable Measure, tracking or Classify' in native['protocol']['markdown']
    with pytest.raises(ValueError):
        sample._protocol_handoff(native, 'measure')
    viewer = sample._protocol_result('small_images', 'tiff_yokogawa', dict(goal='3d'))
    assert [step['module'] for step in viewer['protocol']['steps']] == ['3d']


def test_protocol_settings_help_uses_the_existing_source_bound_locale_body(monkeypatch):
    """Scientific help reuses the exact catalog paragraph instead of partial UI terms."""
    import importlib
    monkeypatch.setenv('SPACR_LANGUAGE', 'zh_CN')
    english = importlib.import_module('spacr.qt.i18n_catalogs.en')
    chinese = importlib.import_module('spacr.qt.i18n_catalogs.zh_CN')
    help_ = {name: text for name, value, text in sample._protocol_setting_help('map_barcodes')}
    assert help_['target_sequence'] == chinese.SETTING_TOOLTIPS['target_sequence']
    assert help_['target_sequence'] != english.SETTING_TOOLTIPS['target_sequence']
    assert any('\u3400' <= char <= '\u9fff' for char in help_['target_sequence'])
