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
