"""Protocols filter exact branches and open native tools only after explicit clicks."""
from copy import deepcopy

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtGui import QTextDocument
from PySide6.QtWidgets import QWidget

from spacr.qt.widgets import sample_project as sample


@pytest.fixture
def protocol_data():
    data = deepcopy(sample.workflow_map())
    data['protocol_choices'] = {
        'data': {'images': 'Images', 'features': 'Measured features'},
        'goal': {'quantification': 'Quantify', 'segmentation_training': 'Train segmentation',
                 'exploration': 'Explore'},
        'stage': {'start': 'Raw images', 'prepared': 'Prepared data'},
        'details': {'yokogawa': 'Yokogawa', 'zeiss': 'Zeiss', 'masks': 'Existing masks',
                    'measurements': 'Measurements'},
    }
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
    data['protocol_routes'] = [
        dict(data='images', goal='quantification', stage='start', details=detail, protocol='count')
        for detail in ('yokogawa', 'zeiss')
    ] + [dict(data='images', goal='segmentation_training', stage='prepared', details='masks', protocol='train'),
         dict(data='features', goal='exploration', stage='prepared', details='measurements', protocol='plot')]
    return data


def _choose(dialog, key, value):
    control = dialog._selectors[key]
    index = control.findData(value)
    assert index > 0, (key, value)
    control.setCurrentIndex(index)


def test_exact_branch_filtering_resets_children_and_hides_unneeded_details(qtbot, protocol_data):
    dialog = sample._ProtocolBrowserDialog(data=protocol_data)
    qtbot.addWidget(dialog)
    dialog.show()
    assert not dialog.isModal()
    assert not dialog._selectors['goal'].isEnabled()
    _choose(dialog, 'data', 'images')
    _choose(dialog, 'goal', 'quantification')
    assert dialog._selectors['stage'].currentData() == 'start'
    assert dialog._selectors['details'].isVisible()
    assert dialog._selected_protocol is None
    _choose(dialog, 'details', 'yokogawa')
    assert dialog._selected_protocol == 'count'
    assert set(dialog._diagram.nodes) >= {'mask', 'measure'}
    assert any(edge['from'] == 'mask' and edge['to'] == 'measure' for edge in dialog._diagram.links)
    text = dialog._text.toPlainText()
    assert all(value in text for value in ('Count objects', 'Preview and run Mask.',
                                          'Measure the merged project.', 'Inputs', 'Outputs', 'Checkpoint'))
    _choose(dialog, 'goal', 'segmentation_training')
    assert dialog._selectors['details'].currentData() == 'masks'
    assert not dialog._selectors['details'].isVisible()
    assert dialog._selected_protocol == 'train'
    assert 'measure' not in dialog._diagram.nodes
    _choose(dialog, 'data', 'features')
    assert dialog._selected_protocol == 'plot'
    assert dialog._selectors['goal'].findData('segmentation_training') == -1


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
    assert dialog._selectors['data'].currentData() == 'features'
    assert dialog._selected_protocol == 'plot'
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
    _choose(dialog, 'data', 'images')
    _choose(dialog, 'goal', 'segmentation_training')
    assert 'spacr://module/make_masks' not in dialog._text.toHtml()
    assert '<img src=' in dialog._text.toPlainText()
    assert not any(value.isImageFormat() for value in dialog._text.document().allFormats())
    assert dialog._text.document().loadResource(QTextDocument.ImageResource, QUrl('https://example.invalid/image.png')) is None
    assert not dialog._activate_link(QUrl('spacr://module/make_masks'))


def test_popup_reuses_its_window_and_preserves_selections_until_input_changes(qtbot, monkeypatch, protocol_data):
    monkeypatch.setattr(sample, 'workflow_map', lambda: protocol_data)
    owner = QWidget()
    qtbot.addWidget(owner)
    first = sample._show_protocols(owner)
    qtbot.addWidget(first)
    _choose(first, 'data', 'images')
    _choose(first, 'goal', 'quantification')
    _choose(first, 'details', 'zeiss')
    first.close()
    assert sample._show_protocols(owner) is first
    assert first._selectors['details'].currentData() == 'zeiss'
    assert sample._show_protocols(owner, '/chosen/project', {'protocol_data': 'features'}) is first
    assert first._selectors['data'].currentData() == 'features'
    assert first._selected_protocol == 'plot'


def test_selected_route_orients_the_walkthrough_at_its_starting_module(qtbot, protocol_data):
    for route in protocol_data['protocol_routes']:
        if route['protocol'] == 'count':
            route['start_module'] = 'measure'
    dialog = sample._ProtocolBrowserDialog(data=protocol_data)
    qtbot.addWidget(dialog)
    _choose(dialog, 'data', 'images')
    _choose(dialog, 'goal', 'quantification')
    _choose(dialog, 'details', 'yokogawa')
    assert dialog._diagram.nodes['measure'].highlighted
    assert not dialog._diagram.nodes['mask'].highlighted
    assert 'Preview and run Mask.' in dialog._text.toPlainText()
    assert 'step-measure' in dialog._text.toHtml()


def test_metadata_prefills_stage_after_goal_selection_without_inspecting_source(qtbot, protocol_data, tmp_path):
    protocol_data['protocol_routes'].append(dict(data='images', goal='quantification', stage='prepared',
                                                details='masks', protocol='count', start_module='measure'))
    dialog = sample._ProtocolBrowserDialog(source_path=str(tmp_path / 'not-scanned'),
        context={'protocol_data': 'images', 'protocol_stage': 'prepared'}, data=protocol_data)
    qtbot.addWidget(dialog)
    assert dialog._selectors['data'].currentData() == 'images'
    assert dialog._selectors['goal'].currentData() is None
    _choose(dialog, 'goal', 'quantification')
    assert dialog._selectors['stage'].currentData() == 'prepared'
    assert dialog._selectors['details'].currentData() == 'masks'


def test_shipped_protocols_cover_every_existing_pipeline_and_render_native_routes(qtbot):
    data = sample.workflow_map()
    assert set(data['pathways']) <= {guide['pathway'] for guide in data['protocols'].values()}
    dialog = sample._ProtocolBrowserDialog(data=data)
    qtbot.addWidget(dialog)
    dialog.show()
    for pathway in data['pathways']:
        protocol = next(key for key, guide in data['protocols'].items() if guide['pathway'] == pathway)
        route = next(row for row in data['protocol_routes'] if row['protocol'] == protocol)
        dialog._preferred = {}
        dialog._cascade(-1)
        for key in dialog._levels:
            _choose(dialog, key, route[key])
        assert dialog._selected_protocol == protocol
        assert all(step['module'] in dialog._diagram.nodes for step in data['protocols'][protocol]['steps'])
        assert data['protocols'][protocol]['title'] in dialog._text.toPlainText()


@pytest.mark.parametrize('source,context,expected', [
    ('/selected/figure.pdf', {}, {'data': 'images', 'goal': 'assays', 'details': 'pdf'}),
    ('/selected/plate/merged/field.npy', {'generated_outputs': {'artifacts': [{'kind': 'merged'}]}},
     {'data': 'images_masks', 'goal': 'quantification', 'stage': 'prepared', 'details': 'merged_project'}),
    ('/selected/measurements.db', {'database_schemas': [{'objects': True, 'numeric': True}]},
     {'data': 'features', 'goal': 'exploration', 'details': 'database'}),
    ('/selected/reads.fastq.gz', {}, {'data': 'fastq', 'goal': 'sequencing', 'stage': 'start'}),
])
def test_native_input_hints_use_metadata_without_scanning_files(qtbot, source, context, expected):
    dialog = sample._ProtocolBrowserDialog(source_path=source, context=context)
    qtbot.addWidget(dialog)
    for key, value in expected.items():
        assert dialog._selectors[key].currentData() == value
    if expected['data'] == 'fastq':
        assert dialog._selectors['details'].currentData() is None


def test_help_me_choose_explains_compatible_choices_without_selecting_or_opening(qtbot, protocol_data):
    dialog = sample._ProtocolBrowserDialog(data=protocol_data)
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog._help_buttons['data'].isEnabled()
    assert not dialog._help_buttons['goal'].isEnabled()
    assert not dialog._diagram_host.isVisible()
    assert dialog._help_buttons['data'].isChecked()
    text = dialog._text.toPlainText()
    assert 'Images' in text and 'Measured features' in text and 'Quantify' in text
    assert dialog._selectors['data'].currentData() is None
    _choose(dialog, 'data', 'images')
    assert dialog._help_buttons['goal'].isChecked()
    assert 'Count objects' in dialog._text.toPlainText()
    _choose(dialog, 'goal', 'quantification')
    assert dialog._help_buttons['details'].isChecked()
    assert 'Yokogawa' in dialog._text.toPlainText()
    _choose(dialog, 'details', 'yokogawa')
    assert dialog._diagram_host.isVisible()
    selected = {key: control.currentData() for key, control in dialog._selectors.items()}
    dialog._help_buttons['goal'].click()
    text = dialog._text.toPlainText()
    assert 'Count objects' in text and 'Train a model' in text
    assert {key: control.currentData() for key, control in dialog._selectors.items()} == selected
    dialog._help_buttons['goal'].click()
    assert 'Preview and run Mask.' in dialog._text.toPlainText()
    assert dialog._selected_protocol == 'count'
