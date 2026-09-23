"""Workflow diagrams show real branches and expose stable node/edge explanations."""
from __future__ import annotations

import pytest

pytest.importorskip('PySide6')
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from spacr.qt.widgets import workflow_diagram as wd
from spacr.qt.widgets.sample_project import SampleProjectDialog

pytestmark = pytest.mark.qt


def test_pipeline_rows_are_graphs_and_keep_the_independent_sequencing_branch(qtbot):
    dialog = SampleProjectDialog()
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == 'Pipeline overviews'
    assert len(dialog.diagrams) == len(dialog._entries)
    index = next(i for i, e in enumerate(dialog._entries) if e['id'] == 'pooled_screen')
    graph = dialog.diagrams[index]
    pairs = {(edge['from'], edge['to']) for edge in graph.links}
    assert ('classify_merged', 'map_barcodes') not in pairs
    assert ('classify_merged', 'regression') in pairs
    assert ('map_barcodes', 'regression') in pairs
    assert set(graph.nodes) == set(dialog._entries[index]['modules']) | {
        'input:images', 'input:external_masks', 'input:fastq'}
    for key in graph.nodes:
        prose = wd.node_description(graph.data, key)
        assert 'Inputs' in prose and 'Outputs' in prose
        for artifact in graph.data['modules'][key]['inputs'] + graph.data['modules'][key]['outputs']:
            assert graph.data['artifacts'][artifact]['title'] in prose


def test_full_network_contains_every_module_and_every_documented_handoff(qtbot):
    dialog = wd.SpacrFlowchartDialog()
    qtbot.addWidget(dialog)
    data = wd.workflow_map()
    assert set(dialog.view.nodes) == set(data['modules'])
    actual = {(e['from'], e['to']): e for e in dialog.view.links}
    for edge in data['connections']:
        found = actual[edge['from'], edge['to']]
        assert found['kind'] == 'documented'
        assert found['artifacts'] == edge['artifacts']
        assert found['handoff'] == edge['handoff']
    for source, module in data['modules'].items():
        for target, other in data['modules'].items():
            shared = set(module['outputs']) & set(other['inputs'])
            if source != target and shared:
                assert (source, target) in actual
    assert len(dialog.view.links) > len(data['connections'])


def test_keyboard_selectors_show_node_and_connection_details_without_resizing(qtbot):
    dialog = wd.SpacrFlowchartDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    size = dialog.details.size()
    bounds = {key: node.sceneBoundingRect() for key, node in dialog.view.nodes.items()}
    dialog.module_picker.setCurrentIndex(dialog.module_picker.findData('measure'))
    assert 'Measure' in dialog.details.toPlainText()
    assert 'measurements/measurements.db' in dialog.details.toPlainText()
    number = next(i for i, e in enumerate(dialog.view.edges)
                  if e.edge['from'] == 'ops' and e.edge['to'] == 'regression')
    dialog.edge_picker.setCurrentIndex(dialog.edge_picker.findData(number))
    assert 'OPS → Regression' in dialog.details.toPlainText()
    assert 'not a direct CSV handoff' in dialog.details.toPlainText()
    assert dialog.details.size() == size
    assert bounds == {key: node.sceneBoundingRect() for key, node in dialog.view.nodes.items()}
    inferred = next(e for e in dialog.view.edges if e.edge['kind'] == 'compatible')
    dialog.view.describe_edge(inferred)
    assert 'not an automatic transfer' in dialog.details.toPlainText()


def test_hovering_a_node_or_arrow_updates_the_same_fixed_text_box(qtbot):
    data = wd.workflow_map()
    data['modules'] = {k: data['modules'][k] for k in ('mask', 'measure')}
    dialog = wd.SpacrFlowchartDialog(data=data)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    graph = dialog.view
    graph.fit_diagram()
    size = dialog.details.size()
    point = graph.mapFromScene(graph.nodes['measure'].sceneBoundingRect().center())
    qtbot.mouseMove(graph.viewport(), graph.viewport().rect().topLeft())
    qtbot.mouseMove(graph.viewport(), point)
    qtbot.waitUntil(lambda: 'Measure' in dialog.details.toPlainText())
    edge = graph.edges[0]
    point = graph.mapFromScene(edge.path().pointAtPercent(.5))
    qtbot.mouseMove(graph.viewport(), point)
    qtbot.waitUntil(lambda: 'Mask → Measure' in dialog.details.toPlainText())
    qtbot.mouseClick(graph.viewport(), Qt.LeftButton, pos=point)
    assert edge.highlighted
    assert dialog.details.size() == size


def test_help_menu_opens_and_reuses_the_flowchart(qtbot):
    from spacr.qt.app import MainWindow
    window = MainWindow()
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    action = window.findChild(QAction, 'SpacrFlowchartAction')
    assert action is not None and action.text() == 'spaCR flowchart'
    action.trigger()
    first = window._spacr_flowchart
    assert first.isVisible()
    first.close()
    action.trigger()
    assert window._spacr_flowchart is first and first.isVisible()


def test_diagram_surface_uses_eighty_percent_alpha_and_keeps_text_opaque(qtbot):
    dialog = wd.DiagramDialog()
    qtbot.addWidget(dialog)
    dialog.resize(200, 100)
    dialog.show()
    qtbot.waitExposed(dialog)
    pixel = dialog.grab().toImage().pixelColor(100, 50)
    assert pixel.alpha() == pytest.approx(204, abs=1)
    assert dialog.windowOpacity() == 1


def test_columns_follow_documented_inputs_and_converge_barcodes_with_classify(qtbot):
    dialog = wd.SpacrFlowchartDialog()
    qtbot.addWidget(dialog)
    nodes = dialog.view.nodes
    core = ['mask', 'measure', 'annotate', 'classify_merged', 'regression']
    positions = [nodes[key].pos().x() for key in core]
    assert positions == sorted(set(positions))
    assert all(nodes[key].pos().y() == 0 for key in core)
    assert nodes['map_barcodes'].pos().x() == nodes['classify_merged'].pos().x()
    assert nodes['map_barcodes'].pos().y() != nodes['classify_merged'].pos().y()
    for edge in dialog.view.links:
        if edge['kind'] == 'documented':
            assert nodes[edge['from']].pos().x() < nodes[edge['to']].pos().x()
    rectangles = [node.sceneBoundingRect() for node in nodes.values()]
    for i, rectangle in enumerate(rectangles):
        assert all(not rectangle.intersects(other) for other in rectangles[i + 1:])


@pytest.mark.parametrize('compact', [False, True])
def test_independent_branches_align_instead_of_crossing(compact):
    keys = ['first_input', 'second_input', 'second_output', 'first_output']
    edges = [dict(zip(('from', 'to', 'kind'), pair + ('documented',)))
             for pair in [('first_input', 'first_output'),
                          ('second_input', 'second_output')]]
    positions = wd._positions(keys, edges, compact=compact)
    for edge in edges:
        source, target = positions[edge['from']], positions[edge['to']]
        assert source.x() < target.x()
        assert source.y() == target.y()
    assert positions['first_input'].y() != positions['second_input'].y()
    assert positions == wd._positions(keys, edges, compact=compact)
    assert keys == ['first_input', 'second_input', 'second_output', 'first_output']


def test_node_and_edge_explanations_link_their_localized_api_pages(qtbot, monkeypatch):
    from spacr.qt import i18n
    from spacr.qt.help_search import api_url
    monkeypatch.setenv(i18n.ENV_LANGUAGE, 'fr')
    dialog = wd.SpacrFlowchartDialog()
    qtbot.addWidget(dialog)
    dialog.view.describe_node('mask')
    assert api_url('spacr.core') in dialog.details.toHtml()
    edge = next(e for e in dialog.view.edges
                if e.edge['from'] == 'mask' and e.edge['to'] == 'measure')
    dialog.view.describe_edge(edge)
    html = dialog.details.toHtml()
    assert api_url('spacr.core') in html
    assert api_url('spacr.measure') in html
    assert '?lang=fr' in html
    assert dialog.details.openExternalLinks()
