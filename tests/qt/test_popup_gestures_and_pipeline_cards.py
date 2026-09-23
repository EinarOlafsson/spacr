"""Popup gestures, arrow joins and persistent pipeline explanations."""
import pytest

pytest.importorskip('PySide6')
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QLabel, QSlider, QScrollArea, QVBoxLayout
from spacr.qt.widgets import glass
from spacr.qt.widgets.sample_project import SampleProjectDialog
from spacr.qt.widgets.workflow_diagram import DiagramDialog, WorkflowView, workflow_map

pytestmark = pytest.mark.qt


def mouse(widget, kind, local, global_point, buttons):
    event = QMouseEvent(kind, QPointF(local), QPointF(global_point),
                       Qt.LeftButton if kind != QEvent.MouseMove else Qt.NoButton,
                       buttons, Qt.NoModifier)
    QApplication.sendEvent(widget, event)


def test_passive_child_drag_moves_popup_but_slider_owns_its_gesture(qtbot):
    dialog = DiagramDialog()
    qtbot.addWidget(dialog)
    layout = QVBoxLayout(dialog)
    label = QLabel('Drag this explanation')
    slider = QSlider(Qt.Horizontal)
    scroll = QScrollArea()
    scroll.setWidget(label)
    scroll.setWidgetResizable(True)
    layout.addWidget(scroll)
    layout.addWidget(slider)
    dialog.resize(400, 200)
    dialog.move(100, 100)
    dialog.show()
    qtbot.waitExposed(dialog)
    before = dialog.pos()
    local = label.rect().center()
    start = label.mapToGlobal(local)
    mouse(label, QEvent.MouseButtonPress, local, start, Qt.LeftButton)
    mouse(label, QEvent.MouseMove, local, start + QPoint(50, 30), Qt.LeftButton)
    mouse(label, QEvent.MouseButtonRelease, local, start + QPoint(50, 30), Qt.NoButton)
    assert dialog.pos() == before + QPoint(50, 30)
    before = dialog.pos()
    qtbot.mouseClick(slider, Qt.LeftButton, pos=QPoint(slider.width() - 20, slider.height() // 2))
    assert slider.value() > 0
    assert dialog.pos() == before
    mouse(label, QEvent.MouseMove, local, start + QPoint(100, 70), Qt.NoButton)
    assert dialog.pos() == before


@pytest.mark.parametrize('edges', [Qt.LeftEdge, Qt.TopEdge, Qt.LeftEdge | Qt.TopEdge, Qt.RightEdge | Qt.TopEdge])
def test_resize_pointer_contains_blue_pixels_and_center_hotspot(qapp, edges):
    cursor = glass._blue_resize_cursor(edges)
    pixels = cursor.pixmap().toImage()
    assert cursor.hotSpot() == QPoint(16, 16)
    assert any((lambda c: c.alpha() > 100 and c.blue() > 220 and c.red() < 60)(pixels.pixelColor(x, y))
               for x in range(pixels.width()) for y in range(pixels.height()))


def test_sloped_edges_meet_arrow_base_at_center_without_entering_head(qtbot):
    graph = WorkflowView(workflow_map(), compact=True)
    qtbot.addWidget(graph)
    for edge in graph.edges:
        endpoint = edge.path().currentPosition()
        head = edge.arrowhead
        assert endpoint == (head[1] + head[2]) / 2
        assert endpoint.y() == edge.end.y()
        assert endpoint.x() < edge.end.x()
        previous = edge.path().elementAt(edge.path().elementCount() - 2)
        assert previous.y == endpoint.y()
        assert previous.x < endpoint.x()
        assert edge.shape().contains(edge.end - QPointF(4, 0))


def test_pipeline_keeps_all_cards_and_selection_changes_only_border(qtbot):
    dialog = SampleProjectDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    graph = dialog.diagrams[0]
    panel = dialog.details
    expected = set(graph.nodes) | {edge['from'] + '→' + edge['to'] for edge in graph.links}
    assert set(panel.cards) == expected
    original = panel.toPlainText()
    positions = {key: card.geometry() for key, card in panel.cards.items()}
    scroll = panel.verticalScrollBar().value()
    graph.describe_node('measure')
    assert panel.cards['measure'].property('selected')
    edge = next(e for e in graph.edges if e.edge['from'] == 'mask' and e.edge['to'] == 'measure')
    graph.describe_edge(edge)
    assert panel.cards['mask→measure'].property('selected')
    assert not panel.cards['measure'].property('selected')
    assert panel.toPlainText() == original
    assert panel.verticalScrollBar().value() == scroll
    assert positions == {key: card.geometry() for key, card in panel.cards.items()}
    assert panel.width() == dialog.list.width()
    sizes = dialog.splitter.sizes()
    dialog.splitter.moveSplitter(sizes[0] - 80, 1)
    assert dialog.details.height() > sizes[1]
    for key in graph.keys:
        if key.startswith('input:'):
            continue
        html = panel.cards[key].findChild(QLabel).text()
        assert html.endswith('</a>')
        assert 'API</a>' in html


def test_every_pipeline_has_external_inputs_and_assays_are_independent(qtbot):
    dialog = SampleProjectDialog()
    qtbot.addWidget(dialog)
    for entry, graph in zip(dialog._entries, dialog.diagrams):
        assert entry['inputs'] and any(key.startswith('input:') for key in graph.nodes)
        assert not entry['modules'][0].startswith('input:')
    i = next(i for i, e in enumerate(dialog._entries) if e['id'] == 'parasite_assay')
    graph = dialog.diagrams[i]
    pairs = {(edge['from'], edge['to']) for edge in graph.links}
    assert {('measure', assay) for assay in ('recruitment', 'invasion', 'replication')} <= pairs
    assert ('input:plaque_images', 'analyze_plaques') in pairs
    assert ('measure', 'analyze_plaques') not in pairs
    pooled = dialog.diagrams[0]
    pairs = {(edge['from'], edge['to']) for edge in pooled.links}
    assert {('gate_editor', 'classify_merged'), ('umap', 'classify_merged'), ('foreign', 'measure')} <= pairs
    assert pooled.nodes['map_barcodes'].pos().x() == pooled.nodes['classify_merged'].pos().x()
