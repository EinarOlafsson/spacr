"""Actual workflow-node title painting keeps complete localized text in its box."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRectF, QSettings
from PySide6.QtGui import QImage, QPainter

from spacr.qt import i18n, preferences, theme
from spacr.qt.widgets import workflow_diagram as workflow


class RecordingPainter(QPainter):
    def __init__(self, image):
        super().__init__(image)
        self.text_calls = []

    def drawText(self, *args):
        result = super().drawText(*args)
        if len(args) == 3 and isinstance(args[0], QRectF):
            box, flags, text = args
            self.text_calls.append((QRectF(box),
                                    self.boundingRect(box, flags, text),
                                    text, self.font().pixelSize()))
        return result


@pytest.fixture
def private_style(qapp, tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "workflow-font.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: settings)
    previous = qapp.styleSheet()
    qapp.setStyleSheet(theme.stylesheet("dark"))
    yield
    qapp.setStyleSheet(previous)


def _title_paint(node):
    image = QImage(280, 200, QImage.Format_ARGB32)
    image.fill(0)
    painter = RecordingPainter(image)
    try:
        node.paint(painter, None)
        return painter.text_calls[0]
    finally:
        painter.end()


@pytest.mark.parametrize("language", ["en", "de", "is", "pt"])
@pytest.mark.parametrize("compact", [False, True])
def test_current_translated_workflow_titles_fit_their_painted_box(
        private_style, qtbot, language, compact):
    preferences.set_language(language)
    data = workflow.workflow_map()
    keys = ["cellpose_masks", "simulation", "ml_analyze", "host_pathogen"]
    view = workflow.WorkflowView(data, keys=keys, compact=compact)
    qtbot.addWidget(view)
    before_nodes = {key: node.sceneBoundingRect() for key, node in view.nodes.items()}
    before_edges = [edge.path() for edge in view.edges]
    transform = view.transform()
    for key, node in view.nodes.items():
        box, measured, text, pixels = _title_paint(node)
        assert text == i18n.tr(data["modules"][key]["name"])
        assert box.adjusted(-.01, -.01, .01, .01).contains(measured), (
            language, key, text, box, measured, pixels)
        assert box == QRectF(14, 9, 232, 66 if compact else 40)
        assert node.sceneBoundingRect() == before_nodes[key]
        assert text in workflow.node_description(data, key)
    assert [edge.path() for edge in view.edges] == before_edges
    assert view.transform() == transform


@pytest.mark.parametrize("title,compact,expected_pixels", [
    ("QC", False, 17), ("QC", True, 24), ("", False, 17), ("", True, 24)])
def test_short_and_empty_titles_keep_the_original_font(
        private_style, qtbot, title, compact, expected_pixels):
    data = {"modules": {"qc_dashboard": {"name": title}}, "artifacts": {}}
    view = workflow.WorkflowView(data, compact=compact)
    qtbot.addWidget(view)
    box, measured, painted, pixels = _title_paint(view.nodes["qc_dashboard"])
    assert painted == title
    assert pixels == expected_pixels
    if title:
        assert box.contains(measured)
