"""Spatial path buttons choose the right dialog and retain cancelled input."""
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QFormLayout, QToolButton, QWidget

from spacr.qt.screens.map_barcodes import _SpatialTranscriptomicsPanel
from spacr.qt.i18n import tr

pytestmark = pytest.mark.qt


@pytest.mark.parametrize("kind", ["folder", "save", "open"])
@pytest.mark.parametrize("cancelled", [False, True], ids=["selected", "cancelled"])
def test_spatial_path_button_uses_its_picker_and_preserves_cancelled_input(
        qtbot, monkeypatch, tmp_path, kind, cancelled):
    host = QWidget()
    qtbot.addWidget(host)
    form = QFormLayout(host)
    caption = "Spatial input"
    edit = _SpatialTranscriptomicsPanel._path_row(
        host, form, caption, "Choose a spatial input.",
        folder=kind == "folder", save=kind == "save")
    previous = str(tmp_path / "previous input")
    chosen = "" if cancelled else str(tmp_path / "chosen input")
    edit.setText(previous)
    calls = []

    def picker(picker_kind):
        def choose(parent, title, current):
            calls.append((picker_kind, parent, title, current))
            return chosen if picker_kind == "folder" else (chosen, "")
        return choose

    for picker_kind, method in (
            ("folder", "getExistingDirectory"),
            ("save", "getSaveFileName"),
            ("open", "getOpenFileName")):
        monkeypatch.setattr(QFileDialog, method, picker(picker_kind))

    button = edit.parentWidget().findChild(QToolButton)
    assert button is not None
    host.show()
    qtbot.mouseClick(button, Qt.LeftButton)

    assert calls == [(kind, host, tr(caption), previous)]
    assert edit.text() == (previous if cancelled else chosen)
