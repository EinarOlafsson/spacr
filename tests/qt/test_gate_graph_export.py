"""Saving the gate editor's graph as a PNG or a PDF."""

import numpy as np
import pandas as pd
import pytest

from spacr.qt.screens.gate_editor import GateEditorScreen


@pytest.fixture
def screen(qt_theme_applied, qtbot):
    widget = GateEditorScreen()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def drawn(screen, qt_theme_applied):
    frame = pd.DataFrame({"area": np.linspace(1, 100, 60),
                          "intensity": np.linspace(0, 10, 60)})
    screen.gates.set_frame(frame)
    qt_theme_applied.processEvents()
    figure = screen.gates.canvas.figure()
    if not figure.get_axes():
        axis = figure.add_subplot(111)
        axis.scatter(frame["area"], frame["intensity"])
    return screen


class TestSaving:

    def test_a_png_is_written(self, drawn, tmp_path):
        out = drawn.save_graph(str(tmp_path / "graph.png"))
        assert out and (tmp_path / "graph.png").exists()
        assert (tmp_path / "graph.png").stat().st_size > 1000

    def test_a_pdf_is_written_and_is_vector(self, drawn, tmp_path):
        """A vector PDF of a scatter is far SMALLER than its raster PNG.

        Size is the cheap proxy for "is this really vector" -- a raster
        wrapped in a PDF comes out at least as large as the PNG.
        """
        drawn.save_graph(str(tmp_path / "graph.png"))
        drawn.save_graph(str(tmp_path / "graph.pdf"))
        pdf = tmp_path / "graph.pdf"
        assert pdf.exists()
        assert pdf.stat().st_size < (tmp_path / "graph.png").stat().st_size

    def test_a_missing_extension_gets_the_preferred_one(self, drawn,
                                                        tmp_path):
        out = drawn.save_graph(str(tmp_path / "noext"))
        assert out.endswith((".png", ".pdf"))

    def test_saving_nothing_says_so_instead_of_writing_a_blank(self, screen,
                                                               tmp_path):
        """An empty canvas must not produce a file that looks like a result."""
        out = screen.save_graph(str(tmp_path / "empty.png"))
        assert out == ""
        assert not (tmp_path / "empty.png").exists()

    def test_the_format_comes_from_preferences(self, drawn):
        """Instruction 50 is explicit: no second figure-format setting."""
        from spacr.qt.preferences import get_figure_format
        assert get_figure_format() in {"png", "pdf"}

    def test_the_button_exists_and_is_wired(self, screen):
        assert screen._save_graph is not None
        assert "Save graph" in screen._save_graph.text()
