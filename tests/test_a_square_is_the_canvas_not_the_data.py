""""Square" has to mean the FIGURE, and it has to reach the file.

Instruction 147 B.

    "aspect ration should be in terms of the graph dimentions not the within
     graph dimentions. it is to be able to save the graph as a square but now
     it just changes the graph internally. instead of aspect ratio call it
     what you want but there should be an option to make the graph a perfect
     square."

`FastPlot.set_aspect_ratio` calls `viewBox.setAspectLocked(True, ratio=...)`,
which ties ONE Y UNIT to n X UNITS. That is a statement about the DATA. It is
exactly right for a Q-Q, whose 45-degree diagonal means nothing unless the
axes share a scale, and it is not what "save the graph as a square" asks for
-- nor what its old name, "Aspect ratio", suggested to anybody.

Both exist now and are named apart:

    Shape: square / wide / tall / free        the CANVAS, what gets exported
    Lock axis scales (1 y unit = n x units)   the DATA, unchanged, renamed

THE FILE IS MEASURED HERE, not the screen. A square on screen that is written
out at the widget's accidental proportions has not done the job, so each of
the three export paths is exercised and the page it produced is read back off
the disk: the PNG's pixels, the SVG's declared width and height, and the PDF's
own MediaBox.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


def _frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "feature": [f"fraction:grna[g{i}_1]" for i in range(n)],
        "coefficient": rng.normal(0, 0.5, n),
        "p_value": rng.uniform(0.001, 0.99, n),
    })


@pytest.fixture()
def volcano(qtbot):
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    # DELIBERATELY NOT SQUARE. A widget that already happened to be square
    # would let a control that does nothing pass every assertion below.
    plot.resize(800, 400)
    plot.show()
    qtbot.waitExposed(plot)
    plot.set_results(_frame())
    return plot


def _png_size(path):
    from PySide6.QtGui import QImage

    image = QImage(str(path))
    assert not image.isNull(), f"{path} is not readable as an image"
    return image.width(), image.height()


def _svg_size(path):
    """The width and height the SVG declares, in its own units."""
    head = open(path, encoding="utf-8").read(2000)
    width = re.search(r'width="([0-9.]+)', head)
    height = re.search(r'height="([0-9.]+)', head)
    assert width and height, head[:400]
    return float(width.group(1)), float(height.group(1))


def _pdf_page_size(path):
    """The PDF's own MediaBox, in points."""
    raw = open(path, "rb").read()
    box = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+"
                    rb"([\d.]+)\s+([\d.]+)\s*\]", raw)
    assert box, "no MediaBox in the PDF"
    left, bottom, right, top = (float(box.group(i)) for i in range(1, 5))
    return right - left, top - bottom


# --------------------------------------------------------------------------- #
#  The two controls are different things, and are named apart
# --------------------------------------------------------------------------- #

def test_the_canvas_shape_and_the_axis_lock_are_two_controls(volcano):
    """One is the page, one is the data. Conflating them is the whole report."""
    from spacr.qt.widgets.fast_plots import menu_entries, menu_groups

    entries = [action.text() for action
               in menu_entries(volcano.build_style_menu())]
    groups = menu_groups(volcano.build_style_menu())

    assert "Shape" in groups, groups
    for shape in ("square", "wide", "tall", "free"):
        assert shape in entries, entries
    assert any("Lock axis scales" in text for text in entries), entries


def test_the_shape_does_not_touch_the_data_lock(volcano):
    """Setting the page square must not silently tie the axes together: on a
    volcano, one coefficient unit is not one -log10(p) unit and pretending so
    would squash the plot into a sliver."""
    volcano.set_canvas_shape("square")

    assert volcano.aspect_ratio() is None


def test_the_data_lock_does_not_touch_the_page(volcano):
    volcano.set_aspect_ratio(1.0)

    assert volcano.canvas_shape() == "free"


def test_an_unknown_shape_is_refused_by_name(volcano):
    with pytest.raises(ValueError) as raised:
        volcano.set_canvas_shape("roundish")

    assert "square" in str(raised.value)


def test_the_menu_shows_which_shape_is_in_force(volcano):
    from spacr.qt.widgets.fast_plots import menu_entries

    volcano.set_canvas_shape("tall")

    ticked = {action.text() for action
              in menu_entries(volcano.build_style_menu())
              if action.isCheckable() and action.isChecked()}
    assert "tall" in ticked, ticked
    assert "square" not in ticked, ticked


def test_picking_a_shape_off_the_menu_applies_it(volcano):
    """Driving the entry proves it is wired, not merely populated."""
    from spacr.qt.widgets.fast_plots import menu_entries

    entry = next(a for a in menu_entries(volcano.build_style_menu())
                 if a.text() == "square")

    entry.trigger()

    assert volcano.canvas_shape() == "square"


# --------------------------------------------------------------------------- #
#  It reaches the FILE. Measured off the disk, in all three formats.
# --------------------------------------------------------------------------- #

def test_a_square_canvas_writes_a_square_png(volcano, tmp_path):
    path = tmp_path / "square.png"
    volcano.set_canvas_shape("square")

    volcano.export(str(path))

    width, height = _png_size(path)
    assert width == height, f"the PNG is {width}x{height}"


def test_a_free_canvas_still_writes_the_widgets_own_proportion(volcano,
                                                               tmp_path):
    """The default must not change. A shape nobody asked for is as wrong as a
    shape that does nothing."""
    path = tmp_path / "free.png"

    volcano.export(str(path))

    width, height = _png_size(path)
    assert width != height, f"a free canvas came out square at {width}x{height}"


def test_a_square_canvas_writes_a_square_svg(volcano, tmp_path):
    path = tmp_path / "square.svg"
    volcano.set_canvas_shape("square")

    volcano.export(str(path))

    width, height = _svg_size(path)
    assert width == pytest.approx(height, rel=0.01), f"{width} x {height}"


def test_a_square_canvas_writes_a_square_pdf_page(volcano, tmp_path):
    """The PDF's own MediaBox, read out of the bytes."""
    path = tmp_path / "square.pdf"
    volcano.set_canvas_shape("square")

    volcano.export(str(path))

    width, height = _pdf_page_size(path)
    assert width == pytest.approx(height, rel=0.01), f"{width} x {height}"


@pytest.mark.parametrize("shape,ratio", [("wide", 2 / 3), ("tall", 1.5)])
def test_wide_and_tall_reach_the_page_as_well(volcano, tmp_path, shape, ratio):
    path = tmp_path / f"{shape}.pdf"
    volcano.set_canvas_shape(shape)

    volcano.export(str(path))

    width, height = _pdf_page_size(path)
    assert height / width == pytest.approx(ratio, rel=0.01)


def test_an_explicit_page_size_still_wins_over_the_shape(volcano, tmp_path):
    """A user who typed a page in millimetres said something more specific
    than "square", and the more specific answer is the one that holds."""
    path = tmp_path / "typed.pdf"
    volcano.set_canvas_shape("square")
    volcano.set_export_size(120.0, 60.0)

    volcano.export(str(path))

    width, height = _pdf_page_size(path)
    assert height / width == pytest.approx(0.5, rel=0.01)


def test_the_shape_is_what_export_size_reports(volcano):
    """One place every export path reads, so one place the shape has to land."""
    volcano.set_canvas_shape("square")

    width_mm, height_mm = volcano.export_size()

    assert height_mm == pytest.approx(width_mm)


# --------------------------------------------------------------------------- #
#  And the screen, so the file is not letterboxed into the difference
# --------------------------------------------------------------------------- #

def test_the_plot_is_held_at_the_shape_on_screen_too(volcano, qtbot):
    volcano.set_canvas_shape("square")
    qtbot.wait(1)

    assert volcano.plot.height() == pytest.approx(volcano.plot.width(), abs=2)


def test_going_back_to_free_releases_the_plot(volcano, qtbot):
    volcano.set_canvas_shape("square")
    qtbot.wait(1)

    volcano.set_canvas_shape("free")
    volcano.resize(800, 400)
    qtbot.wait(1)

    assert volcano.plot.maximumHeight() > volcano.plot.width(), (
        "the plot is still pinned to the square it was released from")
