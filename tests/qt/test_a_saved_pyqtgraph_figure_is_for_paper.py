"""Instruction 150, the pyqtgraph half.

    "when a graph is saved and the user is in dark mode white elements are
     changed to black for saving (text lines, etc)."

A PNG saved from a dark session with a transparent background looks fine in a
file manager's dark preview and DISAPPEARS when it is pasted into a
manuscript. The user finds out at the point of writing the paper, which is the
most expensive possible moment.

The design in one line, and it is easy to get wrong: THE CHROME FLIPS, THE
DATA DOES NOT. A blanket white-to-black would turn a white data point black,
which on a volcano is the colour of "not a hit" — it would change what the
figure says.

Measured on the PIXELS, as the instruction requires, not judged by eye.
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt

from PySide6.QtGui import QImage                                 # noqa: E402

from spacr.figure_style import saved_figure_appearance           # noqa: E402
from spacr.qt.widgets.fast_plots import FastPlot                 # noqa: E402

DATA_COLOUR = "#C4441C"


def _dark_plot(qtbot):
    plot = FastPlot()
    qtbot.addWidget(plot)
    plot.restyle(background="#0d0e10", foreground="#FFFFFF")
    rng = np.random.default_rng(0)
    plot.plot.plot(np.arange(40), rng.normal(size=40), pen=None, symbol="o",
                   symbolBrush=DATA_COLOUR, symbolSize=9)
    plot.plot.setTitle("a title", color="#FFFFFF")
    return plot


def _colours(path) -> Counter:
    image = QImage(str(path)).convertToFormat(QImage.Format_RGB32)
    return Counter(image.pixelColor(x, y).name()
                   for y in range(0, image.height(), 2)
                   for x in range(0, image.width(), 2))


def _axis_ink(plot) -> str:
    pen = plot.plot.getAxis("bottom").pen()
    return pen.color().name() if pen is not None else ""


# -- what lands in the file --------------------------------------------------

def test_a_figure_saved_from_a_dark_session_has_a_light_page(qtbot, tmp_path):
    out = tmp_path / "p.png"
    _dark_plot(qtbot).export(str(out))

    colours = _colours(out)
    assert colours.most_common(1)[0][0] == "#ffffff"


def test_the_chrome_is_dark_in_the_file(qtbot, tmp_path):
    out = tmp_path / "p.png"
    _dark_plot(qtbot).export(str(out))

    colours = _colours(out)
    dark = sum(n for name, n in colours.items() if int(name[1:3], 16) < 90)
    assert dark > 0, "nothing dark was written — the axes are still white"


def test_the_data_keeps_the_colour_it_had_on_screen(qtbot, tmp_path):
    """It carries the CLAIM; inverting it changes what the figure says."""
    out = tmp_path / "p.png"
    _dark_plot(qtbot).export(str(out))

    assert any(name.lower().startswith("#c4") for name in _colours(out))


def test_the_rule_reaches_every_format_and_not_two_out_of_three(qtbot, tmp_path):
    """PDF and SVG paint the scene; PNG goes through pyqtgraph's exporter."""
    import spacr.figure_style as style

    asked = []
    original = style.export_colour
    style.export_colour = lambda c, kind, look=None: (
        asked.append(kind), original(c, kind, look))[1]
    try:
        plot = _dark_plot(qtbot)
        for suffix in ("png", "pdf", "svg"):
            asked.clear()
            plot.export(str(tmp_path / f"p.{suffix}"))
            assert "chrome" in asked, f"{suffix} did not go through the rule"
    finally:
        style.export_colour = original


def test_the_data_is_never_even_offered_to_the_rule(qtbot, tmp_path):
    """Not passed with a different kind. Not passed."""
    import spacr.figure_style as style

    asked = []
    original = style.export_colour
    style.export_colour = lambda c, kind, look=None: (
        asked.append(kind), original(c, kind, look))[1]
    try:
        _dark_plot(qtbot).export(str(tmp_path / "p.png"))
    finally:
        style.export_colour = original
    assert "data" not in asked


# -- and nothing on screen moves ---------------------------------------------

@pytest.mark.parametrize("suffix", ["png", "pdf", "svg"])
def test_the_plot_on_screen_is_what_it_was_before_the_save(qtbot, tmp_path,
                                                           suffix):
    """A user watching a plot while it saves must not see it flash."""
    plot = _dark_plot(qtbot)
    before = _axis_ink(plot)

    plot.export(str(tmp_path / f"p.{suffix}"))
    assert _axis_ink(plot) == before


def test_a_save_that_fails_still_puts_the_plot_back(qtbot, monkeypatch,
                                                    tmp_path):
    plot = _dark_plot(qtbot)
    before = _axis_ink(plot)

    def explode(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(plot, "_write_export", explode)
    with pytest.raises(OSError):
        plot.export(str(tmp_path / "p.png"))
    assert _axis_ink(plot) == before


def test_the_gallery_tile_is_the_screen_version_not_the_print_one(qtbot):
    """139 C: the tile and the file differ on purpose."""
    plot = _dark_plot(qtbot)
    before = _axis_ink(plot)

    assert plot.snapshot(240) is not None
    assert _axis_ink(plot) == before


# -- the look itself ---------------------------------------------------------

def test_print_is_the_default_and_names_a_light_page():
    look = saved_figure_appearance()
    assert look.mode == "print"
    assert look.ground == "#FFFFFF"
    assert look.flip is True


def test_as_on_screen_changes_nothing_at_all():
    look = saved_figure_appearance("screen")
    assert look.flip is False
    assert look.ground is None


def test_transparent_keeps_the_old_behaviour_for_compositing():
    look = saved_figure_appearance("transparent")
    assert look.transparent is True
    assert look.ground is None
