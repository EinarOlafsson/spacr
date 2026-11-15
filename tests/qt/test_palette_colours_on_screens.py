"""Where three screens spend the active palette: a well highlight, a run status, a canvas surface.

* the plate grid outlines the selected well in the accent colour only when
  that well is on the plate;
* Run History paints each status in its palette colour, and a cancelled or
  unknown status in none (the table's ordinary text), never in the error red;
* the Image Scatter canvas gets a page surface and a soft border of its own.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor, QPixmap  # noqa: E402

from spacr.qt import theme  # noqa: E402
from spacr.qt.theme import active_palette  # noqa: E402

pytestmark = pytest.mark.qt


def test_plate_grid_outlines_only_a_selected_well_on_the_plate(subtests, qtbot):
    """A selection off the 1x2 plate draws no accent; an on-plate well does."""
    from spacr.qt.screens import plate_view

    def _accent_pixels(selection):
        widget = plate_view.PlateGridWidget()
        qtbot.addWidget(widget)
        widget.resize(400, 300)
        frame = pd.DataFrame({"row_index": [1, 1], "column_index": [1, 2],
                              "n": [5, 7], "value": [float("nan"), 2.5]})
        frame.attrs["n_rows"] = 1
        frame.attrs["n_cols"] = 2
        widget.set_plate(frame, vmin=0.0, vmax=5.0)
        if selection is not None:
            widget.select(*selection)
            assert widget.selected_well() == selection
        pixmap = QPixmap(widget.size())
        widget.render(pixmap)
        image = pixmap.toImage()
        accent = QColor(active_palette()["accent"]).rgb()
        return sum(image.pixel(x, y) == accent
                   for x in range(0, image.width(), 2)
                   for y in range(0, image.height(), 2))

    unselected = _accent_pixels(None)
    for case_id, selection, outlined in [
            ("off the end of the plate", (9, 9), False),
            ("row zero is off the plate", (0, 1), False),
            ("column past the plate", (1, 3), False),
            ("the first well", (1, 1), True),
            ("the second well", (1, 2), True)]:
        with subtests.test(msg=case_id):
            pixels = _accent_pixels(selection)
            if outlined:
                assert pixels > unselected
            else:
                assert pixels == unselected


def _record(run_id, status):
    """One ``search_runs`` record, shaped as run_journal emits it."""
    return {
        "dir": Path("/nonexistent/spacr-runs") / run_id,
        "run_id": run_id, "app_key": "classify", "status": status,
        "start_utc": "2026-08-29T10:11:12+00:00", "end_utc": "",
        "elapsed_s": 1.5,
        "performance": {"wall_s": 1.5, "process_cpu_s": 0.9,
                        "input_files": 1, "input_bytes": 2048,
                        "output_files": 1, "output_bytes": 4096},
        "settings": {}, "inputs": {}, "outputs": {}, "models": {},
        "warnings": [], "failure": "", "environment": {},
    }


def test_run_history_paints_each_status_in_its_own_colour(
        subtests, qtbot, qt_theme_applied, monkeypatch):
    """success/failed/running/interrupted/corrupt get their colour; cancelled and unknown get none."""
    from spacr.qt.screens.run_history import RunHistoryScreen

    statuses = ["success", "failed", "running", "interrupted", "corrupt",
                "cancelled", "something-new"]
    monkeypatch.setattr(
        "spacr.qt.screens.run_history.search_runs",
        lambda: [_record(f"r-{status}", status) for status in statuses])
    screen = RunHistoryScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.refresh()
    cells = {screen._table.item(row, 2).text(): screen._table.item(row, 2)
             for row in range(screen._table.rowCount())}
    assert sorted(cells) == sorted(statuses)

    palette = active_palette()
    expected = {"success": palette["success"], "failed": palette["error"],
                "running": palette["warning"],
                "interrupted": palette["error"], "corrupt": palette["error"],
                "cancelled": None, "something-new": None}
    for status, colour in expected.items():
        with subtests.test(msg=status):
            if colour is None:
                assert cells[status].data(Qt.ForegroundRole) is None
            else:
                assert (cells[status].foreground().color().name()
                        == QColor(colour).name())


def test_image_scatter_canvas_gets_a_surface_of_its_own(subtests, qapp):
    """The canvas rule is a page surface at the opacity, a soft border and the md radius."""
    from spacr.qt.screens import image_scatter as isc

    for case_id, palette, opacity in [
            ("active palette, half opacity", active_palette(), 0.5),
            ("dark palette, opaque", dict(theme.DARK_PALETTE, theme="dark"),
             None),
            ("light palette, quarter opacity",
             dict(theme.LIGHT_PALETTE, theme="light"), 0.25)]:
        with subtests.test(msg=case_id):
            surface = theme.block_surface("surface", palette.get("theme"),
                                          opacity)
            assert isc._image_scatter_qss(palette, opacity) == (
                f"\nQFrame#{isc.CANVAS_OBJECT} {{\n"
                f"    background: {surface};\n"
                f"    border: 1px solid {palette['border_soft']};\n"
                f"    border-radius: {theme.RADIUS['md']}px;\n"
                "}\n")
            assert isc.CANVAS_OBJECT == "ImageScatterCanvas"
