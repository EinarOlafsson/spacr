"""QC dashboard paths that a well-themed, well-built screen never takes.

The transparency call that decorates the card column, a redraw over a layout
holding something that is not a widget, and a status line whose style cannot
be resolved. None of them may cost the user a verdict.

Offscreen, CPU-only, offline.
"""
from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import qc_dashboard as qc_module           # noqa: E402

pytestmark = pytest.mark.qt


def _project(tmp_path):
    """A minimal project with one measurements table the reader can open."""
    folder = tmp_path / "measurements"
    folder.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(folder / "measurements.db"))
    try:
        connection.execute(
            "CREATE TABLE cell (object_label INTEGER, cell_area REAL, "
            "measurement_ndim INTEGER, measurement_units TEXT)")
        connection.execute("INSERT INTO cell VALUES (1, 120.0, 2, 'px')")
        connection.commit()
    finally:
        connection.close()
    return str(tmp_path)


# ---------------------------------------------------------------------------
# The decoration that cannot be applied
# ---------------------------------------------------------------------------

def test_a_scroll_area_that_cannot_be_made_transparent_still_shows_verdicts(
        qtbot, monkeypatch, tmp_path):
    """Losing the transparency call costs a background, not the dashboard."""
    from PySide6.QtWidgets import QScrollArea

    from spacr.qt import theme

    real = theme.make_transparent

    def _refuse_scroll_areas(widget, *args, **kwargs):
        """Fail only for the card column, the way a themeless area would."""
        if isinstance(widget, QScrollArea):
            raise RuntimeError("no palette to make this transparent against")
        return real(widget, *args, **kwargs)

    monkeypatch.setattr(theme, "make_transparent", _refuse_scroll_areas)

    screen = qc_module.QCDashboardScreen(
        threaded=False, src=_project(tmp_path))
    qtbot.addWidget(screen)

    dashboard = screen.dashboard()
    assert dashboard is not None
    assert [card.key for card in dashboard.cards] == [
        "segmentation", "units", "leakage", "plate", "agreement"]
    assert "Measurement units" in screen.visible_text()


# ---------------------------------------------------------------------------
# Redrawing over something that is not a card
# ---------------------------------------------------------------------------

def test_a_redraw_clears_the_column_even_of_items_that_hold_no_widget(
        qtbot, tmp_path):
    """A spacer in the card column is discarded like everything else.

    A stretch is a layout item that holds no widget. Clearing the column as
    though every item held one raises part-way through, and the read that
    raises is reported as a failed one: the cards are taken off the screen
    and none go back, leaving the user an error where a verdict was.
    """
    screen = qc_module.QCDashboardScreen(
        threaded=False, src=_project(tmp_path))
    qtbot.addWidget(screen)
    before = screen.visible_text()
    assert screen._card_labels, "there are cards to be cleared"
    screen._cards_layout.addStretch(1)
    padded = screen._cards_layout.count()
    assert screen._cards_layout.itemAt(padded - 1).widget() is None, (
        "the stretch is the item that holds no widget")

    started = screen.refresh(force=True)

    assert started is True, "the redraw ran to the end without raising"
    assert screen._status.property("spacrError") != "true"
    assert screen._cards_layout.count() < padded, (
        "the stretch was taken out with the cards")
    assert screen._card_labels, "and the cards were drawn again"
    assert screen.visible_text() == before
    assert all(screen._cards_layout.itemAt(i).widget() is not None
               for i in range(screen._cards_layout.count()))


# ---------------------------------------------------------------------------
# A status line with no style
# ---------------------------------------------------------------------------

def test_a_status_line_with_no_style_still_reports_the_error(
        qtbot, monkeypatch, tmp_path):
    """When the style cannot be resolved the message and flag are still set."""
    screen = qc_module.QCDashboardScreen(threaded=False)
    qtbot.addWidget(screen)
    monkeypatch.setattr(type(screen._status), "style",
                        lambda self: None, raising=False)

    screen.set_source(str(tmp_path / "not-a-folder"))

    assert screen.status_text()
    assert screen._status.property("spacrError") == "true"
    assert screen.dashboard() is None
