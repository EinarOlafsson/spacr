"""The sweep panel when the inputs, the result or the picture is not there.

``tests/test_the_sweep_button.py`` and its neighbours run the real engine over
a small synthetic screen and read the answers off the table. What is driven
here is what the panel does when the run cannot happen or produced nothing:
each of those is a sentence in the status line, and a panel that says nothing
looks like a broken button.

The engine is real throughout -- the result the pictures are drawn from comes
out of an actual sweep -- so the pictures are drawn by matplotlib's Agg canvas
exactly as the panel draws them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog


@pytest.fixture()
def inputs():
    """A small screen: guide A moves ``real``, nothing moves ``noise``."""
    rng = np.random.default_rng(0)
    n = 80
    a = rng.random(n)
    cells = pd.DataFrame({
        "plateID": ["plate1"] * 40 + ["plate2"] * 40,
        "rowID": [f"r{i}" for i in range(n)],
        "columnID": ["c1"] * n,
        "real": a * 3.0 + rng.normal(0, 0.2, n),
        "noise": rng.normal(0, 1, n),
        "pred": rng.random(n),
    })
    counts = pd.DataFrame(
        [{"prc": f"plate{1 + i // 40}_r{i}_c1", "grna": g,
          "fraction": (a[i] if g == "A" else 1 - a[i])}
         for i in range(n) for g in ("A", "B")])
    return cells, counts


@pytest.fixture()
def panel(qtbot, inputs):
    from spacr.qt.widgets.sweep_panel import SweepPanel

    cells, counts = inputs
    widget = SweepPanel(lambda: cells, lambda: counts, threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture()
def swept(panel):
    """A panel that has actually run, so there is a result to draw."""
    assert panel.start() is True
    assert panel._result is not None
    return panel


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_inputs_that_cannot_be_read_are_named_not_swallowed(qtbot):
    from spacr.qt.widgets.sweep_panel import SweepPanel

    def explode():
        raise ValueError("no merged database is open")

    widget = SweepPanel(explode, lambda: pd.DataFrame({"a": [1]}),
                        threaded=False)
    qtbot.addWidget(widget)
    assert widget.start() is False
    assert "Could not read the inputs: no merged database is open" in \
        widget.status.text()


def test_unreadable_scores_do_not_stop_the_sweep(qtbot, inputs):
    """Circularity is a column the answer can do without; the sweep is not."""
    from spacr.qt.widgets.sweep_panel import SweepPanel

    cells, counts = inputs

    def explode():
        raise OSError("the score CSVs are on an unmounted drive")

    widget = SweepPanel(lambda: cells, lambda: counts, threaded=False,
                        scores_provider=explode)
    qtbot.addWidget(widget)
    assert widget.start() is True
    assert widget._result is not None


def test_a_sweep_that_returns_nothing_says_so(panel):
    panel._done(None)
    assert panel.status.text() == "The sweep returned nothing."
    assert panel.run_button.isEnabled()
    assert not panel.save_button.isEnabled()


def test_a_failed_sweep_says_why_and_gives_the_button_back(panel):
    panel.run_button.setEnabled(False)
    panel._failed("the worker died")
    assert "The sweep did not finish: the worker died" in panel.status.text()
    assert panel.run_button.isEnabled()


def test_before_a_run_there_is_no_table_and_no_picture(panel):
    assert panel.rows().empty
    assert panel.figure() is None
    assert panel.show_picture() is None
    assert "Run the sweep first" in panel.status.text()
    assert panel.save() == ""


# ---------------------------------------------------------------------------
# Reading the table back
# ---------------------------------------------------------------------------

def test_a_long_table_says_how_much_of_it_is_on_screen(swept, monkeypatch):
    """2,000 rows is the page; the rest are in the file, and it says so."""
    real = swept.rows()
    assert not real.empty
    many = pd.concat([real] * (2001 // max(1, len(real)) + 1),
                     ignore_index=True)
    monkeypatch.setattr(type(swept), "rows", lambda self: many)
    swept._refill()
    assert swept.table.rowCount() == 2000
    assert f"Showing the first 2,000 of {len(many):,}" in swept.status.text()


def test_the_selected_row_is_the_gene_the_pictures_are_about(swept):
    assert swept.table.rowCount() > 0
    swept.table.selectRow(0)
    chosen = swept.table.item(0, 1).text().strip()
    assert swept.selected_gene() == chosen


def test_with_no_survivors_there_is_no_gene_to_draw(swept):
    """One gene's fingerprint is the one picture that needs a subject."""
    import dataclasses

    swept.table.clearSelection()
    swept._result = dataclasses.replace(swept._result,
                                        table=swept._result.table.iloc[:0])
    assert swept.selected_gene() is None
    assert swept.figure(kind="profile") is None


def test_the_picture_that_cannot_be_drawn_is_a_sentence_not_an_empty_window(
        swept, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("matplotlib said no")

    monkeypatch.setattr(type(swept), "figure", explode)
    assert swept.show_picture() is None
    assert "That picture could not be drawn: matplotlib said no" in \
        swept.status.text()


def test_nothing_to_draw_names_the_picture_and_the_q_filter(swept,
                                                            monkeypatch):
    monkeypatch.setattr(type(swept), "figure",
                        lambda self, path=None, kind=None: None)
    assert swept.show_picture() is None
    assert "Nothing to draw for" in swept.status.text()
    assert "Loosen the q filter" in swept.status.text()


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def test_a_cancelled_save_writes_nothing(swept, monkeypatch, tmp_path):
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    assert swept.save() == ""
    assert list(tmp_path.iterdir()) == []


def test_saving_writes_the_whole_table_and_the_picture_beside_it(swept,
                                                                 monkeypatch,
                                                                 tmp_path):
    """The file is every row, not the page the table is showing."""
    target = tmp_path / "sweep.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    assert swept.save() == str(target)

    written = pd.read_csv(target)
    assert len(written) == len(swept._result.table)
    assert len(written) >= swept.table.rowCount()
    assert (tmp_path / "sweep.png").is_file()
    assert f"Saved {len(written):,} rows" in swept.status.text()


def test_a_figure_that_will_not_render_does_not_lose_the_csv(swept,
                                                             monkeypatch,
                                                             tmp_path):
    target = tmp_path / "sweep.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    monkeypatch.setattr(type(swept), "figure",
                        lambda self, path=None, kind=None: (
                            _ for _ in ()).throw(RuntimeError("no Agg")))
    assert swept.save() == str(target)
    assert target.is_file()
    assert not (tmp_path / "sweep.png").exists()
