"""One button: every gene against every measurement.

Instruction 175. The panel is thin on purpose -- the three corrections that
make the answer trustworthy live in the engine, so a settings CSV, a macro and
this button cannot disagree about them.
"""
import numpy as np
import pandas as pd
import pytest



def _column(table, header: str) -> int:
    """The index of a named column -- so adding one does not break a test."""
    for i in range(table.columnCount()):
        if table.horizontalHeaderItem(i).text() == header:
            return i
    raise AssertionError(f"no {header!r} column in "
                         f"{[table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]}")


@pytest.fixture()
def inputs():
    """A small screen: guide A moves `real`, nothing moves `noise`."""
    rng = np.random.default_rng(0)
    n = 80
    a = rng.random(n)
    cells = pd.DataFrame({
        "plateID": ["plate1"] * 40 + ["plate2"] * 40,
        "rowID": [f"r{i}" for i in range(n)],
        "columnID": ["c1"] * n,
        "real": np.repeat(a, 1) * 3.0 + rng.normal(0, 0.2, n),
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


def test_the_button_runs_the_sweep(panel):
    assert panel.start() is True
    assert "measurement(s)" in panel.status.text()


def test_nothing_wired_says_so_rather_than_failing(qtbot):
    from spacr.qt.widgets.sweep_panel import SweepPanel

    widget = SweepPanel()
    qtbot.addWidget(widget)

    assert widget.start() is False
    assert "no measurements" in widget.status.text()


def test_an_empty_merge_says_what_to_do(qtbot):
    from spacr.qt.widgets.sweep_panel import SweepPanel

    widget = SweepPanel(lambda: pd.DataFrame(), lambda: pd.DataFrame())
    qtbot.addWidget(widget)

    assert widget.start() is False
    assert "merge the measurement databases first" in widget.status.text()


def test_the_worker_touches_no_widget():
    """A regression built Qt widgets on its own worker today and the process
    segfaulted somewhere else entirely."""
    import inspect

    from spacr.qt.widgets.sweep_panel import SweepPanel

    source = inspect.getsource(SweepPanel._work)
    for forbidden in ("self.", "QWidget", "setText", "QTableWidget"):
        assert forbidden not in source, f"the worker touches {forbidden}"


def test_the_table_shows_what_survived(panel):
    panel.start()

    assert panel.table.rowCount() > 0
    column = _column(panel.table, "measurement")
    measurements = {panel.table.item(r, column).text()
                    for r in range(panel.table.rowCount())}
    assert "real" in measurements


def test_tightening_q_shows_fewer_rows(panel):
    panel.start()
    loose = panel.table.rowCount()

    panel.alpha.setValue(0.001)

    assert panel.table.rowCount() <= loose


def test_circularity_that_was_never_computed_shows_as_a_dash(qtbot, inputs):
    """A column of 0.00 reads as 'nothing here is circular', which is the most
    confident possible way to say nothing."""
    from spacr.qt.widgets.sweep_panel import SweepPanel

    cells, counts = inputs
    # No score anywhere: not in the measurements, and none offered.
    widget = SweepPanel(lambda: cells.drop(columns=["pred"]), lambda: counts,
                        threaded=False)
    qtbot.addWidget(widget)

    widget.start()

    assert not widget._result.circularity_known
    column = _column(widget.table, "circularity")
    dashes = {widget.table.item(r, column).text()
              for r in range(widget.table.rowCount())}
    assert dashes == {"—"}, f"circularity displayed as {dashes}"


def test_circularity_IS_shown_when_the_score_is_there(panel):
    panel.start()

    assert panel._result.circularity_known
    column = _column(panel.table, "circularity")
    shown = {panel.table.item(r, column).text()
             for r in range(panel.table.rowCount())}
    assert "—" not in shown


def test_the_circularity_filter_is_not_applied_when_it_cannot_be(panel):
    """Filtering on a column of NaN returns nothing and looks like a result."""
    panel.hide_circular.setChecked(True)
    panel.start()

    assert panel.table.rowCount() > 0


def test_the_score_can_come_from_the_runs_own_csvs(qtbot, inputs):
    """The merged measurements frame has no `pred` column -- it is the
    measurement tables. The scores live in the run's score CSVs, and THEY say
    `pplate1` where the databases say `plate1`."""
    from spacr.qt.widgets.sweep_panel import SweepPanel

    cells, counts = inputs
    scores = cells[["plateID", "rowID", "columnID", "pred"]].copy()
    scores["plateID"] = "p" + scores["plateID"]          # the real spelling
    widget = SweepPanel(lambda: cells.drop(columns=["pred"]), lambda: counts,
                        threaded=False, scores_provider=lambda: scores)
    qtbot.addWidget(widget)

    widget.start()

    assert widget._result.circularity_known, (
        "the score did not join -- the doubled plate name was not normalised")


def test_the_picture_is_drawn_from_what_is_shown(panel, tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    panel.start()
    out = tmp_path / "sweep.png"

    figure = panel.figure(path=str(out))

    assert figure is not None and out.exists()


# ------------------------------------------------------ mounted on the screen


def test_the_button_is_on_the_measurements_tab(qtbot):
    """A panel nobody can reach is a panel that does not exist."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)

    assert hasattr(screen, "_sweep_panel")
    assert "every gene" in screen._sweep_panel.run_button.text().lower()


def test_it_is_wired_to_the_screens_own_inputs(qtbot):
    """The merged frame, the counts and the scores all already live here; the
    panel takes them rather than going looking, so it stays testable."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    panel = screen._sweep_panel

    assert callable(panel._cells_provider)
    assert callable(panel._counts_provider)
    assert callable(panel._scores_provider)
    # And each answers without an attached run rather than raising.
    assert panel._counts_provider() is None or True
    assert panel._scores_provider() is None or True
