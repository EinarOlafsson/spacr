"""Two runs on screen at once, bounded, with a still for the rest (116).

    "every regression run should have its own interactive volcano plot"

The STATE half shipped in d4113297: a run keeps its level, its colouring, its
axis pins, its effect cut and its selection, and gets them back. This is the
other half, and the BOUND is the substance of it rather than a caveat on it.

WHY NOT N LIVE VOLCANOES. 129 measured live pyqtgraph tiles at 74.99 ms per
window-drag frame against 5.19 ms for photographs, on a 16.7 ms budget. Two
runs is what a comparison needs; twelve is what makes the screen unusable.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


def _results(folder, seed=0, rows=120):
    """A coefficient table shaped like one `perform_regression` writes."""
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    frame = pd.DataFrame({
        "feature": [f"gene_fraction:gene[{i:05d}]" for i in range(rows)],
        "coefficient": rng.normal(0, 1, rows),
        "p_value": rng.uniform(1e-6, 1, rows),
    })
    frame["q_value"] = frame["p_value"].clip(upper=0.99)
    path = os.path.join(folder, "results.csv")
    frame.to_csv(path, index=False)
    return folder


def _screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    return screen


# ---------------------------------------------------------------------------
# the bound
# ---------------------------------------------------------------------------

def test_one_run_is_live_until_a_second_is_asked_for(qtbot, tmp_path):
    screen = _screen(qtbot)
    assert screen.live_run_count() == 1
    assert screen.MAX_LIVE_RUNS == 2


def test_a_second_run_opens_beside_the_first(qtbot, tmp_path):
    """Two volcanoes on screen together, each its own live widget."""
    screen = _screen(qtbot)
    one = _results(str(tmp_path / "ols_1"), seed=1)
    two = _results(str(tmp_path / "ols_2"), seed=2)
    assert screen._results_panel.load(one)

    assert screen.open_run_beside({"folder": two}) is True
    assert screen.live_run_count() == 2
    beside = screen._compare_panel
    assert beside is not None
    # ITS OWN volcano, not the loaded run's.
    assert beside.volcano is not screen._results_panel.volcano
    assert os.path.abspath(beside.run_folder()) == os.path.abspath(two)
    assert os.path.abspath(screen._results_panel.run_folder()) == \
        os.path.abspath(one)


def test_the_third_run_is_refused_and_the_bound_is_stated(qtbot, tmp_path):
    """A bound discovered by a refusal with no reason is indistinguishable
    from a broken button."""
    screen = _screen(qtbot)
    screen._results_panel.load(_results(str(tmp_path / "ols_1"), seed=1))
    screen.open_run_beside({"folder": _results(str(tmp_path / "ols_2"),
                                               seed=2)})

    third = _results(str(tmp_path / "ols_3"), seed=3)
    assert screen.open_run_beside({"folder": third}) is False
    assert screen.live_run_count() == 2

    said = screen._console._current_stdout.toPlainText()
    assert "Two runs can be live at once" in said
    # The MEASUREMENT, not "this would be slow".
    assert "75 ms" in said and "5 ms" in said and "17 ms" in said


def test_a_run_is_not_opened_beside_itself(qtbot, tmp_path):
    """Two views of one run is not the comparison that was asked for."""
    screen = _screen(qtbot)
    one = _results(str(tmp_path / "ols_1"), seed=1)
    screen._results_panel.load(one)
    assert screen.open_run_beside({"folder": one}) is False
    assert screen.live_run_count() == 1


def test_a_folder_with_no_results_says_so(qtbot, tmp_path):
    screen = _screen(qtbot)
    empty = str(tmp_path / "nothing")
    os.makedirs(empty)
    assert screen.open_run_beside({"folder": empty}) is False
    assert "No results in" in screen._console._current_stdout.toPlainText()


def test_opening_beside_is_deliberate_and_has_its_own_gesture(qtbot,
                                                              tmp_path):
    """Nothing opens a second run on its own: it is a menu entry, and the
    row has to be a run that produced something."""
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    folder = _results(str(tmp_path / "ols_1"))
    panel.update_run(panel.record_run("ols_1", folder=folder), status="ok")

    heard = []
    panel.compare_requested.connect(heard.append)
    menu = panel._build_run_menu(panel._all_rows())
    beside = [action for action in menu.actions()
              if action.data() == "beside"]
    assert len(beside) == 1
    assert beside[0].toolTip().startswith("Show this run's own volcano")

    # Through the real dispatch. `_run_menu` itself ends in `QMenu.exec`, a
    # C++ event loop a test cannot enter and return from.
    assert panel._apply_run_menu("beside", panel._all_rows()) is True
    assert [row["run"] for row in heard] == ["ols_1"]


def test_a_running_run_is_not_offered_beside(qtbot):
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.record_run("ols_1")                       # still going
    verbs = [action.data()
             for action in panel._build_run_menu(panel._all_rows()).actions()]
    assert "beside" not in verbs


def test_several_rows_selected_offer_no_beside(qtbot, tmp_path):
    """Beside compares TWO runs. Four selected rows is not a comparison."""
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    for index in range(3):
        panel.update_run(panel.record_run(f"ols_{index}",
                                          folder=str(tmp_path)), status="ok")
    verbs = [action.data()
             for action in panel._build_run_menu(panel._all_rows()).actions()]
    assert "beside" not in verbs


# ---------------------------------------------------------------------------
# the photograph
# ---------------------------------------------------------------------------

def test_closing_the_run_beside_keeps_its_photograph(qtbot, tmp_path):
    """A STILL STANDS IN FOR A RUN THAT IS NOT LIVE.

    The state was never in the widget, so making it live again is cheap --
    which is what the bound buys.
    """
    screen = _screen(qtbot)
    screen._results_panel.load(_results(str(tmp_path / "ols_1"), seed=1))
    two = _results(str(tmp_path / "ols_2"), seed=2)
    screen.open_run_beside({"folder": two})
    screen._compare_panel.volcano.resize(320, 240)

    assert screen.close_run_beside() is True
    assert screen.live_run_count() == 1
    photo = screen.run_photograph(two)
    assert photo is not None and not photo.isNull()
    assert photo.width() > 0 and photo.height() > 0


def test_a_run_never_opened_beside_has_no_still(qtbot, tmp_path):
    screen = _screen(qtbot)
    assert screen.run_photograph(str(tmp_path / "ols_9")) is None
    assert screen.run_photograph("") is None


def test_the_slot_is_free_again_after_closing(qtbot, tmp_path):
    screen = _screen(qtbot)
    screen._results_panel.load(_results(str(tmp_path / "ols_1"), seed=1))
    screen.open_run_beside({"folder": _results(str(tmp_path / "ols_2"),
                                               seed=2)})
    screen.close_run_beside()
    assert screen.open_run_beside(
        {"folder": _results(str(tmp_path / "ols_3"), seed=3)}) is True
    assert screen.live_run_count() == 2


def test_closing_nothing_is_not_an_error(qtbot):
    assert _screen(qtbot).close_run_beside() is False


# ---------------------------------------------------------------------------
# a deleted run takes both halves with it
# ---------------------------------------------------------------------------

def test_deleting_the_run_beside_closes_it_and_drops_its_still(qtbot,
                                                               tmp_path):
    """"Deleting a run takes its plot state with it and leaves the others
    intact" -- and a photograph of a run that no longer exists is the same
    stale answer its state would have been."""
    screen = _screen(qtbot)
    screen._results_panel.load(_results(str(tmp_path / "ols_1"), seed=1))
    two = _results(str(tmp_path / "ols_2"), seed=2)
    screen.open_run_beside({"folder": two})
    assert screen.live_run_count() == 2

    screen._on_runs_removed([{"run": "ols_2", "folder": two}])
    assert screen.live_run_count() == 1
    assert screen._compare_panel is None
    assert screen.run_photograph(two) is None


# ---------------------------------------------------------------------------
# the frame budget, measured the way 129 measured it
# ---------------------------------------------------------------------------

def test_a_still_is_cheaper_to_paint_than_the_live_plot(qtbot, tmp_path):
    """MEASURED, not asserted by eye.

    129's numbers are 74.99 ms per window-drag frame for a live pyqtgraph
    tile against 5.19 ms for a photograph. The absolute figures belong to
    that machine; the RATIO is the property the bound rests on, so that is
    what is held here -- and generously, so a slow CI box cannot make the
    architecture look wrong.
    """
    import time

    from PySide6.QtGui import QPainter, QPixmap

    screen = _screen(qtbot)
    screen._results_panel.load(_results(str(tmp_path / "ols_1"), seed=1,
                                        rows=800))
    two = _results(str(tmp_path / "ols_2"), seed=2, rows=800)
    assert screen.open_run_beside({"folder": two})

    plot = screen._compare_panel.volcano
    plot.resize(480, 360)
    qtbot.wait(20)

    def _paint(passes=5):
        started = time.perf_counter()
        for _ in range(passes):
            target = QPixmap(plot.size())
            plot.render(target)
        return (time.perf_counter() - started) / passes

    live = _paint()
    still = plot.grab()

    def _blit(passes=5):
        started = time.perf_counter()
        for _ in range(passes):
            target = QPixmap(still.size())
            painter = QPainter(target)
            painter.drawPixmap(0, 0, still)
            painter.end()
        return (time.perf_counter() - started) / passes

    photograph = _blit()
    assert photograph < live, (
        f"a still painted in {photograph * 1000:.2f} ms and the live plot in "
        f"{live * 1000:.2f} ms; the bound rests on the still being cheaper")
