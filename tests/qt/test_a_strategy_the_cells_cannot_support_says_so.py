"""Choosing a strategy the loaded cells cannot support says so on the control.

The Cells tab's menu offers ten strategies whatever is on screen, which is
right -- a menu that hid the entries a table cannot support would be a menu
that never taught anybody what else there is. What it may not do is leave
"Run the strategy" lit over a table with no measurement columns and no
score, so that the answer arrives a boosted tree later.

These drive the real widgets: the combo box is set, the event loop is
spun, and the button's own ``isEnabled`` and tooltip are read -- not the
function that computes them.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtWidgets import QApplication                     # noqa: E402

from spacr import regression_annotation as ra                  # noqa: E402
from spacr.qt.widgets.annotation_strategy_panel import (       # noqa: E402
    AnnotationStrategyPanel)

WELLS = tuple(f"r1_c{i}" for i in range(1, 7))


def _rows(n: int = 120, *, score: bool = True, measured: bool = True,
          seed: int = 0) -> pd.DataFrame:
    """A plate of objects: identity always, a score and measurements by ask."""
    rng = np.random.default_rng(seed)
    wells = [WELLS[i % len(WELLS)] for i in range(n)]
    frame = pd.DataFrame({
        "plateID": "plate1",
        "rowID": [w.split("_")[0] for w in wells],
        "columnID": [w.split("_")[1] for w in wells],
        "fieldID": "f1",
        "object_label": np.arange(n),
    })
    if measured:
        frame["cell_area"] = rng.random(n) * 100.0
        frame["cell_channel_1_mean_intensity"] = rng.random(n) * 50.0
        frame["cell_perimeter"] = rng.random(n) * 10.0
    if score:
        base = frame["cell_area"] if measured else pd.Series(rng.random(n))
        frame["pred"] = 1.0 / (1.0 + np.exp(
            -(np.asarray(base, dtype=float) / 30.0 - 1.5
              + rng.normal(0.0, 0.3, n))))
    return frame


#: What the panel is pointed at. One mutable holder rather than one panel
#: per table: a panel is around a hundred widgets and Qt keeps every one of
#: them for the session.
_SHOWING: dict = {"frame": None}


@pytest.fixture(scope="module")
def _one_panel():
    widget = AnnotationStrategyPanel(
        objects_provider=lambda: _SHOWING["frame"],
        wells_provider=lambda: (),
        score_provider=lambda: "pred",
        folder_provider=lambda: "",
        threaded=False)
    yield widget
    try:
        widget.shutdown()
    except Exception:                                        # noqa: BLE001
        pass
    widget.close()


@pytest.fixture()
def panel(_one_panel):
    """The shared panel, its controls back at their defaults."""
    _SHOWING["frame"] = _rows()
    _one_panel._positive_wells.setText("")
    _one_panel._negative_wells.setText("")
    _one_panel._label_column.setText("")
    _one_panel._wells.setText("")
    _one_panel._budget.setValue(10)
    _one_panel._holdout.setValue(0.34)
    _one_panel._seed.setValue(0)
    _one_panel._result = None
    _one_panel._running = False
    _one_panel._report.setPlainText("")
    _one_panel.set_strategy("top_score_random")
    _one_panel.refresh()
    QApplication.processEvents()
    return _one_panel


def _choose(panel, key: str) -> None:
    """Pick a strategy the way a user does, and let the panel settle."""
    index = panel._menu.findData(key)
    assert index >= 0, f"{key} is not on the menu"
    panel._menu.setCurrentIndex(index)
    QApplication.processEvents()


# --------------------------------------------------------------------------- #
#  The button greys with the reason, before anything is run
# --------------------------------------------------------------------------- #

def test_a_measured_screen_leaves_the_button_lit(panel):
    """A control that CAN act is lit, so the greying below means something."""
    for key in ("top_score_random", "diversity", "uncertainty",
                "score_strata", "random_holdout"):
        _choose(panel, key)
        assert panel._run_button.isEnabled(), key


def test_a_fitting_strategy_greys_itself_on_a_table_with_no_measurements(
        panel):
    """No measurement columns, so nothing to fit -- said on the button."""
    _SHOWING["frame"] = _rows(measured=False)
    panel.refresh()
    _choose(panel, "diversity")
    assert not panel._run_button.isEnabled()
    tip = panel._run_button.toolTip()
    assert "measurement" in tip
    assert "Diversity sampling over clusters" in tip
    # AND WHERE IT CAN BE READ WITHOUT HOVERING. The tab is often the first
    # thing a user opens, and a tooltip is not an answer they will find.
    assert "measurement" in panel._status.text()
    # Nothing was fitted to learn that.
    assert panel.result() is None


def test_the_plain_random_draw_stays_offered_on_that_same_table(panel):
    """Rule nine: the unbiased sample is available whatever else is not."""
    _SHOWING["frame"] = _rows(measured=False)
    panel.refresh()
    _choose(panel, "random_holdout")
    assert panel._run_button.isEnabled()
    assert panel.run() is True
    result = panel.result()
    assert result is not None, panel.report_text()
    assert "fits no model" in panel.report_text()
    assert "measured on the hold-out" in panel._status.text()


def test_with_no_score_column_every_entry_greys_and_names_itself(panel):
    """Nothing to label the cells with, so no entry can be measured."""
    _SHOWING["frame"] = _rows(score=False)
    panel.refresh()
    for entry in ra.STRATEGIES:
        _choose(panel, entry.key)
        assert not panel._run_button.isEnabled(), entry.key
        assert "nothing to label the cells with" in \
            panel._run_button.toolTip(), entry.key
        assert entry.title in panel._run_button.toolTip()


def test_naming_an_annotation_column_lights_the_button_again(panel):
    """A column somebody annotated is the other way to have labels."""
    frame = _rows(score=False)
    frame["verdict"] = ["yes", "no"] * (len(frame) // 2)
    _SHOWING["frame"] = frame
    panel.refresh()
    _choose(panel, "top_score_random")
    assert not panel._run_button.isEnabled()
    panel._label_column.setText("verdict")
    QApplication.processEvents()
    # NO OTHER CONTROL WAS TOUCHED. Typing into the field is what re-checks,
    # because a button that stayed grey until something else happened reads
    # as a button that does not work.
    assert panel._run_button.isEnabled()


# --------------------------------------------------------------------------- #
#  The anchored strategy, whose reason is a field on this very form
# --------------------------------------------------------------------------- #

def test_control_anchors_greys_until_both_control_lists_are_named(panel):
    _choose(panel, "control_anchors")
    assert not panel._run_button.isEnabled()
    assert "control well" in panel._run_button.toolTip()

    panel._positive_wells.setText("r1_c1")
    QApplication.processEvents()
    assert not panel._run_button.isEnabled()
    assert "negative control well" in panel._run_button.toolTip()

    panel._negative_wells.setText("r1_c2")
    QApplication.processEvents()
    assert panel._run_button.isEnabled()
    assert "hold-out" in panel._run_button.toolTip()


def test_emptying_a_control_list_greys_it_again(panel):
    _choose(panel, "control_anchors")
    panel._positive_wells.setText("r1_c1")
    panel._negative_wells.setText("r1_c2")
    QApplication.processEvents()
    assert panel._run_button.isEnabled()
    panel._negative_wells.setText("")
    QApplication.processEvents()
    assert not panel._run_button.isEnabled()
    assert "negative control well" in panel._run_button.toolTip()


# --------------------------------------------------------------------------- #
#  What the greying may not cost
# --------------------------------------------------------------------------- #

def test_the_reason_never_overwrites_the_report_of_a_finished_run(panel):
    """A run's numbers stay on screen while another entry is being read."""
    _choose(panel, "random_holdout")
    assert panel.run() is True
    written = panel.report_text()
    assert written
    _choose(panel, "control_anchors")
    assert not panel._run_button.isEnabled()
    assert panel.report_text() == written


def test_a_pre_flight_that_raises_does_not_take_the_strategy_away(panel,
                                                                 monkeypatch):
    """The run checks everything this does; a cheap check is not a veto."""
    def explode(*_args, **_kwargs):
        raise RuntimeError("the pre-flight fell over")

    monkeypatch.setattr(ra, "missing_requirement", explode)
    _choose(panel, "top_score_random")
    assert panel.reason() == ""
    assert panel._run_button.isEnabled()


def test_every_entry_is_still_on_the_menu_whatever_the_table_carries(panel):
    """Greyed, not hidden: the menu is what says what else there is."""
    _SHOWING["frame"] = _rows(score=False, measured=False)
    panel.refresh()
    QApplication.processEvents()
    assert panel._menu.count() == len(ra.STRATEGIES)
    keys = [panel._menu.itemData(i) for i in range(panel._menu.count())]
    assert keys == list(ra.strategy_keys())
