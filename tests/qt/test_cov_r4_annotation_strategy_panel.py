"""The annotation panel's refusals: bad keys, dead providers, an empty save.

The strategies themselves, the menu, and the ordinary run are held by
``test_the_cell_tab_offers_annotation_strategies`` and
``test_a_strategy_the_cells_cannot_support_says_so``. What is pinned here is
what the panel does when something it depends on is not there:

* a strategy key that is not on the menu -- the panel is built before the
  montage has decided what it is showing, and an unknown key must grey the
  run button with the module's own sentence rather than raise into Qt;
* a provider that raises -- the montage supplies the object rows, the score
  column and the run folder, and all three are read while it is still
  loading;
* a table that disappears between the pre-flight and the run;
* the save that has to ask for a folder, and the answer "cancel".

Every one of these was a way to reach a fit with nothing to fit, or to leave
the panel silent about why nothing happened.
"""
from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from spacr import regression_annotation as ra                  # noqa: E402
from spacr.qt.widgets import annotation_strategy_panel as asp   # noqa: E402
from spacr.qt.widgets.annotation_strategy_panel import (        # noqa: E402
    NOTHING_RUN, AnnotationStrategyPanel, wells_of_plans)

WELLS = tuple(f"r1_c{i}" for i in range(1, 7))

#: What the shared panel's providers answer, and how often they were asked.
#: An ``Exception`` value is raised instead of returned.
_SHOWING: dict = {"objects": None, "score": "pred", "folder": "", "asked": 0}


def _rows(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """A measured plate with a score, enough for a strategy to be runnable."""
    rng = np.random.default_rng(seed)
    wells = [WELLS[i % len(WELLS)] for i in range(n)]
    frame = pd.DataFrame({
        "plateID": "plate1",
        "rowID": [w.split("_")[0] for w in wells],
        "columnID": [w.split("_")[1] for w in wells],
        "fieldID": "f1",
        "object_label": np.arange(n),
        "cell_area": rng.random(n) * 100.0,
        "cell_channel_1_mean_intensity": rng.random(n) * 50.0,
        "cell_perimeter": rng.random(n) * 10.0,
    })
    frame["pred"] = 1.0 / (1.0 + np.exp(
        -(frame["cell_area"] / 30.0 - 1.5 + rng.normal(0.0, 0.3, n))))
    return frame


def _answer(name):
    """Return what the montage is showing, or raise what it would raise."""
    value = _SHOWING[name]
    if isinstance(value, Exception):
        raise value
    return value


def _objects_provider():
    _SHOWING["asked"] += 1
    return _answer("objects")


@pytest.fixture(scope="module")
def _one_panel():
    """ONE panel for the file. A panel is around a hundred widgets."""
    widget = AnnotationStrategyPanel(
        objects_provider=_objects_provider,
        wells_provider=lambda: WELLS,
        score_provider=lambda: _answer("score"),
        folder_provider=lambda: _answer("folder"),
        threaded=False)
    yield widget
    try:
        widget.shutdown()
    except Exception:                                        # noqa: BLE001
        pass
    widget.close()


@pytest.fixture()
def panel(_one_panel):
    """The shared panel with its providers and controls back at defaults."""
    _SHOWING.update({"objects": _rows(), "score": "pred", "folder": "",
                     "asked": 0})
    _one_panel.set_strategy("top_score_random")
    _one_panel._positive_wells.setText("")
    _one_panel._negative_wells.setText("")
    _one_panel._label_column.setText("")
    _one_panel._wells.setText(", ".join(WELLS))
    _one_panel._budget.setValue(10)
    _one_panel._holdout.setValue(0.25)
    _one_panel._result = None
    _one_panel._running = False
    _one_panel._report.setPlainText("")
    _one_panel.refresh()
    _SHOWING["asked"] = 0
    return _one_panel


# ---------------------------------------------------------------------------
# the wells a montage names
# ---------------------------------------------------------------------------

def test_a_plan_with_a_blank_well_contributes_no_well():
    """A well with no name is not a well the positives can come from.

    The plans come from the montage, where a row that failed to resolve
    carries an empty identifier. Keeping it would put an empty string in the
    control-well list and select cells from a well that does not exist.
    """
    plans = [SimpleNamespace(wells=[SimpleNamespace(well="   "),
                                    SimpleNamespace(well="r1_c1"),
                                    SimpleNamespace(well=None),
                                    SimpleNamespace(well="r1_c1"),
                                    SimpleNamespace(well="r1_c2")])]

    assert wells_of_plans(plans) == ("r1_c1", "r1_c2")


# ---------------------------------------------------------------------------
# the controls
# ---------------------------------------------------------------------------

def test_typing_in_a_filled_field_does_not_re_read_the_montage(panel):
    """The pre-flight runs on emptiness, not on every keystroke.

    Re-checking asks the montage for its object rows, which over a plate is a
    concatenation; doing it per character typed made the control-well field
    unusable. Only a field that starts or stops being empty can change the
    answer, so only that re-reads.
    """
    panel._positive_wells.setText("r1_c1")
    after_first = _SHOWING["asked"]
    assert after_first > 0

    panel._positive_wells.setText("r1_c1, r1_c2")

    assert _SHOWING["asked"] == after_first

    panel._positive_wells.setText("")

    assert _SHOWING["asked"] > after_first


def test_a_key_that_is_not_on_the_menu_selects_nothing(panel):
    """``set_strategy`` reports availability instead of guessing."""
    before = panel.strategy_key()

    assert panel.set_strategy("no_such_strategy") is False
    assert panel.strategy_key() == before

    assert panel.set_strategy("uncertainty") is True
    assert panel.strategy_key() == "uncertainty"


def test_an_unregistered_strategy_greys_the_run_button_and_says_why(panel):
    """The panel outlives the module's table, so it must survive a stale key.

    A key the module does not know is a refusal with the menu in it, on the
    control -- not a traceback out of a combo box's signal. The description is
    cleared as the selection changes; the greying follows on the next refresh,
    which is what the montage calls when its data change.
    """
    with pytest.raises(ra.AnnotationStrategyError) as refusal:
        ra.strategy("no_such_strategy")
    expected = str(refusal.value)

    panel._menu.addItem("Not a registered strategy", "no_such_strategy")
    try:
        panel._menu.setCurrentIndex(panel._menu.count() - 1)

        assert panel.about_text() == ""
        assert panel.reason() == expected

        panel.refresh()

        assert panel._run_button.isEnabled() is False
        assert panel._run_button.toolTip() == expected
    finally:
        panel._menu.setCurrentIndex(0)
        panel._menu.removeItem(panel._menu.count() - 1)

    assert panel.about_text() != ""


# ---------------------------------------------------------------------------
# providers that will not answer
# ---------------------------------------------------------------------------

def test_a_score_provider_that_raises_falls_back_to_pred(panel):
    """A montage still loading must not lose the request its columns name."""
    _SHOWING["score"] = "cv_predictions"
    panel._objects_provider = _objects_provider

    assert panel.request().score_column == "cv_predictions"

    _SHOWING["score"] = RuntimeError("the montage is still loading")

    assert panel.request().score_column == "pred"


def test_an_object_provider_that_raises_leaves_nothing_to_run(panel, caplog):
    """"Cannot answer" and "no rows" reach the user as the same sentence."""
    assert panel.request() is not None

    _SHOWING["objects"] = RuntimeError("the montage is still loading")

    with caplog.at_level(logging.DEBUG, logger=asp.LOG.name):
        assert panel.request() is None
        assert "no cells to choose from" in panel.reason()
    assert any("could not read the object rows" in record.getMessage()
               for record in caplog.records)


def test_a_table_that_vanishes_between_the_check_and_the_run(panel):
    """The pre-flight and the request read the montage separately.

    The rows are fetched again to build the request, and a montage that was
    cleared in between used to reach the fit with ``None`` for a frame.
    """
    frames = [_SHOWING["objects"], None, None]

    def _vanishing():
        _SHOWING["asked"] += 1
        return frames.pop(0) if frames else None

    panel._objects_provider = _vanishing
    try:
        assert panel.run() is False
        assert "no cells to choose from" in panel._status.text()
    finally:
        panel._objects_provider = _objects_provider

    assert NOTHING_RUN not in panel._status.text()


def test_a_runner_that_fails_says_so_rather_than_staying_busy(panel):
    """The worker itself raising must clear the running flag.

    Without this the panel is permanently "a strategy is already running" and
    the run button never comes back.
    """
    panel._running = True
    panel._refresh_controls()
    assert panel._run_button.isEnabled() is False

    panel._jobs.job_failed.emit("the worker thread died")

    assert panel._running is False
    assert panel._status.text() == (
        "The strategy failed: the worker thread died")
    assert panel._run_button.isEnabled() is True


# ---------------------------------------------------------------------------
# saving
# ---------------------------------------------------------------------------

class _Result:
    """The one method ``save`` calls on a result."""

    def __init__(self, folders):
        self.folders = folders

    def write(self, folder):
        self.folders.append(folder)
        os.makedirs(folder, exist_ok=True)
        return {"selected": os.path.join(folder, "selected.csv")}


def test_a_cancelled_folder_chooser_writes_nothing(panel, monkeypatch):
    """Cancel means cancel: no folder, no files, no status claiming any.

    The chooser is opened at the folder the montage suggests; a provider that
    raises costs the suggestion, not the save.
    """
    asked = []
    _SHOWING["folder"] = RuntimeError("no run folder yet")
    panel._result = _Result([])
    monkeypatch.setattr(asp, "QFileDialog", SimpleNamespace(
        getExistingDirectory=lambda *args: asked.append(args) or ""))

    assert panel.save() == {}

    assert len(asked) == 1
    assert asked[0][-1] == "", "a provider that raised still opened the chooser"
    assert panel._result.folders == []


def test_a_chosen_folder_is_written_into(panel, monkeypatch, tmp_path):
    """The counterpart, with no provider at all to suggest a start folder."""
    written_to = []
    panel._result = _Result(written_to)
    panel._folder_provider = None
    monkeypatch.setattr(asp, "QFileDialog", SimpleNamespace(
        getExistingDirectory=lambda *args: str(tmp_path)))
    try:
        written = panel.save()
    finally:
        panel._folder_provider = lambda: _answer("folder")

    assert list(written) == ["selected"]
    assert written_to == [os.path.join(
        str(tmp_path), f"annotation_{panel.strategy_key()}")]
    assert "Wrote 1 file(s)" in panel._status.text()


def test_a_runner_that_will_not_stop_does_not_stop_the_close(panel, caplog):
    """Closing the montage may not be blocked by a worker that will not go."""
    def refuse():
        raise RuntimeError("the thread would not join")

    original = panel._jobs.shutdown
    panel._jobs.shutdown = refuse
    try:
        with caplog.at_level(logging.DEBUG, logger=asp.LOG.name):
            panel.shutdown()
    finally:
        panel._jobs.shutdown = original

    assert any("would not shut down" in record.getMessage()
               for record in caplog.records)
