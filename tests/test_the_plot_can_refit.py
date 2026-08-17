"""Right-click the regression plot and fit the screen another way.

Asked for on 2026-08-16: "when all the analasees are done id like to be able
to right click on the regression plot and choose a different regression and
the other related settings as well as FDR etc."

The arithmetic of a re-fit is in :mod:`spacr.refit` and pinned by
tests/test_a_refit_is_a_new_run.py. THIS file pins the gesture: that the
action is on the menu and separated from the restyling entries above it, that
the panel refuses rather than crashes when it does not know which settings
produced the table, and that the panel asks for the run rather than starting
one it could not stop.
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


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feature": [f"fraction:grna[{i // 3}_{i % 3}]" for i in range(n)],
        "coefficient": rng.normal(0, .5, n),
        "p_value": rng.uniform(size=n),
        "condition": list(rng.choice(["nc", "pc", "other"], n,
                                     p=[.10, .05, .85])),
    })


def _settings(**over):
    settings = {
        "count_data": ["/data/screen/counts.csv"],
        "score_data": ["/data/screen/scores.csv"],
        "regression_type": "ols",
        "multiple_testing_method": "fdr_bh",
        "plot": False,
        "src": "/data/screen/results/ols",
    }
    settings.update(over)
    return settings


def _panel(qtbot):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    return panel


def _menu_items(plot):
    """Every entry on the plot's right-click menu, separators marked.

    Built rather than shown: `QMenu.exec` blocks on a modal event loop and is
    not patchable -- a first version of this helper reached in to swap it out
    and hung the whole suite rather than failing.
    """
    return ["|" if action.isSeparator() else action.text()
            for action in plot.build_style_menu().actions()]


# --------------------------------------------------------------------------- #
#  The gesture
# --------------------------------------------------------------------------- #

def test_the_volcano_offers_a_refit(qtbot):
    items = _menu_items(_panel(qtbot).volcano)

    assert any("Re-fit" in item for item in items), items


def test_it_is_separated_from_the_restyling(qtbot):
    """Everything else on that menu changes how the figure looks. A user
    reaching for "Point size" must not be one slip away from starting a
    fit."""
    items = _menu_items(_panel(qtbot).volcano)

    refit = next(i for i, text in enumerate(items) if "Re-fit" in text)
    size = next(i for i, text in enumerate(items) if "Point size" in text)
    assert "|" in items[size:refit], items


def test_a_plot_nobody_offered_it_to_does_not_have_it(qtbot):
    """The Q-Q draws a simulation and a sweep trial too. A widget that knew
    how to re-fit would offer it where there is nothing to re-fit."""
    items = _menu_items(_panel(qtbot).qq)

    assert not any("Re-fit" in item for item in items), items


# --------------------------------------------------------------------------- #
#  When it does not know what produced the table
# --------------------------------------------------------------------------- #

def test_an_unknown_run_is_refused_with_a_sentence(qtbot):
    """The user right-clicked a graph. A traceback is not an answer to
    that, and neither is a form whose only content is a disabled button."""
    panel = _panel(qtbot)
    panel.set_frame(_frame(), source="")

    assert panel.ask_refit() is False
    assert "settings" in panel.status_text()


def test_it_does_not_open_a_dialog_it_cannot_fill(qtbot, monkeypatch):
    opened = []
    monkeypatch.setattr("spacr.qt.widgets.refit_dialog.ask_refit",
                        lambda *a, **k: opened.append(a) or None)

    panel = _panel(qtbot)
    panel.set_frame(_frame(), source="")
    panel.ask_refit()

    assert opened == []


# --------------------------------------------------------------------------- #
#  When it does
# --------------------------------------------------------------------------- #

def test_it_asks_for_the_run_rather_than_starting_one(qtbot, monkeypatch):
    """The panel has no worker, no console and no Stop button. A widget that
    started a background fit with none of those is a run the user can neither
    watch nor stop."""
    from spacr.refit import refit_settings

    panel = _panel(qtbot)
    panel.set_frame(_frame(), source="")
    panel.set_run_settings(_settings())
    monkeypatch.setattr(
        "spacr.qt.widgets.refit_dialog.ask_refit",
        lambda base, parent=None: refit_settings(base, regression_type="rlm"))

    with qtbot.waitSignal(panel.refit_requested, timeout=1000) as caught:
        assert panel.ask_refit() is True

    assert caught.args[0]["regression_type"] == "rlm"


def test_cancelling_asks_for_nothing(qtbot, monkeypatch):
    panel = _panel(qtbot)
    panel.set_frame(_frame(), source="")
    panel.set_run_settings(_settings())
    monkeypatch.setattr("spacr.qt.widgets.refit_dialog.ask_refit",
                        lambda *a, **k: None)

    emitted = []
    panel.refit_requested.connect(emitted.append)

    assert panel.ask_refit() is False
    assert emitted == []


def test_what_it_reset_is_said_on_the_panel(qtbot, monkeypatch):
    """Switching off a penalty drops its weight. The user is entitled to
    read that before the run starts, not infer it from a folder name."""
    from spacr.refit import refit_settings

    panel = _panel(qtbot)
    panel.set_frame(_frame(), source="")
    panel.set_run_settings(_settings(regression_type="lasso", alpha=0.3))
    monkeypatch.setattr(
        "spacr.qt.widgets.refit_dialog.ask_refit",
        lambda base, parent=None: refit_settings(base, regression_type="ols"))
    panel.ask_refit()

    assert "alpha" in panel.status_text()


# --------------------------------------------------------------------------- #
#  Which settings the panel believes produced the table
# --------------------------------------------------------------------------- #

def test_a_new_table_does_not_inherit_the_last_ones_settings(qtbot):
    """Carrying them over offers to re-fit a screen the panel is no longer
    showing -- with a model this table was never fitted with."""
    panel = _panel(qtbot)
    panel.set_run_settings(_settings(regression_type="quantile"))
    panel.set_frame(_frame(seed=1), source="")

    assert panel._run_settings is None


def test_the_settings_are_found_beside_a_table_opened_off_disk(qtbot, tmp_path):
    from spacr.utils import save_settings

    run = tmp_path / "results" / "ols"
    run.mkdir(parents=True)
    _frame().to_csv(run / "results.csv", index=False)
    save_settings(dict(_settings(src=str(run))), name="regression")

    panel = _panel(qtbot)
    assert panel.load(str(run)) is True

    assert panel._run_settings is not None
    assert panel._run_settings["regression_type"] == "ols"


def test_the_runs_own_settings_win_over_the_file(qtbot, tmp_path):
    """The shared settings/ copy is overwritten by every LATER run of the
    same screen, so on a second run the file describes the wrong one."""
    from spacr.utils import save_settings

    run = tmp_path / "results" / "ols"
    run.mkdir(parents=True)
    _frame().to_csv(run / "results.csv", index=False)
    save_settings(dict(_settings(src=str(run), regression_type="quantile")),
                  name="regression")

    panel = _panel(qtbot)
    panel.load(str(run))
    panel.set_run_settings(_settings(regression_type="rlm"))

    assert panel._run_settings["regression_type"] == "rlm"


# --------------------------------------------------------------------------- #
#  The dialog
# --------------------------------------------------------------------------- #

def test_the_dialog_says_where_the_refit_will_land(qtbot):
    """"its output must not silently replace the run the user is looking
    at". The folder rule is asked, not predicted, so the sentence stays
    true."""
    from spacr.qt.widgets.refit_dialog import RefitDialog

    dialog = RefitDialog(_settings())
    qtbot.addWidget(dialog)

    assert "results" in dialog._notice.text()


def test_the_dialog_warns_before_it_drops_a_penalty(qtbot):
    from spacr.qt.widgets.refit_dialog import RefitDialog

    dialog = RefitDialog(_settings(regression_type="lasso", alpha=0.3))
    qtbot.addWidget(dialog)
    dialog._type.setCurrentIndex(dialog._type.findData("ols"))

    assert "alpha" in dialog._notice.text()


def test_the_penalty_box_is_off_for_a_model_with_no_penalty(qtbot):
    """It is not ignored, it is REFUSED -- so leaving the box live would
    collect a number the run rejects."""
    from spacr.qt.widgets.refit_dialog import RefitDialog

    dialog = RefitDialog(_settings(regression_type="lasso", alpha=0.3))
    qtbot.addWidget(dialog)

    assert dialog._alpha.isEnabled()
    dialog._type.setCurrentIndex(dialog._type.findData("ols"))
    assert not dialog._alpha.isEnabled()


def test_changing_only_the_correction_is_a_real_request(qtbot):
    """Comparing thirteen corrections on one fit is what the results-folder
    rule was written for, so "as before" has to be an option for the model."""
    from spacr.qt.widgets.refit_dialog import RefitDialog

    dialog = RefitDialog(_settings())
    qtbot.addWidget(dialog)
    dialog._type.setCurrentIndex(0)
    dialog._correction.setCurrentIndex(
        dialog._correction.findData("bonferroni"))

    settings, notes = dialog.settings()
    assert settings["multiple_testing_method"] == "bonferroni"
    assert settings["regression_type"] == "ols"


def test_every_correction_the_run_accepts_is_offered(qtbot):
    from spacr.multiple_testing import METHODS
    from spacr.qt.widgets.refit_dialog import RefitDialog

    dialog = RefitDialog(_settings())
    qtbot.addWidget(dialog)

    offered = {dialog._correction.itemData(i)
               for i in range(dialog._correction.count())}
    assert set(METHODS) <= offered, set(METHODS) - offered


def test_every_regression_the_run_accepts_is_offered(qtbot):
    from spacr.ml import REGRESSION_TYPES
    from spacr.qt.widgets.refit_dialog import RefitDialog

    dialog = RefitDialog(_settings())
    qtbot.addWidget(dialog)

    offered = {dialog._type.itemData(i) for i in range(dialog._type.count())}
    assert set(REGRESSION_TYPES) <= offered, set(REGRESSION_TYPES) - offered
