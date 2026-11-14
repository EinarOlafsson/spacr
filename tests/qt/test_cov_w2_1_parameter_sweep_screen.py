"""The Parameter Sweep workbench: a row that cannot be matched by id, and where the sweep lives.

The form, seeding, starting and row activation are covered in
tests/qt/test_parameter_sweep_workbench.py.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.qt.screens.parameter_sweep import (
    APP_KEY, _make_screen, build_parameter_sweep_card, sweepable,
)


class _Inline:
    """A JobRunner stand-in that runs the job where the test can see it."""

    def __init__(self):
        self.jobs = []

    def submit(self, job, handler):
        self.jobs.append(job)
        handler(job())


@pytest.fixture
def screen(qtbot):
    widget = _make_screen()
    qtbot.addWidget(widget)
    widget._runner = _Inline()
    return widget


def _results_frame(rows=2, status="ok", folder=""):
    return pd.DataFrame([{
        "trial_id": index,
        "status": status,
        "regression_type": "ols",
        "inference": "parametric",
        "analysis_unit": "well",
        "n_below_alpha": index,
        "positive_rank": index + 1,
        "folder": folder,
        "error_type": "ValueError",
        "error": "no rows survived filtration",
    } for index in range(rows)])


# ---------------------------------------------------------------------------
# Opening one row


def test_a_trial_id_that_cannot_be_matched_falls_back_to_the_row(screen,
                                                                 tmp_path,
                                                                 monkeypatch):
    """The table may be sorted, so the id is tried first — but not trusted."""
    _results_frame(rows=1).to_csv(tmp_path / "sweep_results.csv", index=False)
    screen.destination.setText(str(tmp_path))
    screen.load_results()
    screen.table.setCurrentCell(0, 0)
    monkeypatch.setattr(type(screen.results), "load",
                        lambda self, folder: True)

    class _Hostile(pd.DataFrame):
        """A results frame whose ``trial_id`` column refuses to be read.

        ``_constructor_from_mgr`` is overridden alongside ``_constructor``
        because pandas rebuilds a subclass through it on every internal
        operation. Overriding ``_constructor`` alone sends pandas down
        ``_Hostile(mgr)``, which DeprecationWarns on the pandas floor
        setup.py declares and errors in the job that installs it -- and the
        hostile ``__getitem__`` this class exists for would go missing from
        any frame pandas rebuilt the other way.
        """

        @property
        def _constructor(self):
            return _Hostile

        def _constructor_from_mgr(self, mgr, axes):
            return _Hostile._from_mgr(mgr, axes=axes)

        def __getitem__(self, key):
            if isinstance(key, str) and key == "trial_id":
                raise RuntimeError("this column is not readable")
            return super().__getitem__(key)

    screen._results = _Hostile(screen._results)

    screen._on_row_activated()

    assert "loaded from disk" in screen.trial_status.text()


# ---------------------------------------------------------------------------
# Where the sweep lives


def test_only_the_regression_module_has_a_sweep():
    """The axes and the row round trip are specific to it."""
    assert sweepable("regression") is True
    assert sweepable(APP_KEY) is False
    assert sweepable("umap") is False


def test_the_card_wraps_the_same_panel_the_screen_is(qtbot):
    """One factory, so the card and the tile cannot drift."""
    panel, card = build_parameter_sweep_card(None)
    qtbot.addWidget(card)

    assert panel.parent() is not None
    assert card.body_layout.indexOf(panel) >= 0
    assert hasattr(panel, "space")
