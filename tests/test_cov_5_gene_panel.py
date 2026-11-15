"""The gene tile never shows a blank field, and never takes the panel down.

The panel reads reference files this install may not have, on a worker
thread, while a window is being closed around it. Each branch here is a place
where the honest answer is a sentence -- "no gene number could be parsed",
"could not write that file" -- and where an exception instead would abort a
QThread teardown or leave the reader looking at empty labels that read as
"measured, found nothing".
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from tests.test_clicking_a_gene_says_what_it_is import (  # noqa: E402
    REAL_ROWS)

pytestmark = pytest.mark.qt

COLUMNS = ["feature", "coefficient", "p_value", "grna", "condition", "gene",
           "n_grna", "n_gene"]


@pytest.fixture()
def results() -> pd.DataFrame:
    frame = pd.DataFrame(REAL_ROWS, columns=COLUMNS)
    frame["q_value"] = frame["p_value"]
    frame["multiple_testing_method"] = "none"
    return frame


@pytest.fixture()
def panel(qtbot, results):
    """A panel warmed inline and shown, which is the state a user is in."""
    from spacr.qt.widgets.gene_panel import GenePanel

    widget = GenePanel(frame_provider=lambda: results, threaded=False)
    qtbot.addWidget(widget)
    widget.show()
    return widget


# --------------------------------------------------------------------------- #
#  Warming up
# --------------------------------------------------------------------------- #

def test_a_reference_this_install_lacks_does_not_stop_the_warm_up(monkeypatch):
    """A failing tile warm-up still returns the annotation columns.

    The warm-up is an optimisation: it pre-reads indices so a click does not
    read a file. An install without one of those reference files must still
    open the panel -- the click path says what it could not resolve, which is
    the same answer either way.
    """
    from spacr import gene_tile
    from spacr.qt.widgets.gene_panel import warm_annotation

    def _explode(term):
        raise FileNotFoundError("no grna reference on this install")

    monkeypatch.setattr(gene_tile, "gene_tile", _explode)

    columns = warm_annotation(["fraction:grna[239740_3]"])

    assert isinstance(columns, tuple)


def test_the_warm_up_is_started_once_however_often_it_is_asked(panel):
    """A second request while one is under way starts no second worker.

    Two warm-ups read the same 360 ms of CSV twice and race to deliver into
    the same panel; the loser overwrites the winner's columns.
    """
    assert panel.warm_now() is False        # the show already started it
    assert panel.warm_now() is False


# --------------------------------------------------------------------------- #
#  Saving the topology through the dialog
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#  Going away
# --------------------------------------------------------------------------- #

def test_shutting_down_a_panel_whose_worker_is_already_gone_is_quiet(panel):
    """A runner whose C++ half has gone reports nothing and raises nothing.

    This runs from ``QApplication.aboutToQuit`` while a whole window is
    closing, so the runner can already be destroyed. There is nothing left to
    stop and nothing to report.
    """
    class _Gone:
        def shutdown(self):
            raise RuntimeError("Internal C++ object already deleted")

    panel._runner = _Gone()

    assert panel._shut_down_warming() is None


def test_a_panel_collected_at_interpreter_shutdown_swallows_everything(panel):
    """``__del__`` never raises, whatever state the panel is in.

    It runs during garbage collection and at interpreter shutdown, where the
    module globals may already be None. An exception there is printed and
    ignored by Python anyway, and the one thing worth doing is the shutdown
    attempt.
    """
    class _Hostile:
        def shutdown(self):
            raise MemoryError("the interpreter is already tearing down")

    real = panel._runner
    panel._runner = _Hostile()
    try:
        assert type(panel).__del__(panel) is None
    finally:
        panel._runner = real


# --------------------------------------------------------------------------- #
#  Reading the terms out of a frame
# --------------------------------------------------------------------------- #

def test_a_frame_whose_feature_column_cannot_be_read_warms_nothing():
    """A frame that raises on access contributes no terms instead of failing.

    The warm-up is handed whatever the screen currently holds, which can be a
    proxy or a partly-built frame. Nothing here is worth taking the panel
    down for.
    """
    from spacr.qt.widgets.gene_panel import _terms_of

    class _Hostile:
        columns = ["feature", "coefficient"]

        def __getitem__(self, key):
            raise KeyError("the column is registered but not materialised")

    assert _terms_of(_Hostile()) == []
    assert _terms_of(None) == []
    assert _terms_of(pd.DataFrame({"coefficient": [1.0]})) == []
    assert _terms_of(pd.DataFrame({"feature": ["a", "b"]})) == ["a", "b"]
