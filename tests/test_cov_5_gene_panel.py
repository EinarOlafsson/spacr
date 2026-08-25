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
    AMBIGUOUS_ROWS, CLEAN_ROWS, REAL_ROWS)

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


def _lower(widget) -> str:
    return widget._known.toPlainText()


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


def test_the_panel_can_be_pointed_at_another_results_frame(panel, results):
    """``set_frame_provider`` reaches the half that reads the frame.

    The numbers on the tile are the frame's, not a copy. A provider that
    stopped at the outer panel would leave the tile showing the previous
    screen's effect and p-value under the new screen's gene.
    """
    moved = results.copy()
    moved["coefficient"] = moved["coefficient"] * -1.0

    panel.set_frame_provider(lambda: moved)
    panel.show_feature("fraction:grna[239740_3]")

    assert panel.summary._frame_provider() is moved


def test_a_term_with_no_gene_behind_it_says_so_rather_than_showing_nothing(
        panel, monkeypatch):
    """A resolvable term whose genes have no record gets a sentence.

    The panel's contract is that nothing is ever blank: "no annotation for
    this gene" and "this gene has no product recorded" are different facts,
    and empty fields would be read as the second.
    """
    from spacr import gene_facts

    monkeypatch.setattr(gene_facts, "facts_for", lambda genes: {})

    panel.show_feature("fraction:grna[239740_3]")

    assert "No gene number could be parsed" in _lower(panel)
    assert panel._facts == ()


# --------------------------------------------------------------------------- #
#  Saving the topology through the dialog
# --------------------------------------------------------------------------- #

def test_the_save_dialog_offers_a_filename_built_from_the_gene(
        panel, tmp_path, monkeypatch):
    """The suggested name carries the gene, and the status line the path.

    A folder of ``deeptmhmm.csv`` files is a folder nobody can use. The
    status line is the only confirmation the write happened at all.
    """
    from spacr.qt.widgets import gene_panel as gp

    panel.show_feature("fraction:grna[239740_3]")
    target = tmp_path / "topology.csv"
    asked = {}

    def _dialog(parent, caption, suggested, filter_):
        asked["suggested"] = suggested
        return str(target), filter_

    monkeypatch.setattr(gp.QFileDialog, "getSaveFileName", staticmethod(_dialog))

    panel._ask_to_save_topology()

    assert "239740" in asked["suggested"]
    assert target.exists()
    assert str(target) in panel._status.text()


def test_a_cancelled_save_writes_nothing_and_says_nothing(
        panel, tmp_path, monkeypatch):
    """Dismissing the dialog leaves the status line and the folder alone.

    An empty path is the user saying no; reporting a write, or attempting
    one, would put a file called nothing in the working directory.
    """
    from spacr.qt.widgets import gene_panel as gp

    panel.show_feature("fraction:grna[239740_3]")
    before = panel._status.text()
    monkeypatch.setattr(gp.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    monkeypatch.chdir(tmp_path)

    panel._ask_to_save_topology()

    assert panel._status.text() == before
    assert list(tmp_path.iterdir()) == []


def test_a_write_that_fails_says_which_file_and_why(
        panel, tmp_path, monkeypatch):
    """An unwritable destination is reported on the status line, not raised.

    A traceback out of a menu action closes nothing and explains nothing; the
    user needs the path and the reason so they can pick another folder.
    """
    from spacr.qt.widgets import gene_panel as gp

    panel.show_feature("fraction:grna[239740_3]")
    blocker = tmp_path / "blocker"
    blocker.write_text("this is a file, not a folder")
    doomed = blocker / "topology.csv"
    monkeypatch.setattr(gp.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(doomed), "")))

    panel._ask_to_save_topology()

    assert "Could not write" in panel._status.text()
    assert str(doomed) in panel._status.text()
    assert not doomed.exists()


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
