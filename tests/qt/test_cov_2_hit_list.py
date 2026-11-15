"""What the Hit List screen does when there is nothing to act on.

Every path here is one where the user has asked for something the screen
cannot do yet: exporting before a folder is loaded, investigating with no row
selected, moving a filter on an empty screen. None of them may raise, because
they run inside Qt slots where an exception is a crash, and none of them may
be silent, because a button that appears to do nothing reads as a broken
build. Each one has to say what is missing.

Offscreen, CPU-only, offline.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt                                    # noqa: E402

from spacr.qt.screens import hit_list as screen_module           # noqa: E402

pytestmark = pytest.mark.qt


def _gene_frame():
    return pd.DataFrame({
        "feature": [f"gene_fraction:gene[{g}]" for g in ("100", "200", "300")],
        "coefficient": [2.4, -1.8, 0.9],
        "std_err": [0.30, 0.25, 0.40],
        "p_value": [1e-6, 4e-5, 0.02],
        "condition": ["other", "other", "other"],
        "n_gene": [48, 44, 30],
    })


def _grna_frame():
    rows = [("100_1", 2.2), ("100_2", 2.9), ("100_3", 1.7),
            ("200_1", -1.9), ("200_2", 1.4), ("300_1", 0.9)]
    return pd.DataFrame({
        "feature": [f"fraction:grna[{g}]" for g, _ in rows],
        "grna": [g for g, _ in rows],
        "coefficient": [c for _, c in rows]})


@pytest.fixture()
def folder(tmp_path):
    """A results folder laid out the way ``perform_regression`` writes one."""
    root = tmp_path / "results" / "pred" / "ols"
    root.mkdir(parents=True)
    _gene_frame().to_csv(root / "results_gene.csv", index=False)
    _grna_frame().to_csv(root / "results_grna.csv", index=False)
    pd.concat([_gene_frame(), _grna_frame()], ignore_index=True).to_csv(
        root / "results.csv", index=False)
    return str(root)


@pytest.fixture()
def empty_screen(qtbot):
    """The screen with no folder — what a user sees when it first opens."""
    widget = screen_module.HitListScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# loading nothing
# ---------------------------------------------------------------------------

def test_loading_a_blank_folder_asks_for_one_instead_of_reading_it(
        empty_screen):
    """An empty path is a prompt, not a read of the current directory.

    ``load_folder("")`` happens whenever the folder box is cleared and
    Return is pressed. Passing the empty string through would send a worker
    at ``""`` -- which resolves to the process's working directory -- and
    the user would get an unexplained failure about a folder they never
    named.
    """
    empty_screen.load_folder("   ")

    assert "Choose a regression results folder." in empty_screen._summary.text()
    assert empty_screen._all is None
    assert empty_screen.last_error == ""
    assert empty_screen._table.topLevelItemCount() == 0


def test_pressing_return_in_the_folder_box_loads_what_was_typed(
        qtbot, empty_screen, folder):
    """The Return key is the second way in, beside the Browse button.

    A box that only worked through a modal file dialog cannot be driven by
    paste-and-Return, which is how a path from a terminal gets into the app.
    """
    empty_screen._folder_edit.setText(folder)
    qtbot.keyClick(empty_screen._folder_edit, Qt.Key_Return)

    assert empty_screen._all is not None
    assert empty_screen._table.topLevelItemCount() > 0
    assert empty_screen.last_error == ""


def test_moving_a_filter_before_a_folder_is_loaded_does_nothing_quietly(
        empty_screen):
    """The filter bar is live from the start, and must survive being used.

    Its controls are connected in the constructor, so the first thing a user
    touches can easily be a spin box on an empty screen. Filtering ``None``
    would raise inside a ``valueChanged`` slot -- a crash on an idle screen.
    """
    before = empty_screen._summary.text()

    empty_screen._q_spin.setValue(0.01)
    empty_screen._guides_spin.setValue(3)
    empty_screen._query.setText("anything")

    assert empty_screen._shown is None
    assert empty_screen._table.topLevelItemCount() == 0
    assert empty_screen._summary.text() == before, \
        "nothing was filtered, so nothing is reported"


# ---------------------------------------------------------------------------
# exporting
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# investigating
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the cell formatter
# ---------------------------------------------------------------------------

def test_a_cell_that_is_not_a_number_is_shown_as_it_stands():
    """A non-numeric cell is printed, not turned into an em dash or a crash.

    The formatter runs over every cell of every row. A value that is not a
    number -- a condition label that reached a numeric column, a ``None``
    from a backend that writes no p-value -- must not take the redraw down,
    and must not be silently replaced by a missing-value marker, which would
    hide that something unexpected is in the table.
    """
    assert screen_module._number("not a number") == "not a number"
    assert screen_module._number(None) == "None"
    assert screen_module._number(float("nan")) == "—"
    assert screen_module._number(0.5) == "0.5"
    assert screen_module._number(1e-6) == "1e-06"
