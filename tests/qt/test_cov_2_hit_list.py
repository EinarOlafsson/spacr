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


@pytest.fixture()
def loaded_screen(qtbot, folder):
    """The screen with a real list on it, built inline."""
    widget = screen_module.HitListScreen(folder=folder, threaded=False,
                                         regression_type="ols")
    qtbot.addWidget(widget)
    return widget


class _NoPathDialog:
    """A file dialog the user cancelled."""

    @staticmethod
    def getSaveFileName(*_args, **_kwargs):        # noqa: N802 - Qt spelling
        return "", ""


def _dialog_returning(path):
    class _Dialog:
        """A file dialog the user chose ``path`` in."""

        @staticmethod
        def getSaveFileName(*_args, **_kwargs):    # noqa: N802 - Qt spelling
            return str(path), ""

    return _Dialog


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

def test_export_before_a_list_exists_says_so_and_opens_no_dialog(
        empty_screen, monkeypatch):
    """The export buttons must not put up a file dialog with nothing to write.

    Asking where to save and then writing an empty file is worse than
    refusing: the user ends up with a plausible-looking CSV of nothing. The
    stand-in dialog fails the test if it is reached at all.
    """
    class _Forbidden:
        @staticmethod
        def getSaveFileName(*_args, **_kwargs):    # noqa: N802 - Qt spelling
            raise AssertionError("a dialog was opened with nothing to export")

    monkeypatch.setattr(screen_module, "QFileDialog", _Forbidden)

    empty_screen._ask_and_export("csv", "Export hit list", "CSV (*.csv)")

    assert "no hit list to export yet" in empty_screen._summary.text()


def test_a_cancelled_export_dialog_writes_nothing(loaded_screen, tmp_path,
                                                  monkeypatch):
    """Cancelling the save dialog leaves the folder exactly as it was.

    The empty path a cancelled dialog returns must not be treated as a
    filename; writing to ``""`` raises inside a click handler.
    """
    monkeypatch.setattr(screen_module, "QFileDialog", _NoPathDialog)
    before = loaded_screen._summary.text()

    loaded_screen._ask_and_export("csv", "Export hit list", "CSV (*.csv)")

    assert list(tmp_path.glob("*.csv")) == []
    assert loaded_screen._summary.text() == before


def test_choosing_a_path_in_the_dialog_writes_the_filtered_rows(
        loaded_screen, tmp_path, monkeypatch):
    """The button writes what is on screen, through the path the dialog gave.

    Both halves matter: the file has to land where the user pointed, and it
    has to hold the filtered rows rather than the whole list.
    """
    target = tmp_path / "hits.csv"
    monkeypatch.setattr(screen_module, "QFileDialog", _dialog_returning(target))
    loaded_screen._direction.setCurrentText("up")
    shown = len(loaded_screen._shown)
    assert 0 < shown < len(loaded_screen._all), "the filter has to bite"

    loaded_screen._ask_and_export("csv", "Export hit list", "CSV (*.csv)")

    assert target.exists()
    assert len(pd.read_csv(target)) == shown
    assert "written to hits.csv" in loaded_screen._summary.text()


# ---------------------------------------------------------------------------
# investigating
# ---------------------------------------------------------------------------

def test_investigating_with_nothing_selected_asks_for_a_selection(
        loaded_screen):
    """The button is always enabled, so it has to answer with no row selected.

    Nothing is emitted: the workbench must never open on a gene the user did
    not pick.
    """
    loaded_screen._table.setCurrentItem(None)
    emitted = []
    loaded_screen.investigate_requested.connect(emitted.append)

    loaded_screen._on_investigate_selected()

    assert "Select one hit to investigate." in loaded_screen._summary.text()
    assert emitted == []


def test_investigating_a_row_the_filters_have_since_dropped_says_so(
        loaded_screen):
    """A selection can outlive the list it was made from.

    The rows on screen and the filtered list are two objects, and a filter
    applied between the click and the handler leaves the table showing a gene
    the list no longer holds. Emitting anyway would open the workbench on a
    gene with no evidence behind it, so the handler names the gene and stops.

    The stale state is set directly because that is the only way to hold it
    still: redrawing the table is what normally clears it.
    """
    item = loaded_screen._table.topLevelItem(0)
    gene = str(item.data(0, Qt.UserRole))
    loaded_screen._table.setCurrentItem(item)
    loaded_screen._shown = loaded_screen._all.filter(query="no-such-gene-at-all")
    assert loaded_screen._shown.gene(gene) is None

    emitted = []
    loaded_screen.investigate_requested.connect(emitted.append)

    loaded_screen._on_investigate_selected()

    message = loaded_screen._summary.text()
    assert gene in message
    assert "no longer in the filtered list" in message
    assert emitted == []


def test_investigating_a_live_row_emits_its_evidence(loaded_screen):
    """The other half: a selected row that is still there is handed over whole.

    The workbench needs the provenance -- folder, guides, well support, FDR --
    not just the gene name, because it resolves the hit back to single cells.
    """
    item = loaded_screen._table.topLevelItem(0)
    gene = str(item.data(0, Qt.UserRole))
    loaded_screen._table.setCurrentItem(item)
    emitted = []
    loaded_screen.investigate_requested.connect(emitted.append)

    loaded_screen._on_investigate_selected()

    assert len(emitted) == 1
    payload = emitted[0]
    assert payload["gene"] == gene
    assert payload["folder"] == loaded_screen._shown.source
    assert set(payload) >= {"effect", "guides", "guide_agreement", "n_guides",
                            "well_support", "fdr", "phenotype"}


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
