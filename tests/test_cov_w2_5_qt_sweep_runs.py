"""The runs tab never marks a run it cannot show, or hides one it cannot mark.

A run row is a promise that something can be opened from it, so the paths
worth driving are the ones where the promise cannot be kept: a folder that
holds no results table, a status column an old CSV never had, a delete of a
run that is still writing. Each is refused in words rather than by doing
nothing, and the words are what these tests read.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from PySide6.QtWidgets import QFileDialog                        # noqa: E402

from spacr.qt.widgets import sweep_runs as sr                    # noqa: E402
from spacr.qt.widgets.sweep_runs import SweepRunsPanel           # noqa: E402


def _coefficients(rows=5, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feature": [f"fraction:grna[{seed}_{i}]" for i in range(rows)],
        "coefficient": rng.normal(0, 0.5, rows),
        "p_value": rng.uniform(size=rows),
    })


def _run_folder(root, name, seed=1):
    """A folder shaped like a finished run: a coefficient table in it."""
    folder = root / "results" / name
    folder.mkdir(parents=True, exist_ok=True)
    _coefficients(seed=seed).to_csv(folder / "results.csv", index=False)
    return str(folder)


@pytest.fixture
def panel(qtbot):
    made = SweepRunsPanel()
    qtbot.addWidget(made)
    return made


# ---------------------------------------------------------------------------
# reading a row
# ---------------------------------------------------------------------------

def test_a_folder_that_cannot_be_asked_holds_no_workspace(monkeypatch):
    """The restore offer is withdrawn quietly rather than raising on a click."""
    from spacr import workspace

    def explode(folder):
        raise OSError("the drive went away")

    monkeypatch.setattr(workspace, "has_workspace", explode)

    assert sr._has_workspace("/anywhere") is False
    assert sr._has_workspace("") is False


def test_ordering_columns_of_nothing_is_nothing():
    """A tab opened before any run has no columns to order."""
    assert sr.ordered_columns(None) == []


def test_ordering_keeps_a_column_nobody_listed():
    """Hiding it would send the user out of the application to read it."""
    frame = pd.DataFrame(columns=["a_column_nobody_listed", "status", "run"])

    ordered = sr.ordered_columns(frame)

    assert "a_column_nobody_listed" in ordered
    assert ordered.index("run") < ordered.index("a_column_nobody_listed")


# ---------------------------------------------------------------------------
# loading the sweep's own table
# ---------------------------------------------------------------------------

def test_loading_nothing_loads_nothing(panel):
    """An empty folder argument is not a path."""
    assert panel.load("") is False
    assert panel.reload() is False


def test_a_csv_path_is_taken_as_the_table_itself(panel, tmp_path):
    """A caller may name the file rather than the folder it is in."""
    path = tmp_path / "sweep_results.csv"
    pd.DataFrame({"trial_id": [1, 2], "status": ["ok", "failed"],
                  "seconds": [1.0, 2.0]}).to_csv(path, index=False)

    assert panel.load(path) is True
    assert len(panel._frame) == 2
    assert panel.reload() is True


# ---------------------------------------------------------------------------
# opening a run off disk
# ---------------------------------------------------------------------------

def test_the_folder_chooser_answer_is_opened(panel, monkeypatch, tmp_path):
    """What the dialog returned is what is loaded."""
    folder = _run_folder(tmp_path, "ols_1")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: folder))

    assert panel.load_run_from_disk() is True
    assert panel.loaded_run_folder() == folder


def test_settings_that_cannot_be_read_leave_the_run_openable(panel, tmp_path,
                                                              monkeypatch):
    """A missing settings file costs the columns, not the run."""
    from spacr import refit

    def explode(table):
        raise ValueError("that settings file is not readable")

    monkeypatch.setattr(refit, "settings_of_run", explode)
    folder = _run_folder(tmp_path, "ols_1")

    assert panel.load_run_from_disk(folder) is True
    assert panel.loaded_run_folder() == folder


# ---------------------------------------------------------------------------
# which run is loaded
# ---------------------------------------------------------------------------

def test_a_key_naming_nothing_leaves_the_mark_where_it_was(panel, tmp_path):
    """A tick against nothing reads as an answer, so there is none."""
    folder = _run_folder(tmp_path, "ols_1")
    panel.load_run_from_disk(folder)

    assert panel.set_loaded_run(None) is False
    assert panel.set_loaded_run("   ") is False
    assert panel.set_loaded_run("/not/a/run") is False
    assert panel.loaded_run_folder() == folder


def test_a_row_that_is_not_a_dict_has_no_key():
    """The key is read off a mapping or it is empty."""
    assert SweepRunsPanel._row_key(None) == ""
    assert SweepRunsPanel._row_key(["ols_1"]) == ""
    assert SweepRunsPanel._row_key({"run": "ols_1"}) == "ols_1"
    assert SweepRunsPanel._row_key({"run": 3}) == ""


def test_loading_a_run_with_no_key_is_refused(panel):
    """A record naming neither a folder nor a run cannot be loaded."""
    assert panel.load_this_run(None) is False
    assert panel.load_this_run({"status": "ok"}) is False


def test_the_rows_of_an_empty_panel_are_empty(panel):
    """Before any run there is nothing to read off the frame."""
    assert panel._all_rows() == []
    assert panel.selected_trial() is None
    assert panel.selected_runs() == []


# ---------------------------------------------------------------------------
# describing what is on disk
# ---------------------------------------------------------------------------

def test_a_size_is_readable():
    """Bytes, kB, MB — whichever the number needs."""
    assert sr._readable_size(512) == "512 B"
    assert sr._readable_size(2048).endswith("KB")
    assert sr._readable_size(5 * 1024 * 1024).endswith("MB")


# ---------------------------------------------------------------------------
# the row menu
# ---------------------------------------------------------------------------

def test_an_unknown_menu_entry_does_nothing(panel, tmp_path):
    """A menu whose exec was dismissed hands back an empty verb."""
    panel.load_run_from_disk(_run_folder(tmp_path, "ols_1"))

    assert panel._apply_run_menu("", panel._all_rows()) is False
    assert panel._apply_run_menu("load", []) is False


def test_the_menu_can_remove_and_compare(panel, tmp_path):
    """Both entries act, and both say so through their own signal."""
    panel.load_run_from_disk(_run_folder(tmp_path, "ols_1"))
    compared = []
    panel.compare_requested.connect(compared.append)
    record = panel.loaded_run()

    assert panel._apply_run_menu("beside", [record]) is True
    assert compared[0]["folder"] == record["folder"]
    assert panel._apply_run_menu("remove", [record]) is True
    assert panel._recorded == {}


# ---------------------------------------------------------------------------
# the summary line
# ---------------------------------------------------------------------------

def test_a_column_that_is_not_whole_is_left_alone():
    """Only columns that are entirely whole numbers are cast back."""
    frame = pd.DataFrame({"seconds": [1.5, 2.5], "trial_id": [1.0, np.nan],
                          "run": ["a", "b"]})

    out = SweepRunsPanel._keep_whole_numbers_whole(
        frame, ["seconds", "trial_id", "run", "absent"])

    assert str(out["seconds"].dtype) == "float64"
    assert str(out["trial_id"].dtype) == "Int64"
    assert pd.api.types.is_string_dtype(out["run"])
    assert out["run"].tolist() == ["a", "b"]


# ---------------------------------------------------------------------------
# the mark, and what it may not point at
# ---------------------------------------------------------------------------

def test_a_third_run_of_the_same_name_falls_back_to_its_path(panel, tmp_path):
    """When the parent is taken too, the whole path is the only unique name."""
    made = []
    for index in range(3):
        folder = tmp_path / "screen" / f"copy{index}" / "results" / "ols_3"
        folder.mkdir(parents=True)
        _coefficients(seed=index).to_csv(folder / "results.csv", index=False)
        made.append(str(folder))
    panel._recorded = {
        1: {"run": "ols_3", "folder": made[0], "status": "ok"},
        2: {"run": os.path.join("results", "ols_3"), "folder": made[1],
            "status": "ok"},
    }
    panel._next_handle = 2

    assert panel._name_for_folder(made[2]) == made[2]


def test_painting_the_mark_before_there_is_a_table_does_nothing(panel):
    """The mark is a column of a frame that does not exist yet."""
    panel._paint_the_loaded_mark()               # must not raise

    assert panel._frame is None


# ---------------------------------------------------------------------------
# reading the selection back
# ---------------------------------------------------------------------------

def test_a_selection_pointing_past_the_table_is_no_selection(panel, tmp_path):
    """A stale row index must not index into a shorter frame."""
    path = tmp_path / "sweep_results.csv"
    pd.DataFrame({"trial_id": [1], "status": ["ok"],
                  "seconds": [1.0]}).to_csv(path, index=False)
    panel.load(path)
    table = panel.table.table
    table.selectRow(0)
    for column in range(table.columnCount()):
        item = table.item(0, column)
        if item is not None:
            item.setData(0x0100, 99)

    assert panel.selected_trial() is None
    assert panel.selected_runs() == []


def test_a_selected_cell_with_no_row_behind_it_is_skipped(panel, tmp_path):
    """Only cells that name a frame row contribute to the selection."""
    path = tmp_path / "sweep_results.csv"
    pd.DataFrame({"trial_id": [1], "status": ["ok"],
                  "seconds": [1.0]}).to_csv(path, index=False)
    panel.load(path)
    table = panel.table.table
    table.selectRow(0)
    item = table.item(0, 0)
    item.setData(0x0100, None)

    rows = panel.selected_runs()

    assert len(rows) == 1, "the row is still found through its other cells"


# ---------------------------------------------------------------------------
# the modal that guards the destructive half
# ---------------------------------------------------------------------------

def test_the_confirmation_defaults_to_no(panel, monkeypatch):
    """This one cannot be undone, so No is the default answer."""
    from PySide6.QtWidgets import QMessageBox

    asked = []

    def question(parent, title, message, buttons, default):
        asked.append((title, message, default))
        return default

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))

    assert panel._confirm_deletion("delete it?", ["/tmp/run"]) is False
    assert asked[0][0] == "Delete runs from disk"
    assert asked[0][2] == QMessageBox.No


def test_the_confirmation_accepts_a_yes(panel, monkeypatch):
    """A deliberate yes is the only thing that goes ahead."""
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.Yes))

    assert panel._confirm_deletion("delete it?", ["/tmp/run"]) is True


# ---------------------------------------------------------------------------
# choosing which run is loaded
# ---------------------------------------------------------------------------

def test_choosing_the_run_that_is_already_loaded_announces_nothing(panel,
                                                                   tmp_path):
    """A choice that changes nothing is not a reload of everything."""
    folder = _run_folder(tmp_path, "ols_1")
    panel.load_run_from_disk(folder)
    seen = []
    panel.loaded_run_changed.connect(seen.append)

    assert panel.set_loaded_run(folder) is True

    assert seen == []


# ---------------------------------------------------------------------------
# the loaded mark and the table's own cells
# ---------------------------------------------------------------------------

def test_a_float_column_that_cannot_be_cast_is_left_as_it_is():
    """Whole-looking numbers too large for Int64 stay floats."""
    frame = pd.DataFrame({"trial_id": [1e30, 2e30]})

    out = SweepRunsPanel._keep_whole_numbers_whole(frame, ["trial_id"])

    assert str(out["trial_id"].dtype) == "float64"


# ---------------------------------------------------------------------------
# the keyboard, and the still under the table
# ---------------------------------------------------------------------------

def _delete_key(panel):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    return QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier)


def test_a_key_on_something_else_is_not_this_panels_business(panel):
    """Only the table's own Delete is intercepted."""
    assert panel.eventFilter(panel, _delete_key(panel)) is False
