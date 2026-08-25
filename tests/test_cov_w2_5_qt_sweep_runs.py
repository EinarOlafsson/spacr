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

def test_a_row_that_is_not_a_row_is_not_ok():
    """Anything that is not a mapping cannot be the loaded run."""
    assert sr._is_ok(None) is False
    assert sr._is_ok(["ok"]) is False


@pytest.mark.parametrize("status", [None, "", "   ", "nan", "NaN"])
def test_a_missing_status_counts_as_ok(status):
    """An older sweep's CSV need not carry the column at all."""
    assert sr._is_ok({"status": status}) is True
    assert sr._is_ok({}) is True


@pytest.mark.parametrize("status", ["failed", "running", "killed"])
def test_anything_else_is_not_ok(status):
    """A row claiming ok is a row a click will try to open results from."""
    assert sr._is_ok({"status": status}) is False


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


def test_a_folder_with_no_results_table_says_where_it_looked(panel, tmp_path):
    """Not a wipe: the runs this session made are still on the table."""
    handle = panel.record_run("run A", folder=str(tmp_path))
    assert handle

    assert panel.load(tmp_path) is False
    assert sr.RESULTS_FILENAME in panel._status.text()
    assert "run A" in [str(row["run"]) for _i, row in panel._frame.iterrows()]


def test_a_results_csv_that_cannot_be_parsed_is_reported(panel, tmp_path):
    """A broken CSV is a sentence over the table, not a traceback."""
    broken = tmp_path / "sweep_results.csv"
    broken.write_text('a,b\n"unterminated\n1,2,3,4\n')

    assert panel.load(tmp_path) is False
    assert "Could not read" in panel._status.text()


def test_an_empty_sweep_says_it_has_recorded_no_trials(panel):
    """An empty frame with no source still explains itself."""
    assert panel.set_frame(pd.DataFrame()) is False
    assert "recorded no trials yet" in panel._status.text()


def test_a_csv_path_is_taken_as_the_table_itself(panel, tmp_path):
    """A caller may name the file rather than the folder it is in."""
    path = tmp_path / "sweep_results.csv"
    pd.DataFrame({"trial_id": [1, 2], "status": ["ok", "failed"],
                  "seconds": [1.0, 2.0]}).to_csv(path, index=False)

    assert panel.load(path) is True
    assert len(panel._frame) == 2
    assert panel.reload() is True


# ---------------------------------------------------------------------------
# this session's runs
# ---------------------------------------------------------------------------

def test_a_stale_handle_is_refused_rather_than_inventing_a_row(panel):
    """A phantom row is worse than a missing one."""
    assert panel.update_run(999, status="ok") is False


def test_a_run_starts_as_running_and_is_not_loadable(panel, tmp_path):
    """A row claiming ok is a row a click will try to open results from."""
    folder = _run_folder(tmp_path, "ols_1")

    handle = panel.record_run("run A", folder=folder)

    row = panel._recorded[handle]
    assert row["status"] == sr.STATUS_RUNNING
    assert panel.loaded_run() is None
    assert "still going" in panel._status.text()


def test_a_settings_dict_puts_the_sweeps_own_columns_on_the_row(panel):
    """A run and a trial are two rows of one table, not two tables."""
    name = next(iter(sr.RUN_SETTING_COLUMNS))

    handle = panel.record_run("run A", settings={name: "chosen",
                                                "not_a_sweep_column": 1})

    row = panel._recorded[handle]
    assert row[name] == "chosen"
    assert "not_a_sweep_column" not in row


# ---------------------------------------------------------------------------
# opening a run off disk
# ---------------------------------------------------------------------------

def test_cancelling_the_folder_chooser_opens_nothing(panel, monkeypatch):
    """An empty answer from the dialog is a cancel."""
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))

    assert panel.load_run_from_disk() is False


def test_the_folder_chooser_answer_is_opened(panel, monkeypatch, tmp_path):
    """What the dialog returned is what is loaded."""
    folder = _run_folder(tmp_path, "ols_1")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: folder))

    assert panel.load_run_from_disk() is True
    assert panel.loaded_run_folder() == folder


def test_something_that_is_not_a_path_is_refused(panel):
    """A caller handing over the wrong type gets False, not a TypeError."""
    assert panel.load_run_from_disk(object()) is False


def test_a_folder_with_no_run_in_it_is_named(panel, tmp_path):
    """Being told nothing happened is the failure this repository keeps fixing."""
    empty = tmp_path / "not_a_run"
    empty.mkdir()

    assert panel.load_run_from_disk(str(empty)) is False
    assert str(empty) in panel._status.text()
    assert "No run in" in panel._status.text()


def test_opening_a_run_twice_is_one_row(panel, tmp_path):
    """The second open would be indistinguishable from a re-run."""
    folder = _run_folder(tmp_path, "ols_1")

    assert panel.load_run_from_disk(folder) is True
    assert panel.load_run_from_disk(folder) is True

    assert len(panel._recorded) == 1


def test_a_run_off_disk_carries_its_own_settings(panel, tmp_path,
                                                  monkeypatch):
    """Two runs cannot be compared on the settings that differ otherwise."""
    from spacr import refit

    name = next(iter(sr.RUN_SETTING_COLUMNS))
    monkeypatch.setattr(refit, "settings_of_run",
                        lambda table: {name: "from disk"})
    folder = _run_folder(tmp_path, "ols_1")

    panel.load_run_from_disk(folder)

    assert panel.loaded_run()[name] == "from disk"


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


def test_two_runs_with_the_same_basename_get_distinguishable_names(panel,
                                                                    tmp_path):
    """`ols_3` is readable and ambiguous across two screens."""
    first = _run_folder(tmp_path / "screen_a", "ols_3", seed=1)
    second = _run_folder(tmp_path / "screen_b", "ols_3", seed=2)

    panel.load_run_from_disk(first)
    panel.load_run_from_disk(second)

    names = {str(row.get("run")) for row in panel._recorded.values()}
    assert len(names) == 2
    assert "ols_3" in names
    assert any(os.sep in name for name in names)


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


def test_loading_the_selected_run_again_still_announces_it(panel, tmp_path):
    """An explicit action resynchronises the views even with no change."""
    folder = _run_folder(tmp_path, "ols_1")
    panel.load_run_from_disk(folder)
    record = panel.loaded_run()
    seen = []
    panel.trial_activated.connect(seen.append)

    assert panel.load_this_run(record) is True

    assert len(seen) == 1
    assert seen[0]["folder"] == folder


def test_double_clicking_nothing_loads_nothing(panel):
    """With no selection there is no run to open."""
    panel._on_double_click()                     # must not raise

    assert panel.loaded_run() is None


def test_the_rows_of_an_empty_panel_are_empty(panel):
    """Before any run there is nothing to read off the frame."""
    assert panel._all_rows() == []
    assert panel.selected_trial() is None
    assert panel.selected_runs() == []


# ---------------------------------------------------------------------------
# describing what is on disk
# ---------------------------------------------------------------------------

def test_a_folder_that_is_not_there_holds_nothing(tmp_path):
    """A user deciding whether to destroy a fit is told there is nothing."""
    assert SweepRunsPanel.describe_folder("") == "nothing on disk"
    assert SweepRunsPanel.describe_folder(
        str(tmp_path / "gone")) == "nothing on disk"


def test_an_empty_folder_says_it_is_empty(tmp_path):
    """Zero files is stated rather than shown as an empty list."""
    empty = tmp_path / "empty"
    empty.mkdir()

    assert SweepRunsPanel.describe_folder(str(empty)) == "an empty folder"


def test_a_folder_is_described_in_the_words_a_decision_needs(tmp_path):
    """Figures, CSVs and everything else, counted and sized."""
    folder = tmp_path / "run"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.png").write_bytes(b"x" * 10)
    (folder / "b.pdf").write_bytes(b"x" * 10)
    (folder / "c.csv").write_text("a,b\n1,2\n")
    (folder / "sub" / "notes.txt").write_text("hello")

    described = SweepRunsPanel.describe_folder(str(folder))

    assert "2 figures" in described
    assert "1 CSV" in described
    assert "1 other file" in described
    assert described.endswith("B") or described.endswith("MB")


def test_a_file_that_cannot_be_sized_is_skipped(tmp_path, monkeypatch):
    """A file that vanishes mid-walk must not stop the description."""
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"x" * 10)
    real_getsize = os.path.getsize

    def sometimes(path):
        if str(path).endswith(".png"):
            raise OSError("gone")
        return real_getsize(path)

    monkeypatch.setattr(os.path, "getsize", sometimes)

    assert SweepRunsPanel.describe_folder(str(folder)) == "an empty folder"


def test_a_size_is_readable():
    """Bytes, kB, MB — whichever the number needs."""
    assert sr._readable_size(512) == "512 B"
    assert sr._readable_size(2048).endswith("KB")
    assert sr._readable_size(5 * 1024 * 1024).endswith("MB")


# ---------------------------------------------------------------------------
# removing and deleting
# ---------------------------------------------------------------------------

def test_removing_nothing_removes_nothing(panel, tmp_path):
    """A gesture with no rows behind it is not a removal."""
    panel.load_run_from_disk(_run_folder(tmp_path, "ols_1"))

    assert panel.remove_runs([]) == 0
    assert panel.remove_runs([{"status": "ok"}]) == 0
    assert len(panel._recorded) == 1


def test_removing_a_run_takes_the_row_and_keeps_the_folder(panel, tmp_path):
    """The safe half: Reload brings it back, and the status line says so."""
    folder = _run_folder(tmp_path, "ols_1")
    panel.load_run_from_disk(folder)
    record = panel.loaded_run()
    gone = []
    panel.runs_removed.connect(gone.extend)

    assert panel.remove_runs([record]) == 1

    assert panel._recorded == {}
    assert os.path.isdir(folder), "the folder on disk is untouched"
    assert "Reload brings it back" in panel._status.text()
    assert panel.loaded_run() is None
    assert len(gone) == 1


def test_removing_a_sweep_trial_matches_it_by_trial_number(panel, tmp_path):
    """A trial's row has no folder, so the key alone would remove nothing."""
    path = tmp_path / "sweep_results.csv"
    pd.DataFrame({"trial_id": [1, 2], "status": ["ok", "ok"],
                  "seconds": [1.0, 2.0]}).to_csv(path, index=False)
    panel.load(path)
    rows = panel._all_rows()

    assert panel.remove_runs([rows[0]]) == 1

    assert len(panel._frame) == 1
    assert str(panel._frame.iloc[0]["trial_id"]) == "2"


def test_removing_every_trial_leaves_no_sweep_frame(panel, tmp_path):
    """An emptied sweep frame is None, not a zero-row table."""
    path = tmp_path / "sweep_results.csv"
    pd.DataFrame({"trial_id": [1], "status": ["ok"],
                  "seconds": [1.0]}).to_csv(path, index=False)
    panel.load(path)

    assert panel.remove_runs(panel._all_rows()) == 1
    assert panel._sweep_frame is None


def test_a_run_that_is_still_going_is_not_deleted(panel, tmp_path):
    """A folder deleted underneath a run leaves half a result."""
    folder = _run_folder(tmp_path, "ols_1")
    handle = panel.record_run("run A", folder=folder)
    record = dict(panel._recorded[handle])

    assert panel.delete_runs_from_disk([record], confirm=lambda *a: True) == 0
    assert os.path.isdir(folder)
    assert "still going and cannot be deleted" in panel._status.text()
    assert "run A" in panel._status.text()


def test_a_run_with_no_folder_has_nothing_to_delete(panel):
    """The message distinguishes "no folder" from "refused"."""
    handle = panel.record_run("run A")
    panel.update_run(handle, status="ok")
    record = dict(panel._recorded[handle])

    assert panel.delete_runs_from_disk([record],
                                       confirm=lambda *a: True) == 0
    assert "no folder on disk" in panel._status.text()


def test_declining_the_confirmation_deletes_nothing(panel, tmp_path):
    """The default answer is No: this one cannot be undone."""
    folder = _run_folder(tmp_path, "ols_1")
    panel.load_run_from_disk(folder)
    asked = []

    def refuse(message, folders):
        asked.append((message, folders))
        return False

    assert panel.delete_runs_from_disk([panel.loaded_run()],
                                       confirm=refuse) == 0
    assert os.path.isdir(folder)
    assert folder in asked[0][0]
    assert "cannot be undone" in asked[0][0]


def test_deleting_a_run_removes_the_folder_and_the_row(panel, tmp_path):
    """The unsafe half, taken only after an explicit yes."""
    folder = _run_folder(tmp_path, "ols_1")
    panel.load_run_from_disk(folder)

    assert panel.delete_runs_from_disk([panel.loaded_run()],
                                       confirm=lambda *a: True) == 1

    assert not os.path.exists(folder)
    assert panel._recorded == {}
    assert "Deleted 1 run folder" in panel._status.text()


@pytest.mark.xfail(strict=True,
                   reason="delete_runs_from_disk builds `keep` from the "
                          "decorated failure strings ('<folder> (reason)'), so "
                          "no plain folder path ever matches and a row whose "
                          "folder survived is removed from the table anyway")
def test_a_folder_that_will_not_delete_keeps_its_row(panel, tmp_path,
                                                     monkeypatch):
    """A row whose folder survived must not disappear from the table.

    The row and the folder are one claim: the folder is still on disk with
    the whole run in it, so the table has to keep offering it. Losing the row
    leaves a finished run that only Reload can find again, and the status
    line says the opposite.
    """
    import shutil

    folder = _run_folder(tmp_path, "ols_1")
    panel.load_run_from_disk(folder)

    def refuse(path):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(shutil, "rmtree", refuse)

    assert panel.delete_runs_from_disk([panel.loaded_run()],
                                       confirm=lambda *a: True) == 0

    assert "Could not delete" in panel._status.text()
    assert len(panel._recorded) == 1


# ---------------------------------------------------------------------------
# the row menu
# ---------------------------------------------------------------------------

def test_an_unknown_menu_entry_does_nothing(panel, tmp_path):
    """A menu whose exec was dismissed hands back an empty verb."""
    panel.load_run_from_disk(_run_folder(tmp_path, "ols_1"))

    assert panel._apply_run_menu("", panel._all_rows()) is False
    assert panel._apply_run_menu("load", []) is False


def test_the_menu_will_not_load_a_run_that_is_still_going(panel, tmp_path):
    """The guard is on the seam a test drives, not only on the paint."""
    handle = panel.record_run("run A", folder=_run_folder(tmp_path, "ols_1"))
    record = dict(panel._recorded[handle])

    assert panel._apply_run_menu("load", [record]) is False
    assert panel.loaded_run() is None


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


def test_the_menu_will_not_restore_a_workspace_that_is_not_there(panel,
                                                                 tmp_path):
    """The offer is withdrawn rather than emitting a request nothing answers."""
    panel.load_run_from_disk(_run_folder(tmp_path, "ols_1"))
    asked = []
    panel.workspace_restore_requested.connect(asked.append)

    assert panel._apply_run_menu("restore", [panel.loaded_run()]) is False
    assert asked == []


def test_the_menu_restores_a_workspace_that_is_there(panel, tmp_path,
                                                     monkeypatch):
    """With a bundle beside the run, the request goes out."""
    from spacr import workspace

    monkeypatch.setattr(workspace, "has_workspace", lambda folder: True)
    panel.load_run_from_disk(_run_folder(tmp_path, "ols_1"))
    asked = []
    panel.workspace_restore_requested.connect(asked.append)

    assert panel._apply_run_menu("restore", [panel.loaded_run()]) is True
    assert len(asked) == 1


def test_the_menu_can_delete_from_disk(panel, tmp_path, monkeypatch):
    """The destructive entry goes through the same confirmed path."""
    folder = _run_folder(tmp_path, "ols_1")
    panel.load_run_from_disk(folder)
    monkeypatch.setattr(SweepRunsPanel, "_confirm_deletion",
                        lambda self, message, folders: True)

    assert panel._apply_run_menu("delete", panel._all_rows()) is True
    assert not os.path.exists(folder)


# ---------------------------------------------------------------------------
# the summary line
# ---------------------------------------------------------------------------

def test_the_summary_counts_what_did_not_work(panel, tmp_path):
    """A sweep whose trials mostly failed still writes a full-looking table."""
    path = tmp_path / "sweep_results.csv"
    pd.DataFrame({"trial_id": [1, 2, 3],
                  "status": ["ok", "failed", "failed"],
                  "seconds": [1.0, 2.0, 3.0]}).to_csv(path, index=False)

    panel.load(path)

    note = panel._status.text()
    assert "3 runs" in note
    assert "2 of which did not produce a regression" in note
    assert "Loaded: trial 1" in note, "the one trial that ran is the loaded one"


def test_the_summary_separates_this_session_from_the_sweep(panel, tmp_path):
    """Two halves of one table, counted separately."""
    path = tmp_path / "sweep_results.csv"
    pd.DataFrame({"trial_id": [1, 2], "status": ["ok", "ok"],
                  "seconds": [1.0, 2.0]}).to_csv(path, index=False)
    panel.record_run("run A", folder=_run_folder(tmp_path, "ols_1"))

    panel.load(path)

    note = panel._status.text()
    assert "1 from this session" in note
    assert "2 from the sweep" in note


def test_whole_numbers_stay_whole_across_a_concat(panel, tmp_path):
    """A run has no trial number, and the missing value is real."""
    path = tmp_path / "sweep_results.csv"
    pd.DataFrame({"trial_id": [1, 2], "status": ["ok", "ok"],
                  "seconds": [1.0, 2.0]}).to_csv(path, index=False)
    panel.record_run("run A", folder=_run_folder(tmp_path, "ols_1"))

    panel.load(path)

    trials = panel._frame["trial_id"]
    assert str(trials.dtype) == "Int64"
    assert trials.isna().any(), "the session run has no trial number"


def test_a_column_that_is_not_whole_is_left_alone():
    """Only columns that are entirely whole numbers are cast back."""
    frame = pd.DataFrame({"seconds": [1.5, 2.5], "trial_id": [1.0, np.nan],
                          "run": ["a", "b"]})

    out = SweepRunsPanel._keep_whole_numbers_whole(
        frame, ["seconds", "trial_id", "run", "absent"])

    assert str(out["seconds"].dtype) == "float64"
    assert str(out["trial_id"].dtype) == "Int64"
    assert str(out["run"].dtype) == "object"


def test_no_run_is_loaded_is_said_rather_than_left_blank(panel, tmp_path):
    """Several runs and no choice made is a state the user has to resolve."""
    path = tmp_path / "sweep_results.csv"
    pd.DataFrame({"trial_id": [1, 2, 3],
                  "status": ["ok", "ok", "ok"],
                  "seconds": [1.0, 2.0, 3.0]}).to_csv(path, index=False)

    panel.load(path)

    assert "No run is loaded — pick one" in panel._status.text()


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


def test_an_announcement_for_a_row_the_table_does_not_hold_says_nothing(
        panel, tmp_path):
    """A view must never be handed a record it cannot open."""
    panel.load_run_from_disk(_run_folder(tmp_path, "ols_1"))
    seen = []
    panel.loaded_run_changed.connect(seen.append)
    panel._loaded_key = "/a/key/naming/nothing"

    assert panel._announce_the_loaded_run("") is False
    assert seen == []


def test_painting_the_mark_before_there_is_a_table_does_nothing(panel):
    """The mark is a column of a frame that does not exist yet."""
    panel._paint_the_loaded_mark()               # must not raise

    assert panel._frame is None


def test_painting_the_mark_skips_rows_the_table_does_not_carry(panel,
                                                               tmp_path):
    """A cell with no frame row behind it is left alone rather than indexed."""
    path = tmp_path / "sweep_results.csv"
    pd.DataFrame({"trial_id": [1, 2], "status": ["ok", "ok"],
                  "seconds": [1.0, 2.0]}).to_csv(path, index=False)
    panel.load(path)
    table = panel.table.table
    column = list(panel._frame.columns).index(sr.LOADED_COLUMN)
    table.item(0, column).setData(0x0100, None)
    table.item(1, column).setData(0x0100, 99)

    panel._paint_the_loaded_mark()               # must not raise

    assert table.item(0, column).text() in ("", sr.LOADED_MARK)


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
# the row menu, built and shown
# ---------------------------------------------------------------------------

def test_the_menu_offers_load_restore_remove_beside_and_delete(panel,
                                                               tmp_path):
    """Everything a user opens the menu for is in it, in that order."""
    panel.load_run_from_disk(_run_folder(tmp_path, "ols_1"))

    menu = panel._build_run_menu([panel.loaded_run()])

    verbs = [action.data() for action in menu.actions() if action.data()]
    assert verbs == ["load", "restore", "remove", "beside", "delete"]
    restore = next(a for a in menu.actions() if a.data() == "restore")
    assert not restore.isEnabled(), "no workspace bundle beside this run"
    assert "saved no workspace" in restore.toolTip()


def test_the_menu_greys_everything_a_running_run_cannot_do(panel, tmp_path):
    """Greyed and saying why, not silently ignored."""
    handle = panel.record_run("run A", folder=_run_folder(tmp_path, "ols_1"))

    menu = panel._build_run_menu([dict(panel._recorded[handle])])

    by_verb = {action.data(): action for action in menu.actions()
               if action.data()}
    assert not by_verb["load"].isEnabled()
    assert "still going" in by_verb["load"].toolTip()
    assert not by_verb["remove"].isEnabled()
    assert not by_verb["delete"].isEnabled()
    assert "beside" not in by_verb, "a running run has nothing to show beside"
    assert any("still going and cannot be deleted" in action.text()
               for action in menu.actions())


def test_a_multi_row_menu_offers_only_what_applies_to_all_of_them(panel,
                                                                  tmp_path):
    """Load, restore and beside are single-run entries."""
    panel.load_run_from_disk(_run_folder(tmp_path, "ols_1", seed=1))
    panel.load_run_from_disk(_run_folder(tmp_path, "ols_2", seed=2))

    menu = panel._build_run_menu(panel._all_rows())

    verbs = [action.data() for action in menu.actions() if action.data()]
    assert verbs == ["remove", "delete"]
    assert "2 runs" in menu.actions()[0].text()


def test_the_menu_acts_on_the_row_it_was_opened_over(panel, tmp_path,
                                                     monkeypatch):
    """Right-clicking an unselected row selects it first, then acts on it.

    The menu is stood in for because ``QMenu.exec`` is a C++ event loop: a
    test that entered it would hang rather than tell anyone what the entry
    does. Everything either side of it — the selection fallback and the
    dispatch — is the real method.
    """
    panel.load_run_from_disk(_run_folder(tmp_path, "ols_1"))
    panel.table.table.clearSelection()
    positions = []

    class OneEntry:
        def __init__(self, verb):
            self._verb = verb

        def data(self):
            return self._verb

        def exec(self, position):
            positions.append(position)
            return self

    monkeypatch.setattr(panel, "_build_run_menu",
                        lambda records: OneEntry("remove"))
    item = panel.table.table.item(0, 0)
    where = panel.table.table.visualItemRect(item).center()

    panel._run_menu(where)

    assert len(positions) == 1
    assert panel._recorded == {}, "the row under the cursor was removed"


def test_a_menu_over_empty_space_does_nothing(panel, tmp_path):
    """There is no row under the cursor, so there is nothing to act on."""
    from PySide6.QtCore import QPoint

    panel.load_run_from_disk(_run_folder(tmp_path, "ols_1"))
    panel.table.table.clearSelection()

    panel._run_menu(QPoint(4000, 4000))          # must not raise

    assert len(panel._recorded) == 1


# ---------------------------------------------------------------------------
# choosing which run is loaded
# ---------------------------------------------------------------------------

def test_choosing_a_run_by_folder_moves_the_mark_and_the_views(panel,
                                                               tmp_path):
    """Moving the mark and leaving the views on the previous run is the bug."""
    first = _run_folder(tmp_path, "ols_1", seed=1)
    second = _run_folder(tmp_path, "ols_2", seed=2)
    panel.load_run_from_disk(first)
    panel.load_run_from_disk(second)
    seen = []
    panel.loaded_run_changed.connect(lambda row: seen.append(row["folder"]))

    assert panel.set_loaded_run(first) is True

    assert panel.loaded_run_folder() == first
    assert seen == [first]
    assert "Loaded:" in panel._status.text()


def test_choosing_the_run_that_is_already_loaded_announces_nothing(panel,
                                                                   tmp_path):
    """A choice that changes nothing is not a reload of everything."""
    folder = _run_folder(tmp_path, "ols_1")
    panel.load_run_from_disk(folder)
    seen = []
    panel.loaded_run_changed.connect(seen.append)

    assert panel.set_loaded_run(folder) is True

    assert seen == []


def test_choosing_a_run_by_name_works_too(panel, tmp_path):
    """A row with no folder is named by its run label."""
    handle = panel.record_run("run A")
    panel.update_run(handle, status="ok")

    assert panel.set_loaded_run("run A") is True
    assert panel.loaded_run()["run"] == "run A"


# ---------------------------------------------------------------------------
# the loaded mark and the table's own cells
# ---------------------------------------------------------------------------

def test_painting_the_mark_skips_a_cell_that_is_not_there(panel, tmp_path):
    """A table rebuilt underneath the mark may have fewer cells than rows."""
    path = tmp_path / "sweep_results.csv"
    pd.DataFrame({"trial_id": [1, 2], "status": ["ok", "ok"],
                  "seconds": [1.0, 2.0]}).to_csv(path, index=False)
    panel.load(path)
    column = list(panel._frame.columns).index(sr.LOADED_COLUMN)
    panel.table.table.takeItem(0, column)

    panel._paint_the_loaded_mark()               # must not raise

    assert panel.table.table.item(0, column) is None


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


def test_delete_on_the_selection_removes_it_from_the_list(panel, tmp_path):
    """The safe half on the bare key; deleting from disk needs words."""
    folder = _run_folder(tmp_path, "ols_1")
    panel.load_run_from_disk(folder)
    panel.table.table.selectRow(0)

    handled = panel.eventFilter(panel.table.table, _delete_key(panel))

    assert handled is True
    assert panel._recorded == {}
    assert os.path.isdir(folder)


def test_delete_refuses_a_run_that_is_still_going(panel, tmp_path):
    """`update_run` would be left writing to a handle with nothing to show."""
    panel.record_run("run A", folder=_run_folder(tmp_path, "ols_1"))
    panel.table.table.selectRow(0)

    handled = panel.eventFilter(panel.table.table, _delete_key(panel))

    assert handled is True
    assert len(panel._recorded) == 1
    assert "still going and cannot be deleted" in panel._status.text()


def test_a_key_on_something_else_is_not_this_panels_business(panel):
    """Only the table's own Delete is intercepted."""
    assert panel.eventFilter(panel, _delete_key(panel)) is False


def test_the_menu_loads_a_finished_run(panel, tmp_path):
    """The entry a user opens the menu for actually loads."""
    first = _run_folder(tmp_path, "ols_1", seed=1)
    second = _run_folder(tmp_path, "ols_2", seed=2)
    panel.load_run_from_disk(first)
    panel.load_run_from_disk(second)

    assert panel._apply_run_menu("load", [panel._all_rows()[0]]) is True
    assert panel.loaded_run_folder() == first


def test_no_still_is_shown_until_a_provider_offers_one(panel, tmp_path):
    """A frame around nothing is worse than no frame."""
    panel.load_run_from_disk(_run_folder(tmp_path, "ols_1"))
    panel.table.table.selectRow(0)

    panel.set_photo_provider(None)

    assert panel.photograph_shown() is None


def test_a_provider_that_raises_costs_the_still_and_nothing_else(panel,
                                                                 tmp_path):
    """The table stays usable when the still cannot be fetched."""
    panel.load_run_from_disk(_run_folder(tmp_path, "ols_1"))
    panel.table.table.selectRow(0)

    def explode(folder):
        raise OSError("no still for that run")

    panel.set_photo_provider(explode)

    assert panel.photograph_shown() is None
    assert panel.selected_trial() is not None


def test_a_provider_that_offers_a_still_gets_it_painted(panel, tmp_path):
    """A real pixmap is scaled to the panel and shown."""
    from PySide6.QtGui import QPixmap

    panel.resize(400, 300)
    panel.load_run_from_disk(_run_folder(tmp_path, "ols_1"))
    panel.table.table.selectRow(0)
    still = QPixmap(64, 48)
    still.fill()

    panel.set_photo_provider(lambda folder: still)

    shown = panel.photograph_shown()
    assert shown is not None
    assert not shown.isNull()
