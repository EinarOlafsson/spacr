"""The Runs tab: which rows it holds, which one is loaded, and what it says.

A run row is a promise that something can be opened from it, so the paths
worth driving are the ones where the promise cannot be kept: a folder that
holds no results table, a CSV that cannot be parsed, a status column an old
CSV never had, a table that already names its own sources, a delete of a run
that is still writing, a menu or key over no row at all.  Each is refused in
words rather than by doing nothing, and the words are what these tests read.

One general test per behaviour family, each walking a case table with pytest's
``subtests`` fixture; every case builds its own panel.  ``QMenu.exec`` and the
confirmation modal are stood in for, so nothing blocks a headless run.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from PySide6.QtCore import QEvent, QPoint, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QPixmap  # noqa: E402
from PySide6.QtWidgets import QFileDialog  # noqa: E402

from spacr.qt.widgets import sweep_runs as sr  # noqa: E402
from spacr.qt.widgets.sweep_runs import SweepRunsPanel  # noqa: E402

pytestmark = pytest.mark.qt


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


def _sweep_csv(root, statuses, name="sweep_results.csv"):
    """A sweep's own table: one trial per status, numbered from 1."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    n = len(statuses)
    pd.DataFrame({"trial_id": list(range(1, n + 1)), "status": statuses,
                  "seconds": [float(i) for i in range(1, n + 1)]}
                 ).to_csv(path, index=False)
    return path


def _delete_key():
    return QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier)


@pytest.fixture
def make_panel(qtbot):
    """Build a fresh, empty Runs tab per case."""
    def make():
        panel = SweepRunsPanel()
        qtbot.addWidget(panel)
        return panel
    return make


def test_is_ok_reads_only_a_missing_or_ok_status_as_loadable(subtests):
    """A missing status is ok (old CSVs); anything else named is not."""
    cases = [
        ("not a mapping: None", None, False),
        ("not a mapping: a list", ["ok"], False),
        ("no status key", {}, True),
        ("status None", {"status": None}, True),
        ("status empty", {"status": ""}, True),
        ("status blank", {"status": "   "}, True),
        ("status nan text", {"status": "nan"}, True),
        ("status NaN text", {"status": "NaN"}, True),
        ("status float nan", {"status": float("nan")}, True),
        ("status ok", {"status": "ok"}, True),
        ("status ok padded", {"status": " ok "}, True),
        ("status failed", {"status": "failed"}, False),
        ("status running", {"status": "running"}, False),
        ("status killed", {"status": "killed"}, False),
        ("status OK upper", {"status": "OK"}, False),
    ]
    for case_id, row, expected in cases:
        with subtests.test(msg=case_id):
            assert sr._is_ok(row) is expected


def test_describe_folder_counts_what_a_delete_would_destroy(
        tmp_path, monkeypatch, subtests):
    """Figures, CSVs and other files, counted, pluralised and sized."""
    def files(spec):
        folder = tmp_path / f"run{len(list(tmp_path.iterdir()))}"
        folder.mkdir()
        for name, data in spec.items():
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return str(folder)

    cases = [
        ("an empty path", lambda: "", "nothing on disk"),
        ("a folder that is not there", lambda: str(tmp_path / "gone"),
         "nothing on disk"),
        ("an empty folder", lambda: files({}), "an empty folder"),
        ("figures, a CSV and another file, nested",
         lambda: files({"a.png": b"x" * 10, "b.pdf": b"x" * 10,
                        "c.csv": b"a,b\n1,2\n", "sub/notes.txt": b"hello"}),
         "2 figures, 1 CSV, 1 other file, 33 B"),
        ("singular and plural counts",
         lambda: files({"a.svg": b"x", "b.tsv": b"x", "c.csv": b"x",
                        "d.json": b"x", "e.log": b"x"}),
         "1 figure, 2 CSVs, 2 other files, 5 B"),
        ("sizes cross into kilobytes",
         lambda: files({"big.jpg": b"x" * 2048}), "1 figure, 2.0 KB"),
    ]
    for case_id, build, expected in cases:
        with subtests.test(msg=case_id):
            assert SweepRunsPanel.describe_folder(build()) == expected

    with subtests.test(msg="a file that cannot be sized is skipped"):
        folder = files({"a.png": b"x" * 10, "b.csv": b"x" * 3})
        real_getsize = os.path.getsize

        def sometimes(path):
            if str(path).endswith(".png"):
                raise OSError("gone")
            return real_getsize(path)

        with monkeypatch.context() as patch:
            patch.setattr(os.path, "getsize", sometimes)
            assert SweepRunsPanel.describe_folder(folder) == "1 CSV, 3 B"


def test_loading_a_sweep_table_composes_rows_or_says_why_not(
        make_panel, tmp_path, subtests):
    """A table becomes rows with runs and sources; a bad one is a sentence."""
    with subtests.test(msg="a folder with no results table says where"):
        panel = make_panel()
        assert panel.record_run("run A", folder=str(tmp_path))
        assert panel.load(tmp_path) is False
        assert sr.RESULTS_FILENAME in panel._status.text()
        assert [str(r["run"]) for _i, r in panel._frame.iterrows()] == [
            "run A"]

    with subtests.test(msg="a CSV that cannot be parsed is reported"):
        panel = make_panel()
        broken_dir = tmp_path / "broken"
        broken_dir.mkdir()
        (broken_dir / "sweep_results.csv").write_text(
            'a,b\n"unterminated\n1,2,3,4\n', encoding="utf-8")
        assert panel.load(broken_dir) is False
        assert "Could not read" in panel._status.text()

    with subtests.test(msg="an empty sweep has recorded no trials"):
        panel = make_panel()
        assert panel.set_frame(pd.DataFrame()) is False
        assert "recorded no trials yet" in panel._status.text()

    with subtests.test(msg="a table that names its sources keeps them"):
        panel = make_panel()
        named = pd.DataFrame({"trial_id": [1, 2],
                              "source": ["on disk", "on disk"],
                              "r_squared": [0.41, 0.52]})
        assert panel.set_frame(named, source="/x/named.csv") is True
        assert panel._frame["source"].tolist() == ["on disk", "on disk"]
        assert panel._frame["run"].tolist() == ["trial 1", "trial 2"]
        anonymous = pd.DataFrame({"trial_id": [7], "r_squared": [0.3]})
        assert panel.set_frame(anonymous, source="/x/plain.csv") is True
        assert panel._frame["source"].tolist() == [sr.SOURCE_SWEEP]
        assert panel._frame["run"].tolist() == ["trial 7"]

    with subtests.test(msg="whole numbers stay whole across the concat"):
        panel = make_panel()
        path = _sweep_csv(tmp_path / "whole", ["ok", "ok"])
        panel.record_run("run A", folder=_run_folder(tmp_path / "whole",
                                                     "ols_1"))
        panel.load(path)
        trials = panel._frame["trial_id"]
        assert str(trials.dtype) == "Int64"
        assert trials.isna().tolist() == [True, False, False]


def test_the_summary_line_counts_runs_failures_and_halves(
        make_panel, tmp_path, subtests):
    """The sentence over the table counts what is true of the rows it holds."""
    def from_csv(statuses, session=False):
        def load(panel, root):
            path = _sweep_csv(root, statuses)
            if session:
                panel.record_run("run A", folder=_run_folder(root, "ols_1"))
            panel.load(path)
        return load

    def from_frame(frame):
        def load(panel, _root):
            panel.set_frame(frame, source="/x/table.csv")
        return load

    cases = [
        # id, loader, starts with, present, absent
        ("failures are counted",
         from_csv(["ok", "failed", "failed"]), "3 runs",
         ["2 of which did not produce a regression",
          "Loaded: trial 1"], []),
        ("this session and the sweep are counted apart",
         from_csv(["ok", "ok"], session=True), "3 runs",
         ["1 from this session", "2 from the sweep"], []),
        ("several finished runs and no choice made",
         from_csv(["ok", "ok", "ok"]), "3 runs",
         ["No run is loaded — pick one"], ["did not produce"]),
        ("no status column means no failures",
         from_frame(pd.DataFrame({"trial_id": [1, 2],
                                  "r_squared": [0.4, 0.5]})),
         "2 runs", [], ["did not produce a regression", "still going"]),
        ("a status column is counted, running apart",
         from_frame(pd.DataFrame({"trial_id": [1, 2, 3],
                                  "status": ["ok", "failed",
                                             sr.STATUS_RUNNING],
                                  "r_squared": [0.4, 0.5, 0.6]})),
         "3 runs", ["1 of which did not produce a regression",
                    "1 still going"], []),
    ]
    for index, (case_id, load, start, present, absent) in enumerate(cases):
        with subtests.test(msg=case_id):
            panel = make_panel()
            load(panel, tmp_path / f"summary{index}")
            said = panel._status.text()
            assert said.startswith(start), said
            for text in present:
                assert text in said, said
            for text in absent:
                assert text not in said, said


def test_session_and_disk_runs_become_one_row_each(
        make_panel, tmp_path, monkeypatch, subtests):
    """Recording and opening runs: one row per run, refusals in words."""
    from spacr import refit

    name = next(iter(sr.RUN_SETTING_COLUMNS))

    with subtests.test(msg="a stale handle is refused"):
        assert make_panel().update_run(999, status="ok") is False

    with subtests.test(msg="a recorded run starts running and unloadable"):
        panel = make_panel()
        handle = panel.record_run("run A",
                                  folder=_run_folder(tmp_path / "rec", "ols_1"))
        assert panel._recorded[handle]["status"] == sr.STATUS_RUNNING
        assert panel.loaded_run() is None
        assert "still going" in panel._status.text()

    with subtests.test(msg="a settings dict puts only the sweep's columns"):
        panel = make_panel()
        handle = panel.record_run("run A", settings={name: "chosen",
                                                    "not_a_sweep_column": 1})
        row = panel._recorded[handle]
        assert row[name] == "chosen"
        assert "not_a_sweep_column" not in row

    with subtests.test(msg="cancelling the folder chooser opens nothing"):
        panel = make_panel()
        with monkeypatch.context() as patch:
            patch.setattr(QFileDialog, "getExistingDirectory",
                          staticmethod(lambda *a, **k: ""))
            assert panel.load_run_from_disk() is False
        assert panel._recorded == {}

    with subtests.test(msg="something that is not a path is refused"):
        panel = make_panel()
        assert panel.load_run_from_disk(object()) is False
        assert panel._recorded == {}

    with subtests.test(msg="a folder with no run in it is named"):
        panel = make_panel()
        empty = tmp_path / "not_a_run"
        empty.mkdir()
        assert panel.load_run_from_disk(str(empty)) is False
        assert str(empty) in panel._status.text()
        assert "No run in" in panel._status.text()

    with subtests.test(msg="opening a run twice is one row"):
        panel = make_panel()
        folder = _run_folder(tmp_path / "twice", "ols_1")
        assert panel.load_run_from_disk(folder) is True
        assert panel.load_run_from_disk(folder) is True
        assert len(panel._recorded) == 1
        assert panel.loaded_run_folder() == folder

    with subtests.test(msg="a run off disk carries its own settings"):
        panel = make_panel()
        with monkeypatch.context() as patch:
            patch.setattr(refit, "settings_of_run",
                          lambda table: {name: "from disk"})
            panel.load_run_from_disk(_run_folder(tmp_path / "own", "ols_1"))
        assert panel.loaded_run()[name] == "from disk"

    with subtests.test(msg="two runs with one basename get two names"):
        panel = make_panel()
        first = _run_folder(tmp_path / "screen_a", "ols_3", seed=1)
        second = _run_folder(tmp_path / "screen_b", "ols_3", seed=2)
        panel.load_run_from_disk(first)
        panel.load_run_from_disk(second)
        names = sorted(str(row.get("run")) for row in panel._recorded.values())
        assert names == sorted(["ols_3", os.path.join("results", "ols_3")])


def test_choosing_the_loaded_run_moves_the_mark_and_announces_it(
        make_panel, tmp_path, subtests):
    """A choice that resolves moves mark and views; anything else says nothing."""
    with subtests.test(msg="by folder"):
        panel = make_panel()
        first = _run_folder(tmp_path / "pick", "ols_1", seed=1)
        second = _run_folder(tmp_path / "pick", "ols_2", seed=2)
        panel.load_run_from_disk(first)
        panel.load_run_from_disk(second)
        seen = []
        panel.loaded_run_changed.connect(lambda row: seen.append(row["folder"]))
        assert panel.set_loaded_run(first) is True
        assert panel.loaded_run_folder() == first
        assert seen == [first]
        assert "Loaded:" in panel._status.text()

    with subtests.test(msg="by name, for a run with no folder"):
        panel = make_panel()
        handle = panel.record_run("run A")
        panel.update_run(handle, status="ok")
        assert panel.set_loaded_run("run A") is True
        assert panel.loaded_run()["run"] == "run A"

    with subtests.test(msg="loading the loaded run again still announces it"):
        panel = make_panel()
        folder = _run_folder(tmp_path / "again", "ols_1")
        panel.load_run_from_disk(folder)
        seen = []
        panel.trial_activated.connect(seen.append)
        assert panel.load_this_run(panel.loaded_run()) is True
        assert [row["folder"] for row in seen] == [folder]

    with subtests.test(msg="double-clicking nothing loads nothing"):
        panel = make_panel()
        panel._on_double_click()
        assert panel.loaded_run() is None

    with subtests.test(msg="an announcement for a row not held says nothing"):
        panel = make_panel()
        panel.load_run_from_disk(_run_folder(tmp_path / "ghost", "ols_1"))
        seen = []
        panel.loaded_run_changed.connect(seen.append)
        panel._loaded_key = "/a/key/naming/nothing"
        assert panel._announce_the_loaded_run("") is False
        assert seen == []


def test_removing_and_deleting_runs_touch_only_what_was_asked(
        make_panel, tmp_path, monkeypatch, subtests):
    """Remove takes rows and keeps folders; delete needs a yes and a finished run."""
    yes = lambda *a: True  # noqa: E731

    with subtests.test(msg="removing nothing removes nothing"):
        panel = make_panel()
        panel.load_run_from_disk(_run_folder(tmp_path / "none", "ols_1"))
        assert panel.remove_runs([]) == 0
        assert panel.remove_runs([{"status": "ok"}]) == 0
        assert len(panel._recorded) == 1

    with subtests.test(msg="removing a run keeps its folder"):
        panel = make_panel()
        folder = _run_folder(tmp_path / "keep", "ols_1")
        panel.load_run_from_disk(folder)
        gone = []
        panel.runs_removed.connect(gone.extend)
        assert panel.remove_runs([panel.loaded_run()]) == 1
        assert panel._recorded == {}
        assert os.path.isdir(folder)
        assert "Reload brings it back" in panel._status.text()
        assert panel.loaded_run() is None
        assert [row["folder"] for row in gone] == [folder]

    with subtests.test(msg="a sweep trial is matched by its trial number"):
        panel = make_panel()
        panel.load(_sweep_csv(tmp_path / "trial", ["ok", "ok"]))
        assert panel.remove_runs([panel._all_rows()[0]]) == 1
        assert [str(t) for t in panel._frame["trial_id"]] == ["2"]

    with subtests.test(msg="removing every trial leaves no sweep frame"):
        panel = make_panel()
        panel.load(_sweep_csv(tmp_path / "every", ["ok"]))
        assert panel.remove_runs(panel._all_rows()) == 1
        assert panel._sweep_frame is None

    with subtests.test(msg="a run still going is not deleted"):
        panel = make_panel()
        folder = _run_folder(tmp_path / "going", "ols_1")
        handle = panel.record_run("run A", folder=folder)
        record = dict(panel._recorded[handle])
        assert panel.delete_runs_from_disk([record], confirm=yes) == 0
        assert os.path.isdir(folder)
        assert "still going and cannot be deleted" in panel._status.text()
        assert "run A" in panel._status.text()

    with subtests.test(msg="a run with no folder has nothing to delete"):
        panel = make_panel()
        handle = panel.record_run("run A")
        panel.update_run(handle, status="ok")
        record = dict(panel._recorded[handle])
        assert panel.delete_runs_from_disk([record], confirm=yes) == 0
        assert "no folder on disk" in panel._status.text()
        assert len(panel._recorded) == 1

    with subtests.test(msg="declining the confirmation deletes nothing"):
        panel = make_panel()
        folder = _run_folder(tmp_path / "decline", "ols_1")
        panel.load_run_from_disk(folder)
        asked = []

        def refuse(message, folders):
            asked.append((message, folders))
            return False

        assert panel.delete_runs_from_disk([panel.loaded_run()],
                                           confirm=refuse) == 0
        assert os.path.isdir(folder)
        assert len(asked) == 1
        assert folder in asked[0][0]
        assert "cannot be undone" in asked[0][0]

    with subtests.test(msg="a yes deletes the folder and the row"):
        panel = make_panel()
        folder = _run_folder(tmp_path / "delete", "ols_1")
        panel.load_run_from_disk(folder)
        assert panel.delete_runs_from_disk([panel.loaded_run()],
                                           confirm=yes) == 1
        assert not os.path.exists(folder)
        assert panel._recorded == {}
        assert "Deleted 1 run folder" in panel._status.text()

    with subtests.test(msg="a folder that will not delete keeps its row"):
        panel = make_panel()
        folder = _run_folder(tmp_path / "stuck", "ols_1")
        panel.load_run_from_disk(folder)

        def refuse_rmtree(path):
            raise OSError(13, "Permission denied")

        with monkeypatch.context() as patch:
            patch.setattr(shutil, "rmtree", refuse_rmtree)
            assert panel.delete_runs_from_disk([panel.loaded_run()],
                                               confirm=yes) == 0
        assert "Could not delete" in panel._status.text()
        assert len(panel._recorded) == 1
        assert os.path.isdir(folder)


def test_the_run_menu_and_delete_key_offer_only_what_applies(
        make_panel, tmp_path, monkeypatch, subtests):
    """Entries per selection, greyed with a reason, and each one really acts."""
    from spacr import workspace

    def verbs(menu):
        return [action.data() for action in menu.actions() if action.data()]

    with subtests.test(msg="one finished run: every entry, in order"):
        panel = make_panel()
        panel.load_run_from_disk(_run_folder(tmp_path / "one", "ols_1"))
        menu = panel._build_run_menu([panel.loaded_run()])
        assert verbs(menu) == ["load", "save_state", "restore", "remove",
                               "beside", "delete"]
        restore = next(a for a in menu.actions() if a.data() == "restore")
        assert restore.isEnabled() is False
        assert "saved no workspace" in restore.toolTip()

    with subtests.test(msg="a running run: greyed and saying why"):
        panel = make_panel()
        handle = panel.record_run("run A",
                                  folder=_run_folder(tmp_path / "run", "ols_1"))
        menu = panel._build_run_menu([dict(panel._recorded[handle])])
        by_verb = {a.data(): a for a in menu.actions() if a.data()}
        assert by_verb["load"].isEnabled() is False
        assert "still going" in by_verb["load"].toolTip()
        assert by_verb["remove"].isEnabled() is False
        assert by_verb["delete"].isEnabled() is False
        assert "beside" not in by_verb
        assert any("still going and cannot be deleted" in a.text()
                   for a in menu.actions())

    with subtests.test(msg="several runs: only what applies to all"):
        panel = make_panel()
        panel.load_run_from_disk(_run_folder(tmp_path / "many", "ols_1", 1))
        panel.load_run_from_disk(_run_folder(tmp_path / "many", "ols_2", 2))
        menu = panel._build_run_menu(panel._all_rows())
        assert verbs(menu) == ["save_state", "remove", "delete"]
        assert "2 runs" in menu.actions()[0].text()

    with subtests.test(msg="load refuses a running run, takes a finished one"):
        panel = make_panel()
        handle = panel.record_run("run A",
                                  folder=_run_folder(tmp_path / "ld", "ols_0"))
        assert panel._apply_run_menu(
            "load", [dict(panel._recorded[handle])]) is False
        assert panel.loaded_run() is None
        panel = make_panel()
        first = _run_folder(tmp_path / "ld", "ols_1", seed=1)
        second = _run_folder(tmp_path / "ld", "ols_2", seed=2)
        panel.load_run_from_disk(first)
        panel.load_run_from_disk(second)
        assert panel._apply_run_menu("load", [panel._all_rows()[0]]) is True
        assert panel.loaded_run_folder() == first

    for case_id, bundled, applied, requests in (
            ("restore withdraws a workspace that is not there", False, False,
             0),
            ("restore requests a workspace that is there", True, True, 1)):
        with subtests.test(msg=case_id):
            panel = make_panel()
            asked = []
            panel.workspace_restore_requested.connect(asked.append)
            with monkeypatch.context() as patch:
                if bundled:
                    patch.setattr(workspace, "has_workspace",
                                  lambda folder: True)
                panel.load_run_from_disk(
                    _run_folder(tmp_path / f"ws{requests}", "ols_1"))
                assert panel._apply_run_menu(
                    "restore", [panel.loaded_run()]) is applied
            assert len(asked) == requests

    with subtests.test(msg="delete goes through the confirmed path"):
        panel = make_panel()
        folder = _run_folder(tmp_path / "del", "ols_1")
        panel.load_run_from_disk(folder)
        with monkeypatch.context() as patch:
            patch.setattr(SweepRunsPanel, "_confirm_deletion",
                          lambda self, message, folders: True)
            assert panel._apply_run_menu("delete", panel._all_rows()) is True
        assert not os.path.exists(folder)

    class _OneEntry:
        def __init__(self, verb, shown):
            self._verb, self._shown = verb, shown

        def data(self):
            return self._verb

        def exec(self, position):
            self._shown.append(position)
            return self

    with subtests.test(msg="the menu acts on the row it was opened over"):
        panel = make_panel()
        panel.load_run_from_disk(_run_folder(tmp_path / "over", "ols_1"))
        panel.table.table.clearSelection()
        shown = []
        with monkeypatch.context() as patch:
            patch.setattr(panel, "_build_run_menu",
                          lambda records: _OneEntry("remove", shown))
            item = panel.table.table.item(0, 0)
            panel._run_menu(panel.table.table.visualItemRect(item).center())
        assert len(shown) == 1
        assert panel._recorded == {}

    with subtests.test(msg="a menu over empty space does nothing"):
        panel = make_panel()
        panel.load_run_from_disk(_run_folder(tmp_path / "empty", "ols_1"))
        panel.table.table.clearSelection()
        shown = []
        with monkeypatch.context() as patch:
            patch.setattr(panel, "_build_run_menu",
                          lambda records: _OneEntry("remove", shown))
            panel._run_menu(QPoint(4000, 4000))
        assert shown == []
        assert len(panel._recorded) == 1

    with subtests.test(msg="Delete on the selection removes it from the list"):
        panel = make_panel()
        folder = _run_folder(tmp_path / "key", "ols_1")
        panel.load_run_from_disk(folder)
        panel.table.table.selectRow(0)
        assert panel.eventFilter(panel.table.table, _delete_key()) is True
        assert panel._recorded == {}
        assert os.path.isdir(folder)

    with subtests.test(msg="Delete refuses a run that is still going"):
        panel = make_panel()
        panel.record_run("run A", folder=_run_folder(tmp_path / "kr", "ols_1"))
        panel.table.table.selectRow(0)
        assert panel.eventFilter(panel.table.table, _delete_key()) is True
        assert len(panel._recorded) == 1
        assert "still going and cannot be deleted" in panel._status.text()


def test_the_mark_and_the_still_skip_what_is_not_there(
        make_panel, tmp_path, subtests):
    """Painting the loaded mark or the still never indexes a missing part."""
    with subtests.test(msg="a cell whose row is missing or out of range"):
        panel = make_panel()
        panel.load(_sweep_csv(tmp_path / "mark", ["ok", "ok"]))
        table = panel.table.table
        column = list(panel._frame.columns).index(sr.LOADED_COLUMN)
        table.item(0, column).setData(0x0100, None)
        table.item(1, column).setData(0x0100, 99)
        panel._paint_the_loaded_mark()
        assert table.item(0, column).text() in ("", sr.LOADED_MARK)

    with subtests.test(msg="a cell that is not there"):
        panel = make_panel()
        panel.load(_sweep_csv(tmp_path / "cell", ["ok", "ok"]))
        column = list(panel._frame.columns).index(sr.LOADED_COLUMN)
        panel.table.table.takeItem(0, column)
        panel._paint_the_loaded_mark()
        assert panel.table.table.item(0, column) is None

    def provider_raising(folder):
        raise OSError("no still for that run")

    still = QPixmap(64, 48)
    still.fill()
    cases = [
        ("no provider shows no still", None, False),
        ("a provider that raises costs only the still", provider_raising,
         False),
        ("a provider's still is painted", lambda folder: still, True),
    ]
    for case_id, provider, painted in cases:
        with subtests.test(msg=case_id):
            panel = make_panel()
            panel.resize(400, 300)
            panel.load_run_from_disk(
                _run_folder(tmp_path / case_id.replace(" ", "_"), "ols_1"))
            panel.table.table.selectRow(0)
            panel.set_photo_provider(provider)
            shown = panel.photograph_shown()
            if painted:
                assert shown is not None and not shown.isNull()
            else:
                assert shown is None
            assert panel.selected_trial() is not None
