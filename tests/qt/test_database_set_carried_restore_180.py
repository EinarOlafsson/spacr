"""Copied workspace databases remain real, ordered inputs to the restored set."""
import copy
from pathlib import Path
import shutil
import sqlite3
import threading

import pytest

pytest.importorskip("PySide6")
from spacr import workspace
from spacr.qt.widgets import database_set as module


def _plate(root, name):
    folder = root / name
    database = folder / "measurements" / "measurements.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, area REAL)")
        connection.execute("INSERT INTO cell VALUES (?, 'r1', 'c1', 12.0)", (name,))
    return folder, database


def _saved(qtbot, tmp_path, names=("plate1",)):
    plates = [_plate(tmp_path, name) for name in names]
    widget = module.DatabaseSetWidget(value=[str(p) for p, _ in plates], mode="folder", threaded=False)
    qtbot.addWidget(widget)
    run = tmp_path / "saved"
    workspace.save(run, workspace.collect({"databases": widget}), mode="copy")
    document = workspace.load(run)
    relocated = tmp_path / "relocated"
    shutil.move(run, relocated)
    return widget, plates, relocated, document


@pytest.mark.parametrize("remove_folder", [True, False])
def test_relocated_database_is_read_without_guessing_a_source_folder(qtbot, tmp_path, remove_folder):
    widget, plates, run, document = _saved(qtbot, tmp_path)
    if remove_folder:
        shutil.rmtree(plates[0][0])
    else:
        plates[0][1].unlink()
    before = copy.deepcopy(document)
    report = workspace.restore({"databases": widget}, document, run_dir=run)
    assert report["restored"] == ["databases"]
    carried = next(row["path"] for row in report["files"] if row["state"] == "carried")
    assert widget.database_paths() == [carried]
    assert widget.sources() == [str(plates[0][0])]
    assert widget.workspace_state()["databases"] == [carried]
    with sqlite3.connect(f"file:{widget.database_paths()[0]}?mode=ro", uri=True) as connection:
        assert connection.execute("SELECT area FROM cell").fetchall() == [(12.0,)]
    assert widget.plan() is not None
    assert document == before


def test_existing_and_carried_databases_keep_saved_order(qtbot, tmp_path):
    widget, plates, run, document = _saved(qtbot, tmp_path, ("plate1", "plate2"))
    shutil.rmtree(plates[0][0])
    report = workspace.restore({"databases": widget}, document, run_dir=run)
    carried = next(row["path"] for row in report["files"] if row["state"] == "carried")
    assert widget.database_paths() == [carried, str(plates[1][1])]
    assert widget.sources() == [str(p) for p, _ in plates]


@pytest.mark.parametrize("damage", ["corrupt", "unverified", "escape"])
def test_unusable_copies_are_not_attached(qtbot, tmp_path, damage):
    widget, plates, run, document = _saved(qtbot, tmp_path)
    shutil.rmtree(plates[0][0])
    record = next(row for row in document["files"] if row.get("copied"))
    if damage == "corrupt":
        (run / record["copied"]).write_bytes(b"not the saved database")
    elif damage == "unverified":
        record.pop("sha256", None)
    else:
        record["copied"] = "../outside.db"
    report = workspace.restore({"databases": widget}, document, run_dir=run)
    assert not widget.sources()
    assert not widget.database_paths()
    assert report["restored"] == []
    assert report["skipped"]
    assert not any(row["state"] == "carried" for row in report["files"])


@pytest.mark.parametrize("shape", ["misaligned", "duplicate"])
def test_missing_reference_does_not_use_an_ambiguous_database_list(qtbot, tmp_path, shape):
    folder, database = _plate(tmp_path, "plate1")
    widget = module.DatabaseSetWidget(mode="folder", threaded=False)
    qtbot.addWidget(widget)
    sources = [str(tmp_path / "gone")] * (2 if shape == "duplicate" else 1)
    assert not widget.apply_workspace_state({"sources": sources, "databases": [str(database), str(database)]})
    assert not widget.database_paths()


@pytest.mark.parametrize("action", ["replace", "remove_readd", "clear", "add", "shutdown"])
def test_pending_restore_respects_current_selection(qtbot, tmp_path, monkeypatch, action):
    from PySide6.QtCore import QThread

    original, _ = _plate(tmp_path, "original")
    replacement, _ = _plate(tmp_path, "replacement")
    entered, release = threading.Event(), threading.Event()
    worker_threads = []
    real = module._restored_source_paths
    def blocked(*args):
        worker_threads.append(QThread.currentThread())
        entered.set()
        assert release.wait(5)
        return [], {}
    monkeypatch.setattr(module, "_restored_source_paths", blocked)
    widget = module.DatabaseSetWidget(mode="folder", threaded=True)
    qtbot.addWidget(widget)
    try:
        assert widget.apply_workspace_state({"sources": [str(original)]})
        qtbot.waitUntil(entered.is_set, timeout=5000)
        assert worker_threads[0] != widget.thread()
        if action == "replace":
            widget.set_value([str(original), str(replacement)])
            expected = [str(original), str(replacement)]
        elif action == "remove_readd":
            assert widget.remove_source(str(original))
            widget.add_sources([str(original)])
            expected = [str(original)]
        elif action == "clear":
            widget.clear()
            expected = []
        elif action == "add":
            widget.add_sources([str(replacement)])
            expected = [str(replacement)]
        else:
            release.set()
            widget.shutdown()
            expected = [str(original)]
        release.set()
        qtbot.waitUntil(lambda: not widget._presence_jobs.active_jobs(), timeout=5000)
        assert widget.sources() == expected
    finally:
        release.set()
        widget.shutdown()


def test_removing_one_pending_source_keeps_the_other_carried_database(qtbot, tmp_path, monkeypatch):
    first, first_db = _plate(tmp_path, "first")
    second, second_db = _plate(tmp_path, "second")
    carried_first, carried_second = tmp_path / "first.db", tmp_path / "second.db"
    shutil.copy2(first_db, carried_first)
    shutil.copy2(second_db, carried_second)
    shutil.rmtree(first)
    shutil.rmtree(second)
    entered, release = threading.Event(), threading.Event()
    real = module._restored_source_paths
    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return real(*args)
    monkeypatch.setattr(module, "_restored_source_paths", blocked)
    widget = module.DatabaseSetWidget(mode="folder", threaded=True)
    qtbot.addWidget(widget)
    try:
        widget.apply_workspace_state({"mode": "folder", "sources": [str(first), str(second)],
                                      "databases": [str(carried_first), str(carried_second)]})
        qtbot.waitUntil(entered.is_set, timeout=5000)
        assert widget.remove_source(str(first))
        release.set()
        qtbot.waitUntil(lambda: widget.database_paths() == [str(carried_second)], timeout=5000)
        assert widget.sources() == [str(second)]
    finally:
        release.set()
        widget.shutdown()
