"""Run Compare lists the runs of a project folder that was moved.

Lesson 50 on the workstation: the snapshot project was prepared on another
machine, so every row of its ``artifacts.db`` recorded that machine's
absolute project root and output paths. Picking the folder at its new place
listed no runs, because the registry was filtered by the recorded root.
"""
from __future__ import annotations

import os
import shutil
import sqlite3

import pytest

from spacr import run_compare as rc
from spacr.artifacts import ARTIFACTS_DB_ENV, Registry


def _measurements(path, per_field):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute('CREATE TABLE "cell" (plateID TEXT, rowID TEXT, '
                           'columnID TEXT, fieldID TEXT, object_label INTEGER)')
        connection.executemany(
            'INSERT INTO "cell" VALUES (?, ?, ?, ?, ?)',
            [("plate1", "r1", "c1", "f1", i) for i in range(per_field)])
        connection.commit()
    finally:
        connection.close()


def _two_runs(root):
    registry = Registry(project=str(root))
    for run_id, objects in (("run-a", 3), ("run-b", 5)):
        db = root / "snapshots" / run_id / "measurements.db"
        _measurements(str(db), objects)
        registry.register(module="measure", kind="measurements-db",
                          path=str(db), role="db", run_id=run_id,
                          settings={"cell_diameter": 30})
    return registry


@pytest.fixture
def moved(tmp_path, monkeypatch):
    monkeypatch.delenv(ARTIFACTS_DB_ENV, raising=False)
    first = tmp_path / "prepared_elsewhere" / "project"
    first.mkdir(parents=True)
    _two_runs(first)
    there = tmp_path / "workstation" / "project"
    there.parent.mkdir()
    shutil.move(str(first), str(there))
    return there


def test_a_moved_project_still_lists_its_runs(moved):
    runs = rc.runs_in(Registry(project=str(moved), create=False), str(moved))

    assert sorted(run.run_id for run in runs) == ["run-a", "run-b"]
    for run in runs:
        assert run.project == str(moved)
        for artifact in run.artifacts:
            assert artifact.path.startswith(str(moved) + os.sep)
            assert os.path.isfile(artifact.path)


def test_a_moved_project_compares_from_its_new_place(moved):
    runs = {run.run_id: run for run in rc.runs_in(
        Registry(project=str(moved), create=False), str(moved))}

    comparison = rc.compare_runs(runs["run-a"], runs["run-b"], force=True)

    overall = {row.metric: (row.a, row.b) for row in comparison.counts.rows
               if row.scope == "overall"}
    assert overall["cell"] == (3, 5)


def test_a_shared_registry_is_still_filtered_by_project(tmp_path, monkeypatch):
    shared = tmp_path / "shared.db"
    monkeypatch.setenv(ARTIFACTS_DB_ENV, str(shared))
    one = tmp_path / "one"
    one.mkdir()
    _two_runs(one)
    other = tmp_path / "other"
    other.mkdir()

    assert rc.runs_in(Registry(project=str(other)), str(other)) == []


def test_the_screen_lists_the_runs_of_a_moved_project(moved, qtbot):
    pytest.importorskip("PySide6")
    from spacr.qt.screens.run_compare import RunCompareScreen

    page = RunCompareScreen()
    qtbot.addWidget(page)
    page.load_project(str(moved))

    assert sorted(run.run_id for run in page.runs()) == ["run-a", "run-b"]
    assert page._a_combo.count() == 2
    assert page.comparison() is not None


def test_relocation_preserves_foreign_or_unlocated_artifacts_and_refuses_other_registry(tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    monkeypatch.delenv(ARTIFACTS_DB_ENV, raising=False)
    root = tmp_path / "project"
    registry = _two_runs(root)
    sample = next(iter(registry.all()))
    old_root = str(tmp_path / "old-root")
    external = str(tmp_path / "outside" / "result.csv")
    old_result = str(tmp_path / "old-root" / "run" / "result.csv")
    rows = [replace(sample, project="", path=""),
            replace(sample, project=old_root, path=external),
            replace(sample, project=old_root, path=old_result),
            replace(sample, project="foreign-drive", path="other-drive/result.csv")]
    proxy = SimpleNamespace(path=registry.path, all=lambda: rows)
    commonpath = os.path.commonpath
    class PathOperations:
        def __getattr__(self, name):
            return getattr(os.path, name)
        def commonpath(self, paths):
            if paths[0] == "foreign-drive":
                raise ValueError("different drives")
            return commonpath(paths)
    class FileOperations:
        path = PathOperations()
        def __getattr__(self, name):
            return getattr(os, name)
    monkeypatch.setattr(rc, "os", FileOperations())
    relocated = rc._rows_of_a_moved_project(proxy, root)
    assert [row.path for row in relocated] == ["", external, str(root / "run/result.csv"), "other-drive/result.csv"]
    assert all(row.project == str(root) for row in relocated)
    assert [row.path for row in rows] == ["", external, old_result, "other-drive/result.csv"]
    proxy.path = str(tmp_path / "unrelated.db")
    assert rc._rows_of_a_moved_project(proxy, root) == []
