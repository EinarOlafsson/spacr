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
