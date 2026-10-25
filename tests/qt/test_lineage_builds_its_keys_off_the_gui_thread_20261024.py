"""The lineage load builds every typed key on the worker, not the GUI thread.

Tutorial captures (lessons 12, 56, 60, 70, 85) froze 1-2 s in
``LineageScreen.set_frames``: the load callback built the forest, and every
row went through ``schema.object_id`` / ``split_object_id`` on the GUI thread.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr import schema
from spacr.qt.screens import lineage as lgs

FIELD = ("plate1", "r1", "c1", "f1")


def _plate(cells=300, per_cell=3):
    def rows(table, labels, parents=None):
        plate, row, column, field_id = FIELD
        frame = pd.DataFrame({"plateID": plate, "rowID": row, "columnID": column,
                              "fieldID": field_id, "object_label": labels,
                              f"{table}_area": 100.0})
        if parents is not None:
            frame["cell_id"] = parents
        return frame
    cell_labels = list(range(1, cells + 1))
    children = [label for label in cell_labels for _ in range(per_cell)]
    return {"cell": rows("cell", cell_labels),
            "nucleus": rows("nucleus", list(range(1, len(children) + 1)), children),
            "pathogen": rows("pathogen", list(range(1, len(children) + 2)), children + [cells + 50])}


@pytest.fixture
def counted(monkeypatch):
    calls = {"object_id": 0, "split_object_id": 0}
    for name in calls:
        original = getattr(schema, name)
        def wrapper(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(schema, name, wrapper)
    return calls


def test_drawing_a_built_tree_computes_no_object_ids(qtbot, qt_theme_applied, counted):
    built = lgs._build_lineage(_plate())
    assert counted["object_id"] > 1000
    counted.update(object_id=0, split_object_id=0)
    screen = lgs.LineageScreen(threaded=False)
    qtbot.addWidget(screen)
    screen._show_built(built)
    assert counted == {"object_id": 0, "split_object_id": 0}
    assert screen.tree.topLevelItemCount() == 300
    assert screen.orphan_list.count() == 1
    assert "missing" in screen.orphan_list.item(0).text()


def test_the_load_job_hands_the_gui_thread_a_built_tree(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    database = tmp_path / "measurements.db"
    database.write_bytes(b"")
    screen = lgs.LineageScreen(threaded=False)
    qtbot.addWidget(screen)
    submitted = []
    monkeypatch.setattr(screen._jobs, "submit", lambda job, done, *a, **k: submitted.append((job, done)))
    monkeypatch.setattr(lgs.lin, "read_object_tables", lambda _path: _plate(cells=5))
    screen._db.setText(str(database))
    screen.load()
    job, done = submitted[0]
    assert done == screen._show_built
    result = job()
    assert isinstance(result, lgs._BuiltLineage) and len(result.forest) == 5
