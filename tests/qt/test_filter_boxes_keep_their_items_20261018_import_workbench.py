"""Typing in the import workbench's regex box never deletes a table item.

Deleting and re-creating Python-made items on every keystroke corrupted
shiboken's wrapper map and crashed the process, so the table is rewritten in
place.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt                                    # noqa: E402
from shiboken6 import isValid                                    # noqa: E402

from spacr.import_plan import plan                               # noqa: E402
from spacr.qt.widgets import import_workbench as iw              # noqa: E402

NAMES = [f"/data/plate1/{w}_T0001F{f:03d}L01A01Z01C{c:02d}.tif"
         for w in ("A01", "A02", "B01") for f in (1, 2) for c in (1, 2)]
NAMES.append("/data/plate1/notes.tif")

REGEXES = [
    r"(?P<wellID>[A-Z]\d+)_T0001F(?P<fieldID>\d+)L01A01Z01C(?P<chanID>\d+)\.tif",
    r"(?P<wellID>A01)_.*",
    r"(?P<wellID>Z99)_.*",
    r"(?P<wellID>A0",
    r"(?P<wellID>[A-Z]\d+)_.*",
    r"",
]


def _snapshot(table):
    return [(table.item(r, 0), table.item(r, 1))
            for r in range(table.rowCount())]


def test_typing_regexes_never_deletes_an_item(qtbot):
    bench = iw.ImportWorkbench(NAMES)
    qtbot.addWidget(bench)
    seen = []
    previous = None
    for text in REGEXES:
        bench.regex.setText(text)
        shown = _snapshot(bench.table)
        for pair in shown:
            seen.extend(pair)
        assert all(isValid(item) for item in seen)
        if previous is not None:
            for old, new in zip(previous, shown):
                assert old[0] is new[0] and old[1] is new[1]
        previous = shown
        made = bench.the_plan()
        expected = [(r.before, r.after) for r in made.renamed] + [
            (n, "no match") for n in made.unmatched]
        assert [(a.text(), b.text()) for a, b in shown] == expected
        for index, (_, cell) in enumerate(shown):
            missed = index >= len(made.renamed)
            assert (cell.toolTip() != "") == missed
            assert (cell.foreground().color() == Qt.red) == missed
            assert shown[index][0].toolTip() == ""
    assert all(isValid(item) for item in seen)


def test_the_table_matches_a_plan_of_the_same_names(qtbot):
    bench = iw.ImportWorkbench(NAMES)
    qtbot.addWidget(bench)
    bench.regex.setText(REGEXES[1])
    bench.regex.setText(REGEXES[0])
    import os
    fresh = plan([os.path.basename(n) for n in NAMES], REGEXES[0],
                 bench.roles(), plate=bench._plate_name())
    assert bench.table.rowCount() == len(fresh.renamed) + len(fresh.unmatched)
    assert bench.table.item(0, 1).text() == fresh.renamed[0].after
