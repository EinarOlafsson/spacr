"""The set of measurement databases a screen will merge.

:class:`spacr.qt.widgets.database_set.DatabaseSetWidget` holds a list of
sources (plate folders or database files), shows one chip each, and puts a
summary of the merge on screen whenever the set changes.  These tests walk
the ways a set is assigned, grown and shrunk, and the summaries for sources
that cannot be merged, asserting the exact sources, chips and summary text.
"""
from __future__ import annotations

import os
import sqlite3

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QFileDialog                             # noqa: E402

from spacr.qt.widgets.database_set import DatabaseSetWidget           # noqa: E402
from spacr.qt.widgets.table_chip import TableChip                     # noqa: E402

pytestmark = pytest.mark.qt


def _measurements_db(path, plate):
    """A minimal measurements database with a one-row ``cell`` table."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE cell (plateID TEXT, wellID TEXT, area REAL)")
        db.execute("INSERT INTO cell VALUES (?, ?, ?)", (plate, "A01", 12.0))
    return path


@pytest.fixture
def plate_folders(tmp_path):
    """Two spaCR plate folders, each with its own measurements database."""
    made = []
    for name in ("plateA", "plateB"):
        folder = tmp_path / name
        _measurements_db(str(folder / "measurements" / "measurements.db"), name)
        made.append(str(folder))
    return made


@pytest.fixture
def make_set(qtbot):
    """Database-set widgets, registered with qtbot."""
    def make(*args, **kwargs):
        widget = DatabaseSetWidget(*args, **kwargs)
        qtbot.addWidget(widget)
        return widget
    return make


def _chip_names(widget):
    """The names on the widget's own chips, in order."""
    return [chip.name for chip in widget.findChildren(TableChip)
            if chip.parent() is widget]


def test_assigning_a_set_keeps_each_source_once(make_set, plate_folders,
                                                subtests):
    """``set_value`` / ``get_value``: blanks and repeats dropped, a single
    source stays a string, and the chip row is rebuilt past a spacer."""
    with subtests.test(msg="blanks and repeats"):
        widget = make_set(mode="folder")
        widget.set_value(["/data/plate1", "", "   ", "/data/plate1",
                          "/data/plate2"])
        assert widget.sources() == ["/data/plate1", "/data/plate2"]
        assert _chip_names(widget) == ["plate1", "plate2"]

    with subtests.test(msg="one source is a string, two are a list"):
        widget = make_set(mode="folder")
        widget.set_value(plate_folders[0])
        assert widget.get_value() == plate_folders[0]
        widget.set_value(plate_folders)
        assert widget.get_value() == plate_folders

    with subtests.test(msg="a spacer in the chip row"):
        widget = make_set(["/data/plate1", "/data/plate2"], mode="folder")
        assert _chip_names(widget) == ["plate1", "plate2"]
        widget._chips.insertStretch(0, 1)
        widget.set_value(["/data/plate3", "/data/plate4"])
        assert _chip_names(widget) == ["plate3", "plate4"]
        assert widget.sources() == ["/data/plate3", "/data/plate4"]


def test_sources_are_added_and_removed_by_label_or_path(
        make_set, plate_folders, monkeypatch, subtests):
    """``choose_sources`` and ``remove_source``: the picker's folder joins,
    a cancel adds nothing, unknown names and the required last source stay."""
    with subtests.test(msg="the picker's folder, then a cancel"):
        widget = make_set(mode="folder")
        for answer in (plate_folders[0], ""):
            monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                                staticmethod(lambda *a, _r=answer, **k: _r))
            widget.choose_sources()
            assert widget.sources() == [plate_folders[0]]

    with subtests.test(msg="by label, by path, and an unknown name"):
        widget = make_set(mode="folder")
        widget.set_value(plate_folders)
        assert widget.remove_source("/somewhere/else") is False
        assert widget.sources() == plate_folders
        assert widget.remove_source(widget._labels()[0]) is True
        assert widget.sources() == [plate_folders[1]]
        assert widget.remove_source(plate_folders[1]) is True
        assert widget.sources() == []

    with subtests.test(msg="the last required source"):
        widget = make_set(mode="folder", min_items=1)
        widget.set_value(plate_folders[:1])
        assert widget.remove_source(plate_folders[0]) is False
        assert widget.sources() == plate_folders[:1]

    with subtests.test(msg="no screen owns the colour-by box"):
        widget = make_set(mode="folder")
        widget.set_value(plate_folders)
        assert widget.colour_by_source.isVisible() is False
        widget._on_colour_toggled(True)
        assert widget.sources() == plate_folders


def test_the_summary_names_sources_that_cannot_be_merged(make_set, tmp_path,
                                                         subtests):
    """A plate never measured and a file that is not a database each give a
    summary naming the problem, and no merge plan."""
    with subtests.test(msg="a plate never measured"):
        unmeasured = tmp_path / "plate_never_run"
        unmeasured.mkdir()
        widget = make_set(mode="folder")
        widget.set_value([str(unmeasured)])
        said = widget.summary.text()
        assert said.startswith(
            "1 of 1 sources have no measurements database yet: ")
        assert str(unmeasured) in said
        assert widget.plan() is None

    with subtests.test(msg="a file that is not a database"):
        fake = tmp_path / "measurements.db"
        fake.write_bytes(b"this is not a SQLite file at all")
        widget = make_set()
        widget.set_value([str(fake)])
        assert widget.summary.text().startswith(
            "could not read 1 database(s) as 'cell': ")
        assert widget.plan() is None
