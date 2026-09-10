"""Several measurement databases, in the two screens instruction 109 names.

The mechanism -- :mod:`spacr.multi_database` -- is tested in
``tests/test_multi_database.py``. THIS file is about the two screens, and it
presses the real controls rather than asserting on a model object, because
instruction 52 was closed on 97 passing tests and the maintainer opened the
app and found the controls unreachable.

So: the chips are counted, their × is clicked, and what the screen then holds
is compared against what was in the files.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.multi_database import SOURCE_COLUMN


def _database(path, plate, *, rows=4, extra=None, table="cell"):
    frame = pd.DataFrame({
        "plateID": [plate] * rows,
        "rowID": [f"r{i + 1}" for i in range(rows)],
        "columnID": ["c1"] * rows,
        "object_label": range(1, rows + 1),
        "area": range(rows),
    })
    if extra:
        frame[extra] = 1.0
    with sqlite3.connect(str(path)) as db:
        frame.to_sql(table, db, index=False)
    return str(path)


@pytest.fixture
def three_plates(tmp_path):
    """Three plates, as a screen is actually acquired: one folder each."""
    paths, counts = [], {}
    for index, rows in enumerate((4, 7, 5), start=1):
        folder = tmp_path / f"plate{index}" / "measurements"
        folder.mkdir(parents=True)
        paths.append(_database(folder / "measurements.db", f"plate{index}",
                               rows=rows))
        counts[f"plate{index}"] = rows
    return paths, counts


@pytest.fixture
def screen(qtbot):
    from spacr.qt.screens.gate_editor import GateEditorScreen

    widget = GateEditorScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _chip_names(layout):
    from spacr.qt.widgets.table_chip import TableChip

    names = []
    for index in range(layout.count()):
        widget = layout.itemAt(index).widget()
        if isinstance(widget, TableChip):
            names.append(widget.name)
    return names


def _chips(layout):
    from spacr.qt.widgets.table_chip import TableChip

    return [layout.itemAt(i).widget() for i in range(layout.count())
            if isinstance(layout.itemAt(i).widget(), TableChip)]


# --------------------------------------------------------------------------- #
#  The Gate Editor
# --------------------------------------------------------------------------- #

def test_three_databases_become_one_removable_working_set(screen, three_plates):
    """"The same three load into the Gate Editor as a working set, removable
    individually." One chip each, named as the provenance column names them."""
    paths, counts = three_plates
    screen.load_paths(paths)

    assert _chip_names(screen._db_chips) == ["plate1", "plate2", "plate3"]
    assert screen._db_chips_label.isVisibleTo(screen)
    # Every one of them can go: with three loaded, none is the last one.
    assert all(chip._close.isVisibleTo(chip) for chip in _chips(screen._db_chips))


def test_a_per_source_count_through_the_screen_matches_the_files(
        screen, three_plates):
    """THE TEST THAT CATCHES SILENT POOLING, driven through the screen.

    The module-level version of this lives in tests/test_multi_database.py.
    This one exists because the screen is what a user touches, and a screen
    that dropped or doubled a source between the merge and the frame it plots
    would pass that one.
    """
    paths, counts = three_plates
    screen.load_paths(paths)

    frame = screen._frame
    assert frame is not None
    assert frame.groupby(SOURCE_COLUMN).size().to_dict() == counts
    assert len(frame) == sum(counts.values())


def test_removing_one_chip_re_merges_the_other_two(screen, three_plates):
    """The × is the control, so the × is what the test presses."""
    paths, counts = three_plates
    screen.load_paths(paths)

    chip = next(c for c in _chips(screen._db_chips) if c.name == "plate2")
    chip._close.click()

    assert _chip_names(screen._db_chips) == ["plate1", "plate3"]
    remaining = {"plate1": counts["plate1"], "plate3": counts["plate3"]}
    assert screen._frame.groupby(SOURCE_COLUMN).size().to_dict() == remaining


def test_removing_down_to_one_leaves_a_working_screen(screen, three_plates):
    """Down to one database is the ordinary single-database session again,
    and the strip goes away rather than showing a set of one."""
    paths, _counts = three_plates
    screen.load_paths(paths)

    for name in ("plate2", "plate3"):
        chip = next(c for c in _chips(screen._db_chips) if c.name == name)
        chip._close.click()

    assert _chip_names(screen._db_chips) == []
    assert not screen._db_chips_label.isVisibleTo(screen)
    assert screen._frame is not None and len(screen._frame) == 4


def test_the_table_working_set_survives_a_multi_database_load(screen,
                                                              three_plates):
    """41's working set has to exist on a merged frame too.

    Before this, `load_paths` never filled the table picker or the table
    working set, so nucleus could not be added to a merged session at all --
    and `_reload_working_set` would then have read only the FIRST database.
    """
    paths, _counts = three_plates
    screen.load_paths(paths)

    assert screen._tables == ["cell"]
    assert screen._table_picker.count() >= 1
    assert _chip_names(screen._chips) == ["cell"]


# --------------------------------------------------------------------------- #
#  Told, and recorded
# --------------------------------------------------------------------------- #

def test_a_collision_is_told_and_the_frame_is_left_alone(screen, tmp_path):
    """Two databases that each hold a ``plate1`` are two experiments."""
    colliding = [_database(tmp_path / "runA.db", "plate1"),
                 _database(tmp_path / "runB.db", "plate1")]
    screen.load_paths(colliding)

    text = screen._source.text()
    assert "plate1" in text
    assert "runA" in text and "runB" in text
    # NOT the qualification advice: it hides the experiment inside the plate
    # id, where nothing can colour by it (instruction 122's correction).
    assert "qualify" not in text
    assert screen._frame is None


def test_what_the_user_chose_about_a_collision_is_recorded(screen,
                                                           three_plates,
                                                           monkeypatch,
                                                           tmp_path):
    """Dropping a database leaves no trace in the surviving frame, so the
    decision is written down where it can be read afterwards."""
    import json

    from spacr import multi_database

    log = tmp_path / "merge_decisions.jsonl"
    monkeypatch.setattr(multi_database, "decision_log_path",
                        lambda: str(log))

    paths, _counts = three_plates
    screen.load_paths(paths)
    chip = next(c for c in _chips(screen._db_chips) if c.name == "plate2")
    chip._close.click()

    assert screen._merge_decision is not None
    written = [json.loads(line) for line in log.read_text().splitlines()]
    outcomes = [record["outcome"] for record in written]
    assert "merged" in outcomes
    assert "resolved" in outcomes
    resolved = next(r for r in written if r["outcome"] == "resolved")
    assert "plate2" in resolved["resolution"]


# --------------------------------------------------------------------------- #
#  Image UMAP
# --------------------------------------------------------------------------- #

@pytest.fixture
def umap_panel(qtbot, qt_theme_applied):
    from spacr.qt.screens.settings_model import SettingsWidgets

    panel = SettingsWidgets("umap")
    panel.build_sections()
    return panel


def test_the_image_umap_source_takes_more_than_one_project(umap_panel,
                                                           tmp_path):
    """`generate_image_umap` has always taken a LIST of source roots; the
    panel was the half that could only ever express one."""
    from spacr.qt.widgets.database_set import DatabaseSetWidget

    widget = umap_panel._widgets["src"]
    assert isinstance(widget, DatabaseSetWidget)

    roots = [str(tmp_path / f"plate{i}") for i in (1, 2, 3)]
    widget.add_sources(roots)

    assert umap_panel.collect()["src"] == roots


def test_the_image_umap_can_colour_the_map_by_source(umap_panel, tmp_path):
    """A merged embedding whose clusters turn out to be the plates is the
    most valuable thing this feature can show, and it can only show it if the
    control to colour by source is reachable."""
    widget = umap_panel._widgets["src"]
    widget.add_sources([str(tmp_path / "plate1"), str(tmp_path / "plate2")])

    assert widget.colour_by_source.isVisibleTo(widget)
    widget.colour_by_source.setChecked(True)

    assert umap_panel.collect()["color_by"] == SOURCE_COLUMN


def test_one_source_does_not_offer_a_colour_by_source_that_says_nothing(
        umap_panel, tmp_path):
    widget = umap_panel._widgets["src"]
    widget.add_sources([str(tmp_path / "plate1")])
    assert not widget.colour_by_source.isVisibleTo(widget)


def test_the_source_control_says_what_the_merge_costs_before_it_runs(
        qtbot, tmp_path):
    """Point 4: the column set a merge produces IS the analysis about to be
    run, so it is reported while the files are still being chosen."""
    from spacr.qt.widgets.database_set import DatabaseSetWidget

    roots = []
    for index, extra in enumerate((None, "perimeter"), start=1):
        folder = tmp_path / f"plate{index}" / "measurements"
        folder.mkdir(parents=True)
        _database(folder / "measurements.db", f"plate{index}", extra=extra)
        roots.append(str(tmp_path / f"plate{index}"))

    widget = DatabaseSetWidget(mode="folder")
    qtbot.addWidget(widget)
    widget.add_sources(roots)

    text = widget.summary.text()
    assert "2 database(s)" in text
    assert "8 rows" in text
    # The measurement that is about to be dropped, NAMED.
    assert "perimeter" in text
    assert _chip_names(widget._chips) == ["plate1", "plate2"]


def test_the_source_control_names_a_plate_collision_without_offering_to_hide_it(
        qtbot, tmp_path):
    from spacr.qt.widgets.database_set import DatabaseSetWidget

    roots = []
    for index in (1, 2):
        folder = tmp_path / f"run{index}" / "measurements"
        folder.mkdir(parents=True)
        _database(folder / "measurements.db", "plate1")
        roots.append(str(tmp_path / f"run{index}"))

    widget = DatabaseSetWidget(mode="folder")
    qtbot.addWidget(widget)
    widget.add_sources(roots)

    text = widget.summary.text()
    assert "plate1" in text
    assert "run1" in text and "run2" in text
    assert "qualify" not in text
    assert "Remove one of them" in text


def test_a_source_whose_database_was_never_written_is_named(qtbot, tmp_path):
    """In folder mode the database is two levels below what the user picked,
    so silence would be indistinguishable from "that plate was measured"."""
    from spacr.qt.widgets.database_set import DatabaseSetWidget

    widget = DatabaseSetWidget(mode="folder")
    qtbot.addWidget(widget)
    widget.add_sources([str(tmp_path / "never_measured")])

    assert "no measurements database" in widget.summary.text()


def test_the_folder_join_is_the_one_get_db_paths_performs(tmp_path):
    """``spacr.utils`` pulls torch, so the settings panel spells this join out
    rather than importing it. This is what keeps the two equal."""
    from spacr.utils import get_db_paths
    from spacr.qt.widgets.database_set import database_for_source

    roots = [str(tmp_path / "plate1"), str(tmp_path / "plate2") + "/"]
    assert [database_for_source(root, "folder") for root in roots] == \
        get_db_paths([root.rstrip("/") for root in roots])


def test_one_source_is_still_a_bare_string(qtbot, tmp_path):
    """A user who chose ONE folder must see exactly what they saw before.

    ``src`` is written to the settings CSV, read by the CLI, replayed by the
    run journal and joined onto by anything doing ``os.path.join(src, ...)``.
    Handing all of those ``['/data/plate1']`` where they used to get
    ``'/data/plate1'`` is the regression a one-element list already caused
    once for ``column_csv``.
    """
    from spacr.qt.widgets.database_set import DatabaseSetWidget

    widget = DatabaseSetWidget(mode="folder")
    qtbot.addWidget(widget)

    widget.add_sources([str(tmp_path / "plate1")])
    assert widget.get_value() == str(tmp_path / "plate1")

    widget.add_sources([str(tmp_path / "plate2")])
    assert widget.get_value() == [str(tmp_path / "plate1"),
                                  str(tmp_path / "plate2")]


def test_the_control_is_reachable_in_the_real_image_umap_screen(qtbot,
                                                                qt_theme_applied):
    """Instruction 52 was closed on 97 passing tests, and the maintainer
    opened the app and found the controls unreachable. So this one opens the
    screen, expands the section the control lives in, and presses the button.
    """
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.database_set import DatabaseSetWidget
    from spacr.qt.widgets.section import Section

    screen = AppScreen("umap")
    qtbot.addWidget(screen)
    screen.resize(1400, 900)

    widget = screen._settings_model._widgets["src"]
    assert isinstance(widget, DatabaseSetWidget)

    section = widget
    while section is not None and not isinstance(section, Section):
        section = section.parentWidget()
    assert section is not None, "the source control is in no settings section"

    screen.show()
    section.set_expanded(True)
    qtbot.waitUntil(lambda: widget.isVisibleTo(screen), timeout=2000)
    assert widget.add_button.isVisibleTo(screen)
    assert widget.add_button.isEnabled()
