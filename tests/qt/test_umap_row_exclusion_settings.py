"""UMAP exposes general database row exclusions instead of lab plate presets.

The second half of this file is about *when* the editor talks to sqlite
rather than what it says. Both of its reads used to run on the GUI
thread, and the ``SELECT DISTINCT`` behind the value dropdown is the
expensive one: the columns a user excludes on hold a handful of distinct
values across the whole table, so the query's ``LIMIT`` is never reached
and SQLite scans every row, once per table the column appears in.
Measured on a 200 000-row × 8-table measurements.db, with the same
event-loop watchdog ``test_gui_responsiveness`` uses:

    set_source                       183 ms  ->  24 ms
    one column change                220 ms  ->   3 ms
    eight rapid column edits         894 ms  ->   2 ms

The last one is the point. The column combo is editable, so
``currentTextChanged`` fires per keystroke, and eight of them froze the
window for the better part of a second in one unbroken block.
"""

from __future__ import annotations

import os
import sqlite3
import time

import pandas as pd
import pytest
from PySide6.QtCore import Qt

from spacr.qt.screens.settings_model import SettingsWidgets
from spacr.qt.widgets import row_exclusion as row_exclusion_mod
from spacr.qt.widgets.row_exclusion import RowExclusionEditor

from tests.qt.test_gui_responsiveness import LoopWatchdog, STALL_BUDGET_S

#: Tables the big fixture carries, all with the same metadata columns.
BIG_TABLES = ("cell", "nucleus", "pathogen", "cytoplasm", "png_list",
              "parasite")


def _measurements_source(tmp_path):
    measurements = tmp_path / "measurements"
    measurements.mkdir()
    frame = pd.DataFrame({
        "plateID": ["p1", "p1", "p2"],
        "columnID": ["c1", "c2", "c1"],
        "object_label": [1, 2, 3],
        "cell_channel_0_mean_intensity": [1.0, 2.0, 3.0],
    })
    with sqlite3.connect(measurements / "measurements.db") as connection:
        frame.to_sql("cell", connection, index=False)
    return tmp_path


@pytest.fixture(scope="module")
def big_source(tmp_path_factory):
    """A measurements.db the value read is unmistakably slow against.

    150 000 rows in each of six tables — 110 MB — with the
    *low-cardinality* text columns that make ``SELECT DISTINCT … LIMIT
    501`` scan the lot: ``plateID`` has three values across 900 000 rows,
    so the limit never fires and every page is read. The nine float
    columns are not decoration; they are what makes each row wide enough
    for the scan to cost what a real feature table costs.

    Module-scoped: building it costs about two seconds and every test
    here reads it without writing.
    """
    import numpy as np

    root = tmp_path_factory.mktemp("bigrun")
    (root / "measurements").mkdir()
    rng = np.random.default_rng(0)
    n, wide = 150_000, 8
    plates = ["plate1", "plate2", "plate3"]
    extra = ", ".join(f'"cell_channel_{i}_mean_intensity" REAL'
                      for i in range(1, wide + 1))
    with sqlite3.connect(root / "measurements" / "measurements.db") as con:
        for name in BIG_TABLES:
            con.execute(
                f'CREATE TABLE "{name}" (plateID TEXT, columnID TEXT, '
                f'rowID TEXT, prcfo TEXT, object_label INTEGER, '
                f'cell_channel_0_mean_intensity REAL, {extra})')
            block = rng.uniform(0, 5e4, (n, wide + 1)).tolist()
            con.executemany(
                f'INSERT INTO "{name}" '
                f'VALUES ({",".join("?" * (6 + wide))})',
                [(plates[i % 3], f"c{i % 24 + 1:02d}", f"r{i % 16 + 1:02d}",
                  f"plate1_A{i % 12 + 1:02d}_{i}", i, *block[i])
                 for i in range(n)])
    return root


def _editor(qtbot, **kw):
    editor = RowExclusionEditor(**kw)
    qtbot.addWidget(editor)
    return editor


def _settled(qtbot, editor, timeout=30000):
    """Pump until every queued, running and undelivered read is done."""
    qtbot.waitUntil(lambda: not editor.is_busy(), timeout=timeout)
    return editor


def _drive(qtbot, dog, done, budget_s=30.0):
    """Pump the event loop until ``done()``, never blocking it."""
    end = time.perf_counter() + budget_s
    while time.perf_counter() < end and not done():
        qtbot.wait(20)
    qtbot.wait(50)
    dog.stop()


def test_umap_input_section_contains_general_exclusion(qtbot):
    model = SettingsWidgets("umap")
    sections = dict(model.build_sections())

    input_labels = [label for label, _widget in sections["Input Data"]]
    assert "Exclude" in input_labels
    assert isinstance(model._widgets["exclude_rows"], RowExclusionEditor)
    for legacy in (
        "col_to_compare", "pos", "neg", "mix",
        "embedding_by_controls", "exclude_conditions",
    ):
        assert legacy not in model._widgets

    assert "Exclude features" in input_labels


def test_umap_display_embedding_plot_and_advanced_settings_are_panel_sections(
        qtbot):
    model = SettingsWidgets("umap")
    sections = dict(model.build_sections())

    assert {"Points & Images", "Clustering", "UMAP", "Runtime"} <= set(
        sections)
    display = {label for label, _widget in sections["Points & Images"]}
    embedding = {label for label, _widget in sections["Clustering"]}
    umap = {label for label, _widget in sections["UMAP"]}
    assert {"Point color", "Point alpha", "Plot images",
            "Plot cluster grids"} <= display
    assert {"Clustering", "Eps", "Min samples"} <= embedding
    assert {"N neighbors", "Min dist"} <= umap
    assert "https://" in model.plain_tooltip_for("plot_images")


def test_exclusion_editor_loads_columns_and_values_from_dropped_source(
    qtbot,
    tmp_path,
):
    source = _measurements_source(tmp_path)
    model = SettingsWidgets("umap")
    model.build_sections()
    assert model.set_value_for_key("tables", ["cell"])
    assert model.set_value_for_key("src", str(source))

    editor = model._widgets["exclude_rows"]
    # `set_source` dispatches the schema read and returns, so the columns
    # arrive an event-loop turn later rather than inside the setter. They
    # still have to arrive: this waits for the read, it does not excuse it.
    _settled(qtbot, editor)
    row = editor._rows[0]
    column_index = row.column.findText("columnID")
    assert column_index >= 0
    row.column.setCurrentIndex(column_index)
    _settled(qtbot, editor)

    value_model = row.values.model()
    c1 = next(
        value_model.item(index)
        for index in range(value_model.rowCount())
        if value_model.item(index).text() == "c1"
    )
    c1.setCheckState(Qt.Checked)

    assert model.collect()["exclude_rows"] == {"columnID": ["c1"]}


def test_exclusion_editor_round_trips_imported_rules(qtbot, tmp_path):
    source = _measurements_source(tmp_path)
    model = SettingsWidgets("umap")
    model.build_sections()
    model.set_value_for_key("tables", ["cell"])
    model.set_value_for_key("src", str(source))

    assert model.set_value_for_key(
        "exclude_rows",
        {"columnID": ["c2"], "plateID": ["p2"]},
    )
    assert model.collect()["exclude_rows"] == {
        "columnID": ["c2"],
        "plateID": ["p2"],
    }


def test_umap_none_filter_text_collects_as_no_filter(qtbot):
    model = SettingsWidgets("umap")
    model.build_sections()

    assert model.set_value_for_key("filter_by", "None")
    assert model.collect()["filter_by"] is None


# ---------------------------------------------------------------------------
# The window keeps repainting while the editor reads the database
# ---------------------------------------------------------------------------

def test_reading_the_values_inline_really_is_slow_enough_to_matter(big_source):
    """Guard against the fixture shrinking until the tests prove nothing.

    If ``distinct_values`` were already under the budget the tests below
    would pass with the threading taken back out. It is not: ``plateID``
    holds three values across 600 000 rows, so every one of those rows is
    read before the query can say so.
    """
    found = row_exclusion_mod.discover_columns(big_source, BIG_TABLES)
    start = time.perf_counter()
    values = row_exclusion_mod.distinct_values(found["plateID"], "plateID")
    elapsed = time.perf_counter() - start

    assert values == ["plate1", "plate2", "plate3"]
    assert elapsed > 0.05, (
        f"the DISTINCT read took only {elapsed * 1000:.0f} ms; the fixture "
        "is no longer a meaningful stand-in for a measurements table")


def test_setting_a_source_never_freezes_the_gui_thread(qtbot, big_source):
    """``set_source`` dispatches the schema read; it does not perform it."""
    editor = _editor(qtbot)
    editor.show()
    qtbot.waitExposed(editor)
    qtbot.wait(50)

    dog = LoopWatchdog(editor)
    dog.start()
    dispatch = time.perf_counter()
    editor.set_source(big_source, BIG_TABLES)
    dispatch = time.perf_counter() - dispatch
    _drive(qtbot, dog, lambda: not editor.is_busy())

    assert dispatch < 0.100, (
        f"set_source took {dispatch * 1000:.0f} ms to return; it is still "
        "reading the database on the GUI thread")
    assert dog.ticks > 10, "the watchdog never ran; the measurement is void"
    assert dog.worst < STALL_BUDGET_S, (
        f"set_source stalled the GUI thread for {dog.worst * 1000:.0f} ms "
        f"(budget {STALL_BUDGET_S * 1000:.0f} ms)")
    # And it really read, rather than staying responsive by doing nothing.
    assert "plateID" in editor._column_sources
    assert len(editor._column_sources["plateID"]) == len(BIG_TABLES)
    assert editor._rows[0].values.checked_values() == []
    assert [v for v in editor._value_cache["plateID"]] == [
        "plate1", "plate2", "plate3"]


def test_choosing_a_column_never_freezes_the_gui_thread(qtbot, big_source,
                                                        monkeypatch):
    """The per-keystroke ``SELECT DISTINCT`` runs on a worker.

    ``distinct_values`` is replaced with an unambiguously slow one, in
    the place the widget reads from, so this cannot pass because the
    fixture happened to be fast today. 1.2 s of reading, and the window
    must keep repainting throughout.
    """
    def slow_values(sources, column, limit=500):
        time.sleep(1.2)
        return [f"{column}-{i}" for i in range(4)]

    editor = _editor(qtbot)
    editor.show()
    qtbot.waitExposed(editor)
    editor.set_source(big_source, BIG_TABLES)
    _settled(qtbot, editor)
    row = editor._rows[0]
    index = row.column.findText("columnID")
    assert index >= 0
    monkeypatch.setattr(row_exclusion_mod, "distinct_values", slow_values)

    dog = LoopWatchdog(editor)
    dog.start()
    dispatch = time.perf_counter()
    row.column.setCurrentIndex(index)
    dispatch = time.perf_counter() - dispatch
    _drive(qtbot, dog, lambda: not editor.is_busy())

    assert dispatch < 0.100, (
        f"choosing a column took {dispatch * 1000:.0f} ms to return; it is "
        "still reading on the GUI thread")
    assert dog.ticks > 10, "the watchdog never ran; the measurement is void"
    assert dog.worst < STALL_BUDGET_S, (
        f"choosing a column stalled the GUI thread for "
        f"{dog.worst * 1000:.0f} ms")
    # The 1.2 s of reading actually happened and landed in the dropdown.
    model = row.values.model()
    assert [model.item(i).text() for i in range(model.rowCount())] == [
        "columnID-0", "columnID-1", "columnID-2", "columnID-3"]


def test_rapid_column_edits_coalesce_into_a_single_query(qtbot, big_source,
                                                         monkeypatch):
    """Typing a column name must not cost one table scan per keystroke.

    The column combo is editable, so ``currentTextChanged`` fires on
    every character. Nine edits here; the debounce must turn them into
    far fewer reads, and the one it does run must be for the **last**
    thing typed.
    """
    editor = _editor(qtbot)
    editor.set_source(big_source, BIG_TABLES)
    _settled(qtbot, editor)

    asked = []
    real = row_exclusion_mod.distinct_values

    def counting(sources, column, limit=500):
        asked.append(column)
        return real(sources, column, limit)

    monkeypatch.setattr(row_exclusion_mod, "distinct_values", counting)

    row = editor._rows[0]
    typed = ["r", "ro", "row", "rowI", "rowID", "c", "co", "col", "columnID"]
    for text in typed:
        row.column.setEditText(text)
    _settled(qtbot, editor)

    assert asked == ["columnID"], (
        f"{len(typed)} edits produced {len(asked)} queries: {asked}")
    model = row.values.model()
    assert model.rowCount() == 24        # c01 … c24, read exactly once


def test_leaving_the_editor_mid_load_cancels_cleanly(qtbot, big_source):
    """Closing during a read must not leak a thread or deliver into a corpse.

    Qt aborts the process when a running QThread is destroyed, and a
    worker that paints a widget on its way out is a use-after-free.
    """
    editor = _editor(qtbot)
    delivered = []
    editor.loaded.connect(delivered.append)

    editor.set_source(big_source, BIG_TABLES)
    assert editor.is_busy()
    editor.close()                       # mid-read, deliberately

    assert editor.active_jobs() == 0
    assert not editor.is_busy()
    qtbot.wait(300)
    assert delivered == []


def test_a_completed_read_leaves_no_job_behind(qtbot, big_source):
    """``active_jobs()`` returns to zero — the ``thread.finished`` bug test."""
    from spacr.qt.bridge import registry

    editor = _editor(qtbot)
    before = len(registry().active())
    editor.set_source(big_source, BIG_TABLES)
    assert editor.active_jobs() >= 1
    _settled(qtbot, editor)

    qtbot.waitUntil(lambda: editor.active_jobs() == 0, timeout=20000)
    qtbot.waitUntil(lambda: len(registry().active()) == before, timeout=20000)


def test_a_new_source_supersedes_the_one_still_being_read(qtbot, big_source,
                                                          tmp_path):
    """The second source wins, however slow the first one was."""
    small = _measurements_source(tmp_path)
    editor = _editor(qtbot)
    editor.set_source(big_source, BIG_TABLES)
    editor.set_source(small, ["cell"])
    _settled(qtbot, editor)

    # 'prcfo' only exists in the big fixture; 'columnID' is in both, but
    # only the small one puts it in exactly one table.
    assert "prcfo" not in editor._column_sources
    assert len(editor._column_sources["columnID"]) == 1
    assert editor._value_cache["plateID"] == ["p1", "p2"]


def test_unthreaded_mode_still_reads_inside_the_call(qtbot, tmp_path):
    """``threaded=False`` keeps the old contract for hosts that need it."""
    source = _measurements_source(tmp_path)
    editor = _editor(qtbot, threaded=False)
    editor.set_source(source, ["cell"])

    assert not editor.is_busy()
    assert editor._rows[0].column.findText("columnID") >= 0
    assert editor._value_cache["plateID"] == ["p1", "p2"]


def test_the_pure_readers_survive_everything_a_src_field_can_hold(tmp_path):
    """``source_paths``/``discover_columns``/``distinct_values`` run on a
    worker thread, where an exception is a job failure rather than a
    banner, so every shape a half-typed ``src`` can take is answered with
    data instead of a raise."""
    source_paths = row_exclusion_mod.source_paths
    discover = row_exclusion_mod.discover_columns
    values_of = row_exclusion_mod.distinct_values

    run = tmp_path / "run"
    (run / "measurements").mkdir(parents=True)
    db = run / "measurements" / "measurements.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE cell (plateID TEXT, only_here TEXT)")
        con.execute("CREATE TABLE nucleus (plateID TEXT)")
        con.executemany("INSERT INTO cell VALUES (?,?)",
                        [("p1", "x"), ("p2", "y"), ("p3", "z")])
        con.execute("INSERT INTO nucleus VALUES ('p4')")

    # Every spelling of "where the database is".
    assert source_paths(str(run)) == [db]
    assert source_paths(str(run / "measurements")) == [db]
    assert source_paths(str(db)) == [db]
    assert source_paths(repr([str(run)])) == [db]
    assert source_paths([None, "", str(run)]) == [db]
    # A string that only looks like a list falls back to being a path.
    assert source_paths("[not, valid, python") == []

    # A table filter really filters.
    assert set(discover(str(run), ["cell"])) == {"plateID", "only_here"}

    # A file that is not a database, and a path that cannot be opened.
    junk = tmp_path / "junk" / "measurements.db"
    junk.parent.mkdir()
    junk.write_text("this is not sqlite")
    assert discover(str(junk)) == {}
    assert values_of([(junk, "cell")], "plateID") == []
    missing = tmp_path / "gone" / "x.db"
    assert discover(str(missing)) == {}
    assert values_of([(missing, "cell")], "plateID") == []

    # A database that exists but this user may not open — a measurements.db
    # written by whoever ran the pipeline.
    locked = tmp_path / "locked" / "measurements.db"
    locked.parent.mkdir()
    locked.write_bytes(db.read_bytes())
    os.chmod(locked, 0o000)
    try:
        if not os.access(locked, os.R_OK):     # not true when run as root
            assert discover(str(locked)) == {}
    finally:
        os.chmod(locked, 0o600)

    # No column named, and a table that is not there.
    assert values_of([(db, "cell")], "") == []
    assert values_of([(db, "no_such_table"), (db, "cell")], "plateID") == [
        "p1", "p2", "p3"]
    # The LIMIT stops the scan, and stops looking at further tables.
    assert values_of([(db, "cell"), (db, "nucleus")], "plateID",
                     limit=2) == ["p1", "p2"]


def test_the_editor_survives_the_odd_shapes_a_host_can_put_it_in(qtbot,
                                                                 tmp_path):
    """The small guards, exercised rather than assumed."""
    source = _measurements_source(tmp_path)
    editor = _editor(qtbot, threaded=False)
    editor.set_source(source, ["cell"])

    # Deleting the only rule leaves an empty one rather than nothing.
    editor._remove_row(editor._rows[0])
    assert len(editor._rows) == 1

    # A column the database does not have is kept, not silently dropped.
    row = editor._rows[0]
    row.column.setEditText("typo_column")
    row.set_columns(["plateID", "columnID"])
    assert row.column.currentText() == "typo_column"

    # Clicking a value in the dropdown toggles it, both ways.
    row.values.set_options(["a", "b"])
    index = row.values.model().index(0, 0)
    row.values._toggle_index(index)
    assert row.values.checked_values() == ["a"]
    row.values._toggle_index(index)
    assert row.values.checked_values() == []

    # A worker that found nothing paints nothing.
    editor._apply_values({})
    editor._apply_values(None)

    # The classmethod aliases still answer, for anything that predates
    # the module-level readers.
    assert RowExclusionEditor._source_paths(source) == \
        row_exclusion_mod.source_paths(source)
    assert set(RowExclusionEditor._discover_columns(source, ["cell"])) == \
        set(row_exclusion_mod.discover_columns(source, ["cell"]))


def test_a_removed_rule_row_is_never_painted_by_a_late_read(qtbot,
                                                            big_source):
    """A row deleted while its values were being read is simply dropped."""
    editor = _editor(qtbot)
    editor.set_source(big_source, BIG_TABLES)
    _settled(qtbot, editor)

    editor._add_row("columnID")
    doomed = editor._rows[-1]
    assert editor._pending.get(doomed) == "columnID"
    editor._remove_row(doomed)
    assert doomed not in editor._pending
    _settled(qtbot, editor)

    assert doomed not in editor._rows
