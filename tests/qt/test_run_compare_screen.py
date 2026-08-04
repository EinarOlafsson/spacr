"""The Run Compare screen: real runs, real registry, real tables.

Every test builds a project with a real
:class:`spacr.artifacts.Registry`, real ``measurements.db`` files and
real ``results.csv`` files, and then reads what the screen drew. Nothing
about the comparison is stubbed — a screen that agrees with a fake diff
is not evidence that it draws a real one.

The screen registers its app at import time, which is the seam it is
supposed to use. That registration is undone here at import, so merely
*collecting* this file cannot change the ``APPS`` table every other test
asserts against; the tests that care register it themselves.
"""
from __future__ import annotations

import csv
import os
import sqlite3

import pytest

pytest.importorskip("PySide6")

from spacr.artifacts import Registry
from spacr.qt.app import APPS, unregister_app

# Snapshot BEFORE the import: reaching this screen initialises the
# ``spacr.qt.screens`` package, which imports every self-registering screen
# there is. Collecting this file must leave the registry exactly as it found
# it, so whatever that import added is taken back out again — this one and
# every other screen's — and the tests below register what they need.
_APPS_BEFORE = {row[0] for row in APPS}

from spacr.qt.screens import run_compare as screen  # noqa: E402

# Not a `del _key` afterwards: another test module may have imported the
# screens package already, in which case the delta is empty, the loop never
# binds the name, and the `del` would fail collection of this whole file.
for _key in sorted({row[0] for row in APPS} - _APPS_BEFORE):
    unregister_app(_key)


BASE_SETTINGS = {
    "cell_diameter": 30,
    "nucleus_diameter": 20,
    "save_png": True,
}


@pytest.fixture
def registered():
    """Guarantee the app is registered for one test, then put it back.

    It used to unregister unconditionally on the way out, on the
    assumption that this fixture was what had put the row there. It is
    not any more: ``app.py`` names this screen in its own
    ``_SELF_REGISTERING_APPS``, so the row already exists, ``register()``
    answers False, and an unconditional teardown DELETES a shipped app
    from the registry for every test that runs afterwards — a
    whole-registry inventory failure in some other file, caused by a
    fixture in this one and blamed on whatever pytest collected next.

    So it undoes only what it did.
    """
    added = screen.register()
    yield screen.APP_KEY
    if added:
        unregister_app(screen.APP_KEY)
    else:
        assert any(row[0] == screen.APP_KEY for row in APPS), (
            "this fixture found the row already registered and the test "
            "removed it; the registry is now short an app")


def _measurements(path, entries):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE cell (plateID TEXT, rowID TEXT, "
                       "columnID TEXT, fieldID TEXT)")
    connection.executemany("INSERT INTO cell VALUES (?, ?, ?, ?)", entries)
    connection.commit()
    connection.close()
    return path


def _objects(plate, wells, fields, per_field):
    return [(plate, "r1", f"c{w + 1}", f"f{f + 1}")
            for w in range(wells) for f in range(fields)
            for _ in range(per_field)]


def _hits(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("feature", "coefficient", "p_value"))
        writer.writerows(rows)
    return path


def _project(tmp_path, *, runs):
    """Build one project holding several runs.

    ``runs`` is ``[(run_id, settings, cell_rows, hit_rows, created_ns), ...]``.
    """
    root = tmp_path / "project"
    root.mkdir(parents=True, exist_ok=True)
    registry = Registry(project=str(root))
    for run_id, settings, cells, hits, created in runs:
        db = _measurements(str(root / "measurements" / f"{run_id}.db"), cells)
        registry.register(module="measure", kind="measurements-db", path=db,
                          settings=settings, run_id=run_id)
        if hits is not None:
            results = _hits(str(root / "results" / f"{run_id}.csv"), hits)
            registry.register(module="regression", kind="regression-results",
                              path=results, settings=settings, run_id=run_id)
        connection = sqlite3.connect(registry.path)
        connection.execute("UPDATE artifacts SET created_ns = ? "
                           "WHERE run_id = ?", (created, run_id))
        connection.commit()
        connection.close()
    return str(root)


def _rows(tree):
    """``{top-level label: [child first column, ...]}`` for one tree."""
    out = {}
    for index in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(index)
        out[item.text(0)] = [item.child(i).text(0)
                             for i in range(item.childCount())]
    return out


def _child(tree, group, key):
    """One child row of ``group`` as a tuple of its cell texts."""
    for index in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(index)
        if item.text(0) != group:
            continue
        for i in range(item.childCount()):
            child = item.child(i)
            if child.text(0) == key:
                return tuple(child.text(c)
                             for c in range(tree.columnCount()))
    raise AssertionError(f"no row {key!r} under {group!r}: {_rows(tree)}")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_the_screen_registers_itself_as_an_app(qapp, registered):
    row = next(r for r in APPS if r[0] == screen.APP_KEY)
    assert row[1] == "Run Compare"
    assert row[3] == "Results & QC"
    from spacr.qt.app import APP_FACTORIES, app_stage
    assert screen.APP_KEY in APP_FACTORIES
    assert app_stage(screen.APP_KEY) == "alpha"


def test_registering_twice_is_a_no_op(qapp, registered):
    assert screen.register() is False
    assert sum(1 for r in APPS if r[0] == screen.APP_KEY) == 1


def test_the_registered_factory_builds_this_screen(qapp, registered, qtbot):
    from spacr.qt.app import APP_FACTORIES
    built = APP_FACTORIES[screen.APP_KEY]()
    qtbot.addWidget(built)
    assert isinstance(built, screen.RunCompareScreen)


def test_the_banner_is_styled_through_the_theme_seam(qapp):
    from spacr.qt.theme import stylesheet, widget_qss_names
    assert "RunCompareBanner" in widget_qss_names()
    qss = stylesheet()
    assert "QFrame#RunCompareBanner" in qss
    # The tables are not restyled here: the shipped QHeaderView chips and
    # ::item:hover accent are what they must inherit.
    assert "RunCompareBanner" not in qss.split("QHeaderView::section")[0][-200:]


# ---------------------------------------------------------------------------
# Loading a project
# ---------------------------------------------------------------------------

def test_a_project_with_no_registry_says_so_rather_than_looking_broken(
        qapp, qtbot, tmp_path):
    empty = tmp_path / "never-run"
    empty.mkdir()
    view = screen.RunCompareScreen()
    qtbot.addWidget(view)
    assert view.load_project(str(empty)) == []
    assert "no artifact registry" in view.verdict_text()
    assert view.comparison() is None


def test_one_run_has_nothing_to_compare_against(qapp, qtbot, tmp_path):
    root = _project(tmp_path, runs=[
        ("only", BASE_SETTINGS, _objects("p1", 1, 1, 3), None, 1)])
    view = screen.RunCompareScreen()
    qtbot.addWidget(view)
    assert len(view.load_project(root)) == 1
    assert "nothing to compare it against" in view.verdict_text()


def test_the_two_newest_runs_are_selected_with_the_newest_as_b(
        qapp, qtbot, tmp_path):
    root = _project(tmp_path, runs=[
        ("old", BASE_SETTINGS, _objects("p1", 1, 1, 3), None, 1),
        ("mid", BASE_SETTINGS, _objects("p1", 1, 1, 4), None, 2),
        ("new", BASE_SETTINGS, _objects("p1", 1, 1, 5), None, 3)])
    view = screen.RunCompareScreen(project=root)
    qtbot.addWidget(view)

    assert [r.run_id for r in view.runs()] == ["new", "mid", "old"]
    a, b = view.selected_runs()
    assert (a.run_id, b.run_id) == ("mid", "new")


def test_the_project_can_be_typed_in(qapp, qtbot, tmp_path):
    root = _project(tmp_path, runs=[
        ("a", BASE_SETTINGS, _objects("p1", 1, 1, 3), None, 1),
        ("b", BASE_SETTINGS, _objects("p1", 1, 1, 3), None, 2)])
    view = screen.RunCompareScreen()
    qtbot.addWidget(view)
    with qtbot.waitSignal(view.project_loaded, timeout=2000):
        view._project_edit.setText(root)
        view._project_edit.returnPressed.emit()
    assert len(view.runs()) == 2


# ---------------------------------------------------------------------------
# The three tabs
# ---------------------------------------------------------------------------

@pytest.fixture
def compared(qapp, qtbot, tmp_path):
    """A screen showing a real comparison with three planted differences."""
    root = _project(tmp_path, runs=[
        ("before", dict(BASE_SETTINGS, cell_diameter=30),
         _objects("p1", 2, 2, 10) + _objects("p2", 2, 2, 10),
         [("gene_a", 2.0, 0.001), ("gene_b", 1.5, 0.01),
          ("gene_c", 1.0, 0.02)], 1),
        ("after", dict(BASE_SETTINGS, cell_diameter=45),
         _objects("p1", 2, 2, 10) + _objects("p2", 2, 2, 6),
         [("gene_a", 2.0, 0.001), ("gene_c", 1.8, 0.02),
          ("gene_d", 0.9, 0.03)], 2),
    ])
    view = screen.RunCompareScreen(project=root)
    qtbot.addWidget(view)
    return view


def test_the_settings_tab_shows_only_what_changed_grouped_by_category(compared):
    from spacr.qt.settings_diff import setting_category
    grouped = _rows(compared._settings_tree)
    category = setting_category("cell_diameter")

    assert list(grouped) == [category]
    assert grouped[category] == ["cell_diameter"]
    assert _child(compared._settings_tree, category, "cell_diameter") == (
        "cell_diameter", "30", "45", "changed")
    for unchanged in ("nucleus_diameter", "save_png"):
        assert all(unchanged not in keys for keys in grouped.values()), (
            f"{unchanged} did not change and must not be on screen")


def test_the_unchanged_toggle_brings_the_rest_back(compared):
    compared._show_all.setChecked(True)
    everything = {key for keys in _rows(compared._settings_tree).values()
                  for key in keys}
    assert {"cell_diameter", "nucleus_diameter", "save_png"} <= everything

    compared._show_all.setChecked(False)
    assert {key for keys in _rows(compared._settings_tree).values()
            for key in keys} == {"cell_diameter"}


def test_the_counts_tab_shows_the_drop_overall_and_per_plate(compared):
    groups = _rows(compared._counts_tree)
    assert list(groups)[0] == "Overall"
    assert set(groups) == {"Overall", "Plate p1", "Plate p2"}

    overall = _child(compared._counts_tree, "Overall", "cell")
    assert overall[1:4] == ("80", "64", "-16")
    assert overall[4] == "-20.0%"

    assert _child(compared._counts_tree, "Plate p1", "cell")[3] == "+0"
    assert _child(compared._counts_tree, "Plate p2", "cell")[3] == "-16"
    assert _child(compared._counts_tree, "Overall", "wells")[3] == "+0"


def test_the_hits_tab_names_what_appeared_vanished_and_moved(compared):
    groups = _rows(compared._hits_tree)
    assert groups["Appeared"] == ["gene_d"]
    assert groups["Vanished"] == ["gene_b"]
    assert groups["Changed rank"] == ["gene_c"]
    assert _child(compared._hits_tree, "Changed rank", "gene_c")[1:4] == (
        "3", "2", "+1")


def test_the_tab_labels_count_what_is_behind_them(compared):
    assert compared._tabs.tabText(0) == "Settings (1)"
    assert compared._tabs.tabText(1).startswith("Counts (")
    assert compared._tabs.tabText(2) == "Hits (1/1)"


def test_the_banner_carries_the_headline(compared):
    text = compared.verdict_text()
    assert "cell" in text and "-20.0%" in text
    assert compared._force_button.isVisibleTo(compared._banner) is False


def test_the_comparison_is_emitted(qapp, qtbot, compared):
    with qtbot.waitSignal(compared.compared, timeout=2000) as caught:
        compared.compare()
    assert caught.args[0].comparable is True
    assert caught.args[0] is compared.comparison()


# ---------------------------------------------------------------------------
# Incomparable runs
# ---------------------------------------------------------------------------

@pytest.fixture
def incomparable(qapp, qtbot, tmp_path):
    """Three runs of three different plates — none of them comparable.

    Three rather than two so that changing the selection lands on another
    incomparable pair; with two runs, any change makes both sides the same
    run, which *is* comparable and would prove nothing.
    """
    root = _project(tmp_path, runs=[
        ("plate_a", BASE_SETTINGS, _objects("plateA", 2, 2, 10), None, 1),
        ("plate_b", BASE_SETTINGS, _objects("plateB", 2, 2, 10), None, 2),
        ("plate_c", BASE_SETTINGS, _objects("plateC", 2, 2, 10), None, 3)])
    view = screen.RunCompareScreen(project=root)
    qtbot.addWidget(view)
    return view


def test_incomparable_runs_are_reported_and_the_tables_stay_empty(incomparable):
    assert incomparable.comparison().comparable is False
    assert "different plates" in incomparable.verdict_text()
    assert "plateB vs plateC" in incomparable.verdict_text(), (
        "the banner has to name the two plates, not just say they differ")
    for tree in (incomparable._settings_tree, incomparable._counts_tree,
                 incomparable._hits_tree):
        assert tree.topLevelItemCount() == 0
    assert incomparable._tabs.tabText(0) == "Settings"


def test_the_blocked_banner_is_marked_for_the_error_style(incomparable):
    assert incomparable._banner.property("blocked") == "true"
    assert incomparable._verdict.property("blocked") == "true"


def test_compare_anyway_draws_the_tables_without_rewriting_the_verdict(
        incomparable):
    assert incomparable._force_button.isVisibleTo(incomparable._banner) is True
    incomparable._force_button.click()

    comparison = incomparable.comparison()
    assert comparison.forced is True
    assert comparison.comparable is True
    assert comparison.comparability.comparable is False
    assert incomparable._counts_tree.topLevelItemCount() > 0


def test_changing_the_selection_drops_the_forcing(incomparable):
    incomparable._force_button.click()
    assert incomparable.comparison().forced is True

    # Move A onto the third run: still an incomparable pair, so the
    # tables must go back to empty rather than staying forced.
    incomparable._a_combo.setCurrentIndex(2)
    assert incomparable.comparison().forced is False
    assert incomparable._counts_tree.topLevelItemCount() == 0


def test_a_version_difference_reaches_the_banner_with_the_tables(
        qapp, qtbot, tmp_path):
    root = _project(tmp_path, runs=[
        ("old", BASE_SETTINGS, _objects("p1", 2, 2, 10), None, 1),
        ("new", BASE_SETTINGS, _objects("p1", 2, 2, 8), None, 2)])
    connection = sqlite3.connect(os.path.join(root, "artifacts.db"))
    connection.execute("UPDATE artifacts SET spacr_version = '1.3.5' "
                       "WHERE run_id = 'old'")
    connection.execute("UPDATE artifacts SET spacr_version = '1.4.0' "
                       "WHERE run_id = 'new'")
    connection.commit()
    connection.close()

    view = screen.RunCompareScreen(project=root)
    qtbot.addWidget(view)
    assert view.comparison().comparable is True
    assert "different spaCR versions" in view.verdict_text()
    assert "1.3.5" in view.verdict_text()
    assert view._counts_tree.topLevelItemCount() > 0
    assert view._banner.property("blocked") == "false"


# ---------------------------------------------------------------------------
# Missing outputs
# ---------------------------------------------------------------------------

def test_a_deleted_database_is_reported_in_the_counts_tab(
        qapp, qtbot, tmp_path):
    root = _project(tmp_path, runs=[
        ("a", BASE_SETTINGS, _objects("p1", 1, 1, 3), None, 1),
        ("b", BASE_SETTINGS, _objects("p1", 1, 1, 3), None, 2)])
    os.remove(os.path.join(root, "measurements", "b.db"))

    view = screen.RunCompareScreen(project=root)
    qtbot.addWidget(view)
    labels = list(_rows(view._counts_tree))
    assert len(labels) == 1
    assert "no longer on disk" in labels[0]


def test_runs_with_no_hit_list_say_so_rather_than_showing_an_empty_table(
        compared, qapp, qtbot, tmp_path):
    root = _project(tmp_path, runs=[
        ("a", BASE_SETTINGS, _objects("p1", 1, 1, 3), None, 1),
        ("b", BASE_SETTINGS, _objects("p1", 1, 1, 3), None, 2)])
    view = screen.RunCompareScreen(project=root)
    qtbot.addWidget(view)
    labels = list(_rows(view._hits_tree))
    assert len(labels) == 1
    assert "no regression results" in labels[0]


def test_identical_settings_say_nothing_changed(qapp, qtbot, tmp_path):
    root = _project(tmp_path, runs=[
        ("a", BASE_SETTINGS, _objects("p1", 1, 1, 3), None, 1),
        ("b", BASE_SETTINGS, _objects("p1", 1, 1, 3), None, 2)])
    view = screen.RunCompareScreen(project=root)
    qtbot.addWidget(view)
    assert list(_rows(view._settings_tree)) == ["No setting changed."]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def test_two_runs_can_be_selected_by_id(qapp, qtbot, tmp_path):
    root = _project(tmp_path, runs=[
        ("one", dict(BASE_SETTINGS, cell_diameter=10),
         _objects("p1", 1, 1, 3), None, 1),
        ("two", dict(BASE_SETTINGS, cell_diameter=20),
         _objects("p1", 1, 1, 3), None, 2),
        ("three", dict(BASE_SETTINGS, cell_diameter=30),
         _objects("p1", 1, 1, 3), None, 3)])
    view = screen.RunCompareScreen(project=root)
    qtbot.addWidget(view)

    view.select("one", "three")
    a, b = view.selected_runs()
    assert (a.run_id, b.run_id) == ("one", "three")
    assert _child(view._settings_tree,
                  view._settings_tree.topLevelItem(0).text(0),
                  "cell_diameter")[1:3] == ("10", "30")


def test_comparing_with_nothing_selected_is_not_a_crash(qapp, qtbot):
    view = screen.RunCompareScreen()
    qtbot.addWidget(view)
    assert view.compare() is None
    assert view.selected_runs() == (None, None)
    # And the toggles must not fire a comparison there is nothing to make.
    view._show_all.setChecked(True)
    assert view.comparison() is None
    view._project_edit.setText("   ")
    view._project_edit.returnPressed.emit()
    assert view.runs() == []


def test_selecting_a_run_id_that_is_not_listed_changes_nothing(
        qapp, qtbot, tmp_path):
    root = _project(tmp_path, runs=[
        ("a", BASE_SETTINGS, _objects("p1", 1, 1, 3), None, 1),
        ("b", BASE_SETTINGS, _objects("p1", 1, 1, 4), None, 2)])
    view = screen.RunCompareScreen(project=root)
    qtbot.addWidget(view)
    before = [r.run_id for r in view.selected_runs()]

    view.select("no-such-run", "nor-this-one")
    assert [r.run_id for r in view.selected_runs()] == before


def test_browsing_to_a_project_loads_it(qapp, qtbot, tmp_path, monkeypatch):
    root = _project(tmp_path, runs=[
        ("a", BASE_SETTINGS, _objects("p1", 1, 1, 3), None, 1),
        ("b", BASE_SETTINGS, _objects("p1", 1, 1, 4), None, 2)])
    view = screen.RunCompareScreen()
    qtbot.addWidget(view)

    monkeypatch.setattr(screen.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: root))
    view._browse_button.click()
    assert len(view.runs()) == 2

    # A cancelled dialog returns "" and must change nothing.
    monkeypatch.setattr(screen.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    view._browse_button.click()
    assert len(view.runs()) == 2


# ---------------------------------------------------------------------------
# The table writers, called directly
# ---------------------------------------------------------------------------

def test_the_table_writers_draw_nothing_for_a_refused_comparison(qapp, qtbot,
                                                                 tmp_path):
    """They are module functions, so their None contract is theirs to keep."""
    from spacr import run_compare as engine

    root = _project(tmp_path, runs=[
        ("a", BASE_SETTINGS, _objects("plateA", 1, 1, 3), None, 1),
        ("b", BASE_SETTINGS, _objects("plateB", 1, 1, 3), None, 2)])
    runs = engine.runs_in(Registry(project=root), root)
    refused = engine.compare_runs(runs[1], runs[0])
    assert refused.settings is None

    for fill, columns in ((screen._fill_settings, screen._SETTINGS_COLUMNS),
                          (screen._fill_counts, screen._COUNT_COLUMNS),
                          (screen._fill_hits, screen._HIT_COLUMNS)):
        tree = screen._tree(columns)
        qtbot.addWidget(tree)
        fill(tree, refused)
        assert tree.topLevelItemCount() == 0


def test_a_count_diff_with_no_overall_rows_draws_only_its_plates(qapp, qtbot):
    from spacr import run_compare as engine

    diff = engine.CountDiff(
        rows=(engine.CountRow("p1", "cell", 10, 8),),
        a=engine.RunCounts(available=True, overall={}),
        b=engine.RunCounts(available=True, overall={}))
    comparison = engine.RunComparison(
        a=engine.RunRef("a"), b=engine.RunRef("b"),
        comparability=engine.Comparability(True), counts=diff)

    tree = screen._tree(screen._COUNT_COLUMNS)
    qtbot.addWidget(tree)
    screen._fill_counts(tree, comparison)
    assert list(_rows(tree)) == ["Plate p1"]


def test_an_unchanged_hit_list_draws_only_the_held_group(qapp, qtbot, tmp_path):
    root = _project(tmp_path, runs=[
        ("a", BASE_SETTINGS, _objects("p1", 1, 1, 3),
         [("gene_a", 2.0, 0.01), ("gene_b", 1.0, 0.02)], 1),
        ("b", BASE_SETTINGS, _objects("p1", 1, 1, 3),
         [("gene_a", 2.0, 0.01), ("gene_b", 1.0, 0.02)], 2)])
    view = screen.RunCompareScreen(project=root)
    qtbot.addWidget(view)

    assert list(_rows(view._hits_tree)) == ["Held rank"]
    assert view._tabs.tabText(2) == "Hits (0/0)"


# ---------------------------------------------------------------------------
# The standalone settings-diff dialog
# ---------------------------------------------------------------------------

def test_the_settings_diff_dialog_tabulates_the_differences(qapp, qtbot):
    """The other consumer of ``settings_diff`` — a modal, two dicts in."""
    from PySide6.QtWidgets import QLabel, QTableWidget

    from spacr.qt.settings_diff import SettingsDiffDialog

    dialog = SettingsDiffDialog(
        {"cell_diameter": 30, "gone": 1},
        {"cell_diameter": 45, "arrived": 2},
        a_label="before", b_label="after")
    qtbot.addWidget(dialog)

    assert "before" in dialog.windowTitle()
    assert "after" in dialog.windowTitle()

    summary = dialog.findChild(QLabel)
    assert "3 differences" in summary.text()
    assert "1 changed" in summary.text()

    table = dialog.findChild(QTableWidget)
    assert table.rowCount() == 3
    cells = {table.item(r, 0).text(): tuple(table.item(r, c).text()
                                            for c in range(1, 4))
             for r in range(table.rowCount())}
    assert cells["cell_diameter"] == ("30", "45", "changed")
    assert cells["arrived"] == ("—", "2", "added")
    assert cells["gone"] == ("1", "—", "removed")

    # Each kind gets its own row colour, so the three are distinguishable
    # without reading the last column.
    colours = {table.item(r, 0).background().color().name()
               for r in range(table.rowCount())}
    assert len(colours) == 3

    dialog.reject()


def test_the_dialog_reads_a_settings_csv_off_disk(qapp, qtbot, tmp_path):
    from PySide6.QtWidgets import QTableWidget

    from spacr.qt.settings_diff import SettingsDiffDialog

    before = tmp_path / "a.csv"
    after = tmp_path / "b.csv"
    before.write_text("Key,Value\ncell_diameter,30\n")
    after.write_text("Key,Value\ncell_diameter,45\n")

    dialog = SettingsDiffDialog(before, after)
    qtbot.addWidget(dialog)
    table = dialog.findChild(QTableWidget)
    assert table.rowCount() == 1
    assert table.item(0, 0).text() == "cell_diameter"
    dialog.reject()
