"""Run comparison: settings, count and hit-list diffs.

Every test here builds two *real* runs — a real
:class:`spacr.artifacts.Registry`, real ``measurements.db`` files, real
``results.csv`` files — with differences chosen in advance, and asserts
the diff finds exactly those and nothing else. Nothing is mocked: a
comparison that agrees with a stub about what changed is not evidence.

The four contracts the screen depends on, stated as tests:

* a setting both runs agree on is **absent** from the default view;
* a count that dropped is found, per plate and overall;
* two runs that cannot be compared are **reported as such**, not diffed;
* rank churn is found when the hit **set** is identical.
"""
from __future__ import annotations

import csv
import os
import sqlite3

import pytest

from spacr import run_compare as rc
from spacr.artifacts import Registry
from spacr.qt.settings_diff import (UNCATEGORISED, diff_settings,
                                    diff_settings_grouped, setting_category)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _write_measurements(path, rows):
    """Write a minimal measurements.db.

    ``rows`` is ``{table: [(plateID, rowID, columnID, fieldID), ...]}``,
    which is the identity every spaCR object table carries.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        for table, entries in rows.items():
            connection.execute(
                f'CREATE TABLE "{table}" (plateID TEXT, rowID TEXT, '
                f'columnID TEXT, fieldID TEXT, object_label INTEGER)')
            connection.executemany(
                f'INSERT INTO "{table}" VALUES (?, ?, ?, ?, ?)',
                [(p, r, c, f, i) for i, (p, r, c, f) in enumerate(entries)])
        connection.commit()
    finally:
        connection.close()
    return path


def _objects(plate, wells, fields, per_field):
    """Rows for ``wells`` x ``fields`` x ``per_field`` objects on one plate."""
    out = []
    for well in range(wells):
        for fieldno in range(fields):
            for _ in range(per_field):
                out.append((plate, "r1", f"c{well + 1}", f"f{fieldno + 1}"))
    return out


def _write_hits(path, rows, *, header=("feature", "coefficient", "p_value")):
    """Write a results CSV in the shape ``spacr.ml`` writes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _project(tmp_path, name, *, settings, counts=None, hits=None,
             version="1.3.6", run_id="", module="measure"):
    """Build one project folder with a registry and one run in it.

    :returns: ``(project_root, RunRef)``.
    """
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    registry = Registry(project=str(root))
    run_id = run_id or f"{name}-run"

    db = str(root / "measurements" / "measurements.db")
    _write_measurements(db, counts if counts is not None else
                        {"cell": _objects("plate1", 2, 2, 3)})
    registry.register(module=module, kind="measurements-db", path=db,
                      role="db", settings=settings, run_id=run_id)

    if hits is not None:
        results = str(root / "results" / "results_significant.csv")
        _write_hits(results, hits)
        registry.register(module="regression", kind="regression-results",
                          path=results, role="results", settings=settings,
                          run_id=run_id)

    # The registry stamps the *running* spaCR version; a comparison has to
    # work on runs recorded by other versions, so it is rewritten here
    # rather than being whatever the test machine happens to be on.
    connection = sqlite3.connect(registry.path)
    try:
        connection.execute("UPDATE artifacts SET spacr_version = ? "
                           "WHERE run_id = ?", (version, run_id))
        connection.commit()
    finally:
        connection.close()

    runs = rc.runs_in(registry, str(root))
    return str(root), next(r for r in runs if r.run_id == run_id)


BASE_SETTINGS = {
    "src": "/data/plate1",
    "cell_diameter": 30,
    "nucleus_diameter": 20,
    "channels": [0, 1, 2],
    "save_png": True,
    "verbose": False,
}


# ---------------------------------------------------------------------------
# Settings diff
# ---------------------------------------------------------------------------

def test_only_the_settings_that_moved_are_in_the_default_view():
    """The whole point: an unchanged key must not be in the diff."""
    a = dict(BASE_SETTINGS)
    b = dict(BASE_SETTINGS, cell_diameter=45)
    diff = diff_settings_grouped(a, b)

    assert [(r.key, r.kind) for r in diff.rows] == [("cell_diameter", "changed")]
    keys = {r.key for r in diff.rows}
    for unchanged in ("nucleus_diameter", "channels", "save_png", "src"):
        assert unchanged not in keys, (
            f"{unchanged} did not change and must not be in the diff")
    assert diff.n_changed == 1
    assert diff.n_same == len(BASE_SETTINGS) - 1


def test_the_full_view_carries_the_unchanged_keys_the_default_hid():
    """The toggle is the *only* difference between the two views."""
    a = dict(BASE_SETTINGS)
    b = dict(BASE_SETTINGS, cell_diameter=45)

    default = diff_settings_grouped(a, b)
    full = diff_settings_grouped(a, b, include_same=True)

    assert full.rows == default.rows, "the toggle must not change what differs"
    everything = {r.key for c in full.categories for r in c.rows + c.same}
    assert everything == set(BASE_SETTINGS)
    assert full.n_same == default.n_same
    assert not any(c.same for c in default.categories)


def test_added_and_removed_keys_are_distinguished_from_changed_ones():
    a = {"cell_diameter": 30, "gone": 1}
    b = {"cell_diameter": 30, "arrived": 2}
    rows = {r.key: r.kind for r in diff_settings_grouped(a, b).rows}
    assert rows == {"gone": "removed", "arrived": "added"}


def test_a_two_hundred_key_diff_is_grouped_by_settings_category():
    """Grouping is what makes a big diff readable, so it is asserted."""
    a = dict(BASE_SETTINGS)
    b = dict(BASE_SETTINGS, cell_diameter=45, save_png=False,
             src="/data/plate1")
    diff = diff_settings_grouped(a, b)

    names = [c.category for c in diff.categories]
    assert len(names) == len(set(names)), "a category must appear once"
    assert setting_category("cell_diameter") in names
    assert setting_category("save_png") in names
    # Every differing row is inside the category its key belongs to.
    for block in diff.categories:
        for row in block.rows:
            assert setting_category(row.key) == block.category


def test_an_uncategorised_key_lands_in_other_rather_than_being_dropped():
    diff = diff_settings_grouped({"not_a_real_spacr_setting": 1},
                                 {"not_a_real_spacr_setting": 2})
    assert setting_category("not_a_real_spacr_setting") == UNCATEGORISED
    assert [c.category for c in diff.categories] == [UNCATEGORISED]
    assert diff.rows[0].key == "not_a_real_spacr_setting"


def test_other_sorts_last_and_declared_categories_keep_panel_order():
    from spacr.settings import categories as declared
    declared_names = list(declared)
    a = {"cell_diameter": 1, "src": "/x", "made_up_key": 1}
    b = {"cell_diameter": 2, "src": "/y", "made_up_key": 2}
    names = [c.category for c in diff_settings_grouped(a, b).categories]

    assert names[-1] == UNCATEGORISED
    real = [n for n in names if n != UNCATEGORISED]
    assert real == sorted(real, key=declared_names.index)


def test_identical_settings_produce_an_empty_diff_that_says_so():
    diff = diff_settings_grouped(BASE_SETTINGS, dict(BASE_SETTINGS))
    assert diff.identical
    assert diff.rows == ()
    assert diff.summary() == "Settings are identical."


def test_the_flat_diff_still_answers_what_it_always_did():
    """`diff_settings` predates the grouping and other callers use it."""
    rows = diff_settings({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4})
    assert [(r.key, r.kind) for r in rows] == [("b", "changed"), ("c", "added")]
    assert all(r.kind != "same" for r in rows)


def test_a_string_and_an_int_that_mean_the_same_thing_are_not_a_change():
    """Settings round-trip through CSV, so "30" and 30 are one value."""
    assert diff_settings({"cell_diameter": "30"}, {"cell_diameter": 30}) == []


def test_the_summary_names_the_categories_that_moved():
    diff = diff_settings_grouped(BASE_SETTINGS,
                                 dict(BASE_SETTINGS, cell_diameter=45))
    summary = diff.summary()
    assert "1 changed" in summary
    assert setting_category("cell_diameter") in summary


def test_the_summary_counts_added_and_removed_separately():
    diff = diff_settings_grouped({"cell_diameter": 30, "gone": 1},
                                 {"cell_diameter": 45, "arrived": 2})
    summary = diff.summary()
    assert "1 changed" in summary
    assert "1 added" in summary
    assert "1 removed" in summary
    assert (diff.n_changed, diff.n_added, diff.n_removed) == (1, 1, 1)


def test_one_category_can_be_asked_for_by_name():
    diff = diff_settings_grouped(BASE_SETTINGS,
                                 dict(BASE_SETTINGS, cell_diameter=45))
    name = setting_category("cell_diameter")
    block = diff.category(name)
    assert block is not None
    assert (block.n_changed, block.n_added, block.n_removed) == (1, 0, 0)
    assert len(block) == 1
    assert diff.category("a category nobody declared") is None


def test_a_diff_row_knows_its_own_category():
    row = diff_settings({"cell_diameter": 1}, {"cell_diameter": 2})[0]
    assert row.category == setting_category("cell_diameter")


def test_the_category_map_falls_back_to_other_when_settings_will_not_import(
        monkeypatch):
    """The diff must still render where the heavy settings module is not."""
    import builtins

    from spacr.qt import settings_diff as sd

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        # `from ..settings import categories` reaches __import__ as the
        # bare name "settings" with a level, not as "spacr.settings".
        if name in ("settings", "spacr.settings"):
            raise ImportError("no settings here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(sd, "_CATEGORY_OF", None)
    monkeypatch.setattr(sd, "_CATEGORY_ORDER", ())
    monkeypatch.setattr(builtins, "__import__", refuse)
    try:
        diff = diff_settings_grouped({"cell_diameter": 1}, {"cell_diameter": 2})
    finally:
        monkeypatch.undo()
        sd._CATEGORY_OF = None
        sd._CATEGORY_ORDER = ()

    assert [c.category for c in diff.categories] == [UNCATEGORISED]
    # And the real map comes back for everyone else.
    assert setting_category("cell_diameter") != UNCATEGORISED


def test_the_offline_value_normaliser_still_understands_csv_round_trips():
    """The fallback for when `run_journal` cannot be imported."""
    from spacr.qt.settings_diff import _normalize
    assert _normalize("true") is True
    assert _normalize("False") is False
    assert _normalize(" 30 ") == 30
    assert _normalize("1.5") == 1.5
    assert _normalize("gbr") == "gbr"
    assert _normalize([1, 2]) == [1, 2]


def test_values_equal_falls_back_when_run_journal_is_unavailable(monkeypatch):
    import builtins

    from spacr.qt.settings_diff import _values_equal

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.endswith("run_journal"):
            raise ImportError("no journal here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert _values_equal("30", 30) is True
    assert _values_equal("30", 31) is False


def test_settings_are_loaded_from_a_dict_a_json_file_or_a_csv(tmp_path):
    import json

    from spacr.qt.settings_diff import _load

    assert _load({"a": 1}) == {"a": 1}

    as_json = tmp_path / "settings.json"
    as_json.write_text(json.dumps({"cell_diameter": 30}))
    assert _load(as_json) == {"cell_diameter": 30}

    as_csv = tmp_path / "settings.csv"
    as_csv.write_text("Key,Value\ncell_diameter,30\nempty\n")
    assert _load(as_csv) == {"cell_diameter": "30", "empty": ""}

    with pytest.raises(ValueError):
        _load(tmp_path / "settings.txt")


# ---------------------------------------------------------------------------
# Runs out of the registry
# ---------------------------------------------------------------------------

def test_runs_are_read_from_the_registry_not_the_filesystem(tmp_path):
    root, run = _project(tmp_path, "proj", settings=BASE_SETTINGS)
    registry = Registry(project=root)

    # Delete the outputs. The run must still be listed, with its settings.
    os.remove(os.path.join(root, "measurements", "measurements.db"))
    runs = rc.runs_in(registry, root)

    assert [r.run_id for r in runs] == [run.run_id]
    assert runs[0].settings["cell_diameter"] == 30
    assert runs[0].modules == ("measure",)
    assert runs[0].kinds == ("measurements-db",)


def test_one_run_groups_every_artifact_it_registered(tmp_path):
    root, _ = _project(tmp_path, "proj", settings=BASE_SETTINGS,
                       hits=[("g1", 1.0, 0.01)])
    runs = rc.runs_in(Registry(project=root), root)
    assert len(runs) == 1
    assert runs[0].modules == ("measure", "regression")
    assert set(runs[0].kinds) == {"measurements-db", "regression-results"}


def test_an_artifact_with_no_run_id_becomes_its_own_run(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    registry = Registry(project=str(root))
    db = _write_measurements(str(root / "measurements" / "measurements.db"),
                             {"cell": _objects("p1", 1, 1, 1)})
    artifact = registry.register(module="measure", kind="measurements-db",
                                 path=db, settings=BASE_SETTINGS)

    runs = rc.runs_in(registry, str(root))
    assert [r.run_id for r in runs] == [f"artifact:{artifact.artifact_id}"]


def test_a_run_whose_artifacts_disagree_about_the_version_reads_as_mixed(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    registry = Registry(project=str(root))
    db = _write_measurements(str(root / "measurements" / "measurements.db"),
                             {"cell": _objects("p1", 1, 1, 1)})
    results = _write_hits(str(root / "results" / "results.csv"),
                          [("g1", 1.0, 0.01)])
    registry.register(module="measure", kind="measurements-db", path=db,
                      settings=BASE_SETTINGS, run_id="r1")
    registry.register(module="regression", kind="regression-results",
                      path=results, settings=BASE_SETTINGS, run_id="r1")
    connection = sqlite3.connect(registry.path)
    try:
        connection.execute("UPDATE artifacts SET spacr_version = '1.3.5' "
                           "WHERE kind = 'measurements-db'")
        connection.execute("UPDATE artifacts SET spacr_version = '1.3.6' "
                           "WHERE kind = 'regression-results'")
        connection.commit()
    finally:
        connection.close()

    assert rc.runs_in(registry, str(root))[0].spacr_version == "mixed"


def test_plates_come_off_the_settings_that_named_the_source():
    assert rc.plates_of({"src": ["/data/plate1", "/data/plate2/"]}) == (
        "plate1", "plate2")
    assert rc.plates_of({"src": "/data/plate1"}) == ("plate1",)
    assert rc.plates_of({}) == ()
    assert rc.plates_of(None) == ()


def test_a_producer_that_recorded_its_plates_is_believed(tmp_path):
    """``extra["plates"]`` beats guessing, and beats counting."""
    root = tmp_path / "proj"
    root.mkdir()
    registry = Registry(project=str(root))
    db = _write_measurements(str(root / "measurements" / "measurements.db"),
                             {"cell": _objects("p1", 1, 1, 1)})
    registry.register(module="measure", kind="measurements-db", path=db,
                      settings=BASE_SETTINGS, run_id="r1",
                      extra={"plates": ["plateX", "plateY"]})
    assert rc.runs_in(registry, str(root))[0].plates == ("plateX", "plateY")


# ---------------------------------------------------------------------------
# Comparability
# ---------------------------------------------------------------------------

def test_runs_of_different_plates_are_reported_not_diffed(tmp_path):
    """The whole reason the check exists: two experiments subtract fine.

    The plate identity comes out of the *database*, because ``src`` is
    cosmetic (:data:`spacr.resume.COSMETIC_SETTINGS`) and the registry
    never stores it. If this ever regressed to reading the settings, the
    blocker would silently stop firing on every registry-loaded run.
    """
    _, a = _project(tmp_path, "a", settings=BASE_SETTINGS,
                    counts={"cell": _objects("plateA", 2, 2, 5)})
    _, b = _project(tmp_path, "b", settings=BASE_SETTINGS,
                    counts={"cell": _objects("plateB", 2, 2, 5)})
    assert a.plates == (), "the registry does not store src"

    comparison = rc.compare_runs(a, b)
    assert comparison.comparable is False
    assert [f.code for f in comparison.comparability.blockers] == [
        "different-plates"]
    assert comparison.settings is None
    assert comparison.counts is None
    assert comparison.hits is None
    assert "plateA" in comparison.headline()
    assert "different plates" in comparison.headline()


def test_runs_of_different_modules_have_nothing_to_line_up(tmp_path):
    _, a = _project(tmp_path, "a", settings=BASE_SETTINGS, module="measure")
    _, b = _project(tmp_path, "b", settings=BASE_SETTINGS, module="mask")
    verdict = rc.comparability(a, b)
    assert not verdict
    assert [f.code for f in verdict.blockers] == ["no-shared-module"]


def test_no_shared_output_kind_blocks_even_when_the_module_matches():
    a = rc.RunRef("a", modules=("measure",), kinds=("measurements-db",))
    b = rc.RunRef("b", modules=("measure",), kinds=("crops",))
    verdict = rc.comparability(a, b)
    assert not verdict
    assert [f.code for f in verdict.blockers] == ["no-shared-kind"]


def test_a_version_difference_is_called_out_but_does_not_block(tmp_path):
    """It explains a count change on its own, so it must be on screen."""
    _, a = _project(tmp_path, "a", settings=BASE_SETTINGS, version="1.3.5")
    _, b = _project(tmp_path, "b", settings=BASE_SETTINGS, version="1.3.6")

    verdict = rc.comparability(a, b)
    assert verdict.comparable is True
    assert verdict.version_changed is True
    assert [f.code for f in verdict.warnings if f.code == "version-changed"]
    message = next(f.message for f in verdict.findings
                   if f.code == "version-changed")
    assert "1.3.5" in message and "1.3.6" in message

    comparison = rc.compare_runs(a, b)
    assert comparison.comparable is True
    assert "different spaCR versions" in comparison.headline()


def test_the_same_version_raises_no_version_finding(tmp_path):
    _, a = _project(tmp_path, "a", settings=BASE_SETTINGS, version="1.3.6")
    _, b = _project(tmp_path, "b", settings=BASE_SETTINGS, version="1.3.6")
    assert rc.comparability(a, b).version_changed is False


def test_partial_plate_overlap_warns_rather_than_blocks():
    a = rc.RunRef("a", modules=("measure",), kinds=("measurements-db",),
                  plates=("p1", "p2"))
    b = rc.RunRef("b", modules=("measure",), kinds=("measurements-db",),
                  plates=("p2", "p3"))
    verdict = rc.comparability(a, b)
    assert verdict.comparable
    assert verdict.shared_plates == ("p2",)
    assert "partial-plate-overlap" in {f.code for f in verdict.warnings}


def test_modules_only_one_run_ran_are_warned_about():
    a = rc.RunRef("a", modules=("measure",), kinds=("measurements-db",))
    b = rc.RunRef("b", modules=("measure", "regression"),
                  kinds=("measurements-db",))
    verdict = rc.comparability(a, b)
    assert verdict.comparable
    warning = next(f for f in verdict.warnings
                   if f.code == "partial-module-overlap")
    assert "regression" in warning.message


def test_a_partial_run_is_flagged_as_not_a_whole_run():
    a = rc.RunRef("a", modules=("measure",), kinds=("measurements-db",),
                  status="partial")
    b = rc.RunRef("b", modules=("measure",), kinds=("measurements-db",))
    verdict = rc.comparability(a, b)
    assert verdict.comparable
    assert "partial-run" in {f.code for f in verdict.warnings}


def test_a_different_project_warns_but_still_compares():
    a = rc.RunRef("a", project="/one", modules=("measure",),
                  kinds=("measurements-db",))
    b = rc.RunRef("b", project="/two", modules=("measure",),
                  kinds=("measurements-db",))
    verdict = rc.comparability(a, b)
    assert verdict.comparable
    assert "different-project" in {f.code for f in verdict.warnings}


def test_comparing_a_run_with_itself_says_so():
    a = rc.RunRef("r1", modules=("measure",), kinds=("measurements-db",))
    verdict = rc.comparability(a, a)
    assert verdict.comparable
    assert "same-run" in {f.code for f in verdict.warnings}


def test_force_produces_the_diffs_a_blocker_would_have_withheld(tmp_path):
    _, a = _project(tmp_path, "a", settings=BASE_SETTINGS,
                    counts={"cell": _objects("plateA", 2, 2, 5)})
    _, b = _project(tmp_path, "b", settings=BASE_SETTINGS,
                    counts={"cell": _objects("plateB", 2, 2, 5)})
    assert rc.compare_runs(a, b).comparable is False

    forced = rc.compare_runs(a, b, force=True)
    assert forced.comparable is True
    assert forced.forced is True
    assert forced.settings is not None
    assert forced.counts is not None
    assert not forced.comparability.comparable, (
        "forcing must not rewrite the verdict — the banner still has to "
        "say the runs are of different plates")


def test_findings_are_ordered_blockers_first():
    a = rc.RunRef("a", modules=("measure",), kinds=("measurements-db",),
                  plates=("p1",), spacr_version="1.0")
    b = rc.RunRef("b", modules=("measure",), kinds=("measurements-db",),
                  plates=("p2",), spacr_version="2.0")
    verdict = rc.comparability(a, b)
    assert verdict.findings[0].blocking is True
    assert verdict.findings[-1].blocking is False


# ---------------------------------------------------------------------------
# Count diff
# ---------------------------------------------------------------------------

def test_a_twelve_percent_drop_in_cells_is_found_overall_and_per_plate(tmp_path):
    """Two plates, one of which lost a quarter of its cells."""
    before = {"cell": _objects("p1", 2, 2, 25) + _objects("p2", 2, 2, 25)}
    after = {"cell": _objects("p1", 2, 2, 25) + _objects("p2", 2, 2, 13)}
    _, a = _project(tmp_path, "a", settings=BASE_SETTINGS, counts=before)
    _, b = _project(tmp_path, "b", settings=BASE_SETTINGS, counts=after)

    diff = rc.compare_runs(a, b).counts
    overall = {r.metric: r for r in diff.overall()}
    assert overall["cell"].a == 200 and overall["cell"].b == 152
    assert overall["cell"].delta == -48
    assert overall["cell"].pct == pytest.approx(-24.0)

    # Per plate: p1 untouched, p2 down.
    p1 = {r.metric: r for r in diff.for_plate("p1")}
    p2 = {r.metric: r for r in diff.for_plate("p2")}
    assert p1["cell"].delta == 0
    assert p2["cell"].delta == -48
    assert diff.worst().metric == "cell"
    assert "-24.0%" in diff.headline()


def test_wells_and_fields_are_counted_once_not_once_per_table(tmp_path):
    rows = {"cell": _objects("p1", 3, 4, 2),
            "nucleus": _objects("p1", 3, 4, 1)}
    _, a = _project(tmp_path, "a", settings=BASE_SETTINGS, counts=rows)
    counts = rc.count_database(a.artifact_of("measurements-db").path)

    assert counts.overall["cell"] == 24
    assert counts.overall["nucleus"] == 12
    assert counts.overall["plates"] == 1
    assert counts.overall["wells"] == 3
    assert counts.overall["fields"] == 12
    assert list(counts.overall).count("wells") == 1


def test_a_table_that_vanished_between_runs_shows_as_missing_not_zero(tmp_path):
    _, a = _project(tmp_path, "a", settings=BASE_SETTINGS,
                    counts={"cell": _objects("p1", 1, 1, 4),
                            "pathogen": _objects("p1", 1, 1, 9)})
    _, b = _project(tmp_path, "b", settings=BASE_SETTINGS,
                    counts={"cell": _objects("p1", 1, 1, 4)})

    diff = rc.diff_counts(rc.count_database(a.artifact_of("measurements-db").path),
                          rc.count_database(b.artifact_of("measurements-db").path))
    pathogen = next(r for r in diff.overall() if r.metric == "pathogen")
    assert pathogen.a == 9
    assert pathogen.b is None
    assert pathogen.delta is None
    assert pathogen.pct is None
    assert pathogen.changed is True


def test_counts_that_all_match_say_so(tmp_path):
    rows = {"cell": _objects("p1", 2, 2, 5)}
    _, a = _project(tmp_path, "a", settings=BASE_SETTINGS, counts=rows)
    _, b = _project(tmp_path, "b", settings=BASE_SETTINGS, counts=rows)
    diff = rc.compare_runs(a, b).counts
    assert diff.changed == ()
    assert diff.worst() is None
    assert diff.headline() == "Every count matched."


def test_a_deleted_database_is_reported_rather_than_counted_as_zero(tmp_path):
    _, a = _project(tmp_path, "a", settings=BASE_SETTINGS)
    _, b = _project(tmp_path, "b", settings=BASE_SETTINGS)
    os.remove(b.artifact_of("measurements-db").path)

    diff = rc.compare_runs(a, b).counts
    assert diff.available is False
    assert "no longer on disk" in diff.note
    assert diff.headline() == diff.note


def test_a_database_with_no_measurement_tables_says_which_file(tmp_path):
    path = str(tmp_path / "empty.db")
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE unrelated (x INTEGER)")
    connection.commit()
    connection.close()

    counts = rc.count_database(path)
    assert counts.available is True
    assert counts.overall == {}
    assert "none of the measurement tables" in counts.note
    assert not counts


def test_counting_nothing_at_all_is_a_note_not_an_exception():
    assert rc.count_database("").note == (
        "no measurements database was registered")
    assert rc.count_database(None).available is False


def test_an_unreadable_database_is_a_note(tmp_path):
    path = tmp_path / "not-a-database.db"
    path.write_bytes(b"this is not sqlite" * 100)
    counts = rc.count_database(str(path))
    assert counts.available is False
    assert counts.note


def test_a_table_without_a_plate_column_is_still_counted(tmp_path):
    path = str(tmp_path / "odd.db")
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE cell (something INTEGER)")
    connection.executemany("INSERT INTO cell VALUES (?)", [(i,) for i in range(7)])
    connection.commit()
    connection.close()

    counts = rc.count_database(path)
    assert counts.overall == {"cell": 7}
    assert counts.per_plate == {}


def test_a_count_row_with_a_zero_baseline_has_no_percentage():
    row = rc.CountRow("overall", "cell", 0, 12)
    assert row.delta == 12
    assert row.pct is None
    assert row.changed is True


def test_metrics_are_ordered_object_tables_then_acquisition():
    a = rc.RunCounts(available=True,
                     overall={"wells": 2, "nucleus": 5, "cell": 9, "extra": 1})
    b = rc.RunCounts(available=True,
                     overall={"wells": 2, "nucleus": 5, "cell": 9, "extra": 1})
    order = [r.metric for r in rc.diff_counts(a, b).overall()]
    assert order == ["cell", "nucleus", "wells", "extra"]


def test_a_run_that_only_ran_mask_is_counted_from_its_object_counts(tmp_path):
    """`object-counts` is the fallback, and only the fallback."""
    root = tmp_path / "maskonly"
    root.mkdir()
    registry = Registry(project=str(root))
    db = _write_measurements(str(root / "measurements" / "measurements.db"),
                             {"cell": _objects("p1", 1, 1, 3)})
    registry.register(module="mask", kind="object-counts", path=db,
                      settings=BASE_SETTINGS, run_id="m1")
    run = rc.runs_in(registry, str(root))[0]
    assert rc._database_of(run) == db
    assert rc.count_database(rc._database_of(run)).overall["cell"] == 3


# ---------------------------------------------------------------------------
# Hit-list diff
# ---------------------------------------------------------------------------

HITS_A = [("gene_a", 2.0, 0.001), ("gene_b", 1.5, 0.01),
          ("gene_c", 1.0, 0.02), ("gene_d", 0.5, 0.04)]


def test_rank_churn_is_detected_when_set_membership_is_identical(tmp_path):
    """The same four hits, reshuffled. Membership says nothing; rank does."""
    shuffled = [("gene_a", 2.0, 0.001), ("gene_b", 0.6, 0.01),
                ("gene_c", 1.8, 0.02), ("gene_d", 0.5, 0.04)]
    _, a = _project(tmp_path, "a", settings=BASE_SETTINGS, hits=HITS_A)
    _, b = _project(tmp_path, "b", settings=BASE_SETTINGS, hits=shuffled)

    diff = rc.compare_runs(a, b).hits
    assert set(diff.a.keys) == set(diff.b.keys), "membership must be identical"
    assert diff.appeared == ()
    assert diff.vanished == ()
    assert {c.key for c in diff.moved} == {"gene_b", "gene_c"}
    assert diff.n_shared == 4
    assert diff.churn == pytest.approx(0.5)
    assert diff.identical is False
    assert "changed rank" in diff.headline()

    moved = {c.key: c for c in diff.moved}
    assert (moved["gene_c"].a_rank, moved["gene_c"].b_rank) == (3, 2)
    assert moved["gene_c"].rank_delta == -1


def test_hits_that_appeared_and_vanished_are_named(tmp_path):
    later = [("gene_a", 2.0, 0.001), ("gene_c", 1.0, 0.02),
             ("gene_e", 3.0, 0.001)]
    _, a = _project(tmp_path, "a", settings=BASE_SETTINGS, hits=HITS_A)
    _, b = _project(tmp_path, "b", settings=BASE_SETTINGS, hits=later)

    diff = rc.compare_runs(a, b).hits
    assert [c.key for c in diff.appeared] == ["gene_e"]
    assert sorted(c.key for c in diff.vanished) == ["gene_b", "gene_d"]
    assert diff.appeared[0].a_rank is None
    assert diff.vanished[0].b_rank is None
    assert "1 appeared" in diff.headline()
    assert "2 vanished" in diff.headline()


def test_an_unchanged_hit_list_reports_no_churn(tmp_path):
    _, a = _project(tmp_path, "a", settings=BASE_SETTINGS, hits=HITS_A)
    _, b = _project(tmp_path, "b", settings=BASE_SETTINGS, hits=HITS_A)
    diff = rc.compare_runs(a, b).hits
    assert diff.identical is True
    assert diff.churn == 0.0
    assert len(diff.held) == 4
    assert diff.headline() == "The same 4 hits, in the same order."


def test_ranking_is_by_absolute_effect_so_a_protective_hit_is_not_buried(tmp_path):
    path = _write_hits(str(tmp_path / "r" / "results.csv"),
                       [("weak", 0.2, 0.5), ("protective", -3.0, 0.001),
                        ("strong", 2.0, 0.001)])
    hits = rc.read_hits(path)
    assert hits.keys == ("protective", "strong", "weak")
    assert hits.score_column == "coefficient"
    assert hits.key_column == "feature"


def test_file_order_is_kept_when_there_is_no_score_column(tmp_path):
    path = _write_hits(str(tmp_path / "r" / "results.csv"),
                       [("third",), ("first",), ("second",)],
                       header=("gene",))
    hits = rc.read_hits(path)
    assert hits.keys == ("third", "first", "second")
    assert hits.score_column == ""
    assert hits.hits[0].score is None


def test_file_order_can_be_forced_over_an_available_score(tmp_path):
    path = _write_hits(str(tmp_path / "r" / "results.csv"), HITS_A)
    assert rc.read_hits(path, score_column="-").keys == (
        "gene_a", "gene_b", "gene_c", "gene_d")


def test_a_duplicated_key_is_ranked_once(tmp_path):
    path = _write_hits(str(tmp_path / "r" / "results.csv"),
                       [("gene_a", 2.0, 0.01), ("gene_a", 1.0, 0.02),
                        ("gene_b", 1.5, 0.01)])
    assert rc.read_hits(path).keys == ("gene_a", "gene_b")


def test_the_top_n_can_be_limited(tmp_path):
    path = _write_hits(str(tmp_path / "r" / "results.csv"), HITS_A)
    assert rc.read_hits(path, limit=2).keys == ("gene_a", "gene_b")


def test_a_csv_with_no_name_column_says_so(tmp_path):
    path = _write_hits(str(tmp_path / "r" / "results.csv"),
                       [(1, 2)], header=("x", "y"))
    hits = rc.read_hits(path)
    assert hits.available is False
    assert "no column naming what was hit" in hits.note


def test_a_missing_hit_list_is_reported_not_invented(tmp_path):
    _, a = _project(tmp_path, "a", settings=BASE_SETTINGS)
    _, b = _project(tmp_path, "b", settings=BASE_SETTINGS)
    diff = rc.compare_runs(a, b).hits
    assert diff.available is False
    assert "no regression results" in diff.note
    assert diff.headline() == diff.note


def test_a_deleted_hit_list_is_reported(tmp_path):
    _, a = _project(tmp_path, "a", settings=BASE_SETTINGS, hits=HITS_A)
    _, b = _project(tmp_path, "b", settings=BASE_SETTINGS, hits=HITS_A)
    os.remove(b.artifact_of("regression-results").path)
    diff = rc.compare_runs(a, b).hits
    assert diff.available is False
    assert "no longer on disk" in diff.note


def test_reading_nothing_at_all_is_a_note():
    assert rc.read_hits("").note == "no regression results were registered"
    assert rc.read_hits(None).available is False


def test_a_results_folder_prefers_the_significant_hits_file(tmp_path):
    root = tmp_path / "proj"
    results = root / "results"
    _write_hits(str(results / "results.csv"),
                [("everything", 0.1, 0.9)])
    _write_hits(str(results / "results_significant.csv"),
                [("called", 2.0, 0.001)])
    registry = Registry(project=str(root))
    registry.register(module="regression", kind="regression-results",
                      path=str(results), settings=BASE_SETTINGS, run_id="r")
    run = rc.runs_in(registry, str(root))[0]
    assert rc._hitlist_of(run).endswith("results_significant.csv")


def test_a_results_folder_falls_back_to_any_results_csv(tmp_path):
    root = tmp_path / "proj"
    results = root / "results"
    _write_hits(str(results / "results_by_well.csv"), [("g", 1.0, 0.01)])
    registry = Registry(project=str(root))
    registry.register(module="regression", kind="regression-results",
                      path=str(results), settings=BASE_SETTINGS, run_id="r")
    run = rc.runs_in(registry, str(root))[0]
    assert rc._hitlist_of(run).endswith("results_by_well.csv")


def test_a_run_with_no_regression_artifact_has_no_hit_list():
    assert rc._hitlist_of(rc.RunRef("r")) == ""


# ---------------------------------------------------------------------------
# The whole comparison
# ---------------------------------------------------------------------------

def test_the_three_diffs_find_exactly_the_three_planted_differences(tmp_path):
    """One setting, one count and one hit differ. Nothing else may."""
    _, a = _project(
        tmp_path, "before",
        settings=dict(BASE_SETTINGS, cell_diameter=30),
        counts={"cell": _objects("p1", 2, 2, 10)},
        hits=[("gene_a", 2.0, 0.001), ("gene_b", 1.5, 0.01)])
    _, b = _project(
        tmp_path, "after",
        settings=dict(BASE_SETTINGS, cell_diameter=45),
        counts={"cell": _objects("p1", 2, 2, 8)},
        hits=[("gene_a", 2.0, 0.001), ("gene_c", 1.5, 0.01)])

    comparison = rc.compare_runs(a, b)
    assert comparison.comparable

    assert [(r.key, r.kind) for r in comparison.settings.rows] == [
        ("cell_diameter", "changed")]

    cell = next(r for r in comparison.counts.overall() if r.metric == "cell")
    assert (cell.a, cell.b) == (40, 32)
    assert cell.pct == pytest.approx(-20.0)
    wells = next(r for r in comparison.counts.overall() if r.metric == "wells")
    assert wells.changed is False

    assert [c.key for c in comparison.hits.appeared] == ["gene_c"]
    assert [c.key for c in comparison.hits.vanished] == ["gene_b"]


def test_the_headline_carries_all_three_answers(tmp_path):
    _, a = _project(tmp_path, "a", settings=BASE_SETTINGS,
                    counts={"cell": _objects("p1", 1, 1, 10)},
                    hits=HITS_A, version="1.3.5")
    _, b = _project(tmp_path, "b", settings=dict(BASE_SETTINGS,
                                                 cell_diameter=45),
                    counts={"cell": _objects("p1", 1, 1, 8)},
                    hits=HITS_A, version="1.3.6")
    headline = rc.compare_runs(a, b).headline()
    assert "different spaCR versions" in headline
    assert "1 changed" in headline
    assert "cell" in headline
    assert "same 4 hits" in headline


def test_run_labels_are_readable_in_a_dropdown(tmp_path):
    _, a = _project(tmp_path, "a", settings=BASE_SETTINGS, version="1.3.6")
    assert "measure" in a.label
    assert "spaCR 1.3.6" in a.label
    assert a.artifact_of("measurements-db") is not None
    assert a.artifact_of("crops") is None


def test_runs_in_is_newest_first_and_can_be_limited(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    registry = Registry(project=str(root))
    for index in range(3):
        db = _write_measurements(
            str(root / "measurements" / f"m{index}.db"),
            {"cell": _objects("p1", 1, 1, index + 1)})
        registry.register(module="measure", kind="measurements-db", path=db,
                          settings=BASE_SETTINGS, run_id=f"run{index}")
        connection = sqlite3.connect(registry.path)
        connection.execute("UPDATE artifacts SET created_ns = ? "
                           "WHERE run_id = ?", (index + 1, f"run{index}"))
        connection.commit()
        connection.close()

    runs = rc.runs_in(registry, str(root))
    assert [r.run_id for r in runs] == ["run2", "run1", "run0"]
    assert [r.run_id for r in rc.runs_in(registry, str(root), limit=2)] == [
        "run2", "run1"]


def test_a_finding_renders_as_its_message():
    finding = rc.Finding("code", rc.WARNING, "Something to know.")
    assert str(finding) == "Something to know."
    assert finding.blocking is False
    assert rc.Finding("c", rc.BLOCKING, "Stop.").blocking is True


def test_a_comparability_with_nothing_to_report_says_so():
    a = rc.RunRef("a", modules=("measure",), kinds=("measurements-db",))
    b = rc.RunRef("b", modules=("measure",), kinds=("measurements-db",))
    verdict = rc.comparability(a, b)
    assert verdict.summary() == "These runs are comparable."
    assert bool(verdict) is True


# ---------------------------------------------------------------------------
# The edges, exercised rather than assumed
# ---------------------------------------------------------------------------

def test_a_run_whose_artifacts_recorded_no_version_reads_as_unknown(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    registry = Registry(project=str(root))
    db = _write_measurements(str(root / "measurements" / "measurements.db"),
                             {"cell": _objects("p1", 1, 1, 1)})
    registry.register(module="measure", kind="measurements-db", path=db,
                      settings=BASE_SETTINGS, run_id="r1")
    connection = sqlite3.connect(registry.path)
    connection.execute("UPDATE artifacts SET spacr_version = ''")
    connection.commit()
    connection.close()

    run = rc.runs_in(registry, str(root))[0]
    assert run.spacr_version == ""
    assert "spaCR" not in run.label
    # No version on either side is not a version *difference*.
    assert rc.comparability(run, run).version_changed is False


def test_a_failed_artifact_makes_the_whole_run_failed(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    registry = Registry(project=str(root))
    db = _write_measurements(str(root / "measurements" / "measurements.db"),
                             {"cell": _objects("p1", 1, 1, 1)})
    registry.register(module="measure", kind="measurements-db", path=db,
                      settings=BASE_SETTINGS, run_id="r1", status="failed")
    run = rc.runs_in(registry, str(root))[0]
    assert run.status == "failed"
    other = rc.RunRef("other", modules=("measure",),
                      kinds=("measurements-db",))
    assert "failed-run" in {f.code for f in rc.comparability(run, other).warnings}


def test_a_module_only_the_first_run_ran_is_named_as_such():
    a = rc.RunRef("a", modules=("measure", "umap"),
                  kinds=("measurements-db",))
    b = rc.RunRef("b", modules=("measure",), kinds=("measurements-db",))
    warning = next(f for f in rc.comparability(a, b).warnings
                   if f.code == "partial-module-overlap")
    assert "only the first ran umap" in warning.message


def test_a_run_that_ran_nothing_is_described_as_nothing():
    a = rc.RunRef("a", modules=(), kinds=())
    b = rc.RunRef("b", modules=("measure",), kinds=("measurements-db",))
    blocker = rc.comparability(a, b).blockers[0]
    assert "nothing" in blocker.message


def test_three_names_are_joined_with_commas_and_an_and():
    a = rc.RunRef("a", modules=("measure",), kinds=("measurements-db",),
                  plates=("p1", "p2", "p3"))
    b = rc.RunRef("b", modules=("measure",), kinds=("measurements-db",),
                  plates=("p9",))
    blocker = rc.comparability(a, b).blockers[0]
    assert "p1, p2 and p3" in blocker.message


def test_a_database_that_cannot_even_be_opened_is_a_note(tmp_path, monkeypatch):
    path = str(tmp_path / "locked.db")
    sqlite3.connect(path).close()

    def refuse(*args, **kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(rc.sqlite3, "connect", refuse)
    counts = rc.count_database(path)
    assert counts.available is False
    assert "could not open the database" in counts.note


def test_a_table_sqlite_will_not_count_is_skipped_rather_than_fatal(tmp_path):
    """A view whose definition no longer resolves is skipped, not fatal."""
    path = str(tmp_path / "broken.db")
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE nucleus (plateID TEXT, rowID TEXT, "
                       "columnID TEXT, fieldID TEXT)")
    connection.execute("INSERT INTO nucleus VALUES ('p1', 'r1', 'c1', 'f1')")
    connection.execute("CREATE VIEW cell AS SELECT * FROM missing_table")
    connection.commit()
    connection.close()

    counts = rc.count_database(path)
    assert "cell" not in counts.overall
    assert counts.overall["nucleus"] == 1


def test_a_view_over_the_object_tables_is_counted_like_a_table(tmp_path):
    path = str(tmp_path / "views.db")
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE raw (plateID TEXT, rowID TEXT, "
                       "columnID TEXT, fieldID TEXT)")
    connection.executemany("INSERT INTO raw VALUES (?, ?, ?, ?)",
                           _objects("p1", 2, 2, 3))
    connection.execute("CREATE VIEW cell AS SELECT * FROM raw")
    connection.commit()
    connection.close()

    counts = rc.count_database(path)
    assert counts.overall["cell"] == 12
    assert counts.overall["wells"] == 2
    assert counts.per_plate["p1"]["cell"] == 12


def test_a_table_without_well_columns_still_reports_its_objects(tmp_path):
    path = str(tmp_path / "sparse.db")
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE cell (plateID TEXT)")
    connection.executemany("INSERT INTO cell VALUES (?)",
                           [("p1",), ("p1",), ("p2",)])
    connection.commit()
    connection.close()

    counts = rc.count_database(path)
    assert counts.overall["cell"] == 3
    assert counts.overall["plates"] == 2
    assert "wells" not in counts.overall
    assert "fields" not in counts.overall
    assert counts.per_plate["p1"]["cell"] == 2
    assert counts.plates == ("p1", "p2")


def test_a_grouping_query_that_fails_yields_no_plates(tmp_path):
    path = str(tmp_path / "db.db")
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE cell (plateID TEXT)")
    connection.commit()
    connection.close()
    inner = sqlite3.connect(path)
    try:
        assert rc._grouped(inner, "no_such_table", "plateID") == []
        assert rc._scalar(inner, "SELECT COUNT(*) FROM no_such_table") is None
        assert rc._scalar(inner, "SELECT plateID FROM cell") is None
    finally:
        inner.close()


def test_a_hit_list_that_cannot_be_read_is_a_note(tmp_path, monkeypatch):
    path = _write_hits(str(tmp_path / "r" / "results.csv"), HITS_A)

    real_open = open

    def refuse(target, *args, **kwargs):
        if str(target) == path:
            raise OSError("permission denied")
        return real_open(target, *args, **kwargs)

    monkeypatch.setattr("builtins.open", refuse)
    hits = rc.read_hits(path)
    assert hits.available is False
    assert "could not read the hit list" in hits.note


def test_a_score_that_is_not_a_number_ranks_last(tmp_path):
    """An unparseable effect size must not out-rank a real one."""
    path = _write_hits(str(tmp_path / "r" / "results.csv"),
                       [("broken", "n/a", 0.5), ("real", 1.0, 0.01)])
    hits = rc.read_hits(path)
    assert hits.keys == ("real", "broken")
    assert hits.hits[0].score == 1.0
    assert hits.hits[1].score is None


def test_hit_list_length_and_truthiness():
    empty = rc.HitList()
    assert len(empty) == 0
    assert bool(empty) is False
    filled = rc.HitList(available=True, hits=(rc.Hit("g", 1, 1.0),))
    assert len(filled) == 1
    assert bool(filled) is True


def test_churn_over_no_shared_hits_is_zero(tmp_path):
    a = rc.HitList(available=True, hits=(rc.Hit("only_a", 1, 1.0),))
    b = rc.HitList(available=True, hits=(rc.Hit("only_b", 1, 1.0),))
    diff = rc.diff_hits(a, b)
    assert diff.n_shared == 0
    assert diff.churn == 0.0


def test_a_hit_list_artifact_that_is_a_folder_with_nothing_in_it(tmp_path):
    root = tmp_path / "proj"
    results = root / "results"
    results.mkdir(parents=True)
    registry = Registry(project=str(root))
    registry.register(module="regression", kind="regression-results",
                      path=str(results), settings=BASE_SETTINGS, run_id="r")
    run = rc.runs_in(registry, str(root))[0]
    assert rc._hitlist_of(run) == str(results)
    assert rc.read_hits(rc._hitlist_of(run)).available is False


def test_a_run_with_no_database_artifact_has_no_database():
    assert rc._database_of(rc.RunRef("r")) == ""


def test_a_run_whose_artifacts_carry_no_settings_still_becomes_a_run(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    registry = Registry(project=str(root))
    db = _write_measurements(str(root / "measurements" / "measurements.db"),
                             {"cell": _objects("p1", 1, 1, 1)})
    results = _write_hits(str(root / "results" / "results.csv"),
                          [("g1", 1.0, 0.01)])
    # The first artifact by sort order carries nothing; the second does.
    registry.register(module="regression", kind="regression-results",
                      path=results, run_id="r1")
    registry.register(module="measure", kind="measurements-db", path=db,
                      run_id="r1")
    run = rc.runs_in(registry, str(root))[0]
    assert run.settings == {}
    assert run.settings_hash == ""
    assert set(run.modules) == {"measure", "regression"}


def test_the_first_artifact_that_has_settings_is_the_one_that_wins(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    registry = Registry(project=str(root))
    db = _write_measurements(str(root / "measurements" / "measurements.db"),
                             {"cell": _objects("p1", 1, 1, 1)})
    results = _write_hits(str(root / "results" / "results.csv"),
                          [("g1", 1.0, 0.01)])
    registry.register(module="regression", kind="regression-results",
                      path=results, run_id="r1")
    registry.register(module="measure", kind="measurements-db", path=db,
                      settings=BASE_SETTINGS, run_id="r1")
    connection = sqlite3.connect(registry.path)
    connection.execute("UPDATE artifacts SET created_ns = 1 "
                       "WHERE kind = 'regression-results'")
    connection.execute("UPDATE artifacts SET created_ns = 2 "
                       "WHERE kind = 'measurements-db'")
    connection.commit()
    connection.close()

    run = rc.runs_in(registry, str(root))[0]
    assert run.settings["cell_diameter"] == 30


def test_blank_entries_in_a_plate_list_are_dropped():
    assert rc.plates_of({"src": ["", "/data/plate1", "   "]}) == ("plate1",)


def test_blank_entries_in_a_recorded_plate_list_are_dropped(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    registry = Registry(project=str(root))
    db = _write_measurements(str(root / "measurements" / "measurements.db"),
                             {"cell": _objects("p1", 1, 1, 1)})
    registry.register(module="measure", kind="measurements-db", path=db,
                      run_id="r1", extra={"plates": ["", "plateX", "  "]})
    assert rc.runs_in(registry, str(root))[0].plates == ("plateX",)


def test_an_acquisition_count_sqlite_refuses_is_left_out(tmp_path, monkeypatch):
    """A metric that cannot be counted is absent, never ``None``."""
    path = str(tmp_path / "db.db")
    _write_measurements(path, {"cell": _objects("p1", 2, 2, 2)})

    real_scalar = rc._scalar

    def refuse(connection, sql, params=()):
        if "COUNT(DISTINCT" in sql:
            return None
        return real_scalar(connection, sql, params)

    monkeypatch.setattr(rc, "_scalar", refuse)
    counts = rc.count_database(path)
    assert counts.overall == {"cell": 8}
    assert counts.per_plate == {"p1": {"cell": 8}}


def test_a_hit_headline_with_no_rank_churn_says_only_what_moved_in_or_out():
    a = rc.HitList(available=True, hits=(rc.Hit("gone", 1, 1.0),))
    b = rc.HitList(available=True, hits=(rc.Hit("new", 1, 1.0),))
    diff = rc.diff_hits(a, b)
    assert diff.moved == ()
    assert diff.headline() == "1 appeared; 1 vanished."
