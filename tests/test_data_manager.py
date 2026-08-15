"""What a project costs, and what may be deleted from it.

Every test here runs against a **real project tree on disk** — real ``.npy``
merged arrays with real headers, a real SQLite measurements database with a
real ``png_list``, real PNG crops, a real ``.pth`` blob — registered through
the real :func:`spacr.artifacts.register_run_outputs`. Nothing is faked,
because the two properties under test are "the number matches the disk" and
"the delete does not touch the originals", and neither can be tested against
a mock of the disk.

The safety properties pinned here:

* the size report equals the bytes ``os.stat`` reports, kind by kind;
* raw images are never a prune candidate — not by default, not when asked
  for by name, not when the registry has a row claiming a module made them;
* a file the registry has never heard of is never a candidate, and stays on
  disk through a prune of everything around it;
* the dry-run list is exactly the set of files the deletion removes;
* a count that disagrees with its write aborts the prune with nothing
  deleted — the property :mod:`tests.test_db_contract` establishes for the
  measurement database, held to here for the registry.

CPU-only, offline, deterministic.
"""
from __future__ import annotations

import json
import os
import sqlite3

import numpy as np
import pytest

from spacr import artifacts as A
from spacr import data_manager as DM
from spacr import ports


# ---------------------------------------------------------------------------
# a real project tree
# ---------------------------------------------------------------------------

#: Bytes written into each fabricated file, chosen so every kind has a
#: distinct, checkable size and no two kinds can be confused by their total.
RAW_BYTES = 4096
CROP_BYTES = 512
MODEL_BYTES = 8192


def _write(path: str, payload: bytes) -> str:
    """Write ``payload`` at ``path``, creating parents. Returns the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)
    return path


def build_project(root: str, *, fields=("A01_1", "A01_2"),
                  crops=True, model=True) -> str:
    """Lay out a spaCR project the way the pipeline leaves one.

    ``orig/`` holds the irreplaceable acquisition. ``merged/`` and ``masks/``
    are what Mask wrote from it, ``measurements/measurements.db`` and
    ``data/`` are what Measure wrote from ``merged/``, and ``model/`` is what
    Classify trained on the crops.
    """
    os.makedirs(root, exist_ok=True)

    for folder in ("orig", "merged", "masks"):
        os.makedirs(os.path.join(root, folder), exist_ok=True)
    for field in fields:
        # The originals: real TIFF-named files off the microscope.
        _write(os.path.join(root, "orig", f"plate1_{field}.tif"),
               b"\x00" * RAW_BYTES)
        # Merged arrays: 3-D with at least two planes, which is the shape
        # contract `measure` declares and `check_ready` enforces.
        np.save(os.path.join(root, "merged", f"plate1_{field}.npy"),
                np.zeros((8, 8, 3), dtype=np.uint16))
        np.save(os.path.join(root, "masks", f"plate1_{field}_cell_mask.npy"),
                np.zeros((8, 8), dtype=np.uint16))

    if crops:
        for index, field in enumerate(fields):
            folder = os.path.join(root, "data", "plate1", f"{field}_png")
            for obj in range(3):
                _write(os.path.join(folder, f"obj_{index}_{obj}.png"),
                       b"\x89PNG" + b"c" * (CROP_BYTES - 4))

    if model:
        _write(os.path.join(root, "model", "epoch_10.pth"),
               b"\x80\x02" + b"m" * (MODEL_BYTES - 2))

    _write(os.path.join(root, "settings", "gen_mask_settings.csv"),
           b"setting,value\nsrc," + root.encode() + b"\n")

    build_measurements_db(root, fields)
    return root


def build_measurements_db(root: str, fields) -> str:
    """Write a measurements database with the tables the ports declare.

    ``png_list`` has to exist and hold rows or ``check_ready('classify')``
    reports the project as not ready, which would make the crops
    un-regenerable and quietly remove them from every plan.
    """
    path = os.path.join(root, "measurements", "measurements.db")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            'CREATE TABLE IF NOT EXISTS png_list ('
            '"plateID" TEXT, "rowID" TEXT, "columnID" TEXT, "fieldID" TEXT, '
            '"png_path" TEXT, "file_name" TEXT)')
        connection.execute(
            'CREATE TABLE IF NOT EXISTS cell ('
            '"plateID" TEXT, "rowID" TEXT, "columnID" TEXT, "fieldID" TEXT, '
            '"object_label" INTEGER, "area" REAL)')
        connection.execute(
            'CREATE TABLE IF NOT EXISTS object_counts ('
            '"count_type" TEXT, "object_count" INTEGER)')
        for index, field in enumerate(fields):
            for obj in range(3):
                connection.execute(
                    'INSERT INTO png_list VALUES (?, ?, ?, ?, ?, ?)',
                    ("plate1", "r1", "c1", f"f{index}",
                     os.path.join(root, "data", "plate1", f"{field}_png",
                                  f"obj_{index}_{obj}.png"),
                     f"obj_{index}_{obj}.png"))
                connection.execute(
                    'INSERT INTO cell VALUES (?, ?, ?, ?, ?, ?)',
                    ("plate1", "r1", "c1", f"f{index}", obj, 100.0 + obj))
        connection.execute(
            'INSERT INTO object_counts VALUES (?, ?)', ("cell", 6))
        connection.commit()
    finally:
        connection.close()
    return path


def register_pipeline(root: str, settings=None) -> A.Registry:
    """Register the runs that produced the tree, through the real hook."""
    settings = dict(settings or {"src": root, "cell_diameter": 30})
    registry = A.open_registry(root)
    for module in ("mask", "measure", "classify"):
        A.register_run_outputs(module, settings, roots=[root],
                               registry=registry, run_id=f"run-{module}")
    return registry


@pytest.fixture()
def project(tmp_path):
    """A registered project on disk, with its registry."""
    root = str(tmp_path / "plate1")
    build_project(root)
    registry = register_pipeline(root)
    return root, registry


def du(path: str) -> int:
    """Bytes under ``path``, the way ``du --apparent-size`` counts them."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, followlinks=False):
        for name in filenames:
            full = os.path.join(dirpath, name)
            if not os.path.islink(full):
                total += os.path.getsize(full)
    return total


def registry_bytes(path: str) -> int:
    """Current registry storage, including transient SQLite sidecars."""
    return sum(
        os.path.getsize(candidate)
        for candidate in (path, f"{path}-wal", f"{path}-shm")
        if os.path.isfile(candidate)
    )


def n_files(path: str) -> int:
    """Files under ``path``, symlinks excluded."""
    if os.path.isfile(path):
        return 1
    return sum(
        1 for dirpath, _d, names in os.walk(path, followlinks=False)
        for name in names if not os.path.islink(os.path.join(dirpath, name)))


# ---------------------------------------------------------------------------
# 1. The size report is the disk
# ---------------------------------------------------------------------------

def test_the_project_total_is_every_byte_on_disk(project):
    """``total_bytes`` equals what walking the tree independently says."""
    root, registry = project
    usage = DM.scan_project(root, registry=registry)
    assert usage.total_bytes == du(root)
    assert usage.total_files == n_files(root)


@pytest.mark.parametrize("folder,kind", [
    ("merged", ports.MERGED_ARRAYS),
    ("masks", ports.MASKS),
    ("data", ports.CROPS),
    ("model", ports.MODEL_WEIGHTS),
])
def test_each_kind_reports_the_bytes_its_folder_actually_holds(
        project, folder, kind):
    """Per-kind bytes are measured, not taken from the registry's record."""
    root, registry = project
    usage = DM.scan_project(root, registry=registry)
    row = usage.kind(kind)
    on_disk = du(os.path.join(root, folder))
    assert on_disk > 0, "the fixture wrote nothing there"
    assert row.size_bytes == on_disk
    assert row.n_files == n_files(os.path.join(root, folder))


def test_the_measurement_database_is_reported_once_not_three_times(project):
    """Three kinds live in one file; its bytes are counted once.

    ``mask`` registers object-counts there, ``measure`` the measurements and
    ``classify`` the predictions. Summing the kinds must still give the size
    of the project, or every number in the report is inflated.
    """
    root, registry = project
    usage = DM.scan_project(root, registry=registry)
    assert sum(row.size_bytes for row in usage.kinds) == usage.total_bytes

    entry = usage.artifact_at(os.path.join(root, "measurements",
                                           "measurements.db"))
    assert entry is not None
    assert len(entry.kinds) >= 2, entry.kinds
    assert entry.kind == ports.MEASUREMENTS_DB
    assert entry.size_bytes == du(os.path.join(root, "measurements",
                                               "measurements.db"))

    # The other two say where their bytes went rather than reading as zero.
    counts = usage.kind(ports.OBJECT_COUNTS)
    assert counts.n_artifacts == 1 and counts.size_bytes == 0
    assert counts.shared_paths == 1
    assert "counted under another kind" in DM.format_usage(usage)


def test_raw_images_are_reported_as_the_largest_thing_nobody_registered(
        project):
    """``orig/`` is measured and labelled, and it is unregistered.

    Nothing in :mod:`spacr.ports` produces raw images, so nothing registers
    them. They still have to appear in the report — a disk breakdown that
    omits the acquisition is useless — and they have to appear as
    unregistered, which is what keeps them out of every plan.
    """
    root, registry = project
    usage = DM.scan_project(root, registry=registry)
    row = usage.kind(ports.RAW_IMAGES)
    assert row.size_bytes == du(os.path.join(root, "orig"))
    assert row.unregistered_bytes == row.size_bytes
    assert row.n_artifacts == 0
    assert usage.unregistered_bytes >= row.size_bytes


def test_a_symlink_is_reported_and_never_walked_into(tmp_path):
    """A link into somebody else's storage is not this project's bytes."""
    outside = tmp_path / "shared"
    outside.mkdir()
    (outside / "huge.npy").write_bytes(b"x" * 100_000)
    root = str(tmp_path / "plate1")
    build_project(root, fields=("A01_1",))
    os.symlink(str(outside), os.path.join(root, "linked"))

    usage = DM.scan_project(root)
    assert usage.symlinks == (os.path.join(root, "linked"),)
    assert usage.total_bytes < 100_000, (
        "the walk followed a symlink out of the project")


def test_a_registry_that_claims_a_missing_path_says_so(project):
    """A registered artifact whose folder was deleted is reported missing."""
    root, registry = project
    import shutil
    shutil.rmtree(os.path.join(root, "masks"))
    usage = DM.scan_project(root, registry=registry)
    assert any(a.kind == ports.MASKS for a in usage.missing)
    assert usage.kind(ports.MASKS).size_bytes == 0


def test_a_project_with_no_registry_scans_and_owns_nothing(tmp_path):
    """Every byte of an unregistered project is unregistered. On purpose."""
    root = str(tmp_path / "plate1")
    build_project(root, fields=("A01_1",))
    usage = DM.scan_project(root)
    assert usage.total_bytes == du(root)
    assert usage.unregistered_bytes == usage.total_bytes
    assert usage.artifacts == ()
    assert not DM.plan_prune(root).candidates


def test_format_usage_names_the_kinds_and_the_unregistered_bytes(project):
    root, registry = project
    text = DM.format_usage(DM.scan_project(root, registry=registry))
    assert "merged arrays" in text
    assert "unregistered" in text
    assert DM.human_bytes(du(root)) in text


def test_human_bytes_uses_the_units_a_disk_quota_is_written_in():
    assert DM.human_bytes(0) == "0 B"
    assert DM.human_bytes(999) == "999 B"
    assert DM.human_bytes(1000) == "1.0 kB"
    assert DM.human_bytes(2_500_000_000) == "2.5 GB"
    assert DM.human_bytes(3_000_000_000_000_000_000).endswith("PB")


# ---------------------------------------------------------------------------
# 2. Originals are never selected
# ---------------------------------------------------------------------------

def test_the_default_plan_never_offers_an_original(project):
    """No candidate is under ``orig/``, and the raw bytes stay put."""
    root, registry = project
    plan = DM.plan_prune(root, registry=registry)
    for candidate in plan.candidates:
        assert not candidate.path.startswith(os.path.join(root, "orig"))
        assert candidate.kind not in DM.ORIGINAL_KINDS


@pytest.mark.parametrize("kind", DM.ORIGINAL_KINDS)
def test_asking_for_an_original_by_name_still_offers_nothing(project, kind):
    """``kinds=['raw-images']`` is not a way in. There is no way in."""
    root, registry = project
    plan = DM.plan_prune(root, registry=registry, kinds=[kind])
    assert plan.candidates == ()
    assert kind not in plan.kinds


def test_an_original_with_a_registry_row_claiming_a_producer_is_still_kept(
        project):
    """The last line of defence, against a registry that says otherwise.

    Registering raw images as though ``mask`` had produced them is exactly
    the shape a buggy producer, a hand-edited registry or a future module
    would take. :func:`is_prunable` must refuse on the kind alone, before it
    ever consults the producer graph.
    """
    root, registry = project
    forged = registry.register(
        module="mask", kind=ports.RAW_IMAGES, role="images",
        path=os.path.join(root, "orig"), project=root,
        settings={"src": root}, status=A.STATUS_COMPLETE)

    reason = DM.is_prunable(forged, root=root, registry=registry)
    assert reason, "a forged producer made an original prunable"
    assert "original" in reason

    plan = DM.plan_prune(root, registry=registry,
                         kinds=[ports.RAW_IMAGES, ports.MERGED_ARRAYS])
    assert all(c.kind != ports.RAW_IMAGES for c in plan.candidates)
    assert os.path.isdir(os.path.join(root, "orig"))


def test_a_kind_no_module_produces_is_not_prunable(project):
    """The general rule the original rule is a special case of."""
    root, registry = project
    assert ports.producers_of(ports.RAW_IMAGES) == ()
    assert ports.producers_of(ports.CHANNEL_STACKS) == (), (
        "a module now declares it produces channel stacks; this test should "
        "become an assertion about the new producer instead")


def test_the_model_and_the_database_are_kept_unless_named(project):
    """Protected kinds are absent by default and explain themselves."""
    root, registry = project
    plan = DM.plan_prune(root, registry=registry)
    kept = {os.path.basename(s.path): s.reason for s in plan.kept}
    assert "model" in kept and "kept by default" in kept["model"]
    assert all(c.kind != ports.MODEL_WEIGHTS for c in plan.candidates)


# ---------------------------------------------------------------------------
# 3. Unregistered means excluded
# ---------------------------------------------------------------------------

def test_an_artifact_absent_from_the_registry_is_never_a_candidate(project):
    """A folder spaCR never registered is not offered, and survives."""
    root, registry = project
    orphan = os.path.join(root, "someone_elses_analysis")
    _write(os.path.join(orphan, "hand_made.csv"), b"gene,score\nA,1\n")

    plan = DM.plan_prune(root, registry=registry)
    assert all(not path.startswith(orphan) for path in plan.paths)

    usage = DM.scan_project(root, registry=registry)
    assert usage.artifact_at(orphan) is None
    assert any(path.startswith(orphan) for path, _ in usage.unregistered)

    if plan.candidates:
        DM.prune(plan, confirm=plan.token, registry=registry)
    assert os.path.isfile(os.path.join(orphan, "hand_made.csv")), (
        "a prune removed a file nothing in the registry claimed")


def test_a_file_dropped_into_a_registered_folder_makes_it_unprunable(project):
    """Unknown content inside a known folder poisons the whole folder.

    The folder's registered fingerprint covers every file under it, so a
    file somebody added afterwards changes it. That is the only defence
    against "I put my only copy of something in ``merged/``", and it has to
    be the fingerprint rather than a count, because a replaced file of the
    same size would otherwise pass.
    """
    root, registry = project
    before = DM.plan_prune(root, registry=registry)
    assert any(c.path == os.path.join(root, "merged")
               for c in before.candidates), "the fixture offers no merged/"

    _write(os.path.join(root, "merged", "my_only_copy.txt"), b"irreplaceable")

    after = DM.plan_prune(root, registry=registry)
    assert all(c.path != os.path.join(root, "merged")
               for c in after.candidates)
    reason = next(s.reason for s in after.kept
                  if s.path == os.path.join(root, "merged"))
    assert "not what was registered" in reason


def test_a_registered_path_outside_the_project_is_not_touched(tmp_path):
    """Containment is checked against the real path, not the string."""
    root = str(tmp_path / "plate1")
    elsewhere = str(tmp_path / "elsewhere")
    build_project(root, fields=("A01_1",))
    os.makedirs(elsewhere, exist_ok=True)
    _write(os.path.join(elsewhere, "shared.npy"), b"x" * 100)

    registry = register_pipeline(root)
    outside = registry.register(
        module="mask", kind=ports.MERGED_ARRAYS, role="merged",
        path=elsewhere, project=root, settings={"src": root})

    usage = DM.scan_project(root, registry=registry)
    assert outside.artifact_id in {a.artifact_id for a in usage.outside}
    assert DM.is_prunable(outside, root=root, registry=registry)
    plan = DM.plan_prune(root, registry=registry)
    assert all(not path.startswith(elsewhere) for path in plan.paths)


# ---------------------------------------------------------------------------
# 4. The dry run is the deletion
# ---------------------------------------------------------------------------

def test_the_plan_lists_exactly_the_files_the_deletion_removes(project):
    """The list a user is shown is the list that goes. Byte for byte."""
    root, registry = project
    plan = DM.plan_prune(root, registry=registry)
    assert plan.candidates, "nothing to prune; the fixture is wrong"

    planned, truncated = plan.file_list()
    assert not truncated
    assert planned, "an empty plan cannot demonstrate anything"

    result = DM.prune(plan, confirm=plan.token, registry=registry)
    assert set(result.removed_files) == set(planned)
    assert result.removed_paths == plan.paths
    for path in planned:
        assert not os.path.exists(path)


def test_the_plan_frees_exactly_the_bytes_it_promised(project):
    """The size the user was shown is the size the disk gives back."""
    root, registry = project
    before = du(root)
    before_registry = registry_bytes(registry.path)
    plan = DM.plan_prune(root, registry=registry)
    promised = plan.total_bytes
    assert promised > 0

    result = DM.prune(plan, confirm=plan.token, registry=registry)
    assert result.freed_bytes == promised
    # Pruning intentionally records its audit facts in the registry before
    # deleting files. SQLite may reuse a page or allocate one plus WAL/SHM
    # sidecars, so the registry's own byte delta is not deleted payload. The
    # promised artifact bytes must disappear exactly either way.
    assert du(root) - registry_bytes(registry.path) == (
        before - before_registry - promised)


def test_a_prune_without_the_token_refuses_and_deletes_nothing(project):
    root, registry = project
    plan = DM.plan_prune(root, registry=registry)
    before = du(root)
    with pytest.raises(DM.ConfirmationRequired):
        DM.prune(plan, confirm="yes", registry=registry)
    with pytest.raises(DM.ConfirmationRequired):
        DM.prune(plan, confirm="", registry=registry)
    assert du(root) == before


def test_a_token_from_one_plan_cannot_authorise_another(project):
    """The token is bound to the paths and sizes it was shown with."""
    root, registry = project
    everything = DM.plan_prune(root, registry=registry)
    merged_only = DM.plan_prune(
        root, registry=registry, paths=[os.path.join(root, "merged")])
    assert merged_only.candidates
    assert everything.token != merged_only.token
    with pytest.raises(DM.ConfirmationRequired):
        DM.prune(everything, confirm=merged_only.token, registry=registry)
    assert os.path.isdir(os.path.join(root, "data"))


def test_a_tree_that_changed_after_the_plan_aborts_the_whole_prune(project):
    """One drifted candidate stops every candidate. Nothing is deleted."""
    root, registry = project
    plan = DM.plan_prune(root, registry=registry)
    assert len(plan.candidates) > 1, "need two candidates to show the abort"
    before = du(root)

    _write(os.path.join(root, "merged", "arrived_late.npy"), b"late")

    with pytest.raises(DM.PruneAborted) as caught:
        DM.prune(plan, confirm=plan.token, registry=registry)
    assert "Nothing was deleted" in str(caught.value)
    assert du(root) == before + 4
    for candidate in plan.candidates:
        assert os.path.exists(candidate.path)


def test_the_plan_explains_every_thing_it_kept(project):
    """A user who expected more space back can read why they did not."""
    root, registry = project
    plan = DM.plan_prune(root, registry=registry)
    assert plan.kept
    assert all(skip.reason for skip in plan.kept)
    text = DM.format_prune_plan(plan)
    assert "cannot be undone" in text
    assert plan.token in text
    assert "get it back" in text


def test_an_empty_plan_is_a_no_op_that_still_needs_its_token(tmp_path):
    root = str(tmp_path / "plate1")
    os.makedirs(root)
    plan = DM.plan_prune(root)
    assert not plan
    assert DM.prune(plan, confirm=plan.token).removed_paths == ()


# ---------------------------------------------------------------------------
# 5. Count first, delete second, verify
# ---------------------------------------------------------------------------

def test_a_count_that_disagrees_with_its_write_aborts_and_removes_nothing(
        project, monkeypatch):
    """The equality is what stops the write, and it stops it in time.

    ``tests/test_db_contract`` establishes for the measurement database that
    neither ``rowid`` nor the declared key identifies a row, and that the
    only property worth asserting is that a write changes exactly what its
    count said. The registry write here is held to the same standard, and
    the ordering is what makes it worth anything: the registry is written
    *before* a single file is removed, so a mismatch leaves the disk
    untouched.
    """
    root, registry = project
    plan = DM.plan_prune(root, registry=registry)
    assert plan.candidates
    before = du(root)
    files_before = {p for p, _ in DM.scan_project(root).unregistered}

    real = DM._count_matching

    def _lying_count(connection, table, predicate, params):
        return real(connection, table, predicate, params) + 1

    monkeypatch.setattr(DM, "_count_matching", _lying_count)

    with pytest.raises(DM.PruneAborted) as caught:
        DM.prune(plan, confirm=plan.token, registry=registry)
    assert "rolled back" in str(caught.value)
    assert "nothing on disk was deleted" in str(caught.value).lower()

    assert du(root) == before
    for candidate in plan.candidates:
        assert os.path.exists(candidate.path)
    assert {p for p, _ in DM.scan_project(root).unregistered} == files_before


def test_the_rollback_leaves_the_registry_exactly_as_it_was(project,
                                                            monkeypatch):
    """A refused write is a write that did not happen."""
    root, registry = project
    plan = DM.plan_prune(root, registry=registry)
    before = {a.artifact_id: (a.extra, a.status)
              for a in registry.by_project(root)}

    monkeypatch.setattr(DM, "_count_matching", lambda *a, **k: 99)
    with pytest.raises(DM.PruneAborted):
        DM.prune(plan, confirm=plan.token, registry=registry)

    monkeypatch.undo()
    after = {a.artifact_id: (a.extra, a.status)
             for a in registry.by_project(root)}
    assert after == before


def test_a_successful_prune_records_what_happened_without_losing_the_recipe(
        project):
    """The row survives the deletion, marked. It is how the data comes back."""
    root, registry = project
    plan = DM.plan_prune(root, registry=registry)
    ids = [i for c in plan.candidates for i in c.artifact_ids]
    result = DM.prune(plan, confirm=plan.token, registry=registry)
    assert result.registry_rows == len(set(ids))

    for artifact_id in ids:
        row = registry.get(artifact_id)
        assert row is not None, "the settings that made it were thrown away"
        assert row.extra.get("pruned_utc")
        assert row.settings_hash
        assert not row.exists


def test_forget_rows_really_deletes_them_and_verifies_the_count(project):
    """The DELETE path, on the same counted predicate."""
    root, registry = project
    plan = DM.plan_prune(root, registry=registry)
    ids = {i for c in plan.candidates for i in c.artifact_ids}
    result = DM.prune(plan, confirm=plan.token, registry=registry,
                      forget_rows=True)
    assert result.forgotten
    assert result.registry_rows == len(ids)
    for artifact_id in ids:
        assert registry.get(artifact_id) is None


def test_the_verified_write_writes_what_it_counted_or_writes_nothing(tmp_path):
    """The helper on its own, against a table with the shape that bit twice.

    ``probe`` declares a column called ``rowID``, so ``rowid`` in this table
    is the plate row and not a row identity — the exact condition under
    which a delete once removed a whole table. The helper never names a row:
    it counts with a predicate and writes with the same one.
    """
    path = str(tmp_path / "probe.db")
    connection = sqlite3.connect(path)
    connection.execute('CREATE TABLE probe ("rowID" TEXT, "value" INTEGER)')
    connection.executemany('INSERT INTO probe VALUES (?, ?)',
                           [("r1", 1), ("r1", 2), ("r2", 3)])
    connection.commit()

    removed = DM._verified_write(connection, "probe", '"rowID" = ?', ["r1"],
                                 'DELETE FROM "probe"', what="test")
    assert removed == 2
    assert connection.execute("SELECT COUNT(*) FROM probe").fetchone()[0] == 1
    connection.close()


def test_the_verified_write_refuses_when_the_count_and_the_write_disagree(
        tmp_path, monkeypatch):
    """A lying count is caught by the equality, not by luck."""
    path = str(tmp_path / "probe.db")
    connection = sqlite3.connect(path)
    connection.execute('CREATE TABLE probe ("rowID" TEXT, "value" INTEGER)')
    connection.executemany('INSERT INTO probe VALUES (?, ?)',
                           [("r1", 1), ("r1", 2), ("r2", 3)])
    connection.commit()

    monkeypatch.setattr(DM, "_count_matching",
                        lambda *a, **k: 99)
    with pytest.raises(DM.PruneAborted) as caught:
        DM._verified_write(connection, "probe", '"rowID" = ?', ["r1"],
                           'DELETE FROM "probe"', what="test")
    assert "99" in str(caught.value)
    connection.close()


# ---------------------------------------------------------------------------
# 6. Shared paths
# ---------------------------------------------------------------------------

def test_the_database_is_never_offered_by_default(project):
    """Three kinds live in one file, and none of them is a default target."""
    root, registry = project
    db = os.path.join(root, "measurements", "measurements.db")
    entry = DM.scan_project(root, registry=registry).artifact_at(db)
    assert entry is not None and len(entry.kinds) >= 2, entry

    plan = DM.plan_prune(root, registry=registry)
    assert all(c.path != db for c in plan.candidates)
    assert "kept by default" in next(s.reason for s in plan.kept
                                     if s.path == db)


def test_one_veto_at_a_shared_path_keeps_the_whole_file(project):
    """A path is prunable only when *every* artifact on it is.

    This is the shape that made ``foreign.py``'s keyed delete destructive:
    two things that are not the same thing sitting behind one identity.
    Here the identity is a path — ``measurements.db`` carries the
    measurements, the object counts and the model's predictions — and
    deleting it for one kind takes the other two with it. So the file is
    offered only when all three could be made again, and one artifact that
    could not keeps it.
    """
    root, registry = project
    db = os.path.join(root, "measurements", "measurements.db")
    named = [ports.MEASUREMENTS_DB, ports.OBJECT_COUNTS, ports.PREDICTIONS]

    # Everything in it is regenerable, so naming the kinds explicitly does
    # offer it. That is the baseline the veto below has to overturn.
    permissive = DM.plan_prune(root, registry=registry, kinds=named)
    assert any(c.path == db for c in permissive.candidates), (
        "the fixture cannot demonstrate a veto if the file is never offered")

    # Now one of the three stops being reproducible: the model's scores came
    # out of a run that did not finish.
    scores = registry.latest(ports.PREDICTIONS, project=root)
    assert scores is not None
    registry.register(module=scores.module, kind=scores.kind,
                      role=scores.role, path=scores.path, project=root,
                      settings_digest=scores.settings_hash,
                      run_id=scores.run_id, status=A.STATUS_PARTIAL)

    plan = DM.plan_prune(root, registry=registry, kinds=named)
    assert all(c.path != db for c in plan.candidates), (
        "the database was offered while one of the kinds in it could not be "
        "made again")
    reason = next(s.reason for s in plan.kept if s.path == db)
    assert "the same path also holds" in reason
    assert os.path.isfile(db)


def test_two_registered_artifacts_one_inside_the_other_keep_each_other(
        project):
    """Nested registrations are kept, because the byte total would lie.

    Bytes are attributed to the innermost registered path, so a plan that
    offered the outer one would free more than it said. Nothing declares
    such a pair today; the guard exists because a plan under-reporting what
    it deletes is the exact failure this module is about.
    """
    root, registry = project
    inner = os.path.join(root, "data", "plate1")
    registry.register(module="measure", kind=ports.CROPS, role="crops",
                      path=inner, project=root,
                      settings={"src": root, "cell_diameter": 30})

    plan = DM.plan_prune(root, registry=registry)
    assert all(c.path not in (inner, os.path.join(root, "data"))
               for c in plan.candidates)
    reasons = [s.reason for s in plan.kept if s.path == inner]
    assert reasons and "inside or around it" in reasons[0]
    assert os.path.isdir(inner)


def test_prune_refuses_a_hand_built_plan_whose_paths_nest(project):
    """The same guard on the execution side, for a plan built elsewhere."""
    root, registry = project
    plan = DM.plan_prune(root, registry=registry)
    crops = next(c for c in plan.candidates if c.kind == ports.CROPS)
    forged = DM.PrunePlan(
        root=root,
        candidates=(crops,
                    DM.PruneCandidate(
                        path=os.path.join(crops.path, "plate1"),
                        kind=ports.CROPS, module="measure",
                        artifact_ids=(), size_bytes=0, n_files=0,
                        inventory_digest="")),
        token="t")
    with pytest.raises(DM.PruneAborted) as caught:
        DM.prune(forged, confirm="t", registry=registry)
    assert "Nothing was deleted" in str(caught.value)
    assert os.path.isdir(crops.path)


def test_the_registry_file_itself_is_never_a_candidate(project):
    """Deleting the provenance to save disk is not a trade this makes."""
    root, registry = project
    forged = registry.register(
        module="measure", kind=ports.CROPS, role="crops",
        path=registry.path, project=root, settings={"src": root})
    assert "provenance record" in DM.is_prunable(forged, root=root,
                                                 registry=registry)


# ---------------------------------------------------------------------------
# 7. Regenerability is checked, not assumed
# ---------------------------------------------------------------------------

def test_nothing_is_prunable_once_its_inputs_are_gone(project):
    """No raw images, no re-run, no prune. The rule that carries the feature.

    Deleting ``orig/`` by hand is what a user does when they think the
    merged arrays are enough. From that moment ``merged/`` is the only copy
    of the pixels, and it stops being prunable — which the plan says in
    words rather than by silently offering less.
    """
    root, registry = project
    import shutil
    assert any(c.kind == ports.MERGED_ARRAYS
               for c in DM.plan_prune(root, registry=registry).candidates)

    shutil.rmtree(os.path.join(root, "orig"))

    plan = DM.plan_prune(root, registry=registry)
    assert all(c.kind != ports.MERGED_ARRAYS for c in plan.candidates)
    reason = next(s.reason for s in plan.kept
                  if s.path == os.path.join(root, "merged"))
    assert "mask could not run here now" in reason


def test_a_partial_run_is_not_prunable(project):
    """A run that lost fields cannot be reproduced by re-running it."""
    root, registry = project
    partial = registry.register(
        module="mask", kind=ports.MERGED_ARRAYS, role="merged",
        path=os.path.join(root, "merged"), project=root,
        settings={"src": root, "cell_diameter": 30},
        status=A.STATUS_PARTIAL)
    reason = DM.is_prunable(partial, root=root, registry=registry)
    assert "partial" in reason


def test_an_artifact_whose_recorded_input_vanished_is_not_prunable(project):
    """Provenance with a hole in it is provenance nobody can replay."""
    root, registry = project
    crops = registry.latest(ports.CROPS, project=root)
    assert crops is not None and crops.inputs, "measure recorded no inputs"
    registry.forget(crops.inputs[0])

    reason = DM.is_prunable(registry.get(crops.artifact_id), root=root,
                            registry=registry)
    assert "no longer in the registry" in reason
    assert all(c.kind != ports.CROPS
               for c in DM.plan_prune(root, registry=registry).candidates)


def test_a_candidate_names_the_module_that_makes_it_again(project):
    root, registry = project
    plan = DM.plan_prune(root, registry=registry)
    by_kind = {c.kind: c for c in plan.candidates}
    assert by_kind[ports.MERGED_ARRAYS].module == "mask"
    assert by_kind[ports.MERGED_ARRAYS].regenerate_with.startswith("re-run mask")
    assert by_kind[ports.CROPS].module == "measure"


def test_a_candidate_reports_what_was_derived_from_it(project):
    """The cost of pruning is not only the bytes."""
    root, registry = project
    plan = DM.plan_prune(root, registry=registry)
    merged = next(c for c in plan.candidates if c.kind == ports.MERGED_ARRAYS)
    assert merged.downstream, (
        "measure's outputs were derived from merged/ and are not reported")


# ---------------------------------------------------------------------------
# 8. Archiving
# ---------------------------------------------------------------------------

def test_archiving_a_whole_project_moves_it_and_leaves_a_record(project,
                                                               tmp_path):
    """The bytes arrive, the origin says where they went, and so does the
    destination's own registry."""
    root, registry = project
    destination = str(tmp_path / "cold_storage" / "plate1")
    before = du(root)

    plan = DM.plan_archive(root, destination, registry=registry)
    assert plan.whole_project
    assert plan.total_bytes == before

    result = DM.archive(plan, confirm=plan.token, registry=registry)
    assert du(destination) >= before
    assert os.path.isfile(result.manifest_path)
    assert os.path.isfile(result.ledger_path)
    assert os.path.isdir(os.path.join(destination, "merged"))
    assert not os.path.exists(os.path.join(root, "merged"))

    with open(result.manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["origin"] == root
    assert manifest["destination"] == destination
    assert manifest["artifacts"], "the manifest carries no provenance"

    with open(result.ledger_path, encoding="utf-8") as handle:
        ledger = json.load(handle)
    assert ledger[0]["destination"] == destination

    moved_registry = A.open_registry(destination)
    rows = moved_registry.by_project(destination)
    assert rows, "the destination registry knows nothing"
    assert any(r.extra.get("archived_from", "").startswith(root) for r in rows)
    # Every artifact travels, including the ones nested inside a moved
    # folder — measurements.db lives under measurements/, and a destination
    # that describes four of a project's seven artifacts is not a record.
    assert {r.kind for r in rows} >= {
        ports.MERGED_ARRAYS, ports.CROPS, ports.MASKS, ports.MODEL_WEIGHTS,
        ports.MEASUREMENTS_DB, ports.SETTINGS_CSV}, sorted(
            {r.kind for r in rows})
    assert {a["kind"] for a in manifest["artifacts"]} == {r.kind for r in rows}


def test_archiving_a_subset_leaves_the_rest_and_marks_the_origin(project,
                                                                 tmp_path):
    root, registry = project
    destination = str(tmp_path / "cold" / "crops")
    crops = os.path.join(root, "data")

    plan = DM.plan_archive(root, destination, registry=registry,
                           paths=[crops])
    assert not plan.whole_project
    assert plan.total_bytes == du(crops)

    DM.archive(plan, confirm=plan.token, registry=registry)
    assert not os.path.exists(crops)
    assert os.path.isdir(os.path.join(root, "merged"))
    assert os.path.isdir(os.path.join(destination, "data"))

    moved = registry.latest(ports.CROPS, project=root)
    assert moved is not None
    assert moved.extra.get("archived_to") == destination


def test_an_archive_without_its_token_moves_nothing(project, tmp_path):
    root, registry = project
    plan = DM.plan_archive(root, str(tmp_path / "cold"), registry=registry)
    with pytest.raises(DM.ConfirmationRequired):
        DM.archive(plan, confirm="please", registry=registry)
    assert os.path.isdir(os.path.join(root, "merged"))


def test_an_archive_never_overwrites_what_is_already_there(project, tmp_path):
    root, registry = project
    destination = str(tmp_path / "cold")
    os.makedirs(os.path.join(destination, "merged"))
    plan = DM.plan_archive(root, destination, registry=registry)
    with pytest.raises(DM.ArchiveError) as caught:
        DM.archive(plan, confirm=plan.token, registry=registry)
    assert "never overwrites" in str(caught.value)
    assert os.path.isdir(os.path.join(root, "merged"))
    assert os.listdir(os.path.join(destination, "merged")) == []


def test_an_archive_into_itself_is_refused_at_planning_time(project):
    root, _registry = project
    with pytest.raises(DM.DataManagerError):
        DM.plan_archive(root, os.path.join(root, "archive"))


def test_archiving_something_outside_the_project_is_refused(project, tmp_path):
    root, registry = project
    with pytest.raises(DM.DataManagerError):
        DM.plan_archive(root, str(tmp_path / "cold"),
                        paths=[str(tmp_path / "elsewhere")])


def test_the_ledger_accumulates_rather_than_overwriting(project, tmp_path):
    """Two archives out of one project leave two records, not one."""
    root, registry = project
    first = DM.plan_archive(root, str(tmp_path / "cold_a"), registry=registry,
                            paths=[os.path.join(root, "data")])
    DM.archive(first, confirm=first.token, registry=registry)
    second = DM.plan_archive(root, str(tmp_path / "cold_b"),
                             registry=registry,
                             paths=[os.path.join(root, "masks")])
    result = DM.archive(second, confirm=second.token, registry=registry)

    with open(result.ledger_path, encoding="utf-8") as handle:
        ledger = json.load(handle)
    assert len(ledger) == 2
    assert {row["destination"] for row in ledger} == {
        str(tmp_path / "cold_a"), str(tmp_path / "cold_b")}


def test_a_whole_project_archive_keeps_the_earlier_records(project, tmp_path):
    """The ledger moves with the data; the origin's history is not lost.

    A whole-project archive moves every child of the root, and the ledger is
    one of them. Reading it before the move is what stops the earlier
    archives from being forgotten because the file that recorded them went
    with the data.
    """
    root, registry = project
    first = DM.plan_archive(root, str(tmp_path / "cold_a"), registry=registry,
                            paths=[os.path.join(root, "data")])
    DM.archive(first, confirm=first.token, registry=registry)

    second = DM.plan_archive(root, str(tmp_path / "cold_b"), registry=registry)
    assert second.whole_project
    result = DM.archive(second, confirm=second.token, registry=registry)

    with open(result.ledger_path, encoding="utf-8") as handle:
        ledger = json.load(handle)
    assert [row["destination"] for row in ledger] == [
        str(tmp_path / "cold_a"), str(tmp_path / "cold_b")]


# ---------------------------------------------------------------------------
# 9. The whole thing, once, in order
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_scan_plan_prune_leaves_a_project_that_can_be_rebuilt(project):
    """The end-to-end claim: what is left is enough to make the rest again."""
    root, registry = project
    usage = DM.scan_project(root, registry=registry)
    assert usage.total_bytes == du(root)

    plan = DM.plan_prune(root, registry=registry)
    freed = plan.total_bytes
    assert freed > 0

    DM.prune(plan, confirm=plan.token, registry=registry)

    # The originals are all still there.
    assert du(os.path.join(root, "orig")) > 0
    # And so is everything needed to run Mask again, which is what regenerates
    # every single thing that was just deleted.
    readiness = ports.check_ready("mask", root=root)
    assert readiness.ok, readiness.reason

    after = DM.scan_project(root, registry=registry)
    assert after.total_bytes == usage.total_bytes - freed
