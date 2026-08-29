"""The data manager's answers when a piece of the picture is missing.

Each test here drives a path the ordinary "registered project, complete
registry, quiet disk" case never reaches: a report with nothing unregistered
in it, a prunability question asked without a registry to answer it with, a
prune whose registry file has gone between the plan and the delete, an
archive of a project that has no provenance at all, and a candidate that
disappears in the moment between being verified and being deleted.

The real project tree from :mod:`tests.test_data_manager` is reused, because
the properties under test are about what is actually on the disk afterwards.
Only the two failures no tree can produce on demand -- ``os.path.realpath``
refusing, a folder removed under the delete loop -- are injected, and each is
aimed at one call.
"""
from __future__ import annotations

import json
import os

import pytest

from spacr import artifacts as A
from spacr import data_manager as DM
from spacr import ports
from tests.test_data_manager import build_project, register_pipeline


@pytest.fixture()
def project(tmp_path):
    """A registered project on disk, with its registry."""
    root = str(tmp_path / "plate1")
    build_project(root)
    return root, register_pipeline(root)


# ---------------------------------------------------------------------------
# _real: a filesystem that will not resolve a path
# ---------------------------------------------------------------------------

def test_a_path_that_cannot_be_resolved_is_used_as_it_was_given(monkeypatch):
    """Containment still answers when ``realpath`` refuses to answer.

    ``_real`` resolves symlinks so a delete cannot be talked out of the
    project through one. On storage where the resolution itself fails -- a
    stale automount, a filesystem that refuses ``lstat`` -- the literal path
    is used instead, so the containment check gives its ordinary answer
    rather than taking the whole disk report down with it.
    """
    def refusing_realpath(path, **kwargs):
        raise OSError(95, "Operation not supported")

    monkeypatch.setattr(DM.os.path, "realpath", refusing_realpath)

    assert DM._real("/projects/plate1/data") == "/projects/plate1/data"
    assert DM._contained("/projects/plate1/data", "/projects/plate1") is True
    assert DM._contained("/elsewhere/data", "/projects/plate1") is False


# ---------------------------------------------------------------------------
# KindUsage.drifted
# ---------------------------------------------------------------------------

def test_a_kind_only_reads_as_drifted_once_the_disk_leaves_the_record(project):
    """Drift is disk against registry, and it is not reported for either alone.

    ``recorded_bytes`` is what the registry stored when the artifact was
    written; ``registered_bytes`` is what is under that artifact now. Raw
    images have no rows at all, so they are not drifted -- they were never
    recorded. Crops match what was recorded until something is dropped into
    the folder, and then they do not, which is the state in which a prune has
    to decline.
    """
    root, registry = project
    usage = DM.scan_project(root, registry=registry)

    raw = usage.kind(ports.RAW_IMAGES)
    assert raw.size_bytes > 0 and raw.n_artifacts == 0
    assert raw.drifted is False

    crops = usage.kind(ports.CROPS)
    assert crops.n_artifacts == 1
    assert crops.recorded_bytes == crops.registered_bytes
    assert crops.drifted is False

    intruder = os.path.join(root, "data", "plate1", "notes.txt")
    with open(intruder, "wb") as handle:
        handle.write(b"someone wrote here after the run")

    after = DM.scan_project(root, registry=registry).kind(ports.CROPS)
    assert after.registered_bytes > after.recorded_bytes
    assert after.drifted is True


# ---------------------------------------------------------------------------
# format_usage
# ---------------------------------------------------------------------------

def test_a_report_with_nothing_unregistered_still_names_what_is_missing(
        tmp_path, monkeypatch):
    """The missing-artifact line is not hidden behind unregistered bytes.

    A project whose every byte is accounted for still has to say when the
    registry claims a path that is no longer there -- that is the one state
    in which a prune must not be trusted.
    """
    monkeypatch.setenv(A.ARTIFACTS_DB_ENV, str(tmp_path / "registry.db"))
    root = str(tmp_path / "plate2")
    os.makedirs(os.path.join(root, "masks"))
    mask = os.path.join(root, "masks", "plate2_A01_1_cell_mask.npy")
    with open(mask, "wb") as handle:
        handle.write(b"\x00" * 128)

    registry = A.open_registry(root)
    registry.register(module="mask", kind=ports.MASKS, path=mask, project=root)
    os.remove(mask)

    usage = DM.scan_project(root, registry=registry)
    assert usage.unregistered_bytes == 0
    assert len(usage.missing) == 1

    report = DM.format_usage(usage)
    assert "1 registered artifact(s) are no longer on disk" in report
    assert "unregistered:" not in report


# ---------------------------------------------------------------------------
# is_prunable without a registry
# ---------------------------------------------------------------------------

def test_without_a_registry_the_input_survival_rule_cannot_be_applied(project):
    """The reason changes when there is no registry to check the inputs in.

    Rule 4 has two halves: every recorded input is still registered and still
    on disk, which needs the registry, and the producing module could run
    here now, which does not. Asked without a registry the first half is
    skipped, and what keeps the artifact is the readiness check instead --
    a different sentence, naming the missing files rather than the missing
    provenance.
    """
    root, registry = project
    crops = next(a for a in registry.by_project(root) if a.kind == ports.CROPS)
    upstream = registry.get(crops.inputs[0])
    assert upstream is not None and os.path.isdir(upstream.path)

    assert DM.is_prunable(crops, root=root, registry=registry) == ""
    assert DM.is_prunable(crops, root=root) == ""

    DM.shutil.rmtree(upstream.path)

    with_registry = DM.is_prunable(crops, root=root, registry=registry)
    assert "is gone, so it cannot be remade" in with_registry
    assert upstream.path in with_registry

    without_registry = DM.is_prunable(crops, root=root)
    assert without_registry != ""
    assert "cannot be remade" not in without_registry
    assert without_registry.startswith("measure could not run here now")


# ---------------------------------------------------------------------------
# _enumerate / PrunePlan.file_list
# ---------------------------------------------------------------------------

def test_a_symlinked_file_inside_a_candidate_is_not_listed_for_deletion(
        tmp_path):
    """A link dropped into a folder is not one of the folder's files.

    The list a user is shown before a delete is the list the delete removes,
    and a recursive delete must never follow a link out of the project: the
    target of that link is somebody else's data.
    """
    outside = tmp_path / "somebody_elses.png"
    outside.write_bytes(b"\x89PNG")
    folder = tmp_path / "plate1" / "data"
    folder.mkdir(parents=True)
    real = folder / "obj_0_0.png"
    real.write_bytes(b"\x89PNG")
    os.symlink(str(outside), str(folder / "linked.png"))

    assert DM._enumerate(str(folder)) == [str(real)]

    plan = DM.PrunePlan(
        root=str(tmp_path / "plate1"),
        candidates=(DM.PruneCandidate(
            path=str(folder), kind=ports.CROPS, module="measure",
            artifact_ids=("a1",), size_bytes=4, n_files=1,
            inventory_digest="deadbeef"),))
    assert plan.file_list() == ((str(real),), False)
    assert outside.exists(), "the link's target was never a candidate"


# ---------------------------------------------------------------------------
# prune with the registry file gone
# ---------------------------------------------------------------------------

def test_a_prune_whose_registry_has_gone_still_deletes_and_marks_no_rows(
        project):
    """The delete is not held hostage by the provenance file disappearing.

    The registry write is step 3 and the delete is step 4; with no registry
    to write to there is nothing to count, so the prune reports zero rows
    marked and removes exactly what the plan named.
    """
    root, registry = project
    plan = DM.plan_prune(root, registry=registry, kinds=[ports.CROPS])
    assert plan.paths == (os.path.join(root, "data"),)

    for suffix in ("", "-wal", "-shm"):
        candidate = os.path.join(root, A.ARTIFACTS_DB_NAME + suffix)
        if os.path.exists(candidate):
            os.remove(candidate)

    result = DM.prune(plan, confirm=plan.token)

    assert result.registry_rows == 0
    assert result.removed_paths == plan.paths
    assert result.freed_bytes == plan.total_bytes
    assert result.n_files == plan.total_files
    assert not os.path.exists(os.path.join(root, "data"))
    assert os.path.isdir(os.path.join(root, "orig")), "the originals stayed"


def test_a_plan_made_without_a_registry_cannot_name_the_downstream_results(
        project):
    """The plan still stands; what it can no longer tell you is what breaks.

    ``downstream`` is the list of results derived from a candidate -- what
    would stop being reproducible once it goes -- and it can only be read out
    of the registry. Handed a scan but no registry, the plan offers the same
    candidates with that warning empty, rather than declining to plan.
    """
    root, registry = project
    with_registry = DM.plan_prune(root, registry=registry, kinds=[ports.CROPS])
    assert with_registry.candidates[0].downstream != ()

    usage = DM.scan_project(root, registry=registry)
    for suffix in ("", "-wal", "-shm"):
        candidate = os.path.join(root, A.ARTIFACTS_DB_NAME + suffix)
        if os.path.exists(candidate):
            os.remove(candidate)

    without_registry = DM.plan_prune(root, usage=usage, kinds=[ports.CROPS])

    assert without_registry.paths == with_registry.paths
    assert without_registry.total_bytes == with_registry.total_bytes
    assert without_registry.candidates[0].downstream == ()


def test_a_candidate_that_vanishes_before_the_delete_is_not_an_error(
        project, monkeypatch):
    """A path removed under the delete loop is already what was wanted.

    The plan is verified against the disk and then acted on, and something
    else on the machine can remove a folder in between -- a pipeline still
    tidying, a second prune. The loop finds nothing to delete, and step 5
    agrees the path is gone, so the run finishes rather than raising
    :class:`PruneIncomplete` over an outcome that is the requested one.
    """
    root, registry = project
    plan = DM.plan_prune(root, registry=registry, kinds=[ports.CROPS])
    doomed = os.path.join(root, "data")
    assert plan.paths == (doomed,)

    real_enumerate = DM._enumerate

    def enumerate_then_lose_it(path):
        listing = real_enumerate(path)
        if path == doomed and os.path.isdir(path):
            DM.shutil.rmtree(path)
        return listing

    monkeypatch.setattr(DM, "_enumerate", enumerate_then_lose_it)

    result = DM.prune(plan, confirm=plan.token)

    assert result.removed_paths == (doomed,)
    assert len(result.removed_files) == plan.total_files
    assert result.freed_bytes == plan.total_bytes
    assert not os.path.exists(doomed)


# ---------------------------------------------------------------------------
# archive without provenance
# ---------------------------------------------------------------------------

def test_an_archive_plan_for_an_empty_project_is_falsy(tmp_path, project):
    """``if plan:`` is how a caller asks whether anything would move."""
    empty = tmp_path / "nothing_here"
    empty.mkdir()
    nothing = DM.plan_archive(str(empty), str(tmp_path / "dest_a"))
    assert bool(nothing) is False
    assert nothing.items == ()

    root, _registry = project
    something = DM.plan_archive(root, str(tmp_path / "dest_b"))
    assert bool(something) is True


def test_a_project_with_no_registry_archives_with_an_empty_provenance_list(
        tmp_path):
    """An unregistered project still moves, and still says where it went.

    There is no registry to read provenance from and none to mark, so the
    manifest carries no artifacts and nothing is registered at the
    destination -- but every file arrives and both records are written, which
    is what makes a folder found nearly empty explainable.
    """
    root = str(tmp_path / "plate3")
    build_project(root)
    assert not os.path.exists(os.path.join(root, A.ARTIFACTS_DB_NAME))
    destination = str(tmp_path / "archive")

    plan = DM.plan_archive(root, destination)
    result = DM.archive(plan, confirm=plan.token)

    assert result.registered == 0
    assert len(result.moved) == len(plan.items)
    assert not os.path.exists(os.path.join(root, "orig"))
    assert os.path.isfile(os.path.join(destination, "orig",
                                       "plate1_A01_1.tif"))
    assert not os.path.exists(os.path.join(destination,
                                           A.ARTIFACTS_DB_NAME))

    with open(result.manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["artifacts"] == []
    assert manifest["origin"] == os.path.abspath(root)

    with open(result.ledger_path, encoding="utf-8") as handle:
        ledger = json.load(handle)
    assert len(ledger) == 1
    assert ledger[0]["destination"] == os.path.abspath(destination)
