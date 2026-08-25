"""The refusals, the unreadable corners and the reports of the data manager.

This module deletes things. Every branch tested here is one where it has to
decline, degrade, or explain instead of proceeding: a folder it cannot read,
a link it must not follow, a registered artifact that is no longer on disk,
a plan whose paths have left the project, a move that did not arrive.

The real project tree from :mod:`tests.test_data_manager` is reused rather
than mocked, because "the number matches the disk" is the property this
module exists for and a fake disk cannot carry it. Only the failures that no
tree can produce on demand -- ``os.stat`` refusing, a move silently doing
nothing -- are injected, and each injection is aimed at exactly one call.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil

import pytest

from spacr import data_manager as DM
from spacr import ports
from tests.test_data_manager import build_project, register_pipeline


@pytest.fixture()
def project(tmp_path):
    """A registered project on disk, with its registry."""
    root = str(tmp_path / "plate1")
    build_project(root)
    return root, register_pipeline(root)


@pytest.fixture()
def unreadable_dir(project):
    """A folder inside the project that the walk is refused entry to."""
    root, registry = project
    locked = os.path.join(root, "locked")
    os.makedirs(locked)
    os.chmod(locked, 0o000)
    try:
        yield root, registry, locked
    finally:
        os.chmod(locked, 0o755)


# ---------------------------------------------------------------------------
# scan_project: what it refuses, and what it cannot read
# ---------------------------------------------------------------------------

def test_scanning_something_that_is_not_a_folder_says_so(tmp_path):
    """A file path where a project was meant would otherwise scan as empty."""
    lonely = tmp_path / "notes.txt"
    lonely.write_text("not a project")
    with pytest.raises(DM.DataManagerError) as excinfo:
        DM.scan_project(str(lonely))
    assert "nothing to measure" in str(excinfo.value)


def test_a_folder_the_walk_cannot_enter_is_reported_not_raised(unreadable_dir):
    """One unreadable corner must not cost the user the rest of the report."""
    root, registry, locked = unreadable_dir
    usage = DM.scan_project(root, registry=registry)
    assert any(locked in problem for problem in usage.errors)
    assert usage.total_bytes > 0, "the rest of the project still measured"


def test_a_file_that_cannot_be_stat_ed_is_an_error_not_a_crash(
        project, monkeypatch):
    """A file that vanishes between listing and sizing is a race, not a bug.

    ``os.walk`` names the file and ``os.stat`` then fails -- the file was
    deleted by the pipeline, or the mount went away. The scan records it and
    keeps counting; raising here would make a size report impossible to get
    from a project that is still being written to.
    """
    root, registry = project
    real_stat = os.stat

    def refusing_stat(path, *args, **kwargs):
        if str(path).endswith("epoch_10.pth"):
            raise PermissionError(13, "Permission denied")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(DM.os, "stat", refusing_stat)
    usage = DM.scan_project(root, registry=registry)
    assert any("epoch_10.pth" in problem for problem in usage.errors)


def test_a_symlinked_file_is_recorded_and_never_counted(project):
    """Its bytes belong to whatever it points at, which may not be ours."""
    root, registry = project
    link = os.path.join(root, "link.tif")
    os.symlink(os.path.join(root, "orig", "plate1_A01_1.tif"), link)
    usage = DM.scan_project(root, registry=registry)
    assert link in usage.symlinks
    assert usage.total_files == len(
        [p for p in _walk_files(root) if not os.path.islink(p)])


def _walk_files(root):
    for dirpath, _dirs, names in os.walk(root, followlinks=False):
        for name in names:
            yield os.path.join(dirpath, name)


# ---------------------------------------------------------------------------
# ProjectUsage accessors and format_usage
# ---------------------------------------------------------------------------

def test_a_kind_the_project_does_not_have_reports_zero_not_none(project):
    """Callers add these up; ``None`` would need a guard at every call site."""
    root, registry = project
    usage = DM.scan_project(root, registry=registry)
    row = usage.kind(ports.SEQUENCING_READS)
    assert row.kind == ports.SEQUENCING_READS
    assert row.size_bytes == 0
    assert row.n_artifacts == 0
    assert row.label


def test_a_kind_with_no_bytes_and_no_artifacts_is_left_out_of_the_report(
        project):
    """A row of zeroes reads as a finding; it is only an absence."""
    root, registry = project
    usage = DM.scan_project(root, registry=registry)
    empty = usage.kind(ports.SEQUENCING_READS)
    padded = dataclasses.replace(usage, kinds=(empty,) + usage.kinds)
    assert empty.label not in DM.format_usage(padded)


def test_the_usage_report_names_missing_artifacts_links_and_unreadable_paths(
        unreadable_dir):
    """Three ways a project disagrees with its registry, in one report."""
    root, registry, locked = unreadable_dir
    shutil.rmtree(os.path.join(root, "model"))
    os.symlink(os.path.join(root, "orig", "plate1_A01_1.tif"),
               os.path.join(root, "link.tif"))
    text = DM.format_usage(DM.scan_project(root, registry=registry))
    assert "no longer on disk" in text
    assert "symlink(s), not followed" in text
    assert "could not read" in text
    assert locked in text


def test_the_usage_object_prints_as_its_report(project):
    """``print(usage)`` is what a user in a notebook actually types."""
    root, registry = project
    usage = DM.scan_project(root, registry=registry)
    assert str(usage) == DM.format_usage(usage)


# ---------------------------------------------------------------------------
# _enumerate and the plan's file list
# ---------------------------------------------------------------------------

def test_enumerating_a_path_that_is_not_there_is_empty_not_an_error(tmp_path):
    """A plan made before a folder went must still be printable."""
    assert DM._enumerate(str(tmp_path / "gone")) == []


def test_a_symlinked_folder_enumerates_as_nothing(tmp_path):
    """Following it is how a recursive delete leaves the project."""
    real = tmp_path / "shared"
    real.mkdir()
    (real / "big.npy").write_bytes(b"x" * 64)
    link = str(tmp_path / "merged")
    os.symlink(str(real), link)
    assert DM._enumerate(link) == []


def test_the_file_list_says_when_it_stopped_counting(project, monkeypatch):
    """A plan over millions of crops must not build a list of millions.

    The cap is :data:`MAX_RECORDED_FILES`; it is lowered here rather than
    writing a hundred thousand files, because what is under test is that the
    truncation is *reported* -- a silent cut would show a user a short list
    and let them believe it was the whole deletion.
    """
    root, registry = project
    plan = DM.plan_prune(root, registry=registry)
    assert plan.candidates, "the fixture offered nothing to prune"
    monkeypatch.setattr(DM, "MAX_RECORDED_FILES", 2)
    paths, truncated = plan.file_list()
    assert truncated is True
    assert len(paths) == 2


def test_the_full_file_list_is_not_marked_truncated(project):
    """The flag has to distinguish a short plan from a cut-off one."""
    root, registry = project
    plan = DM.plan_prune(root, registry=registry)
    paths, truncated = plan.file_list()
    assert truncated is False
    assert len(paths) == plan.total_files


def test_the_plan_prints_as_its_report(project):
    """``print(plan)`` is the confirmation step in a notebook."""
    root, registry = project
    plan = DM.plan_prune(root, registry=registry)
    assert str(plan) == DM.format_prune_plan(plan)


# ---------------------------------------------------------------------------
# plan_prune: what it keeps, and how it explains itself
# ---------------------------------------------------------------------------

def test_an_artifact_the_registry_has_but_the_disk_does_not_is_kept(project):
    """It cannot be deleted and it cannot be counted; it can only be said."""
    root, registry = project
    shutil.rmtree(os.path.join(root, "model"))
    plan = DM.plan_prune(root, registry=registry, kinds=[ports.MODEL_WEIGHTS])
    skips = [s for s in plan.kept if s.path.endswith("model")]
    assert skips, "the vanished artifact was not mentioned at all"
    assert skips[0].reason == "the registry has it but it is not on disk"
    assert skips[0].size_bytes == 0
    assert not plan.candidates


def test_unclaimed_bytes_are_kept_because_nothing_records_what_made_them():
    """The one skip reason that is neither policy nor a safety rule.

    ``other`` means the registry has never heard of the path. There is no
    module to re-run and no settings to re-run it with, so the answer is
    always keep -- and the sentence has to say that rather than "not in the
    kinds you asked for", which would invite the user to ask for it.
    """
    reason = DM._not_selected_reason(DM.OTHER_KIND, [ports.CROPS])
    assert reason == "nothing records what produced this"


def test_a_plan_with_nothing_in_it_says_so_and_offers_no_token(project):
    """An empty plan that printed a token would look like a pending deletion."""
    root, _registry = project
    text = DM.format_prune_plan(DM.PrunePlan(root=root))
    assert text == f"Nothing in {root} can be pruned safely."
    assert "Confirm with token" not in text


def test_the_kept_list_is_cut_short_and_says_how_much_it_hid(project):
    """A hundred skips would bury the candidates the user came to read."""
    root, registry = project
    plan = DM.plan_prune(root, registry=registry)
    assert len(plan.kept) > 1, "the fixture kept nothing to elide"
    text = DM.format_prune_plan(plan, limit=1)
    assert f"… and {len(plan.kept) - 1} more" in text


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------

def test_marking_no_artifacts_writes_no_rows(project):
    """A plan whose candidates carry no registry rows must not open a write."""
    _root, registry = project
    assert DM._mark_artifacts(registry, [], {"pruned_utc": "now"}) == 0


def test_a_candidate_that_left_the_project_aborts_before_any_delete(tmp_path):
    """The last guard between a plan and ``rmtree`` on somebody else's disk.

    A plan is data: it can be loaded from a file, edited, or built by code
    that got the root wrong. :func:`prune` therefore re-checks containment
    itself instead of trusting that :func:`plan_prune` did.
    """
    root = str(tmp_path / "plate1")
    os.makedirs(root)
    elsewhere = tmp_path / "someone_elses"
    elsewhere.mkdir()
    (elsewhere / "irreplaceable.tif").write_bytes(b"x" * 32)
    plan = DM.PrunePlan(
        root=root,
        candidates=(DM.PruneCandidate(
            path=str(elsewhere), kind=ports.CROPS, module="measure",
            artifact_ids=(), size_bytes=32, n_files=1,
            inventory_digest="whatever"),),
        total_bytes=32, total_files=1, token="tok")
    with pytest.raises(DM.PruneAborted) as excinfo:
        DM.prune(plan, confirm="tok")
    assert "not inside" in str(excinfo.value)
    assert (elsewhere / "irreplaceable.tif").exists()


def test_pruning_a_single_file_artifact_removes_the_file(project):
    """Not every artifact is a folder; a candidate can be one file."""
    root, registry = project
    settings_csv = os.path.join(root, "settings", "gen_mask_settings.csv")
    plan = DM.plan_prune(root, registry=registry, kinds=[ports.SETTINGS_CSV])
    assert plan.paths == (settings_csv,)
    DM.prune(plan, confirm=plan.token, registry=registry)
    assert not os.path.exists(settings_csv)
    assert os.path.isdir(os.path.join(root, "settings"))


def test_a_prune_too_big_to_list_still_deletes_and_says_the_list_is_short(
        project, monkeypatch):
    """The recorded file list is a courtesy; the deletion is the contract."""
    root, registry = project
    plan = DM.plan_prune(root, registry=registry, kinds=[ports.CROPS])
    crops = os.path.join(root, "data")
    assert os.path.isdir(crops)
    monkeypatch.setattr(DM, "MAX_RECORDED_FILES", 0)
    result = DM.prune(plan, confirm=plan.token, registry=registry)
    assert result.files_truncated is True
    assert result.removed_files == ()
    assert not os.path.exists(crops)


def test_a_path_that_survives_its_own_deletion_is_reported(project,
                                                           monkeypatch):
    """A busy or permission-locked path leaves the prune half done.

    That is a different state from an abort -- the registry has already
    recorded the artifact as pruned -- so it raises its own type, and says
    the recovery is to run the prune again.
    """
    root, registry = project
    plan = DM.plan_prune(root, registry=registry, kinds=[ports.CROPS])
    monkeypatch.setattr(DM.shutil, "rmtree", lambda *a, **k: None)
    with pytest.raises(DM.PruneIncomplete) as excinfo:
        DM.prune(plan, confirm=plan.token, registry=registry)
    message = str(excinfo.value)
    assert "still there" in message
    assert "run the prune again" in message
    assert os.path.isdir(os.path.join(root, "data"))


# ---------------------------------------------------------------------------
# the archive ledger, plan_archive and archive
# ---------------------------------------------------------------------------

def test_a_ledger_nobody_can_parse_starts_a_new_one(tmp_path):
    """Losing the record about to be written is worse than losing the old."""
    ledger = tmp_path / DM.ARCHIVE_LEDGER_NAME
    ledger.write_text("{ this was half-written when the box went down")
    assert DM._read_ledger(str(ledger)) == []


def test_a_ledger_that_is_not_a_list_is_wrapped_in_one(tmp_path):
    """An older single-record file still has to be carried forward."""
    ledger = tmp_path / DM.ARCHIVE_LEDGER_NAME
    ledger.write_text(json.dumps({"archived_utc": "2020-01-01T00:00:00Z"}))
    assert DM._read_ledger(str(ledger)) == [
        {"archived_utc": "2020-01-01T00:00:00Z"}]


def test_archiving_something_that_is_not_a_folder_says_so(tmp_path):
    """The same refusal as a scan, before a destination is even considered."""
    lonely = tmp_path / "notes.txt"
    lonely.write_text("not a project")
    with pytest.raises(DM.DataManagerError) as excinfo:
        DM.plan_archive(str(lonely), str(tmp_path / "archive"))
    assert "is not a folder" in str(excinfo.value)


def test_a_named_path_that_is_not_there_is_skipped_not_refused(project,
                                                               tmp_path):
    """A stale selection must not stop the entries that do exist from moving."""
    root, registry = project
    plan = DM.plan_archive(
        root, str(tmp_path / "archive"),
        paths=[os.path.join(root, "merged"), os.path.join(root, "gone")],
        registry=registry)
    assert [i.source for i in plan.items] == [os.path.join(root, "merged")]


def test_archiving_an_empty_selection_moves_nothing_and_says_nothing_moved(
        tmp_path):
    """An empty project is archivable; the result is simply empty."""
    root = str(tmp_path / "plate1")
    os.makedirs(root)
    destination = str(tmp_path / "archive")
    plan = DM.plan_archive(root, destination)
    assert plan.items == ()
    result = DM.archive(plan, confirm=plan.token)
    assert result.root == root
    assert result.destination == destination
    assert result.moved == ()
    assert not os.path.exists(os.path.join(destination, "plate1"))


def test_a_move_that_did_not_arrive_stops_the_archive_at_once(project,
                                                              tmp_path,
                                                              monkeypatch):
    """A destination filesystem that accepts a move and drops it is real.

    A full or read-only mount can return from ``shutil.move`` with nothing at
    the far end. The manifest must not be written in that case, because a
    manifest saying seven artifacts arrived is what someone reads *instead*
    of looking at the folder.
    """
    root, registry = project
    destination = str(tmp_path / "archive")
    plan = DM.plan_archive(root, destination, registry=registry)
    assert plan.items, "nothing was planned to move"
    monkeypatch.setattr(DM.shutil, "move", lambda *a, **k: None)
    with pytest.raises(DM.ArchiveError) as excinfo:
        DM.archive(plan, confirm=plan.token, registry=registry)
    assert "not there afterwards" in str(excinfo.value)
    assert not os.path.exists(os.path.join(destination, "manifest.json"))


def test_an_artifact_whose_file_is_already_gone_is_not_registered_at_the_far_end(
        project, tmp_path):
    """The destination registry must describe what actually arrived.

    The registry can name a path that no longer exists -- a file deleted by
    hand after the run. Registering it at the destination anyway would make
    ``by_project`` on the archive list an artifact nobody can open, which is
    the exact failure the manifest exists to prevent.
    """
    root, registry = project
    os.remove(os.path.join(root, "settings", "gen_mask_settings.csv"))
    destination = str(tmp_path / "archive")
    plan = DM.plan_archive(root, destination, registry=registry)
    result = DM.archive(plan, confirm=plan.token, registry=registry)
    assert result.registered >= 1
    from spacr import artifacts as A
    landed = {os.path.basename(a.path)
              for a in A.open_registry(destination).by_project(destination)}
    assert "gen_mask_settings.csv" not in landed
    assert "model" in landed, "the artifacts that did arrive are still there"
    assert not os.path.exists(os.path.join(destination, "settings",
                                           "gen_mask_settings.csv"))


# ---------------------------------------------------------------------------
# is_prunable: every reason an artifact is not regenerable
# ---------------------------------------------------------------------------

@pytest.fixture()
def crops_artifact(project):
    """The registered crops folder, which is prunable as the fixture stands."""
    root, registry = project
    artifact = {a.kind: a for a in registry.by_project(root)}[ports.CROPS]
    assert DM.is_prunable(artifact, root=root, registry=registry) == "", (
        "the fixture artifact must start out prunable, or every test below "
        "would pass for the wrong reason")
    return root, registry, artifact


def test_an_artifact_reached_through_a_symlink_is_never_pruned(crops_artifact):
    """``rmtree`` through a link deletes storage the project does not own."""
    root, registry, artifact = crops_artifact
    link = os.path.join(root, "crops_link")
    os.symlink(os.path.join(root, "data"), link)
    reason = DM.is_prunable(dataclasses.replace(artifact, path=link),
                            root=root, registry=registry)
    assert reason == "it is a symlink; deleting through one leaves the project"
    assert os.path.isdir(os.path.join(root, "data"))


def test_a_kind_no_module_claims_to_produce_cannot_be_remade(crops_artifact):
    """Deleting it would be permanent, whatever the registry row says."""
    root, registry, artifact = crops_artifact
    reason = DM.is_prunable(
        dataclasses.replace(artifact, kind="unicorn-dust"),
        root=root, registry=registry)
    assert "no module declares that it produces unicorn-dust" in reason


def test_a_row_naming_a_module_that_does_not_produce_that_kind_is_kept(
        crops_artifact):
    """The registry disagrees with the ports; re-running would not remake it."""
    root, registry, artifact = crops_artifact
    reason = DM.is_prunable(dataclasses.replace(artifact, module="mask"),
                            root=root, registry=registry)
    assert "mask is not a declared producer" in reason


def test_an_artifact_registered_with_no_fingerprint_is_kept(crops_artifact):
    """With nothing to compare against, "unchanged since the run" is unprovable."""
    root, registry, artifact = crops_artifact
    reason = DM.is_prunable(dataclasses.replace(artifact, fingerprint=""),
                            root=root, registry=registry)
    assert reason == "nothing was on disk when it was registered"


def test_an_artifact_whose_input_has_gone_cannot_be_remade(crops_artifact):
    """The crops came from ``merged/``; without it, re-running produces nothing."""
    root, registry, artifact = crops_artifact
    shutil.rmtree(os.path.join(root, "merged"))
    reason = DM.is_prunable(artifact, root=root, registry=registry)
    assert "is gone, so it cannot be remade" in reason
    assert os.path.join(root, "merged") in reason


def test_a_module_with_no_declared_ports_cannot_be_re_run(crops_artifact,
                                                          monkeypatch):
    """``check_ready`` is what decides "could this run here now?".

    A module it has never heard of is a module nothing can start, so the
    artifact stays. The refusal is injected because a registry row whose
    module is a declared producer of its kind, yet unknown to the port
    table, cannot be built from the ports as they ship.
    """
    root, registry, artifact = crops_artifact

    def unknown(module, **kwargs):
        raise ports.UnknownModule(module)

    monkeypatch.setattr(DM.ports, "check_ready", unknown)
    reason = DM.is_prunable(artifact, root=root, registry=registry)
    assert reason == ("no ports are declared for measure, so it cannot be "
                      "re-run")


def test_an_artifact_already_gone_from_disk_is_not_a_deletion(crops_artifact):
    """Nothing to free, so it must not be counted as bytes a prune recovers."""
    root, registry, artifact = crops_artifact
    shutil.rmtree(os.path.join(root, "data"))
    assert DM.is_prunable(artifact, root=root,
                          registry=registry) == "it is already gone from disk"
