"""The project browser's defensive edges, driven rather than assumed.

Every path here is one a real folder can put the browser on: a registry file
that is no longer a database, a plate deleted while it is being measured, a
directory entry on a mount that went away between the listing and the stat,
a spaCR build in which the ``mask`` module is not declared. The contract
:func:`spacr.projects.scan` states is that none of them raise -- a table that
refuses to appear because one row was odd is worse than a table with one odd
row in it -- and the assertions here are on what the caller gets back
instead.

Filesystem and registry only: no Qt, no torch, no images.
"""
from __future__ import annotations

import errno
import logging
import os
import shutil

import numpy as np

from spacr import artifacts
from spacr import data_manager as _dm
from spacr import ports as _ports
from spacr import projects
from spacr.projects import ProjectSummary, StaleArtifact


# ---------------------------------------------------------------------------
# Fixtures shaped like the folders the browser meets
# ---------------------------------------------------------------------------

def _plate(root, name="plate1", *, merged=1, raw=False):
    """A project folder holding exactly the declared outputs asked for."""
    plate = os.path.join(str(root), name)
    os.makedirs(plate, exist_ok=True)
    for index in range(merged):
        os.makedirs(os.path.join(plate, "merged"), exist_ok=True)
        np.save(os.path.join(plate, "merged", f"f{index}.npy"),
                np.zeros((4, 4, 3), dtype=np.uint16))
    if raw:
        with open(os.path.join(plate, f"{name}_A01_T0001F001L01A01Z01C01.tif"),
                  "wb") as handle:
            handle.write(b"tif")
    return plate


def _wreck(registry_file):
    """Replace a registry with bytes SQLite will not open.

    The shape a half-written file on a share, or a truncated copy, takes:
    the file is there, so it is opened, and it is not a database.
    """
    for suffix in ("-wal", "-shm"):
        side = registry_file + suffix
        if os.path.exists(side):
            os.remove(side)
    with open(registry_file, "wb") as handle:
        handle.write(b"NOT-A-DATABASE" * 64)


# ---------------------------------------------------------------------------
# Dating a port
# ---------------------------------------------------------------------------

def test_a_port_bound_to_no_root_at_all_is_undated_not_stat_ed():
    """``declared_inputs`` with no project resolves ports to an empty target.

    The port names nothing, so there is nothing to date, and the answer is
    the same 0 an absent port gives -- not an exception out of ``os.stat``.
    """
    port = _ports.Port(kind="merged-arrays", role="merged")
    resolved = _ports.resolve_port(port, "")
    assert resolved.target == ""
    assert projects._newest_mtime_ns(resolved) == 0


def test_a_port_that_names_one_folder_is_dated_from_that_folder(tmp_path):
    """A pattern-less port matches no paths, so its own target is the date.

    ``masks/`` is declared as a plain folder: nothing lands in
    :attr:`spacr.ports.ResolvedPort.paths` for it, and dropping the target
    from the sample would leave every such output undated -- which is the
    filesystem half of "last run".
    """
    masks = tmp_path / "masks"
    masks.mkdir()
    os.utime(masks, (1_500_000_000, 1_500_000_000))
    resolved = _ports.resolve_port(
        _ports.Port(kind=_ports.MASKS, role="masks", path="masks"),
        str(tmp_path))
    assert resolved.paths == ()
    assert resolved.target == str(masks)
    assert projects._newest_mtime_ns(resolved) == 1_500_000_000_000_000_000


def test_a_patterned_port_is_dated_from_the_files_it_matched(tmp_path):
    """The paths carry the date when the port itself names no single file."""
    matched = tmp_path / "f0.npy"
    matched.write_bytes(b"x")
    os.utime(matched, (1_000_000_000, 1_000_000_000))
    resolved = _ports.ResolvedPort(
        port=_ports.Port(kind="merged-arrays", role="merged"),
        root=str(tmp_path), target="", paths=(str(matched),),
        exists=True, count=1)
    assert projects._newest_mtime_ns(resolved) == 1_000_000_000_000_000_000


# ---------------------------------------------------------------------------
# What a module says when it has not run
# ---------------------------------------------------------------------------

def test_a_module_with_an_on_disk_signature_says_plainly_that_it_did_not_run(
        tmp_path):
    """"not run here" is only honest for a module that would have left a mark.

    ``mask`` writes ``merged/``, so its absence is an answer. The other
    sentence -- "it writes nothing only it could have written" -- belongs to
    modules with no signature at all, and must not be used for this one.
    """
    empty = tmp_path / "nothing_here"
    empty.mkdir()
    states = {s.module: s for s in projects.module_states(empty)}
    mask = states["mask"]
    assert mask.state == projects.STATE_ABSENT
    assert mask.detectable is True
    assert mask.describe() == "mask: not run here"


# ---------------------------------------------------------------------------
# What counts as a project
# ---------------------------------------------------------------------------

def test_a_registry_file_alone_makes_a_folder_a_project(tmp_path):
    """spaCR recorded something here once; the outputs may since be gone."""
    folder = tmp_path / "emptied_plate"
    folder.mkdir()
    assert projects.looks_like_project(folder) is False
    artifacts.open_registry(str(folder))
    assert os.path.isfile(
        os.path.join(str(folder), artifacts.ARTIFACTS_DB_NAME))
    assert projects.looks_like_project(folder) is True
    # And nothing has "run" in it -- the registry file is the whole evidence.
    assert projects.scan(folder, with_next_steps=False).stage == ""


def test_raw_images_stop_being_evidence_when_mask_is_not_declared(
        tmp_path, monkeypatch):
    """The raw-data clause is the mask module's own input declaration.

    A build with no ``mask`` in :data:`spacr.ports.PORTS` -- a stripped
    install, or a plugin registry that replaced it -- has no declaration to
    ask, so a bare folder of TIFFs is not claimed as a project rather than
    the browser raising ``UnknownModule`` at it.
    """
    plate = _plate(tmp_path, name="fresh", merged=0, raw=True)
    assert projects.looks_like_project(plate) is True
    monkeypatch.delitem(_ports.PORTS, "mask")
    assert projects.looks_like_project(plate) is False


# ---------------------------------------------------------------------------
# Discovery over a filesystem that misbehaves
# ---------------------------------------------------------------------------

class _StaleEntry:
    """A directory entry whose mount went away after it was listed."""

    def __init__(self, entry):
        self.name = entry.name
        self.path = entry.path

    def is_dir(self, *, follow_symlinks=True):
        raise OSError(errno.ESTALE, "Stale file handle")

    def is_file(self, *, follow_symlinks=True):
        raise OSError(errno.ESTALE, "Stale file handle")


class _Listing:
    """What ``os.scandir`` returns: iterable, and a context manager."""

    def __init__(self, entries):
        self._entries = entries

    def __iter__(self):
        return iter(self._entries)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def close(self):
        pass


def test_an_entry_that_cannot_be_stat_ed_is_skipped_and_the_rest_still_listed(
        tmp_path, monkeypatch):
    """One dead mount in a folder of plates costs that one row, not the list."""
    good = _plate(tmp_path, name="good_plate", merged=1)
    (tmp_path / "stale_mount").mkdir()
    real_scandir = os.scandir

    def _scandir(path="."):
        if os.fspath(path) == str(tmp_path):
            return _Listing([_StaleEntry(entry) if entry.name == "stale_mount"
                             else entry for entry in real_scandir(path)])
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", _scandir)
    assert projects.discover([tmp_path], depth=2) == (good,)


# ---------------------------------------------------------------------------
# Summaries built by hand, as a caller may hold them
# ---------------------------------------------------------------------------

def test_a_summary_whose_module_list_was_trimmed_still_names_its_stage():
    """The stage is a fact of its own, not a lookup into ``modules``.

    A caller holding a summary with the per-module detail dropped -- a saved
    row, a filtered list -- still gets the stage back rather than a blank
    cell.
    """
    summary = ProjectSummary(root="/data/p7", name="p7", stage="measure",
                             modules=())
    assert summary.stage_label == "measure"


def test_stale_results_with_nothing_gone_are_counted_as_stale_only():
    """Out of date and no longer there are separate columns of one answer."""
    entry = StaleArtifact(artifact_id="1", kind="measurements-db",
                          module="measure", role="db", path="/data/p7/m.db",
                          reasons=("an input moved on",))
    summary = ProjectSummary(root="/data/p7", known=True, stale=(entry, entry))
    assert summary.staleness_note() == "2 stale"


# ---------------------------------------------------------------------------
# A registry that cannot be read
# ---------------------------------------------------------------------------

def _registered_plate(tmp_path, name="plate1"):
    """A plate with one registered artifact whose file has since gone."""
    plate = _plate(tmp_path, name=name, merged=1)
    store = artifacts.open_registry(plate)
    crops = os.path.join(plate, "data")
    os.makedirs(crops)
    store.register(project=plate, kind=_ports.CROPS, role="crops", path=crops,
                   module="measure", settings={"src": plate})
    os.rmdir(crops)
    return plate, store


def test_a_row_whose_provenance_cannot_be_read_is_dropped_not_raised(
        tmp_path, caplog):
    """The same rows that reported a gone result report nothing once unreadable.

    ``is_stale`` re-opens the registry per row, so a file that stops being a
    database between the listing and the check takes the verdict with it. The
    row is left out of both columns and the reason goes to the log.
    """
    plate, store = _registered_plate(tmp_path)
    records = list(store.by_project(plate))
    assert len(records) == 1
    stale, gone = projects._stale_of(store, records)
    assert (len(stale), len(gone)) == (0, 1)

    _wreck(store.path)
    with caplog.at_level(logging.DEBUG, logger="spacr.projects"):
        assert projects._stale_of(store, records) == ((), ())
    assert any(records[0].artifact_id in record.getMessage()
               for record in caplog.records), caplog.text


def test_a_registry_that_stops_being_readable_leaves_the_project_listed(
        tmp_path):
    """``known`` drops to False; the row keeps everything disk can answer."""
    plate, store = _registered_plate(tmp_path)
    usage = _dm.scan_project(plate, registry=store)
    _wreck(store.path)
    summary = projects.scan(plate, registry=store, usage=usage,
                            with_next_steps=False)
    assert summary.has_registry is True
    assert summary.known is False
    assert summary.n_artifacts == 0
    assert summary.stage == "mask"
    assert summary.last_run_source == projects.SOURCE_FILESYSTEM
    assert summary.staleness_note() == "unknown — nothing recorded"


def test_a_corrupt_registry_does_not_stop_the_project_being_summarised(
        tmp_path):
    """The measurement reconciles against the registry, so it fails with it.

    Opening and walking both go through ``artifacts.db``. Neither may turn a
    listable folder into a raised ``sqlite3.DatabaseError``: the row appears,
    unmeasured, carrying the reason it could not be measured.
    """
    plate, store = _registered_plate(tmp_path)
    _wreck(store.path)
    summary = projects.scan(plate, with_next_steps=False)
    assert summary.exists is True
    assert summary.stage == "mask"
    assert summary.known is False
    assert summary.size_bytes == 0
    assert any("not a database" in problem for problem in summary.errors), \
        summary.errors
    assert "could not read" in projects.format_project(summary)


def test_one_unreadable_registry_does_not_take_the_whole_browse_with_it(
        tmp_path):
    """Every other plate on the disk still gets a row."""
    _plate(tmp_path, name="healthy", merged=1)
    _broken, store = _registered_plate(tmp_path, name="wrecked")
    _wreck(store.path)
    summaries = projects.browse([tmp_path], with_next_steps=False)
    assert sorted(s.name for s in summaries) == ["healthy", "wrecked"]


# ---------------------------------------------------------------------------
# A project deleted underneath the scan
# ---------------------------------------------------------------------------

def test_a_project_deleted_while_it_is_measured_is_reported_gone_not_raised(
        tmp_path, monkeypatch):
    """The folder is there when it is checked and not when it is walked.

    A cleanup job, or the user's own ``rm -rf``, between the two. The
    measurement raises :class:`spacr.data_manager.DataManagerError` for real
    here -- the walk is the shipped one -- and the browser turns it into the
    row it shows for any folder that is gone.
    """
    plate = _plate(tmp_path, name="doomed", merged=1)
    measure = _dm.scan_project

    def _delete_then_measure(root, **kwargs):
        shutil.rmtree(root)
        return measure(root, **kwargs)

    monkeypatch.setattr(_dm, "scan_project", _delete_then_measure)
    summary = projects.scan(plate, with_next_steps=False)
    assert summary.exists is False
    assert summary.note() == "folder is gone"
    assert projects.format_project(summary).splitlines() == [
        summary.root, "  the folder is gone"]
