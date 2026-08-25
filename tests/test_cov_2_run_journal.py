"""The run journal's refusal to fail a run over its own record-keeping.

Every path here is one where the journal cannot do its job: git is not there,
a package list will not enumerate, a settings value is not a path, a file
cannot be hashed, the manifest cannot be written. The rule the whole module is
built on is that none of these may take down the pipeline that was actually
producing results -- and that none of them may be silent either, because a
manifest that quietly omits an output reads as a run that did not make one.

The other half is reading the journal back: a history scan walks folders other
processes are writing to and deleting, and it has to survive all of it.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from spacr import run_journal as RJ


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """A sandboxed runs root, so nothing here touches the user's journal."""
    folder = tmp_path / "runs"
    folder.mkdir()
    monkeypatch.setattr(RJ, "runs_root", lambda: folder)
    return folder


def _write_run(runs_dir, name, manifest, *, log_text=None):
    """Lay out one run folder by hand, the way a finished run leaves one."""
    directory = runs_dir / name
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "settings.json").write_text(json.dumps({"app_key": "mask"}))
    if log_text is not None:
        (directory / "log.txt").write_text(log_text)
    return directory


# ---------------------------------------------------------------------------
# the environment snapshot
# ---------------------------------------------------------------------------

def test_a_checkout_git_cannot_read_reports_no_commit(monkeypatch):
    """A non-zero ``git rev-parse`` means there is no commit to record.

    An installed wheel is not a checkout, and a checkout with no commits is
    not one either. Recording the empty string as a commit hash would put a
    provenance field in the manifest that names nothing.
    """
    def _fail(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=128,
                                           stdout="", stderr="not a git repo")

    monkeypatch.setattr(RJ.subprocess, "run", _fail)
    assert RJ._git_hash() is None


def test_git_being_absent_altogether_reports_no_commit(monkeypatch):
    """No ``git`` on PATH is a machine, not a failure.

    Clusters routinely have no git. Letting the ``FileNotFoundError`` out
    would stop a run before it started, over a field nobody needs.
    """
    def _explode(*_args, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'git'")

    monkeypatch.setattr(RJ.subprocess, "run", _explode)
    assert RJ._git_hash() is None


def test_a_package_list_that_will_not_enumerate_is_empty_not_fatal(caplog):
    """A broken site-packages costs the package list, never the run.

    The environment snapshot is taken while the run is starting. A
    distribution with unreadable metadata is a real thing on shared machines,
    and it must degrade to "no packages recorded" with a line saying why.

    The list is cached for the whole process, so the cache is emptied on both
    sides of the swap: leaving an empty list in it would give every later run
    in this session an environment block with no packages in it.
    """
    real = importlib.metadata.distributions

    def _explode():
        raise OSError("site-packages is not readable")

    try:
        importlib.metadata.distributions = _explode
        RJ._installed_packages.cache_clear()
        with caplog.at_level("WARNING"):
            packages = RJ._installed_packages()
    finally:
        importlib.metadata.distributions = real
        RJ._installed_packages.cache_clear()

    assert packages == {}
    assert any("Could not enumerate installed packages" in r.message
               for r in caplog.records)
    assert RJ._installed_packages(), "the real list comes back afterwards"


# ---------------------------------------------------------------------------
# atomic writes
# ---------------------------------------------------------------------------

def test_the_temp_file_refusing_to_go_does_not_lose_the_write(tmp_path,
                                                              monkeypatch):
    """Cleanup of the scratch file must never undo a completed write.

    ``os.replace`` has already put the content where it belongs by the time
    the temporary name is unlinked. An error raised there would propagate out
    of the manifest writer and report a successful run as failed.
    """
    real_unlink = Path.unlink

    def _refuse(self, *args, **kwargs):
        if self.name.startswith("."):
            raise OSError(13, "Permission denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _refuse)

    target = tmp_path / "manifest.json"
    RJ._atomic_write_text(target, '{"ok": true}')

    assert json.loads(target.read_text()) == {"ok": True}


# ---------------------------------------------------------------------------
# walking settings
# ---------------------------------------------------------------------------

def test_a_settings_dict_that_contains_itself_is_walked_once():
    """A cycle in the settings must not become an infinite recursion.

    Panels build nested settings by reference, and a dict that ends up
    holding itself is a bug in the caller, not a reason for the journal to
    hang while trying to record it.
    """
    inner = {"src": "/data"}
    inner["self"] = inner
    values = dict(RJ._walk_setting_values(inner))

    assert values["src"] == "/data"
    assert not any(key.count("self") > 1 for key in values), \
        "the cycle was followed more than once"


def test_a_settings_list_that_contains_itself_is_walked_once():
    """The same cycle guard, for the sequence branch.

    A list is walked by a different arm of the same function, so a guard on
    only one of them leaves the other able to recurse forever.
    """
    values = ["a"]
    values.append(values)
    walked = dict(RJ._walk_setting_values({"channels": values}))

    assert walked["channels[0]"] == "a"
    assert len(walked) < 5, f"the cycle was expanded: {walked}"


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------

def test_numpy_not_being_loaded_records_no_numpy_state(monkeypatch):
    """A run that never imported numpy has no numpy state to record.

    ``None`` is the honest answer, and it must be written: leaving the key
    out entirely makes a manifest from a numpy-free run look like one where
    the capture failed.
    """
    monkeypatch.setitem(sys.modules, "numpy", None)

    seeds = RJ.extract_seeds({})

    assert seeds["numpy_random_state_sha256"] is None
    assert "numpy_random_state_error" not in seeds


def test_a_random_state_that_will_not_be_read_is_reported_as_an_error(
        monkeypatch):
    """Both random-state captures record their own failure rather than raise.

    They run inside ``open_run``, before the pipeline does anything. An
    exception here would mean a library in a strange state stopped a run that
    would otherwise have worked.
    """
    numpy_stub = types.SimpleNamespace(
        random=types.SimpleNamespace(
            get_state=lambda: (_ for _ in ()).throw(RuntimeError("no state"))))
    torch_stub = types.SimpleNamespace(
        initial_seed=lambda: (_ for _ in ()).throw(RuntimeError("no seed")))
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    seeds = RJ.extract_seeds({})

    assert "no state" in seeds["numpy_random_state_error"]
    assert "no seed" in seeds["torch_seed_error"]
    assert "numpy_random_state_sha256" not in seeds


# ---------------------------------------------------------------------------
# finding paths in the settings
# ---------------------------------------------------------------------------

def test_a_multi_line_setting_is_not_treated_as_a_path(tmp_path):
    """A newline in a path-named setting means it is not a path.

    Path-looking keys hold notes and multi-line text often enough that
    hashing whatever they contain would walk the wrong tree, so a value with
    a line break in it is passed over before anything is resolved.
    """
    plate = tmp_path / "plate1"
    plate.mkdir()

    candidates = RJ._setting_path_candidates({
        "src": str(plate),
        "src_notes": "first line\nsecond line",
    })

    found = {key for key, _path, _out in candidates}
    assert "src" in found
    assert "src_notes" not in found


def test_a_setting_that_cannot_be_resolved_to_a_path_is_skipped():
    """A home directory that does not exist is not a path this run can use.

    ``~nobody/x`` raises out of ``expanduser``, and the candidate list is
    built before the pipeline runs -- so a typo in one setting must not stop
    every other path from being recorded.
    """
    candidates = RJ._setting_path_candidates({
        "src": "~no_such_user_zzz/plate1",
        "dst": "/tmp/spacr-output",
    })

    found = {key for key, _path, _out in candidates}
    assert "src" not in found
    assert "dst" in found


# ---------------------------------------------------------------------------
# walking files
# ---------------------------------------------------------------------------

def test_an_unresolvable_exclusion_still_lets_the_walk_run(tmp_path):
    """A bad entry in the exclusion list must not stop the inventory.

    The exclusions are the journal's own folders. If one of them cannot be
    resolved the walk falls back to comparing them unresolved, which is
    strictly less accurate and infinitely better than no inventory at all.
    """
    (tmp_path / "a.txt").write_text("one")

    found = list(RJ._iter_files(tmp_path, [Path("bad\x00root")]))

    assert [p.name for p in found] == ["a.txt"]


def test_a_path_that_cannot_be_resolved_yields_nothing_and_does_not_raise():
    """An unresolvable root is an empty inventory, not an exception.

    This runs while a finished run is writing its manifest. Raising here
    would report a completed pipeline as failed.
    """
    assert list(RJ._iter_files(Path("nope\x00nope"), [])) == []


def test_a_path_that_is_neither_a_file_nor_a_folder_yields_nothing(tmp_path):
    """A setting naming something that is not there contributes no files."""
    assert list(RJ._iter_files(tmp_path / "not-here", [])) == []


# ---------------------------------------------------------------------------
# file records
# ---------------------------------------------------------------------------

def test_a_file_that_cannot_be_read_gets_no_record(tmp_path):
    """A file that stats but will not open produces no provenance record.

    A record with no digest in it would claim the file was fingerprinted.
    The caller turns the ``None`` into a warning that names the file.
    """
    blocked = tmp_path / "locked.bin"
    blocked.write_bytes(b"secret")
    os.chmod(blocked, 0o000)
    try:
        assert RJ._file_record(blocked) is None
    finally:
        os.chmod(blocked, 0o644)


def test_a_missing_file_has_no_inventory_signature(tmp_path):
    """The before/after signature of a file that is not there is ``None``.

    The baseline is taken over paths the settings named, which routinely do
    not exist yet -- an output folder is the ordinary case.
    """
    assert RJ._inventory_signature(tmp_path / "not-yet") is None


# ---------------------------------------------------------------------------
# the Run object's own record-keeping
# ---------------------------------------------------------------------------

def test_a_model_path_that_cannot_be_inspected_is_a_warning_not_a_crash(
        runs_dir):
    """Recording a model must never be able to fail a training run.

    The warning goes into the manifest where the model record would have
    been, so the run says "the checkpoint could not be recorded" rather than
    reading as a run that used no model.
    """
    with RJ.open_run("classify", {}) as run:
        run.record_model("cyto", "bad\x00path.pth")

    assert "cyto" not in run.model_files
    assert any("cyto" in w and "could not be recorded" in w
               for w in run.provenance_warnings)
    manifest = json.loads((run.dir / "manifest.json").read_text())
    assert any("could not be recorded" in w
               for w in manifest["provenance_warnings"])


def test_an_output_that_cannot_be_attached_is_a_warning_not_a_crash(runs_dir):
    """A missing file to attach costs the copy, never the run.

    ``attach_output`` is called after the work is done, so raising here would
    throw away results that are already on disk.
    """
    with RJ.open_run("measure", {}) as run:
        assert run.attach_output("/nonexistent/result.csv") is None

    assert any("could not be attached" in w for w in run.provenance_warnings)


def test_recording_an_output_hashes_it_when_hashing_is_on(runs_dir, tmp_path):
    """``record_output`` is the explicit half of the output inventory.

    It has to obey the same ``hash_inputs`` switch as ``record_input``: a run
    that did not ask for hashing must not pay for it, and one that did must
    get the digest.
    """
    result = tmp_path / "table.csv"
    result.write_text("a,b\n1,2\n")

    with RJ.open_run("measure", {"hash_inputs": True}) as run:
        run.record_output(result, setting_key="dst")

    record = run.output_hashes[str(result.resolve())]
    assert len(record["sha256"]) == 64
    assert record["setting_keys"] == ["dst"]


def test_recording_an_output_is_free_when_hashing_is_off(runs_dir, tmp_path):
    """The default is off, and off means nothing is read."""
    result = tmp_path / "table.csv"
    result.write_text("a,b\n1,2\n")

    with RJ.open_run("measure", {}) as run:
        run.record_output(result)

    assert run.output_hashes == {}


def test_a_file_in_a_recorded_folder_that_will_not_hash_is_named(runs_dir,
                                                                 tmp_path):
    """One unreadable file in an input folder is reported, not skipped.

    The manifest claims to fingerprint everything the run read. A file that
    could not be hashed has to be said out loud, or the manifest is a
    complete-looking record of an incomplete read.
    """
    folder = tmp_path / "plate"
    folder.mkdir()
    (folder / "good.tif").write_bytes(b"pixels")
    blocked = folder / "locked.tif"
    blocked.write_bytes(b"pixels")
    os.chmod(blocked, 0o000)
    try:
        with RJ.open_run("mask", {"hash_inputs": True}) as run:
            run.record_input(folder, setting_key="src")
    finally:
        os.chmod(blocked, 0o644)

    assert str(folder / "good.tif") in run.input_hashes
    assert str(blocked) not in run.input_hashes
    assert any("could not hash provenance file" in w and "locked.tif" in w
               for w in run.provenance_warnings)


def test_an_empty_input_folder_is_reported_as_holding_nothing(runs_dir,
                                                              tmp_path):
    """A folder that exists and holds no files is worth saying.

    "src pointed at an empty folder" and "src was hashed" produce the same
    empty input list, and the first is almost always a mistake the user wants
    to hear about before the run finishes.
    """
    empty = tmp_path / "empty"
    empty.mkdir()

    with RJ.open_run("mask", {"hash_inputs": True}) as run:
        run.record_input(empty, setting_key="src")

    assert run.input_hashes == {}
    assert any("no regular provenance files found" in w
               for w in run.provenance_warnings)


def test_a_path_that_cannot_even_be_expanded_is_reported(runs_dir):
    """``record_input`` swallows whatever the path layer throws.

    A home directory that does not exist raises from ``expanduser``, before
    any walking happens. The run carries on with a warning naming the path.
    """
    with RJ.open_run("mask", {"hash_inputs": True}) as run:
        run.record_input("~no_such_user_zzz/plate", setting_key="src")

    assert run.input_hashes == {}
    assert any("could not record provenance path" in w
               for w in run.provenance_warnings)


# ---------------------------------------------------------------------------
# the final output inventory
# ---------------------------------------------------------------------------

def test_an_output_folder_that_was_never_created_inventories_nothing(runs_dir,
                                                                     tmp_path):
    """A run that wrote nothing has nothing to inventory, and says nothing.

    The destination folder named in the settings not existing at the end is
    the normal outcome of a run that failed early or produced no files.
    """
    with RJ.open_run("measure", {
            "dst": str(tmp_path / "never-made" / "out.csv"),
            "hash_inputs": True}) as run:
        pass

    assert run.output_hashes == {}
    assert run.provenance_warnings == []


def test_an_output_file_that_cannot_be_hashed_is_named_in_the_warnings(
        runs_dir, tmp_path):
    """A new output that will not open is reported, not silently dropped.

    Silently dropping it makes the manifest list one output where the run
    made two, and a reader comparing manifests would see a run that produced
    less than it did.
    """
    destination = tmp_path / "out"
    destination.mkdir()
    blocked = destination / "locked.csv"

    try:
        with RJ.open_run("measure", {"dst": str(destination / "table.csv"),
                                     "hash_inputs": True}):
            (destination / "table.csv").write_text("a,b\n1,2\n")
            blocked.write_text("secret")
            os.chmod(blocked, 0o000)
    finally:
        os.chmod(blocked, 0o644)

    manifest = json.loads(
        next(runs_dir.glob("*/manifest.json")).read_text())
    assert str(destination / "table.csv") in manifest["output_hashes"]
    assert str(blocked) not in manifest["output_hashes"]
    assert any("could not hash changed output file" in w
               for w in manifest["provenance_warnings"])


# ---------------------------------------------------------------------------
# closing a run when the record-keeping itself fails
# ---------------------------------------------------------------------------

def test_a_log_that_cannot_be_copied_does_not_fail_the_run(runs_dir,
                                                           monkeypatch,
                                                           caplog):
    """The run's own log copy is a convenience, and it says when it failed.

    A folder with no ``log.txt`` reads as "nothing was logged". The warning
    is what distinguishes that from "the copy failed", and the run finishes
    either way.
    """
    from spacr import logging_util

    def _explode():
        raise OSError("no log directory")

    monkeypatch.setattr(logging_util, "log_path", _explode)

    with caplog.at_level("WARNING"):
        with RJ.open_run("mask", {}) as run:
            pass

    assert not (run.dir / "log.txt").exists()
    assert (run.dir / "manifest.json").exists(), "the manifest still landed"
    assert any("could not copy the last" in r.message for r in caplog.records)


def test_a_final_provenance_failure_is_recorded_in_the_manifest(runs_dir,
                                                                monkeypatch):
    """The output inventory failing must not lose the manifest with it.

    The manifest is the run. Its provenance section being incomplete is a
    warning; its absence would be an unreproducible run.
    """
    def _explode(self):
        raise RuntimeError("the disk went away")

    monkeypatch.setattr(RJ.Run, "_capture_final_provenance", _explode)

    with RJ.open_run("measure", {}) as run:
        pass

    manifest = json.loads((run.dir / "manifest.json").read_text())
    assert any("final output provenance failed" in w
               for w in manifest["provenance_warnings"])
    assert "the disk went away" in " ".join(manifest["provenance_warnings"])


def test_a_manifest_that_cannot_be_written_does_not_mask_the_real_error(
        runs_dir, monkeypatch, caplog):
    """The pipeline's own exception is the one the caller has to see.

    A failure while finalising the record during unwinding would replace the
    exception that explains what went wrong with one about bookkeeping.
    """
    # The opening manifest is written before the body runs and has to keep
    # working; it is the closing one that must not mask the pipeline error.
    real_write = RJ.Run._write_manifest
    calls = []

    def _explode_on_close(self):
        calls.append(self)
        if len(calls) > 1:
            raise OSError("read-only filesystem")
        return real_write(self)

    monkeypatch.setattr(RJ.Run, "_write_manifest", _explode_on_close)

    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError, match="segmentation blew up"):
            with RJ.open_run("mask", {}):
                raise ValueError("segmentation blew up")

    assert len(calls) == 2, "the closing write was reached"

    assert any("Could not finalize run manifest" in r.message
               for r in caplog.records)


def test_a_workspace_that_cannot_be_saved_does_not_fail_the_run(runs_dir,
                                                                monkeypatch,
                                                                caplog):
    """The workspace bundle is a convenience beside the results.

    A run that produced results must not be reported as failed because a
    panel could not describe itself.
    """
    from spacr import workspace

    def _explode(*_args, **_kwargs):
        raise RuntimeError("no GUI to describe")

    monkeypatch.setattr(workspace, "save_for_run", _explode)

    with caplog.at_level("ERROR"):
        with RJ.open_run("mask", {}) as run:
            pass

    manifest = json.loads((run.dir / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert any("Could not save the workspace" in r.message
               for r in caplog.records)


# ---------------------------------------------------------------------------
# reading the journal back
# ---------------------------------------------------------------------------

def test_a_history_scan_of_a_journal_that_is_not_there_is_empty(tmp_path,
                                                                monkeypatch,
                                                                caplog):
    """A first run on a fresh machine has no journal to list.

    The history panel opens before anything has ever been run, so a missing
    folder is the ordinary first state and must produce an empty list with a
    line in the log, not an exception in a paint handler.
    """
    monkeypatch.setattr(RJ, "runs_root", lambda: tmp_path / "never-made")

    with caplog.at_level("WARNING"):
        assert RJ.search_runs() == []

    assert any("Could not enumerate run history" in r.message
               for r in caplog.records)


def test_a_manifest_whose_warnings_are_one_string_still_shows_them(runs_dir):
    """Older manifests wrote a single warning as a plain string.

    Reading it as a sequence would spell the warning out one character per
    row in the history panel; ignoring it would drop the only thing the run
    had to say.
    """
    _write_run(runs_dir, "run-a", {
        "app_key": "mask", "status": "success",
        "start_utc": "2026-01-01T00:00:00+00:00",
        "warnings": "one thing went wrong",
    })

    record = RJ.search_runs()[0]

    assert "one thing went wrong" in record["warnings"]


def test_a_legacy_run_has_its_warnings_read_out_of_the_log(runs_dir):
    """A manifest with no warnings block still has a log tail worth mining.

    Runs written before warnings were structured are still in users'
    journals, and the history panel is where they are read. Lines that
    announce themselves as warnings are surfaced; ordinary lines are not.
    """
    _write_run(runs_dir, "run-legacy", {
        "app_key": "mask", "status": "success",
        "start_utc": "2026-01-01T00:00:00+00:00",
    }, log_text="starting up\nWARNING: two fields had no mask\nall done\n")

    record = RJ.search_runs()[0]

    assert "WARNING: two fields had no mask" in record["warnings"]
    assert "all done" not in record["warnings"]


def test_a_log_that_cannot_be_read_is_reported_rather_than_skipped(runs_dir):
    """An unreadable log is itself worth showing in the history row.

    Otherwise a legacy run whose log is locked looks exactly like one that
    logged no warnings at all.
    """
    directory = _write_run(runs_dir, "run-locked", {
        "app_key": "mask", "status": "success",
        "start_utc": "2026-01-01T00:00:00+00:00",
    }, log_text="WARNING: something\n")
    os.chmod(directory / "log.txt", 0o000)
    try:
        record = RJ.search_runs()[0]
    finally:
        os.chmod(directory / "log.txt", 0o644)

    assert any("log.txt unreadable" in w for w in record["warnings"])


def test_a_start_time_with_no_timezone_still_sorts(runs_dir):
    """Manifests written before the timestamps carried a zone still sort.

    Comparing a naive datetime against an aware one raises, and it would
    raise inside the sort -- taking the whole history panel down because of
    one old run folder.
    """
    _write_run(runs_dir, "run-naive", {
        "app_key": "mask", "status": "success",
        "start_utc": "2026-01-01T00:00:00",
    })
    _write_run(runs_dir, "run-aware", {
        "app_key": "mask", "status": "success",
        "start_utc": "2026-02-01T00:00:00+00:00",
    })

    found = RJ.search_runs()

    assert [r["run_id"] for r in found] == ["run-aware", "run-naive"]


def test_a_run_folder_deleted_mid_scan_still_appears_in_the_history(
        runs_dir, monkeypatch):
    """Runs are deleted by other processes while the history is being read.

    The sort reaches for each folder's modification time, and the folder may
    be gone by then. Losing the whole listing over one vanished run -- or
    raising into the panel -- is the failure this guard prevents; the record
    already read is still shown.
    """
    _write_run(runs_dir, "run-gone", {
        "app_key": "mask", "status": "success",
        "start_utc": "2026-01-01T00:00:00+00:00",
    })
    real_read = RJ._read_run_record

    def _read_then_vanish(reference):
        record = real_read(reference)
        shutil.rmtree(Path(reference), ignore_errors=True)
        return record

    monkeypatch.setattr(RJ, "_read_run_record", _read_then_vanish)

    found = RJ.search_runs()

    assert [r["run_id"] for r in found] == ["run-gone"]
    assert not (runs_dir / "run-gone").exists()
