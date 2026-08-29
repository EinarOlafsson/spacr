"""Quarantine's refusals and its rollbacks.

Moving a merged array out of measurement is the one operation in the field
browser that can lose data, so every failure it can meet has to end in one of
two states: the array where it started, or the array in quarantine with a
ledger beside it.  The tests here drive the paths that decide that -- a folder
that is not a plate's, a damaged ledger, a filesystem without hard links, a
removal the kernel refuses, and a field that measurement regenerated while the
ledger was being written.
"""
from __future__ import annotations

import errno
import getpass
import json
import os
import stat

import numpy as np
import pytest

from spacr import qc_quarantine
from spacr.qc_quarantine import QuarantineError

FIELD = "plate1_A01_1"
_OPEN_DESCRIPTORS = "/proc/self/fd"


def _plate(tmp_path):
    """Create ``plate1/merged`` holding one small merged array."""
    merged = tmp_path / "plate1" / "merged"
    merged.mkdir(parents=True)
    np.save(merged / f"{FIELD}.npy", np.arange(24).reshape(2, 3, 4))
    return merged


def _cross_device_link(monkeypatch):
    """Make ``os.link`` report a filesystem that cannot hard-link."""
    def refuse(source, destination, **kwargs):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", refuse)


def _unlink_refusing(monkeypatch, blocked):
    """Make ``os.unlink`` refuse ``blocked`` and behave normally elsewhere.

    ``blocked=None`` refuses every path.  Each refusal names the path it was
    asked to remove, so a caller that meets two of them in a row can be asked
    which one it let out.
    """
    real_unlink = os.unlink

    def refuse(path, *args, **kwargs):
        if blocked is None or os.fspath(path) == os.fspath(blocked):
            raise PermissionError(errno.EACCES, "Operation not permitted",
                                  os.fspath(path))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", refuse)


# -- the two folders this module owns ---------------------------------------

def test_a_folder_that_is_not_a_plates_merged_one_is_refused(tmp_path):
    """Only ``<plate>/merged`` may be used to derive a quarantine folder."""
    stray = tmp_path / "plate1" / "norm_channel_stack"
    stray.mkdir(parents=True)

    with pytest.raises(ValueError) as caught:
        qc_quarantine.quarantine_dir_for(stray)

    assert "expected a plate's 'merged' folder" in str(caught.value)
    assert str(stray) in str(caught.value)
    assert list(tmp_path.joinpath("plate1").iterdir()) == [stray], (
        "a refused folder gets no sibling quarantine directory")


def test_a_record_path_outside_the_quarantine_folder_is_refused(tmp_path):
    """A sidecar is only ever addressed inside ``merged_quarantined``."""
    merged = _plate(tmp_path)

    with pytest.raises(ValueError) as caught:
        qc_quarantine.quarantine_record_path(merged, FIELD)

    assert "expected a 'merged_quarantined' folder" in str(caught.value)

    with pytest.raises(ValueError):
        qc_quarantine.restore_field(merged, FIELD)

    assert (merged / f"{FIELD}.npy").is_file(), (
        "a refused restore moves nothing"
    )


# -- the ledger --------------------------------------------------------------

def test_a_damaged_sidecar_is_replaced_and_the_damage_is_recorded(tmp_path):
    """Unparseable JSON left by an older run must not block a quarantine."""
    merged = _plate(tmp_path)
    quarantine = qc_quarantine.quarantine_dir_for(merged)
    quarantine.mkdir()
    sidecar = qc_quarantine.quarantine_record_path(quarantine, FIELD)
    sidecar.write_text("{ this was truncated by a full disk", encoding="utf-8")

    moved = qc_quarantine.quarantine_field(merged, FIELD, who="reviewer")

    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["prior_record_error"].startswith("JSONDecodeError: ")
    assert [event["action"] for event in record["events"]] == ["quarantined"]
    assert record["quarantined_by"] == "reviewer"
    assert moved.is_file() and not (merged / f"{FIELD}.npy").exists()


def test_a_sidecar_that_is_not_an_object_is_replaced_with_a_note(tmp_path):
    """A JSON array where a record belongs is damage too, not history."""
    merged = _plate(tmp_path)
    quarantine = qc_quarantine.quarantine_dir_for(merged)
    quarantine.mkdir()
    sidecar = qc_quarantine.quarantine_record_path(quarantine, FIELD)
    sidecar.write_text('["quarantined"]', encoding="utf-8")

    qc_quarantine.quarantine_field(merged, FIELD, who="reviewer")

    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["prior_record_error"] == (
        "the previous sidecar was not a JSON object")
    assert record["events"][-1]["by"] == "reviewer"


def test_an_unserialisable_record_leaves_no_temporary_file_behind(tmp_path):
    """A half-written ledger must never survive the write that failed."""
    folder = tmp_path / "plate1" / qc_quarantine.QUARANTINE_DIRNAME
    folder.mkdir(parents=True)
    target = folder / f"{FIELD}.npy.quarantine.json"

    with pytest.raises(TypeError):
        qc_quarantine._write_record(target, {"by": object()})

    assert not target.exists()
    assert list(folder.iterdir()) == [], (
        "the temporary the encoder wrote into is removed")


def test_a_ledger_folder_that_cannot_be_written_raises_without_a_file(
        tmp_path):
    """A read-only quarantine folder fails before a temporary is even made."""
    folder = tmp_path / "plate1" / qc_quarantine.QUARANTINE_DIRNAME
    folder.mkdir(parents=True)
    target = folder / f"{FIELD}.npy.quarantine.json"
    folder.chmod(0o500)
    try:
        with pytest.raises(OSError) as caught:
            qc_quarantine._write_record(target, {"version": 1})
        assert caught.value.errno == errno.EACCES
    finally:
        folder.chmod(0o700)

    assert list(folder.iterdir()) == []


def test_a_temporary_that_cannot_be_removed_does_not_mask_the_real_error(
        tmp_path, monkeypatch):
    """The encoder's failure is what the caller sees, not the cleanup's."""
    folder = tmp_path / "plate1" / qc_quarantine.QUARANTINE_DIRNAME
    folder.mkdir(parents=True)
    target = folder / f"{FIELD}.npy.quarantine.json"
    real_unlink = os.unlink

    def refuse_temporaries(path, *args, **kwargs):
        if os.fspath(path).endswith(".tmp"):
            raise PermissionError(errno.EACCES, "Operation not permitted")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", refuse_temporaries)

    with pytest.raises(TypeError):
        qc_quarantine._write_record(target, {"by": object()})

    leftovers = [path.name for path in folder.iterdir()]
    assert leftovers and all(name.endswith(".tmp") for name in leftovers), (
        "the cleanup really was refused, and the encoder's error still won")
    assert not target.exists()


def test_an_unknown_account_is_recorded_as_unknown(tmp_path, monkeypatch):
    """A container with no passwd entry still gets an auditable actor."""
    def no_such_account():
        raise OSError("no login name for this uid")

    monkeypatch.setattr(getpass, "getuser", no_such_account)
    merged = _plate(tmp_path)

    moved = qc_quarantine.quarantine_field(merged, FIELD, who="   ")

    sidecar = qc_quarantine.quarantine_record_path(moved.parent, FIELD)
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["quarantined_by"] == "unknown"


# -- moving the array --------------------------------------------------------

def test_a_symlinked_field_is_never_moved(tmp_path):
    """A symlink in ``merged`` points somewhere this module does not own."""
    merged = _plate(tmp_path)
    outside = tmp_path / "elsewhere.npy"
    np.save(outside, np.zeros((2, 2)))
    link = merged / "plate1_A02_1.npy"
    link.symlink_to(outside)

    with pytest.raises(QuarantineError) as caught:
        qc_quarantine.quarantine_field(merged, "plate1_A02_1")

    assert "refusing to move symlink" in str(caught.value)
    assert link.is_symlink() and outside.is_file()
    quarantine = qc_quarantine.quarantine_dir_for(merged)
    assert not (quarantine / "plate1_A02_1.npy").exists()


def test_quarantining_a_field_that_is_not_there_names_the_missing_array(
        tmp_path):
    """A scorecard row for a deleted field is a missing file, not a crash."""
    merged = _plate(tmp_path)

    with pytest.raises(FileNotFoundError) as caught:
        qc_quarantine.quarantine_field(merged, "plate1_Z99_9")

    assert caught.value.args[0] == merged / "plate1_Z99_9.npy"
    quarantine = qc_quarantine.quarantine_dir_for(merged)
    assert list(quarantine.iterdir()) == [], (
        "nothing is written for a field that was never there")


def test_a_link_failure_that_is_not_about_hard_links_is_raised(
        tmp_path, monkeypatch):
    """A full disk is not a filesystem without links, and is not retried."""
    merged = _plate(tmp_path)
    source = merged / f"{FIELD}.npy"
    original = source.read_bytes()

    def no_space(link_source, link_destination, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "link", no_space)

    with pytest.raises(OSError) as caught:
        qc_quarantine.quarantine_field(merged, FIELD)

    assert caught.value.errno == errno.ENOSPC
    assert source.read_bytes() == original
    quarantine = qc_quarantine.quarantine_dir_for(merged)
    assert not (quarantine / f"{FIELD}.npy").exists()


def test_a_filesystem_without_hard_links_still_moves_the_array(
        tmp_path, monkeypatch):
    """The copy fallback keeps the bytes, the metadata and the no-overwrite rule."""
    merged = _plate(tmp_path)
    source = merged / f"{FIELD}.npy"
    # 0o666 and an old timestamp: the mode a umask strips from the bare
    # O_CREAT, and a time only a stat copy can carry across.
    source.chmod(0o666)
    os.utime(source, (1234567890, 1234567890))
    original = source.read_bytes()
    _cross_device_link(monkeypatch)

    moved = qc_quarantine.quarantine_field(
        merged, FIELD, flags=["cell:empty_field"], who="reviewer")

    assert moved.read_bytes() == original
    assert not source.exists()
    assert stat.S_IMODE(moved.stat().st_mode) == 0o666
    assert moved.stat().st_mtime == 1234567890, (
        "the copy carries the field's own time, not the copy's")
    assert np.load(moved).shape == (2, 3, 4)
    sidecar = qc_quarantine.quarantine_record_path(moved.parent, FIELD)
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["qc_flags"] == ["cell:empty_field"]
    assert record["quarantined_path"] == str(moved)


@pytest.mark.skipif(not os.path.isdir(_OPEN_DESCRIPTORS),
                    reason="open descriptors are only countable on /proc")
def test_an_unreadable_source_leaves_no_half_copy_behind(
        tmp_path, monkeypatch):
    """A copy that cannot be started drops both the file and the handle.

    The destination is opened before the source is, so a source that cannot
    be read leaves a descriptor with nothing on the other end of it.  A
    quarantine sweep over a plate of unreadable fields would run the process
    out of descriptors long before it ran out of fields.
    """
    merged = _plate(tmp_path)
    source = merged / f"{FIELD}.npy"
    original = source.read_bytes()
    _cross_device_link(monkeypatch)
    source.chmod(0o000)
    open_before = len(os.listdir(_OPEN_DESCRIPTORS))
    try:
        with pytest.raises(PermissionError):
            qc_quarantine.quarantine_field(merged, FIELD)
    finally:
        source.chmod(0o600)

    assert len(os.listdir(_OPEN_DESCRIPTORS)) == open_before, (
        "the descriptor opened for the destination is closed on the way out")
    quarantine = qc_quarantine.quarantine_dir_for(merged)
    assert list(quarantine.iterdir()) == [], (
        "an empty file at the quarantine name would read as a moved field")
    assert source.read_bytes() == original


def test_a_copied_source_that_cannot_be_removed_leaves_no_duplicate(
        tmp_path, monkeypatch):
    """Two copies of one field would be measured twice; the copy is undone."""
    merged = _plate(tmp_path)
    source = merged / f"{FIELD}.npy"
    original = source.read_bytes()
    _cross_device_link(monkeypatch)
    _unlink_refusing(monkeypatch, source)

    with pytest.raises(PermissionError):
        qc_quarantine.quarantine_field(merged, FIELD)

    assert source.read_bytes() == original
    quarantine = qc_quarantine.quarantine_dir_for(merged)
    assert list(quarantine.iterdir()) == []


def test_a_linked_source_that_cannot_be_removed_rolls_the_link_back(
        tmp_path, monkeypatch):
    """A hard link the source outlives is a duplicate, and is unlinked again."""
    merged = _plate(tmp_path)
    source = merged / f"{FIELD}.npy"
    original = source.read_bytes()
    _unlink_refusing(monkeypatch, source)

    with pytest.raises(PermissionError):
        qc_quarantine.quarantine_field(merged, FIELD)

    assert source.read_bytes() == original
    quarantine = qc_quarantine.quarantine_dir_for(merged)
    assert list(quarantine.iterdir()) == []


def test_a_move_that_can_neither_finish_nor_roll_back_names_both_paths(
        tmp_path, monkeypatch):
    """When both names survive, the error says so instead of claiming a move."""
    merged = _plate(tmp_path)
    source = merged / f"{FIELD}.npy"
    original = source.read_bytes()
    quarantine = qc_quarantine.quarantine_dir_for(merged)
    _unlink_refusing(monkeypatch, None)

    with pytest.raises(QuarantineError) as caught:
        qc_quarantine.quarantine_field(merged, FIELD)

    message = str(caught.value)
    assert "could not remove source" in message
    assert str(source) in message and str(quarantine / f"{FIELD}.npy") in message
    assert isinstance(caught.value.__cause__, OSError)
    assert source.read_bytes() == original
    assert (quarantine / f"{FIELD}.npy").read_bytes() == original


# -- the two rollbacks that cannot complete ----------------------------------

def test_a_field_regenerated_during_the_ledger_write_is_not_overwritten(
        tmp_path, monkeypatch):
    """Measurement re-creating the field blocks the rollback, and is kept."""
    merged = _plate(tmp_path)
    source = merged / f"{FIELD}.npy"
    original = source.read_bytes()

    def regenerate_then_fail(path, record):
        source.write_bytes(b"regenerated by a concurrent measurement")
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(qc_quarantine, "_write_record", regenerate_then_fail)

    with pytest.raises(QuarantineError) as caught:
        qc_quarantine.quarantine_field(merged, FIELD)

    message = str(caught.value)
    assert "could not write" in message and "could not restore" in message
    assert isinstance(caught.value.__cause__, OSError)
    assert source.read_bytes() == b"regenerated by a concurrent measurement"
    quarantine = qc_quarantine.quarantine_dir_for(merged)
    assert (quarantine / f"{FIELD}.npy").read_bytes() == original, (
        "the moved array is still there to be recovered by hand")


def test_a_restore_that_cannot_be_undone_says_the_field_is_still_out(
        tmp_path, monkeypatch):
    """A blocked return leaves the array in ``merged`` and says the ledger lied."""
    merged = _plate(tmp_path)
    original = (merged / f"{FIELD}.npy").read_bytes()
    moved = qc_quarantine.quarantine_field(merged, FIELD, who="reviewer")
    quarantine = moved.parent

    def regenerate_then_fail(path, record):
        moved.write_bytes(b"a second quarantined copy")
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(qc_quarantine, "_write_record", regenerate_then_fail)

    with pytest.raises(QuarantineError) as caught:
        qc_quarantine.restore_field(quarantine, FIELD, who="reviewer")

    message = str(caught.value)
    assert "could not update" in message and "could not return" in message
    assert isinstance(caught.value.__cause__, OSError)
    assert (merged / f"{FIELD}.npy").read_bytes() == original
    assert moved.read_bytes() == b"a second quarantined copy"


# -- listing -----------------------------------------------------------------

def test_an_unreadable_quarantine_folder_lists_nothing(tmp_path):
    """A folder the user cannot read is an empty listing, not a traceback."""
    merged = _plate(tmp_path)
    quarantine = qc_quarantine.quarantine_dir_for(merged)
    quarantine.mkdir()
    (quarantine / f"{FIELD}.npy").write_bytes(b"\x93NUMPY")
    quarantine.chmod(0o000)
    try:
        assert qc_quarantine.list_quarantined(merged) == []
    finally:
        quarantine.chmod(0o700)

    assert qc_quarantine.list_quarantined(merged) == [FIELD], (
        "the same folder lists its field once it can be read")


def test_a_copy_that_can_be_neither_finished_nor_cleaned_up_keeps_both(
        tmp_path, monkeypatch):
    """When no removal is permitted the caller sees the real error, not the tidy-up.

    Both names then hold the array.  That is the honest outcome of a
    filesystem that refuses every unlink, and hiding it behind the cleanup's
    own ``OSError`` would report a failure that has nothing to do with why the
    move stopped.
    """
    merged = _plate(tmp_path)
    source = merged / f"{FIELD}.npy"
    original = source.read_bytes()
    _cross_device_link(monkeypatch)
    _unlink_refusing(monkeypatch, None)

    with pytest.raises(PermissionError) as caught:
        qc_quarantine.quarantine_field(merged, FIELD)

    quarantine = qc_quarantine.quarantine_dir_for(merged)
    assert caught.value.errno == errno.EACCES
    assert caught.value.filename == str(source), (
        "the refusal that stopped the move is the one raised, not the one "
        "the tidy-up met on the destination")
    assert str(quarantine / f"{FIELD}.npy") not in str(caught.value)
    assert source.read_bytes() == original
    assert (quarantine / f"{FIELD}.npy").read_bytes() == original
