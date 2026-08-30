"""What a workspace records about one file, and what it leaves out.

Every key in this record is optional except ``role``, ``path`` and ``exists``,
and each omission means something specific. A missing ``sha256`` says the bytes
were not hashed; a ``sha256`` of "" would say they were hashed and came out
empty. A workspace is what a run is reconstructed from, so the difference
between "not recorded" and "recorded as nothing" is the difference between a
reconstruction that asks and one that proceeds on a wrong answer.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_a_file_that_is_not_there_records_only_that():
    """The except: no size, no mtime, no kind -- just ``exists: False``.

    A workspace is written from settings that may name a path the user has
    since moved. Recording a size of 0 for it would make the reconstruction
    look complete.
    """
    from spacr.workspace import _file_record

    record = _file_record("src", Path("/nonexistent/plate1/measurements.db"))

    assert record["exists"] is False
    assert "size" not in record and "kind" not in record


def test_a_directory_is_recorded_as_one_and_not_measured(tmp_path):
    """The is_dir early return: a folder has no size worth recording here."""
    from spacr.workspace import _file_record

    record = _file_record("src", tmp_path)

    assert record["exists"] is True
    assert record["kind"] == "directory"
    assert "size" not in record


def test_a_file_recorded_without_hashing_carries_no_digest(tmp_path):
    """Arc 309 -> 313: ``want_hash`` False, so no sha256 key at all.

    Hashing is skipped for large inputs, and the ABSENCE of the key is the
    signal. An empty string would read as a hash that was taken.
    """
    from spacr.workspace import _file_record

    target = tmp_path / "measurements.db"
    target.write_bytes(b"0123456789")

    record = _file_record("src", target, want_hash=False)

    assert record["kind"] == "file"
    assert record["size"] == 10
    assert "sha256" not in record


def test_a_file_recorded_with_hashing_carries_its_digest(tmp_path):
    """The taken side, which is what makes the absence above meaningful."""
    from spacr.workspace import _file_record

    target = tmp_path / "measurements.db"
    target.write_bytes(b"0123456789")

    record = _file_record("src", target, want_hash=True)

    assert record.get("sha256")


def test_a_hash_that_could_not_be_taken_leaves_the_key_out(tmp_path,
                                                           monkeypatch):
    """Arc 311 -> 313: ``hash_file`` answered falsy.

    It returns nothing for a file it could not read -- a permission error, or
    one being written as it is read. The record still describes the file's
    size and mtime, and simply does not claim a digest.
    """
    from spacr import workspace

    target = tmp_path / "measurements.db"
    target.write_bytes(b"0123456789")

    monkeypatch.setattr(workspace, "hash_file", lambda _path: "")

    record = workspace._file_record("src", target, want_hash=True)

    assert record["size"] == 10
    assert "sha256" not in record
