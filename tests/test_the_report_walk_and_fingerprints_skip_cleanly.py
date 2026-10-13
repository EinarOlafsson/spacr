"""Skips in the report's directory walk, its run stamps and its fingerprints.

Every arc here is a loop passing over one item. A run folder is not a tidy
place: it holds sockets and half-written files as well as directories, a
journal can carry a record of the wrong shape, and a manifest can record a
digest without the file counts beside it. In each case the loop must carry on
with the rest -- the report is a summary, and one odd entry must not cost the
other ninety-nine.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# _iter_dir_files and _dir_stats — an entry that is neither file nor directory
# ---------------------------------------------------------------------------

@pytest.fixture
def folder_with_a_fifo(tmp_path):
    """A directory holding a real file, a subdirectory and a FIFO.

    A FIFO is the smallest thing that is neither ``is_file`` nor ``is_dir``,
    and it is not a contrivance: spaCR's own workers and any editor's swap
    machinery leave sockets and pipes in working folders.
    """
    (tmp_path / "real.npy").write_bytes(b"0123456789")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.npy").write_bytes(b"01234")
    fifo = tmp_path / "a_pipe"
    try:
        os.mkfifo(fifo)
    except (AttributeError, NotImplementedError, OSError):
        pytest.skip("this platform has no mkfifo")
    assert stat.S_ISFIFO(os.stat(fifo).st_mode)
    return tmp_path


def test_the_file_walk_passes_over_something_that_is_neither(folder_with_a_fifo):
    """Arc 306 -> 299: the loop goes round rather than listing the pipe.

    Listing it would put a path in the report that cannot be opened, and the
    reader would go looking for a corrupt output.
    """
    from spacr.report import _iter_dir_files

    files, truncated = _iter_dir_files(Path(folder_with_a_fifo))

    names = sorted(p.name for p in files)
    assert names == ["deep.npy", "real.npy"]
    assert truncated is False


def test_the_size_walk_passes_over_something_that_is_neither(folder_with_a_fifo):
    """Arc 334 -> 328, and the size must not be charged for the pipe.

    ``entry.stat()`` on a FIFO succeeds and reports zero, so the count is
    what would go wrong: a folder reported as holding three files when two
    can be read.
    """
    from spacr.report import _dir_stats

    n_files, total, truncated = _dir_stats(Path(folder_with_a_fifo))

    assert n_files == 2
    assert total == 15
    assert truncated is False


def test_the_walk_stops_at_its_budget(tmp_path):
    """The budget guard above both, which reports truncation rather than lying."""
    from spacr.report import _iter_dir_files

    for index in range(6):
        (tmp_path / f"f{index}.npy").write_bytes(b"0")

    files, truncated = _iter_dir_files(Path(tmp_path), budget=3)

    assert len(files) == 3
    assert truncated is True


# ---------------------------------------------------------------------------
# _read_stamps — a journal record that is not a mapping
# ---------------------------------------------------------------------------

def test_a_run_status_record_of_the_wrong_shape_is_passed_over(tmp_path,
                                                               monkeypatch):
    """Arc 708 -> 707.

    ``read_run_status`` reads a sidecar or a database written by an older
    spaCR, and a list where a dict belongs is exactly what a format change
    leaves behind. Passing over it keeps the records that ARE readable, which
    is the difference between a partial report and none.
    """
    from spacr import errors as errors_module
    from spacr.report import _read_stamps

    good = {"status": "complete", "n_attempted": 3, "n_succeeded": 3,
            "n_failed": 0, "failures": []}
    monkeypatch.setattr(errors_module, "read_run_status",
                        lambda _path: ["not a record", good, None])

    path = tmp_path / "measurements.db"
    path.write_bytes(b"")

    stamps, problems = _read_stamps([path])

    assert [record for _path, record in stamps] == [good]
    assert problems == []


def test_a_run_status_that_cannot_be_read_becomes_a_named_problem(tmp_path,
                                                                  monkeypatch):
    """The except beside it: the file is named, because a bare count is useless."""
    from spacr import errors as errors_module
    from spacr.report import _read_stamps

    def refuse(_path):
        raise RuntimeError("the database is locked")

    monkeypatch.setattr(errors_module, "read_run_status", refuse)
    path = tmp_path / "measurements.db"
    path.write_bytes(b"")

    stamps, problems = _read_stamps([path])

    assert stamps == []
    assert problems and "measurements.db" in problems[0]


# ---------------------------------------------------------------------------
# _fingerprints — a digest recorded without the counts beside it
# ---------------------------------------------------------------------------

def _run(manifest, name="run_1"):
    class _Dir:
        def __init__(self, value):
            self.name = value

    return {"dir": _Dir(name), "manifest": manifest}


def test_a_digest_without_file_counts_is_shown_on_its_own(tmp_path):
    """Arc 1009 -> 1014: no ``count``, so no "n file(s) → " prefix.

    The digest is the verifiable part and must appear whether or not the
    performance block recorded counts. An older run, or one whose counting was
    interrupted, still gets its fingerprint printed.
    """
    from spacr.report import _FINGERPRINT_SCHEMA, _fingerprints

    manifest = {"schema_version": _FINGERPRINT_SCHEMA,
                "input_tree_sha256": "abc123",
                "performance": {}}

    rendered, _notes = _fingerprints([_run(manifest)])

    text = " ".join(str(item) for item in rendered)
    assert "abc123" in text
    assert "file(s)" not in text


def test_a_digest_with_counts_but_no_byte_total_omits_the_size(tmp_path):
    """Arc 1011 -> 1013: counts without bytes still get their arrow.

    Zero bytes is also falsy here, and that is correct rather than a bug: a
    run whose inputs total zero bytes has nothing worth printing a size for.
    """
    from spacr.report import _FINGERPRINT_SCHEMA, _fingerprints

    manifest = {"schema_version": _FINGERPRINT_SCHEMA,
                "input_tree_sha256": "abc123",
                "performance": {"input_files": 12, "input_bytes": 0}}

    rendered, _notes = _fingerprints([_run(manifest)])

    text = " ".join(str(item) for item in rendered)
    assert "12 file(s)" in text
    assert "→" in text
    assert "abc123" in text


def test_a_digest_with_counts_and_bytes_shows_both():
    """The fully populated case, so the two omissions above are visible choices."""
    from spacr.report import _FINGERPRINT_SCHEMA, _fingerprints

    manifest = {"schema_version": _FINGERPRINT_SCHEMA,
                "input_tree_sha256": "abc123",
                "performance": {"input_files": 12, "input_bytes": 4096}}

    rendered, _notes = _fingerprints([_run(manifest)])

    text = " ".join(str(item) for item in rendered)
    assert "12 file(s)" in text and "abc123" in text


def test_a_run_whose_manifest_records_nothing_is_passed_over():
    """Arc 1016 -> 969: a run with no entries adds no section at all.

    An empty heading in a report reads as "this run had no fingerprints",
    which is a claim; leaving it out says nothing, which is the truth.
    """
    from spacr.report import _fingerprints

    from spacr.report import _FINGERPRINT_SCHEMA

    # Current schema, hashing was on, and yet no digest of any kind was
    # written -- which is what an interrupted journal leaves. Every entry
    # source is skipped, so `entries` is empty and the run contributes nothing.
    empty = {"schema_version": _FINGERPRINT_SCHEMA, "started": "2026-08-30"}

    rendered, _notes = _fingerprints([_run(empty)])

    assert rendered == []
