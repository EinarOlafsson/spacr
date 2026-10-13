"""Fingerprinting an artifact, and the one-line verdicts a user reads.

The fingerprint decides whether a downstream stage re-runs, so the METHOD it
used is carried beside the digest: a full hash and a size-and-mtime stamp are
not comparable, and a run that compared one against the other would either
re-run everything or nothing.
"""
from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# content_fingerprint
# ---------------------------------------------------------------------------

def test_a_small_file_is_hashed_whole(tmp_path):
    """Under the limit, so the digest is of the bytes themselves."""
    from spacr.artifacts import content_fingerprint

    target = tmp_path / "measurements.db"
    target.write_bytes(b"0123456789")

    print_ = content_fingerprint(target)

    assert print_.digest
    assert print_.size_bytes == 10
    assert bool(print_)


def test_two_identical_small_files_fingerprint_alike(tmp_path):
    """What a content hash is FOR: same bytes, same answer.

    A copy moved between folders must not look like new work.
    """
    from spacr.artifacts import content_fingerprint

    (tmp_path / "a.db").write_bytes(b"same bytes")
    (tmp_path / "b.db").write_bytes(b"same bytes")

    assert (content_fingerprint(tmp_path / "a.db").digest
            == content_fingerprint(tmp_path / "b.db").digest)


def test_a_file_over_the_limit_is_stamped_rather_than_read(tmp_path):
    """The size guard, and the method that records which was used.

    Hashing a 300 GB measurement database to decide whether to re-run would
    cost more than the re-run. The method name is what stops a stamp being
    compared against a hash.
    """
    from spacr.artifacts import content_fingerprint

    target = tmp_path / "big.db"
    target.write_bytes(b"0123456789")

    stamped = content_fingerprint(target, full_hash_limit=4)
    hashed = content_fingerprint(target, full_hash_limit=1024)

    assert stamped.digest and hashed.digest
    assert stamped.method != hashed.method
    assert stamped.digest != hashed.digest


def test_a_path_that_is_not_there_fingerprints_as_nothing(tmp_path):
    """The falsy fingerprint, which __bool__ exists to make readable.

    "Nothing was fingerprinted" and "this fingerprinted to an empty digest"
    are the same value, so the caller checks truthiness rather than the digest
    string.
    """
    from spacr.artifacts import content_fingerprint

    print_ = content_fingerprint(tmp_path / "no_such_file")

    assert not print_
    assert not print_.digest


# ---------------------------------------------------------------------------
# The one-line summaries
# ---------------------------------------------------------------------------

def _artifact(**changes):
    from spacr.artifacts import Artifact

    fields = dict(artifact_id="a1", project="p", kind="measurements",
                  role="output", path="/data/plate1/measurements.db",
                  module="measure", run_id="r1", settings_hash="s1",
                  spacr_version="1.0", created_ns=0, created_utc="",
                  fingerprint="f", fingerprint_method="sha256",
                  size_bytes=10, n_files=1, status="ok", settings={},
                  extra={})
    fields.update(changes)
    return Artifact(**fields)


def test_an_artifact_names_itself_with_its_kind_module_and_path():
    """The line a run log carries, which is how a user finds the file."""
    text = str(_artifact())

    assert "a1" in text
    assert "measurements" in text
    assert "measure" in text
    assert "/data/plate1/measurements.db" in text


def test_a_current_artifact_says_current_and_nothing_else():
    """No reasons, so no dash and no trailing empty clause."""
    from spacr.artifacts import Staleness

    text = str(Staleness(artifact_id="a1", stale=False))

    assert text == "a1: current"
    assert "—" not in text


def test_a_stale_artifact_lists_why():
    """The reasons are the actionable half; the verdict alone is not."""
    from spacr.artifacts import Staleness

    text = str(Staleness(artifact_id="a1", stale=True,
                         reasons=("settings changed", "upstream is newer")))

    assert text.startswith("a1: stale")
    assert "settings changed" in text and "upstream is newer" in text


def test_a_stale_verdict_is_truthy_and_a_current_one_is_not():
    """__bool__, so ``if staleness:`` reads as "if it is stale"."""
    from spacr.artifacts import Staleness

    assert Staleness(artifact_id="a1", stale=True)
    assert not Staleness(artifact_id="a1", stale=False)
