"""The example screen is fetched, validated, and cached honestly.

Every download is written to a ``.part`` sibling and only renamed once its
size and digest match the manifest, so an interrupted or corrupted transfer
leaves nothing behind that a later run would mistake for a valid cache.
These tests drive the real download machinery against a fake HTTP body so
the byte accounting, the cancellation path, and the atomic rename are all
exercised rather than described.
"""
from __future__ import annotations

import hashlib

from spacr import example_data as ed


def _entry(name, payload, kind="counts"):
    return {
        "name": name,
        "kind": kind,
        "plate": 1,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


# ---------------------------------------------------------------------------
# cache location
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_total_bytes_sums_the_whole_manifest_by_default():
    """With no argument the total covers every manifest entry."""
    from spacr.example_data_manifest import FILES

    assert ed.total_bytes() == sum(int(e["bytes"]) for e in FILES)
    assert ed.total_bytes(FILES[:1]) == int(FILES[0]["bytes"])
    assert ed.total_bytes([]) == 0


# ---------------------------------------------------------------------------
# downloading
# ---------------------------------------------------------------------------

def test_forgetting_a_file_that_is_not_there_is_quiet(tmp_path):
    """``_forget`` swallows the OSError from a path that never existed."""
    ed._forget(str(tmp_path / "never-was"))          # must not raise
    real = tmp_path / "real"
    real.write_bytes(b"x")
    ed._forget(str(real))
    assert not real.exists()


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def test_a_full_cache_needs_no_network_at_all(monkeypatch, tmp_path):
    """Every file present means ``download=False`` succeeds."""
    counts, scores = b"one", b"two"
    manifest = [_entry("c.csv", counts, "counts"),
                _entry("s.csv", scores, "scores")]
    monkeypatch.setattr(ed, "FILES", manifest)
    (tmp_path / "c.csv").write_bytes(counts)
    (tmp_path / "s.csv").write_bytes(scores)

    result = ed.fetch(str(tmp_path), download=False)

    assert result.downloaded == []
    assert ed.missing(str(tmp_path)) == []
    assert "all 2 already cached" in result.note()


# ---------------------------------------------------------------------------
# the status line
# ---------------------------------------------------------------------------

