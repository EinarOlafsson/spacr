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
import io
import os
from pathlib import Path

import pytest

from spacr import example_data as ed


def _entry(name, payload, kind="counts"):
    return {
        "name": name,
        "kind": kind,
        "plate": 1,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


class _Body(io.BytesIO):
    """A minimal stand-in for the object ``urlopen`` returns."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _serve(monkeypatch, payload, *, error=None):
    """Point ``urlopen`` at ``payload`` (or make it raise ``error``)."""
    import urllib.request

    def fake_urlopen(url, timeout=None):
        if error is not None:
            raise error
        return _Body(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


# ---------------------------------------------------------------------------
# cache location
# ---------------------------------------------------------------------------

def test_the_cache_follows_xdg_when_nothing_overrides_it(monkeypatch, tmp_path):
    """``XDG_CACHE_HOME`` decides where the example screen is kept."""
    monkeypatch.delenv("SPACR_EXAMPLE_DATA", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    assert ed.cache_folder() == os.path.join(
        str(tmp_path / "xdg"), "spacr", "example_data")


def test_without_xdg_the_cache_falls_back_to_the_home_dot_cache(
        monkeypatch, tmp_path):
    """A missing ``XDG_CACHE_HOME`` still gives a per-user cache path."""
    monkeypatch.delenv("SPACR_EXAMPLE_DATA", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    where = ed.cache_folder()

    assert where == os.path.join(
        str(tmp_path / "home"), ".cache", "spacr", "example_data")


def test_the_override_wins_over_xdg(monkeypatch, tmp_path):
    """``SPACR_EXAMPLE_DATA`` is used verbatim, expanded."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("SPACR_EXAMPLE_DATA", str(tmp_path / "chosen"))

    assert ed.cache_folder() == str(tmp_path / "chosen")


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_a_file_that_is_not_there_is_not_whole(tmp_path):
    """A missing path fails validation instead of raising."""
    entry = _entry("absent.csv", b"abc")

    assert ed.is_whole(str(tmp_path / "absent.csv"), entry) is False


def test_a_directory_where_a_file_belongs_is_not_whole(tmp_path):
    """``getsize`` raising is a validation failure, not a crash."""
    entry = _entry("thing.csv", b"abcdef")
    # A path that exists but cannot be sized the way a file is: the size
    # probe has to survive it.
    bad = tmp_path / "thing.csv"
    bad.mkdir()

    def explode(path):
        raise OSError("no size for you")

    saved = os.path.getsize
    try:
        os.path.getsize = explode
        assert ed.is_whole(str(bad), entry) is False
    finally:
        os.path.getsize = saved


def test_the_right_size_with_the_wrong_bytes_is_not_whole(tmp_path):
    """The digest, not just the length, decides."""
    entry = _entry("thing.csv", b"correct!")
    path = tmp_path / "thing.csv"
    path.write_bytes(b"WRONG!!!")
    assert path.stat().st_size == entry["bytes"]

    assert ed.is_whole(str(path), entry) is False


def test_total_bytes_sums_the_whole_manifest_by_default():
    """With no argument the total covers every manifest entry."""
    from spacr.example_data_manifest import FILES

    assert ed.total_bytes() == sum(int(e["bytes"]) for e in FILES)
    assert ed.total_bytes(FILES[:1]) == int(FILES[0]["bytes"])
    assert ed.total_bytes([]) == 0


# ---------------------------------------------------------------------------
# downloading
# ---------------------------------------------------------------------------

def test_a_good_download_lands_atomically_and_reports_progress(
        monkeypatch, tmp_path):
    """The ``.part`` file is renamed only after it validates."""
    payload = b"a,b\n" * 5000
    entry = _entry("good.csv", payload)
    _serve(monkeypatch, payload)
    seen = []

    got = ed._download(entry, str(tmp_path),
                       progress=lambda n, r, t: seen.append((n, r, t)))

    assert got == str(tmp_path / "good.csv")
    assert Path(got).read_bytes() == payload
    assert not (tmp_path / "good.csv.part").exists()
    assert seen, "progress was never called"
    assert seen[-1] == ("good.csv", len(payload), len(payload))
    assert all(total == len(payload) for _, _, total in seen)


def test_a_cancelled_download_keeps_nothing(monkeypatch, tmp_path):
    """Cancelling deletes the partial file and names the asset."""
    payload = b"z" * 4096
    entry = _entry("cancel.csv", payload)
    _serve(monkeypatch, payload)

    with pytest.raises(ed.ExampleDataError) as caught:
        ed._download(entry, str(tmp_path), cancelled=lambda: True)

    assert "cancel.csv" in str(caught.value)
    assert "cancelled" in str(caught.value)
    assert list(tmp_path.iterdir()) == []


def test_a_network_failure_explains_that_the_first_fetch_needs_one(
        monkeypatch, tmp_path):
    """A URLError becomes an ExampleDataError naming the URL and the cause."""
    from urllib.error import URLError

    entry = _entry("net.csv", b"payload")
    _serve(monkeypatch, b"", error=URLError("dns is asleep"))

    with pytest.raises(ed.ExampleDataError) as caught:
        ed._download(entry, str(tmp_path))

    message = str(caught.value)
    assert "net.csv" in message
    assert ed.BASE_URL in message
    assert "network connection" in message
    assert list(tmp_path.iterdir()) == []


def test_a_truncated_download_is_thrown_away_and_says_how_short(
        monkeypatch, tmp_path):
    """Bytes that do not match the manifest are never renamed into place."""
    entry = _entry("short.csv", b"x" * 100)
    _serve(monkeypatch, b"x" * 40)

    with pytest.raises(ed.ExampleDataError) as caught:
        ed._download(entry, str(tmp_path))

    message = str(caught.value)
    assert "incomplete or corrupt" in message
    assert "100" in message and "40" in message
    assert "pressing again is safe" in message
    assert list(tmp_path.iterdir()) == []


def test_a_partial_file_that_vanished_is_reported_as_zero_bytes(
        monkeypatch, tmp_path):
    """The size probe survives the ``.part`` file disappearing mid-check."""
    entry = _entry("gone.csv", b"y" * 64)
    _serve(monkeypatch, b"y" * 8)
    real_is_whole = ed.is_whole

    def eat_the_partial(path, entry_):
        # The validator's own filesystem call is what the reporting path
        # then repeats; deleting between the two is the race it guards.
        if str(path).endswith(".part"):
            os.remove(path)
            return False
        return real_is_whole(path, entry_)

    monkeypatch.setattr(ed, "is_whole", eat_the_partial)

    with pytest.raises(ed.ExampleDataError) as caught:
        ed._download(entry, str(tmp_path))

    assert "got 0" in str(caught.value)


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

def test_fetch_downloads_only_what_is_missing(monkeypatch, tmp_path):
    """A file already valid in the cache is not fetched again."""
    counts = b"gene,count\na,1\n"
    scores = b"cell,score\n1,0.5\n"
    manifest = [_entry("c.csv", counts, "counts"),
                _entry("s.csv", scores, "scores")]
    monkeypatch.setattr(ed, "FILES", manifest)
    (tmp_path / "c.csv").write_bytes(counts)
    asked = []

    def fake_download(entry, folder, progress=None, cancelled=None):
        asked.append(entry["name"])
        path = os.path.join(folder, entry["name"])
        Path(path).write_bytes(scores)
        return path

    monkeypatch.setattr(ed, "_download", fake_download)

    result = ed.fetch(str(tmp_path))

    assert asked == ["s.csv"]
    assert result.downloaded == [str(tmp_path / "s.csv")]
    assert result.counts == [str(tmp_path / "c.csv")]
    assert result.scores == [str(tmp_path / "s.csv")]
    assert result.folder == str(tmp_path)
    assert set(result.files) == {str(tmp_path / "c.csv"),
                                 str(tmp_path / "s.csv")}


def test_fetch_refuses_to_use_the_network_when_told_not_to(
        monkeypatch, tmp_path):
    """``download=False`` on an empty cache is an error, not a silent fetch."""
    manifest = [_entry("c.csv", b"one", "counts"),
                _entry("s.csv", b"two", "scores")]
    monkeypatch.setattr(ed, "FILES", manifest)

    with pytest.raises(ed.ExampleDataError) as caught:
        ed.fetch(str(tmp_path), download=False)

    assert "2 of the 2" in str(caught.value)
    assert str(tmp_path) in str(caught.value)


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


def test_fetch_uses_the_cache_folder_when_none_is_named(monkeypatch, tmp_path):
    """No folder argument means the configured cache directory."""
    payload = b"only"
    manifest = [_entry("c.csv", payload, "counts")]
    monkeypatch.setattr(ed, "FILES", manifest)
    monkeypatch.setenv("SPACR_EXAMPLE_DATA", str(tmp_path / "cache"))
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "c.csv").write_bytes(payload)

    result = ed.fetch()

    assert result.folder == str(tmp_path / "cache")
    assert result.counts == [str(tmp_path / "cache" / "c.csv")]


# ---------------------------------------------------------------------------
# the status line
# ---------------------------------------------------------------------------

def test_an_empty_result_says_there_is_no_example_data():
    """Nothing fetched reads as nothing, not as a zero-count sentence."""
    assert ed.Fetched([], [], [], "/nowhere").note() == "No example data."


def test_a_mixed_result_counts_the_downloads_and_the_cache_hits():
    """The note distinguishes what arrived now from what was already here."""
    result = ed.Fetched(["/x/a.csv"], ["/x/b.csv"], ["/x/a.csv"], "/x")

    note = result.note()

    assert "downloaded 1" in note
    assert "1 already cached" in note
    assert "1 count table(s)" in note and "1 score table(s)" in note
    assert "/x" in note


def test_a_fully_downloaded_result_does_not_mention_the_cache():
    """With nothing cached the note carries no cache clause."""
    note = ed.Fetched(["/x/a.csv"], [], ["/x/a.csv"], "/x").note()

    assert "downloaded 1" in note
    assert "already cached" not in note
