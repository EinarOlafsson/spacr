"""Where :mod:`spacr.example_data` keeps the example screen, and how it gets there.

Every download is written to a ``.part`` sibling and only renamed once its
size and digest match the manifest, so an interrupted or corrupted transfer
leaves nothing that a later run would mistake for a valid cache.  The tests
drive the real download machinery against a fake HTTP body and assert the
exact paths, progress calls, status notes and refusal messages.
"""
from __future__ import annotations

import hashlib
import io
import os
import urllib.request
from pathlib import Path
from urllib.error import URLError

import pytest

from spacr import example_data as ed


def _entry(name, payload, kind="counts"):
    return {"name": name, "kind": kind, "plate": 1, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


class _Body(io.BytesIO):
    """A minimal stand-in for the object ``urlopen`` returns."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _serve(monkeypatch, payload, *, error=None):
    """Point ``urlopen`` at ``payload`` (or make it raise ``error``); return the asked URLs."""
    asked = []

    def fake_urlopen(url, timeout=None):
        asked.append((url, timeout))
        if error is not None:
            raise error
        return _Body(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return asked


def test_cache_folder_prefers_the_override_then_xdg_then_home(monkeypatch, tmp_path, subtests):
    """``SPACR_EXAMPLE_DATA`` wins verbatim, else ``XDG_CACHE_HOME``, else ``~/.cache``."""
    home = str(tmp_path / "home")
    xdg = str(tmp_path / "xdg")
    cases = [
        ("override_wins_over_xdg", {"SPACR_EXAMPLE_DATA": str(tmp_path / "chosen"),
                                    "XDG_CACHE_HOME": xdg},
         str(tmp_path / "chosen")),
        ("override_is_user_expanded", {"SPACR_EXAMPLE_DATA": "~/picked"},
         os.path.join(home, "picked")),
        ("xdg_when_nothing_overrides", {"SPACR_EXAMPLE_DATA": None, "XDG_CACHE_HOME": xdg},
         os.path.join(xdg, "spacr", "example_data")),
        ("home_dot_cache_without_xdg", {"SPACR_EXAMPLE_DATA": None, "XDG_CACHE_HOME": None},
         os.path.join(home, ".cache", "spacr", "example_data")),
    ]
    for case_id, env, expected in cases:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as patch:
                patch.setenv("HOME", home)
                patch.setenv("USERPROFILE", home)
                for key, value in env.items():
                    if value is None:
                        patch.delenv(key, raising=False)
                    else:
                        patch.setenv(key, value)
                assert ed.cache_folder() == expected


def test_is_whole_needs_the_right_size_and_the_right_digest(monkeypatch, tmp_path, subtests):
    """Only a file of the manifest size AND digest is whole; a size probe that fails is not whole."""
    payload = b"correct!"
    entry = _entry("thing.csv", payload)
    (tmp_path / "good.csv").write_bytes(payload)
    (tmp_path / "wrong.csv").write_bytes(b"WRONG!!!")
    (tmp_path / "short.csv").write_bytes(b"correct")
    (tmp_path / "folder").mkdir()

    def unsizable(path):
        raise OSError("no size for you")

    cases = [
        ("the_matching_file", "good.csv", None, True),
        ("a_missing_path", "absent.csv", None, False),
        ("right_size_wrong_bytes", "wrong.csv", None, False),
        ("wrong_size", "short.csv", None, False),
        ("a_size_probe_that_raises", "folder", unsizable, False),
    ]
    for case_id, name, getsize, expected in cases:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as patch:
                if getsize is not None:
                    patch.setattr(os.path, "getsize", getsize)
                assert ed.is_whole(str(tmp_path / name), entry) is expected


def _vanish_the_partial(monkeypatch):
    """Delete the ``.part`` file inside validation, the race the size report guards."""
    real_is_whole = ed.is_whole

    def eat(path, entry):
        if str(path).endswith(".part"):
            os.remove(path)
            return False
        return real_is_whole(path, entry)

    monkeypatch.setattr(ed, "is_whole", eat)


DOWNLOAD_FAILURES = [
    ("cancelled", b"z" * 4096, b"z" * 4096, None, lambda: True, None,
     "the download of asset.csv was cancelled"),
    ("network_failure", b"payload", b"", URLError("dns is asleep"), None, None,
     f"could not download asset.csv from {ed.BASE_URL}/asset.csv: <urlopen error "
     "dns is asleep>. The example screen is a release asset, so this needs a "
     "network connection the first time."),
    ("truncated", b"x" * 100, b"x" * 40, None, None, None,
     "asset.csv arrived incomplete or corrupt: expected 100 bytes, got 40. "
     "Nothing was kept, so pressing again is safe."),
    ("partial_vanished_before_sizing", b"y" * 64, b"y" * 8, None, None,
     _vanish_the_partial,
     "asset.csv arrived incomplete or corrupt: expected 64 bytes, got 0. "
     "Nothing was kept, so pressing again is safe."),
]


def test_download_lands_atomically_or_keeps_nothing(monkeypatch, tmp_path, subtests):
    """A valid body is renamed into place with exact progress; every failure names its cause and keeps no file."""
    payload = b"a,b\n" * 5000
    asked = _serve(monkeypatch, payload)
    seen = []
    got = ed._download(_entry("good.csv", payload), str(tmp_path / "ok"),
                       progress=lambda *call: seen.append(call))
    assert got == str(tmp_path / "ok" / "good.csv")
    assert Path(got).read_bytes() == payload
    assert sorted(p.name for p in (tmp_path / "ok").iterdir()) == ["good.csv"]
    assert seen == [("good.csv", len(payload), len(payload))]
    assert asked == [(f"{ed.BASE_URL}/good.csv", 60)]

    for case_id, manifest, body, error, cancelled, rig, message in DOWNLOAD_FAILURES:
        with subtests.test(msg=case_id):
            folder = tmp_path / case_id
            with monkeypatch.context() as patch:
                _serve(patch, body, error=error)
                if rig is not None:
                    rig(patch)
                with pytest.raises(ed.ExampleDataError) as caught:
                    ed._download(_entry("asset.csv", manifest), str(folder),
                                 cancelled=cancelled)
            assert str(caught.value) == message
            assert list(folder.iterdir()) == []


def test_fetch_downloads_only_what_is_missing_or_refuses(monkeypatch, tmp_path, subtests):
    """fetch reuses valid cache files, downloads the rest, refuses offline gaps and unknown kinds."""
    counts, scores = b"gene,count\na,1\n", b"cell,score\n1,0.5\n"
    manifest = [_entry("c.csv", counts, "counts"), _entry("s.csv", scores, "scores")]
    monkeypatch.setattr(ed, "FILES", manifest)
    payloads = {"c.csv": counts, "s.csv": scores}
    asked = []

    def fake_download(entry, folder, progress=None, cancelled=None):
        asked.append(entry["name"])
        path = os.path.join(folder, entry["name"])
        Path(path).write_bytes(payloads[entry["name"]])
        return path

    monkeypatch.setattr(ed, "_download", fake_download)

    with subtests.test(msg="only_the_missing_file_is_downloaded"):
        folder = tmp_path / "half"
        folder.mkdir()
        (folder / "c.csv").write_bytes(counts)
        result = ed.fetch(str(folder))
        assert asked == ["s.csv"]
        assert result == ed.Fetched(counts=[str(folder / "c.csv")],
                                    scores=[str(folder / "s.csv")],
                                    downloaded=[str(folder / "s.csv")],
                                    folder=str(folder))
        assert result.files == [str(folder / "c.csv"), str(folder / "s.csv")]

    with subtests.test(msg="the_cache_folder_is_used_when_none_is_named"):
        asked.clear()
        cache = tmp_path / "cache"
        cache.mkdir()
        for name, data in payloads.items():
            (cache / name).write_bytes(data)
        monkeypatch.setenv("SPACR_EXAMPLE_DATA", str(cache))
        result = ed.fetch(kind="counts")
        assert asked == []
        assert result == ed.Fetched(counts=[str(cache / "c.csv")], scores=[],
                                    downloaded=[], folder=str(cache))

    with subtests.test(msg="download_false_refuses_an_incomplete_cache"):
        asked.clear()
        empty = tmp_path / "empty"
        with pytest.raises(ed.ExampleDataError) as caught:
            ed.fetch(str(empty), download=False)
        assert str(caught.value) == (
            f"2 of the 2 example files are not cached in {empty}, and "
            "downloading was not allowed.")
        assert asked == []

    with subtests.test(msg="a_misspelled_kind_touches_nothing"):
        asked.clear()
        keep = tmp_path / "keep"
        keep.mkdir()
        (keep / "c.csv.part").write_bytes(b"retained interrupted download")
        (keep / "user-notes.csv").write_bytes(b"reviewed,keep\n1,yes\n")
        before = {p.name: p.read_bytes() for p in keep.iterdir()}
        network = _serve(monkeypatch, b"")
        with pytest.raises(ValueError) as caught:
            ed.fetch(keep, kind="count")
        assert str(caught.value) == (
            "no example files of kind 'count'; the manifest has ['counts', 'scores']")
        assert (asked, network) == ([], [])
        assert {p.name: p.read_bytes() for p in keep.iterdir()} == before


def test_fetched_note_says_what_arrived_and_what_was_cached(subtests):
    """The status line distinguishes downloads from cache hits and is exact for every mix."""
    cases = [
        ("nothing_at_all", ed.Fetched([], [], [], "/nowhere"), "No example data."),
        ("mixed", ed.Fetched(["/x/a.csv"], ["/x/b.csv"], ["/x/a.csv"], "/x"),
         "Example screen ready (downloaded 1, 1 already cached): 1 count table(s) "
         "and 1 score table(s) in /x."),
        ("all_downloaded", ed.Fetched(["/x/a.csv"], [], ["/x/a.csv"], "/x"),
         "Example screen ready (downloaded 1): 1 count table(s) and 0 score "
         "table(s) in /x."),
        ("all_cached", ed.Fetched(["/x/a.csv"], ["/x/b.csv", "/x/c.csv"], [], "/x"),
         "Example screen ready (all 3 already cached): 1 count table(s) and 2 "
         "score table(s) in /x."),
    ]
    for case_id, result, expected in cases:
        with subtests.test(msg=case_id):
            assert result.note() == expected
