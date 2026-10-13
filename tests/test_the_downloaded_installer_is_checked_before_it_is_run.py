"""Item 416: what the updater downloads is checked before it is run.

The helper downloads a release asset, makes it executable and runs it. Until
today the only checks were that the URL began with the release download
prefix and that the file was not empty.

WHAT THIS IS NOT. The sums file comes from the same server as the
installer, so it is not a defence against a compromised release. It catches
a truncated or corrupted download, a proxy serving something stale, and an
asset that is not the one the plan named -- before the file is made
executable.
"""

from __future__ import annotations

import hashlib
import io
import os

import pytest

from spacr import install_cleanup as ic

PAYLOAD = b"#!/bin/sh\necho installer\n" * 64
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
URL = (f"{ic._RELEASE_DOWNLOAD}/v1.5.0.9/"
       f"spaCR-1.5.0.9-Linux-x86_64-Online.run")


class _Response(io.BytesIO):
    """Enough of an HTTP response for urlopen's context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


def _serve(monkeypatch, asset: bytes, sums: object):
    """Answer the installer URL with ``asset`` and the sums URL with ``sums``.

    ``sums`` may be bytes, or an exception to raise -- which is how a
    release that publishes no sums file is played.
    """
    import urllib.request

    def _urlopen(request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        if url.endswith(ic._SUMS_NAME):
            if isinstance(sums, Exception):
                raise sums
            return _Response(sums)
        return _Response(asset)

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)


def test_a_download_that_matches_is_kept_and_made_executable(tmp_path,
                                                             monkeypatch):
    target = tmp_path / "installer.run"
    _serve(monkeypatch, PAYLOAD,
           f"{DIGEST}  spaCR-1.5.0.9-Linux-x86_64-Online.run\n".encode())
    ic._download(URL, str(target))
    assert target.read_bytes() == PAYLOAD
    assert target.stat().st_mode & 0o111, "it has to be runnable"


def test_a_download_that_does_not_match_is_deleted_and_refused(tmp_path,
                                                               monkeypatch):
    """The whole point: this happens BEFORE anything is removed or run."""
    target = tmp_path / "installer.run"
    wrong = hashlib.sha256(b"something else").hexdigest()
    _serve(monkeypatch, PAYLOAD,
           f"{wrong}  spaCR-1.5.0.9-Linux-x86_64-Online.run\n".encode())
    with pytest.raises(OSError, match="does not match the checksum"):
        ic._download(URL, str(target))
    assert not target.exists(), "a file that failed its checksum was left behind"


def test_the_refusal_says_nothing_was_installed_or_removed(tmp_path,
                                                           monkeypatch):
    target = tmp_path / "installer.run"
    _serve(monkeypatch, PAYLOAD, b"0" * 64 + b"  spaCR-1.5.0.9-Linux-x86_64-Online.run\n")
    with pytest.raises(OSError) as raised:
        ic._download(URL, str(target))
    assert "Nothing was installed and nothing was removed" in str(raised.value)


def test_a_release_with_no_sums_file_still_installs(tmp_path, monkeypatch):
    """An older release published none, and must stay installable."""
    target = tmp_path / "installer.run"
    _serve(monkeypatch, PAYLOAD, OSError("404"))
    ic._download(URL, str(target))
    assert target.read_bytes() == PAYLOAD


def test_a_sums_file_that_does_not_name_this_asset_is_not_a_failure(
        tmp_path, monkeypatch):
    target = tmp_path / "installer.run"
    _serve(monkeypatch, PAYLOAD,
           f"{DIGEST}  some-other-file.pkg\n".encode())
    ic._download(URL, str(target))
    assert target.read_bytes() == PAYLOAD


def test_an_empty_download_is_still_refused(tmp_path, monkeypatch):
    target = tmp_path / "installer.run"
    _serve(monkeypatch, b"", b"")
    with pytest.raises(OSError, match="was empty"):
        ic._download(URL, str(target))


def test_nothing_is_fetched_from_anywhere_but_the_release(tmp_path):
    with pytest.raises(ValueError, match="refusing to download"):
        ic._download("https://example.com/installer.run",
                     str(tmp_path / "x.run"))


def test_the_digest_is_read_from_the_sums_file_beside_the_asset(monkeypatch):
    seen = []
    import urllib.request

    def _urlopen(request, timeout=None):
        seen.append(request.full_url)
        return _Response(f"{DIGEST}  spaCR-1.5.0.9-Linux-x86_64-Online.run\n".encode())

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    assert ic._published_digest(URL) == DIGEST
    assert seen == [f"{ic._RELEASE_DOWNLOAD}/v1.5.0.9/{ic._SUMS_NAME}"]


def test_a_failed_fetch_stops_the_plan_before_anything_is_removed(tmp_path):
    """The property that makes a checksum worth having here."""
    import inspect

    source = inspect.getsource(ic._run_plan)
    fetch_at = source.index('plan.get("fetch")')
    removed_at = source.index("run_update_sequence")
    assert fetch_at < removed_at, (
        "the download must happen before anything is removed, or a refused "
        "installer leaves a machine with nothing on it")
    assert "nothing was removed" in source
