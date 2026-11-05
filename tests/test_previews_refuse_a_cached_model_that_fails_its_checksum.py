"""A preview refuses cached model bytes a full run would refuse, and hashes a file once."""
from __future__ import annotations

import hashlib
import os

import pytest

from spacr import model_zoo as mz

PAYLOAD = b"the bytes the catalogue row was published with"
COMMUNITY = {
    "key": "community_vtest", "name": "community_vtest_e1", "kind": "cellpose",
    "uri": "https://example.invalid/community_vtest_e1",
    "sha256": hashlib.sha256(PAYLOAD).hexdigest(), "version": "1",
}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An empty model cache, an empty digest memory and a worker thread.

    A QApplication left by an earlier test on the same worker would make the
    main thread Qt's GUI thread, where nothing is hashed; the tests that mean
    the GUI thread say so.
    """
    monkeypatch.setenv("SPACR_MODEL_CACHE", str(tmp_path))
    monkeypatch.setattr(mz, "_on_the_qt_gui_thread", lambda: False)
    monkeypatch.setattr(mz, "_VERIFIED_DIGESTS", {})
    monkeypatch.setattr(mz, "fetch", lambda *a, **k: pytest.fail("fetched"))
    return tmp_path


def _bundled_entry():
    """A bundled catalogue row with a checksum and a uri."""
    raw = next(r for r in mz.BUNDLED_REMOTE_MODELS if r.get("sha256") and r.get("uri"))
    return mz._entry_from_mapping(raw)


def _count_hashes(monkeypatch):
    """Count how often a file is actually read for hashing."""
    calls = []
    real = mz.sha256_file
    monkeypatch.setattr(
        mz, "sha256_file", lambda p, *a, **k: (calls.append(str(p)), real(p, *a, **k))[1])
    return calls


def test_a_matching_community_file_is_used_and_hashed_once(cache, monkeypatch):
    entry = mz._entry_from_mapping(COMMUNITY)
    (cache / entry.name).write_bytes(PAYLOAD)
    calls = _count_hashes(monkeypatch)
    for _ in range(3):
        assert mz._ensure_catalogue_model_file(entry, download=False) == cache / entry.name
    assert len(calls) == 1


def test_a_mismatching_community_file_is_refused_with_the_expected_hash(cache):
    entry = mz._entry_from_mapping(COMMUNITY)
    target = cache / entry.name
    target.write_bytes(b"something else")
    with pytest.raises(mz.ModelUnreadable) as error:
        mz._ensure_catalogue_model_file(entry, download=False)
    text = str(error.value)
    assert str(target) in text and entry.sha256 in text and "Model Zoo" in text
    assert hashlib.sha256(b"something else").hexdigest() in text


def test_a_mismatching_bundled_file_is_refused(cache):
    entry = _bundled_entry()
    (cache / entry.name).write_bytes(b"not the published checkpoint")
    with pytest.raises(mz.ModelUnreadable, match=entry.sha256):
        mz._ensure_catalogue_model_file(entry, download=False)


def test_a_matching_bundled_file_is_used(cache, monkeypatch):
    entry = _bundled_entry()
    target = cache / entry.name
    target.write_bytes(PAYLOAD)
    monkeypatch.setattr(mz, "sha256_file", lambda p, *a, **k: entry.sha256.lower())
    assert mz._ensure_catalogue_model_file(entry, download=False) == target


def test_a_changed_mtime_verifies_again(cache, monkeypatch):
    entry = mz._entry_from_mapping(COMMUNITY)
    target = cache / entry.name
    target.write_bytes(PAYLOAD)
    calls = _count_hashes(monkeypatch)
    mz._ensure_catalogue_model_file(entry, download=False)
    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    mz._ensure_catalogue_model_file(entry, download=False)
    assert len(calls) == 2
    target.write_bytes(b"replaced bytes")
    with pytest.raises(mz.ModelUnreadable):
        mz._ensure_catalogue_model_file(entry, download=False)


def test_a_planning_caller_can_opt_out_of_the_check(cache):
    entry = mz._entry_from_mapping(COMMUNITY)
    target = cache / entry.name
    target.write_bytes(b"something else")
    assert mz._ensure_catalogue_model_file(entry, download=False, verify=False) == target


def test_the_gui_thread_does_not_hash_an_unverified_file(cache, monkeypatch):
    entry = mz._entry_from_mapping(COMMUNITY)
    target = cache / entry.name
    target.write_bytes(b"something else")
    calls = _count_hashes(monkeypatch)
    monkeypatch.setattr(mz, "_on_the_qt_gui_thread", lambda: True)
    assert mz._ensure_catalogue_model_file(entry, download=False) == target
    assert calls == []
    monkeypatch.setattr(mz, "_on_the_qt_gui_thread", lambda: False)
    with pytest.raises(mz.ModelUnreadable):
        mz._ensure_catalogue_model_file(entry, download=False)
    monkeypatch.setattr(mz, "_on_the_qt_gui_thread", lambda: True)
    with pytest.raises(mz.ModelUnreadable):
        mz._ensure_catalogue_model_file(entry, download=False)
    assert len(calls) == 1


def test_a_matching_older_version_is_still_found(cache):
    entry = mz._entry_from_mapping(COMMUNITY)
    (cache / entry.name).write_bytes(b"stale")
    versioned = cache / "community_vtest_e1_v2"
    versioned.write_bytes(PAYLOAD)
    assert mz._ensure_catalogue_model_file(entry, download=False) == versioned


def test_the_gui_thread_returns_an_unhashed_older_version_not_a_missing_file(
        cache, monkeypatch):
    entry = mz._entry_from_mapping(COMMUNITY)
    first = cache / "community_vtest_e1_v2"
    second = cache / "community_vtest_e1_v3"
    first.write_bytes(b"an older publication")
    second.write_bytes(PAYLOAD)
    (cache / "community_vtest_e1_v1").mkdir()
    calls = _count_hashes(monkeypatch)
    monkeypatch.setattr(mz, "_on_the_qt_gui_thread", lambda: True)
    assert mz._ensure_catalogue_model_file(entry, download=False) == first
    assert calls == []
    monkeypatch.setattr(mz, "_on_the_qt_gui_thread", lambda: False)
    assert mz._ensure_catalogue_model_file(entry, download=False) == second
    assert len(calls) == 2
    monkeypatch.setattr(mz, "_on_the_qt_gui_thread", lambda: True)
    assert mz._ensure_catalogue_model_file(entry, download=False) == second
    assert len(calls) == 2
