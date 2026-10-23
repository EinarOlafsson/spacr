"""Timepose release plumbing in the model zoo: fetch, token, files, device, loader edges.

A fake release is registered and the download is replaced by a fake writer, so
no token, network or GPU is used.
"""
from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import pytest

from spacr import model_zoo

FILES = {
    "weights/w.pt": b"weights",
    "fakepose288/__init__.py": b"",
    "fakepose288/infer.py": (b"CONTRACT = 'spacr.timepose/1'\n"
                             b"def load_model(folder, device):\n"
                             b"    return {'folder': folder, 'device': device}\n"),
}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _record(key="timepose_288", contract="spacr.timepose/1", entry="fakepose288.infer"):
    return {"key": key, "kind": "timepose", "name": key,
            "repo_id": "someone/private-timepose", "revision": "b" * 40,
            "entry": entry, "contract": contract,
            "weights": "weights/w.pt", "display_name": "Timepose 288",
            "files": {name: _sha(data) for name, data in FILES.items()}}


def _writer(content=None, calls=None):
    def download(repo_id, filename, revision, repo_type, local_dir, token):
        if calls is not None:
            calls.append(filename)
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((content or FILES)[filename])
        return str(path)
    return download


@pytest.fixture
def release(monkeypatch):
    record = _record()
    monkeypatch.setattr(model_zoo, "BUNDLED_REMOTE_MODELS",
                        tuple(model_zoo.BUNDLED_REMOTE_MODELS) + (record,))
    monkeypatch.setattr(model_zoo, "_TIMEPOSE_VERIFIED", {})
    monkeypatch.setattr(model_zoo, "_TIMEPOSE_MODELS", {})
    monkeypatch.setattr(model_zoo, "_timepose_token", lambda token=None: "hf_x")
    yield record
    for name in list(sys.modules):
        if name.startswith("_spacr_timepose_288") or name.startswith("_spacr_bad"):
            sys.modules.pop(name)


def test_fetch_of_a_timepose_entry_returns_the_weights_file(release, tmp_path, monkeypatch):
    real = model_zoo._timepose_files

    def files(key, dest=None, **kwargs):
        return real(key, dest=dest, downloader=_writer(), **kwargs)
    monkeypatch.setattr(model_zoo, "_timepose_files", files)
    entry = types.SimpleNamespace(kind="timepose", key=release["key"])
    path = model_zoo.fetch(entry, tmp_path)
    assert path.name == "w.pt"
    assert path.read_bytes() == b"weights"
    assert path.parent.parent.parent == tmp_path / "timepose" / release["key"]


def test_an_unknown_release_key_is_refused():
    with pytest.raises(model_zoo.ModelZooError) as caught:
        model_zoo._timepose_record("no_such_release_288")
    assert "no_such_release_288" in str(caught.value)


def test_a_missing_file_is_not_verified_and_a_stale_stamp_is_rehashed(release, tmp_path):
    path = tmp_path / "f.bin"
    assert model_zoo._timepose_verified(path, "0" * 64) is False
    path.write_bytes(b"abc")
    sha = _sha(b"abc")
    assert model_zoo._timepose_verified(path, sha) is True
    assert model_zoo._TIMEPOSE_VERIFIED[str(path)]
    assert model_zoo._timepose_verified(path, sha) is True
    path.write_bytes(b"abcdef")
    assert model_zoo._timepose_verified(path, sha) is False


def test_access_failures_are_recognised_by_name_or_status():
    for name in ("GatedRepoError", "RepositoryNotFoundError", "LocalTokenNotFoundError"):
        assert model_zoo._timepose_access_failure(type(name, (Exception,), {})("x"))
    error = Exception("x")
    error.response = types.SimpleNamespace(status_code=403)
    assert model_zoo._timepose_access_failure(error)
    error.response = types.SimpleNamespace(status_code=500)
    assert not model_zoo._timepose_access_failure(error)
    assert not model_zoo._timepose_access_failure(ValueError("plain"))


def test_the_token_is_the_given_one_the_hub_one_or_the_environment(monkeypatch):
    assert model_zoo._timepose_token("given") == "given"
    hub = types.ModuleType("huggingface_hub")
    hub.get_token = lambda: "from-hub"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert model_zoo._timepose_token() == "from-hub"
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert model_zoo._timepose_token() == "from-env"
    monkeypatch.delenv("HF_TOKEN")
    assert model_zoo._timepose_token() is None


def test_a_verified_cache_downloads_nothing(release, tmp_path):
    model_zoo._timepose_files(release["key"], dest=tmp_path, downloader=_writer())
    calls = []

    def never(**kwargs):
        calls.append(kwargs)
    folder = model_zoo._timepose_files(release["key"], dest=tmp_path, downloader=never)
    assert calls == []
    assert (folder / "weights" / "w.pt").read_bytes() == b"weights"


def test_without_a_downloader_the_hub_downloader_is_used_and_progress_is_reported(
        release, tmp_path, monkeypatch):
    calls = []
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = _writer(calls=calls)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    seen = []
    folder = model_zoo._timepose_files(release["key"], dest=tmp_path,
                                       progress=lambda done, total: seen.append((done, total)))
    assert sorted(calls) == sorted(FILES)
    assert seen == [(1, 3), (2, 3), (3, 3)]
    assert (folder / "weights" / "w.pt").exists()


def test_an_error_that_is_not_an_access_error_propagates(release, tmp_path):
    def broken(**kwargs):
        raise RuntimeError("disk full")
    with pytest.raises(RuntimeError, match="disk full"):
        model_zoo._timepose_files(release["key"], dest=tmp_path, downloader=broken)


def test_a_wrong_download_that_cannot_be_deleted_still_raises_the_mismatch(
        release, tmp_path, monkeypatch):
    def refuse_unlink(self, *args, **kwargs):
        raise OSError("locked")
    monkeypatch.setattr(Path, "unlink", refuse_unlink)
    bad = dict(FILES, **{"weights/w.pt": b"other"})
    with pytest.raises(model_zoo.ChecksumMismatch):
        model_zoo._timepose_files(release["key"], dest=tmp_path, downloader=_writer(bad))


def test_the_device_follows_the_request_then_shared_accelerator(monkeypatch):
    from spacr.accelerator import Accelerator

    assert model_zoo._timepose_device("CPU") == "cpu"
    assert model_zoo._timepose_device("cuda:1") == "cuda:1"
    for kind in ('cuda', 'mps', 'cpu'):
        found = Accelerator(kind=kind, device=kind, label='Test device')
        monkeypatch.setattr('spacr.accelerator.resolve', lambda: found)
        assert model_zoo._timepose_device('auto') == kind

    def unavailable():
        raise ImportError('torch unavailable')

    monkeypatch.setattr('spacr.accelerator.resolve', unavailable)
    assert model_zoo._timepose_device() == "cpu"


def test_a_cached_model_is_returned_without_a_second_download(release, tmp_path):
    first = model_zoo._load_timepose(release["key"], device="cpu", dest=tmp_path,
                                     downloader=_writer())
    calls = []
    second = model_zoo._load_timepose(release["key"], device="cpu", dest=tmp_path,
                                      downloader=_writer(calls=calls))
    assert second[0] is first[0] and second[1] is first[1]
    assert calls == []


def test_an_adapter_that_fails_to_import_is_not_left_registered(release, tmp_path, monkeypatch):
    broken = dict(FILES, **{"fakepose288/__init__.py": b"raise ImportError('boom')\n"})
    record = dict(_record(key="timepose_288"), files={n: _sha(d) for n, d in broken.items()})
    monkeypatch.setattr(model_zoo, "BUNDLED_REMOTE_MODELS", (record,))
    alias = f"_spacr_{record['key']}_{record['revision'][:12]}"
    with pytest.raises(ImportError, match="boom"):
        model_zoo._load_timepose(record["key"], device="cpu", dest=tmp_path,
                                 downloader=_writer(broken))
    assert alias not in sys.modules


def test_an_adapter_with_another_contract_is_refused(release, tmp_path, monkeypatch):
    record = dict(_record(key="timepose_288"), contract="spacr.timepose/9")
    monkeypatch.setattr(model_zoo, "BUNDLED_REMOTE_MODELS", (record,))
    with pytest.raises(model_zoo.ModelZooError) as caught:
        model_zoo._load_timepose(record["key"], device="cpu", dest=tmp_path,
                                 downloader=_writer())
    assert "spacr.timepose/9" in str(caught.value)
    assert not model_zoo._TIMEPOSE_MODELS


def test_an_entry_that_names_a_package_only_uses_the_package_as_the_adapter(
        release, tmp_path, monkeypatch):
    files = dict(FILES, **{"fakepose288/__init__.py": (
        b"CONTRACT = 'spacr.timepose/1'\n"
        b"def load_model(folder, device):\n    return 'm'\n")})
    record = dict(_record(key="timepose_288", entry="fakepose288"),
                  files={n: _sha(d) for n, d in files.items()})
    monkeypatch.setattr(model_zoo, "BUNDLED_REMOTE_MODELS", (record,))
    adapter, model = model_zoo._load_timepose(record["key"], device="cpu", dest=tmp_path,
                                              downloader=_writer(files))
    assert model == "m"
    assert adapter.CONTRACT == "spacr.timepose/1"
