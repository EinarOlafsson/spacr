"""Timepose release plumbing in the model zoo: fetch, token, files, device, loader edges.

A fake release is registered and the download is replaced by a fake writer, so
no token, network or GPU is used.
"""
from __future__ import annotations

import hashlib
import sys
import types

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


