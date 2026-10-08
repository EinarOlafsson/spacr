"""A catalogue key must resolve to the intended local model without fetching."""
from __future__ import annotations

import hashlib

import pytest

from spacr import model_zoo, validate


def _entry(key, name, *, kind="classifier", path="", uri="", sha256=""):
    return model_zoo.ModelEntry(
        key=key, name=name, kind=kind, source="remote", path=path,
        uri=uri, sha256=sha256,
    )


def test_explicit_local_paths_win_and_blank_requests_do_not_fetch(
        tmp_path, monkeypatch):
    local = tmp_path / "trained.pth"
    local.write_bytes(b"local model")
    monkeypatch.setattr(model_zoo, "catalogue", lambda **_kw: pytest.fail(
        "local model resolution queried the remote catalogue"))
    assert model_zoo._ensure_model_file("", download=False) is None
    assert model_zoo._ensure_model_file(str(local), download=False) == local


def test_catalogue_local_copy_and_metadata_without_uri_are_truthful(
        tmp_path, monkeypatch):
    local = tmp_path / "already-downloaded.pth"
    local.write_bytes(b"verified on this machine")
    entries = (
        _entry("present", local.name, path=str(local)),
        _entry("not-published", "missing.pth"),
    )
    monkeypatch.setattr(model_zoo, "catalogue", lambda **_kw: entries)
    assert model_zoo._ensure_model_file("present", download=False) == local
    assert model_zoo._ensure_model_file("not-published", download=False) is None


def test_versioned_cache_refuses_wrong_bytes_and_keeps_searching(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    destination = home / "models"
    destination.mkdir(parents=True)
    wrong = destination / "classifier_v2.pth"
    right = destination / "classifier_v3.pth"
    wrong.write_bytes(b"different model")
    right.write_bytes(b"the published model")
    digest = hashlib.sha256(right.read_bytes()).hexdigest()
    entry = _entry("classifier-key", "classifier.pth", uri="file:///unused",
                   sha256=digest)
    monkeypatch.setattr(model_zoo, "catalogue", lambda **_kw: (entry,))
    monkeypatch.setattr(model_zoo, "_spacr_home", lambda: home)
    assert model_zoo._ensure_model_file("classifier-key", download=False) == right
    right.write_bytes(b"changed after publication")
    assert model_zoo._ensure_model_file("classifier-key", download=False) == (
        destination / "classifier.pth")


def test_validation_accepts_only_a_correct_kind_catalogue_key_without_fetch(
        tmp_path, monkeypatch):
    entries = (
        _entry("trained-classifier", "classifier.pth", kind="classifier",
               uri="file:///unused", sha256="a" * 64),
        _entry("trained-cellpose", "cellpose.pth", kind="cellpose",
               uri="file:///unused", sha256="b" * 64),
    )
    monkeypatch.setattr(model_zoo, "catalogue", lambda **_kw: entries)
    monkeypatch.setattr(model_zoo, "_spacr_home", lambda: tmp_path)
    monkeypatch.setattr(model_zoo, "fetch", lambda *_a, **_k: pytest.fail(
        "validation downloaded checkpoint bytes"))

    correct = validate._check_required_paths(
        {"apply_model_to_dataset": True,
         "model_path": "trained-classifier",
         "custom_model": "trained-cellpose"}, "classify")
    assert not [p for p in correct if p.setting in ("model_path", "custom_model")]

    wrong = validate._check_required_paths(
        {"apply_model_to_dataset": True,
         "model_path": "trained-cellpose",
         "custom_model": "trained-classifier"}, "classify")
    assert {p.setting for p in wrong if p.is_error} == {
        "model_path", "custom_model"}

    unknown = validate._check_required_paths(
        {"apply_model_to_dataset": True, "model_path": "not-a-zoo-model"},
        "classify")
    assert {p.setting for p in unknown if p.is_error} == {"model_path"}


def test_unknown_artifact_key_stays_a_missing_file_error(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from spacr import torch_artifacts

    missing = tmp_path / "unknown-checkpoint.pth"
    monkeypatch.setattr(model_zoo, "_ensure_model_file", lambda *_a, **_k: None)
    with pytest.raises(FileNotFoundError, match="unknown-checkpoint.pth"):
        torch_artifacts.load_model_artifact(str(missing), map_location="cpu")
    assert not missing.exists()
