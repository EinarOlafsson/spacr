"""Headless catalogue selection downloads and reuses checked model bytes."""

import hashlib
from pathlib import Path

import pytest

from spacr import model_zoo as zoo


@pytest.fixture
def remote_model(tmp_path, monkeypatch):
    payload = b"test checkpoint bytes"
    entry = zoo.ModelEntry(
        key="remote_cells", name="remote_cells.CP_model", kind="cellpose",
        uri="https://models.example.test/cells", source="community",
        sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(zoo, "catalogue", lambda **_kwargs: [entry])
    monkeypatch.setattr(zoo, "_spacr_home", lambda: tmp_path)
    monkeypatch.setattr(zoo, "_on_the_qt_gui_thread", lambda: False)
    requests = []

    def open_bytes(uri, **_kwargs):
        requests.append(uri)
        return iter([payload]), len(payload)

    monkeypatch.setattr(zoo, "open_uri", open_bytes)
    return entry, payload, requests


def test_missing_named_model_downloads_and_second_use_is_cached(remote_model):
    entry, payload, requests = remote_model
    first = zoo._ensure_model_file(entry.key, kinds=("cellpose",))
    assert first.read_bytes() == payload
    assert zoo._ensure_model_file(entry.key, kinds=("cellpose",)) == first
    assert requests == [entry.uri]


def test_corrupt_cache_is_preserved_and_replacement_is_reused(
        remote_model, tmp_path):
    entry, payload, requests = remote_model
    original = tmp_path / "models" / entry.name
    original.parent.mkdir()
    original.write_bytes(b"corrupted cached checkpoint")
    replacement = zoo._ensure_model_file(entry.key, kinds=("cellpose",))
    assert replacement != original
    assert replacement.read_bytes() == payload
    assert original.read_bytes() == b"corrupted cached checkpoint"
    assert zoo._ensure_model_file(entry.key, kinds=("cellpose",)) == replacement
    assert requests == [entry.uri]


def test_preflight_recognizes_downloadable_model_without_fetching(remote_model):
    entry, _payload, requests = remote_model
    expected = zoo._ensure_model_file(entry.key, kinds=("cellpose",), download=False)
    assert expected.name == entry.name
    assert not expected.exists()
    assert requests == []
    from spacr.validate import _check_required_paths
    assert not [problem for problem in _check_required_paths(
        {"custom_model": entry.key}, "mask") if problem.setting == "custom_model"]


def test_wrong_model_kind_never_downloads(remote_model):
    entry, _payload, requests = remote_model
    with pytest.raises(zoo.ModelZooError, match="requires.*classifier"):
        zoo._ensure_model_file(entry.key, kinds=("classifier",))
    assert requests == []


def test_missing_local_path_is_not_replaced_by_an_unrelated_model(remote_model):
    _entry, _payload, requests = remote_model
    assert zoo._ensure_model_file("/missing/remote_cells.CP_model") is None
    assert zoo._ensure_model_file("unknown_model") is None
    assert requests == []


def test_bad_download_still_fails_checksum_and_leaves_no_checkpoint(
        remote_model, monkeypatch, tmp_path):
    entry, _payload, _requests = remote_model
    monkeypatch.setattr(zoo, "open_uri",
                        lambda *_args, **_kwargs: (iter([b"wrong bytes"]), 11))
    with pytest.raises(zoo.ChecksumMismatch):
        zoo._ensure_model_file(entry.key, kinds=("cellpose",))
    assert list((tmp_path / "models").iterdir()) == []


def test_gui_thread_never_starts_a_missing_checkpoint_download(
        remote_model, monkeypatch):
    entry, _payload, requests = remote_model
    monkeypatch.setattr(zoo, "_on_the_qt_gui_thread", lambda: True)
    with pytest.raises(zoo.ModelUnreadable, match="Model Zoo"):
        zoo._ensure_model_file(entry.key, kinds=("cellpose",))
    assert requests == []


def test_cellpose_notebook_and_mask_run_resolver_accept_remote_key(remote_model):
    entry, payload, requests = remote_model
    from spacr.utils import _resolve_cellpose_pretrained
    path = _resolve_cellpose_pretrained(entry.key, object_type="cell")
    assert Path(path).read_bytes() == payload
    assert _resolve_cellpose_pretrained(entry.name) == path
    assert requests == [entry.uri]


def test_cellpose3_worker_resolver_accepts_remote_key(remote_model, monkeypatch):
    entry, payload, requests = remote_model
    from dataclasses import replace
    from spacr._segmentation_backends import _cellpose3_model
    converted = replace(entry, kind="cellpose3")
    monkeypatch.setattr(zoo, "catalogue", lambda **_kwargs: [converted])
    path = _cellpose3_model(entry.key, "nucleus")
    assert Path(path).read_bytes() == payload
    assert requests == [entry.uri]


def test_cellpose_dino_worker_accepts_remote_key(remote_model, monkeypatch):
    entry, payload, requests = remote_model
    from dataclasses import replace
    from spacr._segmentation_backends import _cellpose_dino_model
    monkeypatch.setattr(zoo, "catalogue", lambda **_kwargs: [
        replace(entry, kind="cellpose_dino")])
    assert Path(_cellpose_dino_model(entry.key)).read_bytes() == payload
    assert requests == [entry.uri]


def test_prefixed_model_validation_does_not_download(remote_model, monkeypatch):
    entry, payload, requests = remote_model
    from dataclasses import replace
    from spacr._segmentation_backends import _prefixed_model, _prefixed_model_ok
    monkeypatch.setattr(zoo, "catalogue", lambda **_kwargs: [
        replace(entry, kind="stardist")])
    assert _prefixed_model_ok("stardist:" + entry.key)
    assert requests == []
    assert Path(_prefixed_model("stardist", entry.key)).read_bytes() == payload
    assert requests == [entry.uri]


def test_classifier_artifact_loader_accepts_remote_key(remote_model, monkeypatch):
    from dataclasses import replace
    import torch
    from spacr.torch_artifacts import load_model_artifact
    entry, payload, requests = remote_model
    converted = replace(entry, kind="classifier")
    monkeypatch.setattr(zoo, "catalogue", lambda **_kwargs: [converted])
    expected = torch.nn.Linear(2, 1)
    paths = []

    def load(path, **_kwargs):
        paths.append(Path(path))
        return expected

    monkeypatch.setattr(torch, "load", load)
    loaded, _metadata = load_model_artifact(entry.key)
    assert loaded is expected
    assert paths[0].read_bytes() == payload
    assert requests == [entry.uri]


@pytest.mark.parametrize("object_type", ["cell", "invalid"])
def test_missing_real_object_classifier_downloads_then_uses_normal_bundle_validation(
        remote_model, monkeypatch, object_type):
    import io
    import joblib
    from dataclasses import replace
    from spacr.object_classifier import _real_classifier_bundles

    entry, _payload, requests = remote_model
    bundle = {"object_type": object_type, "roles": ["cell"]}
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer)
    payload = buffer.getvalue()
    entry = replace(entry, kind="classifier", name="real_cells.joblib",
                    sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(zoo, "catalogue", lambda **_kwargs: [entry])

    def open_bundle(uri, **_kwargs):
        requests.append(uri)
        return iter([payload]), len(payload)

    monkeypatch.setattr(zoo, "open_uri", open_bundle)
    if object_type == "cell":
        assert _real_classifier_bundles(entry.key) == {"cell": bundle}
        assert _real_classifier_bundles(entry.key) == {"cell": bundle}
    else:
        with pytest.raises(ValueError, match="not a real / not-real classifier"):
            _real_classifier_bundles(entry.key)
    assert requests == [entry.uri]


def test_cli_lists_models_that_are_not_downloaded(remote_model, capsys):
    from spacr.cli import main
    entry, _payload, requests = remote_model
    assert main(["--list-models"]) == 0
    assert entry.key in capsys.readouterr().out
    assert requests == []
