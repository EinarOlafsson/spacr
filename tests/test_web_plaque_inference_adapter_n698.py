"""Desktop segmentation call contracts; no real checkpoint inference."""
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
import tifffile

from tools.web_mask_generator import plaque


@pytest.fixture
def job(tmp_path):
    source = tmp_path / "input.tif"
    pixels = np.arange(2 * 5 * 7, dtype=np.uint16).reshape(2, 5, 7)
    tifffile.imwrite(source, pixels, metadata={"axes": "CYX"}, photometric="minisblack")
    weights = tmp_path / "trusted.CP_model"
    weights.write_bytes(b"operator-provisioned fake checkpoint; no load")
    return tmp_path / "result.zip", dict(
        input_file=source, checkpoint=weights,
        expected_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        model_id="toxoplasma_plaque_v4", model_version="v4",
        effective_settings={"diameter": 52, "flow_threshold": 0.6, "CP_prob": -1.5})


class Model:
    def __init__(self, arrays=True, error=None):
        self.calls = []
        self.arrays = arrays
        self.error = error
    def eval(self, image, **kwargs):
        self.calls.append((image.copy(), kwargs))
        if self.error:
            raise self.error
        shape = image.shape[:2]
        mask = np.zeros(shape, dtype=np.uint32)
        mask[1:3, 2:4] = 12
        outputs = [np.zeros((*shape, 3), dtype=np.uint8), np.full((2, *shape), 0.25, dtype=np.float32), np.full(shape, -2.5, dtype=np.float32)] if self.arrays else []
        return mask, outputs, None


def test_actual_desktop_segmenter_receives_exact_plane_and_settings(job, monkeypatch):
    target, arguments = job
    model = Model()
    seen = []
    monkeypatch.setattr(plaque, "_load_model", lambda path, digest=None: seen.append(path) or model)
    assert plaque.run_plaque_download(target, **arguments) == target
    assert seen == [arguments["checkpoint"]]
    assert len(model.calls) == 1
    image, settings = model.calls[0]
    np.testing.assert_array_equal(image, np.moveaxis(tifffile.imread(arguments["input_file"]), 0, -1))
    assert settings == {"channel_axis": -1, "diameter": 52.0, "flow_threshold": 0.6, "cellprob_threshold": -1.5}
    with zipfile.ZipFile(target) as archive:
        np.testing.assert_array_equal(np.load(io.BytesIO(archive.read("flows.npy"))), np.full((2, 5, 7), 0.25, dtype=np.float32))
        assert np.load(io.BytesIO(archive.read("cell_probability.npy"))).min() == -2.5
        assert tifffile.imread(io.BytesIO(archive.read("mask.tif"))).max() == 12
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["settings"]["pipeline"] == "spacr.plaque.segment_plaque_image"
    assert manifest["settings"]["input_selection"]["channels"] == [0, 1]
    assert manifest["settings"]["diameter"] == 52
    assert manifest["model"]["weights_sha256"] == arguments["expected_sha256"]


def test_explicit_native_scale_and_missing_backend_arrays(job, monkeypatch):
    target, arguments = job
    model = Model(arrays=False)
    monkeypatch.setattr(plaque, "_load_model", lambda path, digest=None: model)
    arguments["effective_settings"] = {"diameter": None}
    plaque.run_plaque_download(target, **arguments)
    assert model.calls[0][1] == {"channel_axis": -1, "diameter": None, "flow_threshold": 0.4, "cellprob_threshold": 0.0}
    with zipfile.ZipFile(target) as archive:
        assert "flows.npy" not in archive.namelist()
        manifest = json.loads(archive.read("manifest.json"))
    assert set(manifest["unavailable"]) == {"flows", "cell_probability"}


@pytest.mark.parametrize("changes", [
    {"expected_sha256": "00" * 32}, {"expected_sha256": "x"},
    {"effective_settings": {}}, {"effective_settings": {"diameter": "auto"}},
    {"effective_settings": {"diameter": 0}}, {"effective_settings": {"diameter": -2}},
    {"effective_settings": {"diameter": float("nan")}}, {"effective_settings": {"diameter": True}},
    {"effective_settings": {"diameter": 20, "flow_threshold": -1}},
    {"effective_settings": {"diameter": 20, "normalize": False}},
])
def test_invalid_identity_and_unresolved_settings_never_load_model(job, changes, monkeypatch):
    target, arguments = job
    arguments.update(changes)
    target.write_bytes(b"old result")
    monkeypatch.setattr(plaque, "_load_model", lambda path, digest=None: pytest.fail("invalid job loaded model"))
    with pytest.raises(ValueError):
        plaque.run_plaque_download(target, **arguments)
    assert target.read_bytes() == b"old result"


@pytest.mark.parametrize("failure", ["load", "eval"])
def test_model_failures_preserve_existing_result(job, monkeypatch, failure):
    target, arguments = job
    target.write_bytes(b"old result")
    if failure == "load":
        def loader(path, digest=None):
            raise RuntimeError("checkpoint unavailable")
    else:
        loader = lambda path, digest=None: Model(error=RuntimeError("inference failed"))
    monkeypatch.setattr(plaque, "_load_model", loader)
    with pytest.raises(RuntimeError):
        plaque.run_plaque_download(target, **arguments)
    assert target.read_bytes() == b"old result"


def test_grey_input_uses_desktop_two_dimensional_channel_contract(job, monkeypatch):
    target, arguments = job
    source = arguments["input_file"]
    tifffile.imwrite(source, np.ones((5, 7), dtype=np.uint16), metadata={"axes": "YX"}, photometric="minisblack")
    model = Model()
    monkeypatch.setattr(plaque, "_load_model", lambda path, digest=None: model)
    plaque.run_plaque_download(target, **arguments)
    assert model.calls[0][0].shape == (5, 7)
    assert model.calls[0][1]["channel_axis"] is None


def test_production_loader_delegates_to_the_existing_desktop_loader(monkeypatch):
    import sys
    import types
    from pathlib import Path
    seen = []
    marker = object()
    monkeypatch.setitem(sys.modules, "spacr.submodules", types.SimpleNamespace(_plaque_cellpose_model=lambda path, digest=None: seen.append(path) or marker))
    assert plaque._load_model(Path("/operator/model.CP_model")) is marker
    assert seen == ["/operator/model.CP_model"]


def test_no_tuple_flow_output_does_not_create_placeholder_arrays(job, monkeypatch):
    target, arguments = job
    model = Model()
    original = model.eval
    model.eval = lambda *args, **kwargs: (original(*args, **kwargs)[0],)
    monkeypatch.setattr(plaque, "_load_model", lambda path, digest=None: model)
    plaque.run_plaque_download(target, **arguments)
    with zipfile.ZipFile(target) as archive:
        assert "flows.npy" not in archive.namelist()
        assert "cell_probability.npy" not in archive.namelist()


def test_excess_channels_cannot_be_silently_ignored_by_cellpose(job, monkeypatch):
    target, arguments = job
    tifffile.imwrite(arguments["input_file"], np.ones((4, 5, 7), dtype=np.uint16), metadata={"axes": "CYX"}, photometric="minisblack")
    monkeypatch.setattr(plaque, "_load_model", lambda *a: pytest.fail("excess channels loaded model"))
    with pytest.raises(ValueError, match="at most three"):
        plaque.run_plaque_download(target, **arguments)
    assert not target.exists()
