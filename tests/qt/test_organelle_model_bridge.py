"""The curation bridge preserves the real CPU model and ledger contracts."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from spacr.qt import organelle_modes as om


@pytest.fixture
def cpu_model(monkeypatch):
    """A deterministic logit model; no pretrained accuracy is implied."""
    torch = pytest.importorskip("torch")
    from spacr import accelerator

    monkeypatch.setattr(accelerator, "torch_device", lambda: torch.device("cpu"))
    model = torch.nn.Conv2d(1, 1, kernel_size=1).cpu()
    with torch.no_grad():
        model.weight.fill_(1)
        model.bias.zero_()
    return model.eval()


@pytest.fixture
def field():
    image = np.zeros((64, 64), dtype=np.uint16)
    image[12:21, 14:25] = 100
    image[45, 45] = 100
    return image


@pytest.mark.parametrize("source", ["supplied", "checkpoint"])
def test_unet_bridge_runs_real_cpu_model_and_preserves_input(tmp_path, cpu_model, field, source):
    import torch

    original = field.copy()
    path = tmp_path / "user-model.pt"
    model = cpu_model
    digest = None
    if source == "checkpoint":
        torch.save(cpu_model, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        model = None
    params = om.DEFAULT_PARAMS._replace(unet_model_path=str(path))
    labels = om.segment(field, "unet", params, min_area=5, model=model)
    expected = np.zeros(field.shape, dtype=np.int32)
    expected[12:21, 14:25] = 1
    np.testing.assert_array_equal(labels, expected)
    np.testing.assert_array_equal(field, original)
    assert np.issubdtype(labels.dtype, np.integer)
    assert next(cpu_model.parameters()).device.type == "cpu"
    if digest is not None:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    else:
        assert not path.exists(), "an explicitly supplied model needs no checkpoint"


def test_unet_probability_cutoff_and_empty_field_are_respected(cpu_model, field):
    params = om.DEFAULT_PARAMS._replace(unet_threshold=0.999999)
    assert not om.segment(field, "unet", params, model=cpu_model).any()
    assert not om.segment(np.zeros_like(field), "unet", om.DEFAULT_PARAMS,
                          model=cpu_model).any()


def test_unet_network_skeleton_is_thinner_and_retains_the_large_component(cpu_model, field):
    params = om.DEFAULT_PARAMS._replace(skeletonize=True)
    labels = om.segment(field, "unet", params, min_area=5, model=cpu_model)
    assert labels.shape == field.shape
    assert labels.max() == 1
    assert 0 < np.count_nonzero(labels) < 99
    assert labels[16, 19] == 1
    assert labels[45, 45] == 0


def test_unet_missing_model_reports_the_path_without_touching_pixels(tmp_path, field):
    path = tmp_path / "missing.pt"
    original = field.copy()
    params = om.DEFAULT_PARAMS._replace(unet_model_path=str(path))
    with pytest.raises(ValueError, match="organelle_unet_model_path") as caught:
        om.segment(field, "unet", params)
    assert str(path) in str(caught.value)
    assert not path.exists()
    np.testing.assert_array_equal(field, original)


def test_method_ledger_records_only_relevant_settings_as_json():
    params = om.DEFAULT_PARAMS._replace(
        ridge_sigmas=(0.5, 1.5, 3.5), ridge_filter="sato",
        log_threshold=0.35, unet_model_path="models/local.pt", unet_threshold=0.7)
    ridge = om.provenance("ridge", params)
    assert ridge["ridge_sigmas"] == [0.5, 1.5, 3.5]
    assert ridge["ridge_filter"] == "sato"
    assert "log_threshold" not in ridge and "unet_model_path" not in ridge
    assert json.loads(json.dumps(ridge)) == ridge
    unet = om.provenance("unet", params)
    assert unet == {"unet_model_path": "models/local.pt", "unet_threshold": 0.7,
                    "skeletonize": False}
    assert om.provenance("otsu", params) == {}
