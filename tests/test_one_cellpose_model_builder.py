"""Merge 5: one Cellpose model builder, one governed OPS eval, one diameter rule."""
from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from spacr import accelerator as acc
from tests.test_cellpose_api_contract import contract_cellpose  # noqa: F401

PACKAGE = Path(__file__).resolve().parents[1] / "spacr"

#: file -> functions that must build their Cellpose model through the builder.
BUILDER_CALLERS = {
    "utils.py": ("_choose_model",),
    "object.py": ("_mask_model_setup", "_robustness_segmenter"),
    "pipeline_v2.py": ("stream_masks_from_stack",),
    "spacr_cellpose.py": ("identify_masks_finetune",),
    "segmentation_uncertainty.py": ("_make_segmenter",),
    "ops_large_sites.py": ("preset_models",),
    "ops_engine.py": ("_cellpose_model",),
    "qt/screens/make_masks.py": ("load_cellpose_model",),
    "qt/widgets/preview_contract.py": ("preview_cellpose_model",),
}


def _function(path, name):
    tree = ast.parse((PACKAGE / path).read_text(encoding="utf-8"))
    return next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == name)


@pytest.mark.parametrize("path,name", [(p, n) for p, names in BUILDER_CALLERS.items() for n in names])
def test_every_caller_builds_through_the_shared_builder(path, name):
    body = _function(path, name)
    calls = {getattr(n.func, "id", getattr(n.func, "attr", None))
             for n in ast.walk(body) if isinstance(n, ast.Call)}
    assert "build_cellpose_model" in calls
    assert "CellposeModel" not in calls


def test_ops_large_sites_and_the_ops_diameters_use_the_shared_routes():
    for name in ("_eval_labels", "_eval_flows"):
        calls = {getattr(n.func, "id", getattr(n.func, "attr", None))
                 for n in ast.walk(_function("ops_large_sites.py", name)) if isinstance(n, ast.Call)}
        assert "_ops_mask_eval" in calls and "_governed_eval" not in calls
    source = (PACKAGE / "ops_engine.py").read_text(encoding="utf-8")
    assert source.count("_ops_diameter(") >= 4
    assert 'float(given["diameter"])' not in source


class _Recorder:
    """Stands in for ``CellposeModel`` and keeps what it was built with."""

    built = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).built.append(kwargs)


@pytest.fixture
def recorder(monkeypatch):
    import cellpose.models as cp_models
    from spacr import utils

    _Recorder.built = []
    monkeypatch.setattr(cp_models, "CellposeModel", _Recorder)
    monkeypatch.setattr(utils, "cp_models", cp_models)
    return _Recorder


@pytest.mark.parametrize("precision,expected", [("auto", None), ("bf16", True), ("fp32", False)])
def test_precision_is_honoured_by_ops_and_the_large_site_preset(
        recorder, monkeypatch, precision, expected):
    from spacr.ops_large_sites import preset_models
    from spacr.ops_engine import _cellpose_model

    monkeypatch.setattr(acc, "_CACHED", acc.Accelerator(kind="cuda", device="cuda:0", label="GPU"))
    settings = {"mask_inference_precision": precision}
    _cellpose_model(settings, True)
    preset_models(settings, gpu=True)
    assert [k.get("use_bfloat16") for k in recorder.built] == [expected] * 3


def test_a_pinned_model_file_is_loaded_by_the_preset(recorder, monkeypatch, tmp_path):
    from spacr import model_zoo
    from spacr.ops_large_sites import preset_models

    pinned = tmp_path / "nuclei.pt"
    pinned.write_bytes(b"x")
    monkeypatch.setattr(model_zoo, "_resolve_pinned_model_file",
                        lambda settings, key, kinds=(): pinned if key == "seg_nucleus_model" else None)
    preset_models({"seg_nucleus_model": "cpsam", "cell_model": "cpsam"})
    assert recorder.built[0]["pretrained_model"] == str(pinned)
    assert "pretrained_model" not in recorder.built[1]


def test_ops_resolves_names_and_refuses_an_unknown_one(recorder):
    from spacr.ops_engine import _cellpose_model
    from spacr.model_zoo import ModelZooError

    _cellpose_model({"ops_sbs_nucleus_model": "cyto3"}, False)
    built = dict(recorder.built[-1])
    assert built.pop("pretrained_model").endswith("cpsam_v2")
    assert built == {"gpu": False, "use_bfloat16": False}
    with pytest.raises(ModelZooError):
        _cellpose_model({"ops_sbs_nucleus_model": "no_such_model_anywhere"}, False)


def test_the_session_cache_is_keyed_by_the_resolved_model_and_kwargs(recorder):
    from spacr.utils import build_cellpose_model

    store = {}

    def cache(identity, factory):
        key = repr(identity)
        if key not in store:
            store[key] = factory()
        return store[key]

    one = build_cellpose_model("cpsam", cache=cache, gpu=False)
    two = build_cellpose_model("cyto3", cache=cache, gpu=False)
    assert one is two and len(recorder.built) == 1


@pytest.mark.parametrize("value,expected", [
    (None, {}), ("", {}), (0, {}), ("0", {}), ("auto", {}), ("Auto", {}),
    (30, {"diameter": 30.0}), ("45.5", {"diameter": 45.5})])
def test_ops_diameters_follow_the_shared_rule(value, expected):
    from spacr.ops_engine import _ops_diameter

    assert _ops_diameter(value, "nucleus") == expected


def test_ops_diameter_is_divided_by_the_downscale():
    from spacr.ops_engine import _ops_diameter

    assert _ops_diameter("40", "cell", 2) == {"diameter": 20.0}


def test_large_site_evaluation_goes_through_the_ops_governed_eval(monkeypatch):
    from spacr import ops_engine, ops_large_sites as ols

    seen = []

    def fake(model, image, settings, *, site, **extra):
        seen.append((site, extra))
        return np.ones((4, 4), np.int32), [None, np.zeros((2, 4, 4)), np.zeros((4, 4))]

    monkeypatch.setattr(ops_engine, "_ops_mask_eval", fake)
    labels = ols._eval_labels(object(), np.zeros((4, 4)), {}, 12.5)
    flows = ols._eval_flows(object(), np.zeros((4, 4)), {})
    assert labels.dtype == np.int32 and flows[0].shape == (2, 4, 4)
    assert seen[0] == ("ops_large_sites", {"normalize": False, "diameter": 12.5})
    assert seen[1] == ("ops_large_sites", {"normalize": False, "compute_masks": False})
    assert ops_engine._ops_tile_batch({"mask_eval_batch_size": 3}, "ops_large_sites", 1) == 8


def _field():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:128, :128]
    image = np.zeros((128, 128), np.float32)
    for cy, cx in ((32, 32), (32, 92), (92, 60)):
        image += 800 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 8.0 ** 2))
    return image + rng.normal(0, 8, image.shape).astype(np.float32)


@pytest.mark.skipif(not os.path.isfile(os.path.expanduser("~/.cellpose/models/cpsam")),
                    reason="the cpsam weights are not cached")
def test_masks_equal_the_hand_built_model_on_a_test_field(monkeypatch):
    from cellpose import models as cp_models
    from spacr.ops_engine import _cellpose_model

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    image = _field()

    def segment(model):
        torch.manual_seed(0)
        return np.asarray(model.eval(image, diameter=20.0, batch_size=1)[0])

    shared_model = _cellpose_model({}, False)
    hand_kwargs = {"gpu": False, "use_bfloat16": False}
    pinned = getattr(shared_model, "pretrained_model", None)
    if pinned is not None and str(pinned).endswith("cpsam_v2"):
        hand_kwargs["pretrained_model"] = str(pinned)
    hand = segment(cp_models.CellposeModel(**hand_kwargs))
    shared = segment(shared_model)
    assert hand.max() >= 1
    np.testing.assert_array_equal(shared, hand)
