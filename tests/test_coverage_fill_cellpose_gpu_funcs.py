"""Coverage-fill for spacr_cellpose's Cellpose-model functions, driven
with a MOCKED CellposeModel so every branch runs on CPU deterministically
(no GPU, no real weights).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import spacr_cellpose as SC


# ---------------------------------------------------------------------------
# Mock CellposeModel
# ---------------------------------------------------------------------------

class _FakeModel:
    """Stand-in for cellpose.models.CellposeModel."""
    def __init__(self, *a, **k):
        self.pretrained_model = k.get("pretrained_model", "fake")

    def eval(self, x=None, **kwargs):
        # Return the 4-tuple shape (mask, flows, styles, diams).
        arr = np.asarray(x)
        h, w = arr.shape[:2] if arr.ndim >= 2 else (8, 8)
        mask = np.zeros((h, w), dtype=np.uint16)
        mask[2:5, 2:5] = 1
        flows = [np.zeros((h, w, 3), dtype=np.float32),
                 np.zeros((3, h, w), dtype=np.float32),
                 np.zeros((h, w), dtype=np.float32),
                 np.zeros((h, w), dtype=np.float32)]
        return mask, flows, None, None


@pytest.fixture
def _mock_cellpose(monkeypatch):
    import types
    fake_models = types.SimpleNamespace(CellposeModel=_FakeModel)
    monkeypatch.setattr(SC, "cp_models", fake_models)
    # Silence display() + heavy plot helper.
    monkeypatch.setattr(SC, "display", lambda *a, **k: None, raising=False)
    yield


def _make_img_dir(tmp_path: Path, n=2, channels=3, size=16) -> Path:
    import tifffile
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 2000, size=(size, size, channels)
                            ).astype(np.uint16)
        tifffile.imwrite(str(tmp_path / f"img_{i}.tif"), arr)
    return tmp_path


# ---------------------------------------------------------------------------
# identify_masks_finetune
# ---------------------------------------------------------------------------

class TestIdentifyMasksFinetune:
    def _settings(self, src, **over):
        s = {
            "src": str(src), "model_name": "cyto", "custom_model": None,
            "diameter": 30, "flow_threshold": 0.4, "CP_prob": 0.0,
            "grayscale": False, "save": False, "normalize": True,
            "channels": [0, 1, 2], "percentiles": [2, 98], "invert": False,
            "verbose": False, "resize": False, "target_height": 16,
            "target_width": 16, "remove_background": False,
            "background": 100, "Signal_to_noise": 5, "rescale": False,
            "resample": False, "fill_in": False, "batch_size": 2,
            "plot": False,
        }
        s.update(over)
        return s

    def test_custom_model_not_found_returns(self, tmp_path, _mock_cellpose):
        s = self._settings(tmp_path, custom_model=str(tmp_path / "nope.pth"))
        # Custom model missing → early return (lines 100-102).
        assert SC.identify_masks_finetune(s) is None

    def test_no_images_returns(self, tmp_path, _mock_cellpose):
        (tmp_path / "masks").mkdir(exist_ok=True)
        s = self._settings(tmp_path)  # empty dir → no images (133-135).
        assert SC.identify_masks_finetune(s) is None

    def test_normalize_path_runs(self, tmp_path, _mock_cellpose):
        _make_img_dir(tmp_path)
        s = self._settings(tmp_path, normalize=True)
        SC.identify_masks_finetune(s)

    def test_non_normalize_resize_path(self, tmp_path, _mock_cellpose):
        _make_img_dir(tmp_path)
        s = self._settings(tmp_path, normalize=False, resize=True)
        SC.identify_masks_finetune(s)

    def test_save_and_fill_in(self, tmp_path, _mock_cellpose):
        _make_img_dir(tmp_path)
        s = self._settings(tmp_path, save=True, fill_in=True)
        SC.identify_masks_finetune(s)
        assert (tmp_path / "masks").exists()

    def test_grayscale_and_verbose(self, tmp_path, monkeypatch,
                                      _mock_cellpose):
        _make_img_dir(tmp_path)
        # verbose path calls print_mask_and_flows — stub it.
        import spacr.plot as PL
        monkeypatch.setattr(PL, "print_mask_and_flows",
                            lambda *a, **k: None)
        s = self._settings(tmp_path, grayscale=True, verbose=True)
        SC.identify_masks_finetune(s)

    @pytest.mark.parametrize("model_name", ["cyto2", "nucleus", "cyto",
                                              "other"])
    def test_channel_selection_per_model(self, tmp_path, _mock_cellpose,
                                            model_name):
        _make_img_dir(tmp_path)
        s = self._settings(tmp_path, model_name=model_name)
        SC.identify_masks_finetune(s)

    def test_custom_model_present(self, tmp_path, _mock_cellpose):
        _make_img_dir(tmp_path)
        model_file = tmp_path / "custom.pth"; model_file.write_bytes(b"x")
        s = self._settings(tmp_path, custom_model=str(model_file))
        SC.identify_masks_finetune(s)


# ---------------------------------------------------------------------------
# generate_masks_from_imgs + check_cellpose_models
# ---------------------------------------------------------------------------

class TestGenerateAndCheck:
    def test_generate_masks_from_imgs_normalize(self, tmp_path,
                                                   _mock_cellpose):
        _make_img_dir(tmp_path)
        model = _FakeModel()
        SC.generate_masks_from_imgs(
            str(tmp_path), model, "cyto", batch_size=2, diameter=30,
            cellprob_threshold=0.0, flow_threshold=0.4, grayscale=False,
            save=True, normalize=True, channels=[0, 1, 2],
            percentiles=[2, 98], invert=False, plot=False, resize=False,
            target_height=16, target_width=16, remove_background=False,
            background=100, Signal_to_noise=5, verbose=True)
        assert (tmp_path / "cyto").exists()

    def test_generate_masks_non_normalize_resize(self, tmp_path,
                                                   _mock_cellpose):
        _make_img_dir(tmp_path)
        model = _FakeModel()
        SC.generate_masks_from_imgs(
            str(tmp_path), model, "nucleus", batch_size=1, diameter=30,
            cellprob_threshold=0.0, flow_threshold=0.4, grayscale=True,
            save=False, normalize=False, channels=[0], percentiles=[2, 98],
            invert=False, plot=False, resize=True, target_height=16,
            target_width=16, remove_background=False, background=100,
            Signal_to_noise=5, verbose=False)

    def test_check_cellpose_models(self, tmp_path, _mock_cellpose):
        _make_img_dir(tmp_path)
        settings = {
            "src": str(tmp_path), "batch_size": 2, "diameter": 30,
            "CP_prob": 0.0, "flow_threshold": 0.4, "grayscale": False,
            "save": False, "normalize": True, "channels": [0, 1, 2],
            "percentiles": [2, 98], "invert": False, "plot": False,
            "resize": False, "target_height": 16, "target_width": 16,
            "remove_background": False, "background": 100,
            "Signal_to_noise": 5, "verbose": False,
        }
        SC.check_cellpose_models(settings)


class _BadEvalModel:
    def __init__(self, *a, **k): self.pretrained_model = "bad"
    def eval(self, x=None, **k):
        return (np.zeros((8, 8), dtype=np.uint16), None)  # 2-tuple → raise


class TestRemainingBranches:
    def test_eval_unexpected_length_raises_identify(self, tmp_path,
                                                       monkeypatch):
        import types
        monkeypatch.setattr(SC, "cp_models",
                            types.SimpleNamespace(CellposeModel=_BadEvalModel))
        monkeypatch.setattr(SC, "display", lambda *a, **k: None,
                            raising=False)
        _make_img_dir(tmp_path)
        s = {
            "src": str(tmp_path), "model_name": "cyto", "custom_model": None,
            "diameter": 30, "flow_threshold": 0.4, "CP_prob": 0.0,
            "grayscale": False, "save": False, "normalize": True,
            "channels": [0, 1, 2], "percentiles": [2, 98], "invert": False,
            "verbose": False, "resize": False, "target_height": 16,
            "target_width": 16, "remove_background": False, "background": 100,
            "Signal_to_noise": 5, "rescale": False, "resample": False,
            "fill_in": False, "batch_size": 2, "plot": False,
        }
        with pytest.raises(ValueError):
            SC.identify_masks_finetune(s)

    def test_eval_unexpected_length_raises_generate(self, tmp_path):
        _make_img_dir(tmp_path)
        with pytest.raises(ValueError):
            SC.generate_masks_from_imgs(
                str(tmp_path), _BadEvalModel(), "cyto", batch_size=1,
                diameter=30, cellprob_threshold=0.0, flow_threshold=0.4,
                grayscale=False, save=False, normalize=False,
                channels=[0], percentiles=[2, 98], invert=False, plot=False,
                resize=False, target_height=16, target_width=16,
                remove_background=False, background=100, Signal_to_noise=5,
                verbose=False)

    def test_no_cuda_branch(self, tmp_path, _mock_cellpose, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        _make_img_dir(tmp_path)
        s = {
            "src": str(tmp_path), "model_name": "cyto", "custom_model": None,
            "diameter": 30, "flow_threshold": 0.4, "CP_prob": 0.0,
            "grayscale": False, "save": False, "normalize": True,
            "channels": [0, 1, 2], "percentiles": [2, 98], "invert": False,
            "verbose": False, "resize": False, "target_height": 16,
            "target_width": 16, "remove_background": False, "background": 100,
            "Signal_to_noise": 5, "rescale": False, "resample": False,
            "fill_in": False, "batch_size": 2, "plot": False,
        }
        SC.identify_masks_finetune(s)

    def test_parse_per_image_ndarray_item(self):
        # A per-image ndarray flows item → f0=item, rest None (line 63).
        masks = [np.zeros((4, 4), dtype=np.int32)]
        flows = [np.zeros((4, 4), dtype=np.float32)]  # ndarray item
        out = SC.parse_cellpose4_output((masks, flows))
        assert out[1][0] is not None and out[2] == [None]

    def test_identify_verbose_resize(self, tmp_path, _mock_cellpose,
                                       monkeypatch):
        import spacr.plot as PL
        monkeypatch.setattr(PL, "print_mask_and_flows", lambda *a, **k: None)
        _make_img_dir(tmp_path)
        s = {
            "src": str(tmp_path), "model_name": "cyto", "custom_model": None,
            "diameter": 30, "flow_threshold": 0.4, "CP_prob": 0.0,
            "grayscale": False, "save": False, "normalize": True,
            "channels": [0, 1, 2], "percentiles": [2, 98], "invert": False,
            "verbose": True, "resize": True, "target_height": 16,
            "target_width": 16, "remove_background": False, "background": 100,
            "Signal_to_noise": 5, "rescale": False, "resample": False,
            "fill_in": False, "batch_size": 2, "plot": False,
        }
        SC.identify_masks_finetune(s)

    def test_generate_plot_resize(self, tmp_path, _mock_cellpose,
                                    monkeypatch):
        import spacr.plot as PL
        monkeypatch.setattr(PL, "print_mask_and_flows", lambda *a, **k: None)
        _make_img_dir(tmp_path)
        SC.generate_masks_from_imgs(
            str(tmp_path), _FakeModel(), "cyto", batch_size=1, diameter=30,
            cellprob_threshold=0.0, flow_threshold=0.4, grayscale=False,
            save=False, normalize=True, channels=[0, 1, 2],
            percentiles=[2, 98], invert=False, plot=True, resize=True,
            target_height=16, target_width=16, remove_background=False,
            background=100, Signal_to_noise=5, verbose=False)


def test_ipython_display_fallback_on_import(monkeypatch):
    # Force IPython.display import to fail → the no-op display fallback
    # branch runs at module import (spacr_cellpose lines 10-13).
    import builtins, importlib, sys
    real_import = builtins.__import__
    def _block(name, *a, **k):
        if name == "IPython.display" or name == "IPython":
            raise ImportError("blocked")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", _block)
    sys.modules.pop("spacr.spacr_cellpose", None)
    mod = importlib.import_module("spacr.spacr_cellpose")
    assert callable(mod.display)   # the fallback no-op
    # Restore the normally-imported module for other tests.
    monkeypatch.undo()
    sys.modules.pop("spacr.spacr_cellpose", None)
    importlib.import_module("spacr.spacr_cellpose")
