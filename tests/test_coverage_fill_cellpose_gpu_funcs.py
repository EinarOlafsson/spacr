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

_MISSING = object()


def _validate_eval_call(x, channel_axis):
    """Reject anything the real ``CellposeModel.eval`` would reject.

    The mock used to be ``def eval(self, x=None, **kwargs)``, which swallowed
    ``channel_axis`` whole. spaCR passed the hard-coded ``channel_axis=3``
    that Cellpose 4 cannot accept for any shape spaCR produces, and 15 tests
    sailed over it while every real run raised.

    Rather than re-assert a value (which drifts from the library), hand the
    exact ``(x, channel_axis)`` pair to the real validator Cellpose 4 uses:
    ``CellposeModel.eval`` calls ``transforms.convert_image(x,
    channel_axis=channel_axis, ...)``. It is pure numpy, CPU-only and
    offline. If spaCR ever passes an axis Cellpose rejects, these tests fail
    with the production error.
    """
    assert channel_axis is not _MISSING, (
        "model.eval() was called without channel_axis; spaCR must pass it "
        "explicitly so the value is covered by this contract."
    )
    from cellpose import transforms
    return transforms.convert_image(np.asarray(x), channel_axis=channel_axis)


class _CellposeRecorder:
    """Everything the fake ``CellposeModel`` saw, across every instance.

    ``identify_masks_finetune`` and ``check_cellpose_models`` build their own
    model, so a test has no reference to hold. The fixture installs one of
    these on :class:`_FakeModel` and every instance registers itself here, which
    is what lets a test assert WHICH checkpoint was loaded, on which device, and
    what pixels reached the network.
    """

    def __init__(self):
        self.models = []
        self.displayed = []

    @property
    def eval_calls(self):
        """Every ``eval()`` call, in call order, across every model."""
        return [call for model in self.models for call in model.eval_calls]

    @property
    def pretrained(self):
        """``pretrained_model=`` of each constructed model, in order."""
        return [m.init_kwargs.get("pretrained_model") for m in self.models]


class _FakeModel:
    """Stand-in for cellpose.models.CellposeModel."""

    #: Recorder installed by the ``_mock_cellpose`` fixture. None outside a
    #: test, so a stray construction cannot leak into the next one.
    recorder = None

    def __init__(self, *a, **k):
        self.init_args = a
        self.init_kwargs = dict(k)
        self.pretrained_model = k.get("pretrained_model", "fake")
        self.eval_calls = []
        if _FakeModel.recorder is not None:
            _FakeModel.recorder.models.append(self)

    def eval(self, x=None, channel_axis=_MISSING, **kwargs):
        converted = _validate_eval_call(x, channel_axis)
        image = np.asarray(x)
        # Keep the pixels, not just the shape: normalize=True/False and
        # resize=True/False are only distinguishable from what was handed over.
        self.eval_calls.append({"shape": image.shape, "dtype": image.dtype,
                                "image": image.copy(),
                                "channel_axis": channel_axis, **kwargs})
        # Return the Cellpose 4 shape (mask, flows, styles).
        h, w = converted.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint16)
        mask[2:5, 2:5] = 1
        flows = [np.zeros((h, w, 3), dtype=np.float32),
                 np.zeros((3, h, w), dtype=np.float32),
                 np.zeros((h, w), dtype=np.float32),
                 np.zeros((h, w), dtype=np.float32)]
        return mask, flows, None


@pytest.fixture
def _mock_cellpose(monkeypatch):
    """Install the fake CellposeModel; yields the call recorder."""
    import types
    recorder = _CellposeRecorder()
    monkeypatch.setattr(_FakeModel, "recorder", recorder)
    fake_models = types.SimpleNamespace(CellposeModel=_FakeModel)
    monkeypatch.setattr(SC, "cp_models", fake_models)
    # display() is the notebook hook check_cellpose_models pushes its settings
    # table through — record it rather than dropping it on the floor.
    monkeypatch.setattr(SC, "display",
                        lambda *a, **k: recorder.displayed.append(a),
                        raising=False)
    yield recorder


def _make_img_dir(tmp_path: Path, n=2, channels=3, size=16) -> Path:
    import tifffile
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 2000, size=(size, size, channels)
                            ).astype(np.uint16)
        tifffile.imwrite(str(tmp_path / f"img_{i}.tif"), arr)
    return tmp_path


# ---------------------------------------------------------------------------
# check_cellpose_models
# ---------------------------------------------------------------------------

class TestGenerateAndCheck:
    def test_check_cellpose_models(self, tmp_path, _mock_cellpose):
        """Cellpose 4 ships exactly one stock model, so the "check the models"
        sweep is a sweep of one: cpsam, and no pre-SAM cyto/nuclei folders."""
        src = _make_img_dir(tmp_path / "check")
        settings = {
            "src": str(src), "batch_size": 2, "diameter": 30,
            "CP_prob": 0.0, "flow_threshold": 0.4, "grayscale": False,
            "save": False, "normalize": True, "channels": [0, 1, 2],
            "percentiles": [2, 98], "invert": False, "plot": False,
            "resize": False, "target_height": 16, "target_width": 16,
            "remove_background": False, "background": 100,
            "Signal_to_noise": 5, "verbose": False,
        }
        SC.check_cellpose_models(settings)

        # exactly one model was built, and it is the stock checkpoint
        assert _mock_cellpose.pretrained == ["cpsam"]
        # ...whose masks go to a folder named after it, and only that one
        assert sorted(p.name for p in src.iterdir() if p.is_dir()) == ["cpsam"]
        # one eval per image, at the percentile loader's target resolution
        calls = _mock_cellpose.eval_calls
        assert [c["shape"] for c in calls] == [(16, 16, 3), (16, 16, 3)]
        assert all(c["cellprob_threshold"] == 0.0 for c in calls)
        # and the resolved settings table was pushed to display()
        (frame,), = _mock_cellpose.displayed
        assert list(frame.columns) == ["setting_key", "setting_value"]
        table = dict(frame.values)
        assert table["src"] == str(src)
        assert table["diameter"] == "30"


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
    # ...and calling it is a silent no-op (executes the fallback body).
    assert mod.display("anything", 2, key=3) is None
    # Restore the normally-imported module for other tests.
    monkeypatch.undo()
    sys.modules.pop("spacr.spacr_cellpose", None)
    importlib.import_module("spacr.spacr_cellpose")


def _settings(src, **over):
    """The identify_masks_finetune settings dict used across this module."""
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


class TestIdentifyMasksProgress938:
    """Item 938: the Progress line counts images done over images in all."""

    def test_progress_counts_files_done_over_all_files(self, tmp_path,
                                                       _mock_cellpose, capsys):
        import re
        src = _make_img_dir(tmp_path / "five", n=5)
        SC.identify_masks_finetune(_settings(src, batch_size=2))
        lines = [m.groups() for m in re.finditer(
            r"Progress: (\d+)/(\d+), operation_type: generate cellpose masks",
            capsys.readouterr().out)]
        assert [int(a) for a, _b in lines] == [1, 2, 3, 4, 5]
        assert {int(b) for _a, b in lines} == {5}
