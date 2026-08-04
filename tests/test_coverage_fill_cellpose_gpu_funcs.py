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

    def reset(self):
        """Forget everything — lets one test run two contrasting passes."""
        self.models.clear()
        self.displayed.clear()

    @property
    def eval_calls(self):
        """Every ``eval()`` call, in call order, across every model."""
        return [call for model in self.models for call in model.eval_calls]

    @property
    def pretrained(self):
        """``pretrained_model=`` of each constructed model, in order."""
        return [m.init_kwargs.get("pretrained_model") for m in self.models]

    @property
    def gpu_flags(self):
        """``gpu=`` of each constructed model, in order."""
        return [m.init_kwargs.get("gpu") for m in self.models]

    @property
    def devices(self):
        """``str(device=)`` of each constructed model, in order."""
        return [str(m.init_kwargs.get("device")) for m in self.models]


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
        # Return the 4-tuple shape (mask, flows, styles, diams).
        h, w = converted.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint16)
        mask[2:5, 2:5] = 1
        flows = [np.zeros((h, w, 3), dtype=np.float32),
                 np.zeros((3, h, w), dtype=np.float32),
                 np.zeros((h, w), dtype=np.float32),
                 np.zeros((h, w), dtype=np.float32)]
        return mask, flows, None, None


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


@pytest.fixture
def _plot_calls(monkeypatch):
    """Record every ``print_mask_and_flows(stack, mask, flows)`` call.

    Both product functions do ``from .plot import print_mask_and_flows`` inside
    the function body, so replacing the module attribute is enough. Recording
    instead of no-op'ing is the point: ``verbose``/``plot`` exist to render the
    mask, and a silent stub cannot tell "rendered" from "skipped".
    """
    import spacr.plot as PL
    calls = []
    monkeypatch.setattr(
        PL, "print_mask_and_flows",
        lambda stack, mask, flows, **k: calls.append(
            {"stack": np.asarray(stack), "mask": np.asarray(mask),
             "flows": flows}))
    return calls


@pytest.fixture
def _quiet_resize_preview(monkeypatch):
    """Stop ``resize_images_and_labels(..., show_example=True)`` from drawing.

    It builds a 20x20-inch 4-panel figure per batch purely as a human preview;
    the tests below resize on purpose, so skip the render.
    """
    import spacr.plot as PL
    monkeypatch.setattr(PL, "plot_resize", lambda *a, **k: None)


def _make_img_dir(tmp_path: Path, n=2, channels=3, size=16) -> Path:
    import tifffile
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 2000, size=(size, size, channels)
                            ).astype(np.uint16)
        tifffile.imwrite(str(tmp_path / f"img_{i}.tif"), arr)
    return tmp_path


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


# ---------------------------------------------------------------------------
# identify_masks_finetune
# ---------------------------------------------------------------------------

class TestIdentifyMasksFinetune:
    def _settings(self, src, **over):
        return _settings(src, **over)

    def test_custom_model_not_found_returns(self, tmp_path, _mock_cellpose):
        s = self._settings(tmp_path, custom_model=str(tmp_path / "nope.pth"))
        # Custom model missing → early return (lines 100-102).
        assert SC.identify_masks_finetune(s) is None

    def test_no_images_returns(self, tmp_path, _mock_cellpose):
        (tmp_path / "masks").mkdir(exist_ok=True)
        s = self._settings(tmp_path)  # empty dir → no images (133-135).
        assert SC.identify_masks_finetune(s) is None

    def test_normalize_path_runs(self, tmp_path, _mock_cellpose):
        """normalize=True routes through the percentile loader.

        That loader is the only one that honours ``settings['channels']`` and
        the only one that rescales each channel to an exact 0..1 span, so both
        are asserted on the pixels the network actually received.
        """
        src = _make_img_dir(tmp_path / "norm")
        SC.identify_masks_finetune(
            self._settings(src, normalize=True, channels=[0]))

        calls = _mock_cellpose.eval_calls
        assert len(calls) == 2                      # one eval per image
        for call in calls:
            # channels=[0] leaves one channel, which spaCR squeezes away — so
            # Cellpose must be told there is NO channel axis.
            assert call["shape"] == (16, 16)
            assert call["channel_axis"] is None
            assert call["dtype"] == np.float64
            # rescale_intensity(..., out_range=(0, 1)) => exact extremes
            assert call["image"].min() == 0.0
            assert call["image"].max() == 1.0
            # spaCR normalised already; Cellpose must not do it a second time.
            assert call["normalize"] is False
            assert call["diameter"] == 30
            assert call["cellprob_threshold"] == 0.0
            assert call["flow_threshold"] == 0.4

        # Contrast: the same images through normalize=False. The raw loader
        # ignores `channels` entirely, so all three survive and a channel axis
        # comes back — none of the assertions above can be reading a constant.
        _mock_cellpose.reset()
        raw_src = _make_img_dir(tmp_path / "raw")
        SC.identify_masks_finetune(
            self._settings(raw_src, normalize=False, channels=[0]))
        raw = _mock_cellpose.eval_calls
        assert [c["shape"] for c in raw] == [(16, 16, 3), (16, 16, 3)]
        assert [c["channel_axis"] for c in raw] == [-1, -1]

    def test_non_normalize_resize_path(self, tmp_path, _mock_cellpose,
                                       _quiet_resize_preview):
        """normalize=False keeps the source resolution; resize=True shrinks it.

        The raw loader has no target_height/target_width of its own, so this is
        the only path where ``resize`` decides what size the network sees.
        """
        src = _make_img_dir(tmp_path / "resized", size=32)
        SC.identify_masks_finetune(
            self._settings(src, normalize=False, resize=True,
                           target_height=16, target_width=16))

        calls = _mock_cellpose.eval_calls
        assert len(calls) == 2
        assert [c["shape"] for c in calls] == [(16, 16, 3), (16, 16, 3)]
        assert all(c["channel_axis"] == -1 for c in calls)

        # Contrast: resize=False hands over the untouched 32x32 source, with
        # the same target_height/target_width in the settings.
        _mock_cellpose.reset()
        untouched = _make_img_dir(tmp_path / "untouched", size=32)
        SC.identify_masks_finetune(
            self._settings(untouched, normalize=False, resize=False,
                           target_height=16, target_width=16))
        assert [c["shape"] for c in _mock_cellpose.eval_calls] == \
            [(32, 32, 3), (32, 32, 3)]

    def test_save_and_fill_in(self, tmp_path, _mock_cellpose):
        _make_img_dir(tmp_path)
        s = self._settings(tmp_path, save=True, fill_in=True)
        SC.identify_masks_finetune(s)
        assert (tmp_path / "masks").exists()

    def test_grayscale_and_verbose(self, tmp_path, capsys, _plot_calls,
                                   _mock_cellpose):
        """grayscale=True is a printed no-op under Cellpose 4; verbose=True
        renders every image.

        ``eval(channels=...)`` is deprecated in Cellpose 4, so there is no
        channel pair left to force — spaCR says so and passes the image through
        unchanged. That "unchanged" is what the shapes below pin.
        """
        src = _make_img_dir(tmp_path / "gray")
        SC.identify_masks_finetune(
            self._settings(src, grayscale=True, verbose=True))
        out = capsys.readouterr().out

        assert "grayscale=True has no effect under Cellpose 4" in out
        assert "Cellpose settings: Model: cpsam" in out      # the verbose line

        # grayscale did NOT collapse anything: all three channels still ran.
        calls = _mock_cellpose.eval_calls
        assert [c["shape"] for c in calls] == [(16, 16, 3), (16, 16, 3)]
        assert all(c["channel_axis"] == -1 for c in calls)

        # verbose=True rendered both images, with the model's own mask.
        assert len(_plot_calls) == 2
        for call in _plot_calls:
            assert call["stack"].shape == (16, 16, 3)
            assert call["mask"].shape == (16, 16)
            assert sorted(np.unique(call["mask"])) == [0, 1]
            assert call["mask"][2:5, 2:5].all()          # the fake model's blob
            assert len(call["flows"]) == 4

        # Contrast: neither flag is a constant. grayscale=False drops the
        # notice, verbose=False renders nothing at all.
        _mock_cellpose.reset()
        _plot_calls.clear()
        quiet = _make_img_dir(tmp_path / "quiet")
        SC.identify_masks_finetune(
            self._settings(quiet, grayscale=False, verbose=False))
        quiet_out = capsys.readouterr().out
        assert "grayscale=True has no effect" not in quiet_out
        assert "Cellpose settings: Model:" not in quiet_out
        assert _plot_calls == []
        assert len(_mock_cellpose.eval_calls) == 2      # it still segmented

    @pytest.mark.parametrize("model_name,notice", [
        ("cyto2", "predates Cellpose-SAM"),
        ("nucleus", "predates Cellpose-SAM"),
        ("cyto", "predates Cellpose-SAM"),
        ("other", "Unknown Cellpose model"),
    ])
    def test_channel_selection_per_model(self, tmp_path, capsys, monkeypatch,
                                         _mock_cellpose, model_name, notice):
        """There is no per-model channel pair left to select.

        Pre-SAM spaCR mapped cyto / cyto2 / nucleus onto different
        ``eval(channels=...)`` pairs. Cellpose 4 deprecated that argument and
        ships one checkpoint, so every name must resolve to ``cpsam`` and hand
        the network the SAME image — the only thing that differs per name is
        the substitution notice. What still selects channels is
        ``settings['channels']``, proved by the second half.
        """
        import spacr.utils as sutils
        # The notices are printed once per run; give this test a private
        # dedup set so it sees its own and no other test loses theirs.
        monkeypatch.setattr(sutils, "_REPORTED_CELLPOSE_NOTICES", set())

        src = _make_img_dir(tmp_path / model_name)
        SC.identify_masks_finetune(self._settings(src, model_name=model_name))
        out = capsys.readouterr().out

        assert _mock_cellpose.pretrained == ["cpsam"]
        assert notice in out
        assert repr(model_name) in out
        assert "Loaded model: cpsam" in out

        calls = _mock_cellpose.eval_calls
        assert len(calls) == 2
        assert [c["shape"] for c in calls] == [(16, 16, 3), (16, 16, 3)]
        assert all(c["channel_axis"] == -1 for c in calls)

        # Contrast: channels, not the model name, is what changes the input.
        _mock_cellpose.reset()
        one_channel = _make_img_dir(tmp_path / f"{model_name}_c0")
        SC.identify_masks_finetune(
            self._settings(one_channel, model_name=model_name, channels=[0]))
        assert _mock_cellpose.pretrained == ["cpsam"]       # still cpsam
        assert [c["shape"] for c in _mock_cellpose.eval_calls] == \
            [(16, 16), (16, 16)]

    def test_custom_model_present(self, tmp_path, capsys, _mock_cellpose):
        """An existing custom_model is the checkpoint that gets loaded."""
        src = _make_img_dir(tmp_path / "custom")
        model_file = tmp_path / "custom.pth"
        model_file.write_bytes(b"x")
        SC.identify_masks_finetune(
            self._settings(src, custom_model=str(model_file)))

        # The checkpoint path — not 'cpsam' — is what CellposeModel was built
        # with. spaCR used to hard-code 'cpsam' here, which silently discarded
        # every model its own Train-Cellpose module produced.
        assert _mock_cellpose.pretrained == [str(model_file)]
        assert f"Loaded model: {model_file}" in capsys.readouterr().out
        assert len(_mock_cellpose.eval_calls) == 2

        # Contrast: the same settings with custom_model=None load the stock
        # weights, so the assertion above is reading the custom_model branch.
        _mock_cellpose.reset()
        stock = _make_img_dir(tmp_path / "stock")
        SC.identify_masks_finetune(self._settings(stock, custom_model=None))
        assert _mock_cellpose.pretrained == ["cpsam"]


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

    def test_generate_masks_non_normalize_resize(self, tmp_path, capsys,
                                                 _mock_cellpose,
                                                 _quiet_resize_preview):
        """normalize=False + resize=True shrinks a 32x32 source to the target
        before the model sees it; save=False leaves the output folder empty."""
        src = _make_img_dir(tmp_path / "resized", size=32)
        model = _FakeModel()
        SC.generate_masks_from_imgs(
            str(src), model, "nucleus", batch_size=1, diameter=30,
            cellprob_threshold=0.0, flow_threshold=0.4, grayscale=True,
            save=False, normalize=False, channels=[0], percentiles=[2, 98],
            invert=False, plot=False, resize=True, target_height=16,
            target_width=16, remove_background=False, background=100,
            Signal_to_noise=5, verbose=False)

        # batch_size=1 => two batches, one eval each, both at the target size.
        assert [c["shape"] for c in model.eval_calls] == [(16, 16, 3)] * 2
        assert all(c["channel_axis"] == -1 for c in model.eval_calls)
        assert all(c["diameter"] == 30 for c in model.eval_calls)
        # channels=[0] is honoured only by the percentile loader, so three
        # channels survive here — that is why channel_axis is -1 above.
        assert "grayscale=True has no effect under Cellpose 4" in \
            capsys.readouterr().out
        # save=False: the model's own folder is created but nothing lands in it.
        assert (src / "nucleus").is_dir()
        assert list((src / "nucleus").iterdir()) == []

        # Contrast: resize=False hands the model the 32x32 source untouched,
        # and save=True actually writes one mask per image.
        other = _make_img_dir(tmp_path / "untouched", size=32)
        model2 = _FakeModel()
        SC.generate_masks_from_imgs(
            str(other), model2, "nucleus", batch_size=1, diameter=30,
            cellprob_threshold=0.0, flow_threshold=0.4, grayscale=False,
            save=True, normalize=False, channels=[0], percentiles=[2, 98],
            invert=False, plot=False, resize=False, target_height=16,
            target_width=16, remove_background=False, background=100,
            Signal_to_noise=5, verbose=False)
        assert [c["shape"] for c in model2.eval_calls] == [(32, 32, 3)] * 2
        assert sorted(p.name for p in (other / "nucleus").iterdir()) == \
            ["img_0.tif", "img_1.tif"]

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


class _BadEvalModel:
    def __init__(self, *a, **k): self.pretrained_model = "bad"
    def eval(self, x=None, channel_axis=_MISSING, **k):
        # Still honours the channel_axis contract; only the arity is wrong.
        _validate_eval_call(x, channel_axis)
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

    def test_no_cuda_branch(self, tmp_path, capsys, _mock_cellpose,
                            monkeypatch):
        """Without CUDA the model is built for the CPU, and says so."""
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        cpu_src = _make_img_dir(tmp_path / "cpu")
        SC.identify_masks_finetune(_settings(cpu_src))

        assert _mock_cellpose.gpu_flags == [False]
        assert _mock_cellpose.devices == ["cpu"]
        assert "Torch CUDA is not available, using CPU" in \
            capsys.readouterr().out
        assert len(_mock_cellpose.eval_calls) == 2

        # Contrast, so the device is not being read off a constant: the same
        # call with CUDA reported available picks cuda:0 and prints nothing.
        # Building the device object needs no real GPU, so this holds on a
        # CPU-only box too.
        _mock_cellpose.reset()
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        gpu_src = _make_img_dir(tmp_path / "gpu")
        SC.identify_masks_finetune(_settings(gpu_src))
        assert _mock_cellpose.gpu_flags == [True]
        assert _mock_cellpose.devices == ["cuda:0"]
        assert "Torch CUDA is not available" not in capsys.readouterr().out

    def test_parse_per_image_ndarray_item(self):
        # A per-image ndarray flows item → f0=item, rest None (line 63).
        masks = [np.zeros((4, 4), dtype=np.int32)]
        flows = [np.zeros((4, 4), dtype=np.float32)]  # ndarray item
        out = SC.parse_cellpose4_output((masks, flows))
        assert out[1][0] is not None and out[2] == [None]

    def test_identify_verbose_resize(self, tmp_path, _mock_cellpose,
                                     _plot_calls):
        """resize=True restores BOTH the mask and the rendered stack to the
        source resolution after the model has run at the target resolution."""
        src = _make_img_dir(tmp_path / "resized", size=32)
        SC.identify_masks_finetune(
            _settings(src, verbose=True, resize=True, save=True,
                      target_height=16, target_width=16))

        # the model ran at 16x16 (the percentile loader resizes on load) ...
        assert [c["shape"] for c in _mock_cellpose.eval_calls] == \
            [(16, 16, 3)] * 2
        # ... and everything the user sees or keeps came back at 32x32.
        assert len(_plot_calls) == 2
        for call in _plot_calls:
            assert call["stack"].shape == (32, 32, 3)
            assert call["mask"].shape == (32, 32)
            assert call["mask"].max() == 1          # the fake model's one label

        import tifffile
        masks = sorted((src / "masks").glob("*.tif"))
        assert [p.name for p in masks] == ["img_0.tif", "img_1.tif"]
        assert [tifffile.imread(str(p)).shape for p in masks] == \
            [(32, 32), (32, 32)]

        # Contrast: resize=False leaves everything at the loader's resolution,
        # so the 32x32 above is not a constant.
        _mock_cellpose.reset()
        _plot_calls.clear()
        kept = _make_img_dir(tmp_path / "kept", size=32)
        SC.identify_masks_finetune(
            _settings(kept, verbose=True, resize=False, save=True,
                      target_height=16, target_width=16))
        assert [c["shape"] for c in _mock_cellpose.eval_calls] == \
            [(16, 16, 3)] * 2
        assert [c["stack"].shape for c in _plot_calls] == [(16, 16, 3)] * 2
        assert [c["mask"].shape for c in _plot_calls] == [(16, 16)] * 2
        assert [tifffile.imread(str(p)).shape
                for p in sorted((kept / "masks").glob("*.tif"))] == \
            [(16, 16), (16, 16)]

    def test_generate_plot_resize(self, tmp_path, _mock_cellpose, _plot_calls,
                                  _quiet_resize_preview):
        """plot=True renders every image, and resize=True restores the mask and
        the rendered stack to the source resolution first."""
        src = _make_img_dir(tmp_path / "resized", size=32)
        model = _FakeModel()
        SC.generate_masks_from_imgs(
            str(src), model, "cyto", batch_size=1, diameter=30,
            cellprob_threshold=0.0, flow_threshold=0.4, grayscale=False,
            save=False, normalize=False, channels=[0, 1, 2],
            percentiles=[2, 98], invert=False, plot=True, resize=True,
            target_height=16, target_width=16, remove_background=False,
            background=100, Signal_to_noise=5, verbose=False)

        assert [c["shape"] for c in model.eval_calls] == [(16, 16, 3)] * 2
        assert len(_plot_calls) == 2
        for call in _plot_calls:
            assert call["stack"].shape == (32, 32, 3)   # restored for display
            assert call["mask"].shape == (32, 32)
            assert call["mask"].max() == 1
            assert len(call["flows"]) == 4

        # Contrast: plot=False renders nothing and resize=False leaves the
        # model looking at the 32x32 source.
        _plot_calls.clear()
        quiet = _make_img_dir(tmp_path / "quiet", size=32)
        model2 = _FakeModel()
        SC.generate_masks_from_imgs(
            str(quiet), model2, "cyto", batch_size=1, diameter=30,
            cellprob_threshold=0.0, flow_threshold=0.4, grayscale=False,
            save=False, normalize=False, channels=[0, 1, 2],
            percentiles=[2, 98], invert=False, plot=False, resize=False,
            target_height=16, target_width=16, remove_background=False,
            background=100, Signal_to_noise=5, verbose=False)
        assert _plot_calls == []
        assert [c["shape"] for c in model2.eval_calls] == [(32, 32, 3)] * 2

    @pytest.mark.xfail(strict=True, reason=(
        "generate_masks_from_imgs discards the loader's orig_dims. On the "
        "normalize=True branch it overwrites them with the shapes of the "
        "ALREADY target-resized images, so `if resize: resize back to "
        "orig_dims` is a no-op and masks are written at target_height x "
        "target_width instead of the source resolution. identify_masks_finetune "
        "keeps the loader's orig_dims and restores correctly."))
    def test_generate_normalize_resize_restores_source_resolution(
            self, tmp_path, _mock_cellpose, _quiet_resize_preview):
        """A saved mask must line up with the image it segmented.

        Same 32x32 source and 16x16 target as
        ``test_identify_verbose_resize``, which does restore. Here the mask is
        written at 16x16, so it cannot be overlaid on its own source image.
        """
        import tifffile
        src = _make_img_dir(tmp_path / "norm_resize", size=32)
        model = _FakeModel()
        SC.generate_masks_from_imgs(
            str(src), model, "cyto", batch_size=1, diameter=30,
            cellprob_threshold=0.0, flow_threshold=0.4, grayscale=False,
            save=True, normalize=True, channels=[0, 1, 2],
            percentiles=[2, 98], invert=False, plot=False, resize=True,
            target_height=16, target_width=16, remove_background=False,
            background=100, Signal_to_noise=5, verbose=False)

        assert [c["shape"] for c in model.eval_calls] == [(16, 16, 3)] * 2
        written = sorted((src / "cyto").glob("*.tif"))
        assert [p.name for p in written] == ["img_0.tif", "img_1.tif"]
        assert [tifffile.imread(str(p)).shape for p in written] == \
            [(32, 32), (32, 32)]


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
