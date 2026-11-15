"""Table-driven tests of ``spacr.spacr_cellpose``: the Cellpose 4 output parser and the two mask-generation loops.

``parse_cellpose4_output`` is walked over the batched, per-image and 2-D
layouts and the shapes it refuses.  ``identify_masks_finetune`` and
``generate_masks_from_imgs`` run against a recording ``CellposeModel``
double whose ``eval`` declares the full Cellpose 4 signature with
``channel_axis=MISSING_CHANNEL_AXIS`` and hands
the call to ``check_cellpose_eval_call`` (the real Cellpose input
validator), so each case pins which checkpoint was loaded on which device,
the exact pixels and arguments the network received, what was rendered and
which masks were written at which resolution.  CPU-only and offline.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

from spacr import spacr_cellpose as SC  # noqa: E402
from tests.cellpose_api_contract import eval_arguments  # noqa: E402
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call  # noqa: E402

BLOB = (slice(2, 5), slice(2, 5))
GRAYSCALE_NOTICE = ("grayscale=True has no effect under Cellpose 4: the "
                    "channel pair (eval channels=) is deprecated and ignored.")


class _Recorder:
    """Every model built and every ``eval`` call made, across instances."""

    def __init__(self):
        self.models = []

    def reset(self):
        self.models.clear()

    @property
    def calls(self):
        return [call for model in self.models for call in model.calls]

    @property
    def shapes(self):
        return [call["image"].shape for call in self.calls]

    @property
    def pretrained(self):
        return [m.init_kwargs.get("pretrained_model") for m in self.models]


class _FakeModel:
    """Stand-in for ``cellpose.models.CellposeModel``."""

    recorder = None

    def __init__(self, *args, **kwargs):
        self.init_kwargs = dict(kwargs)
        self.pretrained_model = kwargs.get("pretrained_model", "fake")
        self.calls = []
        if _FakeModel.recorder is not None:
            _FakeModel.recorder.models.append(self)

    def _record(self, x, channel_axis, kwargs):
        (converted,) = check_cellpose_eval_call(x, channel_axis)
        self.calls.append({**kwargs, "image": np.asarray(x).copy()})
        h, w = converted.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint16)
        mask[BLOB] = 1
        return mask, h, w

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
             rescale=None, diameter=None, flow_threshold=0.4,
             cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
             max_size_fraction=0.4, niter=None, augment=False,
             tile_overlap=0.1, bsize=None, compute_masks=True,
             progress=None):
        mask, h, w = self._record(x, channel_axis, eval_arguments(locals()))
        flows = [np.zeros((h, w, 3), dtype=np.float32),
                 np.zeros((3, h, w), dtype=np.float32),
                 np.zeros((h, w), dtype=np.float32),
                 np.zeros((h, w), dtype=np.float32)]
        return mask, flows, None


class _TwoTupleModel(_FakeModel):
    """Honours the eval contract but returns a two-tuple."""

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
             rescale=None, diameter=None, flow_threshold=0.4,
             cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
             max_size_fraction=0.4, niter=None, augment=False,
             tile_overlap=0.1, bsize=None, compute_masks=True,
             progress=None):
        mask, _h, _w = self._record(x, channel_axis, eval_arguments(locals()))
        return mask, None


@pytest.fixture
def cellpose(monkeypatch):
    """Install the fake model and silence the resize preview; yields the recorder."""
    recorder = _Recorder()
    monkeypatch.setattr(_FakeModel, "recorder", recorder)
    monkeypatch.setattr(SC, "cp_models",
                        types.SimpleNamespace(CellposeModel=_FakeModel))
    monkeypatch.setattr(SC, "display", lambda *a, **k: None, raising=False)
    import spacr.plot as PL
    monkeypatch.setattr(PL, "plot_resize", lambda *a, **k: None)
    return recorder


@pytest.fixture
def plots(monkeypatch):
    """Record every ``print_mask_and_flows(stack, mask, flows)`` call."""
    import spacr.plot as PL
    calls = []
    monkeypatch.setattr(
        PL, "print_mask_and_flows",
        lambda stack, mask, flows, **k: calls.append(
            {"stack": np.asarray(stack), "mask": np.asarray(mask),
             "flows": flows}))
    return calls


def _images(folder, n=2, channels=3, size=16):
    """``n`` random uint16 ``(size, size, channels)`` TIFFs in ``folder``."""
    import tifffile
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        tifffile.imwrite(str(folder / f"img_{i}.tif"), rng.integers(
            0, 2000, size=(size, size, channels)).astype(np.uint16))
    return folder


def _masks(folder):
    """``{name: array}`` of the TIFF masks written in ``folder``."""
    import tifffile
    return {p.name: tifffile.imread(str(p))
            for p in sorted(folder.glob("*.tif"))}


def _blob(size):
    mask = np.zeros((size, size), dtype=np.uint16)
    mask[BLOB] = 1
    return mask


def _settings(src, **over):
    """The identify_masks_finetune settings used across the cases."""
    settings = {
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
    settings.update(over)
    return settings


def test_parse_cellpose4_output_normalises_every_flow_layout(subtests):
    """Batched, per-image and single 2-D outputs become per-image flow lists; unknown layouts are refused."""
    a, b, c, d = (np.full((4, 4), v, dtype=np.float32) for v in range(4))
    one = [np.zeros((4, 4), dtype=np.int32)]
    two = [np.zeros((4, 4)), np.zeros((4, 4))]
    per_image = [
        ("per-image-str-item", one, ["scalar-item"], [None], [None], [None],
         [None]),
        ("per-image-partial-list", one, [[a]], [a], [None], [None], [None]),
        ("per-image-full-tuple", one, [(a, b, c, d)], [a], [b], [c], [d]),
        ("per-image-empty-list", one, [[]], [None], [None], [None], [None]),
        ("per-image-ndarray-item", one, [a], [a], [None], [None], [None]),
        ("2d-mask-three-flows", np.zeros((4, 4), dtype=np.int32), [a, b, c],
         [a], [b], [c], [None]),
    ]
    for case_id, masks, flows, f0, f1, f2, f3 in per_image:
        with subtests.test(msg=case_id):
            out = SC.parse_cellpose4_output((masks, flows))
            assert out[0] is masks
            for got, expected in zip(out[1:], (f0, f1, f2, f3)):
                assert len(got) == len(expected)
                for g, e in zip(got, expected):
                    assert g is e

    with subtests.test(msg="batched-four-arrays"):
        masks = np.zeros((2, 8, 8), dtype=np.int32)
        flow0 = np.arange(2 * 64, dtype=np.float32).reshape(2, 8, 8)
        flow1 = np.arange(3 * 2 * 64, dtype=np.float32).reshape(3, 2, 8, 8)
        flow2 = flow0 + 1000
        flow3 = flow0 + 2000
        out = SC.parse_cellpose4_output((masks, [flow0, flow1, flow2, flow3]))
        assert out[0] is masks
        for i in range(2):
            assert np.array_equal(out[1][i], flow0[i])
            assert np.array_equal(out[2][i], flow1[:, i])
            assert np.array_equal(out[3][i], flow2[i])
            assert np.array_equal(out[4][i], flow3[i])
        assert [len(f) for f in out[1:]] == [2, 2, 2, 2]

    class _NoLen:
        pass

    refused = [
        ("flows-not-a-sequence", (one, "not-a-list"),
         "Unrecognized Cellpose flows type: <class 'str'>"),
        ("masks-without-length", (_NoLen(), [[], [], []]),
         f"Cannot determine number of images in masks (type={_NoLen})"),
        ("three-flows-for-two-images", (two, [a, b, c]),
         "Unrecognized Cellpose flows format: type=<class 'list'>, len=3"),
    ]
    for case_id, output, message in refused:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError) as raised:
                SC.parse_cellpose4_output(output)
            assert str(raised.value) == message


def test_identify_masks_finetune_loads_the_right_model_and_feeds_it_the_right_pixels(
        tmp_path, cellpose, plots, capsys, monkeypatch, subtests):
    """Model choice, device, normalisation, channels, resize, verbose, save and fill-in each reach the run."""
    import spacr.accelerator as accelerator
    import spacr.utils as sutils
    import torch
    from spacr import model_zoo
    from spacr.qt.i18n import tr

    def run(name, **over):
        cellpose.reset()
        plots.clear()
        capsys.readouterr()
        src = _images(tmp_path / name, size=over.pop("size", 16))
        result = SC.identify_masks_finetune(_settings(src, **over))
        return src, result, capsys.readouterr().out

    with subtests.test(msg="missing-custom-model-is-refused"):
        missing = str(tmp_path / "nope.pth")
        with pytest.raises(model_zoo.ModelZooError) as refused:
            SC.identify_masks_finetune(
                _settings(tmp_path, custom_model=missing))
        assert repr(missing) in str(refused.value)
        assert "stock weights were not substituted" in str(refused.value)
        assert cellpose.models == []

    with subtests.test(msg="unknown-model-name-is-refused"):
        src = _images(tmp_path / "other")
        before = {p: p.read_bytes() for p in src.rglob("*") if p.is_file()}
        with pytest.raises(model_zoo.ModelZooError) as refused:
            SC.identify_masks_finetune(_settings(src, model_name="other"))
        expected = tr(
            "Cellpose model {name} is not an installed stock name, an existing checkpoint or an exact model-zoo entry. Choose a listed model or provide its checkpoint; stock weights were not substituted.",
            name=repr("other"))
        expected += "\n" + model_zoo._model_resolution_context(
            "other", setting="model_name", kinds=("cellpose",))
        assert str(refused.value) == expected
        assert cellpose.models == []
        assert {p: p.read_bytes() for p in before} == before

    with subtests.test(msg="no-images"):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert SC.identify_masks_finetune(_settings(empty)) is None
        assert cellpose.calls == []
        assert (empty / "masks").is_dir()
        assert "Either no images were found in" in capsys.readouterr().out

    with subtests.test(msg="percentiles-rescale-one-channel"):
        src, result, out = run("norm", channels=[0])
        assert result is None
        assert cellpose.pretrained == ["cpsam"]
        assert len(cellpose.calls) == 2
        for call in cellpose.calls:
            assert call["image"].shape == (16, 16)
            assert call["image"].dtype == np.float64
            assert (call["image"].min(), call["image"].max()) == (0.0, 1.0)
            assert {k: call[k] for k in (
                "channel_axis", "normalize", "diameter", "flow_threshold",
                "cellprob_threshold", "rescale", "resample", "progress")} == {
                "channel_axis": None, "normalize": False, "diameter": 30,
                "flow_threshold": 0.4, "cellprob_threshold": 0.0,
                "rescale": None, "resample": False, "progress": True}
        assert "Loaded model: cpsam" in out
        assert "Found 2 Images with 0 masks. Generating masks for 2 images" \
            in out
        assert _masks(src / "masks") == {}

    with subtests.test(msg="no-percentiles-cellpose-normalises"):
        run("cp-norm", percentiles=None, rescale=2.0)
        assert [c["normalize"] for c in cellpose.calls] == [True, True]
        assert [c["rescale"] for c in cellpose.calls] == [2.0, 2.0]
        assert cellpose.shapes == [(16, 16, 3)] * 2

    raw = [
        # id, overrides, source size, eval shapes
        ("raw-ignores-channels", {"normalize": False, "channels": [0]}, 16,
         [(16, 16, 3)] * 2),
        ("raw-resize-shrinks", {"normalize": False, "resize": True}, 32,
         [(16, 16, 3)] * 2),
        ("raw-no-resize-keeps-source", {"normalize": False}, 32,
         [(32, 32, 3)] * 2),
    ]
    for case_id, over, size, shapes in raw:
        with subtests.test(msg=case_id):
            run(case_id, size=size, **over)
            assert cellpose.shapes == shapes
            assert [c["channel_axis"] for c in cellpose.calls] == [-1, -1]

    with subtests.test(msg="save-and-fill-in"):
        src, _, _ = run("saved", save=True, fill_in=True)
        assert {k: v.tolist() for k, v in _masks(src / "masks").items()} == {
            "img_0.tif": _blob(16).tolist(), "img_1.tif": _blob(16).tolist()}

    with subtests.test(msg="grayscale-notice-and-verbose-renders"):
        _, _, out = run("gray", grayscale=True, verbose=True)
        assert GRAYSCALE_NOTICE in out
        assert ("Cellpose settings: Model: cpsam, channels: [0, 1, 2], "
                "normalisation: percentiles [2, 98], diameter:30, "
                "flow_threshold:0.4, cellprob_threshold:0.0") in out
        assert cellpose.shapes == [(16, 16, 3)] * 2
        assert [c["channel_axis"] for c in cellpose.calls] == [-1, -1]
        assert len(plots) == 2
        for call in plots:
            assert call["stack"].shape == (16, 16, 3)
            assert np.array_equal(call["mask"], _blob(16))
            assert len(call["flows"]) == 4

    with subtests.test(msg="quiet-run-renders-nothing"):
        _, _, out = run("quiet")
        assert GRAYSCALE_NOTICE not in out
        assert "Cellpose settings: Model:" not in out
        assert plots == []
        assert len(cellpose.calls) == 2

    for name in ("cyto2", "nucleus", "cyto"):
        with subtests.test(msg=f"legacy-name-{name}-resolves-to-cpsam"):
            monkeypatch.setattr(sutils, "_REPORTED_CELLPOSE_NOTICES", set())
            _, _, out = run(name, model_name=name)
            assert cellpose.pretrained == ["cpsam"]
            assert "predates Cellpose-SAM" in out and repr(name) in out
            assert cellpose.shapes == [(16, 16, 3)] * 2
            run(f"{name}_c0", model_name=name, channels=[0])
            assert cellpose.pretrained == ["cpsam"]
            assert cellpose.shapes == [(16, 16)] * 2

    with subtests.test(msg="custom-model-is-loaded"):
        model_file = tmp_path / "custom.pth"
        model_file.write_bytes(b"x")
        _, _, out = run("custom", custom_model=str(model_file))
        assert cellpose.pretrained == [str(model_file)]
        assert f"Loaded model: {model_file}" in out
        assert len(cellpose.calls) == 2

    cpu, cuda = torch.device("cpu"), torch.device("cuda:0")
    for case_id, gpu, device, said in [
            ("no-cuda-uses-cpu", False, cpu,
             "No GPU available to spaCR, using CPU"),
            ("cuda-segments-on-gpu", True, cuda, "Segmenting on test CUDA")]:
        with subtests.test(msg=case_id), monkeypatch.context() as patch:
            patch.setattr(accelerator, "cellpose_gpu", lambda g=gpu: g)
            patch.setattr(accelerator, "cellpose_kwargs",
                          lambda g=gpu, d=device: {"gpu": g, "device": d})
            patch.setattr(accelerator, "describe", lambda: "test CUDA")
            _, _, out = run(case_id)
            assert [m.init_kwargs for m in cellpose.models] == [
                {"pretrained_model": "cpsam", "gpu": gpu, "device": device}]
            assert said in out
            assert len(cellpose.calls) == 2

    for case_id, resize, size in [("verbose-resize-restores-source", True, 32),
                                  ("verbose-no-resize-keeps-loader", False,
                                   16)]:
        with subtests.test(msg=case_id):
            src, _, _ = run(case_id, size=32, verbose=True, resize=resize,
                            save=True)
            assert cellpose.shapes == [(16, 16, 3)] * 2
            assert [c["stack"].shape for c in plots] == [(size, size, 3)] * 2
            assert [c["mask"].shape for c in plots] == [(size, size)] * 2
            assert [c["mask"].max() for c in plots] == [1, 1]
            assert {k: v.shape for k, v in _masks(src / "masks").items()} == {
                "img_0.tif": (size, size), "img_1.tif": (size, size)}

    with subtests.test(msg="two-tuple-eval-is-refused"):
        monkeypatch.setattr(SC, "cp_models",
                            types.SimpleNamespace(CellposeModel=_TwoTupleModel))
        with pytest.raises(ValueError) as refused:
            run("bad-eval")
        assert str(refused.value) == \
            "Unexpected number of return values from model.eval()"


def test_generate_masks_from_imgs_writes_one_mask_per_image_at_the_source_resolution(
        tmp_path, cellpose, plots, capsys, subtests):
    """Loader, resize, plot, save and verbose choices each change exactly what the model sees and what is kept."""
    def run(name, model=None, size=16, **over):
        plots.clear()
        capsys.readouterr()
        src = _images(tmp_path / name, size=size)
        model = model or _FakeModel()
        kwargs = dict(
            model_name="cyto", batch_size=1, diameter=30,
            cellprob_threshold=0.0, flow_threshold=0.4, grayscale=False,
            save=False, normalize=False, channels=[0, 1, 2],
            percentiles=[2, 98], invert=False, plot=False, resize=False,
            target_height=16, target_width=16, remove_background=False,
            background=100, Signal_to_noise=5, verbose=False)
        kwargs.update(over)
        assert SC.generate_masks_from_imgs(str(src), model, **kwargs) is None
        return src / kwargs["model_name"], model, capsys.readouterr().out

    cases = [
        # id, overrides, source size, eval shapes, written mask shape or None,
        # plotted stack shape or None
        ("normalize-save-verbose", dict(normalize=True, save=True,
                                        verbose=True, batch_size=2), 16,
         [(16, 16, 3)] * 2, (16, 16), None),
        ("raw-resize-grayscale-unsaved", dict(model_name="nucleus",
                                              grayscale=True, resize=True,
                                              channels=[0]), 32,
         [(16, 16, 3)] * 2, None, None),
        ("raw-untouched-saved", dict(model_name="nucleus", save=True,
                                     channels=[0]), 32,
         [(32, 32, 3)] * 2, (32, 32), None),
        ("plot-resize-restores-for-display", dict(plot=True, resize=True), 32,
         [(16, 16, 3)] * 2, None, (32, 32, 3)),
        ("plot-off-untouched", dict(), 32, [(32, 32, 3)] * 2, None, None),
        ("normalize-resize-restores-saved-masks",
         dict(normalize=True, resize=True, save=True), 32,
         [(16, 16, 3)] * 2, (32, 32), None),
    ]
    for case_id, over, size, shapes, written, plotted in cases:
        with subtests.test(msg=case_id):
            folder, model, out = run(case_id, size=size, **over)
            assert [c["image"].shape for c in model.calls] == shapes
            assert [c["channel_axis"] for c in model.calls] == [-1, -1]
            assert [{k: c[k] for k in ("normalize", "diameter", "rescale",
                                       "resample", "progress",
                                       "cellprob_threshold",
                                       "flow_threshold")}
                    for c in model.calls] == [
                {"normalize": False, "diameter": 30, "rescale": None,
                 "resample": False, "progress": False,
                 "cellprob_threshold": 0.0, "flow_threshold": 0.4}] * 2
            assert folder.is_dir()
            masks = _masks(folder)
            if written is None:
                assert masks == {}
            else:
                assert sorted(masks) == ["img_0.tif", "img_1.tif"]
                for mask in masks.values():
                    assert mask.shape == written
                    # the model's 3x3 blob, scaled back when it was resized
                    scale = written[0] // 16 if over.get("resize") else 1
                    expected = np.zeros(written, dtype=np.uint16)
                    expected[2 * scale:5 * scale, 2 * scale:5 * scale] = 1
                    assert np.array_equal(mask, expected)
            if plotted is None:
                assert plots == []
            else:
                assert [c["stack"].shape for c in plots] == [plotted] * 2
                assert [c["mask"].shape for c in plots] == [plotted[:2]] * 2
                assert [c["mask"].max() for c in plots] == [1, 1]
                assert [len(c["flows"]) for c in plots] == [4, 4]
            assert (GRAYSCALE_NOTICE in out) is bool(over.get("grayscale"))
            assert ("Cellpose settings: Model: cyto, channels: [0, 1, 2], "
                    "diameter:30, flow_threshold:0.4, cellprob_threshold:0.0"
                    in out) is bool(over.get("verbose"))

    with subtests.test(msg="two-tuple-eval-is-refused"):
        with pytest.raises(ValueError) as refused:
            run("bad-eval", model=_TwoTupleModel(), channels=[0])
        assert str(refused.value) == \
            "Unexpected number of return values from model.eval()"
