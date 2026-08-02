"""The Cellpose-4 ``channel_axis`` contract for :mod:`spacr.spacr_cellpose`.

``spacr_cellpose`` hard-coded ``channel_axis=3`` in the two ``model.eval``
calls that back the GUI's *Cellpose Masks* / *Cellpose All* buttons and
:func:`spacr.submodules.analyze_plaques`. Cellpose 4 rejects that value for
BOTH shapes spaCR's loaders produce, so every one of those runs raised:

    (H, W, C) -> IndexError: tuple index out of range
    (H, W)    -> ValueError: 2D image provided, but channel_axis is not None

``object.py`` had already been fixed to ``channel_axis=-1`` at seven sites;
the fix never reached here. These tests are written against the real
``cellpose.transforms.convert_image`` (pure numpy, CPU, offline) rather than
a hand-written stand-in, so they cannot drift from the library the way the
old ``def eval(self, x=None, **kwargs)`` mock did.
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest
import tifffile

import matplotlib
matplotlib.use("Agg")

from cellpose import transforms

from spacr import spacr_cellpose as SC


# The only two shapes spacr.io's loaders hand to model.eval: a channels-last
# stack, or a 2-D image (single-channel loads are squeezed by both callers).
_LOADER_SHAPES = [(16, 16), (16, 16, 1), (16, 16, 2), (16, 16, 3), (16, 16, 5)]


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape,expected", [
    ((16, 16), None),          # 2-D greyscale -> must be None
    ((16, 16, 1), -1),
    ((16, 16, 3), -1),
    ((4, 16, 16, 3), -1),      # defensive: anything >=3-D is channels-last
])
def test_cellpose_channel_axis_picks_the_axis_cellpose_accepts(shape, expected):
    assert SC.cellpose_channel_axis(np.zeros(shape, dtype=np.float32)) is expected


def test_cellpose_channel_axis_accepts_a_plain_list():
    """It normalises through np.asarray, so a nested list works too."""
    assert SC.cellpose_channel_axis([[1.0, 2.0], [3.0, 4.0]]) is None
    assert SC.cellpose_channel_axis([[[1.0], [2.0]], [[3.0], [4.0]]]) == -1


# ---------------------------------------------------------------------------
# Against the real Cellpose validator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", _LOADER_SHAPES)
def test_real_cellpose_accepts_the_chosen_axis(shape):
    """convert_image is what CellposeModel.eval calls; it must not raise."""
    x = np.random.default_rng(0).random(shape).astype(np.float32)
    out = transforms.convert_image(x, channel_axis=SC.cellpose_channel_axis(x))
    assert out.shape == (16, 16, 3)          # Cellpose 4 always emits 3 channels


@pytest.mark.parametrize("shape", _LOADER_SHAPES)
def test_real_cellpose_rejects_the_old_hard_coded_three(shape):
    """The regression itself: channel_axis=3 fails on every loader shape."""
    x = np.random.default_rng(0).random(shape).astype(np.float32)
    with pytest.raises((IndexError, ValueError)):
        transforms.convert_image(x, channel_axis=3)


def test_two_d_also_rejects_minus_one():
    """-1 alone is not the fix: a 2-D image needs channel_axis=None.

    This is why the axis is chosen per image instead of copying object.py's
    constant -- object.py only ever passes channels-last stacks.
    """
    x = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="2D image provided"):
        transforms.convert_image(x, channel_axis=-1)


# ---------------------------------------------------------------------------
# The callers, not just the callee
# ---------------------------------------------------------------------------

class _AxisRecordingModel:
    """Fake CellposeModel that defers to the real Cellpose validator."""

    def __init__(self, *a, **k):
        self.pretrained_model = k.get("pretrained_model", "fake")
        self.calls = []

    def eval(self, x=None, channel_axis="__absent__", **kwargs):
        assert channel_axis != "__absent__", "spaCR must pass channel_axis"
        arr = np.asarray(x)
        converted = transforms.convert_image(arr, channel_axis=channel_axis)
        self.calls.append({"ndim": arr.ndim, "channel_axis": channel_axis})
        h, w = converted.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint16)
        mask[3:9, 3:9] = 1
        flows = [np.zeros((h, w, 3), dtype=np.float32),
                 np.zeros((3, h, w), dtype=np.float32),
                 np.zeros((h, w), dtype=np.float32),
                 np.zeros((h, w), dtype=np.float32)]
        return mask, flows, None, None


def _write_imgs(d: pathlib.Path, n=2, channels=3, size=16):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for i in range(n):
        arr = rng.integers(0, 3000, size=(size, size, channels)).astype(np.uint16)
        tifffile.imwrite(str(d / f"im_{i}.tif"), np.squeeze(arr))
    return d


def _settings(src, **over):
    s = {
        "src": str(src), "model_name": "cyto", "custom_model": None,
        "diameter": 30, "flow_threshold": 0.4, "CP_prob": 0.0,
        "grayscale": False, "save": True, "normalize": True,
        "channels": [0, 1, 2], "percentiles": [2, 98], "invert": False,
        "verbose": False, "resize": False, "target_height": 16,
        "target_width": 16, "remove_background": False, "background": 100,
        "Signal_to_noise": 5, "rescale": False, "resample": False,
        "fill_in": False, "batch_size": 2, "plot": False,
    }
    s.update(over)
    return s


@pytest.fixture
def patched_model(monkeypatch):
    import types
    holder = {}

    def _factory(*a, **k):
        holder["model"] = _AxisRecordingModel(*a, **k)
        return holder["model"]

    monkeypatch.setattr(SC, "cp_models",
                        types.SimpleNamespace(CellposeModel=_factory))
    monkeypatch.setattr(SC, "display", lambda *a, **k: None, raising=False)
    return holder


@pytest.mark.parametrize("channels,chan_list,want_axis,want_ndim", [
    (3, [0, 1, 2], -1, 3),      # channels-last stack
    (1, [0], None, 2),          # squeezed to 2-D
])
def test_identify_masks_finetune_passes_a_usable_axis(
        tmp_path, patched_model, channels, chan_list, want_axis, want_ndim):
    """The GUI's Cellpose Masks path: masks are produced, not an IndexError."""
    src = _write_imgs(tmp_path / "src", channels=channels)
    SC.identify_masks_finetune(_settings(src, channels=chan_list))

    calls = patched_model["model"].calls
    assert len(calls) == 2, "both images were segmented"
    for call in calls:
        assert call["ndim"] == want_ndim
        assert call["channel_axis"] is want_axis

    masks = sorted((src / "masks").glob("*.tif"))
    assert len(masks) == 2, "no masks were written"
    for m in masks:
        arr = tifffile.imread(str(m))
        assert arr.shape == (16, 16)
        assert int(arr.max()) == 1


@pytest.mark.parametrize("channels,chan_list,want_axis", [
    (3, [0, 1, 2], -1),
    (1, [0], None),
])
def test_generate_masks_from_imgs_passes_a_usable_axis(
        tmp_path, channels, chan_list, want_axis):
    """check_cellpose_models / Cellpose All path."""
    src = _write_imgs(tmp_path / "src", channels=channels)
    model = _AxisRecordingModel()
    SC.generate_masks_from_imgs(
        str(src), model, "cpsam", batch_size=2, diameter=30,
        cellprob_threshold=0.0, flow_threshold=0.4, grayscale=False,
        save=True, normalize=True, channels=chan_list, percentiles=[2, 98],
        invert=False, plot=False, resize=False, target_height=16,
        target_width=16, remove_background=False, background=100,
        Signal_to_noise=5, verbose=False)

    assert len(model.calls) == 2
    assert {c["channel_axis"] for c in model.calls} == {want_axis}
    assert len(list((src / "cpsam").glob("*.tif"))) == 2


def test_analyze_plaques_settings_run_through_real_cellpose_path(
        tmp_path, patched_model, monkeypatch):
    """analyze_plaques' own settings dict must survive the real eval call.

    analyze_plaques delegates segmentation to identify_masks_finetune, so it
    inherited the same defect: the user's plaque assay raised IndexError
    before a single mask was written.

    The handoff is captured (no network: ``download_models`` is stubbed, and
    the bundled checkpoint is redirected into tmp_path rather than written
    into the installed package), then the captured settings are replayed
    through the REAL identify_masks_finetune.
    """
    from spacr import submodules as SUB
    from spacr import utils as UTILS

    src = _write_imgs(tmp_path / "src", channels=3)
    monkeypatch.setattr(UTILS, "download_models", lambda *a, **k: None)

    captured = {}
    real_identify = SC.identify_masks_finetune       # before it is patched

    def _capture(settings):
        captured["settings"] = dict(settings)
        raise RuntimeError("__handoff__")

    monkeypatch.setattr(SC, "identify_masks_finetune", _capture)

    with pytest.raises(RuntimeError, match="__handoff__"):
        SUB.analyze_plaques({"src": str(src), "masks": True})

    resolved = captured["settings"]
    # It is the plaque model that gets selected, and it is a custom_model
    # (which is what makes this path hit identify_masks_finetune's eval).
    assert resolved["custom_model"].endswith(
        "toxo_plaque_cyto_e25000_X1120_Y1120.CP_model")

    # Replay through the real function with the checkpoint redirected to a
    # tmp stand-in, so nothing is written inside the package tree.
    stand_in = tmp_path / "toxo_plaque_cyto_e25000_X1120_Y1120.CP_model"
    stand_in.write_bytes(b"stand-in checkpoint")
    resolved["custom_model"] = str(stand_in)
    resolved["save"] = True

    real_identify(resolved)

    calls = patched_model["model"].calls
    assert calls, "analyze_plaques' settings never reached model.eval"
    assert {c["channel_axis"] for c in calls} == {-1}
    assert len(list((src / "masks").glob("*.tif"))) == 2


# ---------------------------------------------------------------------------
# The other arity Cellpose 4 returns from the same call sites
# ---------------------------------------------------------------------------

class _ThreeTupleModel(_AxisRecordingModel):
    """Cellpose 4 returns (masks, flows, styles) when it has no diameters.

    Both call sites branch on ``len(output)``; only the 4-tuple leg was ever
    exercised, so the ``len(output) == 3`` unpack went untested at both.
    """

    def eval(self, x=None, channel_axis="__absent__", **kwargs):
        mask, flows, _, _ = super().eval(x=x, channel_axis=channel_axis, **kwargs)
        return mask, flows, None


def test_identify_masks_finetune_unpacks_a_three_tuple(tmp_path, monkeypatch):
    import types
    holder = {}
    monkeypatch.setattr(SC, "cp_models", types.SimpleNamespace(
        CellposeModel=lambda *a, **k: holder.setdefault("m", _ThreeTupleModel(*a, **k))))
    monkeypatch.setattr(SC, "display", lambda *a, **k: None, raising=False)

    src = _write_imgs(tmp_path / "src", channels=3)
    SC.identify_masks_finetune(_settings(src))

    assert len(list((src / "masks").glob("*.tif"))) == 2
    assert {c["channel_axis"] for c in holder["m"].calls} == {-1}


def test_generate_masks_from_imgs_unpacks_a_three_tuple(tmp_path):
    src = _write_imgs(tmp_path / "src", channels=3)
    model = _ThreeTupleModel()
    SC.generate_masks_from_imgs(
        str(src), model, "cpsam", batch_size=2, diameter=30,
        cellprob_threshold=0.0, flow_threshold=0.4, grayscale=False,
        save=True, normalize=True, channels=[0, 1, 2], percentiles=[2, 98],
        invert=False, plot=False, resize=False, target_height=16,
        target_width=16, remove_background=False, background=100,
        Signal_to_noise=5, verbose=False)

    assert len(list((src / "cpsam").glob("*.tif"))) == 2


# ---------------------------------------------------------------------------
# Source-level guard: nobody may hard-code a positive channel_axis again
# ---------------------------------------------------------------------------

def test_no_eval_call_in_spacr_hard_codes_a_positive_channel_axis():
    """A positive literal channel_axis is always wrong for a 2-D/3-D image.

    Cellpose 4 indexes ``x.shape[channel_axis]``, so for the <=3-D arrays
    spaCR produces the only valid literals are ``-1``, ``0`` and ``None``.
    This is the guard that would have caught the original ``channel_axis=3``
    at every site at once, including the seven in object.py.
    """
    pkg = pathlib.Path(SC.__file__).resolve().parent
    offenders = []
    for path in sorted(pkg.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "eval"):
                continue
            for kw in node.keywords:
                if kw.arg != "channel_axis":
                    continue
                val = kw.value
                if isinstance(val, ast.Constant) and isinstance(val.value, int) \
                        and not isinstance(val.value, bool) and val.value > 0:
                    offenders.append(f"{path.name}:{node.lineno} -> {val.value}")

    assert offenders == [], (
        "model.eval() called with a positive literal channel_axis; Cellpose 4 "
        f"raises IndexError/ValueError for spaCR's image shapes: {offenders}"
    )
