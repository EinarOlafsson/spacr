"""Cellpose-SAM does not ignore ``diameter``, and the UI must not say it does.

The Live Preview settings dialog disabled the diameter spinner whenever
the model was ``cpsam`` and hung the tooltip "Ignored by Cellpose-SAM" on
it. Both halves were false, and they cost the user a control that
measurably changes what cpsam segments.

What Cellpose 4 actually does — ``CellposeModel._run_cp``, cellpose
4.0.7::

    if diameter is not None:
        image_scaling = 30. / diameter

i.e. the image is resampled so objects land near the network's 30 px
working size. Nothing auto-estimates anything.

Measured on the GPU (RTX 3090, driver 580.173.02, cellpose 4.0.7, torch
2.9.1+cu128) through the panel's own worker entry point
``live_preview._segment_multi``, on a real 1994x1994 micrograph
(``plate1_E01_10.tif``), model ``cpsam``, flow 0.4, cellprob 0.0:

    ===============  =======  ========
    diameter          cells    nuclei
    ===============  =======  ========
    unset (None)           66        65
    30                     66        65
    60                     71        63
    ===============  =======  ========

``30`` matching "unset" is the trap: 30/30 is a no-op rescale, and 30 is
the spinner's default, so anyone who checked the claim by nudging
nothing saw no change and believed it.

These tests are CPU-only and offline. The GPU numbers above are the
evidence for *why* the control matters; what is asserted here is that
cellpose still contains the rescale, that the panel's request carries the
value, and that the dialog no longer disables or mislabels it.
"""
from __future__ import annotations

import re
import sys
import types

import numpy as np
import pytest

from spacr.qt.widgets import live_preview as LP

from tests.cellpose_api_contract import (
    MISSING_CHANNEL_AXIS,
    emulate_pretrained_model,
    eval_arguments,
    init_arguments,
)
from tests.conftest import check_cellpose_eval_call


# ---------------------------------------------------------------------------
# Fake Cellpose — shaped like the installed cellpose 4 API, nothing else
# ---------------------------------------------------------------------------

class _RecordingCellposeModel:
    """Records construction kwargs and every ``eval`` call.

    Both signatures are the installed cellpose 4.0.7 ones, written out with
    the real defaults and no ``**kwargs``, so an argument Cellpose 4 dropped
    (``interp``, ``tile``, ``net_avg``, ``model_loaded``) raises ``TypeError``
    here rather than being absorbed.

    ``eval`` mirrors cellpose 4's real return shape ``(masks, flows,
    styles)`` and, unlike a pure spy, actually *honours* ``diameter`` the
    way cellpose does — it rescales by ``30 / diameter`` and emits a
    number of objects that depends on that scale. A mock that ignored the
    argument would have agreed with the bug.

    ``__init__`` reproduces cellpose 4's *behaviour* as well as its parameter
    list: ``model_type=`` is warned about and thrown away, so
    :attr:`loaded_model` follows ``pretrained_model`` alone. That is how a
    caller still selecting weights through ``model_type=`` silently gets
    ``cpsam``.
    """

    instances: list = []

    def __init__(self, gpu=False, pretrained_model="cpsam", model_type=None,
                 diam_mean=None, device=None, nchan=None, use_bfloat16=True):
        self.kwargs = init_arguments(locals())
        self.loaded_model = emulate_pretrained_model(pretrained_model,
                                                     model_type)
        self.calls: list = []
        type(self).instances.append(self)

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
             invert=False, rescale=None, diameter=None, flow_threshold=0.4,
             cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
             max_size_fraction=0.4, niter=None, augment=False,
             tile_overlap=0.1, bsize=256, compute_masks=True, progress=None):
        # The panel hands over one 2-D plane and lets cellpose detect the
        # axis, so the value is checked but not required to be present.
        check_cellpose_eval_call(x, channel_axis, require_channel_axis=False)
        self.calls.append(eval_arguments(locals()))
        image = x
        scaling = 1.0 if diameter is None else 30.0 / diameter
        mask = np.zeros(image.shape[:2], dtype=np.uint16)
        # More objects when the image is upscaled, fewer when it shrinks —
        # the direction cellpose's own rescale produces.
        count = max(1, int(round(4 * scaling)))
        for label in range(1, count + 1):
            row = 2 * label
            mask[row:row + 1, 1:4] = label
        flow_rgb = np.zeros(image.shape[:2] + (3,), dtype=np.uint8)
        return mask, [flow_rgb], None


@pytest.fixture
def fake_cellpose(monkeypatch):
    """Install a fake ``cellpose.models`` for the duration of one test."""
    _RecordingCellposeModel.instances = []
    models = types.ModuleType("cellpose.models")
    models.CellposeModel = _RecordingCellposeModel
    pkg = types.ModuleType("cellpose")
    pkg.models = models
    monkeypatch.setitem(sys.modules, "cellpose", pkg)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    return _RecordingCellposeModel


def _request(**kwargs) -> LP.PreviewRequest:
    image = np.zeros((24, 24, 2), dtype=np.uint16)
    image[..., 0] = 120
    image[..., 1] = 220
    kwargs.setdefault("image", image)
    kwargs.setdefault("model", "cpsam")
    return LP.PreviewRequest(**kwargs)


def _count(mask) -> int:
    return int(len(np.unique(mask)) - 1)


# ===========================================================================
# 1. The library contract the tooltip now states
# ===========================================================================

def test_cellpose_4_still_rescales_by_thirty_over_diameter():
    """The claim the tooltip makes, checked against the installed cellpose.

    If this fails, cellpose changed what ``diameter`` does and
    :data:`live_preview.DIAMETER_TOOLTIP` has to be re-checked before it
    ships another sentence about it.
    """
    cellpose = pytest.importorskip("cellpose.models")
    import inspect
    source = inspect.getsource(cellpose.CellposeModel)
    assert re.search(r"30\.?\s*/\s*diameter", source), (
        "cellpose.models.CellposeModel no longer rescales by 30/diameter — "
        "re-measure before trusting DIAMETER_TOOLTIP")
    assert "diameter" in inspect.signature(cellpose.CellposeModel.eval).parameters


def test_the_tooltip_says_what_cellpose_does():
    tip = LP.DIAMETER_TOOLTIP
    assert "30/diameter" in tip
    assert "ignored" not in tip.lower()
    assert "cellpose-sam uses it" in tip.lower()


# ===========================================================================
# 2. The value reaches cellpose
# ===========================================================================

@pytest.mark.parametrize("diameter", [15.0, 30.0, 60.0])
def test_the_requested_diameter_is_handed_to_cpsam(fake_cellpose, diameter):
    LP._segment_multi(_request(diameter=diameter))
    model = fake_cellpose.instances[-1]
    assert model.kwargs["pretrained_model"] == "cpsam"
    assert model.calls[-1]["diameter"] == diameter


def test_zero_means_unset_not_zero(fake_cellpose):
    """The spinner's 0 is "let cellpose decide", not a 0 px object.

    ``30. / 0`` would be a ZeroDivisionError inside cellpose, so the
    worker translates 0 to ``None`` — which is also what makes 0 and 30
    behave identically (30/30 = 1, and None skips the rescale).
    """
    LP._segment_multi(_request(diameter=0.0))
    assert fake_cellpose.instances[-1].calls[-1]["diameter"] is None


def test_diameter_changes_what_comes_back_for_cpsam(fake_cellpose):
    """End to end through ``_segment_multi``, with the rescale modelled.

    The GPU numbers in the module docstring are the real version of this
    (66 -> 71 cells between unset and 60); here the shape of the
    dependency is asserted without a GPU.
    """
    small = LP._segment_multi(_request(diameter=15.0))[0]["cell"]
    default = LP._segment_multi(_request(diameter=30.0))[0]["cell"]
    large = LP._segment_multi(_request(diameter=60.0))[0]["cell"]

    assert _count(small) > _count(default) > _count(large)
    assert _count(default) == _count(
        LP._segment_multi(_request(diameter=0.0))[0]["cell"]), \
        "30 and 'unset' are the same rescale — that is why the bug hid"


def test_flow_and_cellprob_reach_cpsam_too(fake_cellpose):
    """The other two knobs the panel used to describe as SAM-ignored."""
    LP._segment_multi(_request(diameter=30.0, flow_threshold=0.7,
                               cellprob=-1.5))
    call = fake_cellpose.instances[-1].calls[-1]
    assert call["flow_threshold"] == 0.7
    assert call["cellprob_threshold"] == -1.5


# ===========================================================================
# 3. The panel carries it
# ===========================================================================

def test_the_panel_puts_its_diameter_in_the_request(qtbot):
    panel = LP.LivePreviewPanel()
    qtbot.addWidget(panel)
    panel._model_box.setCurrentIndex(panel._model_box.findText("cpsam"))
    panel._diameter.setValue(72.0)
    panel._image = np.zeros((16, 16), dtype=np.uint16)

    request = panel._build_request()
    assert request.model == "cpsam"
    assert request.diameter == 72.0
    assert panel.current_params()["diameter"] == 72.0


def test_the_dialog_leaves_the_diameter_usable_on_cpsam(qtbot):
    """The user-visible half of the fix, on the widget the user touches.

    Before: ``setEnabled(False)`` plus "Ignored by Cellpose-SAM", so the
    value the request carried was whatever the spinner happened to hold
    and the user had no way to change it.
    """
    panel = LP.LivePreviewPanel()
    qtbot.addWidget(panel)
    panel._model_box.setCurrentIndex(panel._model_box.findText("cpsam"))
    panel.open_live_settings()
    try:
        assert panel._diameter.isEnabled()
        assert panel._diameter.toolTip() == ""
        label = panel._diameter._spacr_setting_label
        assert "30/diameter" in label.toolTip()
        assert "href=" in label.toolTip()
        # No `_spacr_api_dot`: the live-preview dialog passes
        # `api_dots=False`. The link is still in the label's tooltip, which
        # is what the assertion above checks and what this test is about.
        panel._diameter.setValue(60.0)
    finally:
        panel._live_settings_dialog.close()

    panel._image = np.zeros((16, 16), dtype=np.uint16)
    assert panel._build_request().diameter == 60.0
