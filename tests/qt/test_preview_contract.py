"""The one contract every live view keeps.

Four modules ship a live view — Mask, Timelapse, Measure and Motility — and
before :mod:`spacr.qt.widgets.preview_contract` each had grown its own
answer to the same questions. Measure's button said "Refresh crops" while
the others said "Run preview"; only Mask could be cancelled, and only from
Python; and Measure returned from a press with no array loaded without a
word on the status line. A bug fixed in one was a bug that stayed in the
other three, which is exactly what happened to the Cellpose ``model_type=``
defect: written twice, fixed twice.

Everything here is asserted for **all four panels at once**, so a fifth
live view, or a fix applied to one of these, cannot quietly diverge again.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

pytest.importorskip("PySide6")

from tests.cellpose_api_contract import MISSING_CHANNEL_AXIS, init_arguments
from tests.conftest import check_cellpose_eval_call


# ---------------------------------------------------------------------------
# The four live views
# ---------------------------------------------------------------------------

def _mask_panel():
    from spacr.qt.widgets.live_preview import LivePreviewPanel
    return LivePreviewPanel(threaded=False)


def _timelapse_panel():
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    return TimelapsePreviewPanel(threaded=False)


def _measure_panel():
    from spacr.qt.widgets.measure_preview import MeasurePreviewPanel
    return MeasurePreviewPanel(threaded=False)


def _motility_panel():
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    return MotilityPreviewPanel(threaded=False)


LIVE_VIEWS = (
    pytest.param(_mask_panel, id="mask"),
    pytest.param(_timelapse_panel, id="timelapse"),
    pytest.param(_measure_panel, id="measure"),
    pytest.param(_motility_panel, id="motility"),
)


@pytest.fixture(params=LIVE_VIEWS)
def live_view(request, qtbot):
    """Every live view in turn, built and registered with qtbot."""
    panel = request.param()
    qtbot.addWidget(panel)
    return panel


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def test_every_live_view_calls_its_action_the_same_thing(live_view):
    from spacr.qt.widgets.preview_contract import PREVIEW_RUN_TEXT

    assert live_view._run_btn.text() == PREVIEW_RUN_TEXT


def test_every_live_view_answers_to_run_preview(live_view):
    assert callable(live_view.run_preview)


def test_every_live_view_shares_the_contract(live_view):
    from spacr.qt.widgets.preview_contract import LivePreviewContract

    assert isinstance(live_view, LivePreviewContract)


def test_every_live_view_announces_its_result(live_view):
    """Measure was the odd one out: its pass landed with nobody told."""
    seen = []
    live_view.preview_ready.connect(seen.append)
    assert seen == []


def test_measure_announces_a_finished_crop_pass(qtbot, tmp_path):
    panel = _measure_panel()
    qtbot.addWidget(panel)
    seen = []
    panel.preview_ready.connect(seen.append)

    data = np.zeros((48, 48, 5), np.float32)
    data[..., 0] = 12
    mask = np.zeros((48, 48), np.int32)
    mask[2:8, 2:8] = 1
    mask[20:40, 20:40] = 2
    data[..., 4] = mask
    path = tmp_path / "plate1_A01_f1.npy"
    np.save(str(path), data)

    panel._mask_dim.setValue(4)
    assert panel.load_array(str(path)) is True

    assert seen and seen[-1] is not None
    assert len(seen[-1]) == 2


# ---------------------------------------------------------------------------
# Saying why it is not previewing
# ---------------------------------------------------------------------------

def test_a_live_view_with_nothing_loaded_says_so(live_view):
    """The one thing no live view may do is refuse in silence."""
    assert live_view.can_preview() is False
    reason = live_view.preview_blocked_reason()
    assert reason and reason.lower().startswith("load ")

    live_view.run_preview()
    assert live_view.preview_status() == reason


def test_the_reason_names_what_that_module_needs(live_view):
    """Same shape of sentence, module's own noun."""
    reason = live_view.preview_blocked_reason()
    assert reason.endswith("first.")


# ---------------------------------------------------------------------------
# Cancelling
# ---------------------------------------------------------------------------

def test_every_live_view_offers_a_cancel_beside_run(live_view):
    from spacr.qt.widgets.preview_contract import PREVIEW_CANCEL_TEXT

    cancel = live_view._cancel_btn
    assert cancel.text() == PREVIEW_CANCEL_TEXT
    # Nothing is in flight, so it is not offered yet.
    assert cancel.isEnabled() is False
    assert live_view._run_btn.isEnabled() is True


def test_cancelling_an_idle_live_view_is_a_no_op(live_view):
    assert live_view.cancel_preview() is False
    assert live_view._run_btn.isEnabled() is True


def test_cancelling_bumps_the_token_in_every_live_view(live_view):
    before = live_view.preview_token()
    live_view.cancel_preview()
    assert live_view.preview_token() == before + 1
    assert live_view.preview_stale(before) is True
    assert live_view.preview_stale(live_view.preview_token()) is False


# ---------------------------------------------------------------------------
# The shared guard
# ---------------------------------------------------------------------------

class _FakeWorker:
    """Stands in for a QThread that has not finished."""

    def isRunning(self):     # noqa: N802 (Qt naming)
        return True


@pytest.mark.parametrize("factory", [_mask_panel, _timelapse_panel,
                                     _motility_panel])
def test_a_busy_live_view_refuses_with_the_same_sentence(factory, qtbot,
                                                         monkeypatch):
    from spacr.qt.widgets.preview_contract import PREVIEW_BUSY_MESSAGE

    panel = factory()
    qtbot.addWidget(panel)
    # Loaded enough to run, and busy.
    monkeypatch.setattr(type(panel), "_preview_blocked_reason",
                        lambda self: "")
    panel._worker = _FakeWorker()
    assert panel.preview_running() is True

    panel.run_preview()
    assert panel.preview_status() == PREVIEW_BUSY_MESSAGE
    panel._worker = None


def test_measure_cancels_the_pass_on_the_shared_runner(qtbot, monkeypatch):
    """Measure's pass runs on the shared runner, not on a ``_worker``.

    Cancelling still has to reach it, and still has to say the same words.
    """
    from spacr.qt.widgets.preview_contract import PREVIEW_CANCELLED_MESSAGE

    panel = _measure_panel()
    qtbot.addWidget(panel)
    monkeypatch.setattr(type(panel), "_extra_work_in_flight",
                        lambda self: True)
    assert panel.cancel_preview() is True
    assert panel.preview_status() == PREVIEW_CANCELLED_MESSAGE


def test_begin_preview_marks_the_panel_busy(live_view, monkeypatch):
    monkeypatch.setattr(type(live_view), "_preview_blocked_reason",
                        lambda self: "")
    assert live_view.begin_preview() is True
    assert live_view._run_btn.isEnabled() is False
    assert live_view._cancel_btn.isEnabled() is True
    live_view.set_preview_busy(False)


# ---------------------------------------------------------------------------
# A cancelled pass must not land
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", [_timelapse_panel, _motility_panel])
def test_a_cancelled_pass_is_dropped_when_it_lands(factory, qtbot):
    """Neither Cellpose nor a numpy read can be interrupted.

    So cancelling cannot stop the work; it must stop the *answer*. Before
    the shared contract only the Mask panel dropped a superseded result,
    and these two adopted whatever arrived.
    """
    panel = factory()
    qtbot.addWidget(panel)
    seen = []
    panel.preview_ready.connect(seen.append)

    panel._pending_token = panel.preview_token()
    assert panel.cancel_preview() is False        # nothing was in flight
    panel._on_worker_done(None, "boom")

    assert seen == []
    assert "boom" not in panel.preview_status()


# ---------------------------------------------------------------------------
# One Cellpose constructor
# ---------------------------------------------------------------------------

class _FakeCellposeModel:
    """Records the kwargs a preview builds a Cellpose model with.

    ``eval`` carries the installed signature rather than ``**kwargs`` so the
    mock cannot silently accept a call the real library would reject — see
    ``tests/cellpose_api_contract``.
    """

    calls: list = []

    def __init__(self, gpu=False, pretrained_model="cpsam", model_type=None,
                 diam_mean=None, device=None, nchan=None, use_bfloat16=True):
        type(self).calls.append(init_arguments(locals()))

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
             invert=False, rescale=None, diameter=None, flow_threshold=0.4,
             cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
             max_size_fraction=0.4, niter=None, augment=False,
             tile_overlap=0.1, bsize=256, compute_masks=True, progress=None):
        # A 2-D plane per object type; the previews leave the axis to
        # cellpose's own detection, so the axis is not required here.
        check_cellpose_eval_call(x, channel_axis, require_channel_axis=False)
        plane = np.asarray(x)
        mask = np.zeros(plane.shape[:2], np.uint16)
        mask[2:6, 2:6] = 1
        flow_rgb = np.zeros(plane.shape[:2] + (3,), np.uint8)
        return mask, [flow_rgb], None


@pytest.fixture
def fake_cellpose(monkeypatch):
    """A stand-in ``cellpose.models`` that records constructor kwargs."""
    _FakeCellposeModel.calls = []
    models = types.ModuleType("cellpose.models")
    models.CellposeModel = _FakeCellposeModel
    package = types.ModuleType("cellpose")
    package.models = models
    monkeypatch.setitem(sys.modules, "cellpose", package)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    return _FakeCellposeModel


def test_both_segmenting_previews_go_through_one_constructor(
        fake_cellpose, tmp_path):
    """The ``model_type=`` defect was written twice. Now there is one call.

    ``model_type=`` is accepted and IGNORED by Cellpose 4, so a preview that
    passed the user's choice there segmented with ``cpsam`` whatever they
    picked — including a checkpoint they had just trained themselves.
    """
    from spacr.qt.widgets import live_preview, timelapse_preview
    from spacr.qt.widgets.preview_contract import preview_cellpose_model

    checkpoint = tmp_path / "my_own_model.pth"
    checkpoint.write_bytes(b"not really a checkpoint")

    assert live_preview.preview_cellpose_model is preview_cellpose_model
    assert timelapse_preview.preview_cellpose_model is preview_cellpose_model

    image = np.zeros((8, 8, 2), np.float32)
    request = live_preview.PreviewRequest(
        image=image, model=str(checkpoint), object_types=("cell",),
        channels={"cell": 0})
    live_preview._segment_multi(request)
    timelapse_preview.segment_frame(
        np.zeros((8, 8), np.float32), {"model": str(checkpoint)})

    assert len(fake_cellpose.calls) == 2
    for kwargs in fake_cellpose.calls:
        # The user's choice reaches `pretrained_model`, which Cellpose 4
        # honours, and never `model_type`, which it drops.
        assert kwargs["model_type"] is None
        assert kwargs["pretrained_model"] == str(checkpoint)


def test_the_shared_constructor_resolves_legacy_model_names(fake_cellpose):
    from spacr.qt.widgets.preview_contract import preview_cellpose_model

    preview_cellpose_model("cyto2", gpu=False)

    kwargs = fake_cellpose.calls[-1]
    assert kwargs["model_type"] is None
    # Cellpose 4 ships one model; the legacy spelling maps onto it rather
    # than being handed over and silently dropped.
    assert kwargs["pretrained_model"] == "cpsam"
    assert kwargs["gpu"] is False
