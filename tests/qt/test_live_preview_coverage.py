"""Behavioural tests for ``spacr.qt.widgets.live_preview``.

Companion to ``test_live_preview.py``. Where that file covers the happy
paths, this one drives the parts that used to be untested: the pure image
helpers at odd channel counts, ``_segment_multi`` against a fake Cellpose,
the ``_ZoomView`` event handlers, drag & drop, the settings-change →
recompute path, the Propagate toggle, and the worker lifecycle including
cancellation mid-compute.

Everything is CPU-only, offline and deterministic. The ONLY thing stubbed is
Cellpose itself (a GPU model download) — every other call goes into the real
product code and is asserted on by value.
"""
from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile

from PySide6.QtCore import QEvent, QPoint, QPointF, QThread, Qt, QMimeData, QUrl
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QFileDialog

from spacr.qt.widgets import live_preview as LP

from tests.cellpose_api_contract import (
    MISSING_CHANNEL_AXIS,
    emulate_pretrained_model,
    eval_arguments,
    init_arguments,
)
from tests.conftest import check_cellpose_eval_call


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _qapp(qapp):
    """Every test here can end up building a QPixmap, and QPixmap aborts the
    process outright when no QGuiApplication exists yet."""
    return qapp


@pytest.fixture
def gray_tif(tmp_path: Path) -> Path:
    """A 48x48 uint16 tile with a bright square, so masks have something
    to sit on and intensity filters have a real gradient to bite on."""
    arr = np.full((48, 48), 100, dtype=np.uint16)
    arr[8:24, 8:24] = 3000
    arr[30:36, 30:36] = 600
    p = tmp_path / "gray.tif"
    tifffile.imwrite(str(p), arr)
    return p


@pytest.fixture
def rgb_tif(tmp_path: Path) -> Path:
    arr = np.zeros((32, 32, 3), dtype=np.uint16)
    arr[..., 0] = 1000
    arr[..., 1] = 2000
    arr[..., 2] = 3000
    p = tmp_path / "rgb.tif"
    tifffile.imwrite(str(p), arr)
    return p


class _Evt:
    """Duck-typed stand-in for QDragEnterEvent / QDropEvent."""

    def __init__(self, mime):
        self._mime = mime
        self.accepted = False
        self.ignored = False

    def mimeData(self):
        return self._mime

    def acceptProposedAction(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


def _mime_for(*paths) -> QMimeData:
    m = QMimeData()
    m.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return m


def _panel(qtbot):
    p = LP.LivePreviewPanel()
    qtbot.addWidget(p)
    # The shipped default is "color (random)", which is right for a user --
    # a fixed colour is a coin flip against the image -- and useless for a
    # test asserting on rendered pixels. Every rendering test below was
    # relying on "auto" being the default without saying so; pin it here so
    # the dependency is explicit and the colours are reproducible.
    # `test_random_is_the_default_outline_colour` in
    # tests/qt/test_live_preview_view_sync.py owns the default itself.
    p._outline_colour.setCurrentText("auto")
    return p


def _wait_loaded(qtbot, panel, timeout=5000):
    """Pump the event loop until the panel's asynchronous load has landed.

    The GUI load paths return before the image exists -- that is the whole
    point of them -- so a test that drives one waits rather than asserting on
    the next line.
    """
    qtbot.waitUntil(lambda: not panel._image_loaders, timeout=timeout)
    qtbot.wait(10)


def _pixmap_pixel(view, x=0, y=0):
    """Read back an actual rendered pixel from a ``_ZoomView``."""
    item = view._pixmap_item
    assert item is not None, "view has no pixmap"
    img = item.pixmap().toImage()
    c = img.pixelColor(x, y)
    return (c.red(), c.green(), c.blue())


# ---------------------------------------------------------------------------
# Fake Cellpose — the only externality stubbed anywhere in this file
# ---------------------------------------------------------------------------

class _FakeCellposeModel:
    """Records how it was constructed and what images it was handed.

    Both signatures are the installed cellpose 4.0.7 ones, written out with
    the real defaults and no ``**kwargs``. ``loaded_model`` is what the real
    library would end up running: ``model_type=`` is warned about and dropped
    in v4.0.1+, so it is ``pretrained_model`` — defaulting to ``cpsam`` —
    whatever the caller asked for through ``model_type``.
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
        # One 2-D plane per object type, axis auto-detected by cellpose.
        check_cellpose_eval_call(x, channel_axis, require_channel_axis=False)
        recorded = eval_arguments(locals())
        image = x
        self.calls.append({"image": np.array(image, copy=True), **recorded})
        mask = np.zeros(image.shape[:2], dtype=np.uint16)
        # One object whose size encodes the channel mean, so tests can prove
        # which slice reached the model.
        mask[2:6, 2:6] = 1
        flow_rgb = np.full(image.shape[:2] + (3,),
                           int(np.clip(image.mean(), 0, 255)), dtype=np.uint8)
        return mask, [flow_rgb], None


@pytest.fixture
def fake_cellpose(monkeypatch):
    """Install a fake ``cellpose.models`` for the duration of one test."""
    _FakeCellposeModel.instances = []
    models = types.ModuleType("cellpose.models")
    models.CellposeModel = _FakeCellposeModel
    pkg = types.ModuleType("cellpose")
    pkg.models = models
    monkeypatch.setitem(sys.modules, "cellpose", pkg)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    return _FakeCellposeModel


# ===========================================================================
# Pure helpers
# ===========================================================================

class TestImageHelpers:
    def test_load_preview_image_reads_png_through_pil(self, tmp_path):
        from PIL import Image
        arr = np.zeros((6, 8), dtype=np.uint8)
        arr[2:4, 3:5] = 200
        Image.fromarray(arr).save(tmp_path / "tile.png")
        out = LP.load_preview_image(tmp_path / "tile.png")
        assert out.shape == (6, 8)
        assert np.array_equal(out, arr)

    def test_full_range_max_uses_dtype_max_for_integers(self):
        assert LP._full_range_max(np.zeros((2, 2), np.uint16)) == 65535.0
        assert LP._full_range_max(np.zeros((2, 2), np.uint8)) == 255.0

    def test_full_range_max_floats_assume_unit_range_until_exceeded(self):
        assert LP._full_range_max(np.array([[0.0, 0.5]])) == 1.0
        assert LP._full_range_max(np.array([[0.0, 4.0]])) == 4.0
        # An empty float array has no max to take; the unit range is the
        # documented fallback.
        assert LP._full_range_max(np.zeros((0, 0), np.float32)) == 1.0

    def test_raw_view_of_16bit_data_reads_dark(self):
        """Un-normalised means "map the full bit depth", not "clip to 255"."""
        img = np.full((4, 4), 32768, dtype=np.uint16)
        out = LP._to_uint8(img, normalise=False)
        assert out.dtype == np.uint8
        assert abs(int(out[0, 0]) - 128) <= 1

    def test_normalised_stretch_puts_percentiles_at_the_ends(self):
        img = np.linspace(0, 1000, 100, dtype=np.float32).reshape(10, 10)
        out = LP._to_uint8(img, normalise=True, lo_pct=2.0, hi_pct=98.0)
        assert out.min() == 0 and out.max() == 255

    def test_normalise_of_a_constant_image_is_all_black(self):
        # hi <= lo, so there is no stretch to apply.
        out = LP._to_uint8(np.full((5, 5), 7, np.uint16), normalise=True)
        assert out.shape == (5, 5)
        assert not out.any()

    def test_rgb_normalises_each_channel_independently(self):
        img = np.zeros((10, 10, 3), np.uint16)
        img[..., 0] = np.arange(100).reshape(10, 10)          # 0..99
        img[..., 1] = np.arange(100).reshape(10, 10) * 10     # 0..990
        img[..., 2] = 5                                        # constant
        out = LP._to_uint8(img, normalise=True)
        assert out.shape == (10, 10, 3)
        # Both varying channels stretch to the full range despite the 10x
        # difference in raw scale; the constant channel has no range so it
        # stays at zero.
        assert out[..., 0].max() == 255 and out[..., 1].max() == 255
        assert out[..., 2].max() == 0

    def test_rgb_raw_view_scales_by_bit_depth(self):
        img = np.zeros((4, 4, 3), np.uint16)
        img[..., 1] = 65535
        out = LP._to_uint8(img, normalise=False)
        assert out[0, 0, 0] == 0
        assert out[0, 0, 1] == 255

    def test_single_channel_axis_collapses_to_grayscale(self):
        """BUG FIX: (H, W, 1) used to fall through to the 2-D branch and come
        back with a trailing axis, which numpy_to_qpixmap then fed to Qt with
        a stride three times the real row length."""
        img = np.zeros((6, 6, 1), np.uint16)
        img[2:4] = 65535
        out = LP._to_uint8(img, normalise=False)
        assert out.shape == (6, 6)
        assert out[2, 0] == 255 and out[0, 0] == 0

    def test_more_than_four_channels_maps_the_first_three(self):
        """A 5-channel tile is a real spaCR shape; it used to escape the RGB
        branch entirely (only 2/3/4 were handled)."""
        img = np.zeros((4, 4, 5), np.uint16)
        img[..., 0] = 65535
        img[..., 3] = 65535      # ignored — beyond RGB
        out = LP._to_uint8(img, normalise=False)
        assert out.shape == (4, 4, 3)
        assert tuple(out[0, 0]) == (255, 0, 0)

    def test_boundary_mask_marks_the_outline_only(self):
        m = np.zeros((10, 10), np.int32)
        m[3:7, 3:7] = 1
        b = LP._boundary_mask(m)
        assert b[3, 3] and b[6, 6]      # corners of the square
        assert not b[0, 0]              # far background
        # The interior of a 4x4 square is empty (every pixel touches an edge),
        # so grow it and check a genuine interior pixel instead.
        m2 = np.zeros((10, 10), np.int32)
        m2[2:9, 2:9] = 1
        b2 = LP._boundary_mask(m2)
        assert not b2[5, 5]
        assert b2[2, 5]


class TestOverlay:
    def test_outline_colour_override_beats_the_per_object_colour(self):
        img = np.zeros((20, 20), np.uint8)
        mask = np.zeros((20, 20), np.int32)
        mask[5:15, 5:15] = 1
        out = LP.overlay_masks(img, {"cell": mask}, outline_rgb=(240, 60, 60))
        assert tuple(out[5, 10]) == (240, 60, 60)

    def test_thickness_is_clamped_to_five(self):
        img = np.zeros((30, 30), np.uint8)
        mask = np.zeros((30, 30), np.int32)
        mask[10:20, 10:20] = 1
        five = LP.overlay_masks(img, {"cell": mask}, outline_thickness=5)
        huge = LP.overlay_masks(img, {"cell": mask}, outline_thickness=99)
        assert np.array_equal(five, huge)

    def test_rgb_source_keeps_its_own_pixels_outside_the_outline(self):
        img = np.zeros((16, 16, 3), np.uint16)
        img[..., 2] = 65535               # solid blue tile
        mask = np.zeros((16, 16), np.int32)
        mask[4:8, 4:8] = 1
        out = LP.overlay_masks(img, {"cell": mask}, normalise=False)
        assert out.shape == (16, 16, 3)
        assert tuple(out[0, 0]) == (0, 0, 255)          # untouched blue
        assert tuple(out[4, 5]) == LP.OBJECT_COLORS["cell"]

    def test_empty_and_none_masks_are_skipped(self):
        img = np.zeros((8, 8), np.uint8)
        out = LP.overlay_masks(
            img, {"cell": None, "nucleus": np.zeros((8, 8), np.int32)})
        assert not out.any()

    def test_mask_from_a_differently_sized_image_is_ignored(self):
        """BUG FIX: a mask left over from a previous image used to raise
        ``IndexError: boolean index did not match indexed array`` here."""
        img = np.zeros((32, 32), np.uint8)
        stale = np.zeros((16, 16), np.int32)
        stale[4:8, 4:8] = 1
        good = np.zeros((32, 32), np.int32)
        good[20:28, 20:28] = 1
        out = LP.overlay_masks(img, {"nucleus": stale, "cell": good})
        assert out.shape == (32, 32, 3)
        assert tuple(out[20, 24]) == LP.OBJECT_COLORS["cell"]
        # Nothing from the stale mask leaked into the top-left corner.
        assert not out[:16, :16].any()

    def test_single_channel_source_image_does_not_break_the_overlay(self):
        """BUG FIX: (H, W, 1) gave ``rgb`` a single channel, and assigning a
        3-tuple colour into it raised a broadcast ValueError."""
        img = np.zeros((16, 16, 1), np.uint16)
        mask = np.zeros((16, 16), np.int32)
        mask[4:8, 4:8] = 1
        out = LP.overlay_masks(img, {"cell": mask})
        assert out.shape == (16, 16, 3)
        assert tuple(out[4, 5]) == LP.OBJECT_COLORS["cell"]

    def test_legacy_overlay_mask_shim_matches_the_cell_overlay(self):
        img = np.zeros((12, 12), np.uint8)
        mask = np.zeros((12, 12), np.int32)
        mask[3:9, 3:9] = 1
        assert np.array_equal(LP.overlay_mask(img, mask),
                              LP.overlay_masks(img, {"cell": mask}))


class TestNumpyToQPixmap:
    def test_grayscale_uint8_becomes_a_grey_rgb_pixmap(self):
        arr = np.zeros((4, 6), np.uint8)
        arr[1, 2] = 200
        pm = LP.numpy_to_qpixmap(arr)
        assert (pm.width(), pm.height()) == (6, 4)
        img = pm.toImage()
        c = img.pixelColor(2, 1)
        assert (c.red(), c.green(), c.blue()) == (200, 200, 200)

    def test_uint16_input_is_converted_through_to_uint8(self):
        arr = np.full((3, 3), 65535, np.uint16)
        pm = LP.numpy_to_qpixmap(arr, normalise=False)
        c = pm.toImage().pixelColor(0, 0)
        assert (c.red(), c.green(), c.blue()) == (255, 255, 255)

    @pytest.mark.parametrize("channels", [1, 2, 4, 5])
    def test_odd_channel_counts_render_without_reading_past_the_buffer(
            self, channels):
        """BUG FIX: the stride handed to QImage is ``w * 3``, so anything but
        three uint8 channels made Qt read ``h*w*3`` bytes out of an ``h*w*C``
        buffer — a heap over-read for C < 3 and garbage for C > 3."""
        arr = np.zeros((5, 7, channels), np.uint8)
        arr[..., 0] = 128
        pm = LP.numpy_to_qpixmap(arr)
        assert (pm.width(), pm.height()) == (7, 5)
        c = pm.toImage().pixelColor(0, 0)
        # Channel 0 always lands on red; a single channel greys out.
        assert c.red() == 128
        if channels == 1:
            assert (c.green(), c.blue()) == (128, 128)
        else:
            assert (c.green(), c.blue()) == (0, 0)


class TestSelectChannel:
    def test_picks_the_requested_channel_from_a_stack(self):
        img = np.zeros((4, 4, 3), np.uint16)
        img[..., 1] = 7
        assert np.array_equal(LP._select_channel(img, 1),
                              np.full((4, 4), 7, np.uint16))

    def test_channel_index_wraps_around(self):
        img = np.zeros((4, 4, 3), np.uint16)
        img[..., 2] = 9
        assert np.array_equal(LP._select_channel(img, 5), img[..., 2])

    def test_two_dimensional_image_is_returned_squeezed(self):
        img = np.arange(16, dtype=np.uint16).reshape(4, 4)
        assert np.array_equal(LP._select_channel(img, 3), img)


# ===========================================================================
# Post-segmentation filtering
# ===========================================================================

class TestApplySizeFilter:
    @staticmethod
    def _two_objects():
        m = np.zeros((20, 20), np.int32)
        m[2:4, 2:4] = 1          # 4 px
        m[8:16, 8:16] = 2        # 64 px
        return m

    def test_min_area_keeps_only_the_big_object(self):
        out = LP._apply_size_filter(
            self._two_objects(), {"cell_min_area": 10}, "cell")
        assert int((out > 0).sum()) == 64
        assert sorted(np.unique(out)) == [0, 1]

    def test_settings_present_but_all_neutral_is_an_identity(self):
        mask = self._two_objects()
        out = LP._apply_size_filter(
            mask,
            {"cell_min_area": 0, "cell_max_area": 0,
             "cell_remove_border_objects": False,
             "cell_min_intensity_percentile": 0,
             "cell_max_intensity_percentile": 100},
            "cell")
        assert out is mask

    def test_none_mask_is_returned_untouched(self):
        assert LP._apply_size_filter(None, {"cell_min_area": 5}, "cell") is None

    def test_unparseable_values_fall_back_to_the_default(self):
        """A stray string in the settings dict must not take the filter out;
        it falls back to the neutral default, which here means only the
        max_area rule survives."""
        out = LP._apply_size_filter(
            self._two_objects(),
            {"cell_min_area": "not-a-number", "cell_max_area": 10}, "cell")
        assert int((out > 0).sum()) == 4          # 64 px object removed

    def test_none_value_falls_back_to_the_default(self):
        out = LP._apply_size_filter(
            self._two_objects(),
            {"cell_min_area": None, "cell_max_area": 10}, "cell")
        assert int((out > 0).sum()) == 4

    def test_legacy_min_size_key_is_honoured_as_a_fallback(self):
        out = LP._apply_size_filter(
            self._two_objects(), {"cell_min_size": 10}, "cell")
        assert int((out > 0).sum()) == 64

    def test_remove_border_drops_edge_touching_objects(self):
        m = np.zeros((20, 20), np.int32)
        m[0:5, 0:5] = 1          # touches the top-left edge
        m[8:14, 8:14] = 2        # interior
        out = LP._apply_size_filter(
            m, {"cell_remove_border_objects": True}, "cell")
        assert int((out > 0).sum()) == 36
        assert out[0, 0] == 0

    def test_intensity_percentile_drops_the_dim_object(self):
        m = np.zeros((20, 20), np.int32)
        m[2:6, 2:6] = 1
        m[10:14, 10:14] = 2
        inten = np.zeros((20, 20), np.float32)
        inten[2:6, 2:6] = 10.0        # dim
        inten[10:14, 10:14] = 900.0   # bright
        out = LP._apply_size_filter(
            m, {"cell_min_intensity_percentile": 50}, "cell",
            intensity_img=inten)
        surviving = out[10:14, 10:14]
        assert (surviving > 0).all()
        assert not out[2:6, 2:6].any()

    def test_a_failing_filter_returns_the_unfiltered_mask(self, monkeypatch):
        import spacr.utils as SU
        mask = self._two_objects()

        def _boom(*a, **k):
            raise RuntimeError("filter exploded")
        monkeypatch.setattr(SU, "_filter_objects", _boom)
        out = LP._apply_size_filter(mask, {"cell_min_area": 10}, "cell")
        assert np.array_equal(out, mask)


# ===========================================================================
# _segment_multi against a fake Cellpose
# ===========================================================================

class TestSegmentMulti:
    def _req(self, **kw):
        img = np.zeros((16, 16, 3), np.uint16)
        img[..., 0] = 100
        img[..., 1] = 200
        img[..., 2] = 300
        kw.setdefault("image", img)
        return LP.PreviewRequest(**kw)

    def test_cpsam_is_loaded_as_a_pretrained_model(self, fake_cellpose):
        masks, flows = LP._segment_multi(self._req(model="cpsam"))
        model = fake_cellpose.instances[-1]
        assert model.kwargs["pretrained_model"] == "cpsam"
        # model_type= is in cellpose 4's signature but is warned about and
        # dropped, so "not passed" now means "left at the library default".
        assert model.kwargs["model_type"] is None
        assert model.loaded_model == "cpsam"
        assert set(masks) == {"cell"}
        assert masks["cell"].dtype == np.int32
        assert int(masks["cell"].max()) == 1

    def test_a_trained_checkpoint_reaches_the_weights(self, fake_cellpose,
                                                      tmp_path):
        """A model the user trained must be the model the preview runs.

        This was pinned as "choosing cyto2 must load something other than
        cpsam", which cellpose 4 cannot satisfy: cyto2 is gone, and
        `_resolve_cellpose_pretrained` maps the legacy names onto cpsam on
        purpose instead of pretending. The demand was for a bug.

        What genuinely was broken is here: the panel passed the name as
        `model_type=`, which cellpose 4 accepts and ignores, so a
        checkpoint trained in spaCR's own Train Cellpose module was
        discarded and the stock weights ran instead.
        """
        checkpoint = tmp_path / "my_finetuned_model"
        checkpoint.write_bytes(b"not really weights, but it exists")
        LP._segment_multi(self._req(model=str(checkpoint)))
        assert fake_cellpose.instances[-1].loaded_model == str(checkpoint)

    def test_a_legacy_name_resolves_to_cpsam_rather_than_pretending(
            self, fake_cellpose):
        """The other half, so the fix is not undone by passing names through."""
        LP._segment_multi(self._req(model="cyto2"))
        assert fake_cellpose.instances[-1].loaded_model == "cpsam"

    @pytest.mark.parametrize("available", [True, False])
    def test_gpu_flag_follows_torch(self, fake_cellpose, monkeypatch,
                                    available):
        """Stubbed both ways so the assertion does not depend on whether the
        machine running the suite happens to have a GPU."""
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: available)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        LP._segment_multi(self._req())
        assert fake_cellpose.instances[-1].kwargs["gpu"] is available

    def test_gpu_falls_back_to_false_when_torch_is_unusable(
            self, fake_cellpose, monkeypatch):
        broken = types.ModuleType("torch")   # no .cuda at all
        monkeypatch.setitem(sys.modules, "torch", broken)
        LP._segment_multi(self._req())
        assert fake_cellpose.instances[-1].kwargs["gpu"] is False

    def test_each_object_gets_its_own_channel(self, fake_cellpose):
        req = self._req(object_types=("cell", "nucleus"),
                        channels={"cell": 0, "nucleus": 2})
        masks, _ = LP._segment_multi(req)
        assert set(masks) == {"cell", "nucleus"}
        calls = fake_cellpose.instances[-1].calls
        assert len(calls) == 2
        assert calls[0]["image"].mean() == pytest.approx(100.0)
        assert calls[1]["image"].mean() == pytest.approx(300.0)

    def test_tuning_parameters_reach_the_model(self, fake_cellpose):
        LP._segment_multi(self._req(diameter=42.5, flow_threshold=0.7,
                                    cellprob=-1.25))
        call = fake_cellpose.instances[-1].calls[0]
        assert call["diameter"] == pytest.approx(42.5)
        assert call["flow_threshold"] == pytest.approx(0.7)
        assert call["cellprob_threshold"] == pytest.approx(-1.25)

    def test_zero_diameter_means_auto(self, fake_cellpose):
        LP._segment_multi(self._req(diameter=0))
        assert fake_cellpose.instances[-1].calls[0]["diameter"] is None

    def test_background_is_thresholded_not_subtracted(self, fake_cellpose):
        """What is above the background keeps its value.

        This used to assert 150.0 -- the preview subtracted the background
        and clipped at zero. The pipeline thresholds instead
        (``single_channel[single_channel < background] = 0`` in
        :func:`spacr.io._normalize_img_batch`), and subtraction is a
        different image: it moves every bright pixel down.
        """
        req = self._req(
            channels={"cell": 1},
            preprocess_settings={"remove_background_cell": True,
                                 "cell_background": 50})
        LP._segment_multi(req)
        # Channel 1 is a flat 200, which is above 50, so it is left alone.
        assert fake_cellpose.instances[-1].calls[0]["image"].mean() == \
            pytest.approx(200.0)

    def test_everything_below_the_background_is_zeroed(self, fake_cellpose):
        req = self._req(
            channels={"cell": 0},
            preprocess_settings={"remove_background_cell": True,
                                 "cell_background": 5000})
        LP._segment_multi(req)
        assert fake_cellpose.instances[-1].calls[0]["image"].max() == 0

    def test_the_background_value_is_read_per_object(self, fake_cellpose):
        """`{obj}_background`, not a plain `background`.

        The worker read `background`, which nothing writes -- the panel
        emits `{obj}_background` -- so the value was always the 100.0
        default and moving the spinbox did nothing.
        """
        req = self._req(
            channels={"cell": 1},
            preprocess_settings={"remove_background_cell": True,
                                 "cell_background": 5000,
                                 "background": 1})
        LP._segment_multi(req)
        assert fake_cellpose.instances[-1].calls[0]["image"].max() == 0, (
            "the per-object key must win over the generic one")

    def test_flows_are_captured_per_object(self, fake_cellpose):
        _, flows = LP._segment_multi(self._req(channels={"cell": 2}))
        assert set(flows) == {"cell"}
        assert flows["cell"].shape == (16, 16, 3)
        # The fake encodes the channel mean (300 -> clipped to 255).
        assert int(flows["cell"][0, 0, 0]) == 255

    def test_a_nested_flow_list_is_unwrapped(self, fake_cellpose, monkeypatch):
        """Cellpose hands back ``flows`` as a per-image list, so for a batched
        call ``flows[0]`` is itself ``[rgb, dP, cellprob, ...]``."""
        rgb = np.full((16, 16, 3), 77, np.uint8)

        def _eval(self, image, **kw):
            dP = np.zeros((2,) + image.shape[:2], np.float32)
            return np.zeros(image.shape[:2], np.uint16), [[rgb, dP]], None
        monkeypatch.setattr(fake_cellpose, "eval", _eval, raising=True)
        _, flows = LP._segment_multi(self._req())
        assert np.array_equal(flows["cell"], rgb)

    def test_a_list_of_masks_is_unwrapped(self, fake_cellpose, monkeypatch):
        def _eval(self, image, **kw):
            return [np.ones(image.shape[:2], np.uint16)], [None], None
        monkeypatch.setattr(fake_cellpose, "eval", _eval, raising=True)
        masks, _ = LP._segment_multi(self._req())
        assert masks["cell"].shape == (16, 16)
        assert int(masks["cell"].max()) == 1

    def test_a_model_without_flows_still_returns_masks(
            self, fake_cellpose, monkeypatch):
        def _eval(self, image, **kw):
            return (np.zeros(image.shape[:2], np.uint16),)   # no flows entry
        monkeypatch.setattr(fake_cellpose, "eval", _eval, raising=True)
        masks, flows = LP._segment_multi(self._req())
        assert "cell" in masks
        assert flows == {}


# ===========================================================================
# Worker
# ===========================================================================

class TestPreviewWorker:
    def _drain(self, worker):
        got = {}
        worker.finished_masks.connect(
            lambda m, e, t: got.setdefault("masks", (m, e, t)))
        worker.flows_ready.connect(
            lambda f, t: got.setdefault("flows", (f, t)))
        return got

    def test_run_emits_masks_and_flows_with_the_token(self, monkeypatch):
        mask = np.ones((4, 4), np.int32)
        flow = np.zeros((4, 4, 3), np.uint8)
        monkeypatch.setattr(LP, "_segment_multi",
                            lambda req: ({"cell": mask}, {"cell": flow}))
        w = LP._PreviewWorker(LP.PreviewRequest(image=np.zeros((4, 4))),
                              token=7)
        got = self._drain(w)
        w.run()                      # run the body inline — no thread needed
        assert got["masks"][0]["cell"] is mask
        assert got["masks"][1] == ""
        assert got["masks"][2] == 7
        assert got["flows"] == ({"cell": flow}, 7)

    def test_a_masks_only_return_yields_empty_flows(self, monkeypatch):
        monkeypatch.setattr(
            LP, "_segment_multi",
            lambda req: {"cell": np.zeros((4, 4), np.int32)})
        w = LP._PreviewWorker(LP.PreviewRequest(image=np.zeros((4, 4))))
        got = self._drain(w)
        w.run()
        assert got["flows"][0] == {}

    def test_a_failure_is_reported_as_an_error_string(self, monkeypatch):
        def _boom(req):
            raise ValueError("no model")
        monkeypatch.setattr(LP, "_segment_multi", _boom)
        w = LP._PreviewWorker(LP.PreviewRequest(image=np.zeros((4, 4))),
                              token=3)
        got = self._drain(w)
        w.run()
        assert got["masks"][0] is None
        assert "no model" in got["masks"][1]
        assert got["masks"][2] == 3
        assert "flows" not in got


# ===========================================================================
# _ZoomView
# ===========================================================================

class TestZoomView:
    def _pair(self, qtbot):
        a, b = LP._ZoomView(), LP._ZoomView()
        qtbot.addWidget(a)
        qtbot.addWidget(b)
        a.set_peer(b)
        b.set_peer(a)
        pm = LP.numpy_to_qpixmap(np.zeros((40, 40), np.uint8))
        a.set_pixmap(pm)
        b.set_pixmap(pm)
        return a, b

    def test_set_pixmap_resets_zoom_and_fits(self, qtbot):
        a, _ = self._pair(qtbot)
        assert a.scale_factor() == 1.0
        assert a._user_zoomed is False
        assert a._scene.sceneRect().width() == 40

    def test_zoom_broadcasts_to_the_peer_exactly_once(self, qtbot):
        a, b = self._pair(qtbot)
        seen = []
        a.zoom_changed.connect(seen.append)
        b.zoom_changed.connect(seen.append)
        a._apply_zoom(2.0, broadcast=True)
        assert a.scale_factor() == pytest.approx(2.0)
        # BUG FIX: the broadcast used to set the PEER's ``_syncing`` flag
        # before calling into it, and that flag makes ``_apply_zoom`` return
        # immediately — so the second canvas never tracked the first at all.
        assert b.scale_factor() == pytest.approx(2.0)
        assert seen == [pytest.approx(2.0), pytest.approx(2.0)]
        # Both guards are clear again, so the next gesture from either view
        # still propagates.
        assert a._syncing is False and b._syncing is False
        b._apply_zoom(0.5, broadcast=True)
        assert a.scale_factor() == pytest.approx(1.0)
        assert b.scale_factor() == pytest.approx(1.0)

    def test_a_syncing_view_ignores_zoom_requests(self, qtbot):
        a, _ = self._pair(qtbot)
        a._syncing = True
        a._apply_zoom(3.0)
        assert a.scale_factor() == 1.0

    def test_reset_zoom_returns_to_fit(self, qtbot):
        a, _ = self._pair(qtbot)
        a._apply_zoom(2.5)
        assert a._user_zoomed is True
        a.reset_zoom()
        assert a.scale_factor() == 1.0
        assert a._user_zoomed is False

    def test_reset_zoom_without_a_pixmap_is_safe(self, qtbot):
        v = LP._ZoomView()
        qtbot.addWidget(v)
        v.reset_zoom()
        assert v.scale_factor() == 1.0

    def test_plain_wheel_zooms_both_views(self, qtbot):
        a, b = self._pair(qtbot)
        up = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0),
                         QPoint(0, 120), Qt.NoButton, Qt.NoModifier,
                         Qt.ScrollUpdate, False)
        a.wheelEvent(up)
        assert a.scale_factor() == pytest.approx(1.20)
        assert b.scale_factor() == pytest.approx(1.20)
        down = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0),
                           QPoint(0, -120), Qt.NoButton, Qt.NoModifier,
                           Qt.ScrollUpdate, False)
        a.wheelEvent(down)
        assert a.scale_factor() == pytest.approx(1.20 * 0.833)

    def test_shift_wheel_scrolls_instead_of_zooming(self, qtbot):
        a, _ = self._pair(qtbot)
        ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0),
                         QPoint(0, 120), Qt.NoButton, Qt.ShiftModifier,
                         Qt.ScrollUpdate, False)
        a.wheelEvent(ev)
        assert a.scale_factor() == 1.0

    def test_resize_refits_until_the_user_zooms(self, qtbot, qapp):
        a, _ = self._pair(qtbot)
        a.resize(200, 200)
        a.show()
        qapp.processEvents()
        small = a.transform().m11()
        a.resize(400, 400)
        qapp.processEvents()
        # A 40 px tile fitted into twice the canvas is drawn twice as large.
        assert a.transform().m11() == pytest.approx(2 * small, rel=0.05)
        a._apply_zoom(2.0)
        fixed = a.transform().m11()
        a.resize(700, 700)
        qapp.processEvents()
        assert a.transform().m11() == pytest.approx(fixed)   # user zoom kept

    def test_mouse_move_reports_image_coordinates(self, qtbot):
        a, _ = self._pair(qtbot)
        a.resize(40, 40)
        a.resetTransform()
        seen = []
        a.hover_pixel.connect(lambda x, y: seen.append((x, y)))
        a.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, QPointF(11, 13),
                                     Qt.NoButton, Qt.NoButton, Qt.NoModifier))
        assert len(seen) == 1
        pt = a.mapToScene(QPoint(11, 13))
        assert seen[0] == (int(pt.x()), int(pt.y()))

    def test_mouse_move_without_a_pixmap_emits_nothing(self, qtbot):
        v = LP._ZoomView()
        qtbot.addWidget(v)
        seen = []
        v.hover_pixel.connect(lambda x, y: seen.append((x, y)))
        v.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, QPointF(1, 1),
                                     Qt.NoButton, Qt.NoButton, Qt.NoModifier))
        assert seen == []


# ===========================================================================
# Drag & drop
# ===========================================================================

class TestDragAndDrop:
    def test_a_dropped_tif_is_loaded(self, qtbot, gray_tif):
        p = _panel(qtbot)
        enter = _Evt(_mime_for(gray_tif))
        p.dragEnterEvent(enter)
        assert enter.accepted and not enter.ignored
        move = _Evt(_mime_for(gray_tif))
        p.dragMoveEvent(move)
        assert move.accepted
        drop = _Evt(_mime_for(gray_tif))
        p.dropEvent(drop)
        assert drop.accepted
        _wait_loaded(qtbot, p)
        assert p._image is not None and p._image.shape == (48, 48)
        assert p._path_label.text() == str(gray_tif)

    def test_a_non_image_drop_is_refused_by_every_handler(self, qtbot,
                                                          tmp_path):
        p = _panel(qtbot)
        doc = tmp_path / "notes.txt"
        doc.write_text("x")
        for handler in (p.dragEnterEvent, p.dragMoveEvent, p.dropEvent):
            ev = _Evt(_mime_for(doc))
            handler(ev)
            assert ev.ignored and not ev.accepted
        assert p._image is None

    def test_a_drop_without_urls_is_refused(self, qtbot):
        p = _panel(qtbot)
        ev = _Evt(QMimeData())
        p.dragEnterEvent(ev)
        assert ev.ignored

    def test_a_remote_url_is_not_treated_as_a_file(self, qtbot):
        p = _panel(qtbot)
        m = QMimeData()
        m.setUrls([QUrl("https://example.com/tile.tif")])
        ev = _Evt(m)
        p.dragEnterEvent(ev)
        assert ev.ignored

    def test_the_first_supported_file_in_a_multi_drop_wins(self, qtbot,
                                                           tmp_path, gray_tif):
        p = _panel(qtbot)
        doc = tmp_path / "readme.md"
        doc.write_text("x")
        assert p._dropped_image_path(_Evt(_mime_for(doc, gray_tif))) == \
            str(gray_tif)

    def test_the_canvases_do_not_swallow_drops(self, qtbot):
        p = _panel(qtbot)
        assert p.acceptDrops() is True
        assert p._src_view.acceptDrops() is False
        assert p._mask_view.acceptDrops() is False


# ===========================================================================
# Panel: loading, hover, pick
# ===========================================================================

class TestPanelIO:
    def test_pick_file_loads_the_chosen_path(self, qtbot, monkeypatch,
                                             gray_tif):
        p = _panel(qtbot)
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: (str(gray_tif), "")))
        p._pick_file()
        _wait_loaded(qtbot, p)
        assert p._image is not None
        assert p._path_label.text() == str(gray_tif)

    def test_pick_file_cancelled_loads_nothing(self, qtbot, monkeypatch):
        p = _panel(qtbot)
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: ("", "")))
        p._pick_file()
        assert p._image is None

    def test_loading_a_new_image_drops_everything_derived_from_the_old_one(
            self, qtbot, gray_tif, rgb_tif):
        """BUG FIX: ``_raw_masks`` and ``_flows`` used to survive a reload, so
        the next filter change re-ran the old image's masks through
        ``overlay_masks`` and raised IndexError once the sizes differed."""
        p = _panel(qtbot)
        p.load_image(gray_tif)
        p._raw_masks = {"cell": np.ones((48, 48), np.int32)}
        p._masks = {"cell": np.ones((48, 48), np.int32)}
        p._flows = {"cell": np.zeros((48, 48, 3), np.uint8)}
        assert p.load_image(rgb_tif) is True
        assert p._raw_masks == {} and p._masks == {} and p._flows == {}
        # And the recompute that a filter change triggers is now a no-op
        # rather than a crash.
        p._compartment_widgets["cell"]["min_area"].setValue(5)
        assert p._masks == {}

    def test_hover_before_an_image_is_loaded_is_a_no_op(self, qtbot):
        p = _panel(qtbot)
        before = p._hover_label.text()
        p._on_hover(3, 3)
        assert p._hover_label.text() == before

    def test_hover_on_a_multichannel_image_lists_every_channel(self, qtbot,
                                                               rgb_tif):
        p = _panel(qtbot)
        p.load_image(rgb_tif)
        p._on_hover(4, 5)
        assert "channels=(1000, 2000, 3000)" in p._hover_label.text()
        assert "x=   4" in p._hover_label.text()

    def test_hover_ignores_empty_and_missing_masks(self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        p._masks = {"cell": None, "nucleus": np.zeros((0, 0), np.int32)}
        p._on_hover(10, 10)
        text = p._hover_label.text()
        assert "intensity=3000" in text
        assert "#" not in text

    def test_hover_reports_the_object_area(self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        m = np.zeros((48, 48), np.int32)
        m[10:20, 10:20] = 4
        p._masks = {"cell": m}
        p._on_hover(12, 12)
        assert "cell=#4 area=100px" in p._hover_label.text()

    def test_hover_outside_a_smaller_mask_reports_intensity_only(
            self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        p._masks = {"cell": np.ones((4, 4), np.int32)}
        p._on_hover(40, 40)
        assert "cell=" not in p._hover_label.text()
        assert "intensity=" in p._hover_label.text()

    def test_hover_keeps_scanning_past_a_mask_that_does_not_cover_the_pixel(
            self, qtbot, gray_tif):
        """The out-of-range mask must not end the scan — the nucleus mask
        behind it still has to be reported."""
        p = _panel(qtbot)
        p.load_image(gray_tif)
        nuc = np.zeros((48, 48), np.int32)
        nuc[38:44, 38:44] = 3
        p._masks = {
            "cell": np.ones((4, 4), np.int32),        # too small to reach
            "pathogen": np.zeros((48, 48), np.int32),  # covers it, but is bg
            "nucleus": nuc,
        }
        p._on_hover(40, 40)
        text = p._hover_label.text()
        assert "cell=" not in text and "pathogen=" not in text
        assert "nucleus=#3 area=36px" in text


# ===========================================================================
# Panel: rendering
# ===========================================================================

class TestPanelRendering:
    def test_refresh_without_an_image_is_a_no_op(self, qtbot):
        p = _panel(qtbot)
        p._refresh_canvases()
        assert p._src_view._pixmap_item is None

    def test_both_canvases_show_the_source_until_masks_exist(self, qtbot,
                                                             gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        assert _pixmap_pixel(p._src_view, 0, 0) == \
            _pixmap_pixel(p._mask_view, 0, 0)

    def test_overlay_mode_draws_outlines_on_the_right_canvas(self, qtbot,
                                                             gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        m = np.zeros((48, 48), np.int32)
        m[10:20, 10:20] = 1
        p._masks = {"cell": m}
        p._refresh_canvases()
        # 'auto' draws a random colour now, not the compartment's fixed green.
        assert _pixmap_pixel(p._mask_view, 10, 10) == \
            p._auto_outline_colour("cell")

    def test_outline_colour_choice_repaints_the_overlay(self, qtbot,
                                                        gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        m = np.zeros((48, 48), np.int32)
        m[10:20, 10:20] = 1
        p._masks = {"cell": m}
        assert p._outline_rgb() is None                 # 'auto'
        p._outline_colour.setCurrentText("red")         # fires _refresh
        assert p._outline_rgb() == (240, 60, 60)
        assert _pixmap_pixel(p._mask_view, 10, 10) == (240, 60, 60)

    def test_random_outline_choice_repaints_each_label_stably(self, qtbot,
                                                               gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        mask = np.zeros((48, 48), np.int32)
        mask[5:14, 5:14] = 1
        mask[30:42, 30:42] = 2
        p._masks = {"cell": mask}

        p._outline_colour.setCurrentText("color (random)")
        first = (
            _pixmap_pixel(p._mask_view, 5, 9),
            _pixmap_pixel(p._mask_view, 30, 35),
        )
        assert first[0] != first[1]
        assert first[0] != (0, 0, 0)
        assert first[1] != (0, 0, 0)

        p._refresh_canvases()
        second = (
            _pixmap_pixel(p._mask_view, 5, 9),
            _pixmap_pixel(p._mask_view, 30, 35),
        )
        assert second == first

    def test_masks_mode_shows_labels_not_outlines(self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        m = np.zeros((48, 48), np.int32)
        m[10:20, 10:20] = 1
        p._masks = {"cell": m}
        p._view_mode.setCurrentText("Masks")
        # Interior of the object is filled, not just its boundary.
        assert _pixmap_pixel(p._mask_view, 15, 15) != (0, 0, 0)
        assert _pixmap_pixel(p._mask_view, 0, 0) == (0, 0, 0)

    def test_label_rgb_skips_missing_wrong_sized_and_empty_masks(
            self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        p._masks = {
            "cell": None,
            "nucleus": np.ones((8, 8), np.int32),        # wrong size
            "pathogen": np.zeros((48, 48), np.int32),    # empty
        }
        assert not p._label_rgb().any()

    def test_label_rgb_shades_labels_within_the_object_colour(self, qtbot,
                                                              gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        m = np.zeros((48, 48), np.int32)
        m[4:8, 4:8] = 1
        m[20:24, 20:24] = 2
        p._masks = {"nucleus": m}
        rgb = p._label_rgb()
        # Under 'auto' the Masks view is tinted with the run's random colour.
        base = np.array(p._auto_outline_colour("nucleus"))
        for y, x, lbl in ((5, 5, 1), (21, 21, 2)):
            shade = 0.5 + 0.5 * ((lbl % 7) / 6.0)
            assert np.allclose(rgb[y, x], np.clip(base * shade, 0, 255)
                               .astype(np.uint8), atol=1)
        assert (rgb[0, 0] == 0).all()

    def test_flows_mode_renders_the_flow_image(self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        flow = np.zeros((48, 48, 3), np.uint8)
        flow[..., 0] = 210
        p._view_mode.setCurrentText("Flows")
        p._on_flows_ready({"cell": flow})
        assert p._flows["cell"] is flow
        assert _pixmap_pixel(p._mask_view, 0, 0) == (210, 0, 0)

    def test_flows_ready_while_another_mode_is_showing_does_not_repaint(
            self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        before = _pixmap_pixel(p._mask_view, 0, 0)
        p._on_flows_ready({"cell": np.full((48, 48, 3), 200, np.uint8)})
        assert _pixmap_pixel(p._mask_view, 0, 0) == before

    def test_flows_rgb_without_flows_is_black(self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        out = p._flows_rgb()
        assert out.shape == (48, 48, 3)
        assert not out.any()

    def test_flows_rgb_max_blends_and_skips_non_rgb_entries(self, qtbot,
                                                            gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        a = np.zeros((48, 48, 3), np.uint8); a[..., 0] = 40
        b = np.zeros((48, 48, 3), np.uint8); b[..., 0] = 90; b[..., 1] = 10
        p._flows = {"cell": a, "nucleus": b,
                    "pathogen": np.zeros((48, 48), np.uint8),   # 2-D, skipped
                    "organelle": None}
        out = p._flows_rgb()
        assert tuple(out[0, 0]) == (90, 10, 0)

    def test_flows_rgb_ignores_a_mismatched_shape(self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        a = np.full((48, 48, 3), 30, np.uint8)
        p._flows = {"cell": a, "nucleus": np.full((8, 8, 3), 200, np.uint8)}
        out = p._flows_rgb()
        assert out.shape == (48, 48, 3)
        assert int(out.max()) == 30

    def test_flows_mode_with_no_flows_falls_back_to_the_overlay(self, qtbot,
                                                                gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        m = np.zeros((48, 48), np.int32)
        m[10:20, 10:20] = 1
        p._masks = {"cell": m}
        p._view_mode.setCurrentText("Flows")     # but _flows is empty
        assert _pixmap_pixel(p._mask_view, 10, 10) == \
            p._auto_outline_colour("cell")

    @pytest.mark.parametrize("channels", [1, 5])
    def test_a_tif_with_an_odd_channel_count_loads_and_renders(
            self, qtbot, tmp_path, channels):
        """End-to-end for the channel-count fix: tifffile round-trips both
        shapes faithfully, so both reach ``_refresh_canvases`` from a plain
        Choose-image."""
        arr = np.zeros((20, 24, channels), np.uint16)
        arr[..., 0] = 65535
        path = tmp_path / f"c{channels}.tif"
        tifffile.imwrite(str(path), arr)
        p = _panel(qtbot)
        p._normalise_check.setChecked(False)   # raw view: 65535 -> 255
        assert p.load_image(path) is True
        assert p._image.shape == (20, 24, channels)
        assert p._src_view._pixmap_item.pixmap().size().toTuple() == (24, 20)
        r, g, b = _pixmap_pixel(p._src_view, 0, 0)
        assert r == 255
        assert (g, b) == ((255, 255) if channels == 1 else (0, 0))
        # And an overlay on top of it still lands.
        m = np.zeros((20, 24), np.int32)
        m[5:12, 5:12] = 1
        p._masks = {"cell": m}
        p._refresh_canvases()
        assert _pixmap_pixel(p._mask_view, 5, 5) == \
            p._auto_outline_colour("cell")

    def test_normalise_toggle_repaints_the_source_canvas(self, qtbot,
                                                         gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        # Raw: the 3000/65535 square reads dark. Normalised: it saturates.
        p._normalise_check.setChecked(False)
        raw = _pixmap_pixel(p._src_view, 10, 10)
        p._normalise_check.setChecked(True)
        norm = _pixmap_pixel(p._src_view, 10, 10)
        assert raw[0] < 20
        assert norm[0] == 255

    def test_percentile_spinners_repaint(self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        p._lo_pct.setValue(0.0)
        p._hi_pct.setValue(100.0)
        assert _pixmap_pixel(p._src_view, 10, 10)[0] == 255
        assert _pixmap_pixel(p._src_view, 0, 0)[0] == 0


# ===========================================================================
# Panel: settings plumbing
# ===========================================================================

class TestPanelSettings:
    def test_apply_settings_survives_a_junk_value(self, qtbot):
        p = _panel(qtbot)
        p._diameter.setValue(30.0)
        p.apply_settings({"diameter": "thirty", "flow_threshold": 0.9})
        # The exception aborts the copy, but the panel stays usable and the
        # whole dict is still cached for the request builder.
        assert p._diameter.value() == pytest.approx(30.0)
        assert p._settings["flow_threshold"] == 0.9

    def test_apply_settings_ignores_none_channels_and_unknown_models(
            self, qtbot):
        p = _panel(qtbot)
        p._cell_channel.setValue(3)
        p.apply_settings({"cell_channel": None, "nucleus_channel": None,
                          "model_name": "not-a-model"})
        assert p._cell_channel.value() == 3
        assert p._model_box.currentText() == "cpsam"

    def test_object_type_cell_plus_nucleus_expands_to_two(self, qtbot):
        p = _panel(qtbot)
        p._object_box.setCurrentText("cell + nucleus")
        assert p._selected_object_types() == ("cell", "nucleus")
        assert p._primary_object() == "cell"
        p._object_box.setCurrentText("pathogen")
        assert p._selected_object_types() == ("pathogen",)

    def test_obj_channel_routes_per_compartment(self, qtbot):
        p = _panel(qtbot)
        p._cell_channel.setValue(2)
        p._nucleus_channel.setValue(5)
        assert p._obj_channel("cell") == 2
        assert p._obj_channel("nucleus") == 5
        assert p._obj_channel("pathogen") == 0
        assert p._obj_channel("organelle") == 0

    def test_build_request_carries_widget_state(self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        p._model_box.setCurrentText("cyto3")
        p._diameter.setValue(17.0)
        p._flow.setValue(0.15)
        p._prob.setValue(-2.0)
        p._cell_channel.setValue(1)
        p._nucleus_channel.setValue(2)
        p._object_box.setCurrentText("cell + nucleus")
        p.apply_settings({"some_pipeline_key": "kept"})
        req = p._build_request()
        assert req.model == "cyto3"
        assert req.diameter == pytest.approx(17.0)
        assert req.flow_threshold == pytest.approx(0.15)
        assert req.cellprob == pytest.approx(-2.0)
        # Every object type carries a channel now, not just the two the
        # panel used to expose. `channels.get(obj, 0)` used to fall back to
        # 0 for pathogen and organelle, so selecting either segmented the
        # cell channel. Asserted per key so adding a sixth object type does
        # not fail this test for the wrong reason.
        assert req.channels["cell"] == 1
        assert req.channels["nucleus"] == 2
        assert set(req.channels) >= {"cell", "nucleus", "pathogen",
                                     "organelle"}
        assert req.object_types == ("cell", "nucleus")
        # The cached Mask-app settings and the compartment widgets are merged
        # into one dict used for both pre and post.
        assert req.preprocess_settings["some_pipeline_key"] == "kept"
        assert req.preprocess_settings is req.postprocess_settings

    def test_widget_value_reads_each_widget_kind(self, qtbot):
        p = _panel(qtbot)
        p._common_widgets["remove_background"].setChecked(True)
        assert p._widget_value(p._common_widgets["remove_background"]) is True
        assert p._widget_value(p._model_box) == "cpsam"
        p._common_widgets["background"].setValue(17)
        assert p._widget_value(p._common_widgets["background"]) == 17

    def test_compartment_settings_cover_every_compartment_and_field(self,
                                                                    qtbot):
        p = _panel(qtbot)
        s = p._compartment_settings()
        for comp in LP.COMPARTMENTS:
            for suffix, *_ in LP.COMPARTMENT_FIELDS:
                assert f"{comp}_{suffix}" in s
        assert s["adjust_cells"] is False
        assert s["remove_background_cell"] is False

    def test_compartment_defaults_match_the_pipeline_defaults(self, qtbot):
        """BUG FIX: the panel defaulted ``*_min_intensity_percentile`` to 1,
        ``*_max_intensity_percentile`` to 99, ``*_perimeter_fraction`` to 0.5
        and ``*_intensity_percentile`` to 50, against the pipeline's 0 / 100 /
        0 / 75. The first two switched the intensity filter ON for every
        preview — with two objects it dropped both — and Propagate wrote all
        four into the main settings panel behind the user's back."""
        from spacr.settings import (
            set_default_settings_preprocess_generate_masks as defaults)
        pipeline = defaults({})
        p = _panel(qtbot)
        live = p._compartment_settings()
        checked = 0
        for comp in LP.COMPARTMENTS:
            for suffix, *_ in LP.COMPARTMENT_FIELDS:
                key = f"{comp}_{suffix}"
                if key not in pipeline:
                    continue
                assert live[key] == pipeline[key], f"{key} drifted"
                checked += 1
        assert checked >= 30, "the comparison found almost nothing to check"

    def test_default_filters_are_neutral_so_a_preview_shows_raw_masks(
            self, qtbot):
        """The consequence of the defaults above: an untuned panel must not
        filter anything out of a Cellpose result."""
        p = _panel(qtbot)
        mask = np.zeros((20, 20), np.int32)
        mask[2:4, 2:4] = 1          # dim + tiny
        mask[8:16, 8:16] = 2        # bright + large
        inten = np.full((20, 20), 10.0, np.float32)
        inten[8:16, 8:16] = 900.0
        out = LP._apply_size_filter(mask, p._compartment_settings(), "cell",
                                    intensity_img=inten)
        assert np.array_equal(out, mask)

    def test_settings_for_propagation_maps_the_main_panel_keys(self, qtbot):
        p = _panel(qtbot)
        p._model_box.setCurrentText("nuclei")
        p._cell_channel.setValue(1)
        p._nucleus_channel.setValue(2)
        p._diameter.setValue(25.0)
        p._flow.setValue(0.33)
        p._prob.setValue(1.5)
        p._lo_pct.setValue(3.0)
        s = p.settings_for_propagation()
        assert s["model_name"] == "nuclei"
        assert s["cell_channel"] == 1 and s["nucleus_channel"] == 2
        assert s["cell_diameter"] == pytest.approx(25.0)
        assert s["cell_FT"] == pytest.approx(0.33)
        assert s["cell_CP_prob"] == pytest.approx(1.5)
        assert s["normalize"] is True
        assert s["lower_percentile"] == pytest.approx(3.0)

    def test_a_bad_field_kind_is_rejected_loudly(self, qtbot, monkeypatch):
        """The compartment table is hand-written; a typo in the ``kind``
        column must fail at construction rather than silently skip a knob."""
        monkeypatch.setattr(LP, "COMPARTMENT_FIELDS",
                            (("min_area", "Min area", "wobble", None),))
        with pytest.raises(ValueError, match="wobble"):
            LP.LivePreviewPanel()

    def test_tooltips_fall_back_when_spacr_descriptions_are_unavailable(
            self, qtbot, monkeypatch):
        broken = types.ModuleType("spacr.settings")   # no `descriptions`
        monkeypatch.setitem(sys.modules, "spacr.settings", broken)
        p = _panel(qtbot)
        assert p._compartment_widgets["cell"]["min_area"].toolTip() == \
            "Min area (px²) for cell objects."

    def test_a_widget_that_refuses_a_connection_does_not_break_the_panel(
            self, qtbot, monkeypatch):
        """Defensive branch: the live-refilter wiring swallows a refused
        connect so one bad widget cannot take the whole panel out."""
        class _BadSignal:
            def connect(self, *a, **k):
                raise TypeError("incompatible signature")

        class _BadWidget:
            valueChanged = _BadSignal()

        real = LP.LivePreviewPanel._all_compartment_widgets

        def _with_bad(self):
            return list(real(self)) + [_BadWidget()]
        monkeypatch.setattr(LP.LivePreviewPanel, "_all_compartment_widgets",
                            _with_bad)
        p = _panel(qtbot)
        assert p._compartment_widgets["cell"]["min_area"] is not None


# ===========================================================================
# Settings change -> recompute (no Cellpose re-run)
# ===========================================================================

class TestRecomputeOnSettingsChange:
    @staticmethod
    def _stub_two_objects(monkeypatch, counter):
        def _stub(req):
            counter.append(1)
            m = np.zeros(req.image.shape[:2], np.int32)
            m[2:5, 2:5] = 1          # 9 px, dim
            m[10:26, 10:26] = 2      # 256 px, bright
            return {"cell": m}
        monkeypatch.setattr(LP, "_segment_multi", _stub)

    def _run(self, qtbot, p):
        got = []
        p.preview_ready.connect(got.append)
        p.run_preview()
        qtbot.waitUntil(lambda: len(got) == 1, timeout=5000)
        return got

    def test_changing_a_filter_refilters_without_rerunning_cellpose(
            self, qtbot, monkeypatch, gray_tif):
        calls = []
        self._stub_two_objects(monkeypatch, calls)
        p = _panel(qtbot)
        p.load_image(gray_tif)
        got = self._run(qtbot, p)
        assert len(calls) == 1
        # The default compartment settings are the pipeline's neutral ones, so
        # an untuned preview shows exactly what Cellpose returned.
        assert int((p._masks["cell"] > 0).sum()) == 9 + 256
        assert int(p._masks["cell"].max()) == 2
        assert np.array_equal(got[0]["cell"], p._masks["cell"])

        # Tighten the minimum area — the small object must disappear and
        # Cellpose must NOT be asked again.
        p._compartment_widgets["cell"]["min_area"].setValue(50)
        assert len(calls) == 1
        assert int((p._masks["cell"] > 0).sum()) == 256
        assert not p._masks["cell"][2:5, 2:5].any()      # the 9 px one is gone
        assert (p._masks["cell"][10:26, 10:26] > 0).all()
        # The raw Cellpose output is untouched — that is what makes the
        # re-filter possible without another pass.
        assert int((p._raw_masks["cell"] > 0).sum()) == 9 + 256

        # And relaxing it again brings the object back.
        p._compartment_widgets["cell"]["min_area"].setValue(0)
        assert int((p._masks["cell"] > 0).sum()) == 9 + 256
        assert len(calls) == 1

    def test_the_status_line_reports_the_object_count(self, qtbot,
                                                      monkeypatch, gray_tif):
        self._stub_two_objects(monkeypatch, [])
        p = _panel(qtbot)
        p.load_image(gray_tif)
        self._run(qtbot, p)
        assert p._status.text() == "Found cell=2."
        p._compartment_widgets["cell"]["min_area"].setValue(50)
        assert p._status.text() == "Found cell=1."

    def test_toggling_a_checkbox_filter_also_recomputes(self, qtbot,
                                                        monkeypatch, gray_tif):
        calls = []

        def _stub(req):
            calls.append(1)
            m = np.zeros(req.image.shape[:2], np.int32)
            m[0:6, 0:6] = 1          # touches the border
            m[20:30, 20:30] = 2      # interior
            return {"cell": m}
        monkeypatch.setattr(LP, "_segment_multi", _stub)
        p = _panel(qtbot)
        p.load_image(gray_tif)
        self._run(qtbot, p)
        assert int(p._masks["cell"].max()) == 2
        p._compartment_widgets["cell"]["remove_border_objects"].setChecked(True)
        assert int(p._masks["cell"].max()) == 1
        assert len(calls) == 1

    def test_recompute_is_a_no_op_before_any_run(self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        emitted = []
        p.preview_ready.connect(emitted.append)
        p._compartment_widgets["cell"]["min_area"].setValue(99)
        assert p._masks == {}
        assert emitted == []

    def test_recompute_without_an_image_is_a_no_op(self, qtbot):
        p = _panel(qtbot)
        p._raw_masks = {"cell": np.ones((4, 4), np.int32)}
        p._recompute_masks()
        assert p._masks == {}


# ===========================================================================
# Comparison scrubber
# ===========================================================================

class TestCompareScrubber:
    def _snap(self, p, n_objects, summary="cell=1"):
        m = np.zeros((48, 48), np.int32)
        for i in range(n_objects):
            m[2 + i * 6:6 + i * 6, 2:6] = i + 1
        p._snapshot_run({"cell": m}, [summary])
        return m

    def test_the_row_stays_hidden_until_a_second_run(self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        self._snap(p, 1)
        assert p._compare_row.isVisibleTo(p) is False
        assert p._compare_label.text() == "1/1"
        self._snap(p, 2)
        assert p._compare_row.isVisibleTo(p) is True
        assert p._compare_slider.maximum() == 1
        assert p._compare_slider.value() == 1        # newest selected
        assert p._compare_label.text() == "2/2"

    def test_snapshot_without_an_image_records_nothing(self, qtbot):
        p = _panel(qtbot)
        p._snapshot_run({"cell": np.ones((4, 4), np.int32)}, ["cell=1"])
        assert p._history == []

    def test_history_is_capped_at_fifty_runs(self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        for i in range(60):
            p._snapshot_run({}, [f"run{i}"])
        assert len(p._history) == 50
        assert p._history[-1]["summary"] == "run59"
        assert p._history[0]["summary"] == "run10"
        assert p._compare_slider.maximum() == 49

    def test_scrubbing_back_redraws_that_run(self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        first = self._snap(p, 1, "cell=1")
        self._snap(p, 2, "cell=2")
        p._model_box.setCurrentText("cyto2")
        p._compare_slider.setValue(0)
        label = p._compare_label.text()
        assert label.startswith("1/2")
        assert "cpsam/cell" in label          # the model used for that run
        assert "cell=1" in label
        # The older run's single object is what got repainted, in the current
        # 'auto' colour — the scrubber honours the outline setting now.
        assert _pixmap_pixel(p._mask_view, 2, 2) == \
            p._auto_outline_colour("cell")
        assert first[2, 2] == 1

    def test_scrubbing_to_a_maskless_run_shows_the_source(self, qtbot,
                                                          gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        p._snapshot_run({}, ["cell=0"])
        p._snapshot_run({}, ["cell=0"])
        p._compare_slider.setValue(0)
        assert _pixmap_pixel(p._mask_view, 10, 10) == \
            _pixmap_pixel(p._src_view, 10, 10)

    def test_an_out_of_range_index_is_ignored(self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        before = p._compare_label.text()
        p._on_compare_scrub(0)       # history is empty
        p._on_compare_scrub(-1)
        assert p._compare_label.text() == before


# ===========================================================================
# Worker lifecycle + cancellation
# ===========================================================================

class _Gate:
    """A stub segmentation that blocks until the test releases it."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def __call__(self, req):
        self.calls += 1
        self.entered.set()
        self.release.wait(timeout=10)
        m = np.zeros(req.image.shape[:2], np.int32)
        m[4:12, 4:12] = 1
        return {"cell": m}, {"cell": np.zeros(req.image.shape[:2] + (3,),
                                              np.uint8)}


class TestWorkerLifecycle:
    def test_the_worker_is_never_scheduled_for_c_plus_plus_deletion(
            self, qtbot, monkeypatch, gray_tif):
        """Repo rule: no ``worker.deleteLater``. Two owners for one PySide6
        object is a measured segfault (see spacr.qt.bridge.make_thread). If
        deleteLater were re-added, touching the worker after the event loop
        flushed would raise "Internal C++ object already deleted"."""
        monkeypatch.setattr(
            LP, "_segment_multi",
            lambda req: {"cell": np.zeros(req.image.shape[:2], np.int32)})
        p = _panel(qtbot)
        p.load_image(gray_tif)
        got = []
        p.preview_ready.connect(got.append)
        p.run_preview()
        worker = p._worker
        qtbot.waitUntil(lambda: len(got) == 1, timeout=5000)
        qtbot.waitUntil(lambda: worker.isFinished(), timeout=5000)
        for _ in range(3):
            qtbot.wait(20)          # flush any deferred deletes
        assert worker.isFinished() is True            # C++ object still alive
        assert worker.token == p._run_token

    def test_the_completion_handler_runs_on_the_gui_thread(
            self, qtbot, monkeypatch, gray_tif):
        """``finished_masks`` is emitted inside ``run()``; the receiver has to
        be a bound method of a GUI-thread QObject or Qt picks a direct
        connection and the handler touches widgets off-thread."""
        monkeypatch.setattr(
            LP, "_segment_multi",
            lambda req: {"cell": np.ones(req.image.shape[:2], np.int32)})
        p = _panel(qtbot)
        p.load_image(gray_tif)
        threads = []
        p.preview_ready.connect(
            lambda _m: threads.append(QThread.currentThread()))
        p.run_preview()
        qtbot.waitUntil(lambda: len(threads) == 1, timeout=5000)
        assert threads[0] is p.thread()
        assert threads[0] is QThread.currentThread()

    def test_workers_do_not_accumulate_across_runs(self, qtbot, monkeypatch,
                                                   gray_tif):
        """Each worker is parented to the panel, so without an explicit
        release they would pile up as children — each pinning a full-size
        preview image — for the lifetime of the panel."""
        monkeypatch.setattr(
            LP, "_segment_multi",
            lambda req: {"cell": np.zeros(req.image.shape[:2], np.int32)})
        p = _panel(qtbot)
        p.load_image(gray_tif)
        got = []
        p.preview_ready.connect(got.append)
        for i in range(4):
            p.run_preview()
            qtbot.waitUntil(lambda i=i: len(got) == i + 1, timeout=5000)
            qtbot.waitUntil(lambda: not p._worker.isRunning(), timeout=5000)
        assert len(p.findChildren(LP._PreviewWorker)) == 1

    def test_a_second_run_while_one_is_in_flight_is_refused(
            self, qtbot, monkeypatch, gray_tif):
        gate = _Gate()
        monkeypatch.setattr(LP, "_segment_multi", gate)
        p = _panel(qtbot)
        p.load_image(gray_tif)
        try:
            p.run_preview()
            assert gate.entered.wait(timeout=5)
            assert p._run_btn.isEnabled() is False
            p.run_preview()
            assert p._status.text() == "Preview already running."
            assert gate.calls == 1
        finally:
            gate.release.set()
            qtbot.waitUntil(lambda: p._worker is not None
                            and p._worker.isFinished(), timeout=5000)

    def test_cancelling_mid_compute_discards_the_result(self, qtbot,
                                                        monkeypatch, gray_tif):
        gate = _Gate()
        monkeypatch.setattr(LP, "_segment_multi", gate)
        p = _panel(qtbot)
        p.load_image(gray_tif)
        emitted = []
        p.preview_ready.connect(emitted.append)
        p.run_preview()
        assert gate.entered.wait(timeout=5)

        assert p.cancel_preview() is True
        assert p._status.text() == "Preview cancelled."
        assert p._run_btn.isEnabled() is True

        gate.release.set()
        qtbot.waitUntil(lambda: p._worker.isFinished(), timeout=5000)
        for _ in range(3):
            qtbot.wait(20)          # let the queued results land
        assert emitted == []                      # the answer was dropped
        assert p._masks == {} and p._raw_masks == {} and p._flows == {}
        assert p._status.text() == "Preview cancelled."
        assert p._run_btn.isEnabled() is True     # usable again

    def test_cancel_with_nothing_running_just_bumps_the_token(self, qtbot):
        p = _panel(qtbot)
        before = p._run_token
        assert p.cancel_preview() is False
        assert p._run_token == before + 1
        assert p._status.text() == ""

    def test_loading_a_new_image_mid_compute_drops_the_stale_result(
            self, qtbot, monkeypatch, gray_tif, rgb_tif):
        """The regression this guards: a result computed from a 48x48 tile
        landing after a 32x32 tile was loaded used to reach ``overlay_masks``
        and raise IndexError."""
        gate = _Gate()
        monkeypatch.setattr(LP, "_segment_multi", gate)
        p = _panel(qtbot)
        p.load_image(gray_tif)
        emitted = []
        p.preview_ready.connect(emitted.append)
        p.run_preview()
        assert gate.entered.wait(timeout=5)

        assert p.load_image(rgb_tif) is True       # cancels the run in flight
        gate.release.set()
        qtbot.waitUntil(lambda: p._worker.isFinished(), timeout=5000)
        for _ in range(3):
            qtbot.wait(20)
        assert emitted == []
        assert p._masks == {}
        assert p._image.shape == (32, 32, 3)
        assert p._run_btn.isEnabled() is True

    def test_a_stale_flow_result_is_dropped_too(self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        p._on_flows_ready({"cell": np.ones((48, 48, 3), np.uint8)},
                          p._run_token + 5)
        assert p._flows == {}
        p._on_flows_ready({"cell": np.ones((48, 48, 3), np.uint8)},
                          p._run_token)
        assert set(p._flows) == {"cell"}

    def test_a_failed_run_reports_the_error_and_emits_none(self, qtbot,
                                                           monkeypatch,
                                                           gray_tif):
        def _boom(req):
            raise RuntimeError("cellpose is not installed")
        monkeypatch.setattr(LP, "_segment_multi", _boom)
        p = _panel(qtbot)
        p.load_image(gray_tif)
        got = []
        p.preview_ready.connect(got.append)
        p.run_preview()
        qtbot.waitUntil(lambda: len(got) == 1, timeout=5000)
        assert got == [None]
        assert "cellpose is not installed" in p._status.text()
        qtbot.waitUntil(lambda: p._run_btn.isEnabled(), timeout=5000)

    def test_an_empty_mask_dict_is_reported_not_rendered(self, qtbot,
                                                         monkeypatch,
                                                         gray_tif):
        monkeypatch.setattr(LP, "_segment_multi", lambda req: {})
        p = _panel(qtbot)
        p.load_image(gray_tif)
        p.run_preview()
        qtbot.waitUntil(
            lambda: p._status.text() == "Preview returned no masks.",
            timeout=5000)
        assert p._masks == {}

    def test_a_result_arriving_after_the_image_was_cleared_is_ignored(
            self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        p._image = None
        p._on_worker_done({"cell": np.ones((4, 4), np.int32)}, "")
        assert p._status.text() == "Preview returned no masks."
        assert p._masks == {}

    def test_release_worker_is_safe_with_nothing_to_release(self, qtbot):
        p = _panel(qtbot)
        p._release_worker()
        assert p._worker is None


# ===========================================================================
# Live settings dialog + propagation
# ===========================================================================

class TestPropagation:
    def test_propagate_settings_without_a_callback_is_a_no_op(self, qtbot):
        p = _panel(qtbot)
        p.propagate_settings()          # must not raise
        assert p._propagate_cb is None

    def test_propagate_settings_pushes_the_mapped_dict(self, qtbot):
        p = _panel(qtbot)
        seen = []
        p.set_propagate_callback(seen.append)
        p._model_box.setCurrentText("cyto3")
        p._compartment_widgets["cell"]["min_area"].setValue(321)
        p.propagate_settings()
        assert len(seen) == 1
        assert seen[0]["model_name"] == "cyto3"
        assert seen[0]["cell_min_area"] == 321

    def test_a_throwing_callback_is_swallowed(self, qtbot):
        p = _panel(qtbot)
        calls = []

        def _bad(_settings):
            calls.append(1)
            raise RuntimeError("main panel is gone")
        p.set_propagate_callback(_bad)
        p.propagate_settings()          # must not raise
        assert calls == [1]
        # And the panel is still wired up for the next push.
        good = []
        p.set_propagate_callback(good.append)
        p.propagate_settings()
        assert len(good) == 1

    def test_the_propagate_toggle_pushes_now_and_on_every_later_edit(
            self, qtbot):
        p = _panel(qtbot)
        seen = []
        p.set_propagate_callback(seen.append)
        p.open_live_settings()
        dlg = p._live_settings_dialog
        try:
            assert seen == []
            dlg._propagate_btn.setChecked(True)
            assert len(seen) == 1                    # initial push
            assert seen[-1]["cell_diameter"] == pytest.approx(30.0)

            p._diameter.setValue(55.0)
            assert len(seen) > 1
            assert seen[-1]["cell_diameter"] == pytest.approx(55.0)

            p._compartment_widgets["nucleus"]["min_area"].setValue(77)
            assert seen[-1]["nucleus_min_area"] == 77

            n = len(seen)
            dlg._propagate_btn.setChecked(False)
            p._diameter.setValue(12.0)
            assert len(seen) == n                    # disconnected
            assert p.settings_for_propagation()["cell_diameter"] == \
                pytest.approx(12.0)
        finally:
            dlg.close()

    def test_one_unusable_source_does_not_break_the_toggle(self, qtbot):
        """Defensive branch: a source widget whose signal refuses connect or
        disconnect must not stop the remaining sources being (un)wired."""
        class _BadSignal:
            def connect(self, *a, **k):
                raise RuntimeError("wrapped C/C++ object has been deleted")

            def disconnect(self, *a, **k):
                raise RuntimeError("wrapped C/C++ object has been deleted")

        class _BadWidget:
            valueChanged = _BadSignal()

        p = _panel(qtbot)
        seen = []
        p.set_propagate_callback(seen.append)
        p.open_live_settings()
        dlg = p._live_settings_dialog
        try:
            dlg._propagate_sources.append(_BadWidget())
            dlg._propagate_btn.setChecked(True)
            assert len(seen) == 1
            p._diameter.setValue(21.0)               # good sources still wired
            assert seen[-1]["cell_diameter"] == pytest.approx(21.0)
            n = len(seen)
            dlg._propagate_btn.setChecked(False)
            p._diameter.setValue(9.0)
            assert len(seen) == n                    # and still unwired
        finally:
            dlg.close()

    def test_the_dialogs_run_button_drives_the_panel(self, qtbot, monkeypatch,
                                                     gray_tif):
        monkeypatch.setattr(
            LP, "_segment_multi",
            lambda req: {"cell": np.ones(req.image.shape[:2], np.int32)})
        p = _panel(qtbot)
        p.load_image(gray_tif)
        p.open_live_settings()
        dlg = p._live_settings_dialog
        got = []
        p.preview_ready.connect(got.append)
        try:
            dlg._run_btn.click()
            qtbot.waitUntil(lambda: len(got) == 1, timeout=5000)
            assert set(got[0]) == {"cell"}
        finally:
            dlg.close()


class TestLiveSettingsDialog:
    def test_reopening_focuses_the_existing_dialog(self, qtbot):
        p = _panel(qtbot)
        p.open_live_settings()
        first = p._live_settings_dialog
        p.open_live_settings()
        assert p._live_settings_dialog is first
        assert first.isVisible()
        first.close()
        assert p._live_settings_dialog is None

    def test_closing_the_dialog_refreshes_the_canvases(self, qtbot, gray_tif):
        p = _panel(qtbot)
        p.load_image(gray_tif)
        m = np.zeros((48, 48), np.int32)
        m[10:20, 10:20] = 1
        p._masks = {"cell": m}
        p.open_live_settings()
        p._outline_colour.setCurrentText("white")
        p._live_settings_dialog.close()
        assert p._live_settings_dialog is None
        assert _pixmap_pixel(p._mask_view, 10, 10) == (240, 240, 240)

    def test_widgets_are_returned_to_the_panel_on_close(self, qtbot):
        p = _panel(qtbot)
        p.open_live_settings()
        dlg = p._live_settings_dialog
        assert p._model_box.parent() is not p
        p._diameter.setValue(44.0)
        dlg.close()
        assert p._model_box.parent() is p
        assert p._diameter.parent() is p
        assert p._diameter.value() == pytest.approx(44.0)   # value survives

    def test_changing_the_object_regates_an_open_dialog(self, qtbot):
        p = _panel(qtbot)
        p.open_live_settings()
        dlg = p._live_settings_dialog
        try:
            p._object_box.setCurrentText("pathogen")
            assert dlg._compartment_groupboxes["pathogen"].isVisibleTo(dlg)
            assert not dlg._compartment_groupboxes["cell"].isVisibleTo(dlg)
            assert dlg._compartment_groupboxes["pathogen"].title() == \
                "Pathogen (primary object)"
            assert p._cell_channel.isEnabled() is False
            assert p._nucleus_channel.isEnabled() is False
        finally:
            dlg.close()

    def test_the_regate_hook_survives_a_broken_dialog(self, qtbot):
        """A dialog that has already lost its C++ half must not stop the
        object combo from changing."""
        p = _panel(qtbot)
        calls = []

        class _Broken:
            def refresh_visibility(self):
                calls.append(1)
                raise RuntimeError("dialog already destroyed")
        p._live_settings_dialog = _Broken()
        p._object_box.setCurrentText("organelle")   # fires the hook
        assert calls == [1]
        assert p._selected_object_types() == ("organelle",)
        p._live_settings_dialog = None

    def test_percentile_bounds_grey_out_when_normalise_is_off(self, qtbot):
        p = _panel(qtbot)
        p.open_live_settings()
        try:
            assert p._lo_pct.isEnabled() and p._hi_pct.isEnabled()
            p._normalise_check.setChecked(False)
            assert not p._lo_pct.isEnabled()
            assert p._lo_pct.toolTip() == ""
            assert "href=" in p._lo_pct._spacr_setting_label.toolTip()
            # The API link dot used to be asserted here too. This dialog now
            # passes `api_dots=False` -- 68 of them on one form read as
            # texture rather than as an affordance -- so there is nothing to
            # assert. What mattered about the row survives and is still
            # checked above: greying the field out must not take the help
            # away with it, and the help lives on the label.
            p._normalise_check.setChecked(True)
            assert p._lo_pct.isEnabled()
            assert "href=" in p._lo_pct._spacr_setting_label.toolTip()
        finally:
            p._live_settings_dialog.close()

    def test_the_dialog_still_opens_without_a_screen(self, qtbot, monkeypatch):
        """Headless / detached-display fallback: ``screen()`` returns None and
        ``None.availableGeometry()`` raises, so the dialog takes its default
        size instead of failing to open."""
        monkeypatch.setattr(LP.LiveSettingsDialog, "screen",
                            lambda self: None, raising=False)
        p = _panel(qtbot)
        dlg = LP.LiveSettingsDialog(p)
        qtbot.addWidget(dlg)
        try:
            assert dlg.width() == 1400 and dlg.height() == 720
        finally:
            dlg.close()

    def test_every_compartment_field_has_a_row_in_its_panel(self, qtbot):
        p = _panel(qtbot)
        p.open_live_settings()
        dlg = p._live_settings_dialog
        try:
            for comp in LP.COMPARTMENTS:
                box = dlg._compartment_groupboxes[comp]
                for suffix, *_ in LP.COMPARTMENT_FIELDS:
                    w = p._compartment_widgets[comp][suffix]
                    assert w.parent() is box or w.parentWidget() is box, (
                        f"{comp}.{suffix} was not rehomed into its panel")
            assert p._adjust_cells.parentWidget() is \
                dlg._compartment_groupboxes["cell"]
        finally:
            dlg.close()
