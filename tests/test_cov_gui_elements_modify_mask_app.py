"""CPU coverage for spacr.gui_elements.ModifyMaskApp — the Tk hand-editing app
for segmentation masks.

The whole class is driven headlessly: a hidden Tk root is shrunk to a 64x64
canvas (by overriding ``winfo_screenheight``) so every resize/overlay round
trip stays tiny, and the canvas is told to report its configured size the way
a mapped widget would. Mouse interaction is replayed through a tiny event
stub, so draw / brush / erase / magic-wand / dividing-line flows are exercised
exactly as Tk would deliver them.

Everything is offline and file-backed: a temp folder holds four synthetic
images (uint16 grey TIFF, RGB PNG, RGBA PNG, canvas-sized TIFF) plus a
``masks/`` subfolder, which covers every branch of ``load_image_and_mask``.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

# Canvas geometry: ModifyMaskApp derives the canvas from the screen height,
# so a patched screen height of 164 yields a 64x64 canvas.
CANVAS = 64
SCREEN_H = CANVAS + 100


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class FakeEvent:
    """Minimal stand-in for a Tk event (only .x/.y/.num are ever read)."""

    def __init__(self, x, y, num=1):
        self.x = x
        self.y = y
        self.num = num


def _blobs(h, w, dtype=np.uint16):
    """Deterministic 2-D image: two bright discs on a faintly noisy floor."""
    yy, xx = np.mgrid[:h, :w]
    img = np.zeros((h, w), dtype=np.float64)
    for cy, cx, r, val in (
        (h // 4, w // 4, max(3, h // 10), 40000.0),
        (2 * h // 3, 2 * w // 3, max(4, h // 8), 60000.0),
    ):
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = val
    img += (yy + xx) % 97 + 100  # deterministic, non-constant background
    return img.astype(dtype)


def _labels(h, w):
    """uint8 label mask matching the two discs of :func:`_blobs`."""
    yy, xx = np.mgrid[:h, :w]
    m = np.zeros((h, w), dtype=np.uint8)
    m[(yy - h // 4) ** 2 + (xx - w // 4) ** 2 <= max(3, h // 10) ** 2] = 1
    m[(yy - 2 * h // 3) ** 2 + (xx - 2 * w // 3) ** 2 <= max(4, h // 8) ** 2] = 2
    return m


def _write_tif(path, arr):
    import tifffile
    tifffile.imwrite(str(path), arr)


def _write_png(path, arr):
    from PIL import Image
    Image.fromarray(arr).save(str(path))


def _enter_zoom(app, box=((4, 4), (60, 60))):
    """Turn zoom mode on and commit a zoom rectangle, populating zoom_* state."""
    app.toggle_zoom_mode()
    app.set_zoom_rectangle_start(FakeEvent(*box[0]))
    app.set_zoom_rectangle_end(FakeEvent(*box[1]))
    return app


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_stray_figures():
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


@pytest.fixture
def mask_dir(tmp_path):
    """Folder of four editable images + a partially populated masks/ folder."""
    folder = tmp_path / "imgs"
    folder.mkdir()
    masks = folder / "masks"
    masks.mkdir()

    # index 0 — 2-D uint16, shape differs from the canvas, uint8 mask on disk
    _write_tif(folder / "a_gray16.tif", _blobs(60, 70))
    _write_tif(masks / "a_gray16.tif", _labels(60, 70))

    # index 1 — RGB uint8 PNG, no mask on disk
    grey8 = (_blobs(48, 48).astype(np.float64) / 257.0).astype(np.uint8)
    rgb = np.stack([grey8, grey8, grey8], axis=-1)
    _write_png(folder / "b_rgb.png", rgb)

    # index 2 — RGBA uint8 PNG (alpha channel must be stripped), no mask
    rgba = np.concatenate(
        [rgb, np.full((48, 48, 1), 255, dtype=np.uint8)], axis=-1
    )
    _write_png(folder / "c_rgba.png", rgba)

    # index 3 — exactly canvas-sized uint16, with a NON-uint8 mask on disk
    _write_tif(folder / "d_exact.tif", _blobs(CANVAS, CANVAS))
    _write_tif(masks / "d_exact.tif", _labels(CANVAS, CANVAS).astype(np.uint16) * 1000)

    # a non-image file that the extension filter must drop
    (folder / "notes.txt").write_text("not an image")
    return folder


@pytest.fixture
def app(tk_root, mask_dir):
    from spacr.gui_elements import ModifyMaskApp

    # Shrink the derived canvas so every resize/overlay stays fast.
    tk_root.winfo_screenheight = lambda: SCREEN_H
    a = ModifyMaskApp(tk_root, str(mask_dir), 1.0)
    # A never-mapped canvas reports 1x1; make it report its configured size,
    # which is what display_zoomed_image sees on a real (mapped) window.
    a.canvas.winfo_width = lambda: a.canvas_width
    a.canvas.winfo_height = lambda: a.canvas_height
    return a


# ---------------------------------------------------------------------------
# construction / setup_*
# ---------------------------------------------------------------------------

def test_init_discovers_images_and_builds_every_toolbar(app, mask_dir):
    assert app.image_filenames == [
        "a_gray16.tif", "b_rgb.png", "c_rgba.png", "d_exact.tif"
    ]
    assert app.masks_folder == os.path.join(str(mask_dir), "masks")
    assert app.current_image_index == 0
    assert app.canvas_width == CANVAS and app.canvas_height == CANVAS

    # first pair loaded, remembered at its native size, then stretched
    assert app.original_size == (60, 70)
    assert app.image.shape == (CANVAS, CANVAS)
    assert app.mask.shape == (CANVAS, CANVAS)
    assert app.image.dtype == np.uint16
    assert app.mask.max() == 2  # the two labelled discs survived the stretch

    # initialize_flags state
    assert app.zoom_active is False
    assert app.drawing is False
    assert app.magic_wand_active is False
    assert app.brush_active is False
    assert app.dividing_line_active is False
    assert app.dividing_line_coords == []
    assert app.current_dividing_line is None
    assert app.zoom_scale == 1
    assert app.lower_quantile.get() == "1.0"
    assert app.upper_quantile.get() == "99.9"
    assert app.magic_wand_tolerance.get() == "1000"

    # widgets built by the four setup_* helpers
    assert app.tolerance_entry.get() == "1000"
    assert app.max_pixels_entry.get() == "1000"
    assert app.brush_size_entry.get() == "10"
    assert app.min_area_entry.get() == "100"
    assert app.lower_entry.get() == "1.0"
    assert app.upper_entry.get() == "99.9"
    assert app.intensity_label.cget("text") == "Image: N/A"
    assert app.mask_value_label.cget("text") == "Mask: N/A"
    assert app.pixel_count_label.cget("text") == "Area: N/A"
    assert int(app.canvas.cget("width")) == CANVAS
    assert app.canvas.bind("<Motion>") != ""


# ---------------------------------------------------------------------------
# load_image_and_mask
# ---------------------------------------------------------------------------

def test_load_image_and_mask_uint16_image_with_uint8_mask(app):
    img, mask = app.load_image_and_mask(0)
    assert img.shape == (60, 70) and img.dtype == np.uint16
    assert mask.shape == (60, 70) and mask.dtype == np.uint8
    assert sorted(np.unique(mask).tolist()) == [0, 1, 2]


def test_load_image_and_mask_rgb_is_converted_to_grey_uint16(app):
    img, mask = app.load_image_and_mask(1)
    assert img.ndim == 2 and img.shape == (48, 48)
    assert img.dtype == np.uint16
    assert img.max() == 65535  # rescaled to fill the 16-bit range
    # no mask file on disk -> a fresh empty uint8 mask
    assert mask.shape == (48, 48) and mask.dtype == np.uint8
    assert mask.max() == 0


def test_load_image_and_mask_strips_the_rgba_alpha_channel(app):
    img, mask = app.load_image_and_mask(2)
    assert img.ndim == 2 and img.shape == (48, 48)
    assert img.dtype == np.uint16
    # identical RGB payload to index 1, so the greyscale result must match
    ref, _ = app.load_image_and_mask(1)
    assert np.array_equal(img, ref)
    assert mask.max() == 0


def test_load_image_and_mask_rescales_a_non_uint8_mask(app):
    img, mask = app.load_image_and_mask(3)
    assert img.shape == (CANVAS, CANVAS) and img.dtype == np.uint16
    assert mask.dtype == np.uint8
    # on-disk labels were 0/1000/2000 -> rescaled onto 0..255
    assert sorted(np.unique(mask).tolist()) == [0, 127, 255]


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def test_normalize_image_clips_to_percentiles_and_keeps_dtype(app):
    img = np.arange(10000, dtype=np.uint16).reshape(100, 100)
    out = app.normalize_image(img, 10.0, 90.0)
    assert out.dtype == np.uint16
    assert out.min() == 0 and out.max() == 65535
    lo = np.percentile(img, 10.0)
    hi = np.percentile(img, 90.0)
    assert out[img <= lo].max() == 0
    assert out[img >= hi].min() == 65535


def test_resize_arrays_stretches_to_canvas_and_uses_the_image_dtype(app):
    img = _blobs(40, 50)
    mask = _labels(40, 50)
    out_img, out_mask = app.resize_arrays(img, mask)
    assert out_img.shape == (CANVAS, CANVAS)
    assert out_mask.shape == (CANVAS, CANVAS)
    assert out_img.dtype == np.uint16
    # the mask inherits the IMAGE dtype - resize_arrays reads img.dtype only
    assert out_mask.dtype == np.uint16
    # order=0 for the mask: no interpolated label values are invented
    assert set(np.unique(out_mask).tolist()).issubset({0, 1, 2})


def test_resize_arrays_honours_the_scale_factor_before_stretching(app):
    """A coarse pre-scale loses detail even though the output is canvas-sized."""
    img = _blobs(64, 64)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[32, 32] = 3  # a single pixel: survives at 1.0, is dropped at 0.1
    app.scale_factor = 1.0
    _, fine = app.resize_arrays(img, mask)
    app.scale_factor = 0.1
    _, coarse = app.resize_arrays(img, mask)
    assert fine.shape == coarse.shape == (CANVAS, CANVAS)
    assert fine.max() == 3
    assert coarse.max() == 0


def test_get_scaling_factors_returns_image_over_canvas(app):
    assert app.get_scaling_factors(200, 100, 50, 25) == (4.0, 4.0)
    assert app.get_scaling_factors(64, 64, 64, 64) == (1.0, 1.0)


def test_canvas_and_image_coordinates_round_trip(app):
    # after resize_arrays the image is exactly canvas-sized -> 1:1 mapping
    assert app.canvas_to_image(13, 21) == (13, 21)
    assert app.image_to_canvas(13, 21) == (13, 21)


def test_canvas_to_image_scales_by_the_image_over_canvas_ratio(app):
    app.image = np.zeros((CANVAS * 2, CANVAS * 4), dtype=np.uint16)
    assert app.canvas_to_image(10, 10) == (40, 20)
    assert app.image_to_canvas(40, 20) == (10, 10)


# ---------------------------------------------------------------------------
# update_mouse_info
# ---------------------------------------------------------------------------

def test_update_mouse_info_reports_intensity_label_and_area(app):
    app.mask[:] = 0
    app.mask[10:20, 10:20] = 7
    app.update_mouse_info(FakeEvent(12, 12))
    assert app.intensity_label.cget("text") == f"Intensity: {app.image[12, 12]}"
    assert app.mask_value_label.cget("text") == "Mask: 7"
    assert app.pixel_count_label.cget("text") == "Area: 100"


def test_update_mouse_info_background_pixel_has_no_area(app):
    app.mask[:] = 0
    app.update_mouse_info(FakeEvent(5, 5))
    assert app.mask_value_label.cget("text") == "Mask: 0"
    assert app.pixel_count_label.cget("text") == "Area: N/A"


def test_update_mouse_info_outside_the_image_is_all_na(app):
    app.update_mouse_info(FakeEvent(CANVAS + 5, CANVAS + 5))
    assert app.intensity_label.cget("text") == "Intensity: N/A"
    assert app.mask_value_label.cget("text") == "Mask: N/A"
    assert app.pixel_count_label.cget("text") == "Area: N/A"


def test_update_mouse_info_in_zoom_mode_reads_the_zoom_arrays(app):
    _enter_zoom(app)
    app.zoom_mask[:] = 0
    app.zoom_mask[0:6, 0:6] = 3
    app.zoom_image_orig[2, 2] = 4242
    app.update_mouse_info(FakeEvent(2, 2))
    assert app.intensity_label.cget("text") == "Intensity: 4242"
    assert app.mask_value_label.cget("text") == "Mask: 3"


def test_update_mouse_info_in_zoom_mode_without_zoom_arrays_is_na(app):
    app.zoom_active = True
    app.zoom_image_orig = None
    app.zoom_mask = None
    app.update_mouse_info(FakeEvent(3, 3))
    assert app.intensity_label.cget("text") == "Intensity: N/A"
    assert app.mask_value_label.cget("text") == "Mask: N/A"
    assert app.pixel_count_label.cget("text") == "Area: N/A"


# ---------------------------------------------------------------------------
# display
# ---------------------------------------------------------------------------

def test_update_display_dispatches_on_the_zoom_flag(app, monkeypatch):
    calls = []
    monkeypatch.setattr(app, "display_image", lambda: calls.append("full"))
    monkeypatch.setattr(app, "display_zoomed_image", lambda: calls.append("zoom"))
    app.zoom_active = False
    app.update_display()
    app.zoom_active = True
    app.update_display()
    assert calls == ["full", "zoom"]


def test_display_image_clears_a_pending_zoom_rectangle(app):
    rect = app.canvas.create_rectangle(1, 1, 10, 10, outline="red")
    app.zoom_rectangle_id = rect
    app.display_image()
    assert app.zoom_rectangle_id is None
    assert rect not in app.canvas.find_all()
    assert app.tk_image.width() == CANVAS
    assert app.tk_image.height() == CANVAS


def test_display_image_falls_back_to_default_percentiles_when_blank(app):
    app.lower_quantile.set("")
    app.upper_quantile.set("")
    app.display_image()
    assert app.tk_image.width() == CANVAS


def test_display_zoomed_image_populates_the_zoom_state(app):
    app.zoom_active = True
    app.zoom_rectangle_start = (8, 8)
    app.zoom_rectangle_end = (56, 56)
    app.display_zoomed_image()
    assert (app.zoom_x0, app.zoom_y0, app.zoom_x1, app.zoom_y1) == (8, 8, 56, 56)
    assert app.zoom_image.shape == (CANVAS, CANVAS)
    assert app.zoom_image_orig.shape == (CANVAS, CANVAS)
    assert app.zoom_mask.shape == (CANVAS, CANVAS)
    assert app.zoom_mask.dtype == np.uint8
    assert app.zoom_scale == pytest.approx((CANVAS * CANVAS) / (48 * 48))
    assert app.tk_image.width() == CANVAS


def test_display_zoomed_image_normalizes_corners_of_the_box(app):
    """The rectangle corners may be dragged in any order."""
    app.zoom_active = True
    app.zoom_rectangle_start = (50, 50)
    app.zoom_rectangle_end = (10, 12)
    app.display_zoomed_image()
    assert (app.zoom_x0, app.zoom_x1) == (10, 50)
    assert (app.zoom_y0, app.zoom_y1) == (12, 50)


def test_display_zoomed_image_is_a_noop_without_a_rectangle(app):
    app.zoom_active = True
    app.zoom_rectangle_start = None
    app.zoom_rectangle_end = None
    app.display_zoomed_image()
    assert app.zoom_image is None
    assert app.zoom_mask is None


def test_display_zoomed_image_skips_painting_before_the_canvas_is_mapped(app):
    """An unmapped canvas reports 0/1 px; the slices happen but nothing renders."""
    painted = app.tk_image  # the full-image render done by load_first_image
    app.canvas.winfo_height = lambda: 0
    app.canvas.winfo_width = lambda: 0
    app.zoom_active = True
    app.zoom_rectangle_start = (4, 4)
    app.zoom_rectangle_end = (40, 40)
    app.display_zoomed_image()
    # the raw slices were taken, but no resize/overlay/paint happened
    assert app.zoom_image.shape == (36, 36)
    assert app.zoom_mask.shape == (36, 36)
    assert app.zoom_image_orig.shape == (36, 36)
    assert app.tk_image is painted


def test_overlay_mask_on_image_blends_only_labelled_pixels(app):
    img = np.full((8, 8), 25600, dtype=np.uint16)  # -> 100 in 8-bit
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    out = app.overlay_mask_on_image(img, mask, alpha=0.5)
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8
    assert np.all(out[0, 0] == 100)  # background keeps the plain intensity

    np.random.seed(0)
    colors = np.random.randint(0, 255, size=(2, 3), dtype=np.uint8)
    expected = np.clip(100 * 0.5 + colors[1].astype(float) * 0.5, 0, 255).astype(np.uint8)
    assert np.array_equal(out[2, 2], expected)


def test_overlay_mask_on_image_accepts_rgb_input_and_empty_masks(app):
    img = np.full((6, 6, 3), 512, dtype=np.uint16)
    mask = np.zeros((6, 6), dtype=np.uint8)
    out = app.overlay_mask_on_image(img, mask)
    assert out.shape == (6, 6, 3)
    assert np.all(out == 2)  # 512 / 256, nothing blended


# ---------------------------------------------------------------------------
# navigation + save
# ---------------------------------------------------------------------------

def test_next_and_previous_image_navigate_and_reset_state(app):
    app.zoom_active = True
    app.next_image()
    assert app.current_image_index == 1
    assert app.zoom_active is False  # initialize_flags ran
    assert app.original_size == (48, 48)
    assert app.image.shape == (CANVAS, CANVAS)
    assert app.mask.max() == 0  # index 1 has no mask on disk

    app.previous_image()
    assert app.current_image_index == 0
    assert app.original_size == (60, 70)
    assert app.mask.max() == 2


def test_previous_image_is_a_noop_at_the_first_index(app):
    before = app.mask.copy()
    app.previous_image()
    assert app.current_image_index == 0
    assert np.array_equal(app.mask, before)


def test_next_image_is_a_noop_at_the_last_index(app):
    app.current_image_index = len(app.image_filenames) - 1
    before = app.mask.copy()
    app.next_image()
    assert app.current_image_index == len(app.image_filenames) - 1
    assert np.array_equal(app.mask, before)


def test_save_mask_resizes_back_to_the_original_and_relabels(app, mask_dir):
    import imageio.v2 as imageio
    app.mask[:] = 0
    app.mask[4:14, 4:14] = 9
    app.mask[40:52, 40:52] = 9  # same label, two disjoint blobs
    app.save_mask()

    saved = imageio.imread(str(mask_dir / "masks" / "a_gray16.tif"))
    assert saved.shape == (60, 70)  # written at the ORIGINAL size
    assert sorted(np.unique(saved).tolist()) == [0, 1, 2]  # relabelled


def test_save_mask_keeps_the_mask_as_is_and_creates_the_masks_folder(app, mask_dir):
    import shutil
    import imageio.v2 as imageio

    app.current_image_index = 3
    app.image, app.mask = app.load_image_and_mask(3)
    app.original_size = app.image.shape
    app.image, app.mask = app.resize_arrays(app.image, app.mask)
    assert app.mask.shape == app.original_size == (CANVAS, CANVAS)

    shutil.rmtree(mask_dir / "masks")  # force the os.makedirs branch
    app.mask[:] = 0
    app.mask[2:6, 2:6] = 4
    app.save_mask()

    out = mask_dir / "masks" / "d_exact.tif"
    assert out.exists()
    saved = imageio.imread(str(out))
    assert saved.shape == (CANVAS, CANVAS)
    assert saved.max() == 1
    assert int((saved > 0).sum()) == 16


def test_save_mask_is_a_noop_past_the_last_image(app, mask_dir):
    app.current_image_index = len(app.image_filenames)
    before = sorted(os.listdir(str(mask_dir / "masks")))
    app.save_mask()
    assert sorted(os.listdir(str(mask_dir / "masks"))) == before


# ---------------------------------------------------------------------------
# zoom rectangle
# ---------------------------------------------------------------------------

def test_set_zoom_rectangle_start_only_records_while_zooming(app):
    app.set_zoom_rectangle_start(FakeEvent(5, 6))
    assert app.zoom_rectangle_start is None
    app.zoom_active = True
    app.set_zoom_rectangle_start(FakeEvent(5, 6))
    assert app.zoom_rectangle_start == (5, 6)


def test_update_zoom_box_replaces_the_preview_rectangle(app):
    app.zoom_active = True
    app.set_zoom_rectangle_start(FakeEvent(2, 2))
    app.update_zoom_box(FakeEvent(20, 30))
    first = app.zoom_rectangle_id
    assert first is not None
    assert app.canvas.coords(first) == [2.0, 2.0, 20.0, 30.0]

    app.update_zoom_box(FakeEvent(40, 50))
    assert app.zoom_rectangle_id != first
    assert first not in app.canvas.find_all()
    assert app.zoom_rectangle_end == (40, 50)


def test_update_zoom_box_ignored_without_a_start_corner(app):
    app.zoom_active = True
    app.zoom_rectangle_start = None
    app.update_zoom_box(FakeEvent(9, 9))
    assert app.zoom_rectangle_id is None
    assert app.zoom_rectangle_end is None


def test_set_zoom_rectangle_end_renders_and_drops_the_preview(app):
    app.zoom_active = True
    app.set_zoom_rectangle_start(FakeEvent(8, 8))
    app.update_zoom_box(FakeEvent(56, 56))
    preview = app.zoom_rectangle_id

    app.set_zoom_rectangle_end(FakeEvent(56, 56))
    assert app.zoom_rectangle_id is None
    assert preview not in app.canvas.find_all()
    assert (app.zoom_x0, app.zoom_y0, app.zoom_x1, app.zoom_y1) == (8, 8, 56, 56)
    assert app.zoom_mask.shape == (CANVAS, CANVAS)
    # motion goes back to the read-out handler once the box is committed
    assert app.canvas.bind("<Motion>") != ""
    assert app.canvas.bind("<Button-1>") == ""
    assert app.canvas.bind("<Button-3>") == ""


def test_set_zoom_rectangle_end_ignored_when_zoom_is_off(app):
    app.zoom_active = False
    app.set_zoom_rectangle_end(FakeEvent(5, 5))
    assert app.zoom_rectangle_end is None


def test_apply_zoom_on_enter_finalizes_a_pending_rectangle(app):
    app.zoom_active = True
    app.zoom_rectangle_start = (4, 4)
    app.apply_zoom_on_enter(FakeEvent(40, 40))
    assert app.zoom_rectangle_end == (40, 40)
    assert (app.zoom_x0, app.zoom_x1) == (4, 40)


def test_apply_zoom_on_enter_ignored_without_a_start_corner(app):
    app.zoom_active = True
    app.zoom_rectangle_start = None
    app.apply_zoom_on_enter(FakeEvent(9, 9))
    assert app.zoom_rectangle_end is None


# ---------------------------------------------------------------------------
# mode toggles
# ---------------------------------------------------------------------------

def test_toggle_zoom_mode_on_then_off(app):
    app.brush_active = True
    app.drawing = True
    app.magic_wand_active = True
    app.toggle_zoom_mode()
    assert app.zoom_active is True
    assert app.drawing is False and app.brush_active is False
    assert app.magic_wand_active is False and app.erase_active is False
    assert app.dividing_line_active is False
    assert app.zoom_btn.cget("text") == "Zoom ON"
    assert app.draw_btn.cget("text") == "Draw"
    assert app.erase_btn.cget("text") == "Erase"
    assert app.magic_wand_btn.cget("text") == "Magic Wand"
    assert app.brush_btn.cget("text") == "Brush"
    assert app.dividing_line_btn.cget("text") == "Dividing Line"
    assert app.canvas.bind("<Button-1>") != ""
    assert app.canvas.bind("<Button-3>") != ""

    app.zoom_rectangle_start = (1, 1)
    app.zoom_rectangle_end = (30, 30)
    app.zoom_mask = np.zeros((4, 4), dtype=np.uint8)
    app.zoom_image = np.zeros((4, 4), dtype=np.uint16)
    app.toggle_zoom_mode()
    assert app.zoom_active is False
    assert app.zoom_btn.cget("text") == "Zoom"
    assert app.zoom_rectangle_start is None and app.zoom_rectangle_end is None
    assert app.zoom_rectangle_id is None
    assert (app.zoom_x0, app.zoom_y0, app.zoom_x1, app.zoom_y1) == (None,) * 4
    assert app.zoom_mask is None and app.zoom_image is None
    assert app.zoom_image_orig is None
    assert app.canvas.bind("<Button-1>") == ""
    assert app.canvas.bind("<Motion>") != ""  # rebound to update_mouse_info


def test_toggle_brush_mode_on_then_off(app):
    app.toggle_brush_mode()
    assert app.brush_active is True
    assert app.erase_active is False
    assert app.drawing is False and app.magic_wand_active is False
    assert app.brush_btn.cget("text") == "Brush ON"
    assert app.canvas.bind("<B1-Motion>") != ""
    assert app.canvas.bind("<B3-Motion>") != ""
    assert app.canvas.bind("<ButtonRelease-1>") != ""
    assert app.canvas.bind("<ButtonRelease-3>") != ""

    app.toggle_brush_mode()
    assert app.brush_active is False
    assert app.brush_btn.cget("text") == "Brush"
    assert app.canvas.bind("<B1-Motion>") == ""
    assert app.canvas.bind("<ButtonRelease-3>") == ""


def test_toggle_draw_mode_on_then_off(app):
    app.toggle_draw_mode()
    assert app.drawing is True
    assert app.draw_coordinates == []
    assert app.magic_wand_active is False and app.brush_active is False
    assert app.erase_active is False
    assert app.draw_btn.cget("text") == "Draw ON"
    assert app.canvas.bind("<B1-Motion>") != ""
    assert app.canvas.bind("<ButtonRelease-1>") != ""

    app.toggle_draw_mode()
    assert app.drawing is False
    assert app.draw_btn.cget("text") == "Draw"
    assert app.canvas.bind("<B1-Motion>") == ""
    assert app.canvas.bind("<ButtonRelease-1>") == ""


def test_toggle_magic_wand_mode_on_then_off(app):
    app.toggle_magic_wand_mode()
    assert app.magic_wand_active is True
    assert app.drawing is False and app.brush_active is False
    assert app.erase_active is False
    assert app.magic_wand_btn.cget("text") == "Magic Wand ON"
    assert app.canvas.bind("<Button-1>") != ""
    assert app.canvas.bind("<Button-3>") != ""

    app.toggle_magic_wand_mode()
    assert app.magic_wand_active is False
    assert app.magic_wand_btn.cget("text") == "Magic Wand"
    assert app.canvas.bind("<Button-1>") == ""
    assert app.canvas.bind("<Button-3>") == ""


def test_toggle_erase_mode_on_then_off_once_the_flag_exists(app):
    # Draw mode seeds erase_active (initialize_flags does not), so this is the
    # only order in which the Erase button currently works.
    app.toggle_draw_mode()
    app.toggle_draw_mode()

    app.toggle_erase_mode()
    assert app.erase_active is True
    assert app.drawing is False and app.brush_active is False
    assert app.magic_wand_active is False
    assert app.erase_btn.cget("text") == "Erase ON"
    assert app.canvas.bind("<Button-1>") != ""

    app.toggle_erase_mode()
    assert app.erase_active is False
    assert app.erase_btn.cget("text") == "Erase"
    assert app.canvas.bind("<Button-1>") == ""


def test_toggle_erase_mode_works_on_a_freshly_loaded_image(app):
    app.toggle_erase_mode()
    assert app.erase_active is True
    assert app.erase_btn.cget("text") == "Erase ON"


def test_toggle_dividing_line_mode_on_then_off(app):
    app.toggle_dividing_line_mode()
    assert app.dividing_line_active is True
    assert app.drawing is False and app.brush_active is False
    assert app.magic_wand_active is False and app.erase_active is False
    assert app.dividing_line_btn.cget("text") == "Dividing Line ON"
    assert app.canvas.bind("<Button-1>") != ""
    assert app.canvas.bind("<ButtonRelease-1>") != ""

    app.toggle_dividing_line_mode()
    assert app.dividing_line_active is False
    assert app.dividing_line_btn.cget("text") == "Dividing Line"
    assert app.canvas.bind("<Button-1>") == ""
    assert app.canvas.bind("<ButtonRelease-1>") == ""


# ---------------------------------------------------------------------------
# brush
# ---------------------------------------------------------------------------

def _set_entry(entry, value):
    entry.delete(0, "end")
    entry.insert(0, value)


def test_apply_brush_records_the_stroke_and_release_paints_it(app):
    app.mask[:] = 0
    _set_entry(app.brush_size_entry, "6")

    app.apply_brush(FakeEvent(20, 20))
    assert app.brush_path == [(20, 20, 6)]
    assert app.last_brush_coord == (20, 20)
    app.apply_brush(FakeEvent(26, 20))
    assert len(app.brush_path) == 8  # 1 + the 7 rasterized line pixels
    assert all(size == 6 for _, _, size in app.brush_path)
    assert app.last_brush_coord == (26, 20)
    assert app.canvas.find_withtag("temp_line") != ()

    app.apply_brush_release(FakeEvent(26, 20))
    assert not hasattr(app, "brush_path")
    assert app.canvas.find_withtag("temp_line") == ()
    assert app.mask[20, 20] == 255
    assert app.mask[0, 0] == 0
    assert app.mask.max() == 255


def test_apply_brush_release_without_a_stroke_is_a_noop(app):
    before = app.mask.copy()
    app.apply_brush_release(FakeEvent(1, 1))
    assert np.array_equal(app.mask, before)


def test_erase_brush_records_the_stroke_and_release_clears_it(app):
    app.mask[:] = 50
    _set_entry(app.brush_size_entry, "4")

    app.erase_brush(FakeEvent(10, 10))
    app.erase_brush(FakeEvent(10, 16))
    assert len(app.erase_path) == 8
    assert app.last_erase_coord == (10, 16)

    app.erase_brush_release(FakeEvent(10, 16))
    assert not hasattr(app, "erase_path")
    assert app.mask[10, 10] == 0
    assert app.mask[0, 0] == 50


def test_erase_brush_release_without_a_stroke_is_a_noop(app):
    before = app.mask.copy()
    app.erase_brush_release(FakeEvent(1, 1))
    assert np.array_equal(app.mask, before)


def test_apply_brush_release_in_zoom_mode_writes_back_to_the_full_mask(app):
    _enter_zoom(app)
    app.mask[:] = 0
    app.zoom_mask[:] = 0
    app.brush_path = [(30, 30, 8)]
    app.apply_brush_release(FakeEvent(30, 30))
    assert not hasattr(app, "brush_path")
    # the stroke landed inside the [4:60, 4:60] zoom window of the full mask
    assert app.mask.max() == 255
    assert app.mask[:4, :].max() == 0
    assert app.mask[60:, :].max() == 0
    assert app.mask[:, :4].max() == 0


def test_erase_brush_release_in_zoom_mode_writes_back_to_the_full_mask(app):
    _enter_zoom(app)
    app.mask[:] = 200
    app.zoom_mask[:] = 200
    app.erase_path = [(30, 30, 8)]
    app.erase_brush_release(FakeEvent(30, 30))
    assert not hasattr(app, "erase_path")
    assert app.mask[4:60, 4:60].min() == 0  # a hole was punched
    assert app.mask[:4, :].min() == 200  # outside the window untouched


# ---------------------------------------------------------------------------
# erase whole object
# ---------------------------------------------------------------------------

def test_erase_object_removes_the_whole_label_under_the_cursor(app):
    app.mask[:] = 0
    app.mask[10:20, 10:20] = 5
    app.mask[30:40, 30:40] = 6
    app.erase_object(FakeEvent(15, 15))
    assert app.mask[10:20, 10:20].max() == 0
    assert app.mask[30:40, 30:40].min() == 6


def test_erase_object_on_background_changes_nothing(app):
    app.mask[:] = 0
    app.mask[30:40, 30:40] = 6
    before = app.mask.copy()
    app.erase_object(FakeEvent(2, 2))
    assert np.array_equal(app.mask, before)


def test_erase_object_in_zoom_mode_maps_the_click_back_to_the_original(app):
    _enter_zoom(app, ((0, 0), (CANVAS, CANVAS)))
    app.mask[:] = 0
    app.mask[10:20, 10:20] = 5
    app.erase_object(FakeEvent(15, 15))
    assert app.mask.max() == 0


def test_erase_object_out_of_bounds_in_zoom_mode_returns_early(app, capsys):
    _enter_zoom(app, ((0, 0), (CANVAS, CANVAS)))
    app.mask[:] = 0
    app.mask[10:20, 10:20] = 5
    app.erase_object(FakeEvent(-20, -20))
    assert "out of bounds" in capsys.readouterr().out
    assert app.mask.max() == 5  # nothing was erased


# ---------------------------------------------------------------------------
# magic wand
# ---------------------------------------------------------------------------

def test_apply_magic_wand_add_respects_tolerance(app):
    image = np.zeros((10, 10), dtype=np.uint16)
    image[3:7, 3:7] = 500
    mask = np.zeros((10, 10), dtype=np.uint8)
    out = app.apply_magic_wand(image, mask, (5, 5), tolerance=10, maximum=1000,
                               action="add")
    assert out is mask  # mutated in place and returned
    assert mask[3:7, 3:7].min() == 255
    assert mask[0, 0] == 0
    assert int((mask > 0).sum()) == 16


def test_apply_magic_wand_stops_at_the_maximum(app):
    image = np.zeros((20, 20), dtype=np.uint16)
    mask = np.zeros((20, 20), dtype=np.uint8)
    app.apply_magic_wand(image, mask, (10, 10), tolerance=10, maximum=5,
                         action="add")
    assert int((mask > 0).sum()) == 5


def test_apply_magic_wand_erase_zeroes_the_region(app):
    image = np.zeros((10, 10), dtype=np.uint16)
    image[3:7, 3:7] = 500
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 255
    app.apply_magic_wand(image, mask, (5, 5), tolerance=10, maximum=1000,
                         action="erase")
    assert mask.max() == 0


def test_magic_wand_normal_adds_to_the_full_mask(app):
    app.image[:] = 0
    app.image[20:30, 20:30] = 1000
    app.mask[:] = 0
    app.magic_wand_normal((25, 25), tolerance=10, action="add")
    assert app.mask[20:30, 20:30].min() == 255
    assert app.mask[0, 0] == 0


def test_magic_wand_normal_falls_back_on_a_bad_max_pixels_entry(app, capsys):
    _set_entry(app.max_pixels_entry, "not-a-number")
    app.image[:] = 0
    app.image[20:30, 20:30] = 1000
    app.mask[:] = 0
    app.magic_wand_normal((25, 25), tolerance=10, action="add")
    assert "Invalid maximum value" in capsys.readouterr().out
    assert app.mask[25, 25] == 255


def test_use_magic_wand_left_adds_and_right_erases(app):
    app.magic_wand_tolerance.set("10")
    app.image[:] = 0
    app.image[20:30, 20:30] = 1000
    app.mask[:] = 0

    app.use_magic_wand(FakeEvent(25, 25, num=1))
    assert app.mask[25, 25] == 255
    assert int((app.mask > 0).sum()) == 100

    app.use_magic_wand(FakeEvent(25, 25, num=3))
    assert app.mask.max() == 0


def test_use_magic_wand_dispatches_to_the_zoomed_variant(app):
    _enter_zoom(app, ((0, 0), (CANVAS, CANVAS)))
    app.magic_wand_tolerance.set("10")
    app.image[:] = 0
    app.image[20:30, 20:30] = 1000
    app.mask[:] = 0
    app.display_zoomed_image()  # refresh zoom_* from the edited arrays

    app.use_magic_wand(FakeEvent(25, 25, num=1))
    assert app.mask[25, 25] == 255
    assert app.mask[0, 0] == 0


def test_use_magic_wand_tolerates_a_bad_max_pixels_entry(app):
    _set_entry(app.max_pixels_entry, "not-a-number")
    app.magic_wand_tolerance.set("10")
    app.image[:] = 0
    app.image[20:30, 20:30] = 1000
    app.mask[:] = 0
    app.use_magic_wand(FakeEvent(25, 25, num=1))
    assert app.mask[25, 25] == 255


def test_magic_wand_zoomed_returns_when_the_zoom_state_is_missing(app, capsys):
    app.zoom_image_orig = None
    app.zoom_mask = None
    before = app.mask.copy()
    app.magic_wand_zoomed((5, 5), 10, "add")
    assert "not initialized" in capsys.readouterr().out
    assert np.array_equal(app.mask, before)


def test_magic_wand_zoomed_returns_for_an_out_of_bounds_seed(app, capsys):
    _enter_zoom(app)
    before = app.mask.copy()
    app.magic_wand_zoomed((CANVAS + 10, 2), 10, "add")
    assert "out of bounds" in capsys.readouterr().out
    assert np.array_equal(app.mask, before)


def test_magic_wand_zoomed_add_writes_through_to_the_full_mask(app):
    _enter_zoom(app, ((0, 0), (CANVAS, CANVAS)))
    app.image[:] = 0
    app.image[20:30, 20:30] = 1000
    app.mask[:] = 0
    app.display_zoomed_image()
    app.magic_wand_zoomed((25, 25), 10, "add")
    assert app.mask[20:30, 20:30].min() == 255
    assert app.mask[0, 0] == 0


def test_magic_wand_zoomed_erase_clears_the_full_mask_window(app):
    _enter_zoom(app, ((0, 0), (CANVAS, CANVAS)))
    app.image[:] = 0
    app.image[20:30, 20:30] = 1000
    app.mask[:] = 0
    app.mask[20:30, 20:30] = 255
    app.display_zoomed_image()
    app.magic_wand_zoomed((25, 25), 10, "erase")
    assert app.mask[20:30, 20:30].max() == 0


def test_magic_wand_zoomed_falls_back_on_a_bad_max_pixels_entry(app, capsys):
    _enter_zoom(app, ((0, 0), (CANVAS, CANVAS)))
    _set_entry(app.max_pixels_entry, "not-a-number")
    app.image[:] = 0
    app.image[20:30, 20:30] = 1000
    app.mask[:] = 0
    app.display_zoomed_image()
    app.magic_wand_zoomed((25, 25), 10, "add")
    assert "Invalid maximum value" in capsys.readouterr().out
    assert app.mask[25, 25] == 255


# ---------------------------------------------------------------------------
# freehand drawing
# ---------------------------------------------------------------------------

def test_draw_collects_coordinates_and_previews_the_polygon(app):
    app.toggle_draw_mode()
    app.draw(FakeEvent(5, 5))
    assert app.draw_coordinates == [(5, 5)]
    assert not hasattr(app, "current_line")  # nothing to connect yet
    app.draw(FakeEvent(15, 5))
    assert app.draw_coordinates == [(5, 5), (15, 5)]
    assert app.canvas.coords(app.current_line) == [5.0, 5.0, 15.0, 5.0]


def test_draw_is_ignored_outside_draw_mode(app):
    app.draw_coordinates = []
    app.drawing = False
    app.draw(FakeEvent(3, 3))
    assert app.draw_coordinates == []


def test_draw_on_zoomed_mask_rasterizes_the_polygon(app):
    out = app.draw_on_zoomed_mask([(5, 5), (25, 5), (25, 25), (5, 25), (5, 5)])
    assert out.shape == (CANVAS, CANVAS)
    assert out.dtype == np.uint8
    assert out[15, 15] == 255
    assert out[1, 1] == 0


def test_finish_drawing_fills_the_polygon_into_the_mask(app):
    app.toggle_draw_mode()
    app.mask[:] = 0
    for pt in ((10, 10), (30, 10), (30, 30), (10, 30)):
        app.draw(FakeEvent(*pt))
    app.finish_drawing(FakeEvent(10, 30))
    assert app.mask[20, 20] == 255
    assert app.mask[5, 5] == 0
    assert app.draw_coordinates == []


def test_finish_drawing_needs_more_than_two_points(app):
    app.toggle_draw_mode()
    app.mask[:] = 0
    app.draw(FakeEvent(1, 1))
    app.draw(FakeEvent(2, 2))
    app.finish_drawing(FakeEvent(2, 2))
    assert app.mask.max() == 0
    assert len(app.draw_coordinates) == 2  # left untouched


def test_finish_drawing_in_zoom_mode_merges_into_the_original_mask(app):
    _enter_zoom(app, ((0, 0), (CANVAS, CANVAS)))
    app.mask[:] = 0
    app.drawing = True
    app.draw_coordinates = []
    for pt in ((10, 10), (40, 10), (40, 40), (10, 40)):
        app.draw(FakeEvent(*pt))
    app.finish_drawing(FakeEvent(10, 40))
    assert app.mask[25, 25] == 255
    assert app.mask[2, 2] == 0


def test_finish_drawing_if_active_only_fires_in_draw_mode(app, monkeypatch):
    calls = []
    monkeypatch.setattr(app, "finish_drawing", lambda ev: calls.append(ev))
    app.draw_coordinates = [(1, 1), (2, 2), (3, 3)]
    app.drawing = False
    app.finish_drawing_if_active(FakeEvent(1, 1))
    assert calls == []
    app.drawing = True
    app.finish_drawing_if_active(FakeEvent(1, 1))
    assert len(calls) == 1


def test_update_original_mask_merges_with_a_pixelwise_maximum(app):
    app.mask[:] = 0
    app.mask[10:12, 10:12] = 7
    patch = np.full((20, 20), 3, dtype=np.uint8)
    app.update_original_mask(patch, 5, 25, 5, 25)
    assert app.mask[10, 10] == 7  # the stronger existing label survives
    assert app.mask[6, 6] == 3  # the patch fills the empty pixels
    assert app.mask[0, 0] == 0  # outside the box nothing changed


def test_update_original_mask_from_zoom_writes_back_the_window(app):
    _enter_zoom(app, ((8, 8), (56, 56)))
    app.mask[:] = 0
    app.zoom_mask[:] = 0
    app.zoom_mask[10:50, 10:50] = 9
    app.update_original_mask_from_zoom()
    assert app.mask[8:56, 8:56].max() == 9
    assert app.mask[:8, :].max() == 0
    assert app.mask[56:, :].max() == 0


# ---------------------------------------------------------------------------
# dividing line
# ---------------------------------------------------------------------------

def test_start_dividing_line_records_the_first_point(app):
    app.toggle_dividing_line_mode()
    app.start_dividing_line(FakeEvent(7, 8))
    assert app.dividing_line_coords == [(7, 8)]
    assert app.canvas.coords(app.current_dividing_line) == [7.0, 8.0, 7.0, 8.0]


def test_start_dividing_line_ignored_when_the_mode_is_off(app):
    app.dividing_line_coords = []
    app.start_dividing_line(FakeEvent(1, 2))
    assert app.dividing_line_coords == []
    assert app.current_dividing_line is None


def test_update_dividing_line_preview_extends_the_stroke(app):
    app.toggle_dividing_line_mode()
    app.start_dividing_line(FakeEvent(4, 4))
    app.update_dividing_line_preview(FakeEvent(4, 20))
    assert app.dividing_line_coords == [(4, 4), (4, 20)]
    assert app.canvas.coords(app.current_dividing_line) == [4.0, 4.0, 4.0, 20.0]


def test_update_dividing_line_preview_ignored_without_a_stroke(app):
    app.dividing_line_active = True
    app.dividing_line_coords = []
    app.update_dividing_line_preview(FakeEvent(4, 20))
    assert app.dividing_line_coords == []


def test_update_dividing_line_preview_in_zoom_mode_converts_coordinates(app):
    _enter_zoom(app, ((0, 0), (CANVAS, CANVAS)))
    app.dividing_line_active = True
    app.dividing_line_coords = [(4, 4)]
    app.current_dividing_line = app.canvas.create_line(4, 4, 4, 4)
    app.update_dividing_line_preview(FakeEvent(20, 30))
    assert app.dividing_line_coords == [(4, 4), (20, 30)]
    assert app.canvas.coords(app.current_dividing_line) == [4.0, 4.0, 20.0, 30.0]


def test_finish_dividing_line_cuts_the_object_in_two_and_relabels(app):
    app.toggle_dividing_line_mode()
    app.mask[:] = 0
    app.mask[10:40, 10:40] = 1
    app.start_dividing_line(FakeEvent(25, 5))
    app.update_dividing_line_preview(FakeEvent(25, 20))
    app.finish_dividing_line(FakeEvent(25, 50))

    assert app.current_dividing_line is None
    assert app.dividing_line_coords == []
    assert app.dividing_line_active is False
    assert app.dividing_line_btn.cget("text") == "Dividing Line"
    assert app.mask.max() == 2
    assert app.mask[20, 15] != app.mask[20, 35]
    assert app.mask[20, 25] == 0  # the cut itself is background
    assert app.canvas.bind("<Button-1>") == ""


def test_finish_dividing_line_ignored_when_the_mode_is_off(app):
    app.dividing_line_active = False
    app.dividing_line_coords = []
    before = app.mask.copy()
    app.finish_dividing_line(FakeEvent(1, 1))
    assert app.dividing_line_coords == []
    assert np.array_equal(app.mask, before)


def test_finish_dividing_line_in_zoom_mode_writes_through(app):
    _enter_zoom(app, ((0, 0), (CANVAS, CANVAS)))
    app.mask[:] = 0
    app.mask[10:40, 10:40] = 1
    app.display_zoomed_image()
    app.dividing_line_active = True
    app.dividing_line_coords = [(25, 5)]
    app.current_dividing_line = app.canvas.create_line(25, 5, 25, 5)
    app.finish_dividing_line(FakeEvent(25, 50))
    assert app.current_dividing_line is None
    assert app.mask.max() == 2
    assert app.mask[20, 15] != app.mask[20, 35]


def test_apply_dividing_line_without_coordinates_is_a_noop(app):
    app.dividing_line_coords = []
    before = app.mask.copy()
    app.apply_dividing_line()
    assert np.array_equal(app.mask, before)


# ---------------------------------------------------------------------------
# single-function buttons
# ---------------------------------------------------------------------------

def test_apply_normalization_pushes_the_entries_and_repaints(app, monkeypatch):
    calls = []
    monkeypatch.setattr(app, "update_display", lambda: calls.append(1))
    _set_entry(app.lower_entry, "5.0")
    _set_entry(app.upper_entry, "95.0")
    app.apply_normalization()
    assert app.lower_quantile.get() == "5.0"
    assert app.upper_quantile.get() == "95.0"
    assert calls == [1]


def test_fill_objects_fills_holes_then_relabels(app):
    app.mask[:] = 0
    app.mask[10:30, 10:30] = 4
    app.mask[15:25, 15:25] = 0  # a hole punched in the middle
    app.fill_objects()
    assert app.mask[20, 20] == 1  # hole filled and relabelled from 1
    assert app.mask.max() == 1
    assert int((app.mask > 0).sum()) == 400


def test_relabel_objects_renumbers_components_consecutively(app):
    app.mask[:] = 0
    app.mask[2:6, 2:6] = 200
    app.mask[40:46, 40:46] = 90
    app.relabel_objects()
    assert sorted(np.unique(app.mask).tolist()) == [0, 1, 2]
    assert app.mask[3, 3] == 1
    assert app.mask[42, 42] == 2


def test_clear_objects_zeroes_the_mask(app):
    app.mask[:] = 5
    app.clear_objects()
    assert app.mask.max() == 0
    assert app.mask.shape == (CANVAS, CANVAS)


def test_invert_mask_swaps_foreground_and_background(app):
    app.mask[:] = 0
    app.mask[:, :32] = 1
    app.invert_mask()
    assert app.mask[0, 0] == 0
    assert app.mask[0, 40] == 1
    assert app.mask.max() == 1
    assert int((app.mask > 0).sum()) == CANVAS * 32


def test_remove_small_objects_drops_components_below_min_area(app):
    app.mask[:] = 0
    app.mask[2:5, 2:5] = 1  # 9 px  -> removed
    app.mask[30:45, 30:45] = 2  # 225 px -> kept
    _set_entry(app.min_area_entry, "100")
    app.remove_small_objects()
    assert app.mask[3, 3] == 0
    assert app.mask[35, 35] == 2


def test_remove_small_objects_falls_back_on_a_bad_min_area(app, capsys):
    app.mask[:] = 0
    app.mask[2:5, 2:5] = 1
    app.mask[30:45, 30:45] = 2
    _set_entry(app.min_area_entry, "")
    app.remove_small_objects()
    assert "Invalid minimum area" in capsys.readouterr().out
    assert app.mask[3, 3] == 0
    assert app.mask[35, 35] == 2


# ---------------------------------------------------------------------------
# defensive / buggy paths
# ---------------------------------------------------------------------------

def test_display_zoomed_image_survives_a_zero_width_selection(app):
    app.zoom_active = True
    app.zoom_rectangle_start = (10, 10)
    app.zoom_rectangle_end = (10, 40)
    app.display_zoomed_image()
    assert app.zoom_image.size == 0
    assert app.zoom_mask.shape == (30, 0)
