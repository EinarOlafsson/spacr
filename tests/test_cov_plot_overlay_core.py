"""CPU-only coverage for the two big overlay renderers in ``spacr.plot``.

Targets ``plot_image_mask_overlay`` (the current renderer, deterministic
seeded colormaps, cell/nucleus/pathogen/organelle aware) and its legacy
sibling ``plot_image_mask_overlay_magenta_outlines``.  Both are ~340-line
closures over half a dozen nested helpers that the rest of the suite never
executes.

Every branch is driven with a tiny (48x48) synthetic ``.npy`` stack laid
out the way ``preprocess_generate_masks`` writes them --
``[intensity planes..., cell_mask, nucleus_mask, pathogen_mask]`` -- so the
mask-dimension arithmetic inside both functions is exercised for real:

* ``mode='outlines'`` vs ``mode='masks'``
* ``all_on_all`` / ``all_outlines`` / per-channel outline dispatch
* ``filter_dict`` area+intensity filtering (objects kept, dropped, and the
  zero-object bookkeeping branch)
* ``export_tiffs``, ``save_pdf``
* the ``n_labels <= 0`` colormap guard (negative label plane)
* the empty-mask "no objects" panel

Assertions are on rendered pixel content (the arrays handed to
``imshow``), emitted files and captured stdout -- not on "it didn't
raise".  ``plot_image_mask_overlay`` uses seeded colormaps so its output is
compared exactly; the legacy function reseeds from ``random.randint`` on
every call, so only alpha-driven support (which pixels are painted at all)
is asserted there.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

tiff = pytest.importorskip("tifffile")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Synthetic stack builders
# ---------------------------------------------------------------------------

SHAPE = (48, 48)

# Geometry (kept constant so tests can address individual objects):
#   cell     -> label 1 at (12,12) r=7,  label 2 at (34,34) r=7
#   nucleus  -> label 1 at (12,12) r=3   (only inside cell 1)
#   pathogen -> label 1 at (13,13) r=2   (only inside cell 1)
# => the neighbourhood of cell 2 contains a cell outline and nothing else,
#    which is what the colour-assignment assertions key off.
CELL_2_BOX = (np.s_[25:44], np.s_[25:44])


def _labels(centers_radii):
    """Disc label image; label i+1 for the i-th (cy, cx, r) entry."""
    h, w = SHAPE
    yy, xx = np.mgrid[:h, :w]
    lbl = np.zeros(SHAPE, dtype=np.int32)
    for i, (cy, cx, r) in enumerate(centers_radii, start=1):
        lbl[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return lbl


def cell_mask():
    return _labels([(12, 12, 7), (34, 34, 7)])


def nucleus_mask():
    return _labels([(12, 12, 3)])


def pathogen_mask():
    return _labels([(13, 13, 2)])


def _make_stack(tmp_path, n_intensity=3, masks=(), dtype=np.uint16, name="fov.npy"):
    """Write ``<tmp>/stack/<name>`` and return (path, stack array).

    The file lives one directory below the "project" root so that the
    functions' ``dirname(dirname(file))/results/overlay`` PDF path lands
    inside ``tmp_path``.
    """
    rng = np.random.default_rng(7)
    planes = [rng.integers(100, 4000, size=SHAPE).astype(np.float64)
              for _ in range(n_intensity)]
    planes += [np.asarray(m, dtype=np.float64) for m in masks]
    stack = np.stack(planes, axis=-1).astype(dtype)
    stack_dir = tmp_path / "stack"
    stack_dir.mkdir(exist_ok=True)
    path = stack_dir / name
    np.save(path, stack)
    return str(path), stack


def _panel(fig, index):
    """The RGB array that was handed to ``imshow`` for one subplot."""
    return np.asarray(fig.axes[index].images[0].get_array())


def _has_color(panel, rgb, region=None):
    sub = panel[region] if region is not None else panel
    return bool(np.any(np.all(np.isclose(sub, rgb, atol=1e-6), axis=-1)))


def _normalized(stack, channel, percentiles=(2, 98)):
    """The grayscale plane a panel shows before any overlay is applied."""
    plane = stack[..., channel]
    if plane.dtype in (np.uint16, np.uint8):
        plane = plane.astype(np.float32)
    v_min, v_max = np.percentile(plane, percentiles)
    return np.clip((plane - v_min) / (v_max - v_min + 1e-5), 0, 1)


def _painted(fig, index, stack, channel):
    """Boolean map of pixels the overlay changed relative to the raw channel.

    Checking against the un-overlaid grayscale is the only reliable test:
    a saturated overlay colour can still have r == b (hue 1/3 does), so
    "is this pixel grayscale" is not a usable proxy.
    """
    panel = _panel(fig, index)
    gray = _normalized(stack, channel)
    return np.any(~np.isclose(panel, gray[..., None]), axis=-1)


RED, GREEN, BLUE, MAGENTA = (1., 0., 0.), (0., .5019607843137255, 0.), (0., 0., 1.), (1., 0., 1.)


# ---------------------------------------------------------------------------
# plot_image_mask_overlay -- outline mode, the common path
# ---------------------------------------------------------------------------

def test_overlay_outlines_titles_colors_and_pdf(tmp_path):
    """Three masks -> one titled panel per channel, per-object colours, a PDF."""
    from spacr.plot import plot_image_mask_overlay

    path, stack = _make_stack(
        tmp_path, 3, [cell_mask(), nucleus_mask(), pathogen_mask()])

    fig = plot_image_mask_overlay(
        path, [0, 1, 2], cell_channel=0, nucleus_channel=1, pathogen_channel=2,
        figuresize=2, thickness=1, save_pdf=True, mode="outlines")

    assert [a.get_title() for a in fig.axes] == [
        "cell (channel 0)", "nucleus (channel 1)",
        "pathogen (channel 2)", "combined objects"]
    # Every panel is the full field, RGB, in [0, 1].
    for i in range(4):
        p = _panel(fig, i)
        assert p.shape == (*SHAPE, 3)
        assert p.min() >= 0.0 and p.max() <= 1.0

    # Each intensity channel only carries its own object's outline colour.
    assert _has_color(_panel(fig, 0), RED)
    assert not _has_color(_panel(fig, 0), BLUE)
    assert _has_color(_panel(fig, 1), BLUE)
    assert _has_color(_panel(fig, 2), GREEN)

    pdf = tmp_path / "results" / "overlay" / "fov.pdf"
    assert pdf.is_file() and pdf.stat().st_size > 0
    assert pdf.read_bytes()[:4] == b"%PDF"


def test_overlay_combined_panel_support_is_union_of_masks(tmp_path):
    """The last panel paints exactly the union of all mask pixels on black."""
    from spacr.plot import plot_image_mask_overlay

    cell, nuc, pat = cell_mask(), nucleus_mask(), pathogen_mask()
    path, _ = _make_stack(tmp_path, 3, [cell, nuc, pat])

    fig = plot_image_mask_overlay(
        path, [0, 1, 2], 0, 1, 2, figuresize=2, thickness=1, save_pdf=False)

    combined = _panel(fig, -1)
    painted = combined.max(axis=-1) > 0
    union = (cell > 0) | (nuc > 0) | (pat > 0)
    assert np.array_equal(painted, union)
    # Background is left pure black.
    assert np.all(combined[~union] == 0)


def test_overlay_label_offsets_keep_objects_distinct(tmp_path):
    """Overlapping masks are offset, so nucleus/pathogen recolour cell pixels."""
    from spacr.plot import plot_image_mask_overlay

    cell, nuc, pat = cell_mask(), nucleus_mask(), pathogen_mask()
    path, _ = _make_stack(tmp_path, 3, [cell, nuc, pat])

    fig = plot_image_mask_overlay(
        path, [0, 1, 2], 0, 1, 2, figuresize=2, thickness=1, save_pdf=False)
    combined = _panel(fig, -1)

    # Pathogen is drawn last -> its pixels differ from the cell-only pixels
    # they sit on top of.
    cell_only = (cell == 1) & (nuc == 0) & (pat == 0)
    pat_px = combined[pat == 1]
    cell_px = combined[cell_only]
    assert len(np.unique(pat_px.reshape(-1, 3), axis=0)) == 1
    assert not np.allclose(pat_px[0], cell_px[0])
    # Two different cells get two different colours.
    c1 = combined[(cell == 1) & (nuc == 0) & (pat == 0)][0]
    c2 = combined[cell == 2][0]
    assert not np.allclose(c1, c2)


def test_overlay_is_deterministic_across_calls(tmp_path):
    """Seeded colormaps -> byte-identical panels for two identical calls."""
    from spacr.plot import plot_image_mask_overlay

    path, _ = _make_stack(tmp_path, 3, [cell_mask(), nucleus_mask(), pathogen_mask()])
    kw = dict(figuresize=2, thickness=1, save_pdf=False, mode="masks")

    fig_a = plot_image_mask_overlay(path, [0, 1, 2], 0, 1, 2, **kw)
    panels_a = [_panel(fig_a, i).copy() for i in range(4)]
    plt.close(fig_a)
    fig_b = plot_image_mask_overlay(path, [0, 1, 2], 0, 1, 2, **kw)
    panels_b = [_panel(fig_b, i).copy() for i in range(4)]

    for a, b in zip(panels_a, panels_b):
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# plot_image_mask_overlay -- mask (filled) mode and the all_* dispatch
# ---------------------------------------------------------------------------

def test_overlay_masks_mode_paints_only_inside_its_own_mask(tmp_path):
    """mode='masks', per-channel dispatch: channel 1 is recoloured on nucleus."""
    from spacr.plot import plot_image_mask_overlay

    cell, nuc, pat = cell_mask(), nucleus_mask(), pathogen_mask()
    path, stack = _make_stack(tmp_path, 3, [cell, nuc, pat])

    fig = plot_image_mask_overlay(
        path, [0, 1, 2], 0, 1, 2, figuresize=2, save_pdf=False, mode="masks")

    # Channel 1 is the nucleus channel: exactly the nucleus is repainted --
    # not the cell it sits in, not the pathogen next to it.
    assert np.array_equal(_painted(fig, 1, stack, 1), nuc > 0)
    assert np.array_equal(_painted(fig, 0, stack, 0), cell > 0)
    assert np.array_equal(_painted(fig, 2, stack, 2), pat > 0)
    # The single nucleus is filled with one flat colour.
    inside = np.unique(_panel(fig, 1)[nuc == 1].reshape(-1, 3), axis=0)
    assert len(inside) == 1


def test_overlay_all_on_all_masks_mode_paints_every_channel(tmp_path):
    """all_on_all + mode='masks' -> every panel is painted on every object."""
    from spacr.plot import plot_image_mask_overlay

    cell, nuc, pat = cell_mask(), nucleus_mask(), pathogen_mask()
    path, stack = _make_stack(tmp_path, 3, [cell, nuc, pat])

    fig = plot_image_mask_overlay(
        path, [0, 1, 2], 0, 1, 2, figuresize=2, save_pdf=False,
        mode="masks", all_on_all=True)

    union = (cell > 0) | (nuc > 0) | (pat > 0)
    for i in range(3):
        # Every panel is repainted on exactly the union of all three masks.
        assert np.array_equal(_painted(fig, i, stack, i), union)


def test_overlay_all_on_all_outlines_mode_draws_all_three_colors(tmp_path):
    """all_on_all + mode='outlines' -> red, blue and green on every panel."""
    from spacr.plot import plot_image_mask_overlay

    path, _ = _make_stack(tmp_path, 3, [cell_mask(), nucleus_mask(), pathogen_mask()])

    fig = plot_image_mask_overlay(
        path, [0, 1, 2], 0, 1, 2, figuresize=2, thickness=1, save_pdf=False,
        mode="outlines", all_on_all=True)

    for i in range(3):
        p = _panel(fig, i)
        assert _has_color(p, RED)
        assert _has_color(p, BLUE)
        assert _has_color(p, GREEN)


def test_overlay_all_outlines_decorates_maskless_channel(tmp_path):
    """A channel with no mask of its own gets every outline in its own colour."""
    from spacr.plot import plot_image_mask_overlay

    path, _ = _make_stack(tmp_path, 4, [cell_mask(), nucleus_mask(), pathogen_mask()])

    fig = plot_image_mask_overlay(
        path, [0, 1, 2, 3], 0, 1, 2, figuresize=2, thickness=1, save_pdf=False,
        mode="outlines", all_outlines=True)

    assert [a.get_title() for a in fig.axes] == [
        "cell (channel 0)", "nucleus (channel 1)", "pathogen (channel 2)",
        "channel 3", "combined objects"]
    extra = _panel(fig, 3)
    assert _has_color(extra, RED)
    assert _has_color(extra, BLUE)
    assert _has_color(extra, GREEN)
    # Around the lone cell (no nucleus/pathogen there) only red is used.
    assert _has_color(extra, RED, region=CELL_2_BOX)
    assert not _has_color(extra, GREEN, region=CELL_2_BOX)


def test_overlay_all_outlines_masks_mode_fills_maskless_channel(tmp_path):
    """all_outlines + mode='masks' fills the maskless channel with objects."""
    from spacr.plot import plot_image_mask_overlay

    cell, nuc, pat = cell_mask(), nucleus_mask(), pathogen_mask()
    path, stack = _make_stack(tmp_path, 4, [cell, nuc, pat])

    fig = plot_image_mask_overlay(
        path, [0, 1, 2, 3], 0, 1, 2, figuresize=2, save_pdf=False,
        mode="masks", all_outlines=True)

    union = (cell > 0) | (nuc > 0) | (pat > 0)
    assert np.array_equal(_painted(fig, 3, stack, 3), union)
    # The channels that own a mask still only show their own object.
    assert np.array_equal(_painted(fig, 2, stack, 2), pat > 0)


def test_overlay_maskless_channel_left_untouched_without_all_outlines(tmp_path):
    """Without all_* flags a maskless channel stays pure grayscale."""
    from spacr.plot import plot_image_mask_overlay

    path, stack = _make_stack(tmp_path, 4, [cell_mask(), nucleus_mask(), pathogen_mask()])

    fig = plot_image_mask_overlay(
        path, [0, 1, 2, 3], 0, 1, 2, figuresize=2, thickness=1, save_pdf=False)

    extra = _panel(fig, 3)
    assert np.array_equal(extra[..., 0], extra[..., 1])
    assert np.array_equal(extra[..., 1], extra[..., 2])
    # And it is the percentile-normalised intensity plane.
    plane = stack[..., 3].astype(np.float32)
    v_min, v_max = np.percentile(plane, (2, 98))
    expected = np.clip((plane - v_min) / (v_max - v_min + 1e-5), 0, 1)
    assert np.allclose(extra[..., 0], expected)


def test_overlay_organelle_channel_adds_a_fourth_object(tmp_path):
    """organelle_channel shifts every mask dimension and adds a yellow panel."""
    from spacr.plot import plot_image_mask_overlay

    organelle = _labels([(34, 34, 3)])
    path, _ = _make_stack(
        tmp_path, 4, [cell_mask(), nucleus_mask(), pathogen_mask(), organelle])

    fig = plot_image_mask_overlay(
        path, [0, 1, 2, 3], 0, 1, 2, organelle_channel=3,
        figuresize=2, thickness=1, save_pdf=False, mode="outlines")

    assert [a.get_title() for a in fig.axes] == [
        "cell (channel 0)", "nucleus (channel 1)", "pathogen (channel 2)",
        "organelle (channel 3)", "combined objects"]
    assert _has_color(_panel(fig, 3), (1.0, 1.0, 0.0))
    # The organelle really came from the last plane, not the pathogen plane.
    combined = _panel(fig, -1)
    assert (combined.max(axis=-1) > 0)[organelle == 1].all()


# ---------------------------------------------------------------------------
# plot_image_mask_overlay -- degenerate inputs
# ---------------------------------------------------------------------------

def test_overlay_without_any_mask_renders_no_objects_panel(tmp_path):
    """All object channels None -> plain channels plus a blank 'no objects'."""
    from spacr.plot import plot_image_mask_overlay

    path, _ = _make_stack(tmp_path, 3, [])

    fig = plot_image_mask_overlay(
        path, [0, 1], cell_channel=None, nucleus_channel=None,
        pathogen_channel=None, figuresize=2, save_pdf=False)

    assert [a.get_title() for a in fig.axes] == [
        "channel 0", "channel 1", "no objects"]
    assert np.all(_panel(fig, -1) == 0)
    # Untouched channels stay grayscale.
    for i in (0, 1):
        p = _panel(fig, i)
        assert np.array_equal(p[..., 0], p[..., 2])


def test_overlay_empty_mask_plane_yields_black_combined_panel(tmp_path):
    """A mask plane with no labels leaves the combined panel entirely black."""
    from spacr.plot import plot_image_mask_overlay

    path, _ = _make_stack(tmp_path, 2, [np.zeros(SHAPE, dtype=np.int32)])

    fig = plot_image_mask_overlay(
        path, [0, 1], cell_channel=0, nucleus_channel=None,
        pathogen_channel=None, figuresize=2, thickness=1, save_pdf=False)

    assert fig.axes[-1].get_title() == "combined objects"
    assert np.all(_panel(fig, -1) == 0)
    # Nothing was drawn on the cell channel either.
    p = _panel(fig, 0)
    assert np.array_equal(p[..., 0], p[..., 2])


def test_overlay_negative_label_plane_hits_empty_colormap_guard(tmp_path):
    """A plane whose max label is negative -> n_labels<=0 black colormap."""
    from spacr.plot import plot_image_mask_overlay

    negative = np.full(SHAPE, -1, dtype=np.float32)
    path, _ = _make_stack(tmp_path, 2, [negative], dtype=np.float32, name="neg.npy")

    fig = plot_image_mask_overlay(
        path, [0, 1], cell_channel=0, nucleus_channel=None,
        pathogen_channel=None, figuresize=2, save_pdf=False, mode="masks")

    # Alpha is 0 everywhere (no pixel is > 0), so nothing is overlaid.
    p = _panel(fig, 0)
    assert np.array_equal(p[..., 0], p[..., 2])
    assert np.all(_panel(fig, -1) == 0)


def test_overlay_float_stack_is_not_rescaled(tmp_path):
    """float32 stacks skip the uint->float cast and render identically."""
    from spacr.plot import plot_image_mask_overlay

    masks = [cell_mask(), nucleus_mask(), pathogen_mask()]
    p_u16, _ = _make_stack(tmp_path, 3, masks, dtype=np.uint16, name="u16.npy")
    p_f32, _ = _make_stack(tmp_path, 3, masks, dtype=np.float32, name="f32.npy")

    fig_u = plot_image_mask_overlay(p_u16, [0, 1, 2], 0, 1, 2, figuresize=2,
                                    thickness=1, save_pdf=False)
    panels_u = [_panel(fig_u, i).copy() for i in range(4)]
    plt.close(fig_u)
    fig_f = plot_image_mask_overlay(p_f32, [0, 1, 2], 0, 1, 2, figuresize=2,
                                    thickness=1, save_pdf=False)
    for i, ref in enumerate(panels_u):
        assert np.allclose(_panel(fig_f, i), ref)


def test_overlay_single_channel_renders(tmp_path):
    """A one-channel overlay should still produce image + combined panels."""
    from spacr.plot import plot_image_mask_overlay

    path, _ = _make_stack(tmp_path, 3, [cell_mask(), nucleus_mask(), pathogen_mask()])

    fig = plot_image_mask_overlay(
        path, [0], cell_channel=0, nucleus_channel=1, pathogen_channel=2,
        figuresize=2, thickness=1, save_pdf=False)

    assert len(fig.axes) == 2
    assert fig.axes[-1].get_title() == "combined objects"


# ---------------------------------------------------------------------------
# plot_image_mask_overlay -- filter_dict and tiff export
# ---------------------------------------------------------------------------

def test_overlay_filter_dict_drops_and_keeps_objects(tmp_path, capsys):
    """Area/intensity filtering removes only the objects outside the window."""
    from spacr.plot import plot_image_mask_overlay

    cell, nuc, pat = cell_mask(), nucleus_mask(), pathogen_mask()
    path, _ = _make_stack(tmp_path, 3, [cell, nuc, pat])

    fig = plot_image_mask_overlay(
        path, [0, 1, 2], 0, 1, 2, figuresize=2, thickness=1, save_pdf=False,
        filter_dict={"cell": [(0, 10), (0, 65000)],          # both cells too big
                     "nucleus": [(0, 10 ** 7), (0, 65000)]})  # everything kept

    out = capsys.readouterr().out
    assert "Before filtering cell: 2 objects" in out
    assert "After filtering cell: 0 objects" in out
    assert "Before filtering nucleus: 2 objects" not in out
    assert "Before filtering nucleus: 1 objects" in out
    assert "After filtering nucleus: 1 objects" in out
    # 'pathogen' was not in filter_dict -> untouched.
    assert "filtering pathogen" not in out

    painted = _panel(fig, -1).max(axis=-1) > 0
    assert np.array_equal(painted, (nuc > 0) | (pat > 0))
    assert not _has_color(_panel(fig, 0), RED)   # no cells left to outline


def test_overlay_filter_dict_intensity_window_selects_one_object(tmp_path, capsys):
    """The intensity half of the filter keeps only the bright object."""
    from spacr.plot import plot_image_mask_overlay

    cell = cell_mask()
    intensity = np.full(SHAPE, 100.0)
    intensity[cell == 2] = 50000.0          # only cell 2 is bright
    stack = np.stack([intensity, intensity, cell.astype(float)], axis=-1)
    stack_dir = tmp_path / "stack"
    stack_dir.mkdir(exist_ok=True)
    path = str(stack_dir / "bright.npy")
    np.save(path, stack.astype(np.float32))

    fig = plot_image_mask_overlay(
        path, [0, 1], cell_channel=0, nucleus_channel=None, pathogen_channel=None,
        figuresize=2, thickness=1, save_pdf=False,
        filter_dict={"cell": [(0, 10 ** 7), (10000, 65000)]})

    out = capsys.readouterr().out
    assert "Before filtering cell: 2 objects" in out
    assert "After filtering cell: 1 objects" in out
    painted = _panel(fig, -1).max(axis=-1) > 0
    assert np.array_equal(painted, cell == 2)


def test_overlay_filter_dict_on_label_free_mask_reports_zero(tmp_path, capsys):
    """Filtering an empty mask takes the num_objects == 0 bookkeeping branch."""
    from spacr.plot import plot_image_mask_overlay

    path, _ = _make_stack(tmp_path, 2, [np.zeros(SHAPE, dtype=np.int32)])

    fig = plot_image_mask_overlay(
        path, [0, 1], cell_channel=0, nucleus_channel=None, pathogen_channel=None,
        figuresize=2, save_pdf=False,
        filter_dict={"cell": [(5, 500), (0, 65000)]})

    out = capsys.readouterr().out
    assert "Before filtering cell: 0 objects" in out
    assert "After filtering cell: 0 objects" in out
    # Both averages fall back to the literal 0, not NaN.
    assert "Average area cell: 0.00 pixels, Average intensity: 0.00" in out
    assert "nan" not in out
    assert np.all(_panel(fig, -1) == 0)


def test_overlay_export_tiffs_writes_one_uint16_file_per_plane(tmp_path):
    """export_tiffs dumps every plane of the stack, mask planes included."""
    from spacr.plot import plot_image_mask_overlay

    path, stack = _make_stack(tmp_path, 3, [cell_mask(), nucleus_mask(), pathogen_mask()])

    plot_image_mask_overlay(
        path, [0, 1, 2], 0, 1, 2, figuresize=2, thickness=1, save_pdf=False,
        export_tiffs=True)

    tiff_dir = tmp_path / "results" / "fov" / "tiff"
    written = sorted(os.listdir(tiff_dir))
    assert written == [f"fov_channel_{i}.tiff" for i in range(stack.shape[-1])]
    for i in range(stack.shape[-1]):
        arr = tiff.imread(str(tiff_dir / f"fov_channel_{i}.tiff"))
        assert arr.dtype == np.uint16
        assert np.array_equal(arr, stack[..., i].astype(np.uint16))


# ---------------------------------------------------------------------------
# plot_image_mask_overlay_magenta_outlines (legacy variant)
# ---------------------------------------------------------------------------

def test_magenta_variant_outlines_every_matched_channel(tmp_path):
    """Each channel with a mask gets its own object outlined in magenta."""
    from spacr.plot import plot_image_mask_overlay_magenta_outlines

    path, _ = _make_stack(tmp_path, 3, [cell_mask(), nucleus_mask(), pathogen_mask()])

    fig = plot_image_mask_overlay_magenta_outlines(
        path, [0, 1, 2], cell_channel=0, nucleus_channel=1, pathogen_channel=2,
        figuresize=2, thickness=1, save_pdf=True, mode="outlines")

    assert [a.get_title() for a in fig.axes] == [
        "Image - Channel 0", "Image - Channel 1", "Image - Channel 2",
        "Combined Objects Image"]
    for i in range(3):
        assert _has_color(_panel(fig, i), MAGENTA)
        assert not _has_color(_panel(fig, i), RED)

    pdf = tmp_path / "results" / "overlay" / "fov.pdf"
    assert pdf.is_file() and pdf.read_bytes()[:4] == b"%PDF"


def test_magenta_variant_reads_masks_from_the_tail_of_the_stack(tmp_path):
    """cell/-3, nucleus/-2, pathogen/-1: the outlines match the right planes."""
    from spacr.plot import plot_image_mask_overlay_magenta_outlines

    cell, nuc, pat = cell_mask(), nucleus_mask(), pathogen_mask()
    path, _ = _make_stack(tmp_path, 3, [cell, nuc, pat])

    fig = plot_image_mask_overlay_magenta_outlines(
        path, [0, 1, 2], 0, 1, 2, figuresize=2, thickness=1, save_pdf=False)

    # Channel 1 is the nucleus channel: its magenta pixels must sit on the
    # nucleus, which lives far from cell 2.
    nuc_panel = _panel(fig, 1)
    magenta_px = np.all(np.isclose(nuc_panel, MAGENTA, atol=1e-6), axis=-1)
    assert magenta_px.any()
    ys, xs = np.nonzero(magenta_px)
    assert ys.max() < 25 and xs.max() < 25
    # The combined panel covers the union of the three masks.
    painted = _panel(fig, -1).max(axis=-1) > 0
    assert np.array_equal(painted, (cell > 0) | (nuc > 0) | (pat > 0))


def test_magenta_variant_all_on_all_outlines_uses_object_colors(tmp_path):
    """all_on_all draws every outline on every channel in its object colour."""
    from spacr.plot import plot_image_mask_overlay_magenta_outlines

    path, _ = _make_stack(tmp_path, 3, [cell_mask(), nucleus_mask(), pathogen_mask()])

    fig = plot_image_mask_overlay_magenta_outlines(
        path, [0, 1, 2], 0, 1, 2, figuresize=2, thickness=1, save_pdf=False,
        mode="outlines", all_on_all=True)

    for i in range(3):
        p = _panel(fig, i)
        assert not _has_color(p, MAGENTA)
        assert _has_color(p, GREEN)   # pathogen
        assert _has_color(p, BLUE)    # nucleus
        assert _has_color(p, RED)     # cell
        # Around the lone cell only the cell colour is used.
        assert _has_color(p, RED, region=CELL_2_BOX)
        assert not _has_color(p, GREEN, region=CELL_2_BOX)


def test_magenta_variant_masks_mode_paints_only_object_pixels(tmp_path):
    """mode='masks' fills each channel's own object and nothing else."""
    from spacr.plot import plot_image_mask_overlay_magenta_outlines

    cell, nuc, pat = cell_mask(), nucleus_mask(), pathogen_mask()
    path, stack = _make_stack(tmp_path, 3, [cell, nuc, pat])

    fig = plot_image_mask_overlay_magenta_outlines(
        path, [0, 1, 2], 0, 1, 2, figuresize=2, save_pdf=False, mode="masks")

    for idx, mask in ((0, cell), (1, nuc), (2, pat)):
        assert np.array_equal(_painted(fig, idx, stack, idx), mask > 0)
        # One flat colour per label.
        p = _panel(fig, idx)
        assert len(np.unique(p[mask > 0].reshape(-1, 3), axis=0)) == len(
            np.unique(mask[mask > 0]))


def test_magenta_variant_masks_mode_all_on_all(tmp_path):
    """all_on_all + mode='masks' confines painting to the mask union."""
    from spacr.plot import plot_image_mask_overlay_magenta_outlines

    cell, nuc, pat = cell_mask(), nucleus_mask(), pathogen_mask()
    path, stack = _make_stack(tmp_path, 3, [cell, nuc, pat])

    fig = plot_image_mask_overlay_magenta_outlines(
        path, [0, 1, 2], 0, 1, 2, figuresize=2, save_pdf=False,
        mode="masks", all_on_all=True)

    union = (cell > 0) | (nuc > 0) | (pat > 0)
    for i in range(3):
        assert np.array_equal(_painted(fig, i, stack, i), union)


def test_magenta_variant_all_outlines_only_touches_object_borders(tmp_path):
    """all_outlines paints a maskless channel strictly on object contours."""
    from spacr.plot import plot_image_mask_overlay_magenta_outlines

    cell, nuc, pat = cell_mask(), nucleus_mask(), pathogen_mask()
    path, stack = _make_stack(tmp_path, 4, [cell, nuc, pat])

    fig = plot_image_mask_overlay_magenta_outlines(
        path, [0, 1, 2, 3], 0, 1, 2, figuresize=2, thickness=1, save_pdf=False,
        mode="outlines", all_outlines=True)

    drawn = _painted(fig, 3, stack, 3)
    assert drawn.any()
    from scipy.ndimage import binary_dilation
    union = (cell > 0) | (nuc > 0) | (pat > 0)
    near_object = binary_dilation(union, iterations=2)
    # Contours hug the objects and never fill them completely.
    assert not drawn[~near_object].any()
    assert drawn.sum() < union.sum()


def test_magenta_variant_all_outlines_masks_mode(tmp_path):
    """all_outlines + mode='masks' fills the maskless channel."""
    from spacr.plot import plot_image_mask_overlay_magenta_outlines

    cell, nuc, pat = cell_mask(), nucleus_mask(), pathogen_mask()
    path, stack = _make_stack(tmp_path, 4, [cell, nuc, pat])

    fig = plot_image_mask_overlay_magenta_outlines(
        path, [0, 1, 2, 3], 0, 1, 2, figuresize=2, save_pdf=False,
        mode="masks", all_outlines=True)

    union = (cell > 0) | (nuc > 0) | (pat > 0)
    assert np.array_equal(_painted(fig, 3, stack, 3), union)


def test_magenta_variant_all_outlines_uses_consistent_object_colors(tmp_path):
    """A maskless channel must colour objects the same way all_on_all does."""
    from spacr.plot import plot_image_mask_overlay_magenta_outlines

    path, _ = _make_stack(tmp_path, 4, [cell_mask(), nucleus_mask(), pathogen_mask()])

    fig = plot_image_mask_overlay_magenta_outlines(
        path, [0, 1, 2, 3], 0, 1, 2, figuresize=2, thickness=1, save_pdf=False,
        mode="outlines", all_outlines=True)

    extra = _panel(fig, 3)
    # Only the cell outline exists around cell 2, and cells are red.
    assert _has_color(extra, RED, region=CELL_2_BOX)
    assert not _has_color(extra, GREEN, region=CELL_2_BOX)


@pytest.mark.parametrize(
    "cell_ch, nuc_ch, pat_ch, masks_key",
    [
        (0, None, None, ("cell",)),
        (0, 1, None, ("cell", "nucleus")),
        (0, None, 2, ("cell", "pathogen")),
        (None, 1, 2, ("nucleus", "pathogen")),
        (None, None, 2, ("pathogen",)),
    ],
)
def test_magenta_variant_mask_dim_arithmetic(tmp_path, cell_ch, nuc_ch, pat_ch,
                                             masks_key):
    """Every cell/nucleus/pathogen combination reads the correct tail planes."""
    from spacr.plot import plot_image_mask_overlay_magenta_outlines

    available = {"cell": cell_mask(), "nucleus": nucleus_mask(),
                 "pathogen": pathogen_mask()}
    # Stack tail order is always cell, nucleus, pathogen -- only the present
    # ones are written, which is exactly what the negative indices assume.
    order = [k for k in ("cell", "nucleus", "pathogen") if k in masks_key]
    masks = [available[k] for k in order]
    path, _ = _make_stack(tmp_path, 3, masks, name=f"{'_'.join(order)}.npy")

    fig = plot_image_mask_overlay_magenta_outlines(
        path, [0, 1, 2], cell_channel=cell_ch, nucleus_channel=nuc_ch,
        pathogen_channel=pat_ch, figuresize=2, thickness=1, save_pdf=False)

    union = np.zeros(SHAPE, dtype=bool)
    for k in order:
        union |= available[k] > 0
    painted = _panel(fig, -1).max(axis=-1) > 0
    assert np.array_equal(painted, union)


def test_magenta_variant_filter_dict_filters_every_object_type(tmp_path, capsys):
    """filter_dict is applied to cell, nucleus and pathogen alike."""
    from spacr.plot import plot_image_mask_overlay_magenta_outlines

    cell, nuc, pat = cell_mask(), nucleus_mask(), pathogen_mask()
    path, _ = _make_stack(tmp_path, 3, [cell, nuc, pat])

    keep_all = [(0, 10 ** 7), (0, 65000)]
    fig = plot_image_mask_overlay_magenta_outlines(
        path, [0, 1, 2], 0, 1, 2, figuresize=2, thickness=1, save_pdf=False,
        filter_dict={"cell": [(0, 10), (0, 65000)],   # drops both cells
                     "nucleus": keep_all,
                     "pathogen": keep_all})

    out = capsys.readouterr().out
    for kind, before, after in (("pathogen", 1, 1), ("nucleus", 1, 1),
                                ("cell", 2, 0)):
        assert f"Before filtering {kind}: {before} objects" in out
        assert f"After filtering {kind}: {after} objects" in out

    painted = _panel(fig, -1).max(axis=-1) > 0
    assert np.array_equal(painted, (nuc > 0) | (pat > 0))
    assert not _has_color(_panel(fig, 0), MAGENTA)


def test_magenta_variant_filter_dict_on_empty_mask(tmp_path, capsys):
    """Filtering a label-free plane reports zeros instead of NaN averages."""
    from spacr.plot import plot_image_mask_overlay_magenta_outlines

    path, _ = _make_stack(tmp_path, 3, [np.zeros(SHAPE, dtype=np.int32)],
                          name="empty.npy")

    fig = plot_image_mask_overlay_magenta_outlines(
        path, [0, 1, 2], cell_channel=None, nucleus_channel=None,
        pathogen_channel=2, figuresize=2, save_pdf=False,
        filter_dict={"pathogen": [(5, 500), (0, 65000)]})

    out = capsys.readouterr().out
    assert "Before filtering pathogen: 0 objects" in out
    assert "After filtering pathogen: 0 objects" in out
    assert "nan" not in out
    assert np.all(_panel(fig, -1) == 0)


def test_magenta_variant_export_tiffs(tmp_path):
    """The legacy variant exports the same per-plane uint16 TIFFs."""
    from spacr.plot import plot_image_mask_overlay_magenta_outlines

    path, stack = _make_stack(tmp_path, 3, [cell_mask(), nucleus_mask(),
                                            pathogen_mask()], name="mag.npy")

    plot_image_mask_overlay_magenta_outlines(
        path, [0, 1, 2], 0, 1, 2, figuresize=2, thickness=1, save_pdf=False,
        export_tiffs=True)

    tiff_dir = tmp_path / "results" / "mag" / "tiff"
    assert sorted(os.listdir(tiff_dir)) == [
        f"mag_channel_{i}.tiff" for i in range(stack.shape[-1])]
    arr = tiff.imread(str(tiff_dir / "mag_channel_5.tiff"))
    assert np.array_equal(arr, stack[..., 5].astype(np.uint16))


def test_magenta_variant_single_channel_still_renders(tmp_path):
    """One channel: the legacy variant has no ndarray-wrapping guard, so it works."""
    from spacr.plot import plot_image_mask_overlay_magenta_outlines

    cell = cell_mask()
    path, _ = _make_stack(tmp_path, 3, [cell, nucleus_mask(), pathogen_mask()],
                          name="one.npy")

    fig = plot_image_mask_overlay_magenta_outlines(
        path, [0], cell_channel=0, nucleus_channel=1, pathogen_channel=2,
        figuresize=2, thickness=1, save_pdf=False)

    assert len(fig.axes) == 2
    assert fig.axes[0].get_title() == "Image - Channel 0"
    assert _has_color(_panel(fig, 0), MAGENTA)
