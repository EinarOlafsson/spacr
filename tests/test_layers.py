"""The layer model, tested without a display.

Every assertion here is about a rule a scientist depends on and cannot see
being broken: which layer wins, what opacity does to a pixel, and — the one
that matters most — whether a point and a mask at the same world coordinate
land on the same pixel when the voxels are not cubes.

There is deliberately no ``QApplication`` anywhere in this file. That is the
structural claim of :mod:`spacr.layers` and it is asserted outright in
:func:`test_the_model_does_not_import_qt`: five later features are written
against this model, and if the model needed a widget none of them could be
tested either.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from spacr.layers import (Blending, Canvas, Colormap, FieldKey, ImageLayer,
                          LabelsLayer, Layer, LayerError, LayerEvent,
                          LayerStack, PointsLayer, Shape, ShapesLayer, Spacing,
                          colormap, label_color, label_colors, to_rgba)
from spacr.selection import OBJECT_KEY_COLUMNS


# ---------------------------------------------------------------------------
# The structural claim
# ---------------------------------------------------------------------------

def test_the_model_does_not_import_qt():
    """`spacr.layers` must be importable with no GUI stack behind it.

    Checked in a fresh interpreter rather than against ``sys.modules`` here,
    because a Qt test earlier in the same session would already have imported
    PySide6 and made the assertion pass for the wrong reason.
    """
    code = ("import spacr.layers, sys; "
            "bad = [m for m in sys.modules if m.split('.')[0] in "
            "('PySide6', 'shiboken6', 'qtpy')]; "
            "print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", (
        f"importing spacr.layers dragged in {out.stdout.strip()}; the model "
        f"has to stay testable without a display")


# ---------------------------------------------------------------------------
# Spacing — the part that is a science error when it is wrong
# ---------------------------------------------------------------------------

def test_spacing_maps_indices_to_world_and_back():
    sp = Spacing.from_map({"z": 2.0, "y": 0.65, "x": 0.65}, units="um")
    assert sp.axes == ("z", "y", "x")
    assert sp.ndim == 3
    assert sp.to_world((3, 10, 20)) == pytest.approx((6.0, 6.5, 13.0))
    assert sp.to_data((6.0, 6.5, 13.0)) == pytest.approx((3.0, 10.0, 20.0))


def test_spacing_carries_an_origin_so_a_crop_keeps_its_place():
    sp = Spacing.from_map({"y": 1.0, "x": 1.0}, origin={"y": 100.0, "x": 50.0})
    assert sp.to_world((0, 0)) == pytest.approx((100.0, 50.0))
    assert sp.to_data((105.0, 55.0)) == pytest.approx((5.0, 5.0))


@pytest.mark.parametrize("bad", [0.0, float("nan"), float("inf")])
def test_a_zero_or_infinite_voxel_size_is_refused(bad):
    """The silent one. A collapsed axis draws a plausible wrong slice."""
    with pytest.raises(LayerError, match="collapses the axis|scale"):
        Spacing(scale=(1.0, bad))


def test_spacing_extent_reaches_the_outer_edge_of_the_end_voxels():
    """Half a voxel each side — at 2 um z-steps that is a visible slab."""
    sp = Spacing.from_map({"z": 2.0, "y": 1.0, "x": 1.0})
    extent = sp.extent((4, 10, 10))
    assert extent["z"] == pytest.approx((-1.0, 7.0))
    assert extent["y"] == pytest.approx((-0.5, 9.5))


def test_spacing_rejects_mismatched_axes_and_duplicate_names():
    with pytest.raises(LayerError, match="does not match"):
        Spacing(scale=(1.0, 1.0), axes=("y",))
    with pytest.raises(LayerError, match="unique"):
        Spacing(scale=(1.0, 1.0), axes=("y", "y"))
    with pytest.raises(LayerError, match="no axis"):
        Spacing.isotropic(2).axis_index("z")


def test_spacing_describes_itself_for_a_status_bar():
    sp = Spacing.from_map({"z": 2.0, "y": 0.65, "x": 0.65}, units="um")
    assert sp.describe() == "z 2, y 0.65, x 0.65 um"
    assert sp.rescaled(z=1.5).scale == (1.5, 0.65, 0.65)
    assert sp.translated(x=10.0).translate == (0.0, 0.0, 10.0)


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

def test_canvas_round_trips_a_world_point_to_the_pixel_it_lands_on():
    canvas = Canvas(origin=(0.0, 0.0), step=(2.0, 0.5), shape=(10, 10))
    world = canvas.world_at(3, 4)
    assert world == {"y": 6.0, "x": 2.0}
    assert canvas.pixel_at(world) == pytest.approx((3.0, 4.0))


def test_a_canvas_fitted_to_a_layer_keeps_the_world_aspect_ratio():
    """An anisotropic stack seen from the side must not be squashed."""
    volume = np.zeros((5, 20, 20), dtype=np.uint8)
    layer = ImageLayer(volume, spacing=Spacing.from_map(
        {"z": 4.0, "y": 1.0, "x": 1.0}, units="um"))
    side = Canvas.covering(layer, width=40, axes=("z", "x"))
    # 5 slices x 4 um = 20 um of z against 20 um of x: a square field of view.
    assert side.height == 40
    assert side.step[0] * side.height == pytest.approx(
        side.step[1] * side.width)


def test_canvas_for_grid_is_the_identity_sampling_of_a_layer():
    sp = Spacing.from_map({"y": 2.0, "x": 0.5}, origin={"y": 3.0, "x": -1.0})
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    layer = ImageLayer(data, spacing=sp, contrast_limits=(0.0, 11.0),
                       colormaps="gray")
    canvas = Canvas.for_grid(sp, layer.shape)
    assert canvas.shape == (3, 4)
    rgb, coverage = layer.render(canvas)
    assert coverage.min() == 1.0
    # Every element comes back once, in order.
    assert rgb[..., 0] * 11.0 == pytest.approx(data, abs=1e-4)


def test_canvas_zoom_holds_the_world_point_under_the_cursor():
    canvas = Canvas(origin=(0.0, 0.0), step=(1.0, 1.0), shape=(20, 20))
    before = canvas.world_at(5, 7)
    zoomed = canvas.zoomed(2.0, centre=(5, 7))
    assert zoomed.world_at(5, 7) == pytest.approx(before)
    assert zoomed.step == pytest.approx((0.5, 0.5))


def test_canvas_resize_holds_the_field_of_view_not_the_zoom():
    canvas = Canvas(origin=(0.0, 0.0), step=(1.0, 1.0), shape=(10, 10))
    bigger = canvas.resized(20, 20)
    assert bigger.step == pytest.approx((0.5, 0.5))
    assert bigger.step[0] * bigger.height == pytest.approx(10.0)


def test_canvas_pan_moves_by_whole_canvas_pixels():
    canvas = Canvas(origin=(0.0, 0.0), step=(2.0, 3.0), shape=(4, 4))
    moved = canvas.panned(1, -1)
    assert moved.origin == pytest.approx((2.0, -3.0))


def test_a_degenerate_canvas_is_refused():
    with pytest.raises(LayerError, match="two entries"):
        Canvas(origin=(0.0,), step=(1.0, 1.0), shape=(4, 4))
    with pytest.raises(LayerError, match="share axis"):
        Canvas(origin=(0.0, 0.0), step=(1.0, 1.0), shape=(4, 4),
               axes=("y", "y"))
    with pytest.raises(LayerError, match="positive"):
        Canvas(origin=(0.0, 0.0), step=(1.0, 1.0), shape=(0, 4))
    with pytest.raises(LayerError, match="non-zero"):
        Canvas(origin=(0.0, 0.0), step=(0.0, 1.0), shape=(4, 4))
    with pytest.raises(LayerError, match="positive"):
        Canvas(origin=(0.0, 0.0), step=(1.0, 1.0), shape=(4, 4)).zoomed(0.0)


def test_fitting_a_canvas_to_an_axis_nothing_spans_says_so():
    stack = LayerStack()
    stack.add_image(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(LayerError, match="spans"):
        stack.canvas(axes=("z", "x"))


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec, expected", [
    ("red", (1.0, 0.0, 0.0, 1.0)),
    ("#00ff00", (0.0, 1.0, 0.0, 1.0)),
    ("#0f0", (0.0, 1.0, 0.0, 1.0)),
    ("#0000ff80", (0.0, 0.0, 1.0, 128 / 255)),
    ((0.0, 0.5, 1.0), (0.0, 0.5, 1.0, 1.0)),
    ((0, 128, 255), (0.0, 128 / 255, 1.0, 1.0)),
])
def test_colours_parse_from_every_form_a_caller_has(spec, expected):
    assert to_rgba(spec) == pytest.approx(expected, abs=1e-6)


def test_a_nonsense_colour_is_refused_rather_than_defaulted():
    with pytest.raises(LayerError, match="unknown colour"):
        to_rgba("chartreuse-ish")
    with pytest.raises(LayerError, match="3 or 4"):
        to_rgba((1.0, 0.0))
    with pytest.raises(LayerError, match="cannot read a colour"):
        to_rgba(object())
    with pytest.raises(LayerError, match="hex"):
        to_rgba("#12345")


def test_a_colormap_is_a_ramp_and_a_colour_name_makes_one():
    cm = colormap("magenta")
    assert cm.map(0.0) == pytest.approx([0.0, 0.0, 0.0])
    assert cm.map(1.0) == pytest.approx([1.0, 0.0, 1.0])
    assert cm.map(0.5) == pytest.approx([0.5, 0.0, 0.5])
    assert colormap(cm) is cm
    assert colormap("gray").map([0.0, 1.0]).shape == (2, 3)


def test_an_unknown_colormap_name_is_refused():
    with pytest.raises(LayerError, match="unknown colormap"):
        colormap("definitely-not-a-colormap")
    with pytest.raises(LayerError, match="at least two"):
        Colormap("bad", ["red"])
    with pytest.raises(LayerError, match="stops"):
        Colormap("bad", ["red", "blue"], stops=[0.0])
    with pytest.raises(LayerError, match="increase"):
        Colormap("bad", ["red", "blue"], stops=[1.0, 0.0])


def test_label_colours_are_stable_vivid_and_zero_is_background():
    assert label_color(0) == (0.0, 0.0, 0.0)
    assert label_color(17) == label_color(17)
    assert label_color(17) != label_color(18)
    assert max(label_color(17)) > 0.5
    grid = label_colors(np.array([[0, 1], [2, 1]]))
    assert grid.shape == (2, 2, 3)
    assert tuple(grid[0, 0]) == (0.0, 0.0, 0.0)
    assert tuple(grid[0, 1]) == tuple(grid[1, 1])
    assert label_colors(np.array([], dtype=int)).shape == (0, 3)


# ---------------------------------------------------------------------------
# Compositing: order, opacity, visibility
# ---------------------------------------------------------------------------

def _flat_stack():
    """Two full-frame single-colour image layers, red under green."""
    ones = np.ones((4, 4), dtype=np.float32)
    stack = LayerStack()
    stack.add_image(ones, name="red", colormaps="red",
                    contrast_limits=(0.0, 1.0))
    stack.add_image(ones, name="green", colormaps="green",
                    contrast_limits=(0.0, 1.0))
    return stack, Canvas.for_grid(stack[0].spacing, stack[0].shape)


def test_layer_order_decides_which_colour_survives():
    """The whole point of a stack: index 0 is the bottom."""
    stack, canvas = _flat_stack()
    assert stack.names == ("red", "green")
    top_green = stack.render(canvas)
    assert top_green[0, 0] == pytest.approx([0.0, 1.0, 0.0])

    stack.move("green", 0)
    assert stack.names == ("green", "red")
    top_red = stack.render(canvas)
    assert top_red[0, 0] == pytest.approx([1.0, 0.0, 0.0])
    assert not np.array_equal(top_green, top_red)


def test_raise_and_lower_walk_a_layer_through_the_order():
    stack = LayerStack()
    for name in ("a", "b", "c"):
        stack.add_image(np.zeros((2, 2), dtype=np.uint8), name=name)
    assert stack.names == ("a", "b", "c")
    stack.raise_layer("a")
    assert stack.names == ("b", "a", "c")
    stack.to_top("b")
    assert stack.names == ("a", "c", "b")
    stack.lower_layer("b")
    assert stack.names == ("a", "b", "c")
    stack.to_bottom("c")
    assert stack.names == ("c", "a", "b")
    # Already at the end: a no-op, not an error and not a wrap-around.
    stack.raise_layer("b")
    assert stack.names == ("c", "a", "b")
    stack.lower_layer("c")
    assert stack.names == ("c", "a", "b")


@pytest.mark.parametrize("opacity", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_opacity_is_exactly_the_fraction_it_says(opacity):
    """`out = under*(1-a) + over*a`, with no gamma and no rounding fudge."""
    stack, canvas = _flat_stack()
    stack["green"].opacity = opacity
    out = stack.render(canvas)
    assert out[0, 0] == pytest.approx([1.0 - opacity, opacity, 0.0], abs=1e-6)


def test_opacity_is_clamped_not_refused_and_zero_skips_the_layer():
    stack, canvas = _flat_stack()
    stack["green"].opacity = 4.2
    assert stack["green"].opacity == 1.0
    stack["green"].opacity = -1.0
    assert stack["green"].opacity == 0.0
    assert stack.render(canvas)[0, 0] == pytest.approx([1.0, 0.0, 0.0])
    with pytest.raises(LayerError, match="must be a number"):
        stack["green"].opacity = "half"
    with pytest.raises(LayerError, match="finite"):
        stack["green"].opacity = float("nan")


def test_hiding_a_layer_changes_the_pixels_and_showing_it_puts_them_back():
    stack, canvas = _flat_stack()
    both = stack.render(canvas)
    stack["green"].visible = False
    hidden = stack.render(canvas)
    assert not np.array_equal(both, hidden)
    assert hidden[0, 0] == pytest.approx([1.0, 0.0, 0.0])
    stack["green"].visible = True
    assert np.array_equal(stack.render(canvas), both)


@pytest.mark.parametrize("mode, expected", [
    (Blending.TRANSLUCENT, [0.5, 0.5, 0.0]),
    (Blending.ADDITIVE, [1.0, 0.5, 0.0]),
    (Blending.OPAQUE, [0.5, 0.5, 0.0]),
    (Blending.MULTIPLY, [0.5, 0.0, 0.0]),
    (Blending.MINIMUM, [0.5, 0.0, 0.0]),
])
def test_every_blending_mode_composites_the_way_it_is_documented(mode, expected):
    stack, canvas = _flat_stack()
    stack["green"].blending = mode
    stack["green"].opacity = 0.5
    assert stack.render(canvas)[0, 0] == pytest.approx(expected, abs=1e-6)


def test_opaque_hardens_partial_coverage_into_a_curtain():
    """The one mode where coverage and opacity are treated differently."""
    dst = np.zeros((1, 1, 3), dtype=np.float32)
    src = np.ones((1, 1, 3), dtype=np.float32)
    coverage = np.full((1, 1), 0.25, dtype=np.float32)
    translucent, _ = Blending.apply(dst, src, coverage, 1.0,
                                    Blending.TRANSLUCENT)
    opaque, _ = Blending.apply(dst, src, coverage, 1.0, Blending.OPAQUE)
    assert translucent[0, 0, 0] == pytest.approx(0.25)
    assert opaque[0, 0, 0] == pytest.approx(1.0)
    # ...and it is still fadeable.
    half, _ = Blending.apply(dst, src, coverage, 0.5, Blending.OPAQUE)
    assert half[0, 0, 0] == pytest.approx(0.5)


def test_an_unknown_blending_mode_is_refused():
    with pytest.raises(LayerError, match="unknown blending"):
        Blending.check("screen")
    with pytest.raises(LayerError, match="unknown blending"):
        ImageLayer(np.zeros((2, 2)), blending="screen")


def test_render_uint8_and_rgba_agree_with_render():
    stack, canvas = _flat_stack()
    rgb = stack.render(canvas)
    assert np.array_equal(stack.render_uint8(canvas),
                          np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8))
    rgba = stack.render_rgba(canvas)
    assert np.array_equal(rgba[..., :3], rgb)
    assert rgba[..., 3].min() == pytest.approx(1.0)


def test_an_empty_stack_renders_black_rather_than_raising():
    stack = LayerStack()
    canvas = Canvas(origin=(0.0, 0.0), step=(1.0, 1.0), shape=(3, 3))
    assert stack.render(canvas).shape == (3, 3, 3)
    assert stack.render(canvas).max() == 0.0
    assert stack.describe() == "no layers"
    assert stack.world_extent() == {}


# ---------------------------------------------------------------------------
# The alignment claim
# ---------------------------------------------------------------------------

def test_a_point_and_a_label_at_one_world_coordinate_land_on_one_pixel():
    """The reason there is a world coordinate system at all.

    The mask is on a 0.65 um grid and the points layer is on a 1.3 um grid —
    a downsampled centroid table, which is exactly what a preview produces.
    They are the same object, so they must draw on the same canvas pixel; if
    the model reasoned in array indices they would be a factor of two apart.
    """
    fine = Spacing.from_map({"y": 0.65, "x": 0.65}, units="um")
    coarse = Spacing.from_map({"y": 1.3, "x": 1.3}, units="um")

    mask = np.zeros((40, 40), dtype=np.int32)
    mask[20:24, 30:34] = 7
    labels = LabelsLayer(mask, name="cells", spacing=fine)

    # The centroid of object 7 in world units...
    centre = {"y": fine.to_world((21.5, 31.5))[0],
              "x": fine.to_world((21.5, 31.5))[1]}
    # ...expressed on the coarse grid the points layer happens to use.
    points = PointsLayer(np.array([coarse.to_data((centre["y"], centre["x"]))]),
                         name="centroids", spacing=coarse, size=1.0,
                         face_color="white")

    stack = LayerStack([labels, points])
    canvas = Canvas.covering(stack, height=80)

    row, col = canvas.pixel_at(centre)
    assert labels.label_at_world(canvas.world_at(row, col)) == 7
    assert points.nearest(canvas.world_at(row, col)) == 0

    # And in the rendered picture: the point's pixels are a subset of the
    # label's pixels, which is only true if both were put through the world.
    _, label_coverage = labels.render(canvas)
    _, point_coverage = points.render(canvas)
    assert point_coverage.max() > 0
    assert np.all(label_coverage[point_coverage > 0] > 0)


def test_anisotropy_is_honoured_in_the_orthogonal_plane():
    """A 4 um z-step is four times a 1 um y-step, and the side view says so."""
    volume = np.zeros((6, 10, 10), dtype=np.uint8)
    volume[3, 5, 5] = 255
    sp = Spacing.from_map({"z": 4.0, "y": 1.0, "x": 1.0}, units="um")
    layer = ImageLayer(volume, spacing=sp, contrast_limits=(0.0, 255.0),
                       colormaps="gray")
    # Slice z=12 um is index 3; y=5 um is index 5.
    top = Canvas(origin=(0.0, 0.0), step=(1.0, 1.0), shape=(10, 10),
                 axes=("y", "x"), depth={"z": 12.0})
    side = Canvas(origin=(0.0, 0.0), step=(4.0, 1.0), shape=(6, 10),
                  axes=("z", "x"), depth={"y": 5.0})
    assert layer.render(top)[0][5, 5, 0] == pytest.approx(1.0)
    assert layer.render(side)[0][3, 5, 0] == pytest.approx(1.0)
    # One slice either side is empty, i.e. z was not silently read as 1 um.
    assert layer.render(top.at_depth(z=8.0))[0].max() == 0.0


def test_a_two_dimensional_layer_shows_on_every_slice_of_a_three_d_canvas():
    """A 2-D overlay occupies all of z; a 3-D one does not."""
    flat = LabelsLayer(np.ones((4, 4), dtype=np.int32), name="flat",
                       spacing=Spacing.from_map({"y": 1.0, "x": 1.0}))
    canvas = Canvas(origin=(0.0, 0.0), step=(1.0, 1.0), shape=(4, 4),
                    depth={"z": 99.0})
    assert flat.render(canvas)[1].min() == 1.0


def test_a_stack_refuses_to_mix_micrometres_with_pixels():
    stack = LayerStack()
    stack.add_image(np.zeros((4, 4)), name="um",
                    spacing=Spacing.isotropic(2, 0.65, units="um"))
    assert stack.units == "um"
    with pytest.raises(LayerError, match="measured in"):
        stack.add_image(np.zeros((4, 4)), name="px",
                        spacing=Spacing.isotropic(2, 1.0, units="px"))
    with pytest.raises(LayerError, match="cannot become"):
        stack["um"].spacing = Spacing.isotropic(2, 1.0, units="px")
    assert len(stack) == 1


# ---------------------------------------------------------------------------
# Image layers
# ---------------------------------------------------------------------------

def test_channels_composite_additively_inside_one_image_layer():
    """Green plus magenta overlapping reads white, not "whichever is second"."""
    data = np.zeros((2, 4, 4), dtype=np.float32)
    data[0, :, :2] = 1.0          # green channel, left half
    data[1, :, 1:3] = 1.0         # magenta channel, middle
    layer = ImageLayer(data, channel_axis=0, colormaps=["green", "magenta"],
                       contrast_limits=(0.0, 1.0))
    canvas = Canvas.for_grid(layer.spacing, layer.shape)
    rgb, _ = layer.render(canvas)
    assert rgb[0, 0] == pytest.approx([0.0, 1.0, 0.0])   # green only
    assert rgb[0, 1] == pytest.approx([1.0, 1.0, 1.0])   # overlap
    assert rgb[0, 2] == pytest.approx([1.0, 0.0, 1.0])   # magenta only
    assert rgb[0, 3] == pytest.approx([0.0, 0.0, 0.0])   # neither


def test_a_channel_can_be_turned_off_without_splitting_the_layer():
    data = np.ones((2, 2, 2), dtype=np.float32)
    layer = ImageLayer(data, channel_axis=0, colormaps=["green", "magenta"],
                       contrast_limits=(0.0, 1.0))
    canvas = Canvas.for_grid(layer.spacing, layer.shape)
    assert layer.render(canvas)[0][0, 0] == pytest.approx([1.0, 1.0, 1.0])
    layer.set_channel_visible(1, False)
    assert layer.channel_is_visible(1) is False
    assert layer.render(canvas)[0][0, 0] == pytest.approx([0.0, 1.0, 0.0])


def test_contrast_limits_default_to_a_percentile_stretch_and_stick():
    """Computed once. A per-view stretch makes two crops look like two
    exposures of the same field."""
    rng = np.random.default_rng(0)
    data = rng.integers(100, 4000, size=(64, 64)).astype(np.uint16)
    layer = ImageLayer(data)
    lo, hi = layer.contrast_limits()
    assert lo == pytest.approx(np.percentile(data, 2.0))
    assert hi == pytest.approx(np.percentile(data, 98.0))
    assert layer.contrast_limits() == (lo, hi)
    layer.set_contrast_limits(0.0, 65535.0)
    assert layer.contrast_limits() == (0.0, 65535.0)
    layer.auto_contrast()
    assert layer.contrast_limits() == pytest.approx((lo, hi))


def test_a_flat_image_still_gets_usable_limits():
    """A blank field must not produce lo == hi and a divide by zero."""
    layer = ImageLayer(np.full((4, 4), 7.0))
    lo, hi = layer.contrast_limits()
    assert hi > lo
    canvas = Canvas.for_grid(layer.spacing, layer.shape)
    assert np.isfinite(layer.render(canvas)[0]).all()


def test_inverted_contrast_limits_are_refused():
    layer = ImageLayer(np.zeros((2, 2)))
    with pytest.raises(LayerError, match="must increase"):
        layer.set_contrast_limits(10.0, 1.0)
    with pytest.raises(LayerError, match="must increase"):
        ImageLayer(np.zeros((2, 2)), contrast_limits=(10.0, 1.0))


def test_an_image_needs_two_or_three_spatial_axes():
    with pytest.raises(LayerError, match="2 or 3 spatial axes"):
        ImageLayer(np.zeros((4,)))
    with pytest.raises(LayerError, match="2 or 3 spatial axes"):
        ImageLayer(np.zeros((2, 3, 4, 5)))
    with pytest.raises(LayerError, match="at least a 1-D array"):
        ImageLayer(np.float32(1.0))
    # ...unless one of them is channels.
    assert ImageLayer(np.zeros((2, 3, 4, 5)), channel_axis=0).shape == (3, 4, 5)


def test_channel_metadata_lengths_are_checked():
    data = np.zeros((2, 4, 4))
    with pytest.raises(LayerError, match="colormaps"):
        ImageLayer(data, channel_axis=0, colormaps=["red"])
    with pytest.raises(LayerError, match="channel names"):
        ImageLayer(data, channel_axis=0, channel_names=["a"])
    with pytest.raises(LayerError, match="visibilities"):
        ImageLayer(data, channel_axis=0, channel_visible=[True])
    with pytest.raises(LayerError, match="contrast limits"):
        ImageLayer(data, channel_axis=0,
                   contrast_limits=[(0, 1), (0, 1), (0, 1)])
    layer = ImageLayer(data, channel_axis=0)
    assert layer.n_channels == 2
    assert layer.channel_names == ("ch0", "ch1")
    assert layer.channel_data(0).shape == (4, 4)


def test_a_single_channel_image_is_grey_and_a_multichannel_one_is_not():
    assert ImageLayer(np.zeros((4, 4))).colormap.name == "gray"
    two = ImageLayer(np.zeros((2, 4, 4)), channel_axis=0)
    assert [c.name for c in two.colormaps] == ["green", "magenta"]
    two.set_colormap("red", 1)
    assert two.colormaps[1].name == "red"
    two.colormap = "blue"
    assert two.colormap.name == "blue"


def test_pixels_outside_the_data_are_transparent_not_black():
    """A layer smaller than the canvas must not paint the rest of it."""
    layer = ImageLayer(np.ones((2, 2)), contrast_limits=(0.0, 1.0))
    canvas = Canvas(origin=(0.0, 0.0), step=(1.0, 1.0), shape=(4, 4))
    rgb, coverage = layer.render(canvas)
    assert coverage[0, 0] == 1.0
    assert coverage[3, 3] == 0.0
    assert rgb[3, 3] == pytest.approx([0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Labels and object identity
# ---------------------------------------------------------------------------

def _field_key(**overrides):
    values = {c: v for c, v in zip(FieldKey.columns(),
                                   ("plate1", "A", "1", "1"))}
    values.update(overrides)
    return FieldKey(values=values)


def test_a_clicked_label_names_the_object_the_measurement_table_knows():
    """The whole reason a labels layer carries a field key."""
    mask = np.zeros((10, 10), dtype=np.int32)
    mask[2:5, 2:5] = 17
    layer = LabelsLayer(mask, field=_field_key(),
                        spacing=Spacing.from_map({"y": 2.0, "x": 2.0},
                                                 units="um"))
    world = {"y": 6.0, "x": 6.0}          # element (3, 3) -> label 17
    assert layer.label_at_world(world) == 17
    key = layer.object_key_at_world(world)
    assert key == "plate1_A_1_1_17"

    # ...and it is the key `spacr.selection` builds from a measurement row,
    # not a parallel scheme that happens to look similar.
    import pandas as pd
    from spacr.selection import object_keys as selection_keys
    row = pd.DataFrame([{c: v for c, v in zip(
        OBJECT_KEY_COLUMNS, ("plate1", "A", "1", "1", 17))}])
    assert key == selection_keys(row)[0]


def test_background_and_outside_are_not_objects():
    layer = LabelsLayer(np.zeros((4, 4), dtype=np.int32), field=_field_key())
    assert layer.label_at_world({"y": 1.0, "x": 1.0}) == 0
    assert layer.object_key_at_world({"y": 1.0, "x": 1.0}) is None
    assert layer.label_at_world({"y": -99.0, "x": 0.0}) == 0
    assert layer.label_at_world({"y": 99.0, "x": 0.0}) == 0


def test_a_labels_layer_without_a_field_says_so_rather_than_guessing():
    layer = LabelsLayer(np.ones((4, 4), dtype=np.int32))
    assert layer.object_key_at_world({"y": 1.0, "x": 1.0}) is None
    with pytest.raises(LayerError, match="no field key"):
        layer.object_keys()


def test_object_keys_come_back_in_the_order_they_were_asked_for():
    mask = np.zeros((4, 4), dtype=np.int32)
    mask[0, 0] = 3
    mask[1, 1] = 1
    layer = LabelsLayer(mask, field=_field_key())
    assert list(layer.labels()) == [1, 3]
    assert list(layer.object_keys([3, 1])) == ["plate1_A_1_1_3",
                                              "plate1_A_1_1_1"]


def test_a_timelapse_field_key_keeps_the_frames_apart():
    key = FieldKey(values={c: v for c, v in zip(
        FieldKey.columns(timelapse=True), ("plate1", "A", "1", "1", "5"))},
        timelapse=True)
    assert key.object_key(9) == "plate1_A_1_1_5_9"


def test_an_incomplete_field_key_is_refused_at_construction():
    with pytest.raises(LayerError, match="missing"):
        FieldKey(values={"plateID": "plate1"})


def test_a_field_key_takes_only_the_key_columns_off_a_row():
    key = FieldKey.from_row({c: v for c, v in zip(
        OBJECT_KEY_COLUMNS, ("p", "A", "1", "1", 4))} | {"area": 99.0})
    assert set(key.values) == set(FieldKey.columns())
    assert key.object_key(4) == "p_A_1_1_4"


def test_labels_draw_object_colours_and_leave_the_background_alone():
    mask = np.zeros((4, 4), dtype=np.int32)
    mask[0, 0] = 5
    layer = LabelsLayer(mask)
    canvas = Canvas.for_grid(layer.spacing, layer.shape)
    rgb, coverage = layer.render(canvas)
    assert coverage[0, 0] == 1.0
    assert coverage[3, 3] == 0.0
    assert rgb[0, 0] == pytest.approx(label_color(5), abs=1e-6)


def test_a_labels_layer_needs_whole_numbers():
    with pytest.raises(LayerError, match="integer labels"):
        LabelsLayer(np.array([[0.5, 1.5], [2.5, 3.5]]))
    with pytest.raises(LayerError, match="2-D or 3-D"):
        LabelsLayer(np.zeros((2, 2, 2, 2), dtype=int))
    # Whole-valued floats are accepted and converted, because that is what
    # comes back out of a resize.
    assert LabelsLayer(np.array([[0.0, 1.0]])).data.dtype.kind == "i"


def test_replacing_label_data_on_a_different_grid_is_refused():
    layer = LabelsLayer(np.zeros((4, 4), dtype=np.int32))
    with pytest.raises(LayerError, match="Replace the layer"):
        layer.data = np.zeros((8, 8), dtype=np.int32)
    layer.data = np.ones((4, 4), dtype=np.int32)
    assert layer.labels().tolist() == [1]


def test_the_brush_is_a_ball_in_world_space_not_in_array_elements():
    """A 5 um brush on a 2 um z-step covers fewer slices than rows."""
    volume = np.zeros((9, 9, 9), dtype=np.int32)
    layer = LabelsLayer(volume, spacing=Spacing.from_map(
        {"z": 2.0, "y": 0.5, "x": 0.5}, units="um"))
    centre = layer.to_world((4, 4, 4))
    painted = layer.paint(centre, 3, radius=2.0)
    assert painted > 0
    touched = np.argwhere(volume == 3)
    z_span = touched[:, 0].max() - touched[:, 0].min() + 1
    y_span = touched[:, 1].max() - touched[:, 1].min() + 1
    assert z_span == 3          # +-2 um at 2 um steps
    assert y_span == 9          # +-2 um at 0.5 um steps
    # Painting the same value again changes nothing.
    assert layer.paint(centre, 3, radius=2.0) == 0
    # ...and a brush entirely off the array is a no-op, not an IndexError.
    assert layer.paint({"z": 999.0, "y": 999.0, "x": 999.0}, 3,
                       radius=1.0) == 0


def test_the_selected_label_is_remembered_and_announced():
    seen = []
    stack = LayerStack()
    stack.subscribe(seen.append)
    layer = stack.add_labels(np.ones((2, 2), dtype=np.int32))
    layer.selected_label = 4
    assert layer.selected_label == 4
    assert [e.detail for e in seen if e.kind == "changed"] == ["selected_label"]
    layer.field = _field_key()
    assert layer.field is not None
    with pytest.raises(LayerError, match="FieldKey"):
        layer.field = "plate1"


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------

def test_points_are_stored_in_data_units_and_read_out_in_world_units():
    sp = Spacing.from_map({"y": 2.0, "x": 0.5}, origin={"y": 1.0, "x": 0.0})
    layer = PointsLayer(np.array([[3.0, 4.0]]), spacing=sp)
    assert layer.world[0] == pytest.approx([7.0, 2.0])
    i = layer.add_world({"y": 7.0, "x": 2.0})
    assert layer.data[i] == pytest.approx([3.0, 4.0])


def test_points_can_be_added_removed_and_carry_properties():
    layer = PointsLayer(np.zeros((0, 2)), size=4.0)
    layer.add((1.0, 1.0), category="parasite")
    layer.add((2.0, 2.0), size=8.0, category="host")
    assert len(layer.data) == 2
    assert list(layer.properties["category"]) == ["parasite", "host"]
    assert layer.size.tolist() == [4.0, 8.0]
    layer.remove(0)
    assert list(layer.properties["category"]) == ["host"]
    assert layer.size.tolist() == [8.0]
    with pytest.raises(LayerError, match="no point"):
        layer.remove(5)
    with pytest.raises(LayerError, match="coordinates"):
        layer.add((1.0, 2.0, 3.0))


def test_a_property_column_out_of_step_with_its_points_is_refused():
    layer = PointsLayer(np.zeros((2, 2)), properties={"n": [1, 2]})
    with pytest.raises(LayerError, match="out of step"):
        layer.data = np.zeros((5, 2))
    layer.properties.clear()
    layer.data = np.zeros((5, 2))
    assert len(layer.data) == 5
    with pytest.raises(LayerError, match="values for"):
        PointsLayer(np.zeros((2, 2)), properties={"n": [1]})
    with pytest.raises(LayerError, match="sizes"):
        PointsLayer(np.zeros((2, 2)), size=[1.0])
    with pytest.raises(LayerError, match="N, ndim"):
        PointsLayer(np.zeros((2, 2, 2)))


def test_a_point_is_a_circle_in_the_world_not_in_the_array():
    """An anisotropic grid must not turn a round marker into an ellipse."""
    sp = Spacing.from_map({"y": 4.0, "x": 1.0}, units="um")
    layer = PointsLayer(np.array([[5.0, 20.0]]), spacing=sp, size=8.0,
                        face_color="white")
    canvas = Canvas(origin=(0.0, 0.0), step=(1.0, 1.0), shape=(41, 41),
                    axes=("y", "x"), units="um")
    _, coverage = layer.render(canvas)
    rows = np.flatnonzero(coverage.any(axis=1))
    cols = np.flatnonzero(coverage.any(axis=0))
    # 8 um across in both directions, because the canvas is 1 um per pixel.
    assert rows.max() - rows.min() == cols.max() - cols.min()


def test_a_point_out_of_the_slice_is_not_drawn_and_a_near_one_shrinks():
    sp = Spacing.from_map({"z": 1.0, "y": 1.0, "x": 1.0}, units="um")
    layer = PointsLayer(np.array([[5.0, 10.0, 10.0]]), spacing=sp, size=6.0)
    on = Canvas(origin=(0.0, 0.0), step=(1.0, 1.0), shape=(20, 20),
                depth={"z": 5.0})
    near = on.at_depth(z=7.0)
    far = on.at_depth(z=12.0)
    assert layer.render(on)[1].sum() > layer.render(near)[1].sum() > 0
    assert layer.render(far)[1].max() == 0.0


def test_points_have_an_optional_border_and_pick_the_nearest_point():
    layer = PointsLayer(np.array([[5.0, 5.0]]), size=6.0, face_color="white",
                        border_color="red", border_width=1.0)
    canvas = Canvas(origin=(0.0, 0.0), step=(1.0, 1.0), shape=(11, 11))
    rgb, _ = layer.render(canvas)
    assert rgb[5, 5] == pytest.approx([1.0, 1.0, 1.0])
    assert rgb[5, 2] == pytest.approx([1.0, 0.0, 0.0])
    assert layer.nearest({"y": 5.0, "x": 5.0}) == 0
    assert layer.nearest({"y": 0.0, "x": 0.0}) is None
    assert PointsLayer().nearest({"y": 0.0, "x": 0.0}) is None
    layer.face_color = "green"
    layer.border_color = "blue"
    layer.border_width = 0.0
    layer.set_size(2.0)
    assert layer.size.tolist() == [2.0]


def test_an_empty_points_layer_does_not_drag_the_view_to_the_origin():
    stack = LayerStack()
    stack.add_image(np.zeros((10, 10)), spacing=Spacing.from_map(
        {"y": 1.0, "x": 1.0}, origin={"y": 500.0, "x": 500.0}))
    stack.add_points(name="counts")
    extent = stack.world_extent()
    assert extent["y"][0] > 400.0


# ---------------------------------------------------------------------------
# Shapes — the ROI the Measure item will read
# ---------------------------------------------------------------------------

def test_a_rectangle_rasterises_onto_another_layer_s_grid():
    """Drawn on a preview, honoured on the full-resolution mask."""
    fine = Spacing.from_map({"y": 0.5, "x": 0.5}, units="um")
    coarse = Spacing.from_map({"y": 2.0, "x": 2.0}, units="um")
    shapes = ShapesLayer(spacing=coarse)
    shapes.add_rectangle((1.0, 1.0), (3.0, 3.0))       # 2-6 um in both axes

    grid = Canvas.for_grid(fine, (20, 20))
    mask = shapes.mask(grid)
    rows = np.flatnonzero(mask.any(axis=1))
    # 2 um .. 6 um at 0.5 um per element, half-open: element 4 (2.0 um) is in,
    # element 12 (6.0 um) is not, so two ROIs sharing an edge partition the
    # pixels between them instead of both claiming the seam.
    assert rows.min() == 4 and rows.max() == 11
    assert mask.sum() == 8 * 8


def test_two_rois_sharing_an_edge_do_not_both_claim_it():
    """The reason the boundary rule is half-open rather than inclusive."""
    shapes = ShapesLayer()
    shapes.add_rectangle((0.0, 0.0), (5.0, 10.0))
    shapes.add_rectangle((5.0, 0.0), (10.0, 10.0))
    grid = Canvas.for_grid(Spacing.isotropic(2), (10, 10))
    top = shapes.mask(grid, [0])
    bottom = shapes.mask(grid, [1])
    assert not (top & bottom).any()
    assert (top | bottom).sum() == top.sum() + bottom.sum()


def test_an_ellipse_and_a_polygon_enclose_what_they_should():
    shapes = ShapesLayer()
    shapes.add_ellipse((0.0, 0.0), (10.0, 10.0))
    grid = Canvas.for_grid(Spacing.isotropic(2), (11, 11))
    mask = shapes.mask(grid)
    assert mask[5, 5]
    assert not mask[0, 0]
    assert mask.sum() < 121

    poly = ShapesLayer()
    poly.add_polygon([(0.0, 0.0), (0.0, 10.0), (10.0, 10.0)])
    triangle = poly.mask(grid)
    assert triangle[1, 9]
    assert not triangle[9, 1]


def test_an_open_shape_encloses_nothing_but_still_draws():
    shapes = ShapesLayer()
    shapes.add_path([(0.0, 0.0), (0.0, 9.0)], edge_width=1.0,
                    edge_color="yellow")
    grid = Canvas.for_grid(Spacing.isotropic(2), (10, 10))
    assert shapes.mask(grid).sum() == 0
    rgb, coverage = shapes.render(grid)
    assert coverage[0, 4] > 0
    assert rgb[0, 4] == pytest.approx([1.0, 1.0, 0.0])


def test_shapes_are_only_drawn_on_the_slice_they_were_drawn_in():
    sp = Spacing.from_map({"z": 1.0, "y": 1.0, "x": 1.0})
    shapes = ShapesLayer(spacing=sp, ndim=3)
    shapes.add_rectangle((4.0, 1.0, 1.0), (4.0, 5.0, 5.0))
    here = Canvas(origin=(0.0, 0.0), step=(1.0, 1.0), shape=(8, 8),
                  depth={"z": 4.0})
    assert shapes.mask(here).any()
    assert not shapes.mask(here.at_depth(z=6.0)).any()
    # ...and not at all in a plane the layer does not span.
    assert not shapes.mask(Canvas(origin=(0.0, 0.0), step=(1.0, 1.0),
                                  shape=(4, 4), axes=("q", "r"))).any()


def test_shapes_are_added_removed_and_validated():
    shapes = ShapesLayer()
    i = shapes.add_rectangle((0.0, 0.0), (2.0, 2.0), name="roi")
    assert len(shapes) == 1
    assert shapes.shapes[i].name == "roi"
    assert shapes.remove(0).kind == "rectangle"
    with pytest.raises(LayerError, match="no shape"):
        shapes.remove(0)
    with pytest.raises(LayerError, match="expected a Shape"):
        shapes.add("a rectangle")
    with pytest.raises(LayerError, match="unknown shape kind"):
        Shape("blob", [(0.0, 0.0), (1.0, 1.0)])
    with pytest.raises(LayerError, match="at least two"):
        Shape("polygon", [(0.0, 0.0)])
    with pytest.raises(LayerError, match="three vertices"):
        Shape("polygon", [(0.0, 0.0), (1.0, 1.0)])
    with pytest.raises(LayerError, match="two opposite corners"):
        Shape("rectangle", [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
    with pytest.raises(LayerError, match="3-D"):
        ShapesLayer().add(Shape("polygon", [(0, 0, 0), (1, 1, 1), (2, 2, 2)]))
    with pytest.raises(LayerError, match="same number of axes"):
        ShapesLayer([Shape("polygon", [(0, 0), (1, 1), (2, 2)]),
                     Shape("polygon", [(0, 0, 0), (1, 1, 1), (2, 2, 2)])])


def test_a_shapes_layer_extent_covers_its_vertices():
    shapes = ShapesLayer(spacing=Spacing.from_map({"y": 2.0, "x": 2.0}))
    assert shapes.world_extent()["y"] == (0.0, 0.0)
    shapes.add_rectangle((1.0, 1.0), (3.0, 4.0))
    assert shapes.world_extent()["y"] == pytest.approx((2.0, 6.0))
    assert shapes.world_extent()["x"] == pytest.approx((2.0, 8.0))


# ---------------------------------------------------------------------------
# The stack as a list
# ---------------------------------------------------------------------------

def test_layers_are_reachable_by_name_index_and_object():
    stack = LayerStack()
    a = stack.add_image(np.zeros((2, 2)), name="a")
    b = stack.add_labels(np.zeros((2, 2), dtype=int), name="b")
    assert stack[0] is a and stack["b"] is b
    assert stack[-1] is b
    assert stack.index(b) == 1 and stack.index("a") == 0
    assert "a" in stack and a in stack and "z" not in stack
    assert list(stack) == [a, b]
    assert stack[0:1] == [a]
    assert len(stack) == 2
    with pytest.raises(KeyError, match="no layer named"):
        stack["nope"]
    with pytest.raises(LayerError, match="no layer named"):
        stack.index("nope")
    with pytest.raises(LayerError, match="no layer at index"):
        stack.index(9)
    with pytest.raises(LayerError, match="not in this stack"):
        LayerStack().index(a)


def test_a_duplicate_name_is_suffixed_rather_than_refused():
    """Adding a second mask should get you one, not an error dialog."""
    stack = LayerStack()
    stack.add_labels(np.zeros((2, 2), dtype=int), name="mask")
    stack.add_labels(np.zeros((2, 2), dtype=int), name="mask")
    stack.add_labels(np.zeros((2, 2), dtype=int), name="mask")
    assert stack.names == ("mask", "mask [1]", "mask [2]")
    # Renaming onto a taken name gets the smallest free suffix of it — which
    # for this layer is the name it already had.
    assert stack.rename("mask [2]", "mask") == "mask [2]"
    assert stack.rename("mask [2]", "mask [2]") == "mask [2]"
    stack["mask"].name = "cells"
    assert "cells" in stack.names
    with pytest.raises(LayerError, match="non-blank"):
        stack.rename("cells", "   ")


def test_a_layer_belongs_to_one_stack_at_a_time():
    stack = LayerStack()
    layer = stack.add_image(np.zeros((2, 2)), name="a")
    assert layer.stack is stack
    with pytest.raises(LayerError, match="already in this stack"):
        stack.append(layer)
    with pytest.raises(LayerError, match="expected a Layer"):
        stack.append(np.zeros((2, 2)))
    stack.remove(layer)
    assert layer.stack is None
    assert len(stack) == 0


def test_removing_the_selected_layer_selects_a_neighbour():
    stack = LayerStack()
    a = stack.add_image(np.zeros((2, 2)), name="a")
    b = stack.add_image(np.zeros((2, 2)), name="b")
    assert stack.selected is a and stack.selected_index == 0
    stack.select(b)
    assert stack.selected is b
    stack.remove(b)
    assert stack.selected is a
    stack.remove(a)
    assert stack.selected is None and stack.selected_index == -1
    stack.add_image(np.zeros((2, 2)), name="c")
    stack.select(None)
    assert stack.selected is None


def test_clear_empties_the_stack_and_announces_every_removal():
    seen = []
    stack = LayerStack()
    for name in ("a", "b", "c"):
        stack.add_image(np.zeros((2, 2)), name=name)
    stack.subscribe(seen.append)
    stack.clear()
    assert len(stack) == 0
    assert [e.kind for e in seen] == ["removed"] * 3


def test_every_change_a_view_must_repaint_for_is_announced():
    events = []
    stack = LayerStack()
    stack.subscribe(events.append)
    layer = stack.add_image(np.zeros((4, 4)), name="img")
    layer.visible = False
    layer.opacity = 0.5
    layer.blending = Blending.ADDITIVE
    layer.set_colormap("red")
    layer.spacing = Spacing.isotropic(2, 2.0)
    stack.rename(layer, "renamed")
    stack.add_labels(np.zeros((4, 4), dtype=int), name="mask")
    stack.move("mask", 0)
    stack.select("mask")
    kinds = [e.kind for e in events]
    assert kinds == ["inserted", "changed", "changed", "changed", "changed",
                     "changed", "renamed", "inserted", "moved", "selected"]
    assert [e.detail for e in events if e.kind == "changed"] == [
        "visible", "opacity", "blending", "colormap", "spacing"]
    assert all(e.kind in LayerEvent.REPAINT or e.kind in ("renamed", "selected")
               for e in events)


def test_setting_a_property_to_what_it_already_is_says_nothing():
    """Otherwise every slider drag repaints twice."""
    events = []
    stack = LayerStack()
    layer = stack.add_image(np.zeros((2, 2)))
    stack.subscribe(events.append)
    layer.visible = True
    layer.opacity = 1.0
    layer.blending = Blending.TRANSLUCENT
    layer.name = layer.name
    assert events == []


def test_unsubscribing_stops_the_callbacks():
    events = []
    stack = LayerStack()
    stack.subscribe(events.append)
    stack.subscribe(events.append)      # idempotent
    stack.add_image(np.zeros((2, 2)))
    assert len(events) == 1
    assert stack.unsubscribe(events.append) is True
    assert stack.unsubscribe(events.append) is False
    stack.add_image(np.zeros((2, 2)))
    assert len(events) == 1
    with pytest.raises(LayerError, match="callable"):
        stack.subscribe("not a function")


def test_a_layer_outside_a_stack_can_still_be_renamed_and_changed():
    layer = ImageLayer(np.zeros((2, 2)), name="lonely")
    layer.name = "still lonely"
    layer.visible = False
    assert layer.name == "still lonely"
    assert layer.visible is False
    with pytest.raises(LayerError, match="non-blank"):
        ImageLayer(np.zeros((2, 2)), name="  ")
    with pytest.raises(LayerError, match="spacing has"):
        ImageLayer(np.zeros((2, 2)), spacing=Spacing.isotropic(3))
    with pytest.raises(LayerError, match="must be a Spacing"):
        layer.spacing = (1.0, 1.0)
    with pytest.raises(LayerError, match="spatial axes"):
        layer.spacing = Spacing.isotropic(3)


def test_pick_finds_the_topmost_visible_thing_under_a_pixel():
    stack = LayerStack()
    stack.add_image(np.zeros((10, 10)), name="img")
    mask = np.zeros((10, 10), dtype=np.int32)
    mask[5, 5] = 2
    labels = stack.add_labels(mask, name="mask", field=_field_key())
    points = stack.add_points(np.array([[5.0, 5.0]]), name="dots", size=1.0)
    canvas = Canvas.for_grid(labels.spacing, labels.shape)

    layer, world, value = stack.pick(canvas, 5, 5)
    assert layer is points and value == 0
    points.visible = False
    layer, world, value = stack.pick(canvas, 5, 5)
    assert layer is labels and value == 2
    assert labels.object_key_at_world(world) == "plate1_A_1_1_2"
    labels.visible = False
    assert stack.pick(canvas, 5, 5)[0] is None
    assert stack.pick(canvas, 0, 0)[0] is None


def test_pick_finds_a_shape_under_the_cursor():
    stack = LayerStack()
    shapes = stack.add_shapes(name="rois")
    shapes.add_rectangle((2.0, 2.0), (6.0, 6.0))
    canvas = Canvas(origin=(0.0, 0.0), step=(1.0, 1.0), shape=(10, 10))
    assert stack.pick(canvas, 4, 4)[2] == 0
    assert stack.pick(canvas, 9, 9)[0] is None


def test_the_stack_describes_itself_top_first():
    stack = LayerStack()
    stack.add_image(np.zeros((2, 2)), name="under")
    stack.add_labels(np.zeros((2, 2), dtype=int), name="over", opacity=0.5)
    lines = stack.describe().splitlines()
    assert lines[0].startswith("over (labels)")
    assert "50%" in lines[0]
    assert lines[1].startswith("under (image)")
    stack["under"].visible = False
    assert "hidden" in stack.describe().splitlines()[1]


# ---------------------------------------------------------------------------
# The edges: guards, degenerate inputs and the accessors a view reads
# ---------------------------------------------------------------------------

def test_colour_and_colormap_odds_and_ends():
    assert to_rgba("red", alpha=0.25)[3] == pytest.approx(0.25)
    with pytest.raises(LayerError, match="hex"):
        to_rgba("#gggggg")
    cm = colormap("#ff0000")
    assert cm.map(1.0) == pytest.approx([1.0, 0.0, 0.0])
    assert colormap((0.0, 0.0, 1.0)).map(1.0) == pytest.approx([0.0, 0.0, 1.0])
    ramp = Colormap("two", ["black", "white"])
    assert ramp.colors.shape == (2, 3)
    assert ramp.stops.tolist() == [0.0, 1.0]
    assert hash(ramp) == hash(Colormap("two", ["black", "white"]))
    assert ramp != "not a colormap"
    assert Colormap("stops", ["black", "red", "white"],
                    stops=[0.0, 0.25, 1.0]).map(0.25) == pytest.approx(
                        [1.0, 0.0, 0.0])


def test_a_four_dimensional_spacing_names_its_extra_axes():
    sp = Spacing.isotropic(5)
    assert sp.axes == ("d0", "d1", "z", "y", "x")
    assert sp.has_axis("d0") and not sp.has_axis("t")


def test_every_way_a_spacing_can_be_malformed_is_caught():
    with pytest.raises(LayerError, match="sequence of numbers"):
        Spacing(scale="not numbers")
    with pytest.raises(LayerError, match="at least one axis"):
        Spacing(scale=())
    with pytest.raises(LayerError, match="translate has"):
        Spacing(scale=(1.0, 1.0), translate=(0.0,))
    with pytest.raises(LayerError, match="must be finite"):
        Spacing(scale=(1.0, 1.0), translate=(0.0, float("inf")))
    sp = Spacing.isotropic(2)
    with pytest.raises(LayerError, match="index"):
        sp.to_world((1.0,))
    with pytest.raises(LayerError, match="world point"):
        sp.to_data((1.0,))
    with pytest.raises(LayerError, match="shape"):
        sp.extent((4,))


def test_a_canvas_can_be_fitted_to_a_bare_extent():
    canvas = Canvas.covering({"y": (0.0, 10.0), "x": (0.0, 20.0)})
    assert canvas.height == 512
    assert canvas.width == 1024
    assert canvas.units == "px"
    square = Canvas.covering({"y": (0.0, 4.0), "x": (0.0, 4.0)}, height=8,
                             width=8)
    assert square.shape == (8, 8)
    assert square.zoomed(2.0).step == pytest.approx(
        (square.step[0] / 2, square.step[1] / 2))


class _BareLayer(Layer):
    """A layer that implements nothing, to prove the base class refuses to
    guess rather than drawing an empty rectangle."""

    @property
    def ndim(self):
        return 2


def test_the_base_layer_refuses_to_guess_what_it_cannot_know():
    layer = _BareLayer(name="bare")
    assert layer.shape == ()
    assert layer.axes == ("y", "x")
    assert layer.to_world((1, 2)) == {"y": 1.0, "x": 2.0}
    assert layer.to_data({"y": 1.0, "x": 2.0}) == (1.0, 2.0)
    with pytest.raises(NotImplementedError):
        layer.world_extent()
    with pytest.raises(NotImplementedError):
        layer.render(Canvas(origin=(0.0, 0.0), step=(1.0, 1.0), shape=(2, 2)))
    assert repr(layer).startswith("<_BareLayer 'bare'")


def test_image_layer_accessors_and_degenerate_data():
    data = np.arange(8, dtype=np.float32).reshape(2, 4)
    layer = ImageLayer(data)
    assert layer.data is data
    empty = ImageLayer(np.zeros((0, 4)))
    assert empty.contrast_limits() == (0.0, 1.0)
    flat = ImageLayer(np.zeros((4, 4)))
    flat.auto_contrast()
    lo, hi = flat.contrast_limits()
    assert hi > lo
    mixed = ImageLayer(np.zeros((2, 4, 4)), channel_axis=0,
                       contrast_limits=[None, (0.0, 1.0)])
    assert mixed.contrast_limits(1) == (0.0, 1.0)
    assert mixed.contrast_limits(0)[1] > mixed.contrast_limits(0)[0]


def test_points_accessors_and_the_cases_that_draw_nothing():
    single = PointsLayer([2.0, 3.0])
    assert single.data.shape == (1, 2)
    assert single.face_color[3] == 1.0
    assert single.border_color[:3] == (0.0, 0.0, 0.0)
    assert single.border_width == 0.0
    assert PointsLayer().world_extent()["y"] == (0.0, 0.0)

    canvas = Canvas(origin=(0.0, 0.0), step=(1.0, 1.0), shape=(8, 8))
    assert PointsLayer().render(canvas)[1].max() == 0.0
    # A zero-size point, and one entirely off the canvas, draw nothing.
    assert PointsLayer([[2.0, 2.0]], size=0.0).render(canvas)[1].max() == 0.0
    assert PointsLayer([[99.0, 99.0]], size=2.0).render(canvas)[1].max() == 0.0
    # A point smaller than a pixel and centred between two of them covers
    # neither, rather than covering the one it is nearest.
    assert PointsLayer([[2.5, 2.5]], size=0.2).render(canvas)[1].max() == 0.0
    # ...and a layer that does not live in the plane being drawn.
    off = PointsLayer([[1.0, 1.0]], spacing=Spacing(scale=(1.0, 1.0),
                                                    axes=("q", "r")))
    assert off.render(canvas)[1].max() == 0.0


def test_shapes_render_their_fill_and_survive_a_repeated_vertex():
    shapes = ShapesLayer()
    shapes.add_polygon([(1.0, 1.0), (1.0, 6.0), (6.0, 6.0), (6.0, 1.0)],
                       face_color=(0.0, 1.0, 0.0, 1.0), edge_width=0.0)
    shapes.add_path([(0.0, 0.0), (0.0, 0.0)], edge_width=1.0,
                    edge_color="white")
    grid = Canvas.for_grid(Spacing.isotropic(2), (8, 8))
    rgb, coverage = shapes.render(grid)
    assert rgb[3, 3] == pytest.approx([0.0, 1.0, 0.0])
    assert coverage[3, 3] == 1.0
    assert coverage[0, 0] == 1.0        # the degenerate path is still a dot
    # A shape with a fully transparent face contributes no fill.
    faint = ShapesLayer()
    faint.add_rectangle((1.0, 1.0), (5.0, 5.0), face_color=(1.0, 1.0, 1.0, 0.0),
                        edge_width=0.0)
    assert faint.render(grid)[1].max() == 0.0
    # ...and one out of the plane is not drawn at all.
    solid = ShapesLayer(spacing=Spacing.from_map({"z": 1.0, "y": 1.0,
                                                  "x": 1.0}), ndim=3)
    solid.add_rectangle((4.0, 1.0, 1.0), (4.0, 5.0, 5.0))
    assert solid.render(Canvas(origin=(0.0, 0.0), step=(1.0, 1.0),
                               shape=(8, 8), depth={"z": 0.0}))[1].max() == 0.0


def test_a_negative_stack_index_counts_from_the_top():
    stack = LayerStack()
    stack.add_image(np.zeros((2, 2)), name="a")
    stack.add_image(np.zeros((2, 2)), name="b")
    assert stack.index(-1) == 1
    assert stack.index(-2) == 0
    with pytest.raises(LayerError, match="no layer at index"):
        stack.index(-3)


class _NoDimensionsLayer(Layer):
    """A layer that never says how many axes it has."""


def test_a_layer_that_never_declares_its_axes_cannot_be_built():
    with pytest.raises(NotImplementedError):
        _NoDimensionsLayer(name="nameless")


def test_per_point_sizes_and_a_replaced_points_array_of_the_wrong_width():
    layer = PointsLayer(np.zeros((3, 2)), size=[1.0, 2.0, 3.0])
    assert layer.size.tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(LayerError, match="is 3-D"):
        layer.data = np.zeros((3, 3))


def test_an_ellipse_is_filled_when_it_is_drawn_not_only_when_it_is_masked():
    shapes = ShapesLayer()
    shapes.add_ellipse((0.0, 0.0), (8.0, 8.0), face_color=(1.0, 0.0, 0.0, 1.0),
                       edge_width=0.0)
    grid = Canvas.for_grid(Spacing.isotropic(2), (9, 9))
    rgb, coverage = shapes.render(grid)
    assert rgb[4, 4] == pytest.approx([1.0, 0.0, 0.0])
    assert coverage[0, 0] == 0.0


def test_an_empty_shapes_layer_does_not_drag_the_view_to_the_origin():
    stack = LayerStack()
    stack.add_image(np.zeros((10, 10)), spacing=Spacing.from_map(
        {"y": 1.0, "x": 1.0}, origin={"y": 500.0, "x": 500.0}))
    stack.add_shapes(name="rois")
    assert stack.world_extent()["y"][0] > 400.0
