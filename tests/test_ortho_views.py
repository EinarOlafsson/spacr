"""``B15`` — the geometry of an orthogonal view.

The thing this class exists to get right is anisotropy, and the reason it needs
its own tests is that getting it wrong is invisible. A confocal stack at
0.65 µm in xy and 2 µm in z drawn one pixel per slice gives a side view three
times too thin: not a broken picture, a slightly flat cell. Every 3-D shape
read off it — sphericity, depth, colocalisation along z — is wrong by a factor
nobody has any reason to suspect.

So the assertions here are on measured geometry, in pixels and in world units,
against stacks with real anisotropic spacing. A test that only checked "three
canvases came back" would pass on the broken version.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.layers import (Canvas, LayerError, LayerStack, OrthoViews, Spacing)


#: 0.65 µm in xy and 2 µm in z — an ordinary spaCR confocal stack, and the one
#: worth testing against, because every axis-order and voxel-size mistake
#: produces a different number on it.
CONFOCAL = Spacing.from_map({'z': 2.0, 'y': 0.65, 'x': 0.65}, units='um')


def volume_stack(shape=(10, 64, 64), spacing=CONFOCAL):
    """A stack holding one anisotropic volume with a bright blob in it."""
    stack = LayerStack()
    data = np.zeros(shape, np.uint16)
    data[shape[0] // 2, 30:34, 30:34] = 4000
    stack.add_image(data, name='volume', spacing=spacing,
                    contrast_limits=(0.0, 4000.0))
    return stack


# ---------------------------------------------------------------------------
# 1. the anisotropy
# ---------------------------------------------------------------------------

def test_the_side_views_are_as_deep_as_the_stack_is_thick():
    """A 10-slice 2 µm stack is 20 µm deep, whatever the slice count is."""
    views = OrthoViews.covering(volume_stack(), width=128)

    assert views.xy.shape == (128, 128)
    # The z extent is 10 slices x 2 µm = 20 µm, and the scale is
    # 64 x 0.65 / 128 = 0.325 µm/px, so the side views are 20/0.325 ≈ 62 px.
    assert views.zx.shape == (62, 128)
    assert views.yz.shape == (128, 62)
    assert views.zx.shape[0] != 10, (
        'the side view is one pixel per slice: the z voxel size was ignored '
        'and every depth read off this picture is wrong by the anisotropy')


def test_every_panel_is_at_the_same_world_scale():
    views = OrthoViews.covering(volume_stack(), width=200)
    steps = {canvas.step for canvas in views.canvases().values()}
    assert len(steps) == 1, 'the panels are at different magnifications'
    assert views.scale == pytest.approx(64 * 0.65 / 200)


def test_halving_the_z_voxel_size_halves_the_side_view():
    """The side view's height is a fact about the sample, not about the file."""
    thick = OrthoViews.covering(volume_stack(spacing=CONFOCAL), width=128)
    thin = OrthoViews.covering(
        volume_stack(spacing=CONFOCAL.rescaled(z=1.0)), width=128)
    assert thin.zx.shape[0] == pytest.approx(thick.zx.shape[0] / 2, abs=1)


def test_an_isotropic_stack_is_square_in_every_plane():
    views = OrthoViews.covering(
        volume_stack(shape=(64, 64, 64),
                     spacing=Spacing.isotropic(3, 1.0)), width=64)
    assert views.xy.shape == views.zx.shape == views.yz.shape == (64, 64)


def test_a_2d_field_has_no_second_plane_and_says_so():
    stack = LayerStack()
    stack.add_image(np.zeros((32, 32), np.uint16), name='field')
    with pytest.raises(LayerError, match='no second plane'):
        OrthoViews.covering(stack)


# ---------------------------------------------------------------------------
# 2. the planes, and what each one shows
# ---------------------------------------------------------------------------

def test_the_panels_share_their_edges_with_the_top_view():
    """The layout is only honest if zx shares x with xy and yz shares y."""
    views = OrthoViews.covering(volume_stack(), width=128)
    assert views.xy.axes == ('y', 'x')
    assert views.zx.axes == ('z', 'x')
    assert views.yz.axes == ('y', 'z')
    assert views.xy.shape[1] == views.zx.shape[1]   # same columns: x
    assert views.xy.shape[0] == views.yz.shape[0]   # same rows: y
    assert views.axes == ('z', 'y', 'x')


def test_each_panel_is_pinned_at_the_axes_it_does_not_span():
    views = OrthoViews.covering(volume_stack(), width=128)
    assert set(views.xy.depth) == {'z'}
    assert set(views.zx.depth) == {'y'}
    assert set(views.yz.depth) == {'x'}


def test_the_crosshair_starts_in_the_middle_of_the_volume():
    """Slice 0 of a confocal stack is usually empty."""
    views = OrthoViews.covering(volume_stack(), width=128)
    assert views.point['z'] == pytest.approx(9.0)  # (-1 .. 19) µm
    assert views.point['y'] == pytest.approx(20.475)


def test_the_crosshair_can_be_placed_where_the_caller_wants_it():
    views = OrthoViews.covering(volume_stack(), width=128,
                                point={'z': 4.0})
    assert views.point['z'] == pytest.approx(4.0)
    assert views.xy.depth['z'] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# 3. moving the crosshair
# ---------------------------------------------------------------------------

def test_moving_z_changes_the_top_view_and_nothing_else():
    views = OrthoViews.covering(volume_stack(), width=128)
    moved = views.at(z=12.0)
    assert moved.xy.depth['z'] == pytest.approx(12.0)
    assert moved.zx.depth == views.zx.depth
    assert moved.yz.depth == views.yz.depth
    assert moved.point['z'] == pytest.approx(12.0)
    # The original is untouched: these are values.
    assert views.xy.depth['z'] == pytest.approx(9.0)


def test_a_slider_past_the_end_shows_the_end_rather_than_nothing():
    views = OrthoViews.covering(volume_stack(), width=128)
    assert views.clamped(z=1e6).point['z'] == pytest.approx(19.0)
    assert views.clamped(z=-1e6).point['z'] == pytest.approx(-1.0)
    # An axis the views do not have is left where it was asked for rather
    # than clamped against an extent that does not exist.
    assert views.clamped(t=3.0).point['t'] == pytest.approx(3.0)


def test_clicking_a_side_view_moves_the_top_view_s_slice():
    """The interaction an orthogonal view is for."""
    views = OrthoViews.covering(volume_stack(), width=128)
    moved = views.at_pixel('zx', 5.0, 64.0)
    expected = views.zx.world_at(5.0, 64.0)
    assert moved.point['z'] == pytest.approx(expected['z'])
    assert moved.point['x'] == pytest.approx(expected['x'])
    assert moved.point['y'] == pytest.approx(views.point['y'])
    assert moved.xy.depth['z'] == pytest.approx(expected['z'])


def test_clicking_a_panel_that_does_not_exist_says_which_do():
    views = OrthoViews.covering(volume_stack(), width=128)
    with pytest.raises(LayerError, match="'xy', 'zx' and 'yz'"):
        views.at_pixel('xz', 0.0, 0.0)


# ---------------------------------------------------------------------------
# 4. the sliders
# ---------------------------------------------------------------------------

def test_the_slider_is_in_world_units_and_steps_by_one_slice():
    views = OrthoViews.covering(volume_stack(), width=128)
    low, high, step = views.slider('z')
    assert (low, high) == (-1.0, 19.0)     # outer edges of the end voxels
    assert step == pytest.approx(2.0)      # the z voxel size, in µm
    assert views.n_slices('z') == 10
    assert views.n_slices('x') == 64


def test_the_slider_step_is_the_finest_voxel_where_layers_disagree():
    """A slider stepping by the coarse layer's z would skip the fine one."""
    stack = volume_stack()
    stack.add_labels(np.zeros((20, 64, 64), np.int32), name='mask',
                     spacing=Spacing.from_map({'z': 1.0, 'y': 0.65, 'x': 0.65},
                                              units='um'))
    views = OrthoViews.covering(stack, width=128)
    assert views.slider('z')[2] == pytest.approx(1.0)


def test_a_slider_on_an_axis_the_views_do_not_have_says_which_they_have():
    views = OrthoViews.covering(volume_stack(), width=128)
    with pytest.raises(LayerError, match="no axis 't'"):
        views.slider('t')


# ---------------------------------------------------------------------------
# 5. zoom, resize and rendering
# ---------------------------------------------------------------------------

def test_zooming_keeps_every_panel_at_one_scale_and_on_the_crosshair():
    views = OrthoViews.covering(volume_stack(), width=128)
    zoomed = views.zoomed(2.0)

    assert zoomed.scale == pytest.approx(views.scale / 2)
    assert len({canvas.step for canvas in zoomed.canvases().values()}) == 1
    for name, canvas in zoomed.canvases().items():
        before = views.canvases()[name].pixel_at(views.point)
        after = canvas.pixel_at(views.point)
        assert after == pytest.approx(before, abs=1e-6), (
            f'the {name} panel slid out from under the crosshair')


def test_a_zoom_factor_that_is_not_a_zoom_raises():
    views = OrthoViews.covering(volume_stack(), width=128)
    with pytest.raises(LayerError, match='must be positive'):
        views.zoomed(0.0)


def test_resizing_keeps_the_crosshair_and_the_alignment():
    views = OrthoViews.covering(volume_stack(), width=128).at(z=12.0)
    bigger = views.resized(256)
    assert bigger.xy.shape[1] == 256
    assert bigger.point['z'] == pytest.approx(12.0)
    assert bigger.xy.shape[1] == bigger.zx.shape[1]
    assert bigger.scale == pytest.approx(views.scale / 2)
    assert bigger.slider('z')[2] == pytest.approx(2.0)


def test_the_three_panels_render_the_same_object_in_all_three_planes():
    """The blob is at z=5, y≈30-33, x≈30-33; every panel must show it."""
    stack = volume_stack()
    views = OrthoViews.covering(stack, width=128).at(
        z=10.0, y=30 * 0.65, x=30 * 0.65)
    pictures = views.render(stack)

    for name, canvas in views.canvases().items():
        row, column = canvas.pixel_at(views.point)
        pixel = pictures[name][int(round(row)), int(round(column))]
        assert pixel.max() > 0.5, (
            f'the {name} panel is not showing the object the crosshair is on')
    # ...and a corner of the top view, far from the blob, is dark.
    assert pictures['xy'][2, 2].max() < 0.1


def test_the_side_view_finds_the_object_at_its_real_depth():
    """The test the squashed side view fails.

    The blob is on slice 5 of a 2 µm stack: 10 µm down, which at 0.325 µm/px is
    row 34 of the ZX panel. One pixel per slice would put it at row 5.
    """
    stack = volume_stack()
    views = OrthoViews.covering(stack, width=128)
    picture = views.render(stack)['zx']
    rows = np.flatnonzero(picture.max(axis=(1, 2)) > 0.5)
    assert len(rows), 'the object is not in the side view at all'
    assert 30 <= rows.mean() <= 38, (
        f'the object is at row {rows.mean():.1f} of the side view; at the '
        f'stack\'s real 2 µm z-spacing it belongs at ~34')


def test_the_description_says_where_the_crosshair_is_and_in_what():
    views = OrthoViews.covering(volume_stack(), width=128).at(z=12.0)
    assert views.describe().startswith('z 12 · y ')
    assert views.describe().endswith('(um)')
