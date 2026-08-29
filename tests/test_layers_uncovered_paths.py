"""The layer-model paths a viewer only reaches on a second, identical action.

Most of what is exercised here is the *no-op* half of a setter: setting a
colormap that is already in use, re-selecting the layer that is already
selected, clicking a pixel where the top layer has nothing. Those are the
common cases in a live viewer -- a slider re-emitting its own value, a paint
loop re-selecting a label, a click landing on background -- and each one has
to be quiet rather than merely harmless, because a repaint per mouse move over
a 2048x2048 field is the difference between a responsive viewer and a stuck
one.

The rest are the fallbacks: a nearly-empty image whose percentile stretch
collapses, a leader panel that is not the first panel, and a matplotlib
colormap name reaching the lazy import that exists so this module does not
have to depend on matplotlib.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.layers import (Canvas, CanvasLink, Colormap, FieldKey, ImageLayer,
                          LabelsLayer, LayerStack, PointsLayer, Shape,
                          ShapesLayer, colormap)


def _canvas(origin=(0.0, 0.0), step=(1.0, 1.0), shape=(4, 4)) -> Canvas:
    return Canvas(origin=origin, step=step, shape=shape, axes=("y", "x"))


def _events(stack: LayerStack) -> list:
    """Subscribe to ``stack`` and return the list its events land in."""
    seen: list = []
    stack.subscribe(seen.append)
    return seen


# -- Colormap ------------------------------------------------------------

def test_a_colormap_reprs_as_its_name_and_its_number_of_stops():
    """``repr`` names the ramp and says how many colours it interpolates.

    A colormap in a traceback or a debugger is otherwise an opaque object with
    two float arrays; the name is the only thing that says which channel's LUT
    went wrong, and the stop count distinguishes a two-colour channel ramp
    from a sampled perceptual map that happens to share a name.
    """
    ramp = Colormap("mito", ["black", "red"])
    assert repr(ramp) == "Colormap('mito', 2 stops)"

    sampled = Colormap("fire", ["black", "red", "orange", "yellow", "white"])
    assert repr(sampled) == "Colormap('fire', 5 stops)"


def test_a_matplotlib_colormap_name_is_sampled_into_a_sixteen_stop_ramp():
    """``"viridis"`` resolves through the lazy matplotlib import, not an error.

    matplotlib is tried last and inside a ``try`` so this module stays
    importable without it. The success half of that ``try`` has to produce a
    real ramp: sixteen samples taken along the matplotlib map, keeping its
    name, and agreeing with matplotlib's own colours at both ends.
    """
    pytest.importorskip("matplotlib")
    from matplotlib import colormaps as mpl_colormaps

    ramp = colormap("viridis")

    assert isinstance(ramp, Colormap)
    assert ramp.name == "viridis"
    assert ramp.colors.shape == (16, 3), "sixteen samples along the map"
    assert ramp.stops[0] == 0.0 and ramp.stops[-1] == 1.0

    # It is matplotlib's viridis and not some grey stand-in: the ends match
    # what matplotlib itself returns, and the ramp is not monochrome.
    reference = mpl_colormaps["viridis"]
    assert np.allclose(ramp.colors[0], reference(0.0)[:3], atol=1e-6)
    assert np.allclose(ramp.colors[-1], reference(1.0)[:3], atol=1e-6)
    assert not np.allclose(ramp.colors[0], ramp.colors[-1])


def test_an_unknown_name_still_raises_rather_than_reaching_matplotlib():
    """A name matplotlib does not have is reported as an unknown colormap."""
    with pytest.raises(Exception) as excinfo:
        colormap("not_a_colormap_anywhere")
    assert "unknown colormap 'not_a_colormap_anywhere'" in str(excinfo.value)


# -- CanvasLink ----------------------------------------------------------

def test_the_leader_panel_is_the_first_locked_one_not_the_first_added():
    """A panel added while the first panel is unlocked follows the locked one.

    The user unlocks panel "a" to look closely at one channel; "a" is still
    first in the link. A panel added afterwards must adopt the window the
    *linked* panels share, which is "b"'s, and leave the detached panel where
    the user put it.
    """
    link = CanvasLink()
    link.add("a", _canvas(origin=(0.0, 0.0), step=(1.0, 1.0)), locked=False)
    link.add("b", _canvas(origin=(50.0, 60.0), step=(2.0, 2.0)))

    # "b" was added while the only existing panel was unlocked, so there was
    # no leader to follow and it kept its own window.
    assert link["a"].origin == (0.0, 0.0)
    assert link["b"].origin == (50.0, 60.0)
    assert link["b"].step == (2.0, 2.0)

    added = link.add("c", _canvas(origin=(999.0, 999.0), step=(9.0, 9.0)))

    assert added.origin == (50.0, 60.0), "followed 'b', the first locked panel"
    assert added.step == (2.0, 2.0)
    assert added.shape == (4, 4), "its own pixel size is its own business"
    assert link["a"].origin == (0.0, 0.0), "the unlocked panel did not move"


def test_replacing_an_unlocked_panels_canvas_leaves_the_others_alone():
    """``set`` on a detached panel drives nothing; on a linked one, all.

    Unlocking is what a user does to inspect one panel without dragging the
    grid along. If the detached panel still led, unlocking would only change
    which direction the coupling ran in.
    """
    link = CanvasLink({"a": _canvas(), "b": _canvas(), "c": _canvas()})
    link.unlock("a")

    link.set("a", _canvas(origin=(30.0, 40.0), step=(3.0, 3.0)))

    assert link["a"].origin == (30.0, 40.0)
    assert link["b"].origin == (0.0, 0.0), "a detached panel does not drive"
    assert link["c"].origin == (0.0, 0.0)
    assert link["b"].step == (1.0, 1.0)

    # And the locked half still couples, so the difference is the lock and
    # not a broken `set`.
    link.set("b", _canvas(origin=(7.0, 8.0), step=(0.5, 0.5)))
    assert link["c"].origin == (7.0, 8.0)
    assert link["c"].step == (0.5, 0.5)
    assert link["a"].origin == (30.0, 40.0), "still detached"


# -- FieldKey ------------------------------------------------------------

def test_from_row_takes_an_explicit_object_type_over_the_rows_own_column():
    """An ``object_type`` argument wins and the row's column is not consulted.

    A caller that already knows which mask it loaded -- the nucleus mask of a
    row that happens to carry ``object_type='cell'`` from the cell table it
    was joined to -- must be able to say so, or the keys it publishes name the
    wrong object.
    """
    row = {"plateID": "plate1", "rowID": "A", "columnID": "1", "fieldID": "1",
           "object_type": "cell", "area": 123.0}

    key = FieldKey.from_row(row, object_type="Nucleus")

    assert key.object_type == "nucleus", "normalised, and not the row's 'cell'"
    assert dict(key.values) == {"plateID": "plate1", "rowID": "A",
                                "columnID": "1", "fieldID": "1"}
    assert "area" not in key.values, "a measurement is not part of an identity"

    # Without the argument the row's own column is what is used, which is the
    # branch this one skips.
    assert FieldKey.from_row(row).object_type == "cell"


# -- ImageLayer ----------------------------------------------------------

def test_setting_the_colormap_a_channel_already_has_notifies_nobody():
    """Re-applying the current LUT is silent; a different one repaints.

    A colormap combo box re-emits its current value whenever the layer list
    rebuilds. If that repainted, switching layers would repaint every channel
    of every layer for nothing.
    """
    stack = LayerStack()
    image = stack.add_image(np.arange(16, dtype=np.float32).reshape(4, 4))
    image.set_colormap("green")
    seen = _events(stack)

    image.set_colormap("green")
    assert seen == [], "the LUT did not change, so nothing repaints"
    assert image.colormap.name == "green"

    image.set_colormap("red")
    assert [e.detail for e in seen] == ["colormap"]
    assert image.colormap.name == "red"


def test_a_sparse_image_stretches_to_min_max_when_the_percentiles_collapse():
    """Contrast falls back to min/max when 2–98% is flat, and not to a hack.

    A field that is background with a handful of bright puncta -- a punctate
    marker, a sparse mask overlay -- has identical 2nd and 98th percentiles.
    Without the fallback the layer would divide by a zero-width window; with a
    fallback that only ever added 1.0 the puncta would clip. The real min and
    max are used, so the brightest punctum sits exactly at the top of the LUT.
    """
    data = np.zeros((100, 10), dtype=np.float64)
    data[0, :5] = 400.0            # 5 bright pixels out of 1000
    image = ImageLayer(data, name="puncta")

    lo, hi = image.contrast_limits()

    assert (lo, hi) == (0.0, 400.0), "the data's own range, not lo+1"
    # The stretch is real: the puncta reach white and the background does not.
    rgb, _ = image.render(Canvas.for_grid(image.spacing, image.shape))
    assert rgb[0, 0, :].max() == pytest.approx(1.0)
    assert rgb[50, 5, :].max() == pytest.approx(0.0)


def test_a_flat_image_widens_its_limits_by_one_rather_than_dividing_by_zero():
    """An all-constant plane still gets an increasing pair of limits."""
    image = ImageLayer(np.full((4, 4), 7.0), name="flat")
    assert image.contrast_limits() == (7.0, 8.0)


def test_setting_the_contrast_limits_a_channel_already_has_notifies_nobody():
    """Re-applying the current limits is silent; moving them repaints.

    Two contrast sliders emit on every drag step, and both emit again when the
    layer is re-selected. Repainting on a value that did not move is a repaint
    per mouse event over the whole canvas.
    """
    stack = LayerStack()
    image = stack.add_image(np.arange(16, dtype=np.float32).reshape(4, 4))
    image.set_contrast_limits(2.0, 9.0)
    seen = _events(stack)

    image.set_contrast_limits(2.0, 9.0)
    assert seen == []
    assert image.contrast_limits() == (2.0, 9.0)

    image.set_contrast_limits(2.0, 10.0)
    assert [e.detail for e in seen] == ["contrast_limits"]
    assert image.contrast_limits() == (2.0, 10.0)


def test_auto_contrast_on_an_empty_channel_clears_it_to_the_default_stretch():
    """A zero-size plane cannot be measured, so its limits go back to unset.

    An empty plane is what a cropped ROI with no rows or a not-yet-loaded tile
    leaves behind. ``np.percentile`` on it raises, so the plane is skipped and
    the channel is returned to the lazily-computed default rather than being
    left holding a stale window from before the crop.
    """
    image = ImageLayer(np.zeros((0, 4)), name="cropped-away")
    image.set_contrast_limits(5.0, 50.0)
    assert image.contrast_limits() == (5.0, 50.0)

    image.auto_contrast()

    assert image.contrast_limits() == (0.0, 1.0), "back to the empty default"


def test_setting_a_channels_visibility_to_what_it_already_is_notifies_nobody():
    """A checkbox re-emitting its current state must not repaint the canvas."""
    stack = LayerStack()
    image = stack.add_image(np.zeros((2, 4, 4)), channel_axis=0)
    seen = _events(stack)

    image.set_channel_visible(1, True)
    assert seen == [], "channel 1 was already visible"
    assert image.channel_is_visible(1) is True

    image.set_channel_visible(1, False)
    assert [e.detail for e in seen] == ["channel_visible"]
    assert image.channel_is_visible(1) is False


# -- LabelsLayer ---------------------------------------------------------

def test_reselecting_the_label_already_selected_notifies_nobody():
    """Hovering back over the same object must not repaint the label overlay.

    A viewer sets ``selected_label`` from every mouse-move; the pointer sits
    inside one object for hundreds of events. Only the crossing into a
    different object is a change.
    """
    stack = LayerStack()
    labels = stack.add_labels(np.array([[0, 1], [2, 2]], dtype=np.int32))
    labels.selected_label = 2
    seen = _events(stack)

    labels.selected_label = 2
    assert seen == []
    assert labels.selected_label == 2

    labels.selected_label = 1
    assert [e.detail for e in seen] == ["selected_label"]
    assert labels.selected_label == 1


# -- PointsLayer ---------------------------------------------------------

def test_per_point_sizes_do_not_become_the_layers_default_for_new_points():
    """One size per point sets those points only; a scalar sets the default.

    ``_default_size`` is what the next added point gets. Reading it off an
    array of per-point sizes would make the last-added point's diameter depend
    on the order the array happened to be in.
    """
    layer = PointsLayer(np.array([[0.0, 0.0], [1.0, 1.0]]), size=10.0)

    layer.set_size([4.0, 20.0])
    assert layer.size.tolist() == [4.0, 20.0]

    layer.add([2.0, 2.0])
    assert layer.size.tolist() == [4.0, 20.0, 10.0], "the default is still 10"

    # A scalar, by contrast, does move the default.
    layer.set_size(6.0)
    layer.add([3.0, 3.0])
    assert layer.size.tolist() == [6.0, 6.0, 6.0, 6.0]


# -- Shape ---------------------------------------------------------------

def test_a_rectangle_given_as_four_corners_keeps_them_in_the_order_given():
    """Four corners are taken as-is; only two are expanded.

    A rectangle round-tripped through a saved ROI comes back as its four
    corners. Expanding those as if they were two opposite ones would drop half
    of them, so the four-corner form is passed through unchanged -- which also
    lets a rotated box survive, since nothing re-derives it from a bounding
    box.
    """
    corners = np.array([[0.0, 0.0], [0.0, 10.0], [6.0, 10.0], [6.0, 0.0]])
    shape = Shape(kind="rectangle", data=corners)

    assert shape.data.shape == (4, 2)
    assert np.array_equal(shape.data, corners), "kept, not re-derived"

    # Two corners are still expanded, so the four-corner path is the branch
    # and not the whole method.
    two = Shape(kind="rectangle", data=np.array([[0.0, 0.0], [6.0, 10.0]]))
    assert two.data.shape == (4, 2)

    with pytest.raises(Exception) as excinfo:
        Shape(kind="rectangle",
              data=np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    assert "two opposite corners or four corners, got 3" in str(excinfo.value)


# -- LayerStack ----------------------------------------------------------

def test_selecting_the_layer_that_is_already_selected_emits_nothing():
    """Re-selecting is silent, so a layer list can set selection freely.

    A Qt list view emits ``currentChanged`` when it is rebuilt, with the same
    row it already had. If that round-tripped back as a ``selected`` event the
    model and the widget would ping-pong.
    """
    stack = LayerStack()
    first = stack.add_image(np.zeros((4, 4)), name="first")
    second = stack.add_image(np.zeros((4, 4)), name="second")
    stack.select(second)
    seen = _events(stack)

    assert stack.select(second) is second
    assert seen == [], "already selected"

    assert stack.select("first") is first
    assert [(e.kind, e.layer) for e in seen] == [("selected", first)]


def test_a_click_on_background_labels_falls_through_to_the_layer_below():
    """Picking skips a labels layer where the label is 0 and keeps going down.

    Background is not a hit. A labels overlay covers the whole field, so if
    label 0 counted the layer underneath could never be clicked anywhere
    outside an object -- which is most of a field.
    """
    stack = LayerStack()
    lower = stack.add_labels(np.array([[7, 7], [7, 7]], dtype=np.int32),
                             name="lower")
    upper = stack.add_labels(np.array([[0, 0], [0, 5]], dtype=np.int32),
                             name="upper")
    canvas = Canvas.for_grid(upper.spacing, upper.shape)

    layer, world, value = stack.pick(canvas, 0.0, 0.0)

    assert layer is lower, "the top layer had only background there"
    assert value == 7
    assert world == {"y": 0.0, "x": 0.0}

    # Where the top layer does have an object, it wins.
    assert stack.pick(canvas, 1.0, 1.0)[0] is upper
    assert stack.pick(canvas, 1.0, 1.0)[2] == 5


def test_a_click_that_misses_every_point_falls_through_to_the_layer_below():
    """A points overlay only claims a click inside a point's own disc.

    Centroid markers are a few pixels wide on a field of thousands. If the
    points layer swallowed every click the image beneath it would become
    unclickable the moment centroids were shown.
    """
    stack = LayerStack()
    labels = stack.add_labels(np.array([[3, 3], [3, 3]], dtype=np.int32),
                              name="masks")
    stack.add_points(np.array([[1.0, 1.0]]), size=0.5, name="centroids")
    canvas = Canvas.for_grid(labels.spacing, labels.shape)

    layer, world, value = stack.pick(canvas, 0.0, 0.0)

    assert layer is labels, "the click missed the marker's disc"
    assert value == 3

    # And a click on the marker does reach the points layer, so the miss is a
    # miss and not a points layer that never answers.
    hit_layer, _, index = stack.pick(canvas, 1.0, 1.0)
    assert isinstance(hit_layer, PointsLayer)
    assert index == 0


def test_a_click_on_nothing_at_all_returns_no_layer_but_still_a_world_point():
    """With every layer missing, ``pick`` still reports where the click was."""
    stack = LayerStack()
    stack.add_shapes([Shape(kind="rectangle",
                            data=np.array([[10.0, 10.0], [12.0, 12.0]]))])
    canvas = _canvas()

    layer, world, value = stack.pick(canvas, 0.0, 0.0)

    assert layer is None and value is None
    assert world == {"y": 0.0, "x": 0.0}


# -- Layer ---------------------------------------------------------------

def test_a_layer_reprs_as_its_type_name_and_grid_shape():
    """``repr`` says which subclass, what it is called and how big it is.

    A stack in a traceback is a list of layers; without this each entry is an
    object id, and the commonest layer bug -- two layers on grids that do not
    match -- is invisible in exactly the place it needs to be read.
    """
    image = ImageLayer(np.zeros((5, 6)), name="dapi")
    assert repr(image) == "<ImageLayer 'dapi' (5, 6)>"

    labels = LabelsLayer(np.zeros((5, 6), dtype=np.int32), name="cell_mask")
    assert repr(labels) == "<LabelsLayer 'cell_mask' (5, 6)>"

    shapes = ShapesLayer(name="rois")
    assert repr(shapes) == "<ShapesLayer 'rois' ()>"
