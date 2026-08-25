"""Two ways a layer model is asked about something that is not there.

A points layer has no grid, so it has no voxel to offer a slice slider; and a
listener that was never subscribed cannot be removed. Both are ordinary states
in a running viewer -- a centroid overlay beside a stack, a panel torn down
twice -- and both must be answered rather than raised on, because they happen
inside paint and teardown paths where an exception is a crash.
"""
from __future__ import annotations

import numpy as np

from spacr.layers import (Canvas, CanvasLink, ImageLayer, LayerStack,
                          OrthoViews, PointsLayer, Spacing)


def _anisotropic_stack():
    """A confocal-shaped volume with a centroid overlay on the same spacing."""
    spacing = Spacing(axes=("z", "y", "x"), scale=(2.0, 0.65, 0.65))
    volume = ImageLayer(np.zeros((5, 10, 10)), name="volume", spacing=spacing)
    points = PointsLayer(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
                         name="centroids", spacing=spacing)
    return LayerStack([volume, points]), spacing


def test_a_gridless_layer_does_not_get_a_vote_on_the_slice_step():
    """A points overlay must not change the z step the slider moves by.

    The step is the finest voxel in the source, and a points layer has no
    voxel at all -- its ``shape`` is ``()``. If it were read as a grid anyway
    the slider would step by whatever fell out of an empty zip, and dragging z
    would stop landing on slices. The overlay is skipped, so the step is still
    the volume's own 2 µm.
    """
    stack, spacing = _anisotropic_stack()
    points = stack["centroids"]
    assert points.shape == (), "a points layer has no grid to measure"

    views = OrthoViews.covering(stack, width=64)

    assert dict(views.steps) == {"z": 2.0, "y": 0.65, "x": 0.65}
    assert views.slider("z")[2] == 2.0
    # And the same volume without the overlay agrees, which is the point:
    # adding centroids to a viewer did not move the slider.
    alone = OrthoViews.covering(LayerStack([ImageLayer(
        np.zeros((5, 10, 10)), name="volume", spacing=spacing)]), width=64)
    assert dict(alone.steps) == dict(views.steps)


def test_unsubscribing_a_listener_that_was_never_there_says_so():
    """``unsubscribe`` reports whether this call is what removed the listener.

    A panel that is closed twice, or one that unsubscribes in both a
    ``closeEvent`` and a destructor, must not raise on the second pass. The
    return value distinguishes the two, so a caller that cares can tell.
    """
    link = CanvasLink({"a": Canvas(origin=(0.0, 0.0), step=(1.0, 1.0),
                                   shape=(4, 4), axes=("y", "x"))})
    seen = []

    assert link.unsubscribe(seen.append) is False, "never subscribed"

    link.subscribe(seen.append)
    assert link.unsubscribe(seen.append) is True, "this call removed it"
    assert link.unsubscribe(seen.append) is False, "and it is gone now"

    link.pan(1.0, 1.0)
    assert seen == [], "a removed listener hears nothing"
