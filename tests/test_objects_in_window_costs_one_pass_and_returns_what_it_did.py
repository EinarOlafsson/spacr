"""`objects_in_window` reads a window once, and returns what it always did.

MEASURED BEFORE THE FIX (372 PART 14-L, section 3, synthetic discs, CPU):

    512 px    131 labels    0.15 s    1.1 ms/label
    1024 px   524 labels    3.0 s     5.7 ms/label
    2048 px   2095 labels   38.9 s   18.6 ms/label

The implementation did ``labels == value`` over the WHOLE window once per
label, so its cost was labels x pixels: a 4096 px window was minutes of
bookkeeping before any model ran. The cost per label grew with the window,
which is the signature the cost test below looks for -- a ratio between two
window sizes at the same object density, so machine load cancels.

THE OUTPUT IS A JOIN KEY'S INPUT, so a faster implementation has to be the
same function. The implementation it replaced is copied below as
``_reference_objects_in_window`` and compared field by field, including the
Python types, on random planted labels: touching objects, objects cut by
every window edge, one label in two pieces, sparse label values, and the
dtypes a segmenter hands back.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from spacr.ops_compose import Window
from spacr.ops_objects import WindowObject, objects_in_window


def _reference_objects_in_window(window, labels, *, canvas=None):
    """The implementation before the fix, verbatim apart from the error path.

    One boolean mask per label over the whole window. Kept here, not in the
    package, so the comparison cannot drift with the code it checks.
    """
    labels = np.asarray(labels)
    found = []
    values = np.unique(labels)
    values = values[values > 0]
    if values.size == 0:
        return ()

    height, width = labels.shape
    canvas_height, canvas_width = canvas if canvas else (None, None)
    for value in values:
        mask = labels == value
        rows, cols = np.nonzero(mask)
        top, bottom = int(rows.min()), int(rows.max())
        left, right = int(cols.min()), int(cols.max())
        touches = []
        if top == 0:
            touches.append(window.top > 0)
        if left == 0:
            touches.append(window.left > 0)
        if bottom == height - 1:
            touches.append(canvas_height is None
                           or window.top + height < canvas_height)
        if right == width - 1:
            touches.append(canvas_width is None
                           or window.left + width < canvas_width)
        found.append(WindowObject(
            window=window, label=int(value),
            centroid_y=float(rows.mean()) + window.top,
            centroid_x=float(cols.mean()) + window.left,
            area=int(mask.sum()),
            bbox=(top + window.top, left + window.left,
                  bottom + window.top, right + window.left),
            clipped=any(touches)))
    return tuple(found)


def _planted(rng, shape):
    """A random label image with every awkward case a segmenter produces."""
    height, width = shape
    labels = np.zeros(shape, dtype=np.int64)
    ys, xs = np.mgrid[:height, :width]
    count = int(rng.integers(1, 30))
    for value in rng.choice(np.arange(1, 60000), size=count, replace=False):
        kind = int(rng.integers(4))
        cy = rng.uniform(-4, height + 4)
        cx = rng.uniform(-4, width + 4)
        if kind == 0:
            radius = rng.uniform(0.5, 9.0)
            mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2
        elif kind == 1:
            half_h, half_w = rng.uniform(0.0, 6.0, 2)
            mask = (np.abs(ys - cy) <= half_h) & (np.abs(xs - cx) <= half_w)
        elif kind == 2:
            radius = rng.uniform(1.0, 4.0)
            mask = (((ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2)
                    | ((ys - (height - 1 - cy)) ** 2 + (xs - cx) ** 2
                       <= radius ** 2))
        else:
            radius = rng.uniform(2.0, 10.0)
            mask = (((ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2)
                    & (rng.random(shape) < 0.6))
        labels[mask] = int(value)
    if rng.random() < 0.5:
        top = int(rng.integers(0, height))
        left = int(rng.integers(0, width))
        tile = int(rng.integers(1, 5))
        patch = labels[top:top + 4 * tile, left:left + 4 * tile]
        ty, tx = np.mgrid[:patch.shape[0], :patch.shape[1]]
        patch[...] = 60001 + (ty // tile) * 4 + (tx // tile)
    if rng.random() < 0.2:
        labels[rng.random(shape) < 0.02] = -3
    return labels


def _canvas(rng, window):
    """None, the window's own far edge, or a canvas beyond or short of it."""
    choice = int(rng.integers(4))
    if choice == 0:
        return None
    if choice == 1:
        return (window.bottom, window.right)
    if choice == 2:
        return (window.bottom + int(rng.integers(1, 50)),
                window.right + int(rng.integers(1, 50)))
    return (window.bottom - 1, window.right + 1)


def _assert_identical(found, expected):
    """Same tuple, same order, same values and the same Python types."""
    assert found == expected
    for one, two in zip(found, expected):
        assert type(one.label) is int and type(two.label) is int
        assert type(one.area) is int
        assert type(one.centroid_y) is float and type(one.centroid_x) is float
        assert type(one.clipped) is bool
        assert all(type(value) is int for value in one.bbox)
        assert repr(one) == repr(two)


@pytest.mark.parametrize("dtype", [np.int32, np.uint16, np.int64, np.uint8,
                                   np.float32, bool])
def test_every_field_matches_the_implementation_it_replaced(dtype):
    """Random planted labels, every dtype a segmenter returns, 40 windows each."""
    rng = np.random.default_rng(372 + np.dtype(dtype).num)
    compared = 0
    for _ in range(40):
        shape = (int(rng.integers(1, 70)), int(rng.integers(1, 70)))
        window = Window(top=int(rng.choice([0, rng.integers(1, 200)])),
                        left=int(rng.choice([0, rng.integers(1, 200)])),
                        height=shape[0], width=shape[1])
        labels = _planted(rng, shape).astype(dtype)
        canvas = _canvas(rng, window)
        expected = _reference_objects_in_window(window, labels, canvas=canvas)
        _assert_identical(objects_in_window(window, labels, canvas=canvas),
                          expected)
        compared += len(expected)
    assert compared >= 40


def test_the_edge_cases_match_too():
    """An empty window, one label filling it, and a pixel in every corner."""
    window = Window(top=5, left=0, height=9, width=7)
    cases = [np.zeros((9, 7), np.int32), np.full((9, 7), 4, np.int32)]
    corners = np.zeros((9, 7), np.int32)
    corners[0, 0], corners[0, 6], corners[8, 0], corners[8, 6] = 1, 2, 3, 4
    cases.append(corners)
    cases.append(np.arange(63, dtype=np.int32).reshape(9, 7))
    for labels in cases:
        for canvas in (None, (14, 7), (40, 40)):
            _assert_identical(
                objects_in_window(window, labels, canvas=canvas),
                _reference_objects_in_window(window, labels, canvas=canvas))


def test_a_non_contiguous_view_is_read_in_raster_order():
    """A transposed or strided view reaches the function as often as a copy."""
    rng = np.random.default_rng(7)
    base = _planted(rng, (60, 50)).astype(np.int32)
    view = base.T[::1, ::-1]
    window = Window(top=30, left=40, height=view.shape[0], width=view.shape[1])
    _assert_identical(objects_in_window(window, view),
                      _reference_objects_in_window(window, view))


def _grid(size, spacing=32, radius=5.5, seed=0):
    """Discs on a jittered grid: the label count scales with the area exactly."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((size, size), np.int32)
    reach = int(np.ceil(radius))
    dy, dx = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    disc = dy ** 2 + dx ** 2 <= radius ** 2
    value = 0
    for cy in range(spacing // 2, size, spacing):
        for cx in range(spacing // 2, size, spacing):
            value += 1
            jy, jx = rng.integers(-6, 7, 2)
            yy, xx = dy[disc] + cy + jy, dx[disc] + cx + jx
            inside = (yy >= 0) & (yy < size) & (xx >= 0) & (xx < size)
            labels[yy[inside], xx[inside]] = value
    return labels, value


def test_the_cost_per_label_does_not_grow_with_the_window():
    """Sixteen times the pixels at the same density is sixteen times the work.

    Before the fix the cost per label rose with the window's pixel count
    (1.1 -> 18.6 ms/label from 512 to 2048 px), so this ratio was ~16 or
    more. The two sizes are timed alternately and the best of three kept,
    so a busy machine slows both rather than one.
    """
    small, small_count = _grid(384)
    large, large_count = _grid(1536)
    assert large_count == 16 * small_count
    canvas = (1 << 20, 1 << 20)
    best = {384: float("inf"), 1536: float("inf")}
    for _ in range(3):
        for size, labels in ((384, small), (1536, large)):
            window = Window(top=512, left=512, height=size, width=size)
            start = time.perf_counter()
            found = objects_in_window(window, labels, canvas=canvas)
            best[size] = min(best[size], time.perf_counter() - start)
            assert len(found) == (small_count if size == 384 else large_count)
    per_label_small = best[384] / small_count
    per_label_large = best[1536] / large_count
    assert per_label_large / per_label_small < 4.0, (
        f"{1e3 * per_label_small:.3f} ms/label at 384 px, "
        f"{1e3 * per_label_large:.3f} ms/label at 1536 px")
