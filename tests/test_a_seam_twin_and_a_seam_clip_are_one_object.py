"""Two windows' views of one nucleus are one object, even when they disagree.

Found on well A1, the first real Phase B run (372 PART 14-L, V11a/V11b):

* 1,048 SEAM DUPLICATES. One nucleus seen whole by two windows, centroids
  within 3 px (736 within 1 px), but Cellpose drew it smaller in one of them
  because the window edge cut its context: area ratio median 0.74 against
  ``DEFAULT_AREA_RATIO`` 0.8, so both were kept and numbered.
* 10,182 REFUSED CLIPS under ``number(strict=True)``. A clip -- 9 x 6 px,
  area 38 -- of a nucleus the neighbouring window saw whole (area 96), whose
  centroid the cut had moved 3.1 px from the whole one's, against a 3.0 px
  per-axis tolerance. The refusal told the operator to raise the overlap,
  which changes nothing.

Every fixture here reproduces one of those geometries with the measured
numbers, and each guard beside it is the case a looser rule would get wrong.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from spacr.ops_compose import Window, windows_over
from spacr.ops_objects import (DEFAULT_AREA_RATIO, DEFAULT_CENTROID_TOLERANCE,
                               ObjectsError, WindowObject, number,
                               objects_in_window, segment_windows, sew)

CANVAS = (64, 96)
LEFT = Window(top=0, left=0, height=64, width=64)
RIGHT = Window(top=0, left=32, height=64, width=64)


def _disc(shape, centre, radius_squared):
    """A digital disc, centred anywhere, as a boolean mask."""
    ys, xs = np.mgrid[:shape[0], :shape[1]]
    return (ys - centre[0]) ** 2 + (xs - centre[1]) ** 2 <= radius_squared


def _in_window(window, mask_in_well):
    """The well-frame mask cropped to one window, as a one-label image."""
    crop = mask_in_well[window.top:window.bottom, window.left:window.right]
    return crop.astype(np.int32)


def test_a_nucleus_the_window_edge_shrank_is_still_one_object():
    """V11a. Both complete, 1 px apart, area ratio 0.73 -- one nucleus."""
    whole = _disc(CANVAS, (32.5, 48), 31)
    shrunk = _disc(CANVAS, (32.5, 47), 22)
    right = objects_in_window(RIGHT, _in_window(RIGHT, whole), canvas=CANVAS)
    left = objects_in_window(LEFT, _in_window(LEFT, shrunk), canvas=CANVAS)
    assert len(left) == len(right) == 1
    a, b = left[0], right[0]
    assert not a.clipped and not b.clipped
    assert b.area == 96 and a.area == 70
    assert a.area / b.area < DEFAULT_AREA_RATIO
    assert math.hypot(a.centroid_y - b.centroid_y,
                      a.centroid_x - b.centroid_x) == pytest.approx(1.0)

    objects = number(sew(list(left) + list(right)))

    assert len(objects) == 1
    assert objects[0].area == 96
    assert objects[0].n_observations == 2


def _crescent_with_a_nucleus_in_it(shape, centre):
    """Two touching nuclei whose centroids nearly coincide.

    A ring open on one side, and a small round nucleus in its hole. Their
    masks are disjoint, but the ring's centroid sits about a pixel from the
    small one's -- the one arrangement where a centroid test cannot tell two
    nuclei apart.
    """
    ys, xs = np.mgrid[:shape[0], :shape[1]]
    distance2 = (ys - centre[0]) ** 2 + (xs - centre[1]) ** 2
    small = distance2 <= 4
    ring = (distance2 >= 9) & (distance2 <= 36) & ~(
        (xs > centre[1] + 1) & (np.abs(ys - centre[0]) <= 2))
    labels = np.zeros(shape, np.int32)
    labels[ring] = 1
    labels[small] = 2
    return labels


def test_two_touching_nuclei_seen_by_both_windows_stay_two():
    """The guard for the disc rule: labels of one window are never merged.

    The ring seen by one window and the small nucleus seen by the other are
    a cross-window pair inside the small one's equivalent disc, so the disc
    rule alone would sew them; what stops it is that each group would then
    hold two labels of the same window.
    """
    truth = _crescent_with_a_nucleus_in_it(CANVAS, (32, 48))
    seen = (objects_in_window(LEFT, truth[:, :64], canvas=CANVAS)
            + objects_in_window(RIGHT, truth[:, 32:], canvas=CANVAS))
    assert len(seen) == 4 and not any(one.clipped for one in seen)
    ring = [one for one in seen if one.area > 50]
    small = [one for one in seen if one.area < 50]
    offset = math.hypot(ring[0].centroid_y - small[1].centroid_y,
                        ring[0].centroid_x - small[1].centroid_x)
    assert offset <= math.sqrt(small[1].area / math.pi)

    objects = number(sew(seen))

    assert len(objects) == 2
    assert sorted(one.area for one in objects) == sorted(
        {one.area for one in seen})
    assert all(one.n_observations == 2 for one in objects)


def test_a_tiny_object_beside_a_big_one_in_one_window_stays_two():
    """Two labels of one label image are two objects, whatever their centroids."""
    window = Window(0, 0, 100, 100)
    big = WindowObject(window=window, label=1, centroid_y=50.0,
                       centroid_x=50.0, area=300, bbox=(40, 40, 60, 60))
    tiny = WindowObject(window=window, label=2, centroid_y=50.0,
                        centroid_x=51.0, area=6, bbox=(49, 50, 51, 52))
    assert 1.0 <= math.sqrt(tiny.area / math.pi)

    assert len(sew([big, tiny])) == 2
    assert len(number(sew([tiny, big]))) == 2


def _measured_clip_and_twin():
    """V11b's geometry exactly: a 9 x 6 clip of area 38, its twin of area 96.

    The clip is what the LEFT window's segmenter drew at its right edge; the
    twin is the whole nucleus the RIGHT window saw, 3.1 px away and more
    than 3 px along the axis the cut ran across.
    """
    clip = np.zeros(CANVAS, np.int32)
    for column, height in zip(range(58, 64), (4, 5, 6, 7, 7, 9)):
        start = 32 - height // 2
        clip[start:start + height, column] = 1
    twin = _disc(CANVAS, (31.5, 64), 31)
    left = objects_in_window(LEFT, _in_window(LEFT, clip), canvas=CANVAS)
    right = objects_in_window(RIGHT, _in_window(RIGHT, twin), canvas=CANVAS)
    return left[0], right[0]


def test_a_clip_the_cut_moved_past_the_tolerance_joins_its_whole_twin():
    """V11b. Numbered without refusal, as the nucleus the other window saw."""
    clip, twin = _measured_clip_and_twin()
    assert clip.clipped and not twin.clipped
    assert clip.area == 38 and twin.area == 96
    assert (clip.bbox[2] - clip.bbox[0] + 1,
            clip.bbox[3] - clip.bbox[1] + 1) == (9, 6)
    assert math.hypot(clip.centroid_y - twin.centroid_y,
                      clip.centroid_x - twin.centroid_x) == pytest.approx(
                          3.1, abs=0.02)
    assert abs(clip.centroid_x - twin.centroid_x) > DEFAULT_CENTROID_TOLERANCE

    objects = number(sew([clip, twin]), strict=True)

    assert len(objects) == 1
    assert objects[0].area == 96
    assert objects[0].window == RIGHT.offset()
    assert objects[0].n_observations == 2


def test_a_clip_joins_one_nucleus_and_cannot_chain_two():
    """A clip matched to two whole nuclei must not make them one object.

    Two small nuclei seen whole by one window, and a clip from its neighbour
    that lies within reach of both. Sewing by single linkage through the
    clip would number two cells once.
    """
    whole = Window(0, 32, 64, 64)
    clip_window = Window(0, 0, 64, 64)
    first = WindowObject(window=whole, label=1, centroid_y=32.0,
                         centroid_x=50.0, area=29, bbox=(29, 47, 35, 53))
    second = WindowObject(window=whole, label=2, centroid_y=32.0,
                          centroid_x=56.0, area=29, bbox=(29, 53, 35, 59))
    clip = WindowObject(window=clip_window, label=1, centroid_y=32.0,
                        centroid_x=53.0, area=8, bbox=(31, 52, 33, 55),
                        clipped=True)

    objects = number(sew([first, clip, second]))

    assert len(objects) == 2
    assert sum(one.n_observations for one in objects) == 3


def test_a_clip_beside_a_whole_neighbour_it_is_not_part_of_is_still_refused():
    """A clip is joined to a whole nucleus it lies in, not to one it lies beside."""
    whole = objects_in_window(RIGHT, _in_window(RIGHT, _disc(
        CANVAS, (20.5, 60), 31)), canvas=CANVAS)[0]
    clip = WindowObject(window=LEFT, label=5, centroid_y=31.0,
                        centroid_x=61.0, area=30, bbox=(27, 58, 35, 63),
                        clipped=True)

    with pytest.raises(ObjectsError):
        number(sew([whole, clip]))


def _run(centres, *, size, overlap, radius, shape, strict=True):
    """Plant discs, tile, segment, sew and number, as the pipeline does."""
    truth = np.zeros(shape, np.int32)
    for index, centre in enumerate(centres, start=1):
        truth[_disc(shape, centre, radius ** 2)] = index

    def segment(window):
        """Crop the truth and relabel it the way a segmenter would."""
        crop = truth[window.top:window.bottom, window.left:window.right]
        out = np.zeros_like(crop)
        for new, value in enumerate([v for v in np.unique(crop) if v > 0],
                                    start=1):
            out[crop == value] = new
        return out

    windows = list(windows_over(shape, size=size, overlap=overlap))
    seen = segment_windows(windows, segment, canvas=shape)
    return number(sew(seen), strict=strict)


def test_the_refusal_states_the_extent_and_both_remedies():
    """The message names what it knows -- the extent -- and both ways out.

    `number` is not told the overlap, so it cannot say which remedy applies;
    it says how large the object is and what each remedy is, including the
    non-strict run that drops it.
    """
    with pytest.raises(ObjectsError) as refusal:
        _run([(100, 200)], size=128, overlap=16, radius=40, shape=(600, 600))
    message = str(refusal.value)
    assert "81 x 81 px" in message
    assert "overlap has to exceed" in message
    assert "strict=False" in message


def test_an_object_across_a_whole_window_is_told_the_windows_are_too_small():
    """No overlap fixes an object that spans a window edge to edge."""
    with pytest.raises(ObjectsError) as refusal:
        _run([(150, 150)], size=64, overlap=16, radius=60, shape=(300, 300))
    message = str(refusal.value)
    assert "no overlap can fix" in message
    assert "64 px" in message


def test_non_strict_drops_what_no_window_saw_whole_and_says_so(caplog):
    """strict=False numbers nothing truncated, and the log carries the count."""
    with caplog.at_level(logging.WARNING, logger="spacr.ops_objects"):
        objects = _run([(100, 200), (400, 400)], size=128, overlap=16,
                       radius=40, shape=(600, 600), strict=False)
    assert objects == () or all(one.area > 3000 for one in objects)
    assert not any(one.centroid_y < 200 for one in objects)
    dropped = [record for record in caplog.records
               if "dropped" in record.getMessage()]
    assert len(dropped) == 1
    assert "1 group" in dropped[0].getMessage()


def test_a_tie_on_area_is_broken_by_window_then_label_not_by_input_order():
    """The kept observation decides the centroid, so it cannot be arrival order."""
    a = WindowObject(window=LEFT, label=7, centroid_y=32.0, centroid_x=48.0,
                     area=96, bbox=(27, 43, 37, 53))
    b = WindowObject(window=RIGHT, label=3, centroid_y=32.0, centroid_x=48.4,
                     area=96, bbox=(27, 43, 37, 53))

    one_way = number(sew([a, b]))
    other_way = number(sew([b, a]))
    by_hand = number([(b, a)])

    assert [one.row() for one in one_way] == [one.row() for one in other_way]
    assert [one.row() for one in by_hand] == [one.row() for one in one_way]
    assert one_way[0].window == LEFT.offset()


def test_two_objects_on_one_rounded_centroid_get_the_same_ids_every_time():
    """Raster order ties on the rounded centroid; the tie is broken by window."""
    first = (WindowObject(window=RIGHT, label=2, centroid_y=10.0001,
                          centroid_x=40.0, area=50, bbox=(6, 36, 14, 44)),)
    second = (WindowObject(window=LEFT, label=9, centroid_y=10.0002,
                           centroid_x=40.0, area=60, bbox=(6, 36, 14, 44)),)

    one_way = number([first, second])
    other_way = number([second, first])

    assert [one.row() for one in one_way] == [one.row() for one in other_way]
    assert one_way[0].window == LEFT.offset()
