"""372 B2-B4: segment window by window, sew the seams, number once.

Every test here plants its own truth -- nuclei at known centres, on a canvas
of known size, tiled by the real `windows_over` -- so the assertions are
against a count and a position that were decided before the code ran, not
against whatever the code happened to produce.

The case that matters most is the one a naive implementation gets wrong in a
way nothing downstream would notice: a nucleus sitting in a window overlap is
segmented TWICE, and emitting it twice would inflate the object count, give
two ids to one cell, and make every per-object measurement downstream double
count it. `test_a_nucleus_in_the_overlap_is_one_object_not_two` is that case.
"""

import numpy as np
import pytest

from spacr.ops_compose import Window, windows_over
from spacr.ops_objects import (
    DEFAULT_CENTROID_TOLERANCE, ObjectsError, PlateObject, WindowObject,
    number, objects_frame, objects_in_window, segment_windows, sew,
)

CANVAS = (600, 600)


def _disc(canvas_shape, centre, radius):
    """A filled circle on a boolean canvas -- one planted nucleus."""
    ys, xs = np.ogrid[:canvas_shape[0], :canvas_shape[1]]
    cy, cx = centre
    return (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2


def _planted(centres, radius=8, shape=CANVAS):
    """A whole-canvas label image with one label per centre."""
    labels = np.zeros(shape, dtype=np.int32)
    for index, centre in enumerate(centres, start=1):
        labels[_disc(shape, centre, radius)] = index
    return labels


def _segmenter(truth):
    """A segmenter that crops planted truth and relabels it per window.

    Relabelling is what a real segmenter does -- it has no idea what the
    neighbouring window called anything -- and it is exactly the condition
    that makes B4's global numbering necessary.
    """
    def segment(window: Window):
        crop = truth[window.top:window.bottom,
                     window.left:window.left + window.width]
        out = np.zeros_like(crop)
        for new, value in enumerate(
                [v for v in np.unique(crop) if v > 0], start=1):
            out[crop == value] = new
        return out
    return segment


def _run(centres, *, size=256, overlap=64, radius=8, shape=CANVAS,
         strict=True):
    truth = _planted(centres, radius=radius, shape=shape)
    windows = list(windows_over(shape, size=size, overlap=overlap))
    seen = segment_windows(windows, _segmenter(truth), canvas=shape)
    return number(sew(seen), strict=strict), windows, seen


def test_isolated_nuclei_are_counted_once_each():
    """The baseline: nothing near a seam, so nothing to sew."""
    centres = [(100, 100), (100, 400), (400, 100), (400, 400)]
    objects, _, _ = _run(centres)

    assert len(objects) == len(centres)
    assert [one.object_id for one in objects] == [1, 2, 3, 4]


def test_a_nucleus_in_the_overlap_is_one_object_not_two():
    """THE case. Two windows segment it; it is still one cell.

    Emitting it twice would inflate the count, give one cell two ids, and
    double count every measurement Phase C makes against those ids -- none of
    which looks like an error downstream. It just looks like more cells.
    """
    # Windows of 256 with 64 overlap start at x = 0, 192, 384, so columns
    # 192-255 belong to two windows. x = 220 with radius 8 spans 212-228:
    # comfortably inside that band and clear of both windows' own edges, so
    # BOTH see the whole nucleus rather than one seeing a truncated arc.
    centres = [(100, 220)]
    objects, windows, seen = _run(centres)

    in_overlap = [one for one in seen if not one.clipped]
    assert len(in_overlap) >= 2, "the fixture did not place it in an overlap"
    assert len(objects) == 1
    assert objects[0].n_observations >= 2


def test_the_surviving_centroid_is_the_planted_one():
    """Sewing must not move the object it kept.

    The centroid is half the join key, so a sew that returned the mean of two
    observations -- or the clipped one -- would put every later phase's
    sampling coordinates slightly off the nucleus.
    """
    centres = [(100, 220)]
    objects, _, _ = _run(centres)

    assert objects[0].centroid_y == pytest.approx(100.0, abs=0.5)
    assert objects[0].centroid_x == pytest.approx(220.0, abs=0.5)


def test_every_planted_nucleus_survives_a_full_tiling():
    """A grid that puts nuclei in interiors, seams and corners at once."""
    centres = [(y, x) for y in range(60, 560, 70) for x in range(60, 560, 70)]
    objects, _, _ = _run(centres)

    assert len(objects) == len(centres)
    found = {(round(one.centroid_y), round(one.centroid_x)) for one in objects}
    assert found == {(y, x) for y, x in centres}


def test_ids_are_raster_order_in_the_well_frame():
    """B4's ordering, which is what makes an id reproducible.

    Ordering by anything a scheduler can vary -- window completion order, a
    dict's iteration -- would give the same data different ids on different
    runs, and nothing downstream could be compared between them.
    """
    centres = [(400, 400), (100, 100), (400, 100), (100, 400)]
    objects, _, _ = _run(centres)

    ordered = [(round(one.centroid_y), round(one.centroid_x))
               for one in objects]
    assert ordered == sorted(ordered)
    assert ordered[0] == (100, 100)
    assert ordered[-1] == (400, 400)


def test_the_same_data_numbered_twice_gives_the_same_ids():
    """Determinism, asserted rather than assumed."""
    centres = [(y, x) for y in range(60, 560, 90) for x in range(60, 560, 90)]
    first, _, _ = _run(centres)
    second, _, _ = _run(centres)

    assert [one.row() for one in first] == [one.row() for one in second]


def test_ids_start_at_one_because_zero_is_background():
    """An object numbered 0 is invisible to any mask comparison."""
    objects, _, _ = _run([(100, 100), (300, 300)])
    assert min(one.object_id for one in objects) == 1


def test_an_object_larger_than_the_overlap_is_refused_not_fragmented():
    """The failure that must not be silent.

    `windows_over`'s contract is that the overlap exceeds the largest object.
    When it does not, every window clips the object, and emitting the pieces
    would count one nucleus as several small ones -- which looks exactly like
    a real result. So it raises, and says what to change.
    """
    # radius 40 -> 80 px across, with an overlap of only 16.
    with pytest.raises(ObjectsError, match="overlap has to exceed"):
        _run([(100, 200)], size=128, overlap=16, radius=40)


def test_the_refusal_can_be_turned_off_to_inspect_a_bad_run():
    """Non-strict emits what it found, having been asked to."""
    objects, _, _ = _run([(100, 200)], size=128, overlap=16, radius=40,
                         strict=False)
    assert objects == () or all(isinstance(o, PlateObject) for o in objects)


def test_a_clipped_observation_never_wins_over_a_complete_one():
    """The complete mask carries the true area; the clipped one does not."""
    window = Window(top=0, left=0, height=100, width=100)
    truth = _planted([(50, 95)], radius=10, shape=(100, 200))
    partial = objects_in_window(window, truth[:100, :100],
                                canvas=(100, 200))
    whole = objects_in_window(Window(top=0, left=60, height=100, width=100),
                              truth[:100, 60:160], canvas=(100, 200))

    assert partial[0].clipped is True
    assert whole[0].clipped is False
    kept = number(sew(list(partial) + list(whole)))
    assert len(kept) == 1
    assert kept[0].area == whole[0].area


def test_an_object_at_the_canvas_edge_is_complete_not_clipped():
    """The well's own edge is where an object genuinely ends.

    Treating it as a seam would refuse every nucleus touching the outside of
    the well -- a whole rim of real cells -- as 'larger than the overlap'.
    """
    window = Window(top=0, left=0, height=100, width=100)
    truth = _planted([(4, 50)], radius=10, shape=(100, 100))
    found = objects_in_window(window, truth, canvas=(100, 100))

    assert len(found) == 1
    assert found[0].clipped is False


def test_two_nuclei_close_together_are_not_merged():
    """The centroid test alone would be enough to marry these. It is not used
    alone: the areas must agree too, so a false match needs two coincidences.
    """
    near = WindowObject(window=Window(0, 0, 100, 100), label=1,
                        centroid_y=50.0, centroid_x=50.0, area=300,
                        bbox=(40, 40, 60, 60))
    other = WindowObject(window=Window(0, 0, 100, 100), label=2,
                         centroid_y=50.0 + DEFAULT_CENTROID_TOLERANCE - 0.5,
                         centroid_x=50.0, area=80, bbox=(45, 40, 55, 60))

    assert len(sew([near, other])) == 2


def test_a_window_sized_label_image_is_required():
    """A mismatch would offset every centroid by the difference."""
    window = Window(top=10, left=10, height=50, width=50)
    with pytest.raises(ObjectsError, match="every centroid"):
        objects_in_window(window, np.zeros((40, 40), dtype=np.int32))


def test_the_table_carries_the_columns_phase_d_specifies():
    """Written from these without a translation layer, per Phase D."""
    objects, _, _ = _run([(100, 100), (300, 300)])
    frame = objects_frame(objects)

    assert list(frame["object_id"]) == [1, 2]
    for column in ("object_id", "centroid_x", "centroid_y", "area",
                   "bbox_top", "bbox_left", "bbox_bottom", "bbox_right"):
        assert column in frame.columns


def test_an_empty_window_contributes_nothing():
    """Most windows of a sparse well are empty; that is not a failure."""
    window = Window(top=0, left=0, height=64, width=64)
    assert objects_in_window(window, np.zeros((64, 64), dtype=np.int32)) == ()
