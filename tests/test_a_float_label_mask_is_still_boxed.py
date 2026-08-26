"""Bounding boxes for masks that are not plain integer arrays.

:func:`spacr.measure._label_bounding_boxes` is the optimisation that lets a
per-object loop touch one object's own pixels instead of the whole field. It
is allowed to answer "no boxes" -- every caller then measures over the whole
field and gets the same numbers -- so the only way it can be wrong is by
raising, or by handing an object somebody else's box.

Two inputs that arrive in practice and are neither an integer array nor a
reason to give up:

* an **empty** mask, from a field where nothing segmented;
* a **float** mask whose labels are whole numbers, which is what a mask round
  tripped through a float image, an ``np.where`` or SQLite's REAL affinity
  comes back as.
"""
import numpy as np

from spacr import measure as M


def test_an_empty_mask_has_no_boxes_and_does_not_reduce_over_it():
    """A field where nothing segmented is an ordinary answer, not an error.

    ``arr.max()`` on an empty array raises, so the size check has to come
    first.
    """
    assert M._label_bounding_boxes(np.zeros((0, 0), dtype=np.int32)) == {}
    assert M._label_bounding_boxes(np.zeros((0, 8), dtype=np.int32)) == {}


def test_a_float_mask_with_whole_number_labels_is_boxed_like_an_integer_one():
    """``2.0`` is label two, not a fractional label.

    The boxes are the ones the integer mask gives, pixel for pixel -- the
    point of the cast is that the crop selects exactly the same pixels in the
    same order, so every reduction over them is unchanged.
    """
    integer = np.zeros((32, 32), dtype=np.int32)
    integer[4:9, 6:11] = 1
    integer[20:26, 18:30] = 2
    floating = integer.astype(np.float64)

    boxes = M._label_bounding_boxes(floating)

    assert boxes == M._label_bounding_boxes(integer)
    assert boxes[1] == (slice(4, 9), slice(6, 11))
    assert boxes[2] == (slice(20, 26), slice(18, 30))
    # The box really is the object: nothing outside it carries the label.
    for label, box in boxes.items():
        assert (floating[box] == label).any()
        outside = floating.copy()
        outside[box] = 0
        assert not (outside == label).any()
