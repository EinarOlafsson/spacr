"""Item 288: Plaque mode's flow pictures and numeric settings at their edges.

``plaque_flow_outputs`` pulls the flow picture and the cell-probability map
out of whatever ``model.eval`` returned, for the preview to draw. It keeps
only what it can draw:

* a single flow array rather than a list is still read;
* a flow picture that is not an RGB image, or a probability map that is not
  2-D, is left out rather than drawn wrongly;
* a flows entry that cannot be converted at all gives nothing, and the
  preview carries on.

``_number`` reads a numeric setting and falls back when it is not a number.
"""
from __future__ import annotations

import numpy as np

from spacr import plaque


def test_a_single_flow_array_is_read_as_the_flow_picture():
    rgb = np.full((4, 5, 3), 300.0)
    found = plaque.plaque_flow_outputs((np.zeros((4, 5)), rgb, None))
    assert found["flow_rgb"].shape == (4, 5, 3)
    assert found["flow_rgb"].dtype == np.uint8
    assert int(found["flow_rgb"].max()) == 255
    assert found["cellprob"] is None


def test_what_cannot_be_drawn_is_left_out():
    grey = np.zeros((4, 5))
    cube = np.zeros((2, 4, 5))
    found = plaque.plaque_flow_outputs((None, [grey, None, cube]))
    assert found == {"flow_rgb": None, "cellprob": None}


def test_flows_that_cannot_be_read_give_nothing():
    class Unreadable:
        def __array__(self, *a, **k):
            raise RuntimeError("tensor on a device that went away")

    found = plaque.plaque_flow_outputs((None, [Unreadable()]))
    assert found == {"flow_rgb": None, "cellprob": None}


def test_a_setting_that_is_not_a_number_takes_the_default():
    assert plaque._number({"x": "wide"}, "x", 3.0) == 3.0
    assert plaque._number({"x": [1]}, "x", None) is None
    assert plaque._number({"x": "2.5"}, "x", 3.0) == 2.5
