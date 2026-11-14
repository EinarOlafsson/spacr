"""Picture settings translate one vocabulary into another without guessing.

The annotator names a setting one way and the crop layer another, and the two
disagree about what a channel is: the annotator's ``channels`` are colour
planes of a picture already drawn, while the crop layer's are which source
array planes to cut. Every branch here is a place where a value that cannot
be translated must be left alone rather than coerced -- an index read as a
colour, or a colour read as an index, produces a crop that looks plausible
and is wrong.
"""
from __future__ import annotations

from spacr import picture_settings as ps


# -- which settings apply ----------------------------------------------------

def test_a_threshold_belongs_only_to_the_picker_that_computes_one():
    """The other pickers produce no probability for a threshold to cut."""
    assert ps.applies_to_picking("picking_threshold", "attributed") is True
    assert ps.applies_to_picking("picking_threshold", "rank") is False


def test_a_setting_this_module_never_heard_of_still_applies():
    """Hiding the unknown would hide every setting added after this one."""
    assert ps.applies_to_picking("some_new_setting", "rank") is True


# -- what shape the cut can take ---------------------------------------------

def test_something_without_settings_cannot_be_asked_about_its_cut():
    """A caller holding a list rather than a settings dict gets False."""
    assert ps.bounding_box_only(["crop_source", "merged"]) is False


def test_the_database_route_can_only_cut_a_box():
    """A coordinate column has no outline to follow."""
    assert ps.bounding_box_only({"crop_source": ps.STREAM_FROM_DB}) is True
