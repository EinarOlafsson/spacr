"""What an ROI refuses to be, and the sentence it describes itself with.

An ROI decides which objects a run measures, so every refusal here prevents a
measurement over a region that is not a region -- and the summary line is what
tells the user, in a status bar, which region they actually applied.
"""
from __future__ import annotations

import numpy as np
import pytest


def _square(**changes):
    from spacr.roi import RegionOfInterest

    fields = dict(kind="polygon",
                  vertices=[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    fields.update(changes)
    return RegionOfInterest(**fields)


def test_a_square_polygon_is_accepted():
    """The baseline the refusals are measured against."""
    assert _square().kind == "polygon"


def test_vertices_of_the_wrong_shape_are_refused_with_the_shape_named():
    """A 1-D or (M, 3) array is not world points.

    The message carries the shape it got, because "an ROI needs an (M, 2)
    array" alone does not tell the user what they passed.
    """
    from spacr.roi import RoiError

    with pytest.raises(RoiError) as excinfo:
        _square(vertices=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    assert "(M, 2)" in str(excinfo.value)
    assert "(2, 3)" in str(excinfo.value)


def test_fewer_than_two_points_is_refused_with_the_count():
    """One point is a click, not a region."""
    from spacr.roi import RoiError

    with pytest.raises(RoiError) as excinfo:
        _square(kind="rectangle", vertices=[[0.0, 0.0]])

    assert "at least two points" in str(excinfo.value)
    assert "got 1" in str(excinfo.value)


def test_a_two_point_polygon_is_refused():
    """A polygon needs three: two points enclose no area.

    The rectangle above is legal with two -- they are opposite corners -- so
    this rule is per kind rather than global, which is why it is a separate
    check and a separate test.
    """
    from spacr.roi import RoiError

    with pytest.raises(RoiError, match="three points"):
        _square(vertices=[[0.0, 0.0], [4.0, 4.0]])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_vertex_is_refused(bad):
    """NaN and infinity, which is what an unparsed coordinate becomes.

    A NaN vertex makes every containment test False, so the ROI would measure
    nothing and report success -- the silent failure the check exists for.
    """
    from spacr.roi import RoiError

    with pytest.raises(RoiError, match="finite"):
        _square(vertices=[[0.0, 0.0], [4.0, 0.0], [bad, 4.0]])


# ---------------------------------------------------------------------------
# RoiSet.describe
# ---------------------------------------------------------------------------

def _set(**changes):
    from spacr.roi import RoiSet

    fields = dict(fields={"*": (_square(),)})
    fields.update(changes)
    return RoiSet(**fields)


def test_a_default_only_set_says_every_field():
    """The ``every field`` wording, which is what ANY_FIELD alone means."""
    text = _set().describe()

    assert "every field" in text
    assert "plus a default" not in text


def test_named_fields_are_counted():
    """The count, so a user can tell one field from four hundred."""
    text = _set(fields={"plate1_A01_F001": (_square(),),
                        "plate1_A02_F001": (_square(),)}).describe()

    assert "2 field(s)" in text
    assert "plus a default" not in text


def test_named_fields_plus_a_default_say_both():
    """The clause that is only added when BOTH are present.

    Saying "2 field(s)" alone would hide that every other field is also being
    measured, which is the difference between a targeted run and a whole-plate
    one.
    """
    from spacr.roi import ANY_FIELD

    text = _set(fields={"plate1_A01_F001": (_square(),),
                        ANY_FIELD: (_square(),)}).describe()

    assert "1 field(s)" in text
    assert "plus a default for the rest" in text


def test_an_inverted_set_says_outside():
    """The word that reverses the meaning of the whole run."""
    assert "measuring outside" in _set(invert=True).describe()
    assert "measuring inside" in _set(invert=False).describe()


def test_the_overlap_rule_states_its_threshold():
    """``centroid`` needs no number and ``overlap`` does.

    A run described as "overlap rule" without the fraction cannot be
    reproduced from its own log.
    """
    assert "centroid rule" in _set(mode="centroid").describe()

    text = _set(mode="overlap", min_overlap=0.25).describe()
    assert "overlap rule at 25%" in text
