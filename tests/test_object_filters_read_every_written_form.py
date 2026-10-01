"""Object filter lists as a settings file or a hand-typed field writes them.

Item 511 made object filters any scikit-image regionprop. A settings file
stores the list as text, and a person types it; these cases are the forms
the list arrives in and the ways a wrong one is refused, plus the engine's
own refusals for measurements that cannot be judged.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt import mask_engine as me


def _two_objects():
    labels = np.zeros((12, 12), np.int32)
    labels[1:4, 1:4] = 1
    labels[6:11, 6:11] = 2
    return labels


def test_empty_and_blank_text_are_no_filters():
    assert me.normalise_filters(None) == []
    assert me.normalise_filters("   ") == []


def test_a_python_literal_and_a_single_dict_are_read():
    literal = "[{'property': 'area', 'min': 10}]"
    assert me.normalise_filters(literal) == [
        {"property": "area", "min": 10.0, "max": None}]
    assert me.normalise_filters({"property": "area", "max": "20"}) == [
        {"property": "area", "min": None, "max": 20.0}]
    assert me.normalise_filters([("area", 1, "")]) == [
        {"property": "area", "min": 1.0, "max": None}]


@pytest.mark.parametrize("written, message", [
    ("[{'property': 'area', 'min': }]", "must be a list of"),
    (["area"], "needs a 'property'"),
    ([{"min": 3}], "needs a 'property'"),
    ([{"property": "area", "min": "ten"}], "must be a number or empty"),
    ([{"property": "area", "max": float("inf")}], "must be finite"),
    ([{"property": "area", "min": 9, "max": 3}], "above its maximum"),
])
def test_a_filter_that_cannot_be_read_says_why(written, message):
    with pytest.raises(ValueError, match=message):
        me.normalise_filters(written)


def test_object_filters_must_map_object_types_to_lists():
    with pytest.raises(ValueError, match="maps each object type"):
        me.settings_filters({"object_filters": "[1, 2]"}, "cell")
    assert me.settings_filters(
        {"object_filters": "{'cell': [{'property': 'area', 'min': 2}]}"},
        "cell") == [{"property": "area", "min": 2.0, "max": None}]


def test_an_entry_with_neither_bound_judges_nothing():
    labels = _two_objects()
    rules = [{"property": "area", "min": None, "max": None},
             {"property": "area", "min": 10, "max": None}]
    removals = me.filter_removals(labels, rules)
    assert [removal.label for removal in removals] == [1]
    assert [f.index for f in removals[0].failed] == [1]


def test_nonfinite_intensity_is_refused_when_mask_generation_asks():
    labels = _two_objects()
    grey = np.ones(labels.shape, np.float64)
    grey[labels == 2] = np.nan
    rules = [{"property": "intensity_mean", "min": 0.5, "max": None}]
    with pytest.raises(ValueError, match="finite object mean intensities"):
        me.filter_removals(labels, rules, grey,
                           require_finite_intensity=True)
    assert me.filter_removals(labels, rules, grey) == []


def test_a_property_skimage_cannot_measure_in_3d_is_named(monkeypatch):
    import skimage.measure

    def refuse(*args, **kwargs):
        raise NotImplementedError("not in 3-D")

    monkeypatch.setattr(skimage.measure, "regionprops_table", refuse)
    labels = np.zeros((3, 6, 6), np.int32)
    labels[1, 1:4, 1:4] = 1
    with pytest.raises(ValueError, match="cannot measure eccentricity on a 3-D"):
        me.filter_removals(
            labels, [{"property": "eccentricity", "min": 0.1}])


def test_a_property_that_fails_on_both_probes_is_left_off_the_list(
        monkeypatch):
    from skimage.measure import _regionprops

    monkeypatch.setattr(me, "_FILTER_CATALOGUE", None)
    monkeypatch.setitem(_regionprops.COL_DTYPES, "zz_never_computes", float)
    try:
        shape, intensity = me._filter_catalogue()
    finally:
        me._FILTER_CATALOGUE = None
    assert "zz_never_computes" not in shape + intensity
    assert "area" in shape and "intensity_mean" in intensity
