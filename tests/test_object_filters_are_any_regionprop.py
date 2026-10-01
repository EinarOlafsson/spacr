"""Item 511: object filters are any scikit-image regionprop the user adds.

The maintainer asked that "the user should be able to pick any sikit-image
regionprop to filter on", opt-in rather than one fixed row per filter. The
engine is :mod:`spacr.qt.mask_engine`'s filter list, and these tests pin:

* every scalar regionprop is offered, enumerated from skimage itself, and
  the intensity statistics only when an intensity image exists;
* a filter on each category of property (size, shape, topology, intensity)
  judges objects exactly as ``regionprops_table`` measures them;
* an intensity property without an intensity image is refused with a clear
  message instead of failing later;
* removing a filter restores what only it hid;
* the old four hard-coded bounds migrate into the list, one system;
* ``regionprops_table`` runs once per mask, whatever the list holds;
* Mask generation's ``object_filters`` setting gives the same labels as the
  Make Masks path for the same list.
"""
from __future__ import annotations

import numpy as np
import pytest
from skimage.measure import regionprops_table

from spacr.qt import mask_engine as engine
from spacr.utils import _filter_objects, _relabel_sequential


def field():
    """Five objects that differ in size, shape, topology and brightness."""
    labels = np.zeros((80, 80), dtype=np.uint16)
    yy, xx = np.mgrid[0:80, 0:80]
    labels[((xx - 15) ** 2 + (yy - 15) ** 2) < 64] = 1
    labels[5:9, 35:70] = 2
    labels[30:50, 10:30] = 3
    labels[36:44, 16:24] = 0
    labels[60:63, 60:63] = 4
    labels[((xx - 55) ** 2 + (yy - 45) ** 2) < 36] = 5
    image = np.zeros((80, 80), dtype=np.float64)
    for index, value in enumerate((100.0, 400.0, 250.0, 50.0, 900.0), start=1):
        image[labels == index] = value
    image += np.linspace(0.0, 5.0, image.size).reshape(image.shape)
    return labels, image


def kept_ids(labels):
    return sorted(int(v) for v in np.unique(labels) if v)


def test_every_scalar_regionprop_is_offered_and_nothing_else():
    shape = engine.filter_properties()
    both = engine.filter_properties(intensity=True)
    for name in ("area", "eccentricity", "solidity", "extent", "perimeter",
                 "axis_major_length", "axis_minor_length", "euler_number",
                 "orientation", "feret_diameter_max"):
        assert name in shape
    for name in ("coords", "image", "bbox", "centroid", "moments", "label",
                 "slice", "inertia_tensor"):
        assert name not in both
    intensity = [name for name in both if name not in shape]
    assert set(intensity) >= {"intensity_mean", "intensity_max", "intensity_min"}
    assert not any(name.startswith("intensity_") for name in shape), (
        "an intensity property was offered with no intensity image")


@pytest.mark.parametrize("name", engine.filter_properties(intensity=True))
def test_each_property_judges_objects_as_regionprops_measures_them(name):
    labels, image = field()
    table = regionprops_table(labels.astype(np.int32), intensity_image=image,
                              properties=["label", name])
    values = dict(zip(table["label"], table[name]))
    ordered = sorted(values.values())
    low, high = ordered[1], ordered[-2]
    expected = sorted(int(k) for k, v in values.items() if low <= v <= high)
    removals = engine.filter_removals(
        labels, [{"property": name, "min": low, "max": high}], image)
    dropped = {removal.label for removal in removals}
    assert sorted(set(values) - dropped) == expected
    for removal in removals:
        assert removal.failed[0].property == name
        assert removal.failed[0].value == pytest.approx(values[removal.label])


def test_a_property_spacr_never_hard_coded_filters_objects():
    labels, image = field()
    holes = engine.filter_removals(labels, [{"property": "euler_number", "min": 1}])
    assert [removal.label for removal in holes] == [3], "the ring has a hole"
    long = engine.filter_removals(labels, [{"property": "eccentricity", "max": 0.9}])
    assert [removal.label for removal in long] == [2]
    bright = engine.filter_removals(
        labels, [{"property": "intensity_max", "max": 500}], image)
    assert [removal.label for removal in bright] == [5]


def test_an_intensity_property_without_an_intensity_image_is_refused():
    """spaCR refuses before scikit-image is called, on every supported skimage.

    intensity_min, not intensity_std: intensity_std arrived in skimage
    0.23, and on the minimum supported 0.22 it is correctly refused as a
    property that release does not have, before the intensity check is
    reached. intensity_min exists in every release spaCR supports.
    """
    labels, _image = field()
    with pytest.raises(ValueError, match="intensity_min measures pixel values"):
        engine.filter_removals(labels, [{"property": "intensity_min", "max": 1}])
    if "intensity_std" in engine.filter_properties(intensity=True):
        with pytest.raises(ValueError,
                           match="intensity_std measures pixel values"):
            engine.filter_removals(
                labels, [{"property": "intensity_std", "max": 1}])
    with pytest.raises(ValueError, match="cannot run on a mask alone"):
        _filter_objects(labels.copy(), None,
                        filters=[{"property": "intensity_mean", "min": 1}])
    assert engine.filters_need_intensity([{"property": "mean_intensity"}])
    assert not engine.filters_need_intensity([{"property": "solidity"}])


def test_removing_a_filter_restores_the_objects_it_hid():
    labels, image = field()
    both = [{"property": "area", "min": 20},
            {"property": "eccentricity", "max": 0.9}]
    hidden, removals = engine.apply_filters(labels, image, both)
    assert kept_ids(hidden) == [1, 3, 5]
    assert {removal.label for removal in removals} == {2, 4}
    restored, removals = engine.apply_filters(labels, image, both[:1])
    assert kept_ids(restored) == [1, 2, 3, 5]
    unfiltered, removals = engine.apply_filters(labels, image, [])
    assert kept_ids(unfiltered) == [1, 2, 3, 4, 5] and removals == []


def test_the_old_hard_coded_bounds_migrate_into_the_list():
    assert engine.legacy_filters() == []
    assert engine.legacy_filters(min_area=20, max_intensity=300.0) == [
        {"property": "area", "min": 20.0, "max": None},
        {"property": "intensity_mean", "min": None, "max": 300.0}]
    labels, image = field()
    legacy = _filter_objects(labels.copy(), image, min_area=20,
                             max_intensity=300.0)
    listed = _filter_objects(labels.copy(), image, filters=engine.legacy_filters(
        min_area=20, max_intensity=300.0))
    np.testing.assert_array_equal(legacy, listed)
    out, removals = engine.filter_report(labels, image, min_area=20)
    assert [(r.label, r.bounds) for r in removals] == [(4, ("min_area",))]


def test_a_filter_list_is_checked_when_it_is_read():
    assert engine.normalise_filters(
        '[{"property": "MeanIntensity", "min": "2", "max": ""}]') == [
        {"property": "intensity_mean", "min": 2.0, "max": None}]
    assert engine.normalise_filters([("solidity", 0.5, None)]) == [
        {"property": "solidity", "min": 0.5, "max": None}]
    with pytest.raises(ValueError, match="not a scalar scikit-image regionprop"):
        engine.normalise_filters([{"property": "coords", "min": 1}])
    with pytest.raises(ValueError, match="above its maximum"):
        engine.normalise_filters([{"property": "area", "min": 5, "max": 1}])
    with pytest.raises(ValueError, match="takes only"):
        engine.normalise_filters([{"property": "area", "low": 5}])


def test_the_setting_is_per_object_type():
    settings = {"object_filters": "{'cell': [{'property': 'solidity', 'min': 0.9}]}"}
    assert engine.settings_filters(settings, "cell") == [
        {"property": "solidity", "min": 0.9, "max": None}]
    assert engine.settings_filters(settings, "nucleus") == []
    assert engine.settings_filters({}, "cell") == []
    with pytest.raises(ValueError, match="not an object type"):
        engine.settings_filters({"object_filters": {"cels": []}}, "cell")


def test_regionprops_table_runs_once_per_mask(monkeypatch):
    import skimage.measure

    labels, image = field()
    calls = []
    real = skimage.measure.regionprops_table

    def counting(*args, **kwargs):
        calls.append(kwargs.get("properties"))
        return real(*args, **kwargs)

    monkeypatch.setattr(skimage.measure, "regionprops_table", counting)
    engine.filter_properties(intensity=True)
    calls.clear()
    _filter_objects(labels.copy(), image, min_area=10, filters=[
        {"property": "solidity", "min": 0.5},
        {"property": "eccentricity", "max": 0.95},
        {"property": "intensity_max", "max": 1000}])
    assert len(calls) == 1
    assert set(calls[0]) == {"label", "area", "solidity", "eccentricity",
                             "intensity_max"}


def test_mask_generation_gives_the_labels_make_masks_gives():
    """The same list, through the Mask run's filter and the editor's."""
    from spacr.object import merge_split_filter_masks

    labels, image = field()
    filters = [{"property": "solidity", "min": 0.8},
               {"property": "intensity_mean", "max": 800},
               {"property": "perimeter", "min": 10}]
    settings = {"object_filters": {"cell": filters}}
    run = merge_split_filter_masks([labels.copy()], [image], settings, "cell")[0]
    edited, _removals = engine.apply_filters(labels, image, filters)
    np.testing.assert_array_equal(run, _relabel_sequential(edited))
    assert kept_ids(edited) == [1, 2, 3]



# ---------------------------------------------------------------------------
# 2026-09-25, the maintainer: "RETIRE Mask's old per-object {obj}_min_area,
# _max_area, _min_intensity, _max_intensity settings from the form; saved
# settings that carry them are migrated into object_filters rows on load."
# ---------------------------------------------------------------------------

_RETIRED = [f"{obj}_{bound}" for obj in ("cell", "nucleus", "pathogen")
            for bound in ("min_area", "max_area", "min_intensity",
                          "max_intensity")]


def test_the_retired_bounds_are_no_longer_mask_settings():
    from spacr import settings as S
    from spacr.validate import RETIRED_SETTINGS

    defaults = S.set_default_settings_preprocess_generate_masks({})
    shown = {key for keys in S.categories.values() for key in keys}
    for key in _RETIRED:
        assert key not in defaults
        assert key not in S.expected_types
        assert key not in S.tooltips
        assert key not in shown
        assert RETIRED_SETTINGS[key] == "object_filters"
    assert defaults["object_filters"] == {}
    assert "object_filters" in shown
    assert set(S.RETIRED_OBJECT_BOUNDS) == set(_RETIRED)


def test_an_old_settings_file_is_migrated_into_rows_on_load(tmp_path):
    import csv

    from spacr.cli import _under_todays_names, load_settings_file
    from spacr.settings import set_default_settings_preprocess_generate_masks

    path = tmp_path / "old_mask_settings.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Key", "Value"))
        writer.writerows([("cell_min_area", 50), ("cell_max_area", 0),
                          ("nucleus_max_intensity", 900.5),
                          ("pathogen_min_area", 0),
                          ("pathogen_min_intensity", 0.0)])
    moved = []
    loaded = _under_todays_names(load_settings_file(str(path)), moved)
    want = {"cell": [{"property": "area", "min": 50.0, "max": None}],
            "nucleus": [{"property": "intensity_mean", "min": None,
                         "max": 900.5}]}
    assert not set(_RETIRED) & set(loaded)
    assert loaded["object_filters"] == want
    assert {problem.setting for problem in moved} == {
        "cell_min_area", "cell_max_area", "nucleus_max_intensity",
        "pathogen_min_area", "pathogen_min_intensity"}
    assert all("object_filters" in problem.message for problem in moved)

    run = set_default_settings_preprocess_generate_masks(dict(loaded))
    assert run["object_filters"] == want
    assert engine.object_filter_area_floor(run, "cell") == 50
    assert engine.object_filter_area_floor(run, "nucleus") == 0


def test_the_mask_defaults_fold_a_script_that_still_passes_the_old_names():
    from spacr.settings import (_get_object_settings,
                                set_default_settings_preprocess_generate_masks)

    run = set_default_settings_preprocess_generate_masks(
        {"cell_min_area": 12, "cell_min_intensity": 3.5, "verbose": False})
    assert "cell_min_area" not in run and "cell_min_intensity" not in run
    assert engine.settings_filters(run, "cell") == [
        {"property": "area", "min": 12.0, "max": None},
        {"property": "intensity_mean", "min": 3.5, "max": None}]
    assert _get_object_settings("cell", run)["min_size"] == 12


def test_a_row_the_file_already_has_wins_over_a_retired_bound():
    from spacr.settings import _fold_object_bounds

    folded = _fold_object_bounds({
        "cell_min_area": 10, "cell_max_intensity": 70,
        "object_filters": '{"cell": [{"property": "area", "min": 99}]}'})
    assert folded["object_filters"] == {"cell": [
        {"property": "area", "min": 99},
        {"property": "intensity_mean", "min": None, "max": 70.0}]}
    assert "cell_min_area" not in folded


def test_an_unusable_retired_bound_is_left_for_the_run_to_refuse():
    """Moving it would turn the old refusal into a silent 'off'."""
    from spacr.settings import _fold_object_bounds
    from spacr.utils import _validated_intensity_bounds

    folded = _fold_object_bounds({"cell_min_intensity": -1,
                                  "nucleus_max_intensity": float("nan")},
                                 quiet=True)
    assert folded["cell_min_intensity"] == -1
    assert "object_filters" not in folded
    with pytest.raises(ValueError):
        _validated_intensity_bounds(folded["cell_min_intensity"], 0)


def test_a_migrated_file_filters_exactly_as_the_old_bounds_did():
    """The pipeline still honours an old saved value, through the row."""
    from spacr.object import merge_split_filter_masks
    from spacr.settings import _fold_object_bounds

    labels = np.zeros((40, 40), np.int32)
    labels[2:4, 2:4] = 1
    labels[10:20, 10:20] = 2
    labels[25:35, 25:35] = 3
    image = np.full(labels.shape, 10.0)
    image[labels == 3] = 500.0
    old = dict(min_area=10, max_area=0, min_intensity=0.0, max_intensity=100.0)
    legacy = _filter_objects(labels.copy(), image, **old)
    folded = _fold_object_bounds({f"cell_{k}": v for k, v in old.items()})
    assert not set(_RETIRED) & set(folded)
    run = merge_split_filter_masks([labels.copy()], [image], folded, "cell")[0]
    np.testing.assert_array_equal(run, _relabel_sequential(legacy))
    assert legacy.max() == 1 and legacy[15, 15] == 1, (
        "only the mid-sized dim object survives the old bounds")
