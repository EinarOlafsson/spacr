"""Item 511 in Make Masks: an "Add a filter" list of any regionprop.

The Filter category starts empty; "Add a filter" offers every scalar
regionprop (intensity statistics only with an image open), each row has a
property, a minimum, a maximum and Remove, the list applies live, is recorded
in the field's ledger, and removing a row brings back what it hid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from tests.qt.test_make_masks_filters_splits_and_inverts import (
    PAIR, ROUND, SPECK, _built, field_and_mask,
)

pytestmark = pytest.mark.qt


@pytest.fixture
def one_field(tmp_path: Path) -> Path:
    import imageio.v2 as imageio

    folder = tmp_path / "field"
    (folder / "masks").mkdir(parents=True)
    image, mask = field_and_mask()
    imageio.imwrite(folder / "a.tif", image)
    imageio.imwrite(folder / "masks" / "a.tif", mask)
    return folder


@pytest.fixture
def screen(qtbot, qt_theme_applied, one_field: Path):
    made = _built(qtbot, one_field)
    yield made
    made._magnifier.close()
    made.close_folded()


def ids_of(mask) -> list:
    return sorted(int(v) for v in np.unique(np.asarray(mask)) if v)


def test_without_an_image_only_shape_properties_are_offered(qtbot):
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        offered = made._filter_list.offered()
        assert offered == list(engine.filter_properties())
        assert "intensity_mean" not in offered
        with pytest.raises(ValueError, match="no intensity image"):
            made._filter_list.add_filter("intensity_mean")
    finally:
        made._magnifier.close()
        made.close_folded()


def test_an_open_field_offers_every_scalar_regionprop(screen):
    assert screen._filter_list.offered() == list(
        engine.filter_properties(intensity=True))


def test_add_a_filter_adds_one_row_with_min_max_and_remove(screen):
    assert screen._filter_list.rows() == []
    screen._filter_property.setCurrentText("eccentricity")
    screen._filter_add.click()
    rows = screen._filter_list.rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["property"] == "eccentricity"
    assert row["min"].text() == "" and row["max"].text() == ""
    assert row["remove"].isVisibleTo(screen._filter_list)
    assert screen._filter_list.filters() == [
        {"property": "eccentricity", "min": None, "max": None}]
    assert ids_of(screen._canvas.mask) == sorted([PAIR, ROUND, SPECK]), (
        "a row with no bounds hides nothing")


def test_a_never_hard_coded_property_hides_objects_and_remove_restores(screen):
    before = ids_of(screen._canvas.mask)
    screen._filter_list.set_filter("eccentricity", None, 0.5)
    assert ids_of(screen._canvas.mask) == sorted([ROUND, SPECK]), (
        "the merged pair is elongated; the disc and the square are not")
    rows = screen._filter_log.toPlainText().splitlines()
    assert len(rows) == 1 and rows[0].startswith(f"Object {PAIR} ")
    assert "maximum eccentricity 0.5 (was" in rows[0]

    entry = screen._log.edits[-1]
    assert entry.kind == "filter"
    assert entry.detail["filters"] == [
        {"property": "eccentricity", "min": None, "max": 0.5}], (
        "the list is recorded in the mask's provenance")

    remove = screen._filter_list.rows()[0]["remove"]
    remove.click()
    assert screen._filter_list.rows() == []
    assert ids_of(screen._canvas.mask) == before
    np.testing.assert_array_equal(screen._canvas.mask, field_and_mask()[1])
    assert screen._log.edits[-1].detail["filters"] == []


def test_removing_one_of_two_rows_restores_only_what_it_hid(screen):
    screen._filter_list.set_filter("area", 20)
    screen._filter_list.set_filter("solidity", None, 0.95)
    hidden = ids_of(screen._canvas.mask)
    assert SPECK not in hidden
    screen._filter_list.remove_filter(1)
    assert ids_of(screen._canvas.mask) == sorted([PAIR, ROUND])
    screen._filter_list.remove_filter(0)
    assert ids_of(screen._canvas.mask) == sorted([PAIR, ROUND, SPECK])


def test_an_edit_made_between_filter_runs_survives_removing_a_row(screen):
    screen._filter_list.set_filter("area", 20)
    assert SPECK not in ids_of(screen._canvas.mask)
    mask = screen._canvas.mask.copy()
    mask[mask == ROUND] = 0
    screen._canvas.mask = mask
    screen._filter_list.remove_filter(0)
    assert ids_of(screen._canvas.mask) == sorted([PAIR, SPECK]), (
        "the speck comes back and the erased object stays erased")


def test_the_editor_and_a_mask_run_agree_on_the_same_list(screen):
    from spacr.utils import _filter_objects, _relabel_sequential

    filters = [{"property": "eccentricity", "max": 0.5},
               {"property": "intensity_mean", "min": 150}]
    screen._filter_list.set_filters(filters)
    image, mask = field_and_mask()
    run = _filter_objects(mask.copy(), image, filters=filters)
    np.testing.assert_array_equal(_relabel_sequential(screen._canvas.mask), run)


def test_the_old_four_bounds_open_as_rows(screen):
    screen._filter_list.set_filters({"min_area": 20, "max_area": 0,
                                     "min_intensity": 0, "max_intensity": 0})
    assert screen._filter_list.filters() == [
        {"property": "area", "min": 20.0, "max": None}]
    assert SPECK not in ids_of(screen._canvas.mask)
