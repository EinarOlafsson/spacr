"""Item 288: the FEATURES table at the edges the ordinary drops do not reach.

``spacr.measure``'s FEATURES table turns hand-picked image and mask files
into a Measure run. Everything here is a case where the table has to say
no, or has to decide something the user did not state:

* a mask token that is not a spaCR object, a numbered organelle past the
  last slot, or nothing at all names no role;
* channel tokens with no number sort after the numbered ones;
* a regex with no field group, or with neither a channel nor a mask group,
  places nothing and says why for every file; a file whose field group
  captured nothing, or that captured neither a channel nor an object, is
  left unassigned with its reason;
* a plate and a well captured by the regex are used for a new row;
* a table with no channel or no mask column lists that as a problem;
* renumbering skips a column number another token already holds;
* the table's settings keep a user's PNG channels and a single crop mode
  given as a string;
* intensities that are not finite, negative or too large, masks of the
  wrong shape, negative or too large, are refused by name;
* a table with nowhere to write is refused before anything is written.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.errors import ConfigurationError
from spacr.measure import (FieldRow, FieldTable, _channel_rank_key,
                           _checked_intensity, _checked_label,
                           _renumber_channels, assign_paths_by_regex,
                           field_table_settings, mask_role_of,
                           measure_from_field_table)


@pytest.mark.parametrize("token, role", [
    (None, None), ("  ", None), ("organelle_999", None), ("teapot", None),
    ("Organelle 2", "organelleb"), ("nuclei", "nucleus")])
def test_mask_tokens_name_a_role_or_none(token, role):
    assert mask_role_of(token) == role


def test_unnumbered_channel_tokens_sort_after_numbered_ones():
    tokens = ["gfp", "C10", "C2", "dapi"]
    assert sorted(tokens, key=_channel_rank_key) == ["C2", "C10", "dapi",
                                                     "gfp"]


def test_a_regex_without_a_field_group_places_nothing():
    result = assign_paths_by_regex(["/d/a_C1.tif", "/d/b_C1.tif"],
                                   r"(?P<channel>C\d)")
    assert result.assigned == []
    assert [reason for _p, reason in result.unassigned] == [
        "the regex names no field group, so there is no row to put this in "
        "-- add (?P<field>...) to it"] * 2


def test_a_regex_without_a_column_group_places_nothing():
    result = assign_paths_by_regex(["/d/a.tif"], r"(?P<field>\w+)\.tif")
    assert result.assigned == []
    assert "neither a channel nor a mask group" in result.unassigned[0][1]


def test_files_the_regex_cannot_place_are_listed_with_their_reason():
    pattern = (r"(?P<plateID>p\d)_(?P<wellID>[A-H]\d\d)_(?P<field>\w*)"
               r"_(?:(?P<channel>C\d)|(?P<mask>\w+)_mask)?\.tif")
    result = assign_paths_by_regex([
        "/d/p1_B03_f1_C1.tif",
        "/d/p1_B03_f1_teapot_mask.tif",
        "/d/p1_B03__C1.tif",
        "/d/p1_B03_f2_.tif",
    ], pattern)
    table = result.table
    assert [(r.label, r.well) for r in table.rows] == [("f1", "B03")]
    assert table.plate == "p1"
    reasons = dict(result.unassigned)
    assert "which is not a spaCR mask type" in \
        reasons["/d/p1_B03_f1_teapot_mask.tif"]
    assert "captured no field name" in reasons["/d/p1_B03__C1.tif"]
    assert "captured neither a channel nor an object" in \
        reasons["/d/p1_B03_f2_.tif"]


def test_a_table_without_channels_or_masks_names_both_gaps():
    table = FieldTable(rows=[FieldRow(label="f1")], n_channels=0, roles=())
    problems = table.problems()
    assert "The table needs at least one channel column." in problems
    assert any("at least one mask column" in p for p in problems)


def test_renumbering_skips_a_column_another_token_holds():
    """Two browsed-in columns past the ranked ones get the next free numbers."""
    table = FieldTable(rows=[FieldRow(label="f", channels={0: "a", 5: "x",
                                                             6: "y"})],
                       n_channels=7)
    _renumber_channels(table, ["C2"], ["C1", "C2"])
    assert table.rows[0].channels == {1: "a", 2: "x", 3: "y"}


def test_the_settings_keep_the_users_png_channels_and_a_single_crop_mode():
    table = FieldTable(rows=[FieldRow(label="f", channels={0: "a", 1: "b"},
                                      masks={"cell": "m"})],
                       n_channels=2, roles=("cell",))
    resolved = field_table_settings(
        table, {"png_dims": [1], "crop_mode": "cell", "cytoplasm": False})
    assert resolved["png_dims"] == [1]
    assert resolved["crop_mode"] == ["cell"]
    assert resolved["channels"] == [0, 1]


@pytest.mark.parametrize("plane, words", [
    (np.array([[1.0, np.nan]]), "contain NaN or infinity"),
    (np.array([[1.5, 2.0]]), "lose precision"),
    (np.array([[-1, 2]]), "must fit the Measure uint16 contract"),
    (np.array([[70000, 2]]), "must fit the Measure uint16 contract"),
])
def test_intensities_measure_cannot_hold_are_refused(plane, words):
    with pytest.raises(ConfigurationError, match=words):
        _checked_intensity(plane, "p_A01_1", "/d/a.tif")


def test_whole_float_intensities_are_kept():
    out = _checked_intensity(np.array([[1.0, 2.0]]), "s", "/d/a.tif")
    assert out.dtype == np.uint16 and out.tolist() == [[1, 2]]


@pytest.mark.parametrize("plane, shape, words", [
    (np.zeros((2, 3), np.int32), (2, 2), "does not match the intensity"),
    (np.array([[-1, 0]]), (1, 2), "cannot contain negative IDs"),
    (np.array([[70000, 0]]), (1, 2), "exceeds the maximum 65535"),
])
def test_masks_measure_cannot_hold_are_refused(plane, shape, words):
    with pytest.raises(ConfigurationError, match=words):
        _checked_label(plane, "p_A01_1", "/d/m.tif", shape)


def test_channels_of_different_shapes_in_one_field_are_refused(tmp_path):
    import tifffile

    paths = {}
    for name, shape in (("c1", (8, 8)), ("c2", (8, 9)), ("m", (8, 8))):
        paths[name] = str(tmp_path / f"{name}.tif")
        tifffile.imwrite(paths[name], np.ones(shape, np.uint16))
    table = FieldTable(rows=[FieldRow(label="f", channels={0: paths["c1"],
                                                             1: paths["c2"]},
                                      masks={"cell": paths["m"]})],
                       n_channels=2, roles=("cell",))
    with pytest.raises(ConfigurationError,
                       match="channel 2 of drawn_A01_1 has shape"):
        measure_from_field_table(table, dst=str(tmp_path / "out"))


def test_a_float_crop_is_resized_without_integer_rounding():
    from spacr.measure import _resize_crop_like_the_run

    crop = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    out = _resize_crop_like_the_run(crop, (8, 6))
    assert out.dtype == np.float32 and out.shape == (6, 8)
    assert 0 < float(out[3, 3]) < 1


def test_a_table_with_nowhere_to_write_is_refused(tmp_path):
    table = FieldTable(rows=[FieldRow(label="f", masks={"cell": "m"})],
                       n_channels=1, roles=("cell",))
    with pytest.raises(ConfigurationError, match="nowhere to write"):
        measure_from_field_table(table)
    assert list(tmp_path.iterdir()) == []
