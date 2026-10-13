"""Item 288: vendor filename conventions where the name and the file disagree.

``spacr.convert`` can read a plate by a microscope's filename convention.
A convention names ONE plane per file, so when the file itself holds more
the file wins:

* a file holding several channels is not pinned to the one channel its
  name mentions;
* a file holding several series (fields) does not keep the single field
  number its name states;
* a file holding a z-stack or a time series drops the name's z or t token.

Also pinned: the vendor spellings of a well that are not a well at all
give ``None``; a custom pattern that does not compile is refused by name;
field numbers are kept only when each field states exactly one; and a LIF
series without a recorded image index is looked up in the file's series
table.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from spacr import convert as cv
from spacr.errors import ConfigurationError


def test_a_custom_pattern_that_does_not_compile_is_refused():
    with pytest.raises(ConfigurationError) as excinfo:
        cv._convention_key("custom", "(?P<wellID")
    assert "custom_regex does not compile" in str(excinfo.value)


@pytest.mark.parametrize("key, token, well", [
    ("cq1", "", None), ("cq1", "0", None), ("cq1", "25", "B01"),
    ("opera_phenix", "well5", None), ("opera_phenix", "r02c03", "B03"),
    ("leica_matrix_screener", "junk", None),
    ("leica_matrix_screener", "U02--V01", "B03"),
])
def test_vendor_well_spellings(key, token, well):
    assert cv._convention_well(key, token) == well


def test_a_container_axis_wins_over_the_names_token():
    meta = {"z_index": 3, "t_index": 2, "stem": "x"}
    kept = cv._inside_wins(meta, {"n_z": 5, "n_t": 4})
    assert kept["z_index"] is None and kept["t_index"] is None
    assert meta["z_index"] == 3, "the caller's dict is not edited"
    assert cv._inside_wins(meta, {"n_z": 1, "n_t": 1}) == meta


NAME = "plate1_B02_T0001F003L01A01Z01C01.tif"


def test_a_multichannel_file_is_not_pinned_to_its_names_channel(tmp_path):
    tifffile.imwrite(str(tmp_path / NAME),
                     np.zeros((3, 8, 8), np.uint16), photometric="minisblack")
    (source,) = cv.scan(str(tmp_path), metadata_type="cellvoyager")
    assert source.n_channels == 3
    assert source.channel is None
    assert source.meta["field_number"] == 3


def test_a_multiseries_file_does_not_keep_one_field_number(tmp_path,
                                                          monkeypatch):
    tifffile.imwrite(str(tmp_path / NAME), np.zeros((8, 8), np.uint16))
    monkeypatch.setattr(cv, "_describe", lambda path, ext: {
        "n_z": 1, "n_t": 1, "n_c": 1, "n_series": 2, "per_series": []})
    sources = cv.scan(str(tmp_path), metadata_type="cellvoyager")
    assert len(sources) == 2
    assert [s.meta["field_number"] for s in sources] == [None, None]
    assert [s.field.rsplit("#", 1)[-1] for s in sources] == ["s1", "s2"]


def _source(field, number):
    return SimpleNamespace(field=field, meta={"field_number": number})


def test_field_numbers_are_kept_only_when_each_field_states_one():
    assert cv._named_field_numbers(
        [_source("a", 3), _source("b", 7)], ["a", "b"]) == {"a": 3, "b": 7}
    assert cv._named_field_numbers(
        [_source("a", 3), _source("a", 4), _source("b", 7)],
        ["a", "b"]) == {}


def test_a_lif_series_without_an_image_index_is_looked_up(monkeypatch):
    frames = []

    class Image:
        def get_frame(self, z, t, c, m):
            frames.append((z, t, c, m))
            return np.full((2, 2), 10 * c + m, np.uint16)

    images = [Image(), Image()]
    monkeypatch.setattr(cv, "_lif_images", lambda path: images)
    monkeypatch.setattr(cv, "_lif_series", lambda imgs: [
        {"lif_image": 0, "lif_tile": 0}, {"lif_image": 1, "lif_tile": 2}])
    source = cv.SourceImage(path="x.lif", plate="p", well="A01", field="f",
                            z=1, t=1, n_channels=2, meta={"series": 1})
    array = cv._read_lif(source)
    assert frames == [(0, 0, 0, 2), (0, 0, 1, 2)]
    assert np.asarray(array).max() == 12
