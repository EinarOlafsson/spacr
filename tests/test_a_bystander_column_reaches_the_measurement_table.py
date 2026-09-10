"""The bystander split is a setting, and it reaches the cell table.

Instruction 388 step 1's second half. :mod:`spacr.bystanders` already
decides WHICH uninfected cells are bystanders and is tested on its own;
this file holds the properties the MEASUREMENT is responsible for:

* the reach is a SETTING with a declared type, a tooltip and a category, so
  the GUI can draw it and ``check_settings`` will accept it;
* it is expressed in measured cell diameters, so the same field at twice
  the magnification gives the same answer;
* the columns obey the four rules every measurement column obeys -- numeric,
  no ``label`` in the name, no ``count`` in the name, and never NaN;
* and the acceptance test instruction 388 names: a well with no parasites
  reports no bystanders.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import measure
from spacr.settings import (categories, expected_types,
                            get_measure_crop_settings, tooltips)

COLUMNS = ("cell_is_bystander", "cell_is_distal", "cell_distance_to_infected")


def _settings(**over):
    base = {"cell_mask_dim": 4, "nucleus_mask_dim": 5,
            "pathogen_mask_dim": 6, "organelle_mask_dim": None,
            "cytoplasm": False, "channels": [0, 1],
            "spatial_measurements": False, "object_distances": False,
            "bystander_measurements": True,
            "bystander_reach_in_diameters": 1.0}
    base.update(over)
    return base


def _field(scale: int = 1, infected=(1,)):
    """Two adjacent PAIRS of cells, the pairs far apart.

    1 and 2 are four pixels apart, so are 3 and 4, and the two pairs are
    fifty apart. Infecting cell 1 makes exactly one bystander; infecting 1
    and 4 makes two. Without a second adjacent pair, "the bystander count
    rises with the infected count" cannot be shown at all -- the first
    version of this fixture put 3 and 4 forty pixels apart and the test
    compared one bystander with one bystander.

    Positions are explicit rather than computed from gaps, so the geometry
    the assertions depend on can be read off the source. Cell sides are 10
    (equivalent diameter 11.28), so at a reach of one diameter cell 2 is
    within reach of cell 1 and nothing else is within reach of anything.

    Held away from the field edge, because a cell clipped by the border is
    dropped from the diameter median -- deliberately, but it would make
    this fixture measure something other than what it looks like.
    """
    side = 10 * scale
    starts = [10 * scale, 24 * scale, 70 * scale, 84 * scale]
    size = 140 * scale
    cell = np.zeros((size, size), dtype=np.int32)
    for label, x in enumerate(starts, start=1):
        cell[10 * scale:10 * scale + side, x:x + side] = label
    # ONE NUCLEUS PER CELL, because `get_components` reads the nucleus mask
    # unconditionally -- there is no "no nuclei" path through it.
    nucleus = np.zeros_like(cell)
    pathogen = np.zeros_like(cell)
    for label, x in enumerate(starts, start=1):
        nucleus[10 * scale + side // 2, x + side // 2] = label
    for n, label in enumerate(infected, start=1):
        x = starts[label - 1]
        pathogen[10 * scale + side // 3, x + side // 3] = n
    return cell, nucleus, pathogen


def _cells(cell, nucleus, pathogen, **over):
    frames = measure._morphological_measurements(
        cell, nucleus, pathogen, None, None, _settings(**over), zernike=None)
    return frames[0]


class TestTheSettingExistsAndIsHonest:

    def test_it_is_off_until_asked_for(self):
        """It adds a column family; nothing should turn that on quietly."""
        assert get_measure_crop_settings({})["bystander_measurements"] is False

    def test_the_reach_defaults_to_one_measured_diameter(self):
        assert get_measure_crop_settings({})[
            "bystander_reach_in_diameters"] == pytest.approx(1.0)

    @pytest.mark.parametrize("key,kind", [
        ("bystander_measurements", bool),
        ("bystander_reach_in_diameters", float)])
    def test_each_is_declared_typed_and_reachable(self, key, kind):
        """An undeclared key has no widget and check_settings refuses it."""
        assert expected_types[key] is kind
        assert key in tooltips and tooltips[key]
        assert any(key in keys for keys in categories.values())

    def test_the_tooltip_says_what_the_reach_is_measured_in(self):
        text = tooltips["bystander_reach_in_diameters"]
        assert "diameter" in text
        assert "pixels" in text or "micrometre" in text


class TestTheColumnsReachTheCellTable:

    def test_nothing_is_added_when_it_is_off(self):
        cell, nucleus, pathogen = _field()
        frame = _cells(cell, nucleus, pathogen, bystander_measurements=False)
        assert not [c for c in frame.columns if "bystander" in c
                    or "distal" in c or "distance_to_infected" in c]

    def test_all_three_arrive_when_it_is_on(self):
        cell, nucleus, pathogen = _field()
        frame = _cells(cell, nucleus, pathogen)
        for column in COLUMNS:
            assert column in frame.columns, sorted(frame.columns)

    def test_the_near_cell_is_a_bystander_and_the_far_ones_are_not(self):
        cell, nucleus, pathogen = _field()
        frame = _cells(cell, nucleus, pathogen).set_index("label")
        assert frame.loc[2, "cell_is_bystander"] == 1
        assert frame.loc[4, "cell_is_distal"] == 1

    def test_an_infected_cell_is_neither_flag(self):
        """Two flags carry three states; the infected cell is 0 in both."""
        row = _cells(*_field()).set_index("label").loc[1]
        assert row["cell_is_bystander"] == 0 and row["cell_is_distal"] == 0


class TestTheAcceptanceTestsInstruction388Names:

    def test_a_well_with_no_parasites_reports_no_bystanders(self):
        """THE FIRST THING TO CHECK, in the instruction's own words.

        It follows from the distance being infinite rather than from a
        special case, which is why that distance is infinite.
        """
        cell, nucleus, _ = _field()
        frame = _cells(cell, nucleus, np.zeros_like(cell))
        assert frame["cell_is_bystander"].sum() == 0
        assert frame["cell_is_distal"].sum() == len(frame)

    def test_the_bystander_count_rises_with_the_infected_count(self):
        cell, nucleus, one = _field(infected=(1,))
        _c2, _n2, two = _field(infected=(1, 4))
        fewer = _cells(cell, nucleus, one)["cell_is_bystander"].sum()
        more = _cells(cell, nucleus, two)["cell_is_bystander"].sum()
        assert more > fewer


class TestTheReachIsALengthNotAPixelCount:

    def test_the_same_field_at_twice_the_size_classifies_the_same(self):
        """The point of expressing the reach in diameters.

        Every distance doubles and so does the median diameter, so the
        answer must not move. A hard-coded pixel reach would call the same
        biology different on a 63x acquisition.
        """
        small = _cells(*_field(scale=1))
        large = _cells(*_field(scale=2))
        assert (list(small["cell_is_bystander"])
                == list(large["cell_is_bystander"]))
        assert list(small["cell_is_distal"]) == list(large["cell_is_distal"])

    def test_a_reach_of_zero_makes_every_uninfected_cell_distal(self):
        """Turning the split off without a second setting."""
        frame = _cells(*_field(), bystander_reach_in_diameters=0.0)
        assert frame["cell_is_bystander"].sum() == 0

    def test_an_unreadable_reach_does_not_invent_bystanders(self):
        """A mis-typed setting must not look like a finding."""
        frame = _cells(*_field(), bystander_reach_in_diameters="wide")
        assert frame["cell_is_bystander"].sum() == 0

    def test_a_wider_reach_catches_more(self):
        near = _cells(*_field(), bystander_reach_in_diameters=1.0)
        far = _cells(*_field(), bystander_reach_in_diameters=10.0)
        assert far["cell_is_bystander"].sum() > near["cell_is_bystander"].sum()


class TestTheFourRulesEveryMeasurementColumnObeys:

    def test_every_new_column_is_numeric(self):
        """The object namespace refuses a non-numeric column."""
        frame = _cells(*_field())
        for column in COLUMNS:
            assert np.issubdtype(frame[column].dtype, np.number), column

    def test_no_new_column_carries_label_or_count_in_its_name(self):
        """`label` is folded into the merge key; `count` is stripped from
        every model matrix."""
        for column in COLUMNS:
            bare = column[len("cell_"):]
            assert "label" not in bare and "count" not in bare

    def test_nothing_is_ever_nan(self):
        """One NaN anywhere deletes the column from every model matrix."""
        frame = _cells(*_field())
        for column in COLUMNS:
            assert frame[column].notna().all(), column

    def test_a_field_with_nothing_infected_uses_the_sentinel_not_infinity(self):
        cell, nucleus, _ = _field()
        frame = _cells(cell, nucleus, np.zeros_like(cell))
        assert (frame["cell_distance_to_infected"] == -1.0).all()
        assert np.isfinite(frame["cell_distance_to_infected"]).all()


class TestItDegradesVisiblyRatherThanQuietly:

    def test_a_field_with_no_measurable_diameter_emits_nothing(self, capsys):
        """No columns beats a column of confident nonsense.

        Every cell clipped by the field edge leaves no diameter to take a
        median of. Emitting the block anyway would give a reach of zero and
        mark every uninfected cell distal -- a wrong answer where a missing
        one is honest.
        """
        cell = np.zeros((30, 30), dtype=np.int32)
        cell[0:10, 0:10] = 1          # touches the border
        cell[20:30, 20:30] = 2        # touches the border
        nucleus = np.zeros_like(cell)
        nucleus[5, 5] = 1
        nucleus[25, 25] = 2
        frame = _cells(cell, nucleus, np.zeros_like(cell))
        assert "cell_is_bystander" not in frame.columns
        assert "no median cell diameter" in capsys.readouterr().out
