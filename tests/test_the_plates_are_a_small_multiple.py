"""The plate id is read right to left, so two plates are never averaged.

An experiment prefix used to collapse them into one picture.

Taking the LEADING token of a `prc` key made `exp_plate1_r1_c1` and
`exp_plate2_r1_c1` both the plate `exp`, so a screen whose plate ids carry a
prefix was drawn as ONE grid with the two plates averaged well by well --
silently, and looking exactly like a plate measured twice as densely. A
positional artefact on one plate and not the other is the entire reason to
look at a plate heatmap, and it disappeared into the mean.

The rule is the one `spacr.ml._split_prc` and `spacr.schema.parse_prcf`
already state, for the reason both give: the plate id is the only component
allowed to contain the separator, so it is the only one that cannot be found
by counting from the left.
"""
from __future__ import annotations

import pytest

from spacr.figures.plates import plate_names


class TestThePlateNameIsReadRightToLeft:

    def _frame(self, keys):
        import pandas as pd

        return pd.DataFrame({"prc": keys})

    def test_a_plain_three_token_key_is_unchanged(self):
        """Every screen this has ever run on. The fix must not move them."""
        from spacr.figures.plates import plate_names

        assert plate_names(self._frame(
            ["plate1_r10_c11", "plate2_r1_c1", "plate1_r2_c3"])) == \
            ["plate1", "plate2"]

    def test_an_experiment_prefix_no_longer_collapses_two_plates(self):
        from spacr.figures.plates import plate_names

        names = plate_names(self._frame(
            ["exp_plate1_r1_c1", "exp_plate2_r1_c1"]))

        assert names == ["exp_plate1", "exp_plate2"], (
            "two plates were collapsed into one and averaged")

    def test_a_plate_id_may_contain_the_separator(self):
        from spacr.figures.plates import plate_names

        assert plate_names(self._frame(
            ["exp1_plate1_r1_c1", "exp1_plate1_r2_c2"])) == ["exp1_plate1"]

    def test_a_malformed_key_is_visible_rather_than_absorbed(self):
        """Dropping it would hide the row inside another plate's grid."""
        from spacr.figures.plates import plate_names

        names = plate_names(self._frame(["plate1_r1_c1", "rubbish"]))

        assert "rubbish" in names

    def test_the_order_is_the_order_the_data_gives(self):
        from spacr.figures.plates import plate_names

        assert plate_names(self._frame(
            ["plate3_r1_c1", "plate1_r1_c1", "plate3_r2_c2"])) == \
            ["plate3", "plate1"]

    def test_it_agrees_with_the_module_that_already_owns_this_rule(self):
        """Two copies of one parsing rule is how they drift apart."""
        from spacr.ml import _split_prc

        for key in ("plate1_r10_c11", "exp_plate1_r1_c1",
                    "exp1_plate1_r2_c2"):
            plate, _row, _column = _split_prc(key)
            assert plate_names(self._frame([key])) == [plate], key

