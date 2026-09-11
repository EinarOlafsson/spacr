"""Every refusal in the regression layout conversions, one at a time.

`spacr/regression_layout.py` says in its own opening what it is for: "a
value is never silently selected when duplicate rows disagree". That
promise is kept entirely by refusals -- eight of them, each naming a
different malformed input -- and instruction 352's sweep found fifteen
statements and fifteen PARTIAL BRANCHES uncovered over a hundred
statements, which is almost exactly that set.

A REFUSAL NOBODY HAS RUN IS A PROMISE NOBODY HAS CHECKED. The risk is not
that the arm is missing but that it names the wrong thing, or raises the
wrong type, or -- worst for this module -- that it does not fire and a
value IS silently selected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.regression_layout import (long_to_wide_regression_data,
                                     normalise_count_table_layout,
                                     wide_to_long_regression_data)


class TestTheWideToLongRefusals:

    def test_something_that_is_not_a_frame(self):
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            wide_to_long_regression_data([{"prc": "p1_A01"}])

    def test_a_named_predictor_column_that_is_not_there(self):
        frame = pd.DataFrame({"prc": ["p1_A01"], "g1": [1.0]})
        with pytest.raises(ValueError, match="absent: g2"):
            wide_to_long_regression_data(frame, predictor_columns=["g1", "g2"])

    def test_a_non_numeric_column_nobody_declared(self):
        """The one refusal that exists to stop a GUESS.

        A text column is either metadata or a mislabelled predictor, and
        the module cannot tell. Choosing either way silently is what this
        arm refuses to do.
        """
        frame = pd.DataFrame({"prc": ["p1_A01"], "g1": [1.0],
                              "notes": ["a comment"]})
        with pytest.raises(ValueError, match="not declared as metadata"):
            wide_to_long_regression_data(frame)
        # AND IT SAYS WHICH COLUMN, because a screen has hundreds.
        with pytest.raises(ValueError, match="notes"):
            wide_to_long_regression_data(frame)

    def test_a_frame_with_no_predictor_at_all(self):
        frame = pd.DataFrame({"prc": ["p1_A01"]})
        with pytest.raises(ValueError, match="No wide independent-variable"):
            wide_to_long_regression_data(frame)

    def test_a_column_asked_to_be_both_identifier_and_predictor(self):
        frame = pd.DataFrame({"prc": ["p1_A01"], "g1": [1.0]})
        with pytest.raises(ValueError, match="both identifiers and predictors"):
            wide_to_long_regression_data(frame, id_columns=["prc", "g1"],
                                         predictor_columns=["g1"])

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_a_value_that_is_not_finite(self, bad):
        """NaN and both infinities, because a fit cannot use any of them."""
        frame = pd.DataFrame({"prc": ["p1_A01", "p1_A02"], "g1": [1.0, bad]})
        with pytest.raises(ValueError, match="must be finite"):
            wide_to_long_regression_data(frame)

    def test_a_well_formed_frame_converts(self):
        """The control: the refusals are not refusing everything."""
        frame = pd.DataFrame({"prc": ["p1_A01", "p1_A02"],
                              "g1": [1.0, 2.0], "g2": [3.0, 4.0]})
        got = wide_to_long_regression_data(frame)
        assert len(got) == 4
        assert set(got.columns) >= {"prc"}


class TestTheLongToWideRefusals:

    @staticmethod
    def _long():
        return pd.DataFrame({"prc": ["p1_A01", "p1_A01"],
                             "grna": ["g1", "g2"], "count": [1.0, 2.0]})

    def test_a_missing_required_column_is_named(self):
        frame = self._long().drop(columns=["count"])
        with pytest.raises(ValueError, match="lacks: count"):
            long_to_wide_regression_data(frame, index_columns=["prc"],
                                         predictor_column="grna",
                                         value_column="count")

    def test_no_index_column_at_all(self):
        with pytest.raises(ValueError, match="At least one index column"):
            long_to_wide_regression_data(self._long(), index_columns=[],
                                         predictor_column="grna",
                                         value_column="count")

    def test_a_value_that_is_not_finite(self):
        frame = self._long()
        frame.loc[1, "count"] = np.inf
        with pytest.raises(ValueError, match="must be finite"):
            long_to_wide_regression_data(frame, index_columns=["prc"],
                                         predictor_column="grna",
                                         value_column="count")

    def test_two_rows_for_one_pair_that_disagree(self):
        """THE REFUSAL THE MODULE'S OPENING SENTENCE IS ABOUT.

        One (well, gRNA) with two different counts. Picking either is a
        silent choice about somebody's data, so it refuses -- and names an
        example, because a screen has thousands of pairs and "some pair
        disagrees" is not something anyone can act on.
        """
        frame = pd.DataFrame({"prc": ["p1_A01", "p1_A01"],
                              "grna": ["g1", "g1"], "count": [1.0, 9.0]})
        with pytest.raises(ValueError, match="conflicting values"):
            long_to_wide_regression_data(frame, index_columns=["prc"],
                                         predictor_column="grna",
                                         value_column="count")
        with pytest.raises(ValueError, match="for example"):
            long_to_wide_regression_data(frame, index_columns=["prc"],
                                         predictor_column="grna",
                                         value_column="count")

    def test_two_rows_that_agree_are_not_a_conflict(self):
        """The same pair twice with the SAME value is not a disagreement."""
        frame = pd.DataFrame({"prc": ["p1_A01", "p1_A01"],
                              "grna": ["g1", "g1"], "count": [4.0, 4.0]})
        got = long_to_wide_regression_data(frame, index_columns=["prc"],
                                           predictor_column="grna",
                                           value_column="count")
        assert float(got.loc[0, "g1"]) == 4.0

    def test_metadata_that_is_not_constant_within_an_observation(self):
        """A wide row has ONE cell per column; two values cannot both go in."""
        frame = pd.DataFrame({"prc": ["p1_A01", "p1_A01"],
                              "gene": ["GENE1", "GENE2"],
                              "grna": ["g1", "g2"], "count": [1.0, 2.0]})
        with pytest.raises(ValueError, match="is not constant within"):
            long_to_wide_regression_data(frame, index_columns=["prc"],
                                         predictor_column="grna",
                                         value_column="count",
                                         metadata_columns=["gene"])


class TestTheCountTableLayout:

    def test_a_layout_nobody_offers(self):
        frame = pd.DataFrame({"grna": ["g1"], "count": [1]})
        with pytest.raises(ValueError, match="choose one of"):
            normalise_count_table_layout(frame, layout="sideways")

    def test_a_long_table_missing_its_columns_is_named(self):
        frame = pd.DataFrame({"prc": ["p1_A01"]})
        with pytest.raises(ValueError, match="lacks"):
            normalise_count_table_layout(frame, layout="long")

    def test_a_long_table_passes_through_and_says_so(self):
        frame = pd.DataFrame({"prc": ["p1_A01"], "grna": ["g1"],
                              "count": [3]})
        got, found = normalise_count_table_layout(frame, layout="long")
        assert found == "long"
        assert set(got.columns) >= {"grna", "count"}
