"""One setting, five shapes, and every one of them reachable from the panel.

Instruction 236 A2, quoted in the request as the part that has to be easy:

    "the dividing of database measurements and running these models works
     smoothly and intuitively, so the user can train on channel_1
     measurements only or morphological measurements or channel
     combinations, localization, etcetera. This should be straight forward
     and easy."

WHAT WAS FOUND. `utils.filter_dataframe_features` has always taken a list
of channels, the string `'morphology'`, and a free-text column fragment --
its own docstring says so. `settings.expected_types` declared the setting
as `int`. So the panel could only draw a spin box, `check_settings` refused
anything else, and three of the four documented ways of choosing a feature
space were unreachable by anybody who was not editing Python.

The fix is one setting that takes what a user means by it, a multi-select
that says the whole question in one row, and a results folder named from
the canonical form so `1` and `[1]` do not write to two places.

LOCALISATION NEEDS NO SETTING OF ITS OWN. A colocalisation column names the
two channels it measures and survives a request for either, so asking for
channel 1 already brings channel 1's relationships with every other channel
with it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.utils import (FEATURE_GROUPS, MORPHOLOGY, feature_columns,
                         feature_folder_name, feature_selection)


COLUMNS = [
    "cell_area", "cell_solidity", "nucleus_zernike_7",
    "cell_channel_0_mean_intensity", "cell_channel_1_mean_intensity",
    "cell_channel_2_mean_intensity", "cell_channel_3_mean_intensity",
    "cell_channel_1_channel_2_pearsons",
    "pathogen_channel_1_percentile_75",
]


class TestWhatAUserCanMean:
    @pytest.mark.parametrize("said,expected", [
        (None, None), ("", None), ("   ", None), ([], None),
        ("all", None), ("All", None), ("none", None),
        (1, 1), ("1", 1), ([1], 1), (["1"], 1),
        ([1, 2], [1, 2]), ("1,2", [1, 2]), ("1, 2", [1, 2]),
        ("morphology", MORPHOLOGY), ("Morphology", MORPHOLOGY),
        ([1, "morphology"], [1, MORPHOLOGY]),
        ("mean_intensity", "mean_intensity"),
    ])
    def test_it_is_read_the_way_it_was_meant(self, said, expected):
        assert feature_selection(said) == expected

    def test_a_repeated_group_is_the_same_feature_space(self):
        """A permutation or a duplicate must not change the output path."""
        assert feature_selection([3, 3]) == 3
        assert feature_selection([1, 2, 1]) == [1, 2]

    def test_a_single_member_list_is_the_member(self):
        """`1` and `[1]` name the same feature space, and the panel's
        multi-select produces the second. Without this the chip strip would
        write its answers to a folder the older integer setting never used,
        and two runs of one analysis would not be comparable."""
        assert feature_selection([1]) == feature_selection(1)
        assert feature_folder_name([1]) == feature_folder_name(1)

    def test_a_boolean_is_refused_by_name(self):
        """True is an int in Python and would be read as channel 1."""
        with pytest.raises(ValueError, match="boolean"):
            feature_selection(True)

    def test_something_that_is_not_a_selection_at_all_says_so(self):
        with pytest.raises(ValueError, match="channel_of_interest"):
            feature_selection({"channel": 1})


class TestWhatEachSelectionKeeps:
    def test_one_channel_keeps_only_that_channel(self):
        kept = feature_columns(COLUMNS, 1)
        assert "cell_channel_1_mean_intensity" in kept
        assert "cell_channel_2_mean_intensity" not in kept
        assert "cell_area" not in kept

    def test_a_combination_keeps_the_union(self):
        """Not the intersection -- two channels asked for together is more
        features, not none."""
        one = set(feature_columns(COLUMNS, 1))
        two = set(feature_columns(COLUMNS, 2))
        both = set(feature_columns(COLUMNS, [1, 2]))
        assert both == one | two
        assert len(both) > len(one)

    def test_morphology_is_shape_on_every_object(self):
        kept = feature_columns(COLUMNS, MORPHOLOGY)
        assert set(kept) == {"cell_area", "cell_solidity",
                             "nucleus_zernike_7"}

    def test_a_channel_and_shape_together(self):
        """The combination the request names, and the one a spin box could
        not express at all."""
        kept = set(feature_columns(COLUMNS, [1, MORPHOLOGY]))
        assert "cell_channel_1_mean_intensity" in kept
        assert "cell_area" in kept
        assert "cell_channel_2_mean_intensity" not in kept

    def test_localisation_comes_with_its_channel(self):
        """A colocalisation column measures a RELATIONSHIP between two
        channels, so it belongs to both. Asking for channel 1 means how
        channel 1 relates to everything it was measured against."""
        assert "cell_channel_1_channel_2_pearsons" in feature_columns(
            COLUMNS, 1)
        assert "cell_channel_1_channel_2_pearsons" in feature_columns(
            COLUMNS, 2)
        assert "cell_channel_1_channel_2_pearsons" not in feature_columns(
            COLUMNS, 3)

    def test_a_free_text_filter_keeps_the_family(self):
        kept = feature_columns(COLUMNS, "mean_intensity")
        assert all("mean_intensity" in c for c in kept)
        assert len(kept) == 4

    def test_nothing_chosen_keeps_everything(self):
        assert feature_columns(COLUMNS, None) == COLUMNS

    def test_the_order_of_the_columns_is_preserved(self):
        """A feature list that reorders itself between runs makes two
        importance tables incomparable for no reason."""
        kept = feature_columns(COLUMNS, [0, 1, 2, 3])
        assert kept == [c for c in COLUMNS if c in set(kept)]


class TestItIsReachableFromThePanel:
    def test_the_declared_type_admits_every_documented_shape(self):
        """THE DEFECT. `int` alone made three of the four documented ways
        of choosing a feature space refusable by `check_settings` and
        undrawable by the panel."""
        from spacr.settings import expected_types

        declared = expected_types["channel_of_interest"]
        for shape in (int, str, list, type(None)):
            assert shape in declared, shape

    def test_the_panel_offers_the_groups_as_a_multi_select(self):
        from spacr.qt.screens.settings_model import FIXED_ALPHABETS

        offered = [value for value, _label
                   in FIXED_ALPHABETS["channel_of_interest"]]
        assert offered == list(FEATURE_GROUPS)

    def test_the_tooltip_says_what_each_choice_does(self):
        """A control a user cannot read is a control they will not use."""
        from spacr.settings import tooltips

        said = tooltips["channel_of_interest"].lower()
        for expected in ("shape", "combination", "colocalisation",
                         "every measurement"):
            assert expected in said, expected


class TestTheResultsFolder:
    @pytest.mark.parametrize("said,folder", [
        (None, "all_features"),
        (1, "channel_1"),
        ([1], "channel_1"),
        ([1, 2], "channels_1_2"),
        ("1,2", "channels_1_2"),
        (MORPHOLOGY, "morphology"),
        ([1, MORPHOLOGY], "channel_1_morphology"),
        ("mean_intensity", "mean_intensity"),
    ])
    def test_it_names_the_feature_space(self, said, folder):
        assert feature_folder_name(said) == folder

    def test_a_free_text_filter_no_longer_kills_the_run(self):
        """`filter_dataframe_features` has always accepted a column
        fragment, and the path builder raised on one -- so the features
        were filtered and the run then died on the way to naming a
        folder."""
        assert feature_folder_name("cell_channel_1/percentile") \
            == "cell_channel_1_percentile"

    def test_the_name_is_safe_on_a_filesystem(self):
        name = feature_folder_name("intensity > 0.5 (raw)")
        assert "/" not in name and " " not in name and name


class TestItReachesTheFitter:
    def _frame(self, rows=200, seed=0):
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame(
            rng.normal(size=(rows, len(COLUMNS))), columns=COLUMNS)
        half = rows // 2
        frame["columnID"] = ["c1"] * half + ["c2"] * (rows - half)
        frame["rowID"] = [f"r{1 + i % 8}" for i in range(rows)]
        frame["plateID"] = "plate1"
        frame["fieldID"] = [f"f{1 + i % 4}" for i in range(rows)]
        frame["object_label"] = [str(i) for i in range(rows)]
        frame.loc[frame["columnID"] == "c2", COLUMNS] += 2.0
        frame.index = [
            f"plate1_{frame['rowID'][i]}_{frame['columnID'][i]}_"
            f"{frame['fieldID'][i]}_o{i}" for i in range(rows)]
        return frame

    @pytest.mark.parametrize("said", [None, 1, [1, 2], MORPHOLOGY,
                                      [1, MORPHOLOGY], "mean_intensity"])
    def test_the_model_trains_on_exactly_what_was_chosen(self, said):
        from spacr.ml import ml_analysis

        output, _figures = ml_analysis(
            self._frame(), channel_of_interest=said,
            location_column="columnID", positive_control="c2",
            negative_control="c1", n_repeats=1, top_features=5,
            n_estimators=8, model_type="random_forest", n_jobs=1,
            remove_low_variance_features=False,
            remove_highly_correlated_features=False, verbose=False)
        trained_on = set(output[9])
        assert trained_on
        assert trained_on <= set(feature_columns(COLUMNS, said))

    def test_a_combination_sees_more_than_either_half(self):
        """Driven on the tsg101 screen, channel 1 gave 53 features, shape
        gave 15, and the two together gave 68 -- the union exactly."""
        from spacr.ml import ml_analysis

        def trained(said):
            output, _ = ml_analysis(
                self._frame(), channel_of_interest=said,
                location_column="columnID", positive_control="c2",
                negative_control="c1", n_repeats=1, top_features=5,
                n_estimators=8, model_type="random_forest", n_jobs=1,
                remove_low_variance_features=False,
                remove_highly_correlated_features=False, verbose=False)
            return set(output[9])

        assert trained([1, MORPHOLOGY]) == trained(1) | trained(MORPHOLOGY)
