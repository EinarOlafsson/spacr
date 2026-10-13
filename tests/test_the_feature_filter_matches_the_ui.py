"""The four shapes a feature filter arrives in, and what each one selects.

``_feature_filter_matches`` exists so the console and the UI select the same
columns from the same filter -- its docstring says "the same public filter
forms as the UI". Two selections that disagree here mean a model trained on
one set of features and a report describing another, which nothing downstream
would notice.
"""
from __future__ import annotations

import pytest


COLUMNS = [
    "cell_area", "cell_perimeter", "cell_eccentricity", "cell_solidity",
    "cell_channel_0_mean_intensity", "cell_channel_1_mean_intensity",
    "cell_channel_2_mean_intensity",
    "nucleus_area", "nucleus_channel_1_mean_intensity",
    "prcfo", "label",
]


def test_morphology_selects_shape_columns_and_no_intensities():
    """The named vocabulary, which is the one form that is not a channel.

    An intensity column slipping into a morphology selection is the failure
    that matters: it makes a "shape only" model depend on staining.
    """
    from spacr.utils import _feature_filter_matches

    picked = _feature_filter_matches(COLUMNS, "morphology")

    assert "cell_area" in picked
    assert "cell_eccentricity" in picked
    assert not [c for c in picked if "mean_intensity" in c]


def test_a_single_channel_selects_only_that_channel():
    """An int, which is what the settings panel stores for one channel."""
    from spacr.utils import _feature_filter_matches

    picked = _feature_filter_matches(COLUMNS, 1)

    assert "cell_channel_1_mean_intensity" in picked
    assert "nucleus_channel_1_mean_intensity" in picked
    assert "cell_channel_0_mean_intensity" not in picked


def test_several_channels_select_all_of_them():
    """A list, which is the multi-channel selection."""
    from spacr.utils import _feature_filter_matches

    picked = _feature_filter_matches(COLUMNS, [0, 2])

    assert "cell_channel_0_mean_intensity" in picked
    assert "cell_channel_2_mean_intensity" in picked
    assert "cell_channel_1_mean_intensity" not in picked


def test_free_text_selects_by_substring():
    """The else: anything else is a column fragment, matched as written.

    This is the form a user reaches for when they want every intensity column
    regardless of channel, and it is why the docstring calls these "public
    filter forms" -- the string is the user's, not a vocabulary.
    """
    from spacr.utils import _feature_filter_matches

    picked = _feature_filter_matches(COLUMNS, "mean_intensity")

    assert len(picked) == 4
    assert all("mean_intensity" in c for c in picked)


def test_a_filter_matching_nothing_selects_nothing():
    """An empty result rather than everything, which is the safe direction.

    Returning all columns for an unmatched filter would silently train on the
    full feature set while the report said otherwise.
    """
    from spacr.utils import _feature_filter_matches

    assert _feature_filter_matches(COLUMNS, "no_such_feature") == []
    assert _feature_filter_matches(COLUMNS, 9) == []


def test_a_non_string_column_name_does_not_break_the_match():
    """``str(column)`` on both sides, so a numeric column header is survivable."""
    from spacr.utils import _feature_filter_matches

    picked = _feature_filter_matches([0, 1, "cell_area"], "area")

    assert picked == ["cell_area"]
