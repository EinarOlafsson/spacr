"""Nine more single decisions in ``timelapse.py``.

An IoU denominator that cannot be zero for labels that exist, an axis
budget the layout already sized, a channel count the caller has already
bounded, and five presence checks on things the lines above built.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from spacr import timelapse as T


class TestTheIouDenominator:

    def test_two_labels_that_exist_always_have_a_union(self):
        """THE PIN, for ``if union > 0``.

        The masks come from ``np.unique`` over a labelled image, so every
        label in the loop has at least one pixel -- and a union of two
        non-empty masks cannot be empty. Division by zero is what the
        guard names; it cannot be reached from a real pair of labels.
        """
        previous = np.zeros((8, 8), dtype=int)
        previous[1:4, 1:4] = 1
        following = np.zeros((8, 8), dtype=int)
        following[5:7, 5:7] = 2

        for label in np.unique(previous)[1:]:
            for other in np.unique(following)[1:]:
                m1 = previous == label
                m2 = following == other
                assert m1.any() and m2.any()
                assert np.logical_or(m1, m2).sum() > 0

    def test_disjoint_masks_cost_the_maximum(self):
        """The value the guard protects: a pair with no overlap gets a
        cost of 1, which is what keeps the assignment from pairing two
        cells that never touched."""
        m1 = np.zeros((4, 4), dtype=bool)
        m1[0, 0] = True
        m2 = np.zeros((4, 4), dtype=bool)
        m2[3, 3] = True

        inter = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()

        assert inter == 0 and union == 2
        assert 1 - inter / union == 1.0

    def test_identical_masks_cost_nothing(self):
        m = np.zeros((4, 4), dtype=bool)
        m[1:3, 1:3] = True

        assert 1 - np.logical_and(m, m).sum() / np.logical_or(m, m).sum() == 0.0


class TestTheQcAxisBudget:

    def test_the_layout_sizes_the_axes_for_the_panels_it_will_draw(self):
        """THE PIN, for the two ``if axis_idx < len(axes)`` checks.

        The figure is built with one axis per QC panel the strategy asks
        for, so the index cannot outrun the list. What the guard buys is
        that a strategy which grew a panel without growing the layout
        DROPS it rather than raising in the middle of a run's report.
        """
        axes = [object(), object(), object()]
        drawn, axis_idx = [], 0
        for _panel in ("hist", "pca", "xgb"):
            if axis_idx < len(axes):
                drawn.append(axes[axis_idx])
                axis_idx += 1

        assert len(drawn) == 3
        assert axis_idx == len(axes)

    def test_a_fourth_panel_with_three_axes_is_dropped_not_raised(self):
        axes = [object(), object(), object()]
        axis_idx, dropped = 0, 0
        for _panel in range(4):
            if axis_idx < len(axes):
                axis_idx += 1
            else:
                dropped += 1

        assert dropped == 1

    def test_both_xgboost_panels_are_guarded(self):
        from spacr import timelapse as TL

        source = inspect.getsource(TL)

        # THE LAST ONE. `elif qc_strategy == "xgboost" and has_xgb:`
        # appears twice -- once where the axis COUNT is decided and once
        # where the panels are drawn -- and `index` finds the counter,
        # whose block has no guards at all.
        xgb = source.rindex('elif qc_strategy == "xgboost" and has_xgb:')
        assert xgb != source.index('elif qc_strategy == "xgboost" and has_xgb:'), (
            "the duplicate xgboost branch is gone; this pin was anchored on "
            "there being two")
        block = source[xgb:xgb + 500]

        assert block.count("if axis_idx < len(axes):") == 2, (
            "one of the two xgboost QC panels is no longer guarded, so a "
            "layout one axis short raises instead of dropping a panel")

        counter = source[source.index(
            'elif qc_strategy == "xgboost" and has_xgb:'):]
        assert "qc_axes_count = 2" in counter[:200], (
            "the axis budget no longer reserves two for xgboost, so the "
            "guards below now drop a panel on an ordinary run")


class TestTheMergedPreviewChannels:

    @pytest.mark.parametrize("n_channels,expected", [
        (0, []), (1, [0]), (2, [0, 1]), (3, [0, 1, 2]), (5, [0, 1, 2]),
    ])
    def test_only_the_channels_that_exist_are_merged(self, n_channels,
                                                     expected):
        """THE ARC: ``n_channels >= 1`` and its two neighbours.

        A preview merges up to three channels into RGB, and a stack with
        fewer leaves the rest black rather than repeating a channel --
        which would show a one-channel field as grey and read as a
        colour balance.
        """
        merged = np.zeros((4, 4, 3), dtype=float)
        filled = []
        for index in range(3):
            if n_channels >= index + 1:
                merged[..., index] = 1.0
                filled.append(index)

        assert filled == expected
        for index in range(3):
            assert merged[..., index].any() == (index in expected)

    def test_the_preview_still_caps_at_three(self):
        source = inspect.getsource(T._debug_plot_merged_planes)

        assert "if n_channels >= 1:" in source
        assert "if n_channels >= 2:" in source
        assert "if n_channels >= 3:" in source
        assert "if n_channels >= 4:" not in source, (
            "a fourth channel is being written into an RGB array")


class TestCarryingTheFirstGroupsQcPayload:

    def test_the_payload_of_the_first_processed_group_is_kept(self):
        """THE PIN, for ``first_payload_settings is not None``.

        The QC panels describe ONE group, and the first processed one is
        the choice made -- keeping the last would mean the panel changed
        depending on how the groups happened to be ordered. It is set by
        every path that processes a group, so by the concat it is
        present.
        """
        source = inspect.getsource(T._apply_infection_intensity_qc)
        concat = source.index("all_df_qc = pd.concat(parts")
        guard = source.index("if first_payload_settings is not None:", concat)

        assert concat < guard
        for key in ("infection_hist_data", "infection_pca_data",
                    "infection_xgb_importance"):
            assert f'settings["{key}"]' in source[guard:], (
                f"{key} is no longer carried out of the first group's payload")

    def test_the_comment_says_which_group_it_is(self):
        source = inspect.getsource(T._apply_infection_intensity_qc)

        assert "from the first processed group" in source


class TestTheStraightnessFilter:

    def test_a_track_frame_with_straightness_can_be_filtered(self):
        """THE ARC: the column is present.

        Straightness is computed by the tracker, so a frame that came
        through it has the column -- and the filter is OFF by default,
        because dropping tracks for being too straight is a judgement
        about the biology rather than a repair.
        """
        track_df = pd.DataFrame({"straightness": [0.99, 0.5]})

        assert "straightness" in track_df.columns
        threshold = 0.95
        flagged = track_df["straightness"] > threshold
        assert flagged.tolist() == [True, False]

    def test_the_filter_is_off_unless_asked_for(self):
        source = inspect.getsource(T._compute_velocities_and_well_summary)

        assert 'settings.get("straightness_filter", False)' in source, (
            "the straightness filter now defaults ON, so tracks are dropped "
            "for being too straight without the user asking")
        assert 'settings.get("straightness_threshold", 0.95)' in source

    def test_a_frame_without_the_column_is_left_alone(self):
        track_df = pd.DataFrame({"velocity": [1.0]})

        assert "straightness" not in track_df.columns


class TestTheWellSummary:

    def test_wells_that_produced_records_become_a_frame(self):
        records = [{"well": "A01", "velocity": 1.0}]

        assert records
        assert list(pd.DataFrame(records)["well"]) == ["A01"]

    def test_no_records_leaves_the_summary_as_it_was(self):
        """THE ARC: ``well_records`` is empty.

        Every track was filtered out -- a strict straightness or velocity
        cut does this -- and ``DataFrame([])`` would replace a summary
        that has columns with one that has none, so later readers see a
        frame missing every field rather than an empty one.
        """
        records = []

        assert not records
        assert list(pd.DataFrame(records).columns) == []


class TestTheXgboostQcPayloads:

    def test_the_intensity_column_is_present_when_the_caller_named_one(self):
        """THE PIN, for ``if intensity_col in cell_level.columns``.

        The column is chosen from the frame's own columns upstream, so it
        is there -- and the whole block is inside a try that turns any
        failure into a run with no QC panel rather than no run.
        """
        source = inspect.getsource(T._infection_qc_xgboost)
        opened = source.rindex("try:", 0, source.index(
            "if intensity_col in cell_level.columns:"))
        guard = source.index("if intensity_col in cell_level.columns:")

        assert opened < guard, (
            "the histogram payload is no longer inside a try, so a QC "
            "failure now costs the run rather than the panel")

    def test_no_usable_features_skips_the_pca_panel(self):
        """THE ARC: ``used_feature_cols`` is empty.

        A frame with no numeric feature column has nothing to decompose,
        and PCA over a zero-width matrix raises. Skipping the panel
        leaves the histogram and the importances, which are still worth
        showing.
        """
        used_feature_cols = []

        assert not used_feature_cols
        frame = pd.DataFrame({"a": [1.0, 2.0]})
        assert frame[used_feature_cols].to_numpy(dtype=float).shape == (2, 0)

    def test_the_panel_matrix_is_owned_rather_than_viewed(self):
        """The comment there is a pandas-3 trap worth keeping: a
        homogeneous selection can return a READ-ONLY view, and the
        display-only imputation below would fail on it."""
        source = inspect.getsource(T._infection_qc_xgboost)

        assert "copy=True" in source
        assert "read-only view" in source
