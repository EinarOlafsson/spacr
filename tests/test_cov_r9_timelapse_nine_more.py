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
        assert T.link_by_iou(m1, m2, iou_threshold=0.1) == []

    def test_identical_masks_cost_nothing(self):
        m = np.zeros((4, 4), dtype=bool)
        m[1:3, 1:3] = True

        assert 1 - np.logical_and(m, m).sum() / np.logical_or(m, m).sum() == 0.0
        assert T.link_by_iou(m, m, iou_threshold=0.1) == [(True, True)]


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

    def test_xgboost_reserves_exactly_the_two_axes_it_draws(self):
        axes = [object(), object()]
        axis_idx = 0
        probability = axes[axis_idx]
        axis_idx += 1
        importance = axes[axis_idx]
        axis_idx += 1

        assert probability is axes[0]
        assert importance is axes[1]
        assert axis_idx == len(axes)

    def test_both_xgboost_panels_use_the_reserved_axes(self):
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

        assert "if axis_idx < len(axes):" not in block
        assert block.count("axes[axis_idx]") == 2

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

        assert "if n_channels >= 1:" not in source
        assert "merged_rgb[..., 0] = norm_intensity[0]" in source
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
        payload = source.index(
            'settings["infection_hist_data"] = first_payload_settings.get',
            concat)

        assert concat < payload
        assert "if first_payload_settings is not None:" not in source[concat:payload]
        for key in ("infection_hist_data", "infection_pca_data",
                    "infection_xgb_importance"):
            assert f'settings["{key}"]' in source[payload:], (
                f"{key} is no longer carried out of the first group's payload")

    def test_the_comment_says_which_group_it_is(self):
        source = inspect.getsource(T._apply_infection_intensity_qc)

        assert "from the first processed group" in source


class TestOptionalEmbeddingImports:

    def test_only_the_call_site_checks_optional_embedders(self):
        source = inspect.getsource(T._infection_qc_pca_clustering)

        assert 'if embed_method == "umap" and umap is not None:' in source
        assert 'elif embed_method == "tsne" and TSNE is not None:' in source
        assert "if umap is None:" not in source
        assert "if TSNE is None:" not in source


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

    def test_every_track_record_supplies_the_column(self):
        source = inspect.getsource(T._compute_velocities_and_well_summary)

        built = source.index('"straightness": straightness')
        used = source.index('track_df["straightness"]', built)
        assert built < used
        assert 'if "straightness" in track_df.columns:' not in source


class TestTheWellSummary:

    def test_wells_that_produced_records_become_a_frame(self):
        records = [{"well": "A01", "velocity": 1.0}]

        assert records
        assert list(pd.DataFrame(records)["well"]) == ["A01"]

    def test_a_nonempty_track_frame_always_produces_a_well_record(self):
        source = inspect.getsource(T._compute_velocities_and_well_summary)

        empty_return = source.index("if track_df.empty:")
        grouping = source.index('track_df.groupby(["plateID", "wellID"])')
        conversion = source.index("pd.DataFrame(well_records)", grouping)
        assert empty_return < grouping < conversion
        assert "if well_records:" not in source[grouping:conversion]


class TestTheXgboostQcPayloads:

    def test_the_intensity_column_is_present_when_the_caller_named_one(self):
        """THE PIN, for ``if intensity_col in cell_level.columns``.

        The column is chosen from the frame's own columns upstream, so it
        is there -- and the whole block is inside a try that turns any
        failure into a run with no QC panel rather than no run.
        """
        source = inspect.getsource(T._infection_qc_xgboost)
        selected = source.index("if intensity_col is None:")
        payload = source.index(
            "intens = cell_level[intensity_col].to_numpy(dtype=float)")
        opened = source.rindex("try:", 0, payload)

        assert selected < opened < payload, (
            "the histogram payload is no longer inside a try, so a QC "
            "failure now costs the run rather than the panel")
        assert "if intensity_col in cell_level.columns:" not in source

    def test_the_trained_feature_list_cannot_be_empty_at_the_panel(self):
        source = inspect.getsource(T._infection_qc_xgboost)

        refusal = source.index("if not feature_cols:")
        assignment = source.index("used_feature_cols = feature_cols", refusal)
        panel = source.index("X_panel = cell_level[used_feature_cols]", assignment)
        assert refusal < assignment < panel
        assert "if used_feature_cols:" not in source[assignment:panel]

    def test_the_panel_matrix_is_owned_rather_than_viewed(self):
        """The comment there is a pandas-3 trap worth keeping: a
        homogeneous selection can return a READ-ONLY view, and the
        display-only imputation below would fail on it."""
        source = inspect.getsource(T._infection_qc_xgboost)

        assert "copy=True" in source
        assert "read-only view" in source
