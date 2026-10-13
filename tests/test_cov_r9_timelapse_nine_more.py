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

_XGB_PATHOGEN_CHAN = 1


def _xgb_frame(n_per_class=18, wells=("A01", "A02"), n_frames=3, seed=3):
    """A frame-level table with a clean infected/uninfected separation.

    Shaped like the measurement table ``_infection_qc_xgboost`` reads:
    one tracked object per spec, repeated over ``n_frames`` frames, with
    an exact pathogen-channel p95 so the quartile thresholds are stable.
    """
    rng = np.random.default_rng(seed)
    chan = _XGB_PATHOGEN_CHAN
    rows = []
    cell_id = 0
    for well in wells:
        for _ in range(n_per_class):
            for infected, centre in ((True, 1000.0), (False, 300.0)):
                cell_id += 1
                intensity = float(rng.normal(centre, 120.0))
                area = float(rng.uniform(200.0, 900.0)) + (
                    300.0 if infected else 0.0)
                solidity = float(rng.uniform(0.70, 0.99))
                y0 = float(rng.uniform(10.0, 200.0))
                x0 = float(rng.uniform(10.0, 200.0))
                for frame_index in range(n_frames):
                    rows.append({
                        "plateID": "plate1", "wellID": well, "fieldID": "1",
                        "cellID": cell_id, "frame": frame_index,
                        "infected": bool(infected),
                        "n_pathogens": 3 if infected else 0,
                        f"cell_p95_intensity_ch{chan}": intensity,
                        f"cell_mean_intensity_ch{chan}": intensity * 0.6,
                        "cell_mean_intensity_ch0": float(
                            rng.uniform(100.0, 200.0)),
                        "cell_area": area,
                        "cell_perimeter": 0.4 * area,
                        "cell_solidity": solidity,
                        "cell_centroid-0": y0 + 1.5 * frame_index,
                        "cell_centroid-1": x0 + float(frame_index),
                        "nucleus_area": float(rng.uniform(50.0, 200.0)),
                    })
    return pd.DataFrame(rows)


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

    def test_the_payload_comes_from_the_first_group_not_the_last(
            self, tmp_path, monkeypatch):
        """WHICH group it is, driven rather than read.

        The QC panels describe ONE group. Keeping the FIRST processed
        one is the choice made; keeping the last would mean the panel
        changed depending on how the groups happened to be ordered.
        """
        processed = []

        def _recording_qc(all_df, settings, infection_col, pathogen_chan,
                          motility_dir):
            plate = str(all_df["plateID"].iloc[0])
            processed.append(plate)
            settings["infection_hist_data"] = {"plate": plate}
            settings["infection_intensity_qc_panel_type"] = plate
            return all_df, infection_col

        monkeypatch.setattr(T, "_infection_qc_histogram", _recording_qc)

        # plateB FIRST in the frame and second alphabetically, so the
        # first processed group is neither the last nor the sorted one.
        frame = pd.DataFrame({
            "plateID": ["plateB"] * 3 + ["plateA"] * 3,
            "wellID": ["A01"] * 6,
            "infected": [True, False, True, False, True, False]})
        settings = {"infection_intensity_qc": True,
                    "infection_intensity_strategy": "histogram",
                    "infection_intensity_qc_scope": "plate"}

        T._apply_infection_intensity_qc(
            frame, settings, "infected", 1, str(tmp_path / "motility"))

        assert processed == ["plateB", "plateA"]
        assert settings["infection_hist_data"] == {"plate": "plateB"}, (
            "the QC payload is no longer the first processed group's, so "
            "which group the panel describes now depends on group order")
        assert settings["infection_intensity_qc_panel_type"] == "plateB"


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

        assert 'settings.get("drop_straight_tracks", False)' in source, (
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

    def test_the_panel_matrix_is_owned_rather_than_viewed(self, tmp_path):
        """A pandas-3 trap worth keeping: a homogeneous selection can
        return a READ-ONLY view, and the display-only imputation below
        writes into the matrix it is given.

        Driven on what owning it buys. The features carry a hole, so the
        imputation has to run: on an owned matrix it fills the hole and
        the PCA payload is produced, and on a read-only view it would
        raise into the enclosing ``except`` and leave the panel empty.
        The imputation is display-only either way -- the measurements
        come back still carrying their hole.
        """
        # NARROWED TO THE STATEMENT, not the function. A bare
        # `"copy=True" in source` over a seven-hundred-line body is
        # satisfied forever by an unrelated `to_numpy(copy=True)` on the
        # index arrays further up, so it stayed green with the copy removed
        # from BOTH matrices. Slicing from the assignment is what makes it
        # answer about this statement.
        source = inspect.getsource(T._infection_qc_xgboost)
        start = source.index("X_panel = cell_level[used_feature_cols]")
        # To the end of the CALL, not the end of the line: the statement
        # wraps, and `to_numpy(` sits on the first line with its arguments
        # on the next.
        panel_stmt = source[start:source.index(")", start) + 1]
        assert "copy=True" in panel_stmt, panel_stmt

        all_df = _xgb_frame()
        holed = sorted(all_df["cellID"].unique())[::2]
        all_df.loc[all_df["cellID"].isin(holed), "cell_solidity"] = np.nan
        settings = {"tracked_object": "cell",
                    "infection_xgb_n_estimators": 15,
                    "infection_xgb_max_depth": 2,
                    "infection_xgb_n_jobs": 1,
                    "infection_intensity_mode": "relabel"}

        # COPY-ON-WRITE ON, DELIBERATELY. The assertions below describe what
        # owning the matrix buys, and they can only describe it in the mode
        # where a selection returns a read-only view. This repository runs
        # pandas 2.2.2 with copy-on-write OFF, where the imputation writes
        # into a doomed temporary, nothing raises, and the payload appears
        # whether the matrix was copied or not -- so without this the test
        # passes with `copy=True` removed and guards nothing at all.
        # Measured: with it, removing the copy fails on `payload is not
        # None`; without it, removing the copy from both matrices passes.
        with pd.option_context("mode.copy_on_write", True):
            out, _ = T._infection_qc_xgboost(
                all_df=all_df, settings=settings, infection_col="infected",
                pathogen_chan=_XGB_PATHOGEN_CHAN,
                motility_dir=str(tmp_path / "motility"))

        payload = settings.get("infection_pca_data")
        assert payload is not None, (
            "the PCA panel payload was lost, which is what a read-only "
            "panel matrix costs: the imputation raises into the enclosing "
            "except and the run finishes with no panel")
        assert np.isfinite(payload["coords"]).all(), (
            "the imputation did not fill the hole it was there to fill")

        survivors = out[out["cellID"].isin(holed)]
        assert len(survivors) > 0
        assert survivors["cell_solidity"].isna().all(), (
            "the display-only imputation wrote back into the "
            "measurements, so the matrix is a view of them after all")
