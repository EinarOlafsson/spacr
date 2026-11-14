"""Branch coverage for the read/score preparation helpers in :mod:`spacr.ml`.

Covers :func:`spacr.ml.process_reads` (CSV loading, ``plate_row``/``prcfo``
identifier splitting, plate stamping, row filtering, fraction-threshold
validation and the gRNA -> org/gene/grna split), :func:`check_normality`'s
verbose branches, :func:`clean_controls` and the non-``prcfo`` metadata path
plus the inversion / normality branches of :func:`process_scores`.

Everything here is CPU-only, offline and operates on tiny synthetic frames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _close_figs():
    """Never let a figure leak out of a test in this module."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

GRNAS = ["TGGT1_GENEA_g1", "TGGT1_GENEA_g2", "TGGT1_GENEB_g1", "TGGT1_GENEB_g2"]


def _reads_frame(plate_col=True, plate_value="plate1"):
    """4 gRNAs x 4 wells with counts 10/20/30/40 -> fractions .1/.2/.3/.4."""
    rows = []
    for row in ("r1", "r2"):
        for col in ("c1", "c2"):
            for i, grna in enumerate(GRNAS):
                rec = {
                    "rowID": row,
                    "columnID": col,
                    "grna": grna,
                    "count": 10 * (i + 1),
                }
                if plate_col:
                    rec["plateID"] = plate_value
                rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# process_reads — CSV path, fraction threshold, org/gene/grna split
# ---------------------------------------------------------------------------

def case_process_reads_from_csv_path_splits_grna_and_filters(tmp_path, capsys):
    """A CSV path is read from disk; the org prefix is stripped from the gRNA."""
    _seen = []
    from spacr.ml import process_reads

    csv = tmp_path / "counts.csv"
    _reads_frame().to_csv(csv, index=False)

    out = process_reads(str(csv), fraction_threshold=0.15, plate="ignored")

    # 4 wells x 4 gRNAs, the 0.1-fraction gRNA dropped from every well
    _seen.append(keep(list(out.columns)))
    _seen.append(keep(len(out)))
    _seen.append(keep(set(out["prc"])))
    # 'TGGT1_GENEA_g1' -> gene GENEA, grna 'GENEA_g1'
    _seen.append(keep(set(out["gene"])))
    _seen.append(keep(set(out["grna"])))
    per_well = out.groupby("prc")["fraction"].sum()
    _seen.append(bool(
        np.allclose(per_well.to_numpy(), 0.9)))

    msg = capsys.readouterr().out
    _seen.append(('Removed 4 of 16 observations' if 'Removed 4 of 16 observations' in (msg) else ("missing", 'Removed 4 of 16 observations')))
    _seen.append(('75.0% retained' if '75.0% retained' in (msg) else ("missing", '75.0% retained')))
    return _seen


def case_process_reads_no_plate_id_column_uses_plate_argument():
    """Without a plateID column the explicit ``plate`` argument is stamped on."""
    _seen = []
    from spacr.ml import process_reads

    df = _reads_frame(plate_col=False)
    out = process_reads(df, fraction_threshold=None, plate="plateX")

    _seen.append(bool(
        out["prc"].str.startswith("plateX_").all()))
    _seen.append(keep(sorted(out["prc"].unique())))
    return _seen


def case_process_reads_no_plate_id_and_no_plate_defaults_to_plate1():
    """plate=None with no plateID column falls back to the literal 'plate1'."""
    _seen = []
    from spacr.ml import process_reads

    out = process_reads(_reads_frame(plate_col=False), fraction_threshold=None,
                        plate=None)

    _seen.append(keep(set(out["prc"])))
    return _seen


def case_process_reads_plate_row_column_is_split():
    """A combined ``plate_row`` column is split into plateID / rowID."""
    _seen = []
    from spacr.ml import process_reads

    df = pd.DataFrame({
        "plate_row": ["plate2_rA"] * 4 + ["plate2_rB"] * 4,
        "columnID": ["c3"] * 8,
        "grna": GRNAS * 2,
        "count": [10, 20, 30, 40] * 2,
    })
    out = process_reads(df, fraction_threshold=None, plate=None)

    _seen.append(keep(set(out["prc"])))
    _seen.append(bool(
        np.allclose(sorted(out.loc[out["prc"] == "plate2_rA_c3", "fraction"]),
                       [0.1, 0.2, 0.3, 0.4])))
    return _seen


def case_process_reads_prcfo_column_overrides_identifiers():
    """A ``prcfo`` column supplies plate/row/column/field/object in one go."""
    _seen = []
    from spacr.ml import process_reads

    df = pd.DataFrame({
        "prcfo": [f"plate3_rB_c5_f1_o{i}" for i in range(4)]
                 + [f"plate3_rC_c5_f1_o{i}" for i in range(4)],
        "grna": GRNAS * 2,
        "count": [10, 20, 30, 40] * 2,
    })
    out = process_reads(df, fraction_threshold=None, plate=None)

    _seen.append(keep(set(out["prc"])))
    _seen.append(keep(len(out)))
    return _seen


def case_process_reads_scalar_filter_column_and_value_drop_rows(capsys):
    """String ``filter_column``/``filter_value`` are promoted to lists."""
    _seen = []
    from spacr.ml import process_reads

    out = process_reads(_reads_frame(), fraction_threshold=None,
                        plate=None, filter_column="rowID", filter_value="r2")

    _seen.append(keep(set(out["prc"])))
    _seen.append(keep(len(out)))
    capsys.readouterr()
    return _seen


def case_process_reads_list_filters_drop_several_values():
    """List filters are applied column-by-column and value-by-value."""
    _seen = []
    from spacr.ml import process_reads

    out = process_reads(_reads_frame(), fraction_threshold=None, plate=None,
                        filter_column=["rowID", "columnID"],
                        filter_value=["r1", "c2"])

    _seen.append(keep(set(out["prc"])))
    _seen.append(keep(len(out)))
    return _seen


def case_process_reads_missing_required_columns_raises():
    """Missing rowID/columnID is reported as a ValueError, not a KeyError."""
    _seen = []
    from spacr.ml import process_reads

    df = pd.DataFrame({"grna": GRNAS, "count": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="must contain") as _exc1:
        process_reads(df, fraction_threshold=None, plate="plate1")
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


@pytest.mark.parametrize("threshold", [1.5, -0.25])
def test_process_reads_out_of_range_threshold_raises(threshold):
    """The fraction is a relative abundance: thresholds outside [0, 1] are rejected."""
    from spacr.ml import process_reads

    with pytest.raises(ValueError, match=r"outside the valid range \[0, 1\]"):
        process_reads(_reads_frame(), fraction_threshold=threshold, plate=None)


def case_process_reads_threshold_removing_everything_raises(capsys):
    """A threshold above the maximum observed fraction is a hard error."""
    _seen = []
    from spacr.ml import process_reads

    with pytest.raises(ValueError) as excinfo:
        process_reads(_reads_frame(), fraction_threshold=0.99, plate=None)
    _seen.append(('ValueError' if isinstance(excinfo.value, ValueError) else type(excinfo.value).__name__))

    _seen.append(('All 16 rows were removed' if 'All 16 rows were removed' in (str(excinfo.value)) else ("missing", 'All 16 rows were removed')))
    _seen.append(('median 0.25' if 'median 0.25' in (str(excinfo.value)) else ("missing", 'median 0.25')))
    _seen.append(('0.0% retained' if '0.0% retained' in (capsys.readouterr().out) else ("missing", '0.0% retained')))
    return _seen


# ---------------------------------------------------------------------------
# check_normality — verbose branches
# ---------------------------------------------------------------------------

def case_check_normality_verbose_non_normal_prints_and_returns_false(capsys):
    """Strongly bimodal data is reported as non-normal when verbose."""
    _seen = []
    from spacr.ml import check_normality

    data = np.concatenate([np.zeros(25), np.ones(25) * 100.0])
    got = check_normality(data, "bimodal", verbose=True)

    out = capsys.readouterr().out
    _seen.append(bool(
        got is False))
    _seen.append(('Shapiro-Wilk Test for bimodal' if 'Shapiro-Wilk Test for bimodal' in (out) else ("missing", 'Shapiro-Wilk Test for bimodal')))
    _seen.append(('the data for bimodal is not normally distributed' if 'the data for bimodal is not normally distributed' in (out.lower()) else ("missing", 'the data for bimodal is not normally distributed')))
    return _seen


def case_check_normality_verbose_normal_prints_and_returns_true(rng, capsys):
    """Gaussian data is reported as normal when verbose."""
    _seen = []
    from spacr.ml import check_normality

    got = check_normality(rng.normal(0.0, 1.0, 200), "gauss", verbose=True)

    out = capsys.readouterr().out
    _seen.append(bool(
        got is True))
    _seen.append(('is normally distributed' if 'is normally distributed' in (out) else ("missing", 'is normally distributed')))
    return _seen


# ---------------------------------------------------------------------------
# clean_controls
# ---------------------------------------------------------------------------

def case_clean_controls_non_list_value_is_a_no_op():
    """A bare (non-list) value is ignored — only lists are removed."""
    _seen = []
    from spacr.ml import clean_controls

    df = pd.DataFrame({"grp": ["a", "b", "c"], "v": [1, 2, 3]})
    out = clean_controls(df, "a", "grp")

    pd.testing.assert_frame_equal(out, df)
    return _seen


def case_clean_controls_removes_each_listed_value(capsys):
    """Every listed value is dropped and announced."""
    _seen = []
    from spacr.ml import clean_controls

    df = pd.DataFrame({"grp": ["a", "b", "c", "a"], "v": [1, 2, 3, 4]})
    out = clean_controls(df, ["a", "c"], "grp")

    _seen.append(keep(list(out["grp"])))
    printed = capsys.readouterr().out
    _seen.append(('Removed data from a' if 'Removed data from a' in (printed) else ("missing", 'Removed data from a')))
    _seen.append(('Removed data from c' if 'Removed data from c' in (printed) else ("missing", 'Removed data from c')))
    return _seen


def case_clean_controls_none_column_is_a_no_op():
    """column=None means 'do not filter'; perform_regression passes
    settings['filter_column'] straight through and None is a legal value."""
    _seen = []
    from spacr.ml import clean_controls

    df = pd.DataFrame({"grp": ["a", "b", "c"], "v": [1, 2, 3]})
    out = clean_controls(df, ["a"], None)

    pd.testing.assert_frame_equal(out, df)
    return _seen


def case_clean_controls_accepts_a_list_of_columns():
    """process_reads documents a list of filter columns, so the well-level
    control filter has to accept the same shape rather than raising
    'unhashable type: list' on the membership test."""
    _seen = []
    from spacr.ml import clean_controls

    df = pd.DataFrame({
        "rowID": ["r1", "r2", "r3", "r4"],
        "columnID": ["c1", "c2", "c3", "c4"],
        "v": [1, 2, 3, 4],
    })
    out = clean_controls(df, ["r1", "c3"], ["rowID", "columnID"])

    # r1 dropped via rowID, c3 dropped via columnID.
    _seen.append(keep(list(out["v"])))
    return _seen


def case_clean_controls_list_skips_columns_the_frame_lacks():
    """Unknown names in the column list are ignored, not an error."""
    _seen = []
    from spacr.ml import clean_controls

    df = pd.DataFrame({"grp": ["a", "b"], "v": [1, 2]})
    out = clean_controls(df, ["a"], ["grp", "not_a_column"])

    _seen.append(keep(list(out["grp"])))
    return _seen


# ---------------------------------------------------------------------------
# process_scores — the non-prcfo metadata branch
# ---------------------------------------------------------------------------

def _wells_df(n_rows=6, per_well=8, plate="p1", seed=3, low=0.2, high=0.8):
    rng = np.random.default_rng(seed)
    recs = []
    plates = plate if isinstance(plate, (list, tuple)) else [plate]
    for p in plates:
        for w in range(n_rows):
            for _ in range(per_well):
                recs.append({
                    "plateID": p,
                    "rowID": f"r{w + 1}",
                    "columnID": "c1",
                    "pred": float(rng.uniform(low, high)),
                })
    return pd.DataFrame(recs)


def case_process_scores_stamps_plate_on_single_plate_frame():
    """A single-plate frame has its plateID replaced by the ``plate`` argument."""
    _seen = []
    from spacr.ml import process_scores

    out, dv = process_scores(_wells_df(), "pred", plate="p9",
                             min_cells_per_well=4, agg_type="mean")

    _seen.append(keep(dv))
    _seen.append(bool(
        out["prc"].str.startswith("p9_").all()))
    _seen.append(keep(len(out)))
    _seen.append(bool(
        (out["cell_count"] == 8).all()))
    return _seen


def case_process_scores_multi_plate_frame_ignores_plate_argument(capsys):
    """Two plates must not be collapsed onto one prc by the ``plate`` argument."""
    _seen = []
    from spacr.ml import process_scores

    df = _wells_df(n_rows=3, per_well=8, plate=["p1", "p2"])
    out, _ = process_scores(df, "pred", plate="pZ", min_cells_per_well=4,
                            agg_type="median")

    printed = capsys.readouterr().out
    _seen.append(("Ignoring the 'plate' argument" if "Ignoring the 'plate' argument" in (printed) else ("missing", "Ignoring the 'plate' argument")))
    _seen.append(('already contains 2 distinct plateIDs' if 'already contains 2 distinct plateIDs' in (printed) else ("missing", 'already contains 2 distinct plateIDs')))
    _seen.append(keep(set(out["prc"])))
    _seen.append(bool(
        not out["prc"].str.startswith("pZ").any()))
    return _seen


def case_process_scores_without_plate_id_or_plate_raises():
    """No plateID column and no plate argument is a hard, explicit error."""
    _seen = []
    from spacr.ml import process_scores

    df = _wells_df().drop(columns=["plateID"])
    with pytest.raises(ValueError, match="no usable 'plateID' column") as _exc1:
        process_scores(df, "pred", plate=None, min_cells_per_well=1)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_process_scores_all_nan_plate_id_and_no_plate_raises():
    """An all-NaN plateID column is just as unusable as a missing one."""
    _seen = []
    from spacr.ml import process_scores

    df = _wells_df()
    df["plateID"] = pd.Series([None] * len(df), dtype=object)
    with pytest.raises(ValueError, match="no usable 'plateID' column") as _exc1:
        process_scores(df, "pred", plate=None, min_cells_per_well=1)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_process_scores_missing_column_id_raises():
    """plateID alone is not enough — rowID and columnID are required too."""
    _seen = []
    from spacr.ml import process_scores

    df = _wells_df().drop(columns=["columnID"])
    with pytest.raises(ValueError,
                       match="must contain 'plateID', 'rowID', and 'columnID'") as _exc1:
        process_scores(df, "pred", plate=None, min_cells_per_well=1)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_process_scores_reciprocal_inversion_warns_and_drops_zeros(capsys):
    """1/x inversion turns zeros into NaN, warns, and drops those objects."""
    _seen = []
    from spacr.ml import process_scores

    df = _wells_df(n_rows=6, per_well=10, seed=11, low=0.25, high=0.75)
    zero_idx = df.index[df["rowID"] == "r1"][:3]
    df.loc[zero_idx, "pred"] = 0.0

    out, dv = process_scores(df, "pred", plate="p1", min_cells_per_well=5,
                             agg_type="mean", invert_dependent_variable=-1)

    printed = capsys.readouterr().out
    _seen.append(('contains 3 zero' if 'contains 3 zero' in (printed) else ("missing", 'contains 3 zero')))
    _seen.append(("Inverted 'pred' as 1/x on raw values." if "Inverted 'pred' as 1/x on raw values." in (printed) else ("missing", "Inverted 'pred' as 1/x on raw values.")))
    _seen.append(keep(dv))
    counts = dict(zip(out["prc"], out["cell_count"]))
    _seen.append(keep(counts["p1_r1_c1"]))
    _seen.append(bool(
        all(counts[f"p1_r{i}_c1"] == 10 for i in range(2, 7))))
    # every aggregated value is a mean of reciprocals of values in [0.25, 0.75]
    _seen.append(bool(
        (out["pred"] >= 1.0 / 0.75).all()))
    return _seen


def case_process_scores_complement_inversion_matches_manual_mean(capsys):
    """1 - x inversion is applied before aggregation."""
    _seen = []
    from spacr.ml import process_scores

    df = _wells_df(n_rows=4, per_well=6, seed=5)
    out, _ = process_scores(df, "pred", plate="p1", min_cells_per_well=2,
                            agg_type="mean", invert_dependent_variable=True)

    _seen.append(("Inverted 'pred' as 1 - x on raw values." if "Inverted 'pred' as 1 - x on raw values." in (capsys.readouterr().out) else ("missing", "Inverted 'pred' as 1 - x on raw values.")))
    expected = (1.0 - df["pred"]).groupby(
        df["plateID"] + "_" + df["rowID"] + "_" + df["columnID"]).mean()
    got = out.set_index("prc")["pred"]
    _seen.append(bool(
        np.allclose(got.loc[expected.index].to_numpy(), expected.to_numpy())))
    return _seen


def case_process_scores_reports_non_normal_response(capsys):
    """A bimodal per-well response is reported as not normally distributed."""
    _seen = []
    from spacr.ml import process_scores

    rng = np.random.default_rng(17)
    recs = []
    for w in range(30):
        centre = 0.05 if w % 2 == 0 else 0.95
        for _ in range(6):
            recs.append({
                "plateID": "p1",
                "rowID": f"r{w + 1}",
                "columnID": "c1",
                "pred": float(np.clip(centre + rng.normal(0, 0.002), 0.001, 0.999)),
            })
    df = pd.DataFrame(recs)

    out, dv = process_scores(df, "pred", plate="p1", min_cells_per_well=3,
                             agg_type="mean")

    printed = capsys.readouterr().out
    _seen.append(('pred is not normally distributed' if 'pred is not normally distributed' in (printed) else ("missing", 'pred is not normally distributed')))
    _seen.append(keep(dv))
    _seen.append(keep(len(out)))
    return _seen


def case_process_scores_transform_renames_response_and_rechecks_normality(capsys):
    """A transform adds a '<transform>_<var>' column and re-runs the normality test."""
    _seen = []
    from spacr.ml import process_scores

    rng = np.random.default_rng(23)
    recs = []
    for w in range(24):
        centre = 0.02 if w % 2 == 0 else 0.9
        for _ in range(5):
            recs.append({
                "plateID": "p1",
                "rowID": f"r{w + 1}",
                "columnID": "c1",
                "pred": float(np.clip(centre + rng.normal(0, 0.003), 0.001, 0.999)),
            })
    df = pd.DataFrame(recs)

    out, dv = process_scores(df, "pred", plate="p1", min_cells_per_well=3,
                             agg_type="mean", transform="sqrt")

    _seen.append(keep(dv))
    _seen.append(('sqrt_pred' if 'sqrt_pred' in (out.columns) else ("missing", 'sqrt_pred')))
    _seen.append(bool(
        np.allclose(out["sqrt_pred"].to_numpy(),
                       np.sqrt(out["pred"].to_numpy()))))
    _seen.append(('sqrt_pred is not normally distributed' if 'sqrt_pred is not normally distributed' in (capsys.readouterr().out) else ("missing", 'sqrt_pred is not normally distributed')))
    return _seen


def case_process_scores_min_cells_per_well_drops_small_wells():
    """Wells below min_cells_per_well never reach the response frame."""
    _seen = []
    from spacr.ml import process_scores

    df = _wells_df(n_rows=5, per_well=10, seed=31)
    # shrink one well to 2 objects
    small = df.index[df["rowID"] == "r5"][2:]
    df = df.drop(index=small).reset_index(drop=True)

    out, _ = process_scores(df, "pred", plate="p1", min_cells_per_well=5,
                            agg_type="quantile")

    _seen.append((("absent", 'p1_r5_c1') if 'p1_r5_c1' not in (set(out["prc"])) else ("present", 'p1_r5_c1')))
    _seen.append(keep(len(out)))
    _seen.append(bool(
        (out["cell_count"] == 10).all()))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_process_reads_from_csv_path_splits_grna_and_filters, [['prc', 'grna', 'fraction', 'gene'], 12, {'plate1_r1_c1', 'plate1_r1_c2', 'plate1_r2_c1', 'plate1_r2_c2'}, {'GENEA', 'GENEB'}, {'GENEA_g2', 'GENEB_g1', 'GENEB_g2'}, True, 'Removed 4 of 16 observations', '75.0% retained'], id="process_reads_from_csv_path_splits_grna_and_filters"),
    pytest.param(case_process_reads_no_plate_id_column_uses_plate_argument, [True, ['plateX_r1_c1', 'plateX_r1_c2', 'plateX_r2_c1', 'plateX_r2_c2']], id="process_reads_no_plate_id_column_uses_plate_argument"),
    pytest.param(case_process_reads_no_plate_id_and_no_plate_defaults_to_plate1, [{'plate1_r1_c1', 'plate1_r1_c2', 'plate1_r2_c1', 'plate1_r2_c2'}], id="process_reads_no_plate_id_and_no_plate_defaults_to_plate1"),
    pytest.param(case_process_reads_plate_row_column_is_split, [{'plate2_rA_c3', 'plate2_rB_c3'}, True], id="process_reads_plate_row_column_is_split"),
    pytest.param(case_process_reads_prcfo_column_overrides_identifiers, [{'plate3_rB_c5', 'plate3_rC_c5'}, 8], id="process_reads_prcfo_column_overrides_identifiers"),
    pytest.param(case_process_reads_scalar_filter_column_and_value_drop_rows, [{'plate1_r1_c1', 'plate1_r1_c2'}, 8], id="process_reads_scalar_filter_column_and_value_drop_rows"),
    pytest.param(case_process_reads_list_filters_drop_several_values, [{'plate1_r2_c1'}, 4], id="process_reads_list_filters_drop_several_values"),
    pytest.param(case_process_reads_missing_required_columns_raises, ['ValueError'], id="process_reads_missing_required_columns_raises"),
    pytest.param(case_process_reads_threshold_removing_everything_raises, ['ValueError', 'All 16 rows were removed', 'median 0.25', '0.0% retained'], id="process_reads_threshold_removing_everything_raises"),
    pytest.param(case_process_scores_stamps_plate_on_single_plate_frame, ['pred', True, 6, True], id="process_scores_stamps_plate_on_single_plate_frame"),
    pytest.param(case_process_scores_multi_plate_frame_ignores_plate_argument, ["Ignoring the 'plate' argument", 'already contains 2 distinct plateIDs', {'p1_r1_c1', 'p1_r2_c1', 'p1_r3_c1', 'p2_r1_c1', 'p2_r2_c1', 'p2_r3_c1'}, True], id="process_scores_multi_plate_frame_ignores_plate_argument"),
    pytest.param(case_process_scores_without_plate_id_or_plate_raises, ['ValueError'], id="process_scores_without_plate_id_or_plate_raises"),
    pytest.param(case_process_scores_all_nan_plate_id_and_no_plate_raises, ['ValueError'], id="process_scores_all_nan_plate_id_and_no_plate_raises"),
    pytest.param(case_process_scores_missing_column_id_raises, ['ValueError'], id="process_scores_missing_column_id_raises"),
    pytest.param(case_process_scores_reciprocal_inversion_warns_and_drops_zeros, ['contains 3 zero', "Inverted 'pred' as 1/x on raw values.", 'pred', 7, True, True], id="process_scores_reciprocal_inversion_warns_and_drops_zeros"),
    pytest.param(case_process_scores_complement_inversion_matches_manual_mean, ["Inverted 'pred' as 1 - x on raw values.", True], id="process_scores_complement_inversion_matches_manual_mean"),
    pytest.param(case_process_scores_reports_non_normal_response, ['pred is not normally distributed', 'pred', 30], id="process_scores_reports_non_normal_response"),
    pytest.param(case_process_scores_transform_renames_response_and_rechecks_normality, ['sqrt_pred', 'sqrt_pred', True, 'sqrt_pred is not normally distributed'], id="process_scores_transform_renames_response_and_rechecks_normality"),
    pytest.param(case_process_scores_min_cells_per_well_drops_small_wells, [("absent", 'p1_r5_c1'), 4, True], id="process_scores_min_cells_per_well_drops_small_wells"),
    pytest.param(case_check_normality_verbose_non_normal_prints_and_returns_false, [True, 'Shapiro-Wilk Test for bimodal', 'the data for bimodal is not normally distributed'], id="check_normality_verbose_non_normal_prints_and_returns_false"),
    pytest.param(case_check_normality_verbose_normal_prints_and_returns_true, [True, 'is normally distributed'], id="check_normality_verbose_normal_prints_and_returns_true"),
    pytest.param(case_clean_controls_non_list_value_is_a_no_op, [], id="clean_controls_non_list_value_is_a_no_op"),
    pytest.param(case_clean_controls_removes_each_listed_value, [['b'], 'Removed data from a', 'Removed data from c'], id="clean_controls_removes_each_listed_value"),
    pytest.param(case_clean_controls_none_column_is_a_no_op, [], id="clean_controls_none_column_is_a_no_op"),
    pytest.param(case_clean_controls_accepts_a_list_of_columns, [[2, 4]], id="clean_controls_accepts_a_list_of_columns"),
    pytest.param(case_clean_controls_list_skips_columns_the_frame_lacks, [['b']], id="clean_controls_list_skips_columns_the_frame_lacks"),
])
def test_ml_reads_scores_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected


