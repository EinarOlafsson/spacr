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

def test_process_reads_from_csv_path_splits_grna_and_filters(tmp_path, capsys):
    """A CSV path is read from disk; the org prefix is stripped from the gRNA."""
    from spacr.ml import process_reads

    csv = tmp_path / "counts.csv"
    _reads_frame().to_csv(csv, index=False)

    out = process_reads(str(csv), fraction_threshold=0.15, plate="ignored")

    # 4 wells x 4 gRNAs, the 0.1-fraction gRNA dropped from every well
    assert list(out.columns) == ["prc", "grna", "fraction", "gene"]
    assert len(out) == 12
    assert set(out["prc"]) == {
        "plate1_r1_c1", "plate1_r1_c2", "plate1_r2_c1", "plate1_r2_c2",
    }
    # 'TGGT1_GENEA_g1' -> gene GENEA, grna 'GENEA_g1'
    assert set(out["gene"]) == {"GENEA", "GENEB"}
    assert set(out["grna"]) == {"GENEA_g2", "GENEB_g1", "GENEB_g2"}
    per_well = out.groupby("prc")["fraction"].sum()
    assert np.allclose(per_well.to_numpy(), 0.9)

    msg = capsys.readouterr().out
    assert "Removed 4 of 16 observations" in msg
    assert "75.0% retained" in msg


def test_process_reads_no_plate_id_column_uses_plate_argument():
    """Without a plateID column the explicit ``plate`` argument is stamped on."""
    from spacr.ml import process_reads

    df = _reads_frame(plate_col=False)
    out = process_reads(df, fraction_threshold=None, plate="plateX")

    assert out["prc"].str.startswith("plateX_").all()
    assert sorted(out["prc"].unique()) == [
        "plateX_r1_c1", "plateX_r1_c2", "plateX_r2_c1", "plateX_r2_c2",
    ]


def test_process_reads_no_plate_id_and_no_plate_defaults_to_plate1():
    """plate=None with no plateID column falls back to the literal 'plate1'."""
    from spacr.ml import process_reads

    out = process_reads(_reads_frame(plate_col=False), fraction_threshold=None,
                        plate=None)

    assert set(out["prc"]) == {
        "plate1_r1_c1", "plate1_r1_c2", "plate1_r2_c1", "plate1_r2_c2",
    }


def test_process_reads_plate_row_column_is_split():
    """A combined ``plate_row`` column is split into plateID / rowID."""
    from spacr.ml import process_reads

    df = pd.DataFrame({
        "plate_row": ["plate2_rA"] * 4 + ["plate2_rB"] * 4,
        "columnID": ["c3"] * 8,
        "grna": GRNAS * 2,
        "count": [10, 20, 30, 40] * 2,
    })
    out = process_reads(df, fraction_threshold=None, plate=None)

    assert set(out["prc"]) == {"plate2_rA_c3", "plate2_rB_c3"}
    assert np.allclose(sorted(out.loc[out["prc"] == "plate2_rA_c3", "fraction"]),
                       [0.1, 0.2, 0.3, 0.4])


def test_process_reads_prcfo_column_overrides_identifiers():
    """A ``prcfo`` column supplies plate/row/column/field/object in one go."""
    from spacr.ml import process_reads

    df = pd.DataFrame({
        "prcfo": [f"plate3_rB_c5_f1_o{i}" for i in range(4)]
                 + [f"plate3_rC_c5_f1_o{i}" for i in range(4)],
        "grna": GRNAS * 2,
        "count": [10, 20, 30, 40] * 2,
    })
    out = process_reads(df, fraction_threshold=None, plate=None)

    assert set(out["prc"]) == {"plate3_rB_c5", "plate3_rC_c5"}
    assert len(out) == 8


def test_process_reads_scalar_filter_column_and_value_drop_rows(capsys):
    """String ``filter_column``/``filter_value`` are promoted to lists."""
    from spacr.ml import process_reads

    out = process_reads(_reads_frame(), fraction_threshold=None,
                        plate=None, filter_column="rowID", filter_value="r2")

    assert set(out["prc"]) == {"plate1_r1_c1", "plate1_r1_c2"}
    assert len(out) == 8
    capsys.readouterr()


def test_process_reads_list_filters_drop_several_values():
    """List filters are applied column-by-column and value-by-value."""
    from spacr.ml import process_reads

    out = process_reads(_reads_frame(), fraction_threshold=None, plate=None,
                        filter_column=["rowID", "columnID"],
                        filter_value=["r1", "c2"])

    assert set(out["prc"]) == {"plate1_r2_c1"}
    assert len(out) == 4


def test_process_reads_missing_required_columns_raises():
    """Missing rowID/columnID is reported as a ValueError, not a KeyError."""
    from spacr.ml import process_reads

    df = pd.DataFrame({"grna": GRNAS, "count": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="must contain"):
        process_reads(df, fraction_threshold=None, plate="plate1")


@pytest.mark.parametrize("threshold", [1.5, -0.25])
def test_process_reads_out_of_range_threshold_raises(threshold):
    """The fraction is a relative abundance: thresholds outside [0, 1] are rejected."""
    from spacr.ml import process_reads

    with pytest.raises(ValueError, match=r"outside the valid range \[0, 1\]"):
        process_reads(_reads_frame(), fraction_threshold=threshold, plate=None)


def test_process_reads_threshold_removing_everything_raises(capsys):
    """A threshold above the maximum observed fraction is a hard error."""
    from spacr.ml import process_reads

    with pytest.raises(ValueError) as excinfo:
        process_reads(_reads_frame(), fraction_threshold=0.99, plate=None)

    assert "All 16 rows were removed" in str(excinfo.value)
    assert "median 0.25" in str(excinfo.value)
    assert "0.0% retained" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# check_normality — verbose branches
# ---------------------------------------------------------------------------

def test_check_normality_verbose_non_normal_prints_and_returns_false(capsys):
    """Strongly bimodal data is reported as non-normal when verbose."""
    from spacr.ml import check_normality

    data = np.concatenate([np.zeros(25), np.ones(25) * 100.0])
    got = check_normality(data, "bimodal", verbose=True)

    out = capsys.readouterr().out
    assert got is False
    assert "Shapiro-Wilk Test for bimodal" in out
    assert "the data for bimodal is not normally distributed" in out.lower()


def test_check_normality_verbose_normal_prints_and_returns_true(rng, capsys):
    """Gaussian data is reported as normal when verbose."""
    from spacr.ml import check_normality

    got = check_normality(rng.normal(0.0, 1.0, 200), "gauss", verbose=True)

    out = capsys.readouterr().out
    assert got is True
    assert "is normally distributed" in out


# ---------------------------------------------------------------------------
# clean_controls
# ---------------------------------------------------------------------------

def test_clean_controls_non_list_value_is_a_no_op():
    """A bare (non-list) value is ignored — only lists are removed."""
    from spacr.ml import clean_controls

    df = pd.DataFrame({"grp": ["a", "b", "c"], "v": [1, 2, 3]})
    out = clean_controls(df, "a", "grp")

    pd.testing.assert_frame_equal(out, df)


def test_clean_controls_removes_each_listed_value(capsys):
    """Every listed value is dropped and announced."""
    from spacr.ml import clean_controls

    df = pd.DataFrame({"grp": ["a", "b", "c", "a"], "v": [1, 2, 3, 4]})
    out = clean_controls(df, ["a", "c"], "grp")

    assert list(out["grp"]) == ["b"]
    printed = capsys.readouterr().out
    assert "Removed data from a" in printed
    assert "Removed data from c" in printed


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


def test_process_scores_stamps_plate_on_single_plate_frame():
    """A single-plate frame has its plateID replaced by the ``plate`` argument."""
    from spacr.ml import process_scores

    out, dv = process_scores(_wells_df(), "pred", plate="p9",
                             min_cell_count=4, agg_type="mean")

    assert dv == "pred"
    assert out["prc"].str.startswith("p9_").all()
    assert len(out) == 6
    assert (out["cell_count"] == 8).all()


def test_process_scores_multi_plate_frame_ignores_plate_argument(capsys):
    """Two plates must not be collapsed onto one prc by the ``plate`` argument."""
    from spacr.ml import process_scores

    df = _wells_df(n_rows=3, per_well=8, plate=["p1", "p2"])
    out, _ = process_scores(df, "pred", plate="pZ", min_cell_count=4,
                            agg_type="median")

    printed = capsys.readouterr().out
    assert "Ignoring the 'plate' argument" in printed
    assert "already contains 2 distinct plateIDs" in printed
    assert set(out["prc"]) == {
        "p1_r1_c1", "p1_r2_c1", "p1_r3_c1",
        "p2_r1_c1", "p2_r2_c1", "p2_r3_c1",
    }
    assert not out["prc"].str.startswith("pZ").any()


def test_process_scores_without_plate_id_or_plate_raises():
    """No plateID column and no plate argument is a hard, explicit error."""
    from spacr.ml import process_scores

    df = _wells_df().drop(columns=["plateID"])
    with pytest.raises(ValueError, match="no usable 'plateID' column"):
        process_scores(df, "pred", plate=None, min_cell_count=1)


def test_process_scores_all_nan_plate_id_and_no_plate_raises():
    """An all-NaN plateID column is just as unusable as a missing one."""
    from spacr.ml import process_scores

    df = _wells_df()
    df["plateID"] = pd.Series([None] * len(df), dtype=object)
    with pytest.raises(ValueError, match="no usable 'plateID' column"):
        process_scores(df, "pred", plate=None, min_cell_count=1)


def test_process_scores_missing_column_id_raises():
    """plateID alone is not enough — rowID and columnID are required too."""
    from spacr.ml import process_scores

    df = _wells_df().drop(columns=["columnID"])
    with pytest.raises(ValueError,
                       match="must contain 'plateID', 'rowID', and 'columnID'"):
        process_scores(df, "pred", plate=None, min_cell_count=1)


def test_process_scores_reciprocal_inversion_warns_and_drops_zeros(capsys):
    """1/x inversion turns zeros into NaN, warns, and drops those objects."""
    from spacr.ml import process_scores

    df = _wells_df(n_rows=6, per_well=10, seed=11, low=0.25, high=0.75)
    zero_idx = df.index[df["rowID"] == "r1"][:3]
    df.loc[zero_idx, "pred"] = 0.0

    out, dv = process_scores(df, "pred", plate="p1", min_cell_count=5,
                             agg_type="mean", invert_dependent_variable=-1)

    printed = capsys.readouterr().out
    assert "contains 3 zero" in printed
    assert "Inverted 'pred' as 1/x on raw values." in printed
    assert dv == "pred"
    counts = dict(zip(out["prc"], out["cell_count"]))
    assert counts["p1_r1_c1"] == 7
    assert all(counts[f"p1_r{i}_c1"] == 10 for i in range(2, 7))
    # every aggregated value is a mean of reciprocals of values in [0.25, 0.75]
    assert (out["pred"] >= 1.0 / 0.75).all()


def test_process_scores_complement_inversion_matches_manual_mean(capsys):
    """1 - x inversion is applied before aggregation."""
    from spacr.ml import process_scores

    df = _wells_df(n_rows=4, per_well=6, seed=5)
    out, _ = process_scores(df, "pred", plate="p1", min_cell_count=2,
                            agg_type="mean", invert_dependent_variable=True)

    assert "Inverted 'pred' as 1 - x on raw values." in capsys.readouterr().out
    expected = (1.0 - df["pred"]).groupby(
        df["plateID"] + "_" + df["rowID"] + "_" + df["columnID"]).mean()
    got = out.set_index("prc")["pred"]
    assert np.allclose(got.loc[expected.index].to_numpy(), expected.to_numpy())


def test_process_scores_reports_non_normal_response(capsys):
    """A bimodal per-well response is reported as not normally distributed."""
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

    out, dv = process_scores(df, "pred", plate="p1", min_cell_count=3,
                             agg_type="mean")

    printed = capsys.readouterr().out
    assert "pred is not normally distributed" in printed
    assert dv == "pred"
    assert len(out) == 30


def test_process_scores_transform_renames_response_and_rechecks_normality(capsys):
    """A transform adds a '<transform>_<var>' column and re-runs the normality test."""
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

    out, dv = process_scores(df, "pred", plate="p1", min_cell_count=3,
                             agg_type="mean", transform="sqrt")

    assert dv == "sqrt_pred"
    assert "sqrt_pred" in out.columns
    assert np.allclose(out["sqrt_pred"].to_numpy(),
                       np.sqrt(out["pred"].to_numpy()))
    assert "sqrt_pred is not normally distributed" in capsys.readouterr().out


def test_process_scores_min_cell_count_drops_small_wells():
    """Wells below min_cell_count never reach the response frame."""
    from spacr.ml import process_scores

    df = _wells_df(n_rows=5, per_well=10, seed=31)
    # shrink one well to 2 objects
    small = df.index[df["rowID"] == "r5"][2:]
    df = df.drop(index=small).reset_index(drop=True)

    out, _ = process_scores(df, "pred", plate="p1", min_cell_count=5,
                            agg_type="quantile")

    assert "p1_r5_c1" not in set(out["prc"])
    assert len(out) == 4
    assert (out["cell_count"] == 10).all()
