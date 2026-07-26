"""Coverage for the tail of spacr.submodules: ``generate_score_heatmap`` and
``post_regression_analysis`` (submodules.py lines 1945-end).

Both entry points are pure pandas/seaborn on CSV inputs, so everything here is
CPU-only, offline and sub-second. All inputs are built as tiny deterministic
CSVs (8 rows x 1 column of a plate) so the expected fractions, means, MAEs,
correlations and propagated effect sizes can be written down by hand and
asserted exactly.

Two defects that used to be pinned here as ``xfail(strict=True)`` are now
fixed, and the tests below are ordinary regression tests for them:

* ``generate_score_heatmap`` -> ``calculate_fraction_mixed_condition`` grouped by
  ``columnID`` but merged on ``column_name``, which never exists on the grouped
  frame -> ``KeyError: 'column_name'`` for every input. It now keys on
  ``columnID`` throughout while still accepting a legacy ``column_name`` CSV.
* the ``'row_number' in merged_df.columns`` guard was a typo for the ``'row_num'``
  helper column the heatmap added *in place*, so ``row_num`` leaked into the
  returned frame, the saved ``*_data.csv`` and the MAE table as a bogus channel.
  The helper now copies its input and the guard tests the right name.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import seaborn as sns

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROWS = [f"r{i}" for i in range(1, 9)]
COL = "c3"
OTHER_COL = "c2"


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# generate_score_heatmap — input builders
# ---------------------------------------------------------------------------

def _write_scores_csv(path, seed, column_field="columnID", value_column="pred"):
    """Per-object classifier scores: one c3 row + one c2 row per plate row."""
    rng = np.random.default_rng(seed)
    recs = []
    for r in ROWS:
        for c in (COL, OTHER_COL):
            recs.append({column_field: c, "rowID": r,
                         value_column: round(float(rng.uniform(0, 1)), 6)})
    df = pd.DataFrame(recs)
    df.to_csv(path, index=False)
    # value of the c3 row per rowID — the groupby mean of a single row is itself
    return {rec["rowID"]: rec[value_column] for rec in recs
            if rec[column_field] == COL}


# Deterministic read counts: sgA rises, sgB falls, sgC is not a control sgRNA.
def _counts(row_idx):
    return {"A": 10 + 3 * row_idx, "B": 40 - 2 * row_idx, "C": 1000}


def _write_mixed_csv(path, grna_names):
    """Per-well sgRNA read counts for the mixed control condition."""
    a, b, c = grna_names
    recs = []
    for i, r in enumerate(ROWS):
        cts = _counts(i)
        for name, key in ((a, "A"), (b, "B"), (c, "C")):
            recs.append({"column_name": COL, "columnID": COL, "rowID": r,
                         "grna_name": name, "count": cts[key]})
        # A well in a different column that the c3 filter must drop.
        recs.append({"column_name": OTHER_COL, "columnID": OTHER_COL,
                     "rowID": r, "grna_name": a, "count": 99999})
    pd.DataFrame(recs).to_csv(path, index=False)
    return {r: _counts(i)["A"] / (_counts(i)["A"] + _counts(i)["B"])
            for i, r in enumerate(ROWS)}


def _model_folder(tmp_path, names=("modelA", "modelB"), with_empty=True):
    """A folder of per-model sub-folders, each holding a scores.csv."""
    folder = tmp_path / "models"
    folder.mkdir()
    truth = {}
    for i, m in enumerate(names):
        d = folder / m
        d.mkdir()
        truth[m] = _write_scores_csv(str(d / "scores.csv"), seed=17 + i)
    if with_empty:
        # sub-folder without the CSV -> exercises the "No such file" branch
        (folder / "model_without_scores").mkdir()
        # a plain file -> exercises the os.path.isdir() guard
        (folder / "stray.txt").write_text("not a folder\n")
    return folder, truth


def _settings(tmp_path, folders, mixed, cv, dst, **overrides):
    settings = {
        "folders": folders,
        "csv_name": "scores.csv",
        "data_column": "pred",
        "csv": str(mixed),
        "cv_csv": str(cv),
        "data_column_cv": "pred_cv",
        "plateID": 1,
        "columnID": COL,
        "control_sgrnas": ["sgA", "sgB"],
        "fraction_grna": "sgA",
        "cmap": "coolwarm",
        "dst": dst,
    }
    settings.update(overrides)
    return settings


# ---------------------------------------------------------------------------
# generate_score_heatmap
# ---------------------------------------------------------------------------

def test_generate_score_heatmap_completes_on_valid_inputs(tmp_path):
    """A fully valid set of score/reads/CV CSVs must yield the merged frame."""
    from spacr.submodules import generate_score_heatmap

    folder, _ = _model_folder(tmp_path)
    mixed = tmp_path / "mixed.csv"
    _write_mixed_csv(str(mixed), ("sgA", "sgB", "sgC"))
    cv = tmp_path / "cv.csv"
    _write_scores_csv(str(cv), seed=3, value_column="pred_cv")
    dst = tmp_path / "out"
    dst.mkdir()

    out = generate_score_heatmap(
        _settings(tmp_path, [str(folder)], mixed, cv, str(dst))
    )
    assert isinstance(out, pd.DataFrame)
    assert {"fraction", "prc", "modelA_pred", "modelB_pred"} <= set(out.columns)
    assert len(out) == len(ROWS)


def test_generate_score_heatmap_end_to_end_values_and_artifacts(
    tmp_path, capsys
):
    """Merged fractions/scores, the MAE table and the three artifacts are correct."""
    from spacr.submodules import generate_score_heatmap

    folder, model_truth = _model_folder(tmp_path)
    mixed = tmp_path / "mixed.csv"
    frac_truth = _write_mixed_csv(str(mixed), ("sgA", "sgB", "sgC"))
    cv = tmp_path / "cv.csv"
    cv_truth = _write_scores_csv(str(cv), seed=3, value_column="pred_cv")
    dst = tmp_path / "out"
    dst.mkdir()

    out = generate_score_heatmap(
        _settings(tmp_path, [str(folder)], mixed, cv, str(dst))
    )

    # -- shape / identity ---------------------------------------------------
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(ROWS)
    assert {"fraction", "prc", "plateID", "rowID", "columnID",
            "modelA_pred", "modelB_pred", "pred_cv"} <= set(out.columns)
    assert sorted(out["prc"]) == sorted(f"plate1_{r}_{COL}" for r in ROWS)
    # plateID is synthesised from settings['plateID'], columns filtered to c3
    assert set(out["plateID"]) == {"plate1"}
    assert set(out["columnID"]) == {COL}

    by_row = out.set_index("rowID")
    for r in ROWS:
        # fraction = control sgRNA A reads / (A + B) reads for that well
        assert by_row.loc[r, "fraction"] == pytest.approx(frac_truth[r])
        # one c3 row per well, so the groupby mean is that row's score
        assert by_row.loc[r, "modelA_pred"] == pytest.approx(model_truth["modelA"][r])
        assert by_row.loc[r, "modelB_pred"] == pytest.approx(model_truth["modelB"][r])
        assert by_row.loc[r, "pred_cv"] == pytest.approx(cv_truth[r])

    # the c2 wells (and the non-control sgRNA) never leak in
    assert 99999 not in out.select_dtypes("number").to_numpy()

    # -- discovery logging --------------------------------------------------
    printed = capsys.readouterr().out
    assert "Found 2 CSV files" in printed
    assert "model_without_scores" in printed and "No such file" in printed
    assert "modelA_pred" in printed and "modelB_pred" in printed

    # -- artifacts ----------------------------------------------------------
    mae_csv = dst / "mae_scores_comparison_plate_1.csv"
    data_csv = dst / "scores_comparison_plate_1_data.csv"
    pdf = dst / "scores_comparison_plate_1.pdf"
    assert mae_csv.is_file() and data_csv.is_file() and pdf.is_file()
    assert pdf.read_bytes()[:4] == b"%PDF"

    saved = pd.read_csv(data_csv)
    assert len(saved) == len(ROWS)
    assert saved["fraction"].tolist() == pytest.approx(out["fraction"].tolist())

    mae = pd.read_csv(mae_csv)
    assert list(mae.columns) == ["Channel", "MAE", "Row"]
    assert {"modelA_pred", "modelB_pred", "pred_cv"} <= set(mae["Channel"])
    mae_idx = mae.set_index(["Channel", "Row"])["MAE"]
    for r in ROWS:
        prc = f"plate1_{r}_{COL}"
        for channel in ("modelA_pred", "modelB_pred", "pred_cv"):
            expected = abs(frac_truth[r] - by_row.loc[r, channel])
            assert mae_idx.loc[(channel, prc)] == pytest.approx(expected)
    # every channel x well pair is present
    assert len(mae) == len(mae["Channel"].unique()) * len(ROWS)


def test_generate_score_heatmap_does_not_leak_row_num(tmp_path):
    """The plate-row sort helper must not survive into the results."""
    from spacr.submodules import generate_score_heatmap

    folder, _ = _model_folder(tmp_path)
    mixed = tmp_path / "mixed.csv"
    _write_mixed_csv(str(mixed), ("sgA", "sgB", "sgC"))
    cv = tmp_path / "cv.csv"
    _write_scores_csv(str(cv), seed=3, value_column="pred_cv")
    dst = tmp_path / "out"
    dst.mkdir()

    out = generate_score_heatmap(
        _settings(tmp_path, [str(folder)], mixed, cv, str(dst))
    )
    mae = pd.read_csv(dst / "mae_scores_comparison_plate_1.csv")
    assert "row_num" not in out.columns
    assert "row_num" not in set(mae["Channel"])


def test_generate_score_heatmap_column_alias_str_folder_and_no_dst(tmp_path):
    """Cover the CV 'column' alias, default control sgRNAs, a str ``folders``
    and ``dst=None``; the 'row_number' guard fires and drops the sort helper."""
    from spacr.submodules import generate_score_heatmap

    folder, model_truth = _model_folder(tmp_path, names=("modelA",),
                                        with_empty=False)
    mixed = tmp_path / "mixed.csv"
    # default control sgRNAs (control_sgrnas=None) + a decoy that must be dropped
    frac_truth = _write_mixed_csv(
        str(mixed), ("TGGT1_220950_1", "TGGT1_233460_4", "TGGT1_999999_9")
    )
    # CV CSV carries 'column' instead of 'columnID' and a 'row_number' score
    cv = tmp_path / "cv.csv"
    cv_truth = _write_scores_csv(str(cv), seed=5, column_field="column",
                                 value_column="row_number")

    out = generate_score_heatmap(_settings(
        tmp_path,
        str(folder),                       # plain string, not a list
        mixed, cv, None,
        data_column_cv="row_number",
        control_sgrnas=None,
        fraction_grna="TGGT1_220950_1",
    ))

    assert len(out) == len(ROWS)
    # 'row_number' is present, so the (mis-named) guard drops the sort helper
    assert "row_number" in out.columns
    assert "row_num" not in out.columns
    by_row = out.set_index("rowID")
    for r in ROWS:
        assert by_row.loc[r, "fraction"] == pytest.approx(frac_truth[r])
        assert by_row.loc[r, "modelA_pred"] == pytest.approx(model_truth["modelA"][r])
        assert by_row.loc[r, "row_number"] == pytest.approx(cv_truth[r])
    # dst is None -> nothing written anywhere under tmp_path
    assert not list(tmp_path.glob("**/*.pdf"))
    assert sorted(p.name for p in tmp_path.glob("*.csv")) == ["cv.csv", "mixed.csv"]


def test_generate_score_heatmap_missing_score_csvs_raises(tmp_path):
    """With no scores.csv anywhere the combined frame is None -> TypeError."""
    from spacr.submodules import generate_score_heatmap

    folder = tmp_path / "models"
    (folder / "modelA").mkdir(parents=True)   # no scores.csv inside
    mixed = tmp_path / "mixed.csv"
    _write_mixed_csv(str(mixed), ("sgA", "sgB", "sgC"))
    cv = tmp_path / "cv.csv"
    _write_scores_csv(str(cv), seed=3, value_column="pred_cv")

    with pytest.raises(TypeError):
        generate_score_heatmap(_settings(tmp_path, [str(folder)], mixed, cv, None))


# ---------------------------------------------------------------------------
# post_regression_analysis
# ---------------------------------------------------------------------------
#
# Designed so the correlation matrix is exactly +-1 everywhere:
#   g1 = v, g3 = 2v          -> corr(g1, g3) = +1
#   g2 = 1 - v, g4 = 3(1-v)  -> corr(g2, g4) = +1, corr(g1, g2) = -1
# After the 0-1 normalisation the matrix becomes
#   g1 [1 0 1 0]  g2 [0 1 0 1]  g3 [1 0 1 0]  g4 [0 1 0 1]
# so with anchors g1=1.0 and g2=0.0 the propagation gives
#   g3 = (1*1 + 1*0)/2 = 0.5     g4 = (1*0 + 1*0.5*0)/2 = 0.0
GRNA_LIST = ["g1", "g2", "g3", "g4"]
ANCHORS = {"g1": 1.0, "g2": 0.0}
EXPECTED_EFFECTS = {"g1": 1.0, "g2": 0.0, "g3": 0.5, "g4": 0.0}


def _write_regression_csv(path):
    v = np.linspace(0.1, 0.6, 6)
    prcs = [f"plate1_r{i+1}_c3" for i in range(len(v))]
    recs = []
    for prc, val in zip(prcs, v):
        recs.append({"prc": prc, "grna": "g2", "fraction": 1.0 - val})
        recs.append({"prc": prc, "grna": "g3", "fraction": 2.0 * val})
        recs.append({"prc": prc, "grna": "g4", "fraction": 3.0 * (1.0 - val)})
        # g1 is split across two rows so pivot_table(aggfunc='sum') matters
        recs.append({"prc": prc, "grna": "g1", "fraction": val / 4.0})
        recs.append({"prc": prc, "grna": "g1", "fraction": 3.0 * val / 4.0})
        # a gRNA outside grna_list that must be filtered out entirely
        recs.append({"prc": prc, "grna": "decoy", "fraction": 42.0})
    pd.DataFrame(recs).to_csv(path, index=False)
    return prcs


@pytest.fixture
def sns_recorder(monkeypatch):
    """Record what post_regression_analysis hands to seaborn, then draw it."""
    calls = {}
    real_heatmap = sns.heatmap
    real_barplot = sns.barplot

    def heatmap(data, *args, **kwargs):
        calls["heatmap"] = data.copy()
        calls["heatmap_kwargs"] = kwargs
        return real_heatmap(data, *args, **kwargs)

    def barplot(*args, **kwargs):
        calls["barplot"] = (list(kwargs.get("x")), list(kwargs.get("y")))
        return real_barplot(*args, **kwargs)

    monkeypatch.setattr(sns, "heatmap", heatmap)
    monkeypatch.setattr(sns, "barplot", barplot)
    return calls


def test_post_regression_analysis_saves_matrix_and_effect_sizes(tmp_path):
    """save=True writes the correlation matrix, effect sizes and both PDFs."""
    from spacr.submodules import post_regression_analysis

    csv = tmp_path / "regression.csv"
    _write_regression_csv(str(csv))

    assert post_regression_analysis(str(csv), ANCHORS, GRNA_LIST, save=True) is None

    out_dir = tmp_path / "post_regression_analysis_results"
    assert out_dir.is_dir()
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "correlation_matrix.csv",
        "correlation_matrix_heatmap.pdf",
        "effect_sizes.csv",
        "effect_sizes_barplot.pdf",
    ]
    for pdf in out_dir.glob("*.pdf"):
        assert pdf.read_bytes()[:4] == b"%PDF"

    corr = pd.read_csv(out_dir / "correlation_matrix.csv", index_col=0)
    # the decoy gRNA is filtered out before pivoting
    assert list(corr.index) == GRNA_LIST
    assert list(corr.columns) == GRNA_LIST
    expected_corr = np.array([
        [1.0, -1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0, 1.0],
    ])
    assert corr.to_numpy() == pytest.approx(expected_corr)

    effects = pd.read_csv(out_dir / "effect_sizes.csv", index_col=0)
    got = effects.iloc[:, 0].to_dict()
    assert got == pytest.approx(EXPECTED_EFFECTS)


def test_post_regression_analysis_no_save_only_creates_folder(tmp_path, sns_recorder):
    """save=False plots the same numbers but writes nothing to disk."""
    from spacr.submodules import post_regression_analysis

    csv = tmp_path / "regression.csv"
    _write_regression_csv(str(csv))

    post_regression_analysis(str(csv), ANCHORS, GRNA_LIST, save=False)

    out_dir = tmp_path / "post_regression_analysis_results"
    assert out_dir.is_dir()
    assert list(out_dir.iterdir()) == []

    plotted_corr = sns_recorder["heatmap"]
    assert list(plotted_corr.index) == GRNA_LIST
    assert plotted_corr.loc["g1", "g3"] == pytest.approx(1.0)
    assert plotted_corr.loc["g1", "g2"] == pytest.approx(-1.0)
    assert sns_recorder["heatmap_kwargs"]["cmap"] == "coolwarm"

    names, values = sns_recorder["barplot"]
    assert names == GRNA_LIST
    assert dict(zip(names, values)) == pytest.approx(EXPECTED_EFFECTS)
    # anchors keep exactly the effect size supplied by the caller
    for grna, size in ANCHORS.items():
        assert values[names.index(grna)] == pytest.approx(size)
    # a gRNA perfectly correlated with the strong anchor outranks one that is
    # perfectly anti-correlated with it
    assert values[names.index("g3")] > values[names.index("g4")]


def test_post_regression_analysis_propagation_is_sequential(tmp_path, sns_recorder):
    """Propagation walks the index in order and reuses already-updated values.

    With anchors g1=0.8, g2=0.2 and the +-1 correlation design above:
      g3 = (1*0.8 + 1*0.0)/2 = 0.40      (g4 still 0 at this point)
      g4 = (1*0.2 + 1*0.0)/2 = 0.10      (g3's fresh 0.40 has weight 0 here)
    """
    from spacr.submodules import post_regression_analysis

    csv = tmp_path / "regression.csv"
    _write_regression_csv(str(csv))
    post_regression_analysis(str(csv), {"g1": 0.8, "g2": 0.2}, GRNA_LIST, save=False)

    names, values = sns_recorder["barplot"]
    effects = dict(zip(names, values))
    assert effects == pytest.approx({"g1": 0.8, "g2": 0.2, "g3": 0.4, "g4": 0.1})
    # the correlation matrix seaborn was handed is the raw (un-normalised) one
    corr = sns_recorder["heatmap"]
    assert corr.to_numpy().min() == pytest.approx(-1.0)
    assert corr.to_numpy().max() == pytest.approx(1.0)
