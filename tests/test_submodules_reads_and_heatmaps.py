"""Table-driven tests of the read-count / score comparison figures in ``spacr.submodules``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture:

* ``compare_reads_to_scores``: the per-well read and class fractions it
  plots (as line data), every legacy key spelling, the ``inf`` / ``NaN``
  ratio repair, list inputs, saving and the refusals;
* ``generate_score_heatmap``: the merged frame, the MAE table and the three
  saved artefacts, plate stamping and the ``row_num`` sort-key clash;
* ``post_regression_analysis``: the correlation matrix and the sequential
  effect-size propagation;
* ``_style_colour_bar``: a colour bar loses its frame, a bar-less figure is
  left alone.

Every input is a tiny deterministic CSV written per case, so each fraction,
mean, MAE, correlation and effect size can be written down by hand.
"""
from __future__ import annotations

import functools
import re

import numpy as np
import pandas as pd
import pytest
import seaborn as sns

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import spacr.submodules  # noqa: E402
from spacr.submodules import (  # noqa: E402
    _style_colour_bar,
    compare_reads_to_scores,
    generate_score_heatmap,
    post_regression_analysis,
)


@pytest.fixture(autouse=True)
def _close_figs():
    """No figure leaves a test in this module."""
    yield
    plt.close("all")


def _case_dir(tmp_path, name):
    d = tmp_path / re.sub(r"[^A-Za-z0-9_]+", "_", name)
    d.mkdir()
    return d


def _raw_reader(mp):
    """The reader's documented raw-file mode, so legacy spellings reach the function."""
    from spacr.tabular import read_table
    mp.setattr(spacr.submodules, "read_table", functools.partial(read_table, canonicalise=False))


# ---------------------------------------------------------------------------
# compare_reads_to_scores
# ---------------------------------------------------------------------------

PC = "TGGT1_220950_1"
NC = "TGGT1_233460_4"
FILLER = "TGGT1_111111_1"
ROWS4 = ("r1", "r2", "r3", "r4")
# rowID -> (pc units, nc units)
EMPIRICAL = {"r1": (90, 10), "r2": (80, 20), "r3": (70, 30), "r4": (60, 40)}
X_PC = [0.6, 0.7, 0.8, 0.9]
X_NC = [0.1, 0.2, 0.3, 0.4]


def _reads(with_plate=True):
    """Per-gRNA reads: in c3 the pc gRNA climbs 10->40 while nc falls 40->10;
    c4 is a flat 10/10 split."""
    recs = []
    for i, row in enumerate(ROWS4):
        for col, counts in (("c3", {PC: 10 * (i + 1), NC: 10 * (4 - i), FILLER: 5}),
                            ("c4", {PC: 10, NC: 10, FILLER: 5})):
            for grna, cnt in counts.items():
                rec = {"rowID": row, "columnID": col, "grna_name": grna, "count": cnt}
                if with_plate:
                    rec["plateID"] = "plate1"
                recs.append(rec)
    return pd.DataFrame(recs)


def _calls(with_plate=True, n_class1=None):
    """Per-object classifier calls: c3 row i has i+1 of 6 objects class 1; c4 1 of 2."""
    recs = []
    for i, row in enumerate(ROWS4):
        n1 = (i + 1) if n_class1 is None else n_class1[i]
        for col, (k1, k0) in (("c3", (n1, 6 - n1)), ("c4", (1, 1))):
            for label, n in ((1, k1), (0, k0)):
                for _ in range(n):
                    rec = {"rowID": row, "columnID": col, "cv_predictions": label}
                    if with_plate:
                        rec["plateID"] = "plate1"
                    recs.append(rec)
    return pd.DataFrame(recs)


def _lines(fig):
    """{label: (x, y)} for every line on the figure's first axes."""
    return {ln.get_label(): (np.asarray(ln.get_xdata()), np.asarray(ln.get_ydata()))
            for ln in fig.axes[0].lines}


def _without(frame, mask):
    return frame[~mask(frame)].reset_index(drop=True)


def _zeroed_r4(frame):
    frame = frame.copy()
    frame.loc[(frame["rowID"] == "r4") & frame["grna_name"].isin([PC, NC]), "count"] = 0
    return frame


def _r13(frame):
    frame = frame.copy()
    frame["rowID"] = frame["rowID"].replace({"r1": "r13"})
    return frame


C1 = [4 / 6, 3 / 6, 2 / 6, 1 / 6]


def test_compare_reads_to_scores_plots_read_and_class_fractions_per_well(
        subtests, tmp_path, capsys):
    """The plotted lines carry exactly the per-well read fraction, class-1
    fraction and pc/nc ratio; legacy spellings join, and a missing key or
    column is refused by name."""
    reads, calls = _reads(), _calls()
    no_plate = (_reads(False), _calls(False))
    vector = pd.Series([0.1, 0.2, 0.3, 0.4], name="my_vector")
    cases = [
        # (id, [(reads, scores), ...], kwargs, raw reader, expected)
        # expected: {"lines": {label: (x, y)}} / {"nc_lines": ...} / ("raises", type, text)
        #           / "none" (columns listed)
        ("one pair: pc and nc figures", [(reads, calls)], {},
         False, {"lines": {"class_1_fraction": (X_PC, C1), f"{PC}_fraction": (X_PC, [.8, .6, .4, .2]),
                           "nc_fraction": (X_PC, [.4, .3, .2, .1])},
                 "nc_lines": {"class_1_fraction": (X_NC, C1[::-1])}}),
        ("column / value select the c4 wells", [(reads, calls)],
         {"column": "columnID", "value": "c4"}, False,
         {"lines": {f"{PC}_fraction": (X_PC, [0.5] * 4), "class_1_fraction": (X_PC, [0.5] * 4)}}),
        ("the built-in 16-row mixture", [(_r13(reads), _r13(calls))], {"empirical_dict": None},
         False, {"lines": {"nc_fraction": ([0.3, 0.8, 0.9], [0.7, 0.2, 0.1])}}),
        ("declared default save paths skip saving", [(reads, calls)], {"save_paths": "default"},
         False, {"lines": {"class_1_fraction": (X_PC, C1)}}),
        ("a bare y column name is one line", [(reads, calls)],
         {"y_columns": "class_1_fraction"}, False, {"n_lines": 1, "only_line": C1}),
        ("a y vector is one unlabelled line", [(reads, calls)], {"y_columns": vector}, False,
         {"vector": [0.4, 0.3, 0.2, 0.1]}),
        ("no nc reads: inf ratio becomes 0",
         [(_without(reads, lambda f: (f["rowID"] == "r4") & (f["grna_name"] == NC)), calls)],
         {"y_columns": ["fraction_ratio"]}, False,
         {"lines": {"fraction_ratio": (X_PC, [0.0, 3 / 2, 2 / 3, 1 / 4])}}),
        ("no nc reads: that well's pc fraction is 1",
         [(_without(reads, lambda f: (f["rowID"] == "r4") & (f["grna_name"] == NC)), calls)],
         {}, False, {"lines": {f"{PC}_fraction": (X_PC, [1.0, .6, .4, .2])}}),
        ("no control reads: NaN ratio becomes 0", [(_zeroed_r4(reads), calls)],
         {"y_columns": ["fraction_ratio"]}, False,
         {"lines": {"fraction_ratio": (X_PC, [0.0, 3 / 2, 2 / 3, 1 / 4])}}),
        ("no control reads: the 0/0 pc fraction is dropped", [(_zeroed_r4(reads), calls)], {},
         False, {"lines": {f"{PC}_fraction": (X_PC[1:], [.6, .4, .2])}}),
        ("an all-positive classifier", [(reads, calls.assign(cv_predictions=1))], {}, False,
         {"lines": {"class_1_fraction": (X_PC, [1.0] * 4)}}),
        ("plate stamps a plateID the CSVs lack", [no_plate], {"plate": "p9"}, False,
         {"lines": {"class_1_fraction": (X_PC, C1)}}),
        ("legacy row header on the reads", [(_reads(False).rename(columns={"rowID": "row"}),
                                             _calls(False))], {"plate": None, "as_list": True},
         False, "raises_or_lines"),
        ("row and row_name headers together",
         [(_reads(False).rename(columns={"rowID": "row_name"}).assign(
             row=lambda f: f["row_name"]), _calls(False))], {"as_list": True}, False,
         {"lines": {f"{PC}_fraction": (X_PC, [.8, .6, .4, .2])}}),
        ("two plates with legacy column spellings average per x",
         [(_reads(False).rename(columns={"columnID": "column"}),
           _calls(False).rename(columns={"rowID": "row_name"})),
          (_reads(False).rename(columns={"columnID": "column_name"}),
           _calls(False, n_class1=[5, 5, 5, 5]).rename(columns={"rowID": "row_name"}))],
         {}, False, {"lines": {"class_1_fraction": (X_PC, [(4 / 6 + 5 / 6) / 2, (3 / 6 + 5 / 6) / 2,
                                                            (2 / 6 + 5 / 6) / 2, (1 / 6 + 5 / 6) / 2])},
                     "printed": "Reads: 48 Scores: 64"}),
        ("raw files: both reads spellings repaired",
         [(_reads(False).rename(columns={"rowID": "row", "columnID": "column"}),
           _calls(False).rename(columns={"rowID": "row_name"})),
          (_reads(False).rename(columns={"rowID": "row_name", "columnID": "column_name"}),
           _calls(False).rename(columns={"rowID": "row_name"}))], {}, True,
         {"joined_plates": ["plate1", "plate2"], "joined_rows": 8}),
        ("reads without a plateID are refused", [no_plate], {}, False,
         ("raises", ValueError, "Cannot find plate, row or column")),
        ("reads without a rowID are refused", [(reads.drop(columns=["rowID"]), calls)], {}, False,
         ("raises", ValueError, "Cannot find plate, row or column")),
        ("scores without a columnID are refused", [(reads, calls.drop(columns=["columnID"]))], {},
         False, ("raises", ValueError, "Cannot find 'plateID', 'rowID', or 'columnID'")),
        ("mismatched list lengths are refused", [(reads, calls), (reads, None)], {}, False,
         ("raises", ValueError, "")),
        ("an unknown y column lists the frame's columns", [(reads, calls)],
         {"y_columns": ["not_a_real_column"]}, False, "none"),
        ("an unknown bare y column lists the frame's columns", [(reads, calls)],
         {"y_columns": "not_a_column"}, False, "none"),
        ("a non-string y column lists the frame's columns", [(reads, calls)],
         {"y_columns": (any,)}, False, "none"),
    ]
    for case_id, pairs, kw, raw, expected in cases:
        with subtests.test(msg=case_id):
            folder = _case_dir(tmp_path, case_id)
            rps, sps = [], []
            for i, (r, s) in enumerate(pairs):
                rp = folder / f"reads{i}.csv"
                r.to_csv(rp, index=False)
                rps.append(str(rp))
                if s is not None:
                    sp = folder / f"scores{i}.csv"
                    s.to_csv(sp, index=False)
                    sps.append(str(sp))
            kw = dict(kw)
            as_list = kw.pop("as_list", False) or len(pairs) > 1
            call = dict(empirical_dict=EMPIRICAL, save_paths=[None, None])
            call.update(kw)
            if call["save_paths"] == "default":
                del call["save_paths"]
            args = (rps, sps) if as_list else (rps[0], sps[0])
            joined = []
            capsys.readouterr()
            with pytest.MonkeyPatch.context() as mp:
                if raw:
                    _raw_reader(mp)
                    mp.setattr(spacr.submodules, "display", joined.append)
                if isinstance(expected, tuple):
                    with pytest.raises(expected[1], match=re.escape(expected[2])):
                        compare_reads_to_scores(*args, **call)
                    continue
                figs = compare_reads_to_scores(*args, **call)
            printed = capsys.readouterr().out
            if expected == "none":
                assert figs is None
                assert "columns in dataframe:" in printed
                for col in ("pc_fraction", "nc_fraction", "class_1_fraction",
                            f"{PC}_fraction", "fraction_ratio"):
                    assert f"\n{col}\n" in printed
                continue
            if expected == "raises_or_lines":
                # a bare legacy 'row' header is canonicalised on read like any other
                assert len(figs) == 2
                assert np.allclose(_lines(figs[0])["class_1_fraction"][1], C1)
                continue
            assert isinstance(figs, list) and len(figs) == 2
            assert list(folder.glob("*.pdf")) == []
            for key, fig in (("lines", figs[0]), ("nc_lines", figs[1])):
                lines = _lines(fig)
                for label, (x, y) in expected.get(key, {}).items():
                    np.testing.assert_allclose(np.round(lines[label][0], 3), x)
                    np.testing.assert_allclose(lines[label][1], y)
            if "n_lines" in expected:
                assert len(figs[0].axes[0].lines) == expected["n_lines"]
                np.testing.assert_allclose(figs[0].axes[0].lines[0].get_xdata(), X_PC)
                np.testing.assert_allclose(figs[0].axes[0].lines[0].get_ydata(),
                                           expected["only_line"])
            if "printed" in expected:
                assert expected["printed"] in printed
            if "vector" in expected:
                ax = figs[0].axes[0]
                assert len(ax.lines) == 1 and ax.get_legend() is None
                np.testing.assert_allclose(ax.lines[0].get_xdata(), X_PC)
                np.testing.assert_allclose(ax.lines[0].get_ydata(), expected["vector"])
                assert ax.get_ylabel() == "Fraction"
                assert figs[0]._spacr_spec["y"] == "my_vector"
                np.testing.assert_allclose(figs[0]._spacr_data["my_vector"], expected["vector"])
            if "joined_plates" in expected:
                frame = joined[0]
                assert sorted(frame["prc"].str.split("_").str[0].unique()) == expected["joined_plates"]
                assert len(frame) == expected["joined_rows"]

    with subtests.test(msg="figure chrome and both PDFs"):
        folder = _case_dir(tmp_path, "saved")
        rp, sp, pc_pdf, nc_pdf = (folder / n for n in ("r.csv", "s.csv", "pc.pdf", "nc.pdf"))
        reads.to_csv(rp, index=False)
        calls.to_csv(sp, index=False)
        capsys.readouterr()
        fig_pc, fig_nc = compare_reads_to_scores(str(rp), str(sp), empirical_dict=EMPIRICAL,
                                                 save_paths=[str(pc_pdf), str(nc_pdf)])
        ax = fig_pc.axes[0]
        assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == (
            "pc_fraction", "Fraction", "Line Plot")
        assert fig_nc.axes[0].get_xlabel() == "nc_fraction"
        assert ax.get_legend().get_title().get_text() == "Legend"
        assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
        assert set(_lines(fig_pc)) == {"class_1_fraction", f"{PC}_fraction", "nc_fraction"}
        for pdf in (pc_pdf, nc_pdf):
            assert pdf.read_bytes()[:4] == b"%PDF"
        printed = capsys.readouterr().out
        assert f"Plot saved to {pc_pdf}" in printed and f"Plot saved to {nc_pdf}" in printed


# ---------------------------------------------------------------------------
# generate_score_heatmap
# ---------------------------------------------------------------------------

ROWS8 = [f"r{i}" for i in range(1, 9)]
COL, OTHER_COL = "c3", "c2"


def _write_scores(path, seed, value_column="pred", column_key="columnID", plate=None):
    """One c3 and one c2 row per plate row; returns the c3 value per rowID."""
    rng = np.random.default_rng(seed)
    recs = []
    for r in ROWS8:
        for c in (COL, OTHER_COL):
            rec = {"rowID": r, value_column: round(float(rng.uniform(0, 1)), 6)}
            if column_key:
                rec[column_key] = c
            if plate:
                rec["plateID"] = plate
            recs.append(rec)
    pd.DataFrame(recs).to_csv(path, index=False)
    return {rec["rowID"]: rec[value_column] for rec in recs
            if not column_key or rec[column_key] == COL}


def _read_counts(i):
    return {"A": 10 + 3 * i, "B": 40 - 2 * i, "C": 1000}


def _write_mixed(path, names=("sgA", "sgB", "sgC"), plate=None, column_key="columnID"):
    """Per-well sgRNA reads; a c2 decoy well the column filter must drop."""
    recs = []
    for i, r in enumerate(ROWS8):
        cts = _read_counts(i)
        for name, key in zip(names, "ABC"):
            rec = {column_key: COL, "rowID": r, "grna_name": name, "count": cts[key]}
            if plate:
                rec["plateID"] = plate
            recs.append(rec)
        if not plate:
            recs.append({column_key: OTHER_COL, "rowID": r, "grna_name": names[0],
                         "count": 99999})
    pd.DataFrame(recs).to_csv(path, index=False)
    return {r: _read_counts(i)["A"] / (_read_counts(i)["A"] + _read_counts(i)["B"])
            for i, r in enumerate(ROWS8)}


def _heatmap_case(folder, *, models=("modelA", "modelB"), strays=True, cv_value="pred_cv",
                  cv_column_key="columnID", plate=None, mixed_names=("sgA", "sgB", "sgC"),
                  mixed_column_key="columnID", dst=True, **over):
    models_dir = folder / "models"
    models_dir.mkdir()
    truth = {}
    for i, m in enumerate(models):
        (models_dir / m).mkdir()
        truth[m] = _write_scores(models_dir / m / "scores.csv", 17 + i, plate=plate)
    if strays:
        (models_dir / "model_without_scores").mkdir()
        (models_dir / "stray.txt").write_text("not a folder\n", encoding="utf-8")
    truth["fraction"] = _write_mixed(folder / "mixed.csv", mixed_names, plate=plate,
                                     column_key=mixed_column_key)
    truth["cv"] = _write_scores(folder / "cv.csv", 3, cv_value, column_key=cv_column_key,
                                plate=plate)
    out = folder / "out"
    if dst:
        out.mkdir()
    settings = {"folders": [str(models_dir)], "csv_name": "scores.csv", "data_column": "pred",
                "csv": str(folder / "mixed.csv"), "cv_csv": str(folder / "cv.csv"),
                "data_column_cv": cv_value, "plateID": 1, "columnID": COL,
                "control_sgrnas": ["sgA", "sgB"], "fraction_grna": "sgA", "cmap": "coolwarm",
                "dst": str(out) if dst else None}
    settings.update(over)
    return settings, truth


def test_generate_score_heatmap_merges_scores_reads_and_cv_per_well(subtests, tmp_path, capsys):
    """The merged frame holds, per c3 well, the control read fraction and
    every model's mean score; the MAE table and artefacts match it, and the
    sort key never survives as a channel."""
    cases = [
        # (id, case kwargs, raw reader, expected)
        ("valid inputs with artefacts", {}, False,
         {"models": ("modelA", "modelB"), "cv": "pred_cv", "plate": "plate1", "saved": True}),
        ("a str folder, default controls, legacy CV column, no dst",
         {"models": ("modelA",), "strays": False, "cv_value": "row_number", "cv_column_key": "column",
          "mixed_names": ("TGGT1_220950_1", "TGGT1_233460_4", "TGGT1_999999_9"),
          "dst": False, "control_sgrnas": None, "fraction_grna": "TGGT1_220950_1",
          "str_folder": True}, False,
         {"models": ("modelA",), "cv": "row_number", "plate": "plate1"}),
        ("a CV channel called row_num is not the sort key",
         {"cv_value": "row_num", "dst": False}, False,
         {"models": ("modelA", "modelB"), "cv": None, "plate": "plate1"}),
        ("plateID None keeps the CSVs' own plate", {"plate": "plate7", "strays": False,
                                                    "models": ("modelA",), "dst": False,
                                                    "plateID": None}, False,
         {"models": ("modelA",), "cv": "pred_cv", "plate": "plate7"}),
        ("a plate number overrides the CSVs' plate", {"plate": "plate7", "strays": False,
                                                       "models": ("modelA",), "dst": False,
                                                       "plateID": 2}, False,
         {"models": ("modelA",), "cv": "pred_cv", "plate": "plate2"}),
        ("raw file: a legacy column_name reads table", {"mixed_column_key": "column_name",
                                                        "dst": False}, True,
         {"models": ("modelA", "modelB"), "cv": "pred_cv", "plate": "plate1"}),
        ("a CV table without a column key names it", {"cv_column_key": None, "dst": False},
         False, ("raises", KeyError, "columnID")),
        ("no scores.csv anywhere", {"models": (), "dst": False}, False,
         ("raises", TypeError, "")),
    ]
    for case_id, kw, raw, expected in cases:
        with subtests.test(msg=case_id):
            folder = _case_dir(tmp_path, case_id)
            kw = dict(kw)
            str_folder = kw.pop("str_folder", False)
            settings, truth = _heatmap_case(folder, **kw)
            if str_folder:
                settings["folders"] = settings["folders"][0]
            capsys.readouterr()
            with pytest.MonkeyPatch.context() as mp:
                if raw:
                    _raw_reader(mp)
                if isinstance(expected, tuple):
                    with pytest.raises(expected[1]) as excinfo:
                        generate_score_heatmap(settings)
                    assert expected[2] in str(excinfo.value)
                    continue
                out = generate_score_heatmap(settings)
            printed = capsys.readouterr().out
            assert len(out) == 8 and "row_num" not in out.columns
            assert set(out["prc"]) == {f"{expected['plate']}_{r}_{COL}" for r in ROWS8}
            assert set(out["columnID"]) == {COL}
            by_row = out.set_index("rowID")
            for r in ROWS8:
                assert by_row.loc[r, "fraction"] == pytest.approx(truth["fraction"][r])
                for m in expected["models"]:
                    assert by_row.loc[r, f"{m}_pred"] == pytest.approx(truth[m][r])
                if expected["cv"]:
                    assert by_row.loc[r, expected["cv"]] == pytest.approx(truth["cv"][r])
            assert 99999 not in out.select_dtypes("number").to_numpy()
            if expected.get("saved"):
                assert "Found 2 CSV files" in printed
                assert "model_without_scores" in printed and "No such file" in printed
                assert "modelA_pred" in printed and "modelB_pred" in printed
                dst = folder / "out"
                assert sorted(p.name for p in dst.iterdir()) == [
                    "mae_scores_comparison_plate_1.csv", "scores_comparison_plate_1.pdf",
                    "scores_comparison_plate_1_data.csv"]
                assert (dst / "scores_comparison_plate_1.pdf").read_bytes()[:4] == b"%PDF"
                saved = pd.read_csv(dst / "scores_comparison_plate_1_data.csv")
                assert saved["fraction"].tolist() == pytest.approx(out["fraction"].tolist())
                assert "row_num" not in saved.columns
                mae = pd.read_csv(dst / "mae_scores_comparison_plate_1.csv")
                assert list(mae.columns) == ["Channel", "MAE", "Row"]
                assert sorted(set(mae["Channel"])) == ["modelA_pred", "modelB_pred", "pred_cv"]
                assert len(mae) == 3 * 8
                idx = mae.set_index(["Channel", "Row"])["MAE"]
                for r in ROWS8:
                    for channel in ("modelA_pred", "modelB_pred", "pred_cv"):
                        assert idx.loc[(channel, f"plate1_{r}_{COL}")] == pytest.approx(
                            abs(truth["fraction"][r] - by_row.loc[r, channel]))
            else:
                assert list(folder.glob("**/*.pdf")) == []

    with subtests.test(msg="a score CSV's own row_num index never reaches the frame"):
        folder = _case_dir(tmp_path, "row_num index")
        settings, _ = _heatmap_case(folder, strays=False, dst=False)
        for m in ("modelA", "modelB"):
            path = folder / "models" / m / "scores.csv"
            pd.read_csv(path).assign(row_num=range(16)).to_csv(path, index=False)
        out = generate_score_heatmap(settings)
        assert "row_num" not in out.columns and len(out) == 8


# ---------------------------------------------------------------------------
# post_regression_analysis
# ---------------------------------------------------------------------------

GRNA_LIST = ["g1", "g2", "g3", "g4"]
CORR = np.array([[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0],
                 [1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]])


def _write_regression_csv(path):
    """g1/g3 rise with v, g2/g4 fall; g1 split over two rows; a decoy gRNA."""
    recs = []
    for i, val in enumerate(np.linspace(0.1, 0.6, 6)):
        prc = f"plate1_r{i + 1}_c3"
        recs += [{"prc": prc, "grna": "g2", "fraction": 1.0 - val},
                 {"prc": prc, "grna": "g3", "fraction": 2.0 * val},
                 {"prc": prc, "grna": "g4", "fraction": 3.0 * (1.0 - val)},
                 {"prc": prc, "grna": "g1", "fraction": val / 4.0},
                 {"prc": prc, "grna": "g1", "fraction": 3.0 * val / 4.0},
                 {"prc": prc, "grna": "decoy", "fraction": 42.0}]
    pd.DataFrame(recs).to_csv(path, index=False)


def test_post_regression_analysis_propagates_anchor_effects_through_correlation(
        subtests, tmp_path):
    """The correlation matrix of the listed gRNAs is exact, effect sizes
    propagate sequentially from the anchors, and ``save`` writes exactly the
    two tables and two PDFs."""
    cases = [
        # (id, anchors, save, expected effect sizes, expected files)
        ("anchors 1 / 0, saved", {"g1": 1.0, "g2": 0.0}, True,
         {"g1": 1.0, "g2": 0.0, "g3": 0.5, "g4": 0.0},
         ["correlation_matrix.csv", "correlation_matrix_heatmap.pdf", "effect_sizes.csv",
          "effect_sizes_barplot.pdf"]),
        ("anchors 1 / 0, not saved", {"g1": 1.0, "g2": 0.0}, False,
         {"g1": 1.0, "g2": 0.0, "g3": 0.5, "g4": 0.0}, []),
        ("propagation reuses already-updated values", {"g1": 0.8, "g2": 0.2}, False,
         {"g1": 0.8, "g2": 0.2, "g3": 0.4, "g4": 0.1}, []),
    ]
    for case_id, anchors, save, effects, files in cases:
        with subtests.test(msg=case_id):
            folder = _case_dir(tmp_path, case_id)
            csv = folder / "regression.csv"
            _write_regression_csv(csv)
            calls = {}
            with pytest.MonkeyPatch.context() as mp:
                real_heatmap, real_barplot = sns.heatmap, sns.barplot

                def heatmap(data, *a, **kw):
                    calls["heatmap"] = (data.copy(), kw)
                    return real_heatmap(data, *a, **kw)

                def barplot(*a, **kw):
                    calls["barplot"] = (list(kw["x"]), list(kw["y"]))
                    return real_barplot(*a, **kw)

                mp.setattr(sns, "heatmap", heatmap)
                mp.setattr(sns, "barplot", barplot)
                assert post_regression_analysis(str(csv), anchors, GRNA_LIST, save=save) is None
            corr, kw = calls["heatmap"]
            assert list(corr.index) == GRNA_LIST and list(corr.columns) == GRNA_LIST
            np.testing.assert_allclose(corr.to_numpy(), CORR)
            assert kw["cmap"] == "coolwarm"
            names, values = calls["barplot"]
            assert names == GRNA_LIST
            assert dict(zip(names, values)) == pytest.approx(effects)
            out_dir = folder / "post_regression_analysis_results"
            assert sorted(p.name for p in out_dir.iterdir()) == files
            if save:
                for pdf in out_dir.glob("*.pdf"):
                    assert pdf.read_bytes()[:4] == b"%PDF"
                saved = pd.read_csv(out_dir / "correlation_matrix.csv", index_col=0)
                assert list(saved.index) == GRNA_LIST
                np.testing.assert_allclose(saved.to_numpy(), CORR)
                saved_effects = pd.read_csv(out_dir / "effect_sizes.csv", index_col=0)
                assert saved_effects.iloc[:, 0].to_dict() == pytest.approx(effects)


# ---------------------------------------------------------------------------
# _style_colour_bar
# ---------------------------------------------------------------------------

def test_style_colour_bar_restyles_only_a_second_axes(subtests):
    """The last of two or more axes loses its frame and takes the theme ink;
    a single-axes figure keeps every spine and tick colour."""
    from matplotlib.figure import Figure

    from spacr.submodules import resolve_ink, theme_target

    ink = matplotlib.colors.to_rgba(resolve_ink(theme_target()))
    for case_id, n_axes in (("one axes", 1), ("heatmap and bar", 2), ("three axes", 3)):
        with subtests.test(msg=case_id):
            fig = Figure()
            axes = [fig.add_subplot(1, n_axes, i + 1) for i in range(n_axes)]
            axes[0].plot([0, 1], [0, 1])
            before = [[s.get_visible() for s in ax.spines.values()] for ax in axes]
            assert _style_colour_bar(fig) is None
            assert len(fig.axes) == n_axes
            for i, ax in enumerate(axes):
                spines = [s.get_visible() for s in ax.spines.values()]
                if n_axes > 1 and i == n_axes - 1:
                    assert spines == [False] * 4
                    ticks = ax.yaxis.get_major_ticks()[0]
                    assert matplotlib.colors.to_rgba(ticks.tick1line.get_color()) == ink
                    assert matplotlib.colors.to_rgba(ticks.label1.get_color()) == ink
                else:
                    assert spines == before[i] == [True] * 4
