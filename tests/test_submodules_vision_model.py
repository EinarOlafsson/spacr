"""Table-driven tests of the legacy vision-model explainer ``spacr.submodules.interperate_vision_model``.

Two general tests, each walking a case table with pytest's ``subtests``
fixture:

* the join: how a per-object scores CSV (every key spelling, the modern
  ``columnID`` schema, a partial scores file) lines up with the merged
  measurements, the cross-compartment ratio expansion, and the refusals for
  a key repeated on either side or a time-lapse database;
* the explainers: random-forest importance, its compartment / channel
  grouping, ``include_all``, integer channel ids, permutation importance,
  SHAP (with and without subsampling) and the saved CSVs.

``spacr.io._read_and_merge_data`` is replaced by a fake returning a frame
shaped like the real output, so no database, network or GPU is involved.  A
recording ``RandomForestClassifier`` subclass captures the feature matrix the
function built.  Every case runs in its own folder and monkeypatch context.
"""
from __future__ import annotations

import functools
import os
import re

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

import spacr.io  # noqa: E402
import spacr.submodules  # noqa: E402
from spacr.io import MergeCardinalityError  # noqa: E402
from spacr.submodules import interperate_vision_model  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    plt.close("all")
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# synthetic data shaped like io._read_and_merge_data output
# ---------------------------------------------------------------------------

def _grid(n):
    """``n`` unique (plate, row, column, field) metadata tuples."""
    return [("plate1", f"r{i % 3 + 1}", f"c{i % 4 + 1}", f"f{i}") for i in range(n)]


def _metadata(g):
    return {"object_label": [f"o{i}" for i in range(1, len(g) + 1)],
            "plateID": [x[0] for x in g], "rowID": [x[1] for x in g],
            "column_name": [x[2] for x in g], "fieldID": [x[3] for x in g]}


def _full_frame(n, seed=0, zero_pathogen_row=None):
    """Four compartments with an ``_area`` each, two channel intensities, a
    two-channel colocalization on a ``cells_`` prefix and one feature that
    names no compartment; the label follows ``cell_channel_0_mean_intensity``."""
    rng = np.random.default_rng(seed)
    g = _grid(n)
    signal = rng.normal(size=n)
    cols = _metadata(g)
    cols.update({
        "cell_area": rng.uniform(1000.0, 4000.0, n),
        "nucleus_area": rng.uniform(100.0, 900.0, n),
        "pathogen_area": rng.uniform(10.0, 90.0, n),
        "cytoplasm_area": rng.uniform(500.0, 3000.0, n),
        "cell_channel_0_mean_intensity": 1500.0 + 400.0 * signal,
        "nucleus_channel_1_mean_intensity": rng.uniform(200.0, 900.0, n),
        "cells_channel_2_channel_3_colocalization": rng.uniform(0.0, 1.0, n),
        "field_focus_score": rng.uniform(0.0, 1.0, n),
    })
    df = pd.DataFrame(cols)
    if zero_pathogen_row is not None:
        df.loc[zero_pathogen_row, "pathogen_area"] = 0.0
    return df, g, (signal > 0).astype(int)


def _small_frame(n, seed=0):
    """Four features with distinct base names (no ratio partners); the label
    follows ``cell_area``."""
    rng = np.random.default_rng(seed)
    g = _grid(n)
    signal = rng.normal(size=n)
    cols = _metadata(g)
    cols.update({
        "cell_area": 2000.0 + 500.0 * signal,
        "cells_channel_0_channel_1_colocalization": rng.uniform(0.0, 1.0, n),
        "nucleus_channel_2_mean_intensity": rng.uniform(200.0, 900.0, n),
        "pathogen_channel_3_mean_intensity": rng.uniform(50.0, 400.0, n),
    })
    return pd.DataFrame(cols), g, (signal > 0).astype(int)


def _scores(g, labels, keys=None, score_column="score"):
    """The per-object prediction table that is merged onto the frame."""
    data = {"plateID": [x[0] for x in g], "fieldID": [x[3] for x in g],
            "object": np.arange(1, len(g) + 1), score_column: labels}
    data.update(keys if keys is not None else {
        "rowID": [x[1] for x in g], "column_name": [x[2] for x in g],
        "object_label": np.arange(1, len(g) + 1)})
    return pd.DataFrame(data)


def _settings(src, scores, **over):
    s = {"src": str(src), "scores": str(scores),
         "tables": ["cell", "nucleus", "pathogen", "cytoplasm"],
         "channels": ["channel_0", "channel_1", "channel_2", "channel_3"],
         "score_column": "score", "feature_importance": True,
         "permutation_importance": False, "shap": False, "shap_sample": False,
         "top_features": 4, "include_all": False, "nuclei_limit": 100,
         "pathogen_limit": 100, "n_jobs": 1, "save": False}
    s.update(over)
    return s


def _install(mp, df, rec):
    """Fake the merged-measurements reader and record every forest fit."""
    from sklearn.ensemble import RandomForestClassifier as _RF

    def _fake(locs, tables, verbose=False, nuclei_limit=None, pathogen_limit=None, **kw):
        rec["merge"] = {"locs": list(locs), "tables": list(tables), "verbose": verbose,
                        "nuclei_limit": nuclei_limit, "pathogen_limit": pathogen_limit}
        return df.copy(), []

    class _RecordingForest(_RF):
        def fit(self, X, y, sample_weight=None):
            rec["fits"].append((X.copy(), y.copy()))
            return super().fit(X, y, sample_weight)

    mp.setattr(spacr.io, "_read_and_merge_data", _fake)
    mp.setattr(spacr.submodules, "RandomForestClassifier", _RecordingForest)


def _raw_reader(mp):
    """The reader's documented raw-file mode, so legacy spellings reach the join."""
    from spacr.tabular import read_table
    mp.setattr(spacr.submodules, "read_table", functools.partial(read_table, canonicalise=False))


def _bar_axes():
    return [ax for num in plt.get_fignums() for ax in plt.figure(num).axes if ax.patches]


def _run(tmp_path, case_id, capsys, frame, scores, *, raw_reader=False, make_src=True, **over):
    """Write the scores CSV, run the explainer, return everything observable."""
    base = tmp_path / re.sub(r"[^A-Za-z0-9_]+", "_", case_id)
    base.mkdir()
    src = base / "plate"
    if make_src:
        src.mkdir()
    path = base / "scores.csv"
    scores.to_csv(path, index=False)
    rec = {"fits": [], "merge": None}
    capsys.readouterr()
    plt.close("all")
    with pytest.MonkeyPatch.context() as mp:
        _install(mp, frame, rec)
        if raw_reader:
            _raw_reader(mp)
        try:
            out, error = interperate_vision_model(_settings(src, path, **over)), None
        except MergeCardinalityError as exc:
            out, error = None, exc
    return dict(out=out, error=error, rec=rec, src=src, path=path,
                printed=capsys.readouterr().out)


# ---------------------------------------------------------------------------
# the join
# ---------------------------------------------------------------------------

def test_interperate_vision_model_joins_scores_onto_the_measurements(
        subtests, tmp_path, capsys):
    """Every key spelling of the scores file joins one row per scored object,
    ratios are expanded with zero denominators neutralised, and a key repeated
    on either side is refused by a message naming file, key and value."""
    with subtests.test(msg="no settings fail on the missing src key"):
        with pytest.raises(KeyError) as excinfo:
            interperate_vision_model(None)
        assert excinfo.value.args == ("src",)

    full, g24, lab24 = _full_frame(24, seed=2)
    alt = np.arange(24) % 2
    rows, cols = [x[1] for x in g24], [x[2] for x in g24]
    small, g20, lab20 = _small_frame(20, seed=17)
    doubled = _scores(g20, lab20)
    doubled = pd.concat([doubled, doubled.iloc[[0]]], ignore_index=True)
    timelapse = small.assign(timeID=1)
    timelapse = pd.concat([timelapse, timelapse.iloc[[0]].assign(timeID=2)],
                          ignore_index=True)
    frame30, g30, _ = _full_frame(30, seed=3)
    zero, gz, labz = _full_frame(24, seed=4, zero_pathogen_row=0)
    cases = [
        # (id, frame, scores, run kwargs, expected labels fitted or error fragments)
        ("rowID / column_name / object_label", full, _scores(g24, alt), {}, alt),
        ("legacy row_name / column_name", full,
         _scores(g24, alt, {"row_name": rows, "column_name": cols}), {}, alt),
        ("raw file: both spellings, the specific one names the well", full,
         _scores(g24, alt, {"row": ["r99"] * 24, "row_name": rows,
                            "column": ["c99"] * 24, "column_name": cols}),
         {"raw_reader": True}, alt),
        ("raw file: row and column_name", full,
         _scores(g24, alt, {"row": rows, "column_name": cols}), {"raw_reader": True}, alt),
        ("raw file: row_name and column", full,
         _scores(g24, alt, {"row_name": rows, "column": cols}), {"raw_reader": True}, alt),
        ("the modern columnID schema", small.rename(columns={"column_name": "columnID"}),
         _scores(g20, lab20, {"rowID": [x[1] for x in g20], "columnID": [x[2] for x in g20],
                              "object_label": np.arange(1, 21)}),
         {"tables": ["cell"], "top_features": 2}, lab20),
        ("the merge is inner: unscored objects are dropped", frame30,
         _scores(g30[:12], np.arange(12) % 2), {}, np.arange(12) % 2),
        ("a score listed twice is refused", small, doubled,
         {"tables": ["cell"], "top_features": 2},
         ["{path}", "object_label", repr(g20[0][3])]),
        ("a measurement row listed twice is refused",
         pd.concat([small, small.iloc[[0]]], ignore_index=True), _scores(g20, lab20),
         {"tables": ["cell"], "top_features": 2}, ["the merged measurements", "!TIMELAPSE"]),
        ("a time-lapse database says so", timelapse, _scores(g20, lab20),
         {"tables": ["cell"], "top_features": 2},
         ["TIMELAPSE", "timeID", "spacr.ml.interperate_vision_model"]),
    ]
    for case_id, frame, scores, kw, expected in cases:
        with subtests.test(msg=case_id):
            r = _run(tmp_path, case_id, capsys, frame, scores, **{"top_features": 3, **kw})
            if isinstance(expected, list):
                assert r["out"] is None and r["rec"]["fits"] == []
                message = str(r["error"])
                for fragment in expected:
                    fragment = fragment.replace("{path}", str(r["path"]))
                    if fragment.startswith("!"):
                        assert fragment[1:] not in message
                    else:
                        assert fragment in message, fragment
                continue
            assert r["error"] is None
            X, y = r["rec"]["fits"][0]
            assert y.tolist() == list(expected)
            assert len(X) == len(expected)
            assert "score" not in X.columns
            assert set(r["out"]) == {"feature_importance", "feature_importance_compartment",
                                     "feature_importance_channel"}

    with subtests.test(msg="ratios are expanded and a zero denominator becomes 0"):
        r = _run(tmp_path, "ratios", capsys, zero, _scores(gz, labz), top_features=3)
        X, _ = r["rec"]["fits"][0]
        assert X.shape == (24, 20)
        assert {"nucleus_cell_area", "cytoplasm_cell_area", "cell_nucleus_area",
                "pathogen_cytoplasm_area"} <= set(X.columns)
        np.testing.assert_allclose(X["nucleus_cell_area"], zero["nucleus_area"] / zero["cell_area"])
        assert X.loc[0, "cell_pathogen_area"] == 0.0 and X.loc[0, "pathogen_cell_area"] == 0.0
        assert np.isfinite(X.to_numpy()).all()
        assert "Expanded dataframe to 25 columns with relative features" in r["printed"]

    with subtests.test(msg="every explainer off still reads, merges and expands"):
        df, g, lab = _full_frame(24, seed=1)
        r = _run(tmp_path, "all off", capsys, df, _scores(g, lab), feature_importance=False,
                 permutation_importance=False, shap=False)
        assert r["out"] == {} and r["rec"]["fits"] == []
        assert r["rec"]["merge"] == {
            "locs": [os.path.join(str(r["src"]), "measurements", "measurements.db")],
            "tables": ["cell", "nucleus", "pathogen", "cytoplasm"], "verbose": True,
            "nuclei_limit": 100, "pathogen_limit": 100}
        assert "Expanded dataframe to 25 columns with relative features" in r["printed"]
        assert plt.get_fignums() == []


# ---------------------------------------------------------------------------
# the explainers
# ---------------------------------------------------------------------------

def _check_groups(out, include_all=False):
    """Group tables are the per-feature importance summed by group label."""
    fi = out["feature_importance"]
    for key, name in (("feature_importance_compartment", "compartment"),
                      ("feature_importance_channel", "channel")):
        table = out[key]
        assert list(table.columns) == [name, f"{name}_importance_sum"]
        body = table.iloc[:-1] if include_all else table
        if include_all:
            assert table[name].tolist().count("all") == 1 and table[name].iloc[-1] == "all"
            assert table[f"{name}_importance_sum"].iloc[-1] == pytest.approx(
                body[f"{name}_importance_sum"].sum())
        for group, total in zip(body[name], body[f"{name}_importance_sum"]):
            assert total == pytest.approx(fi.loc[fi[name] == group, "importance"].sum())
    chan = out["feature_importance_channel"]
    body = chan.iloc[:-1] if include_all else chan
    assert body["channel_importance_sum"].sum() == pytest.approx(1.0)


@pytest.mark.heavy  # its SHAP rows were a measured parallel memory amplifier
def test_interperate_vision_model_ranks_groups_and_saves_importances(
        subtests, tmp_path, capsys):
    """Each explainer returns its exact table shape topped by the feature the
    label was built from; the importances group by compartment and channel,
    plots show the top N, and ``save`` writes one CSV per table."""
    full40, g40, lab40 = _full_frame(40, seed=5)
    small40, gs40, labs40 = _small_frame(40, seed=8)
    small20, gs20, labs20 = _small_frame(20, seed=9)
    small24, gs24, labs24 = _small_frame(24, seed=12)
    small120, gs120, labs120 = _small_frame(120, seed=10)
    full24, g24, lab24 = _full_frame(24, seed=7)
    cases = [
        # (id, frame, scores, settings overrides, expected output keys)
        ("forest importance: ranked, grouped and plotted", full40, _scores(g40, lab40),
         {"top_features": 4},
         {"feature_importance", "feature_importance_compartment", "feature_importance_channel"}),
        ("include_all appends one total row per group table", full24, _scores(g24, lab24),
         {"top_features": 3, "include_all": True},
         {"feature_importance", "feature_importance_compartment", "feature_importance_channel"}),
        ("integer channel ids group cleanly", full24, _scores(g24, lab24),
         {"channels": [0, 1, 2, 3], "top_features": 3},
         {"feature_importance", "feature_importance_compartment", "feature_importance_channel"}),
        ("permutation importance next to the forest", small40, _scores(gs40, labs40),
         {"tables": ["cell", "nucleus", "pathogen"], "permutation_importance": True,
          "top_features": 2},
         {"feature_importance", "feature_importance_compartment", "feature_importance_channel",
          "permutation_importance"}),
        ("permutation importance on its own", small24, _scores(gs24, labs24),
         {"tables": ["cell"], "feature_importance": False, "permutation_importance": True,
          "top_features": 2}, {"permutation_importance"}),
        ("SHAP over every object, indexed by compartment and channel", small20,
         _scores(gs20, labs20), {"tables": ["cell", "nucleus", "pathogen"], "shap": True,
                                 "top_features": 4},
         {"feature_importance", "feature_importance_compartment", "feature_importance_channel",
          "shap"}),
        ("SHAP on its own", small20, _scores(gs20, labs20),
         {"tables": ["cell"], "feature_importance": False, "shap": True, "top_features": 2},
         {"shap"}),
        ("SHAP subsampled on a small plate", small24, _scores(gs24, labs24),
         {"tables": ["cell"], "shap": True, "shap_sample": True, "top_features": 2},
         {"feature_importance", "feature_importance_compartment", "feature_importance_channel",
          "shap"}),
        ("SHAP subsampled to one percent", small120, _scores(gs120, labs120),
         {"tables": ["cell"], "channels": ["channel_0"], "shap": True, "shap_sample": True,
          "top_features": 2},
         {"feature_importance", "feature_importance_compartment", "feature_importance_channel",
          "shap"}),
        ("save writes one CSV per table", full24, _scores(g24, lab24),
         {"top_features": 3, "save": True, "make_src": False},
         {"feature_importance", "feature_importance_compartment", "feature_importance_channel"}),
    ]
    for case_id, frame, scores, over, keys in cases:
        with subtests.test(msg=case_id):
            r = _run(tmp_path, case_id, capsys, frame, scores, **over)
            out = r["out"]
            assert set(out) == keys
            top = over["top_features"]
            if "feature_importance" in keys:
                fi = out["feature_importance"]
                assert list(fi.columns) == ["feature", "importance", "compartment", "channel"]
                assert fi["importance"].is_monotonic_decreasing
                assert fi["importance"].sum() == pytest.approx(1.0)
                assert "Feature Importance ..." in r["printed"]
                _check_groups(out, include_all=over.get("include_all", False))
                ax = [a for a in _bar_axes()
                      if a.get_title() == f"Top {top} Features - Feature Importance"]
                assert len(ax) == 1 and len(ax[0].patches) == min(top, len(fi))
                # the SHAP summary redraws its own x label onto the current axes
                assert ax[0].get_xlabel() == ("SHAP value (impact on model output)"
                                              if "shap" in keys else "Importance")
                if "shap" not in keys:
                    assert ax[0].get_ylim()[0] > ax[0].get_ylim()[1], "best feature on top"
            if "permutation_importance" in keys:
                perm = out["permutation_importance"]
                assert list(perm.columns) == ["feature", "importance"] and len(perm) == 4
                assert perm["importance"].is_monotonic_decreasing
                assert perm.iloc[0]["feature"] == "cell_area"
                assert perm.iloc[0]["importance"] > 0.1
                assert perm.iloc[-1]["importance"] == pytest.approx(0.0, abs=0.05)
                assert "Permutation Importance ..." in r["printed"]
                ax = [a for a in _bar_axes()
                      if a.get_title() == f"Top {top} Features - Permutation Importance"]
                assert len(ax) == 1 and len(ax[0].patches) == top
                assert ax[0].get_ylim()[0] > ax[0].get_ylim()[1]
            if "shap" in keys:
                shap_df = out["shap"]
                assert list(shap_df.columns.names) == ["compartment", "channel"]
                assert np.isfinite(shap_df.to_numpy()).all()
                assert "SHAP Analysis ..." in r["printed"]
            if case_id.startswith("forest"):
                fi = out["feature_importance"]
                assert len(fi) == 20
                assert fi.iloc[0]["feature"] == "cell_channel_0_mean_intensity"
                assert fi.loc[fi["feature"] == "field_focus_score", "compartment"].isna().all()
                assert fi.loc[fi["feature"] == "cell_area", "channel"].iloc[0] == "morphology"
                comp = set(out["feature_importance_compartment"]["compartment"])
                assert {"cell", "nucleus", "pathogen", "cytoplasm", "cell-nucleus",
                        "nucleus-pathogen"} <= comp
                chan = set(out["feature_importance_channel"]["channel"])
                assert {"channel_0", "channel_1", "morphology", "channel_2-channel_3"} <= chan
                assert out["feature_importance_compartment"][
                    "compartment_importance_sum"].sum() == pytest.approx(
                        fi.loc[fi["compartment"].notna(), "importance"].sum())
                ax = [a for a in _bar_axes() if a.get_title().endswith("Feature Importance")][0]
                assert "cell_channel_0_mean_intensity" in [t.get_text() for t in ax.get_yticklabels()]
            if case_id.startswith("integer"):
                chan = out["feature_importance_channel"]
                assert set(chan["channel"]) == {"channel_0", "channel_1", "channel_2-channel_3",
                                                "morphology"}
            if case_id.startswith("SHAP over every"):
                shap_df = out["shap"]
                assert shap_df.shape == (20, 4)
                assert set(shap_df.columns) == {("cell", "morphology"),
                                                ("cell", "channel_0 + channel_1"),
                                                ("nucleus", "channel_2"),
                                                ("pathogen", "channel_3")}
                assert shap_df.abs().mean().idxmax() == ("cell", "morphology")
            if case_id == "SHAP on its own":
                assert out["shap"].shape == (20, 2)
            if case_id == "SHAP subsampled on a small plate":
                assert out["shap"].shape == (1, 2)
            if case_id == "SHAP subsampled to one percent":
                assert out["shap"].shape == (1, 2)
            if case_id.startswith("permutation importance on"):
                assert r["rec"]["fits"] and plt.get_fignums()
            if case_id.startswith("save"):
                dst = r["src"] / "results"
                assert sorted(p.name for p in dst.iterdir()) == [
                    "feature_importance.csv", "feature_importance_channel.csv",
                    "feature_importance_compartment.csv"]
                for name, table in out.items():
                    saved = pd.read_csv(dst / f"{name}.csv", index_col=0)
                    assert saved.columns.tolist() == table.columns.tolist()
                    assert saved.iloc[:, 0].tolist() == table.iloc[:, 0].tolist()
                    np.testing.assert_allclose(saved.iloc[:, 1], table.iloc[:, 1])
