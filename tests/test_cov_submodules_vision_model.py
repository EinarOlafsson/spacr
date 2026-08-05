"""CPU-only coverage for ``spacr.submodules.interperate_vision_model``.

This is the *legacy* copy of the vision-model explainer (``spacr.ml`` carries a
newer one). It joins a per-object score CSV onto the merged measurements
frame, expands cross-compartment ratio features, then runs random-forest
importance, permutation importance and SHAP, grouping the result by
compartment and by channel.

Everything here runs against a fake ``io._read_and_merge_data`` so no sqlite
DB, no network and no CUDA are involved; the frames are shaped exactly like
the real helper's output. A recording ``RandomForestClassifier`` subclass is
installed where the test needs to assert on the feature matrix the function
actually built (the function only returns the importance tables).

Bugs found while writing this file were pinned with ``xfail(strict=True)``
asserting the CORRECT behaviour; all four are now fixed and the tests at the
bottom of the file are their plain regression tests:

* ``if settings['feature_importance'] or settings['feature_importance']``
  (a duplicated condition) left ``model`` / ``feature_importance_df``
  unbound when only permutation importance or only SHAP was requested,
* ``shap_sample`` drew ``int(len(X)/100)`` rows, i.e. zero rows for any
  experiment with fewer than 100 objects,
* ``group_feature_class`` regex-matched ``settings['channels']``, so the
  documented integer channel ids blew up,
* the merge keys used the legacy ``column_name`` while
  ``io._read_and_merge_data`` emits ``columnID`` (both are accepted now).
"""
from __future__ import annotations

import os
import types

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

# Imported up front so the slow torch/cellpose import chain behind
# spacr.submodules is paid at collection time, not charged to one test.
import spacr.io  # noqa: E402
import spacr.submodules  # noqa: E402


# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------

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
    """The five join columns; object_label carries the 'oN' prefix form."""
    return {
        "object_label": [f"o{i}" for i in range(1, len(g) + 1)],
        "plateID": [x[0] for x in g],
        "rowID": [x[1] for x in g],
        "column_name": [x[2] for x in g],
        "fieldID": [x[3] for x in g],
    }


def _measurement_frame(n, seed=0, zero_pathogen_row=None):
    """Full four-compartment frame: every compartment owns an ``_area`` column
    (so the ratio expansion has partners), two channel intensities, one
    two-channel colocalization on a ``cells_`` prefix, and one feature that
    belongs to no compartment at all."""
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


def _minimal_frame(n, seed=0):
    """Four numeric features with pairwise-distinct base names, so the ratio
    expansion adds nothing and the feature space stays exactly four wide.

    The names are chosen to drive every branch of
    ``extract_compartment_channel``: a ``cells_`` prefix that gets renamed to
    ``cell``, a feature naming two channels at once, one feature per
    remaining channel token, and a morphology feature with no channel token.
    """
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


def _install_fake_merge(monkeypatch, df, recorder=None):
    """Patch ``spacr.io._read_and_merge_data`` (resolved lazily inside the
    function under test) with a recording fake."""
    def _fake(locs, tables, verbose=False, nuclei_limit=None,
              pathogen_limit=None, **kwargs):
        if recorder is not None:
            recorder.update({"locs": list(locs), "tables": list(tables),
                             "verbose": verbose, "nuclei_limit": nuclei_limit,
                             "pathogen_limit": pathogen_limit})
        return df.copy(), []

    monkeypatch.setattr(spacr.io, "_read_and_merge_data", _fake)


def _write_scores(path, g, labels, extra=None, score_column="score"):
    """Write the per-object prediction CSV that gets merged onto the frame."""
    data = {
        "plateID": [x[0] for x in g],
        "fieldID": [x[3] for x in g],
        "object": np.arange(1, len(g) + 1),
        score_column: labels,
    }
    data.update(extra if extra is not None else {
        "rowID": [x[1] for x in g],
        "column_name": [x[2] for x in g],
        "object_label": np.arange(1, len(g) + 1),
    })
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def _record_forest(monkeypatch, store):
    """Install a RandomForestClassifier that records every (X, y) it is fit on."""
    from sklearn.ensemble import RandomForestClassifier as _RF

    class _RecordingForest(_RF):
        def fit(self, X, y, sample_weight=None):
            store.append((X.copy(), y.copy()))
            return super().fit(X, y, sample_weight)

    monkeypatch.setattr(spacr.submodules, "RandomForestClassifier",
                        _RecordingForest)


def _settings(src, scores, **over):
    """Every key ``interperate_vision_model`` reads; no defaults helper exists
    for the submodules copy."""
    s = {
        "src": str(src),
        "scores": str(scores),
        "tables": ["cell", "nucleus", "pathogen", "cytoplasm"],
        "channels": ["channel_0", "channel_1", "channel_2", "channel_3"],
        "score_column": "score",
        "feature_importance": True,
        "permutation_importance": False,
        "shap": False,
        "shap_sample": False,
        "top_features": 4,
        "include_all": False,
        "nuclei_limit": 100,
        "pathogen_limit": 100,
        "n_jobs": 1,
        "save": False,
    }
    s.update(over)
    return s


def _nested(name):
    """Rebuild a closure-free nested helper of ``interperate_vision_model``.

    ``create_extended_radar_plot`` has both of its call sites commented out
    (submodules.py lines 1616-1617) and the two ``... is None`` default
    branches are unreachable because the call sites always pass the argument;
    rebuilding the real code object is the only way to execute that product
    code. The helpers close over nothing, so module globals are enough.
    """
    from spacr.submodules import interperate_vision_model

    code = next(c for c in interperate_vision_model.__code__.co_consts
                if isinstance(c, types.CodeType) and c.co_name == name)
    assert code.co_freevars == (), f"{name} unexpectedly closes over state"
    return types.FunctionType(code, spacr.submodules.__dict__, name)


def _bar_axes():
    """Every open Axes that carries bar patches."""
    return [ax for num in plt.get_fignums() for ax in plt.figure(num).axes
            if ax.patches]


# ===========================================================================
# read_and_preprocess_data — merge, aliases, ratio expansion
# ===========================================================================

def test_no_settings_fails_on_the_missing_src_key():
    """``settings=None`` degrades to an empty dict, so the first thing the
    function needs — the measurements DB under ``src`` — is what it reports."""
    from spacr.submodules import interperate_vision_model

    with pytest.raises(KeyError) as excinfo:
        interperate_vision_model(None)
    assert excinfo.value.args[0] == "src"


def test_all_explainers_disabled_returns_an_empty_output_dict(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    """With feature_importance/permutation/shap all off the function still
    reads, merges and expands the frame, then returns nothing to report."""
    from spacr.submodules import interperate_vision_model

    df, g, labels = _measurement_frame(24, seed=1)
    rec = {}
    _install_fake_merge(monkeypatch, df, rec)
    src = tmp_path / "plateA"
    src.mkdir()
    scores = _write_scores(tmp_path / "scores_a.csv", g, labels)

    out = interperate_vision_model(_settings(
        src, scores, feature_importance=False, permutation_importance=False,
        shap=False))

    assert out == {}
    # the DB location and the object-count caps are derived from settings
    assert rec["locs"] == [str(src) + "/measurements/measurements.db"]
    assert rec["tables"] == ["cell", "nucleus", "pathogen", "cytoplasm"]
    assert rec["verbose"] is True
    assert rec["nuclei_limit"] == 100 and rec["pathogen_limit"] == 100
    # 5 metadata + 8 measured + 12 cross-compartment ratio columns
    assert "Expanded dataframe to 25 columns with relative features" \
        in capsys.readouterr().out
    assert not plt.get_fignums(), "nothing should have been plotted"


def test_legacy_row_and_column_aliases_drive_the_merge(tmp_path, monkeypatch):
    """A scores CSV with row/row_name (and column/column_name) instead of
    rowID/columnID still joins: the later alias wins, so only 'row_name' can
    make the merge succeed."""
    from spacr.submodules import interperate_vision_model

    df, g, _sig = _measurement_frame(24, seed=2)
    labels = np.arange(len(g)) % 2
    _install_fake_merge(monkeypatch, df)
    fits = []
    _record_forest(monkeypatch, fits)

    src = tmp_path / "plateB"
    src.mkdir()
    scores = _write_scores(tmp_path / "scores_b.csv", g, labels, extra={
        "row": ["WRONG"] * len(g),
        "row_name": [x[1] for x in g],
        "column": ["WRONG"] * len(g),
        "column_name": [x[2] for x in g],
    })

    out = interperate_vision_model(_settings(src, scores, top_features=3))

    X, y = fits[0]
    assert len(X) == len(g), "the aliased rowID must line every object up"
    assert y.tolist() == labels.tolist()
    assert "score" not in X.columns
    assert set(out) == {"feature_importance", "feature_importance_compartment",
                        "feature_importance_channel"}


def test_merge_is_inner_so_unscored_objects_are_dropped(tmp_path, monkeypatch):
    """Only objects present in the scores CSV survive into the feature matrix."""
    from spacr.submodules import interperate_vision_model

    df, g, _sig = _measurement_frame(30, seed=3)
    scored = g[:12]
    labels = np.arange(len(scored)) % 2
    _install_fake_merge(monkeypatch, df)
    fits = []
    _record_forest(monkeypatch, fits)

    src = tmp_path / "plateC"
    src.mkdir()
    scores = _write_scores(tmp_path / "scores_c.csv", scored, labels)

    interperate_vision_model(_settings(src, scores, top_features=3))

    X, y = fits[0]
    assert len(X) == 12
    assert sorted(set(y)) == [0, 1]


def test_comparison_columns_expand_and_neutralise_division_by_zero(
        tmp_path, monkeypatch, capsys):
    """Cross-compartment ratios are added as ``<num>_<den><base>`` columns,
    and a zero denominator is turned into 0 rather than inf/NaN."""
    from spacr.submodules import interperate_vision_model

    df, g, labels = _measurement_frame(24, seed=4, zero_pathogen_row=0)
    _install_fake_merge(monkeypatch, df)
    fits = []
    _record_forest(monkeypatch, fits)

    src = tmp_path / "plateD"
    src.mkdir()
    scores = _write_scores(tmp_path / "scores_d.csv", g, labels)

    interperate_vision_model(_settings(src, scores, top_features=3))

    X, _y = fits[0]
    # 8 measured + 12 ratios (4 compartments x 3 partners, pairwise names
    # collide with the direct ones)
    assert X.shape == (24, 20)
    for col in ("nucleus_cell_area", "cytoplasm_cell_area",
                "cell_nucleus_area", "pathogen_cytoplasm_area"):
        assert col in X.columns
    # '<numerator>_<denominator>_area'
    assert np.allclose(X["nucleus_cell_area"],
                       df["nucleus_area"] / df["cell_area"])
    # object 0 has pathogen_area == 0: inf -> pd.NA -> 0
    assert X.loc[0, "cell_pathogen_area"] == 0.0
    assert X.loc[0, "pathogen_cell_area"] == 0.0
    assert np.isfinite(X.to_numpy()).all()
    assert "Expanded dataframe to 25 columns" in capsys.readouterr().out


# ===========================================================================
# Step 1 — random-forest feature importance + compartment/channel grouping
# ===========================================================================

def test_feature_importance_table_is_ranked_and_plotted(tmp_path, monkeypatch,
                                                        capsys):
    """The RF importance table covers every numeric feature, sums to 1, ranks
    the only signal-carrying feature first, and the bar chart shows the top N
    with the y axis inverted."""
    from spacr.submodules import interperate_vision_model

    df, g, labels = _measurement_frame(40, seed=5)
    _install_fake_merge(monkeypatch, df)
    src = tmp_path / "plateE"
    src.mkdir()
    scores = _write_scores(tmp_path / "scores_e.csv", g, labels)

    out = interperate_vision_model(_settings(src, scores, top_features=4))

    fi = out["feature_importance"]
    # group_feature_class annotates the same frame in place
    assert list(fi.columns) == ["feature", "importance", "compartment", "channel"]
    assert len(fi) == 20
    assert fi["importance"].is_monotonic_decreasing
    assert fi["importance"].sum() == pytest.approx(1.0)
    assert fi.iloc[0]["feature"] == "cell_channel_0_mean_intensity"
    assert "Feature Importance ..." in capsys.readouterr().out

    bars = _bar_axes()
    assert len(bars) == 1
    ax = bars[0]
    assert len(ax.patches) == 4
    assert ax.get_xlabel() == "Importance"
    assert ax.get_title() == "Top 4 Features - Feature Importance"
    bottom, top = ax.get_ylim()
    assert bottom > top, "invert_yaxis() should put the best feature on top"
    labels_drawn = [t.get_text() for t in ax.get_yticklabels()]
    assert "cell_channel_0_mean_intensity" in labels_drawn


def test_importance_is_grouped_by_compartment_and_by_channel(tmp_path,
                                                             monkeypatch):
    """Compartment/channel groups sum the per-feature importance; features
    naming two groups get a hyphenated label, features naming no compartment
    are dropped, and features naming no channel fall back to 'morphology'."""
    from spacr.submodules import interperate_vision_model

    df, g, labels = _measurement_frame(40, seed=6)
    _install_fake_merge(monkeypatch, df)
    src = tmp_path / "plateF"
    src.mkdir()
    scores = _write_scores(tmp_path / "scores_f.csv", g, labels)

    out = interperate_vision_model(_settings(src, scores, top_features=4))

    fi = out["feature_importance"]
    comp = out["feature_importance_compartment"]
    chan = out["feature_importance_channel"]

    assert list(comp.columns) == ["compartment", "compartment_importance_sum"]
    assert list(chan.columns) == ["channel", "channel_importance_sum"]

    comp_groups = set(comp["compartment"])
    assert {"cell", "nucleus", "pathogen", "cytoplasm"} <= comp_groups
    # ratio features name two compartments -> hyphen-joined, in tables order
    assert "cell-nucleus" in comp_groups
    assert "nucleus-pathogen" in comp_groups
    for group in comp_groups:
        expected = fi.loc[fi["compartment"] == group, "importance"].sum()
        got = comp.loc[comp["compartment"] == group,
                       "compartment_importance_sum"].iloc[0]
        assert got == pytest.approx(expected)
    # 'field_focus_score' matches no compartment -> NaN -> dropped by groupby
    assert fi.loc[fi["feature"] == "field_focus_score",
                  "compartment"].isna().all()
    assert comp["compartment_importance_sum"].sum() == pytest.approx(
        fi.loc[fi["compartment"].notna(), "importance"].sum())

    chan_groups = set(chan["channel"])
    assert {"channel_0", "channel_1", "morphology"} <= chan_groups
    # one feature names two channels at once
    assert "channel_2-channel_3" in chan_groups
    # every feature lands in some channel group, so the total is the whole 1.0
    assert chan["channel_importance_sum"].sum() == pytest.approx(1.0)
    assert fi.loc[fi["feature"] == "cell_area", "channel"].iloc[0] == "morphology"


def test_include_all_appends_a_total_row_to_both_group_tables(tmp_path,
                                                              monkeypatch):
    """include_all=True adds a trailing 'all' row holding the group total."""
    from spacr.submodules import interperate_vision_model

    df, g, labels = _measurement_frame(24, seed=7)
    _install_fake_merge(monkeypatch, df)
    src = tmp_path / "plateG"
    src.mkdir()
    scores = _write_scores(tmp_path / "scores_g.csv", g, labels)

    out = interperate_vision_model(_settings(src, scores, top_features=3,
                                             include_all=True))

    for key, name in (("feature_importance_compartment", "compartment"),
                      ("feature_importance_channel", "channel")):
        table = out[key]
        assert table[name].iloc[-1] == "all"
        assert list(table[name]).count("all") == 1
        total = table[f"{name}_importance_sum"].iloc[-1]
        assert total == pytest.approx(table[f"{name}_importance_sum"][:-1].sum())
    assert out["feature_importance_channel"][
        "channel_importance_sum"].iloc[-1] == pytest.approx(1.0)


# ===========================================================================
# Step 2 — permutation importance
# ===========================================================================

def test_permutation_importance_ranks_the_predictive_feature(tmp_path,
                                                             monkeypatch,
                                                             capsys):
    """Shuffling the signal feature costs accuracy; shuffling the noise
    features costs nothing, so the table is topped by the signal."""
    from spacr.submodules import interperate_vision_model

    df, g, labels = _minimal_frame(40, seed=8)
    _install_fake_merge(monkeypatch, df)
    src = tmp_path / "plateH"
    src.mkdir()
    scores = _write_scores(tmp_path / "scores_h.csv", g, labels)

    out = interperate_vision_model(_settings(
        src, scores, tables=["cell", "nucleus", "pathogen"],
        permutation_importance=True, top_features=2))

    perm = out["permutation_importance"]
    assert list(perm.columns) == ["feature", "importance"]
    assert len(perm) == 4
    assert perm["importance"].is_monotonic_decreasing
    assert perm.iloc[0]["feature"] == "cell_area"
    assert perm.iloc[0]["importance"] > 0.1
    assert perm.iloc[-1]["importance"] == pytest.approx(0.0, abs=0.05)
    assert "Permutation Importance ..." in capsys.readouterr().out

    titles = [ax.get_title() for ax in _bar_axes()]
    assert "Top 2 Features - Feature Importance" in titles
    assert "Top 2 Features - Permutation Importance" in titles
    perm_ax = [ax for ax in _bar_axes()
               if ax.get_title().endswith("Permutation Importance")][0]
    assert len(perm_ax.patches) == 2
    bottom, top = perm_ax.get_ylim()
    assert bottom > top


# ===========================================================================
# Step 3 — SHAP
# ===========================================================================

def test_shap_frame_is_indexed_by_compartment_and_channel(tmp_path,
                                                          monkeypatch,
                                                          capsys):
    """Without subsampling every merged object is explained, and the SHAP
    matrix is re-labelled with a (compartment, channel) MultiIndex covering
    the 'cells' rename, the multi-channel join and the morphology fallback."""
    from spacr.submodules import interperate_vision_model

    df, g, labels = _minimal_frame(20, seed=9)
    _install_fake_merge(monkeypatch, df)
    src = tmp_path / "plateI"
    src.mkdir()
    scores = _write_scores(tmp_path / "scores_i.csv", g, labels)

    out = interperate_vision_model(_settings(
        src, scores, tables=["cell", "nucleus", "pathogen"],
        shap=True, shap_sample=False, top_features=4))

    shap_df = out["shap"]
    assert shap_df.shape == (20, 4)
    assert list(shap_df.columns.names) == ["compartment", "channel"]
    assert set(shap_df.columns) == {
        ("cell", "morphology"),                    # cell_area
        ("cell", "channel_0 + channel_1"),         # cells_..._colocalization
        ("nucleus", "channel_2"),
        ("pathogen", "channel_3"),
    }
    assert np.isfinite(shap_df.to_numpy()).all()
    # the label was built from cell_area, so that is where the signal sits
    mean_abs = shap_df.abs().mean()
    assert mean_abs.idxmax() == ("cell", "morphology")
    assert mean_abs[("cell", "morphology")] > 0.0
    assert "SHAP Analysis ..." in capsys.readouterr().out


def test_shap_sample_explains_one_percent_of_the_objects(tmp_path, monkeypatch):
    """shap_sample=True subsamples ``int(len(X)/100)`` rows before explaining."""
    from spacr.submodules import interperate_vision_model

    df, g, labels = _minimal_frame(120, seed=10)
    _install_fake_merge(monkeypatch, df)
    src = tmp_path / "plateJ"
    src.mkdir()
    scores = _write_scores(tmp_path / "scores_j.csv", g, labels)

    out = interperate_vision_model(_settings(
        src, scores, tables=["cell"], channels=["channel_0"],
        shap=True, shap_sample=True, top_features=2))

    shap_df = out["shap"]
    assert shap_df.shape == (int(120 / 100), 2)
    assert list(shap_df.columns.names) == ["compartment", "channel"]
    assert np.isfinite(shap_df.to_numpy()).all()


# ===========================================================================
# save
# ===========================================================================

def test_save_writes_one_csv_per_output_table(tmp_path, monkeypatch):
    """save=True creates <src>/results and dumps every produced table."""
    from spacr.submodules import interperate_vision_model

    df, g, labels = _measurement_frame(24, seed=11)
    _install_fake_merge(monkeypatch, df)
    src = tmp_path / "plateK"          # deliberately absent: makedirs must build it
    scores = _write_scores(tmp_path / "scores_k.csv", g, labels)

    out = interperate_vision_model(_settings(src, scores, top_features=3,
                                             save=True))

    dst = src / "results"
    assert dst.is_dir()
    assert sorted(p.name for p in dst.iterdir()) == [
        "feature_importance.csv",
        "feature_importance_channel.csv",
        "feature_importance_compartment.csv",
    ]
    saved = pd.read_csv(dst / "feature_importance.csv", index_col=0)
    assert saved["feature"].tolist() == out["feature_importance"]["feature"].tolist()
    assert np.allclose(saved["importance"], out["feature_importance"]["importance"])
    saved_chan = pd.read_csv(dst / "feature_importance_channel.csv", index_col=0)
    assert saved_chan["channel_importance_sum"].sum() == pytest.approx(1.0)


# ===========================================================================
# nested helpers that the call sites can never reach
# ===========================================================================

def test_radar_plot_helper_draws_a_closed_polar_loop():
    """``create_extended_radar_plot`` builds a filled polar axes whose ticks
    are the labels it was handed. Both call sites are commented out, so the
    helper is rebuilt from its code object."""
    radar = _nested("create_extended_radar_plot")

    values = [0.4, 0.3, 0.2]
    radar(values, ["cell", "nucleus", "pathogen"], "SHAP by compartment")

    axes = [ax for num in plt.get_fignums() for ax in plt.figure(num).axes]
    assert len(axes) == 1
    ax = axes[0]
    assert ax.name == "polar"
    assert ax.get_title() == "SHAP by compartment"
    assert [t.get_text() for t in ax.get_xticklabels()] == \
        ["cell", "nucleus", "pathogen"]
    # the loop is closed: 4 points for 3 labels, first == last
    (line,) = ax.lines
    xs, ys = line.get_xydata().T
    assert len(ys) == 4
    assert ys[0] == pytest.approx(ys[-1] if len(set(values)) > 1 else values[0])
    assert list(ys[:3]) == pytest.approx(values)
    assert xs[0] == pytest.approx(0.0)
    assert len(ax.patches) == 1, "the area under the curve is filled"


def test_comparison_columns_default_to_the_four_spacr_compartments():
    """``generate_comparison_columns`` defaults to cell/nucleus/pathogen/
    cytoplasm when no compartment list is given (the call site always passes
    one, so the default is rebuilt from the code object)."""
    gen = _nested("generate_comparison_columns")

    df = pd.DataFrame({"cell_area": [10.0, 20.0],
                       "nucleus_area": [5.0, 5.0],
                       "pathogen_area": [2.0, 0.0]})

    out, comparisons = gen(df, None)

    assert comparisons == {
        "cell_area": ["nucleus_area", "pathogen_area"],
        "nucleus_area": ["cell_area", "pathogen_area"],
        "pathogen_area": ["cell_area", "nucleus_area"],
    }
    assert out["nucleus_cell_area"].tolist() == [0.5, 0.25]
    assert out["cell_pathogen_area"].tolist() == [5.0, 0.0]  # 20/0 -> inf -> 0
    assert out["pathogen_nucleus_area"].tolist() == [0.4, 0.0]
    assert np.isfinite(out.to_numpy()).all()


def test_group_feature_class_defaults_to_the_four_compartments():
    """``group_feature_class`` falls back to the standard compartment list and
    drops features that match none of them."""
    group = _nested("group_feature_class")

    imp = pd.DataFrame({
        "feature": ["cell_area", "nucleus_cell_area", "pathogen_x", "zzz"],
        "importance": [0.4, 0.3, 0.2, 0.1],
    })

    summed = group(imp.copy(), None, "compartment", False)

    assert summed.set_index("compartment")["compartment_importance_sum"].to_dict() \
        == pytest.approx({"cell": 0.4, "cell-nucleus": 0.3, "pathogen": 0.2})
    assert "zzz" not in summed["compartment"].tolist()
    assert summed["compartment_importance_sum"].sum() == pytest.approx(0.9)


# ===========================================================================
# regression tests for bugs that were once pinned with xfail(strict=True)
# ===========================================================================

def test_permutation_importance_without_feature_importance(tmp_path,
                                                           monkeypatch):
    """Permutation importance must be usable on its own."""
    from spacr.submodules import interperate_vision_model

    df, g, labels = _minimal_frame(24, seed=12)
    _install_fake_merge(monkeypatch, df)
    src = tmp_path / "plateL"
    src.mkdir()
    scores = _write_scores(tmp_path / "scores_l.csv", g, labels)

    out = interperate_vision_model(_settings(
        src, scores, tables=["cell"], feature_importance=False,
        permutation_importance=True, top_features=2))

    assert "permutation_importance" in out
    assert len(out["permutation_importance"]) == 4


def test_shap_without_feature_importance(tmp_path, monkeypatch):
    """SHAP must be usable without also asking for RF feature importance."""
    from spacr.submodules import interperate_vision_model

    df, g, labels = _minimal_frame(20, seed=13)
    _install_fake_merge(monkeypatch, df)
    src = tmp_path / "plateM"
    src.mkdir()
    scores = _write_scores(tmp_path / "scores_m.csv", g, labels)

    out = interperate_vision_model(_settings(
        src, scores, tables=["cell"], feature_importance=False,
        shap=True, shap_sample=False, top_features=2))

    assert "shap" in out
    assert len(out["shap"]) == 20


def test_shap_sample_on_a_small_plate(tmp_path, monkeypatch):
    """shap_sample must degrade gracefully below 100 objects."""
    from spacr.submodules import interperate_vision_model

    df, g, labels = _minimal_frame(24, seed=14)
    _install_fake_merge(monkeypatch, df)
    src = tmp_path / "plateN"
    src.mkdir()
    scores = _write_scores(tmp_path / "scores_n.csv", g, labels)

    out = interperate_vision_model(_settings(
        src, scores, tables=["cell"], shap=True, shap_sample=True,
        top_features=2))

    assert len(out["shap"]) >= 1


def test_integer_channel_ids_are_accepted(tmp_path, monkeypatch):
    """``channels`` is an int list everywhere else in spacr (and in this
    function's own docstring), so [0, 1, 2, 3] must group cleanly."""
    from spacr.submodules import interperate_vision_model

    df, g, labels = _measurement_frame(24, seed=15)
    _install_fake_merge(monkeypatch, df)
    src = tmp_path / "plateO"
    src.mkdir()
    scores = _write_scores(tmp_path / "scores_o.csv", g, labels)

    out = interperate_vision_model(_settings(src, scores, channels=[0, 1, 2, 3],
                                             top_features=3))

    chan = out["feature_importance_channel"]
    assert len(chan) >= 2
    # every feature still lands in a channel group (unmatched -> morphology)
    assert chan["channel_importance_sum"].sum() == pytest.approx(1.0)


def test_modern_columnid_schema_from_read_and_merge_data(tmp_path, monkeypatch):
    """The frame handed back by ``io._read_and_merge_data`` names the plate
    column ``columnID``; the explainer has to cope with it."""
    from spacr.submodules import interperate_vision_model

    df, g, labels = _minimal_frame(20, seed=16)
    df = df.rename(columns={"column_name": "columnID"})
    _install_fake_merge(monkeypatch, df)
    src = tmp_path / "plateP"
    src.mkdir()
    scores = _write_scores(tmp_path / "scores_p.csv", g, labels, extra={
        "rowID": [x[1] for x in g],
        "columnID": [x[2] for x in g],
        "object_label": np.arange(1, len(g) + 1),
    })

    out = interperate_vision_model(_settings(src, scores, tables=["cell"],
                                             top_features=2))

    assert len(out["feature_importance"]) == 4


def test_a_score_listed_twice_for_one_object_is_caught_not_absorbed(
    tmp_path, monkeypatch
):
    """The scores join is one_to_one; a repeated score row is the right half.

    A scores CSV concatenated from two runs lists an object twice. The join
    then duplicates that object's measurement row, so it is fitted twice and
    weighted double in every importance below — with no row count, no
    warning and no visible difference in the tables that come back. The
    declared cardinality turns it into a MergeCardinalityError that names the
    scores file, the key and the offending value instead.
    """
    from spacr.io import MergeCardinalityError

    from spacr.submodules import interperate_vision_model

    df, g, labels = _minimal_frame(20, seed=17)
    _install_fake_merge(monkeypatch, df)
    src = tmp_path / "plateQ"
    src.mkdir()

    scores_path = tmp_path / "scores_q.csv"
    _write_scores(scores_path, g, labels)
    scores = pd.read_csv(scores_path)
    doubled = pd.concat([scores, scores.iloc[[0]]], ignore_index=True)
    doubled.to_csv(scores_path, index=False)

    # What the join used to do with it: one measurement row silently cloned.
    # Same five keys, and the same 'oN' -> 'N' object fixup, as the function.
    keys = ["plateID", "rowID", "columnID", "fieldID", "object_label"]
    left = df.rename(columns={"column_name": "columnID"}).copy()
    left["object_label"] = left["object_label"].str.replace("o", "")
    right = doubled.rename(columns={"column_name": "columnID"}).copy()
    right["object_label"] = right["object"].astype(str)
    grown = left.merge(right[keys + ["score"]], on=keys, how="inner")
    assert len(grown) == len(left) + 1

    with pytest.raises(MergeCardinalityError) as excinfo:
        interperate_vision_model(_settings(src, scores_path, tables=["cell"],
                                           top_features=2))

    # The message has to say WHICH file and WHICH key, or the user is back to
    # pandas' "Merge keys are not unique in right dataset".
    message = str(excinfo.value)
    assert str(scores_path) in message
    assert "object_label" in message
    # ...and it has to name the duplicated identity itself, not just the key.
    assert repr(g[0][3]) in message      # the fieldID of the doubled row


def test_a_measurement_row_listed_twice_is_caught_too(tmp_path, monkeypatch):
    """The LEFT half of one_to_one, which many_to_one let through in silence.

    A measurements frame that repeats an identity is what a timelapse database
    looks like through this key: ``prcfo`` carries the timepoint and this join
    does not, so every frame of an object is a separate row with the same
    ``(plate, row, column, field, object)``. Under ``many_to_one`` that joined
    cleanly and gave each frame the same score — one object entering the forest
    once per frame, weighted by how long it was imaged. The row count does not
    give it away either, because the join is inner.
    """
    from spacr.io import MergeCardinalityError

    from spacr.submodules import interperate_vision_model

    df, g, labels = _minimal_frame(20, seed=18)
    # Two frames of the same field/object: identical identity, different pixels.
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    _install_fake_merge(monkeypatch, df)
    src = tmp_path / "plateR"
    src.mkdir()

    scores_path = _write_scores(tmp_path / "scores_r.csv", g, labels)

    # Under the old contract this merged happily and cloned nothing visible:
    # 21 measurement rows in, 21 rows out, one object counted twice.
    keys = ["plateID", "rowID", "columnID", "fieldID", "object_label"]
    left = df.rename(columns={"column_name": "columnID"}).copy()
    left["object_label"] = left["object_label"].str.replace("o", "")
    right = pd.read_csv(scores_path).rename(columns={"column_name": "columnID"})
    right["object_label"] = right["object"].astype(str)
    right[keys] = right[keys].astype(str)
    left[keys] = left[keys].astype(str)
    absorbed = left.merge(right[keys + ["score"]], on=keys, how="inner",
                          validate="many_to_one")
    assert len(absorbed) == len(df)

    with pytest.raises(MergeCardinalityError) as excinfo:
        interperate_vision_model(_settings(src, scores_path, tables=["cell"],
                                           top_features=2))
    assert "the merged measurements" in str(excinfo.value)
    # Nothing in this frame says "timelapse", so nothing may claim it is one.
    assert "TIMELAPSE" not in str(excinfo.value)


def test_a_timelapse_database_says_so_instead_of_naming_only_the_key(
    tmp_path, monkeypatch
):
    """The one cause that is a property of the database, not of a bad file.

    On a timelapse ``_read_and_merge_data`` returns one row per object PER
    FRAME (``prcfo`` carries the timepoint) and the scores CSV holds one crop
    per frame too, so BOTH sides repeat under this timepoint-less key. The
    retracted comment claimed the opposite — that a timelapse was the reason
    the left side could not be asserted unique — and left the user with
    pandas' "Merge keys are not unique", which names neither the timepoint nor
    the newer explainer that does handle it.
    """
    from spacr.io import MergeCardinalityError

    from spacr.submodules import interperate_vision_model

    df, g, labels = _minimal_frame(20, seed=19)
    # Two frames of the same object, spelled the way the DB spells them.
    df["timeID"] = 1
    second = df.iloc[[0]].copy()
    second["timeID"] = 2
    df = pd.concat([df, second], ignore_index=True)
    _install_fake_merge(monkeypatch, df)
    src = tmp_path / "plateS"
    src.mkdir()

    scores_path = _write_scores(tmp_path / "scores_s.csv", g, labels)

    with pytest.raises(MergeCardinalityError) as excinfo:
        interperate_vision_model(_settings(src, scores_path, tables=["cell"],
                                           top_features=2))
    message = str(excinfo.value)
    assert "TIMELAPSE" in message
    assert "timeID" in message
    assert "spacr.ml.interperate_vision_model" in message
