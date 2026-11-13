"""CPU-only coverage for the SHAP / vision-model interpretation block of ``spacr.ml``.

Covers the tail of ``spacr/ml.py`` (lines ~2377-end):

* ``shap_analysis``            — SHAP summary figure for a fitted estimator,
* ``find_optimal_threshold``   — F1-maximising probability cut-off,
* ``_calculate_similarity``    — control-similarity metrics, including the
  singular-covariance fallback, the per-row ``safe_similarity`` swallow and the
  outer ``except`` guard (both reached by real failure injection),
* ``interperate_vision_model`` — the full RF / permutation / SHAP explainer with
  its compartment + channel radar plots.

The end-to-end test for ``interperate_vision_model`` runs against a *real*
synthetic ``measurements.db`` so ``io._read_and_merge_data`` executes for real;
the branch-level tests swap in a fast fake merge so the file stays quick.

No network, no CUDA, no TensorFlow.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

# Imported up front (rather than lazily inside the first test that needs them)
# so the multi-second torch/cellpose/skimage import chain behind spacr.utils is
# paid at collection time instead of being charged to one arbitrary test.
import spacr.io  # noqa: E402,F401
import spacr.ml  # noqa: E402,F401
import spacr.utils  # noqa: E402,F401


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
# synthetic data builders
# ---------------------------------------------------------------------------

_ROWS = ["r1", "r2"]
_COLUMNS = [f"c{i}" for i in range(1, 11)]
_FIELDS = [f"f{i}" for i in range(1, 11)]


def _grid():
    """(plateID, rowID, columnID, fieldID) tuples — one per object."""
    return [("plate1", r, c, f) for r in _ROWS for c in _COLUMNS for f in _FIELDS]


def _metadata_cols(grid):
    n = len(grid)
    return {
        "object_label": np.arange(1, n + 1),
        "plateID": [g[0] for g in grid],
        "rowID": [g[1] for g in grid],
        "columnID": [g[2] for g in grid],
        "fieldID": [g[3] for g in grid],
        "prcf": [f"{g[0]}_{g[1]}_{g[2]}_{g[3]}" for g in grid],
        "prc": [f"{g[0]}_{g[1]}_{g[2]}" for g in grid],
    }


def _build_measurements_db(src, rng):
    """Write ``<src>/measurements/measurements.db`` with cell + cytoplasm tables.

    Column names are chosen so that ``extract_compartment_channel`` inside
    ``interperate_vision_model`` sees every branch it has: a ``cells_*``
    prefix (renamed to ``cell``), a morphology feature with no channel token,
    and one feature per channel_0..channel_3.

    Returns ``(db_path, signal)`` where ``signal`` drives the label so the
    random forest has something real to learn.
    """
    grid = _grid()
    n = len(grid)
    signal = rng.normal(size=n)

    cell = _metadata_cols(grid)
    cell["cell_channel_0_mean_intensity"] = 1500.0 + 400.0 * signal
    cell["cell_channel_1_mean_intensity"] = rng.uniform(500, 3000, n)

    cyto = _metadata_cols(grid)
    cyto["cytoplasm_channel_2_mean_intensity"] = rng.uniform(100, 900, n)
    cyto["cytoplasm_channel_3_mean_intensity"] = rng.uniform(100, 900, n)

    meas = os.path.join(str(src), "measurements")
    os.makedirs(meas, exist_ok=True)
    db = os.path.join(meas, "measurements.db")
    con = sqlite3.connect(db)
    try:
        pd.DataFrame(cell).to_sql("cell", con, index=False)
        pd.DataFrame(cyto).to_sql("cytoplasm", con, index=False)
    finally:
        con.close()
    return db, signal


def _write_scores_csv(path, grid, labels, extra_cols):
    """Write the per-object prediction CSV that gets merged onto the DB."""
    data = {
        "plateID": [g[0] for g in grid],
        "fieldID": [g[3] for g in grid],
        "object": np.arange(1, len(grid) + 1),
        "cv_predictions": labels,
    }
    data.update(extra_cols)
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return df


def _fake_merged_frame(n_objects, rng):
    """A DataFrame shaped like the output of ``io._read_and_merge_data``."""
    grid = _grid()[:n_objects]
    cols = _metadata_cols(grid)
    # object_label comes back from the real helper as the 'oN' string form.
    cols["object_label"] = [f"o{i}" for i in range(1, len(grid) + 1)]
    signal = rng.normal(size=len(grid))
    cols["cell_channel_0_mean_intensity"] = 1500.0 + 400.0 * signal
    cols["nucleus_channel_1_mean_intensity"] = rng.uniform(200, 900, len(grid))
    cols["cell_area"] = rng.uniform(200, 4000, len(grid))
    df = pd.DataFrame(cols)
    labels = (signal > 0).astype(int)
    return df, grid, labels


def _install_fake_merge(monkeypatch, df, recorder):
    """Patch ``spacr.io._read_and_merge_data`` with a recording fake."""
    def _fake(locs, tables, verbose=False, nuclei_limit=None,
              pathogen_limit=None, **kwargs):
        recorder.update({"locs": list(locs), "tables": list(tables),
                         "verbose": verbose, "nuclei_limit": nuclei_limit,
                         "pathogen_limit": pathogen_limit})
        return df.copy(), []

    monkeypatch.setattr(spacr.io, "_read_and_merge_data", _fake)


@pytest.fixture()
def gallery(monkeypatch):
    """Every figure the run announces, as ``(title, folder)``.

    THE FIGURES MOVED TO PYQTGRAPH, so `plt.get_fignums()` is empty by
    design and reading it would silently pass forever. A drawn figure is
    observed where the user meets it: announced to the figure sink, with
    its data beside it in the bundle. That is a real artefact rather than a
    private hook, so a test that reads it is testing the delivery as well as
    the drawing.
    """
    from spacr import figure_sink

    seen = []
    figure_sink.set_file_sink(
        lambda path, title: seen.append((str(title or ""),
                                         os.path.dirname(str(path)))))
    yield seen
    figure_sink.clear_sink()


def _names_in(folder):
    """The labels the figure in ``folder`` was drawn from.

    Every bundle carries the numbers behind its picture, which for a radar
    or a bar chart is exactly the list of labels -- so the assertion reads
    the file the user would open.
    """
    frame = pd.read_csv(os.path.join(folder, "data.csv"))
    column = "name" if "name" in frame.columns else frame.columns[0]
    return set(frame[column].astype(str))


def _titled(gallery, word):
    """Folders of the announced figures whose title contains ``word``."""
    return [folder for title, folder in gallery if word in title]


def _polar_axes():
    """Every polar Axes currently held by an open figure.

    Kept for the ONE assertion that still wants to prove there are none:
    the radar moved to pyqtgraph, and a matplotlib polar axis appearing
    again would mean the second renderer had come back.
    """
    out = []
    for num in plt.get_fignums():
        for ax in plt.figure(num).axes:
            if ax.name == "polar":
                out.append(ax)
    return out


# ===========================================================================
# shap_analysis
# ===========================================================================


def _row_labels(plot):
    """The feature names on a pyqtgraph plot's left axis.

    The beeswarm's rows ARE its labels, so this is the pyqtgraph spelling of
    what these tests used to read off `ax.get_yticklabels()`.
    """
    axis = plot.plot.getAxis("left")
    return {str(text) for group in (axis._tickLevels or [])
            for _value, text in group}


def case_shap_analysis_returns_closed_summary_figure():
    """shap_analysis explains X_test and hands back a closed summary figure
    whose y ticks are the explained feature names."""
    _seen = []
    from sklearn.linear_model import LogisticRegression

    from spacr.ml import shap_analysis

    rng = np.random.default_rng(0)
    feats = ["cell_area", "nucleus_area", "cell_channel_0_mean_intensity"]
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=feats)
    y = (X["cell_area"] * 2 + rng.normal(scale=0.1, size=60) > 0).astype(int)
    X_train, X_test = X.iloc[:40], X.iloc[40:]
    model = LogisticRegression(max_iter=500).fit(X_train, y[:40])

    plot = shap_analysis(model, X_train, X_test)

    # A PYQTGRAPH BEESWARM, not a matplotlib figure. `shap.summary_plot`
    # draws into a figure it makes itself, so it could not be handed a
    # scene -- it was the last thing on the explanation path keeping a
    # second renderer alive, and two renderers is how one screen comes to
    # produce two pictures of one number.
    _seen.append(bool(
        not isinstance(plot, matplotlib.figure.Figure)))
    _seen.append(bool(
        _row_labels(plot) >= set(feats)))
    # And nothing was left on the pyplot stack, because nothing went on it.
    _seen.append(keep(plt.get_fignums()))
    plot.deleteLater()
    return _seen


def case_shap_analysis_tree_model_labels_every_feature():
    """A tree model routes through TreeExplainer; the returned figure still
    carries one beeswarm row per feature."""
    _seen = []
    from sklearn.ensemble import RandomForestRegressor

    from spacr.ml import shap_analysis

    rng = np.random.default_rng(1)
    feats = ["f_a", "f_b"]
    X = pd.DataFrame(rng.normal(size=(30, 2)), columns=feats)
    y = X["f_a"] * 3.0 + rng.normal(scale=0.05, size=30)
    model = RandomForestRegressor(n_estimators=8, random_state=0, n_jobs=1)
    model.fit(X, y)

    plot = shap_analysis(model, X, X.iloc[:10])

    _seen.append(bool(
        _row_labels(plot) >= set(feats)))
    plot.deleteLater()
    return _seen


def case_shap_analysis_tree_classifier_selects_the_positive_class():
    """A binary tree explainer's output axis is not an interaction matrix."""
    _seen = []
    from sklearn.ensemble import RandomForestClassifier

    from spacr.ml import shap_analysis

    rng = np.random.default_rng(2)
    feats = ["cell_area", "pathogen_area", "nucleus_area"]
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=feats)
    y = (X["cell_area"] - X["pathogen_area"] > 0).astype(int)
    model = RandomForestClassifier(
        n_estimators=8, random_state=0, n_jobs=1
    ).fit(X, y)

    plot = shap_analysis(model, X, X.iloc[:10])

    _seen.append(bool(
        _row_labels(plot) >= set(feats)))
    plot.deleteLater()
    return _seen


# ===========================================================================
# find_optimal_threshold
# ===========================================================================

def case_find_optimal_threshold_picks_the_f1_maximising_cut():
    """On a cleanly separable problem the returned threshold really is the
    F1 optimum over the precision-recall sweep."""
    _seen = []
    from sklearn.metrics import f1_score

    from spacr.ml import find_optimal_threshold

    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    proba = np.array([0.05, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.95])

    t = float(find_optimal_threshold(y_true, proba))

    _seen.append(bool(
        0.3 < t <= 0.7))
    _seen.append(bool(
        f1_score(y_true, (proba >= t).astype(int)) == pytest.approx(1.0)))
    return _seen


def case_find_optimal_threshold_ignores_undefined_f1_points():
    """Precision-recall sweeps can contain points where precision and recall
    are both 0; the F1 there is undefined and must not win the argmax."""
    _seen = []
    from sklearn.metrics import f1_score

    from spacr.ml import find_optimal_threshold

    # At thresholds 0.8 / 0.9 every predicted positive is a true negative
    # -> precision == recall == 0 -> 0/0 -> NaN.
    y_true = np.array([1, 0, 0])
    proba = np.array([0.1, 0.9, 0.8])

    t = float(find_optimal_threshold(y_true, proba))

    best = max(f1_score(y_true, (proba >= c).astype(int)) for c in proba)
    _seen.append(bool(
        f1_score(y_true, (proba >= t).astype(int)) == pytest.approx(best)))
    return _seen


# ===========================================================================
# _calculate_similarity
# ===========================================================================

def _similarity_frame(rng, n=24, constant_feature=False):
    data = {
        "columnID": ["c1"] * (n // 2) + ["c2"] * (n - n // 2),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    }
    if constant_feature:
        # Zero variance -> the standardised column is all zeros -> the
        # covariance matrix is exactly singular.
        data["f3"] = np.ones(n)
    else:
        data["f3"] = rng.normal(size=n)
    return pd.DataFrame(data)


def case_calculate_similarity_regularises_a_singular_covariance():
    """A zero-variance feature makes np.cov singular; the LinAlgError branch
    regularises the diagonal so the Mahalanobis columns still come out finite."""
    _seen = []
    from sklearn.preprocessing import StandardScaler

    from spacr.ml import _calculate_similarity

    rng = np.random.default_rng(7)
    feats = ["f1", "f2", "f3"]
    df = _similarity_frame(rng, constant_feature=True)

    # Precondition: the un-regularised inverse really does blow up, so the
    # except-branch below is genuinely exercised (not dead code).
    cov = np.cov(StandardScaler().fit_transform(df[feats]), rowvar=False)
    with pytest.raises(np.linalg.LinAlgError) as _exc1:
        np.linalg.inv(cov)
    _seen.append(('LinAlgError' if isinstance(_exc1.value, np.linalg.LinAlgError) else type(_exc1.value).__name__))

    out = _calculate_similarity(df.copy(), feats, "columnID", "c1", "c2")

    for col in ("similarity_to_pos_mahalanobis", "similarity_to_neg_mahalanobis"):
        assert col in out.columns
        assert out[col].notna().all()
        assert np.isfinite(out[col].to_numpy()).all()
        assert (out[col].to_numpy() >= 0).all()
    return _seen


def case_calculate_similarity_swallows_a_failing_metric(monkeypatch):
    """safe_similarity turns a raising distance function into NaN without
    taking the other metrics down with it."""
    _seen = []
    rng = np.random.default_rng(8)
    feats = ["f1", "f2", "f3"]
    df = _similarity_frame(rng)

    calls = {"n": 0}

    def _exploding_euclidean(*args, **kwargs):
        calls["n"] += 1
        raise ValueError("synthetic distance failure")

    monkeypatch.setattr(spacr.ml, "euclidean", _exploding_euclidean)

    out = spacr.ml._calculate_similarity(df.copy(), feats, "columnID", "c1", "c2")

    _seen.append(bool(
        calls["n"] == 2 * len(df)))
    _seen.append(bool(
        out["similarity_to_pos_euclidean"].isna().all()))
    _seen.append(bool(
        out["similarity_to_neg_euclidean"].isna().all()))
    # Everything downstream of euclidean still computed normally.
    _seen.append(bool(
        out["similarity_to_pos_cosine"].notna().all()))
    _seen.append(bool(
        out["similarity_to_neg_braycurtis"].notna().all()))
    return _seen


def case_calculate_similarity_reports_and_returns_on_assignment_failure(capsys):
    """If writing a similarity column raises, the outer guard prints the error
    and returns the partially-populated frame instead of propagating."""
    _seen = []
    from spacr.ml import _calculate_similarity

    class _ExplodingFrame(pd.DataFrame):
        """DataFrame that refuses one specific column assignment."""

        @property
        def _constructor(self):
            return _ExplodingFrame

        def _constructor_from_mgr(self, mgr, axes):
            # Required on pandas 2.2.x, harmless on 2.3+. A subclass that
            # overrides only `_constructor` sends pandas down
            # `self._constructor(mgr)` on every internal reconstruction, and
            # 2.2 DeprecationWarns there ("Passing a BlockManager to
            # _ExplodingFrame is deprecated"). The min-deps CI job pins the
            # declared floor pandas==2.2.1 and turns warnings into errors, so
            # this test failed there and nowhere else -- a property of the
            # fixture, not of spacr.ml. pandas 2.3 stopped warning; overriding
            # the hook is what makes the fixture correct on BOTH, and it
            # preserves the subclass either way.
            return _ExplodingFrame._from_mgr(mgr, axes=axes)

        def __setitem__(self, key, value):
            if key == "similarity_to_pos_cosine":
                raise RuntimeError("synthetic assignment failure")
            super().__setitem__(key, value)

    rng = np.random.default_rng(9)
    df = _ExplodingFrame(_similarity_frame(rng))

    out = _calculate_similarity(df, ["f1", "f2", "f3"], "columnID", "c1", "c2")

    printed = capsys.readouterr().out
    _seen.append(('Error calculating similarity scores' if 'Error calculating similarity scores' in (printed) else ("missing", 'Error calculating similarity scores')))
    _seen.append(('synthetic assignment failure' if 'synthetic assignment failure' in (printed) else ("missing", 'synthetic assignment failure')))
    sim_cols = [c for c in out.columns if c.startswith("similarity_")]
    # Only the two euclidean columns made it in before the failure.
    _seen.append(keep(sim_cols))
    return _seen


def case_calculate_similarity_accepts_list_valued_controls():
    """List controls average across several wells; the resulting reference is
    the mean of the pooled subset."""
    _seen = []
    from scipy.spatial.distance import euclidean as _euc

    from spacr.ml import _calculate_similarity

    rng = np.random.default_rng(10)
    n = 30
    df = pd.DataFrame({
        "columnID": (["c1", "c2", "c3"] * 10)[:n],
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "f3": rng.normal(size=n),
    })
    feats = ["f1", "f2", "f3"]
    expected_pos = df[df["columnID"].isin(["c2", "c3"])][feats].mean()

    out = _calculate_similarity(df.copy(), feats, "columnID", ["c2", "c3"], ["c1"])

    manual = _euc(df.loc[0, feats].to_numpy(float), expected_pos.to_numpy(float))
    _seen.append(bool(
        out.loc[0, "similarity_to_pos_euclidean"] == pytest.approx(manual)))
    # Chebyshev is the max coordinate gap — check it against the same reference.
    manual_cheb = np.abs(df.loc[0, feats].to_numpy(float)
                         - expected_pos.to_numpy(float)).max()
    _seen.append(bool(
        out.loc[0, "similarity_to_pos_chebyshev"] == pytest.approx(manual_cheb)))
    return _seen


# ===========================================================================
# interperate_vision_model
# ===========================================================================
#
# ORDERING NOTE: the cheap fake-merge cases come first on purpose. SHAP's
# explainer machinery has a one-off multi-second warm-up, and letting a small
# fixture absorb it keeps the real-database end-to-end test at the bottom of
# this module comfortably fast.


def case_interperate_vision_model_maps_legacy_score_columns(tmp_path, monkeypatch, gallery):
    """Legacy row/column names map to the canonical object-join columns.

    A missing ``object_label`` is also filled from ``object``.
    """
    _seen = []
    from spacr.ml import interperate_vision_model

    rng = np.random.default_rng(11)
    df, grid, labels = _fake_merged_frame(24, rng)
    rec = {}
    _install_fake_merge(monkeypatch, df, rec)

    src = tmp_path / "plateA"
    src.mkdir()
    scores_csv = tmp_path / "legacy_scores.csv"
    # Use one supported legacy spelling for each coordinate.  Conflicting
    # aliases are resolved centrally by the schema layer and are a separate
    # collision-policy contract, not part of this merge test.
    _write_scores_csv(scores_csv, grid, labels, {
        "row_name": [g[1] for g in grid],
        "column": [g[2] for g in grid],
    })

    merged = interperate_vision_model({
        "src": str(src),
        "scores": str(scores_csv),
        "tables": ["cell"],
        "score_column": "cv_predictions",
        "feature_importance": True,
        "permutation_importance": False,
        "shap": False,
        "top_features": 3,
        "n_jobs": 1,
        "save": False,
    })

    _seen.append(bool(
        len(merged) == len(grid)))
    _seen.append(bool(
        set(merged["rowID"]) == {g[1] for g in grid}))
    _seen.append(bool(
        set(merged["columnID"]) == {g[2] for g in grid}))
    # object_label was populated from the 'object' column and 'o'-stripped
    _seen.append(bool(
        merged["object_label"].tolist() == [str(i) for i in range(1, len(grid) + 1)]))
    # the DB location handed to the merge helper is derived from src
    _seen.append(bool(
        rec["locs"] == [os.path.join(str(src), "measurements", "measurements.db")]))
    _seen.append(keep(rec["tables"]))
    _seen.append(bool(
        rec["nuclei_limit"] == 1000 and rec["pathogen_limit"] == 1000))

    # feature-importance chart was drawn with the top-N features
    drawn = _titled(gallery, "Feature Importance")
    _seen.append(bool(
        drawn))
    _seen.append(('Top 3 Features' if 'Top 3 Features' in (gallery[0][0]) else ("missing", 'Top 3 Features')))
    _seen.append(keep(len(_names_in(drawn[0]))))
    _seen.append(bool(
        not _polar_axes()))
    return _seen


def case_interperate_vision_model_keeps_canonical_score_columns(tmp_path, monkeypatch, gallery):
    """When the scores CSV already carries rowID/columnID/object_label none of
    the legacy fallbacks fire and the merge still lines up 1:1."""
    _seen = []
    from spacr.ml import interperate_vision_model

    rng = np.random.default_rng(12)
    df, grid, labels = _fake_merged_frame(24, rng)
    _install_fake_merge(monkeypatch, df, {})

    src = tmp_path / "plateB"
    src.mkdir()
    scores_csv = tmp_path / "canonical_scores.csv"
    _write_scores_csv(scores_csv, grid, labels, {
        "rowID": [g[1] for g in grid],
        "columnID": [g[2] for g in grid],
        "object_label": np.arange(1, len(grid) + 1),
    })

    merged = interperate_vision_model({
        "src": str(src),
        "scores": str(scores_csv),
        "tables": ["cell"],
        "score_column": "cv_predictions",
        "feature_importance": True,
        "permutation_importance": True,
        "shap": False,
        "top_features": 2,
        "n_jobs": 1,
        "save": False,
    })

    _seen.append(bool(
        len(merged) == len(grid)))
    _seen.append(bool(
        merged["cv_predictions"].tolist() == list(labels)))
    # both charts (feature importance + permutation importance) exist
    titles = [title for title, _folder in gallery]
    _seen.append(bool(
        any("Feature Importance" in t for t in titles)))
    _seen.append(bool(
        any("Permutation Importance" in t for t in titles)))
    _seen.append(bool(
        all("Top 2 Features" in t for t in titles)))
    return _seen


def case_interperate_vision_model_shap_without_subsampling(tmp_path, monkeypatch, gallery):
    """shap_sample=False explains every merged object rather than a 1% draw,
    and the SHAP matrix is aggregated into compartment/channel radar plots."""
    _seen = []
    from spacr.ml import interperate_vision_model

    rng = np.random.default_rng(13)
    df, grid, labels = _fake_merged_frame(12, rng)
    _install_fake_merge(monkeypatch, df, {})

    src = tmp_path / "plateC"
    src.mkdir()
    scores_csv = tmp_path / "shap_scores.csv"
    _write_scores_csv(scores_csv, grid, labels, {
        "rowID": [g[1] for g in grid],
        "columnID": [g[2] for g in grid],
        "object_label": np.arange(1, len(grid) + 1),
    })

    merged = interperate_vision_model({
        "src": str(src),
        "scores": str(scores_csv),
        "tables": ["cell"],
        "score_column": "cv_predictions",
        "feature_importance": True,
        "permutation_importance": False,
        "shap": True,
        "shap_sample": False,
        "top_features": 3,
        "n_jobs": 1,
        "save": False,
    })

    _seen.append(bool(
        len(merged) == len(grid)))

    radars = _titled(gallery, "SHAP Importance by")
    _seen.append(keep(len(radars)))
    label_sets = [_names_in(folder) for folder in radars]
    compartments = next(s for s in label_sets if "nucleus" in s)
    _seen.append(keep(compartments))
    channels = next(s for s in label_sets if "morphology" in s)
    _seen.append(keep(channels))
    _seen.append(bool(
        not _polar_axes()))
    return _seen


def case_interperate_vision_model_shap_sample_on_a_small_plate(tmp_path, monkeypatch):
    """shap_sample must degrade gracefully when there are <100 objects."""
    _seen = []
    from spacr.ml import interperate_vision_model

    rng = np.random.default_rng(19)
    df, grid, labels = _fake_merged_frame(24, rng)
    _install_fake_merge(monkeypatch, df, {})

    src = tmp_path / "plateH"
    src.mkdir()
    scores_csv = tmp_path / "small_plate_scores.csv"
    _write_scores_csv(scores_csv, grid, labels, {
        "rowID": [g[1] for g in grid],
        "columnID": [g[2] for g in grid],
        "object_label": np.arange(1, len(grid) + 1),
    })

    merged = interperate_vision_model({
        "src": str(src),
        "scores": str(scores_csv),
        "tables": ["cell"],
        "score_column": "cv_predictions",
        "feature_importance": True,
        "permutation_importance": False,
        "shap": True,
        "shap_sample": True,
        "top_features": 3,
        "n_jobs": 1,
        "save": False,
    })

    _seen.append(bool(
        isinstance(merged, pd.DataFrame)))
    _seen.append(bool(
        len(merged) == len(grid)))
    return _seen


def case_interperate_vision_model_none_settings_uses_defaults(tmp_path, monkeypatch):
    """Passing None falls back to set_interperate_vision_model_defaults; the
    placeholder 'path' src is snapshotted before the run blows up on it."""
    _seen = []
    from spacr.ml import interperate_vision_model

    rng = np.random.default_rng(14)
    df, _g, _l = _fake_merged_frame(8, rng)
    _install_fake_merge(monkeypatch, df, {})
    monkeypatch.chdir(tmp_path)

    # ``scores`` defaults to the placeholder ``path``.  The tabular reader
    # rejects that suffix before attempting I/O, but the settings snapshot is
    # still written first.
    from spacr.tabular import TabularFormatError

    with pytest.raises(TabularFormatError) as _exc1:
        interperate_vision_model(None)
    _seen.append(('TabularFormatError' if isinstance(_exc1.value, TabularFormatError) else type(_exc1.value).__name__))

    saved = tmp_path / "path" / "settings" / "interperate_vision_model.csv"
    _seen.append(bool(
        saved.is_file()))
    snap_df = pd.read_csv(saved)
    defaults = dict(zip(snap_df["Key"], snap_df["Value"].astype(str)))
    _seen.append(keep(defaults["src"]))
    _seen.append(keep(defaults["score_column"]))
    _seen.append(keep(defaults["top_features"]))
    return _seen


def case_interperate_vision_model_save_writes_importance_tables(tmp_path, monkeypatch):
    """With save=True both importance tables are written under ``<src>/results``."""
    _seen = []
    from spacr.ml import interperate_vision_model

    rng = np.random.default_rng(15)
    df, grid, labels = _fake_merged_frame(24, rng)
    _install_fake_merge(monkeypatch, df, {})

    src = tmp_path / "plateD"
    src.mkdir()
    scores_csv = tmp_path / "save_scores.csv"
    _write_scores_csv(scores_csv, grid, labels, {
        "rowID": [g[1] for g in grid],
        "columnID": [g[2] for g in grid],
        "object_label": np.arange(1, len(grid) + 1),
    })

    interperate_vision_model({
        "src": str(src),
        "scores": str(scores_csv),
        "tables": ["cell"],
        "score_column": "cv_predictions",
        "feature_importance": True,
        "permutation_importance": True,
        "shap": False,
        "top_features": 3,
        "n_jobs": 1,
        "save": True,
    })

    results_dir = src / "results"
    written = {p.name: pd.read_csv(p) for p in results_dir.glob("*.csv")}
    _seen.append(keep(set(written)))
    for frame in written.values():
        assert list(frame.columns) == ["feature", "importance"]
        assert len(frame) == 3            # the full table, not the top-N slice
        assert frame["importance"].is_monotonic_decreasing
    _seen.append(keep(set(written["feature_importance.csv"]["feature"])))
    return _seen


def case_interperate_vision_model_save_true_writes_a_real_csv(tmp_path, monkeypatch):
    """settings['save']=True must persist the importance table to disk."""
    _seen = []
    from spacr.ml import interperate_vision_model

    rng = np.random.default_rng(16)
    df, grid, labels = _fake_merged_frame(24, rng)
    _install_fake_merge(monkeypatch, df, {})
    monkeypatch.chdir(tmp_path)

    src = tmp_path / "plateE"
    src.mkdir()
    scores_csv = tmp_path / "save_real_scores.csv"
    _write_scores_csv(scores_csv, grid, labels, {
        "rowID": [g[1] for g in grid],
        "columnID": [g[2] for g in grid],
        "object_label": np.arange(1, len(grid) + 1),
    })

    interperate_vision_model({
        "src": str(src),
        "scores": str(scores_csv),
        "tables": ["cell"],
        "score_column": "cv_predictions",
        "feature_importance": True,
        "permutation_importance": False,
        "shap": False,
        "top_features": 3,
        "n_jobs": 1,
        "save": True,
    })

    hits = list(tmp_path.rglob("feature_importance.csv"))
    _seen.append(bool(
        hits))
    return _seen


def case_interperate_vision_model_permutation_without_feature_importance(
        tmp_path, monkeypatch):
    """permutation_importance must be usable on its own."""
    _seen = []
    from spacr.ml import interperate_vision_model

    rng = np.random.default_rng(17)
    df, grid, labels = _fake_merged_frame(24, rng)
    _install_fake_merge(monkeypatch, df, {})

    src = tmp_path / "plateF"
    src.mkdir()
    scores_csv = tmp_path / "perm_scores.csv"
    _write_scores_csv(scores_csv, grid, labels, {
        "rowID": [g[1] for g in grid],
        "columnID": [g[2] for g in grid],
        "object_label": np.arange(1, len(grid) + 1),
    })

    merged = interperate_vision_model({
        "src": str(src),
        "scores": str(scores_csv),
        "tables": ["cell"],
        "score_column": "cv_predictions",
        "feature_importance": False,
        "permutation_importance": True,
        "shap": False,
        "top_features": 3,
        "n_jobs": 1,
        "save": False,
    })

    _seen.append(bool(
        isinstance(merged, pd.DataFrame)))
    _seen.append(bool(
        len(merged) == len(grid)))
    return _seen


def case_interperate_vision_model_shap_without_feature_importance(
        tmp_path, monkeypatch):
    """shap must be usable without also asking for RF feature importance."""
    _seen = []
    from spacr.ml import interperate_vision_model

    rng = np.random.default_rng(18)
    df, grid, labels = _fake_merged_frame(12, rng)
    _install_fake_merge(monkeypatch, df, {})

    src = tmp_path / "plateG"
    src.mkdir()
    scores_csv = tmp_path / "shap_only_scores.csv"
    _write_scores_csv(scores_csv, grid, labels, {
        "rowID": [g[1] for g in grid],
        "columnID": [g[2] for g in grid],
        "object_label": np.arange(1, len(grid) + 1),
    })

    merged = interperate_vision_model({
        "src": str(src),
        "scores": str(scores_csv),
        "tables": ["cell"],
        "score_column": "cv_predictions",
        "feature_importance": False,
        "permutation_importance": False,
        "shap": True,
        "shap_sample": False,
        "top_features": 3,
        "n_jobs": 1,
        "save": False,
    })

    _seen.append(bool(
        isinstance(merged, pd.DataFrame)))
    _seen.append(bool(
        len(merged) == len(grid)))
    return _seen


# ---------------------------------------------------------------------------
# End-to-end: real measurements.db through io._read_and_merge_data
# ---------------------------------------------------------------------------

def case_interperate_vision_model_end_to_end_real_db(tmp_path, capsys, gallery):
    """Feature importance + permutation importance + SHAP all run against a
    real sqlite measurements DB, and the SHAP contributions are folded into
    compartment/channel radar plots covering every channel token."""
    _seen = []
    from spacr.ml import interperate_vision_model

    rng = np.random.default_rng(3)
    src = tmp_path / "plate1"
    src.mkdir()
    _db, signal = _build_measurements_db(src, rng)
    grid = _grid()
    labels = (signal > 0).astype(int)
    scores_csv = tmp_path / "scores.csv"
    _write_scores_csv(scores_csv, grid, labels,
                      {"rowID": [g[1] for g in grid],
                       "columnID": [g[2] for g in grid],
                       "object_label": np.arange(1, len(grid) + 1)})

    merged = interperate_vision_model({
        "src": str(src),
        "scores": str(scores_csv),
        "tables": ["cell", "cytoplasm"],
        "score_column": "cv_predictions",
        "feature_importance": True,
        "permutation_importance": True,
        "shap": True,
        "shap_sample": True,
        "top_features": 5,
        "n_jobs": 1,
        "save": False,
    })

    # ---- merged frame -----------------------------------------------------
    _seen.append(bool(
        isinstance(merged, pd.DataFrame)))
    _seen.append(bool(
        len(merged) == len(grid)))
    _seen.append(('cv_predictions' if 'cv_predictions' in (merged.columns) else ("missing", 'cv_predictions')))
    _seen.append(bool(
        set(merged["cv_predictions"].unique()) <= {0, 1}))
    for col in ("cell_channel_0_mean_intensity", "cell_channel_1_mean_intensity",
                "cytoplasm_channel_2_mean_intensity",
                "cytoplasm_channel_3_mean_intensity", "cells_per_well"):
        assert col in merged.columns
    # the 'o' prefix was stripped before the join
    _seen.append(bool(
        not merged["object_label"].str.startswith("o").any()))

    # ---- settings snapshot ------------------------------------------------
    saved = src / "settings" / "interperate_vision_model.csv"
    _seen.append(bool(
        saved.is_file()))
    saved_df = pd.read_csv(saved)
    snap = dict(zip(saved_df["Key"], saved_df["Value"].astype(str)))
    _seen.append(keep(snap["score_column"]))

    # ---- all three explainers announced themselves ------------------------
    printed = capsys.readouterr().out
    _seen.append(('Feature Importance ...' if 'Feature Importance ...' in (printed) else ("missing", 'Feature Importance ...')))
    _seen.append(('Permutation Importance ...' if 'Permutation Importance ...' in (printed) else ("missing", 'Permutation Importance ...')))
    _seen.append(('SHAP Analysis ...' if 'SHAP Analysis ...' in (printed) else ("missing", 'SHAP Analysis ...')))

    # ---- radar plots ------------------------------------------------------
    radars = _titled(gallery, "SHAP Importance by")
    _seen.append(keep(len(radars)))
    label_sets = [_names_in(folder) for folder in radars]

    compartment_labels = next(s for s in label_sets if "cell" in s)
    # 'cells_per_well' must have been folded into the 'cell' compartment.
    _seen.append(keep(compartment_labels))

    channel_labels = next(s for s in label_sets if "morphology" in s)
    _seen.append(bool(
        {"channel_0", "channel_1", "channel_2", "channel_3",
            "morphology"} <= channel_labels))
    # 5 individual channels + every unordered pair of them
    _seen.append(bool(
        len(channel_labels) == 5 + (5 * 4) // 2))
    _seen.append(('channel_0 + channel_1' if 'channel_0 + channel_1' in (channel_labels) else ("missing", 'channel_0 + channel_1')))

    # ONE VALUE PER LABEL in the data beside the picture. The old assertion
    # counted vertices on the matplotlib line and allowed for the closing
    # one; a pyqtgraph radar closes its own polygon inside `add_radar`, so
    # what is worth asserting is that the recorded data has no phantom
    # extra row.
    for folder in radars:
        frame = pd.read_csv(os.path.join(folder, "data.csv"))
        assert len(frame) == len(set(frame["name"])), folder
    _seen.append(bool(
        not _polar_axes()))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_calculate_similarity_swallows_a_failing_metric, [True, True, True, True, True], id="calculate_similarity_swallows_a_failing_metric"),
    pytest.param(case_interperate_vision_model_maps_legacy_score_columns, [True, True, True, True, True, ['cell'], True, True, 'Top 3 Features', 3, True], id="interperate_vision_model_maps_legacy_score_columns"),
    pytest.param(case_interperate_vision_model_keeps_canonical_score_columns, [True, True, True, True, True], id="interperate_vision_model_keeps_canonical_score_columns"),
    pytest.param(case_interperate_vision_model_shap_without_subsampling, [True, 2, {'cell', 'nucleus', 'cell + nucleus'}, {'channel_0', 'channel_1', 'morphology', 'channel_0 + channel_1', 'channel_0 + morphology', 'channel_1 + morphology'}, True], id="interperate_vision_model_shap_without_subsampling"),
    pytest.param(case_interperate_vision_model_shap_sample_on_a_small_plate, [True, True], id="interperate_vision_model_shap_sample_on_a_small_plate"),
    pytest.param(case_interperate_vision_model_none_settings_uses_defaults, ['TabularFormatError', True, 'path', 'cv_predictions', '30'], id="interperate_vision_model_none_settings_uses_defaults"),
    pytest.param(case_interperate_vision_model_save_writes_importance_tables, [{'feature_importance.csv', 'permutation_importance.csv'}, {'cell_channel_0_mean_intensity', 'nucleus_channel_1_mean_intensity', 'cell_area'}], id="interperate_vision_model_save_writes_importance_tables"),
    pytest.param(case_interperate_vision_model_save_true_writes_a_real_csv, [True], id="interperate_vision_model_save_true_writes_a_real_csv"),
    pytest.param(case_interperate_vision_model_permutation_without_feature_importance, [True, True], id="interperate_vision_model_permutation_without_feature_importance"),
    pytest.param(case_interperate_vision_model_shap_without_feature_importance, [True, True], id="interperate_vision_model_shap_without_feature_importance"),
    pytest.param(case_interperate_vision_model_end_to_end_real_db, [True, True, 'cv_predictions', True, True, True, 'cv_predictions', 'Feature Importance ...', 'Permutation Importance ...', 'SHAP Analysis ...', 2, {'cell', 'cytoplasm', 'cell + cytoplasm'}, True, True, 'channel_0 + channel_1', True], id="interperate_vision_model_end_to_end_real_db"),
    pytest.param(case_shap_analysis_returns_closed_summary_figure, [True, True, []], id="shap_analysis_returns_closed_summary_figure"),
    pytest.param(case_shap_analysis_tree_model_labels_every_feature, [True], id="shap_analysis_tree_model_labels_every_feature"),
    pytest.param(case_shap_analysis_tree_classifier_selects_the_positive_class, [True], id="shap_analysis_tree_classifier_selects_the_positive_class"),
    pytest.param(case_find_optimal_threshold_picks_the_f1_maximising_cut, [True, True], id="find_optimal_threshold_picks_the_f1_maximising_cut"),
    pytest.param(case_find_optimal_threshold_ignores_undefined_f1_points, [True], id="find_optimal_threshold_ignores_undefined_f1_points"),
    pytest.param(case_calculate_similarity_regularises_a_singular_covariance, ['LinAlgError'], id="calculate_similarity_regularises_a_singular_covariance"),
    pytest.param(case_calculate_similarity_accepts_list_valued_controls, [True, True], id="calculate_similarity_accepts_list_valued_controls"),
    pytest.param(case_calculate_similarity_reports_and_returns_on_assignment_failure, ['Error calculating similarity scores', 'synthetic assignment failure', ['similarity_to_pos_euclidean', 'similarity_to_neg_euclidean']], id="calculate_similarity_reports_and_returns_on_assignment_failure"),
])
def test_ml_shap_vision_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected

