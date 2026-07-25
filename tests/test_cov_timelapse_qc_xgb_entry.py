"""Branch coverage for the tail of ``spacr.timelapse``.

Covers the two symbols at the end of the module:

* ``_infection_qc_xgboost`` -- the supervised infection-relabelling QC step
  (feature selection, per-well training-set curation, correlation pruning,
  relabel/remove modes, ambiguous-band dropping and the three QC payloads).
* ``automated_motility_assay`` -- the public entry point that stitches the
  merged-npy reader, the QC dispatcher, the velocity summariser, the SQLite
  writer and the panel makers together.

Everything here is CPU-only and offline. The heavy leaf helpers that live
*outside* this region (the per-group regionprops worker and the giant panel
plotter) are replaced by recorders so the orchestration logic itself is what
gets exercised; every test asserts on real outputs (returned frames, emitted
files, DB tables, payload dicts or raised exception types).
"""
from __future__ import annotations

import builtins
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

# Imported once at collection time: spacr.timelapse pulls in torch/cellpose and
# is slow to load, and one test below arms an import hook that would otherwise
# make the very first lazy import of the module explode.
import spacr.timelapse  # noqa: E402,F401


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    import matplotlib.pyplot as plt
    yield
    plt.close("all")


PATHOGEN_CHAN = 1
KEY_COLS = ["plateID", "wellID", "fieldID", "cellID"]


def _build_all_df(cell_specs, n_frames=3, seed=0, pathogen_chan=PATHOGEN_CHAN):
    """Build a frame-level measurement table from explicit per-cell specs.

    ``cell_specs`` is a list of ``(wellID, infected, intensity)`` tuples; one
    tracked object is emitted per spec, repeated over ``n_frames`` frames.

    The pathogen-channel p95 intensity is exactly the requested value on every
    frame so that the per-cell median used by the QC step is exact and the
    quartile thresholds are predictable.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cid, (well, infected, intensity) in enumerate(cell_specs, start=1):
        area0 = float(rng.uniform(200.0, 900.0)) + (300.0 if infected else 0.0)
        solidity0 = float(rng.uniform(0.70, 0.99))
        y0 = float(rng.uniform(10.0, 200.0))
        x0 = float(rng.uniform(10.0, 200.0))
        for f in range(n_frames):
            rows.append(
                {
                    "plateID": "plate1",
                    "wellID": well,
                    "fieldID": "1",
                    "cellID": cid,
                    "frame": f,
                    "infected": bool(infected),
                    "n_pathogens": 3 if infected else 0,
                    f"cell_p95_intensity_ch{pathogen_chan}": float(intensity),
                    f"cell_mean_intensity_ch{pathogen_chan}": float(intensity) * 0.6
                    + float(rng.normal(0, 2.0)),
                    "cell_mean_intensity_ch0": float(rng.uniform(100.0, 200.0)),
                    "cell_area": area0 + float(rng.normal(0, 5.0)),
                    "cell_perimeter": 0.4 * area0 + float(rng.normal(0, 3.0)),
                    "cell_solidity": solidity0 + float(rng.normal(0, 0.005)),
                    "cell_centroid-0": y0 + 1.5 * f,
                    "cell_centroid-1": x0 + 1.0 * f,
                    "nucleus_area": float(rng.uniform(50.0, 200.0)),
                }
            )
    return pd.DataFrame(rows)


def _separable_specs(n_per_class=18, wells=("A01", "A02"), seed=3):
    """Two wells with a clean infected/uninfected intensity separation."""
    rng = np.random.default_rng(seed)
    specs = []
    for well in wells:
        for _ in range(n_per_class):
            specs.append((well, True, float(rng.normal(1000.0, 120.0))))
            specs.append((well, False, float(rng.normal(300.0, 120.0))))
    return specs


def _settings(**over):
    base = {
        "tracked_object": "cell",
        "infection_xgb_n_estimators": 15,
        "infection_xgb_max_depth": 2,
        "infection_xgb_n_jobs": 1,
        "infection_intensity_mode": "relabel",
    }
    base.update(over)
    return base


def _run_xgb(all_df, settings, motility_dir, infection_col="infected",
             pathogen_chan=PATHOGEN_CHAN):
    from spacr.timelapse import _infection_qc_xgboost
    return _infection_qc_xgboost(
        all_df=all_df,
        settings=settings,
        infection_col=infection_col,
        pathogen_chan=pathogen_chan,
        motility_dir=str(motility_dir),
    )


# ===========================================================================
# _infection_qc_xgboost -- early fallbacks
# ===========================================================================

def test_xgboost_import_failure_falls_back_to_histogram(tmp_path, monkeypatch, capsys):
    """ImportError on ``import xgboost`` routes to the histogram QC."""
    all_df = _build_all_df(_separable_specs())
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "xgboost" or name.startswith("xgboost."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    settings = _settings()
    out_df, col = _run_xgb(all_df, settings, tmp_path)
    monkeypatch.undo()

    txt = capsys.readouterr().out
    assert "XGBoost not installed" in txt
    # histogram QC relabels and leaves a PNG behind
    assert col == "adjusted_infected"
    assert settings["infection_intensity_qc_panel_type"] == "histogram"
    assert os.path.isfile(settings["infection_intensity_qc_panel_path"])
    assert "infection_prob" not in out_df.columns


def test_pathogen_chan_none_falls_back_without_relabelling(tmp_path, capsys):
    all_df = _build_all_df(_separable_specs())
    settings = _settings()
    out_df, col = _run_xgb(all_df, settings, tmp_path, pathogen_chan=None)

    txt = capsys.readouterr().out
    assert "pathogen_chan is None" in txt
    # histogram QC cannot find cell_*_ch None -> labels untouched
    assert col == "infected"
    assert "adjusted_infected" not in out_df.columns
    assert settings["infection_intensity_qc_panel_path"] is None
    assert settings["infection_hist_data"] is None


def test_stale_adjusted_and_prob_columns_are_dropped(tmp_path):
    """Columns left over from a previous QC run must be recomputed, not merged."""
    all_df = _build_all_df(_separable_specs())
    all_df["adjusted_infected"] = 7
    all_df["adjusted_infected_x"] = 8
    all_df["infection_prob"] = 0.123
    all_df["infection_prob_y"] = 0.456

    out_df, col = _run_xgb(all_df, _settings(), tmp_path)

    assert col == "adjusted_infected"
    assert "adjusted_infected_x" not in out_df.columns
    assert "infection_prob_y" not in out_df.columns
    # freshly computed, not the stale sentinel values
    assert set(out_df["adjusted_infected"].unique()) <= {0, 1}
    assert not np.allclose(out_df["infection_prob"].to_numpy(), 0.123)


def test_infection_column_recovered_from_infect_like_column(tmp_path, capsys):
    all_df = _build_all_df(_separable_specs())
    all_df = all_df.rename(columns={"infected": "is_infected_flag"})
    settings = _settings()

    out_df, col = _run_xgb(all_df, settings, tmp_path, infection_col="infected")

    txt = capsys.readouterr().out
    assert "not found; using 'is_infected_flag' instead" in txt
    assert col == "adjusted_infected"
    assert out_df["adjusted_infected"].dtype.kind == "i"


def test_infection_column_created_from_n_pathogens(tmp_path, capsys):
    all_df = _build_all_df(_separable_specs()).drop(columns=["infected"])
    settings = _settings()

    out_df, col = _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    assert "_infected_from_n_pathogens" in txt
    assert col == "adjusted_infected"
    # the derived label column was materialised on the frame-level table
    assert "_infected_from_n_pathogens" in out_df.columns
    derived = out_df["_infected_from_n_pathogens"].to_numpy()
    expected = (out_df["n_pathogens"] > 0).astype(int).to_numpy()
    assert np.array_equal(derived, expected)


def test_no_label_and_no_n_pathogens_falls_back(tmp_path, capsys):
    """No infection label at all -> histogram QC, which is itself a no-op."""
    all_df = _build_all_df(_separable_specs()).drop(
        columns=["infected", "n_pathogens"]
    )
    # rename the cell_* pathogen intensity so the histogram helper also bails
    all_df = all_df.rename(
        columns={
            f"cell_p95_intensity_ch{PATHOGEN_CHAN}": (
                f"nucleus_p95_intensity_ch{PATHOGEN_CHAN}"
            ),
            f"cell_mean_intensity_ch{PATHOGEN_CHAN}": (
                f"nucleus_mean_intensity_ch{PATHOGEN_CHAN}"
            ),
        }
    )
    settings = _settings()
    out_df, col = _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    assert "No infection label and no 'n_pathogens'" in txt
    assert col == "infected"
    assert "adjusted_infected" not in out_df.columns
    assert out_df.shape == all_df.shape


@pytest.mark.parametrize("missing", KEY_COLS)
def test_missing_key_column_raises_keyerror(tmp_path, missing):
    all_df = _build_all_df(_separable_specs()).drop(columns=[missing])
    with pytest.raises(KeyError, match=missing):
        _run_xgb(all_df, _settings(), tmp_path)


def test_unknown_tracked_object_falls_back_to_cell(tmp_path, capsys):
    all_df = _build_all_df(_separable_specs())
    settings = _settings(tracked_object="mitochondrion")

    out_df, col = _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    assert "Unknown tracked_object" in txt
    assert col == "adjusted_infected"
    assert settings["infection_xgb_importance"]["tracked_object"] == "cell"
    assert settings["infection_pca_data"]["tracked_object"] == "cell"


def test_no_intensity_column_for_tracked_object_falls_back(tmp_path, capsys):
    """tracked_object='nucleus' but only cell_* intensities exist."""
    all_df = _build_all_df(_separable_specs())
    settings = _settings(tracked_object="nucleus")

    out_df, col = _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    assert "No pathogen-channel intensity column found" in txt
    # histogram QC took over and did relabel using cell_p95_intensity_ch1
    assert col == "adjusted_infected"
    assert settings["infection_intensity_qc_panel_type"] == "histogram"
    assert settings["infection_hist_data"]["intensity_col"] == (
        f"cell_p95_intensity_ch{PATHOGEN_CHAN}"
    )


def test_all_degenerate_features_falls_back(tmp_path, capsys):
    """Every cell_* column constant -> no usable features."""
    specs = [("A01", i % 2 == 0, 500.0) for i in range(40)]
    all_df = _build_all_df(specs)
    for c in list(all_df.columns):
        if c.startswith("cell_"):
            all_df[c] = 1.0

    settings = _settings()
    out_df, col = _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    assert "No usable cell_* feature columns" in txt
    # histogram QC also bails: no intensity variation
    assert col == "infected"
    assert "adjusted_infected" not in out_df.columns


def test_too_few_infected_cells_falls_back(tmp_path, capsys):
    specs = [("A01", i < 5, 900.0 if i < 5 else 200.0) for i in range(40)]
    all_df = _build_all_df(specs)
    settings = _settings()

    out_df, col = _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    assert "Too few infected or uninfected cells overall" in txt
    # histogram QC still runs on 40 cells with 2 distinct intensities
    assert col == "adjusted_infected"
    assert settings["infection_hist_data"] is not None


def test_non_finite_intensities_fall_back(tmp_path, capsys):
    all_df = _build_all_df(_separable_specs())
    all_df[f"cell_p95_intensity_ch{PATHOGEN_CHAN}"] = np.nan
    settings = _settings()

    out_df, col = _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    assert "No finite intensities for infected/uninfected" in txt
    # histogram QC drops every row on dropna -> no relabelling
    assert col == "infected"
    assert "adjusted_infected" not in out_df.columns


# ===========================================================================
# _infection_qc_xgboost -- per-well training curation
# ===========================================================================

def _curation_specs():
    """Deterministic layout exercising balanced sampling + single-class skip.

    infected intensities: 16x1000 (A01), 16x600 (A02), 6x1000 (A03)
        -> 75th percentile = 1000 -> extreme positives = A01(16) + A03(6)
    uninfected intensities: 8x100 (A01), 16x400 (A02), 6x400 (A03)
        -> 25th percentile = 175 -> extreme negatives = A01(8)
    so A01 has both classes (16 vs 8) and A03 is positives-only.
    """
    specs = []
    specs += [("A01", True, 1000.0)] * 16
    specs += [("A01", False, 100.0)] * 8
    specs += [("A02", True, 600.0)] * 16
    specs += [("A02", False, 400.0)] * 16
    specs += [("A03", True, 1000.0)] * 6
    specs += [("A03", False, 400.0)] * 6
    return specs


def test_balanced_per_well_sampling_and_single_class_skip(tmp_path, capsys):
    all_df = _build_all_df(_curation_specs(), seed=11)
    settings = _settings(infection_xgb_min_cells_per_class=5)

    out_df, col = _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    assert "Extreme-intensity candidates: infected=22, uninfected=8" in txt
    # A01: 16 pos vs 8 neg -> balanced down to 8 per class
    assert "wells used=1, positives=8, negatives=8" in txt
    assert "Wells skipped due to single class in extreme set: plate1_A03" in txt
    assert col == "adjusted_infected"


def _curation_specs_negative_heavy():
    """Mirror image of ``_curation_specs``: the extreme set is negative-heavy.

    infected intensities: 8x1000 (A01), 16x600 (A02)
        -> 75th percentile = 1000 -> extreme positives = A01(8)
    uninfected intensities: 16x100 (A01), 16x400 (A02)
        -> 25th percentile = 100  -> extreme negatives = A01(16)
    """
    specs = []
    specs += [("A01", True, 1000.0)] * 8
    specs += [("A01", False, 100.0)] * 16
    specs += [("A02", True, 600.0)] * 16
    specs += [("A02", False, 400.0)] * 16
    return specs


def test_negative_heavy_well_downsamples_negatives(tmp_path, capsys):
    all_df = _build_all_df(_curation_specs_negative_heavy(), seed=24)
    settings = _settings(infection_xgb_min_cells_per_class=5)

    out_df, col = _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    assert "Extreme-intensity candidates: infected=8, uninfected=16" in txt
    # positives kept whole, negatives sampled down to match
    assert "wells used=1, positives=8, negatives=8" in txt
    assert col == "adjusted_infected"


def test_small_well_keeps_all_extremes(tmp_path, capsys):
    all_df = _build_all_df(_curation_specs(), seed=12)
    settings = _settings(infection_xgb_min_cells_per_class=1000)

    _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    # min_per_class not reached -> the "keep everything" branch
    assert "wells used=1, positives=16, negatives=8" in txt


def test_bad_random_state_falls_back_to_default_seed(tmp_path):
    all_df = _build_all_df(_curation_specs(), seed=13)
    settings = _settings(
        infection_xgb_min_cells_per_class=5,
        infection_xgb_random_state="not-an-int",
    )
    out_df, col = _run_xgb(all_df, settings, tmp_path)
    assert col == "adjusted_infected"
    assert out_df["adjusted_infected"].notna().all()


def test_no_well_with_both_classes_falls_back(tmp_path, capsys):
    """Every well is single-class in the extreme set -> skip XGBoost QC."""
    specs = []
    # well A01: only infected; well A02: only uninfected
    specs += [("A01", True, 1000.0 + i) for i in range(20)]
    specs += [("A01", False, 500.0 + i) for i in range(20)]
    specs += [("A02", True, 700.0 + i) for i in range(20)]
    specs += [("A02", False, 100.0 + i) for i in range(20)]
    all_df = _build_all_df(specs, seed=14)
    settings = _settings()

    out_df, col = _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    assert "No wells with both infected and uninfected" in txt
    assert col == "adjusted_infected"          # histogram QC took over
    assert settings["infection_intensity_qc_panel_type"] == "histogram"


# ===========================================================================
# _infection_qc_xgboost -- feature matrix handling
# ===========================================================================

def test_correlated_feature_is_pruned(tmp_path, capsys):
    all_df = _build_all_df(_separable_specs(), seed=5)
    all_df["cell_bbox_area"] = all_df["cell_area"] * 2.0 + 1.0  # r == 1.0

    settings = _settings()
    _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    assert "Removing highly correlated features" in txt
    used = settings["infection_xgb_importance"]["feature_names"]
    assert "cell_area" in used
    assert "cell_bbox_area" not in used


def test_feature_selection_excludes_other_channels_and_centroids(tmp_path):
    all_df = _build_all_df(_separable_specs(), seed=6)
    settings = _settings(infection_xgb_corr_threshold=1.5)  # disable pruning

    _run_xgb(all_df, settings, tmp_path)

    used = settings["infection_xgb_importance"]["feature_names"]
    assert used, "expected a non-empty feature list"
    assert all(f.startswith("cell_") for f in used)
    assert not any("centroid" in f for f in used)
    assert "cell_mean_intensity_ch0" not in used
    assert "nucleus_area" not in used
    assert f"cell_p95_intensity_ch{PATHOGEN_CHAN}" in used


def test_all_nonfinite_feature_column_is_zero_imputed(tmp_path):
    """A ±inf column survives the degeneracy filter and is zeroed out."""
    all_df = _build_all_df(_separable_specs(), seed=7)
    all_df["cell_solidity"] = np.where(
        np.arange(len(all_df)) % 2 == 0, np.inf, -np.inf
    )
    settings = _settings(infection_xgb_corr_threshold=1.5)

    out_df, col = _run_xgb(all_df, settings, tmp_path)

    assert col == "adjusted_infected"
    used = settings["infection_xgb_importance"]["feature_names"]
    assert "cell_solidity" in used
    # a constant (zeroed) feature can carry no gain
    imp = dict(
        zip(used, settings["infection_xgb_importance"]["feature_importances"])
    )
    assert imp["cell_solidity"] == 0.0
    # PCA still produced finite coordinates despite the inf column
    coords = settings["infection_pca_data"]["coords"]
    assert np.isfinite(coords).all()


# ===========================================================================
# _infection_qc_xgboost -- relabel / remove / ambiguous band
# ===========================================================================

def test_relabel_mode_payloads_and_columns(tmp_path, capsys):
    all_df = _build_all_df(_separable_specs(), seed=8)
    n_rows_before = len(all_df)
    settings = _settings(infection_xgb_drop_ambiguous=False)

    out_df, col = _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    assert "Relabel mode" in txt
    assert col == "adjusted_infected"
    # nothing dropped when the ambiguous band is disabled
    assert len(out_df) == n_rows_before
    assert out_df["adjusted_infected"].dtype.kind == "i"
    assert out_df["infection_prob"].between(0.0, 1.0).all()

    hist = settings["infection_hist_data"]
    assert hist["intensity_col"] == f"cell_p95_intensity_ch{PATHOGEN_CHAN}"
    assert hist["pathogen_chan"] == PATHOGEN_CHAN
    assert hist["log_transform"] is False
    assert hist["bin_edges"].size >= 11
    assert hist["intensities_inf"].size + hist["intensities_uninf"].size == 72

    pca = settings["infection_pca_data"]
    assert pca["method_label"] == "PCA"
    assert pca["coords"].shape == (72, 2)
    assert pca["labels"].shape == (72,)

    imp = settings["infection_xgb_importance"]
    assert len(imp["feature_names"]) == len(imp["feature_importances"])
    assert all(v >= 0.0 for v in imp["feature_importances"])
    # sorted descending by gain
    assert imp["feature_importances"] == sorted(
        imp["feature_importances"], reverse=True
    )

    # no PNG for the xgboost strategy
    assert settings["infection_intensity_qc_panel_type"] == "xgboost"
    assert settings["infection_intensity_qc_panel_path"] is None


def test_relabel_mode_with_margin_leaves_uncertain_labels_untouched(tmp_path):
    all_df = _build_all_df(_separable_specs(), seed=9)
    settings = _settings(
        infection_xgb_drop_ambiguous=False,
        infection_xgb_margin=0.49,
    )
    out_df, col = _run_xgb(all_df, settings, tmp_path)

    probs = out_df["infection_prob"].to_numpy()
    adj = out_df["adjusted_infected"].to_numpy()
    # with a 0.49 margin only near-certain probabilities may flip
    mid = (probs > 0.01) & (probs < 0.99)
    orig = out_df["infected"].astype(int).to_numpy()
    assert np.array_equal(adj[mid], orig[mid])
    assert np.array_equal(adj[probs >= 0.99], np.ones(int((probs >= 0.99).sum()), int))


def test_remove_mode_drops_disagreements(tmp_path, capsys):
    """A block of mislabelled cells is removed rather than relabelled."""
    specs = _separable_specs(seed=10)
    # flip labels of 6 high-intensity cells: the model will disagree
    specs = [
        (w, (not inf) if (i % 6 == 0) else inf, val)
        for i, (w, inf, val) in enumerate(specs)
    ]
    all_df = _build_all_df(specs, seed=10)
    settings = _settings(
        infection_intensity_mode="remove",
        infection_xgb_margin=0.1,
        infection_xgb_drop_ambiguous=False,
    )

    out_df, col = _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    assert "Remove mode: removed" in txt
    assert col == "adjusted_infected"
    n_cells_out = out_df[KEY_COLS].drop_duplicates().shape[0]
    assert n_cells_out < 72
    assert len(out_df) == n_cells_out * 3      # whole tracks removed
    assert out_df["infection_prob"].notna().all()


def test_remove_mode_zero_margin_has_no_ambiguous_band(tmp_path, capsys):
    specs = _separable_specs(seed=15)
    specs = [
        (w, (not inf) if (i % 7 == 0) else inf, val)
        for i, (w, inf, val) in enumerate(specs)
    ]
    all_df = _build_all_df(specs, seed=15)
    settings = _settings(
        infection_intensity_mode="remove",
        infection_xgb_margin=0.0,
        infection_xgb_drop_ambiguous=False,
    )
    out_df, col = _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    assert "margin=0.00" in txt
    # with margin 0 every disagreement is a removal, so what survives agrees
    assert np.array_equal(
        out_df["adjusted_infected"].to_numpy(),
        (out_df["infection_prob"].to_numpy() >= 0.5).astype(int),
    )


def test_unknown_mode_is_treated_as_relabel(tmp_path, capsys):
    all_df = _build_all_df(_separable_specs(), seed=16)
    n_before = len(all_df)
    settings = _settings(
        infection_intensity_mode="nonsense",
        infection_xgb_drop_ambiguous=False,
    )
    out_df, _ = _run_xgb(all_df, settings, tmp_path)
    assert "Relabel mode" in capsys.readouterr().out
    assert len(out_df) == n_before


def test_ambiguous_band_drops_cells_and_swaps_reversed_bounds(tmp_path, capsys):
    all_df = _build_all_df(_separable_specs(), seed=17)
    settings = _settings(
        infection_xgb_drop_ambiguous=True,
        # reversed + out of range on purpose: must clip then swap to [0, 1]
        infection_xgb_ambiguous_low=1.4,
        infection_xgb_ambiguous_high=-0.2,
    )
    out_df, col = _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    assert "ambiguous XGBoost probability in [0.00, 1.00]" in txt
    # the whole [0, 1] band is ambiguous -> every cell dropped
    assert out_df.empty
    assert col == "adjusted_infected"


def test_ambiguous_band_partial_drop(tmp_path, capsys):
    all_df = _build_all_df(_separable_specs(), seed=18)
    n_cells_before = all_df[KEY_COLS].drop_duplicates().shape[0]
    settings = _settings(
        infection_xgb_drop_ambiguous=True,
        infection_xgb_ambiguous_low=0.45,
        infection_xgb_ambiguous_high=0.55,
    )
    out_df, col = _run_xgb(all_df, settings, tmp_path)

    n_cells_after = out_df[KEY_COLS].drop_duplicates().shape[0]
    assert n_cells_after <= n_cells_before
    probs = out_df["infection_prob"].to_numpy()
    assert not ((probs >= 0.45) & (probs <= 0.55)).any()


def test_rows_with_nan_keys_keep_their_original_label(tmp_path):
    """Rows whose group key is NaN never reach cell_level and are back-filled."""
    all_df = _build_all_df(_separable_specs(), seed=19)
    all_df["fieldID"] = all_df["fieldID"].astype(object)
    orphan = all_df["cellID"] == 1
    all_df.loc[orphan, "fieldID"] = None
    orig_label = int(all_df.loc[orphan, "infected"].iloc[0])

    settings = _settings(infection_xgb_drop_ambiguous=False)
    out_df, col = _run_xgb(all_df, settings, tmp_path)

    kept = out_df[out_df["fieldID"].isna()]
    assert len(kept) == 3
    assert (kept["adjusted_infected"] == orig_label).all()
    assert kept["infection_prob"].isna().all()
    assert out_df["adjusted_infected"].dtype.kind == "i"


# ===========================================================================
# _infection_qc_xgboost -- payload failure paths
# ===========================================================================

def test_pca_payload_failure_is_swallowed(tmp_path, monkeypatch, capsys):
    all_df = _build_all_df(_separable_specs(), seed=20)
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("sklearn blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    settings = _settings(infection_xgb_drop_ambiguous=False)
    out_df, col = _run_xgb(all_df, settings, tmp_path)
    monkeypatch.undo()

    txt = capsys.readouterr().out
    assert "Could not compute histogram/PCA payloads" in txt
    assert settings["infection_pca_data"] is None
    # the histogram payload was built before the failure point
    assert settings["infection_hist_data"] is not None
    # ... and the QC result itself is unaffected
    assert col == "adjusted_infected"
    assert settings["infection_xgb_importance"] is not None


def test_feature_importance_failure_is_swallowed(tmp_path, capsys):
    all_df = _build_all_df(_separable_specs(), seed=21)
    settings = _settings(
        infection_xgb_drop_ambiguous=False,
        infection_xgb_top_features="twenty",   # int() raises inside the try
    )
    out_df, col = _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    assert "Could not compute feature importances" in txt
    assert settings["infection_xgb_importance"] is None
    assert col == "adjusted_infected"
    assert settings["infection_intensity_qc_panel_type"] == "xgboost"


def test_zero_top_features_yields_empty_importance_payload(tmp_path, capsys):
    all_df = _build_all_df(_separable_specs(), seed=22)
    settings = _settings(
        infection_xgb_drop_ambiguous=False,
        infection_xgb_top_features=0,
    )
    _run_xgb(all_df, settings, tmp_path)

    txt = capsys.readouterr().out
    assert "Top XGBoost features (gain)" not in txt
    imp = settings["infection_xgb_importance"]
    assert imp["feature_names"] == []
    assert imp["feature_importances"] == []


def test_histogram_payload_skipped_when_too_few_cells(tmp_path):
    """Fewer than 10 surviving cells -> no histogram payload, PCA still built."""
    all_df = _build_all_df(_separable_specs(), seed=23)
    settings = _settings(
        infection_xgb_drop_ambiguous=True,
        infection_xgb_ambiguous_low=0.0,
        infection_xgb_ambiguous_high=0.999999,
    )
    out_df, col = _run_xgb(all_df, settings, tmp_path)

    assert settings["infection_hist_data"] is None
    assert col == "adjusted_infected"


# ===========================================================================
# automated_motility_assay
# ===========================================================================

TABLE = "timelapse_object_measurements"


def _make_src(tmp_path, n_channels=3, n_files=2, name="assay"):
    src = tmp_path / name
    merged = src / "merged"
    merged.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for i in range(n_files):
        arr = rng.integers(0, 500, size=(n_channels, 12, 12)).astype(np.uint16)
        np.save(merged / f"plate1_A01_1_{i}.npy", arr)
    return src


def _write_db(src, df, table=TABLE):
    mdir = src / "measurements"
    mdir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(mdir / "measurements.db") as con:
        df.to_sql(table, con, if_exists="replace", index=False)
    return mdir / "measurements.db"


def _assay_settings(src, **over):
    base = {
        "src": str(src),
        "channels": [0, 1, 2],
        "cell_channel": 2,
        "nucleus_channel": 0,
        "pathogen_channel": PATHOGEN_CHAN,
        "n_jobs": 1,
        "infection_intensity_qc": True,
        "infection_intensity_strategy": "xgboost",
        "infection_intensity_qc_scope": "combined",
        "infection_xgb_n_estimators": 15,
        "infection_xgb_max_depth": 2,
        "infection_xgb_n_jobs": 1,
        "reuse_existing_measurements": True,
    }
    base.update(over)
    return base


@pytest.fixture
def panel_calls(monkeypatch):
    """Replace the (very heavy) panel plotter with a call recorder."""
    import spacr.timelapse as tl
    calls = []

    def _rec(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(tl, "_make_intensity_motility_panel", _rec)
    return calls


def test_reuse_from_db_runs_full_xgboost_pipeline(tmp_path, panel_calls, capsys):
    from spacr.timelapse import automated_motility_assay

    src = _make_src(tmp_path)
    df = _build_all_df(_separable_specs(n_per_class=15), n_frames=4, seed=30)
    _write_db(src, df)

    settings = _assay_settings(src)
    out = automated_motility_assay(settings)

    txt = capsys.readouterr().out
    assert "Loaded ORIGINAL measurements from DB" in txt
    assert "Reusing ORIGINAL smoothed measurements" in txt

    # adjusted labels exist on the returned (adjusted) frame
    assert "adjusted_infected" in out.columns
    assert "infection_prob" in out.columns

    # adjusted CSV emitted with the strategy in its name
    csv_path = src / "measurements" / f"{TABLE}_adjusted_xgboost.csv"
    assert csv_path.is_file()
    assert "adjusted_infected" in pd.read_csv(csv_path).columns

    # the canonical SQLite table holds the PRE-QC snapshot only
    with sqlite3.connect(src / "measurements" / "measurements.db") as con:
        tables = {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        canonical = pd.read_sql_query(f"SELECT * FROM {TABLE}", con)
        wells = pd.read_sql_query(f"SELECT * FROM {TABLE}_well_motility", con)
    assert f"{TABLE}_well_motility" in tables
    assert "adjusted_infected" not in canonical.columns
    assert set(wells["wellID"]) == {"A01", "A02"}
    assert {"mean_velocity_all", "velocity_unit"}.issubset(wells.columns)
    assert (wells["velocity_unit"] == "µm/min").all()

    # correlation CSV + debug plot PDF
    assert (src / "measurements" / "velocity_feature_correlations.csv").is_file()
    assert list((src / "motility_plots").glob("merged_planes_*.pdf"))

    # both panels were requested, with the strategy encoded in the tag
    assert [c["label_tag"] for c in panel_calls] == [
        "mask_xgboost",
        "adjusted_xgboost",
    ]
    assert panel_calls[0]["infection_col"] == "infected"
    assert panel_calls[1]["infection_col"] == "adjusted_infected"
    assert panel_calls[0]["n_channels"] == 3
    assert panel_calls[0]["vel_unit"] == "µm/min"


def test_missing_proba_column_warns_and_keeps_all_rows(tmp_path, panel_calls, capsys):
    from spacr.timelapse import automated_motility_assay

    src = _make_src(tmp_path)
    df = _build_all_df(_separable_specs(n_per_class=15), n_frames=4, seed=31)
    _write_db(src, df)

    # the default proba column name does not exist on the QC output
    settings = _assay_settings(src, infection_xgb_proba_column="does_not_exist")
    out = automated_motility_assay(settings)

    txt = capsys.readouterr().out
    assert "no XGBoost probability/score column was found" in txt
    assert "_ambiguous_flag" not in out.columns


def test_auto_detected_infection_prob_column_is_used(tmp_path, panel_calls, capsys):
    from spacr.timelapse import automated_motility_assay

    src = _make_src(tmp_path)
    df = _build_all_df(_separable_specs(n_per_class=15), n_frames=4, seed=32)
    _write_db(src, df)

    settings = _assay_settings(src, infection_xgb_proba_column=None)
    out = automated_motility_assay(settings)

    txt = capsys.readouterr().out
    # fell through the "xgb"-named candidates to the "infection"-named ones
    assert "no XGBoost probability/score column was found" not in txt
    # cell-level QC already removed the [0.25, 0.75] band, so no track is left
    # inside the open (0.25, 0.75) interval
    probs = (
        out.groupby(KEY_COLS)["infection_prob"].mean().dropna().to_numpy()
    )
    assert not ((probs > 0.25) & (probs < 0.75)).any()


def test_xgb_named_column_triggers_ambiguous_track_drop(tmp_path, panel_calls, capsys):
    from spacr.timelapse import automated_motility_assay

    src = _make_src(tmp_path)
    df = _build_all_df(_separable_specs(n_per_class=15), n_frames=4, seed=33)
    # a non-cell_* column whose name matches the "xgb"+"prob" auto-detection
    df["nucleus_xgb_prob"] = np.where(df["cellID"] % 3 == 0, 0.5, 0.95)
    _write_db(src, df)

    settings = _assay_settings(src, infection_xgb_proba_column=None)
    out = automated_motility_assay(settings)

    txt = capsys.readouterr().out
    assert "ambiguous XGBoost tracks (0.25 < proba < 0.75)" in txt
    assert "_ambiguous_flag" not in out.columns
    # every surviving track sits outside the ambiguous band
    assert (out["nucleus_xgb_prob"] == 0.95).all()


def test_per_well_qc_scope(tmp_path, panel_calls, capsys):
    from spacr.timelapse import automated_motility_assay

    src = _make_src(tmp_path)
    df = _build_all_df(
        _separable_specs(n_per_class=15, wells=("A01", "A02")),
        n_frames=4,
        seed=34,
    )
    _write_db(src, df)

    settings = _assay_settings(
        src,
        infection_intensity_qc_scope="per_well",
        infection_xgb_drop_ambiguous=False,
    )
    out = automated_motility_assay(settings)

    assert "adjusted_infected" in out.columns
    assert set(out["wellID"]) == {"A01", "A02"}
    assert out["adjusted_infected"].notna().all()


# --- recompute-from-merged paths ------------------------------------------

def _fake_group_df(n_frames=4, seed=40):
    return _build_all_df(_separable_specs(n_per_class=15), n_frames=n_frames,
                         seed=seed)


def test_recompute_from_merged_sequential(tmp_path, panel_calls, monkeypatch, capsys):
    import spacr.timelapse as tl

    src = _make_src(tmp_path, n_files=1)
    seen = []

    def _fake_process(args):
        seen.append(args)
        return _fake_group_df()

    monkeypatch.setattr(tl, "_process_merged_group", _fake_process)

    settings = _assay_settings(src, reuse_existing_measurements=False, n_jobs=1)
    out = tl.automated_motility_assay(settings)

    txt = capsys.readouterr().out
    assert "Using n_jobs=1" in txt
    assert "Combined raw measurements" in txt
    assert "After smoothing" in txt
    # one (plate, well, field) group -> one worker call, with the right args
    assert len(seen) == 1
    assert seen[0][0] == str(src)
    assert seen[0][2] == 3                     # n_channels
    assert seen[0][5] == PATHOGEN_CHAN
    assert "adjusted_infected" in out.columns
    # DB was created from scratch
    with sqlite3.connect(src / "measurements" / "measurements.db") as con:
        stored = pd.read_sql_query(f"SELECT * FROM {TABLE}", con)
    assert len(stored) == 15 * 2 * 2 * 4       # cells x classes x wells x frames


def test_recompute_uses_pool_when_n_jobs_gt_one(tmp_path, panel_calls, monkeypatch):
    import multiprocessing
    import spacr.timelapse as tl

    src = _make_src(tmp_path, n_files=1)
    created = {}

    class _FakePool:
        def __init__(self, processes=None):
            created["processes"] = processes

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, iterable):
            created["n_args"] = len(list(iterable))
            return [fn(a) for a in iterable]

    monkeypatch.setattr(multiprocessing, "Pool", _FakePool)
    monkeypatch.setattr(tl, "_process_merged_group", lambda args: _fake_group_df())

    settings = _assay_settings(src, reuse_existing_measurements=False, n_jobs=3)
    out = tl.automated_motility_assay(settings)

    assert created["processes"] == 3
    assert created["n_args"] == 1
    assert not out.empty


def test_njobs_none_derives_from_cpu_count(tmp_path, panel_calls, monkeypatch, capsys):
    import multiprocessing
    import spacr.timelapse as tl

    src = _make_src(tmp_path, n_files=1)
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 2)
    monkeypatch.setattr(tl, "_process_merged_group", lambda args: _fake_group_df())

    settings = _assay_settings(src, reuse_existing_measurements=False, n_jobs=None)
    out = tl.automated_motility_assay(settings)

    assert "Using n_jobs=1" in capsys.readouterr().out
    assert not out.empty


def test_db_reuse_with_missing_table_recomputes(tmp_path, panel_calls, monkeypatch,
                                                capsys):
    import spacr.timelapse as tl

    src = _make_src(tmp_path, n_files=1)
    _write_db(src, pd.DataFrame({"a": [1]}), table="some_other_table")
    monkeypatch.setattr(tl, "_process_merged_group", lambda args: _fake_group_df())

    settings = _assay_settings(src)
    out = tl.automated_motility_assay(settings)

    txt = capsys.readouterr().out
    assert "Failed to reuse existing measurements" in txt
    assert not out.empty


def test_db_reuse_with_wrong_columns_recomputes(tmp_path, panel_calls, monkeypatch,
                                                capsys):
    import spacr.timelapse as tl

    src = _make_src(tmp_path, n_files=1)
    _write_db(src, pd.DataFrame({"plateID": ["p"], "wellID": ["A01"]}))
    monkeypatch.setattr(tl, "_process_merged_group", lambda args: _fake_group_df())

    settings = _assay_settings(src)
    out = tl.automated_motility_assay(settings)

    txt = capsys.readouterr().out
    assert "empty or missing required columns" in txt
    assert not out.empty


def test_missing_merged_dir_raises(tmp_path):
    from spacr.timelapse import automated_motility_assay

    src = tmp_path / "no_merged"
    src.mkdir()
    with pytest.raises(FileNotFoundError, match="No merged directory"):
        automated_motility_assay(_assay_settings(src))


def test_empty_merged_dir_raises(tmp_path):
    from spacr.timelapse import automated_motility_assay

    src = tmp_path / "empty_merged"
    (src / "merged").mkdir(parents=True)
    (src / "merged" / "README.txt").write_text("not an npy")
    with pytest.raises(FileNotFoundError, match="No .npy files found"):
        automated_motility_assay(_assay_settings(src))


@pytest.mark.parametrize("channels", [[], "not-a-list"])
def test_bad_channels_setting_raises_valueerror(tmp_path, channels):
    from spacr.timelapse import automated_motility_assay

    src = _make_src(tmp_path, n_files=1)
    with pytest.raises(ValueError, match="non-empty list of channels"):
        automated_motility_assay(_assay_settings(src, channels=channels))


def test_infected_derived_from_n_pathogens(tmp_path, panel_calls, capsys):
    from spacr.timelapse import automated_motility_assay

    src = _make_src(tmp_path)
    df = _build_all_df(_separable_specs(n_per_class=15), n_frames=4, seed=41)
    df = df.drop(columns=["infected"])
    _write_db(src, df)

    settings = _assay_settings(src, infection_intensity_qc=False)
    out = automated_motility_assay(settings)

    assert out["infected"].dtype == bool
    expected = out["n_pathogens"] > 0
    assert np.array_equal(out["infected"].to_numpy(), expected.to_numpy())
    assert "Tracks (mask-based)" in capsys.readouterr().out


def test_no_infection_information_defaults_to_false(tmp_path, panel_calls, capsys):
    from spacr.timelapse import automated_motility_assay

    src = _make_src(tmp_path)
    df = _build_all_df(_separable_specs(n_per_class=15), n_frames=4, seed=42)
    df = df.drop(columns=["infected", "n_pathogens"])
    _write_db(src, df)

    settings = _assay_settings(src, infection_intensity_qc=False)
    out = automated_motility_assay(settings)

    assert (~out["infected"]).all()
    assert "infected=0, uninfected=60" in capsys.readouterr().out


def test_qc_disabled_uses_plain_adjusted_csv_name_and_one_panel(
    tmp_path, panel_calls, capsys
):
    from spacr.timelapse import automated_motility_assay

    src = _make_src(tmp_path)
    df = _build_all_df(_separable_specs(n_per_class=15), n_frames=4, seed=43)
    _write_db(src, df)

    settings = _assay_settings(
        src, infection_intensity_qc=False, infection_intensity_strategy="none"
    )
    out = automated_motility_assay(settings)

    assert (src / "measurements" / f"{TABLE}_adjusted.csv").is_file()
    assert "adjusted_infected" not in out.columns
    # infection_col stayed 'infected' -> the adjusted panel is skipped
    assert [c["label_tag"] for c in panel_calls] == ["mask_none"]


def test_panels_can_be_disabled(tmp_path, panel_calls):
    from spacr.timelapse import automated_motility_assay

    src = _make_src(tmp_path)
    df = _build_all_df(_separable_specs(n_per_class=15), n_frames=4, seed=44)
    _write_db(src, df)

    settings = _assay_settings(
        src, make_mask_panel=False, make_adjusted_panel=False
    )
    out = automated_motility_assay(settings)

    assert panel_calls == []
    assert "adjusted_infected" in out.columns


def test_adjusted_csv_failure_is_reported_but_not_fatal(
    tmp_path, panel_calls, monkeypatch, capsys
):
    from spacr.timelapse import automated_motility_assay

    src = _make_src(tmp_path)
    df = _build_all_df(_separable_specs(n_per_class=15), n_frames=4, seed=45)
    _write_db(src, df)

    real_to_csv = pd.DataFrame.to_csv

    def _flaky_to_csv(self, path_or_buf=None, *args, **kwargs):
        name = os.path.basename(path_or_buf) if isinstance(path_or_buf, str) else ""
        if "_adjusted" in name:
            raise OSError("disk on fire")
        return real_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", _flaky_to_csv)

    settings = _assay_settings(src)
    out = automated_motility_assay(settings)

    txt = capsys.readouterr().out
    assert "failed to save adjusted CSV" in txt
    assert "disk on fire" in txt
    # the run continued: velocity correlations and the DB were still written
    assert not out.empty
    assert (src / "measurements" / "velocity_feature_correlations.csv").is_file()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG: when every merged group yields an empty frame, "
        "pd.concat([]) raises ValueError('No objects to concatenate') before "
        "the intended RuntimeError('No measurements were produced ...') can "
        "be raised"
    ),
)
def test_no_measurements_raises_runtimeerror(tmp_path, panel_calls, monkeypatch):
    import spacr.timelapse as tl

    src = _make_src(tmp_path, n_files=1)
    monkeypatch.setattr(
        tl, "_process_merged_group", lambda args: pd.DataFrame()
    )
    settings = _assay_settings(src, reuse_existing_measurements=False)
    with pytest.raises(RuntimeError, match="No measurements were produced"):
        tl.automated_motility_assay(settings)
