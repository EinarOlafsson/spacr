"""CPU coverage for the infection-QC panel helpers in ``spacr.timelapse``.

Covers four private helpers used by ``summarise_tracks_from_merged``:

* ``_compute_intensity_percentiles_per_channel`` -- per-frame / per-object
  intensity percentiles, including all of its early-return guards.
* ``_make_adjusted_qc_panel`` -- the 3-panel PNG (PCA / XGBoost / histogram)
  built purely from payloads stashed in ``settings``.
* ``_load_measurements_from_db`` -- SQLite reader with its failure paths.
* ``_infection_qc_histogram`` -- histogram-based infection relabelling
  (relabel / remove modes, log transform, threshold fallback, graph on/off).

Everything is synthetic, offline and Agg-only.
"""
from __future__ import annotations

import functools
import inspect

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# fixtures / builders
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture(autouse=True)
def _png_figure_preference(monkeypatch):
    """Say which figure format these tests are asserting.

    ``_make_adjusted_qc_panel`` writes through ``spacr.plot.save_figure``,
    which follows the user's figure-format preference and rewrites the file
    extension to match. Under pytest there is no preference store, so the
    preference falls back to ``spacr.plot.DEFAULT_FIGURE_FORMAT`` -- PDF. The
    panel tests below assert an exact ``.png`` filename, so they state the
    preference instead of inheriting the shipped default.

    (``_infection_qc_histogram`` is unaffected either way: it pins ``fmt="png"``
    on purpose, because the panel reads that file back with ``mpimg.imread``.)
    """
    import spacr.plot as P
    monkeypatch.setattr(P, "figure_output_preferences", lambda: ("png", 200))


def _qc_df(n_cells=30, frames=2, chan=2, col="p95", seed=0,
           plate="plate1", well="A01"):
    """Per-frame per-cell table with a clean infected / uninfected split.

    Uninfected cells sit around 100-200 intensity, infected around 5000-9000,
    so the histogram threshold search has an obvious answer.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cells):
        infected = i >= n_cells // 2
        base = 5000.0 + 4000.0 * rng.random() if infected else 100.0 + 80.0 * rng.random()
        for f in range(frames):
            rows.append(
                {
                    "plateID": plate,
                    "wellID": well,
                    "fieldID": 1,
                    "cellID": i + 1,
                    "frame": f,
                    "infected": bool(infected),
                    f"cell_{col}_intensity_ch{chan}": float(base + rng.random()),
                }
            )
    return pd.DataFrame(rows)


def _per_cell_positive(df, col="adjusted_infected"):
    """Number of distinct cells whose label is True."""
    return int(df.groupby("cellID")[col].max().astype(bool).sum())


# ---------------------------------------------------------------------------
# _compute_intensity_percentiles_per_channel
# ---------------------------------------------------------------------------

def _labelled_stack(T=3, H=16, W=16):
    mask = np.zeros((T, H, W), dtype=np.int32)
    for t in range(T):
        mask[t, 2:6, 2:6] = 1
        mask[t, 9:14, 9:14] = 2
    return mask


def _case_percentiles_normal_path_values_and_columns():
    observed = {}
    from spacr.timelapse import _compute_intensity_percentiles_per_channel

    mask = _labelled_stack(T=2)
    inten = np.zeros((2, 16, 16, 3), dtype=float)
    # Channel 1 gets a deterministic ramp so percentiles are predictable.
    inten[:, :, :, 1] = 7.0
    inten[0, 2:6, 2:6, 1] = np.arange(16, dtype=float).reshape(4, 4)
    inten[0, 9:14, 9:14, 1] = 100.0

    out = _compute_intensity_percentiles_per_channel(
        mask, inten, 1, "cell", percentiles=(25, 50, 75)
    )

    observed['list(out.columns)'] = list(out.columns)
    # 2 frames x 2 objects
    observed['len(out)'] = len(out)
    observed["set(out['cell_label'])"] = set(out['cell_label'])
    observed["set(out['frame'])"] = set(out['frame'])

    r = out[(out["frame"] == 0) & (out["cell_label"] == 1)].iloc[0]
    observed["r['cell_p50_intensity_ch1']"] = r['cell_p50_intensity_ch1']
    observed["r['cell_p25_intensity_ch1']"] = r['cell_p25_intensity_ch1']

    r2 = out[(out["frame"] == 0) & (out["cell_label"] == 2)].iloc[0]
    observed["r2['cell_p75_intensity_ch1']"] = r2['cell_p75_intensity_ch1']
    # frame 1 is the flat 7.0 background inside both labels
    r3 = out[out["frame"] == 1].iloc[0]
    observed["r3['cell_p50_intensity_ch1']"] = r3['cell_p50_intensity_ch1']
    observed["out['cell_label'].dtype.kind in 'iu'"] = bool(out['cell_label'].dtype.kind in 'iu')
    return observed


_EXPECT_percentiles_normal_path_values_and_columns = {
    'list(out.columns)': ['frame', 'cell_label', 'cell_p25_intensity_ch1', 'cell_p50_intensity_ch1', 'cell_p75_intensity_ch1'],
    'len(out)': 4,
    "set(out['cell_label'])": {1, 2},
    "set(out['frame'])": {0, 1},
    "r['cell_p50_intensity_ch1']": pytest.approx(7.5),
    "r['cell_p25_intensity_ch1']": pytest.approx(np.percentile(np.arange(16.0), 25)),
    "r2['cell_p75_intensity_ch1']": pytest.approx(100.0),
    "r3['cell_p50_intensity_ch1']": pytest.approx(7.0),
    "out['cell_label'].dtype.kind in 'iu'": True,
}


def _case_percentiles_label_as_track_id_renames_column():
    observed = {}
    from spacr.timelapse import _compute_intensity_percentiles_per_channel

    mask = _labelled_stack(T=1)
    inten = np.full((1, 16, 16, 2), 3.0)

    out = _compute_intensity_percentiles_per_channel(
        mask, inten, 0, "cell", percentiles=(50,), label_as_track_id=True
    )
    observed["'track_id' in out.columns"] = bool('track_id' in out.columns)
    observed["'cell_label' not in out.columns"] = bool('cell_label' not in out.columns)
    observed["out['cell_p50_intensity_ch0'].tolist()"] = out['cell_p50_intensity_ch0'].tolist()
    return observed


_EXPECT_percentiles_label_as_track_id_renames_column = {
    "'track_id' in out.columns": True,
    "'cell_label' not in out.columns": True,
    "out['cell_p50_intensity_ch0'].tolist()": [3.0, 3.0],
}


def _case_percentiles_none_intensity_stack_returns_empty_frame():
    observed = {}
    from spacr.timelapse import _compute_intensity_percentiles_per_channel

    out = _compute_intensity_percentiles_per_channel(
        _labelled_stack(), None, 0, "nucleus"
    )
    observed['out.empty'] = bool(out.empty)
    observed['list(out.columns)'] = list(out.columns)

    out_tid = _compute_intensity_percentiles_per_channel(
        _labelled_stack(), None, 0, "nucleus", label_as_track_id=True
    )
    observed['list(out_tid.columns)'] = list(out_tid.columns)
    return observed


_EXPECT_percentiles_none_intensity_stack_returns_empty_frame = {
    'out.empty': True,
    'list(out.columns)': ['frame', 'nucleus_label'],
    'list(out_tid.columns)': ['frame', 'track_id'],
}


def _case_percentiles_bad_channel_index_returns_empty_frame(chan):
    observed = {}
    from spacr.timelapse import _compute_intensity_percentiles_per_channel

    mask = _labelled_stack(T=1)
    inten = np.ones((1, 16, 16, 3), dtype=float)
    out = _compute_intensity_percentiles_per_channel(mask, inten, chan, "pathogen")
    observed['out.empty'] = bool(out.empty)
    observed['list(out.columns)'] = list(out.columns)
    return observed


_EXPECT_percentiles_bad_channel_index_returns_empty_frame = {
    'out.empty': True,
    'list(out.columns)': ['frame', 'pathogen_label'],
}


def _case_percentiles_skips_empty_frames_and_returns_empty_when_all_empty():
    observed = {}
    from spacr.timelapse import _compute_intensity_percentiles_per_channel

    mask = np.zeros((3, 8, 8), dtype=np.int32)
    mask[1, 1:4, 1:4] = 4  # only frame 1 has an object
    inten = np.full((3, 8, 8, 1), 2.5)

    out = _compute_intensity_percentiles_per_channel(mask, inten, 0, "cytoplasm",
                                                     percentiles=(50,))
    observed['len(out)'] = len(out)
    observed["out['frame'].tolist()"] = out['frame'].tolist()
    observed["out['cytoplasm_label'].tolist()"] = out['cytoplasm_label'].tolist()

    # No labels at all anywhere -> the "no dfs" return.
    empty = _compute_intensity_percentiles_per_channel(
        np.zeros((2, 8, 8), dtype=np.int32), inten[:2], 0, "cell"
    )
    observed['empty.empty'] = bool(empty.empty)
    observed['list(empty.columns)'] = list(empty.columns)
    return observed


_EXPECT_percentiles_skips_empty_frames_and_returns_empty_when_all_empty = {
    'len(out)': 1,
    "out['frame'].tolist()": [1],
    "out['cytoplasm_label'].tolist()": [4],
    'empty.empty': True,
    'list(empty.columns)': ['frame', 'cell_label'],
}


def _case_percentiles_negative_only_labels_are_skipped():
    """np.any() is truthy for a -1 mask but there are no labels > 0."""
    observed = {}
    from spacr.timelapse import _compute_intensity_percentiles_per_channel

    mask = np.zeros((1, 8, 8), dtype=np.int32)
    mask[0, 0:3, 0:3] = -1
    inten = np.ones((1, 8, 8, 1), dtype=float)

    out = _compute_intensity_percentiles_per_channel(mask, inten, 0, "cell")
    observed['out.empty'] = bool(out.empty)
    observed['list(out.columns)'] = list(out.columns)
    return observed


_EXPECT_percentiles_negative_only_labels_are_skipped = {
    'out.empty': True,
    'list(out.columns)': ['frame', 'cell_label'],
}


def _case_percentiles_all_nan_object_is_dropped():
    """A label whose pixels are all NaN yields no record (vals.size == 0)."""
    observed = {}
    from spacr.timelapse import _compute_intensity_percentiles_per_channel

    mask = _labelled_stack(T=1)
    inten = np.full((1, 16, 16, 1), 5.0)
    inten[0, 2:6, 2:6, 0] = np.nan  # label 1 is entirely non-finite

    out = _compute_intensity_percentiles_per_channel(mask, inten, 0, "cell",
                                                     percentiles=(50,))
    observed["out['cell_label'].tolist()"] = out['cell_label'].tolist()
    observed["out['cell_p50_intensity_ch0'].tolist()"] = out['cell_p50_intensity_ch0'].tolist()
    return observed


_EXPECT_percentiles_all_nan_object_is_dropped = {
    "out['cell_label'].tolist()": [2],
    "out['cell_p50_intensity_ch0'].tolist()": [5.0],
}


def _case_percentiles_all_objects_nan_returns_empty():
    observed = {}
    from spacr.timelapse import _compute_intensity_percentiles_per_channel

    mask = _labelled_stack(T=1)
    inten = np.full((1, 16, 16, 1), np.nan)
    out = _compute_intensity_percentiles_per_channel(mask, inten, 0, "cell")
    observed['out.empty'] = bool(out.empty)
    observed['list(out.columns)'] = list(out.columns)
    return observed


_EXPECT_percentiles_all_objects_nan_returns_empty = {
    'out.empty': True,
    'list(out.columns)': ['frame', 'cell_label'],
}


# ---------------------------------------------------------------------------
# _make_adjusted_qc_panel
# ---------------------------------------------------------------------------

def _panel_df(plate="plate1", well="A01"):
    return pd.DataFrame({"plateID": [plate] * 4, "wellID": [well] * 4})


def _full_payload(rng=None, log=False, thr=1.5, chan=2):
    rng = rng or np.random.default_rng(1)
    return {
        "infection_hist_data": {
            "intensities_inf": rng.normal(5.0, 1.0, 40),
            "intensities_uninf": rng.normal(1.0, 0.5, 40),
            "bin_edges": np.linspace(-2, 9, 21),
            "thr_val": thr,
            "pathogen_chan": chan,
            "log_transform": log,
        },
        "infection_pca_data": {
            "coords": rng.normal(size=(30, 3)),
            "labels": np.array([True, False] * 15),
            "method_label": "UMAP",
        },
        "infection_xgb_importance": {
            "feature_names": ["cell_area", "cell_p95_intensity_ch2"],
            "feature_importances": [0.7, 0.3],
        },
    }


def _case_adjusted_qc_panel_full_payload_writes_png(tmp_path):
    observed = {}
    from spacr.timelapse import _make_adjusted_qc_panel

    out_dir = tmp_path / "motility"  # deliberately absent -> exercises makedirs
    settings = _full_payload()
    _make_adjusted_qc_panel(_panel_df(), "adjusted_infected", str(out_dir),
                            settings, "adjusted")

    expected = out_dir / "infection_qc_panel_adjusted_plate1_A01.png"
    observed['expected.is_file()'] = bool(expected.is_file())
    observed['expected.stat().st_size > 1000'] = bool(expected.stat().st_size > 1000)
    observed['expected.read_bytes()[:8]'] = expected.read_bytes()[:8]
    return observed


_EXPECT_adjusted_qc_panel_full_payload_writes_png = {
    'expected.is_file()': True,
    'expected.stat().st_size > 1000': True,
    'expected.read_bytes()[:8]': b'\x89PNG\r\n\x1a\n',
}


def _case_adjusted_qc_panel_log_and_multi_well_tag(tmp_path, capsys):
    observed = {}
    from spacr.timelapse import _make_adjusted_qc_panel

    df = pd.DataFrame(
        {"plateID": ["plate1"] * 4, "wellID": ["A01", "A01", "B02", "B02"]}
    )
    settings = _full_payload(log=True)
    _make_adjusted_qc_panel(df, "adjusted_infected", str(tmp_path), settings, "mask")

    expected = tmp_path / "infection_qc_panel_mask_plate1_MULTI_WELLS.png"
    observed['expected.is_file()'] = bool(expected.is_file())
    observed['str(expected) in capsys.readouterr().out'] = bool(str(expected) in capsys.readouterr().out)
    return observed


_EXPECT_adjusted_qc_panel_log_and_multi_well_tag = {
    'expected.is_file()': True,
    'str(expected) in capsys.readouterr().out': True,
}


def _case_adjusted_qc_panel_no_threshold_and_no_channel(tmp_path):
    """thr_val None skips the axvline, pathogen_chan None uses the plain label."""
    observed = {}
    from spacr.timelapse import _make_adjusted_qc_panel

    settings = _full_payload(thr=None, chan=None)
    settings["infection_hist_data"]["thr_val"] = None
    settings["infection_hist_data"]["pathogen_chan"] = None
    _make_adjusted_qc_panel(_panel_df(), "adjusted_infected", str(tmp_path),
                            settings, "adjusted")
    observed["(tmp_path / 'infection_qc_panel_adjusted_plate1_A01.png').is_file()"] = bool((tmp_path / 'infection_qc_panel_adjusted_plate1_A01.png').is_file())
    return observed


_EXPECT_adjusted_qc_panel_no_threshold_and_no_channel = {
    "(tmp_path / 'infection_qc_panel_adjusted_plate1_A01.png').is_file()": True,
}


def _case_adjusted_qc_panel_empty_payloads_render_placeholders(tmp_path):
    """No hist / no PCA / no XGB -> all three 'no data' placeholder branches."""
    observed = {}
    from spacr.timelapse import _make_adjusted_qc_panel

    settings = {}
    _make_adjusted_qc_panel(_panel_df(), "adjusted_infected", str(tmp_path),
                            settings, "adjusted")
    observed["(tmp_path / 'infection_qc_panel_adjusted_plate1_A01.png').is_file()"] = bool((tmp_path / 'infection_qc_panel_adjusted_plate1_A01.png').is_file())
    return observed


_EXPECT_adjusted_qc_panel_empty_payloads_render_placeholders = {
    "(tmp_path / 'infection_qc_panel_adjusted_plate1_A01.png').is_file()": True,
}


def _case_adjusted_qc_panel_malformed_payloads_render_placeholders(tmp_path):
    """Present-but-unusable payloads take the same placeholder branches."""
    observed = {}
    from spacr.timelapse import _make_adjusted_qc_panel

    settings = {
        # values present but no bins -> histogram placeholder
        "infection_hist_data": {
            "intensities_inf": [1.0, 2.0],
            "intensities_uninf": [0.1],
            "bin_edges": [],
        },
        # coords is 1-D -> shape check fails -> PCA placeholder
        "infection_pca_data": {
            "coords": np.arange(10.0),
            "labels": np.zeros(10, dtype=bool),
        },
        # name/value length mismatch -> XGB placeholder
        "infection_xgb_importance": {
            "feature_names": ["a", "b", "c"],
            "feature_importances": [1.0],
        },
    }
    _make_adjusted_qc_panel(_panel_df(plate="p9", well="H12"),
                            "adjusted_infected", str(tmp_path), settings, "xgb")
    observed["(tmp_path / 'infection_qc_panel_xgb_p9_H12.png').is_file()"] = bool((tmp_path / 'infection_qc_panel_xgb_p9_H12.png').is_file())
    return observed


_EXPECT_adjusted_qc_panel_malformed_payloads_render_placeholders = {
    "(tmp_path / 'infection_qc_panel_xgb_p9_H12.png').is_file()": True,
}


def _case_adjusted_qc_panel_pca_labels_none_takes_placeholder(tmp_path):
    observed = {}
    from spacr.timelapse import _make_adjusted_qc_panel

    settings = _full_payload()
    settings["infection_pca_data"] = {"coords": np.zeros((5, 2)), "labels": None}
    _make_adjusted_qc_panel(_panel_df(), "adjusted_infected", str(tmp_path),
                            settings, "adjusted")
    observed["(tmp_path / 'infection_qc_panel_adjusted_plate1_A01.png').is_file()"] = bool((tmp_path / 'infection_qc_panel_adjusted_plate1_A01.png').is_file())
    return observed


_EXPECT_adjusted_qc_panel_pca_labels_none_takes_placeholder = {
    "(tmp_path / 'infection_qc_panel_adjusted_plate1_A01.png').is_file()": True,
}


def _case_adjusted_qc_panel_meta_tag_without_plate_columns(tmp_path):
    """No plateID/wellID at all -> MULTI_PLATES_MULTI_WELLS filename."""
    observed = {}
    from spacr.timelapse import _make_adjusted_qc_panel

    _make_adjusted_qc_panel(pd.DataFrame({"cellID": [1, 2]}), "adjusted_infected",
                            str(tmp_path), _full_payload(), "adjusted")
    observed["(tmp_path / 'infection_qc_panel_adjusted_MULTI_PLATES_MULTI_WELLS.png').is_file()"] = bool((tmp_path / 'infection_qc_panel_adjusted_MULTI_PLATES_MULTI_WELLS.png').is_file())
    return observed


_EXPECT_adjusted_qc_panel_meta_tag_without_plate_columns = {
    "(tmp_path / 'infection_qc_panel_adjusted_MULTI_PLATES_MULTI_WELLS.png').is_file()": True,
}


# ---------------------------------------------------------------------------
# _infection_qc_histogram
# ---------------------------------------------------------------------------

def _case_infection_qc_histogram_relabel_writes_png_and_payload(tmp_path, capsys):
    observed = {}
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df()
    settings = {}
    out, col = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    observed['col'] = col
    observed["'adjusted_infected' in out.columns"] = bool('adjusted_infected' in out.columns)
    observed['len(out) == len(df)'] = bool(len(out) == len(df))
    # The synthetic split is clean, so the histogram must reproduce it exactly.
    observed['_per_cell_positive(out)'] = _per_cell_positive(out)
    observed["out['adjusted_infected'].dtype"] = out['adjusted_infected'].dtype

    png = tmp_path / "infection_intensity_histogram_plate1_A01.png"
    observed['png.is_file()'] = bool(png.is_file())
    observed["settings['infection_intensity_qc_panel_type']"] = settings['infection_intensity_qc_panel_type']
    observed["settings['infection_intensity_qc_panel_path'] == str(png)"] = bool(settings['infection_intensity_qc_panel_path'] == str(png))

    payload = settings["infection_hist_data"]
    observed["payload['intensity_col']"] = payload['intensity_col']
    observed["payload['pathogen_chan']"] = payload['pathogen_chan']
    observed["payload['log_transform'] is False"] = bool(payload['log_transform'] is False)
    observed["payload['intensities_inf'].size"] = payload['intensities_inf'].size
    observed["payload['intensities_uninf'].size"] = payload['intensities_uninf'].size
    observed["payload['bin_edges'].size"] = payload['bin_edges'].size
    observed["200.0 < payload['thr_val'] < 9000.0"] = bool(200.0 < payload['thr_val'] < 9000.0)

    observed["'Automatic intensity threshold' in capsys.readouterr().out"] = bool('Automatic intensity threshold' in capsys.readouterr().out)
    return observed


_EXPECT_infection_qc_histogram_relabel_writes_png_and_payload = {
    'col': 'adjusted_infected',
    "'adjusted_infected' in out.columns": True,
    'len(out) == len(df)': True,
    '_per_cell_positive(out)': 15,
    "out['adjusted_infected'].dtype": bool,
    'png.is_file()': True,
    "settings['infection_intensity_qc_panel_type']": 'histogram',
    "settings['infection_intensity_qc_panel_path'] == str(png)": True,
    "payload['intensity_col']": 'cell_p95_intensity_ch2',
    "payload['pathogen_chan']": 2,
    "payload['log_transform'] is False": True,
    "payload['intensities_inf'].size": 15,
    "payload['intensities_uninf'].size": 15,
    "payload['bin_edges'].size": 65,
    "200.0 < payload['thr_val'] < 9000.0": True,
    "'Automatic intensity threshold' in capsys.readouterr().out": True,
}


def _case_infection_qc_histogram_no_intensity_column(tmp_path, capsys):
    observed = {}
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df()
    settings = {}
    out, col = _infection_qc_histogram(df, settings, "infected", 7, str(tmp_path))

    observed['col'] = col
    observed['out is df'] = bool(out is df)
    observed["'adjusted_infected' not in out.columns"] = bool('adjusted_infected' not in out.columns)
    observed["settings['infection_hist_data'] is None"] = bool(settings['infection_hist_data'] is None)
    observed["settings['infection_intensity_qc_panel_path'] is None"] = bool(settings['infection_intensity_qc_panel_path'] is None)
    observed["settings['infection_intensity_qc_panel_type']"] = settings['infection_intensity_qc_panel_type']
    observed["'None of' in capsys.readouterr().out"] = bool('None of' in capsys.readouterr().out)
    observed['not list(tmp_path.iterdir())'] = bool(not list(tmp_path.iterdir()))
    return observed


_EXPECT_infection_qc_histogram_no_intensity_column = {
    'col': 'infected',
    'out is df': True,
    "'adjusted_infected' not in out.columns": True,
    "settings['infection_hist_data'] is None": True,
    "settings['infection_intensity_qc_panel_path'] is None": True,
    "settings['infection_intensity_qc_panel_type']": 'histogram',
    "'None of' in capsys.readouterr().out": True,
    'not list(tmp_path.iterdir())': True,
}


def _case_infection_qc_histogram_falls_back_to_mean_column(tmp_path):
    observed = {}
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df(col="mean")
    settings = {"infection_intensity_qc_graphs": False}
    out, col = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    observed['col'] = col
    observed["settings['infection_hist_data']['intensity_col']"] = settings['infection_hist_data']['intensity_col']
    return observed


_EXPECT_infection_qc_histogram_falls_back_to_mean_column = {
    'col': 'adjusted_infected',
    "settings['infection_hist_data']['intensity_col']": 'cell_mean_intensity_ch2',
}


def _case_infection_qc_histogram_prefers_p95_over_mean(tmp_path):
    observed = {}
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df()
    df["cell_mean_intensity_ch2"] = 1.0  # useless column, must be ignored
    settings = {"infection_intensity_qc_graphs": False}
    _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))
    observed["settings['infection_hist_data']['intensity_col']"] = settings['infection_hist_data']['intensity_col']
    return observed


_EXPECT_infection_qc_histogram_prefers_p95_over_mean = {
    "settings['infection_hist_data']['intensity_col']": 'cell_p95_intensity_ch2',
}


def _case_infection_qc_histogram_too_few_cells_skips(tmp_path, capsys):
    observed = {}
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df(n_cells=8)
    settings = {}
    out, col = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    observed['col'] = col
    observed["'adjusted_infected' not in out.columns"] = bool('adjusted_infected' not in out.columns)
    observed["settings['infection_hist_data'] is None"] = bool(settings['infection_hist_data'] is None)
    observed["settings['infection_intensity_qc_panel_path'] is None"] = bool(settings['infection_intensity_qc_panel_path'] is None)
    observed["'Too few cells' in capsys.readouterr().out"] = bool('Too few cells' in capsys.readouterr().out)
    return observed


_EXPECT_infection_qc_histogram_too_few_cells_skips = {
    'col': 'infected',
    "'adjusted_infected' not in out.columns": True,
    "settings['infection_hist_data'] is None": True,
    "settings['infection_intensity_qc_panel_path'] is None": True,
    "'Too few cells' in capsys.readouterr().out": True,
}


def _case_infection_qc_histogram_constant_intensity_skips(tmp_path, capsys):
    observed = {}
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df(n_cells=40)
    df["cell_p95_intensity_ch2"] = 500.0  # nunique == 1
    settings = {}
    out, col = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    observed['col'] = col
    observed["settings['infection_hist_data'] is None"] = bool(settings['infection_hist_data'] is None)
    observed["'Too few cells or no intensity variation' in capsys.readouterr().out"] = bool('Too few cells or no intensity variation' in capsys.readouterr().out)
    return observed


_EXPECT_infection_qc_histogram_constant_intensity_skips = {
    'col': 'infected',
    "settings['infection_hist_data'] is None": True,
    "'Too few cells or no intensity variation' in capsys.readouterr().out": True,
}


def _case_infection_qc_histogram_drops_stale_adjusted_columns(tmp_path):
    """Re-running on a DB-loaded frame must not produce _x / _y merge suffixes."""
    observed = {}
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df()
    df["adjusted_infected"] = False
    df["adjusted_infected_x"] = 1
    df["adjusted_infected_y"] = 2
    settings = {"infection_intensity_qc_graphs": False}
    out, col = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    observed["[c for c in out.columns if c.startswith('adjusted_infected')]"] = [c for c in out.columns if c.startswith('adjusted_infected')]
    observed['_per_cell_positive(out)'] = _per_cell_positive(out)
    return observed


_EXPECT_infection_qc_histogram_drops_stale_adjusted_columns = {
    "[c for c in out.columns if c.startswith('adjusted_infected')]": ['adjusted_infected'],
    '_per_cell_positive(out)': 15,
}


def _case_infection_qc_histogram_percentile_fallback(tmp_path, capsys):
    """No bin reaches the infected fraction -> percentile fallback threshold."""
    observed = {}
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df()
    df["infected"] = False  # nothing is infected -> frac_inf is all zero
    settings = {
        "infection_intensity_qc_graphs": False,
        "infection_hist_percentile": 10.0,
    }
    out, col = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    thr = settings["infection_hist_data"]["thr_val"]
    raw = df.groupby("cellID")["cell_p95_intensity_ch2"].mean()
    observed['thr == pytest.approx(float(np.nanpercentile(raw.to_numpy(), 10.0)))'] = bool(thr == pytest.approx(float(np.nanpercentile(raw.to_numpy(), 10.0))))
    observed['_per_cell_positive(out) == int((raw >= thr).sum())'] = bool(_per_cell_positive(out) == int((raw >= thr).sum()))
    observed["settings['infection_hist_data']['intensities_inf'].size"] = settings['infection_hist_data']['intensities_inf'].size
    observed["'Could not find bin with infected' in capsys.readouterr().out"] = bool('Could not find bin with infected' in capsys.readouterr().out)
    return observed


_EXPECT_infection_qc_histogram_percentile_fallback = {
    'thr == pytest.approx(float(np.nanpercentile(raw.to_numpy(), 10.0)))': True,
    '_per_cell_positive(out) == int((raw >= thr).sum())': True,
    "settings['infection_hist_data']['intensities_inf'].size": 0,
    "'Could not find bin with infected' in capsys.readouterr().out": True,
}


def _case_infection_qc_histogram_clamps_bins_fraction_and_percentile(tmp_path):
    observed = {}
    from spacr.timelapse import _infection_qc_histogram

    settings = {
        "infection_intensity_n_bins": 4,          # clamped up to 10
        "infection_intensity_frac_infected": 0.1,  # clamped up to 0.5
        "infection_hist_percentile": -20.0,        # clamped up to 0.0
        "infection_intensity_qc_graphs": False,
    }
    _infection_qc_histogram(_qc_df(), settings, "infected", 2, str(tmp_path))
    observed["settings['infection_hist_data']['bin_edges'].size"] = settings['infection_hist_data']['bin_edges'].size

    settings2 = {
        "infection_intensity_n_bins": 5000,        # clamped down to 256
        "infection_intensity_frac_infected": 5.0,  # clamped down to 0.95
        "infection_hist_percentile": 500.0,        # clamped down to 100.0
        "infection_intensity_qc_graphs": False,
    }
    _infection_qc_histogram(_qc_df(), settings2, "infected", 2, str(tmp_path))
    observed["settings2['infection_hist_data']['bin_edges'].size"] = settings2['infection_hist_data']['bin_edges'].size
    return observed


_EXPECT_infection_qc_histogram_clamps_bins_fraction_and_percentile = {
    "settings['infection_hist_data']['bin_edges'].size": 11,
    "settings2['infection_hist_data']['bin_edges'].size": 257,
}


def _case_infection_qc_histogram_remove_mode_drops_conflicting_cells(tmp_path, capsys):
    observed = {}
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df(seed=3)
    # Two cells whose mask label contradicts their intensity.
    df.loc[df["cellID"] == 1, "infected"] = True    # dim but called infected
    df.loc[df["cellID"] == 30, "infected"] = False  # bright but called uninfected
    settings = {
        "infection_intensity_mode": "REMOVE",  # case-insensitive
        "infection_intensity_qc_graphs": False,
    }
    out, col = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    observed['col'] = col
    observed["set(out['cellID'])"] = set(out['cellID'])
    observed['len(out) == len(df) - 4'] = bool(len(out) == len(df) - 4)
    observed['_per_cell_positive(out)'] = _per_cell_positive(out)
    observed["'Removed 2 cells' in capsys.readouterr().out"] = bool('Removed 2 cells' in capsys.readouterr().out)
    observed['not list(tmp_path.iterdir())'] = bool(not list(tmp_path.iterdir()))
    return observed


_EXPECT_infection_qc_histogram_remove_mode_drops_conflicting_cells = {
    'col': 'adjusted_infected',
    "set(out['cellID'])": set(range(1, 31)) - {1, 30},
    'len(out) == len(df) - 4': True,
    '_per_cell_positive(out)': 14,
    "'Removed 2 cells' in capsys.readouterr().out": True,
    'not list(tmp_path.iterdir())': True,
}


def _case_infection_qc_histogram_remove_mode_without_conflicts_keeps_all(tmp_path):
    observed = {}
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df()
    settings = {
        "infection_intensity_mode": "remove",
        "infection_intensity_qc_graphs": False,
    }
    out, col = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    observed['len(out) == len(df)'] = bool(len(out) == len(df))
    observed['_per_cell_positive(out)'] = _per_cell_positive(out)
    return observed


_EXPECT_infection_qc_histogram_remove_mode_without_conflicts_keeps_all = {
    'len(out) == len(df)': True,
    '_per_cell_positive(out)': 15,
}


def _case_infection_qc_histogram_unknown_mode_falls_back_to_relabel(tmp_path, capsys):
    observed = {}
    from spacr.timelapse import _infection_qc_histogram

    settings = {
        "infection_intensity_mode": "not-a-mode",
        "infection_intensity_qc_graphs": False,
    }
    df = _qc_df()
    out, _ = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))
    observed['len(out) == len(df)'] = bool(len(out) == len(df))
    observed["'mode=relabel' in capsys.readouterr().out"] = bool('mode=relabel' in capsys.readouterr().out)
    return observed


_EXPECT_infection_qc_histogram_unknown_mode_falls_back_to_relabel = {
    'len(out) == len(df)': True,
    "'mode=relabel' in capsys.readouterr().out": True,
}


def _case_infection_qc_histogram_relabels_disagreeing_cells(tmp_path, capsys):
    """Mask says infected but intensity is dim -> the label is flipped."""
    observed = {}
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df()
    df.loc[df["cellID"] == 2, "infected"] = True  # dim cell mislabelled
    settings = {"infection_intensity_qc_graphs": False}
    out, _ = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    observed["bool(out.loc[out['cellID'] == 2, 'adjusted_infected'].iloc[0]) is False"] = bool(bool(out.loc[out['cellID'] == 2, 'adjusted_infected'].iloc[0]) is False)
    observed['_per_cell_positive(out)'] = _per_cell_positive(out)
    observed["'Adjusted infection labels for 1 cells' in capsys.readouterr().out"] = bool('Adjusted infection labels for 1 cells' in capsys.readouterr().out)
    return observed


_EXPECT_infection_qc_histogram_relabels_disagreeing_cells = {
    "bool(out.loc[out['cellID'] == 2, 'adjusted_infected'].iloc[0]) is False": True,
    '_per_cell_positive(out)': 15,
    "'Adjusted infection labels for 1 cells' in capsys.readouterr().out": True,
}


def _case_infection_qc_histogram_nan_cells_fall_back_to_mask_label(tmp_path):
    """Cells dropped by dropna() get their label back from the mask column."""
    observed = {}
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df(n_cells=32)
    df.loc[df["cellID"] == 32, "cell_p95_intensity_ch2"] = np.nan
    df.loc[df["cellID"] == 31, "cell_p95_intensity_ch2"] = np.inf
    settings = {"infection_intensity_qc_graphs": False}
    out, _ = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    observed["out['adjusted_infected'].isna().sum()"] = out['adjusted_infected'].isna().sum()
    # both were in the infected half of the synthetic split
    observed["bool(out.loc[out['cellID'] == 32, 'adjusted_infected'].iloc[0]) is True"] = bool(bool(out.loc[out['cellID'] == 32, 'adjusted_infected'].iloc[0]) is True)
    observed["bool(out.loc[out['cellID'] == 31, 'adjusted_infected'].iloc[0]) is True"] = bool(bool(out.loc[out['cellID'] == 31, 'adjusted_infected'].iloc[0]) is True)
    return observed


_EXPECT_infection_qc_histogram_nan_cells_fall_back_to_mask_label = {
    "out['adjusted_infected'].isna().sum()": 0,
    "bool(out.loc[out['cellID'] == 32, 'adjusted_infected'].iloc[0]) is True": True,
    "bool(out.loc[out['cellID'] == 31, 'adjusted_infected'].iloc[0]) is True": True,
}


def _case_infection_qc_histogram_graphs_disabled_writes_nothing(tmp_path, capsys):
    observed = {}
    from spacr.timelapse import _infection_qc_histogram

    settings = {"infection_intensity_qc_graphs": False}
    _infection_qc_histogram(_qc_df(), settings, "infected", 2, str(tmp_path))

    observed["settings['infection_intensity_qc_panel_path'] is None"] = bool(settings['infection_intensity_qc_panel_path'] is None)
    observed["settings['infection_hist_data'] is not None"] = bool(settings['infection_hist_data'] is not None)
    observed['not list(tmp_path.iterdir())'] = bool(not list(tmp_path.iterdir()))
    observed["'skipping histogram plot' in capsys.readouterr().out"] = bool('skipping histogram plot' in capsys.readouterr().out)
    return observed


_EXPECT_infection_qc_histogram_graphs_disabled_writes_nothing = {
    "settings['infection_intensity_qc_panel_path'] is None": True,
    "settings['infection_hist_data'] is not None": True,
    'not list(tmp_path.iterdir())': True,
    "'skipping histogram plot' in capsys.readouterr().out": True,
}


def _case_infection_qc_histogram_png_filename_uses_meta_tag(tmp_path):
    observed = {}
    from spacr.timelapse import _infection_qc_histogram

    df = pd.concat(
        [_qc_df(plate="plateA", well="C03"), _qc_df(plate="plateB", well="C03", seed=9)],
        ignore_index=True,
    )
    # keep cellIDs unique per plate so the groupby key is well formed
    df.loc[df["plateID"] == "plateB", "cellID"] += 100
    settings = {}
    _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    observed["(tmp_path / 'infection_intensity_histogram_MULTI_PLATES_C03.png').is_file()"] = bool((tmp_path / 'infection_intensity_histogram_MULTI_PLATES_C03.png').is_file())
    return observed


_EXPECT_infection_qc_histogram_png_filename_uses_meta_tag = {
    "(tmp_path / 'infection_intensity_histogram_MULTI_PLATES_C03.png').is_file()": True,
}


def _case_infection_qc_histogram_log_threshold_applied_in_log_space(tmp_path):
    observed = {}
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df()
    settings = {
        "infection_intensity_log": True,
        "infection_intensity_qc_graphs": False,
    }
    out, _ = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    thr_log = settings["infection_hist_data"]["thr_val"]
    raw = df.groupby("cellID")["cell_p95_intensity_ch2"].mean()
    expected = int((raw >= 10.0 ** thr_log).sum())
    observed['expected < len(raw)'] = bool(expected < len(raw))
    observed['_per_cell_positive(out) == expected'] = bool(_per_cell_positive(out) == expected)
    return observed


_EXPECT_infection_qc_histogram_log_threshold_applied_in_log_space = {
    'expected < len(raw)': True,
    '_per_cell_positive(out) == expected': True,
}


def _case_infection_qc_histogram_log_payload_is_log_transformed(tmp_path):
    """The plotting payload is in log space when log_transform is on."""
    observed = {}
    from spacr.timelapse import _infection_qc_histogram

    df = _qc_df()
    settings = {
        "infection_intensity_log": True,
        "infection_intensity_qc_graphs": True,
    }
    _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    payload = settings["infection_hist_data"]
    observed["payload['log_transform'] is True"] = bool(payload['log_transform'] is True)
    raw = df.groupby("cellID")["cell_p95_intensity_ch2"].mean().to_numpy()
    # The helper offsets by eps = max(min(positive) / 2, 1e-6) before log10.
    eps = max(raw.min() * 0.5, 1e-6)
    both = np.concatenate([payload["intensities_inf"], payload["intensities_uninf"]])
    observed['both.max() == pytest.approx(np.log10(raw.max() + eps))'] = bool(both.max() == pytest.approx(np.log10(raw.max() + eps)))
    observed['both.min() == pytest.approx(np.log10(raw.min() + eps))'] = bool(both.min() == pytest.approx(np.log10(raw.min() + eps)))
    # log compresses the 2 orders of magnitude between the two populations
    observed['both.max() - both.min() < 2.0'] = bool(both.max() - both.min() < 2.0)
    observed["(tmp_path / 'infection_intensity_histogram_plate1_A01.png').is_file()"] = bool((tmp_path / 'infection_intensity_histogram_plate1_A01.png').is_file())
    return observed


_EXPECT_infection_qc_histogram_log_payload_is_log_transformed = {
    "payload['log_transform'] is True": True,
    'both.max() == pytest.approx(np.log10(raw.max() + eps))': True,
    'both.min() == pytest.approx(np.log10(raw.min() + eps))': True,
    'both.max() - both.min() < 2.0': True,
    "(tmp_path / 'infection_intensity_histogram_plate1_A01.png').is_file()": True,
}


def _case_infection_qc_histogram_output_feeds_adjusted_panel(tmp_path):
    """End-to-end: the payload the histogram stores renders the panel."""
    observed = {}
    from spacr.timelapse import _infection_qc_histogram, _make_adjusted_qc_panel

    df = _qc_df()
    settings = {"infection_intensity_qc_graphs": False}
    out, col = _infection_qc_histogram(df, settings, "infected", 2, str(tmp_path))

    panel_dir = tmp_path / "panel"
    _make_adjusted_qc_panel(out, col, str(panel_dir), settings, "adjusted")

    png = panel_dir / "infection_qc_panel_adjusted_plate1_A01.png"
    observed['png.is_file()'] = bool(png.is_file())
    observed['png.read_bytes()[:8]'] = png.read_bytes()[:8]
    return observed


_EXPECT_infection_qc_histogram_output_feeds_adjusted_panel = {
    'png.is_file()': True,
    'png.read_bytes()[:8]': b'\x89PNG\r\n\x1a\n',
}


# ---------------------------------------------------------------------------
# behaviour tests: one per public-function contract, one parametrised row per case
# ---------------------------------------------------------------------------

def _run_case(case, tmp_path, monkeypatch, capsys):
    """Call a case helper with only the fixtures its signature asks for."""
    wanted = inspect.signature(case).parameters
    available = {"tmp_path": tmp_path, "monkeypatch": monkeypatch, "capsys": capsys}
    return case(**{k: v for k, v in available.items() if k in wanted})


_P = functools.partial


PERCENTILE_CASES = [
    pytest.param(_case_percentiles_normal_path_values_and_columns, _EXPECT_percentiles_normal_path_values_and_columns, id="normal-path-values-and-columns"),
    pytest.param(_case_percentiles_label_as_track_id_renames_column, _EXPECT_percentiles_label_as_track_id_renames_column, id="label-as-track-id-renames-column"),
    pytest.param(_case_percentiles_none_intensity_stack_returns_empty_frame, _EXPECT_percentiles_none_intensity_stack_returns_empty_frame, id="none-intensity-stack-returns-empty-frame"),
    pytest.param(_P(_case_percentiles_bad_channel_index_returns_empty_frame, chan=None), _EXPECT_percentiles_bad_channel_index_returns_empty_frame, id="bad-channel-index-returns-empty-frame-none"),
    pytest.param(_P(_case_percentiles_bad_channel_index_returns_empty_frame, chan=-1), _EXPECT_percentiles_bad_channel_index_returns_empty_frame, id="bad-channel-index-returns-empty-frame-negative"),
    pytest.param(_P(_case_percentiles_bad_channel_index_returns_empty_frame, chan=5), _EXPECT_percentiles_bad_channel_index_returns_empty_frame, id="bad-channel-index-returns-empty-frame-past-end"),
    pytest.param(_case_percentiles_skips_empty_frames_and_returns_empty_when_all_empty, _EXPECT_percentiles_skips_empty_frames_and_returns_empty_when_all_empty, id="skips-empty-frames-and-returns-empty-when-all-empty"),
    pytest.param(_case_percentiles_negative_only_labels_are_skipped, _EXPECT_percentiles_negative_only_labels_are_skipped, id="negative-only-labels-are-skipped"),
    pytest.param(_case_percentiles_all_nan_object_is_dropped, _EXPECT_percentiles_all_nan_object_is_dropped, id="all-nan-object-is-dropped"),
    pytest.param(_case_percentiles_all_objects_nan_returns_empty, _EXPECT_percentiles_all_objects_nan_returns_empty, id="all-objects-nan-returns-empty"),
]


@pytest.mark.parametrize("case, expected", PERCENTILE_CASES)
def test_percentiles_per_channel_are_exact_and_return_an_empty_frame_with_the_label_column_when_nothing_can_be_measured(case, expected, tmp_path, monkeypatch, capsys):
    """Per-object intensity percentiles: values, columns, track_id rename, skipped frames and every empty-frame guard."""
    observed = _run_case(case, tmp_path, monkeypatch, capsys)
    assert observed == expected


PANEL_CASES = [
    pytest.param(_case_adjusted_qc_panel_full_payload_writes_png, _EXPECT_adjusted_qc_panel_full_payload_writes_png, id="full-payload-writes-png"),
    pytest.param(_case_adjusted_qc_panel_log_and_multi_well_tag, _EXPECT_adjusted_qc_panel_log_and_multi_well_tag, id="log-and-multi-well-tag"),
    pytest.param(_case_adjusted_qc_panel_no_threshold_and_no_channel, _EXPECT_adjusted_qc_panel_no_threshold_and_no_channel, id="no-threshold-and-no-channel"),
    pytest.param(_case_adjusted_qc_panel_empty_payloads_render_placeholders, _EXPECT_adjusted_qc_panel_empty_payloads_render_placeholders, id="empty-payloads-render-placeholders"),
    pytest.param(_case_adjusted_qc_panel_malformed_payloads_render_placeholders, _EXPECT_adjusted_qc_panel_malformed_payloads_render_placeholders, id="malformed-payloads-render-placeholders"),
    pytest.param(_case_adjusted_qc_panel_pca_labels_none_takes_placeholder, _EXPECT_adjusted_qc_panel_pca_labels_none_takes_placeholder, id="pca-labels-none-takes-placeholder"),
    pytest.param(_case_adjusted_qc_panel_meta_tag_without_plate_columns, _EXPECT_adjusted_qc_panel_meta_tag_without_plate_columns, id="meta-tag-without-plate-columns"),
]


@pytest.mark.parametrize("case, expected", PANEL_CASES)
def test_adjusted_qc_panel_writes_a_png_named_for_its_plate_and_wells_with_placeholders_for_unusable_payloads(case, expected, tmp_path, monkeypatch, capsys):
    """The 3-panel PNG: file name tag, makedirs, printed path, placeholder axes for missing or malformed payloads."""
    observed = _run_case(case, tmp_path, monkeypatch, capsys)
    assert observed == expected


HISTOGRAM_CASES = [
    pytest.param(_case_infection_qc_histogram_relabel_writes_png_and_payload, _EXPECT_infection_qc_histogram_relabel_writes_png_and_payload, id="relabel-writes-png-and-payload"),
    pytest.param(_case_infection_qc_histogram_no_intensity_column, _EXPECT_infection_qc_histogram_no_intensity_column, id="no-intensity-column"),
    pytest.param(_case_infection_qc_histogram_falls_back_to_mean_column, _EXPECT_infection_qc_histogram_falls_back_to_mean_column, id="falls-back-to-mean-column"),
    pytest.param(_case_infection_qc_histogram_prefers_p95_over_mean, _EXPECT_infection_qc_histogram_prefers_p95_over_mean, id="prefers-p95-over-mean"),
    pytest.param(_case_infection_qc_histogram_too_few_cells_skips, _EXPECT_infection_qc_histogram_too_few_cells_skips, id="too-few-cells-skips"),
    pytest.param(_case_infection_qc_histogram_constant_intensity_skips, _EXPECT_infection_qc_histogram_constant_intensity_skips, id="constant-intensity-skips"),
    pytest.param(_case_infection_qc_histogram_drops_stale_adjusted_columns, _EXPECT_infection_qc_histogram_drops_stale_adjusted_columns, id="drops-stale-adjusted-columns"),
    pytest.param(_case_infection_qc_histogram_percentile_fallback, _EXPECT_infection_qc_histogram_percentile_fallback, id="percentile-fallback"),
    pytest.param(_case_infection_qc_histogram_clamps_bins_fraction_and_percentile, _EXPECT_infection_qc_histogram_clamps_bins_fraction_and_percentile, id="clamps-bins-fraction-and-percentile"),
    pytest.param(_case_infection_qc_histogram_remove_mode_drops_conflicting_cells, _EXPECT_infection_qc_histogram_remove_mode_drops_conflicting_cells, id="remove-mode-drops-conflicting-cells"),
    pytest.param(_case_infection_qc_histogram_remove_mode_without_conflicts_keeps_all, _EXPECT_infection_qc_histogram_remove_mode_without_conflicts_keeps_all, id="remove-mode-without-conflicts-keeps-all"),
    pytest.param(_case_infection_qc_histogram_unknown_mode_falls_back_to_relabel, _EXPECT_infection_qc_histogram_unknown_mode_falls_back_to_relabel, id="unknown-mode-falls-back-to-relabel"),
    pytest.param(_case_infection_qc_histogram_relabels_disagreeing_cells, _EXPECT_infection_qc_histogram_relabels_disagreeing_cells, id="relabels-disagreeing-cells"),
    pytest.param(_case_infection_qc_histogram_nan_cells_fall_back_to_mask_label, _EXPECT_infection_qc_histogram_nan_cells_fall_back_to_mask_label, id="nan-cells-fall-back-to-mask-label"),
    pytest.param(_case_infection_qc_histogram_graphs_disabled_writes_nothing, _EXPECT_infection_qc_histogram_graphs_disabled_writes_nothing, id="graphs-disabled-writes-nothing"),
    pytest.param(_case_infection_qc_histogram_png_filename_uses_meta_tag, _EXPECT_infection_qc_histogram_png_filename_uses_meta_tag, id="png-filename-uses-meta-tag"),
    pytest.param(_case_infection_qc_histogram_log_threshold_applied_in_log_space, _EXPECT_infection_qc_histogram_log_threshold_applied_in_log_space, id="log-threshold-applied-in-log-space"),
    pytest.param(_case_infection_qc_histogram_log_payload_is_log_transformed, _EXPECT_infection_qc_histogram_log_payload_is_log_transformed, id="log-payload-is-log-transformed"),
    pytest.param(_case_infection_qc_histogram_output_feeds_adjusted_panel, _EXPECT_infection_qc_histogram_output_feeds_adjusted_panel, id="output-feeds-adjusted-panel"),
]


@pytest.mark.parametrize("case, expected", HISTOGRAM_CASES)
def test_infection_qc_histogram_relabels_or_removes_cells_by_intensity_and_records_the_payload(case, expected, tmp_path, monkeypatch, capsys):
    """Histogram relabelling: modes, clamps, fallbacks, skip paths, payload, PNG and printed messages."""
    observed = _run_case(case, tmp_path, monkeypatch, capsys)
    assert observed == expected
