"""Synthetic CPU-only coverage for the DB/CSV plotting block of ``spacr.plot``.

Covers ``plot_data_from_db``, ``plot_data_from_csv``, ``plot_region`` and
``plot_image_grid`` (plot.py lines ~3970-4381).

Everything runs on small deterministic fixtures:

* a ``measurements.db`` holding a ``saliency_image_correlations`` table
  (the short-circuit branch of ``plot_data_from_db``),
* a full ``cell``/``cytoplasm``/``nucleus``/``pathogen`` schema so
  ``io._read_and_merge_data`` + the ``recruitment`` derivation run for real,
* per-plate CSVs for ``plot_data_from_csv``,
* a merged ``.npy`` FOV + PNG crops + an activation DB for ``plot_region``.

No network, no CUDA, no Cellpose.
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


# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


N_FIELDS = 3          # objects per well
COLUMNS = [f"c{i}" for i in range(1, 11)]
ROWS = ["r1", "r2"]


# ---------------------------------------------------------------------------
# synthetic measurement DBs
# ---------------------------------------------------------------------------

def _well_grid():
    """(plateID, rowID, columnID, fieldID) tuples, N_FIELDS objects per well."""
    out = []
    for row in ROWS:
        for col in COLUMNS:
            for f in range(N_FIELDS):
                out.append(("plate1", row, col, f"f{f + 1}"))
    return out


def _saliency_frame(rng):
    """A ``saliency_image_correlations``-style table (one row per object)."""
    grid = _well_grid()
    n = len(grid)
    # Row r1 gets a clearly different mean than r2 so the group test is real.
    offset = np.array([0.0 if r == "r1" else 1.5 for _, r, _, _ in grid])
    return pd.DataFrame({
        "plateID": [g[0] for g in grid],
        "rowID": [g[1] for g in grid],
        "columnID": [g[2] for g in grid],
        "fieldID": [g[3] for g in grid],
        "object_label": np.arange(1, n + 1),
        "saliency_correlation": rng.normal(0.0, 0.25, n) + offset,
        "other_metric": rng.normal(5.0, 1.0, n),
    })


def _make_saliency_db(dirpath, rng):
    """Write ``<dirpath>/measurements/measurements.db`` with the saliency table."""
    meas = os.path.join(dirpath, "measurements")
    os.makedirs(meas, exist_ok=True)
    db = os.path.join(meas, "measurements.db")
    con = sqlite3.connect(db)
    try:
        _saliency_frame(rng).to_sql("saliency_image_correlations", con, index=False)
    finally:
        con.close()
    return db


def _entity_frame(rng, entity, with_cell_id):
    """One measurement table with the metadata columns spacr joins on."""
    grid = _well_grid()
    n = len(grid)
    cols = {
        "object_label": np.arange(1, n + 1),
        "plateID": [g[0] for g in grid],
        "rowID": [g[1] for g in grid],
        "columnID": [g[2] for g in grid],
        "fieldID": [g[3] for g in grid],
        "prcf": [f"{g[0]}_{g[1]}_{g[2]}_{g[3]}" for g in grid],
        "prc": [f"{g[0]}_{g[1]}_{g[2]}" for g in grid],
    }
    if with_cell_id:
        # Parent-cell link: an INTEGER label (spacr prefixes 'o' itself).
        cols["cell_id"] = np.arange(1, n + 1)
    offset = np.array([0.0 if g[1] == "r1" else 900.0 for g in grid])
    for ch in (0, 1):
        cols[f"{entity}_channel_{ch}_mean_intensity"] = (
            rng.uniform(500, 3000, n) + (offset if ch == 1 else 0.0)
        )
    cols[f"{entity}_area"] = rng.uniform(200, 4000, n)
    return pd.DataFrame(cols)


def _make_merge_db(dirpath, rng):
    """Write a measurements.db with cell/cytoplasm/nucleus/pathogen tables."""
    meas = os.path.join(dirpath, "measurements")
    os.makedirs(meas, exist_ok=True)
    db = os.path.join(meas, "measurements.db")
    con = sqlite3.connect(db)
    try:
        for entity in ("cell", "cytoplasm"):
            _entity_frame(rng, entity, with_cell_id=False).to_sql(entity, con, index=False)
        for entity in ("nucleus", "pathogen"):
            _entity_frame(rng, entity, with_cell_id=True).to_sql(entity, con, index=False)
    finally:
        con.close()
    return db


def _db_settings(src, **over):
    """Minimal settings for plot_data_from_db (defaults fill the rest)."""
    s = {
        "src": src,
        "database": "measurements.db",
        "table_names": "saliency_image_correlations",
        "data_column": "saliency_correlation",
        "grouping_column": "condition",
        "graph_name": "Fig1",
        "graph_type": "jitter",
        "cell_types": ["HeLa", "U2OS"],
        "cell_plate_metadata": [["r1"], ["r2"]],
        "representation": "well",
        "theme": "deep",
        "save": True,
        "verbose": False,
    }
    s.update(over)
    return s


# ---------------------------------------------------------------------------
# plot_data_from_db
# ---------------------------------------------------------------------------

def test_plot_data_from_db_saliency_table_plots_and_saves(tmp_path, rng):
    """The `saliency_image_correlations` short-circuit reads the table via
    _read_db, annotates conditions and renders + persists a spacrGraph."""
    from spacr.plot import plot_data_from_db

    src = tmp_path / "plate1"
    src.mkdir()
    _make_saliency_db(str(src), rng)

    settings = _db_settings(str(src))
    fig, results_df, df = plot_data_from_db(settings)

    assert isinstance(fig, matplotlib.figure.Figure)
    # annotate_conditions produced exactly the two host-cell conditions
    assert sorted(df["condition"].unique().tolist()) == ["HeLa", "U2OS"]
    assert df["host_cells"].isna().sum() == 0
    # prc was built from plate/row/column
    assert df["prc"].iloc[0].startswith("plate1_r")
    assert set(df["prc"]) == {f"plate1_{r}_{c}" for r in ROWS for c in COLUMNS}
    # the stats frame carries the between-group comparison
    assert not results_df.empty
    assert "HeLa vs U2OS (saliency_correlation)" in set(results_df["Comparison"])
    # settings snapshot + the spacrGraph outputs were written
    assert (src / "settings" / "Fig1_plot_settings_db.csv").is_file()
    out_dir = src / "results" / "Fig1"
    stem = "Fig1_saliency_correlation_condition_jitter"
    assert (out_dir / f"{stem}.pdf").is_file()
    saved_stats = pd.read_csv(out_dir / f"{stem}_stats.csv")
    assert len(saved_stats) == len(results_df)
    # settings['dst'] is set to <src>/results by the function
    assert settings["dst"] == os.path.join(str(src), "results")


def test_plot_data_from_db_recruitment_from_merged_tables(tmp_path, rng):
    """table_names as a list goes through io._read_and_merge_data and the
    'recruitment' data column is derived from pathogen/cytoplasm intensities."""
    from spacr.plot import plot_data_from_db

    src = tmp_path / "plate1"
    src.mkdir()
    _make_merge_db(str(src), rng)

    settings = _db_settings(
        str(src),
        table_names=["cell", "cytoplasm", "nucleus", "pathogen"],
        data_column="recruitment",
        channel_of_interest=1,
        graph_name="Recruit",
        graph_type="bar",
        save=False,
    )
    fig, results_df, df = plot_data_from_db(settings)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert "recruitment" in df.columns
    # recruitment == pathogen / cytoplasm for channel 1, recomputed here
    expected = (df["pathogen_channel_1_mean_intensity"]
                / df["cytoplasm_channel_1_mean_intensity"])
    np.testing.assert_allclose(df["recruitment"].to_numpy(),
                               expected.to_numpy(), rtol=1e-9)
    assert np.isfinite(df["recruitment"].to_numpy()).all()
    assert sorted(df["condition"].unique().tolist()) == ["HeLa", "U2OS"]
    assert "HeLa vs U2OS (recruitment)" in set(results_df["Comparison"])
    # merged frame keeps one row per object across the four joined tables
    assert len(df) == len(ROWS) * len(COLUMNS) * N_FIELDS
    # save=False -> spacrGraph wrote nothing, but plot_data_from_db still
    # created the (empty) output directory
    out_dir = src / "results" / "Recruit"
    assert out_dir.is_dir()
    assert list(out_dir.iterdir()) == []


def test_plot_data_from_db_missing_data_column_returns_none(tmp_path, rng):
    """An absent data_column short-circuits with None (after listing columns)."""
    from spacr.plot import plot_data_from_db

    src = tmp_path / "plate1"
    src.mkdir()
    _make_saliency_db(str(src), rng)

    out = plot_data_from_db(_db_settings(str(src), data_column="not_a_column"))
    assert out is None
    # nothing was plotted
    assert plt.get_fignums() == []
    # no graph output directory was created
    assert not (src / "results" / "Fig1").exists()


def test_plot_data_from_db_missing_grouping_column_returns_none(tmp_path, rng):
    """An absent grouping_column short-circuits with None."""
    from spacr.plot import plot_data_from_db

    src = tmp_path / "plate1"
    src.mkdir()
    _make_saliency_db(str(src), rng)

    out = plot_data_from_db(_db_settings(str(src), grouping_column="nope"))
    assert out is None
    assert not (src / "results" / "Fig1").exists()


def test_plot_data_from_db_rejects_non_str_non_list_src(tmp_path):
    """src must be a str or a list of str."""
    from spacr.plot import plot_data_from_db

    with pytest.raises(ValueError, match="src must be a string or a list"):
        plot_data_from_db(_db_settings(12345))


def test_plot_data_from_db_src_list_and_dropna_failures_are_caught(tmp_path, rng, capsys):
    """Two source dirs are concatenated, 'database' is broadcast to a list, and
    the three dropna() guards survive missing annotation columns."""
    from spacr.plot import plot_data_from_db

    src_a = tmp_path / "plateA"
    src_b = tmp_path / "plateB"
    src_a.mkdir()
    src_b.mkdir()
    _make_saliency_db(str(src_a), rng)
    _make_saliency_db(str(src_b), rng)

    settings = _db_settings(
        [str(src_a), str(src_b)],
        # *_types are None while *_plate_metadata is not -> annotate_conditions
        # never creates host_cells/pathogen/treatment, so every dropna raises.
        cell_types=None,
        cell_plate_metadata=[["r1"]],
        pathogen_types=None,
        pathogen_plate_metadata=[["c1"]],
        treatments=None,
        treatment_plate_metadata=[["c2"]],
        grouping_column="rowID",
        graph_name="Multi",
        save=False,
    )
    fig, results_df, df = plot_data_from_db(settings)

    err = capsys.readouterr().out
    assert "Could not drop NaN values from 'host_cell' column" in err
    assert "Could not drop NaN values from 'pathogen' column" in err
    assert "Could not drop NaN values from 'treatment' column" in err

    # 'database' was broadcast to one entry per src
    assert settings["database"] == ["measurements.db", "measurements.db"]
    assert settings["dst"] == os.path.join(str(src_a), "results")
    # both plates concatenated -> twice the rows of a single DB
    assert len(df) == 2 * len(ROWS) * len(COLUMNS) * N_FIELDS
    assert sorted(df["rowID"].unique().tolist()) == ROWS
    assert isinstance(fig, matplotlib.figure.Figure)
    assert not results_df.empty
    # save_settings appends '_list' when src is a list
    assert (src_a / "settings" / "Multi_plot_settings_db_list.csv").is_file()


# ---------------------------------------------------------------------------
# plot_data_from_csv
# ---------------------------------------------------------------------------

def _csv_frame(rng, n_wells=10, groups=("ctrl", "treat"), prc=None):
    rows = []
    for gi, group in enumerate(groups):
        for w in range(n_wells):
            rows.append({
                "prc": prc(gi, w) if prc else None,
                "group": group,
                "value": float(rng.normal(10.0 + 4.0 * gi, 1.0)),
            })
    df = pd.DataFrame(rows)
    if prc is None:
        df = df.drop(columns=["prc"])
    return df


def _csv_settings(src, **over):
    s = {
        "src": src,
        "data_column": "value",
        "grouping_column": "group",
        "graph_name": "CsvFig",
        "graph_type": "box",
        "save": True,
        "y_lim": None,
        "log_y": False,
        "log_x": False,
        "representation": "object",
        "theme": "deep",
        "remove_outliers": False,
        "verbose": False,
    }
    s.update(over)
    return s


def _csv_stem(settings):
    return (f"{settings['graph_name']}_{settings['data_column']}"
            f"_{settings['grouping_column']}_{settings['graph_type']}")


def test_plot_data_from_csv_adds_plate_and_common_columns(tmp_path, rng):
    """A CSV without plateID gets plateID='plate1' and common='spacr'."""
    from spacr.plot import plot_data_from_csv

    csv = tmp_path / "data.csv"
    _csv_frame(rng).to_csv(csv, index=False)

    settings = _csv_settings(str(csv))
    fig, results_df = plot_data_from_csv(settings)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert not results_df.empty
    assert "ctrl vs treat (value)" in set(results_df["Comparison"])

    out_dir = tmp_path / "results" / "CsvFig"
    saved = pd.read_csv(out_dir / f"{_csv_stem(settings)}_data.csv")
    assert set(saved["plateID"]) == {"plate1"}
    assert set(saved["common"]) == {"spacr"}
    assert len(saved) == 20


def test_plot_data_from_csv_two_sources_split_prc_and_keep_groups(tmp_path, rng):
    """Two CSVs are concatenated, 'prc' is split into plateID/rowID/columnID and
    keep_groups filters the grouping column."""
    from spacr.plot import plot_data_from_csv

    files = []
    for i in (1, 2):
        f = tmp_path / f"plate{i}.csv"
        _csv_frame(
            rng,
            groups=("ctrl", "treat", "drop_me"),
            prc=lambda gi, w, i=i: f"plate{i}_r{gi + 1}_c{w + 1}",
        ).to_csv(f, index=False)
        files.append(str(f))

    settings = _csv_settings(
        files, keep_groups=["ctrl", "treat"], representation="well",
        graph_name="CsvTwo", graph_type="bar")
    fig, results_df = plot_data_from_csv(settings)

    assert isinstance(fig, matplotlib.figure.Figure)
    out_dir = tmp_path / "results" / "CsvTwo"
    saved = pd.read_csv(out_dir / f"{_csv_stem(settings)}_data.csv")
    # 'drop_me' was filtered out by keep_groups
    assert set(saved["group"]) == {"ctrl", "treat"}
    # representation='well' aggregated by prc -> one row per prc/group pair,
    # for two plates x two kept groups x ten columns
    assert len(saved) == 2 * 2 * 10
    # the prc split produced real plate ids from BOTH files
    assert set(p.split("_")[0] for p in saved["prc"]) == {"plate1", "plate2"}


def test_plot_data_from_csv_prc_split_failure_is_caught(tmp_path, rng, capsys):
    """A prc value that does not split into exactly 3 parts is reported, not raised."""
    from spacr.plot import plot_data_from_csv

    csv = tmp_path / "bad_prc.csv"
    _csv_frame(rng, prc=lambda gi, w: f"plate1_r{gi + 1}_c{w + 1}_f1").to_csv(
        csv, index=False)

    settings = _csv_settings(str(csv), graph_name="CsvBad")
    fig, results_df = plot_data_from_csv(settings)

    assert "Could not split the prc column" in capsys.readouterr().out
    assert isinstance(fig, matplotlib.figure.Figure)
    out_dir = tmp_path / "results" / "CsvBad"
    saved = pd.read_csv(out_dir / f"{_csv_stem(settings)}_data.csv")
    # plateID stayed the synthesised value because the split failed
    assert set(saved["plateID"]) == {"plate1"}


def test_plot_data_from_csv_keep_groups_string_is_normalised(tmp_path, rng):
    """A string keep_groups is promoted to a list in-place and does NOT filter."""
    from spacr.plot import plot_data_from_csv

    csv = tmp_path / "data.csv"
    _csv_frame(rng).to_csv(csv, index=False)

    settings = _csv_settings(str(csv), keep_groups="ctrl", graph_name="CsvStr")
    fig, _ = plot_data_from_csv(settings)

    assert settings["keep_groups"] == ["ctrl"]
    out_dir = tmp_path / "results" / "CsvStr"
    saved = pd.read_csv(out_dir / f"{_csv_stem(settings)}_data.csv")
    # both groups survived - the string branch only normalises
    assert set(saved["group"]) == {"ctrl", "treat"}


def test_plot_data_from_csv_remove_outliers_and_verbose(tmp_path, rng, capsys):
    """remove_outliers strips per-group IQR outliers before plotting."""
    from spacr.plot import plot_data_from_csv

    df = _csv_frame(rng)
    df.loc[0, "value"] = 5000.0          # extreme outlier in group 'ctrl'
    csv = tmp_path / "outlier.csv"
    df.to_csv(csv, index=False)

    settings = _csv_settings(str(csv), remove_outliers=True, verbose=True,
                             graph_name="CsvOut")
    fig, _ = plot_data_from_csv(settings)

    assert isinstance(fig, matplotlib.figure.Figure)
    out_dir = tmp_path / "results" / "CsvOut"
    saved = pd.read_csv(out_dir / f"{_csv_stem(settings)}_data.csv")
    assert saved["value"].max() < 5000.0
    assert len(saved) < len(df)
    # verbose=True routed the frame through IPython display()
    assert capsys.readouterr().out != ""


def test_plot_data_from_csv_rejects_non_str_non_list_src():
    from spacr.plot import plot_data_from_csv

    with pytest.raises(ValueError, match="src must be a string or a list"):
        plot_data_from_csv(_csv_settings(3.14))


# ---------------------------------------------------------------------------
# plot_image_grid
# ---------------------------------------------------------------------------

def _write_png(path, rng, mode="RGB", size=16):
    from PIL import Image
    if mode == "RGB":
        arr = rng.integers(0, 255, size=(size, size, 3)).astype(np.uint8)
    else:
        arr = rng.integers(0, 255, size=(size, size)).astype(np.uint8)
    Image.fromarray(arr, mode=mode).save(path)
    return str(path)


def test_plot_image_grid_rgb_pads_grid_with_black_tiles(tmp_path, rng):
    """3 RGB images -> 2x2 grid; the 4th tile is a filled black placeholder."""
    from spacr.plot import plot_image_grid

    paths = [_write_png(tmp_path / f"img{i}.png", rng) for i in range(3)]
    fig = plot_image_grid(paths, percentiles=(2, 98))

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 4
    # every axis has exactly one image and no visible frame
    for ax in fig.axes:
        assert len(ax.images) == 1
        assert not ax.axison
    # the three real tiles are 16x16 RGB, normalised into 0..255 uint8
    real = [ax.images[0].get_array() for ax in fig.axes[:3]]
    for arr in real:
        assert arr.shape == (16, 16, 3)
        assert arr.dtype == np.uint8
        assert arr.min() == 0 and arr.max() == 255
    # the padding tile is the 1x3 black square
    pad = np.asarray(fig.axes[3].images[0].get_array())
    assert pad.shape == (1, 3)
    assert pad.max() == 0
    assert fig.get_facecolor()[:3] == (0.0, 0.0, 0.0)


def test_plot_image_grid_grayscale_normalises_to_full_range(tmp_path, rng):
    """Single-channel PNGs take the 2-D normalisation branch."""
    from spacr.plot import plot_image_grid

    paths = [_write_png(tmp_path / f"g{i}.png", rng, mode="L") for i in range(4)]
    fig = plot_image_grid(paths, percentiles=(5, 95))

    assert len(fig.axes) == 4            # perfect square -> no padding tiles
    for ax in fig.axes:
        arr = np.asarray(ax.images[0].get_array())
        assert arr.ndim == 2
        assert arr.dtype == np.uint8
        assert arr.min() == 0 and arr.max() == 255


def test_plot_image_grid_normalises_raw_ndarrays(tmp_path, rng, monkeypatch):
    """The nested normaliser has a non-PIL branch that returns a float array
    instead of re-wrapping in PIL. Force it by making Image.open hand back
    ndarrays, and check the tiles come through as float32 scaled to 0..1."""
    from PIL import Image as PILImage
    from spacr.plot import plot_image_grid

    paths = [_write_png(tmp_path / f"n{i}.png", rng) for i in range(2)]
    real_open = PILImage.open

    def _open_as_array(path, *args, **kwargs):
        with real_open(path, *args, **kwargs) as im:
            return np.asarray(im).astype(np.float64)

    monkeypatch.setattr(PILImage, "open", _open_as_array)
    fig = plot_image_grid(paths, percentiles=(2, 98))

    assert len(fig.axes) == 4
    for ax in fig.axes[:2]:
        arr = np.asarray(ax.images[0].get_array())
        assert arr.dtype == np.float32       # not the uint8 PIL round-trip
        assert arr.shape == (16, 16, 3)
        assert arr.min() == pytest.approx(0.0)
        assert arr.max() == pytest.approx(1.0)


def test_plot_image_grid_single_image(tmp_path, rng):
    """One image should still render a 1x1 grid."""
    from spacr.plot import plot_image_grid

    path = _write_png(tmp_path / "solo.png", rng)
    fig = plot_image_grid([path], percentiles=(2, 98))
    assert len(fig.axes) == 1
    assert fig.axes[0].images[0].get_array().shape == (16, 16, 3)


# ---------------------------------------------------------------------------
# plot_region
# ---------------------------------------------------------------------------

FOV_NAME = "plate1_r1_c1_f1"


def _make_region_src(tmp_path, rng, png_stem=FOV_NAME, act_stem=FOV_NAME,
                     n_crops=3):
    """Build a src folder with merged/, measurements/ and the PNG + activation
    assets plot_region resolves."""
    src = tmp_path / "region_src"
    (src / "merged").mkdir(parents=True)
    (src / "measurements").mkdir(parents=True)
    png_dir = src / "data" / "cell_png"
    png_dir.mkdir(parents=True)
    act_dir = src / "datasets" / "activation" / "saliency_image"
    act_dir.mkdir(parents=True)

    # merged FOV: 2 intensity planes + cell mask + nucleus mask
    stack = np.zeros((64, 64, 4), dtype=np.uint16)
    stack[..., 0] = rng.integers(100, 4000, (64, 64))
    stack[..., 1] = rng.integers(100, 4000, (64, 64))
    cell = np.zeros((64, 64), np.uint16)
    cell[8:28, 8:28] = 1
    cell[36:60, 36:60] = 2
    nucleus = np.zeros((64, 64), np.uint16)
    nucleus[14:22, 14:22] = 1
    nucleus[44:52, 44:52] = 2
    stack[..., 2] = cell
    stack[..., 3] = nucleus
    np.save(src / "merged" / f"{FOV_NAME}.npy", stack)

    png_paths, act_paths = [], []
    for i in range(n_crops):
        png_paths.append(_write_png(png_dir / f"{png_stem}_o{i + 1}.png", rng))
        act_paths.append(_write_png(act_dir / f"{act_stem}_o{i + 1}.png", rng))

    meta = {
        "plateID": ["plate1"] * n_crops,
        "rowID": ["r1"] * n_crops,
        "columnID": ["c1"] * n_crops,
        "fieldID": ["f1"] * n_crops,
        "cell_id": [f"o{i + 1}" for i in range(n_crops)],
    }
    con = sqlite3.connect(src / "measurements" / "measurements.db")
    try:
        pd.DataFrame({**meta, "png_path": png_paths}).to_sql(
            "png_list", con, index=False)
    finally:
        con.close()
    con = sqlite3.connect(src / "measurements" / "activation.db")
    try:
        pd.DataFrame({**meta, "png_path": act_paths}).to_sql(
            "saliency_image_list", con, index=False)
    finally:
        con.close()
    return src


@pytest.fixture
def small_overlay_canvas(monkeypatch):
    """Keep plot_region fast.

    plot_region hardcodes ``figuresize=10``, so the mask-overlay figure is
    40x10 inches and ``save_figure_as_pdf`` re-rasterises it at dpi=600
    (~8 s per call). The real product function still runs end to end - we only
    shrink the canvas it returns before plot_region saves it.
    """
    import spacr.plot as P
    real = P.plot_image_mask_overlay

    def _shrunk(*args, **kwargs):
        fig = real(*args, **kwargs)
        if fig is not None:
            fig.set_size_inches(4, 1)
        return fig

    monkeypatch.setattr(P, "plot_image_mask_overlay", _shrunk)
    return _shrunk


def _region_settings(src, **over):
    s = {
        "src": str(src),
        "name": f"{FOV_NAME}.npy",
        "channels": [0, 1],
        "cell_channel": 0,
        "nucleus_channel": 1,
        "pathogen_channel": None,
        "percentiles": (2, 98),
        "activation_mode": "saliency_image",
        "activation_db": "activation.db",
        "mode": "outlines",
        "export_tiffs": False,
    }
    s.update(over)
    return s


def test_plot_region_writes_three_pdfs(tmp_path, rng, small_overlay_canvas):
    """With PNG crops and activation maps present all three figures render and
    are saved as PDFs under <src>/results/<name>/."""
    from spacr.plot import plot_region

    src = _make_region_src(tmp_path, rng)
    fig_1, fig_2, fig_3 = plot_region(_region_settings(src))

    for fig in (fig_1, fig_2, fig_3):
        assert isinstance(fig, matplotlib.figure.Figure)
    # the mask overlay has one axis per channel plus the combined-objects axis
    assert len(fig_1.axes) == 3
    # 3 crops -> 2x2 grid for both image grids
    assert len(fig_2.axes) == 4
    assert len(fig_3.axes) == 4

    dst = src / "results" / FOV_NAME
    for suffix in ("mask_overlay", "png_grid", "activation_grid"):
        pdf = dst / f"{FOV_NAME}_{suffix}.pdf"
        assert pdf.is_file(), f"missing {pdf}"
        assert pdf.stat().st_size > 0
        assert pdf.read_bytes()[:4] == b"%PDF"


def test_plot_region_without_matching_crops_returns_none_figures(
        tmp_path, rng, capsys, small_overlay_canvas):
    """When neither the PNG nor the activation table has a path containing the
    FOV name, only the mask overlay is produced."""
    from spacr.plot import plot_region

    src = _make_region_src(tmp_path, rng, png_stem="other_fov",
                           act_stem="other_fov")
    fig_1, fig_2, fig_3 = plot_region(_region_settings(src))

    assert isinstance(fig_1, matplotlib.figure.Figure)
    assert fig_2 is None
    assert fig_3 is None
    out = capsys.readouterr().out
    assert "Could not find any cropped PNGs" in out
    assert "Could not find any activation maps" in out

    dst = src / "results" / FOV_NAME
    assert (dst / f"{FOV_NAME}_mask_overlay.pdf").is_file()
    assert not (dst / f"{FOV_NAME}_png_grid.pdf").exists()
    assert not (dst / f"{FOV_NAME}_activation_grid.pdf").exists()


def test_plot_region_masks_mode_and_tiff_export(tmp_path, rng, small_overlay_canvas):
    """mode='masks' + export_tiffs=True exercises the alternative overlay path
    and writes one TIFF per plane of the merged stack."""
    from spacr.plot import plot_region

    src = _make_region_src(tmp_path, rng)
    fig_1, fig_2, fig_3 = plot_region(
        _region_settings(src, mode="masks", export_tiffs=True))

    assert all(isinstance(f, matplotlib.figure.Figure)
               for f in (fig_1, fig_2, fig_3))
    tiff_dir = src / "results" / FOV_NAME / "tiff"
    tiffs = sorted(p.name for p in tiff_dir.glob("*.tiff"))
    assert tiffs == [f"{FOV_NAME}_channel_{i}.tiff" for i in range(4)]
