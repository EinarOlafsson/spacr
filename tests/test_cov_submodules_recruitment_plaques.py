"""CPU coverage for the recruitment / plaque slice of ``spacr.submodules``.

Covers :func:`spacr.submodules.analyze_recruitment` (source-path
normalisation, condition annotation, overlay plotting, object filters,
recruitment ratios, per-well grouping and CSV export) and
:func:`spacr.submodules.analyze_plaques` (mask folder selection, per-image
region statistics and the sqlite export).

Everything heavy is injected away: the measurements DB read is replaced by
a synthetic merged DataFrame, the plotting entry points by recorders and
Cellpose segmentation / HuggingFace model downloads by no-ops. The real
``spacr.utils`` filtering / grouping helpers and the real
``spacr.io._results_to_csv`` writer still run, so the assertions are about
genuine numbers and genuine files on disk.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate across tests."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


CHANNELS = (0, 1, 2, 3)


def _merged_recruitment_df(wells=(("r1", "c1"), ("r4", "c4")), n_cells=4,
                           cell_area=1000.0, extra_wells=()):
    """Build a merged cell/nucleus/pathogen/cytoplasm table.

    Mirrors the shape ``spacr.io._read_and_merge_data`` returns: one row per
    cell, ``prcfo`` index, plate metadata columns, ``cells_per_well`` and one
    intensity family per channel for every compartment.
    """
    rows = []
    for rid, cid in list(wells) + list(extra_wells):
        for i in range(n_cells):
            label = f"o{i + 1}"
            prc = f"plate1_{rid}_{cid}"
            prcf = f"{prc}_1"
            rec = {
                "prcfo": f"{prcf}_{label}",
                "plateID": "plate1",
                "rowID": rid,
                "columnID": cid,
                "fieldID": "1",
                "object_label": label,
                "prc": prc,
                "prcf": prcf,
                "file_name": f"{prcf}",
                "cells_per_well": float(n_cells),
                "cell_area": cell_area + i,
                "nucleus_area": 300.0 + i,
                "pathogen_area": 120.0 + i,
                "cytoplasm_area": 700.0 + i,
            }
            for chan in CHANNELS:
                # Deterministic, distinct per compartment so ratios are exact.
                rec[f"cell_channel_{chan}_mean_intensity"] = 100.0 + chan
                rec[f"nucleus_channel_{chan}_mean_intensity"] = 200.0 + chan
                rec[f"cytoplasm_channel_{chan}_mean_intensity"] = 50.0 + chan
                rec[f"pathogen_channel_{chan}_mean_intensity"] = 400.0 + chan
                rec[f"pathogen_channel_{chan}_percentile_75"] = 500.0 + chan
                rec[f"pathogen_channel_{chan}_outside_mean"] = 10.0 + chan
                rec[f"pathogen_channel_{chan}_outside_75_percentile"] = 20.0 + chan
                rec[f"pathogen_channel_{chan}_periphery_mean"] = 30.0 + chan
                rec[f"cell_channel_{chan}_percentile_95"] = 1000.0 + 10 * i
            rows.append(rec)
    df = pd.DataFrame(rows).set_index("prcfo")
    return df


class _Recorder:
    """Callable that records every invocation (and can be made to explode)."""

    def __init__(self, exc=None, result=None):
        self.calls = []
        self.exc = exc
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.exc is not None:
            raise self.exc
        return self.result


@pytest.fixture
def recruitment_env(monkeypatch, tmp_path):
    """Patch the heavy collaborators of analyze_recruitment.

    Returns a dict of the recorders so tests can assert on the calls, plus a
    mutable ``df`` slot holding the DataFrame ``_read_and_merge_data`` hands
    back.
    """
    import spacr.io as SIO
    import spacr.plot as SPLT

    state = {"df": _merged_recruitment_df(), "read_calls": []}

    def fake_read_and_merge_data(locs, tables, **kwargs):
        state["read_calls"].append({"locs": list(locs), "tables": list(tables),
                                    "kwargs": dict(kwargs)})
        return state["df"].copy(), []

    overlay = _Recorder()
    plot_controls = _Recorder()
    plot_recruitment = _Recorder()

    monkeypatch.setattr(SIO, "_read_and_merge_data", fake_read_and_merge_data)
    monkeypatch.setattr(SPLT, "plot_image_mask_overlay", overlay)
    monkeypatch.setattr(SPLT, "_plot_controls", plot_controls)
    monkeypatch.setattr(SPLT, "_plot_recruitment", plot_recruitment)

    state.update({"overlay": overlay, "plot_controls": plot_controls,
                  "plot_recruitment": plot_recruitment})
    return state


def _base_settings(src, **over):
    settings = {
        "src": str(src),
        "plot": False,
        "plot_control": False,
        "target_intensity_min": 0,
        "figuresize": 4,
    }
    settings.update(over)
    return settings


# ===========================================================================
# analyze_recruitment — happy path
# ===========================================================================

def test_analyze_recruitment_writes_csvs_and_exact_ratios(tmp_path, recruitment_env):
    """Full pass: ratios are the literal pathogen/cytoplasm quotient, the
    per-well table is the mean over the wells' cells, and both CSVs land in
    ``<src>/results``."""
    from spacr.submodules import analyze_recruitment

    src = tmp_path / "plate"
    src.mkdir()
    cells, wells = analyze_recruitment(_base_settings(src))

    # 2 wells x 4 cells, nothing filtered out.
    assert len(cells) == 8
    assert len(wells) == 2

    # channel_of_interest defaults to 2 -> pathogen(402)/cytoplasm(52).
    assert np.allclose(cells["recruitment"], 402.0 / 52.0)
    # _calculate_recruitment ran for every channel in channel_dims; the last
    # channel (3) wins the shared column names.
    assert np.allclose(cells["pathogen_cytoplasm_mean_mean"], 403.0 / 53.0)
    assert np.allclose(cells["pathogen_cell_q75_mean"], 503.0 / 103.0)
    assert np.allclose(cells["pathogen_outside_nucleus_mean_mean"], 13.0 / 203.0)
    assert np.allclose(cells["pathogen_periphery_cytoplasm_mean_mean"], 33.0 / 53.0)

    # annotate_conditions mapped column -> pathogen and row -> treatment.
    assert set(cells["condition"]) == {"HeLa_pathogen_1_cm",
                                       "HeLa_pathogen_2_lovastatin"}

    results = src / "results"
    assert (results / "cells.csv").exists()
    assert (results / "wells.csv").exists()
    on_disk = pd.read_csv(results / "wells.csv")
    assert len(on_disk) == 2
    # Numeric columns were averaged per well.
    assert np.allclose(wells["cells_per_well"], 4.0)

    # The DB location analyze_recruitment asked for.
    assert recruitment_env["read_calls"][0]["locs"] == [
        str(src) + "/measurements/measurements.db"]
    assert recruitment_env["read_calls"][0]["tables"] == [
        "cell", "nucleus", "pathogen", "cytoplasm"]

    # _plot_recruitment is unconditional: once by PV, once by well.
    assert len(recruitment_env["plot_recruitment"].calls) == 2
    assert recruitment_env["plot_recruitment"].calls[0][0][1] == "by PV"
    assert recruitment_env["plot_recruitment"].calls[1][0][1] == "by well"
    # plot / plot_control were off.
    assert recruitment_env["overlay"].calls == []
    assert recruitment_env["plot_controls"].calls == []

    # save_settings snapshotted the resolved settings.
    assert (src / "settings" / "recruitment.csv").exists()


def test_analyze_recruitment_drops_unannotated_wells(tmp_path, recruitment_env):
    """A well that matches none of the cell/pathogen/treatment locations gets
    a NA condition and is dropped before anything is measured."""
    from spacr.submodules import analyze_recruitment

    src = tmp_path / "plate"
    src.mkdir()
    recruitment_env["df"] = _merged_recruitment_df(extra_wells=(("r9", "c9"),))

    settings = _base_settings(
        src,
        cell_types=["HeLa"],
        cell_plate_metadata=[["c1", "c4"]],
    )
    cells, wells = analyze_recruitment(settings)

    assert len(cells) == 8, "the r9/c9 well should have been dropped"
    assert "plate1_r9_c9" not in set(cells["prc"])
    assert len(wells) == 2


# ===========================================================================
# analyze_recruitment — src normalisation (lines 769-776)
# ===========================================================================

def test_analyze_recruitment_moves_loose_db_into_measurements(tmp_path,
                                                              recruitment_env):
    """``src`` given as ``<dir>/measurements.db`` is rewritten to ``<dir>``
    and the DB is physically moved into ``<dir>/measurements/``."""
    from spacr.submodules import analyze_recruitment

    src = tmp_path / "plate"
    src.mkdir()
    loose_db = src / "measurements.db"
    with sqlite3.connect(loose_db) as con:
        con.execute("CREATE TABLE t (a INTEGER)")

    settings = _base_settings(loose_db)
    analyze_recruitment(settings)

    assert settings["src"] == str(src)
    assert not loose_db.exists()
    moved = src / "measurements" / "measurements.db"
    assert moved.exists()
    with sqlite3.connect(moved) as con:
        names = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "t" in names
    assert recruitment_env["read_calls"][0]["locs"] == [str(moved)]


def test_analyze_recruitment_existing_measurements_dir_is_not_clobbered(
        tmp_path, recruitment_env):
    """When ``<dir>/measurements`` already exists the loose DB is left alone —
    the makedirs/move branch is skipped."""
    from spacr.submodules import analyze_recruitment

    src = tmp_path / "plate"
    (src / "measurements").mkdir(parents=True)
    keeper = src / "measurements" / "measurements.db"
    keeper.write_bytes(b"already here")
    loose_db = src / "measurements.db"
    loose_db.write_bytes(b"loose")

    settings = _base_settings(loose_db)
    analyze_recruitment(settings)

    assert settings["src"] == str(src)
    assert loose_db.exists(), "existing measurements dir must not trigger a move"
    assert keeper.read_bytes() == b"already here"


@pytest.mark.xfail(strict=True, reason=(
    "BUG: analyze_recruitment strips only the filename from a "
    "<src>/measurements/measurements.db path, leaving src pointing at the "
    "measurements folder, then reads <src>/measurements/measurements.db "
    "again -> a doubled 'measurements/measurements' path that never exists."))
def test_analyze_recruitment_db_inside_measurements_dir(tmp_path,
                                                        recruitment_env):
    """Passing the canonical ``<plate>/measurements/measurements.db`` path
    should read that very file."""
    from spacr.submodules import analyze_recruitment

    src = tmp_path / "plate"
    (src / "measurements").mkdir(parents=True)
    db = src / "measurements" / "measurements.db"
    db.write_bytes(b"db")

    analyze_recruitment(_base_settings(db))

    assert recruitment_env["read_calls"][0]["locs"] == [str(db)]


# ===========================================================================
# analyze_recruitment — size-range defaulting (lines 812-818)
# ===========================================================================

def test_analyze_recruitment_none_size_ranges_become_unbounded(tmp_path,
                                                               recruitment_env):
    """``None`` size ranges are replaced in-place with ``[0, 10**100]`` and
    therefore filter nothing out."""
    from spacr.submodules import analyze_recruitment

    src = tmp_path / "plate"
    src.mkdir()
    settings = _base_settings(
        src,
        cell_size_range=None,
        nucleus_size_range=None,
        pathogen_size_range=None,
    )
    cells, _ = analyze_recruitment(settings)

    unbounded = [0, 10 ** 100]
    assert settings["cell_size_range"] == unbounded
    assert settings["nucleus_size_range"] == unbounded
    assert settings["pathogen_size_range"] == unbounded
    assert len(cells) == 8


# ===========================================================================
# analyze_recruitment — overlay plotting (lines 820-837)
# ===========================================================================

def test_analyze_recruitment_plots_first_n_merged_files(tmp_path,
                                                        recruitment_env):
    """With ``plot=True`` the first ``plot_nr + 1`` merged files get an
    overlay, using the configured channel dims."""
    from spacr.submodules import analyze_recruitment

    src = tmp_path / "plate"
    merged = src / "merged"
    merged.mkdir(parents=True)
    made = []
    for i in range(5):
        p = merged / f"field_{i}.npy"
        p.write_bytes(b"")
        made.append(str(p))

    analyze_recruitment(_base_settings(src, plot=True, plot_nr=1))

    overlay = recruitment_env["overlay"]
    assert len(overlay.calls) == 2, "plot_nr=1 -> indices 0 and 1 only"
    called_paths = [c[0][0] for c in overlay.calls]
    assert set(called_paths).issubset(set(made))
    args, kwargs = overlay.calls[0]
    # (file_path, channel_dims, cell_chann_dim, nucleus_chann_dim, pathogen_chann_dim)
    assert args[1] == [0, 1, 2, 3]
    assert args[2:] == (3, 0, 2)
    assert kwargs == {"figuresize": 10, "normalize": True, "thickness": 3,
                      "save_pdf": True}


def test_analyze_recruitment_missing_merged_dir_skips_plotting(tmp_path,
                                                               recruitment_env):
    """``plot=True`` without a ``merged`` folder is a silent no-op, not a crash."""
    from spacr.submodules import analyze_recruitment

    src = tmp_path / "plate"
    src.mkdir()
    cells, _ = analyze_recruitment(_base_settings(src, plot=True))

    assert recruitment_env["overlay"].calls == []
    assert len(cells) == 8


def test_analyze_recruitment_overlay_failure_is_swallowed(tmp_path, monkeypatch,
                                                          recruitment_env,
                                                          capsys):
    """A raising overlay plotter is caught and reported; the analysis still
    completes and writes its CSVs."""
    import spacr.plot as SPLT
    from spacr.submodules import analyze_recruitment

    boom = _Recorder(exc=RuntimeError("no display for you"))
    monkeypatch.setattr(SPLT, "plot_image_mask_overlay", boom)

    src = tmp_path / "plate"
    merged = src / "merged"
    merged.mkdir(parents=True)
    (merged / "field_0.npy").write_bytes(b"")

    cells, wells = analyze_recruitment(_base_settings(src, plot=True))

    out = capsys.readouterr().out
    assert "Failed to plot images with outlines" in out
    assert "no display for you" in out
    assert len(boom.calls) == 1
    assert len(cells) == 8
    assert (src / "results" / "cells.csv").exists()


# ===========================================================================
# analyze_recruitment — object filters (lines 839-847)
# ===========================================================================

def test_analyze_recruitment_cell_size_filter_drops_rows(tmp_path,
                                                         recruitment_env):
    """The cell area filter is a strict open interval on ``cell_area``."""
    from spacr.submodules import analyze_recruitment

    src = tmp_path / "plate"
    src.mkdir()
    # areas are 1000, 1001, 1002, 1003 per well -> keep only 1002.
    settings = _base_settings(src, cell_size_range=[1001, 1003])
    cells, _ = analyze_recruitment(settings)

    assert len(cells) == 2  # one surviving cell in each of the two wells
    assert set(np.round(cells["cell_area"])) == {1002}


def test_analyze_recruitment_target_intensity_min_filter(tmp_path,
                                                         recruitment_env,
                                                         capsys):
    """A non-zero ``target_intensity_min`` filters on
    ``cell_channel_<coi>_percentile_95``."""
    from spacr.submodules import analyze_recruitment

    src = tmp_path / "plate"
    src.mkdir()
    # percentile_95 values are 1000, 1010, 1020, 1030 per well.
    cells, _ = analyze_recruitment(
        _base_settings(src, target_intensity_min=1015))

    assert len(cells) == 4
    assert cells["cell_channel_2_percentile_95"].min() > 1015
    assert "After channel 2 filtration" in capsys.readouterr().out


def test_analyze_recruitment_none_chann_dims_skip_all_filters(tmp_path,
                                                              recruitment_env):
    """With every ``*_chann_dim`` set to None no object filter runs, so rows
    far outside the size ranges survive."""
    from spacr.submodules import analyze_recruitment

    src = tmp_path / "plate"
    src.mkdir()
    settings = _base_settings(
        src,
        cell_chann_dim=None,
        nucleus_chann_dim=None,
        pathogen_chann_dim=None,
        cell_size_range=[10 ** 6, 10 ** 7],      # would remove everything
        nucleus_size_range=[10 ** 6, 10 ** 7],
        pathogen_size_range=[10 ** 6, 10 ** 7],
    )
    cells, _ = analyze_recruitment(settings)

    assert len(cells) == 8


# ===========================================================================
# analyze_recruitment — well grouping / controls (lines 855-869)
# ===========================================================================

def test_analyze_recruitment_cells_per_well_threshold(tmp_path,
                                                      recruitment_env):
    """Wells below ``cells_per_well`` are dropped, and their cells with them."""
    from spacr.submodules import analyze_recruitment

    src = tmp_path / "plate"
    src.mkdir()
    small = _merged_recruitment_df(wells=(("r1", "c1"),), n_cells=2)
    big = _merged_recruitment_df(wells=(("r4", "c4"),), n_cells=5)
    recruitment_env["df"] = pd.concat([small, big])

    cells, wells = analyze_recruitment(_base_settings(src, cells_per_well=3))

    assert len(wells) == 1
    assert set(cells["prc"]) == {"plate1_r4_c4"}
    assert len(cells) == 5


def test_analyze_recruitment_plot_control_invoked(tmp_path, recruitment_env):
    """``plot_control=True`` forwards the mask-channel list and the channel of
    interest to ``_plot_controls``."""
    from spacr.submodules import analyze_recruitment

    src = tmp_path / "plate"
    src.mkdir()
    analyze_recruitment(_base_settings(src, plot_control=True))

    controls = recruitment_env["plot_controls"]
    assert len(controls.calls) == 1
    args, kwargs = controls.calls[0]
    # mask_chans = [nucleus_chann_dim, pathogen_chann_dim, cell_chann_dim]
    assert args[1] == [0, 2, 3]
    assert args[2] == 2
    assert kwargs == {"figuresize": 5}


# ===========================================================================
# analyze_plaques
# ===========================================================================

def _write_label_tif(path: Path, blobs=((10, 10, 4), (40, 40, 6))):
    """Write a uint16 label tif with circular blobs of the given radii."""
    tifffile = pytest.importorskip("tifffile")
    img = np.zeros((64, 64), dtype=np.uint16)
    yy, xx = np.mgrid[:64, :64]
    for idx, (cy, cx, r) in enumerate(blobs, start=1):
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = idx
    tifffile.imwrite(str(path), img)
    return img


@pytest.fixture
def plaque_env(monkeypatch):
    """Neutralise the model download + Cellpose segmentation."""
    import spacr.utils as SU
    import spacr.spacr_cellpose as SCP

    download = _Recorder(result="/nowhere")
    finetune = _Recorder()
    monkeypatch.setattr(SU, "download_models", download)
    monkeypatch.setattr(SCP, "identify_masks_finetune", finetune)
    return {"download": download, "finetune": finetune}


def test_analyze_plaques_masks_false_reads_existing_masks(tmp_path, plaque_env):
    """With ``masks=False`` no segmentation runs; the pre-existing tifs under
    ``<src>/masks`` are measured straight into plaques_analysis.db."""
    from spacr.submodules import analyze_plaques

    src = tmp_path / "plaques"
    masks = src / "masks"
    masks.mkdir(parents=True)
    img_a = _write_label_tif(masks / "a.tif")
    _write_label_tif(masks / "b.tif", blobs=((20, 20, 5),))
    # Decoys that must be ignored: wrong extension and a directory.
    (masks / "notes.txt").write_text("ignore me")
    (masks / "sub.tif").mkdir()

    settings = {"src": str(src), "masks": False}
    assert analyze_plaques(settings) is None

    assert plaque_env["finetune"].calls == [], "masks=False must not segment"
    assert len(plaque_env["download"].calls) == 1
    assert settings["dst"] == str(masks)
    assert settings["custom_model"].endswith(
        "toxo_plaque_cyto_e25000_X1120_Y1120.CP_model")

    db = masks / "plaques_analysis.db"
    assert db.exists()
    with sqlite3.connect(db) as con:
        summary = pd.read_sql("SELECT * FROM summary", con)
        stats = pd.read_sql("SELECT * FROM stats", con)
        details = pd.read_sql("SELECT * FROM details", con)

    assert set(summary["file"]) == {"a.tif", "b.tif"}
    a_row = summary.set_index("file").loc["a.tif"]
    assert a_row["object_count"] == 2
    expected_sizes = sorted(np.bincount(img_a.ravel())[1:].tolist())
    assert a_row["average_size"] == pytest.approx(np.mean(expected_sizes))

    b_stats = stats.set_index("file").loc["b.tif"]
    assert b_stats["plaque_count"] == 1
    # A single region has zero spread.
    assert b_stats["std_dev_size"] == pytest.approx(0.0)

    # One details row per plaque: 2 in a.tif + 1 in b.tif.
    assert len(details) == 3
    assert sorted(details[details["file"] == "a.tif"]["plaque_size"]) == \
        expected_sizes


def test_analyze_plaques_masks_true_runs_segmentation_first(tmp_path,
                                                            plaque_env,
                                                            monkeypatch):
    """``masks=True`` calls identify_masks_finetune with the resolved settings
    and then analyses whatever it produced in ``<src>/masks``."""
    import spacr.spacr_cellpose as SCP
    from spacr.submodules import analyze_plaques

    src = tmp_path / "plaques"
    src.mkdir()
    seen = {}

    def fake_finetune(settings):
        seen.update(settings)
        dst = Path(settings["dst"])
        dst.mkdir(parents=True, exist_ok=True)
        _write_label_tif(dst / "seg.tif", blobs=((15, 15, 3), (45, 45, 3),
                                                 (15, 45, 3)))

    monkeypatch.setattr(SCP, "identify_masks_finetune", fake_finetune)

    analyze_plaques({"src": str(src), "masks": True})

    assert seen["dst"] == str(src / "masks")
    assert seen["diameter"] == 30          # get_analyze_plaque_settings default
    assert seen["custom_model"].endswith(".CP_model")

    with sqlite3.connect(src / "masks" / "plaques_analysis.db") as con:
        stats = pd.read_sql("SELECT * FROM stats", con)
    assert len(stats) == 1
    assert stats.loc[0, "plaque_count"] == 3
    assert stats.loc[0, "average_size"] > 0


def test_analyze_plaques_replaces_previous_tables_and_counts(tmp_path, plaque_env):
    """Re-running over the same folder overwrites the tables rather than
    appending to them."""
    from spacr.submodules import analyze_plaques

    src = tmp_path / "plaques"
    masks = src / "masks"
    masks.mkdir(parents=True)
    _write_label_tif(masks / "a.tif")

    analyze_plaques({"src": str(src), "masks": False})
    analyze_plaques({"src": str(src), "masks": False})

    with sqlite3.connect(masks / "plaques_analysis.db") as con:
        summary = pd.read_sql("SELECT * FROM summary", con)
        details = pd.read_sql("SELECT * FROM details", con)
    assert len(summary) == 1
    assert len(details) == 2


# ===========================================================================
# count_phenotypes
# ===========================================================================

def _png_list_db(db_path: Path):
    """png_list table with 4 plate/row/column groups and known value counts."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("p1", "r1", "c1", 1), ("p1", "r1", "c1", 1), ("p1", "r1", "c1", 2),
        ("p1", "r1", "c2", 2), ("p1", "r1", "c2", 2),
        ("p1", "r2", "c1", 1),
        ("p1", "r2", "c2", 1), ("p1", "r2", "c2", 2),
    ]
    df = pd.DataFrame(rows, columns=["plateID", "rowID", "columnID", "value"])
    df["png_path"] = [f"/x/{i}.png" for i in range(len(df))]
    with sqlite3.connect(db_path) as con:
        df.to_sql("png_list", con, index=False)
    return df


def test_count_phenotypes_pivots_counts_per_well(tmp_path, monkeypatch):
    """count_phenotypes appends measurements/measurements.db to a folder src
    and writes one row per plate/row/column with a column per unique value."""
    import spacr.submodules as SUB

    shown = _Recorder()
    monkeypatch.setattr(SUB, "display", shown, raising=False)
    monkeypatch.chdir(tmp_path)          # the stray os.makedirs('src/results')

    _png_list_db(tmp_path / "measurements" / "measurements.db")

    settings = {"src": str(tmp_path), "annotation_column": "value"}
    assert SUB.count_phenotypes(settings) is None
    assert settings["src"] == os.path.join(str(tmp_path),
                                           "measurements/measurements.db")

    out_csv = tmp_path / "measurements" / "phenotype_counts.csv"
    assert out_csv.exists()
    counts = pd.read_csv(out_csv, index_col=0)
    assert list(counts.columns) == ["value_1", "value_2"]
    assert set(counts.index) == {"p1_r1_c1", "p1_r1_c2", "p1_r2_c1", "p1_r2_c2"}
    assert counts.loc["p1_r1_c1"].tolist() == [2, 1]
    assert counts.loc["p1_r1_c2"].tolist() == [0, 2]
    assert counts.loc["p1_r2_c1"].tolist() == [1, 0]
    assert counts.loc["p1_r2_c2"].tolist() == [1, 1]

    # display() got the per-well nunique table.
    assert len(shown.calls) == 1
    grouped = shown.calls[0][0][0]
    assert sorted(grouped["unique_count"].tolist()) == [1, 1, 2, 2]


@pytest.mark.xfail(strict=True, reason=(
    "BUG: count_phenotypes runs os.makedirs(os.path.join('src','results')) on "
    "a hard-coded relative path, creating a stray ./src/results directory in "
    "the caller's cwd; the value is immediately overwritten so the makedirs "
    "is dead code."))
def test_count_phenotypes_does_not_litter_cwd(tmp_path, monkeypatch):
    """An explicit measurements.db path should not create anything outside the
    measurements folder."""
    import spacr.submodules as SUB

    monkeypatch.setattr(SUB, "display", _Recorder(), raising=False)
    monkeypatch.chdir(tmp_path)

    db = tmp_path / "measurements" / "measurements.db"
    _png_list_db(db)
    SUB.count_phenotypes({"src": str(db), "annotation_column": "value"})

    assert (tmp_path / "measurements" / "phenotype_counts.csv").exists()
    assert not (tmp_path / "src").exists()
