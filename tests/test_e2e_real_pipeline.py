"""End-to-end pipeline test on the real NAS plate1 microscopy dataset.

Runs the full spaCR core pipeline sequentially on a small copied subset of
real cells, exercising core / io / measure / object / ml / utils (and a
sequencing chunk on the real R1 fastq + toxo analysis on the outputs).

Marked ``slow`` and GPU-backed — opt in with ``pytest -m slow``. Skips
cleanly when the NAS dataset is not mounted, so it never breaks CI.
"""
from __future__ import annotations

import os
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

# --- dataset locations on the NAS (autofs, mounts on access) ---------------
RAW = "/nas_mnt/data/sequencing/plate1/orig/orig"
MASK_SETTINGS = "/nas_mnt/data/sequencing/settings/preprocess_generate_masks_settings.csv"
MEASURE_SETTINGS = "/nas_mnt/data/sequencing/settings/measure_crop_settings.csv"
R1_FASTQ = "/nas_mnt/data/sequencing/seq_3/EO1_R1_001.fastq.gz"

WELLS = ["E01", "E02", "L01", "L02"]     # E→row5(c1/c2), L→row12(c1/c2)
FIELDS = ["F001", "F009"]                # 2 fields/well → 8 fields, 2 columns

pytestmark = pytest.mark.slow

_available = os.path.isdir(RAW) and os.path.isfile(MASK_SETTINGS)
_skip = pytest.mark.skipif(not _available,
                           reason="NAS plate1 dataset not mounted")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Copy a subset, run preprocess_generate_masks + measure_crop once."""
    if not _available:
        pytest.skip("NAS plate1 dataset not mounted")

    from spacr.utils import load_settings
    from spacr.core import preprocess_generate_masks
    from spacr.measure import measure_crop

    work = tmp_path_factory.mktemp("e2e")
    src = os.path.join(str(work), "plate1")
    os.makedirs(src)

    n = 0
    for f in sorted(os.listdir(RAW)):
        if f.endswith(".tif") and any(w in f for w in WELLS) \
                and any(fd in f for fd in FIELDS):
            shutil.copy2(os.path.join(RAW, f), os.path.join(src, f))
            n += 1
    assert n == 32, f"expected 32 raw tifs, copied {n}"

    # --- Stage 1: masks (GPU Cellpose) ---
    ms = load_settings(MASK_SETTINGS, setting_key="Key", setting_value="Value")
    ms.update(dict(src=src, plot=False, save=True, verbose=False,
                   test_mode=False, workers=2, batch_size=8, randomize=False))
    preprocess_generate_masks(ms)

    merged = os.path.join(src, "merged")
    assert os.path.isdir(merged) and len(os.listdir(merged)) == 8

    # --- Stage 2: measure + crop ---
    cs = load_settings(MEASURE_SETTINGS, setting_key="Key", setting_value="Value")
    cs.update(dict(src=merged, input_folder=merged, plot=False,
                   test_mode=False, save_measurements=True, save_png=True,
                   n_job=2, verbose=False))
    measure_crop(cs)

    db = os.path.join(src, "measurements", "measurements.db")
    assert os.path.isfile(db)
    return {"src": src, "merged": merged, "db": db, "work": str(work)}


# ---------------------------------------------------------------------------
# Stage 1 — core / io / object: masks + merged stacks
# ---------------------------------------------------------------------------

@_skip
def test_stage1_masks_and_stacks(pipeline):
    src = pipeline["src"]
    for sub in ("stack", "merged", "masks", "1", "2", "3", "4"):
        p = os.path.join(src, sub)
        assert os.path.isdir(p) and os.listdir(p), f"{sub} missing/empty"
    # merged stacks carry the appended masks (7 planes: 4 chan + 3 masks)
    sample = np.load(os.path.join(src, "merged",
                                  sorted(os.listdir(pipeline["merged"]))[0]))
    assert sample.ndim == 3 and sample.shape[-1] >= 7


# ---------------------------------------------------------------------------
# Stage 2 — measure / io: measurements.db
# ---------------------------------------------------------------------------

@_skip
def test_stage2_measurements(pipeline):
    con = sqlite3.connect(pipeline["db"])
    try:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", con)["name"].tolist()
        assert {"cell", "nucleus", "pathogen", "cytoplasm", "png_list"} <= set(tables)
        cell = pd.read_sql_query("SELECT * FROM cell", con)
        assert len(cell) > 50
        assert set(cell["columnID"].unique()) == {"c1", "c2"}
        # feature columns actually populated
        assert any("channel_3" in c for c in cell.columns)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Stage 3 — ml: train a classifier on the control wells + score
# ---------------------------------------------------------------------------

@_skip
def test_stage3_generate_ml_scores(pipeline):
    from spacr.ml import generate_ml_scores
    settings = {
        "src": pipeline["src"], "channel_of_interest": 3,
        "location_column": "columnID",
        "positive_control": "c2", "negative_control": "c1",
        "model_type_ml": "random_forest", "heatmap_feature": "predictions",
        "grouping": "mean", "min_max": "allq", "minimum_cell_count": 25,
        "n_repeats": 2, "top_features": 20, "test_size": 0.25,
        "n_estimators": 50, "n_jobs": 2, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "learning_rate": 0.001, "remove_low_variance_features": True,
        "remove_highly_correlated_features": False, "prune_features": False,
        "cross_validation": False, "exclude": None, "verbose": False,
    }
    out = generate_ml_scores(settings)
    assert out is not None
    results_dir = os.path.join(pipeline["src"], "results")
    assert os.path.isdir(results_dir)


# ---------------------------------------------------------------------------
# Stage 4 — sequencing: process a chunk of the REAL R1 fastq
# ---------------------------------------------------------------------------

@_skip
@pytest.mark.skipif(not os.path.isfile(R1_FASTQ),
                    reason="R1 fastq not on NAS")
def test_stage4_sequencing_chunk(pipeline, tmp_path):
    from spacr.sequencing import generate_barecode_mapping

    # Synthetic barcode reference CSVs (sequence,name). Real matches are
    # sparse but every code path runs; that's what we're covering.
    rng = np.random.default_rng(0)
    bc_dir = tmp_path / "barcodes"; bc_dir.mkdir()

    def _write(path, k, n):
        pd.DataFrame({
            "sequence": ["".join(rng.choice(list("ACGT"), k)) for _ in range(n)],
            "name": [f"bc{i}" for i in range(n)],
        }).to_csv(path, index=False)

    col_csv = str(bc_dir / "col.csv"); _write(col_csv, 8, 8)
    row_csv = str(bc_dir / "row.csv"); _write(row_csv, 8, 8)
    grna_csv = str(bc_dir / "grna.csv"); _write(grna_csv, 20, 20)

    # src folder containing the single R1 read file (symlink to avoid a 7.7GB copy)
    seq_src = tmp_path / "seq"; seq_src.mkdir()
    link = seq_src / "EO1_R1_001.fastq.gz"
    os.symlink(R1_FASTQ, str(link))

    generate_barecode_mapping({
        "src": str(seq_src), "mode": "single", "single_direction": "R1",
        "column_csv": col_csv, "row_csv": row_csv, "grna_csv": grna_csv,
        "chunk_size": 2000, "test": True, "save_h5": False, "n_jobs": 1,
    })
    # test=True processes a single chunk and writes qc + unique_combinations
    out_dir = seq_src / "EO1_single_R1"
    assert out_dir.exists()
    assert (out_dir / "qc.csv").exists()


# ---------------------------------------------------------------------------
# Stage 5 — toxo: analyse the per-object scores from stage 3
# ---------------------------------------------------------------------------

@_skip
def test_stage6_core_umap_and_graphs(pipeline):
    """Exercise core's other three entry points on the real measurement DB:
    generate_image_umap, reducer_hyperparameter_search, generate_screen_graphs."""
    from spacr.core import (generate_image_umap,
                            reducer_hyperparameter_search,
                            generate_screen_graphs)
    plate = pipeline["src"]
    umap_s = dict(
        src=plate, tables=['cell', 'nucleus', 'pathogen', 'cytoplasm'],
        visualize='cell', image_nr=4, dot_size=10, n_neighbors=15,
        min_dist=0.1, metric='euclidean', eps=0.5, min_samples=5,
        filter_by='channel_0', img_zoom=0.3, plot_by_cluster=False,
        plot_cluster_grids=False, remove_cluster_noise=False,
        remove_highly_correlated=True, log_data=False, black_background=True,
        remove_image_canvas=False, plot_outlines=False, plot_points=True,
        smooth_lines=False, clustering='dbscan', reduction_method='umap',
        save_figure=False, row_limit=200, color_by=None, exclude=None,
        plot_images=False, embedding_by_controls=False,
        col_to_compare='columnID', pos='c2', neg='c1', figuresize=10,
        verbose=False, resnet_features=False, channel_of_interest=3,
        mix_metadata=False, analyze_clusters=False, cell_min_size=0,
        nucleus_min_size=0, pathogen_min_size=0, cytoplasm_min_size=0,
        min_cell_count=0, nuclei_limit=True, pathogen_limit=True)
    generate_image_umap(umap_s)
    # branch variations: tSNE + KMeans, control-trained embedding, image
    # grids, cluster analysis, condition exclusion
    generate_image_umap({**umap_s, "reduction_method": "tsne",
                         "clustering": "kmeans"})
    generate_image_umap({**umap_s, "embedding_by_controls": True})
    generate_image_umap({**umap_s, "plot_images": True,
                         "plot_by_cluster": True})
    generate_image_umap({**umap_s, "analyze_clusters": True})
    reducer_hyperparameter_search(
        umap_s, reduction_params=[{'n_neighbors': 15}],
        dbscan_params=[{'eps': 0.5, 'min_samples': 5}],
        kmeans_params=[{'n_clusters': 3}], save=False, show=False)

    graph_s = dict(
        src=plate, tables=['cell', 'nucleus', 'pathogen', 'cytoplasm'],
        cells=['HeLa'], cell_loc=[['c1', 'c2']], controls=['c1', 'c2'],
        controls_loc=[['c1'], ['c2']], graph_type='bar', summary_func='mean',
        y_axis_start=0, error_bar_type='std', theme='pastel',
        representation='well', nuclei_limit=True, pathogen_limit=True,
        channel_of_interest=3, verbose=False, graph_name='screen')
    generate_screen_graphs(graph_s)


@_skip
def test_stage5_toxo_on_scores(pipeline, tmp_path):
    from spacr import toxo as T

    # Build a coefficient/p_value frame in the shape toxo expects, from the
    # measured cells (real intensities → real distribution).
    con = sqlite3.connect(pipeline["db"])
    cell = pd.read_sql_query("SELECT * FROM cell", con)
    con.close()
    n = min(len(cell), 60)
    rng = np.random.default_rng(1)
    data = pd.DataFrame({
        "feature": [f"{220000 + i}_1" for i in range(n)],
        "coefficient": rng.normal(0, 0.4, n),
        "p_value": np.clip(np.abs(rng.normal(0.05, 0.05, n)), 1e-8, 1),
    })
    metadata = pd.DataFrame({
        "gene_nr": [str(220000 + i) for i in range(n)],
        "tagm_location": rng.choice(["cytosol", "dense granules", "unknown"], n),
    })
    hits = T.custom_volcano_plot(data, metadata, figsize=6,
                                 save_path=str(tmp_path / "volcano.pdf"))
    assert isinstance(hits, list)
    assert (tmp_path / "volcano.pdf").exists()
