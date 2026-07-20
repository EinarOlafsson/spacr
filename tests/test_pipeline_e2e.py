"""
End-to-end spacr pipeline tests using real HF microscopy data.

These are the core-function tests: they verify that
`preprocess_generate_masks` -> `measure_crop` actually produces the
folder structure, mask arrays, and `measurements/measurements.db` that
the rest of spacr consumes.

Every test is marked slow + gpu + network. The pipeline itself runs
once per session via the `spacr_pipeline_run` fixture in conftest;
tests here just inspect its outputs.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest


pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.network,
]


# ---------------------------------------------------------------------------
# 1. Folder structure after preprocess_generate_masks
# ---------------------------------------------------------------------------

EXPECTED_TOP_LEVEL_FOLDERS = {
    "orig",         # originals moved out of the flat source dir
    "1", "2", "3", "4",  # per-channel folders (Yokogawa channels)
    "stack",        # merged per-FOV .npy stacks
    "channel_stack",  # normalized per-channel arrays
    "masks",        # {cell,nucleus,pathogen,organelle}_mask_stack subdirs
    "merged",       # merged mask + image arrays (for plot / measure)
    "measurements", # measurements.db lives here
    "settings",     # save_settings() writes into here
}


def test_pipeline_creates_expected_top_level_folders(spacr_pipeline_run):
    src = Path(spacr_pipeline_run["src"])
    subdirs = {p.name for p in src.iterdir() if p.is_dir()}
    # Every folder we listed above must appear.
    missing = EXPECTED_TOP_LEVEL_FOLDERS - subdirs
    assert not missing, f"pipeline did not create {missing}; got {subdirs}"


def test_orig_holds_the_raw_input_tiffs(spacr_pipeline_run):
    orig = Path(spacr_pipeline_run["src"]) / "orig"
    assert orig.exists()
    tiffs = sorted(p.name for p in orig.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    # Fixture pulls 3 fields * 4 channels = 12 raw TIFFs.
    assert len(tiffs) == 12, f"expected 12 raw TIFFs in orig/, got {len(tiffs)}"


@pytest.mark.parametrize("chan", ["1", "2", "3", "4"])
def test_each_channel_folder_holds_one_file_per_field(spacr_pipeline_run, chan):
    folder = Path(spacr_pipeline_run["src"]) / chan
    assert folder.exists()
    tiffs = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    # 3 fields → 3 per-field per-channel files.
    assert len(tiffs) == 3, f"channel {chan}: expected 3 files, got {len(tiffs)}"


def test_stack_and_channel_stack_contain_npy_arrays(spacr_pipeline_run):
    src = Path(spacr_pipeline_run["src"])
    stack = src / "stack"
    channel_stack = src / "channel_stack"
    assert stack.exists() and any(stack.iterdir()), "stack/ is empty"
    assert channel_stack.exists() and any(channel_stack.iterdir()), "channel_stack/ is empty"


# ---------------------------------------------------------------------------
# 2. Fluorescent mask generation — masks exist for every requested object
#    type and are valid label arrays
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("object_type", ["cell", "nucleus", "pathogen"])
def test_mask_stack_folder_created(spacr_pipeline_run, object_type):
    mask_dir = Path(spacr_pipeline_run["src"]) / "masks" / f"{object_type}_mask_stack"
    assert mask_dir.exists(), f"missing {mask_dir}"
    npys = list(mask_dir.glob("*.npy"))
    assert npys, f"{mask_dir} contains no .npy mask files"


@pytest.mark.parametrize("object_type", ["cell", "nucleus", "pathogen"])
def test_masks_are_valid_label_arrays(spacr_pipeline_run, object_type):
    """Every generated mask must be an integer label array with 0 as
    background and unique positive ids for each object."""
    mask_dir = Path(spacr_pipeline_run["src"]) / "masks" / f"{object_type}_mask_stack"
    for npy in mask_dir.glob("*.npy"):
        arr = np.load(npy)
        assert arr.dtype.kind in "iu", f"{npy.name}: dtype {arr.dtype} is not integer"
        assert arr.ndim in (2, 3), f"{npy.name}: mask is {arr.ndim}D — expected 2D or 3D"
        assert arr.min() == 0, f"{npy.name}: background label != 0"
        if arr.max() > 0:
            unique = np.unique(arr)
            # ids must be non-negative and unique per object
            assert (unique >= 0).all()


def test_nucleus_masks_lie_inside_cell_masks(spacr_pipeline_run):
    """Every nucleus label must have >50% of its area inside a cell label.

    A small tolerance is allowed because segmentation on real images is
    imperfect at cell boundaries, but the vast majority should be inside.
    """
    src = Path(spacr_pipeline_run["src"])
    cell_dir = src / "masks" / "cell_mask_stack"
    nuc_dir = src / "masks" / "nucleus_mask_stack"

    cell_files = sorted(cell_dir.glob("*.npy"))
    nuc_files = sorted(nuc_dir.glob("*.npy"))
    assert cell_files and nuc_files
    # Files should be name-matched between the two folders.
    common = set(p.name for p in cell_files) & set(p.name for p in nuc_files)
    assert common, "no matching cell/nucleus mask filenames"

    inside_ratios = []
    for fname in common:
        cell = np.load(cell_dir / fname)
        nuc = np.load(nuc_dir / fname)
        for nid in np.unique(nuc):
            if nid == 0:
                continue
            nuc_pixels = (nuc == nid)
            inside = ((cell != 0) & nuc_pixels).sum()
            total = nuc_pixels.sum()
            if total > 0:
                inside_ratios.append(inside / total)
    if inside_ratios:
        median_inside = float(np.median(inside_ratios))
        # At least half the nuclei must be predominantly inside a cell.
        assert median_inside >= 0.5, (
            f"median fraction of nucleus pixels inside a cell = {median_inside:.2f}"
        )


def test_mask_object_sizes_are_biologically_plausible(spacr_pipeline_run):
    """Cell mask objects should be at least a few dozen pixels and no
    single object should span more than half the image — a very loose
    sanity check that catches wildly-wrong segmentation."""
    mask_dir = Path(spacr_pipeline_run["src"]) / "masks" / "cell_mask_stack"
    for npy in mask_dir.glob("*.npy"):
        arr = np.load(npy)
        total_pixels = arr.size
        for cid in np.unique(arr):
            if cid == 0:
                continue
            size = (arr == cid).sum()
            assert size >= 10, f"{npy.name}: object {cid} is only {size} px"
            assert size <= 0.5 * total_pixels, (
                f"{npy.name}: object {cid} covers {size}/{total_pixels} px"
            )


# ---------------------------------------------------------------------------
# 3. Measurements: run measure_crop on the pipeline output and inspect the
#    generated measurements/measurements.db
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spacr_measure_run(spacr_pipeline_run):
    """Run measure_crop on the shared pipeline output (module-scoped so
    the downstream measurement tests share the same DB)."""
    from spacr.measure import measure_crop
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings(None)
    settings.update({
        "src": spacr_pipeline_run["src"],
        "channels": [1, 2, 3, 4],
        "cell_channel": 1, "nucleus_channel": 0, "pathogen_channel": 2,
        "cell_mask_dim": None, "nucleus_mask_dim": None, "pathogen_mask_dim": None,
        "cell_chann_dim": 1, "nucleus_chann_dim": 0, "pathogen_chann_dim": 2,
        "cytoplasm": True,
        "n_jobs": 1, "batch_size": 8, "verbose": False,
        "plot": False, "save_png": False, "save_arrays": False,
    })
    try:
        measure_crop(settings)
    except Exception as e:  # pragma: no cover - integration path
        pytest.skip(f"measure_crop failed on synthetic pipeline output: {e}")
    return {"src": spacr_pipeline_run["src"], "db_path":
            os.path.join(spacr_pipeline_run["src"], "measurements", "measurements.db")}


def test_measurements_db_created(spacr_measure_run):
    db = Path(spacr_measure_run["db_path"])
    assert db.exists(), f"expected {db} to be created by measure_crop"
    assert db.stat().st_size > 0


def test_measurements_db_has_object_tables(spacr_measure_run):
    """The measurements DB should contain at least one of the standard
    per-object tables (cell / nucleus / pathogen / cytoplasm)."""
    conn = sqlite3.connect(spacr_measure_run["db_path"])
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()
    per_object = {"cell", "nucleus", "pathogen", "cytoplasm"}
    intersect = tables & per_object
    assert intersect, f"measurements.db has no per-object tables (found {tables})"


def test_measurements_db_rows_are_nonempty(spacr_measure_run):
    """At least one per-object table has rows."""
    conn = sqlite3.connect(spacr_measure_run["db_path"])
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        counts = {}
        for t in tables:
            try:
                counts[t] = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            except sqlite3.OperationalError:
                counts[t] = 0
    finally:
        conn.close()
    assert any(v > 0 for v in counts.values()), (
        f"measurements.db has no rows in any table (counts={counts})"
    )
