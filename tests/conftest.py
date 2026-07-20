"""
Shared pytest fixtures + synthetic-data builders for the spacr test suite.

Everything in here is DETERMINISTIC (fixed seeds) so failures can be
reproduced from a git hash alone. Fixtures are session-scoped where the
generated object is read-only; per-test fixtures reset writable state.

Fixtures provided:
    tmp_project_dir    per-test temp dir that gets wiped after the test
    rng                numpy Generator seeded to 0
    synth_image_2d     2-D uint16 grayscale "microscopy" image
    synth_image_3d     3-D uint16 image (Z, H, W)
    synth_image_stack  4-D uint16 stack (T, C, H, W)
    synth_mask_2d      2-D int label mask with N connected blobs
    synth_masks_multi  dict of cell/nucleus/pathogen label masks
    synth_measurements pandas DataFrame with typical spacr columns
    synth_sqlite_db    file-backed sqlite with a minimal spacr schema
    dark_style         style_out dict returned by set_dark_style() with
                       a hidden Tk root; scope='function' to keep Tk
                       state fresh across tests.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the in-tree spacr importable without an editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Headless matplotlib for CI / test runs.
os.environ.setdefault("MPLBACKEND", "Agg")

# Try to import matplotlib once with the Agg backend fixed. If unavailable,
# individual tests that need it will skip themselves.
try:  # pragma: no cover - import side effect only
    import matplotlib
    matplotlib.use("Agg", force=True)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Basic infra fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Deterministic numpy Generator."""
    return np.random.default_rng(0)


@pytest.fixture
def tmp_project_dir(tmp_path):
    """A fresh temp directory laid out like a spacr project."""
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    (tmp_path / "measurements").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# Synthetic image fixtures
# ---------------------------------------------------------------------------

def _place_blobs(shape, n_blobs, rng, radius_range=(6, 14), max_intensity=60000):
    """Draw n_blobs bright circular blobs on a dark background."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    img = np.zeros(shape, dtype=np.uint16)
    for _ in range(n_blobs):
        cy = int(rng.integers(20, h - 20))
        cx = int(rng.integers(20, w - 20))
        r = int(rng.integers(*radius_range))
        intensity = int(rng.integers(int(max_intensity * 0.4), max_intensity))
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        img[mask] = np.maximum(img[mask], intensity)
    # Add a bit of gaussian background noise so np.min != np.max in flat regions.
    img = img + rng.integers(50, 200, size=shape, dtype=np.uint16)
    return img.astype(np.uint16)


@pytest.fixture
def synth_image_2d(rng):
    """256x256 uint16 grayscale image with ~8 bright blobs on dark background."""
    return _place_blobs((256, 256), n_blobs=8, rng=rng)


@pytest.fixture
def synth_image_3d(rng):
    """3-D image (Z=5, H=128, W=128) uint16."""
    return np.stack([_place_blobs((128, 128), n_blobs=6, rng=rng) for _ in range(5)])


@pytest.fixture
def synth_image_stack(rng):
    """4-D (T=3, C=2, H=128, W=128) uint16 timelapse-ish stack."""
    return np.stack(
        [
            np.stack([_place_blobs((128, 128), n_blobs=5, rng=rng) for _ in range(2)])
            for _ in range(3)
        ]
    )


# ---------------------------------------------------------------------------
# Synthetic label-mask fixtures
# ---------------------------------------------------------------------------

def _labeled_blobs(shape, n_blobs, rng, radius_range=(8, 16)):
    """Return an int32 label image where each blob has a unique id starting at 1."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    lbl = np.zeros(shape, dtype=np.int32)
    next_id = 1
    for _ in range(n_blobs):
        cy = int(rng.integers(20, h - 20))
        cx = int(rng.integers(20, w - 20))
        r = int(rng.integers(*radius_range))
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        # Skip if it would overlap an existing label (keep them disjoint).
        if lbl[mask].max() != 0:
            continue
        lbl[mask] = next_id
        next_id += 1
    return lbl


@pytest.fixture
def synth_mask_2d(rng):
    """256x256 int32 label mask, 6 disjoint blobs (ids 1..N)."""
    return _labeled_blobs((256, 256), n_blobs=6, rng=rng)


@pytest.fixture
def synth_masks_multi(rng):
    """Dict of aligned cell/nucleus/pathogen label masks for one 256x256 field."""
    cell = _labeled_blobs((256, 256), n_blobs=5, rng=rng, radius_range=(20, 30))
    # Nucleus sits inside cells; smaller radius, centered on cell centroids.
    nucleus = np.zeros_like(cell)
    from scipy.ndimage import center_of_mass
    for cell_id in np.unique(cell):
        if cell_id == 0:
            continue
        cy, cx = center_of_mass(cell == cell_id)
        yy, xx = np.mgrid[: cell.shape[0], : cell.shape[1]]
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= 6 ** 2
        nucleus[m] = cell_id
    # Pathogens: 0-2 small blobs scattered inside random cells.
    pathogen = np.zeros_like(cell)
    next_id = 1
    for _ in range(int(rng.integers(0, 3))):
        cell_ids = [i for i in np.unique(cell) if i != 0]
        if not cell_ids:
            break
        cid = int(rng.choice(cell_ids))
        cy, cx = center_of_mass(cell == cid)
        yy, xx = np.mgrid[: cell.shape[0], : cell.shape[1]]
        offset_y = int(rng.integers(-10, 11))
        offset_x = int(rng.integers(-10, 11))
        m = (yy - (cy + offset_y)) ** 2 + (xx - (cx + offset_x)) ** 2 <= 3 ** 2
        m = m & (cell == cid)  # keep pathogen inside its cell
        if m.any():
            pathogen[m] = next_id
            next_id += 1
    return {"cell": cell, "nucleus": nucleus, "pathogen": pathogen}


# ---------------------------------------------------------------------------
# Synthetic DataFrames & sqlite
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_measurements(rng):
    """A DataFrame with typical spacr measurement columns for 40 objects."""
    n = 40
    plates = ["plate1"] * n
    rows = rng.integers(1, 9, size=n)  # A..H analog
    cols = rng.integers(1, 13, size=n)
    wells = [f"{chr(ord('A')+r-1)}{c:02d}" for r, c in zip(rows, cols)]
    fields = rng.integers(1, 4, size=n)
    prcs = [f"{p}_{w}_{f}" for p, w, f in zip(plates, wells, fields)]
    return pd.DataFrame(
        {
            "plate": plates,
            "row": rows,
            "column": cols,
            "well": wells,
            "field": fields,
            "prc": prcs,
            "object_label": np.arange(1, n + 1),
            "cell_area": rng.uniform(200, 4000, size=n),
            "cell_channel_0_mean_intensity": rng.uniform(500, 40000, size=n),
            "cell_channel_1_mean_intensity": rng.uniform(500, 40000, size=n),
            "nucleus_area": rng.uniform(80, 900, size=n),
            "pathogen_count": rng.integers(0, 5, size=n),
        }
    )


@pytest.fixture
def synth_sqlite_db(tmp_path, synth_measurements):
    """A file-backed sqlite database with a minimal spacr-ish schema."""
    db_path = tmp_path / "measurements.db"
    con = sqlite3.connect(db_path)
    try:
        synth_measurements.to_sql("cell", con, index=False)
        # A dummy annotation table many spacr helpers assume exists.
        anno = pd.DataFrame(
            {
                "prc": synth_measurements["prc"].unique(),
                "annotation": 0,
            }
        )
        anno.to_sql("png_list", con, index=False)
    finally:
        con.close()
    return db_path


# ---------------------------------------------------------------------------
# GUI / Tk fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tk_root():
    """A hidden Tk root; skips if there is no display available."""
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display available for Tk: {e}")
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


@pytest.fixture
def dark_style(tk_root):
    """The style_out dict returned by set_dark_style()."""
    from tkinter import ttk
    from spacr.gui_elements import set_dark_style
    return set_dark_style(ttk.Style(), parent_frame=None)
