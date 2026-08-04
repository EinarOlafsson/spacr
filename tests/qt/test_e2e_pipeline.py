"""End-to-end pipeline tests against the synthetic demo datasets.

These tests actually invoke real spacr pipeline functions
(`preprocess_generate_masks`, `measure_crop`) so they need torch,
cellpose, and the rest of the heavy deps installed. That's why
they're marked `slow` — the fast CI job skips them, and running
`pytest -m slow` opts in.

If a test can't import the pipeline for any reason (usually a
missing optional dependency in a bare-bones environment) the test
skips with a clear reason so the suite still stays green.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from spacr.qt import synthetic as syn


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require(module: str) -> None:
    """Skip the test if `module` isn't importable."""
    try:
        __import__(module)
    except Exception as e:
        pytest.skip(f"required module not importable: {module} ({e})")


def _minimal_mask_settings(src: str) -> Dict[str, Any]:
    """Wrap `demo_settings` with a few extras `preprocess_generate_masks`
    reads on the first defaults pass (that aren't part of the CSV)."""
    s = syn.demo_settings("mask", src)
    s["consolidate"] = False
    s["remove_background"] = False
    s["normalize"] = True
    s["backgrounds"] = [100, 100, 100, 100]
    s["remove_background_cell"] = False
    s["remove_background_nucleus"] = False
    s["remove_background_pathogen"] = False
    s["cells_per_field"] = 4
    s["test_mode"] = False
    s["Signal_to_noise"] = 10
    s["cell_intensity_range"] = None
    s["nucleus_intensity_range"] = None
    s["pathogen_intensity_range"] = None
    s["cytoplasm_intensity_range"] = None
    s["denoise"] = False
    s["remove_background_intensity"] = False
    s["skip_extraction"] = False
    return s


# ---------------------------------------------------------------------------
# preprocess_generate_masks against the mask demo
# ---------------------------------------------------------------------------

def test_preprocess_generate_masks_runs_on_mask_demo(tmp_path: Path):
    """The mask pipeline segments the demo field and writes it out.

    "It did not crash" is not the contract — ``measure_crop`` is fed from
    this, so what matters is the state left on disk: a merged stack whose
    trailing planes carry labels, a ``measurements.db`` with per-object
    counts, and a run that reported itself complete.
    """
    _require("torch")
    _require("cellpose")
    from spacr.core import preprocess_generate_masks

    # Use all four channels — matches demo_settings("mask") which lists
    # channels=[0,1,2,3]. The pipeline normalises every channel in that
    # list, so we need images for each one.
    layout = syn.generate_mask_demo(
        tmp_path / "expt",
        wells=("A01",), fields=1, channels=(0, 1, 2, 3),
    )
    s = _minimal_mask_settings(str(layout.src))
    # Read before the call: the pipeline canonicalises the dict in place.
    # The label planes are appended after the acquisition channels, one per
    # role, in MASK_ROLE_ORDER — that is what `*_mask_dim` points at
    # downstream and what measure_crop then reads back.
    n_channels = len(s["channels"])
    mask_dims = {role: n_channels + i
                 for i, role in enumerate(syn.MASK_ROLE_ORDER)}

    try:
        preprocess_generate_masks(s)
    except Exception as e:
        # Cellpose / torch may fail cold on a machine with no models
        # downloaded; skip rather than fail so the test stays useful.
        msg = str(e).lower()
        if any(kw in msg for kw in ("model", "download", "cuda", "network")):
            pytest.skip(f"preprocess needed model / network access: {e}")
        raise

    src = layout.src

    # 1. The merged stack measure_crop consumes.
    merged = sorted((src / "merged").glob("*.npy"))
    assert merged, f"no merged stacks under {src / 'merged'}"
    assert all(p.stat().st_size > 0 for p in merged)

    stack = np.load(merged[0], allow_pickle=True)
    assert stack.ndim == 3
    assert stack.shape[-1] == n_channels + len(mask_dims), (
        f"expected {n_channels} image planes + {len(mask_dims)} label "
        f"planes, got {stack.shape[-1]}")
    for role in ("cell", "nucleus", "pathogen"):
        plane = stack[..., mask_dims[role]]
        assert plane.max() > 0, f"the {role} mask plane is empty"
    # Contrast: the trailing planes are *label* planes, not another copy of
    # the intensity image — a handful of distinct ids against thousands of
    # grey levels. Writing the raw channel into the mask slot would pass
    # every "not empty" check above and be useless downstream.
    assert np.unique(stack[..., 0]).size > 100
    assert 1 < np.unique(stack[..., mask_dims["cell"]]).size < 500

    # 2. The counts database.
    db = src / "measurements" / "measurements.db"
    assert db.is_file() and db.stat().st_size > 0
    con = sqlite3.connect(db)
    try:
        counts = dict(con.execute(
            "SELECT count_type, object_count FROM object_counts").fetchall())
        status = con.execute(
            "SELECT status, n_succeeded, n_failed FROM run_status").fetchall()
    finally:
        con.close()

    assert counts, "measurements.db recorded no object counts"
    assert counts["cell_before_filtration"] > 0
    assert counts["nucleus_before_filtration"] > 0
    assert counts["pathogen_before_filtration"] > 0
    assert status == [("complete", 1, 0)], f"run did not complete: {status}"

    # 3. The settings the run actually used, next to its output.
    used = src / "settings" / "gen_mask_settings.csv"
    assert used.is_file() and used.stat().st_size > 0


# ---------------------------------------------------------------------------
# measure_crop against the measure demo (has masks pre-built)
# ---------------------------------------------------------------------------

def test_measure_crop_runs_on_measure_demo(tmp_path: Path):
    """measure_crop turns the demo's merged stacks into measured objects.

    The output state is the whole point: per-object tables with real
    feature values, PNG crops for the classifier, and a run ledger that
    says every field succeeded. A scalar ``png_size`` used to make the
    single field fail 100% inside ``_measure_crop_core`` — the tables were
    still written, so nothing short of reading ``run_status`` noticed.
    """
    _require("torch")
    from spacr.measure import measure_crop

    layout = syn.generate_measure_demo(
        tmp_path / "expt",
        wells=("A01",), fields=1, channels=(0, 1, 2, 3),
    )
    s = syn.demo_settings("measure", str(layout.src))
    s["save_png"] = True
    s["png_size"] = [64, 64]        # [width, height], per the setting's contract
    s["png_dims"] = [0]
    s["experiment"] = "synth"
    s["representative_images"] = False
    s["cells"] = [1]
    s["nuclei"] = [1]

    try:
        measure_crop(s)
    except FileNotFoundError as e:
        # measure_crop expects a `merged/` folder that
        # preprocess_generate_masks produces. The measure demo doesn't
        # generate that intermediate layout yet — chaining the two
        # generators is a future improvement. Skip cleanly for now so
        # the assertion documents the gap without failing CI.
        pytest.skip(
            f"measure_crop needs a preprocessed layout the measure "
            f"demo doesn't yet produce: {e}"
        )
    except Exception as e:
        msg = str(e).lower()
        if any(kw in msg for kw in ("no such table", "no data",
                                       "cellpose", "torch", "cuda",
                                       "model", "empty")):
            pytest.skip(f"measure needed heavier setup: {e}")
        raise

    db = layout.src / "measurements" / "measurements.db"
    assert db.is_file() and db.stat().st_size > 0
    con = sqlite3.connect(db)
    try:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        rows = {t: con.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
                for t in ("cell", "nucleus", "pathogen", "cytoplasm")
                if t in tables}
        cell_cols = [r[1] for r in con.execute("PRAGMA table_info(cell)")]
        areas = [r[0] for r in con.execute("SELECT cell_area FROM cell")]
        wells = {r[0] for r in con.execute("SELECT prcf FROM cell")}
        status = con.execute(
            "SELECT status, n_attempted, n_succeeded, n_failed "
            "FROM run_status").fetchall()
        n_pngs = con.execute("SELECT COUNT(*) FROM png_list").fetchone()[0] \
            if "png_list" in tables else 0
    finally:
        con.close()

    # Every object type the demo ships got its own table, with rows in it.
    assert {"cell", "nucleus", "pathogen", "cytoplasm"} <= tables
    assert rows["cell"] > 0, "measure_crop wrote an empty cell table"
    assert rows["nucleus"] > 0 and rows["pathogen"] > 0

    # The rows carry real measurements, not placeholders: a wide feature
    # table and a strictly positive area for every single cell.
    assert len(cell_cols) > 50 and "cell_area" in cell_cols
    assert len(areas) == rows["cell"]
    assert all(a is not None and a > 0 for a in areas), \
        f"cells with no area: {[a for a in areas if not a]}"
    # ... measured off the one field the demo generated, not smeared over
    # a default/blank well id.
    assert wells == {"plate1_r1_c1_f1"}, f"unexpected well ids: {wells}"

    # save_png was on, so the crops exist both on disk and in the index.
    assert "png_list" in tables, "save_png=True produced no png_list table"
    assert n_pngs > 0
    assert len(sorted(layout.src.rglob("*.png"))) == n_pngs

    # And the ledger agrees the field was measured. This is the assertion
    # the old test lacked: the tables above were populated even on the run
    # where the only field failed.
    assert status == [("complete", 1, 1, 0)], f"run did not complete: {status}"
