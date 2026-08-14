"""Synthetic-data coverage for the measure_crop hot path.

Drives ``_measure_crop_core`` (the per-field worker that loads a merged
stack, filters/relabels masks, computes morphology + intensity features,
writes them to SQLite, and crops per-object PNGs) and the ``measure_crop``
orchestrator directly on a hand-built merged ``.npy`` stack — no GPU, no
Cellpose, no HF download. This covers the large block of measure.py that
otherwise only runs behind the slow/gpu e2e pipeline.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest

from spacr.settings import get_measure_crop_settings


def _build_merged_stack(masks, rng, n_channels=4, with_organelle=False):
    """Assemble a merged (H, W, C) stack: intensity channels then mask
    slices at the dims measure_crop expects (cell=4, nucleus=5, pathogen=6,
    optional organelle=7)."""
    cell = masks["cell"].astype(np.uint16)
    nucleus = masks["nucleus"].astype(np.uint16)
    pathogen = masks["pathogen"].astype(np.uint16)
    H, W = cell.shape
    chans = []
    for c in range(n_channels):
        # Signal correlated with the cell mask so intensity features vary.
        base = rng.integers(50, 200, size=(H, W)).astype(np.uint16)
        base[cell > 0] += 3000
        chans.append(base)
    layers = chans + [cell, nucleus, pathogen]
    if with_organelle:
        organelle = np.zeros_like(cell)
        organelle[nucleus > 0] = nucleus[nucleus > 0]
        layers.append(organelle.astype(np.uint16))
    return np.stack(layers, axis=-1).astype(np.uint16)


def _settings_for(merged_dir, **over):
    s = get_measure_crop_settings(settings={})
    s.update({
        "src": str(merged_dir),
        "channels": [0, 1, 2, 3],
        "cell_mask_dim": 4, "nucleus_mask_dim": 5, "pathogen_mask_dim": 6,
        "png_dims": [0, 1, 2], "png_size": [64, 64],
        "save_measurements": True, "save_png": True, "save_arrays": False,
        "plot": False, "verbose": False, "timelapse": False,
        "crop_mode": ["cell"], "normalize": [1, 99], "normalize_by": "png",
        "experiment": "exp", "n_jobs": 1, "test_mode": False,
        # measure_crop enables cytoplasm when cell+nucleus/pathogen exist;
        # _exclude_objects then keeps cells that have a nucleus + cytoplasm.
        "cytoplasm": True,
    })
    s.update(over)
    return s


def _write_stack(tmp_path, data, name="plate1_A01_F001.npy"):
    merged = tmp_path / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    # The pipeline creates measurements/ upstream; _merge_and_save_to_database
    # opens <parent>/measurements/measurements.db without mkdir.
    (tmp_path / "measurements").mkdir(parents=True, exist_ok=True)
    np.save(merged / name, data)
    return merged, name


# ---------------------------------------------------------------------------
# _measure_crop_core
# ---------------------------------------------------------------------------

def test_measure_crop_core_writes_measurements_and_pngs(tmp_path, synth_masks_multi, rng):
    from spacr.measure import _measure_crop_core
    data = _build_merged_stack(synth_masks_multi, rng)
    merged, name = _write_stack(tmp_path, data)
    settings = _settings_for(merged)

    index, avg_time, cells, figs = _measure_crop_core(0, [], name, settings)

    assert index == 0
    # cells is the unique-label array of the cell mask (includes 0 bg).
    assert np.max(cells) >= 1
    # measurements.db written one level up from merged/
    db = tmp_path / "measurements" / "measurements.db"
    assert db.is_file()
    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert any("cell" in t for t in tables)
    # PNG crops written under a cell_png folder.
    pngs = list(tmp_path.rglob("*.png"))
    assert pngs, "expected at least one cropped PNG"


def test_measure_crop_core_cytoplasm_and_bounding_box(tmp_path, synth_masks_multi, rng):
    from spacr.measure import _measure_crop_core
    data = _build_merged_stack(synth_masks_multi, rng)
    merged, name = _write_stack(tmp_path, data)
    settings = _settings_for(
        merged, cytoplasm=True, cytoplasm_min_size=1,
        use_bounding_box=True, nucleus_min_size=1, pathogen_min_size=1,
        cell_min_size=1)
    index, avg_time, cells, figs = _measure_crop_core(0, [], name, settings)
    assert index == 0
    assert (tmp_path / "measurements" / "measurements.db").is_file()


def test_measure_crop_core_nucleus_and_pathogen_crop_modes(tmp_path, synth_masks_multi, rng):
    from spacr.measure import _measure_crop_core
    data = _build_merged_stack(synth_masks_multi, rng)
    merged, name = _write_stack(tmp_path, data)
    settings = _settings_for(
        merged, crop_mode=["nucleus", "pathogen"],
        png_size=[[48, 48], [32, 32]])
    index, avg_time, cells, figs = _measure_crop_core(0, [], name, settings)
    assert index == 0


def test_measure_crop_core_save_arrays(tmp_path, synth_masks_multi, rng):
    from spacr.measure import _measure_crop_core
    data = _build_merged_stack(synth_masks_multi, rng)
    merged, name = _write_stack(tmp_path, data)
    settings = _settings_for(merged, save_arrays=True, save_png=False)
    index, avg_time, cells, figs = _measure_crop_core(0, [], name, settings)
    assert index == 0
    assert isinstance(cells, np.ndarray), "the worker must not return its failure sentinel"
    arrays = list(tmp_path.rglob("region_array/*.npy"))
    assert arrays, "expected saved region arrays"


def test_measure_crop_core_organelle_summary(tmp_path, synth_masks_multi, rng):
    """Organelle-per-parent summaries for cell / nucleus / pathogen parents."""
    from spacr.measure import _measure_crop_core
    data = _build_merged_stack(synth_masks_multi, rng, with_organelle=True)
    # Guarantee a pathogen object inside a nucleus so the pathogen-parent
    # summary branch runs (synth pathogens are random and often absent).
    nucleus = data[:, :, 5]
    ys, xs = np.where(nucleus > 0)
    if ys.size:
        cy, cx = int(ys[0]), int(xs[0])
        data[cy:cy + 3, cx:cx + 3, 6] = 1        # pathogen slice (dim 6)
    merged, name = _write_stack(tmp_path, data)
    settings = _settings_for(
        merged, organelle_mask_dim=7, organelle_min_size=0,
        summarize_organelles_by=["cell", "nucleus", "pathogen"])
    index, avg_time, cells, figs = _measure_crop_core(0, [], name, settings)
    assert index == 0


def test_measure_crop_core_no_cell_mask(tmp_path, synth_masks_multi, rng):
    """cell_mask_dim=None → uninfected path, cytoplasm disabled."""
    from spacr.measure import _measure_crop_core
    data = _build_merged_stack(synth_masks_multi, rng)
    merged, name = _write_stack(tmp_path, data)
    settings = _settings_for(merged, cell_mask_dim=None, crop_mode=["nucleus"])
    index, avg_time, cells, figs = _measure_crop_core(0, [], name, settings)
    assert index == 0


def test_measure_crop_core_float_input_converted(tmp_path, synth_masks_multi, rng):
    from spacr.measure import _measure_crop_core
    data = _build_merged_stack(synth_masks_multi, rng).astype(np.float32)
    merged, name = _write_stack(tmp_path, data)
    settings = _settings_for(merged, verbose=True)
    index, avg_time, cells, figs = _measure_crop_core(0, [], name, settings)
    assert index == 0


# ---------------------------------------------------------------------------
# measure_crop orchestrator
# ---------------------------------------------------------------------------

def test_measure_crop_end_to_end_single_field(tmp_path, synth_masks_multi, rng):
    from spacr.measure import measure_crop
    data = _build_merged_stack(synth_masks_multi, rng)
    merged, name = _write_stack(tmp_path, data)
    settings = _settings_for(merged, n_jobs=1)
    # measure_crop mutates settings['src']; pass a copy.
    measure_crop(dict(settings))
    assert (tmp_path / "measurements" / "measurements.db").is_file()


def test_measure_crop_core_plot_path(tmp_path, synth_masks_multi, rng):
    """plot=True exercises the before/after figures + PNG grid assembly."""
    from spacr.measure import _measure_crop_core
    data = _build_merged_stack(synth_masks_multi, rng)
    merged, name = _write_stack(tmp_path, data)
    settings = _settings_for(merged, plot=True)
    index, avg_time, cells, figs = _measure_crop_core(0, [], name, settings)
    assert index == 0
    # plot=True populates the figure dict with before/after/pngs entries.
    assert any(k.endswith("__before_filtration") for k in figs)


def test_measure_crop_core_dilate_and_cytoplasm_crop(tmp_path, synth_masks_multi, rng):
    """crop_mode cytoplasm + organelle, with PNG dilation enabled."""
    from spacr.measure import _measure_crop_core
    data = _build_merged_stack(synth_masks_multi, rng, with_organelle=True)
    merged, name = _write_stack(tmp_path, data)
    settings = _settings_for(
        merged, organelle_mask_dim=7,
        crop_mode=["cytoplasm", "organelle"],
        png_size=[[48, 48], [40, 40]],
        dialate_pngs=[True, True], dialate_png_ratios=[0.1, 0.1])
    index, avg_time, cells, figs = _measure_crop_core(0, [], name, settings)
    assert index == 0


def test_measure_crop_rejects_bool_normalize(tmp_path, synth_masks_multi, rng, capsys):
    """normalize=True (bool) is invalid — measure_crop warns and returns."""
    from spacr.measure import measure_crop
    data = _build_merged_stack(synth_masks_multi, rng)
    merged, name = _write_stack(tmp_path, data)
    settings = _settings_for(merged, normalize=True)
    measure_crop(dict(settings))
    out = capsys.readouterr().out
    assert "normalize" in out.lower()


def test_measure_crop_rejects_bad_normalize_by(tmp_path, synth_masks_multi, rng, capsys):
    from spacr.measure import measure_crop
    data = _build_merged_stack(synth_masks_multi, rng)
    merged, name = _write_stack(tmp_path, data)
    settings = _settings_for(merged, normalize=[1, 99], normalize_by="bogus")
    measure_crop(dict(settings))
    out = capsys.readouterr().out
    assert "normalize_by" in out


def test_measure_crop_appends_merged_when_missing(tmp_path, synth_masks_multi, rng, capsys):
    """src not ending in 'merged' → measure_crop appends merged/ and warns."""
    from spacr.measure import measure_crop
    data = _build_merged_stack(synth_masks_multi, rng)
    _write_stack(tmp_path, data)
    settings = _settings_for(tmp_path, n_jobs=1)  # src = tmp_path, not .../merged
    measure_crop(dict(settings))
    out = capsys.readouterr().out
    assert "merged" in out


def test_measure_crop_core_timelapse_nucleus_relabel(tmp_path, synth_masks_multi, rng):
    """timelapse_objects='nucleus' relabels cells to nucleus ids and re-saves."""
    from spacr.measure import _measure_crop_core
    data = _build_merged_stack(synth_masks_multi, rng)
    merged, name = _write_stack(tmp_path, data)
    settings = _settings_for(
        merged, timelapse=True, save_png=False,
        timelapse_objects="nucleus")
    index, avg_time, cells, figs = _measure_crop_core(0, [], name, settings)
    assert index == 0


# ---------------------------------------------------------------------------
# generate_cellpose_train_set + get_object_counts
# ---------------------------------------------------------------------------

def test_generate_cellpose_train_set(tmp_path, rng):
    from spacr.measure import generate_cellpose_train_set
    import tifffile
    # A source folder with a masks/ subfolder and images alongside.
    folder = tmp_path / "expA"
    (folder / "masks").mkdir(parents=True)
    # mask with >=5 objects (kept) and one with <5 (skipped)
    keep = np.zeros((32, 32), np.uint16)
    for i in range(1, 7):
        keep[i * 4:i * 4 + 2, 2:4] = i
    few = np.zeros((32, 32), np.uint16)
    few[0:2, 0:2] = 1
    for nm, m in [("keep.tif", keep), ("few.tif", few)]:
        tifffile.imwrite(folder / "masks" / nm, m)
        tifffile.imwrite(folder / nm, rng.integers(0, 255, (32, 32)).astype(np.uint16))

    dst = tmp_path / "train"
    generate_cellpose_train_set([str(folder)], str(dst), min_objects=5)
    imgs = list((dst / "imgs").glob("*.tif"))
    masks = list((dst / "masks").glob("*.tif"))
    assert any("keep" in p.name for p in masks)
    assert not any("few" in p.name for p in masks)  # below min_objects
    assert len(imgs) == len(masks)


def test_get_object_counts(tmp_path):
    from spacr.measure import get_object_counts
    meas = tmp_path / "measurements"
    meas.mkdir()
    con = sqlite3.connect(meas / "measurements.db")
    con.execute("CREATE TABLE object_counts (file_name TEXT, count_type TEXT, object_count INT)")
    con.executemany(
        "INSERT INTO object_counts VALUES (?,?,?)",
        [("f1", "cell", 5), ("f2", "cell", 7), ("f1", "nucleus", 4)])
    con.commit(); con.close()
    df = get_object_counts(str(tmp_path))
    row = df[df["count_type"] == "cell"].iloc[0]
    assert row["total_object_count"] == 12
    assert row["avg_object_count_per_file_name"] == 6
