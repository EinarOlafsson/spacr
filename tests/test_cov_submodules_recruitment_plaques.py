"""Shared recruitment / plaque test builders plus ``count_phenotypes`` and the empty plaque folder.

``_merged_recruitment_df``, ``_base_settings``, ``_Recorder``, ``_write_label_tif``
and ``plaque_env`` are imported by other suites.  ``analyze_recruitment`` and
``analyze_plaques`` are tested in ``tests/test_submodules_well_assays.py``.
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

    import spacr.submodules as SUB

    download = _Recorder(result="/nowhere")
    finetune = _Recorder()
    monkeypatch.setattr(SU, "download_models", download)
    monkeypatch.setattr(SCP, "identify_masks_finetune", finetune)
    # The default plaque model is a 1.2 GB zoo checkpoint since 2026-09-21;
    # resolving it for real would download it.
    monkeypatch.setattr(SUB, "_resolve_plaque_model",
                        lambda settings, fetch=True: "/models/plaque.CP_model")
    return {"download": download, "finetune": finetune}


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
                                           "measurements", "measurements.db")

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


def test_segmenting_a_plaque_folder_without_images_builds_no_model(tmp_path, monkeypatch):
    """A source with no plaque image segments nothing and never loads Cellpose."""
    import spacr.submodules as SUB

    src = tmp_path / "plate"
    src.mkdir()
    (src / "notes.txt").write_text("not an image", encoding="utf-8")
    (src / "nested.tif").mkdir()
    dst = tmp_path / "plate" / "masks"

    def no_model(_path):
        raise AssertionError("a folder with no images must not build the model")

    monkeypatch.setattr(SUB, "_plaque_cellpose_model", no_model)
    assert SUB._segment_plaque_folder({"src": str(src), "dst": str(dst)}, "model") == 0
    assert dst.is_dir() and list(dst.iterdir()) == []
