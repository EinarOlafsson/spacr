"""The sbs_cells phase and the provenance-matched mask resume (items 1055, 1056).

A masks-only run (objects, sbs_cells, ph_flatfield, ph_segment) followed by the
phases that read the stored masks gives the tables of one full run, byte for
byte; a changed weights md5 or mask setting refuses; the sbs_cells phase alone
writes the masks decode writes.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tifffile")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")
pytest.importorskip("cellpose.models")

from spacr import ops_engine
from spacr.image_store import _merged_store_path, field_key, read_labels
from spacr.ops_store import read_table, write_table
from tests.test_ops_objects_resume_958 import _CountingModel, stitched  # noqa: F401
from tests.test_ops_phenotype_run_v2 import SIZE, _ThresholdModel, _write_tiles
from tests.test_ops_provenance_975 import V1, V2, weights  # noqa: F401
from tests.test_sbs_cells_1025 import _ProbabilityModel

MASKS = ("objects", "sbs_cells", "ph_flatfield", "ph_segment")
LATER = ("decode", "ph_measure")
FULL = ("objects", "decode", "ph_flatfield", "ph_segment", "ph_measure")
PH = {"ops_ph_channels": "DAPI,GFP,AF750", "ops_flatfield": "none",
      "ops_ph_downsample": 1, "ops_ph_texture": False, "ops_ph_image_format": "npz",
      "ops_sbs_cell_method": "cellpose", "ops_cross_field_ownership": False}


def _run(stitched, folder, monkeypatch, phases, **extra):
    """Run phases on a copy of the stitched well that also holds placed phenotype fields."""
    root, settings, library = stitched
    if not (folder / "measurements").exists():
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root / "stitched" / "measurements", folder / "measurements")
        _write_tiles(folder / "tiles")
        frame = pd.DataFrame([{"plate": "synthetic", "well": "A1", "site": site,
                               "centre_y": 1000.0 + 100 * site, "centre_x": 1000.0,
                               "sbs_site": 0, "is_anchor": int(site == 0)}
                              for site in (0, 1)])
        write_table(str(folder / "measurements.db"), "ops_phenotype",
                    ops_engine._ph_transform_columns(frame, 0.25, 0.0, (SIZE, SIZE)))

    def fake(s, gpu, key="ops_sbs_nucleus_model"):
        """One stand-in per segmentation role."""
        if key == "ops_sbs_cell_model":
            return _ProbabilityModel()
        if key == "ops_sbs_nucleus_model":
            return _CountingModel()
        return _ThresholdModel(grow=0 if "nucleus" in key else 5)

    monkeypatch.setattr(ops_engine, "_cellpose_model", fake)
    monkeypatch.setattr(ops_engine, "_SBS_CELL_MODELS", {})
    return ops_engine.run_ops(
        {**settings, **PH, "dst_root": str(folder), "phenotype_source": str(folder / "tiles"),
         **extra}, phases=phases, library=library)


def _digest(folder):
    """A hash of every table and mask of the run, without paths or timings."""
    out = {}
    db = str(folder / "measurements.db")
    for name in ("ops_objects", "ops_barcodes"):
        out[name] = read_table(db, name)
    out["cells"] = pd.read_parquet(folder / "ops_sbs_cells_A1.parquet") \
        if (folder / "ops_sbs_cells_A1.parquet").exists() else None
    store = _merged_store_path(str(folder / "sbs"))
    for site in range(0, 8):
        try:
            for label in ("sbs_cell", "sbs_nucleus"):
                out[f"{label}{site}"] = np.asarray(read_labels(store, label, key=field_key("A1", site)))
        except (OSError, ValueError, KeyError):
            break
    for part in sorted((folder / "A1" / "phenotype_cells").glob("site-*.parquet")):
        out[part.name] = pd.read_parquet(part)
    for part in sorted((folder / "A1" / "phenotype_masks").glob("site-*.npz")):
        with np.load(part) as data:
            out[part.name] = {key: data[key] for key in sorted(data.files)}
    digest = hashlib.sha256()
    for key in sorted(out):
        value = out[key]
        if value is None:
            continue
        digest.update(key.encode())
        if isinstance(value, pd.DataFrame):
            drop = [c for c in value.columns if any(t in c.lower() for t in ("path", "time", "second"))]
            digest.update(value.drop(columns=drop).to_csv(index=False).encode())
        elif isinstance(value, dict):
            for name in sorted(value):
                digest.update(np.ascontiguousarray(value[name]).tobytes())
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _report(folder):
    with open(folder / "A1" / "ops_report.json", encoding="utf-8") as handle:
        return json.load(handle)


def test_a_masks_only_run_then_the_rest_equals_one_full_run(
        stitched, weights, tmp_path, monkeypatch):
    """The key test: byte-identical objects, cells, calls and phenotype tables."""
    _run(stitched, tmp_path / "full", monkeypatch, FULL)
    _run(stitched, tmp_path / "split", monkeypatch, MASKS)
    models = _report(tmp_path / "split")["provenance"]["cellpose_models"]
    assert set(models) == {"sbs_nuclei", "sbs_cells", "ph_nuclei", "ph_cells"}
    assert all(record["settings_digest"] for record in models.values())
    result = _run(stitched, tmp_path / "split", monkeypatch, LATER)
    assert result["wells"]["A1"]["decode"]["sbs_cells"]["reused"] is True
    assert _digest(tmp_path / "split") == _digest(tmp_path / "full")


def test_the_sbs_cells_phase_alone_writes_the_masks_decode_writes(
        stitched, weights, tmp_path, monkeypatch):
    _run(stitched, tmp_path / "inside", monkeypatch, ("objects", "decode"),
         ops_sbs_cell_method="cellpose")
    _run(stitched, tmp_path / "alone", monkeypatch, ("objects", "sbs_cells"),
         ops_sbs_cell_method="cellpose")
    store_a = _merged_store_path(str(tmp_path / "alone" / "sbs"))
    store_b = _merged_store_path(str(tmp_path / "inside" / "sbs"))
    sites = 0
    for site in range(8):
        try:
            first = np.asarray(read_labels(store_a, "sbs_cell", key=field_key("A1", site)))
        except (OSError, ValueError, KeyError):
            break
        assert np.array_equal(first, np.asarray(
            read_labels(store_b, "sbs_cell", key=field_key("A1", site))))
        sites += 1
    assert sites and (tmp_path / "alone" / "ops_sbs_cells_A1.parquet").exists()
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "alone" / "ops_sbs_cells_A1.parquet"),
                                  pd.read_parquet(tmp_path / "inside" / "ops_sbs_cells_A1.parquet"))
    assert _report(tmp_path / "alone")["provenance"]["cellpose_models"]["sbs_cells"]["made_by"] \
        == "sbs_cells"


def test_a_mismatched_weights_md5_refuses(stitched, weights, tmp_path, monkeypatch):
    _run(stitched, tmp_path, monkeypatch, MASKS, ops_cellpose_model="cpsam_v1")
    for phases in (("decode",), ("ph_measure",)):
        with pytest.raises(ValueError, match="md5"):
            _run(stitched, tmp_path, monkeypatch, phases, ops_cellpose_model="cpsam_v2")


def test_a_changed_mask_setting_refuses_and_the_rest_is_free(
        stitched, weights, tmp_path, monkeypatch):
    _run(stitched, tmp_path, monkeypatch, MASKS)
    with pytest.raises(ValueError, match="ops_sbs_cell_threshold"):
        _run(stitched, tmp_path, monkeypatch, ("decode",), ops_sbs_cell_threshold=0.5)
    with pytest.raises(ValueError, match="ops_ph_cell_diameter"):
        _run(stitched, tmp_path, monkeypatch, ("ph_measure",), ops_ph_cell_diameter=33)
    with pytest.raises(ValueError, match="batch_size"):
        _run(stitched, tmp_path, monkeypatch, ("ph_measure",), batch_size=3)
    _run(stitched, tmp_path, monkeypatch, LATER)
    _run(stitched, tmp_path, monkeypatch, ("decode",), ops_read_threshold=20)
