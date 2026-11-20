"""OPS support readers retain native labels, identities and legacy storage behavior."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("zarr")
pytest.importorskip("pyarrow")

from spacr import crop_store, ops_decode, ops_engine, ops_run_infra
from spacr.image_store import PlateWriter, field_key
from spacr.tabular import read_table


def _crop(label, has_mask=True):
    """One deterministic raw crop with distinct cell and nucleus labels."""
    pixels = (np.arange(60).reshape(3, 4, 5) + 100 * label).astype(np.uint16)
    mask = np.array([[0, 0, 1, 1, 0], [0, 1, 2, 2, 1],
                     [1, 2, 2, 1, 0], [0, 1, 1, 0, 0]], np.uint8)
    if not has_mask:
        mask = np.zeros_like(mask)
    return {"object_label": label, "pixels": pixels, "mask": mask,
            "bbox_y0": 0, "bbox_x0": 0, "bbox_y1": 4, "bbox_x1": 5,
            "margin_px": 0, "clipped_at_edge": False}


def _native(root, well, *, source="ops_ph_crops", commit=True, empty=False, marked=True):
    """Write two typed native crops, or a committed empty well, using the shared writer."""
    writer = crop_store.CropPoolWriter(root, "P1", well, "cell", 3,
        mask_dtype=np.uint8, crops_per_shard=2, attributes={"source": source})
    expected = {}
    if not empty:
        for label, object_id, has_mask in ((1, 41, True), (2, None, False)):
            crop = _crop(label, has_mask)
            identity = f"P1_{well}_7_{label}"
            writer.add([{**crop, "object_id": identity, "prcfo": identity,
                         "ops_crop": marked, "ops_object_id": object_id,
                         "site": 7, "label": label, "has_mask": has_mask,
                         "cy": 2, "cx": 2, "window_edge": False, "reason": "linked"}], field=7)
            expected[identity] = (crop["pixels"], crop["mask"] if has_mask else None)
    writer.commit() if commit else writer.close()
    return expected


def _legacy(root, well):
    """Write a legacy TIFF-in-TAR crop through the existing OPS archive writer."""
    crop = _crop(3)
    identity = f"P1_{well}_7_3"
    path = root / "data" / "cell" / "P1" / well / "shard-0000.tar"
    path.parent.mkdir(parents=True)
    frame = ops_engine._ph_write_shard(str(path), [(identity + ".tif",
        ops_engine._ph_crop_bytes(crop["pixels"], crop["mask"], ["DAPI", "GFP", "AF750"]),
        {"plate": "P1", "well": well, "site": 7, "label": 3,
         "prcfo": identity, "object_id": 43})])
    frame.insert(0, "shard", str(path))
    return frame, {identity: (crop["pixels"], crop["mask"])}


def test_crop_index_preserves_legacy_and_uses_only_committed_ops_native_wells(tmp_path):
    """Native OPS commits supersede TAR per well, including empty commits; other pools do not."""
    legacy, _expected = _legacy(tmp_path, "A1")
    pd.testing.assert_frame_equal(ops_run_infra.crop_index(str(tmp_path), "P1"), legacy)
    for well in ("A2", "B1", "C1", "D1", "F1"):
        _legacy(tmp_path, well)
    _native(tmp_path, "A1")
    _native(tmp_path, "B1", source="measure")
    _native(tmp_path, "C1", commit=False)
    _native(tmp_path, "D1", empty=True)
    _native(tmp_path, "E1", marked=False)
    _native(tmp_path, "F1")
    (crop_store.class_dir(tmp_path, "P1", "F1", "cell") / "mask" / "0" / "zarr.json").unlink()
    index = ops_run_infra.crop_index(str(tmp_path), "P1")
    assert index["well"].value_counts().to_dict() == {"A1": 2, "A2": 1, "B1": 1, "C1": 1, "F1": 1}
    native = index[index["well"] == "A1"].sort_values("label")
    assert native["crop_format"].tolist() == ["ome-zarr", "ome-zarr"]
    assert native["crop_object_id"].tolist() == native["prcfo"].tolist()
    assert native["object_id"].iloc[0] == 41 and native["object_id"].iloc[1] is None
    assert all(Path(path).is_dir() for path in native["shard"])
    assert ops_run_infra.crop_index(str(tmp_path), "P1", ["D1"]).empty
    assert ops_run_infra.crop_index(str(tmp_path), "another_plate").empty


def test_stream_crops_reads_mixed_formats_in_bounded_batches_with_exact_masks(tmp_path, monkeypatch):
    """Mixed streams preserve pixels, ternary labels, absent masks, stable order and bounded reads."""
    _legacy(tmp_path, "A1")
    expected = _native(tmp_path, "A1")
    _frame, old = _legacy(tmp_path, "A2")
    expected.update(old)
    calls = []
    original = crop_store.read_crops

    def read_batch(root, rows, **kwargs):
        """Record native requests while reading the real public crop pools."""
        calls.append(len(rows))
        return original(root, rows, **kwargs)

    monkeypatch.setattr(crop_store, "read_crops", read_batch)
    plain = list(ops_run_infra.stream_crops(str(tmp_path), "P1", shuffle=False, buffer=1))
    assert [row["prcfo"] for row, _pixels, _mask in plain] == list(
        ops_run_infra.crop_index(str(tmp_path), "P1")["prcfo"])
    assert calls == [1, 1]
    assert len(plain) == len(expected)
    for row, pixels, mask in plain:
        wanted, wanted_mask = expected[row["prcfo"]]
        assert pixels.dtype == np.uint16
        np.testing.assert_array_equal(pixels, wanted)
        if wanted_mask is None:
            assert mask is None
        else:
            assert mask.dtype == np.uint8 and set(np.unique(mask)) == {0, 1, 2}
            np.testing.assert_array_equal(mask, wanted_mask)
    first = [row["prcfo"] for row, _pixels, _mask in
             ops_run_infra.stream_crops(str(tmp_path), "P1", seed=7, buffer=2)]
    again = [row["prcfo"] for row, _pixels, _mask in
             ops_run_infra.stream_crops(str(tmp_path), "P1", seed=7, buffer=2)]
    assert first == again and set(first) == set(expected)


def test_native_crop_stress_accounting_does_not_decode_pools_as_tar(tmp_path, monkeypatch):
    """Existing TAR metrics stay TAR-only while native store files enter plate accounting."""
    _native(tmp_path, "A1")

    def refuse_tar(*args, **kwargs):
        """Fail if a native pool is accidentally passed to the archive decoder."""
        raise AssertionError("native pool passed to TAR reader")

    monkeypatch.setattr(ops_engine, "_ph_read_crop", refuse_tar)
    well, plate = ops_run_infra.stress_rows(str(tmp_path), "P1", ["A1"], {}, random_reads=1)
    assert well["tar_shards"] is None and well["tar_crops"] is None
    assert plate["nas_plate_bytes"] > 0 and plate["nas_plate_files"] > 0


def test_mask_comparison_discovers_native_fields_and_preserves_explicit_templates(tmp_path):
    """Default discovery reads committed labels, while an explicit NPZ template retains priority."""
    import tifffile

    cells = np.zeros((12, 14), np.uint32)
    cells[2:8, 3:10] = 9
    nuclei = np.zeros_like(cells)
    nuclei[3:6, 4:7] = 9
    mask_root = tmp_path / "A1" / "phenotype_masks"
    writer = PlateWriter(mask_root, "P1")
    writer.write_field("A1", 4, None, {"cells": cells, "nuclei": nuclei})
    field = writer.path / field_key("A1", 4)
    np.testing.assert_array_equal(ops_decode._read_labels(str(field), "Nuclei"), nuclei)
    np.savez(mask_root / "site-4.npz", cells=np.zeros_like(cells), nuclei=np.zeros_like(nuclei))
    reference = tmp_path / "reference.tif"
    tifffile.imwrite(reference, cells)
    settings = {"dst_root": str(tmp_path), "plate": "P1", "ops_ph_image_format": "ome-zarr",
                "ops_compare_fields": ["A1:4"], "ops_compare_reference_objects": ["Cells"],
                "ops_compare_reference_masks": str(reference)}
    ops_decode.run_compare_masks(settings)
    agreement = read_table(tmp_path / "results" / "ops_compare" / "mask_agreement.csv",
                           canonicalise=False, report=None)
    assert agreement.loc[agreement["site"] == "all", "f1"].item() == 1.0
    ops_decode.run_compare_masks({**settings, "ops_compare_spacr_masks": str(mask_root / "site-4.npz")})
    agreement = read_table(tmp_path / "results" / "ops_compare" / "mask_agreement.csv",
                           canonicalise=False, report=None)
    assert agreement.loc[agreement["site"] == "all", "tp"].item() == 0


def test_flatfield_validation_matches_legacy_with_native_holdout_labels(tmp_path, monkeypatch):
    """The unchanged estimator sees exact held-out labels through committed native fields."""
    from spacr import ops_flatfield
    from tests.test_ops_flatfield_v2_912 import _run, field_image

    settings = _run(tmp_path, ops_ph_image_format="npz")
    legacy = ops_flatfield.run_flat_phases(settings, ("flat_nuclear", "flat_validate"))
    for well, site in (("A1", 1), ("B1", 2)):
        _pixels, labels = field_image(site)
        writer = PlateWriter(Path(settings["dst_root"]) / well / "phenotype_masks", "P1")
        writer.write_field(well, site, None, {"nuclei": labels, "cells": labels})
    seen = []
    original = ops_engine._ph_read_masks

    def read_masks(path):
        """Inspect the resolved path while retaining the shared mask reader and exact labels."""
        seen.append(str(path))
        return original(path)

    monkeypatch.setattr(ops_engine, "_ph_read_masks", read_masks)
    native = ops_flatfield.run_flat_phases({**settings, "ops_ph_image_format": "ome-zarr"}, ("flat_validate",))
    assert len(seen) == 2 and all(".ome.zarr/" in path for path in seen)
    np.testing.assert_equal(native["plates"]["P1"]["validation"],
                            legacy["plates"]["P1"]["validation"])
