"""Real OPS phases preserve pixels, masks, identities and resume across the default switch."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from spacr import ops_engine
from spacr.cli import resolve_module, resolve_settings
from spacr.crop_store import class_dir, read_commit
from spacr.ops_run_infra import crop_index, stream_crops
from spacr.tabular import read_database, read_table
from tests import test_ops_phenotype_run_v2 as fixture


models = fixture.models


def _settings(root, historical, masks):
    """Resolve the real format defaults while keeping the tiny fixture settings fixed."""
    path = None
    if historical:
        path = root / "old-settings.json"
        path.write_text(json.dumps({"spacr_settings_version": 948}), encoding="utf-8")
    resolved = resolve_settings(resolve_module("ops"), str(path) if path else None)
    formats = {key: resolved[key] for key in ("ops_ph_image_format", "ops_crop_image_format")}
    assert formats == ({"ops_ph_image_format": "npz", "ops_crop_image_format": "tiff-tar"}
                       if historical else {"ops_ph_image_format": "ome-zarr", "ops_crop_image_format": "ome-zarr"})
    return {**fixture.SETTINGS, **formats, "dst_root": str(root / "run"),
            "phenotype_source": str(root / "tiles"), "ops_design": str(root / "design.csv"),
            "ops_merged_root": str(root / "nas"), "ops_crop_masks": masks,
            "ops_crop_control_fraction": 0.0, "ops_ph_measure_gpu": False}


def _run(root, historical, masks):
    """Segment, measure, pair planted sequencing cells and run the real downstream phases."""
    root.mkdir()
    settings = _settings(root, historical, masks)
    sites = (0, 1) if masks else (1, 2, 10)
    fixture._write_tiles(root / "tiles", sites=sites)
    run = root / "run"
    run.mkdir()
    db = str(run / "measurements.db")
    fixture._placed(db, sites=sites)
    ops_engine.run_ops(settings, phases=("ph_segment", "ph_measure"))
    cells = read_table(ops_engine._ph_cells_path(str(run), "A1"), canonicalise=False, report=None)
    fixture._sequencing(db, root, cells)
    report = ops_engine.run_ops(settings, phases=fixture.FULL)
    return settings, report


def _files(root):
    """Exact bytes of persisted store files, excluding no production payloads."""
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("masks", [False, True])
def test_new_ops_defaults_match_old_saved_outputs_and_resume(tmp_path, models, masks):
    """All real phase outputs agree, including ternary/absent masks and typed cell IDs."""
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    old_settings, old_report = _run(old_root, True, masks)
    new_settings, new_report = _run(new_root, False, masks)
    sites = {0, 1} if masks else {1, 2, 10}
    count = 4 * len(sites)
    old_masks = ops_engine._ph_mask_paths(str(old_root / "run"), fixture.PLATE, "A1", old_settings)
    new_masks = ops_engine._ph_mask_paths(str(new_root / "run"), fixture.PLATE, "A1", new_settings)
    assert set(old_masks) == set(new_masks) == sites
    assert not list((new_root / "run" / "A1" / "phenotype_masks").glob("*.npz"))
    for site in old_masks:
        old = ops_engine._ph_read_masks(old_masks[site])
        new = ops_engine._ph_read_masks(new_masks[site])
        assert set(old) == set(new) == {"nuclei", "cells"}
        for role in old:
            assert old[role].dtype == new[role].dtype
            np.testing.assert_array_equal(old[role], new[role])
        old_field = ops_engine._ph_read_merged(ops_engine._ph_merged_path(
            str(old_root / "nas"), fixture.PLATE, "A1", site))
        new_field = ops_engine._ph_read_merged(ops_engine._ph_merged_path(
            str(new_root / "nas"), fixture.PLATE, "A1", site, "ome-zarr"))
        for key in ("channels", "nuclei", "cells", "cytoplasm", "owned"):
            assert old_field[key].dtype == new_field[key].dtype
            np.testing.assert_array_equal(old_field[key], new_field[key])
        assert old_field["meta"] == new_field["meta"]
    for table, key in (("cell", "prcfo"), ("nucleus", "prcfo"),
                       ("cytoplasm", "prcfo"), ("image", "prcf")):
        from spacr import store

        with store.open(old_report["ph_tables"]["store"]) as opened:
            old = opened.query(table=table).sort_values(key).reset_index(drop=True)
        with store.open(new_report["ph_tables"]["store"]) as opened:
            new = opened.query(table=table).sort_values(key).reset_index(drop=True)
        # The legacy tar crops are copied into the store, which adds its
        # crop pointer to the cell table; native crops stay where the crop
        # index points (item 0ap), so the new cell table has no copy.
        extra = [c for c in old.columns if c not in new.columns]
        assert set(extra) <= ({"crop_store", "crop_index"} if table == "cell" else set())
        pd.testing.assert_frame_equal(old.drop(columns=extra), new, check_exact=True)
        if table == "image" and not masks:
            assert old[key].tolist() == ["P_1_r1_c1_f1", "P_1_r1_c1_f10", "P_1_r1_c1_f2"]
    old_crops = list(stream_crops(str(old_root / "nas"), fixture.PLATE, shuffle=False))
    new_crops = list(stream_crops(str(new_root / "nas"), fixture.PLATE, shuffle=False))
    assert len(old_crops) == len(new_crops) == count
    for (old_row, old_image, old_mask), (new_row, new_image, new_mask) in zip(old_crops, new_crops):
        for key in ("prcfo", "plate", "well", "site", "label", "object_id",
                    "cy", "cx", "window_edge", "reason"):
            assert old_row[key] == new_row[key]
        assert old_image.dtype == new_image.dtype == np.uint16
        np.testing.assert_array_equal(old_image, new_image)
        if masks:
            assert old_mask.dtype == new_mask.dtype == np.uint8
            assert set(np.unique(new_mask)) == {0, 1, 2}
            np.testing.assert_array_equal(old_mask, new_mask)
        else:
            assert old_mask is new_mask is None
    assert not list((new_root / "nas").rglob("*.tar"))
    index = crop_index(str(new_root / "nas"), fixture.PLATE)
    assert index["prcfo"].is_unique and len(index) == count
    # The store step copies no committed crop (item 0ap); the crop index
    # gives them back through the one crop reader. With save_png off (the
    # default) they are cut on read from the fields ph_crops wrote to
    # fov.ome.zarr (item 0aq), equal to the stored legacy crops checked above.
    assert ops_engine._ops_store_crops(
        str(new_root / "nas"), fixture.PLATE, "A1", index["prcfo"].iloc[::-1]) == (None, None, None)
    from spacr.crop_store import read_crops

    picked = [index.iloc[i].to_dict() for i in (7, 0, 7)]
    assert all(row["crop_array"] is None and row["ops_crop"] is True
               and row["source_field"] == ops_engine._ph_field_name(fixture.PLATE, "A1", row["site"])
               for row in picked)
    for (image, mask), i in zip(read_crops(str(new_root / "nas"), picked), (7, 0, 7)):
        np.testing.assert_array_equal(image, new_crops[i][1])
        if masks:
            np.testing.assert_array_equal(mask, new_crops[i][2])
    before = _files(new_root / "nas")
    again = ops_engine.run_ops(new_settings, phases=("ph_segment", "ph_merged", "ph_crops"))
    assert again["wells"]["A1"]["ph_fields"]["ph_segment"]["resumed"] == len(sites)
    assert again["wells"]["A1"]["ph_merged"]["resumed"] == len(sites)
    assert again["ph_crops"]["wells"]["A1"]["resumed"] is True
    assert _files(new_root / "nas") == before
    base = class_dir(new_root / "nas", fixture.PLATE, "A1", "cell")
    (base / "CROPS_DONE.json").unlink()
    recovered = ops_engine.run_ops(new_settings, phases=("ph_crops",))
    assert recovered["ph_crops"]["wells"]["A1"]["crops"] == count
    committed = read_commit(new_root / "nas", fixture.PLATE, "A1", "cell")
    assert committed["n_rows"] == count
    assert len({row["object_id"] for row in committed["rows"]}) == count
    assert _files(new_root / "nas") == before
    for phase, marker in (
        ("ph_segment", new_root / "run" / "A1" / "phenotype_masks" / "part_options.json"),
        ("ph_merged", new_root / "nas" / "merged" / "phenotype" / fixture.PLATE / "A1" / "part_options.json"),
    ):
        saved_marker = marker.read_bytes()
        marker.unlink()
        with pytest.raises(ValueError, match="holds parts made with"):
            ops_engine.run_ops(new_settings, phases=(phase,))
        assert not marker.exists()
        marker.write_bytes(saved_marker)
    assert _files(new_root / "nas") == before
