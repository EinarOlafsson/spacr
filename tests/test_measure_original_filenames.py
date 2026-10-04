"""Measure adds original_* columns from a filename-conversion manifest.

A plate renamed by the converter keeps a ``conversion_map.csv`` that maps each
measured field back to the microscope file it came from. After a run every
measurement table with field identity gets ``original_filename``,
``original_path`` and the pre-conversion plate/well/field/channel/z/time.
Without a manifest the database is left exactly as Measure wrote it, and a
resumed run fills the rows it adds as well as the rows written before.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import measure
from spacr.schema import is_provenance_column

FIELDS = ("plate1_A01_1", "plate1_A01_2", "plate1_B02_1")
SOURCES = {
    "plate1_A01_1": ("A01", 1, "exp7/well_A1_pos3.nd2", "W1", "3"),
    "plate1_A01_2": ("A01", 2, "exp7/well_A1_pos4.nd2", "W1", "4"),
    "plate1_B02_1": ("B02", 1, "exp7/well_B2_pos1.nd2", "W2", "1"),
}


def _settings(merged, **over):
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings({})
    settings.update({
        "src": str(merged), "channels": [0, 1],
        "cell_mask_dim": 2, "nucleus_mask_dim": None,
        "pathogen_mask_dim": None, "cell_min_size": 0,
        "nucleus_min_size": 0, "pathogen_min_size": 0,
        "cytoplasm_min_size": 0, "save_png": False, "save_arrays": False,
        "plot": False, "verbose": False, "n_jobs": 1,
        "save_measurements": True,
    })
    settings.update(over)
    return settings


def _write_field(merged, name, seed):
    """Two intensity channels and four labelled cells away from the border."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((128, 128), dtype=np.uint16)
    for number, (y, x) in enumerate(((30, 30), (30, 90), (90, 30), (90, 90)), start=1):
        yy, xx = np.ogrid[:128, :128]
        labels[(yy - y) ** 2 + (xx - x) ** 2 <= 12 ** 2] = number
    image = rng.integers(100, 300, (128, 128, 2)).astype(np.uint16)
    image[labels > 0] += 800
    np.save(merged / f"{name}.npy", np.concatenate([image, labels[..., None]], axis=-1))


def _write_plate(root, fields=FIELDS):
    merged = root / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    for seed, name in enumerate(fields):
        _write_field(merged, name, seed)
    return merged


def _write_manifest(root):
    """The converter's map: two channels and two z planes per field."""
    rows = []
    for name, (well, field, source, source_well, source_field) in SOURCES.items():
        for channel in (1, 2):
            for z in (1, 2):
                rows.append({
                    "target": f"plate1_{well}_T0001F{field:03d}L01A01Z{z:02d}C{channel:02d}.tif",
                    "source": f"/raw/{source}", "source_relpath": source,
                    "plate": "plate1", "well": well, "field": field,
                    "channel": channel, "z": z, "t": 1,
                    "source_plate": "exp7", "source_well": source_well,
                    "source_field": source_field, "source_channel": f"ch{channel}",
                    "source_z": z, "source_t": 0, "status": "converted"})
    path = root / "conversion_map.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _tables(root):
    db = root / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        names = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        return {name: pd.read_sql_query(f'SELECT * FROM "{name}"', conn)
                for name in names}


def _assert_original_values(frame):
    for _, row in frame.iterrows():
        name = row["prcf"] if "file_name" not in frame else row["file_name"]
        stem = name if name in SOURCES else next(
            key for key, value in SOURCES.items()
            if row["prcf"] == "plate1_r{}_c{}_f{}".format(
                ord(value[0][0]) - 64, int(value[0][1:]), value[1]))
        well, _, source, source_well, source_field = SOURCES[stem]
        assert row["original_filename"] == source.rsplit("/", 1)[-1]
        assert row["original_path"] == f"/raw/{source}"
        assert row["original_relpath"] == source
        assert row["original_plate"] == "exp7"
        assert row["original_well"] == source_well
        assert row["original_field"] == source_field
        assert row["original_channel"] == "ch1; ch2"
        assert row["original_z"] == "1; 2"
        assert row["original_t"] == "0"


def test_every_table_with_field_identity_gets_original_columns(tmp_path):
    merged = _write_plate(tmp_path)
    _write_manifest(tmp_path)
    measure.measure_crop(_settings(merged))

    tables = _tables(tmp_path)
    keyed = {name: frame for name, frame in tables.items()
             if "prcf" in frame.columns and len(frame)}
    assert {"cell", "cytoplasm"} <= set(keyed)
    for name, frame in keyed.items():
        assert frame["original_filename"].notna().all(), name
        _assert_original_values(frame)
    assert set(keyed["cell"]["original_filename"]) == {
        "well_A1_pos3.nd2", "well_A1_pos4.nd2", "well_B2_pos1.nd2"}


def test_without_a_manifest_measure_adds_nothing(tmp_path):
    merged = _write_plate(tmp_path)
    measure.measure_crop(_settings(merged))

    for name, frame in _tables(tmp_path).items():
        assert not [column for column in frame.columns
                    if column.startswith("original_")
                    and column not in ("original_dtype", "original_intensity_max")], name
    db = tmp_path / "measurements" / "measurements.db"
    assert measure._add_original_filename_columns(str(db), str(merged)) == {}


def test_a_resumed_run_fills_old_and_new_rows(tmp_path):
    merged = _write_plate(tmp_path, FIELDS[:2])
    measure.measure_crop(_settings(merged, resume=True))
    cells = _tables(tmp_path)["cell"]
    assert "original_filename" not in cells.columns

    _write_manifest(tmp_path)
    _write_field(merged, FIELDS[2], 2)
    measure.measure_crop(_settings(merged, resume=True))

    cells = _tables(tmp_path)["cell"]
    assert set(cells["prcf"]) == {"plate1_r1_c1_f1", "plate1_r1_c1_f2",
                                  "plate1_r2_c2_f1"}
    assert cells["original_filename"].notna().all()
    _assert_original_values(cells)


def test_manifest_inside_the_merged_folder_and_unmatched_fields(tmp_path):
    merged = _write_plate(tmp_path)
    manifest = _write_manifest(tmp_path)
    frame = pd.read_csv(manifest)
    frame[frame.well == "A01"].to_csv(merged / "conversion_map.csv", index=False)
    manifest.unlink()
    measure.measure_crop(_settings(merged))

    cells = _tables(tmp_path)["cell"]
    matched = cells[cells["prcf"] != "plate1_r2_c2_f1"]
    assert matched["original_filename"].notna().all()
    assert cells.loc[cells["prcf"] == "plate1_r2_c2_f1",
                     "original_filename"].isna().all()
    _assert_original_values(matched)


@pytest.mark.parametrize("column", ["original_filename", "original_plate",
                                    "original_field", "original_z"])
def test_original_columns_are_provenance_not_features(column):
    assert is_provenance_column(column)


def test_merge_recomputes_names_measure_already_added(tmp_path):
    from spacr.derived_tables import _restore_original_filenames

    manifest = _write_manifest(tmp_path)
    frame = pd.DataFrame({"prcf": ["plate1_r1_c1_f1"], "original_filename": ["stale"],
                          "original_path": ["stale"]})
    result, _ = _restore_original_filenames(
        frame, {"original_filenames": {"map_path": str(manifest)}})
    assert result["original_filename"].tolist() == ["well_A1_pos3.nd2"]
