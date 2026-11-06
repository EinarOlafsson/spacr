"""Item 774: the objects Measure measured are the objects crops are cut from.

Measure edits the label planes in memory: it merges two cells that share a
nucleus, drops objects outside the size limits and applies the uninfected
rule. Those edits used to stay in memory, so a crop cut by ``object_label``
from ``merged/`` could be a smaller or a different object than the one in
the measurements table. Measure now records every changed plane beside the
merged file, :class:`spacr.crops.MergedField` serves the recorded plane, and
the merged file itself keeps its bytes.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import sqlite3

import numpy as np
import pytest

from spacr import crops
from spacr.settings import get_measure_crop_settings

SIZE = 96


def _field():
    """Two touching cells sharing one nucleus, one lone cell, one speck.

    Cells 1 and 2 are the left and right halves of a block; the nucleus
    straddles their border half and half, so Measure merges cell 2 into
    cell 1. Cell 4 is far below the size limit and is dropped.
    """
    rng = np.random.default_rng(0)
    cell = np.zeros((SIZE, SIZE), np.uint16)
    nucleus = np.zeros((SIZE, SIZE), np.uint16)
    cell[10:40, 10:30] = 1
    cell[10:40, 30:50] = 2
    nucleus[20:30, 25:35] = 1
    cell[60:85, 60:85] = 3
    nucleus[68:76, 68:76] = 2
    cell[88:90, 5:7] = 4
    planes = [rng.integers(100, 400, (SIZE, SIZE)).astype(np.uint16)
              + (cell > 0).astype(np.uint16) * 2000 * (c + 1)
              for c in range(2)]
    return np.stack(planes + [cell, nucleus], axis=-1).astype(np.uint16)


def _settings(merged_dir, **over):
    settings = get_measure_crop_settings(settings={})
    settings.update({
        "src": str(merged_dir), "channels": [0, 1],
        "cell_mask_dim": 2, "nucleus_mask_dim": 3, "pathogen_mask_dim": None,
        "save_measurements": True, "save_png": False, "save_arrays": False,
        "plot": False, "verbose": False, "timelapse": False,
        "experiment": "exp", "n_jobs": 1, "test_mode": False,
        "cytoplasm": False, "cell_min_size": 20, "nucleus_min_size": 1,
        "pathogen_min_size": 0, "cytoplasm_min_size": 0,
        "radial_dist": False, "calculate_correlation": False,
        "homogeneity": False,
    })
    settings.update(over)
    return settings


def _run(tmp_path, data=None, **over):
    from spacr.measure import _measure_crop_core

    root = tmp_path / "plate1"
    merged = root / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    (root / "measurements").mkdir(parents=True, exist_ok=True)
    name = "plate1_A01_F001.npy"
    path = merged / name
    if data is not None or not path.exists():
        np.save(path, _field() if data is None else data)
    crops.clear_field_cache()
    index, _, _cells, _, error = _measure_crop_core(
        0, [], name, _settings(merged, **over))
    assert index == 0, error
    return root, str(path)


def _measured_areas(root, table="cell"):
    db = os.path.join(root, "measurements", "measurements.db")
    with contextlib.closing(sqlite3.connect(db, timeout=30.0)) as connection:
        rows = connection.execute(
            f'SELECT object_label, {table}_area FROM "{table}"').fetchall()
    return {int(label): float(area) for label, area in rows}


def _sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def test_the_measured_area_equals_the_stored_label_area(tmp_path):
    before = _field()
    root, path = _run(tmp_path)
    measured = _measured_areas(root)

    assert sorted(measured) == [1, 3]
    assert measured[1] == float(np.count_nonzero(np.isin(before[..., 2], [1, 2])))

    field = crops.open_merged_field(path, mask_dims={"cell": 2, "nucleus": 3})
    assert field.labels("cell") == sorted(measured)
    index = field.label_index("cell")
    for label, area in measured.items():
        assert index.area(label) == area
        y0, y1, x0, x1 = index.bbox(label)
        window = field.read_mask_window("cell", y0, y1, x0, x1)
        assert int(np.count_nonzero(window == label)) == area


def test_the_merged_file_keeps_its_bytes(tmp_path):
    root, path = _run(tmp_path)
    first = _sha(path)
    record = crops._measured_masks_path(path)
    assert os.path.isfile(record)
    assert np.array_equal(np.load(path), _field())

    _run(tmp_path)
    assert _sha(path) == first


def test_a_rerun_with_other_limits_starts_from_the_segmentation(tmp_path):
    _run(tmp_path)
    root, path = _run(tmp_path, cell_min_size=1)

    measured = _measured_areas(root)
    assert 4 in measured and measured[4] == 4.0
    field = crops.open_merged_field(path, mask_dims={"cell": 2, "nucleus": 3})
    assert field.label_index("cell").area(4) == 4


def test_an_unedited_field_has_no_record_and_a_stale_one_is_removed(tmp_path):
    lone = _field()
    lone[..., 2][np.isin(lone[..., 2], [2, 4])] = 0
    lone[..., 3][lone[..., 3] == 1] = 0
    root, path = _run(tmp_path, data=lone)
    assert not os.path.exists(crops._measured_masks_path(path))

    _run(tmp_path, data=_field())
    assert os.path.exists(crops._measured_masks_path(path))
    _run(tmp_path, data=lone)
    assert not os.path.exists(crops._measured_masks_path(path))


def test_a_resegmented_field_ignores_the_old_record(tmp_path):
    root, path = _run(tmp_path)
    resegmented = _field()
    resegmented[..., 2][resegmented[..., 2] == 4] = 0
    resegmented[..., 2][40:45, 10:50] = 5
    np.save(path, resegmented)
    crops.clear_field_cache()

    field = crops.open_merged_field(path, mask_dims={"cell": 2, "nucleus": 3})
    assert field.labels("cell") == [1, 2, 3, 5]


def test_a_cached_field_sees_a_new_record(tmp_path):
    root = tmp_path / "plate1" / "merged"
    root.mkdir(parents=True)
    path = str(root / "plate1_A01_F001.npy")
    np.save(path, _field())
    crops.clear_field_cache()
    dims = {"cell": 2, "nucleus": 3}
    assert crops.open_merged_field(path, mask_dims=dims).labels("cell") == [1, 2, 3, 4]

    _run(tmp_path)
    assert crops.open_merged_field(path, mask_dims=dims).labels("cell") == [1, 3]


def test_the_recorded_cytoplasm_is_the_measured_one(tmp_path):
    root, path = _run(tmp_path, cytoplasm=True, cytoplasm_min_size=150)
    measured = _measured_areas(root, "cytoplasm")
    field = crops.open_merged_field(path, mask_dims={"cell": 2, "nucleus": 3})
    index = field.label_index("cytoplasm")

    assert field.labels("cytoplasm") == sorted(measured)
    for label, area in measured.items():
        assert index.area(label) == area


@pytest.mark.parametrize("pathogen", [False, True])
def test_the_in_memory_cell_plane_is_the_measured_one(tmp_path, pathogen):
    """PNG crops come from the in-memory field; it carries the merged cell."""
    from spacr import measure as M

    data = _field()
    over = {}
    if pathogen:
        pathogen_plane = np.zeros((SIZE, SIZE), np.uint16)
        pathogen_plane[15:18, 15:18] = 1
        pathogen_plane[70:73, 70:73] = 2
        data = np.concatenate([data, pathogen_plane[..., None]], axis=-1)
        over = {"pathogen_mask_dim": 4, "uninfected": True,
                "cytoplasm": True}
    seen = {}
    real = M._record_measured_masks

    def spy(merged_path, settings, masks, cytoplasm_mask=None):
        seen["cell"] = np.array(masks["cell"])
        return real(merged_path, settings, masks, cytoplasm_mask)

    import unittest.mock as mock
    with mock.patch.object(M, "_record_measured_masks", spy):
        root, path = _run(tmp_path, data=data, **over)
    measured = _measured_areas(root)
    assert sorted(measured) == [1, 3]
    for label, area in measured.items():
        assert int(np.count_nonzero(seen["cell"] == label)) == area
    field = crops.open_merged_field(path, mask_dims={"cell": 2, "nucleus": 3})
    assert field.label_index("cell").area(1) == measured[1]
