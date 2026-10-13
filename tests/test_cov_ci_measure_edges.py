"""Measurement adapters preserve unmatched objects and narrow wound bands."""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from spacr import measure


def test_an_uncroppable_nucleus_keeps_its_row_without_a_png(
        tmp_path, monkeypatch):
    from spacr import crop_source

    merged = tmp_path / "merged"
    merged.mkdir()
    image = np.zeros((8, 8, 2), dtype=np.uint16)
    image[2:5, 2:5, 1] = 1
    np.save(merged / "plate1_A01_1.npy", image)
    nuclei = pd.DataFrame([{
        "path_name": "plate1_A01_1.npy", "object_label": 1,
        "plateID": "plate1", "rowID": "A", "columnID": "01", "fieldID": "1",
    }])
    measured = pd.DataFrame({"dna_mean": [1.0]})
    monkeypatch.setattr(crop_source, "crop_at", lambda *_args, **_kwargs: None)

    paths = measure._write_nucleus_crops(
        str(tmp_path), nuclei, measured, {"nucleus_mask_dim": 1},
        channel=0, size=8, folder=str(tmp_path / "crops"))

    assert paths.isna().all()
    assert list((tmp_path / "crops").iterdir()) == []


def test_an_unavailable_mask_plane_leaves_a_cellprofiler_object_unmatched(
        tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    first = np.ones((4, 4, 2), dtype=np.uint16)
    second = np.ones((4, 4, 1), dtype=np.uint16)
    np.save(merged / "plate1_A01_1.npy", first)
    np.save(merged / "plate1_A01_2.npy", second)
    table_path = tmp_path / "nuclei.npy"
    np.save(table_path, np.array([[2, 1, 2, 2]], dtype=float))
    reply = {
        "images": {"1": ["plate1_A01_1_ch0.tif"],
                   "2": ["plate1_A01_2_ch0.tif"]},
        "objects": {"Nuclei": {
            "columns": ["ImageNumber", "ObjectNumber",
                        "Location_Center_X", "Location_Center_Y"],
            "path": str(table_path),
        }},
    }

    result = measure._cellprofiler_tables(reply, str(merged), {
        "cell_mask_dim": 0, "nucleus_mask_dim": 1,
        "timelapse": False,
    })["cellprofiler_nuclei"]

    assert result["object_type"].tolist() == ["nucleus"]
    assert result["object_label"].isna().all()
    assert result["prcfo"].isna().all()


@pytest.mark.parametrize("can_close", [True, False])
def test_a_non_array_cellprofiler_plane_is_rejected_and_closed_when_possible(
        monkeypatch, can_close):
    closed = []

    class UnsupportedPlane:
        def close(self):
            closed.append(True)

    plane = UnsupportedPlane() if can_close else object()
    monkeypatch.setattr(measure.np, "load", lambda *_args, **_kwargs: plane)
    mapping = measure._cellprofiler_overlap_labels(
        ["not-an-array.npy"], np.ones((2, 2), dtype=np.uint16))

    assert mapping == {}
    assert closed == ([True] if can_close else [])


def test_a_small_wound_band_does_not_fit_an_otsu_split(monkeypatch):
    plane = np.ones((10, 30), dtype=float)
    distances = np.ones_like(plane)
    distances[:10, :10] = 0
    monkeypatch.setattr(measure, "_wound_signal", lambda *_args: plane)
    monkeypatch.setattr(measure, "_wound_across", lambda *_args: distances)
    monkeypatch.setattr(measure, "_wound_unsaturated",
                        lambda *_args: np.ones_like(plane, dtype=bool))

    def unexpected_split(_band):
        pytest.fail("a 100-pixel band must not be split")

    monkeypatch.setattr(measure, "_otsu_separation", unexpected_split)
    axis = types.SimpleNamespace(half_band=0.5)
    assert measure._wound_relevel(plane, axis, 1.0, 100, 2) == pytest.approx(1.0)
