"""Overview measurements carry image-level features into fov and join annotations."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.measure import _image_measurement_frame, _set_image_annotations
from spacr.qt.widgets.qc_field_browser import (
    _OverviewField,
    _overview_measurement,
    _overview_measurements_export,
)
from spacr.sp_stats import _compare_image_groups, _representative_images

pytestmark = pytest.mark.qt


def _field(tmp_path, well, column, name, level):
    files = []
    for channel in (1, 2):
        pixels = np.full((32, 32), 100, dtype=np.uint16)
        pixels[8:16, 8:16] = level * channel
        path = tmp_path / f"{well}_{name}_c{channel}.npy"
        np.save(path, pixels)
        files.append((str(channel), str(path)))
    return _OverviewField("plate1", well, "A", column, name, tuple(files))


def test_measurement_row_holds_image_level_features(tmp_path):
    row = _overview_measurement(_field(tmp_path, "A01", 1, "1", 1000), (1, 2))
    assert row["status"] == "ok", row["error"]
    assert row["channel_1_objects_count"] == 1
    assert row["channel_1_otsu_area_fraction"] == pytest.approx(64 / 1024)
    assert row["channels_1_2_manders_m1"] == pytest.approx(1.0)
    assert row["object_source"] == "otsu_per_channel"
    assert row["channel_2_saturation_level"] == 65535


def test_exported_fields_join_annotations_for_comparison(tmp_path):
    fields = [_field(tmp_path, f"A0{column}", column, str(index), 400 + 300 * (column > 2) + index)
              for column in (1, 2, 3, 4) for index in range(3)]
    report = _overview_measurements_export(fields, tmp_path, (1, 2))
    assert report["count"] == 12 and not report["errors"]
    db = tmp_path / "measurements" / "measurements.db"
    _set_image_annotations(db, fields, "cell_type", "HeLa", source=tmp_path,
                           wells=[("plate1", "A01"), ("plate1", "A02")])
    _set_image_annotations(db, fields, "cell_type", "HFF", source=tmp_path,
                           wells=[("plate1", "A03"), ("plate1", "A04")])
    frame = _image_measurement_frame(db)
    assert len(frame) == 12 and frame["cell_type"].notna().all()
    result = _compare_image_groups(frame, ["channel_1_foreground_mean"], ["cell_type"])
    summary = result["summary"].set_index("group")
    assert summary.loc["HFF", "mean"] > summary.loc["HeLa", "mean"]
    ranked = _representative_images(frame, ["cell_type"], method="typical_in_focus", top=2)
    assert set(ranked["cell_type"]) == {"HFF", "HeLa"}
