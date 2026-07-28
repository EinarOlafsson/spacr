import pandas as pd
import pytest


def _object_frame(table, stamp):
    row = {
        "plateID": "plate1",
        "rowID": "A",
        "columnID": "01",
        "fieldID": "1",
        "prcf": "plate1_A_01_1",
        "object_label": 1,
        f"{table}_area": 10.0,
        **stamp,
    }
    if table in {"nucleus", "pathogen"}:
        row["cell_id"] = 1
    return pd.DataFrame([row])


@pytest.mark.parametrize(
    "stamp",
    [
        {
            "measurement_ndim": 2,
            "measurement_units": "px",
            "n_z": 1,
            "voxel_size_z_um": None,
            "voxel_size_xy_um": None,
        },
        {
            "measurement_ndim": 3,
            "measurement_units": "um",
            "n_z": 5,
            "voxel_size_z_um": 1.5,
            "voxel_size_xy_um": 0.25,
        },
    ],
    ids=["2d", "3d"],
)
def test_read_and_merge_data_keeps_one_measurement_stamp(monkeypatch, stamp):
    """Four object tables must not create repeated pandas suffix columns."""
    from spacr import io
    from spacr.utils import MEASUREMENT_STAMP_COLUMNS

    tables = ("cell", "cytoplasm", "nucleus", "pathogen")
    frames = {table: _object_frame(table, stamp) for table in tables}

    def fake_read_db(_loc, requested_tables):
        return [frames[table].copy() for table in requested_tables]

    monkeypatch.setattr(io, "_read_db", fake_read_db)

    merged, object_frames = io._read_and_merge_data(
        ["unused.db"], list(tables), nuclei_limit=None, pathogen_limit=None
    )

    assert len(merged) == 1
    assert len(object_frames) == len(tables)
    assert not any(column.endswith(("_x", "_y")) for column in merged.columns)
    for column in MEASUREMENT_STAMP_COLUMNS:
        assert column in merged.columns
        actual = merged.iloc[0][column]
        expected = stamp[column]
        if expected is None:
            assert pd.isna(actual)
        else:
            assert actual == expected
