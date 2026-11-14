"""The smart-microscope helpers when their inputs are missing or odd."""
from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from spacr import core


def test_a_positions_file_needs_field_x_and_y(tmp_path):
    path = tmp_path / "positions.csv"
    pd.DataFrame({"field": [1], "x": [0.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="lacks the column"):
        core._microscope_positions({"microscope_positions": str(path)})


def _microscope(tmp_path, positions=None):
    import tifffile

    folder = tmp_path / "images"
    folder.mkdir()
    tifffile.imwrite(folder / "plate1_A01_T0001F001L01A01Z01C01.tif",
                     np.arange(2 * 6 * 6, dtype=np.uint16).reshape(2, 6, 6))
    return core._SimulatedMicroscope(
        str(folder), positions if positions is not None else {},
        np.eye(2), {"metadata_type": "cellvoyager"})


def test_the_simulated_microscope_reports_its_stage(tmp_path):
    scope = _microscope(tmp_path)
    scope.set_xy_position(3, 4)
    scope.set_position(7)
    assert (scope.get_x_position(), scope.get_y_position(),
            scope.get_position()) == (3.0, 4.0, 7.0)
    with pytest.raises(RuntimeError, match="snap_image was not called"):
        scope.get_image()


def test_a_simulated_microscope_with_no_known_field_refuses_to_snap(tmp_path):
    scope = _microscope(tmp_path)
    with pytest.raises(ValueError, match="found no image"):
        scope.snap_image()


def test_a_multi_plane_field_is_flattened_to_its_first_plane(tmp_path):
    scope = _microscope(tmp_path)
    path = str(tmp_path / "images" / "plate1_A01_T0001F001L01A01Z01C01.tif")
    assert scope._field(path).shape == (6, 6)
