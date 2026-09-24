"""Automatic conversion must not overwrite or renumber an existing plate."""

import numpy as np
import pytest
import tifffile


@pytest.mark.parametrize("existing", [
    "plate1_A01_T0001F001L01C01.tif",
    "plate2_B03_T0002F002L01A02Z03C01.tif",
    "plate1_AA13_T0001F001L01C01.TIF",
])
@pytest.mark.parametrize("has_log", [False, True])
def test_existing_converted_images_refuse_before_any_write(tmp_path, existing, has_log):
    from spacr.io import convert_to_yokogawa

    tifffile.imwrite(tmp_path / "a_raw.tif", np.full((8, 8), 7, np.uint16))
    tifffile.imwrite(tmp_path / existing, np.full((8, 8), 42, np.uint16))
    if has_log:
        (tmp_path / "rename_log.csv").write_text("Original File,Renamed TIFF\nprevious.tif," + existing + "\n")
        (tmp_path / "rename_log.run_status.json").write_text('{"previous": true}')
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    with pytest.raises(ValueError, match="already contains converted images") as error:
        convert_to_yokogawa(str(tmp_path))

    assert existing in str(error.value)
    assert "cellvoyager" in str(error.value)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_second_conversion_preserves_first_mapping_and_pixels(tmp_path):
    from spacr.io import convert_to_yokogawa

    tifffile.imwrite(tmp_path / "raw.tif", np.full((8, 8), 12, np.uint16))
    convert_to_yokogawa(str(tmp_path))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    with pytest.raises(ValueError, match="already contains converted images"):
        convert_to_yokogawa(str(tmp_path))

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
