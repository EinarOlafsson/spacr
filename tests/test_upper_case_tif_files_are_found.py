"""Raw images with an upper-case or mixed-case extension are read like any other."""
from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
tifffile = pytest.importorskip("tifffile")

_NAME = "plate1_A01_T0001F001L01A01Z01C0{c}{ext}"


def _plate(folder, ext):
    folder.mkdir()
    rng = np.random.default_rng(0)
    for channel in (1, 2):
        data = rng.integers(0, 4000, (32, 32)).astype(np.uint16)
        tifffile.imwrite(str(folder / _NAME.format(c=channel, ext=ext)), data)
    return folder


def _settings(src):
    from spacr.settings import set_default_settings_preprocess_generate_masks

    return set_default_settings_preprocess_generate_masks({
        "src": str(src), "metadata_type": "cellvoyager", "custom_regex": None,
        "channels": [0, 1], "nucleus_channel": 0, "cell_channel": 1,
        "pathogen_channel": None, "organelle_channel": None, "plot": False,
        "batch_size": 1, "test_mode": False, "timelapse": False,
        "normalize": True, "randomize": False, "verbose": False,
    })


@pytest.mark.parametrize("ext", [".TIF", ".Tif", ".tif", ".TIFF"])
def test_raw_image_names_ignores_case(tmp_path, ext):
    from spacr.io import _raw_image_names

    folder = _plate(tmp_path / "plate", ext)
    (folder / ".hidden.TIF").write_bytes(b"")
    assert len(_raw_image_names(str(folder))) == 2
    assert len(_raw_image_names(str(folder), ".tif")) == (2 if ext.lower() == ".tif" else 0)


@pytest.mark.parametrize("ext", [".TIF", ".Tif"])
def test_preprocess_builds_stacks_from_upper_case_tif(tmp_path, ext):
    from spacr.io import preprocess_img_data

    folder = _plate(tmp_path / "plate", ext)
    preprocess_img_data(_settings(folder))
    stacks = list((folder / "stack").glob("*.npy"))
    assert len(stacks) == 1
    assert np.load(stacks[0]).shape[-1] == 2


@pytest.mark.parametrize("metadata_type", ["cellvoyager", "auto"])
def test_verbatim_patterns_match_any_extension_case(metadata_type):
    import re
    from spacr.utils import _get_regex

    pattern = re.compile(_get_regex(metadata_type, "tif"))
    probe = {"cellvoyager": "plate1_A01_T0001F001L01A01Z01C01",
             "auto": "plate1_A01_T0001F001L01C01"}[metadata_type]
    assert all(pattern.fullmatch(probe + ext) for ext in (".tif", ".TIF", ".Tif"))
