"""Item 288: three crop lookups that cannot take their short cut.

* Re-anchoring a frame learns the prefix the first resolved row was moved
  by and reuses it for the rest. A row recorded under a DIFFERENT prefix
  skips that short cut and is still found by the full search; a row under
  the learned prefix whose file is not there is not swapped for some other
  file that does exist.
* ``open_merged_field`` reuses a cached field when the file cannot be
  fingerprinted, but only one for the same path -- another file's cached
  field is never handed back.
* Decoding a crop with a format number spaCR never wrote is refused by name.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from spacr import crops


def _crop(root, name):
    real = root / "plate1" / "data" / "cell_png" / name
    real.parent.mkdir(parents=True, exist_ok=True)
    real.write_bytes(b"\x89PNG\r\n\x1a\n")
    return str(real)


def test_rows_under_another_prefix_or_missing_are_handled_row_by_row(
        tmp_path):
    root = tmp_path / "screen"
    first = _crop(root, "object_1.png")
    second = _crop(root, "object_2.png")
    frame = pd.DataFrame({"png_path": [
        "/old/machine/plate1/data/cell_png/object_1.png",
        "/elsewhere/entirely/plate1/data/cell_png/object_2.png",
        "/old/machine/plate1/data/cell_png/object_3.png",
    ]})

    out, report = crops.reanchor_frame(frame, str(root))

    assert out["png_path"].iloc[0] == first
    assert out["png_path"].iloc[1] == second
    assert out["png_path"].iloc[2].endswith(
        os.path.join("data", "cell_png", "object_3.png"))
    assert not os.path.exists(out["png_path"].iloc[2])
    assert report.n_paths == 3


@pytest.fixture
def two_fields(tmp_path):
    paths = []
    for name in ("a.npy", "b.npy"):
        path = tmp_path / name
        np.save(path, np.zeros((6, 6, 5), np.uint16))
        paths.append(str(path))
    yield paths
    for path in paths:
        for key in [k for k in crops._FIELD_CACHE if k[0] == path]:
            crops._FIELD_CACHE.pop(key, None)


def test_an_unverifiable_file_never_gets_another_files_field(
        two_fields, monkeypatch):
    a, b = two_fields
    field_a = crops.open_merged_field(a)
    monkeypatch.setattr(crops, "_content_fingerprint", lambda path, size: None)

    field_b = crops.open_merged_field(b)
    assert field_b is not field_a
    assert os.path.abspath(field_b.path) == os.path.abspath(b)
    assert crops.open_merged_field(a) is field_a, (
        "the same path, unverifiable, reuses its own cached field")


def test_an_unknown_crop_format_is_refused_by_name():
    from PIL import Image

    image = Image.new("RGB", (4, 4))
    with pytest.raises(crops.CropError) as excinfo:
        crops.decode_crop_image(image, fmt=99)
    assert "unknown crop format 99" in str(excinfo.value)
