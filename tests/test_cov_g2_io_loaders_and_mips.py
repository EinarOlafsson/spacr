"""io edges: mixed-dtype z planes, guarded loader workers, empty Cellpose sets."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile
from PIL import Image

from spacr import io
from tests.test_622_mask_ingest_streaming import PATTERN


def test_z_planes_of_different_dtypes_project_without_overflow(tmp_path):
    for z, (dtype, value) in enumerate(((np.uint8, 200), (np.uint16, 1000)), 1):
        tifffile.imwrite(tmp_path / f"plate_A01_T1F000Z{z}C1.tif",
                         np.full((8, 8), value, dtype))
    io._rename_and_organize_image_files(str(tmp_path), PATTERN,
                                        metadata_type="custom",
                                        save_original_images=False)
    stack = np.load(next((tmp_path / "stack").glob("*.npy")))
    assert int(stack.max()) == 1000


@pytest.fixture
def guarded(monkeypatch):
    import spacr.resource_log as resource_log

    calls = []
    monkeypatch.setattr(resource_log, "_guard_workers",
                        lambda task, workers, unit, **k: calls.append(workers) or 0)
    return calls


def _crops(root):
    for name in ("nc", "pc"):
        folder = root / "train" / name
        folder.mkdir(parents=True)
        for well in ("A01", "A02", "A03", "A04"):
            Image.new("RGB", (8, 8), (30, 80, 200)).save(
                folder / f"plate1_{well}_1_o1.png")


def test_loaders_guard_the_workers_they_ask_for(tmp_path, guarded):
    _crops(tmp_path)
    io.generate_loaders(str(tmp_path), classes=["nc", "pc"], n_jobs=2,
                        image_size=8, validation_split=0.25)
    io.generate_cv_loaders(str(tmp_path), 2, batch_size=2, image_size=8,
                           n_jobs=3)
    assert 2 in guarded and 3 in guarded

