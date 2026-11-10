"""The OPS phenotype measurement's GPU and CPU skew and kurtosis agree (item 872, B13).

Item 788 made both paths compute the moments in float64; this pins the
agreement through the OPS phenotype path itself, with the self-check of
B7: one synthetic field with bright objects of large mean and small
variance (float32's worst case) is measured by :func:`_ph_gpu_check` on the
"card" and on the CPU. The GPU kernels run on the torch CPU device, which
executes the same code, so no GPU is needed.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyarrow")

from skimage.segmentation import expand_labels                     # noqa: E402

from spacr import measure, ops_engine                               # noqa: E402

CPU = torch.device("cpu")


def test_the_ops_gpu_check_finds_skew_and_kurtosis_within_1e_6(
        tmp_path, monkeypatch):
    monkeypatch.setattr(measure, "_measurement_device",
                        lambda settings: CPU if settings.get("measure_gpu")
                        else None)
    rng = np.random.default_rng(7)
    size = 160
    nuclei = np.zeros((size, size), np.int32)
    for label, (r, c) in enumerate(rng.integers(12, size - 12, (25, 2)), 1):
        nuclei[r, c] = label
    nuclei = expand_labels(nuclei, 4)
    cells = expand_labels(nuclei, 6)
    stack = np.stack([30000.0 + rng.gamma(2.0, 2.0, (size, size))
                      for _ in range(3)], axis=-1).astype(np.float32)
    stack[..., 0] += 2000.0 * (nuclei > 0)
    path = tmp_path / "stack.npy"
    np.save(path, stack)
    masks = tmp_path / "site-0.npz"
    np.savez(masks, nuclei=nuclei.astype(np.uint32), cells=cells.astype(np.uint32))
    task = {"site": 0, "files": {}, "channels": ("DAPI", "GFP", "AF750"),
            "mask_path": str(masks), "cell_path": str(tmp_path / "c.parquet"),
            "texture": False, "staged": str(path), "measure_gpu": True}
    check = ops_engine._ph_gpu_check(task, rtol=1e-6)
    assert check["only_in_test"] == [] and check["only_in_reference"] == []
    assert check["rows"] == int(cells.max())
    outside = {entry["column"] for entry in check["outside"]}
    shape_columns = [c for c in outside if "skew" in c or "kurtosis" in c]
    assert shape_columns == []
    assert check["columns"] > 100


def test_the_gpu_check_skips_an_unreadable_field(tmp_path):
    task = {"site": 3, "files": {"DAPI": str(tmp_path / "missing.tif")},
            "channels": ("DAPI",), "mask_path": "", "cell_path": "",
            "texture": False}
    assert ops_engine._ph_gpu_check(task) == {
        "site": 3, "skipped": "the field is unreadable"}
