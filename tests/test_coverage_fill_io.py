"""Coverage-fill for spacr.io pure-logic helpers (no GPU)."""
from __future__ import annotations


import numpy as np

import matplotlib
matplotlib.use("Agg")

from spacr import io as IO


# ---------------------------------------------------------------------------
# _read_mask / _read_db
# ---------------------------------------------------------------------------

def test_read_mask_tif(tmp_path):
    import tifffile
    p = tmp_path / "m.tif"
    tifffile.imwrite(str(p), np.ones((8, 8), dtype=np.uint16))
    out = IO._read_mask(str(p))
    assert out.dtype == np.uint16 and out.shape == (8, 8)


# ---------------------------------------------------------------------------
# _get_avg_object_size
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _generate_time_lists
# ---------------------------------------------------------------------------

def test_generate_time_lists_groups_by_field():
    files = [
        "plate1_A01_1_0.npy", "plate1_A01_1_2.npy", "plate1_A01_1_1.npy",
        "plate1_A02_1_0.npy",
        "malformed.npy",           # skipped (too few parts)
        "plate1_A01_1_bad.npy",    # skipped (bad timepoint)
    ]
    out = IO._generate_time_lists(files)
    # Two groups: (plate1,A01,1) and (plate1,A02,1).
    assert len(out) == 2
    # The A01 group is sorted by timepoint.
    a01 = [g for g in out if len(g) == 3][0]
    assert a01[0].endswith("_0.npy")


# ---------------------------------------------------------------------------
# parse_gz_files
# ---------------------------------------------------------------------------

def test_parse_gz_files(tmp_path):
    for name in ("s1_R1_001.fastq.gz", "s1_R2_001.fastq.gz",
                 "s2_R1_001.fastq.gz"):
        (tmp_path / name).write_bytes(b"")
    out = IO.parse_gz_files(str(tmp_path))
    assert out["s1"]["R1"].endswith("s1_R1_001.fastq.gz")
    assert out["s1"]["R2"].endswith("s1_R2_001.fastq.gz")
    assert "R1" in out["s2"]


# ---------------------------------------------------------------------------
# convert_numpy_to_tiff
# ---------------------------------------------------------------------------

def test_convert_numpy_to_tiff(tmp_path):
    np.save(str(tmp_path / "a.npy"),
            np.ones((8, 8), dtype=np.uint16))
    IO.convert_numpy_to_tiff(str(tmp_path))
    tiffs = list(tmp_path.rglob("*.tif")) + list(tmp_path.rglob("*.tiff"))
    assert tiffs


# ---------------------------------------------------------------------------
# _read_and_join_tables
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _save_figure + _save_settings_to_db
# ---------------------------------------------------------------------------

def test_save_figure(tmp_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    # _save_figure writes to dirname(src)/figure/ — give a nested src
    # so the PDF lands inside tmp_path.
    src = str(tmp_path / "cell" / "objroot")
    IO._save_figure(fig, src, text="test")
    assert list(tmp_path.rglob("*.pdf"))


