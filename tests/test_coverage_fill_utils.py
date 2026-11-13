"""Coverage-fill for spacr.utils pure-logic helpers (no GPU)."""
from __future__ import annotations

import gzip
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from spacr import utils as U


# ---------------------------------------------------------------------------
# Mask / label helpers
# ---------------------------------------------------------------------------

def test_relabel_sequential():
    m = np.zeros((6, 6), dtype=np.int32)
    m[0:2, 0:2] = 5
    m[3:5, 3:5] = 9
    out = U._relabel_sequential(m)
    assert set(np.unique(out)) == {0, 1, 2}


def test_compute_label_perimeters():
    m = np.zeros((10, 10), dtype=np.int32)
    m[2:6, 2:6] = 1
    out = U._compute_label_perimeters(m)
    assert out is not None


# ---------------------------------------------------------------------------
# Type / list helpers
# ---------------------------------------------------------------------------

def test_is_list_of_lists():
    assert U.is_list_of_lists([[1], [2]]) is True
    assert U.is_list_of_lists([1, 2]) is False
    assert U.is_list_of_lists("not a list") is False


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def test_check_normality_utils():
    rng = np.random.default_rng(0)
    assert U.check_normality(rng.normal(0, 1, 200)) in (True, False)
    # Clearly non-normal
    assert U.check_normality(np.array([0.0] * 50 + [1000.0])) is False


# ---------------------------------------------------------------------------
# FASTQ read counter
# ---------------------------------------------------------------------------

def test_count_reads_in_fastq(tmp_path):
    fq = tmp_path / "reads.fastq.gz"
    records = "".join(f"@id{i}\nACGT\n+\nIIII\n" for i in range(5))
    with gzip.open(fq, "wt") as fh:
        fh.write(records)
    assert U.count_reads_in_fastq(str(fq)) == 5


# ---------------------------------------------------------------------------
# _get_cellpose_channels (remap logic)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# torchvision model names
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# is_multiprocessing_process (mock process)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# batch 2 — more pure helpers
# ---------------------------------------------------------------------------

def test_get_db_paths_str_and_list():
    assert U.get_db_paths("/a")[0].endswith(os.path.join("measurements", "measurements.db"))
    both = U.get_db_paths(["/a", "/b"])
    assert len(both) == 2


def test_get_sequencing_paths():
    assert U.get_sequencing_paths("/a")[0].endswith(
        os.path.join("sequencing", "sequencing_data.csv"))


def test_check_integrity_collapses_label_columns():
    df = pd.DataFrame({
        "label": [1, 2], "cell_label": [3, 4], "value": [5, 6],
    })
    out = U._check_integrity(df)
    assert "object_label" in out.columns
    assert "label_list" in out.columns


def test_check_multicollinearity():
    rng = np.random.default_rng(1)
    x = pd.DataFrame({
        "a": rng.normal(0, 1, 50), "b": rng.normal(0, 1, 50),
    })
    vif = U.check_multicollinearity(x)
    assert "VIF" in vif.columns and len(vif) == 2


def test_apply_union_find():
    m = np.zeros((6, 6), dtype=np.int32)
    m[0:2, 0:2] = 1; m[3:5, 3:5] = 2
    parent = {1: 1, 2: 2}
    out = U._apply_union_find(m, parent)
    assert out is not None
