"""Throughput benchmarks of the OPS phenotype run (item 872, Part A tests 4).

Every test here is marked ``slow`` and is left out of the default run. The
field benchmarks also need the Funk et al. 2022 plate A pilot field A1/514,
named by environment variables, and are skipped without it:

* ``SPACR_FUNK_A1_514_STACK``: the field's max-projected ``(H, W, 4)`` stack
  as a ``.npy`` file (DAPI, GFP, A594, AF750);
* ``SPACR_FUNK_A1_514_MASKS``: ``phenotype_masks/site-514.npz`` of the
  reference run (``nuclei`` and ``cells``).

Only ratios are asserted, never absolute seconds; the times are printed so
the run's log records them.
"""
from __future__ import annotations

import os
import sqlite3
import time

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow

CHANNELS = ("DAPI", "GFP", "A594", "AF750")


def _a1_514():
    """The pilot field's stack and masks, or a skip."""
    stack = os.environ.get("SPACR_FUNK_A1_514_STACK")
    masks = os.environ.get("SPACR_FUNK_A1_514_MASKS")
    if not (stack and masks and os.path.isfile(stack) and os.path.isfile(masks)):
        pytest.skip("set SPACR_FUNK_A1_514_STACK and SPACR_FUNK_A1_514_MASKS "
                    "to the plate A pilot field A1/514")
    with np.load(masks) as saved:
        return (np.load(stack).astype(np.float32),
                saved["nuclei"].astype(np.int32), saved["cells"].astype(np.int32))


def _measure_seconds(stack, nuclei, cells):
    """Seconds to measure the field on the CPU, and its table."""
    from spacr.ops_sample import measure_phenotype_field

    tick = time.perf_counter()
    table = measure_phenotype_field(nuclei, cells, stack, CHANNELS, texture=True)
    return time.perf_counter() - tick, table


def test_bounding_box_loops_halve_the_time_per_field(monkeypatch):
    """Part A1 test 4: the per-object loops at most half the whole-field ones."""
    from spacr import measure

    stack, nuclei, cells = _a1_514()
    fast, table = _measure_seconds(stack, nuclei, cells)
    monkeypatch.setattr(measure, "_label_bounding_boxes", lambda mask: {})
    slow, reference = _measure_seconds(stack, nuclei, cells)
    print(f"A1/514 measure: {fast:.1f} s with boxes, {slow:.1f} s without")
    pd.testing.assert_frame_equal(table, reference, check_exact=True)
    assert fast <= 0.5 * slow


def test_one_transaction_per_frame_is_twenty_times_faster(tmp_path):
    """Part A2 test 4: 10,000 rows, one BEGIN against autocommit per row."""
    from spacr.utils import _insert_frame

    frame = pd.DataFrame({"prcfo": [f"p_{i}" for i in range(10_000)],
                          "value": np.arange(10_000, dtype=float)})

    def seconds(path, per_row):
        """Insert ``frame`` into a new file database and time it."""
        conn = sqlite3.connect(str(path), isolation_level=None)
        conn.execute("CREATE TABLE t (prcfo TEXT, value REAL)")
        tick = time.perf_counter()
        if per_row:
            for row in frame.itertuples(index=False):
                conn.execute("INSERT INTO t VALUES (?, ?)", tuple(row))
        else:
            _insert_frame(conn, "t", frame)
        took = time.perf_counter() - tick
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 10_000
        conn.close()
        return took

    framed = seconds(tmp_path / "framed.db", False)
    autocommit = seconds(tmp_path / "rows.db", True)
    print(f"10,000 rows: {framed:.2f} s in one frame, {autocommit:.1f} s row by row")
    assert autocommit >= 20 * framed

