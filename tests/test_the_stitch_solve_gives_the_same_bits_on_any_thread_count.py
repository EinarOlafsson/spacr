"""A stitch rerun must store byte-identical ``ops_geometry`` (S1 (e)).

The shifts the registration measures are integers, so the only floating
point in the stitch is the least-squares solve. A multithreaded BLAS made
that solve's last bits depend on the thread count -- 7e-11 to 1.7e-10 px on
a 333-tile well -- and no rerun matched production. The solve now runs on
one BLAS thread. Pinned here: the same bits whatever the caller's thread
count, the caller's count restored afterwards, and the answer within 1e-6
px of the threaded solve it replaced.

Only the solve is pinned; the process keeps its thread pools. Identity
holds per machine and BLAS build, since another CPU kernel of the same BLAS
may round the last bits differently. Without threadpoolctl the solve runs
threaded, as before.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from spacr.ops_layout import round_well_layout
from spacr.ops_solve import solve_placements

threadpoolctl = pytest.importorskip("threadpoolctl")

PITCH = 1300


def _edges(sites: int = 333, seed: int = 0):
    """A round well's raster edges with integer shifts and the stage's skew.

    333 tiles, the measured well, so the design matrix is large enough for
    OpenBLAS to split the work across threads, which a toy grid is not.
    """
    rng = np.random.default_rng(seed)
    edges = {}
    for a, b, axis in round_well_layout(sites).pairs():
        along = PITCH + int(rng.integers(-3, 4))
        across = int(rng.integers(-9, 10))
        edges[(a, b)] = (along, across) if axis == "rows" else (across, along)
    return edges, list(range(sites))


def _array(placed, sites):
    """The placements as an ``(n, 2)`` array in site order."""
    return np.array([placed[site] for site in sites], dtype=np.float64)


def test_one_and_four_threads_give_the_same_bits():
    """The caller's BLAS thread count does not reach the stored positions."""
    edges, sites = _edges()
    answers = []
    for threads in (1, 4):
        with threadpoolctl.threadpool_limits(limits=threads, user_api="blas"):
            answers.append(_array(solve_placements(edges, sites), sites))
    assert answers[0].tobytes() == answers[1].tobytes()


def test_repeated_solves_give_the_same_bits():
    """Ten solves at four threads, one answer."""
    edges, sites = _edges(seed=1)
    with threadpoolctl.threadpool_limits(limits=4, user_api="blas"):
        seen = {_array(solve_placements(edges, sites), sites).tobytes()
                for _ in range(10)}
    assert len(seen) == 1


def test_the_callers_thread_count_is_restored():
    """Only the solve is pinned; the process keeps its pools."""
    edges, sites = _edges(seed=3)

    def counts():
        return sorted(info["num_threads"]
                      for info in threadpoolctl.threadpool_info()
                      if info["user_api"] == "blas")

    with threadpoolctl.threadpool_limits(limits=2, user_api="blas"):
        before = counts()
        solve_placements(edges, sites)
        assert counts() == before


def test_the_pinned_solve_stays_within_a_micropixel_of_the_threaded_one(
        monkeypatch):
    """The fix changes rounding, not geometry.

    Hiding threadpoolctl runs the solve exactly as it ran before the pin,
    threaded, so this compares the new answer with the old solver's.
    """
    edges, sites = _edges(seed=2)
    pinned = _array(solve_placements(edges, sites), sites)
    monkeypatch.setitem(sys.modules, "threadpoolctl", None)
    with threadpoolctl.threadpool_limits(limits=4, user_api="blas"):
        threaded = _array(solve_placements(edges, sites), sites)
    assert np.abs(pinned - threaded).max() <= 1e-6
    assert pinned[0] == pytest.approx((0.0, 0.0), abs=1e-6)
