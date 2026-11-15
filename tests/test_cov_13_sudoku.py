"""Propagation over a stale or emptied graph returns no mass.

``sudoku_all`` slices the cell pool each round and rebuilds the graph from the
survivors, so :func:`spacr.sudoku.propagate` must answer zeros for a graph of
the wrong size and an empty answer for an emptied pool. The assignment
itself is covered by tests/test_sudoku_degenerate_screens.py.
"""
from __future__ import annotations

import numpy as np

from spacr.sudoku import propagate, similarity_graph


def test_propagating_into_a_graph_of_the_wrong_size_returns_no_mass():
    """Mismatched shapes must yield zeros, never a broadcast answer.

    ``sudoku_all`` slices the cell pool each round and rebuilds the graph from
    the survivors. A stale graph reaching this with the previous round's size
    would otherwise propagate one round's anchors onto another round's cells.
    """
    graph = similarity_graph(np.random.default_rng(0).normal(size=(6, 2)),
                             neighbours=3)
    seeds = np.zeros((4, 2), dtype=np.float32)
    seeds[0, 0] = 1.0

    mass = propagate(graph, seeds)

    assert mass.shape == (4, 2)
    assert not mass.any()


def test_propagating_no_cells_at_all_returns_an_empty_answer():
    """An emptied pool is a normal end state for the sequential rounds."""
    from scipy import sparse

    mass = propagate(sparse.csr_matrix((0, 0)), np.zeros((0, 3)))

    assert mass.shape == (0, 3)
