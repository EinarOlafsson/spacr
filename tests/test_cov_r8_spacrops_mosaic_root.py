"""The mosaic transform builder with no tiles to join, and a dead root guard.

`_compute_mosaic_transforms` keeps the edges that scored well enough,
builds a spanning tree over the tiles they connect, picks the
highest-degree tile as the root, and walks the tree computing each
tile's transform into the root's frame.

A stitch that found no usable pair is an ordinary outcome -- the tiles
may not overlap at all -- and it is answered at the TOP of the function.
That early answer is what makes the later `if root is None` guard
unreachable, and this file pins the two together.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from spacr.spacrops import spacrStitcher


@pytest.fixture()
def stitcher():
    """A stitcher without its constructor's detector setup.

    `__init__` builds an ORB detector and reads options this method
    never touches; the transform builder is arithmetic over the rows it
    is handed.
    """
    return spacrStitcher.__new__(spacrStitcher)


class TestWithNothingToStitch:

    def test_no_rows_gives_no_transforms_and_no_edges(self, stitcher):
        transforms, used = stitcher._compute_mosaic_transforms(
            [], min_score=0.5)
        assert transforms == {}
        assert used == []

    def test_the_answer_is_a_pair_whatever_happens(self, stitcher):
        """Callers unpack two values; an early return must keep that shape."""
        result = stitcher._compute_mosaic_transforms([], min_score=0.5)
        assert isinstance(result, tuple) and len(result) == 2
        transforms, used = result
        assert isinstance(transforms, dict) and isinstance(used, list)

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0, 10.0])
    def test_the_empty_answer_does_not_depend_on_the_threshold(
            self, stitcher, threshold):
        transforms, used = stitcher._compute_mosaic_transforms(
            [], min_score=threshold)
        assert transforms == {} and used == []


class TestTheRootGuardThatCannotFire:
    """`if root is None: return {}, used_edges` is unreachable.

    The function answers an empty tile set at the top:

        nodes = set()
        for r in rows:
            nodes.add(r["pathA"]); nodes.add(r["pathB"])
        nodes = sorted(nodes)
        if not nodes:
            return {}, []

    So by the time the root is chosen, `nodes` is never empty -- and
    `max(nodes, key=...)` on a non-empty sequence always returns a
    value. The `if nodes else None` beside it, and the guard below it,
    are both defending against a state the early return has already
    taken.

    Pinned from the producing side rather than forced: reaching it would
    mean emptying `nodes` between the two, which tests nothing about the
    program.
    """

    def test_an_empty_tile_set_is_answered_at_the_top(self, stitcher):
        source = inspect.getsource(stitcher._compute_mosaic_transforms)
        assert "if not nodes:" in source, (
            "the early empty-node return has gone; the `root is None` guard "
            "below may now be reachable and wants a test of its own")
        early = source.index("if not nodes:")
        root_line = source.index("root = max(nodes")
        assert early < root_line

    def test_any_row_at_all_puts_two_tiles_in_the_node_set(self, stitcher):
        """So `nodes` is non-empty whenever the early return did not fire.

        Every row names both ends of an edge, so one row is two nodes --
        there is no row shape that contributes none.
        """
        source = inspect.getsource(stitcher._compute_mosaic_transforms)
        assert 'nodes.add(r["pathA"]); nodes.add(r["pathB"])' in source, (
            "rows no longer contribute both endpoints; the node set could "
            "now be empty with rows present")

    # NOT DRIVEN with real rows: the tree-building path below the early
    # return calls `_estimate_grid_steps`, which reads detector options the
    # constructor sets -- and constructing a real spacrStitcher builds an ORB
    # detector. That is a stitching test rather than a test of this guard,
    # and it belongs with the stitching suite.
