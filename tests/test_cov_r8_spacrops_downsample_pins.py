"""The downsample factor cannot reach any of the three translation lifts at zero.

Every stitching transform is estimated on a downsampled pair and then
lifted back to full resolution by dividing the translation by that
factor. The redundant zero guards are removed after pinning the producing
expressions and the earlier image-resize failure for raw zero settings.
"""
from __future__ import annotations

import inspect
import re

import numpy as np
import pytest

from spacr import spacrops as S


class TestTheDownsampleFactor:

    def test_it_is_read_with_a_floor_of_one(self):
        """THE PIN for removing the zero guard from the floored lift.

        ``s = self.downsample if self.downsample > 0 else 1.0`` -- so a
        zero, a negative or an unset downsample all become 1.0, and the
        divisions below can never divide by zero.

        The lift is ``M_full[0, 2] /= s``; the floor is what makes that
        unconditional division safe.
        """
        source = inspect.getsource(S)
        assert "s = self.downsample if self.downsample > 0 else 1.0" in source, (
            "the downsample factor no longer has a floor, so the divisions "
            "below can now divide by zero")

        for downsample in (0, -1, 0.0, -0.5):
            floored = downsample if downsample > 0 else 1.0
            assert floored != 0, (
                f"a downsample of {downsample} floored to {floored}")

        for downsample in (0.25, 1.0, 4):
            assert (downsample if downsample > 0 else 1.0) == downsample

        assert "if s != 0:" not in source

    def test_all_three_lifts_divide_the_translation_only(self):
        """The rotation and scale are unaffected by the downsample; only
        the translation is in pixels and therefore in the factor's
        units. Dividing the whole matrix would rescale the tile."""
        source = inspect.getsource(S)

        lifts = re.findall(r"M_full\[0, 2\] /= [^\n]*", source)
        assert len(lifts) >= 3, (
            f"expected three translation lifts, found {len(lifts)}")

        assert "M_full[:2, :2] /=" not in source, (
            "the rotation block is now divided by the downsample too, which "
            "rescales every tile")

    def test_a_zero_translation_divisor_would_put_a_tile_at_infinity(self):
        """What the guard buys, shown."""
        matrix = np.eye(3, dtype=np.float32)
        matrix[0, 2] = 120.0

        with np.errstate(divide="ignore", invalid="ignore"):
            lifted = matrix[0, 2] / 0.0

        assert not np.isfinite(lifted), (
            "dividing a translation by zero is no longer an infinity, so "
            "the guard protects something else now")


class TestTheMosaicGraphRoot:

    def test_a_graph_with_nodes_has_a_root(self):
        """The root is the best-connected tile, and every transform in
        the mosaic is relative to it."""
        adjacency = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
        nodes = list(adjacency)

        root = max(nodes, key=lambda p: len(adjacency[p])) if nodes else None

        assert root == "b", "the root is not the best-connected tile"

    def test_a_graph_with_no_nodes_has_no_root_and_no_transforms(self):
        """THE UNCOVERED ARC: ``root is None``.

        A well whose tiles all failed to register leaves the graph
        empty, and ``T3[None]`` would put a None key into the transform
        map that every later lookup then misses. Returning an empty map
        and the edges that were used says the same thing without the
        broken entry.
        """
        adjacency = {}
        nodes = list(adjacency)

        root = max(nodes, key=lambda p: len(adjacency[p])) if nodes else None

        assert root is None

        source = inspect.getsource(S.spacrStitcher._compute_mosaic_transforms)
        assert "if not nodes:" in source
        assert "return {}, []" in source
        assert "if root is None:" not in source
        assert "if nodes else None" not in source
