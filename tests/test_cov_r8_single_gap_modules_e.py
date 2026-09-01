"""Four more single-decision modules: one driven, three pinned.

Each pin names the code that rules its guard out, so the pin fails when
that code changes rather than the guard quietly coming alive.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# spacr/figures/fast_render.py -- every panel key has a branch
# ---------------------------------------------------------------------------

class TestEveryFastPanelKeyIsHandled:

    def test_an_unknown_key_is_refused_with_the_known_ones_named(self):
        import pandas as pd

        from spacr.figures.fast_render import build_fast_plot

        frame = pd.DataFrame({"feature": ["a"], "coefficient": [1.0],
                              "p_value": [0.01]})
        with pytest.raises(KeyError, match="no pyqtgraph twin"):
            build_fast_plot("not_a_panel", frame)

    def test_the_branch_chain_covers_every_declared_panel(self):
        """THE PIN.

        The final ``elif`` falls through to ``return plot`` for a key
        that matched none of the branches -- and no such key exists: the
        gate at the top raises for anything outside FAST_PANELS, and
        every member of FAST_PANELS has a branch.

        A panel added to the table without a branch would silently
        return an EMPTY plot -- a picture with no data and no complaint,
        which is worse than the KeyError the unknown key gets. That is
        what this catches.
        """
        from spacr.figures import fast_render

        source = inspect.getsource(fast_render.build_fast_plot)
        dedented = textwrap.dedent(source)
        tree = ast.parse(dedented)

        handled = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and len(node.comparators) == 1:
                target = node.comparators[0]
                if isinstance(target, ast.Constant) and \
                        isinstance(target.value, str):
                    handled.add(target.value)
                elif isinstance(target, ast.Tuple):
                    handled.update(e.value for e in target.elts
                                   if isinstance(e, ast.Constant))

        missing = set(fast_render.FAST_PANELS) - handled
        assert missing == {"agreement"}, (
            "only the exhaustive final else may be absent from comparisons; "
            f"got {sorted(missing)}")
        assert "else:" in source and "guide_support" in source


# ---------------------------------------------------------------------------
# spacr/external_masks.py -- the same file reached through two groups
# ---------------------------------------------------------------------------

class TestOneImageCountedOnce:
    """The same file reached through two selected groups."""

    def _group(self, root, role, object_type=None):
        return {"key": f"{role}-{root.name}", "root": str(root),
                "paths": [str(path) for path in sorted(root.glob("*.tif"))],
                "role": role, "object_type": object_type}

    def _tree(self, tmp_path):
        import numpy as np
        import tifffile

        images = tmp_path / "images"
        masks = tmp_path / "masks"
        images.mkdir()
        masks.mkdir()
        for name in ("a.tif", "b.tif"):
            tifffile.imwrite(images / name,
                             np.random.default_rng(1).integers(
                                 0, 255, (16, 16), dtype=np.uint8))
            tifffile.imwrite(masks / name, np.zeros((16, 16), dtype=np.uint16))
        return images, masks

    def test_the_same_folder_selected_twice_is_planned_once(self, tmp_path):
        """THE UNCOVERED ARC.

        Nothing stops a user selecting the same folder as two image
        groups -- a parent and a child, or the same path twice. A
        duplicated source is a second copy of every mask written over
        the first, and a plan that reports twice the work it will do.
        """
        pytest.importorskip("tifffile")
        from spacr import external_masks

        images, masks = self._tree(tmp_path)

        once = external_masks.plan_external_masks({"inputs": [
            self._group(images, "image"),
            self._group(masks, "mask", "cell")]})
        twice = external_masks.plan_external_masks({"inputs": [
            self._group(images, "image"),
            self._group(images, "image"),
            self._group(masks, "mask", "cell")]})

        assert once.errors == [], f"the single-group plan failed: {once.errors}"
        assert once.stems, "the plan found no images to start with"
        assert twice.stems == once.stems, (
            "the same folder selected twice was planned twice")
        assert twice.images == once.images

    def test_a_second_distinct_folder_is_added_not_deduplicated(self,
                                                                 tmp_path):
        """The other side of the same check: the marker is (path, series),
        so two different files are two sources."""
        pytest.importorskip("tifffile")
        import numpy as np
        import tifffile

        from spacr import external_masks

        images, masks = self._tree(tmp_path)
        more = tmp_path / "more"
        more.mkdir()
        tifffile.imwrite(more / "c.tif",
                         np.random.default_rng(2).integers(
                             0, 255, (16, 16), dtype=np.uint8))
        tifffile.imwrite(masks / "c.tif", np.zeros((16, 16), dtype=np.uint16))

        one = external_masks.plan_external_masks({"inputs": [
            self._group(images, "image"),
            self._group(masks, "mask", "cell")]})
        both = external_masks.plan_external_masks({"inputs": [
            self._group(images, "image"),
            self._group(more, "image"),
            self._group(masks, "mask", "cell")]})

        assert len(both.stems) > len(one.stems), (
            "a second, distinct folder added nothing to the plan")


# ---------------------------------------------------------------------------
# spacr/response_distribution.py -- a histogram always has bars
# ---------------------------------------------------------------------------

class TestTheTwoAxisPanelAlwaysHasALegend:

    def test_a_rescaled_transform_draws_both_distributions(self):
        from spacr.response_distribution import panel

        rng = np.random.default_rng(4)
        values = rng.uniform(0.01, 0.99, 300)

        result = panel(values, "logit")
        ax = result["axes"]

        assert ax.get_xlabel()
        assert ax.get_title().startswith("Response before and after")

    def test_matplotlib_makes_one_bar_per_bin_even_with_no_data(self):
        """THE PIN.

        ``if handles:`` guards a legend built from the first patch of
        each axis. ``ax.hist(..., bins=40)`` creates forty rectangles
        whether or not any value lands in them, so the patch list is
        never empty and the guard never fires.

        The legend matters: with two x-axes and no legend, the reader
        cannot tell which distribution is which, and the whole point of
        the panel is the comparison.
        """
        from matplotlib.figure import Figure

        figure = Figure()
        ax = figure.add_subplot(111)
        ax.hist(np.array([]), bins=40, alpha=0.55)

        assert len(ax.patches) == 40, (
            "hist no longer makes a bar per empty bin, so the legend guard "
            "in response_distribution.panel is live")


# ---------------------------------------------------------------------------
# spacr/sudoku.py -- the committed guide is always among the names
# ---------------------------------------------------------------------------

class TestTheCommittedGuideIsAlwaysInTheResult:

    def _run(self, n=24, seed=2):
        from spacr.sudoku import sudoku_all

        rng = np.random.default_rng(seed)
        features = rng.normal(size=(n, 3))
        scores = rng.uniform(0.0, 1.0, n)
        wells = [f"w{i % 4}" for i in range(n)]
        fractions = {f"w{i}": {"g1": 0.6, "g2": 0.4} for i in range(4)}
        ranking = [("g1", 0.9), ("g2", 0.5)]
        return sudoku_all(features, scores, wells, fractions, ranking)

    def test_a_run_assigns_from_the_ranked_guides(self):
        result = self._run()

        assert set(result.names) == {"g1", "g2"}
        assert result.affirm.shape[1] == 2

    def test_the_inner_call_returns_exactly_the_guides_it_was_given(self):
        """THE PIN.

        The per-round call passes the WHOLE guide tuple -- running it
        with the committed guide alone is degenerate -- and ``sudoku``
        returns that tuple back as ``names``, unfiltered. So the
        committed guide is always found in it and ``mine`` is never
        None.

        If ``sudoku`` ever drops a guide from its names (one with no
        cells, say), the round silently records nothing for it while
        still claiming its cells. This fails first.
        """
        from spacr.sudoku import sudoku

        guides = ("g1", "g2", "g3")
        rng = np.random.default_rng(7)
        result = sudoku(rng.normal(size=(12, 2)), rng.uniform(size=12),
                        [f"w{i % 3}" for i in range(12)],
                        {f"w{i}": {"g1": 0.5, "g2": 0.5} for i in range(3)},
                        guides)

        assert result.names == guides, (
            "sudoku no longer returns the guides it was given")

    def test_even_a_guide_with_no_well_fraction_keeps_its_column(self):
        from spacr.sudoku import sudoku

        guides = ("g1", "unheard_of")
        rng = np.random.default_rng(8)
        result = sudoku(rng.normal(size=(10, 2)), rng.uniform(size=10),
                        [f"w{i % 2}" for i in range(10)],
                        {f"w{i}": {"g1": 1.0} for i in range(2)},
                        guides)

        assert result.names == guides
        assert result.affirm.shape[1] == 2
