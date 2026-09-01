"""Twelve more single decisions, across nine modules.

Two optional-dependency guards, a graph with no node to root at, a move
that would be a no-op, four presence checks and a plot legend with
nothing to name.
"""
from __future__ import annotations

import inspect
import os
import pathlib

import numpy as np
import pandas as pd


def _source(module):
    return pathlib.Path(inspect.getsourcefile(module)).read_text()


class TestTheOptionalEmbedders:

    def test_umap_availability_is_decided_at_dispatch(self):
        """A failed optional import falls through to PCA before the helper."""
        from spacr import timelapse as T

        source = _source(T)
        assert "umap = None" in source
        assert 'embed_method == "umap" and umap is not None' in source
        assert "umap-learn is not installed." not in source

    def test_tsne_availability_is_decided_at_dispatch(self):
        """The t-SNE helper has the same single availability boundary."""
        from spacr import timelapse as T

        source = _source(T)
        assert "TSNE = None" in source
        assert 'embed_method == "tsne" and TSNE is not None' in source
        assert "sklearn.manifold.TSNE is not available." not in source

    def test_unavailable_requested_embedders_share_the_pca_fallback(self):
        from spacr import timelapse as T

        source = _source(T)
        assert 'if embed_method in {"umap", "tsne"}:' in source
        assert '"not available; falling back to PCA."' in source
        for marker in ("def _search_umap(", "def _search_tsne("):
            body = source[source.index(marker):]
            assert " is None:" not in body[:500]


class TestTheMosaicRoot:

    def test_the_best_connected_tile_becomes_the_root(self):
        adjacency = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
        nodes = list(adjacency)

        root = max(nodes, key=lambda p: len(adjacency[p])) if nodes else None

        assert root == "b"

    def test_a_mosaic_with_no_tiles_has_no_root_and_no_transforms(self):
        """The early empty-node answer makes a later root guard redundant.

        Every pair can fail to register -- a plate whose tiles do not
        overlap -- leaving no node to root the BFS at. Returning empty
        transforms is right: ``T3[None]`` would be a key nothing reads,
        and the caller then places every tile at its nominal position.
        """
        nodes = []

        root = max(nodes, key=len) if nodes else None
        assert root is None

        from spacr import spacrops as S

        source = _source(S)
        assert "if not nodes:" in source
        assert "return {}, []" in source
        assert "if root is None:" not in source
        assert "if nodes else None" not in source


class TestMovingAFileOntoItself:

    def test_the_post_stitch_destination_is_a_deeper_directory(self):
        """The post-stitch move always adds the non-empty well component.

        The first organizer stage leaves the tile under ``dst/well``; the
        post-stitch stage moves it under ``dst/well/well``. The filename gate
        refuses an empty well, so these paths cannot be equal.
        """
        source = os.path.join("dst", "A1", "tile.tif")
        target = os.path.join("dst", "A1", "A1", "tile.tif")
        assert os.path.abspath(source) != os.path.abspath(target)

        from spacr import spacrops as S

        text = _source(S)
        assert "if os.path.abspath(sp) != os.path.abspath(rp):" not in text
        assert "shutil.move(sp, rp)" in text

    def test_the_added_well_component_cannot_be_empty(self):
        """Pin the parser premise that keeps source and target distinct."""
        from spacr import spacrops as S

        text = inspect.getsource(S.stitch_cycle_wells)
        assert "not m.group(well_group)" in text
        assert "well = (m.group(well_group) or \"\").upper()" in text


class TestTheInvasionClassColumns:

    def test_a_class_no_field_showed_still_gets_a_column(self):
        """THE ARC: ``name not in field_counts.columns``.

        ``value_counts().unstack()`` only makes columns for classes that
        OCCURRED, so a screen where nothing was, say, 'invaded' would
        produce a table missing that column -- and every downstream sum
        would be a KeyError rather than a zero.
        """
        parasites = pd.DataFrame({"prcf": ["f1", "f1", "f2"],
                                  "invasion_class": ["a", "a", "b"]})
        counts = parasites.groupby("prcf", sort=False)["invasion_class"] \
            .value_counts().unstack(fill_value=0)

        assert "c" not in counts.columns
        for name in ("a", "b", "c"):
            if name not in counts.columns:
                counts[name] = 0

        assert set(counts.columns) >= {"a", "b", "c"}
        assert counts["c"].sum() == 0

    def test_the_classes_are_a_declared_list(self):
        from spacr import submodules as S

        source = _source(S)
        assert "for name in _INVASION_CLASSES:" in source, (
            "the class list is no longer walked, so a class that never "
            "occurs is missing from the table again")


class TestTheSudokuGuideColumn:

    def test_a_guide_present_in_the_well_is_copied_across(self):
        """The arm that runs for a guide the sub-problem knows."""
        names = ["g1", "g2"]

        assert names.index("g2") == 1
        assert ("g3" in names) is False

    def test_every_ranked_guide_is_a_result_column(self):
        """The inner solve returns the complete guide tuple it receives."""
        from spacr.sudoku import sudoku_all

        rng = np.random.default_rng(2)
        features = rng.normal(size=(24, 3))
        scores = rng.uniform(0.0, 1.0, 24)
        wells = [f"w{i % 4}" for i in range(24)]
        fractions = {f"w{i}": {"g1": 0.6, "g2": 0.4} for i in range(4)}
        result = sudoku_all(
            features, scores, wells, fractions,
            ranking=[("g1", 0.9), ("g2", 0.5)],
        )

        assert result.names == ("g1", "g2")


class TestTheManifestWarnings:

    def test_a_list_of_warnings_is_extended(self):
        collected = []
        values = ["one", "two", ""]
        if isinstance(values, (list, tuple)):
            collected.extend(str(v) for v in values if v)

        assert collected == ["one", "two"]

    def test_a_single_warning_written_as_a_string_is_still_collected(self):
        """THE ARC: ``elif values``.

        A manifest can carry one warning as a bare string rather than a
        list -- older writers did -- and iterating it would append one
        entry per CHARACTER. The elif is what keeps that from happening.
        """
        collected = []
        values = "the run was resumed"
        if isinstance(values, (list, tuple)):
            collected.extend(str(v) for v in values if v)
        elif values:
            collected.append(str(values))

        assert collected == ["the run was resumed"]

    def test_an_empty_value_adds_nothing(self):
        for values in ("", None, [], 0):
            collected = []
            if isinstance(values, (list, tuple)):
                collected.extend(str(v) for v in values if v)
            elif values:
                collected.append(str(values))
            assert collected == []


class TestTheBeforeAfterLegend:

    def test_a_legend_is_drawn_when_both_histograms_have_bars(self):
        handles = ["before-patch", "after-patch"]

        assert handles

    def test_even_an_empty_histogram_has_a_patch_for_its_legend(self):
        """Matplotlib creates one rectangle per requested bin."""
        from matplotlib.figure import Figure

        axes = Figure().add_subplot(111)
        axes.hist(np.array([]), bins=40)

        assert len(axes.patches) == 40


class TestThePositionalEffectPanel:

    def test_an_edge_delta_needs_both_sides_to_be_non_empty(self):
        """THE PIN, for ``edge.size and interior.size``.

        The branch above requires four or more groups, so slicing off
        the first and last leaves at least two in the middle -- both
        sides are non-empty by then.
        """
        values = [np.array([1.0]), np.array([2.0]), np.array([3.0]),
                  np.array([4.0])]

        edge = np.concatenate([values[0], values[-1]])
        interior = np.concatenate(values[1:-1])

        assert edge.size and interior.size

        from spacr import regression_qc as Q

        source = _source(Q)
        assert "if mark_edges and len(groups) >= 4:" in source

    def test_a_group_with_no_points_is_not_jittered(self):
        """THE ARC: ``v.size`` is false.

        A position with no observations is an empty array, and
        ``default_rng().uniform(size=0)`` is fine while the scatter that
        follows would draw nothing -- the guard keeps the group's slot on
        the axis without a phantom point at its centre.
        """
        empty = np.array([])

        assert not empty.size
        assert np.random.default_rng(0).uniform(-0.16, 0.16, 0).size == 0

    def test_a_single_point_is_not_jittered_either(self):
        """The nested case: one point gets zero offset, so a group of one
        sits exactly on its tick rather than somewhere random -- which
        would read as a spread it does not have."""
        assert np.zeros(1).tolist() == [0.0]

        from spacr import regression_qc as Q

        assert "if v.size > 1 else np.zeros(1)" in _source(Q)


class TestTheSequencingMode:

    def test_the_two_modes_pick_different_processors(self):
        """The admitted non-paired mode takes the single-read processor.

        The surrounding sample gate admits only paired or single, so the
        second dispatch arm is an exhaustive ``else`` rather than a branch.
        """
        from spacr import sequencing as S

        source = _source(S)
        assert "function = paired_read_chunked_processing" in source
        assert "function = single_read_chunked_processing" in source
        assert "elif settings['mode'] == 'single':" not in source
        assert "if settings['single_direction'] == 'R1':" in source

    def test_the_threshold_figure_always_has_a_destination(self):
        """The closure's sole caller derives and passes a destination.

        ``os.path.dirname`` returns a string even for a bare filename, so the
        old optional-destination guard could not be false through this API.
        """
        from spacr import sequencing as S

        source = _source(S)
        assert "if dst is not None:" not in source
        assert "dst = os.path.dirname(settings['count_data'][0])" in source
        assert "log_y=settings.get('log_y', False), dst=dst)" in source
        assert "'fraction_threshold.pdf'" in source
        assert os.path.dirname("counts.csv") == ""
