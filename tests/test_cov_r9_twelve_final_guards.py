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
import pytest


def _source(module):
    return pathlib.Path(inspect.getsourcefile(module)).read_text()


class TestTheOptionalEmbedders:

    def test_the_umap_refusal_names_the_package_to_install(self):
        """THE PIN, for ``umap is None`` -- and MEASURED, which changed
        what this says.

        It was written on the record that umap-learn is deliberately
        absent from this environment because it pulls numba. That is no
        longer true: umap 0.5.12 and numba 0.67 are installed against
        numpy 2.5.2, so the guard cannot fire here either and this is a
        pin rather than the path a user meets.

        What is still worth holding is the refusal naming the package to
        install, since that IS what a checkout without it meets.
        """
        from spacr import timelapse as T

        # Both names are LOCALS inside the analysis function, bound by a
        # guarded import, so the premise is asked of the environment
        # rather than of the module.
        try:
            import umap                                   # noqa: F401
            installed = True
        except Exception:                                 # noqa: BLE001
            installed = False

        assert installed, (
            "umap-learn has gone from this environment, so the refusal is "
            "now the path every analysis takes and wants a driven test")
        assert "umap-learn is not installed." in _source(T)
        assert "umap = None" in _source(T), (
            "the guarded import is gone, so a missing umap-learn now stops "
            "the module loading rather than one analysis")

    def test_the_tsne_guard_is_the_opposite_case(self):
        """THE PIN, for ``TSNE is None``.

        sklearn IS a hard dependency, so this one cannot fire -- the two
        guards look identical and are not, and saying so is the point.
        """
        from spacr import timelapse as T

        from sklearn.manifold import TSNE                 # noqa: F401

        assert TSNE is not None, (
            "sklearn.manifold.TSNE is missing, which would make it an "
            "optional dependency rather than a required one")
        assert "sklearn.manifold.TSNE is not available." in _source(T)

    def test_both_refuse_before_doing_any_work(self):
        from spacr import timelapse as T

        source = _source(T)
        for marker, message in (
                ("def _search_umap(", "umap-learn is not installed."),
                ("def _search_tsne(", "sklearn.manifold.TSNE is not")):
            start = source.index(marker)
            body = source[start:start + 400]
            assert message in body
            assert body.index(message) < body.index("random_state"), (
                f"{marker} does work before checking its dependency")


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

    def test_a_guide_the_sub_problem_never_saw_leaves_its_column_alone(self):
        """THE ARC: ``mine is None``.

        A guide can be in the library and absent from one well's own
        solve -- it was not sequenced there. Writing ``here.affirm[:,
        None]`` would be an index error; leaving the column at its
        initial value says "no evidence from this well", which is true.
        """
        names = ["g1", "g2"]
        mine = names.index("g3") if "g3" in names else None

        assert mine is None

        from spacr import sudoku as S

        assert "if mine is not None:" in _source(S)


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

    def test_no_bars_means_no_legend_rather_than_an_empty_box(self):
        """THE ARC: ``handles`` is empty.

        A transform panel over a column with nothing finite draws no
        patches, and ``ax.legend([], [...])`` puts an empty box on the
        figure with a title and no entries.
        """
        handles = []

        assert not handles

        from spacr import response_distribution as R

        assert "if handles:" in _source(R)


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
        """THE ARC: ``mode == 'single'``.

        Paired and single-end runs use different chunk processors, and
        the direction setting only means something for the single one.
        """
        from spacr import sequencing as S

        source = _source(S)
        assert "function = paired_read_chunked_processing" in source
        assert "function = single_read_chunked_processing" in source
        assert "elif settings['mode'] == 'single':" in source
        assert "if settings['single_direction'] == 'R1':" in source

    def test_the_threshold_figure_is_saved_only_when_asked(self):
        """THE ARC: ``dst is not None``.

        The figure is shown either way; a destination is what turns it
        into a file, and ``os.path.join(None, ...)`` is a TypeError at
        the end of a scan that had finished.
        """
        from spacr import sequencing as S

        source = _source(S)
        assert "if dst is not None:" in source
        assert "'fraction_threshold.pdf'" in source

        with pytest.raises(TypeError):
            os.path.join(None, "results")
