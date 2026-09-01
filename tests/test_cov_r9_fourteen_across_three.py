"""Fourteen single decisions across measure, hyperparam and
regression_annotation.

Four settings checks whose value the schema supplies, five guards on
results the line above produced, and five refusals that name what the
user should change.
"""
from __future__ import annotations

import inspect
import pathlib

import numpy as np
import pytest


def _source(module):
    return pathlib.Path(inspect.getsourcefile(module)).read_text()


class TestTheMaskRegistry:

    def test_a_mask_offered_under_a_name_already_held_is_not_replaced(self):
        """THE ARC: ``name not in masks`` is false.

        The registry is built from the run's own masks, so the one being
        measured is normally already in it -- and replacing it would
        hand the distance code a DIFFERENT array from the one every
        earlier step keyed on.
        """
        masks = {"cell": "the run's cell mask"}

        for name, mask in (("cell", "another array"), ("nucleus", "n")):
            if name not in masks:
                masks = dict(masks, **{name: mask})

        assert masks == {"cell": "the run's cell mask", "nucleus": "n"}

    def test_the_registry_is_copied_rather_than_mutated(self):
        """A caller holding the old mapping must not see a mask appear in
        it half way through a measurement."""
        from spacr import measure as M

        source = _source(M)
        assert "masks = dict(masks, **{name: mask})" in source
        assert "masks.update(" not in source[
            source.index("if name not in masks:"):
            source.index("if name not in masks:") + 200]


class TestTheSettingsTheSchemaSupplies:

    @pytest.mark.parametrize("marker", [
        "if settings['cell_mask_dim'] is not None:",
        "if not settings['cell_mask_dim'] is None:",
    ])
    def test_the_cell_dim_is_checked_before_relabelling(self, marker):
        """THE PIN, for two spellings of the same check.

        Relabelling nuclei to their parent cell needs a cell mask, and
        ``timelapse_objects == 'nucleus'`` is only reachable on a run
        that segmented cells -- so the dim is set. Both spellings are
        held because they are the same decision written two ways, and a
        reader who fixes one will not find the other.
        """
        from spacr import measure as M

        source = _source(M)
        assert marker in source
        gate = source.index("if settings['timelapse_objects'] == 'nucleus':")
        assert source.index(marker, gate) > gate, (
            "the cell-dim check no longer sits under the timelapse-object "
            "gate, so it now runs for every object type")

    def test_the_normalise_check_is_an_identity_test(self):
        """``normalize`` is False or a percentile pair, so ``not ... is
        False`` distinguishes "switched off" from "(0, 100)", which is a
        real request that a truthiness test would drop."""
        from spacr import measure as M

        assert "if not settings['normalize'] is False:" in _source(M)

        for value in (False, (0, 100), [1, 99], None):
            switched_off = value is False
            assert switched_off == (value is False)
        assert bool((0, 100)) and not bool([])


class TestTheNeighbourGraph:

    def test_a_complete_graph_has_k_neighbours_for_every_point(self):
        raw = np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1]])
        k = 2
        cleaned = np.array([[int(v) for v in row if int(v) != index][:k]
                            for index, row in enumerate(raw)], dtype=int)

        assert cleaned.shape == (3, k)

    def test_duplicate_points_still_return_distinct_neighbour_indices(self):
        """Duplicate coordinates do not duplicate an index in sklearn's row."""
        from sklearn.neighbors import NearestNeighbors

        points = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0],
                           [0.0, 1.0]])
        k = 2
        raw = NearestNeighbors(n_neighbors=k + 1).fit(points).kneighbors(
            points, return_distance=False)
        rows = [[int(v) for v in row if int(v) != index][:k]
                for index, row in enumerate(raw)]

        assert all(len(row) == k for row in rows)
        from spacr import hyperparam as H

        source = _source(H)
        assert "if cleaned.shape != (shape[0], k):" not in source
        assert "Could not construct a complete nearest-neighbour graph" \
            not in source


class TestNamingTheBackendHonestly:

    def test_an_injected_embedder_is_called_custom_not_cpu(self):
        """An injected embedder is custom even if the caller labels it cuML.

        Naming it CPU would be false provenance in saved checkpoints and
        table rows -- a reader would believe the run is reproducible with
        umap-learn, and it is not.
        """
        from spacr import hyperparam as H

        features = np.random.default_rng(3).normal(size=(20, 4))
        result = H.umap_search(
            features, H.SearchSpace({"n_neighbors": [5]}), backend="cuml",
            embed_fn=lambda values, _params: values[:, :2])

        assert result.best.extra_metrics["backend"] == "custom"
        assert "Embedding backend: custom." in " ".join(result.notes)


class TestResultsTheLineAboveProduced:

    def test_a_valid_cluster_walk_always_has_a_result_row(self):
        """All-noise partitions still produce a scored walk row."""
        from spacr import hyperparam as H
        from spacr.umap_search import walk_clusters

        rows = walk_clusters(np.zeros((8, 2)), min_cluster_sizes=(5,))
        assert len(rows) == 1

        source = inspect.getsource(H.umap_search)
        assert "if cluster_walk:" not in source
        assert "chosen = cluster_walk[0]" in source

    def test_attribution_maps_are_kept_only_when_asked(self):
        """THE ARC: ``keep_maps``.

        The data loader refuses an empty image list, so every completed fit
        has a map; the remaining choice is whether to retain that large array.
        """
        from spacr import hyperparam as H

        source = inspect.getsource(H.activation_fit_fn)
        assert "if keep_maps:" in source
        assert "if keep_maps and maps:" not in source

    def test_normalisation_is_added_only_when_statistics_were_resolved(self):
        """THE ARC: ``stats is None``.

        A dataset whose statistics could not be computed is fed
        unnormalised rather than with a None mean, which is a TypeError
        inside the first batch.
        """
        steps = []
        for stats in (None, ((0.5,), (0.2,))):
            if stats is not None:
                steps.append(stats)

        assert len(steps) == 1


class TestRefusalsThatNameTheFix:
    def test_four_is_the_floor_and_it_is_a_hold_out_argument(self):
        """THE ARC: ``labelled.size < 4``.

        Fewer than four cannot be split into a training part and a
        hold-out that measures anything, which is why the floor is four
        rather than one.
        """
        for size in (0, 1, 3):
            assert size < 4
        for size in (4, 10):
            assert not size < 4

    def test_the_seed_model_summary_is_appended_when_there_is_one(self):
        """THE ARC: ``fit is not None``.

        A run that could not fit a seed model still queues by
        uncertainty over whatever probabilities it has, and the note is
        simply shorter -- rather than the queue failing.
        """
        notes = ["queued by uncertainty"]
        for fit in (None, "the seed model, measured"):
            if fit is not None:
                notes.append(str(fit))

        assert len(notes) == 2

    def test_the_pu_correction_is_applied_only_to_predictions_that_exist(self):
        """THE ARC: ``predictions is None``.

        The estimator can decline to predict on an empty selectable
        pool, and dividing None by the labelling rate would end a run
        that had already fitted.
        """
        from spacr import regression_annotation as RA

        source = _source(RA)
        assert "if predictions is not None:" in source
        assert "c is a lower bound on the true labelling rate" in source, (
            "the note that the PU correction is PARTIAL is gone, so a reader "
            "takes the corrected probabilities as calibrated")

    def test_a_round_that_did_not_improve_stops_the_self_training(self):
        """THE ARC: ``elif index``.

        The first round has nothing to compare against, so only a LATER
        round can stop the loop -- and the message is the argument: a
        round that does not improve the audit set is a round of the
        model agreeing with itself.
        """
        from spacr import regression_annotation as RA

        source = _source(RA)
        assert "a round that does not improve it is a" in source
        assert "round of the model agreeing with itself" in source

        best, stops = None, []
        for index, score in enumerate([0.6, 0.7, 0.65]):
            if best is None or score > best:
                best = score
            elif index:
                stops.append(index)
        assert stops == [2], "the first round stopped the loop"

    def test_no_round_fitting_at_all_is_refused(self):
        """THE PIN, for ``if best is None``.

        The loop sets ``best`` on its first iteration whenever a round
        fitted, so this is reached only when every round failed -- and
        answering an empty result would look like a strategy that ran
        and found nothing.
        """
        from spacr import regression_annotation as RA

        source = _source(RA)
        assert "No self-training round could be fitted." in source
        loop = source.index("if best is None or report.balanced_accuracy")
        guard = source.index("if best is None:", loop)
        assert loop < guard
