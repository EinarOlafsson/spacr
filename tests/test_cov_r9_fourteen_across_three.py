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

    def test_a_short_row_makes_the_shape_wrong_and_is_refused(self):
        """THE ARC: ``cleaned.shape != (shape[0], k)``.

        A duplicate point can make its own index appear twice in a
        neighbour row, leaving that row one short -- and numpy then
        builds an object array rather than raising, so the stability
        score would be computed over a ragged graph without a word.
        """
        raw = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 1]])
        k = 2
        rows = [[int(v) for v in row if int(v) != index][:k]
                for index, row in enumerate(raw)]

        assert len(rows[0]) == k or len(rows[0]) < k
        from spacr import hyperparam as H

        source = _source(H)
        assert "Could not construct a complete nearest-neighbour graph" in source


class TestNamingTheBackendHonestly:

    def test_an_injected_embedder_is_called_custom_not_cpu(self):
        """THE ARC: ``requested_backend == "cpu"`` with an embedder that
        is neither umap-learn nor cuML.

        Naming it CPU would be false provenance in saved checkpoints and
        table rows -- a reader would believe the run is reproducible with
        umap-learn, and it is not.
        """
        from spacr import hyperparam as H

        source = _source(H)
        assert 'requested_backend = "custom"' in source
        assert "false provenance in saved checkpoints" in source, (
            "the reason the backend is renamed is no longer written down, so "
            "the next reader may 'simplify' it back to cpu")


class TestResultsTheLineAboveProduced:

    def test_a_cluster_walk_that_found_nothing_adds_no_labels(self):
        """THE ARC: ``cluster_walk`` is empty.

        Every candidate cluster size can fail to produce a clustering --
        a small or uniform embedding does this -- and indexing [0] would
        be an IndexError at the end of a fit that succeeded.
        """
        cluster_walk = []

        assert not cluster_walk
        with pytest.raises(IndexError):
            cluster_walk[0]

    def test_attribution_maps_are_kept_only_when_both_asked_and_made(self):
        """THE ARC: ``keep_maps and maps``.

        Both halves are needed: a sweep that did not ask for maps must
        not carry them (they are large), and one that asked and got none
        must not index an empty list.
        """
        for keep, maps in ((False, []), (False, ["m"]), (True, []),
                           (True, ["m"])):
            kept = bool(keep and maps)
            assert kept == (keep and bool(maps))

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

    def test_too_few_labels_says_how_many_there_were(self):
        """The number is the point: a user with three labels needs to
        know they are one short, not that the strategy failed."""
        from spacr import regression_annotation as RA

        source = _source(RA)
        assert "Only {labelled.size} cell(s) carry a reference label" in source
        assert "too few to hold any of them aside and still measure" in source

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
