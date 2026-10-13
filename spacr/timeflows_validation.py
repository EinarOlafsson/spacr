"""Shared supplied-mask linking scores and held-out checks for Timeflows training.

These measurements condition on the provided segmentation and track labels.
They do not measure end-to-end segmentation or whole-movie tracking accuracy.
"""
from __future__ import annotations

import hashlib
import inspect
import math

import numpy as np

from . import timeflows_model as tm
from .timeflows_baseline import link_frames


def temporal_assignment_policy():
    """Describe the default decoder used by both validation entry points."""
    parameters = inspect.signature(tm.link_by_timeflows).parameters
    return {
        'policy': 'distance_gate_before_assignment_with_unmatched_choices',
        'objective': 'minimum_total_distance_plus_unmatched_cost',
        'min_successor': parameters['min_successor'].default,
        'max_distance_diameters': parameters['max_distance'].default,
        'unmatched_cost': 'one distance limit, increased by one float64 ULP for an inclusive boundary',
    }


def scramble(labels, seed):
    """Relabel target objects without changing their shapes or positions.

    :param labels: target label array, with zero reserved for background.
    :param seed: random seed for permuting the nonzero identities.
    :returns: the relabelled array and original-to-shuffled identity mapping.
    """
    ids = np.unique(labels)
    ids = ids[ids != 0]
    mapping = dict(zip(ids.tolist(), np.random.default_rng(seed).permutation(ids).tolist()))
    result = np.zeros_like(labels)
    for old, new in mapping.items():
        result[labels == old] = new
    return result, mapping


def score_pair(labels_t, labels_t1, predictions, seed=0, unknown_successors=()):
    """Score source objects after excluding explicitly unknown successors.

    Shared track identities define the true links before target identities
    are shuffled. An absent target identity means no successor unless the
    source is in ``unknown_successors``; missing annotations are not detected
    automatically. An IoU control is added to the supplied prediction arms.

    :param labels_t: source masks with positive track identities and zero background.
    :param labels_t1: target masks using the same track identities.
    :param predictions: named prediction dictionaries accepted by
        :func:`spacr.timeflows_model.link_by_timeflows`; ``iou`` is reserved.
    :param seed: seed for shuffling target identities.
    :param unknown_successors: source identities omitted from every score.
    :returns: per-object truth, predictions, correctness and motion/density strata.
    """
    scrambled, mapping = scramble(labels_t1, seed)
    here, there = tm.object_centroids(labels_t), tm.object_centroids(labels_t1)
    links = {name: tm.link_by_timeflows(labels_t, scrambled, prediction)
             for name, prediction in predictions.items()}
    links['iou'] = {a: b for a, b, _ in link_frames(labels_t, scrambled)}
    rows = []
    for label, (y, x, diameter) in here.items():
        if label in unknown_successors:
            continue
        truth = mapping.get(label)
        motion = (math.hypot(there[label][0] - y, there[label][1] - x) / max(diameter, 1)
                  if label in there else None)
        neighbours = sum(other != label and math.hypot(oy - y, ox - x) <= 5 * diameter
                         for other, (oy, ox, _) in here.items())
        row = {'label': label, 'has_successor': truth is not None,
               'motion_diameters': motion,
               'motion_bin': ('no_successor' if motion is None else
                              'below_0.5' if motion < .5 else '0.5_to_1' if motion < 1 else 'at_least_1'),
               'neighbours_within_5_diameters': neighbours,
               'density_bin': 'zero' if neighbours == 0 else 'one_to_three' if neighbours <= 3 else 'at_least_four',
               'truth_target': truth,
               'predicted_target': {name: found.get(label) for name, found in links.items()},
               'correct': {name: found.get(label) == truth for name, found in links.items()}}
        rows.append(row)
    return rows


def summarise(rows):
    """Aggregate object-weighted results with explicit missing-data denominators.

    :param rows: per-object records from :func:`score_pair`.
    :returns: overall, motion, density and joint-stratum summaries. Successor
        accuracy is ``None`` where no true successors are available; false
        links and abstentions are counts with their population sizes retained.
    """
    arms = sorted({arm for row in rows for arm in row['correct']})

    def group(selected):
        """Summarize correct successor links and abstentions for the selected ground-truth objects."""
        alive = [row for row in selected if row['has_successor']]
        gone = [row for row in selected if not row['has_successor']]
        return {'sources': len(selected), 'true_successors': len(alive), 'no_successor': len(gone),
                'arms': {arm: {
                    'correct_successor_links': sum(row['correct'][arm] for row in alive),
                    'successor_accuracy': sum(row['correct'][arm] for row in alive) / len(alive) if alive else None,
                    'false_links_without_successor': sum(row['predicted_target'][arm] is not None for row in gone),
                    'abstentions_with_successor': sum(row['predicted_target'][arm] is None for row in alive),
                } for arm in arms}}

    return {'overall': group(rows),
            'motion': {key: group([row for row in rows if row['motion_bin'] == key])
                       for key in ('below_0.5', '0.5_to_1', 'at_least_1', 'no_successor')},
            'density': {key: group([row for row in rows if row['density_bin'] == key])
                        for key in ('zero', 'one_to_three', 'at_least_four')},
            'motion_by_density': {
                f'{motion}/{density}': group([row for row in rows if row['motion_bin'] == motion and row['density_bin'] == density])
                for motion in ('below_0.5', '0.5_to_1', 'at_least_1')
                for density in ('zero', 'one_to_three', 'at_least_four')}}


def _frame_fingerprint(frame):
    """Hash the actual float32, three-channel input consumed by the encoder."""
    array = tm._to_input(frame).numpy()
    digest = hashlib.sha256(str(array.shape).encode('ascii'))
    digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def check_pair_holdout(training_pairs, validation_pairs):
    """Reject validation frames identical to any normalized training input.

    Both endpoints participate. Channel filling, channel truncation and
    float32 conversion follow the model's own input adapter. This detects
    identical inputs across paths or dtypes, not near-duplicates or different
    frames of the same biological movie; the CLI separately rejects shared
    movie paths. Return the validation input fingerprints for provenance.

    :param training_pairs: pairs whose input frames participate in training.
    :param validation_pairs: nonempty held-out pairs with both input endpoints.
    :returns: two normalized-input SHA-256 fingerprints per validation pair.
    :raises ValueError: validation is empty or an input also occurs in training.
    """
    if not validation_pairs:
        raise ValueError('Validation requires at least one held-out pair')
    training = {_frame_fingerprint(frame) for pair in training_pairs
                for frame in (pair.frame_t, pair.frame_t1)}
    validation = []
    for pair in validation_pairs:
        fingerprints = [_frame_fingerprint(pair.frame_t), _frame_fingerprint(pair.frame_t1)]
        if any(fingerprint in training for fingerprint in fingerprints):
            raise ValueError('A held-out validation frame also occurs in the training inputs')
        validation.append(fingerprints)
    return validation


def validate_timeflows(net, pairs, *, device='cpu', seed=0, initial_head=None):
    """Score held-out pairs and controls without changing training state.

    ``pairs`` contain normalized frames and full track-label masks. Call
    :func:`check_pair_holdout` before training to verify exact input separation.
    The result includes per-object rows and displacement/density strata for
    the current model, IoU, zero-motion and oracle controls, plus a copied-frame
    check. Optional ``initial_head`` is a complete snapshot of the head/up
    parameters and buffers from training start. It is called ``initial_head``
    because a resumed model's initial head is not necessarily untrained.

    Module training/evaluation modes, current head weights and CPU/selected
    CUDA random states are restored even if scoring raises. No optimizer is
    touched. The returned scores concern supplied masks, not segmentation
    performance, lineage or whole-movie tracking.

    :param net: Timeflows network evaluated in its current state.
    :param pairs: nonempty full-frame pairs with matching track identities.
    :param device: device used for inference and random-state preservation.
    :param seed: starting seed for target-identity shuffles, incremented per pair.
    :param initial_head: optional complete head/up parameter and buffer snapshot.
    :returns: per-object rows, stratified summaries, copied-frame controls and
        temporal assignment metadata.
    :raises ValueError: pairs are empty or the initial-head keys or shapes differ.
    """
    if not pairs:
        raise ValueError('Validation requires at least one held-out pair')
    torch = tm._torch()
    modes = [(module, module.training) for module in net.modules()]
    current_head = {name: value.detach().clone() for name, value in net.state_dict().items()
                    if name.startswith(('head.', 'up.'))}
    if initial_head is not None:
        if set(initial_head) != set(current_head):
            raise ValueError('Initial head snapshot must contain exactly the head/up state')
        if any(initial_head[name].shape != value.shape for name, value in current_head.items()):
            raise ValueError('Initial head snapshot shapes do not match the model')
    target_device = torch.device(device)
    devices = []
    if target_device.type == 'cuda':
        devices = [target_device.index if target_device.index is not None else torch.cuda.current_device()]
    rows, copied_rows = [], []
    with torch.random.fork_rng(devices=devices):
        try:
            for index, pair in enumerate(pairs):
                targets = tm.time_targets(pair.labels_t, pair.labels_t1)
                predictions = {
                    'trained': tm.predict_pair(net, pair.frame_t, pair.frame_t1, device=device),
                    'zero_motion': {'vector': np.zeros_like(targets['vector']),
                                    'successor': np.ones_like(targets['successor'])},
                    'oracle': {'vector': targets['vector'], 'successor': targets['successor']},
                }
                if initial_head is not None:
                    try:
                        net.load_state_dict(initial_head, strict=False)
                        predictions['initial_head'] = tm.predict_pair(
                            net, pair.frame_t, pair.frame_t1, device=device)
                    finally:
                        net.load_state_dict(current_head, strict=False)
                rows.extend(score_pair(pair.labels_t, pair.labels_t1, predictions, seed + index))
                copied = tm.predict_pair(net, pair.frame_t, pair.frame_t, device=device)
                copied_rows.extend(score_pair(pair.labels_t, pair.labels_t,
                                              {'trained': copied}, seed + index))
        finally:
            for module, mode in modes:
                module.training = mode
    return {'pairs': len(pairs), 'seed': seed,
            'temporal_assignment': temporal_assignment_policy(),
            'scope': 'Linking given supplied full segmentation; not end-to-end tracking accuracy.',
            'results': summarise(rows), 'copied_frame_control': summarise(copied_rows),
            'rows': rows, 'copied_frame_rows': copied_rows}
