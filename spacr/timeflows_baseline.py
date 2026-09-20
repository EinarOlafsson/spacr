"""The plain stitcher the timeflows fork has to beat, and the score that says so.

WHAT IT IS FOR
==============
Step 2 of the timeflows plan (``features/future/426``) is a number, not a
feature. Before a third Cellpose head is written, trained and maintained,
there has to be a baseline: segment every frame independently, link the frames
by mask overlap, and score the result against the ground truth with the
measures the tracking field already agrees on. Whatever the fork produces is
then either better than this or it is not, and the plan says plainly that "the
plain stitcher is already good enough" is a real possible outcome that should
be accepted if it happens. That outcome is only visible if this number exists
first.

WHAT IS IN HERE
===============
* :func:`stitch_by_iou` -- per-frame labels in, one consistent id per object
  out. Hungarian assignment on the IoU between consecutive frames, which is
  the standard overlap stitcher and deliberately nothing cleverer.
* :func:`det_score` and :func:`tra_score` -- the Cell Tracking Challenge's DET
  and TRA, built on the AOGM cost of Matula et al. (2015): the weighted count
  of the edit operations that would turn the computed tracking graph into the
  ground-truth one, normalised by the cost of building the ground truth from
  nothing.
* :func:`identity_switches` -- how often one ground-truth object changes
  computed id. TRA already pays for this, mixed in with everything else; the
  count is reported separately because it is the failure a biologist sees, and
  because a tracker can trade it against segmentation errors without TRA
  moving much.
* :func:`score_tracking` -- all of it in one row, with the raw operation
  counts beside the scores.

WHY THE OPERATION COUNTS ARE PUBLIC
===================================
DET and TRA are one number each and both are dominated by the vertex terms: a
false negative costs ten and a redundant edge costs one, so a tracker can lose
half its links and still score well if its segmentation is good. The counts of
false negatives, false positives, splits, and added, deleted and changed edges
say which half of the problem the number came from. The plan needs exactly
that distinction later -- the gap between linking given perfect segmentation
and end-to-end linking is what decides whether to work on the head or on the
backbone -- so the counts are part of the output rather than an internal.

HOW A TRACKING IS REPRESENTED
=============================
The Cell Tracking Challenge convention, because the metrics are its: a label
stack where THE LABEL IS THE TRACK ID, so an object keeps its label for as
long as it is the same object, plus an optional lineage mapping each label to
its parent label. A division is therefore two new labels whose parent is the
old one; a tracker that cannot express divisions passes no lineage and pays
for the two missing parent edges, which is the honest price of a plain
stitcher.

WHY IT DOES NOT CALL ``spacr.timelapse``
========================================
:func:`spacr.timelapse.link_by_iou` does the same linking and is the one the
pipeline uses. It is not imported here for two reasons. Importing
``spacr.timelapse`` pulls in matplotlib, OpenCV and the tracking backends,
which is a heavy import for a scoring module that has to run inside a training
loop, and its overlap arithmetic is a Python double loop over pairs of boolean
masks -- fine for the frames a run produces, far too slow for scoring a held
out set every epoch. The arithmetic here is a single cross-tabulation per
frame pair. Where the two are meant to agree -- which pairs of labels are
matched at a given threshold -- the tests say so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

#: The AOGM edit weights of the Cell Tracking Challenge (Matula et al., PLoS
#: ONE 2015), which are what make DET and TRA comparable with published
#: numbers. They are not tuning knobs: a false negative costs ten because
#: finding a missed object is the expensive repair, a split costs five, and a
#: false positive costs one because deleting it is cheap. Changing them
#: changes what the score means, so a caller that passes its own has to say so
#: when it reports the number.
AOGM_WEIGHTS = {
    'ns': 5.0,
    'fn': 10.0,
    'fp': 1.0,
    'ed': 1.0,
    'ea': 1.5,
    'ec': 1.0,
}

#: A computed segment is the detection of a ground-truth object when it covers
#: MORE THAN HALF of that object's pixels. More than half is what makes the
#: matching a function: two computed segments cannot both hold more than half
#: of the same object, so no object is detected twice and the matching never
#: needs a tie broken.
DETECTION_OVERLAP = 0.5


def overlap_counts(previous, current):
    """Cross-tabulate the pixels shared by the labels of two frames.

    One pass over the pixels where both frames are labelled, so the cost is
    the image rather than the product of the two label counts.

    :param previous: 2-D label frame.
    :param current: 2-D label frame of the same shape.
    :returns: ``(labels_previous, labels_current, counts)``. ``counts[i, j]``
        is the number of pixels carrying ``labels_previous[i]`` in the first
        frame and ``labels_current[j]`` in the second. Background is excluded
        from both label arrays.
    """
    previous = np.asarray(previous)
    current = np.asarray(current)
    labels_previous = np.unique(previous)
    labels_previous = labels_previous[labels_previous > 0]
    labels_current = np.unique(current)
    labels_current = labels_current[labels_current > 0]
    counts = np.zeros((labels_previous.size, labels_current.size), dtype=float)
    if labels_previous.size == 0 or labels_current.size == 0:
        return labels_previous, labels_current, counts
    both = np.logical_and(previous > 0, current > 0)
    if not both.any():
        return labels_previous, labels_current, counts
    index_previous = {int(label): position
                      for position, label in enumerate(labels_previous)}
    index_current = {int(label): position
                     for position, label in enumerate(labels_current)}
    pairs = np.stack([previous[both].ravel(), current[both].ravel()], axis=1)
    unique_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)
    for (label_previous, label_current), count in zip(unique_pairs, pair_counts):
        counts[index_previous[int(label_previous)],
               index_current[int(label_current)]] = float(count)
    return labels_previous, labels_current, counts


def iou_matrix(previous, current):
    """Intersection over union for every pair of labels in two frames.

    :param previous: 2-D label frame.
    :param current: 2-D label frame of the same shape.
    :returns: ``(labels_previous, labels_current, iou)`` where ``iou[i, j]``
        is in ``[0, 1]``.
    """
    labels_previous, labels_current, counts = overlap_counts(previous, current)
    if labels_previous.size == 0 or labels_current.size == 0:
        return labels_previous, labels_current, counts
    areas_previous = np.array(
        [float((np.asarray(previous) == label).sum())
         for label in labels_previous])
    areas_current = np.array(
        [float((np.asarray(current) == label).sum())
         for label in labels_current])
    union = (areas_previous[:, None] + areas_current[None, :] - counts)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.where(union > 0, counts / union, 0.0)
    return labels_previous, labels_current, iou


def link_frames(previous, current, iou_threshold=0.1):
    """Match the labels of two consecutive frames one to one.

    Hungarian assignment maximising total IoU, then every pair below the
    threshold is dropped. The assignment matters where a cell divides or two
    cells touch: greedy best-match links both children to the parent and
    produces two objects with one id, which the metrics then punish twice.

    :param previous: 2-D label frame.
    :param current: 2-D label frame of the same shape.
    :param iou_threshold: smallest IoU that is accepted as the same object.
    :returns: list of ``(label_previous, label_current, iou)``, ordered by the
        previous frame's labels.
    """
    labels_previous, labels_current, iou = iou_matrix(previous, current)
    if labels_previous.size == 0 or labels_current.size == 0:
        return []
    rows, columns = linear_sum_assignment(-iou)
    matches = []
    for row, column in zip(rows, columns):
        value = float(iou[row, column])
        if value >= iou_threshold:
            matches.append((int(labels_previous[row]),
                            int(labels_current[column]), value))
    return sorted(matches)


def stitch_by_iou(masks, iou_threshold=0.1):
    """Give one id to the same object through a movie, by overlap alone.

    This is the baseline the plan calls "the plain stitcher": each frame is
    segmented independently, consecutive frames are matched by IoU, an object
    that matches nothing starts a new id, and nothing is bridged across a gap.
    It has no notion of a division -- both children are new objects, and the
    parent simply ends -- which is exactly the limitation the time-flow head is
    supposed to remove, so it is left in rather than patched around.

    :param masks: per-frame segmentation, ``(T, Y, X)`` or a sequence of 2-D
        frames, whose labels mean nothing across frames.
    :param iou_threshold: smallest IoU that is accepted as the same object.
    :returns: ``(tracked, table)``. ``tracked`` is a ``(T, Y, X)`` integer
        array in which the label IS the track id; ``table`` is a
        ``DataFrame`` of ``frame``, ``original_label``, ``track_id`` and the
        ``iou`` the link was made on (``NaN`` where the track starts).
    """
    frames = [np.asarray(frame) for frame in masks]
    for index, frame in enumerate(frames):
        if frame.ndim != 2:
            raise ValueError(
                f'stitch_by_iou needs 2-D label frames and frame {index} has '
                f'{frame.ndim} dimensions (shape {frame.shape}); the overlap '
                f'arithmetic would link a volume without complaint and the '
                f'tracks would be fiction')
    tracked = [np.zeros_like(frame, dtype=np.int64) for frame in frames]
    rows = []
    next_track = 1
    assigned = {}
    if frames:
        for label in np.unique(frames[0]):
            if label <= 0:
                continue
            assigned[(0, int(label))] = next_track
            rows.append({'frame': 0, 'original_label': int(label),
                         'track_id': next_track, 'iou': float('nan')})
            next_track += 1
    for index in range(1, len(frames)):
        matches = link_frames(frames[index - 1], frames[index],
                              iou_threshold=iou_threshold)
        linked = {}
        for label_previous, label_current, value in matches:
            track = assigned.get((index - 1, label_previous))
            if track is None:
                continue
            linked[label_current] = (track, value)
        for label in np.unique(frames[index]):
            if label <= 0:
                continue
            label = int(label)
            if label in linked:
                track, value = linked[label]
            else:
                track, value = next_track, float('nan')
                next_track += 1
            assigned[(index, label)] = track
            rows.append({'frame': index, 'original_label': label,
                         'track_id': track, 'iou': value})
    for (index, label), track in assigned.items():
        tracked[index][frames[index] == label] = track
    table = pd.DataFrame(rows, columns=[
        'frame', 'original_label', 'track_id', 'iou'])
    stack = (np.stack(tracked) if tracked
             else np.zeros((0, 0, 0), dtype=np.int64))
    return stack, table


def tracking_graph(masks, lineage=None):
    """Build the acyclic oriented graph the AOGM measures are defined on.

    A label whose frames are not contiguous -- a tracker that bridged a gap,
    or annotation that let an object vanish for a frame and kept its id -- is
    not refused, but the gap earns NO EDGE. The challenge's format has no way
    to say "the same object, not visible for a while", and an edge across the
    gap would credit a link the graph cannot express and the truth does not
    contain. The tracking is scored on the links it can show; the identity it
    claims across the gap is still honoured by
    :func:`identity_switches`, which is where that claim belongs.

    :param masks: label stack in which the label is the track id.
    :param lineage: optional mapping of label to parent label. ``0`` or a
        label that never appears means no parent.
    :returns: dict with ``vertices`` (set of ``(frame, label)``) and ``edges``
        (dict from ``((frame, label), (frame, label))`` to ``'track'`` or
        ``'division'``). Track edges join consecutive frames of one label;
        division edges join a parent's last frame to a child's first.
    """
    frames = [np.asarray(frame) for frame in masks]
    appearances = {}
    for index, frame in enumerate(frames):
        for label in np.unique(frame):
            if label <= 0:
                continue
            appearances.setdefault(int(label), []).append(index)
    vertices = set()
    edges = {}
    for label, seen in appearances.items():
        seen = sorted(seen)
        for frame_index in seen:
            vertices.add((frame_index, label))
        for position in range(len(seen) - 1):
            if seen[position + 1] != seen[position] + 1:
                continue
            edges[((seen[position], label),
                   (seen[position + 1], label))] = 'track'
    for label, parent in dict(lineage or {}).items():
        label, parent = int(label), int(parent)
        if parent <= 0 or parent not in appearances or label not in appearances:
            continue
        last_parent = max(appearances[parent])
        first_child = min(appearances[label])
        if first_child != last_parent + 1:
            continue
        edges[((last_parent, parent), (first_child, label))] = 'division'
    return {'vertices': vertices, 'edges': edges}


def match_vertices(gt_masks, pred_masks):
    """Match every ground-truth object to the computed segment that detects it.

    The Cell Tracking Challenge's criterion: a computed segment detects a
    ground-truth object when it covers more than half of that object's pixels.

    :param gt_masks: ground-truth label stack.
    :param pred_masks: computed label stack of the same shape.
    :returns: ``(matched, shared)``. ``matched`` maps a ground-truth vertex
        ``(frame, label)`` to the computed vertex that detects it; ``shared``
        maps a computed vertex to the number of ground-truth objects it
        detects, which is above one exactly where two objects were segmented
        as one.
    :raises ValueError: when the two stacks have different shapes.
    """
    gt_frames = [np.asarray(frame) for frame in gt_masks]
    pred_frames = [np.asarray(frame) for frame in pred_masks]
    if len(gt_frames) != len(pred_frames):
        raise ValueError(
            f'the ground truth has {len(gt_frames)} frames and the tracking '
            f'has {len(pred_frames)}; they have to be the same movie')
    matched = {}
    shared = {}
    for index, (gt_frame, pred_frame) in enumerate(zip(gt_frames, pred_frames)):
        if gt_frame.shape != pred_frame.shape:
            raise ValueError(
                f'frame {index} is {gt_frame.shape} in the ground truth and '
                f'{pred_frame.shape} in the tracking')
        gt_labels, pred_labels, counts = overlap_counts(gt_frame, pred_frame)
        if gt_labels.size == 0 or pred_labels.size == 0:
            continue
        gt_areas = np.array([float((gt_frame == label).sum())
                             for label in gt_labels])
        for position, label in enumerate(gt_labels):
            area = gt_areas[position]
            if area <= 0:
                continue
            best = int(np.argmax(counts[position]))
            if counts[position, best] > DETECTION_OVERLAP * area:
                pred_vertex = (index, int(pred_labels[best]))
                matched[(index, int(label))] = pred_vertex
                shared[pred_vertex] = shared.get(pred_vertex, 0) + 1
    return matched, shared


def aogm_costs(gt_masks, pred_masks, gt_lineage=None, pred_lineage=None,
               weights=None):
    """Count the edit operations between a computed tracking and the truth.

    Vertex operations follow the detection matching: a ground-truth object
    nothing detects is a false negative, a computed segment that detects
    nothing is a false positive, and a computed segment that detects several
    objects has to be split once per extra object.

    Edge operations are counted on the mapped graph. A computed edge is
    carried onto the ground truth only when BOTH of its endpoints detect
    exactly one ground-truth object each; anything else has an endpoint that
    does not exist in the truth, so the edge is deleted. A ground-truth edge
    with no computed counterpart is added, and one whose counterpart has the
    other semantics -- a division link scored as a plain track link, or the
    reverse -- is changed rather than rebuilt.

    :param gt_masks: ground-truth label stack, label = track id.
    :param pred_masks: computed label stack, label = track id.
    :param gt_lineage: mapping of ground-truth label to parent label.
    :param pred_lineage: mapping of computed label to parent label.
    :param weights: AOGM weights. Anything left out keeps its
        :data:`AOGM_WEIGHTS` value, so ``{'fp': 10.0}`` changes the price of a
        false positive and nothing else.
    :returns: dict of the six operation counts (``fn``, ``fp``, ``ns``,
        ``ed``, ``ea``, ``ec``), the graph sizes (``n_gt_vertices``,
        ``n_pred_vertices``, ``n_gt_edges``, ``n_pred_edges``) and the four
        costs: ``aogm_d`` and ``aogm_d0`` for detection, ``aogm`` and
        ``aogm_0`` for tracking.
    """
    weights = dict(AOGM_WEIGHTS, **dict(weights or {}))
    gt_graph = tracking_graph(gt_masks, gt_lineage)
    pred_graph = tracking_graph(pred_masks, pred_lineage)
    matched, shared = match_vertices(gt_masks, pred_masks)

    false_negatives = len(gt_graph['vertices']) - len(matched)
    false_positives = sum(1 for vertex in pred_graph['vertices']
                          if vertex not in shared)
    splits = sum(count - 1 for count in shared.values() if count > 1)

    reverse = {}
    for gt_vertex, pred_vertex in matched.items():
        if shared.get(pred_vertex, 0) == 1:
            reverse[pred_vertex] = gt_vertex

    mapped = {}
    deleted_edges = 0
    for (source, target), semantic in pred_graph['edges'].items():
        gt_source = reverse.get(source)
        gt_target = reverse.get(target)
        if gt_source is None or gt_target is None:
            deleted_edges += 1
            continue
        if (gt_source, gt_target) not in gt_graph['edges']:
            deleted_edges += 1
            continue
        mapped[(gt_source, gt_target)] = semantic

    added_edges = 0
    changed_edges = 0
    for edge, semantic in gt_graph['edges'].items():
        if edge not in mapped:
            added_edges += 1
        elif mapped[edge] != semantic:
            changed_edges += 1

    aogm_d = (weights['ns'] * splits + weights['fn'] * false_negatives
              + weights['fp'] * false_positives)
    aogm_d0 = weights['fn'] * len(gt_graph['vertices'])
    aogm = (aogm_d + weights['ed'] * deleted_edges
            + weights['ea'] * added_edges + weights['ec'] * changed_edges)
    aogm_0 = aogm_d0 + weights['ea'] * len(gt_graph['edges'])
    return {
        'fn': false_negatives,
        'fp': false_positives,
        'ns': splits,
        'ed': deleted_edges,
        'ea': added_edges,
        'ec': changed_edges,
        'n_gt_vertices': len(gt_graph['vertices']),
        'n_pred_vertices': len(pred_graph['vertices']),
        'n_gt_edges': len(gt_graph['edges']),
        'n_pred_edges': len(pred_graph['edges']),
        'aogm_d': aogm_d,
        'aogm_d0': aogm_d0,
        'aogm': aogm,
        'aogm_0': aogm_0,
    }


def _normalised(cost, reference):
    """Turn an AOGM cost into a score in ``[0, 1]``.

    :param cost: the weighted edit cost of the computed tracking.
    :param reference: the cost of building the ground truth from nothing.
    :returns: ``1 - min(cost, reference) / reference``, and ``1.0`` when there
        is nothing to build, which is the only value that does not punish a
        tracker for an empty ground truth.
    """
    if reference <= 0:
        return 1.0
    return 1.0 - min(cost, reference) / reference


def det_score(gt_masks, pred_masks, weights=None, costs=None):
    """The Cell Tracking Challenge's DET: detection only, no links.

    :param gt_masks: ground-truth label stack.
    :param pred_masks: computed label stack.
    :param weights: AOGM weights; :data:`AOGM_WEIGHTS` when ``None``.
    :param costs: an :func:`aogm_costs` result to reuse instead of measuring
        again.
    :returns: float in ``[0, 1]``; ``1.0`` when every object is detected once.
    """
    costs = costs or aogm_costs(gt_masks, pred_masks, weights=weights)
    return _normalised(costs['aogm_d'], costs['aogm_d0'])


def tra_score(gt_masks, pred_masks, gt_lineage=None, pred_lineage=None,
              weights=None, costs=None):
    """The Cell Tracking Challenge's TRA: detection and links together.

    :param gt_masks: ground-truth label stack, label = track id.
    :param pred_masks: computed label stack, label = track id.
    :param gt_lineage: mapping of ground-truth label to parent label.
    :param pred_lineage: mapping of computed label to parent label.
    :param weights: AOGM weights; :data:`AOGM_WEIGHTS` when ``None``.
    :param costs: an :func:`aogm_costs` result to reuse instead of measuring
        again.
    :returns: float in ``[0, 1]``; ``1.0`` for a tracking identical to the
        truth.
    """
    costs = costs or aogm_costs(gt_masks, pred_masks, gt_lineage,
                                pred_lineage, weights)
    return _normalised(costs['aogm'], costs['aogm_0'])


def identity_switches(gt_masks, pred_masks):
    """Count how often a ground-truth object changes computed identity.

    Each ground-truth track is walked in frame order over the frames where it
    was detected, and every change of computed label is one switch. Frames
    where the object was missed are skipped rather than ending the track, so
    an object that is lost and then found again under a NEW id is counted as a
    switch. That is the failure a biologist reads as "the cell became another
    cell", and a tracker that hides it by dropping the frame should not score
    better for dropping it.

    :param gt_masks: ground-truth label stack, label = track id.
    :param pred_masks: computed label stack, label = track id.
    :returns: dict with ``switches``, ``tracks_with_switches`` and
        ``gt_tracks``.
    """
    matched, _ = match_vertices(gt_masks, pred_masks)
    by_track = {}
    for (frame, label), (_, pred_label) in matched.items():
        by_track.setdefault(int(label), []).append((int(frame), int(pred_label)))
    switches = 0
    broken = 0
    for label, seen in by_track.items():
        ordered = [pred for _, pred in sorted(seen)]
        changes = sum(1 for index in range(len(ordered) - 1)
                      if ordered[index] != ordered[index + 1])
        switches += changes
        if changes:
            broken += 1
    gt_labels = set()
    for frame in gt_masks:
        for label in np.unique(np.asarray(frame)):
            if label > 0:
                gt_labels.add(int(label))
    return {'switches': switches, 'tracks_with_switches': broken,
            'gt_tracks': len(gt_labels)}


def score_tracking(gt_masks, pred_masks, gt_lineage=None, pred_lineage=None,
                   weights=None, name='tracking'):
    """Score one computed tracking against the truth, counts and all.

    :param gt_masks: ground-truth label stack, label = track id.
    :param pred_masks: computed label stack, label = track id.
    :param gt_lineage: mapping of ground-truth label to parent label.
    :param pred_lineage: mapping of computed label to parent label.
    :param weights: AOGM weights; :data:`AOGM_WEIGHTS` when ``None``.
    :param name: what the row is called, so several trackings concatenate into
        one table.
    :returns: dict carrying ``name``, ``det``, ``tra``, ``switches``,
        ``tracks_with_switches``, ``gt_tracks`` and every field of
        :func:`aogm_costs`.
    """
    costs = aogm_costs(gt_masks, pred_masks, gt_lineage, pred_lineage, weights)
    switches = identity_switches(gt_masks, pred_masks)
    row = {
        'name': name,
        'det': det_score(gt_masks, pred_masks, weights=weights, costs=costs),
        'tra': tra_score(gt_masks, pred_masks, gt_lineage, pred_lineage,
                         weights=weights, costs=costs),
    }
    row.update(switches)
    row.update(costs)
    return row


def score_stitcher(gt_masks, segmentation, gt_lineage=None,
                   iou_threshold=0.1, weights=None, name='iou-stitcher'):
    """Run the baseline stitcher on a segmentation and score what it produced.

    This is the number step 2 exists to produce, and the number step 3's fork
    has to beat.

    :param gt_masks: ground-truth label stack, label = track id.
    :param segmentation: per-frame segmentation whose labels mean nothing
        across frames. Pass the ground-truth stack itself to measure LINKING
        GIVEN PERFECT SEGMENTATION, which is the upper bound the plan asks for
        beside the end-to-end number.
    :param gt_lineage: mapping of ground-truth label to parent label.
    :param iou_threshold: smallest IoU the stitcher accepts as the same
        object.
    :param weights: AOGM weights; :data:`AOGM_WEIGHTS` when ``None``.
    :param name: what the row is called.
    :returns: ``(row, tracked, table)`` -- the :func:`score_tracking` row, the
        stitched label stack, and the stitcher's own link table.
    """
    tracked, table = stitch_by_iou(segmentation, iou_threshold=iou_threshold)
    row = score_tracking(gt_masks, tracked, gt_lineage=gt_lineage,
                         pred_lineage=None, weights=weights, name=name)
    row['iou_threshold'] = float(iou_threshold)
    return row, tracked, table


def scores_table(rows):
    """Put scored trackings in one table, best TRA first.

    :param rows: iterable of :func:`score_tracking` results.
    :returns: ``DataFrame`` with the scores first and the operation counts
        after them.
    """
    table = pd.DataFrame(list(rows))
    if len(table) == 0:
        return table
    leading = [column for column in
               ('name', 'det', 'tra', 'switches', 'tracks_with_switches')
               if column in table.columns]
    rest = [column for column in table.columns if column not in leading]
    return table[leading + rest].sort_values(
        'tra', ascending=False).reset_index(drop=True)
