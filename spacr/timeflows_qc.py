"""Check that timelapse annotation carries one label per object over time.

WHAT IT IS FOR
==============
The timeflows plan (``features/future/426``) rests on one claim about the
training data: an object keeps the SAME LABEL in every frame it appears in.
A time-flow head is trained to point from a pixel of an object in frame ``t``
at that object's centre in frame ``t+1``, and the only thing that says which
object in ``t+1`` is the same object is the label. If the annotation recycles
a label after its object dies or leaves, the model is trained to point at an
unrelated cell, and no later score catches it -- the held-out check scrambles
the same wrong labels and agrees with itself.

So this module reads label stacks and produces a table. It answers the four
questions step 1 of the plan asks, and it answers them as measurements rather
than impressions:

* **Is a label ever reused?** Every label that vanishes and comes back is
  listed with how far it moved across the gap, how much its mask overlaps the
  mask it had before the gap, and what a continuous object in the same data
  moves in one frame. A label that reappears on the other side of the field
  with no overlap is flagged; a label that flickers for one frame and returns
  where it was is reported as a gap and not as reuse.
* **How are divisions annotated?** Two new ids, or one child inheriting the
  parent's id. Both conventions exist in real annotation and the loss has to
  know which one it is being trained on.
* **What is the frame interval, and is it constant?** Within a dataset and
  across datasets. A displacement field silently encodes pixels per frame,
  so a set imaged every 30 s mixed with one imaged every 10 min is two
  problems trained as one.
* **How far does an object move between frames?** In pixels, and -- the ratio
  that decides whether the task is easy or hard -- in multiples of its own
  diameter. Displacement much smaller than the object is a field a small
  receptive field can carry. Displacement larger than the object means the
  vector has to point further than the head can see, and several identical
  cells are candidates at the far end of it.

WHY A SCRIPT AND NOT A LOOK
===========================
The defect this looks for is invisible by eye. A recycled id is a correct
frame and a correct frame, and only the pair is wrong. The plan calls it the
most common defect in tracking annotation and the easiest to miss, which is
why the output here is a table with a verdict per check and an exit code, and
why the detail tables behind every verdict are public: a flagged row is a row
somebody has to read, not a number to average.

WHAT IT DOES NOT DO
===================
It judges the ANNOTATION, not a tracker. Scoring a predicted tracking against
a ground truth is :mod:`spacr.timeflows_baseline`, which is step 2 of the same
plan. It holds no graphical code and opens no dialogs.
"""

from __future__ import annotations

import argparse
import os
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .timeflows_baseline import overlap_counts

#: Verdict vocabulary, in the order a reader should worry about them. ``fail``
#: means the training data is not what the plan assumes, ``check`` means a
#: human has to read the detail table before the data is used, ``ok`` means
#: the measurement supports the assumption, and ``unknown`` means the input
#: did not carry what the check needs.
VERDICTS = ('ok', 'check', 'fail', 'unknown')

#: A gap across which the object moved more than this multiple of what a
#: continuous object in the same data moves per frame is called reuse rather
#: than a gap. Three is deliberately generous: the flag is an accusation
#: against the annotation, and a false one costs somebody a day of reading.
REUSE_DISPLACEMENT_FACTOR = 3.0

#: Fraction of a child's area that has to come from one parent object before
#: the child is called that parent's. Half is the Cell Tracking Challenge's
#: own detection criterion and has the property that at most one parent can
#: satisfy it.
CHILD_OVERLAP = 0.5

#: Per-frame displacement, as a multiple of the object's own diameter, above
#: which the time-flow head is being asked to point further than it can see.
#: Below the first number the task is the easy one Cellpose's spatial flows
#: already solve at this scale; above the second the plan's own honest failure
#: mode applies and step 1 is supposed to say so before any model is written.
DISPLACEMENT_EASY = 0.5
DISPLACEMENT_HARD = 1.0


def _frames_of(masks):
    """Return ``masks`` as a list of 2-D integer label frames.

    :param masks: a ``(T, Y, X)`` array, or any sequence of 2-D label frames.
    :returns: list of 2-D ``numpy`` integer arrays, one per frame.
    :raises ValueError: when a frame is not two-dimensional, which is how a
        ``(T, Z, Y, X)`` volume arrives here. Every measurement below reasons
        in the image plane, and a volume would be measured without complaint
        and produce plausible nonsense -- the same trap
        :func:`spacr.timelapse._require_2d_frames` exists for.
    """
    if isinstance(masks, (list, tuple)):
        frames = [np.asarray(frame) for frame in masks]
    else:
        array = np.asarray(masks)
        frames = [array[index] for index in range(array.shape[0])]
    for index, frame in enumerate(frames):
        if frame.ndim != 2:
            raise ValueError(
                f"timeflows QC needs a (T, Y, X) stack of 2-D label frames and "
                f"frame {index} has {frame.ndim} dimension(s) (shape "
                f"{frame.shape}). Every check here measures in the image plane, "
                f"so a volume handed to it would be measured without failing "
                f"and the displacements would be fiction."
            )
    return frames


def label_frames(masks):
    """Measure every labelled object in every frame.

    :param masks: label stack, ``(T, Y, X)`` or a sequence of 2-D frames.
    :returns: ``DataFrame`` with one row per object per frame and the columns
        ``frame``, ``label``, ``area``, ``y``, ``x`` and ``diameter``, where
        the diameter is the area-equivalent circle diameter in pixels.
    """
    frames = _frames_of(masks)
    rows = []
    grid_y = None
    grid_x = None
    for index, frame in enumerate(frames):
        flat = np.asarray(frame).ravel()
        if flat.size == 0:
            continue
        if grid_y is None or grid_y.size != flat.size:
            height, width = frame.shape
            grid_y = np.repeat(np.arange(height, dtype=float), width)
            grid_x = np.tile(np.arange(width, dtype=float), height)
        positive = flat > 0
        if not positive.any():
            continue
        labels = flat[positive].astype(np.int64)
        counts = np.bincount(labels)
        sum_y = np.bincount(labels, weights=grid_y[positive])
        sum_x = np.bincount(labels, weights=grid_x[positive])
        present = np.nonzero(counts)[0]
        for label in present:
            area = float(counts[label])
            rows.append({
                'frame': index,
                'label': int(label),
                'area': area,
                'y': float(sum_y[label] / area),
                'x': float(sum_x[label] / area),
                'diameter': float(np.sqrt(4.0 * area / np.pi)),
            })
    table = pd.DataFrame(
        rows, columns=['frame', 'label', 'area', 'y', 'x', 'diameter'])
    return table.sort_values(['label', 'frame']).reset_index(drop=True)


def label_spans(frames_table):
    """Summarise where each label lives in the movie.

    :param frames_table: the table :func:`label_frames` returns.
    :returns: ``DataFrame`` with one row per label carrying ``label``,
        ``first_frame``, ``last_frame``, ``n_frames``, ``n_missing`` and
        ``missing_frames``. ``n_missing`` counts frames between the first and
        the last appearance in which the label is absent, which is the only
        kind of gap that can hide a recycled id.
    """
    rows = []
    for label, group in frames_table.groupby('label', sort=True):
        seen = sorted(int(frame) for frame in group['frame'])
        first, last = seen[0], seen[-1]
        missing = [frame for frame in range(first, last + 1)
                   if frame not in set(seen)]
        rows.append({
            'label': int(label),
            'first_frame': first,
            'last_frame': last,
            'n_frames': len(seen),
            'n_missing': len(missing),
            'missing_frames': tuple(missing),
        })
    return pd.DataFrame(rows, columns=[
        'label', 'first_frame', 'last_frame', 'n_frames', 'n_missing',
        'missing_frames'])


def displacement_table(frames_table):
    """Per-frame displacement of every label that survives a frame.

    Only CONSECUTIVE frames are measured. A label that reappears after a gap
    contributes nothing here, because the distance it covered is not a
    per-frame displacement and averaging it in would inflate the number this
    module exists to report.

    :param frames_table: the table :func:`label_frames` returns.
    :returns: ``DataFrame`` with ``label``, ``frame``, ``displacement`` in
        pixels, ``diameter`` of the object in the earlier frame, and
        ``displacement_over_diameter``.
    """
    rows = []
    for label, group in frames_table.groupby('label', sort=True):
        ordered = group.sort_values('frame')
        frames = ordered['frame'].to_numpy()
        ys = ordered['y'].to_numpy()
        xs = ordered['x'].to_numpy()
        diameters = ordered['diameter'].to_numpy()
        for index in range(len(frames) - 1):
            if int(frames[index + 1]) != int(frames[index]) + 1:
                continue
            step = float(np.hypot(ys[index + 1] - ys[index],
                                  xs[index + 1] - xs[index]))
            diameter = float(diameters[index])
            rows.append({
                'label': int(label),
                'frame': int(frames[index]),
                'displacement': step,
                'diameter': diameter,
                'displacement_over_diameter': (
                    step / diameter if diameter > 0 else float('nan')),
            })
    return pd.DataFrame(rows, columns=[
        'label', 'frame', 'displacement', 'diameter',
        'displacement_over_diameter'])


def _mask_iou(first, second):
    """Intersection over union of two boolean masks.

    :param first: boolean array.
    :param second: boolean array of the same shape.
    :returns: float in ``[0, 1]``; ``0.0`` when both masks are empty.
    """
    intersection = float(np.logical_and(first, second).sum())
    union = float(np.logical_or(first, second).sum())
    if union == 0.0:
        return 0.0
    return intersection / union


def recycled_labels(masks, frames_table=None, max_displacement=None):
    """List every label that disappears and comes back, and judge each one.

    A recycled id -- the same number given to an unrelated object after the
    first one left -- is the defect the whole plan is exposed to, and it looks
    exactly like a tracking gap from one frame. What separates them is
    distance: a real object that was missed for a frame is found again near
    where it was, and usually still overlapping its own mask. So each gap is
    reported with the distance covered per missing frame, the overlap between
    the mask before the gap and the mask after it, and the distance a
    continuous object in the same data covers in one frame.

    :param masks: label stack, ``(T, Y, X)`` or a sequence of 2-D frames.
    :param frames_table: the table :func:`label_frames` returns, when it has
        already been measured. Measured here when ``None``.
    :param max_displacement: distance per missing frame, in pixels, above
        which a gap is called reuse. ``None`` measures it from the data as
        :data:`REUSE_DISPLACEMENT_FACTOR` times the 95th percentile of the
        continuous per-frame displacement, falling back to the median object
        diameter when nothing in the stack survives a frame.
    :returns: ``DataFrame`` with one row per gap: ``label``, ``gap_start``,
        ``gap_end``, ``gap_frames``, ``displacement``,
        ``displacement_per_frame``, ``iou_across_gap``, ``allowed`` and
        ``verdict`` (``'gap'`` or ``'reuse-suspected'``).
    """
    frames = _frames_of(masks)
    if frames_table is None:
        frames_table = label_frames(frames)
    steps = displacement_table(frames_table)
    if max_displacement is None:
        if len(steps) > 0:
            max_displacement = REUSE_DISPLACEMENT_FACTOR * float(
                np.percentile(steps['displacement'].to_numpy(), 95))
        elif len(frames_table) > 0:
            max_displacement = float(frames_table['diameter'].median())
        else:
            max_displacement = 0.0
    spans = label_spans(frames_table)
    positions = {
        (int(row.frame), int(row.label)): (float(row.y), float(row.x))
        for row in frames_table.itertuples()
    }
    rows = []
    for row in spans.itertuples():
        if row.n_missing == 0:
            continue
        seen = sorted(
            int(frame) for frame in
            frames_table.loc[frames_table['label'] == row.label, 'frame'])
        for index in range(len(seen) - 1):
            before, after = seen[index], seen[index + 1]
            if after == before + 1:
                continue
            gap_frames = after - before - 1
            y0, x0 = positions[(before, int(row.label))]
            y1, x1 = positions[(after, int(row.label))]
            distance = float(np.hypot(y1 - y0, x1 - x0))
            per_frame = distance / float(gap_frames + 1)
            overlap = _mask_iou(
                np.asarray(frames[before]) == row.label,
                np.asarray(frames[after]) == row.label)
            suspect = overlap == 0.0 and per_frame > max_displacement
            rows.append({
                'label': int(row.label),
                'gap_start': before + 1,
                'gap_end': after - 1,
                'gap_frames': gap_frames,
                'displacement': distance,
                'displacement_per_frame': per_frame,
                'iou_across_gap': overlap,
                'allowed': float(max_displacement),
                'verdict': 'reuse-suspected' if suspect else 'gap',
            })
    return pd.DataFrame(rows, columns=[
        'label', 'gap_start', 'gap_end', 'gap_frames', 'displacement',
        'displacement_per_frame', 'iou_across_gap', 'allowed', 'verdict'])


def division_events(masks, child_overlap=CHILD_OVERLAP):
    """Find divisions and say which labelling convention they follow.

    A child is called an object's child when at least ``child_overlap`` of the
    child's own area came from that object. Two or more children of one parent
    in one step is a division, and the convention is read off the ids: either
    one child carries the parent's id forward, or both children are new.

    :param masks: label stack, ``(T, Y, X)`` or a sequence of 2-D frames.
    :param child_overlap: fraction of the child's area that has to come from
        the parent. Default :data:`CHILD_OVERLAP`.
    :returns: ``DataFrame`` with ``frame`` (the frame the children are in),
        ``parent_label``, ``n_children``, ``child_labels`` and ``convention``,
        which is ``'child_inherits_parent'`` or ``'children_are_new'``.
    """
    frames = _frames_of(masks)
    rows = []
    for index in range(len(frames) - 1):
        previous, current = frames[index], frames[index + 1]
        labels_previous, labels_current, counts = overlap_counts(
            previous, current)
        if labels_previous.size == 0 or labels_current.size == 0:
            continue
        areas_current = np.array(
            [float((np.asarray(current) == label).sum())
             for label in labels_current])
        children = {}
        for position, label in enumerate(labels_current):
            area = areas_current[position]
            if area <= 0:
                continue
            fractions = counts[:, position] / area
            best = int(np.argmax(fractions))
            if fractions[best] >= child_overlap:
                children.setdefault(int(labels_previous[best]), []).append(
                    int(label))
        for parent, child_labels in sorted(children.items()):
            if len(child_labels) < 2:
                continue
            inherits = parent in child_labels
            rows.append({
                'frame': index + 1,
                'parent_label': parent,
                'n_children': len(child_labels),
                'child_labels': tuple(sorted(child_labels)),
                'convention': ('child_inherits_parent' if inherits
                               else 'children_are_new'),
            })
    return pd.DataFrame(rows, columns=[
        'frame', 'parent_label', 'n_children', 'child_labels', 'convention'])


def frame_interval_summary(timestamps=None, frame_interval=None,
                           tolerance=1e-6):
    """Reduce frame timing to an interval and a verdict about its constancy.

    :param timestamps: acquisition time of each frame, in any one unit. Two
        frames are the minimum that says anything.
    :param frame_interval: a stated interval, used when no timestamps are
        given. It is reported as stated and never contradicts a measurement.
    :param tolerance: absolute spread between the largest and smallest
        interval that still counts as constant.
    :returns: dict with ``interval``, ``spread``, ``source`` (``'measured'``,
        ``'stated'`` or ``'missing'``) and ``verdict``.
    """
    if timestamps is not None and len(timestamps) >= 2:
        times = np.asarray(timestamps, dtype=float)
        intervals = np.diff(times)
        spread = float(intervals.max() - intervals.min())
        return {
            'interval': float(np.median(intervals)),
            'spread': spread,
            'source': 'measured',
            'verdict': 'ok' if spread <= tolerance else 'check',
        }
    if frame_interval is not None:
        return {
            'interval': float(frame_interval),
            'spread': 0.0,
            'source': 'stated',
            'verdict': 'check',
        }
    return {
        'interval': float('nan'),
        'spread': float('nan'),
        'source': 'missing',
        'verdict': 'unknown',
    }


def _row(dataset, check, value, verdict, detail):
    """Build one row of the QC table.

    :param dataset: name the stack was audited under.
    :param check: the question this row answers.
    :param value: the measurement, as a string.
    :param verdict: one of :data:`VERDICTS`.
    :param detail: a sentence saying what the measurement means.
    :returns: dict with those five keys.
    """
    return {'dataset': dataset, 'check': check, 'value': value,
            'verdict': verdict, 'detail': detail}


def audit_label_consistency(masks, dataset='dataset', timestamps=None,
                            frame_interval=None, max_displacement=None):
    """Answer step 1's four questions about one label stack, as a table.

    :param masks: label stack, ``(T, Y, X)`` or a sequence of 2-D frames.
    :param dataset: the name the rows are reported under.
    :param timestamps: acquisition time of each frame, when it is known.
    :param frame_interval: a stated frame interval, used when there are no
        timestamps.
    :param max_displacement: passed to :func:`recycled_labels`.
    :returns: ``DataFrame`` with the columns ``dataset``, ``check``,
        ``value``, ``verdict`` and ``detail``, one row per check.
    """
    frames = _frames_of(masks)
    table = label_frames(frames)
    spans = label_spans(table)
    steps = displacement_table(table)
    gaps = recycled_labels(frames, frames_table=table,
                           max_displacement=max_displacement)
    divisions = division_events(frames)
    timing = frame_interval_summary(timestamps, frame_interval)

    rows = [
        _row(dataset, 'frames', str(len(frames)), 'ok',
             'frames in the stack'),
        _row(dataset, 'objects', str(len(table)), 'ok',
             'labelled objects summed over frames'),
        _row(dataset, 'labels', str(len(spans)), 'ok',
             'distinct labels in the stack'),
    ]

    suspected = gaps.loc[gaps['verdict'] == 'reuse-suspected'] if len(gaps) \
        else gaps
    if len(gaps) == 0:
        rows.append(_row(dataset, 'label_reuse', '0 gaps', 'ok',
                         'no label disappears and comes back, so no label can '
                         'have been recycled'))
    elif len(suspected) == 0:
        rows.append(_row(
            dataset, 'label_reuse', f'{len(gaps)} gaps, 0 suspected', 'check',
            'every gap closes within the distance a continuous object covers, '
            'so these read as missed detections rather than recycled ids; the '
            'detail table lists them'))
    else:
        rows.append(_row(
            dataset, 'label_reuse',
            f'{len(gaps)} gaps, {len(suspected)} suspected', 'fail',
            'a label comes back with no overlap and further away than a '
            'continuous object travels, which is what a recycled id looks '
            'like; read the detail table before training on this'))

    if len(divisions) == 0:
        rows.append(_row(dataset, 'division_convention', 'no divisions found',
                         'unknown',
                         'nothing in this stack splits into two children, so '
                         'the convention cannot be read off it'))
    else:
        conventions = sorted(set(divisions['convention']))
        counts = divisions['convention'].value_counts()
        value = ', '.join(f'{name} x{int(counts[name])}'
                          for name in conventions)
        if len(conventions) == 1:
            rows.append(_row(dataset, 'division_convention', value, 'ok',
                             'one convention throughout, which is what the '
                             'loss has to be told'))
        else:
            rows.append(_row(dataset, 'division_convention', value, 'fail',
                             'the annotation uses both conventions, so a '
                             'single rule for divisions would be wrong on '
                             'some of it'))

    interval_value = ('unknown' if timing['source'] == 'missing'
                      else f"{timing['interval']:.6g} ({timing['source']})")
    interval_detail = {
        'measured': (f"intervals spread by {timing['spread']:.6g}; the "
                     'displacement field encodes pixels per frame, so a '
                     'varying interval is two problems trained as one'),
        'stated': ('taken from the caller and not measured; timestamps would '
                   'confirm it'),
        'missing': ('no timestamps and no stated interval, so nothing here '
                    'says what a pixel of displacement means in time'),
    }[timing['source']]
    rows.append(_row(dataset, 'frame_interval', interval_value,
                     timing['verdict'], interval_detail))

    if len(steps) == 0:
        rows.append(_row(dataset, 'displacement', 'no object survives a frame',
                         'unknown',
                         'nothing appears in two consecutive frames, so there '
                         'is no displacement to measure'))
    else:
        pixels = steps['displacement'].to_numpy()
        ratios = steps['displacement_over_diameter'].to_numpy()
        ratios = ratios[np.isfinite(ratios)]
        median = float(np.median(pixels))
        p95 = float(np.percentile(pixels, 95))
        rows.append(_row(
            dataset, 'displacement_px',
            f'median {median:.3g}, p95 {p95:.3g}', 'ok',
            'per-frame displacement of objects that survive a frame'))
        if ratios.size:
            ratio_p95 = float(np.percentile(ratios, 95))
            if ratio_p95 <= DISPLACEMENT_EASY:
                verdict = 'ok'
                detail = ('objects move much less than their own size, so the '
                          'time head can see its own target')
            elif ratio_p95 <= DISPLACEMENT_HARD:
                verdict = 'check'
                detail = ('objects move a noticeable fraction of their own '
                          'size; the head needs a receptive field wider than '
                          'the object')
            else:
                verdict = 'fail'
                detail = ('objects move further than their own diameter, '
                          'which is the plan\'s own stated failure mode: the '
                          'vector has to point past what the head can see and '
                          'several identical cells are candidates')
            rows.append(_row(
                dataset, 'displacement_over_diameter',
                f'p95 {ratio_p95:.3g}', verdict, detail))

    return pd.DataFrame(rows, columns=[
        'dataset', 'check', 'value', 'verdict', 'detail'])


def audit_datasets(stacks, timestamps=None, frame_intervals=None,
                   max_displacement=None):
    """Audit several label stacks and compare their frame intervals.

    :param stacks: mapping of dataset name to label stack.
    :param timestamps: optional mapping of dataset name to frame timestamps.
    :param frame_intervals: optional mapping of dataset name to a stated
        interval.
    :param max_displacement: passed to :func:`recycled_labels`.
    :returns: ``DataFrame`` of every dataset's rows, followed by one
        ``frame_interval_across_datasets`` row reported under the dataset name
        ``all``. Two datasets imaged at different intervals are a different
        problem from one dataset imaged unevenly, and only this row sees it.
    """
    timestamps = dict(timestamps or {})
    frame_intervals = dict(frame_intervals or {})
    tables = []
    intervals = {}
    for name, masks in stacks.items():
        table = audit_label_consistency(
            masks, dataset=name, timestamps=timestamps.get(name),
            frame_interval=frame_intervals.get(name),
            max_displacement=max_displacement)
        tables.append(table)
        timing = frame_interval_summary(
            timestamps.get(name), frame_intervals.get(name))
        if timing['source'] != 'missing':
            intervals[name] = timing['interval']
    if len(stacks) > 1:
        if len(intervals) < len(stacks):
            tables.append(pd.DataFrame([_row(
                'all', 'frame_interval_across_datasets',
                f'{len(intervals)} of {len(stacks)} known', 'unknown',
                'some datasets carry no interval, so they cannot be compared '
                'with the ones that do')]))
        else:
            values = np.array(list(intervals.values()), dtype=float)
            spread = float(values.max() - values.min())
            same = spread <= 1e-6
            tables.append(pd.DataFrame([_row(
                'all', 'frame_interval_across_datasets',
                f'{values.min():.6g} to {values.max():.6g}',
                'ok' if same else 'check',
                'one interval across every dataset' if same else
                'the datasets are imaged at different intervals, so the head '
                'has to be conditioned on the interval or trained across the '
                'range rather than on the mixture')]))
    if not tables:
        return pd.DataFrame(columns=[
            'dataset', 'check', 'value', 'verdict', 'detail'])
    return pd.concat(tables, ignore_index=True)


def format_qc_table(table, width=64):
    """Render a QC table as aligned text.

    :param table: the table :func:`audit_label_consistency` returns.
    :param width: how much of the detail sentence to print per row.
    :returns: the table as a string, one line per row, with no trailing
        newline.
    """
    if len(table) == 0:
        return 'no rows'
    columns = ['dataset', 'check', 'value', 'verdict']
    widths = {name: max(len(name), int(table[name].astype(str).str.len().max()))
              for name in columns}
    lines = []
    header = '  '.join(name.upper().ljust(widths[name]) for name in columns)
    lines.append(f'{header}  DETAIL')
    for row in table.itertuples():
        values = {'dataset': row.dataset, 'check': row.check,
                  'value': row.value, 'verdict': row.verdict}
        rendered = '  '.join(str(values[name]).ljust(widths[name])
                             for name in columns)
        detail = str(row.detail)
        if len(detail) > width:
            detail = detail[:width - 1] + '…'
        lines.append(f'{rendered}  {detail}')
    return '\n'.join(lines)


def load_label_stack(path):
    """Read a label stack from a file or a folder of frames.

    :param path: ``.npy`` file, ``.tif``/``.tiff`` file, or a folder holding
        one file per frame, read in sorted name order.
    :returns: ``(T, Y, X)`` integer array.
    :raises ValueError: when the folder holds nothing readable, or the suffix
        is not one of the three above.
    """
    if os.path.isdir(path):
        names = sorted(name for name in os.listdir(path)
                       if name.lower().endswith(('.npy', '.tif', '.tiff')))
        if not names:
            raise ValueError(
                f'{path} holds no .npy, .tif or .tiff frames to read')
        frames = [load_label_stack(os.path.join(path, name))
                  for name in names]
        frames = [frame if frame.ndim == 2 else frame[0] for frame in frames]
        return np.stack(frames)
    lowered = str(path).lower()
    if lowered.endswith('.npy'):
        return np.asarray(np.load(path))
    if lowered.endswith(('.tif', '.tiff')):
        import tifffile
        return np.asarray(tifffile.imread(path))
    raise ValueError(
        f'{path} is not a label stack this reads: give a .npy or .tif file, '
        f'or a folder of them')


def main(argv=None):
    """Run the QC over one or more label stacks and print the table.

    :param argv: command line arguments; ``sys.argv[1:]`` when ``None``.
    :returns: ``0`` when no check failed, ``1`` when one did. A failure here
        is a statement about the ANNOTATION, not about this program: it means
        the data does not support what the timeflows plan assumes about it.
    """
    parser = argparse.ArgumentParser(
        prog='timeflows-label-qc',
        description='Check that timelapse labels identify the same object '
                    'across frames.')
    parser.add_argument('paths', nargs='+',
                        help='label stacks: .npy, .tif, or a folder of frames')
    parser.add_argument('--frame-interval', type=float, default=None,
                        help='stated interval between frames, when the data '
                             'carries no timestamps')
    parser.add_argument('--max-displacement', type=float, default=None,
                        help='pixels per frame above which a gap is called a '
                             'recycled label; measured from the data when '
                             'left out')
    parser.add_argument('--csv', default=None,
                        help='write the table here as well as printing it')
    parser.add_argument('--detail', default=None,
                        help='write the per-gap and per-division tables into '
                             'this folder')
    arguments = parser.parse_args(argv)

    stacks = {}
    for path in arguments.paths:
        stacks[os.path.basename(str(path).rstrip(os.sep)) or str(path)] = \
            load_label_stack(path)
    intervals = ({name: arguments.frame_interval for name in stacks}
                 if arguments.frame_interval is not None else None)
    table = audit_datasets(stacks, frame_intervals=intervals,
                           max_displacement=arguments.max_displacement)
    print(format_qc_table(table))
    if arguments.csv:
        table.to_csv(arguments.csv, index=False)
        print(f'table written to {arguments.csv}')
    if arguments.detail:
        os.makedirs(arguments.detail, exist_ok=True)
        for name, masks in stacks.items():
            frames_table = label_frames(masks)
            recycled_labels(
                masks, frames_table=frames_table,
                max_displacement=arguments.max_displacement).to_csv(
                    os.path.join(arguments.detail, f'{name}_gaps.csv'),
                    index=False)
            division_events(masks).to_csv(
                os.path.join(arguments.detail, f'{name}_divisions.csv'),
                index=False)
            displacement_table(frames_table).to_csv(
                os.path.join(arguments.detail, f'{name}_displacement.csv'),
                index=False)
        print(f'detail tables written to {arguments.detail}')
    return 1 if (table['verdict'] == 'fail').any() else 0


if __name__ == '__main__':
    raise SystemExit(main())
