"""Independent pixel/identity checks for a small unambiguous tracking demo.

This deliberately rejects competing overlap candidates instead of pretending
to validate a general linker. It never imports spaCR's tracking functions.
"""
from collections import defaultdict
import math
import statistics

import numpy as np


def verify_worker_passes(records):
    """Require the displayed baseline/strict/restored values to have actually run.

    Typing may emit extra intermediate values. They may be recorded but cannot
    stand in for the final .1 pass, even when its resulting tracks coincide.
    """
    if (len(records) < 3 or records[0].get('threshold') != .1 or
            records[-1].get('threshold') != .1 or
            not any(r.get('threshold') == 1 for r in records)):
        raise ValueError('The actual workers did not run the requested baseline, strict and restored values')
    if any(r.get('segmented') is not False or r.get('error') != '' for r in records):
        raise ValueError('Every actual worker must confirm no segmentation and no error')
    return dict(observed_results=len(records), initial_threshold=.1, strict_threshold=1,
                final_threshold=.1, segmentation_performed=False)


def reference_partitions(masks, threshold):
    """Connect only unique above-threshold pixel overlaps between adjacent frames."""
    masks = np.asarray(masks)
    if (masks.ndim != 3 or masks.shape[0] < 2 or
            not np.issubdtype(masks.dtype, np.integer) or np.any(masks < 0)):
        raise ValueError('Expected at least two nonnegative integer label frames')
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('The independently checked overlap threshold must be in (0, 1]')
    objects = [{int(label): frame == label for label in np.unique(frame) if label > 0}
               for frame in masks]
    if any(not frame for frame in objects):
        raise ValueError('This bounded teaching check requires objects in every frame')
    identity, groups = {}, defaultdict(set)
    next_identity = 0
    for frame, current in enumerate(objects):
        matched = {}
        if frame:
            previous = objects[frame - 1]
            for label, pixels in current.items():
                candidates = []
                for old_label, old_pixels in previous.items():
                    intersection = int(np.count_nonzero(pixels & old_pixels))
                    union = int(np.count_nonzero(pixels | old_pixels))
                    if intersection / union >= threshold:
                        candidates.append(old_label)
                if len(candidates) > 1:
                    raise ValueError('Competing overlaps need a different independent audit')
                if candidates:
                    old_label = candidates[0]
                    if old_label in matched.values():
                        raise ValueError('A split cannot be represented as an unambiguous continuation')
                    matched[label] = old_label
        for label in current:
            if label in matched:
                group = identity[(frame - 1, matched[label])]
            else:
                next_identity += 1
                group = next_identity
            identity[(frame, label)] = group
            groups[group].add((frame, label))
    return {frozenset(group) for group in groups.values()}


def verify_tracks(masks, records, threshold, *, stats=None, minimum_length=3,
                  displacement_limit=50):
    """Check every object, centroid, track partition and displayed numeric indicator."""
    masks = np.asarray(masks)
    expected = reference_partitions(masks, threshold)
    seen, groups = set(), defaultdict(list)
    expected_objects = set().union(*expected)
    for row in records:
        values = [row.get(key) for key in ('frame', 'original_label', 'track_id')]
        if any(isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or
               not math.isfinite(value) or value != int(value) for value in values):
            raise ValueError('Track identities must be finite integers')
        frame, label, track = map(int, values)
        key = (frame, label)
        if key not in expected_objects or track <= 0:
            raise ValueError('The track table names an absent object or invalid identity')
        if key in seen or any(item[0] == frame for item in groups[track]):
            raise ValueError('An object or track occurs twice in one frame')
        seen.add(key)
        y, x = np.nonzero(masks[frame] == label)
        coordinates = (float(x.mean()), float(y.mean()))
        for name, value in zip(('x', 'y'), coordinates):
            if (not isinstance(row.get(name), (int, float, np.number)) or
                    not math.isfinite(row[name]) or
                    not math.isclose(row[name], value, rel_tol=0, abs_tol=1e-8)):
                raise ValueError('A saved track centroid differs from its actual label pixels')
        groups[track].append((frame, label, *coordinates))
    if seen != expected_objects:
        raise ValueError('The track table omits detected objects')
    actual = {frozenset((row[0], row[1]) for row in rows) for rows in groups.values()}
    if actual != expected:
        raise ValueError('Track continuations differ from unique pixel-overlap evidence')
    lengths = [len(rows) for rows in groups.values()]
    steps = []
    for rows in groups.values():
        ordered = sorted(rows)
        steps.extend(math.hypot(b[2] - a[2], b[3] - a[3]) for a, b in zip(ordered, ordered[1:]))
    numbers = dict(n_frames=len(masks), n_tracks=len(groups),
        mean_length=statistics.mean(lengths), median_length=statistics.median(lengths),
        n_short=sum(n < minimum_length for n in lengths), min_length=minimum_length,
        starts_after_first=sum(min(r[0] for r in rows) > 0 for rows in groups.values()),
        ends_before_last=sum(max(r[0] for r in rows) < len(masks) - 1 for rows in groups.values()),
        suspicious_jumps=sum(step > displacement_limit for step in steps),
        max_step=max(steps, default=0), objects_per_frame=len(seen) / len(masks),
        displacement_limit=displacement_limit)
    if stats is not None:
        for key, value in numbers.items():
            if (key not in stats or not isinstance(stats[key], (int, float, np.number)) or
                    not math.isfinite(stats[key]) or
                    not math.isclose(stats[key], value, rel_tol=0, abs_tol=1e-8)):
                raise ValueError('The displayed tracking indicator differs: ' + key)
    return dict(objects_checked=len(seen), centroid_values_checked=2 * len(seen),
                label_pixels_checked=int(masks.size), iou_threshold=threshold,
                independent_statistics=numbers, biological_validation=False)
