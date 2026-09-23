"""Evaluate Timeflows linking on explicit CTC full masks and tracking markers.

This is linking conditioned on the supplied segmentations, not end-to-end
segmentation/tracking accuracy. GT segmentation may be synthetic; ST is silver
annotation. Neither tracking markers alone nor missing full masks are accepted.
Ambiguous or unmarked segmentation objects are excluded and counted.

Pair selection is evenly spaced before reading pixels and streams one pair at
a time. Reports stratify true successors by displacement/source diameter and
by neighbours within five source diameters. Controls are IoU, zero motion,
oracle targets, a random time head on the checkpoint encoder, and the trained
model with frame t copied to t+1. An untrained head is not expected to score at
random-label chance: position and assignment already supply useful priors.

Checkpoint metadata must name its training movies; evaluation refuses those
movies, including aliases. No weights are downloaded. Outputs go to a new
directory; summary.json is written only after every selected pair completes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import tifffile

from spacr.timeflows_baseline import link_frames
from spacr import timeflows_model as tm


def digest(path):
    """Hash one input without loading the file into memory."""
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def indexed(folder, prefix):
    """Index only complete, two-dimensional CTC filenames, never slice masks."""
    import re

    pattern = re.compile(re.escape(prefix) + r'(\d+)\.tiff?$', re.IGNORECASE)
    found = {}
    for path in Path(folder).glob('*'):
        match = pattern.fullmatch(path.name)
        if match:
            number = int(match[1])
            if number in found:
                raise ValueError(f'Duplicate frame {number} in {folder}')
            found[number] = path
    return found


def selected_pairs(movie, sequence, segmentation, gaps, maximum):
    """Select real endpoints using explicit full segmentation and TRA files."""
    if len(sequence) != 2 or not sequence.isascii() or not sequence.isdecimal():
        raise ValueError('CTC sequences must be two-digit directory names')
    if maximum < 0 or segmentation not in ('GT', 'ST'):
        raise ValueError('Use GT or ST segmentation and a non-negative pair limit')
    movie = Path(movie)
    sources = [indexed(movie / sequence, 't'),
               indexed(movie / f'{sequence}_{segmentation}' / 'SEG', 'man_seg'),
               indexed(movie / f'{sequence}_GT' / 'TRA', 'man_track')]
    usable = set.intersection(*(set(source) for source in sources))
    if not usable:
        raise ValueError(f'No matched images, full {segmentation}/SEG masks and GT/TRA markers in {movie}/{sequence}')
    result = []
    for gap in gaps:
        if gap < 1:
            raise ValueError('Frame gaps must be positive')
        starts = sorted(number for number in usable if number + gap in usable)
        if not starts:
            raise ValueError(f'No complete pairs at gap {gap} in {movie}/{sequence}')
        if maximum and len(starts) > maximum:
            positions = np.linspace(0, len(starts) - 1, maximum).round().astype(int)
            starts = [starts[index] for index in sorted(set(positions.tolist()))]
        for start in starts:
            result.append({'sequence': sequence, 'gap_frames': gap,
                           'frame_numbers': [start, start + gap],
                           'files': [[source[number] for source in sources]
                                     for number in (start, start + gap)]})
    return result


def tracked_masks(segmentation, markers):
    """Use only full objects containing exactly one distinct tracking marker."""
    segmentation, markers = np.asarray(segmentation), np.asarray(markers)
    if segmentation.ndim != 2 or segmentation.shape != markers.shape:
        raise ValueError('Full masks and tracking markers must share a 2-D shape')
    if not np.issubdtype(segmentation.dtype, np.integer) or not np.issubdtype(markers.dtype, np.integer):
        raise ValueError('Annotation masks must contain integer labels')
    if np.any(segmentation < 0) or np.any(markers < 0):
        raise ValueError('Annotation labels must be non-negative')
    mapping = {}
    unmarked = ambiguous = 0
    for label in np.unique(segmentation):
        if not label:
            continue
        ids = np.unique(markers[segmentation == label])
        ids = ids[ids != 0]
        if len(ids) == 1:
            mapping.setdefault(int(ids[0]), []).append(int(label))
        elif len(ids):
            ambiguous += 1
        else:
            unmarked += 1
    output = np.zeros(segmentation.shape, np.int64)
    duplicate = 0
    for track, labels in mapping.items():
        if len(labels) == 1:
            output[segmentation == labels[0]] = track
        else:
            duplicate += len(labels)
    kept = set(np.unique(output)) - {0}
    marker_ids = set(np.unique(markers)) - {0}
    return output, {'retained_tracks': len(kept), 'unmarked_objects': unmarked,
                    'multi_marker_objects': ambiguous, 'duplicate_track_objects': duplicate,
                    'markers_without_retained_full_mask': len(marker_ids - kept),
                    'excluded_track_ids': sorted(int(label) for label in marker_ids - kept)}


def scramble(labels, seed):
    """Relabel target objects without changing their shapes or positions."""
    ids = np.unique(labels)
    ids = ids[ids != 0]
    mapping = dict(zip(ids.tolist(), np.random.default_rng(seed).permutation(ids).tolist()))
    result = np.zeros_like(labels)
    for old, new in mapping.items():
        result[labels == old] = new
    return result, mapping


def score_pair(labels_t, labels_t1, predictions, seed=0, unknown_successors=()):
    """Score sources with known outcomes; incomplete next-frame masks are censored."""
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
    """Aggregate object-weighted results with explicit missing-data denominators."""
    arms = sorted({arm for row in rows for arm in row['correct']})

    def group(selected):
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


def checkpoint_predictors(checkpoint, device, seed):
    """Load a complete checkpoint locally and create trained/random-head controls."""
    import torch
    from cellpose.vit import CPSAM

    torch.set_num_threads(2)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    ps = int(state['up.weight'].shape[-1])
    dtype = state['encoder.encoder.patch_embed.proj.weight'].dtype
    encoder = CPSAM(ps=ps, dtype=dtype).to(dtype=dtype)
    net = tm.TimeflowsNet(tm.CellposeSamFeatures(encoder))
    net.load_state_dict(state, strict=True)
    del state
    trained = {name: value.detach().clone() for name, value in net.state_dict().items()
               if name.startswith(('head.', 'up.'))}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for module in (*net.head.modules(), net.up):
            reset = getattr(module, 'reset_parameters', None)
            if callable(reset):
                reset()
    random = {name: value.detach().clone() for name, value in net.state_dict().items()
              if name.startswith(('head.', 'up.'))}
    net.load_state_dict(trained, strict=False)

    def predict(a, b, random_head=False):
        try:
            if random_head:
                net.load_state_dict(random, strict=False)
            return tm.predict_pair(net, a, b, device=device)
        finally:
            if random_head:
                net.load_state_dict(trained, strict=False)

    return predict


def check_holdout(movie, provenance):
    """Reject missing training provenance and aliases of training movies."""
    movies = provenance.get('movies')
    if not isinstance(movies, list) or not movies or any(not isinstance(path, str) or not path for path in movies):
        raise ValueError('Checkpoint metadata must contain nonempty training movie paths')
    candidate = Path(movie).resolve()
    for path in movies:
        trained = Path(path).resolve()
        if candidate == trained or candidate.is_relative_to(trained) or trained.is_relative_to(candidate):
            raise ValueError(f'Evaluation movie overlaps a training movie: {trained}')


def main(argv=None):
    """Write a reproducible, streaming, stratified linking evaluation."""
    from importlib.metadata import version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--movie', type=Path, required=True)
    parser.add_argument('--sequences', nargs='+', default=['01', '02'])
    parser.add_argument('--segmentation', choices=['GT', 'ST'], required=True)
    parser.add_argument('--data-kind', choices=['synthetic', 'real'], required=True)
    parser.add_argument('--gaps', nargs='+', type=int, default=[1, 3, 6])
    parser.add_argument('--pairs-per-gap', type=int, default=3, help='Evenly spaced pairs per sequence and gap; 0 evaluates all')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.pairs_per_gap < 0 or any(gap < 1 for gap in args.gaps):
        parser.error('pairs-per-gap must be non-negative and gaps positive')
    metadata = args.checkpoint.with_suffix(args.checkpoint.suffix + '.json')
    provenance = json.loads(metadata.read_text())
    check_holdout(args.movie, provenance)
    selections = [pair for sequence in dict.fromkeys(args.sequences)
                  for pair in selected_pairs(args.movie, sequence, args.segmentation,
                                             list(dict.fromkeys(args.gaps)), args.pairs_per_gap)]
    args.out.mkdir(parents=True, exist_ok=False)
    run = {'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': digest(args.checkpoint),
           'checkpoint_metadata': provenance, 'metadata_sha256': digest(metadata),
           'movie': str(args.movie.resolve()), 'data_kind': args.data_kind,
           'segmentation_source': args.segmentation, 'device': args.device, 'seed': args.seed,
           'selected_pairs': len(selections), 'pairs_per_gap': args.pairs_per_gap,
           'gaps_frames': args.gaps,
           'software': {name: version(name) for name in ('numpy', 'scipy', 'torch', 'cellpose', 'tifffile')},
           'evaluator_sha256': digest(__file__), 'model_code_sha256': digest(tm.__file__),
           'scope': 'Linking given supplied full segmentation, not end-to-end segmentation/tracking accuracy.',
           'holdout_check': 'Resolved movie paths and aliases; not a content comparison against all training images.',
           'controls': 'IoU, zero motion, oracle, random time head on checkpoint encoder, copied frame with trained head; random-head chance is not assumed.'}
    (args.out / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    predict = checkpoint_predictors(args.checkpoint, args.device, args.seed)
    rows, stationary_rows = [], []
    started = time.perf_counter()
    with (args.out / 'pairs.jsonl').open('x') as handle:
        for index, selection in enumerate(selections):
            frames, masks, exclusions, inputs = [], [], [], []
            for image_path, mask_path, marker_path in selection['files']:
                image, segmentation, markers = [tifffile.imread(path) for path in (image_path, mask_path, marker_path)]
                if image.ndim != 2 or image.shape != segmentation.shape:
                    raise ValueError('Images and full masks must share a 2-D shape')
                mask, excluded = tracked_masks(segmentation, markers)
                frames.append(tm._normalise(image))
                masks.append(mask)
                exclusions.append(excluded)
                inputs.append([{'path': str(path.resolve()), 'sha256': digest(path)}
                               for path in (image_path, mask_path, marker_path)])
            if frames[0].shape != frames[1].shape:
                raise ValueError('The two frames must have the same shape')
            target = tm.time_targets(*masks)
            predictions = {
                'trained': predict(*frames),
                'random_head': predict(*frames, random_head=True),
                'zero_motion': {'vector': np.zeros_like(target['vector']), 'successor': np.ones_like(target['successor'])},
                'oracle': {'vector': target['vector'], 'successor': target['successor']},
            }
            pair_rows = score_pair(*masks, predictions, seed=args.seed + index,
                                   unknown_successors=exclusions[1]['excluded_track_ids'])
            copied_rows = score_pair(masks[0], masks[0], {'trained': predict(frames[0], frames[0])}, seed=args.seed + index)
            rows.extend(pair_rows)
            stationary_rows.extend(copied_rows)
            record = {key: value for key, value in selection.items() if key != 'files'}
            record.update(inputs=inputs, shape=list(frames[0].shape), exclusions=exclusions,
                          rows=pair_rows, copied_frame_rows=copied_rows)
            handle.write(json.dumps(record, allow_nan=False) + '\n')
            handle.flush()
            print(f'{index + 1}/{len(selections)} pairs complete', flush=True)
    summary = {**run, 'complete': True, 'seconds': time.perf_counter() - started,
               'results': summarise(rows), 'copied_frame_control': summarise(stationary_rows)}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
