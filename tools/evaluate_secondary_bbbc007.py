"""Measure fixed secondary-growth settings on BBBC007's manual paired outlines.

Download BBBC007_v1_images.zip and BBBC007_v1_outlines.zip from
https://bbbc.broadinstitute.org/BBBC007 before running this CPU-only evaluator.
No archive is extracted and no input is modified. JSON records archive hashes,
each field's exclusions, timings and relationship diagnostics.

White outline pixels are unknown, not cell interior or background. Interiors
are four-connected black regions enclosed by the outlines; border-connected
regions and regions smaller than 16 pixels are excluded. A nuclear interior
must place more than half its area in exactly one cell interior, and that cell
must have exactly one candidate nucleus. All eligible nuclear interiors seed
the detector, including those without an unambiguous evaluation partner.

Scores are object IoU/Dice on the paired interior regions, ignoring white cell
outline pixels. They are conditional on manual nuclear seeds and are NOT the
published BBBC007 adjacent-boundary metric, nor end-to-end detection accuracy.
Settings are fixed before evaluation; no train/test fitting or tuning occurs.
Some intensity TIFFs contain RGB channels. These use fixed BT.601 luminance
weights (0.299, 0.587, 0.114), followed by division by 255 for every image.
Distance growth was added after the first intensity comparison; its result is
exploratory on this same dataset, not an independent held-out validation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import zipfile
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
from scipy import ndimage
from skimage.segmentation import find_boundaries

from spacr.qt.mask_engine import primary_secondary_report, secondary_object_instances

SETTINGS = {
    'otsu_sigma2': {'sigma': 2, 'stop': 'threshold', 'stop_algorithm': 'otsu'},
    'fraction_0.4_sigma2': {'sigma': 2, 'stop': 'seed_fraction', 'stop_value': .4},
    'absolute_0.1_sigma2': {'sigma': 2, 'stop': 'absolute', 'stop_value': .1},
    'percentile_60_sigma2': {'sigma': 2, 'stop': 'percentile', 'stop_value': 60},
    'distance_growth_otsu_sigma2': {'sigma': 2, 'stop': 'threshold',
                                  'stop_algorithm': 'otsu', 'growth': 'distance'},
}


def interiors(outline, min_area=16):
    """Label enclosed four-connected interiors and report excluded fragments."""
    labels, _ = ndimage.label(~np.asarray(outline, dtype=bool))
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    sizes = np.bincount(labels.ravel())
    tiny = np.flatnonzero((sizes < min_area) & (sizes > 0))
    reject = np.union1d(border, tiny)
    labels[np.isin(labels, reject)] = 0
    return labels.astype(np.uint16), {'border_components': int(np.count_nonzero(border)),
                                     'tiny_components': int(np.count_nonzero(tiny))}


def paired_interiors(nuclear_outline, cell_outline):
    """Map unambiguous manual cell interiors onto their nuclear IDs."""
    nuclei, nuclear_excluded = interiors(nuclear_outline)
    cells, cell_excluded = interiors(cell_outline)
    candidates = {}
    for label in np.unique(nuclei):
        if not label:
            continue
        overlap = np.bincount(cells[nuclei == label].ravel())
        overlap[0] = 0
        best = int(overlap.argmax())
        if overlap[best] > np.count_nonzero(nuclei == label) / 2:
            candidates.setdefault(best, []).append(int(label))
    reference = np.zeros_like(nuclei)
    pairs = []
    for cell, labels in candidates.items():
        if len(labels) == 1:
            reference[cells == cell] = labels[0]
            pairs.append(labels[0])
    return nuclei, reference, sorted(pairs), {
        'nuclear_excluded': nuclear_excluded, 'cell_excluded': cell_excluded,
        'nuclei': int(np.count_nonzero(np.unique(nuclei))),
        'cell_interiors': int(np.count_nonzero(np.unique(cells))),
        'paired': len(pairs), 'ambiguous_cells': sum(len(ids) > 1 for ids in candidates.values()),
        'unpaired_nuclei': int(np.count_nonzero(np.unique(nuclei))) - len(pairs),
    }


def object_scores(prediction, reference, ids, ignored):
    """Score each matched ID, excluding hand-drawn cell-outline pixels."""
    values = []
    valid = ~np.asarray(ignored, dtype=bool)
    for label in ids:
        predicted = (prediction == label) & valid
        target = (reference == label) & valid
        intersection = int(np.count_nonzero(predicted & target))
        total = int(np.count_nonzero(predicted) + np.count_nonzero(target))
        union = total - intersection
        values.append({'id': int(label), 'iou': intersection / union if union else 0.,
                       'dice': 2 * intersection / total if total else 0.})
    return values


def image_pairs(archive):
    """Match the dataset's three explicit DNA/actin filename conventions."""
    names = {name.split('/', 1)[1]: name for name in archive.namelist() if name.endswith('.tif')}
    pairs = []
    for name in sorted(names):
        if name.endswith('d.tif'):
            actin = name[:-5] + 'f.tif'
        elif '_D_1UL.tif' in name:
            actin = name.replace('_D_1UL.tif', '_F_2UL.tif')
        elif name.endswith('d0.tif'):
            actin = re.sub('d0.tif$', 'd1.tif', name)
        else:
            continue
        if actin not in names:
            raise ValueError(f'Missing actin field for {name}')
        pairs.append((name, actin))
    if len(pairs) * 2 != len(names):
        raise ValueError('Some images did not enter a unique DNA/actin pair.')
    return names, pairs


def read_tiff(archive, name):
    """Decode one archive member without extracting paths to the filesystem."""
    return imageio.imread(io.BytesIO(archive.read(name)), extension='.tif')


def summarize(scores):
    """Pool equally weighted manual cell pairs across fields."""
    if not scores:
        raise ValueError('No unambiguous manual cell pairs to score.')
    iou = np.asarray([row['iou'] for row in scores])
    dice = np.asarray([row['dice'] for row in scores])
    return {'cells': len(scores), 'mean_iou': float(iou.mean()),
            'median_iou': float(np.median(iou)), 'mean_dice': float(dice.mean()),
            'fraction_iou_at_least_0.5': float((iou >= .5).mean())}


def evaluate(images_path, outlines_path):
    """Compare fixed stop rules and distance growth against two baselines.

    Distance growth was added after the initial intensity comparison exposed
    boundary leakage. Its result is exploratory, not held-out validation.
    """
    fields = []
    pooled = {key: [] for key in (*SETTINGS, 'primary_only', 'distance_otsu')}
    example = None
    with zipfile.ZipFile(images_path) as images, zipfile.ZipFile(outlines_path) as outlines:
        names, pairs = image_pairs(images)
        outline_names, outline_pairs = image_pairs(outlines)
        if pairs != outline_pairs:
            raise ValueError('Image and outline archives contain different fields.')
        for dna, actin in pairs:
            image = read_tiff(images, names[actin]).astype(np.float32)
            if image.ndim == 3 and image.shape[-1] == 3:
                image = image @ np.asarray([.299, .587, .114], dtype=np.float32)
            image /= 255.
            nuclear_outline = read_tiff(outlines, outline_names[dna])
            cell_outline = read_tiff(outlines, outline_names[actin])
            nuclei, reference, ids, exclusions = paired_interiors(nuclear_outline, cell_outline)
            record = {'field': actin, 'shape': list(image.shape), 'reference': exclusions, 'methods': {}}
            predictions = {'primary_only': nuclei.copy()}
            for method, settings in SETTINGS.items():
                started = time.perf_counter()
                found = secondary_object_instances(image, nuclei, **settings, min_area=0, fill_holes=True)
                predictions[method] = found.labels
                record['methods'][method] = {'seconds': time.perf_counter() - started,
                                             'threshold': found.level}
                if method == 'otsu_sigma2':
                    blurred = ndimage.gaussian_filter(image, 2)
                    nearest = ndimage.distance_transform_edt(nuclei == 0, return_distances=False,
                                                             return_indices=True)
                    distance = nuclei[tuple(nearest)]
                    distance[(blurred < found.level) & (nuclei == 0)] = 0
                    predictions['distance_otsu'] = distance
            for method, prediction in predictions.items():
                scores = object_scores(prediction, reference, ids, cell_outline)
                pooled[method].extend(scores)
                metrics = record['methods'].setdefault(method, {})
                report = primary_secondary_report(nuclei, prediction)
                metrics.update(summary=summarize(scores), objects=scores,
                               relationships={key: len(value) for key, value in report._asdict().items()})
            fields.append(record)
            if example is None:
                example = image, nuclear_outline, cell_outline, nuclei, reference, predictions
    return {'dataset': 'BBBC007 v1', 'source': 'https://bbbc.broadinstitute.org/BBBC007',
            'archive_sha256': {str(Path(path).name): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                               for path in (images_path, outlines_path)},
            'conversion': __doc__, 'settings': SETTINGS, 'fields': fields,
            'summary': {method: summarize(scores) for method, scores in pooled.items()}}, example


def render_example(example, path):
    """Export a static figure of the first alphabetic field, without cherry-picking."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    image, nuclear_outline, cell_outline, nuclei, reference, predictions = example
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    panels = [('Actin + manual cell outlines', cell_outline),
              ('Manual nuclear interiors', find_boundaries(nuclei)),
              ('Paired reference interiors', find_boundaries(reference)),
              ('Watershed: Otsu, sigma 2', find_boundaries(predictions['otsu_sigma2'])),
              ('Distance growth: Otsu, sigma 2', find_boundaries(predictions['distance_growth_otsu_sigma2'])),
              ('Distance baseline: Otsu', find_boundaries(predictions['distance_otsu']))]
    for axis, (title, boundary) in zip(axes.ravel(), panels):
        axis.imshow(image, cmap='gray', vmin=0, vmax=float(np.percentile(image, 99.5)))
        overlay = np.zeros((*image.shape, 4))
        overlay[boundary > 0] = (1, .4, .1, 1)
        axis.imshow(overlay)
        axis.set_title(title, fontsize=10)
        axis.axis('off')
    figure.suptitle('BBBC007 — A9 p10: manual seeds; fixed settings; no fitting')
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    """Write the full per-field report and an optional inspectable image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--outlines', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    report, example = evaluate(args.images, args.outlines)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if args.figure:
        render_example(example, args.figure)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
