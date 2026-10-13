"""Compare two previously recorded segmentations of the same real image.

These are the Apply tutorial's batch and live-preview outputs, both made with
cpsam but different preprocessing. Neither is ground truth; this is agreement,
not accuracy, and does not run or repair the Model Compare GUI backend.
"""
import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path

import numpy as np
from ops_geometry_example import digest

EXPECTED = {
    'image.tif': 'cc9df7fe55a085f100d633f9cc83af6aca4b654de2c9208bfc80b3044e88c138',
    'batch.tif': '5523bb5bf3a22179f4fb648a20ece1b25036443b0ce91f730a811b0020130dc8',
    'preview.npy': '7250985b9275b73950031d986ae8b389ded26920eac0af929466c38b3f1253da',
}


def independently_check(a, b, comparison):
    """Check reported objects and each matched IoU using the actual pixels."""
    count_a, count_b = (len(np.unique(m[m > 0])) for m in (a, b))
    if (comparison.n_objects_a, comparison.n_objects_b) != (count_a, count_b):
        raise ValueError('Comparison counts disagree with the real nonzero labels')
    if comparison.n_matched != len(comparison.matches):
        raise ValueError('Matched-pair count differs from the actual assignments')
    used_a, used_b = set(), set()
    for label_a, label_b, reported in comparison.matches:
        if label_a in used_a or label_b in used_b:
            raise ValueError('A label was assigned to more than one object')
        used_a.add(label_a); used_b.add(label_b)
        first, second = a == label_a, b == label_b
        expected = np.count_nonzero(first & second) / np.count_nonzero(first | second)
        if not np.isclose(reported, expected, rtol=0, atol=1e-12):
            raise ValueError('Matched IoU differs from independently counted pixels')
    expected_fraction = 2 * len(comparison.matches) / (count_a + count_b)
    if not np.isclose(comparison.iou_matched_fraction, expected_fraction):
        raise ValueError('Matched-object fraction has the wrong denominator')
    return {'objects_a': count_a, 'objects_b': count_b,
            'matched_pairs_pixel_checked': len(comparison.matches),
            'foreground_disagreement_pixels': int(np.count_nonzero((a > 0) != (b > 0)))}


def run(source, destination):
    import tifffile
    from spacr.model_compare import compare_masks
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError('Choose a NEW output folder outside the source images')
    paths = {name: source / name for name in EXPECTED}
    if any(digest(paths[name]) != value for name, value in EXPECTED.items()):
        raise ValueError('Inputs differ from the exact recorded field and its two saved masks')
    image = tifffile.imread(paths['image.tif'])
    a = tifffile.imread(paths['batch.tif'])
    b = np.load(paths['preview.npy'], allow_pickle=False)
    if image.shape != a.shape or a.shape != b.shape or a.shape != (512, 512):
        raise ValueError('All three inputs must describe the same original 512-square field')
    same = compare_masks(a, a, field='self-check')
    if same.ari != 1 or same.iou_matched_fraction != 1 or same.mean_matched_iou != 1:
        raise ValueError('The positive identical-mask control failed')
    comparison = compare_masks(a, b, field='cell_pair_02: batch vs live preview')
    checks = independently_check(a, b, comparison)
    if (checks['objects_a'], checks['objects_b']) != (94, 95) or checks['foreground_disagreement_pixels'] == 0:
        raise ValueError('The recorded difference no longer matches the original Apply example')
    print('One actual 512-square field; same cpsam weights, different preprocessing.', flush=True)
    print(comparison, flush=True)
    print(f'Matched {comparison.n_matched} pairs; mean matched IoU {comparison.mean_matched_iou:.6f}.', flush=True)
    print('Neither mask is ground truth. No model is rerun; no accuracy claim.', flush=True)
    destination.mkdir(parents=True, exist_ok=False)
    values = {key: value for key, value in asdict(comparison).items()
              if key not in {'matches', 'qc_a', 'qc_b'}}
    with (destination / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values))
        writer.writeheader(); writer.writerow(values)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from skimage.segmentation import find_boundaries
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    lo, hi = np.percentile(image, (2, 99))
    for ax, masks, title in [(axes[0], a, 'A: saved batch mask (94 objects)'),
                             (axes[1], b, 'B: saved live-preview mask (95 objects)')]:
        ax.imshow(image, cmap='gray', vmin=lo, vmax=hi)
        edges = np.ma.masked_where(~find_boundaries(masks), np.ones(masks.shape))
        ax.imshow(edges, cmap='autumn', vmin=0, vmax=1)
        ax.set_title(title); ax.set_axis_off()
    different = (a > 0) != (b > 0)
    axes[2].imshow(different, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Foreground disagreement: {checks["foreground_disagreement_pixels"]} pixels')
    axes[2].set_axis_off()
    fig.suptitle('Same real image, different preprocessing — agreement is not accuracy')
    fig.savefig(destination / 'comparison.png', dpi=160); plt.close(fig)
    if any(digest(paths[name]) != value for name, value in EXPECTED.items()):
        raise ValueError('A source changed during comparison')
    report = {'accepted': True, 'scope': 'Pure compare_masks API on two genuine saved segmentations',
              'sources': EXPECTED, 'source_unchanged': True, 'helper_sha256': digest(__file__),
              'comparison': asdict(comparison), 'independent_checks': checks,
              'same_mask_control_passed': True, 'ground_truth_used': False,
              'accuracy_validated': False, 'different_model_weights_compared': False,
              'gui_workflow_completed': False, 'inference_performed': False, 'published': False,
              'artifacts': {p.name: digest(p) for p in sorted(destination.iterdir())}}
    (destination / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
