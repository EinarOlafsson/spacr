"""Recheck preserved live-preview measurements without rerunning inference."""
import hashlib
from pathlib import Path

import numpy as np

from stage_lesson import read


def check_arrays(baseline, variant, outputs, changes):
    """Compare the recorded counts with actual masks and an independent area cut."""
    for array in (baseline, variant):
        if array.ndim != 2 or array.dtype.kind not in 'ui' or np.any(array < 0):
            raise ValueError('Expected nonnegative two-dimensional label masks')
    if baseline.shape != variant.shape or list(baseline.shape) != outputs['cell']['shape']:
        raise ValueError('Preview dimensions changed')
    labels, areas = np.unique(baseline[baseline > 0], return_counts=True)
    count = len(labels)
    filtered = int(np.count_nonzero(areas >= changes['filter']['minimum_area']))
    alternate = int(np.count_nonzero(np.unique(variant)))
    trace = changes['filter']; model = changes['model_option']
    if (count != outputs['cell']['objects'] or count != trace['before']
            or count != trace['restored'] or filtered != trace['after']
            or not 0 < filtered < count
            or trace['model_rerun'] is not False or trace['raw_mask_unchanged'] is not True):
        raise ValueError('The recorded reversible area filter disagrees with the saved mask')
    if (model['rerun_completed'] is not True or alternate != model['objects_after']
            or hashlib.sha256(variant.tobytes()).hexdigest() != model['array_sha256']
            or np.array_equal(baseline, variant)):
        raise ValueError('The recorded parameter rerun is not the saved distinct mask')
    return dict(before=count, minimum_area=trace['minimum_area'], after=filtered,
                restored=count, parameter_variant_objects=alternate,
                preserved_native_observation_not_new_inference=True,
                segmentation_quality_ranking_claimed=False)


def inspect(capture):
    capture = Path(capture)
    outputs = read(capture / 'preview_outputs.json')
    baseline = capture / 'preview_cell.npy'
    if hashlib.sha256(baseline.read_bytes()).hexdigest() != outputs['cell']['sha256']:
        raise ValueError('The original preview file changed')
    changes = read(capture / 'live_variants.json')
    result = check_arrays(np.load(baseline, allow_pickle=False),
                          np.load(capture / 'preview_cell_diameter_variant.npy', allow_pickle=False),
                          outputs, changes)
    result['source_hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [baseline, capture / 'preview_cell_diameter_variant.npy',
         capture / 'preview_outputs.json', capture / 'live_variants.json']}
    return result
