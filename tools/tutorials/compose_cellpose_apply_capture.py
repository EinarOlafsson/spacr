"""Stage the verified Apply batch and an explicitly non-equivalent preview.

This preserves the old equivalence hold. It does not fix preprocessing, card
height or segmentation accuracy; a lesson must disclose all three limits.
"""
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import tifffile

from build_evaluation_example import sha
from cellpose_apply_evidence import minimum_area_labels, require_pixels
from compose_report_capture import _frame, _read
from stage_lesson import DEFAULT_STAGE, read, write


def require_scope(proof, reference):
    if not proof['pipeline']['accepted'] or not reference['accepted']:
        raise ValueError('The batch and independent reference must pass')
    if (proof['accuracy_validated'] is not False or proof['ground_truth_used'] is not False
            or proof['preview']['batch_mask_equal'] is not False
            or reference['preview_default_reference']['differs_from_batch'] is not True):
        raise ValueError('Retain the observed preview mismatch and lack of accuracy validation')
    if not proof['filter']['raw_unchanged'] or not proof['filter']['restored']:
        raise ValueError('Filtering must preserve raw labels and restore the result')


def require_closed_dialogs(frames):
    for name in ('22b_actual_zoomed_filtered_preview','23b_actual_zoomed_restored_preview'):
        if frames[name].get('dialogs'):
            raise ValueError('A dialog covers the narrated unobstructed filter result')


def compose(stage=DEFAULT_STAGE):
    stage=Path(stage); source=stage/'captures/cellpose_apply_native_zoom_v2'
    destination=stage/'captures/cellpose_apply_disclosed_verified_v2'
    if destination.exists():raise FileExistsError('Preserve the previous composition')
    hashes={}
    proof=_read(source/'scientific_acceptance.json',hashes)
    reference=_read(source/'independent_reference.json',hashes)
    require_scope(proof,reference)
    work=Path(proof['private_folder'])
    if sha(reference['checkpoint'])!=reference['checkpoint_sha256']:
        raise ValueError('The reference checkpoint changed')
    for original,value in proof['original_inputs'].items():
        if sha(original)!=value or sha(work/Path(original).name)!=value:
            raise ValueError('An actual source image changed')
        hashes[original]=value;hashes[str(work/Path(original).name)]=value
    for row in reference['files']:
        mask=work/'masks'/row['name']
        if sha(mask)!=row['saved_mask_sha256']:
            raise ValueError('A verified batch mask changed')
        hashes[str(mask)]=sha(mask)
    raw=np.load(source/'preview_cell.npy',allow_pickle=False)
    filtered=np.load(source/'preview_cell_filtered.npy',allow_pickle=False)
    restored=np.load(source/'preview_cell_restored.npy',allow_pickle=False)
    require_pixels(filtered,minimum_area_labels(raw,proof['filter']['cutoff']))
    require_pixels(restored,minimum_area_labels(raw,0))
    if np.array_equal(raw,tifffile.imread(work/'masks/cell_pair_02.tif')):
        raise ValueError('The narrated preview-versus-batch mismatch is absent')
    for name in ('preview_cell.npy','preview_cell_filtered.npy','preview_cell_restored.npy'):
        hashes[str(source/name)]=sha(source/name)
    for path in source.glob('batch_figure_*_arrays.npz'):hashes[str(path)]=sha(path)
    provenance=_read(source/'provenance.json',hashes)
    if not provenance['completed_capture'] or provenance['module']!='cellpose_masks':
        raise ValueError('Expected a completed native Apply recording')
    frames={}
    original_frames=_read(source/'frames.json',hashes)
    require_closed_dialogs(original_frames)
    for key,original in original_frames.items():
        frame=deepcopy(original);path=_frame(source/frame['image'],frame['sha256'],source,hashes)
        frame['image']=os.path.relpath(path,destination);frames[key]=frame
    if any(sha(path)!=value for path,value in hashes.items()):
        raise ValueError('Evidence changed during composition')
    accepted=dict(accepted=True,scope='Verified batch demonstration and explicitly mismatching native preview; not accuracy or preview equivalence',
        native_observation=proof,independent_reference=reference,source_hashes=hashes,
        preview_equivalence_fixed=False,preview_card_height_fixed=False,
        accuracy_validated=False,app_source_modified=False,published=False)
    destination.mkdir()
    write(destination/'frames.json',frames)
    write(destination/'provenance.json',dict(completed_capture=True,module='cellpose_masks',
        sources=[provenance],composition_only=True,app_source_modified=False))
    write(destination/'scientific_acceptance.json',accepted)
    print(destination)
    return accepted


if __name__=='__main__':compose()
