"""Stage verified Curate actions with an explicit external-checkpoint caveat.

The native history-loss gate remains false and preserved. This composition
certifies only pixel/table operations and separate saved-file checkpoints.
"""
from copy import deepcopy
import csv
import os
from pathlib import Path
import json
import numpy as np
import tifffile
from build_evaluation_example import sha
from compose_report_capture import _frame,_read
from curate_evidence import check_mask,check_tracks,painted_disk
from stage_lesson import DEFAULT_STAGE,write


def require_scope(proof):
    if (proof['accepted'] is not False or proof['mask_prior_history_preserved_after_second_save'] is not False
            or proof['mask_history_on_reopen'] ['previous_entries']!=3
            or proof['mask_history_on_reopen']['current_entries']!=0):
        raise ValueError('Preserve the actually observed mask-history loss; do not claim a repair')
    if (proof['synthetic'] is not True or not proof['original_inputs_preserved']
            or not proof['tracks_prior_history_preserved']):
        raise ValueError('Require disclosed synthetic input and preserved originals and track history')
    for key,count in [('external_first_mask_checkpoint',3),('external_second_mask_checkpoint',1)]:
        record=proof[key]
        if not record['source_unchanged'] or record['edit_count']!=count or len(record['copied_sha256'])!=2:
            raise ValueError('Both exact external mask-and-ledger checkpoints are required')
    if not proof['external_first_history_still_preserved']:
        raise ValueError('The first-session checkpoint must survive the second session')


def compose(stage=DEFAULT_STAGE):
    stage=Path(stage);source=stage/'captures/curate_explicit_checkpoints_v1'
    destination=stage/'captures/curate_disclosed_verified_v1'
    if destination.exists():raise FileExistsError('Preserve the previous composition')
    hashes={};proof=_read(source/'scientific_acceptance.json',hashes);require_scope(proof)
    for path,value in proof['original_inputs'].items():
        if sha(path)!=value:raise ValueError('An original synthetic input changed')
        hashes[path]=value
    work=Path(proof['private_folder'])
    for key in ('external_first_mask_checkpoint','external_second_mask_checkpoint'):
        for path,value in proof[key]['copied_sha256'].items():
            if sha(path)!=value:raise ValueError('A protected checkpoint changed')
            hashes[path]=value
    raw=np.load(next(p for p in proof['original_inputs'] if p.endswith('.npy')),allow_pickle=False)
    baseline=raw[...,2].astype(np.int64)
    paint=proof['checks']['05_practice_disk_not_a_real_object']
    first=painted_disk(baseline,paint['actual_centre'],paint['radius'],paint['label']).astype(np.uint16)
    check_mask(tifffile.imread(work/'checkpoint-before-reopen/mask.tif'),first)
    first_log=json.loads((work/'checkpoint-before-reopen/mask.tif.curation.json').read_text())
    second_log=json.loads((work/'checkpoint-after-second-save/mask.tif.curation.json').read_text())
    if first_log!=proof['checks']['07_saved_mask_and_ledger']['ledger'] or second_log!=proof['checks']['10_second_session_saved_history']['ledger']:
        raise ValueError('Protected ledgers differ from the actual recorded sessions')
    if sha(work/'mask.tif')!=sha(work/'checkpoint-after-second-save/mask.tif'):
        raise ValueError('The final saved mask differs from the second checkpoint')
    with Path(next(p for p in proof['original_inputs'] if p.endswith('.csv'))).open() as handle:
        rows=list(csv.DictReader(handle))
    with (work/'tracks.csv').open() as handle:check_tracks(list(csv.DictReader(handle)),rows)
    for path in (work/'mask.tif',work/'tracks.csv',work/'tracks.csv.curation.json'):
        hashes[str(path)]=sha(path)
    provenance=_read(source/'provenance.json',hashes)
    if not provenance['completed_capture']:raise ValueError('Native capture is incomplete')
    original_frames=_read(source/'frames.json',hashes);frames={}
    for key,original in original_frames.items():
        frame=deepcopy(original);path=_frame(source/frame['image'],frame['sha256'],source,hashes)
        frame['image']=os.path.relpath(path,destination);frames[key]=frame
    if any(sha(path)!=value for path,value in hashes.items()):raise ValueError('Evidence changed during composition')
    accepted=dict(accepted=True,scope='Verified practice edits and EXTERNAL exact-byte checkpoints; NOT automatic mask-history preservation',
        native_observation=proof,source_hashes=hashes,mask_history_loss_fixed=False,
        synthetic=True,biological_correction_or_accuracy_validated=False,app_source_modified=False,published=False)
    destination.mkdir();write(destination/'frames.json',frames)
    write(destination/'provenance.json',dict(completed_capture=True,module='curate',sources=[provenance],composition_only=True,app_source_modified=False))
    write(destination/'scientific_acceptance.json',accepted)
    print(destination)
    return accepted


if __name__=='__main__':compose()
