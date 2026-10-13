"""Preserve actual hierarchy evidence and explicitly reject unsafe child crops."""
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import zipfile
from build_evaluation_example import sha
from compose_report_capture import _read, _frame
from lineage_evidence import read_expected, verify_forest
from stage_lesson import DEFAULT_STAGE, REPO, write


def prepare():
    source = DEFAULT_STAGE/'captures/lineage_current_review_v2'
    target = DEFAULT_STAGE/'captures/lineage_critical_review_v1'
    if target.exists():raise FileExistsError('Preserve earlier composition')
    hashes = {}
    provenance = _read(source/'provenance.json',hashes)
    proof = _read(source/'lineage_acceptance.json',hashes)
    hold_path = REPO/'tools/tutorials/evidence/2026-09-10_lineage_crop_identity_hold.json'
    hold = _read(hold_path,hashes)
    if not provenance['completed_capture'] or not proof['accepted'] or proof.get('error'):
        raise ValueError('Require completed current native recording')
    if hold['ready_for_narration'] is not False or proof['remaining_lineage_workers']:
        raise ValueError('Preserve the earlier hold and require finished workers')
    unsafe, parent = proof['unsafe_family_crop_handoff'], proof['crop_handoff']
    if unsafe['scientifically_valid'] or unsafe['recommended']:
        raise ValueError('Unsafe family crops must remain rejected')
    if unsafe['source_object_types'] != ['cell','cell','cell'] or unsafe['source_labels'] != [3,5,1]:
        raise ValueError('Re-evaluate the exact recorded child substitution')
    if parent['requested'] != ['plate1_r12_c1_f1_cell3'] or parent['shown'] != 1 or parent['labels_edited']:
        raise ValueError('Require the actual single-parent counterpart without label edits')
    if not proof['crop_zoom']['escape_restored_grid']:
        raise ValueError('Require native preview recovery')
    database = Path(proof['source']['database'])
    if sha(database) != proof['source']['database_sha256']:
        raise ValueError('Recorded private database changed')
    hashes[str(database)] = sha(database)
    expected = read_expected(database)
    export = Path(proof['api_export']['path'])
    if sha(export) != proof['api_export']['sha256']:
        raise ValueError('Recorded API export changed')
    hashes[str(export)] = sha(export)
    with export.open(newline='') as stream:
        rows = [{k:int(v) if k in ('label','depth','n_children') else v for k,v in r.items()}
                for r in csv.DictReader(stream)]
    verify_forest(rows,[],expected)
    if len(rows) != 7201:raise ValueError('Expected the full uncapped hierarchy')
    frames = {}
    for key, original in _read(source/'frames.json',hashes).items():
        frame = deepcopy(original)
        path = _frame(source/frame['image'],frame['sha256'],source,hashes)
        frame['image'] = os.path.relpath(path,target);frames[key] = frame
    if any(sha(path)!=value for path,value in hashes.items()):
        raise ValueError('Source evidence changed during review')
    target.mkdir()
    write(target/'frames.json',frames)
    write(target/'provenance.json',dict(completed_capture=True,module='lineage',
        composition_only=True,sources=[provenance],app_source_modified=False))
    receipt=dict(accepted=True,scope='Verified containment and parent-only crop route; unsafe family-to-child substitution explicitly rejected',
        native_verification=proof,source_hashes=hashes,
        original_hold=dict(path=str(hold_path),sha256=sha(hold_path),unchanged=True),
        child_crop_fallback_fixed=False,biological_validation=False,published=False)
    write(target/'scientific_acceptance.json',receipt)
    write(REPO/'tools/tutorials/evidence/2026-09-11_lineage_critical_review_verification.json',receipt)
    destination=REPO/'docs/source/_extra/tutorials/examples/Lineage_real_containment_review.zip'
    if destination.exists():raise FileExistsError('Preserve existing evidence archive')
    entries={'README.txt':Path(__file__).with_name('lineage_README.txt').read_bytes(),
             'lineage_api_export.csv':export.read_bytes(),
             'review.json':(json.dumps(receipt,indent=2)+'\n').encode()}
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in entries.items():
            info=zipfile.ZipInfo(name,date_time=(2026,9,11,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(info,data)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() or any(archive.read(name)!=data for name,data in entries.items()):
            raise ValueError('Packaged evidence differs')
    print(target);print(destination,sha(destination))


if __name__ == '__main__':prepare()
