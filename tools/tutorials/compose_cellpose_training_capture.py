"""Join the real source-import refusal, real API command and real plot viewer."""
from copy import deepcopy
import os
from pathlib import Path

from compose_report_capture import _frame, _read
from stage_lesson import DEFAULT_STAGE, read, write
from train_cellpose_example import digest


def compose(stage=DEFAULT_STAGE):
    stage=Path(stage)
    destination=stage/'captures/train_cellpose_explicit_verified'
    if destination.exists():raise FileExistsError('Preserve the previous composition')
    native=stage/'captures/train_cellpose_review_and_pair_viewer_v2'
    command=stage/'captures/train_cellpose_explicit_api_v1'
    review=read(native/'scientific_acceptance.json');api=read(command/'scientific_acceptance.json')
    if review.get('accepted') is not True or api.get('accepted') is not True:
        raise ValueError('Both actual recordings must pass their explicit scoped checks')
    if review['native_source_import']['accepted'] is not False or not review['native_source_import'].get('hold'):
        raise ValueError('The tutorial must retain the native source-import refusal')
    if review['gui_source_control_fixed'] is not False or api['held_out_accuracy_validated'] is not False:
        raise ValueError('The API demonstration is not a GUI repair or accuracy evaluation')
    training=api['training']
    for path,value in training['source_hashes'].items():
        if digest(path)!=value:raise ValueError('An original training input changed')
    if digest(training['result']['checkpoint'])!=training['checkpoint_sha256']:
        raise ValueError('The verified checkpoint changed')
    if digest(training['actual_preview']['path'])!=training['actual_preview']['sha256']:
        raise ValueError('The actual pair figure changed')
    if review['external_viewer']['saved_plots'][0]['sha256']!=training['actual_preview']['sha256']:
        raise ValueError('The external viewer did not show the actual training figure')
    hashes={};frames={};sources=[]
    for prefix,source in [('gui',native),('command',command)]:
        provenance=_read(source/'provenance.json',hashes)
        if not provenance['completed_capture'] or provenance['module']!='train_cellpose':
            raise ValueError('Expected the completed actual Train Cellpose recording')
        sources.append(provenance)
        for key,original in _read(source/'frames.json',hashes).items():
            frame=deepcopy(original)
            path=_frame(source/frame['image'],frame['sha256'],source,hashes)
            frame['image']=os.path.relpath(path,destination);frame['source_capture']=str(source)
            frames[prefix+'_'+key]=frame
    if any(digest(path)!=value for path,value in hashes.items()):
        raise ValueError('A recording changed during composition')
    proof=dict(accepted=True,scope='Disclosed native source-import defect and separate checked training API demonstration',
               native_review=review,api=api,source_hashes=hashes,
               gui_source_control_fixed=False,held_out_accuracy_validated=False,
               application_source_modified=False,published=False)
    destination.mkdir()
    write(destination/'frames.json',frames)
    write(destination/'provenance.json',dict(completed_capture=True,module='train_cellpose',
          sources=sources,composition_only=True,app_source_modified=False))
    write(destination/'scientific_acceptance.json',proof)
    print(destination)
    return proof


if __name__=='__main__':compose()
