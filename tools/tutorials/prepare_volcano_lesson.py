"""Reuse the verified native Volcano recording only while its sources match."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import zipfile
from build_evaluation_example import sha
from compose_report_capture import _frame,_read
from stage_lesson import DEFAULT_STAGE,REPO,read,write
from volcano_evidence import read_family

APPLICATION_FILES=('spacr/qt/screens/volcano.py','spacr/qt/widgets/volcano_explorer.py','spacr/volcano_style.py')


def prepare(stage=DEFAULT_STAGE):
    stage=Path(stage);source=stage/'captures/volcano_native_checkbox_and_exports'
    destination=stage/'captures/volcano_rechecked_verified_v1'
    if destination.exists():raise FileExistsError('Preserve previous composition')
    hashes={};proof=_read(source/'volcano_acceptance.json',hashes)
    provenance=_read(source/'provenance.json',hashes)
    if not (proof['accepted'] and proof['all_source_files_unchanged'] and proof['all_private_inputs_unchanged'] and provenance['completed_capture']):
        raise ValueError('Require completed native workflow and unchanged original data')
    for name in APPLICATION_FILES:
        old=subprocess.check_output(['git','show',provenance['commit']+':'+name],cwd=REPO)
        if old!=(REPO/name).read_bytes():raise ValueError('Relevant application source changed; re-record '+name)
        hashes[str(REPO/name)]=sha(REPO/name)
    for path,value in proof['source_files'].items():
        if sha(path)!=value:raise ValueError('An original Regression output changed')
        hashes[path]=value
    work=Path(proof['private_folder']);data=work/'results'
    for name,value in proof['private_input_hashes'].items():
        if sha(data/name)!=value:raise ValueError('A recorded private input changed')
        hashes[str(data/name)]=value
    rows,family=read_family(data/'guide_permutation_results_long.csv')
    if family!=proof['independent_family']:raise ValueError('The independently checked statistical family differs')
    for path,record in proof['exports'].items():
        if sha(path)!=record['sha256']:raise ValueError('A recorded export changed')
        hashes[path]=record['sha256']
    baseline=read(work/'baseline_style.json');final=read(work/'volcano_style.json')
    if baseline['alpha']!=.05 or baseline['threshold_multiplier']!=3 or baseline['y_neg_log10'] is not True:
        raise ValueError('Baseline style no longer describes the actual initial renderer')
    if final['alpha']!=.05 or final['y_column']!='adjusted_p_value' or final['y_neg_log10'] is not True:
        raise ValueError('The final export did not restore adjusted-P and original alpha')
    if proof['alpha_semantics']!={'below_new_cut':19,'still_source_flagged':0,'new_statistical_test':False}:
        raise ValueError('Do not claim the reference-line change reruns significance testing')
    hashes[str(work/'baseline_style.json')]=sha(work/'baseline_style.json')
    frames={}
    for key,original in _read(source/'frames.json',hashes).items():
        frame=deepcopy(original);path=_frame(source/frame['image'],frame['sha256'],source,hashes)
        frame['image']=os.path.relpath(path,destination);frames[key]=frame
    if any(sha(path)!=value for path,value in hashes.items()):raise ValueError('Evidence changed while preparing lesson')
    destination.mkdir()
    write(destination/'frames.json',frames)
    write(destination/'provenance.json',dict(completed_capture=True,module='volcano_explorer',sources=[provenance],composition_only=True,app_source_modified=False))
    accepted=dict(accepted=True,scope='Existing real guide-family visualization and native Save/Load-style workaround, not new testing',
        workflow=proof,source_hashes=hashes,application_sources_unchanged_since_capture=list(APPLICATION_FILES),
        initial_control_sync_fixed=False,new_analysis=False,synthetic_results=False,published=False)
    write(destination/'scientific_acceptance.json',accepted)
    contents={'README.txt':(Path(__file__).parent/'volcano_README.txt').read_bytes()}
    for name in proof['private_input_hashes']:contents['results/'+name]=(data/name).read_bytes()
    for name in ('baseline_style.json','volcano_style.json'):contents[name]=(work/name).read_bytes()
    contents['source_manifest.json']=(json.dumps(dict(source_sha256=proof['private_input_hashes'],family=family),indent=2)+'\n').encode()
    target=REPO/'docs/source/_extra/tutorials/examples/Volcano_real_guide_family.zip'
    if target.exists():raise FileExistsError('Preserve previous download bundle')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,value in contents.items():
            info=zipfile.ZipInfo(name,date_time=(2026,9,11,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,value)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() or any(archive.read(name)!=value for name,value in contents.items()):raise ValueError('Packaged bytes differ')
    write(destination/'bundle_verification.json',dict(accepted=True,sha256=sha(target),source_bytes_exact=True))
    print(destination);print(target,sha(target))
    return accepted


if __name__=='__main__':prepare()
