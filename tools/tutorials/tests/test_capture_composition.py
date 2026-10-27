"""Partial native recordings compose only with equal sources and exact bytes."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compose_capture_set import compose


def capture(tmp_path,name,visual,source_hash='same-source'):
    root=tmp_path/name;folder=root/'captures/home';folder.mkdir(parents=True)
    (root/'exact-source.json').write_text(json.dumps({'sha256':{'spacr/qt/app.py':source_hash}}), encoding="utf-8")
    (folder/'provenance.json').write_text(json.dumps({'completed_capture':True}), encoding="utf-8")
    image=folder/(visual+'.png');image.write_bytes((visual+'-actual-frame').encode())
    frames={visual:{'image':image.name,'sha256':hashlib.sha256(image.read_bytes()).hexdigest()}}
    (folder/'frames.json').write_text(json.dumps(frames), encoding="utf-8");return folder


def activation_science(folder, project):
    """Synthetic metadata only: no model, CUDA session, media decode or pixels."""
    crops=[dict(name=f'crop{i}.png',source=f'/original/crop{i}.png',sha256=str(i+1)*64,
                identity=dict(prcfo=f'cell{i}')) for i in range(4)]
    model=project+'/model.pth';archive=project+'/real_cell_crops.tar';card=project+'/model.card.json'
    proof=dict(accepted=True,lesson='16_activation',app_source_modified=False,active_jobs_after_close=0,
        provider_contacted=False,reference_device_resolution=dict(device='cuda'),
        preservation={key:True for key in ('original_database_and_sidecars_unchanged',
            'original_crops_model_card_and_marker_unchanged','private_archive_model_and_card_unchanged')},
        inputs=dict(model=model,archive=archive,crops=crops,
            original_hashes={**{row['source']:row['sha256'] for row in crops},
                '/original/model.pth':'a'*64,'/original/model.card.json':'b'*64},
            private_input_hashes={model:'a'*64,card:'b'*64,archive:'c'*64},
            database='/original/measurements.db',database_bundle={
                '/original/measurements.db':dict(sha256='d'*64,signature=[1,2,3,4,5])},crop_format={'channels':'rgb'},
            model_classes=['class0','class1'],model_epochs=1,model_validation_claim=False,biological_claim=False),
        independent_reference=dict(reference_device='cuda:0',source_names=[row['name'] for row in crops],
            original_channel_order_preserved=True,training_input_pixels=196608,training_input_max_error=0),runs=[])
    for method in ('saliency_channel','saliency_image'):
        per_map=49152 if method=='saliency_channel' else 16384
        proof['runs'].append(dict(method=method,
            settings=dict(dataset=archive,model_path=model,cam_type=method,batch_size=4),
            cuda_execution=dict(method=method,model_path=model,observer='loaded_classifier_forward_pre_hook',errors=[],
                loads=[dict(model_path=model,forwards=[dict(parameter_devices=['cuda:0'],buffer_devices=['cuda:0'],
                    input_devices=['cuda:0'],input_shapes=[[4,3,128,128]])])]),
            outcome=dict(finished=True,ok=True,errors=[]),settings_errors=[],
            maps=[dict(path=project+'/'+method+'/'+row['name'],sha256='e'*64,pixels=per_map,
                max_absolute_error=int(i==1 and method=='saliency_channel')) for i,row in enumerate(crops)],
            independent_saved_pixels_checked=4*per_map,
            external_viewer=dict(accepted=True,saved_files_unchanged=True,application_figures_fixed=False,
                saved_plots=[dict(actual_desktop_capture=True)])))
    (folder/'scientific_acceptance.json').write_text(json.dumps(proof))
    return proof


def test_activation_composition_retains_each_verbatim_cuda_proof_across_random_projects(tmp_path):
    first=capture(tmp_path,'first','dataset');second=capture(tmp_path,'second','grid')
    left=activation_science(first,'/private/random-first');right=activation_science(second,'/private/random-second')
    out=tmp_path/'combined/captures/activation'
    compose([first,second],out,scientific_lesson='16_activation')
    result=json.loads((out/'scientific_acceptance.json').read_text())
    assert result['accepted'] is True and result['lesson']=='16_activation'
    assert [row['original_proof'] for row in result['sources']]==[left,right]
    assert [row['selected_scenes'] for row in result['sources']]==[['dataset'],['grid']]
    assert 'cuda_execution' not in result and 'runs' not in result
    for row,source in zip(result['sources'],(first,second)):
        assert row['sha256']==hashlib.sha256((source/'scientific_acceptance.json').read_bytes()).hexdigest()


@pytest.mark.parametrize('fault',[None,'hash','device_index'])
def test_activation_database_copy_metadata_is_not_a_scientific_cohort_difference(tmp_path,fault):
    first=capture(tmp_path,'first','dataset');second=capture(tmp_path,'second','grid')
    activation_science(first,'/private/first');proof=activation_science(second,'/private/second')
    proof['inputs']['database_bundle']['/original/measurements.db']['signature']=[1,99,3,4,100]
    proof['reference_device_resolution']['device']='cuda:0'
    if fault=='hash':proof['inputs']['database_bundle']['/original/measurements.db']['sha256']='f'*64
    elif fault=='device_index':proof['reference_device_resolution']['device']='cuda:1'
    (second/'scientific_acceptance.json').write_text(json.dumps(proof))
    out=tmp_path/'combined/captures/activation'
    if fault:
        with pytest.raises(ValueError):compose([first,second],out,scientific_lesson='16_activation')
        assert not out.exists()
    else:
        compose([first,second],out,scientific_lesson='16_activation')
        result=json.loads((out/'scientific_acceptance.json').read_text())
        assert result['shared_cohort']['database_bundle']=={'measurements.db':'d'*64}
        assert result['sources'][1]['original_proof']==proof


@pytest.mark.parametrize('fault',['missing','failed','crop','archive','duplicate_basename','cuda',
                                 'reference','pixels','model','settings','preservation'])
def test_activation_scientific_mismatch_fails_before_composition(tmp_path,fault):
    first=capture(tmp_path,'first','dataset');second=capture(tmp_path,'second','grid')
    activation_science(first,'/private/first');proof=activation_science(second,'/private/second')
    path=second/'scientific_acceptance.json'
    if fault=='missing':path.rename(path.with_suffix('.held'))
    elif fault=='failed':proof['accepted']=False
    elif fault=='crop':proof['inputs']['crops'][0]['identity']['prcfo']='different-cell'
    elif fault=='archive':proof['inputs']['private_input_hashes'][proof['inputs']['archive']]='f'*64
    elif fault=='duplicate_basename':proof['inputs']['private_input_hashes']['/other/model.pth']='a'*64
    elif fault=='cuda':proof['runs'][0]['cuda_execution']['loads'][0]['forwards'][0]['input_devices']=['cpu']
    elif fault=='reference':proof['independent_reference']['reference_device']='cpu'
    elif fault=='pixels':proof['runs'][0]['maps'][0]['max_absolute_error']=2
    elif fault=='model':proof['runs'][0]['settings']['model_path']='/foreign/model.pth'
    elif fault=='settings':proof['runs'][0]['settings']['batch_size']=3
    else:proof['preservation']['private_archive_model_and_card_unchanged']=False
    if fault!='missing':path.write_text(json.dumps(proof))
    out=tmp_path/'combined/captures/activation'
    with pytest.raises(ValueError):compose([first,second],out,scientific_lesson='16_activation')
    assert not out.exists()


def test_composed_alias_retains_original_frame_and_records_source(tmp_path):
    first=capture(tmp_path,'first','home');second=capture(tmp_path,'second','spaceout')
    original=(second/'spaceout.png').read_bytes();out=tmp_path/'combined/captures/home'
    frames=compose([first,second],out,aliases={'spaceout':'last_scene'})
    assert set(frames)=={'home','last_scene'}
    assert (out/'spaceout.png').read_bytes()==original
    assert (second/'spaceout.png').read_bytes()==original
    assert frames['last_scene']['composition_source']['visual']=='spaceout'


@pytest.mark.parametrize('failure',['source','bytes','duplicate'])
def test_unrelated_or_changed_capture_fails_before_composition(tmp_path,failure):
    first=capture(tmp_path,'first','home');second=capture(tmp_path,'second','home' if failure=='duplicate' else 'other',source_hash='different' if failure=='source' else 'same-source')
    if failure=='bytes':(second/'other.png').write_bytes(b'changed')
    out=tmp_path/'combined/captures/home'
    with pytest.raises(ValueError):compose([first,second],out)
    assert not out.exists()


def test_explicit_recorder_fix_retains_app_identity_and_both_exact_tool_sources(tmp_path):
    first=capture(tmp_path,'first','home');second=capture(tmp_path,'second','other')
    tool='tools/tutorials/capture_workflow_overview.py'
    for folder,version in ((first,'old'),(second,'new')):
        path=folder.parent.parent/'exact-source.json';value=json.loads(path.read_text(encoding="utf-8"))
        value['sha256'][tool]=version;path.write_text(json.dumps(value), encoding="utf-8")
    out=tmp_path/'combined/captures/home'
    with pytest.raises(ValueError):compose([first,second],out)
    compose([first,second],out,recorder_delta_paths=[tool],
            recorder_delta_reason='Select unchanged bundled templates through actual editors.')
    identity=json.loads((out/'exact-source.json').read_text(encoding="utf-8"))
    assert identity['sha256']=={'spacr/qt/app.py':'same-source'}
    assert [row['source_files_sha256'][tool] for row in identity['composition']]==['old','new']
    assert identity['recorder_delta_paths']==[tool]


def test_composition_keeps_each_capture_identity_without_overwriting_stage_evidence(tmp_path):
    source=capture(tmp_path,'source','home')
    stage=tmp_path/'combined';stage.mkdir()
    historical=stage/'exact-source.json';historical.write_bytes(b'earlier stage evidence')
    output=stage/'captures/home'
    compose([source],output)
    assert historical.read_bytes()==b'earlier stage evidence'
    assert json.loads((output/'exact-source.json').read_text(encoding="utf-8"))['sha256']=={'spacr/qt/app.py':'same-source'}


@pytest.mark.parametrize('path',['spacr/qt/app.py','tools/tutorials/../spacr/qt/app.py'])
def test_application_changes_cannot_be_declared_as_recorder_fixes(tmp_path,path):
    first=capture(tmp_path,'first','home');second=capture(tmp_path,'second','other',source_hash='changed')
    with pytest.raises(ValueError):
        compose([first,second],tmp_path/'combined/captures/home',
                recorder_delta_paths=[path],recorder_delta_reason='A proposed application change.')


def test_selected_source_scenes_preserve_full_original_manifest(tmp_path):
    first=capture(tmp_path,'first','kept')
    manifest=json.loads((first/'frames.json').read_text(encoding="utf-8"))
    omitted=first/'superseded.png';omitted.write_bytes(b'original rejected image')
    manifest['superseded']={'image':omitted.name,'sha256':hashlib.sha256(omitted.read_bytes()).hexdigest()}
    (first/'frames.json').write_text(json.dumps(manifest), encoding="utf-8")
    original=(first/'frames.json').read_bytes()
    second=capture(tmp_path,'second','replacement');out=tmp_path/'combined/captures/home'
    result=compose([first,second],out,source_scenes={str(first.resolve()):['kept']})
    assert set(result)=={'kept','replacement'}
    assert (first/'frames.json').read_bytes()==original
    assert omitted.read_bytes()==b'original rejected image'
    assert not (out/'superseded.png').exists()
    receipt=json.loads((out/'provenance.json').read_text(encoding="utf-8"))
    assert receipt['composition'][0]['omitted_source_scenes']==['superseded']


@pytest.mark.parametrize('selection',['missing','empty','unknown_source'])
def test_invalid_source_selection_fails_before_writing(tmp_path,selection):
    first=capture(tmp_path,'first','kept');out=tmp_path/'combined/captures/home'
    choices={str(first.resolve()):['absent'] if selection=='missing' else []}
    if selection=='unknown_source':choices={str(tmp_path/'unknown'):['kept']}
    with pytest.raises(ValueError):compose([first],out,source_scenes=choices)
    assert not out.exists()
