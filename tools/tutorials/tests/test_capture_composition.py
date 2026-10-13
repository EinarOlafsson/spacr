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
    (root/'exact-source.json').write_text(json.dumps({'sha256':{'spacr/qt/app.py':source_hash}}))
    (folder/'provenance.json').write_text(json.dumps({'completed_capture':True}))
    image=folder/(visual+'.png');image.write_bytes((visual+'-actual-frame').encode())
    frames={visual:{'image':image.name,'sha256':hashlib.sha256(image.read_bytes()).hexdigest()}}
    (folder/'frames.json').write_text(json.dumps(frames));return folder


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
        path=folder.parent.parent/'exact-source.json';value=json.loads(path.read_text())
        value['sha256'][tool]=version;path.write_text(json.dumps(value))
    out=tmp_path/'combined/captures/home'
    with pytest.raises(ValueError):compose([first,second],out)
    compose([first,second],out,recorder_delta_paths=[tool],
            recorder_delta_reason='Select unchanged bundled templates through actual editors.')
    identity=json.loads((out/'exact-source.json').read_text())
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
    assert json.loads((output/'exact-source.json').read_text())['sha256']=={'spacr/qt/app.py':'same-source'}


@pytest.mark.parametrize('path',['spacr/qt/app.py','tools/tutorials/../spacr/qt/app.py'])
def test_application_changes_cannot_be_declared_as_recorder_fixes(tmp_path,path):
    first=capture(tmp_path,'first','home');second=capture(tmp_path,'second','other',source_hash='changed')
    with pytest.raises(ValueError):
        compose([first,second],tmp_path/'combined/captures/home',
                recorder_delta_paths=[path],recorder_delta_reason='A proposed application change.')


def test_selected_source_scenes_preserve_full_original_manifest(tmp_path):
    first=capture(tmp_path,'first','kept')
    manifest=json.loads((first/'frames.json').read_text())
    omitted=first/'superseded.png';omitted.write_bytes(b'original rejected image')
    manifest['superseded']={'image':omitted.name,'sha256':hashlib.sha256(omitted.read_bytes()).hexdigest()}
    (first/'frames.json').write_text(json.dumps(manifest))
    original=(first/'frames.json').read_bytes()
    second=capture(tmp_path,'second','replacement');out=tmp_path/'combined/captures/home'
    result=compose([first,second],out,source_scenes={str(first.resolve()):['kept']})
    assert set(result)=={'kept','replacement'}
    assert (first/'frames.json').read_bytes()==original
    assert omitted.read_bytes()==b'original rejected image'
    assert not (out/'superseded.png').exists()
    receipt=json.loads((out/'provenance.json').read_text())
    assert receipt['composition'][0]['omitted_source_scenes']==['superseded']


@pytest.mark.parametrize('selection',['missing','empty','unknown_source'])
def test_invalid_source_selection_fails_before_writing(tmp_path,selection):
    first=capture(tmp_path,'first','kept');out=tmp_path/'combined/captures/home'
    choices={str(first.resolve()):['absent'] if selection=='missing' else []}
    if selection=='unknown_source':choices={str(tmp_path/'unknown'):['kept']}
    with pytest.raises(ValueError):compose([first],out,source_scenes=choices)
    assert not out.exists()
