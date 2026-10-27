"""Synthetic source objects exercise composition paths, not capture acceptance."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import compose_classify_capture as composer


@pytest.fixture
def composition(tmp_path, monkeypatch):
    stage=tmp_path/'physical';stage.mkdir()
    terminal=tmp_path/'old-preparation';terminal.mkdir()
    gui=stage/'captures'/'classify_merged';gui.mkdir(parents=True)
    dataset=stage/'dataset';dataset.mkdir()
    references=tmp_path/'accepted-reference-cards';references.mkdir()
    saved=references/'source-results';saved.mkdir()
    (saved/'original-proof.json').write_text('{}')
    (references/'cards.json').write_text('{}')
    def write(root,name,value):
        (root/name).write_text(json.dumps(value))
    inputs=dict(objects_checked=64,sample_counts={'train':32,'test':32})
    proof=dict(inputs=inputs,native_audits={'accepted-fixture':True})
    metrics=dict(accuracy=.5)
    helper=tmp_path/'prepare_classify_split.py';helper.write_bytes(b'fixture helper')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    write(terminal,'scientific_acceptance.json',dict(accepted=True,preparation_only=True,
        model_started=False,filename_parser_fixed=False,destination='/recorded-capture/dataset',
        inputs=inputs,helper_sha256=digest(helper)))
    write(gui,'scientific_acceptance.json',dict(**proof,independent_test_metrics=metrics))
    write(gui,'batch_acceptance.json',dict(accepted=True))
    for root in [terminal,gui]:
        write(root,'provenance.json',dict(completed_capture=True,module='classify_merged'))
    write(terminal,'frames.json',{'unused_old_terminal':{}})
    clip={}
    for field,sha_field in [('video','sha256'),('receipt','receipt_sha256'),('native_video','native_sha256')]:
        p=gui/('clip.'+field);p.write_bytes(field.encode());clip[field]=p.name;clip[sha_field]=digest(p)
    receipt=gui/clip['receipt']
    receipt.write_text(json.dumps(dict(sha256=clip['sha256'],full_decode_passed=True,audio_streams=0)))
    clip['receipt_sha256']=digest(receipt)
    image=gui/'gui.png';image.write_bytes(b'original GUI PNG fixture')
    write(gui,'frames.json',{'19_setting_classifier_family':dict(image=image.name,sha256=digest(image),clip=clip)})
    cards={}
    for visual in ['prepare_01_source_example','prepare_02_prepare_existing_split','prepare_03_exported_manifest']:
        p=references/(visual+'.png');p.write_bytes(visual.encode())
        cards[visual]=dict(image=p.name,sha256=digest(p),narration_sha256='original-narration')
    import render_classify_input_reference_cards as normal_cards
    calls=[]
    def validate(folder,lesson,repo):
        calls.append(folder);assert folder==references
        return cards
    monkeypatch.setattr(normal_cards,'validate_cards',validate)
    monkeypatch.setattr(composer,'read',lambda p:dict(id='10_classify_cv'))
    def frame(p,expected,root,hashes):
        assert p.resolve().is_relative_to(root)
        composer._same_hash(p,expected,hashes)
        return p.resolve()
    monkeypatch.setattr(composer,'_frame',frame)
    monkeypatch.setattr(composer,'__file__',str(tmp_path/'compose_classify_capture.py'))
    def finished(actual,*,path_mapping):
        assert actual==dataset and path_mapping==(Path('/recorded-capture'),stage)
        return proof
    monkeypatch.setattr(composer,'inspect_finished',finished)
    monkeypatch.setattr(composer,'inspect_metrics',lambda *a,**k:metrics)
    return SimpleNamespace(stage=stage,terminal=terminal,gui=gui,dataset=dataset,
        references=references,destination=stage/'composed',clip=clip,calls=calls)


def test_composition_preserves_three_reference_cards_and_original_native_clip_bindings(composition):
    c=composition;original=(c.gui/'frames.json').read_bytes()
    composer.compose(c.stage,terminal=c.terminal,gui=c.gui,dataset=c.dataset,
        destination=c.destination,recorded_stage='/recorded-capture',preparation_references=c.references)
    frames=json.loads((c.destination/'frames.json').read_text())
    assert c.calls==[c.references]
    assert 'prepare_unused_old_terminal' not in frames
    assert sum(r.get('kind')=='authored_reference_card' for r in frames.values())==3
    joined=frames['gui_19_setting_classifier_family']['clip']
    for field,sha_field in [('video','sha256'),('receipt','receipt_sha256'),('native_video','native_sha256')]:
        assert (c.destination/joined[field]).resolve()==c.gui/c.clip[field]
        assert joined[sha_field]==c.clip[sha_field]
    assert (c.gui/'frames.json').read_bytes()==original
    proof=json.loads((c.destination/'scientific_acceptance.json').read_text())
    assert proof['preparation']['destination']=='/recorded-capture/dataset'
    assert proof['physical_stage']==str(c.stage)


@pytest.mark.parametrize('fault',['changed_clip','escape'])
def test_composition_rejects_changed_or_foreign_clip_before_destination(composition,fault):
    c=composition
    if fault=='changed_clip':(c.gui/c.clip['video']).write_bytes(b'changed')
    else:
        p=c.gui/'frames.json';frames=json.loads(p.read_text())
        frames['19_setting_classifier_family']['clip']['video']='../foreign.mp4'
        p.write_text(json.dumps(frames))
    with pytest.raises(ValueError):
        composer.compose(c.stage,terminal=c.terminal,gui=c.gui,dataset=c.dataset,
            destination=c.destination,recorded_stage='/recorded-capture',preparation_references=c.references)
    assert not c.destination.exists()


def test_stage_copy_keeps_hashes_and_actual_gui_identity_inside_stage(composition):
    c=composition
    identity=c.stage/'exact-source.json';identity.write_text(json.dumps(dict(sha256={'app.py':'actual-gui-source'})))
    gate=c.gui/'host_gate_summary.json';gate.write_text(json.dumps(dict(clips={'actual':dict(passed=True)})))
    composer.compose(c.stage,terminal=c.terminal,gui=c.gui,dataset=c.dataset,
        destination=c.destination,recorded_stage='/recorded-capture',
        preparation_references=c.references,copy_inputs=True)
    assert (c.destination/'exact-source.json').read_bytes()==identity.read_bytes()
    assert (c.destination/'host_gate_summary.json').read_bytes()==gate.read_bytes()
    frames=json.loads((c.destination/'frames.json').read_text())
    for row in frames.values():
        image=(c.destination/row['image']).resolve()
        assert image.is_relative_to(c.destination)
        assert hashlib.sha256(image.read_bytes()).hexdigest()==row['sha256']
        if row.get('clip'):
            for field,sha_field in [('video','sha256'),('receipt','receipt_sha256'),('native_video','native_sha256')]:
                p=(c.destination/row['clip'][field]).resolve()
                assert p.is_relative_to(c.destination)
                assert hashlib.sha256(p.read_bytes()).hexdigest()==row['clip'][sha_field]


@pytest.mark.parametrize('fault',[None,'different_metrics','different_inputs','different_audits'])
def test_normal_two_take_union_requires_same_science_and_keeps_original_native_sources(composition,fault):
    from compose_capture_set import compose as union
    import shutil
    c=composition
    identity=c.stage/'exact-source.json';identity.write_text(json.dumps(dict(sha256={'app.py':'same-source'})))
    retry=c.stage.parent/'retry';gui2=retry/'captures/classify_merged';gui2.mkdir(parents=True)
    shutil.copy2(identity,retry/'exact-source.json')
    for name in ['provenance.json','batch_acceptance.json','scientific_acceptance.json']:
        shutil.copy2(c.gui/name,gui2/name)
    frames=json.loads((c.gui/'frames.json').read_text())
    row=frames['19_setting_classifier_family']
    for name in [row['image'],*[row['clip'][key] for key in ['video','receipt','native_video']]]:
        shutil.copy2(c.gui/name,gui2/('retry_'+name))
    row['image']='retry_'+row['image']
    for key in ['video','receipt','native_video']:row['clip'][key]='retry_'+row['clip'][key]
    (gui2/'frames.json').write_text(json.dumps({'23_batch_finished':row}))
    (c.gui/'host_gate_summary.json').write_text(json.dumps(dict(clips={
        '19_setting_classifier_family':dict(passed=True,distinct_fps=29.5)})))
    (gui2/'host_gate_summary.json').write_text(json.dumps(dict(clips={
        '23_batch_finished':dict(passed=False,distinct_fps=27.7)})))
    if fault:
        p=gui2/'scientific_acceptance.json';proof=json.loads(p.read_text())
        key={'different_metrics':'independent_test_metrics','different_inputs':'inputs','different_audits':'native_audits'}[fault]
        proof[key]={'different':True};p.write_text(json.dumps(proof))
    combined=c.stage/'normal-union'
    union([c.gui,gui2],combined)
    union_frames=json.loads((combined/'frames.json').read_text())
    assert union_frames['19_setting_classifier_family']['composition_source']==dict(
        capture=str(c.gui),visual='19_setting_classifier_family')
    assert union_frames['23_batch_finished']['composition_source']==dict(
        capture=str(gui2),visual='23_batch_finished')
    assert all(not (combined/frame['clip']['native_video']).exists() for frame in union_frames.values())
    (combined/'host_gate_summary.json').write_text(json.dumps(dict(clips={})))
    if fault:
        with pytest.raises(ValueError,match='Every GUI take'):
            composer.compose(c.stage,terminal=c.terminal,gui=combined,dataset=c.dataset,
                destination=c.destination,recorded_stage='/recorded-capture',
                preparation_references=c.references,copy_inputs=True,gui_sources=[c.gui,gui2])
        assert not c.destination.exists()
    else:
        composer.compose(c.stage,terminal=c.terminal,gui=combined,dataset=c.dataset,
            destination=c.destination,recorded_stage='/recorded-capture',
            preparation_references=c.references,copy_inputs=True,gui_sources=[c.gui,gui2])
        frames=json.loads((c.destination/'frames.json').read_text())
        assert {'gui_19_setting_classifier_family','gui_23_batch_finished'}<=set(frames)
        proof=json.loads((c.destination/'scientific_acceptance.json').read_text())
        assert [row['capture'] for row in proof['original_gui_proofs']]==[str(c.gui),str(gui2)]
        gate=json.loads((c.destination/'host_gate_summary.json').read_text())
        assert gate['failed']==['gui_23_batch_finished']
        assert gate['clips']['gui_23_batch_finished']['passed'] is False
        for index,original in enumerate([c.gui,gui2]):
            assert (c.destination/f'_inputs/gui-gates/{index}.json').read_bytes()==(
                original/'host_gate_summary.json').read_bytes()
        for row in frames.values():
            if row.get('clip'):
                p=c.destination/row['clip']['native_video']
                assert p.is_file()
                assert hashlib.sha256(p.read_bytes()).hexdigest()==row['clip']['native_sha256']
