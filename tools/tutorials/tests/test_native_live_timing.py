"""Native reference scenes cover all speech at their original elapsed speed."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'authoring' / 'tools'))
from native_live_timing import longest_scene_timing


def lesson():
    return {'id':'05_home','scenes':[{'visual':'first','narration':'First control.'},
                                   {'visual':'second','narration':'Second control.'}]}


def test_native_reference_uses_each_scenes_longest_voice_without_time_warp():
    tracks=[{'scenes':[{'duration':1.01},{'duration':2.1}]},
            {'scenes':[{'duration':1.9},{'duration':1.7}]}]
    result=longest_scene_timing(lesson(),tracks,{0},fps=30)
    assert result['scenes'][0]['duration']==1.9
    assert result['scenes'][1]['duration']==2.1
    assert result['scenes'][0]['native_live_video'] is True
    assert result['scenes'][1]['native_live_video'] is False
    assert result['scenes'][1]['speech_start']==1.9
    assert result['total_duration']==4
    assert result['playback_speed']==1 and not result['looped_or_stretched']
    assert 'complete_native_clips' not in result


def test_native_reference_rounds_up_to_sufficient_whole_frames():
    result=longest_scene_timing(lesson(),[{'scenes':[{'duration':1.011},{'duration':2.011}]}],{0,1})
    for scene in result['scenes']:
        assert scene['duration']>=scene['longest_narration_duration']
        assert scene['duration']-scene['longest_narration_duration']<1/30
        assert scene['duration']*30==pytest.approx(round(scene['duration']*30))


def test_native_reference_preserves_complete_source_and_apply_actions_after_short_speech():
    tracks = [{'scenes': [{'duration': 14.8}, {'duration': 26.933333333333334}]}]
    result = longest_scene_timing(lesson(), tracks, {0, 1},
                                 native_durations={0: 812 / 30, 1: 1743 / 30})
    assert result['scenes'][0]['duration'] == 812 / 30
    assert result['scenes'][1]['speech_start'] == 812 / 30
    assert result['scenes'][1]['duration'] == 1743 / 30
    assert result['total_duration'] == pytest.approx(2555 / 30)
    assert result['playback_speed'] == 1 and not result['looped_or_stretched']


@pytest.mark.parametrize('tracks', [[],[{'scenes':[]}],
    [{'scenes':[{'duration':float('nan')},{'duration':1}]}],
    [{'scenes':[{'duration':0},{'duration':1}]}]])
def test_missing_mismatched_or_invalid_track_durations_fail(tracks):
    with pytest.raises(ValueError):longest_scene_timing(lesson(),tracks,{0,1})


@pytest.fixture
def frozen_native_stage(tmp_path):
    import json
    from native_live_timing import sha256
    from render_all_voices import (LANGUAGES, narration_dialect, prepare_scene_plans,
                                  resolve_voice_speed, track_fingerprint)
    authored=lesson();identity=authored['id'];root=tmp_path/'production'/identity
    root.mkdir(parents=True)
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,ensure_ascii=False), encoding="utf-8")
    write(root/'lesson.en.json',authored)
    video=root/'recorded.mp4';video.write_bytes(b'explicit nonmedia unit fixture')
    receipt=root/'recorded.capture.json'
    write(receipt,{'sha256':sha256(video),'duration':3.0,
                  'encoded_fps':'30/1','frames':[{'pts': index} for index in range(90)],
                  'full_decode_passed':True,'audio_streams':0})
    write(root/'scenes.json',{'fps':30,'scenes':[{'clip':{
        'video':video.name,'receipt':receipt.name,'sha256':sha256(video),
        'receipt_sha256':sha256(receipt)}},{}]})
    for language,(code,voices) in LANGUAGES.items():
        from copy import deepcopy
        localized=deepcopy(authored)
        if language in ('ja','zh-CN'):
            texts=('最初の操作です。','次の操作です。') if language=='ja' else ('第一项操作。','第二项操作。')
            for scene,text in zip(localized['scenes'],texts):scene['narration']=text
        write(tmp_path/'catalog'/f'lessons_{language}.json',{'lessons':[localized]})
        for voice in voices:
            media=root/'audio'/language/f'{voice}.m4a'
            media.parent.mkdir(parents=True,exist_ok=True)
            media.write_bytes(f'nonmedia unit fixture {language} {voice}'.encode())
            actual_code='b' if language=='en' and voice.startswith('b') else code
            dialect=narration_dialect(language,actual_code,voice);speed=resolve_voice_speed(voice)
            plans=prepare_scene_plans(localized,language,dialect,speed,voice=voice)
            fingerprint,inputs=track_fingerprint(localized,language,actual_code,dialect,
                voice,speed,plans,runtime_identity={'state':'explicit-unit-fixture'})
            write(media.with_suffix('.json'),{'language':language,'voice':voice,
                'media_sha256':sha256(media),'media_bytes':media.stat().st_size,
                'render_fingerprint':fingerprint,'render_inputs':inputs,
                'scenes':[{'text':scene['narration'],'duration':2.0}
                          for scene in localized['scenes']]})
    return tmp_path,root


def test_native_acceptance_requires_all_fifty_current_tracks(frozen_native_stage):
    from native_live_timing import plan
    stage,root=frozen_native_stage;result=plan(stage,'05_home')
    assert result['coverage']['accepted']
    assert result['complete_native_clips'] is True
    assert result['coverage']['track_count']==50
    assert len(result['source_tracks'])==50
    assert result['scenes'][0]['duration']==3.0
    (root/'audio/en/af_heart.m4a').unlink()
    with pytest.raises(FileNotFoundError):plan(stage,'05_home')


@pytest.mark.parametrize('mutation',['audio','receipt','source','fingerprint','coverage'])
def test_changed_native_inputs_and_duration_gaps_fail_closed(frozen_native_stage,mutation):
    import json
    from native_live_timing import plan,checked_native_timing,sha256
    stage,root=frozen_native_stage;accepted=plan(stage,'05_home')
    sidecar=root/'video/native-live-timings.json';sidecar.parent.mkdir()
    sidecar.write_text(json.dumps(accepted), encoding="utf-8");assert checked_native_timing(stage,'05_home')==sidecar
    if mutation=='audio':(root/'audio/en/af_heart.m4a').write_bytes(b'changed')
    elif mutation=='receipt':(root/'recorded.capture.json').write_text('{}', encoding="utf-8")
    elif mutation=='source':
        path=stage/'catalog/lessons_en.json';value=json.loads(path.read_text(encoding="utf-8"))
        value['lessons'][0]['scenes'][0]['narration']='A changed instruction.';path.write_text(json.dumps(value), encoding="utf-8")
    elif mutation=='fingerprint':
        path=root/'audio/en/af_heart.json';value=json.loads(path.read_text(encoding="utf-8"))
        value['render_fingerprint']='changed';path.write_text(json.dumps(value), encoding="utf-8")
    else:
        path=root/'recorded.capture.json';value=json.loads(path.read_text(encoding="utf-8"))
        value['duration']=1;path.write_text(json.dumps(value), encoding="utf-8")
        value['frames']=value['frames'][:30];path.write_text(json.dumps(value), encoding="utf-8")
        path=root/'scenes.json';value=json.loads(path.read_text(encoding="utf-8"))
        value['scenes'][0]['clip']['receipt_sha256']=sha256(root/'recorded.capture.json')
        path.write_text(json.dumps(value), encoding="utf-8")
        assert not plan(stage,'05_home')['coverage']['accepted']
    with pytest.raises(ValueError):checked_native_timing(stage,'05_home')


def test_complete_capture_clock_does_not_round_container_milliseconds_up_to_a_fake_frame():
    from native_live_timing import complete_native_duration
    receipt = {'duration': 13.534, 'encoded_fps': '30/1',
               'frames': [{'pts': index} for index in range(406)]}
    assert complete_native_duration(receipt, 30) == 406 / 30


@pytest.mark.parametrize('mutation', ['missing', 'gap', 'rate', 'duration'])
def test_incomplete_or_inconsistent_native_frame_evidence_fails(mutation):
    from native_live_timing import complete_native_duration
    receipt = {'duration': 1.0, 'encoded_fps': '30/1',
               'frames': [{'pts': index} for index in range(30)]}
    if mutation == 'missing': receipt.pop('frames')
    elif mutation == 'gap': receipt['frames'][10]['pts'] = 11
    elif mutation == 'rate': receipt['encoded_fps'] = '60/1'
    else: receipt['duration'] = 2
    with pytest.raises(ValueError): complete_native_duration(receipt, 30)



@pytest.mark.parametrize('fps,encoded_fps,duration,accepted', [
    (0, '30/1', 1.0, False),
    (30, 'not-a-rate', 1.0, False),
    (30, '30/1', 1.0019, True),
    (30, '30/1', 1.0021, False),
])
def test_complete_capture_duration_boundary_is_explicit(fps, encoded_fps, duration, accepted):
    from native_live_timing import complete_native_duration
    receipt = {'duration': duration, 'encoded_fps': encoded_fps,
               'frames': [{'pts': index} for index in range(30)]}
    if accepted:
        assert complete_native_duration(receipt, fps) == 1.0
    else:
        with pytest.raises(ValueError):
            complete_native_duration(receipt, fps)


@pytest.fixture
def mixed_prose_native_stage(tmp_path, monkeypatch):
    """Adapter boundary only: the engine and immutable-source validators are stubs."""
    from copy import deepcopy
    import json, types
    import native_live_timing as timing
    import published_native_reuse as reuse
    from check_completed_matrix import voice_matrix
    stage=tmp_path/'stage';baseline=tmp_path/'published';baseline.mkdir()
    identity='20_cellpose_masks';root=stage/'production'/identity
    english={'id':identity,'scenes':[{'visual':'first','narration':'Approved current text.'}]}
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    put(root/'lesson.en.json',english)
    inventory=voice_matrix(Path(__file__).resolve().parents[3]/'tools/tutorials/authoring/tools/render_all_voices.py')
    keys=[(language,voice) for language,voices in inventory.items() for voice in voices]
    bindings={'master':{'path':f'media_host/{identity}/video/{identity}_silent.mp4','sha256':'1'*64,'bytes':100},
              'timing':{'path':f'media_host/{identity}/video/native-live-timings.json','sha256':'2'*64,'bytes':200}}
    source={'baseline_root':str(baseline),'baseline_manifest_sha256':'3'*64,
            'baseline_publication_receipt_sha256':'4'*64,'baseline_media_commit':'5'*40,
            'media_root':'https://example.invalid/immutable'}
    original={'source_tracks':[{'language':l,'voice':v} for l,v in keys]}
    record=dict(schema=1,kind='published_native_narration_reuse',lesson=identity,
                fresh_capture=False,raw_capture_revalidated=False,**source,
                published_master=bindings['master'],published_timing=bindings['timing'])
    put(root/reuse.RECORD,record)
    mixed={'schema':1,'kind':'allowlisted_published_prose_audio','source':{
        'lesson':identity,'native_adapter_required':True, 'baseline_root':str(baseline),
        'baseline_manifest_sha256':source['baseline_manifest_sha256'],
        'publication_receipt_sha256':source['baseline_publication_receipt_sha256'],
        'original_visual_clock':dict(kind='native_visual_timing',path=bindings['timing']['path'],
            sha256=bindings['timing']['sha256'],published_bindings=bindings),
        'tracks':[dict(language=l,voice=v,audio_sha256='a'*64) for l,v in keys]}}
    engine=types.ModuleType('published_prose_audio_refresh')
    engine.RECORD='published-prose-audio-refresh.json';engine.KIND=mixed['kind']
    put(root/engine.RECORD,mixed)
    proofs=[]
    for language,voice in keys:
        media=root/f'audio/{language}/{voice}.m4a';media.parent.mkdir(parents=True,exist_ok=True)
        media.write_bytes(f'nonmedia fixture {language}/{voice}'.encode())
        audio_hash=timing.sha256(media)
        put(media.with_suffix('.json'),dict(language=language,voice=voice,
            media_sha256=audio_hash,media_bytes=media.stat().st_size,
            mixed_published_scene_audio=True,scenes=[{'text':'Approved current text.'}]))
        proofs.append(dict(language=language,voice=voice,audio_sha256=audio_hash,
            timing_sha256=timing.sha256(media.with_suffix('.json')),provenance_kind=engine.KIND,
            mixed_record_sha256=timing.sha256(root/engine.RECORD),original_audio_sha256='a'*64))
    calls=[]
    def checked(s,b,i,*,decode):
        assert (s,b,i,decode)==(stage,baseline,identity,False)
        calls.append((i,decode))
        return deepcopy(inventory),deepcopy(proofs)
    engine.checked_tracks=checked
    monkeypatch.setitem(sys.modules,'published_prose_audio_refresh',engine)
    monkeypatch.setattr(reuse,'_published_inputs',lambda b,i:(baseline,original,bindings,source))
    return dict(stage=stage,root=root,baseline=baseline,identity=identity,english=english,
                engine=engine,inventory=inventory,proofs=proofs,original=original,calls=calls,put=put)


def test_mixed_prose_native_dispatch_returns_truthful_fifty_proofs_without_fingerprints(mixed_prose_native_stage):
    import native_live_timing as timing
    e=mixed_prose_native_stage
    tracks,proofs=timing.verified_narration_tracks(e['stage'],e['identity'],e['english'])
    assert len(tracks)==len(proofs)==50 and e['calls']==[(e['identity'],False)]
    assert all(p['provenance_kind']==e['engine'].KIND and 'render_fingerprint' not in p for p in proofs)
    assert {(t['language'],t['voice']) for t in tracks}=={(p['language'],p['voice']) for p in proofs}


@pytest.mark.parametrize('part',['baseline','manifest','receipt','master','clock','native_kind','reuse_kind'])
def test_mixed_prose_native_foreign_bindings_fail_before_engine(mixed_prose_native_stage,part):
    import native_live_timing as timing
    import json
    e=mixed_prose_native_stage;p=e['root']/e['engine'].RECORD;r=json.loads(p.read_text())
    if part=='baseline':r['source']['baseline_root']=str(e['baseline'].parent/'foreign')
    if part=='manifest':r['source']['baseline_manifest_sha256']='foreign'
    if part=='receipt':r['source']['publication_receipt_sha256']='foreign'
    if part=='master':r['source']['original_visual_clock']['published_bindings']['master']['sha256']='foreign'
    if part=='clock':r['source']['original_visual_clock']['sha256']='foreign'
    if part=='native_kind':r['source']['original_visual_clock']['kind']='legacy'
    if part=='reuse_kind':
        q=e['root']/'published-native-reuse.json';v=json.loads(q.read_text());v['kind']='fresh_capture';e['put'](q,v)
    e['put'](p,r)
    with pytest.raises(ValueError):timing.verified_narration_tracks(e['stage'],e['identity'],e['english'])
    assert e['calls']==[]


@pytest.mark.parametrize('part',['missing_native_proof','duplicate_native_proof','retained_conflict','foreign_lesson'])
def test_mixed_prose_native_requires_explicit_scope_and_original_fifty(mixed_prose_native_stage,part):
    import native_live_timing as timing
    e=mixed_prose_native_stage
    if part=='missing_native_proof':e['original']['source_tracks'].pop()
    if part=='duplicate_native_proof':e['original']['source_tracks'][1]=e['original']['source_tracks'][0]
    if part=='retained_conflict':e['put'](e['root']/'published-retained-narration.json',{})
    if part=='foreign_lesson':
        wrong=e['stage']/'production'/'81_sequencing_pathways';e['put'](wrong/e['engine'].RECORD,{})
        with pytest.raises(ValueError,match='three reviewed'):timing.verified_narration_tracks(e['stage'],'81_sequencing_pathways',{})
        assert e['calls']==[];return
    with pytest.raises(ValueError):timing.verified_narration_tracks(e['stage'],e['identity'],e['english'])
    assert e['calls']==[]


@pytest.mark.parametrize('part',['kind','record_hash','original_hash','timing_hash','fresh_fingerprint','missing_current_voice'])
def test_mixed_prose_native_rejects_malformed_current_proof(mixed_prose_native_stage,part):
    import native_live_timing as timing
    e=mixed_prose_native_stage;p=e['proofs'][0]
    if part=='kind':p['provenance_kind']='fresh_tts'
    if part=='record_hash':p['mixed_record_sha256']='foreign'
    if part=='original_hash':p['original_audio_sha256']='foreign'
    if part=='timing_hash':p['timing_sha256']='foreign'
    if part=='fresh_fingerprint':p['render_fingerprint']='invented'
    if part=='missing_current_voice':e['proofs'].pop()
    with pytest.raises(ValueError):timing.verified_narration_tracks(e['stage'],e['identity'],e['english'])
    assert e['calls']==[(e['identity'],False)]


def test_mixed_prose_absence_keeps_existing_retained_dispatch(mixed_prose_native_stage,monkeypatch):
    import native_live_timing as timing
    import published_retained_narration as retained
    e=mixed_prose_native_stage;record=e['root']/e['engine'].RECORD;record.rename(record.with_name(record.name+'.held'))
    e['put'](e['root']/retained.RECORD,{})
    sentinel=([{'old':'track'}],[{'old':'proof'}])
    monkeypatch.setattr(retained,'checked_published_narration',lambda s,i:sentinel)
    assert timing.verified_narration_tracks(e['stage'],e['identity'],e['english'])==sentinel
    assert e['calls']==[]
