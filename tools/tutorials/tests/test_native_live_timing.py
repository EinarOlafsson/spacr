"""Native reference scenes cover all speech at their original elapsed speed."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
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


def test_native_reference_rounds_up_to_sufficient_whole_frames():
    result=longest_scene_timing(lesson(),[{'scenes':[{'duration':1.011},{'duration':2.011}]}],{0,1})
    for scene in result['scenes']:
        assert scene['duration']>=scene['longest_narration_duration']
        assert scene['duration']-scene['longest_narration_duration']<1/30
        assert scene['duration']*30==pytest.approx(round(scene['duration']*30))


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
        path.write_text(json.dumps(value,ensure_ascii=False))
    write(root/'lesson.en.json',authored)
    video=root/'recorded.mp4';video.write_bytes(b'explicit nonmedia unit fixture')
    receipt=root/'recorded.capture.json'
    write(receipt,{'sha256':sha256(video),'duration':3.0,
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
    assert result['coverage']['track_count']==50
    assert len(result['source_tracks'])==50
    (root/'audio/en/af_heart.m4a').unlink()
    with pytest.raises(FileNotFoundError):plan(stage,'05_home')


@pytest.mark.parametrize('mutation',['audio','receipt','source','fingerprint','coverage'])
def test_changed_native_inputs_and_duration_gaps_fail_closed(frozen_native_stage,mutation):
    import json
    from native_live_timing import plan,checked_native_timing,sha256
    stage,root=frozen_native_stage;accepted=plan(stage,'05_home')
    sidecar=root/'video/native-live-timings.json';sidecar.parent.mkdir()
    sidecar.write_text(json.dumps(accepted));assert checked_native_timing(stage,'05_home')==sidecar
    if mutation=='audio':(root/'audio/en/af_heart.m4a').write_bytes(b'changed')
    elif mutation=='receipt':(root/'recorded.capture.json').write_text('{}')
    elif mutation=='source':
        path=stage/'catalog/lessons_en.json';value=json.loads(path.read_text())
        value['lessons'][0]['scenes'][0]['narration']='A changed instruction.';path.write_text(json.dumps(value))
    elif mutation=='fingerprint':
        path=root/'audio/en/af_heart.json';value=json.loads(path.read_text())
        value['render_fingerprint']='changed';path.write_text(json.dumps(value))
    else:
        path=root/'recorded.capture.json';value=json.loads(path.read_text())
        value['duration']=1;path.write_text(json.dumps(value))
        path=root/'scenes.json';value=json.loads(path.read_text())
        value['scenes'][0]['clip']['receipt_sha256']=sha256(root/'recorded.capture.json')
        path.write_text(json.dumps(value))
        assert not plan(stage,'05_home')['coverage']['accepted']
    with pytest.raises(ValueError):checked_native_timing(stage,'05_home')
