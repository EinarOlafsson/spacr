"""One factual static Doctor replacement retains every published source clock."""
from copy import deepcopy
from pathlib import Path
import json,sys
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'authoring/tools')]
import published_native_mix as mix
import render_platform_doctor_reference_card as doctor

@pytest.fixture
def plan():
 original={'scenes':[]};cursor=0.0
 for i,visual in enumerate(doctor.VISUALS):
  duration=29.3 if i==9 else 3.1
  original['scenes'].append(dict(scene=i+1,visual=visual,speech_start=cursor,speech_end=cursor+duration,scene_end=cursor+duration,duration=duration,native_live_video=i in (7,8)));cursor+=duration
 english=dict(id='04_platform_installers',scenes=[dict(visual=v,narration='Instruction '+v) for v in doctor.VISUALS])
 tracks=[dict(scenes=[dict(duration=2,speech_start=i*2,speech_end=i*2+1.48) for i in range(14)]) for _ in range(50)]
 cards={doctor.CARD:dict(kind='authored_reference_card',narration_sha256=mix.identity(english['scenes'][9]['narration']))}
 bindings=dict(master=dict(sha256=doctor.MASTER_SHA),timing=dict(sha256='b'*64))
 edit=dict(schema=1,lesson=english['id'],master_sha256=doctor.MASTER_SHA,visual_timing_sha256='b'*64,scenes=[])
 return english,original,tracks,cards,edit,bindings,{}

def test_exact879_frame_reference_preserves_all14_clocks_flags_and_inputs(plan):
 before=deepcopy(plan);rows=mix.plan_mix(*plan)
 assert [r['visual'] for r in rows]==list(doctor.VISUALS)
 assert [i+1 for i,r in enumerate(rows) if r['native_live_video']]==[8,9]
 assert [i+1 for i,r in enumerate(rows) if 'reference_card' in r]==[10]
 assert rows[9]['duration']==29.3 and round(rows[9]['duration']*30)==879
 for row,source in zip(rows,plan[1]['scenes']):
  assert {k:row[k] for k in ('speech_start','speech_end','scene_end','duration')}=={k:source[k] for k in ('speech_start','speech_end','scene_end','duration')}
 assert plan==before

@pytest.mark.parametrize('fault',['order','extra_card','native_flag','master','doctor_length','speech_overflow','native_edit'])
def test_doctor_refresh_cannot_change_footage_scope_or_narration_coverage(plan,fault):
 a=deepcopy(plan);english,original,tracks,cards,edit,bindings,fences=a
 if fault=='order':english['scenes'][0],english['scenes'][1]=english['scenes'][1],english['scenes'][0]
 elif fault=='extra_card':cards[doctor.VISUALS[0]]=deepcopy(cards[doctor.CARD])
 elif fault=='native_flag':original['scenes'][0]['native_live_video']=True
 elif fault=='master':bindings['master']['sha256']='f'*64;edit['master_sha256']='f'*64
 elif fault=='doctor_length':original['scenes'][9]['duration']=880/30
 elif fault=='speech_overflow':tracks[0]['scenes'][9].update(duration=30,speech_end=48)
 else:edit['scenes']=[dict(visual=doctor.VISUALS[7])]
 with pytest.raises(ValueError):mix.plan_mix(*a)

def test_original_diagnostic_row_required_not_just_matching_summary_words(tmp_path,monkeypatch):
 paths={name:tmp_path/name for name in doctor.SOURCE_HASHES}
 proof=dict(completed_capture=True,verification_recorded_live=True,doctor_gpu_allocation_probe_requested=False,installed_identity=dict(version='1.5.1.3'))
 commands=[dict(scene=doctor.CARD,command=['spacr-doctor','--no-gpu-probe'],accepted=True,returncode=0,output='13 passed, 1 warnings, 0 failed, 0 errored, 4 skipped; missing pylibCZIrw; torch currently uses cpu; no GPU probe was run')]
 paths['accepted-card.png'].write_bytes(b'explicit nonimage source fixture')
 paths['verification-provenance.json'].write_text(json.dumps(proof));paths['verification-commands.json'].write_text(json.dumps(commands))
 spec=dict(scenes=[{} for _ in range(10)],capture_source=dict(frame_origins={doctor.CARD:dict(kind='reference_guidance',capture='original',image_sha256=doctor.digest(paths['accepted-card.png']))},source_captures=[dict(capture='original',provenance=dict(verification_sha256=doctor.digest(paths['verification-provenance.json'])))]))
 spec['scenes'][9]=dict(capture_sha256=doctor.digest(paths['accepted-card.png']));paths['accepted-scene-spec.json'].write_text(json.dumps(spec))
 monkeypatch.setattr(doctor,'SOURCE_HASHES',{name:doctor.digest(path) for name,path in paths.items()})
 assert doctor.checked_sources(paths)==proof
 commands[0]['returncode']=1;paths['verification-commands.json'].write_text(json.dumps(commands))
 # Even an explicitly rebound unit hash cannot turn a failed command into evidence.
 doctor.SOURCE_HASHES['verification-commands.json']=doctor.digest(paths['verification-commands.json'])
 with pytest.raises(ValueError,match='command/results'):doctor.checked_sources(paths)


def test_platform_default_selects_only_exact_existing_binding_and_rejects_missing_or_duplicate(tmp_path,monkeypatch):
    path=tmp_path/'tools/tutorials/idle_tail_policy.json';path.parent.mkdir(parents=True)
    selected=dict(lesson='04_platform_installers',master_sha256='a'*64,visual_timing_sha256='b'*64,scenes=[dict(visual='05_doctor',source_start=206.0333333333333,source_end=235.33333333333331,idle_tail_start=0)])
    other=dict(lesson='07_mask',scenes=[])
    monkeypatch.setattr(mix,'REPO',tmp_path)
    path.write_text(json.dumps(dict(schema=1,lessons=[other,selected])))
    result=mix.platform_source_policy()
    assert result==dict(schema=1,lessons=[selected])
    result['lessons'][0]['scenes'][0]['idle_tail_start']=99
    assert mix.platform_source_policy()['lessons'][0]==selected
    for entries in ([other],[selected,selected]):
        path.write_text(json.dumps(dict(schema=1,lessons=entries)))
        with pytest.raises(ValueError,match='exactly one'):mix.platform_source_policy()
