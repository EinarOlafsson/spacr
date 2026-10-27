"""Offline planner, actual input binding and normal consumer dispatch contracts."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import struct
import sys
import importlib.util
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE),str(HERE/'authoring/tools')]
import published_platform_native_refresh as refresh
spec=importlib.util.spec_from_file_location('published_native_mix',HERE/'published_native_mix.py')
mix=importlib.util.module_from_spec(spec)
sys.modules['published_native_mix']=mix
spec.loader.exec_module(mix)
print('ACTUAL_PRIVATE_IMPORT',refresh.__file__,mix.__file__,flush=True)
from render_platform_maintenance_references import VISUALS

@pytest.fixture
def scenario():
    english=dict(id=refresh.LESSON,scenes=[dict(visual=v,narration='Feature '+v) for v in VISUALS])
    original=dict(scenes=[dict(scene=i,visual=v,native_live_video=i in (8,9),
        speech_start=(i-1)*3,speech_end=i*3,scene_end=i*3,duration=3,
        reference_card=dict(kind='authored_reference_card',sha256='a'*64))
        for i,v in enumerate(VISUALS,1)])
    for i in (8,9):original['scenes'][i-1].pop('reference_card')
    tracks=[dict(scenes=[dict(speech_start=0,speech_end=1.5,duration=2,
                              sentences=[dict(audible_end=1.5)]) for _ in VISUALS]) for _ in range(50)]
    replacements=[dict(selection=dict(source_scene=i,visual=VISUALS[i-1],idle_tail_start=1),
        frames=45,video=f'/actual-{i}.mp4',inventory={'spacr/app.py':'a'*64}) for i in (8,9)]
    return english,original,tracks,replacements,{v:0 for v in VISUALS}

def test_keeps_twelve_complete_published_intervals_and_only_two_fresh_native(scenario):
    rows=refresh.plan_rows(*scenario)
    assert len(rows)==14 and [r['scene'] for r in rows if r['fresh_capture']]==[8,9]
    for old,row in zip(scenario[1]['scenes'],rows):
        if row['scene'] in (8,9):
            assert row['kept_frames']==45 and row['final_frame_hold_frames']==15
            assert row['native_live_video'] is True
        else:
            assert row['source_start_frame']==round(old['speech_start']*30)
            assert row['source_keep_frames']==[[0,90]] and row['final_frame_hold_frames']==0
            assert row['duration']==old['duration'] and row['reference_card']==old['reference_card']
            assert row['original_source']==old
    assert all(a['scene_end']==b['speech_start'] for a,b in zip(rows,rows[1:]))

@pytest.mark.parametrize('fault',['wrong_lesson','wrong_slot','wrong_order','native_reference',
                                 'missing_voice','missing_fence','action_overflow','retained_speech_overflow'])
def test_planner_fails_closed(scenario,fault):
    english,old,tracks,replacements,fences=deepcopy(scenario)
    if fault=='wrong_lesson':english['id']='05_home'
    elif fault=='wrong_slot':replacements[0]['selection']['source_scene']=7
    elif fault=='wrong_order':english['scenes'].reverse()
    elif fault=='native_reference':old['scenes'][0]['native_live_video']=True
    elif fault=='missing_voice':tracks.pop()
    elif fault=='missing_fence':fences.pop(VISUALS[0])
    elif fault=='action_overflow':replacements[0]['selection']['idle_tail_start']=1.6
    else:tracks[0]['scenes'][0]['duration']=3.1
    with pytest.raises(ValueError):refresh.plan_rows(english,old,tracks,replacements,fences)

RULE_PATH=Path('/media/carruthers/mnt3/claude/capture_host_proto/t18/gate_rule.py')
@pytest.mark.parametrize('fault',[None,'loss_only','too_slow','admission_freeze','freeze','slow','unknown'])
def test_uses_existing_campaign_admission_plus_raw_freeze_rejection(fault,tmp_path):
    accepted=refresh.rule({'admission_rule':dict(path=str(RULE_PATH),sha256=refresh.RULE_SHA)})
    gate=dict(passed=True,reasons=[],distinct_fps=29,app_freeze_seconds=[])
    if fault=='loss_only':gate.update(passed=False,reasons=['choppy (capture loss): 27.5 distinct fps'],distinct_fps=27.5)
    elif fault=='too_slow':gate.update(passed=False,reasons=['choppy (capture loss): 26 distinct fps'],distinct_fps=26)
    elif fault:
        key={'admission_freeze':'app_admission_freeze_seconds','freeze':'app_freeze_seconds',
             'slow':'app_slow_seconds','unknown':'app_unknown_seconds'}[fault]
        gate[key]=[1]
    if fault in (None,'loss_only'):refresh.accepted_gate(gate,tmp_path,accepted)
    else:
        with pytest.raises(ValueError):refresh.accepted_gate(gate,tmp_path,accepted)

@pytest.mark.parametrize('fault',['app_delta','unlisted_recorder_delta','unused_allowlist'])
def test_source_inventory_compatibility_is_explicit_and_exact(fault):
    first={'spacr/app.py':'a'*64,'tools/tutorials/capture_home.py':'b'*64}
    second=deepcopy(first);plan={}
    if fault=='app_delta':second['spacr/app.py']='c'*64;plan=dict(recorder_delta_paths=['spacr/app.py'],recorder_delta_reason='not permitted')
    elif fault=='unlisted_recorder_delta':second['tools/tutorials/capture_home.py']='c'*64
    else:plan=dict(recorder_delta_paths=['tools/tutorials/capture_home.py'],recorder_delta_reason='unused')
    with pytest.raises(ValueError):refresh.compatible_sources([dict(inventory=first),dict(inventory=second)],plan)

def test_only_exact_reviewed_recorder_delta_allowed():
    a={'spacr/app.py':'a'*64,'tools/tutorials/capture_home.py':'b'*64};b={**a,'tools/tutorials/capture_home.py':'c'*64}
    refresh.compatible_sources([dict(inventory=a),dict(inventory=b)],dict(
        recorder_delta_paths=['tools/tutorials/capture_home.py'],recorder_delta_reason='Reviewed capture-only correction'))

@pytest.fixture
def captured(tmp_path):
    root=tmp_path/'run';capture=root/'captures/home';capture.mkdir(parents=True)
    visual=VISUALS[7]
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(json.dumps(value).encode() if not isinstance(value,bytes) else value)
    paths={key:capture/(visual+suffix) for key,suffix in [('image','.png'),('video','.mp4'),
        ('native_video','.native.mp4'),('receipt','.capture.json'),('gate','.gate.json'),
        ('host','.host.json'),('pointer','.pointer.json')]}
    paths.update(frames=capture/'frames.json',provenance=capture/'provenance.json',
        exact=root/'exact-source.json',profile=root/'capture-profile.json',host_run=root/'host-run.json',
        summary=capture/'host_gate_summary.json')
    put(paths['image'],b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'+struct.pack('>II',3840,2160))
    put(paths['video'],b'verified video fixture');put(paths['native_video'],b'original native fixture')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=dict(sha256=sha(paths['video']),size=[3840,2160],full_decode_passed=True,audio_streams=0,
                 encoded_fps='30',duration=1,frames=[dict(pts=n) for n in range(30)])
    receipt['native']=dict(video=paths['native_video'].name,sha256=sha(paths['native_video']))
    put(paths['receipt'],receipt)
    summary=dict(passed=True,reasons=[],distinct_fps=29,app_freeze_seconds=[],capture_loss_seconds=[],file=paths['gate'].name)
    put(paths['gate'],summary);put(paths['summary'],dict(clips={visual:summary}))
    put(paths['frames'],{visual:dict(image=paths['image'].name,sha256=sha(paths['image']),clip=dict(
        video=paths['video'].name,receipt=paths['receipt'].name,sha256=sha(paths['video']),
        receipt_sha256=sha(paths['receipt']),native_video=paths['native_video'].name,
        native_sha256=sha(paths['native_video']),gate=summary))})
    put(paths['provenance'],dict(commit='fixture',module='home',app_source_modified=False,completed_capture=False))
    put(paths['exact'],dict(commit='fixture',fresh=True,private_profile=True,sha256={'spacr/app.py':'a'*64}))
    put(paths['profile'],dict(theme='dark',field='data_art_impulse_lens',**{k:False for k in
        ('show_alpha','show_alpha_features','show_alpha_species','spaceout','anonymous_reports_enabled','anonymous_report_notice_due')}))
    profile=json.loads(paths['profile'].read_text())
    profile.update(field_density=.1,field_density_shipped=.1,popup_backdrop='drift',requested_field_held_fps=60)
    put(paths['profile'],profile)
    put(paths['host_run'],dict(app_exit_code=0,recorder_rc=0,failure=None))
    put(paths['host'],{});put(paths['pointer'],[])
    selection=dict(source_scene=8,visual=visual,capture=str(capture),idle_tail_start=.5)
    def bind():selection['files']={key:dict(sha256=sha(path),bytes=path.stat().st_size) for key,path in paths.items()}
    bind()
    return selection,paths,put,bind

@pytest.mark.parametrize('fault',[None,'changed_bytes','wrong_module','foreign_commit','receipt_drift','wrong_slot','native_receipt_drift'])
def test_original_capture_bindings_and_provenance(captured,fault):
    selection,paths,put,bind=captured
    if fault=='changed_bytes':put(paths['video'],b'changed')
    elif fault=='wrong_module':value=json.loads(paths['provenance'].read_text());value['module']='mask';put(paths['provenance'],value);bind()
    elif fault=='foreign_commit':value=json.loads(paths['exact'].read_text());value['commit']='foreign';put(paths['exact'],value);bind()
    elif fault=='receipt_drift':value=json.loads(paths['receipt'].read_text());value['frames'][-1]['pts']=999;put(paths['receipt'],value);bind()
    elif fault=='native_receipt_drift':value=json.loads(paths['receipt'].read_text());value['native']['sha256']='f'*64;put(paths['receipt'],value);bind()
    elif fault=='wrong_slot':selection['source_scene']=7
    accepted=lambda gate,captures:gate['passed']
    if fault:
        with pytest.raises(ValueError):refresh.selected_capture(selection,accepted)
    else:
        selected=refresh.selected_capture(selection,accepted)
        assert selected['frames']==30 and selected['fresh_capture'] is True
        assert selected['source_run_completed'] is False


def test_normal_mixed_dispatch_is_explicit_and_other_modes_unchanged(monkeypatch):
    seen=[]
    monkeypatch.setattr(refresh,'render',lambda *args:seen.append(args) or 'clock')
    assert mix.render('stage','base',None,{},refresh.LESSON,refresh.REVISION,{'reviewed':'plan'})=='clock'
    assert len(seen)==1
    with pytest.raises(ValueError):mix.render('stage','base',None,{},'05_home',refresh.REVISION,{})
    with pytest.raises(ValueError):mix.render('stage','base',{}, {},refresh.LESSON,refresh.REVISION,{})
    with pytest.raises(ValueError):mix.render('stage','base',None,{},refresh.LESSON,'maintenance-v1',{})


@pytest.mark.parametrize('field,value',[('field_density',.2),('field_density_shipped',.2),
                                      ('popup_backdrop','none'),('requested_field_held_fps',30)])
def test_actual_requested_visual_profile_mismatch_is_rejected(captured,field,value):
    selection,paths,put,bind=captured
    profile=json.loads(paths['profile'].read_text());profile[field]=value
    put(paths['profile'],profile);bind()
    with pytest.raises(ValueError,match='profile'):
        refresh.selected_capture(selection,lambda gate,captures:gate['passed'])


def test_native_flags_are_not_integer_claims(scenario):
    scenario[1]['scenes'][7]['native_live_video']=1
    with pytest.raises(ValueError):refresh.plan_rows(*scenario)

@pytest.mark.parametrize('fault',['symlink_root','competing_native_record'])
def test_stage_does_not_escape_or_mix_source_modes(tmp_path,monkeypatch,fault):
    stage=tmp_path/'stage';baseline=tmp_path/'baseline'
    root=stage/'production'/refresh.LESSON
    root.parent.mkdir(parents=True)
    if fault=='symlink_root':
        outside=tmp_path/'outside';outside.mkdir();root.symlink_to(outside,target_is_directory=True)
    else:
        root.mkdir();(root/'published-native-reuse.json').write_text('{}')
    monkeypatch.setattr(refresh,'_published_inputs',lambda *args:(baseline,{}, {}, {}))
    with pytest.raises(ValueError,match='exclusive contained'):
        refresh.inputs(stage,baseline,{}, {})


def final_scenario(scenario):
    english,original,tracks,replacements,fences=deepcopy(scenario)
    original['scenes'][-1]['duration']=26.8
    original['scenes'][-1]['speech_end']=original['scenes'][-1]['scene_end']=39+26.8
    for track in tracks:
        track['scenes'][-1]=dict(speech_start=200,speech_end=226.4,duration=26.8,
                                sentences=[dict(audible_end=225.8)])
    tracks[17]['scenes'][-1]['sentences'][-1]['audible_end']=226.075
    return english,original,tracks,replacements,fences


def test_final_hold_covers_worst_actual_audible_offset_without_trimming_source(scenario):
    values=final_scenario(scenario)
    row=refresh.plan_rows(*values)[-1]
    assert row['source_keep_frames']==[[0,804]] and row['kept_frames']==804
    assert row['final_frame_hold_frames']==9 and row['duration']==27.1
    assert row['source_kind']=='published_interval_with_final_result_hold'
    assert row['original_source']==values[1]['scenes'][-1]
    assert row['fresh_capture'] is False and row['native_live_video'] is False
    assert row['duration']-(226.075-200)>=1


def test_final_hold_does_not_append_frames_when_existing_quiet_tail_suffices(scenario):
    rows=refresh.plan_rows(*scenario)
    assert rows[-1]['source_keep_frames']==[[0,90]]
    assert rows[-1]['final_frame_hold_frames']==0 and rows[-1]['duration']==3
    assert rows[-1]['source_kind']=='unchanged_published_interval'


@pytest.mark.parametrize('fault',['missing_sentences','empty_sentences','missing_audible',
                                 'nan','infinity','before_speech','after_speech','boolean'])
def test_final_hold_rejects_unvalidated_audible_timing(scenario,fault):
    values=final_scenario(scenario)
    spoken=values[2][3]['scenes'][-1]
    if fault=='missing_sentences':spoken.pop('sentences')
    elif fault=='empty_sentences':spoken['sentences']=[]
    elif fault=='missing_audible':spoken['sentences']=[{}]
    else:
        spoken['sentences'][-1]['audible_end']={'nan':float('nan'),'infinity':float('inf'),
            'before_speech':199.9,'after_speech':226.5,'boolean':True}[fault]
    with pytest.raises(ValueError,match='last audible'):
        refresh.plan_rows(*values)


def test_explicit_mixed_clock_preserves_reference_and_native_interval_mapping(scenario,tmp_path,monkeypatch):
    import idle_tail_policy
    english,original,tracks,replacements,fences=deepcopy(scenario)
    stage=tmp_path/'stage';baseline=tmp_path/'baseline';repo=tmp_path/'repo'
    root=stage/'production'/refresh.LESSON
    refresh.write(root/'lesson.en.json',english)
    refresh.write(repo/'tools/tutorials/lessons'/f'{refresh.LESSON}.json',english)
    refresh.write(root/refresh.AUDIO_RECORD,dict(baseline_root=str(baseline)))
    bindings=dict(master=dict(path='original.mp4',sha256='a'*64),
                  timing=dict(path='clock.json',sha256='b'*64))
    monkeypatch.setattr(refresh,'REPO',repo)
    monkeypatch.setattr(refresh,'_published_inputs',lambda *args:
        (baseline,original,bindings,dict(baseline_root=str(baseline))))
    monkeypatch.setattr(refresh,'checked_published_narration',lambda *args:(tracks,[]))
    monkeypatch.setattr(refresh,'rule',lambda plan:None)
    monkeypatch.setattr(refresh,'selected_capture',lambda selected,accepted:
        next(r for r in replacements if r['selection']['source_scene']==selected['source_scene']))
    monkeypatch.setattr(idle_tail_policy,'bind_policy',lambda *args:[dict(native_idle_tails=dict(
        scenes=[dict(visual=k,idle_tail_start=v) for k,v in fences.items()]))])
    plan=dict(schema=1,lesson=refresh.LESSON,revision=refresh.REVISION,
              english_sha256=refresh.identity(english),master_sha256='a'*64,
              visual_timing_sha256='b'*64,replacements=[r['selection'] for r in replacements])
    timing=refresh.inputs(stage,baseline,plan,{})[-1]
    assert timing['preserve_visual_intervals'] is True
    assert timing['playback_speed']==1 and timing['looped_or_stretched'] is False
    assert timing['published_native_mix']['fresh_capture'] is False
    assert [r['scene'] for r in timing['scenes'] if r['fresh_capture']]==[8,9]


def cut_scenario(scenario):
    values=deepcopy(scenario)
    new=values[3][1]
    new['frames']=120
    new['selection'].update(keep_frames=[[0,15],[60,90]],confirmed_idle_frames=[[90,120]],
        omitted_setup_frames=[[15,60]],omitted_setup_reason='Reviewed intervening menu setup',
        final_result_frame=89,idle_tail_start=1.4)
    return values


def test_reviewed_cut_uses_only_kept_native_frames_at_1x_and_retains_raw_proof(scenario):
    values=cut_scenario(scenario)
    row=refresh.plan_rows(*values)[8]
    assert row['source_start_frame']==0 and row['source_keep_frames']==[[0,15],[60,90]]
    assert row['kept_frames']==45 and row['final_frame_hold_frames']==15 and row['duration']==2
    assert row['edited'] is True and row['source_kind']=='reviewed_native_companion_edit'
    assert row['native_companion']['frames']==120
    assert row['native_companion']['selection']==values[3][1]['selection']


@pytest.mark.parametrize('fault',['overlap','unaccounted','outside','noninteger','unordered',
    'wrong_final','missing_idle','missing_reason','blank_reason','keep_too_long',
    'fence_after_kept','unknown_field'])
def test_reviewed_cut_partition_and_shortest_voice_safety(scenario,fault):
    values=cut_scenario(scenario);selected=values[3][1]['selection']
    if fault=='overlap':selected['omitted_setup_frames']=[[14,60]]
    elif fault=='unaccounted':selected['omitted_setup_frames']=[[16,60]]
    elif fault=='outside':selected['confirmed_idle_frames']=[[90,121]]
    elif fault=='noninteger':selected['keep_frames'][0][0]=False
    elif fault=='unordered':selected['keep_frames'].reverse()
    elif fault=='wrong_final':selected['final_result_frame']=90
    elif fault=='missing_idle':selected.pop('confirmed_idle_frames')
    elif fault=='missing_reason':selected.pop('omitted_setup_reason')
    elif fault=='blank_reason':selected['omitted_setup_reason']=' '
    elif fault=='keep_too_long':selected.update(keep_frames=[[0,15],[60,91]],confirmed_idle_frames=[[91,120]],final_result_frame=90)
    elif fault=='fence_after_kept':selected.update(keep_frames=[[0,15],[60,75]],confirmed_idle_frames=[[75,120]],final_result_frame=74,idle_tail_start=1.1)
    else:selected['keep_framez']=selected['keep_frames']
    with pytest.raises(ValueError):refresh.plan_rows(*values)


def test_no_cut_selection_remains_full_native_capture(scenario):
    row=refresh.plan_rows(*scenario)[8]
    assert row['source_keep_frames']==[[0,45]] and row['edited'] is False
    assert row['native_companion']['frames']==45


def test_actual_bound_capture_cut_fence_is_in_edited_elapsed_time(captured):
    selected,_,_,_=captured
    selected.update(keep_frames=[[0,6],[15,24]],confirmed_idle_frames=[[24,30]],
                    omitted_setup_frames=[[6,15]],omitted_setup_reason='Reviewed menu approach',
                    final_result_frame=23,idle_tail_start=.4)
    result=refresh.selected_capture(selected,lambda gate,where:gate['passed'])
    assert result['frames']==30 and result['selection']['keep_frames']==[[0,6],[15,24]]
    selected['idle_tail_start']=.6
    with pytest.raises(ValueError,match='fence'):
        refresh.selected_capture(selected,lambda gate,where:gate['passed'])
