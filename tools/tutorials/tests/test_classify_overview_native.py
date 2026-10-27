"""Bounded offline row, provenance and owning-dispatch contracts; no media run."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

import pytest
import published_classify_overview_native as overview
import published_native_mix as mix


@pytest.fixture
def rows_input():
    english=dict(id=overview.LESSON,scenes=[dict(visual=v,narration='Current '+v) for v in overview.VISUALS])
    tracks=[dict(scenes=[dict(speech_start=0,speech_end=6,duration=6.52,
        sentences=[dict(audible_end=5.82)]) for _ in overview.VISUALS]) for _ in range(50)]
    captures=[dict(selection=dict(source_scene=i,visual=overview.VISUALS[i-1],idle_tail_start=1),
        frames=60,keep=[[0,60]],edited=False,video='/original-'+str(i)+'.mp4')
        for i in range(1,13) if i!=6]
    cv=dict(source_start_frame=7060,source_keep_frames=[[0,135]],kept_frames=135,
        final_frame_hold_frames=61,duration=196/30,native_live_video=True,published_component={})
    return english,tracks,captures,dict(kind='published_classify_cv_interval'),cv


def test_all_twelve_rows_keep_eleven_actual_clips_and_one_published_cv_motion(rows_input):
    before=deepcopy(rows_input);rows=overview.plan_rows(*rows_input)
    assert len(rows)==12 and sum(row['fresh_capture'] for row in rows)==11
    assert rows[5]['source_keep_frames']==[[0,135]] and rows[5]['source_start_frame']==7060
    assert rows[5]['source_kind']=='published_native_cv_interval'
    assert all(row['source_keep_frames']==[[0,60]] for i,row in enumerate(rows) if i!=5)
    assert all(left['scene_end']==right['speech_start'] for left,right in zip(rows,rows[1:]))
    assert rows[-1]['duration']>=6.82
    assert rows_input==before


@pytest.mark.parametrize('fault',['missing_voice','missing_role','wrong_order','late_fence','edited_overflow','final_audible_missing'])
def test_overview_rows_reject_incomplete_scope_or_unsafe_actions(rows_input,fault):
    english,tracks,captures,component,cv=deepcopy(rows_input)
    if fault=='missing_voice':tracks.pop()
    elif fault=='missing_role':captures.pop()
    elif fault=='wrong_order':english['scenes'].reverse()
    elif fault=='late_fence':captures[0]['selection']['idle_tail_start']=6.1
    elif fault=='edited_overflow':captures[0].update(keep=[[0,210]],edited=True)
    else:tracks[0]['scenes'][-1].pop('sentences')
    with pytest.raises(ValueError):overview.plan_rows(english,tracks,captures,component,cv)


@pytest.mark.parametrize('fault',[None,'different_main_base','different_cv_base','stale_cv_identity'])
def test_current_and_published_cv_source_identity_is_exact(tmp_path,fault):
    commit='c'*40;inventory={'spacr/app.py':'a'*64}
    exact=tmp_path/'exact-source.json';exact.write_text(json.dumps(dict(commit=commit,sha256=inventory)))
    entry=dict(path=str(exact),sha256=hashlib.sha256(exact.read_bytes()).hexdigest(),bytes=exact.stat().st_size)
    component=dict(source_capture_lineage=dict(source_files=[entry]))
    captures=[dict(source_commit=commit,inventory=deepcopy(inventory)) for _ in range(11)]
    if fault=='different_main_base':captures[0]['inventory']['spacr/app.py']='b'*64
    elif fault=='different_cv_base':
        exact.write_text(json.dumps(dict(commit='d'*40,sha256=inventory)))
        entry.update(sha256=hashlib.sha256(exact.read_bytes()).hexdigest(),bytes=exact.stat().st_size)
    elif fault=='stale_cv_identity':exact.write_text('{}')
    if fault:
        with pytest.raises(ValueError):overview.same_source(captures,component,dict(source_commit=commit))
    else:overview.same_source(captures,component,dict(source_commit=commit))


@pytest.mark.parametrize('fault',[None,'scientific','batch'])
def test_compatible_physical_retakes_require_equal_workflow_science(fault):
    captures=[]
    for role in ('main','ml'):
        for take in ('first','retry'):
            captures.append(dict(selection=dict(visual=role+'_example',capture='/'+role+'/'+take),
                source_commit='c'*40,inventory={'spacr/app.py':'a'*64},
                scientific=dict(accepted=True,inputs={'dataset':'same'},results={'count':8}),
                batch=dict(accepted=True,metrics={'accuracy':.8}) if role=='ml' else None))
    if fault=='scientific':captures[1]['scientific']['results']['count']=9
    elif fault=='batch':captures[-1]['batch']['metrics']['accuracy']=.9
    if fault:
        with pytest.raises(ValueError,match='different scientific/batch'):
            overview.compatible_takes(captures)
    else:
        overview.compatible_takes(captures)
        assert captures[0]['selection']['capture']!=captures[1]['selection']['capture']


@pytest.mark.parametrize('fault',[None,'undeclared','application','inventory','commit','unused'])
def test_explicit_recorder_only_delta_preserves_original_cv_identity(tmp_path,fault):
    old='c'*40;new='d'*40;driver='tools/tutorials/capture_classify_overview.py'
    inventory={driver:'a'*64,'spacr/app.py':'b'*64}
    exact=tmp_path/'exact-source.json';exact.write_text(json.dumps(dict(commit=old,sha256=inventory)))
    entry=dict(path=str(exact),sha256=hashlib.sha256(exact.read_bytes()).hexdigest(),bytes=exact.stat().st_size)
    component=dict(source_capture_lineage=dict(source_files=[entry]))
    captures=[dict(source_commit=old,inventory=deepcopy(inventory)),
              dict(source_commit=new,inventory={**inventory,driver:'e'*64})]
    plan=dict(source_commit=old,source_commits=[old,new],recorder_delta_paths=[driver],
        recorder_delta_reason='Reviewed recorder tooltip expectation only')
    if fault=='undeclared':plan.pop('recorder_delta_paths')
    elif fault=='application':
        captures[1]['inventory']['spacr/app.py']='f'*64
        plan['recorder_delta_paths'].append('spacr/app.py')
    elif fault=='inventory':captures[1]['inventory']['extra.py']='f'*64
    elif fault=='commit':plan['source_commits']=[old]
    elif fault=='unused':plan['recorder_delta_paths'].append('tools/tutorials/capture_other.py')
    if fault:
        with pytest.raises(ValueError):overview.same_source(captures,component,plan)
    else:
        proof=overview.same_source(captures,component,plan)
        assert proof==dict(source_commits=[old,new],recorder_delta_paths=[driver],
            recorder_delta_reason=plan['recorder_delta_reason'])
        assert json.loads(exact.read_text())['sha256']==inventory


@pytest.mark.parametrize('fault',[None,'default','bad_second','freeze','cursor','unused_scope','wrong_slot'])
def test_reviewed_ml_result_cut_admits_only_clean_whole_source_seconds(fault):
    selection=dict(source_scene=5,source_visual='25_ml_figure_02',reviewed_result_capture_loss=True,
        keep_frames=[[540,600]],omitted_setup_frames=[[0,540]],omitted_setup_reason='Reviewed completed result only',
        confirmed_idle_frames=[[600,975]],final_result_frame=599)
    gate=dict(passed=False,fps=30,frames=975,limits=dict(min_fps=27.9,min_cursor_found=.9),
        capture_loss_seconds=list(range(18))+[21,26,27,28,29,30,31],
        reasons=['choppy (capture loss): 17.5 distinct fps < 27.9'],
        distinct_per_second=[0]*18+[28,29]+[0]*12,
        cursor=dict(found_fraction=1,jump_count=0,jumps=[]),
        top_bar=dict(hidden=True,visible_in=[],sampled_frames=98),
        **{key:[] for key in ('app_admission_freeze_seconds','app_freeze_seconds','app_slow_seconds',
            'app_unknown_seconds','app_gui_sampler_gaps_s','app_stall_windows_s','adapter_bypasses')})
    if fault=='default':selection.pop('reviewed_result_capture_loss')
    elif fault=='bad_second':gate['distinct_per_second'][19]=27
    elif fault=='freeze':gate['app_freeze_seconds']=[18]
    elif fault=='cursor':gate['cursor']['jump_count']=1
    elif fault=='unused_scope':gate['passed']=True
    elif fault=='wrong_slot':selection['source_scene']=4
    if fault:
        with pytest.raises(ValueError):overview.reviewed_result_quality(selection,gate,[[540,600]],975)
    else:
        proof=overview.reviewed_result_quality(selection,gate,[[540,600]],975)
        assert proof['raw_gate_passed'] is False and proof['kept_distinct_per_second']==[28,29]
        assert gate['passed'] is False and len(gate['capture_loss_seconds'])==25


def test_normal_mix_dispatch_only_accepts_explicit_overview_revision(monkeypatch):
    called=[];monkeypatch.setattr(overview,'render',lambda *args:called.append(args) or 'checked-clock')
    assert mix.render('stage','published',None,{},overview.LESSON,overview.REVISION,overview_plan={})=='checked-clock'
    assert len(called)==1
    with pytest.raises(ValueError):mix.render('stage','published',None,{},'05_home',overview.REVISION,overview_plan={})
    with pytest.raises(ValueError):mix.render('stage','published',None,{},overview.LESSON,overview_plan={})


@pytest.fixture
def captured_main(tmp_path):
    run=tmp_path/'run';capture=run/'captures/classify_merged';capture.mkdir(parents=True)
    raw='05_classifier_evaluation_tooltip'
    paths={key:capture/(raw+suffix) for key,suffix in [('image','.png'),('video','.mp4'),
        ('native_video','.native.mp4'),('receipt','.capture.json'),('gate','.gate.json'),
        ('host','.host.json'),('pointer','.pointer.json')]}
    paths.update(frames=capture/'frames.json',provenance=capture/'provenance.json',
        scientific=capture/'scientific_acceptance.json',exact=run/'exact-source.json',
        profile=run/'capture-profile.json',host_run=run/'host-run.json',summary=capture/'host_gate_summary.json')
    def put(path,value):path.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    put(paths['image'],b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'+struct.pack('>II',3840,2160))
    put(paths['video'],b'synthetic video');put(paths['native_video'],b'synthetic native container')
    receipt=dict(sha256=digest(paths['video']),size=[3840,2160],audio_streams=0,full_decode_passed=True,
        duration=1,encoded_fps='30',frames=[dict(pts=i) for i in range(30)],
        native=dict(video=paths['native_video'].name,sha256=digest(paths['native_video'])))
    put(paths['receipt'],receipt)
    gate=dict(passed=True,reasons=[],distinct_fps=29,app_freeze_seconds=[],capture_loss_seconds=[],file=paths['gate'].name)
    put(paths['gate'],gate);put(paths['summary'],dict(clips={raw:gate}))
    put(paths['frames'],{raw:dict(image=paths['image'].name,sha256=digest(paths['image']),clip=dict(
        video=paths['video'].name,sha256=digest(paths['video']),receipt=paths['receipt'].name,
        receipt_sha256=digest(paths['receipt']),native_video=paths['native_video'].name,
        native_sha256=digest(paths['native_video']),gate=gate))})
    put(paths['exact'],dict(commit='c'*40,fresh=True,private_profile=True,sha256={'spacr/app.py':'a'*64}))
    put(paths['provenance'],dict(commit='c'*40,module='classify_merged',app_source_modified=False,completed_capture=False))
    put(paths['host_run'],dict(app_exit_code=0,recorder_rc=0,failure=None))
    put(paths['host'],{});put(paths['pointer'],[])
    put(paths['profile'],dict(theme='dark',field='data_art_impulse_lens',field_density=.1,
        field_density_shipped=.1,popup_backdrop='drift',requested_field_held_fps=60,
        **{key:False for key in ('show_alpha','show_alpha_features','show_alpha_species','spaceout','anonymous_reports_enabled','anonymous_report_notice_due')}))
    folds=[dict(key=key,tooltip=key,displayed_tooltip=key) for key in
        ['classifier_evaluation','explain_cv','activation','train_compare','feature_explorer']]
    put(paths['scientific'],dict(accepted=True,model_started=False,family_restored='cv',
        nested_feature_explorer_opened=True,folds=folds))
    selection=dict(source_scene=7,visual=overview.VISUALS[6],source_visual=raw,capture=str(capture),idle_tail_start=.5)
    def bind():selection['files']={key:dict(sha256=digest(path),bytes=path.stat().st_size) for key,path in paths.items()}
    bind();return selection,paths,put,bind


@pytest.mark.parametrize('fault',[None,'changed_bytes','raw_freeze','wrong_profile','navigation_model','wrong_role'])
def test_actual_capture_binding_admission_and_navigation_contract(captured_main,fault):
    selection,paths,put,bind=captured_main
    if fault=='changed_bytes':put(paths['video'],b'changed')
    elif fault=='raw_freeze':
        gate=json.loads(paths['gate'].read_text());gate['app_freeze_seconds']=[0]
        put(paths['gate'],gate);put(paths['summary'],dict(clips={selection['source_visual']:gate}))
        frames=json.loads(paths['frames'].read_text());frames[selection['source_visual']]['clip']['gate']=gate;put(paths['frames'],frames);bind()
    elif fault=='wrong_profile':
        profile=json.loads(paths['profile'].read_text());profile['requested_field_held_fps']=20;put(paths['profile'],profile);bind()
    elif fault=='navigation_model':
        scientific=json.loads(paths['scientific'].read_text());scientific['model_started']=True;put(paths['scientific'],scientific);bind()
    elif fault=='wrong_role':selection['source_scene']=6
    if fault:
        with pytest.raises(ValueError):overview.selected_capture(selection,lambda *args:True)
    else:
        result=overview.selected_capture(selection,lambda *args:True)
        assert result['frames']==30 and result['keep']==[[0,30]]


def test_renderer_joins_only_published_cv_master_and_actual_eleven_capture_sources(tmp_path,monkeypatch,rows_input):
    import render_visual_master
    rows=overview.plan_rows(*rows_input)
    root=tmp_path/'stage/production/41_classify';root.mkdir(parents=True)
    baseline=tmp_path/'published'
    component=dict(published_master=dict(path='media_host/10_classify_cv/video/10_classify_cv_silent.mp4'))
    timing=dict(lesson=overview.LESSON,scenes=rows,published_native_mix=dict(published_cv=component))
    spec=dict(scenes=[dict(published_source=row) for row in rows])
    monkeypatch.setattr(overview,'inputs',lambda *a:(root,baseline,spec,deepcopy(timing)))
    calls=[]
    def encode(source,row,part,*,terminal_seekable):
        calls.append((str(source),row['scene'],terminal_seekable,deepcopy(row['source_keep_frames'])))
        part.write_bytes(b'encoded fixture')
    def concat(parts,target,listing):
        assert len(parts)==12 and all(part.is_file() for part in parts)
        target.write_bytes(b'joined fixture')
    monkeypatch.setattr(overview,'encode_scene',encode)
    monkeypatch.setattr(render_visual_master,'concat',concat)
    overview.render(tmp_path/'stage',baseline,{}, {})
    assert len(calls)==12
    assert calls[5][0]==str(baseline/component['published_master']['path'])
    assert calls[5][3]==[[0,135]]
    assert all(source.startswith('/original-') for source,scene,_,_ in calls if scene!=6)
    assert [scene for _,scene,seek,_ in calls if seek]==[12]
    policy=json.loads((root/'mixed-idle-tail-policy.json').read_text())
    assert len(policy['lessons'][0]['scenes'])==12
    assert policy['lessons'][0]['scenes'][5]['idle_tail_start']==4.5
