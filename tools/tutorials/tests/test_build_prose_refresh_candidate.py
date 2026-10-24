"""Synthetic component joins and real candidate inventory/policy guards; no codecs."""
from copy import deepcopy
from pathlib import Path
import json,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import build_prose_refresh_candidate as b
from append_staged_lessons import JS_PREFIX,JS_SUFFIX,parse_javascript
from check_completed_matrix import reconcile_audio


def put(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value))
 return path

@pytest.fixture
def candidate_inputs(tmp_path,monkeypatch):
 lesson='71_investigate_hit';baseline=tmp_path/'published';stage=tmp_path/'stage';root=stage/'production'/lesson
 voices=deepcopy(b.engine.POLICIES[lesson]['voices']);keys=[(l,v) for l,vs in voices.items() for v in vs]
 old=dict(id=lesson,title='Hit',description='Original',objectives=['Inspect'],prerequisite='Inputs',app_key='hit',status='ready',narration_voices=voices,scenes=[dict(visual='result',narration='Original sentence.')])
 new=deepcopy(old);new['scenes'][0]['narration']='Approved new sentence.'
 lessons=[old]+[dict(id=f'other_{i}',title=str(i),status='ready',scenes=[dict(visual='still',narration='Keep')]) for i in range(83)]
 catalogs={}
 for name in b.CATALOGS:
  before=dict(schema=1,lessons=deepcopy(lessons));after=deepcopy(before);after['lessons'][0]=deepcopy(new)
  put(baseline/'web/catalog'/name,before);put(stage/'catalog'/name,after);catalogs[name]=after
 runtime=dict(schema=1,lessons=deepcopy(lessons));runtime['lessons'][0].update(silent=f'{lesson}/video/{lesson}_silent.mp4',web=f'{lesson}/web/{lesson}_silent.mp4')
 p=baseline/'web/lesson_catalog.js';p.write_text(JS_PREFIX+json.dumps(runtime)+JS_SUFFIX)
 (baseline/'web/app_v2.js').write_text('unchanged player')
 for path in (f'media_host/{lesson}/video/{lesson}_silent.mp4',f'media_host/{lesson}/web/{lesson}_silent.mp4',f'web/production/{lesson}/poster.jpg'):
  p=baseline/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'original picture')
 tracks=[];original_tracks=[]
 for l,v in keys:
  prefix=f'media_host/{lesson}/audio/{l}/{v}'
  for ending in ('.m4a','.json'):
   p=baseline/(prefix+ending);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}' if ending=='.json' else 'original AAC')
   q=root/f'audio/{l}/{v}{ending}';q.parent.mkdir(parents=True,exist_ok=True);q.write_text('{}' if ending=='.json' else 'new AAC')
  tracks.append(dict(language=l,voice=v,audio_sha256=b.digest(root/f'audio/{l}/{v}.m4a'),timing_sha256=b.digest(root/f'audio/{l}/{v}.json'),original_audio_sha256=b.digest(baseline/(prefix+'.m4a')),provenance_kind=b.engine.KIND,mixed_record_sha256='set later'))
  original_tracks.append(dict(language=l,voice=v,mode='source_coordinate_splice',decoded_true_peak_dbfs=-2.))
 records=[dict(path=p.relative_to(baseline).as_posix(),bytes=p.stat().st_size,sha256=b.digest(p)) for top in ('web','media_host') for p in (baseline/top).rglob('*') if p.is_file()]
 manifest=dict(routes=84,ready_lessons=84,coming_soon=[],catalog_languages=len(b.CATALOGS),files=records,ceiling_bytes=10000000)
 put(baseline/'release-manifest.json',manifest);put(baseline/'publication-receipt.json',dict(commit='actual published fixture'))
 put(baseline/'checks/candidate-browser-checks.json',dict(passed=True,manifest_sha256=b.digest(baseline/'release-manifest.json'),ready_playback_cases=[dict(lesson=l['id'],passed=True) for l in lessons],placeholder_cases=[]))
 put(baseline/'checks/placeholder-mutation-checks.json',dict(passed=True,baseline_before_and_after_passed=True,player_sha256=b.digest(baseline/'web/app_v2.js'),mutations=[dict(guard=g,observed_red=True) for g in ('availability guard','completion guard')]))
 clock=f'media_host/{lesson}/audio/en/af_heart.json'
 source=dict(lesson=lesson,selected_lessons=[lesson],baseline_root=str(baseline),baseline_manifest_sha256=b.digest(baseline/'release-manifest.json'),publication_receipt_sha256=b.digest(baseline/'publication-receipt.json'),voices=voices,native_adapter_required=False,original_visual_clock=dict(kind='legacy',path=clock,sha256=b.digest(baseline/clock)))
 put(root/b.engine.RECORD,dict(schema=1,kind=b.engine.KIND,source=source,tracks=original_tracks))
 for t in tracks:t['mixed_record_sha256']=b.digest(root/b.engine.RECORD)
 monkeypatch.setattr(b,'require_published_script_baseline',lambda *args:None)
 monkeypatch.setattr(b.engine,'sources',lambda *args:({lesson:deepcopy(source)},deepcopy(catalogs)))
 calls=[]
 def checked(*args,decode=False):calls.append(decode);return deepcopy(voices),deepcopy(tracks)
 monkeypatch.setattr(b.engine,'checked_tracks',checked)
 return dict(lesson=lesson,baseline=baseline,stage=stage,root=root,source=source,catalogs=catalogs,runtime=runtime,old=old,new=new,calls=calls,tracks=tracks)


def test_one_final_codec_gate_real_inventory_preserves83_and_explicit_old_clock(candidate_inputs):
 f=candidate_inputs;target=b.build(f['stage'],f['baseline'],f['lesson']);assert f['calls']==[False,True]
 before=b.read(f['baseline']/'release-manifest.json');after=b.read(target/'release-manifest.json')
 old={r['path']:r for r in before['files']};new={r['path']:r for r in after['files']}
 changed={r['path'] for r in before['files'] if r['path'].startswith('web/catalog/') or r['path'].startswith(f'media_host/{f["lesson"]}/audio/')}|{'web/lesson_catalog.js'}
 for path in set(old)-changed:assert new[path]==old[path] and (target/path).read_bytes()==(f['baseline']/path).read_bytes()
 runtime=parse_javascript((target/'web/lesson_catalog.js').read_text());assert runtime['lessons'][1:]==f['runtime']['lessons'][1:]
 clock='media_host/'+runtime['lessons'][0]['visual_timings'];assert (target/clock).read_bytes()==(f['baseline']/f['source']['original_visual_clock']['path']).read_bytes()
 report=b.read(target/'checks/prose-audio-codec-checks.json');expected={(t['language'],t['voice']) for t in f['tracks']};hashes={(t['language'],t['voice']):t['audio_sha256'] for t in f['tracks']}
 assert set(reconcile_audio([report],f['lesson'],expected,hashes))==expected
 assert report['current_whole_track_tts_claimed'] is False and report['human_listening_review'] is False
 assert (target/'checks/placeholder-mutation-checks.json').read_bytes()==(f['baseline']/'checks/placeholder-mutation-checks.json').read_bytes()
 b.validate(target,include_hosted_media=True)


@pytest.mark.parametrize('failure',['foreign_baseline','visual_finish','producer_peak','native_flags','clock_conflict'])
def test_cheap_failure_precedes_any_final_aac(candidate_inputs,monkeypatch,failure):
 f=candidate_inputs
 if failure=='foreign_baseline':f['source']['baseline_root']=str(f['stage'])
 if failure=='visual_finish':
  def reject(*args):raise ValueError('poster drift')
  monkeypatch.setattr(b,'checked_visual',reject)
 if failure=='producer_peak':
  r=b.read(f['root']/b.engine.RECORD);r['tracks'][0]['decoded_true_peak_dbfs']=None;put(f['root']/b.engine.RECORD,r)
 if failure=='native_flags':
  f['source']['native_adapter_required']=True
  r=b.read(f['root']/b.engine.RECORD);r['source']=deepcopy(f['source']);put(f['root']/b.engine.RECORD,r)
 if failure=='clock_conflict':
  relative=f'media_host/{f["lesson"]}/video/original-visual-timings.json';path=put(f['baseline']/relative,dict(wrong='picture clock'))
  manifest=b.read(f['baseline']/'release-manifest.json');manifest['files'].append(dict(path=relative,bytes=path.stat().st_size,sha256=b.digest(path)));put(f['baseline']/'release-manifest.json',manifest)
  f['source']['baseline_manifest_sha256']=b.digest(f['baseline']/'release-manifest.json')
  r=b.read(f['root']/b.engine.RECORD);r['source']=deepcopy(f['source']);put(f['root']/b.engine.RECORD,r)
  checks=b.read(f['baseline']/'checks/candidate-browser-checks.json');checks['manifest_sha256']=f['source']['baseline_manifest_sha256'];put(f['baseline']/'checks/candidate-browser-checks.json',checks)
 with pytest.raises(ValueError):b.build(f['stage'],f['baseline'],f['lesson'])
 assert True not in f['calls'] and not list(f['stage'].glob('release-candidate-*'))


@pytest.mark.parametrize('field',['other83','global_metadata','selected_review'])
def test_reviewed_catalog_scope_rejects_extra_drift(candidate_inputs,field):
 f=candidate_inputs;before=b.read(f['baseline']/'web/catalog/lessons_en.json');after=deepcopy(f['catalogs']['lessons_en.json'])
 if field=='other83':after['lessons'][3]['title']='changed'
 if field=='global_metadata':after['schema']=2
 if field=='selected_review':after['lessons'][0]['scenes'][0]['narration']='Unreviewed'
 with pytest.raises(ValueError):b.catalog_scope(before,after,f['lesson'],f['new'])


@pytest.mark.parametrize('field',['baseline','receipt','clock'])
def test_picture_audio_join_requires_identical_published_source(candidate_inputs,field):
 f=candidate_inputs;s=f['source'];v=dict(lesson=f['lesson'],baseline_root=s['baseline_root'],manifest_sha256=s['baseline_manifest_sha256'],publication_receipt_sha256=s['publication_receipt_sha256'],original_clock={k:s['original_visual_clock'][k] for k in ('path','sha256')})
 b.joined_picture(s,v,f['lesson'])
 if field=='baseline':v['baseline_root']=str(f['stage'])
 if field=='receipt':v['publication_receipt_sha256']='wrong'
 if field=='clock':v['original_clock']['sha256']='wrong'
 with pytest.raises(ValueError):b.joined_picture(s,v,f['lesson'])


def test_original_picture_clock_conflict_rejected(candidate_inputs):
 f=candidate_inputs;source=f['source'];dedicated=f'media_host/{f["lesson"]}/video/original-visual-timings.json'
 with pytest.raises(ValueError,match='clock conflict'):b.runtime_projection(f['runtime'],f['old'],f['new'],f['lesson'],source,{source['original_visual_clock']['path']},{dedicated:{'sha256':'wrong'}},None)


def test_candidate_union_rejects_unselected_bytes_or_deletion():
 old={'selected':{'sha256':'a'},'other':{'sha256':'b'}}
 with pytest.raises(ValueError,match='Unselected'):b.require_union(old,{'selected':{'sha256':'new'},'other':{'sha256':'tampered'}},{'selected'},'selected')
 with pytest.raises(ValueError,match='inventory'):b.require_union(old,{'selected':{'sha256':'new'}},{'selected'},'selected')


@pytest.mark.parametrize('drift',['late_fence','canonical_policy'])
def test_native_fence_fit_and_current_review_binding_fail_closed(tmp_path,drift):
 lesson='20_cellpose_masks';clock=dict(kind='native_visual_timing',lesson=lesson,native_live_video=True,coverage=dict(accepted=True),scenes=[dict(visual='result',speech_start=0.,scene_end=3.)]);path=put(tmp_path/'clock.json',clock)
 master=dict(path=f'media_host/{lesson}/video/{lesson}_silent.mp4',bytes=10,sha256='a'*64);old=dict(lesson=lesson,master_sha256='a'*64,visual_timing_sha256='b'*64,scenes=[dict(visual='result',source_start=0.,source_end=3.,idle_tail_start=1.5)])
 rebound=deepcopy(old);rebound['visual_timing_sha256']=b.digest(path);generated=dict(schema=1,lessons=[rebound]);canonical=[deepcopy(rebound)]
 source=dict(original_visual_clock=dict(path=f'media_host/{lesson}/video/native-live-timings.json'));tracks=[dict(scenes=[dict(speech_start=0.,speech_end=2.)]) for _ in range(50)]
 assert b.checked_native_policy(lesson,dict(native_idle_tails=dict(schema=1,**old)),source,clock,path,master,tracks,generated,canonical)['schema']==1
 if drift=='late_fence':tracks[-1]['scenes'][0]['speech_end']=1.4
 else:canonical[0]['scenes'][0]['idle_tail_start']=1.
 with pytest.raises(ValueError):b.checked_native_policy(lesson,dict(native_idle_tails=dict(schema=1,**old)),source,clock,path,master,tracks,generated,canonical)


def approved80_inputs(tmp_path):
 lesson='80_image_analysis_pathways';approved=deepcopy(b.APPROVED80_SOURCE_POLICY)
 original=dict(kind='native_visual_timing',lesson=lesson,native_live_video=True,coverage=dict(accepted=True),scenes=[dict(scene=i,visual=f['visual'],speech_start=f['source_start'],scene_end=f['source_end'],native_live_video=True) for i,f in enumerate(approved['scenes'],1)])
 current=deepcopy(original);current['mixed_current_narration']='new source-bound timing'
 path=put(tmp_path/'current-native.json',current)
 master=dict(path=f'media_host/{lesson}/video/{lesson}_silent.mp4',sha256=approved['master_sha256'],bytes=121528128)
 timing=dict(path=f'media_host/{lesson}/video/native-live-timings.json',sha256=approved['visual_timing_sha256'],bytes=27172)
 source=dict(original_visual_clock=dict(kind='native_visual_timing',path=timing['path'],sha256=timing['sha256'],published_bindings=dict(master=master,timing=timing)))
 rebound=deepcopy(approved);rebound['visual_timing_sha256']=b.digest(path)
 tracks=[dict(scenes=[dict(speech_start=0.,speech_end=1.) for _ in range(12)]) for _ in range(50)]
 return lesson,source,original,current,path,master,tracks,dict(schema=1,lessons=[rebound]),[deepcopy(rebound)]


def test_80_accepts_only_the_reviewed12_zero_fences_rebound_to_current_timing(tmp_path):
 lesson,source,original,current,path,master,tracks,generated,canonical=approved80_inputs(tmp_path)
 bound=b.checked_native_policy(lesson,{},source,current,path,master,tracks,generated,canonical,original_clock=original)
 assert bound==dict(schema=1,**canonical[0]) and len(bound['scenes'])==12
 assert all(f['idle_tail_start']==0 for f in bound['scenes'])
 assert [(f['visual'],f['source_start'],f['source_end']) for f in bound['scenes']]==[(s['visual'],s['speech_start'],s['scene_end']) for s in original['scenes']]


@pytest.mark.parametrize('drift',['source','nonzero','missing_scene','missing_policy'])
def test_80_unreviewed_source_or_incomplete_policy_rejected(tmp_path,drift):
 lesson,source,original,current,path,master,tracks,generated,canonical=approved80_inputs(tmp_path)
 if drift=='source':source['original_visual_clock']['sha256']='f'*64
 if drift=='nonzero':generated['lessons'][0]['scenes'][0]['idle_tail_start']=.1;canonical=deepcopy(generated['lessons'])
 if drift=='missing_scene':generated['lessons'][0]['scenes'].pop();canonical=deepcopy(generated['lessons'])
 if drift=='missing_policy':generated=None;canonical=[]
 with pytest.raises(ValueError):b.checked_native_policy(lesson,{},source,current,path,master,tracks,generated,canonical,original_clock=original)


def test_native_normal_current_reuse_duration_guard_still_rejects_overflow(tmp_path,monkeypatch):
 import published_native_reuse as reuse
 lesson='20_cellpose_masks';stage=tmp_path/'stage';root=stage/'production'/lesson;repo=tmp_path/'repo';baseline=tmp_path/'baseline';baseline.mkdir()
 english=dict(id=lesson,scenes=[dict(visual='result',narration='Approved')]);put(root/'lesson.en.json',english);put(repo/f'tools/tutorials/lessons/{lesson}.json',english);monkeypatch.setattr(reuse,'REPO',repo)
 master=root/f'video/{lesson}_silent.mp4';master.parent.mkdir(parents=True);master.write_bytes(b'published exact master')
 original=dict(scenes=[dict(scene=1,visual='result',speech_start=0.,scene_end=3.,duration=3.,native_live_video=True)])
 bindings=dict(master=dict(path='master',bytes=master.stat().st_size,sha256=b.digest(master)),timing=dict(path='clock',sha256='c'*64,bytes=100));source=dict(baseline_root=str(baseline),baseline_manifest_sha256='m',baseline_publication_receipt_sha256='r',baseline_media_commit='commit',media_root='root')
 put(root/reuse.RECORD,dict(schema=1,kind='published_native_narration_reuse',lesson=lesson,fresh_capture=False,raw_capture_revalidated=False,**source,published_master=bindings['master'],published_timing=bindings['timing']))
 put(root/'scenes.json',reuse.interval_spec(english,original));monkeypatch.setattr(reuse,'_published_inputs',lambda *args:(baseline,original,bindings,source))
 import native_live_timing
 monkeypatch.setattr(native_live_timing,'verified_narration_tracks',lambda *args:([dict(scenes=[dict(duration=3.01)])]*50,[{}]*50))
 with pytest.raises(ValueError,match='exceeds unchanged'):reuse.current_reuse_timing(stage,lesson)


def test_prior_hit_list_refresh_manifest_has_no_false_claim_in_new20(candidate_inputs):
 f=candidate_inputs;old=b.read(f['baseline']/'release-manifest.json')
 old.update(refreshed_lessons=['48_hit_list'],hit_list_combined_refresh=True,
            hit_list_components=dict(audio_record_sha256='old48',picture_record_sha256='old48picture'),
            original_visual_timings=[dict(lesson='48_hit_list',path='old48clock')],
            web_checks=[dict(lesson='48_hit_list')],metadata_only_updates={'59':{}},
            retained_visual_timing={'48':{}},link_only_updates=['old'],
            prose_components={'oldlesson':'oldproof'},prose_visual_source={'old':'source'},
            human_listening_signoff=False,hosted_web_lessons=[f['lesson']])
 frozen=deepcopy(old);current=b.refresh_manifest_base(old,'20_cellpose_masks')
 assert current['refreshed_lessons']==['20_cellpose_masks'] and current['preserved_lessons']==83
 assert all(key not in current for key in ('hit_list_combined_refresh','hit_list_components',
            'original_visual_timings','web_checks','metadata_only_updates',
            'retained_visual_timing','link_only_updates','prose_components','prose_visual_source'))
 assert current['files']==old['files'] and current['hosted_web_lessons']==old['hosted_web_lessons']
 assert current['ceiling_bytes']==old['ceiling_bytes'] and current['human_listening_signoff'] is False
 assert old==frozen
