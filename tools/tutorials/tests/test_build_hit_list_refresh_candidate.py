"""Synthetic component contracts exercise assembly, not real audio or publication."""
from copy import deepcopy
from pathlib import Path
import json,sys
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'authoring/tools')]
import build_hit_list_refresh_candidate as builder
from stage_lesson import write
from append_staged_lessons import JS_PREFIX,JS_SUFFIX

@pytest.fixture
def prepared(tmp_path,monkeypatch):
 stage,base=tmp_path/'stage',tmp_path/'published';L=builder.LESSON;root=stage/'production'/L
 voices={'en':['af_heart'],'es':['ef_dora','em_alex','em_santa'],'fr':['ff_siwis'],'hi':['hf_alpha','hf_beta','hm_omega','hm_psi'],'it':['if_sara','im_nicola'],'ja':['jf_alpha','jf_gongitsune','jf_nezumi','jf_tebukuro','jm_kumo'],'pt-BR':['pf_dora','pm_alex','pm_santa'],'zh-CN':['zf_xiaobei','zf_xiaoni','zf_xiaoxiao','zf_xiaoyi','zm_yunjian','zm_yunxi','zm_yunxia','zm_yunyang']}
 old=dict(id=L,title='Hit List',description='Task',objectives=['Inspect'],prerequisite='Original prerequisite',app_key='hit_list',host_app_key='regression',scenes=[dict(visual=v,narration='Original paragraph '+str(i),hold_after=.7) for i,v in enumerate(builder.picture48.VISUALS)],narration_voices=voices)
 new=deepcopy(old);new['prerequisite']='Approved conditional prerequisite';new['scenes'][1]['narration']='Approved first sentence. Unchanged rest.'
 canonical={k:v for k,v in new.items() if k!='narration_voices'};write(root/'lesson.en.json',canonical)
 monkeypatch.setattr(builder.audio48,'CURRENT_ENGLISH_SHA256',builder.audio48.common.identity(canonical))
 others=[dict(id=f'{i:02}_fixture',title=f'Lesson{i}',scenes=[dict(visual='view',narration='Original')]) for i in range(1,84)]
 original=dict(lessons=[old,*others],scope='preserve all other objects');current={};baseline_bytes={}
 for name in builder.CATALOGS:
  write(base/'web/catalog'/name,original);updated=deepcopy(original);updated['lessons'][0]=deepcopy(new);write(stage/'catalog'/name,updated);current[name]=deepcopy(new)
 runtime=deepcopy(original)
 for row in runtime['lessons']:row['web']=f'{row["id"]}/web/{row["id"]}_silent.mp4'
 runtime['navigation']={'source_commit':'unchanged'}
 (base/'web/lesson_catalog.js').write_text(JS_PREFIX+json.dumps(runtime)+JS_SUFFIX)
 (base/'web/app_v2.js').write_bytes(b'same normal player');(base/'web/index.html').write_text('same normal index')
 def put(root,relative,data):
  p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);return p
 for row in others:put(base,f'media_host/{row["id"]}/web/{row["id"]}_silent.mp4',('unchanged '+row['id']).encode())
 for role in ('video','web'):put(base,f'media_host/{L}/{role}/{L}_silent.mp4',('original '+role).encode())
 put(base,f'web/production/{L}/poster.jpg',b'original poster')
 clock=dict(schema=1,total_duration=1.1,scenes=[dict(scene=i+1,text=s['narration'],speech_start=i/10,speech_end=(i+1)/10-.01,scene_end=(i+1)/10,duration=.1) for i,s in enumerate(old['scenes'])])
 tracks=[];source_tracks=[]
 for language,offered in voices.items():
  for voice in offered:
   prefix=f'media_host/{L}/audio/{language}/{voice}'
   put(base,prefix+'.m4a',('originalAAC '+language+voice).encode());write(base/(prefix+'.json'),clock)
   path=put(stage,f'production/{L}/audio/{language}/{voice}.m4a',('new mixedAAC '+language+voice).encode());write(path.with_suffix('.json'),dict(unit_fixture='new timing',language=language,voice=voice))
   tracks.append(dict(language=language,voice=voice,audio_sha256=builder.digest(path),timing_sha256=builder.digest(path.with_suffix('.json'))))
   source_tracks.append(dict(language=language,voice=voice))
 oldclock='media_host/'+L+'/audio/en/af_heart.json';original_clock=dict(path=oldclock,sha256=builder.digest(base/oldclock))
 put(stage,f'production/{L}/video/original-visual-timings.json',(base/oldclock).read_bytes())
 rows=[dict(path=p.relative_to(base).as_posix(),sha256=builder.digest(p),bytes=p.stat().st_size) for folder in ('web','media_host') for p in sorted((base/folder).rglob('*')) if p.is_file()]
 manifest=dict(files=rows,routes=84,ready_lessons=84,coming_soon=[],catalog_languages=14,refreshed_lessons=['06_api'],ceiling_bytes=10000000,hosted_web_lessons=[l['id'] for l in original['lessons']])
 write(base/'release-manifest.json',manifest);write(base/'publication-receipt.json',dict(synthetic_unit_fixture=True))
 source=dict(baseline_root=str(base),baseline_manifest_sha256=builder.digest(base/'release-manifest.json'),publication_receipt_sha256=builder.digest(base/'publication-receipt.json'),original_visual_clock=original_clock,tracks=source_tracks)
 visual=dict(baseline_root=str(base),manifest_sha256=source['baseline_manifest_sha256'],publication_receipt_sha256=source['publication_receipt_sha256'],original_clock=deepcopy(original_clock))
 monkeypatch.setattr(builder.audio48,'_source',lambda *a:(deepcopy(source),deepcopy(current)))
 monkeypatch.setattr(builder.picture48,'checked_source',lambda *a:(deepcopy(visual),deepcopy(canonical),deepcopy(clock)))
 calls=[]
 def check_tracks(*args,**kwargs):calls.append(kwargs.copy());return deepcopy(voices),deepcopy(tracks)
 monkeypatch.setattr(builder.audio48,'checked_tracks',check_tracks)
 guard=[];monkeypatch.setattr(builder,'require_published_script_baseline',lambda *a:guard.append('normal-published-guard'))
 paths=dict(master=put(stage,f'production/{L}/video/{L}_silent.mp4',b'finished new master'),web=put(stage,f'web-renditions/{L}/video/{L}_silent.mp4',b'finished new web'),poster=put(stage,f'production/{L}/poster.jpg',b'finished normal poster'),original_clock=root/'video/original-visual-timings.json')
 monkeypatch.setattr(builder.picture48,'checked_picture_timing',lambda s:paths['original_clock'])
 # Exercise the REAL finished-media consumer, with synthetic saved flags only.
 write(root/'current-video-acceptance.json',dict(master_full_decode_passed=True,master_all_presentation_times_checked=True,master_frame_count_matches_scene_boundaries=True,published_legacy_hit_list_header=True,native_fifty_voice_timeline=False,fresh_capture=False,raw_capture_revalidated=False,master_sha256=builder.digest(paths['master']),poster_sha256=builder.digest(paths['poster']),timing_sha256=builder.digest(paths['original_clock'])))
 write(stage/f'web-renditions/{L}/rendition-checks.json',dict(accepted=True,full_decode_passed=True,all_frame_presentation_times_match=True,master_sha256=builder.digest(paths['master']),rendition_sha256=builder.digest(paths['web'])))
 write(root/builder.audio48.RECORD,dict(synthetic_audio_unit_fixture=True));write(root/builder.picture48.RECORD,dict(synthetic_picture_unit_fixture=True))
 baseline_bytes={p.relative_to(base).as_posix():p.read_bytes() for p in base.rglob('*') if p.is_file()}
 return dict(stage=stage,base=base,source=source,visual=visual,current=current,old=old,new=new,runtime=runtime,calls=calls,guard=guard,baseline_bytes=baseline_bytes,paths=paths)


def test_combined_builder_keeps_83_objects_bytes_player_and_explicit_old_clock(prepared):
 p=prepared;candidate=builder.build(p['stage'],p['base']);report=builder.read(candidate/'release-manifest.json')
 assert p['calls']==[{'decode':True}] and p['guard']==['normal-published-guard']
 assert report['hit_list_combined_refresh'] is True and report['refreshed_lessons']==[builder.LESSON] and report['preserved_lessons']==83 and len(report['audio_patch_tracks'])==27
 assert report['fresh_capture'] is False and report['current_whole_track_tts_claimed'] is False and not (candidate/'publication-receipt.json').exists()
 before={r['path']:r for r in builder.read(p['base']/'release-manifest.json')['files']};after={r['path']:r for r in report['files']}
 assert set(after)-set(before)=={builder.CLOCK_PATH} and not set(before)-set(after)
 for row in p['runtime']['lessons'][1:]:
  relative='media_host/'+row['web'];assert after[relative]==before[relative] and (candidate/relative).read_bytes()==p['baseline_bytes'][relative]
 assert (candidate/'web/app_v2.js').read_bytes()==p['baseline_bytes']['web/app_v2.js']
 runtime=builder.parse_javascript((candidate/'web/lesson_catalog.js').read_text());assert runtime['lessons'][1:]==p['runtime']['lessons'][1:] and runtime['navigation']==p['runtime']['navigation']
 assert runtime['lessons'][0]['visual_timings']==builder.CLOCK_PATH.removeprefix('media_host/')
 assert (candidate/builder.CLOCK_PATH).read_bytes()==p['baseline_bytes'][p['source']['original_visual_clock']['path']]
 assert (candidate/builder.CLOCK_PATH).read_bytes()!=(candidate/p['source']['original_visual_clock']['path']).read_bytes()
 for name in builder.CATALOGS:
  original=builder.read(p['base']/'web/catalog'/name);new=builder.read(candidate/'web/catalog'/name)
  assert new['lessons'][1:]==original['lessons'][1:] and new['lessons'][0]==p['new']
 assert all((p['base']/relative).read_bytes()==data for relative,data in p['baseline_bytes'].items()),'Never write through a preserved hardlink'
 with pytest.raises(FileExistsError,match='existing actual candidate'):builder.build(p['stage'],p['base'])

@pytest.mark.parametrize('fault',['baseline','manifest','receipt','clock_path','clock_hash'])
def test_components_must_join_same_actual_baseline_and_original_clock(prepared,fault):
 p=prepared
 if fault=='baseline':p['visual']['baseline_root']=str(p['base'].parent/'different')
 elif fault=='manifest':p['visual']['manifest_sha256']='a'*64
 elif fault=='receipt':p['visual']['publication_receipt_sha256']='b'*64
 elif fault=='clock_path':p['visual']['original_clock']['path']='media_host/48_hit_list/audio/en/changed.json'
 else:p['visual']['original_clock']['sha256']='c'*64
 with pytest.raises(ValueError,match='share one actual'):builder.build(p['stage'],p['base'])
 assert not p['calls'] and not list(p['stage'].glob('release-candidate-*'))

@pytest.mark.parametrize('fault',['other_lesson','other_metadata','voice','missing_catalog','duplicate_object'])
def test_selected_catalog_scope_is_not_a_general_metadata_exception(prepared,fault):
 p=prepared;path=p['stage']/'catalog/lessons_es.json';value=builder.read(path)
 if fault=='other_lesson':value['lessons'][1]['scenes'][0]['narration']='Changed other lesson'
 elif fault=='other_metadata':value['scope']='Changed unrelated metadata'
 elif fault=='voice':value['lessons'][0]['narration_voices']['en'].append('am_adam')
 elif fault=='missing_catalog':path.rename(path.with_name('preserved-missing-catalog.json'))
 else:value['lessons'][1]=deepcopy(value['lessons'][0])
 if fault!='missing_catalog':write(path,value)
 with pytest.raises((ValueError,FileNotFoundError)):builder.build(p['stage'],p['base'])
 assert not p['calls']


def test_normal_publication_guard_cannot_be_bypassed_with_component_records(prepared,monkeypatch):
 p=prepared
 def reject(*args):raise ValueError('Actual published readback/Pages proof missing')
 monkeypatch.setattr(builder,'require_published_script_baseline',reject)
 with pytest.raises(ValueError,match='published readback'):builder.build(p['stage'],p['base'])
 assert not p['calls'] and not list(p['stage'].glob('release-candidate-*'))


def test_finished_poster_drift_rejected_by_independent_visual_checker(prepared):
 p=prepared;p['paths']['poster'].write_bytes(b'changed after finish')
 with pytest.raises(ValueError,match='poster'):builder.build(p['stage'],p['base'])
 assert not p['calls'] and not list(p['stage'].glob('release-candidate-*'))


def test_union_rejects_unselected_byte_change_or_extra_source_asset():
 before={'web/app_v2.js':dict(path='web/app_v2.js',sha256='a'), 'media_host/48_hit_list/audio/en/af_heart.m4a':dict(path='media_host/48_hit_list/audio/en/af_heart.m4a',sha256='b')}
 before[builder.CLOCK_PATH]=dict(path=builder.CLOCK_PATH,sha256='oldclock')
 allowed={'media_host/48_hit_list/audio/en/af_heart.m4a'};after=deepcopy(before);after['web/app_v2.js']['sha256']='changed'
 with pytest.raises(ValueError,match='Unselected'):builder.require_union(before,after,allowed,builder.CLOCK_PATH)
 after=deepcopy(before);after['media_host/48_hit_list/unapproved.bin']={}
 with pytest.raises(ValueError,match='inventory'):builder.require_union(before,after,allowed,builder.CLOCK_PATH)


def test_existing_explicit_clock_is_preserved_instead_of_rebinding_to_new_english(prepared):
 p=prepared;runtime=deepcopy(p['runtime']);relative='media_host/48_hit_list/video/previous-original-clock.json';runtime['lessons'][0]['visual_timings']=relative.removeprefix('media_host/')
 ref=dict(path=relative,sha256='d'*64);inventory={relative:dict(path=relative,sha256='d'*64)}
 after,path=builder.runtime_catalog(runtime,p['old'],p['new'],ref,{'media_host/48_hit_list/audio/en/af_heart.json'},inventory)
 assert path==relative and after['lessons'][0]['visual_timings']==runtime['lessons'][0]['visual_timings'] and runtime['lessons'][0]['prerequisite']==p['old']['prerequisite']
