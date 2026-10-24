"""Small synthetic fixtures exercise contracts, never stand in for published/capture proof."""
from copy import deepcopy
from pathlib import Path
import json,shutil,sys
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'authoring/tools')]
import published_legacy_picture_mix as mix
import render_api_reference_cards as cards
from stage_lesson import write,REPO
from check_completed_matrix import voice_matrix
from append_staged_lessons import JS_PREFIX,JS_SUFFIX

@pytest.fixture
def planned():
 english=dict(id=cards.LESSON,scenes=[dict(visual=v,narration=f'Inspect chapter {i+1}: '+v) for i,v in enumerate(cards.VISUALS)])
 clock=dict(schema=1,total_duration=1.2,scenes=[dict(scene=i+1,text=a['narration'],speech_start=i/5,speech_end=(i+1)/5-.01,scene_end=(i+1)/5,duration=.2) for i,a in enumerate(english['scenes'])])
 refs={i:dict(source_scene=i,visual=cards.VISUALS[i-1],image=cards.filename(i),kind='authored_reference_card',narration_sha256=cards.identity(english['scenes'][i-1]['narration'])) for i in cards.CARDS}
 return english,clock,refs

def test_five_indexed_cards_preserve_complete_chapter2_and_all_six_clocks(planned):
 before=deepcopy(planned);rows=mix.plan(*planned)
 assert [r['scene'] for r in rows if 'reference_card' in r]==[1,3,4,5,6]
 assert rows[1]==dict(scene=2,visual='02_list',source_start_frame=6,source_keep_frames=[[0,6]],kept_frames=6,final_frame_hold_frames=0,edited=False)
 assert [r['source_start_frame'] for r in rows]==list(range(0,36,6))
 assert all(r['source_keep_frames']==[[0,6]] and r['edited'] is False and r['final_frame_hold_frames']==0 for r in rows)
 assert not any('native_live_video' in r for r in rows) and planned==before
 assert rows[2]['visual']==rows[5]['visual'] and rows[2]['reference_card']['image']!=rows[5]['reference_card']['image']
 assert rows[2]['reference_card']['narration_sha256']!=rows[5]['reference_card']['narration_sha256']

@pytest.mark.parametrize('fault',['visual_key','repeat_swap','bool_index','wrong_visual','extra_card','missing_card','clock_gap','clock_index','native','changed_text','clock_kind'])
def test_index_clock_and_card_scope_reject_drift(planned,fault):
 english,clock,refs=deepcopy(planned)
 if fault=='visual_key':refs={r['visual']:r for r in refs.values()}
 elif fault=='repeat_swap':refs[3],refs[6]=refs[6],refs[3]
 elif fault=='bool_index':refs[1]['source_scene']=True
 elif fault=='wrong_visual':refs[3]['visual']='02_list'
 elif fault=='extra_card':refs[2]=deepcopy(refs[1])
 elif fault=='missing_card':refs.pop(4)
 elif fault=='clock_gap':clock['scenes'][1]['speech_start']+=.01
 elif fault=='clock_index':clock['scenes'][5]['scene']=3
 elif fault=='native':clock['scenes'][0]['native_live_video']=True
 elif fault=='changed_text':clock['scenes'][0]['text']='changed'
 else:clock['kind']='native_visual_timing'
 with pytest.raises(ValueError):mix.plan(english,clock,refs)

def test_default10_contract_keeps_original_objects_and_unknown_lessons_refuse():
 import render_classify_input_reference_cards as old
 english=dict(id=mix.LESSON,scenes=[dict(visual=v,narration='Inspect '+v) for v in old.VISUALS])
 clock=dict(schema=1,total_duration=2.2,scenes=[dict(scene=i+1,text=a['narration'],speech_start=i/5,speech_end=(i+1)/5-.01,scene_end=(i+1)/5,duration=.2) for i,a in enumerate(english['scenes'])])
 refs={v:dict(visual=v,image=v+'.png',kind='authored_reference_card',narration_sha256=old.identity(next(a['narration'] for a in english['scenes'] if a['visual']==v))) for v in old.CARDS}
 rows=mix.plan(english,clock,refs)
 expected=[dict(scene=i+1,visual=a['visual'],source_start_frame=i*6,source_keep_frames=[[0,6]],kept_frames=6,final_frame_hold_frames=0,edited=False,**({'reference_card':deepcopy(refs[a['visual']])} if a['visual'] in refs else {})) for i,a in enumerate(english['scenes'])]
 assert rows==expected and mix.contract(mix.LESSON)[:3]==(mix.KIND,11,[2,3,4])
 with pytest.raises(ValueError):mix.contract('07_mask')

def test_content_labels_example_version_and_no_execution_limit_without_claiming_run_result():
 values=cards.definitions();assert tuple(values)==cards.CARDS
 assert 'Example output: 1.5.0.9' in values[1][2]
 assert values[6][1]=='Command instructions' and '--dry-run' not in values[6][2][0]
 assert 'Example dry-run: perform_regression() was not called.' in values[5][2]
 assert 'In this example, nothing was executed.' in values[5][2]
 assert not any('recording' in text.lower() or 'current version' in text.lower() for title,label,lines in values.values() for text in [title,label,*lines])
 assert cards.filename(3)!=cards.filename(6)

@pytest.fixture
def published(tmp_path,monkeypatch,planned):
 from audit_staged_catalogs import CATALOGS
 english,clock,_=deepcopy(planned);base,repo=tmp_path/'published',tmp_path/'repo'
 renderer=repo/'tools/tutorials/authoring/tools/render_all_voices.py';renderer.parent.mkdir(parents=True)
 shutil.copy2(REPO/'tools/tutorials/authoring/tools/render_all_voices.py',renderer)
 declared=voice_matrix(renderer);declared['en']=['af_heart'];assert sum(map(len,declared.values()))==27
 original={**deepcopy(english),'narration_voices':declared};write(repo/'tools/tutorials/lessons'/str(cards.LESSON+'.json'),english)
 for name in CATALOGS:
  value=dict(lessons=[deepcopy(original)])
  write(base/'web/catalog'/name,value);write(repo/'tools/tutorials/authoring/catalog'/name,value)
 runtime=dict(lessons=[{**deepcopy(original),'silent':f'{cards.LESSON}/video/{cards.LESSON}_silent.mp4','web':f'{cards.LESSON}/web/{cards.LESSON}_silent.mp4'}])
 (base/'web/lesson_catalog.js').write_text(JS_PREFIX+json.dumps(runtime)+JS_SUFFIX)
 for lang,voices in declared.items():
  for voice in voices:
   audio=base/f'media_host/{cards.LESSON}/audio/{lang}/{voice}.m4a';audio.parent.mkdir(parents=True,exist_ok=True);audio.write_bytes(('unit fake AAC '+lang+voice).encode())
   value={**deepcopy(clock),'language':lang,'voice':voice,'media_sha256':mix.digest(audio),'render_inputs':dict(lesson=cards.LESSON,language=lang,voice=voice,scenes=[dict(narration=s['narration']) for s in english['scenes']])}
   write(audio.with_suffix('.json'),value)
 master=f'media_host/{cards.LESSON}/video/{cards.LESSON}_silent.mp4';web=f'media_host/{cards.LESSON}/web/{cards.LESSON}_silent.mp4'
 for relative in (master,web,f'web/production/{cards.LESSON}/poster.jpg'):
  path=base/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(('unit source '+relative).encode())
 files=[dict(path=p.relative_to(base).as_posix(),sha256=mix.digest(p),bytes=p.stat().st_size) for root in ('web','media_host') for p in sorted((base/root).rglob('*')) if p.is_file()]
 manifest=dict(files=files,routes=1,ready_lessons=1,hosted_web_lessons=[cards.LESSON]);write(base/'release-manifest.json',manifest)
 n=sum(r['path'].startswith('media_host/') for r in files);commit='a'*40;url='https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/'+commit
 receipt=dict(manifest_sha256=mix.digest(base/'release-manifest.json'),commit=commit,tag='fixture',media_root=url,media_files=n,readback=dict(commit=commit,passed=True,files_expected=n,metadata_matched=n,downloaded_sha256_matched=n,metadata_failures=[],download_failures=[]),pages=dict(index_sha256='b'*64,ready=1,routes=1,hosted_web_copies=1))
 write(base/'publication-receipt.json',receipt);write(base/'checks/published-media-browser-checks.json',dict(passed=True,manifest_sha256=receipt['manifest_sha256'],media_root=url,index_sha256='b'*64,placeholder_cases=[],ready_playback_cases=[dict(lesson=cards.LESSON,passed=True)]))
 monkeypatch.setattr(mix,'REPO',repo)
 monkeypatch.setattr(cards,'PUBLISHED',dict(master=mix.digest(base/master),web=mix.digest(base/web),clock=mix.digest(base/f'media_host/{cards.LESSON}/audio/en/af_heart.json')))
 return base,tmp_path/'stage',repo

def test_prepare_preserves_all54_audio_files_full14_catalog_bytes_and_clock(published):
 base,stage,_=published;mix.prepare(stage,base,cards.LESSON);source,english,clock=mix.checked_source(stage,cards.LESSON)
 assert source['kind']==cards.KIND and len(source['tracks'])==27 and source['fresh_capture'] is False
 assert source['raw_capture_revalidated'] is False and source['current_runtime_freshness_claimed'] is False
 for t in source['tracks']:
  for key in ('audio','timing'):assert (stage/t[key].replace('media_host/','production/',1)).read_bytes()==(base/t[key]).read_bytes()
 for path in (stage/'catalog').iterdir():assert path.read_bytes()==(base/'web/catalog'/path.name).read_bytes()
 assert (stage/'production'/cards.LESSON/'video/original-visual-timings.json').read_bytes()==(base/source['original_clock']['path']).read_bytes()
 with pytest.raises(FileExistsError):mix.prepare(stage,base,cards.LESSON)

@pytest.mark.parametrize('fault',['audio','clock','catalog','source_record','version_master','offerings','unselected_chapter'])
def test_staged_or_authored_drift_refuses_published_retention(published,fault):
 base,stage,repo=published;mix.prepare(stage,base,cards.LESSON);root=stage/'production'/cards.LESSON
 if fault=='audio':(root/'audio/es/ef_dora.m4a').write_bytes(b'changed')
 elif fault=='clock':(root/'video/original-visual-timings.json').write_text('{}')
 elif fault=='catalog':(stage/'catalog/lessons_en.json').write_text('{}')
 elif fault=='source_record':write(root/mix.SOURCE,{})
 elif fault=='version_master':(base/f'media_host/{cards.LESSON}/video/{cards.LESSON}_silent.mp4').write_bytes(b'new')
 else:
  p=repo/'tools/tutorials/authoring/catalog/lessons_en.json';value=mix.read(p)
  if fault=='offerings':value['lessons'][0]['narration_voices']['en'].append('af_bella')
  else:value['lessons'][0]['scenes'][1]['narration']='changed chapter2'
  write(p,value)
 with pytest.raises((ValueError,KeyError)):mix.checked_source(stage,cards.LESSON)

@pytest.fixture
def reference_fixture(published,monkeypatch):
 from PIL import Image
 base,stage,repo=published;mix.prepare(stage,base,cards.LESSON);source,english,_=mix.checked_source(stage,cards.LESSON)
 folder=stage/'production'/cards.LESSON/'reference-cards';(folder/'source-frames').mkdir(parents=True)
 hashes={}
 for index in cards.CARDS:
  p=folder/'source-frames'/f'chapter-{index:03}.png';Image.new('RGB',(12,8),(index,2,3)).save(p);hashes[index]=cards.digest(p)
 monkeypatch.setattr(cards,'SOURCE_FRAMES',hashes)
 style=repo/'tools/tutorials/authoring/tools/render_install_keyframes.py';style.write_text('# bound fixture style')
 rows=[]
 for index in cards.CARDS:
  p=folder/cards.filename(index);Image.new('RGB',(3840,2160),(index,2,3)).save(p)
  rows.append(dict(source_scene=index,visual=cards.VISUALS[index-1],image=p.name,sha256=cards.digest(p),kind='authored_reference_card',narration_sha256=cards.identity(english['scenes'][index-1]['narration'])))
 write(folder/'cards.json',{**cards.header(english,source,repo),'cards':rows})
 return folder,english,source,repo

@pytest.mark.parametrize('fault',[None,'frame','png','swap_repeat','unsafe_image','duplicate','source_hash','style','wording'])
def test_reference_contract_rejects_changed_frame_image_source_or_repeated_identity(reference_fixture,fault):
 folder,english,source,repo=reference_fixture;contract=cards.read(folder/'cards.json')
 if fault=='frame':(folder/'source-frames/chapter-003.png').write_bytes(b'changed')
 elif fault=='png':(folder/cards.filename(6)).write_bytes(b'changed')
 elif fault=='swap_repeat':contract['cards'][1],contract['cards'][-1]=contract['cards'][-1],contract['cards'][1]
 elif fault=='unsafe_image':contract['cards'][0]['image']='../unbound.png'
 elif fault=='duplicate':contract['cards'][-1]['source_scene']=3
 elif fault=='source_hash':source=deepcopy(source);source['original_visuals'][0]['sha256']='f'*64
 elif fault=='style':(repo/'tools/tutorials/authoring/tools/render_install_keyframes.py').write_text('changed')
 elif fault=='wording':contract['definitions_sha256']='f'*64
 write(folder/'cards.json',contract)
 if fault is None:
  checked=cards.validate_cards(folder,english,repo,source);assert list(checked)==list(cards.CARDS)
 else:
  with pytest.raises(ValueError):cards.validate_cards(folder,english,repo,source)

def test_actual_bound_master_web_clock_guards_refuse_even_a_rebound_record(reference_fixture):
 _,_,source,_=reference_fixture
 for key in ('master','web','clock'):
  bad=deepcopy(source)
  if key=='clock':bad['original_clock']['sha256']='f'*64
  else:bad['original_visuals'][0 if key=='master' else 1]['sha256']='f'*64
  with pytest.raises(ValueError):cards.checked_published(bad)

def test_encoding_routes_only_chapter2_to_complete_source_interval(planned,tmp_path,monkeypatch):
 import published_native_edit,render_visual_master
 english,clock,refs=planned;rows=mix.plan(english,clock,refs);source=tmp_path/'source.mp4';source.write_bytes(b'unit not media')
 calls=[]
 monkeypatch.setattr(published_native_edit,'encode_scene',lambda p,row,target:calls.append(('source',p,deepcopy(row),target)))
 monkeypatch.setattr(render_visual_master,'encode_still',lambda p,d,target,fps:calls.append(('card',p,d,target,fps)))
 monkeypatch.setattr(render_visual_master,'concat',lambda parts,target,listing:calls.append(('concat',parts,target)))
 mix.encode_parts(source,rows,tmp_path/'cards',tmp_path/'parts',tmp_path/'target.mp4')
 preserved=[c for c in calls if c[0]=='source'];assert len(preserved)==1 and preserved[0][2]==rows[1]
 assert len([c for c in calls if c[0]=='card'])==5 and calls[-1][0]=='concat'
 with pytest.raises(FileExistsError):mix.encode_parts(source,rows,tmp_path/'cards',tmp_path/'parts',tmp_path/'another.mp4')

def test_only06_master_web_poster_may_change_and_unknown_dispatch_refuses():
 lesson=cards.LESSON;visuals=[f'media_host/{lesson}/video/{lesson}_silent.mp4',f'media_host/{lesson}/web/{lesson}_silent.mp4',f'web/production/{lesson}/poster.jpg']
 kept=['web/lesson_catalog.js','web/catalog/lessons_en.json',f'media_host/{lesson}/audio/en/af_heart.json',f'media_host/{lesson}/audio/es/ef_dora.m4a']
 before={p:dict(path=p,sha256='a'*64,bytes=1) for p in visuals+kept};after=deepcopy(before)
 for p in visuals:after[p]['sha256']='b'*64
 mix.require_candidate_records(before,after,lesson)
 for p in kept:
  bad=deepcopy(after);bad[p]['sha256']='c'*64
  with pytest.raises(ValueError):mix.require_candidate_records(before,bad,lesson)
 with pytest.raises(ValueError):mix.checked_legacy_picture_timing(Path('/not-an-input'),'unknown')

def test_normal06_finish_dispatch_retains_legacy27_not_native(planned,tmp_path,monkeypatch):
 import finish_staged_visual as finish
 import prepare_web_media as web
 english,clock,_=planned;stage=tmp_path/'stage';root=stage/'production'/cards.LESSON;repo=tmp_path/'repo'
 write(root/'lesson.en.json',english);write(repo/'tools/tutorials/lessons'/str(cards.LESSON+'.json'),english)
 write(root/'scenes.json',dict(kind=cards.KIND,scenes=[dict(narration=s['narration']) for s in english['scenes']]))
 write(root/mix.RECORD,{});write(root/'video/original-visual-timings.json',clock)
 master=root/'video'/str(cards.LESSON+'_silent.mp4');master.write_bytes(b'unit mocked encoder result')
 publisher=repo/'tools/tutorials/authoring/tools/publish_tutorials.py';publisher.parent.mkdir(parents=True);publisher.write_text('ENCODE_ARGS=[]\n')
 calls=[];monkeypatch.setattr(finish,'REPO',repo)
 monkeypatch.setattr(mix,'checked_legacy_picture_timing',lambda s,i:(calls.append((s,i)) or root/'video/original-visual-timings.json'))
 monkeypatch.setattr(finish,'checked_native_timing',lambda *a:pytest.fail('Legacy06 must not request native validation'))
 monkeypatch.setattr(finish,'probe',lambda p:dict(streams=[dict(codec_type='video',width=3840,height=2160,r_frame_rate='30/1',nb_frames='36')]))
 monkeypatch.setattr(finish,'timestamps',lambda p:[i/30 for i in range(36)])
 monkeypatch.setattr(finish.subprocess,'run',lambda *a,**k:None)
 monkeypatch.setattr(web,'main',lambda:(root/'poster.jpg').write_bytes(b'unit poster'))
 monkeypatch.setattr(finish,'stage_one',lambda *a,**k:dict(accepted=True))
 receipt=finish.finish(stage,cards.LESSON)
 assert calls==[(stage,cards.LESSON)] and receipt['published_legacy_picture_references'] is True
 assert receipt['native_fifty_voice_timeline'] is False and receipt['retained_narration_offerings']==27
 assert receipt['fresh_capture'] is False and receipt['raw_capture_revalidated'] is False


def test_normal06_poster_requires_checked_record_before_extracting_any_frame(tmp_path,monkeypatch):
 import prepare_web_media as web
 root=tmp_path/'production'/cards.LESSON;write(root/'scenes.json',dict(kind=cards.KIND,scenes=[]));(root/'video').mkdir();(root/'video'/str(cards.LESSON+'_silent.mp4')).write_bytes(b'unit invalid master')
 monkeypatch.setattr(web,'PRODUCTION',tmp_path/'production');monkeypatch.setattr(sys,'argv',['prepare_web_media','--lessons',cards.LESSON])
 def reject(*args):raise ValueError('Actual06 source binding must pass first')
 monkeypatch.setattr(mix,'checked_legacy_picture_timing',reject)
 with pytest.raises(ValueError,match='Actual06 source binding'):web.main()


@pytest.mark.parametrize('changed_poster',[False,True])
def test06_candidate_keeps_every_nonvisual_byte_and_rejects_postfinish_poster_change(published,monkeypatch,changed_poster):
 # Mocked finish evidence checks wiring, never claims actual encoding acceptance.
 base,stage,_=published;mix.prepare(stage,base,cards.LESSON);root=stage/'production'/cards.LESSON
 master=root/'video'/str(cards.LESSON+'_silent.mp4');master.write_bytes(b'unit updated master')
 poster=root/'poster.jpg';poster.write_bytes(b'unit updated poster')
 web=stage/'web-renditions'/cards.LESSON/'video'/master.name;web.parent.mkdir(parents=True);web.write_bytes(b'unit updated web')
 clock=root/'video/original-visual-timings.json'
 write(root/'current-video-acceptance.json',dict(master_full_decode_passed=True,master_all_presentation_times_checked=True,master_frame_count_matches_scene_boundaries=True,published_legacy_picture_references=True,native_fifty_voice_timeline=False,master_sha256=mix.digest(master),timing_sha256=mix.digest(clock),poster_sha256=mix.digest(poster)))
 write(web.parent.parent/'rendition-checks.json',dict(accepted=True,full_decode_passed=True,all_frame_presentation_times_match=True,master_sha256=mix.digest(master),rendition_sha256=mix.digest(web)))
 monkeypatch.setattr(mix,'checked_legacy_picture_timing',lambda s,i:clock)
 write(root/'reference-cards/cards.json',dict(unit_fixture=True));validated=[]
 monkeypatch.setattr(mix,'validate',lambda target,**options:validated.append((target,options)))
 if changed_poster:
  poster.write_bytes(b'changed after normal finish')
  with pytest.raises(ValueError,match='different media/clock'):mix.build_candidate(stage,cards.LESSON)
  assert not list(stage.glob('release-candidate-legacy-pictures-*'))
  return
 candidate=mix.build_candidate(stage,cards.LESSON);before=mix.read(base/'release-manifest.json');after=mix.read(candidate/'release-manifest.json')
 changed={r['path'] for r in after['files'] if r!=next(v for v in before['files'] if v['path']==r['path'])}
 assert changed=={f'media_host/{cards.LESSON}/video/{master.name}',f'media_host/{cards.LESSON}/web/{master.name}',f'web/production/{cards.LESSON}/poster.jpg'}
 assert after['legacy_picture_references']['card_slots']==[1,3,4,5,6]
 assert after['fresh_capture'] is False and validated==[(candidate,dict(include_hosted_media=True))]
 for row in before['files']:
  if row['path'] not in changed:assert (candidate/row['path']).read_bytes()==(base/row['path']).read_bytes()
