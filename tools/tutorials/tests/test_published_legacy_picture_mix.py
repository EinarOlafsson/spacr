"""Synthetic unit sources are not capture/publication proof; real tiny encode checks frame order."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'authoring/tools')]
import published_legacy_picture_mix as mix
import render_classify_input_reference_cards as cards
from stage_lesson import write,REPO
from check_completed_matrix import voice_matrix
from append_staged_lessons import JS_PREFIX,JS_SUFFIX

@pytest.fixture
def planned():
 english=dict(id=mix.LESSON,scenes=[dict(visual=v,narration='Inspect '+v) for v in cards.VISUALS])
 clock=dict(schema=1,total_duration=2.2,scenes=[dict(scene=i+1,text=a['narration'],speech_start=i/5,speech_end=(i+1)/5-.01,scene_end=(i+1)/5,duration=.2) for i,a in enumerate(english['scenes'])])
 refs={v:dict(visual=v,image=v+'.png',kind='authored_reference_card',narration_sha256=cards.identity(next(a['narration'] for a in english['scenes'] if a['visual']==v))) for v in cards.CARDS}
 return english,clock,refs

def test_only_three_card_slots_and_eight_complete_original_intervals(planned):
 before=deepcopy(planned);rows=mix.plan(*planned)
 assert [r['scene'] for r in rows if 'reference_card' in r]==[2,3,4]
 assert [r['source_start_frame'] for r in rows]==list(range(0,66,6))
 assert all(r['source_keep_frames']==[[0,6]] and r['kept_frames']==6 and r['final_frame_hold_frames']==0 for r in rows)
 assert all('native_live_video' not in r for r in rows) and planned==before

@pytest.mark.parametrize('fault',['native','clock_kind','reorder','missing','extra_card','card_text','time_gap','wrong_index','bool_index','nonfinite_total'])
def test_invalid_source_or_card_scope_fails_closed(planned,fault):
 english,clock,refs=deepcopy(planned)
 if fault=='native':clock['scenes'][0]['native_live_video']=True
 elif fault=='clock_kind':clock['kind']='native_visual_timing'
 elif fault=='reorder':english['scenes'][0],english['scenes'][1]=english['scenes'][1],english['scenes'][0]
 elif fault=='missing':refs.pop(cards.CARDS[0])
 elif fault=='extra_card':refs[cards.VISUALS[0]]=deepcopy(next(iter(refs.values())))
 elif fault=='card_text':refs[cards.CARDS[0]]['narration_sha256']='changed'
 elif fault=='time_gap':clock['scenes'][3]['speech_start']+=.01
 elif fault=='bool_index':clock['scenes'][0]['scene']=True
 elif fault=='nonfinite_total':clock['total_duration']=float('nan')
 else:clock['scenes'][2]['scene']=1
 with pytest.raises(ValueError):mix.plan(english,clock,refs)

@pytest.fixture
def published(tmp_path,monkeypatch,planned):
 from audit_staged_catalogs import CATALOGS
 english,clock,_=deepcopy(planned);base,repo=tmp_path/'published',tmp_path/'repo'
 renderer=repo/'tools/tutorials/authoring/tools/render_all_voices.py';renderer.parent.mkdir(parents=True)
 shutil.copy2(REPO/'tools/tutorials/authoring/tools/render_all_voices.py',renderer)
 declared=voice_matrix(renderer);declared['en']=['af_heart'];assert sum(map(len,declared.values()))==27
 original={**deepcopy(english),'narration_voices':declared}
 write(repo/'tools/tutorials/lessons'/str(mix.LESSON+'.json'),english)
 for name in CATALOGS:
  value=dict(lessons=[deepcopy(original)])
  write(base/'web/catalog'/name,value);write(repo/'tools/tutorials/authoring/catalog'/name,value)
 runtime=dict(lessons=[{**deepcopy(original),'silent':f'{mix.LESSON}/video/{mix.LESSON}_silent.mp4','web':f'{mix.LESSON}/web/{mix.LESSON}_silent.mp4'}])
 (base/'web/lesson_catalog.js').write_text(JS_PREFIX+json.dumps(runtime)+JS_SUFFIX)
 for lang,voices in declared.items():
  for voice in voices:
   audio=base/f'media_host/{mix.LESSON}/audio/{lang}/{voice}.m4a';audio.parent.mkdir(parents=True,exist_ok=True);audio.write_bytes(('unit fake AAC '+lang+voice).encode())
   value={**deepcopy(clock), 'language':lang,'voice':voice,'media_sha256':mix.digest(audio),'render_inputs':dict(lesson=mix.LESSON,language=lang,voice=voice,scenes=[dict(narration=s['narration']) for s in english['scenes']])}
   write(audio.with_suffix('.json'),value)
 for relative in (f'media_host/{mix.LESSON}/video/{mix.LESSON}_silent.mp4',f'media_host/{mix.LESSON}/web/{mix.LESSON}_silent.mp4',f'web/production/{mix.LESSON}/poster.jpg'):
  path=base/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'unit source; never a real media proof')
 files=[dict(path=p.relative_to(base).as_posix(),sha256=mix.digest(p),bytes=p.stat().st_size) for root in ('web','media_host') for p in sorted((base/root).rglob('*')) if p.is_file()]
 manifest=dict(files=files,routes=1,ready_lessons=1,hosted_web_lessons=[mix.LESSON]);write(base/'release-manifest.json',manifest)
 n=sum(r['path'].startswith('media_host/') for r in files);commit='a'*40;root='https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/'+commit
 receipt=dict(manifest_sha256=mix.digest(base/'release-manifest.json'),commit=commit,tag='fixture',media_root=root,media_files=n,readback=dict(commit=commit,passed=True,files_expected=n,metadata_matched=n,downloaded_sha256_matched=n,metadata_failures=[],download_failures=[]),pages=dict(index_sha256='b'*64,ready=1,routes=1,hosted_web_copies=1))
 write(base/'publication-receipt.json',receipt);write(base/'checks/published-media-browser-checks.json',dict(passed=True,manifest_sha256=receipt['manifest_sha256'],media_root=root,index_sha256='b'*64,placeholder_cases=[],ready_playback_cases=[dict(lesson=mix.LESSON,passed=True)]))
 monkeypatch.setattr(mix,'REPO',repo)
 return base,tmp_path/'stage',repo

def test_exact_legacy27_prepare_keeps_all_catalog_audio_and_clock_bytes(published):
 base,stage,_=published;mix.prepare(stage,base);source,english,clock=mix.checked_source(stage)
 assert len(source['tracks'])==27 and source['fresh_capture'] is False and source['current_runtime_freshness_claimed'] is False
 assert mix.digest(stage/'production'/mix.LESSON/'video/original-visual-timings.json')==source['original_clock']['sha256']
 for t in source['tracks']:
  for k in ('audio','timing'):assert (stage/t[k].replace('media_host/','production/',1)).read_bytes()==(base/t[k]).read_bytes()

@pytest.mark.parametrize('fault',['audio','catalog','clock','extra_track','source_master','record','symlink','voices'])
def test_published_or_staged_drift_cannot_bypass_retention(published,fault):
 base,stage,repo=published;mix.prepare(stage,base);root=stage/'production'/mix.LESSON
 if fault=='audio':p=root/'audio/en/af_heart.m4a'
 elif fault=='catalog':p=stage/'catalog/lessons_en.json'
 elif fault=='clock':p=root/'video/original-visual-timings.json'
 elif fault=='extra_track':p=root/'audio/en/unknown.m4a'
 elif fault=='source_master':p=base/f'media_host/{mix.LESSON}/video/{mix.LESSON}_silent.mp4'
 elif fault=='record':p=root/mix.SOURCE
 elif fault=='voices':
  p=repo/'tools/tutorials/authoring/catalog/lessons_en.json';value=json.loads(p.read_text());value['lessons'][0]['narration_voices']['en']=[];write(p,value)
  with pytest.raises(ValueError):mix.checked_source(stage)
  return
 else:
  p=root/'audio/en/af_heart.m4a';p.rename(p.with_name('preserved.m4a'));p.symlink_to('preserved.m4a')
  with pytest.raises(ValueError):mix.checked_source(stage)
  return
 p.write_bytes(b'changed')
 with pytest.raises((ValueError,json.JSONDecodeError)):mix.checked_source(stage)

@pytest.fixture
def saved_sources(tmp_path,monkeypatch):
 folder=tmp_path/'sources';folder.mkdir();rows=[]
 wells={'train':[['plate1','r12','c1'],['plate1','r5','c2']],'test':[['plate1','r12','c2'],['plate1','r5','c1']]}
 for split,items in wells.items():
  for well in items:
   for label in ('infected_1','infected_2'):
    for i in range(8):
     name='_'.join(well)+'_'+label+'_'+str(i);rows.append(dict(prcfo=name,filename=name+'.png',well=well,split=split,class_name=label))
 manifest=dict(accepted=True,source_unchanged=True,model_started=False,source_database_sha256='a'*64,class_values={'infected_1':1,'infected_2':2},train_wells=wells['train'],test_wells=wells['test'],records=rows)
 inputs=dict(objects_checked=64,sample_counts=dict(train=32,test=32),actual_wells=wells,source_database_sha256='a'*64)
 proof=dict(accepted=True,preparation=dict(accepted=True,helper_sha256=cards.HELPER,inputs=inputs),native=dict(inputs=inputs))
 commands=[dict(scene=v,accepted=True,returncode=0,output='"objects": 64, "train_objects": 32, "test_objects": 32') for v in ('01_source_example','02_prepare_existing_split','03_exported_manifest')]
 for name,value in [('commands.json',commands),('tutorial_input_manifest.json',manifest),('scientific_acceptance.json',proof)]:write(folder/name,value)
 monkeypatch.setattr(cards,'SOURCES',{name:cards.digest(folder/name) for name in cards.SOURCES})
 return folder,manifest

def test_card_content_uses_actual_manifest_counts_and_explicit_reference_labels(saved_sources):
 folder,manifest=saved_sources;assert cards.checked_sources(folder)==manifest
 values=cards.definitions();assert set(values)==set(cards.CARDS)
 assert all('reference' in label.lower() for title,label,lines in values.values())
 assert '64' in values[cards.CARDS[1]][2][-1] and '32' in values[cards.CARDS[1]][2][-1]
 assert not any('recording' in line.lower() for title,label,lines in values.values() for line in lines)

@pytest.mark.parametrize('fault',['bytes','count','well','command'])
def test_cards_fail_on_tampered_or_even_rebound_inconsistent_sources(saved_sources,monkeypatch,fault):
 folder,manifest=saved_sources
 if fault=='bytes':(folder/'commands.json').write_text('tampered')
 elif fault in ('count','well'):
  if fault=='count':manifest['records'].pop()
  else:manifest['records'][0]['well']=['plate1','r99','c1']
  write(folder/'tutorial_input_manifest.json',manifest);cards.SOURCES['tutorial_input_manifest.json']=cards.digest(folder/'tutorial_input_manifest.json')
 else:
  commands=json.loads((folder/'commands.json').read_text());commands[0]['returncode']=1;write(folder/'commands.json',commands);cards.SOURCES['commands.json']=cards.digest(folder/'commands.json')
 with pytest.raises(ValueError):cards.checked_sources(folder)

@pytest.mark.parametrize('count',[6,7])
def test_tiny_actual_encode_preserves_every_noncard_source_frame_and_only_replaces_three(planned,tmp_path,count):
 from PIL import Image
 english,clock,refs=deepcopy(planned)
 for i,s in enumerate(clock['scenes']):s.update(speech_start=i*count/30,speech_end=(i+1)*count/30-.01,scene_end=(i+1)*count/30,duration=count/30)
 clock['total_duration']=11*count/30;rows=mix.plan(english,clock,refs)
 def frame(value):
  image=Image.new('RGB',(64,48))
  for bit in range(7):
   colour=(235,235,235) if value&(1<<bit) else (20,20,20)
   image.paste(colour,(bit*8,0,bit*8+8,48))
  return image
 raw=b''.join(frame(i).tobytes() for i in range(11*count));source=tmp_path/'source.mp4'
 subprocess.run(['ffmpeg','-v','error','-n','-f','rawvideo','-pixel_format','rgb24','-video_size','64x48','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-threads','2','-crf','12','-pix_fmt','yuv420p',str(source)],input=raw,check=True)
 before=mix.digest(source);directory=tmp_path/'cards';directory.mkdir()
 for n,visual in enumerate(cards.CARDS):frame(90+n).save(directory/(visual+'.png'))
 target=tmp_path/'output.mp4';mix.encode_parts(source,rows,directory,tmp_path/'parts',target)
 decoded=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(target),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
 size=64*48*3;assert len(decoded)//size==11*count
 result=[]
 for i in range(len(decoded)//size):
  frame_bytes=decoded[i*size:(i+1)*size];result.append(sum((frame_bytes[(24*64+bit*8+4)*3]>128)<<bit for bit in range(7)))
 expected=[]
 for i in range(11):expected.extend([89+i]*count if i in (1,2,3) else range(i*count,(i+1)*count))
 assert result==expected and mix.digest(source)==before
 with pytest.raises(FileExistsError):mix.encode_parts(source,rows,directory,tmp_path/'again',source)
 assert mix.digest(source)==before


def test_candidate_scope_rejects_audio_catalog_runtime_or_extra_record_changes():
    visual=[f'media_host/{mix.LESSON}/video/{mix.LESSON}_silent.mp4',f'media_host/{mix.LESSON}/web/{mix.LESSON}_silent.mp4',f'web/production/{mix.LESSON}/poster.jpg']
    kept=['web/lesson_catalog.js','web/catalog/lessons_en.json',f'media_host/{mix.LESSON}/audio/en/af_heart.m4a',f'media_host/{mix.LESSON}/audio/en/af_heart.json']
    original={p:dict(path=p,sha256='a'*64,bytes=10) for p in visual+kept}
    refreshed=deepcopy(original)
    for p in visual:refreshed[p].update(sha256='b'*64,bytes=20)
    mix.require_candidate_records(original,refreshed)
    for changed in kept:
        bad=deepcopy(refreshed);bad[changed]['sha256']='c'*64
        with pytest.raises(ValueError):mix.require_candidate_records(original,bad)
    for bad in ({**refreshed,'new':dict(path='new')},{k:v for k,v in refreshed.items() if k!=kept[0]}):
        with pytest.raises(ValueError):mix.require_candidate_records(original,bad)


def test_normal_finish_dispatch_marks_legacy_not_native_fifty(planned,tmp_path,monkeypatch):
    import finish_staged_visual as finish
    import prepare_web_media as web
    english,clock,_=planned;stage=tmp_path/'stage';root=stage/'production'/mix.LESSON;repo=tmp_path/'repo'
    write(root/'lesson.en.json',english);write(repo/'tools/tutorials/lessons'/str(mix.LESSON+'.json'),english)
    write(root/'scenes.json',dict(kind=mix.KIND,scenes=[dict(narration=s['narration']) for s in english['scenes']]))
    write(root/mix.RECORD,{});write(root/'video/original-visual-timings.json',clock)
    master=root/'video'/str(mix.LESSON+'_silent.mp4');master.write_bytes(b'unit-only mocked encoder result')
    publisher=repo/'tools/tutorials/authoring/tools/publish_tutorials.py';publisher.parent.mkdir(parents=True);publisher.write_text('ENCODE_ARGS=[]\n')
    calls=[]
    monkeypatch.setattr(finish,'REPO',repo)
    monkeypatch.setattr(mix,'checked_legacy_picture_timing',lambda s,i:(calls.append((s,i)) or root/'video/original-visual-timings.json'))
    monkeypatch.setattr(finish,'checked_native_timing',lambda *a:pytest.fail('Legacy branch must not request native validation'))
    monkeypatch.setattr(finish,'probe',lambda p:dict(streams=[dict(codec_type='video',width=3840,height=2160,r_frame_rate='30/1',nb_frames='66')]))
    monkeypatch.setattr(finish,'timestamps',lambda p:[i/30 for i in range(66)])
    monkeypatch.setattr(finish.subprocess,'run',lambda *a,**k:None)
    monkeypatch.setattr(web,'main',lambda:(root/'poster.jpg').write_bytes(b'unit poster'))
    monkeypatch.setattr(finish,'stage_one',lambda *a,**k:dict(accepted=True))
    result=finish.finish(stage,mix.LESSON)
    assert calls==[(stage,mix.LESSON)] and result['native_fifty_voice_timeline'] is False
    assert result['published_legacy_picture_references'] is True and result['retained_narration_offerings']==27
    assert result['fresh_capture'] is False and result['raw_capture_revalidated'] is False
    write(root/'published-native-mix.json',{})
    with pytest.raises(ValueError):finish.finish(stage,mix.LESSON)


def test_normal_poster_dispatch_cannot_fall_back_to_unbound_images(tmp_path,monkeypatch):
    import prepare_web_media as web
    root=tmp_path/'production'/mix.LESSON;write(root/'scenes.json',dict(kind=mix.KIND,scenes=[]));(root/'video').mkdir();(root/'video'/str(mix.LESSON+'_silent.mp4')).write_bytes(b'unit invalid master')
    monkeypatch.setattr(web,'PRODUCTION',tmp_path/'production');monkeypatch.setattr(sys,'argv',['prepare_web_media','--lessons',mix.LESSON])
    def reject(*args):raise ValueError('actual legacy source must validate first')
    monkeypatch.setattr(mix,'checked_legacy_picture_timing',reject)
    with pytest.raises(ValueError,match='actual legacy source'):web.main()


@pytest.mark.parametrize('poster_changed_after_finish',[False,True])
def test_candidate_builder_copies_original_clock_catalogs_and_audio_without_native_conversion(published,monkeypatch,poster_changed_after_finish):
    # Synthetic receipt/probes below test the consumer wiring, not real media acceptance.
    base,stage,repo=published;mix.prepare(stage,base);root=stage/'production'/mix.LESSON
    master=root/'video'/str(mix.LESSON+'_silent.mp4');master.write_bytes(b'unit revised master')
    poster=root/'poster.jpg';poster.write_bytes(b'unit revised poster')
    web=stage/'web-renditions'/mix.LESSON/'video'/master.name;web.parent.mkdir(parents=True);web.write_bytes(b'unit revised web')
    clock=root/'video/original-visual-timings.json'
    write(root/'current-video-acceptance.json',dict(master_full_decode_passed=True,master_all_presentation_times_checked=True,master_frame_count_matches_scene_boundaries=True,published_legacy_picture_references=True,native_fifty_voice_timeline=False,master_sha256=mix.digest(master),timing_sha256=mix.digest(clock),poster_sha256=mix.digest(poster)))
    write(web.parent.parent/'rendition-checks.json',dict(accepted=True,full_decode_passed=True,all_frame_presentation_times_match=True,master_sha256=mix.digest(master),rendition_sha256=mix.digest(web)))
    calls=[]
    monkeypatch.setattr(mix,'checked_legacy_picture_timing',lambda s,i:(calls.append((s,i)) or clock))
    contract=root/'reference-cards/cards.json';write(contract,dict(unit_fixture=True))
    checked=[];monkeypatch.setattr(mix,'validate',lambda target,**options:checked.append((target,options)))
    if poster_changed_after_finish:
        poster.write_bytes(b'changed after normal finish')
        with pytest.raises(ValueError,match='different media/clock'):mix.build_candidate(stage)
        assert not list(stage.glob('release-candidate-legacy-pictures-*')),'Reject before creating a candidate'
        return
    candidate=mix.build_candidate(stage)
    before=mix.read(base/'release-manifest.json');after=mix.read(candidate/'release-manifest.json')
    changed=[r['path'] for r in after['files'] if next(v for v in before['files'] if v['path']==r['path'])!=r]
    assert set(changed)=={f'media_host/{mix.LESSON}/video/{master.name}',f'media_host/{mix.LESSON}/web/{master.name}',f'web/production/{mix.LESSON}/poster.jpg'}
    assert checked==[(candidate,dict(include_hosted_media=True))] and calls==[(stage,mix.LESSON)]*2
    assert after['legacy_picture_refresh'] is True and after['fresh_capture'] is False
    assert after['legacy_picture_references']['card_slots']==[2,3,4]
    for record in before['files']:
        if record['path'] not in changed:assert (candidate/record['path']).read_bytes()==(base/record['path']).read_bytes()
    assert (candidate/'web/lesson_catalog.js').read_bytes()==(base/'web/lesson_catalog.js').read_bytes()


def test_incomplete_publication_readback_refuses_stage_creation(published):
    base,stage,_=published;receipt=mix.read(base/'publication-receipt.json');receipt['readback']['passed']=False;write(base/'publication-receipt.json',receipt)
    with pytest.raises(ValueError,match='readback'):mix.prepare(stage,base)
    assert not stage.exists()


@pytest.mark.parametrize('fault',[None,'png','narration','slots','helper'])
def test_reference_contract_binds_png_narration_slots_and_linked_helper(saved_sources,planned,tmp_path,fault):
    from PIL import Image
    sources,_=saved_sources;english,_,_=planned;folder=tmp_path/'cards';folder.mkdir();shutil.copytree(sources,folder/'source-results')
    repo=tmp_path/'repo';helper=repo/'tools/tutorials/prepare_classify_split.py';helper.parent.mkdir(parents=True);shutil.copy2(REPO/'tools/tutorials/prepare_classify_split.py',helper)
    style=repo/'tools/tutorials/authoring/tools/render_install_keyframes.py';style.parent.mkdir(parents=True);style.write_text('# unit-only style binding')
    rows=[]
    for visual in cards.CARDS:
        path=folder/(visual+'.png');Image.new('RGB',(3840,2160),(12,30,44)).save(path)
        scene=next(a for a in english['scenes'] if a['visual']==visual)
        rows.append(dict(visual=visual,image=path.name,sha256=cards.digest(path),kind='authored_reference_card',narration_sha256=cards.identity(scene['narration'])))
    contract=dict(schema=1,lesson=mix.LESSON,english_sha256=cards.identity(english),definitions_sha256=cards.identity(cards.definitions()),generator_sha256=cards.digest(cards.__file__),style_helper_sha256=cards.digest(style),source_hashes=cards.SOURCES,cards=rows)
    if fault=='png':(folder/rows[0]['image']).write_bytes(b'tampered')
    elif fault=='narration':rows[0]['narration_sha256']='changed'
    elif fault=='slots':rows[0]['visual']=cards.VISUALS[0]
    elif fault=='helper':helper.write_text('changed helper')
    write(folder/'cards.json',contract)
    if fault is None:assert set(cards.validate_cards(folder,english,repo))==set(cards.CARDS)
    else:
        with pytest.raises(ValueError):cards.validate_cards(folder,english,repo)
