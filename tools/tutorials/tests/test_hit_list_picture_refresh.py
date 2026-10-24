"""Synthetic contracts/frames test the repair; they are not capture/science proof."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,shutil,subprocess,sys,zipfile
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'authoring/tools')]
import hit_list_picture_refresh as picture
import render_hit_list_picture_references as refs
from stage_lesson import REPO,write
from append_staged_lessons import JS_PREFIX,JS_SUFFIX

def current_english():
 value=json.loads((REPO/'tools/tutorials/lessons/48_hit_list.json').read_text())
 if picture.identity(value)==picture.ORIGINAL_ENGLISH:
  value['prerequisite']=value['prerequisite'].replace('The current Regression Hits shortcut does not reveal the panel; run python hit_list_companion.py and Browse to the extracted results folder instead.','If the Regression Hits shortcut does not reveal the panel, run python hit_list_companion.py and Browse to the extracted results folder.')
  value['scenes'][1]['narration']=value['scenes'][1]['narration'].replace('Extract the linked example and activate your spaCR environment.','To use the fallback, extract the linked example and activate your spaCR environment.',1)
 assert picture.identity(value)==picture.CURRENT_ENGLISH
 return value

def original_clock(old):
 spans=[(b-a)/30 for a,b in zip(picture.EDGES,picture.EDGES[1:])];elapsed=0.0;scenes=[]
 for i,span in enumerate(spans):
  end=elapsed+span;scenes.append(dict(scene=i+1,text=old['scenes'][i]['narration'],speech_start=elapsed,speech_end=end-.01,scene_end=end,duration=span));elapsed=end
 return dict(schema=1,total_duration=elapsed,scenes=scenes)

def test_exact_partition_preserves_every_moving_frame_and_only_references_first_chapter():
 current=current_english();clock=original_clock(picture.original_english(current));before=deepcopy(clock)
 rows=picture.plan(current,clock,{'route':{},'header':{}})
 assert len(rows)==11 and rows[0]['picture']=='authored_route_reference' and rows[0]['frames']==272
 assert sum(r['frames'] for r in rows)==4144 and clock==before
 assert [(r['source_start_frame'],r['source_end_frame']) for r in rows]==list(zip(picture.EDGES,picture.EDGES[1:]))
 assert all(r['picture']=='original_moving_header_overlay' and r['header_rectangle']==[0,0,3840,72] and r['playback_speed']==1 for r in rows[1:])
 assert not any('native_live_video' in r for r in rows)

@pytest.mark.parametrize('fault',['english','visual','scene2_suffix','limit','clock_kind','clock_gap','clock_index','clock_text','frames','extra_reference'])
def test_scope_sourceclock_and_unchanged_text_fail_closed(fault):
 current=current_english();clock=original_clock(picture.original_english(current));images={'route':{},'header':{}}
 if fault=='english':current['title']='changed'
 elif fault=='visual':current['scenes'][1]['visual']='other'
 elif fault=='scene2_suffix':current['scenes'][1]['narration']+=' Changed.'
 elif fault=='limit':current['prerequisite']=current['prerequisite'].replace('The companion does not connect Investigate. ','')
 elif fault=='clock_kind':clock['kind']='native_visual_timing'
 elif fault=='clock_gap':clock['scenes'][1]['speech_start']+=.1
 elif fault=='clock_index':clock['scenes'][2]['scene']=True
 elif fault=='clock_text':clock['scenes'][1]['text']=current['scenes'][1]['narration'] # old visual clock must retain OLD text
 elif fault=='frames':clock['scenes'][-1]['duration']+=.1;clock['scenes'][-1]['scene_end']+=.1;clock['total_duration']+=.1
 else:images['still_chapter2']={}
 with pytest.raises(ValueError):picture.plan(current,clock,images)

@pytest.fixture
def staged(tmp_path,monkeypatch):
 import build_audio_patch_candidate
 current=current_english();old=picture.original_english(current);clock=original_clock(old)
 base=tmp_path/'published';stage=tmp_path/'stage';repo=tmp_path/'repo';root=stage/'production'/picture.LESSON
 write(root/'lesson.en.json',current);write(repo/'tools/tutorials/lessons/48_hit_list.json',current)
 from check_completed_matrix import voice_matrix
 declared=voice_matrix(REPO/'tools/tutorials/authoring/tools/render_all_voices.py');declared['en']=['af_heart']
 write(base/'web/catalog/lessons_en.json',dict(lessons=[{**old,'narration_voices':declared}]))
 runtime=dict(lessons=[dict(id=picture.LESSON,silent='48_hit_list/video/48_hit_list_silent.mp4',web='48_hit_list/web/48_hit_list_silent.mp4')])
 (base/'web/lesson_catalog.js').write_text(JS_PREFIX+json.dumps(runtime)+JS_SUFFIX)
 write(base/'media_host/48_hit_list/audio/en/af_heart.json',clock)
 for role in ('video','web'):
  p=base/f'media_host/48_hit_list/{role}/48_hit_list_silent.mp4';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(('synthetic '+role).encode())
 poster=base/'web/production/48_hit_list/poster.jpg';poster.parent.mkdir(parents=True);poster.write_bytes(b'synthetic old poster')
 archive=base/'web/examples/Hit_List_real_results_companion.zip';archive.parent.mkdir(parents=True)
 with zipfile.ZipFile(archive,'w') as z:z.writestr('hit_list_companion.py',b'synthetic launcher')
 proof=repo/'tools/tutorials/evidence/2026-09-11_hit_list_companion_verification.json';proof.parent.mkdir(parents=True);proof.write_text('synthetic fixture; not scientific acceptance')
 rows=[dict(path=p.relative_to(base).as_posix(),sha256=picture.digest(p),bytes=p.stat().st_size) for p in base.rglob('*') if p.is_file()]
 write(base/'release-manifest.json',dict(files=rows));write(base/'publication-receipt.json',dict(synthetic_unit_fixture=True))
 monkeypatch.setattr(build_audio_patch_candidate,'require_published_script_baseline',lambda *a:None) # existing normal receipt guard has separate tests
 monkeypatch.setattr(picture,'REPO',repo)
 for name,path in [('MASTER',base/'media_host/48_hit_list/video/48_hit_list_silent.mp4'),('WEB',base/'media_host/48_hit_list/web/48_hit_list_silent.mp4'),('ORIGINAL_CLOCK',base/'media_host/48_hit_list/audio/en/af_heart.json'),('ZIP',archive),('SCIENTIFIC',proof)]:monkeypatch.setattr(picture,name,picture.digest(path))
 monkeypatch.setattr(picture,'LAUNCHER',hashlib.sha256(b'synthetic launcher').hexdigest())
 return stage,base,repo

def test_prepare_keeps_explicit_original_visual_clock_without_touching_current_audio(staged):
 stage,base,_=staged;root=stage/'production'/picture.LESSON
 audio=root/'audio/en/af_heart.json';write(audio,dict(current_audio_separately_checked=True));before=audio.read_bytes()
 picture.prepare(stage,base);source,current,clock=picture.checked_source(stage)
 assert (root/picture.CLOCK).read_bytes()==(base/source['original_clock']['path']).read_bytes()
 assert audio.read_bytes()==before and source['audio_validated'] is False and source['science_recomputed'] is False
 assert source['fresh_capture'] is False and source['raw_capture_revalidated'] is False
 with pytest.raises(FileExistsError):picture.prepare(stage,base)

@pytest.mark.parametrize('fault',['master','web','oldclock','zip','receipt','record','current','canonical'])
def test_source_drift_refuses_even_without_audio_decode(staged,fault):
 stage,base,repo=staged;picture.prepare(stage,base);root=stage/'production'/picture.LESSON
 if fault=='master':p=base/'media_host/48_hit_list/video/48_hit_list_silent.mp4'
 elif fault=='web':p=base/'media_host/48_hit_list/web/48_hit_list_silent.mp4'
 elif fault=='oldclock':p=root/picture.CLOCK
 elif fault=='zip':p=base/'web/examples/Hit_List_real_results_companion.zip'
 elif fault=='receipt':p=repo/'tools/tutorials/evidence/2026-09-11_hit_list_companion_verification.json'
 elif fault=='record':p=root/picture.SOURCE
 elif fault=='current':p=root/'lesson.en.json'
 else:p=repo/'tools/tutorials/lessons/48_hit_list.json'
 p.write_bytes(b'changed')
 with pytest.raises((ValueError,json.JSONDecodeError)):picture.checked_source(stage)

@pytest.fixture
def references(staged,monkeypatch):
 from PIL import Image
 stage,base,repo=staged;picture.prepare(stage,base);source,current,clock=picture.checked_source(stage)
 folder=stage/'production'/picture.LESSON/'picture-references';(folder/'source-frames').mkdir(parents=True);hashes=[]
 for i in range(1,12):
  p=folder/'source-frames'/f'chapter-{i:03}.png';Image.new('RGB',(2560,1440),(i,2,3)).save(p);hashes.append(picture.digest(p))
 monkeypatch.setattr(picture,'FRAME_HASHES',tuple(hashes))
 style=repo/'tools/tutorials/authoring/tools/render_install_keyframes.py';style.parent.mkdir(parents=True);style.write_text('# synthetic style')
 images={}
 for key,name,size in [('route','scene-001--route.png',(3840,2160)),('header','header.png',(3840,72))]:
  Image.new('RGB',size,(5,9,16)).save(folder/name)
  images[key]=dict(image=name,sha256=picture.digest(folder/name),size=list(size),kind='authored_route_reference' if key=='route' else 'authored_feature_header')
 write(folder/'references.json',{**refs.contract(current,source,repo),'images':images})
 return stage,folder,current,source,repo

@pytest.mark.parametrize('fault',[None,'source_png','header_bytes','route_bytes','header_height','header_alpha','extra_image','roi','wording','generator_binding'])
def test_header_geometry_and_reference_sources_are_exact(references,fault):
 from PIL import Image
 stage,folder,current,source,repo=references;value=json.loads((folder/'references.json').read_text())
 if fault=='source_png':(folder/'source-frames/chapter-009.png').write_bytes(b'changed')
 elif fault=='header_bytes':(folder/'header.png').write_bytes(b'changed')
 elif fault=='route_bytes':(folder/'scene-001--route.png').write_bytes(b'changed')
 elif fault in ('header_height','header_alpha'):
  Image.new('RGBA' if fault=='header_alpha' else 'RGB',(3840,74 if fault=='header_height' else 72)).save(folder/'header.png')
  value['images']['header']['sha256']=picture.digest(folder/'header.png')
 elif fault=='extra_image':value['images']['chapter2']={}
 elif fault=='roi':value['definitions_sha256']='changed ROI'
 elif fault=='wording':value['english_sha256']='changed'
 elif fault=='generator_binding':value['generator_sha256']='changed'
 write(folder/'references.json',value)
 if fault is None:assert set(refs.validate_references(folder,current,source,repo))=={'route','header'}
 else:
  with pytest.raises(ValueError):refs.validate_references(folder,current,source,repo)

def test_encoder_retains_full_tail_once_and_never_overwrites_source(tmp_path,monkeypatch):
 import render_visual_master
 calls=[];source=tmp_path/'source.mp4';source.write_bytes(b'synthetic media placeholder')
 monkeypatch.setattr(render_visual_master,'encode_still',lambda *a:calls.append(('still',a)))
 monkeypatch.setattr(render_visual_master,'concat',lambda *a:calls.append(('concat',a)))
 monkeypatch.setattr(picture.subprocess,'run',lambda cmd,**kw:calls.append(('moving',cmd,kw)))
 picture.encode_parts(source,tmp_path/'route.png',tmp_path/'header.png',tmp_path/'work',tmp_path/'target.mp4')
 command=next(c[1] for c in calls if c[0]=='moving');filters=command[command.index('-filter_complex')+1]
 assert filters==picture.moving_filter()=='[0:v]trim=start_frame=272:end_frame=4144,settb=1/30,setpts=N[body];[1:v]format=yuva420p[header];[body][header]overlay=x=0:y=0:shortest=1:format=yuv420[out]'
 assert command[command.index('-frames:v')+1]=='3872' and next(c[1][1] for c in calls if c[0]=='still')==272/30
 assert not any(x in filters for x in ('scale','crop','fps','tpad','select'))
 with pytest.raises(FileExistsError):picture.encode_parts(source,tmp_path/'r',tmp_path/'h',tmp_path/'another',source)
 assert source.read_bytes()==b'synthetic media placeholder'


def test_tiny_real_encode_preserves_each_moving_body_frame_and_only_changes_header(tmp_path,monkeypatch):
 from PIL import Image
 # Encoder-only fixture dimensions; actual source checker remains fixed4K/4144.
 monkeypatch.setattr(picture,'EDGES',(0,6,36));monkeypatch.setattr(picture,'RECT',(0,0,64,8))
 def image(value):
  im=Image.new('RGB',(64,48),(20,20,20))
  for bit in range(6):im.paste((235,235,235) if value&(1<<bit) else (20,20,20),(bit*8,8,bit*8+8,48))
  return im
 source=tmp_path/'source.mp4';raw=b''.join(image(i).tobytes() for i in range(36))
 subprocess.run(['ffmpeg','-v','error','-n','-f','rawvideo','-pixel_format','rgb24','-video_size','64x48','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-threads','2','-crf','12','-pix_fmt','yuv420p',str(source)],input=raw,check=True)
 before=picture.digest(source);route=tmp_path/'route.png';image(60).save(route);header=tmp_path/'header.png';Image.new('RGB',(64,8),(200,40,40)).save(header)
 # Compare the exact selected source YUV planes with the product's shared
 # filter graph BEFORE any second encode. No padding/skipped rows or frames.
 original=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(source),'-vf','trim=start_frame=6:end_frame=36,setpts=PTS-STARTPTS','-frames:v','30','-f','rawvideo','-pix_fmt','yuv420p','pipe:1'])
 filtered=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(source),'-loop','1','-framerate','30','-i',str(header),'-filter_complex_threads','2','-filter_complex',picture.moving_filter(),'-map','[out]','-frames:v','30','-f','rawvideo','-pix_fmt','yuv420p','pipe:1'])
 frame_bytes=64*48*3//2
 assert len(original)==len(filtered)==30*frame_bytes
 # ROI covers8 luma rows /4 rows in each half-resolution chroma plane.
 # Compare ALL bytes outside it in Y,U,V on EVERY retained moving frame.
 planes=[(0,64,48,8),(64*48,32,24,4),(64*48+32*24,32,24,4)]
 for index in range(30):
  start=index*frame_bytes
  for offset,width,height,covered_rows in planes:
   a=start+offset+covered_rows*width;b=start+offset+height*width
   assert original[a:b]==filtered[a:b],(index,offset)
  assert original[start:start+64*8]!=filtered[start:start+64*8]
 target=tmp_path/'result.mp4';picture.encode_parts(source,route,header,tmp_path/'work',target)
 decoded=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(target),'-f','rawvideo','-pix_fmt','rgb24','pipe:1']);size=64*48*3
 assert len(decoded)==36*size
 values=[sum((decoded[(i*48*64+30*64+bit*8+4)*3]>128)<<bit for bit in range(6)) for i in range(36)]
 assert values==[60]*6+list(range(6,36)) and picture.digest(source)==before
 # Header is replaced over every retained moving frame, body still carries each index.
 assert all(decoded[(i*48*64+3*64+30)*3]>150 for i in range(6,36))

@pytest.mark.parametrize('fault',[None,'poster','master','web','timing','native_claim','missing_full_decode'])
def test_finished_visual_media_requires_actual_normal_master_web_poster_bindings(tmp_path,monkeypatch,fault):
 stage=tmp_path/'stage';root=stage/'production'/picture.LESSON;root.mkdir(parents=True)
 paths={}
 for key,rel in [('master','production/48_hit_list/video/48_hit_list_silent.mp4'),('web','web-renditions/48_hit_list/video/48_hit_list_silent.mp4'),('poster','production/48_hit_list/poster.jpg'),('original_clock','production/48_hit_list/video/original-visual-timings.json')]:
  p=stage/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(('synthetic '+key).encode());paths[key]=p
 monkeypatch.setattr(picture,'checked_picture_timing',lambda s:paths['original_clock'])
 acceptance=dict(master_full_decode_passed=True,master_all_presentation_times_checked=True,master_frame_count_matches_scene_boundaries=True,published_legacy_hit_list_header=True,native_fifty_voice_timeline=False,fresh_capture=False,raw_capture_revalidated=False,master_sha256=picture.digest(paths['master']),timing_sha256=picture.digest(paths['original_clock']),poster_sha256=picture.digest(paths['poster']))
 rendition=dict(accepted=True,full_decode_passed=True,all_frame_presentation_times_match=True,master_sha256=picture.digest(paths['master']),rendition_sha256=picture.digest(paths['web']))
 if fault in ('poster','master','web'):paths[fault].write_bytes(b'changed after finish')
 elif fault=='timing':paths['original_clock'].write_bytes(b'changed clock')
 elif fault=='native_claim':acceptance['native_fifty_voice_timeline']=True
 elif fault=='missing_full_decode':rendition['full_decode_passed']=False
 write(root/'current-video-acceptance.json',acceptance);write(stage/'web-renditions/48_hit_list/rendition-checks.json',rendition)
 if fault is None:assert picture.checked_finished_media(stage)==paths
 else:
  with pytest.raises(ValueError):picture.checked_finished_media(stage)

def test_normal_finish_dispatch_uses_old_visual_clock_and_does_not_claim_retained_audio(tmp_path,monkeypatch):
 import finish_staged_visual as finish
 import prepare_web_media as web
 current=current_english();clock=original_clock(picture.original_english(current));stage=tmp_path/'stage';root=stage/'production'/picture.LESSON;repo=tmp_path/'repo'
 write(root/'lesson.en.json',current);write(repo/'tools/tutorials/lessons/48_hit_list.json',current)
 write(root/'scenes.json',dict(kind=picture.KIND,scenes=[dict(narration=s['narration']) for s in current['scenes']]))
 write(root/picture.RECORD,{});write(root/picture.CLOCK,clock);master=root/'video/48_hit_list_silent.mp4';master.write_bytes(b'synthetic master')
 publisher=repo/'tools/tutorials/authoring/tools/publish_tutorials.py';publisher.parent.mkdir(parents=True);publisher.write_text('ENCODE_ARGS=[]\n')
 calls=[];monkeypatch.setattr(picture,'checked_picture_timing',lambda s:(calls.append(s) or root/picture.CLOCK))
 monkeypatch.setattr(finish,'REPO',repo);monkeypatch.setattr(finish,'checked_native_timing',lambda *a:pytest.fail('No native conversion'))
 monkeypatch.setattr(finish,'probe',lambda p:dict(streams=[dict(codec_type='video',width=3840,height=2160,r_frame_rate='30/1',nb_frames='4144')]))
 monkeypatch.setattr(finish,'timestamps',lambda p:[i/30 for i in range(4144)])
 monkeypatch.setattr(finish.subprocess,'run',lambda *a,**k:None)
 monkeypatch.setattr(web,'main',lambda:(root/'poster.jpg').write_bytes(b'synthetic normal poster'))
 monkeypatch.setattr(finish,'stage_one',lambda *a,**k:dict(accepted=True))
 result=finish.finish(stage,picture.LESSON)
 assert calls==[stage] and result['published_legacy_hit_list_header'] is True
 assert result['native_fifty_voice_timeline'] is False and result['audio_validated'] is False
 assert 'retained_narration_offerings' not in result
 assert result['timing_sha256']==picture.digest(root/picture.CLOCK)
 write(root/'published-legacy-picture-mix.json',{})
 with pytest.raises(ValueError,match='explicit48'):finish.finish(stage,picture.LESSON)


def test_normal_poster_dispatch_rejects_unchecked48_before_any_extraction(tmp_path,monkeypatch):
 import prepare_web_media as web
 root=tmp_path/'production'/picture.LESSON;write(root/'scenes.json',dict(kind=picture.KIND,scenes=[]));(root/'video').mkdir();(root/'video/48_hit_list_silent.mp4').write_bytes(b'synthetic invalid master')
 monkeypatch.setattr(web,'PRODUCTION',tmp_path/'production');monkeypatch.setattr(sys,'argv',['prepare_web_media','--lessons',picture.LESSON])
 def reject(*args):raise ValueError('Actual48 picture source must validate first')
 monkeypatch.setattr(picture,'checked_picture_timing',reject)
 with pytest.raises(ValueError,match='Actual48 picture source'):web.main()
