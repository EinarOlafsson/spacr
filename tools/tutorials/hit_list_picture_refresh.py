"""Explicit legacy48 picture repair; source movement retained outside a fixed header.

Current audio/reviews belong to the separate explicit mixed-audio validator. This
module validates visuals only and must not be used as audio/publication proof.
"""
from pathlib import Path
from copy import deepcopy
import argparse,json,math,os,shutil,subprocess,sys
from stage_lesson import REPO,read,write
from check_completed_matrix import digest
LESSON='48_hit_list'
KIND='published_legacy_hit_list_header'
SOURCE='published-hit-list-picture-source.json'
RECORD='published-hit-list-picture-refresh.json'
CLOCK='video/original-visual-timings.json'
CURRENT_ENGLISH='98a0eaaf0bb38a6b0d2f7bead3b40748f9b4f8b544e9192492f538f27a5529ec'
ORIGINAL_ENGLISH='725dbe48aa9ba7956c69806111ba283950734449a30162bf99095a7ff2109f87'
MASTER='66b4363d45bcb9159f5f98719d9f67214a51b8a0771322500eeb183ddac7d6ad'
WEB='6a6e0ff3fd5e834f60742c201438e06cc43e1cb2c6b0996432acdf4b42eedca2'
ORIGINAL_CLOCK='0e6bd875ecc9443d46ea77c1c9f73570fe08bd3c99a69ba1318419c8e9bf1949'
SCIENTIFIC='1fdbe4925928105d550688525df0041c1dfcb2d9391969195dc00ecc4994b2ff'
ZIP='ab1c68e0f2293e5a6a90df180d8ec65b3dc5ba165b72265bf8e5fb54846d6dbb'
LAUNCHER='33280fce3db6e86d95681189876f9e75696f4cd7fe7bca9cb1e96d2322f0863d'
VISUALS=('01b_hit_list_route_failure','01c_explicit_companion_not_native_route','03_all_325_ranked_genes_not_discoveries','04_no_genes_pass_q_point05','06_at_least_two_tested_guides','07_all_tested_guides_same_sign','08_hide_controls_not_recalculate_q','10_positive_effect_magnitude_cut','13_real_gene_returns','17_export_html_complete','18_original_ranked_population_restored')
EDGES=(0,272,628,1202,1519,1806,2206,2500,2892,3176,3665,4144)
RECT=(0,0,3840,72)
FRAME_HASHES=('878b5ce6e4b1d03390eacd4d2eb13540be4566604f00a32f66f12f3c749eaee8','4dbffb7273297f1055e70ca7126ef43b9178e156bacfa2bbc22940dc5be50dbe','437314d395c57d606508d27ee9d7fabb00549d844e6a0027b5d3b1237eced529','5fd0242ca5e172c3057a3f6f97c68d2c83fac76dcd490b71794e7606ef298437','32bc487daca3a8823ac4fb40f2b01320766c9f29b4a1fb8dd29e4a5a5d2cd237','8d46c7dbd276bd0f1a586ad047acb99eefdf92b5f3f8a6200acfd3d92351c25c','19cff64d47a48c7c679b529382f1f06e28d996103dd0c16eefb5a7f0713a0b36','36b2d3f3997b3c860870b78a20ff749e15fa5333d06138afbe58b38926aa8ed4','67978ab915fc10c4e0ac2437b2404da718d43a695364852bcb58a97af7b04b79','393194a6f054f9c1fec20d345b5f0516ee5c122a2fb29a5adbca009191bbd29f','aa9e79329d0f6489d38619016dfdca28ac62ea2cc32b16acbbd7e5c317ef3d42')

def identity(value):
 import hashlib
 return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def regular(root,relative):
 from published_legacy_picture_mix import regular as checked
 return checked(root,relative)
def checked_english(english):
 if identity(english)!=CURRENT_ENGLISH or english.get('id')!=LESSON or tuple(s.get('visual') for s in english.get('scenes',[]))!=VISUALS:raise ValueError('Only exact approved48 authored source')
def original_english(current):
 checked_english(current);old=deepcopy(current)
 old['prerequisite']=old['prerequisite'].replace('If the Regression Hits shortcut does not reveal the panel, run python hit_list_companion.py and Browse to the extracted results folder.','The current Regression Hits shortcut does not reveal the panel; run python hit_list_companion.py and Browse to the extracted results folder instead.',1)
 old['scenes'][1]['narration']=old['scenes'][1]['narration'].replace('To use the fallback, extract the linked example and activate your spaCR environment.','Extract the linked example and activate your spaCR environment.',1)
 if identity(old)!=ORIGINAL_ENGLISH:raise ValueError('Original48 authored binding differs')
 return old

def checked_clock(clock,english):
 sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'))
 from render_visual_master import frame_aligned_durations
 if clock.get('schema')!=1 or clock.get('kind') or clock.get('native_live_video') or len(clock.get('scenes',[]))!=11:raise ValueError('Only original11 legacy chapters')
 edges=[0];elapsed=0.0
 for index,(scene,span) in enumerate(zip(clock['scenes'],frame_aligned_durations(clock['scenes'],30)),1):
  nums=[scene.get(k) for k in ('speech_start','speech_end','scene_end','duration')]
  if any(type(n) not in (int,float) or not math.isfinite(n) for n in nums):raise ValueError('Finite legacy clock required')
  start,speech_end,end,duration=nums
  if (type(scene.get('scene')) is not int or scene['scene']!=index or scene.get('native_live_video') or scene.get('text')!=english['scenes'][index-1]['narration'] or abs(start-elapsed)>1e-6 or not start<speech_end<=end or abs(end-start-duration)>1e-6):raise ValueError('Original indexed clock/text differs')
  elapsed=end;edges.append(edges[-1]+round(span*30))
 if tuple(edges)!=EDGES or type(clock.get('total_duration')) not in (int,float) or not math.isfinite(clock['total_duration']) or abs(clock['total_duration']-elapsed)>1e-6:raise ValueError('Original4144-frame partition differs')
 return edges

def published_source(baseline,current):
 from build_audio_patch_candidate import require_published_script_baseline,legacy_visual_source
 from retain_narration import catalog_lesson,require_unchanged_lesson
 baseline=Path(baseline).resolve();manifest=read(regular(baseline,'release-manifest.json'));require_published_script_baseline(baseline,manifest)
 records={r['path']:r for r in manifest['files']}
 if len(records)!=len(manifest['files']):raise ValueError('Duplicate published paths')
 def bound(relative):
  p=regular(baseline,relative);row=records.get(relative)
  if not row or digest(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise ValueError('Published visual/source bytes differ')
  return p
 old=original_english(current);english=catalog_lesson(bound('web/catalog/lessons_en.json'),LESSON)
 require_unchanged_lesson(old,english,english,voice_inventory=english.get('narration_voices'))
 master=f'media_host/{LESSON}/video/{LESSON}_silent.mp4';web=f'media_host/{LESSON}/web/{LESSON}_silent.mp4'
 clock=legacy_visual_source(baseline,LESSON,english);bound(clock.relative_to(baseline).as_posix());checked_clock(read(clock),old)
 if digest(bound(master))!=MASTER or digest(bound(web))!=WEB or digest(clock)!=ORIGINAL_CLOCK:raise ValueError('Only reviewed48 accepted picture/clock')
 archive=bound('web/examples/Hit_List_real_results_companion.zip')
 import zipfile
 with zipfile.ZipFile(archive) as z:
  import hashlib
  if digest(archive)!=ZIP or hashlib.sha256(z.read('hit_list_companion.py')).hexdigest()!=LAUNCHER:raise ValueError('Original example/launcher differs')
 evidence=regular(REPO,'tools/tutorials/evidence/2026-09-11_hit_list_companion_verification.json')
 if digest(evidence)!=SCIENTIFIC:raise ValueError('Accepted scientific receipt differs')
 return dict(schema=1,kind=KIND,lesson=LESSON,baseline_root=str(baseline),manifest_sha256=digest(baseline/'release-manifest.json'),publication_receipt_sha256=digest(baseline/'publication-receipt.json'),original_master=deepcopy(records[master]),original_web=deepcopy(records[web]),original_clock=dict(path=clock.relative_to(baseline).as_posix(),sha256=digest(clock)),scientific_receipt_sha256=SCIENTIFIC,example_sha256=ZIP,launcher_sha256=LAUNCHER,english_sha256=identity(current),original_english_sha256=identity(old),frame_edges=list(EDGES),header_rectangle=list(RECT),fresh_capture=False,raw_capture_revalidated=False,science_recomputed=False,audio_validated=False),read(clock)

def prepare(stage,baseline):
 if Path(stage).is_symlink() or Path(baseline).is_symlink():raise ValueError('Independent stage/baseline required')
 stage=Path(stage).resolve();root=stage/'production'/LESSON;baseline=Path(baseline).resolve()
 if stage==baseline or stage.is_relative_to(baseline) or baseline.is_relative_to(stage):raise ValueError('Never write into published source')
 current=read(regular(stage,root.relative_to(stage)/'lesson.en.json'));checked_english(current)
 if current!=read(REPO/'tools/tutorials/lessons'/f'{LESSON}.json'):raise ValueError('Current stage/canonical differs')
 source,clock=published_source(baseline,current)
 for rel in (SOURCE,CLOCK):
  if os.path.lexists(root/rel):raise FileExistsError('Preserve earlier source/visual clock')
 target=root/CLOCK;target.parent.mkdir(parents=True,exist_ok=True)
 with (baseline/source['original_clock']['path']).open('rb') as a,target.open('xb') as b:shutil.copyfileobj(a,b)
 write(root/SOURCE,source);return root/SOURCE

def checked_source(stage):
 if Path(stage).is_symlink():raise ValueError('Independent stage required')
 stage=Path(stage).resolve();root=stage/'production'/LESSON
 current=read(regular(stage,root.relative_to(stage)/'lesson.en.json'));checked_english(current)
 if current!=read(REPO/'tools/tutorials/lessons'/f'{LESSON}.json'):raise ValueError('Current stage/canonical differs')
 record=read(regular(stage,root.relative_to(stage)/SOURCE));expected,clock=published_source(record['baseline_root'],current)
 if record!=expected or digest(regular(stage,root.relative_to(stage)/CLOCK))!=ORIGINAL_CLOCK:raise ValueError('Visual source/old clock changed')
 return record,current,clock

def plan(current,clock,references):
 checked_english(current);checked_clock(clock,original_english(current))
 if set(references)!= {'route','header'}:raise ValueError('One route card and one header only')
 return [dict(scene=i+1,visual=visual,source_start_frame=EDGES[i],source_end_frame=EDGES[i+1],frames=EDGES[i+1]-EDGES[i],picture='authored_route_reference' if i==0 else 'original_moving_header_overlay',header_rectangle=None if i==0 else list(RECT),playback_speed=1) for i,visual in enumerate(VISUALS)]

def inputs(stage):
 sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'))
 from render_hit_list_picture_references import validate_references
 source,current,clock=checked_source(stage);root=Path(stage)/'production'/LESSON
 refs=validate_references(root/'picture-references',current,source,REPO)
 rows=plan(current,clock,refs)
 spec=dict(lesson=LESSON,kind=KIND,size=[3840,2160],fps=30,scenes=[dict(visual=s['visual'],narration=s['narration'],published_picture=row) for s,row in zip(current['scenes'],rows)],references=refs)
 return source,current,clock,rows,spec

def moving_filter():
 # Strict yuv420p source plus an opaque header converted on its own branch.
 # Overlay copies the base Y/U/V bytes outside the aligned rectangle; there
 # is no RGB conversion of the base frame. Final H.264 encoding remains lossy.
 return f'[0:v]trim=start_frame={EDGES[1]}:end_frame={EDGES[-1]},settb=1/30,setpts=N[body];[1:v]format=yuva420p[header];[body][header]overlay=x=0:y=0:shortest=1:format=yuv420[out]'

def encode_parts(source,route,header,work,target):
 sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'))
 from render_visual_master import encode_still,concat
 source,work,target=Path(source),Path(work),Path(target)
 if any(os.path.lexists(p) for p in (work,target)) or target.resolve()==source.resolve():raise FileExistsError('Preserve source and old renders')
 work.mkdir();target.parent.mkdir(parents=True,exist_ok=True)
 first=work/'route.mp4';tail=work/'moving.mp4';encode_still(Path(route),EDGES[1]/30,first,30)
 # Original absolute frames272..4143, each retained once. Opaque72px header only;
 # no crop, scaling, speed change, pause, discarded body frame or frozen tail.
 filters=moving_filter()
 subprocess.run(['ffmpeg','-nostdin','-n','-v','error','-threads','2','-i',str(source),'-loop','1','-framerate','30','-i',str(header),'-filter_complex_threads','2','-filter_complex',filters,'-map','[out]','-an','-frames:v',str(EDGES[-1]-EDGES[1]),'-c:v','libx264','-preset','veryfast','-crf','18','-threads','2','-pix_fmt','yuv420p','-video_track_timescale','90000',str(tail)],check=True)
 concat([first,tail],target,work/'concat.txt')

def render(stage):
 from stage_web_renditions import probe
 stage=Path(stage).resolve();root=stage/'production'/LESSON;source,current,clock,rows,spec=inputs(stage)
 target=root/'video'/f'{LESSON}_silent.mp4';work=root/'hit-list-picture-render'
 if any(os.path.lexists(root/p) for p in (RECORD,'scenes.json')):raise FileExistsError('Preserve previous picture metadata')
 master=Path(source['baseline_root'])/source['original_master']['path'];streams=probe(master)['streams']
 if len(streams)!=1 or streams[0].get('codec_type')!='video' or [streams[0].get('width'),streams[0].get('height')]!=[3840,2160] or streams[0].get('r_frame_rate')!='30/1' or streams[0].get('pix_fmt')!='yuv420p' or int(streams[0].get('nb_frames',0))!=EDGES[-1]:raise ValueError('Original master does not fit4144-frame clock')
 encode_parts(master,root/'picture-references'/spec['references']['route']['image'],root/'picture-references'/spec['references']['header']['image'],work,target)
 if inputs(stage)!=(source,current,clock,rows,spec):raise ValueError('Picture inputs changed during render')
 write(root/'scenes.json',spec);write(root/RECORD,dict(source=source,spec_sha256=digest(root/'scenes.json'),master_sha256=digest(target)))
 return checked_picture_timing(stage)

def checked_picture_timing(stage):
 source,current,clock,rows,spec=inputs(stage);root=Path(stage)/'production'/LESSON
 if any((root/name).exists() for name in ('published-native-reuse.json','published-native-edit.json','published-native-mix.json','published-legacy-picture-mix.json')):raise ValueError('Explicit legacy overlay must not coexist with native/other visual claims')
 record=read(regular(stage,root.relative_to(stage)/RECORD));master=regular(stage,root.relative_to(stage)/'video'/f'{LESSON}_silent.mp4')
 if record!=dict(source=source,spec_sha256=digest(root/'scenes.json'),master_sha256=digest(master)) or read(root/'scenes.json')!=spec:raise ValueError('Picture render record differs')
 return root/CLOCK

def checked_finished_media(stage):
 """Visual half only; a combined builder MUST independently check current audio."""
 stage=Path(stage);root=stage/'production'/LESSON;clock=checked_picture_timing(stage)
 master=regular(stage,root.relative_to(stage)/'video'/f'{LESSON}_silent.mp4');poster=regular(stage,root.relative_to(stage)/'poster.jpg')
 web=regular(stage,f'web-renditions/{LESSON}/video/{LESSON}_silent.mp4')
 acceptance=read(root/'current-video-acceptance.json');rendition=read(stage/f'web-renditions/{LESSON}/rendition-checks.json')
 for flag in ('master_full_decode_passed','master_all_presentation_times_checked','master_frame_count_matches_scene_boundaries','published_legacy_hit_list_header'):
  if acceptance.get(flag) is not True:raise ValueError('Actual normal overlay finish required')
 if acceptance.get('native_fifty_voice_timeline') is not False or acceptance.get('fresh_capture') is not False or acceptance.get('raw_capture_revalidated') is not False or acceptance.get('master_sha256')!=digest(master) or acceptance.get('timing_sha256')!=digest(clock) or acceptance.get('poster_sha256')!=digest(poster):raise ValueError('Finish master/oldclock/poster binding differs')
 if any(rendition.get(k) is not True for k in ('accepted','full_decode_passed','all_frame_presentation_times_match')) or rendition.get('master_sha256')!=digest(master) or rendition.get('rendition_sha256')!=digest(web):raise ValueError('Finished web binding differs')
 return dict(master=master,web=web,poster=poster,original_clock=clock)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
 g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--render',action='store_true');p.add_argument('--baseline',type=Path);a=p.parse_args()
 if a.prepare:
  if a.baseline is None:p.error('--prepare requires actual published --baseline')
  print(prepare(a.stage,a.baseline))
 elif a.baseline is not None:p.error('Prepared record owns immutable baseline')
 else:print(render(a.stage))
if __name__=='__main__':main()
