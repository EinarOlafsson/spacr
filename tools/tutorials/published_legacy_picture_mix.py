"""Explicit reviewed legacy picture references; no native or synthesis claim."""
from copy import deepcopy
from pathlib import Path
import argparse,json,math,os,shutil,sys,tempfile
from stage_lesson import REPO,read,write
from check_completed_matrix import digest,voice_matrix
from retain_narration import catalog_lesson,require_unchanged_lesson,require_timing
from append_staged_lessons import parse_javascript
from build_release_candidate import copy_checked
from validate_candidate import validate
KIND='published_legacy_picture_references'
SOURCE='published-legacy-picture-source.json'
RECORD='published-legacy-picture-mix.json'
LESSON='10_classify_cv'
sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'))

def contract(lesson):
 if lesson==LESSON:
  from render_classify_input_reference_cards import checked_lesson,CARDS,identity,validate_cards
  return KIND,11,[2,3,4],checked_lesson,CARDS,identity,validate_cards
 if lesson=='06_api':
  from render_api_reference_cards import KIND as kind,checked_lesson,CARDS,identity,validate_cards
  return kind,6,list(CARDS),checked_lesson,CARDS,identity,validate_cards
 raise ValueError('Only explicit reviewed legacy10/API06 contracts')

def regular(root,relative):
 root=Path(root);relative=Path(relative)
 if root.is_symlink() or relative.is_absolute() or '..' in relative.parts:raise ValueError('Independent bounded source path required')
 p=root
 for part in relative.parts:
  p=p/part
  if p.is_symlink():raise ValueError('Source path must not contain symlinks')
 if not p.is_file() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('Missing bounded source file')
 return p

def _published(baseline,lesson=LESSON):
 from audit_staged_catalogs import CATALOGS
 from build_audio_patch_candidate import require_published_script_baseline,legacy_visual_source
 from published_retained_narration import _authoring_without_generated_aliases
 sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'))
 kind,_,_,checked_lesson,_,_,_=contract(lesson)
 baseline=Path(baseline).resolve();manifest=read(baseline/'release-manifest.json')
 require_published_script_baseline(baseline,manifest)
 records={r['path']:r for r in manifest['files']}
 if len(records)!=len(manifest['files']):raise ValueError('Duplicate published manifest paths')
 def verify(relative):
  p=regular(baseline,relative);r=records.get(relative)
  if not r or p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise ValueError('Published source differs: '+relative)
  return p
 english=catalog_lesson(verify('web/catalog/lessons_en.json'),lesson);checked_lesson(english)
 declared=english.get('narration_voices');allowed=voice_matrix(REPO/'tools/tutorials/authoring/tools/render_all_voices.py')
 if (not isinstance(declared,dict) or set(declared)!=set(allowed) or sum(map(len,declared.values()))!=27
     or declared.get('en')!=['af_heart'] or any(not isinstance(v,list) or not v or len(v)!=len(set(v)) or not set(v)<=set(allowed[k]) for k,v in declared.items())):raise ValueError('Require original exact27 offerings')
 canonical=read(REPO/'tools/tutorials/lessons'/str(lesson+'.json'))
 if 'narration_voices' in canonical and canonical['narration_voices']!=declared:raise ValueError('Authored voice offerings changed')
 require_unchanged_lesson(canonical,english,english,voice_inventory=declared)
 catalogs={};localized={}
 for name in CATALOGS:
  path=verify('web/catalog/'+name);catalogs[name]=path;original=catalog_lesson(path,lesson)
  current=catalog_lesson(REPO/'tools/tutorials/authoring/catalog'/name,lesson);language=name.split('_',1)[1][:-5]
  require_unchanged_lesson(_authoring_without_generated_aliases(current,original,language),original,original,voice_inventory=declared)
  if original.get('narration_voices')!=declared or current.get('narration_voices')!=declared:raise ValueError('Current or published catalog offerings changed')
  if name.startswith('lessons_'):localized[language]=original
 runtime=verify('web/lesson_catalog.js');rows=[l for l in parse_javascript(runtime.read_text())['lessons'] if l['id']==lesson]
 if len(rows)!=1 or rows[0].get('native_idle_tails') or rows[0].get('narration_voices')!=declared:raise ValueError('Require original non-native runtime binding')
 clock=legacy_visual_source(baseline,lesson,english);verify(clock.relative_to(baseline).as_posix())
 tracks=[];prefix=f'media_host/{lesson}/audio/'
 expected={prefix+f'{lang}/{voice}{suffix}' for lang,voices in declared.items() for voice in voices for suffix in ('.m4a','.json')}
 if {p for p in records if p.startswith(prefix)}!=expected:raise ValueError('Published audio inventory is not the original27')
 for lang,voices in declared.items():
  for voice in voices:
   rel=prefix+f'{lang}/{voice}';audio=verify(rel+'.m4a');metadata=verify(rel+'.json')
   require_timing(read(metadata),localized[lang],lang,voice,digest(audio))
   tracks.append(dict(language=lang,voice=voice,audio=rel+'.m4a',timing=rel+'.json',audio_sha256=digest(audio),timing_sha256=digest(metadata)))
 visual_paths=[f'media_host/{lesson}/video/{lesson}_silent.mp4',f'media_host/{lesson}/web/{lesson}_silent.mp4',f'web/production/{lesson}/poster.jpg']
 for rel in visual_paths:verify(rel)
 source=dict(schema=1,kind=kind,lesson=lesson,baseline_root=str(baseline),manifest_sha256=digest(baseline/'release-manifest.json'),publication_receipt_sha256=digest(baseline/'publication-receipt.json'),original_clock=dict(path=clock.relative_to(baseline).as_posix(),sha256=digest(clock)),original_visuals=[deepcopy(records[p]) for p in visual_paths],tracks=tracks,fresh_capture=False,raw_capture_revalidated=False,current_runtime_freshness_claimed=False)
 if lesson=='06_api':
  from render_api_reference_cards import checked_published
  checked_published(source)
 return source,canonical,catalogs,read(clock)

def prepare(stage,baseline,lesson=LESSON):
 if Path(stage).is_symlink() or Path(baseline).is_symlink():raise ValueError('Independent stage and baseline roots required')
 stage=Path(stage).resolve();baseline=Path(baseline).resolve()
 if os.path.lexists(stage) or stage==baseline or stage.is_relative_to(baseline) or baseline.is_relative_to(stage):raise FileExistsError('Require a fresh independent stage')
 source,english,catalogs,clock=_published(baseline,lesson);stage.mkdir(parents=True);root=stage/'production'/lesson
 def copy(src,relative):
  target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True)
  with Path(src).open('rb') as a,target.open('xb') as b:shutil.copyfileobj(a,b)
 for name,path in catalogs.items():copy(path,'catalog/'+name)
 for track in source['tracks']:
  for key in ('audio','timing'):copy(baseline/track[key],track[key].replace('media_host/','production/',1))
 copy(baseline/source['original_clock']['path'],f'production/{lesson}/video/original-visual-timings.json')
 write(root/'lesson.en.json',english);write(root/SOURCE,source)
 return root/SOURCE

def checked_source(stage,lesson=LESSON):
 if Path(stage).is_symlink():raise ValueError('Independent stage required')
 stage=Path(stage).resolve();root=stage/'production'/lesson;record=read(regular(stage,root.relative_to(stage)/SOURCE))
 source,english,catalogs,clock=_published(record['baseline_root'],lesson)
 if record!=source or read(regular(stage,root.relative_to(stage)/'lesson.en.json'))!=english:raise ValueError('Staged legacy source/lesson differs')
 for name,path in catalogs.items():
  if digest(regular(stage,'catalog/'+name))!=digest(path):raise ValueError('Retained full catalog bytes differ')
 for track in source['tracks']:
  for key in ('audio','timing'):
   if digest(regular(stage,track[key].replace('media_host/','production/',1)))!=track[key+'_sha256']:raise ValueError('Retained27 audio/timing bytes differ')
 actual={p.relative_to(root/'audio').as_posix() for p in (root/'audio').rglob('*') if p.is_file()}
 expected={str(Path(t[k]).relative_to(f'media_host/{lesson}/audio')) for t in source['tracks'] for k in ('audio','timing')}
 if actual!=expected or digest(regular(stage,f'production/{lesson}/video/original-visual-timings.json'))!=source['original_clock']['sha256']:raise ValueError('Retained audio/clock inventory changed')
 return source,english,clock

def plan(english,clock,cards):
 sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'))
 kind,count,slots,checked_lesson,CARDS,identity,_=contract(english.get('id'))
 from render_visual_master import frame_aligned_durations
 checked_lesson(english)
 if clock.get('schema')!=1 or clock.get('kind') or clock.get('native_live_video') or len(clock.get('scenes',[]))!=count or set(cards)!=set(CARDS):raise ValueError('Only original indexed legacy clock and exact reference slots')
 cursor=0;end=0.0;rows=[]
 for i,(scene,duration) in enumerate(zip(clock['scenes'],frame_aligned_durations(clock['scenes'],30))):
  authored=english['scenes'][i]
  if type(scene.get('scene')) is not int or scene.get('scene')!=i+1 or scene.get('native_live_video') or scene.get('text')!=authored['narration']:raise ValueError('Legacy indexed source clock changed')
  values=[scene.get(k) for k in ('speech_start','speech_end','scene_end','duration')]
  if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('Invalid legacy clock numbers')
  start,speech_end,stop,span=values
  if abs(start-end)>1e-6 or not start<speech_end<=stop or span<=0 or abs(stop-start-span)>1e-6:raise ValueError('Noncontiguous legacy clock')
  end=stop
  count=round(duration*30)
  row=dict(scene=i+1,visual=authored['visual'],source_start_frame=cursor,source_keep_frames=[[0,count]],kept_frames=count,final_frame_hold_frames=0,edited=False)
  key=i+1 if english['id']=='06_api' else row['visual']
  if key in cards:
   card=cards[key]
   if english['id']=='06_api' and (type(card.get('source_scene')) is not int or card['source_scene']!=i+1 or card.get('visual')!=row['visual']):raise ValueError('Reference index and repeated visual must both match')
   if card.get('kind')!='authored_reference_card' or card.get('narration_sha256')!=identity(authored['narration']):raise ValueError('Reference must bind current unchanged narration')
   row['reference_card']=deepcopy(card)
  rows.append(row);cursor+=count
 if type(clock.get('total_duration')) not in (int,float) or not math.isfinite(clock['total_duration']) or abs(clock['total_duration']-end)>1e-6:raise ValueError('Incomplete legacy clock')
 return rows

def _inputs(stage,lesson=LESSON):
 sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'))
 kind,_,_,_,_,_,validate_cards=contract(lesson)
 source,english,clock=checked_source(stage,lesson);root=Path(stage)/'production'/lesson
 cards=validate_cards(root/'reference-cards',english,REPO,source) if lesson=='06_api' else validate_cards(root/'reference-cards',english,REPO)
 rows=plan(english,clock,cards)
 spec=dict(lesson=lesson,kind=kind,size=[3840,2160],fps=30,scenes=[dict(visual=a['visual'],narration=a['narration'],published_source=row) for a,row in zip(english['scenes'],rows)])
 return source,clock,rows,spec

def encode_parts(source,rows,cards_root,work,target):
 from published_native_edit import encode_scene
 sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'))
 from render_visual_master import encode_still,concat
 if any(os.path.lexists(p) for p in (work,target)):raise FileExistsError('Preserve old render/source bytes')
 if Path(source).resolve()==Path(target).resolve():raise FileExistsError('Never overwrite source master')
 work.mkdir();target.parent.mkdir(parents=True,exist_ok=True);parts=[]
 for row in rows:
  part=work/f'scene-{row["scene"]:02}.mp4'
  if 'reference_card' in row:encode_still(Path(cards_root)/row['reference_card']['image'],row['kept_frames']/30,part,30)
  else:encode_scene(source,row,part)
  parts.append(part)
 concat(parts,target,work/'concat.txt')

def render(stage,lesson=LESSON):
 stage=Path(stage).resolve();root=stage/'production'/lesson;source,clock,rows,spec=_inputs(stage,lesson)
 if any(os.path.lexists(root/name) for name in (RECORD,'scenes.json')):raise FileExistsError('Preserve earlier legacy render metadata')
 from stage_web_renditions import probe
 master=Path(source['baseline_root'])/source['original_visuals'][0]['path'];stream=probe(master)['streams']
 if len(stream)!=1 or stream[0].get('codec_type')!='video' or [stream[0].get('width'),stream[0].get('height')]!=[3840,2160] or stream[0].get('r_frame_rate')!='30/1' or int(stream[0].get('nb_frames',0))!=sum(r['kept_frames'] for r in rows):raise ValueError('Original master does not fit original scene frame clock')
 target=root/'video'/str(lesson+'_silent.mp4');encode_parts(master,rows,root/'reference-cards',root/'legacy-picture-render',target)
 if _inputs(stage,lesson)!=(source,clock,rows,spec):raise ValueError('Source/cards changed during rendering')
 write(root/'scenes.json',spec);write(root/RECORD,dict(source=source,spec_sha256=digest(root/'scenes.json'),rendered_master_sha256=digest(target)))
 return checked_legacy_picture_timing(stage,lesson)

def checked_legacy_picture_timing(stage,identity):
 if identity=='48_hit_list':
  from hit_list_picture_refresh import checked_picture_timing
  return checked_picture_timing(stage)
 contract(identity);lesson=identity
 source,clock,rows,spec=_inputs(stage,lesson);root=Path(stage)/'production'/lesson
 if any((root/name).exists() for name in ('published-native-reuse.json','published-native-edit.json','published-native-mix.json')):raise ValueError('Legacy record cannot coexist with native claims')
 record=read(regular(stage,root.relative_to(stage)/RECORD))
 expected=dict(source=source,spec_sha256=digest(root/'scenes.json'),rendered_master_sha256=digest(root/'video'/str(lesson+'_silent.mp4')))
 if record!=expected or read(root/'scenes.json')!=spec:raise ValueError('Legacy render bindings changed')
 return root/'video/original-visual-timings.json'

def require_candidate_records(before,after,lesson=LESSON):
 allowed={f'media_host/{lesson}/video/{lesson}_silent.mp4',f'media_host/{lesson}/web/{lesson}_silent.mp4',f'web/production/{lesson}/poster.jpg'}
 if not allowed<=set(before) or set(before)!=set(after) or any(after[p]!=before[p] for p in before if p not in allowed):raise ValueError('Only three selected visual assets may differ')

def build_candidate(stage,lesson=LESSON):
 from build_audio_patch_candidate import require_published_script_baseline
 stage=Path(stage).resolve();clock=checked_legacy_picture_timing(stage,lesson);root=stage/'production'/lesson
 source,_,_=checked_source(stage,lesson);baseline=Path(source['baseline_root']);old=read(baseline/'release-manifest.json');require_published_script_baseline(baseline,old)
 acceptance=read(root/'current-video-acceptance.json')
 for key in ('master_full_decode_passed','master_all_presentation_times_checked','master_frame_count_matches_scene_boundaries','published_legacy_picture_references'):
  if acceptance.get(key) is not True:raise ValueError('Actual normal legacy finish required')
 if acceptance.get('native_fifty_voice_timeline') is not False or acceptance['master_sha256']!=digest(root/'video'/str(lesson+'_silent.mp4')) or acceptance['timing_sha256']!=digest(clock) or acceptance.get('poster_sha256')!=digest(regular(stage,root.relative_to(stage)/'poster.jpg')):raise ValueError('Legacy finish describes different media/clock')
 rendition=read(stage/'web-renditions'/lesson/'rendition-checks.json')
 if rendition.get('accepted') is not True or rendition.get('full_decode_passed') is not True or rendition.get('all_frame_presentation_times_match') is not True or rendition.get('master_sha256')!=acceptance['master_sha256'] or rendition.get('rendition_sha256')!=digest(stage/'web-renditions'/lesson/'video'/str(lesson+'_silent.mp4')):raise ValueError('Actual normal web finish must bind the new master')
 replacements={f'media_host/{lesson}/video/{lesson}_silent.mp4':root/'video'/str(lesson+'_silent.mp4'),f'media_host/{lesson}/web/{lesson}_silent.mp4':stage/'web-renditions'/lesson/'video'/str(lesson+'_silent.mp4'),f'web/production/{lesson}/poster.jpg':root/'poster.jpg'}
 target=Path(tempfile.mkdtemp(prefix='release-candidate-legacy-pictures-',dir=stage));records=[]
 for row in old['files']:
  relative=row['path'];path=replacements.get(relative,regular(baseline,relative));expected=digest(path) if relative in replacements else row['sha256']
  copy_checked(path,target/relative,records,target,expected)
 if set(replacements)-{r['path'] for r in old['files']}:raise ValueError('Original visual records missing')
 before={r['path']:r for r in old['files']};after={r['path']:r for r in records}
 require_candidate_records(before,after,lesson)
 # No catalog/runtime/audio/timing changes, no fabricated native sidecar.
 manifest=deepcopy(old);manifest.update(files=sorted(records,key=lambda r:r['path']),refreshed_lessons=[lesson],appended_lessons=[],withdrawn_lessons=[],preserved_lessons=old['routes']-1,script_refresh=False,legacy_picture_refresh=True,release_hold=True,uploaded=False,published=False,fresh_capture=False,raw_capture_revalidated=False)
 manifest['baseline_manifest_sha256']=source['manifest_sha256']
 manifest['legacy_picture_references']=dict(lesson=lesson,source_manifest_sha256=source['manifest_sha256'],original_clock=source['original_clock'],original_visuals=source['original_visuals'],card_slots=contract(lesson)[2],card_contract_sha256=digest(root/'reference-cards/cards.json'),current_runtime_freshness_claimed=False)
 manifest['web_bytes']=sum(r['bytes'] for r in records if r['path'].startswith('web/'));manifest['media_host_bytes']=sum(r['bytes'] for r in records if r['path'].startswith('media_host/'))
 write(target/'release-manifest.json',manifest);validate(target,include_hosted_media=True)
 checked_legacy_picture_timing(stage,lesson);print('CANDIDATE',target,flush=True);return target

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
 group=p.add_mutually_exclusive_group(required=True)
 for flag in ('prepare','render','candidate'):group.add_argument('--'+flag,action='store_true')
 p.add_argument('--baseline',type=Path);p.add_argument('--lesson',default=LESSON,choices=[LESSON,'06_api']);a=p.parse_args()
 if a.prepare:
  if a.baseline is None:p.error('--prepare requires actual delivered --baseline')
  print(prepare(a.stage,a.baseline,a.lesson))
 elif a.baseline is not None:p.error('The prepared record owns immutable baseline for render/candidate')
 elif a.render:print(render(a.stage,a.lesson))
 else:build_candidate(a.stage,a.lesson)
if __name__=='__main__':main()
