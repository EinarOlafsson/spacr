"""Tutorial19 only: retain decoded published scenes and synthesize reviewed scene9.

Unchanged PCM is exact before the unavoidable final lossy AAC encoding. This is
mixed published-scene provenance, never a whole-track current-TTS fingerprint.
"""
from copy import deepcopy
from pathlib import Path
import argparse, hashlib, json, math, os, subprocess, sys
from stage_lesson import REPO, read, write
from check_completed_matrix import digest

LESSON='19_train_cellpose'
RECORD='published-scene-audio-refresh.json'
WORK='published-scene-audio-work'
RATE=24000
CACHE=Path('/media/carruthers/mnt3/claude/n676_stages/sentence-audio-cache')
MODELS=Path('/media/carruthers/mnt3/codex/scratch/tutorials-agent-20261008/kokoro-model-cache')
OLD_EN=('Check the folders, model name and output location, then press Run. '
 'The Figures panel shows the training images above their masks, and the console follows progress. '
 'On this GPU, ten epochs on six images finished in under a minute.')
NEW_EN=OLD_EN.removesuffix(' On this GPU, ten epochs on six images finished in under a minute.')

def identity(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def regular(root,relative):
 root,relative=Path(root),Path(relative)
 if root.is_symlink() or relative.is_absolute() or '..' in relative.parts:raise ValueError('Independent bounded source path required')
 path=root
 for part in relative.parts:
  path=path/part
  if path.is_symlink():raise ValueError('No source/provenance symlinks')
 if not path.is_file():raise ValueError('Missing source/provenance file: '+str(path))
 return path

def _source(stage,baseline):
 """No models or decodes: exact published source, reviewed deletion and inventory."""
 from audit_staged_catalogs import CATALOGS
 from build_audio_patch_candidate import require_published_script_baseline,legacy_visual_source
 from retain_narration import catalog_lesson,require_timing
 from apply_translation_review import translated_lesson
 if baseline is None or Path(stage).is_symlink() or Path(baseline).is_symlink():raise ValueError('Actual independent baseline/stage required')
 stage,baseline=Path(stage).resolve(),Path(baseline).resolve()
 if stage==baseline or stage.is_relative_to(baseline) or baseline.is_relative_to(stage):raise ValueError('Independent mixed stage required')
 manifest=read(baseline/'release-manifest.json');require_published_script_baseline(baseline,manifest)
 records={r['path']:r for r in manifest['files']}
 if len(records)!=len(manifest['files']):raise ValueError('Duplicate baseline paths')
 def checked(relative):
  path=regular(baseline,relative);row=records.get(relative)
  if not row or (path.stat().st_size,digest(path))!=(row['bytes'],row['sha256']):raise ValueError('Published source bytes changed')
  return path
 current={};original={};reviews={}
 for name in CATALOGS:
  old=read(checked('web/catalog/'+name));new=read(regular(stage,'catalog/'+name))
  before=catalog_lesson(baseline/'web/catalog'/name,LESSON);after=catalog_lesson(stage/'catalog'/name,LESSON)
  expected=deepcopy(old);target=next(l for l in expected['lessons'] if l['id']==LESSON)
  target['scenes'][8]=deepcopy(after['scenes'][8])
  if expected!=new:raise ValueError('Only selected scene9 text may change; preserve all other catalog objects')
  a,b=deepcopy(before),deepcopy(after)
  for row in (a['scenes'][8],b['scenes'][8]):
   for key in ('narration','speech_text'):row.pop(key,None)
  if a!=b or after['scenes'][8]['visual']!='train_run_console':raise ValueError('Scene structure, voices or other narration changed')
  if name=='lessons_en.json':
   if before['scenes'][8]['narration']!=OLD_EN or after['scenes'][8]['narration']!=NEW_EN:raise ValueError('Only exact approved English timing-boast deletion')
  else:
   language=name.split('_',1)[1].removesuffix('.json');path=regular(stage,f'production/{LESSON}/review.{language}.json')
   if path.read_bytes()!=(REPO/f'tools/tutorials/lessons/reviews/{LESSON}.{language}.json').read_bytes():raise ValueError('Use actual adopted full source-bound review')
   reviews[language]=digest(path)
  original[name],current[name]=before,after
 english=current['lessons_en.json'];canonical=read(REPO/f'tools/tutorials/lessons/{LESSON}.json')
 if {k:v for k,v in english.items() if k!='narration_voices'}!=canonical or read(regular(stage,f'production/{LESSON}/lesson.en.json'))!=canonical:raise ValueError('Canonical/staged reviewed English differs')
 for language in reviews:
  name=('lessons_' if 'lessons_'+language+'.json' in current else 'captions_')+language+'.json'
  review=read(stage/f'production/{LESSON}/review.{language}.json')
  if translated_lesson(review,english,original[name])!=current[name]:raise ValueError('Full localized review differs')
 voices=english.get('narration_voices')
 if (not isinstance(voices,dict) or voices.get('en')!=['af_heart'] or len(voices)!=8
     or sum(map(len,voices.values()))!=27 or voices!=original['lessons_en.json']['narration_voices']
     or any(not isinstance(v,list) or len(v)!=len(set(v)) or not v for v in voices.values())):raise ValueError('Preserve exact original27 offerings')
 clock=legacy_visual_source(baseline,LESSON,original['lessons_en.json'])
 tracks=[];prefix=f'media_host/{LESSON}/audio/'
 expected={prefix+f'{l}/{v}{s}' for l,vs in voices.items() for v in vs for s in ('.m4a','.json')}
 if {p for p in records if p.startswith(prefix)}!=expected:raise ValueError('Original exact27 inventory differs')
 for language,offered in voices.items():
  for voice in offered:
   stem=prefix+f'{language}/{voice}';audio,timing=checked(stem+'.m4a'),checked(stem+'.json')
   metadata=read(timing);require_timing(metadata,original['lessons_'+language+'.json'],language,voice,digest(audio))
   tracks.append(dict(language=language,voice=voice,audio=stem+'.m4a',timing=stem+'.json',audio_sha256=digest(audio),timing_sha256=digest(timing)))
 return dict(schema=1,kind='mixed_published_scene_audio',lesson=LESSON,baseline_root=str(baseline),
  baseline_manifest_sha256=digest(baseline/'release-manifest.json'),publication_receipt_sha256=digest(baseline/'publication-receipt.json'),
  original_visual_clock=dict(path=clock.relative_to(baseline).as_posix(),sha256=digest(clock)),
  catalogs={n:identity(v) for n,v in current.items()},reviews=reviews,tracks=tracks,
  fresh_capture=False,current_whole_track_tts_claimed=False,human_listening_review=False),current

def splice_pcm(source,old_scenes,replacement,replacement_scene,np):
 """Sample-exact old-scene slices; no normalization, trim, padding or resampling."""
 if source.dtype!=np.int16 or source.ndim!=1 or replacement.dtype!=np.int16 or replacement.ndim!=1 or not len(replacement):raise ValueError('Require nonempty mono s16 PCM')
 if len(old_scenes)<9:raise ValueError('Missing changed scene9')
 cursor=0;pieces=[];scenes=[];proof=[]
 previous=0
 for index,scene in enumerate(old_scenes):
  values=[scene.get(k) for k in ('speech_start','speech_end','scene_end')]
  if any(type(v) not in (int,float) or not math.isfinite(v) for v in values) or not values[0]<values[1]<=values[2]:raise ValueError('Invalid published scene bounds')
  start,end=round(values[0]*RATE),round(values[2]*RATE)
  if start!=previous or end>len(source) or end<=start:raise ValueError('Uncovered published PCM interval')
  previous=end
  if index==8:
   pcm=replacement;row=deepcopy(replacement_scene);offset=cursor/RATE
  else:
   pcm=source[start:end];row=deepcopy(scene);offset=cursor/RATE-scene['speech_start']
  for key in ('speech_start','speech_end','scene_end'):row[key]+=offset
  for sentence in row.get('sentences',[]):
   for key in ('speech_start','speech_end','audible_start','audible_end'):
    if key in sentence:sentence[key]+=offset
  if abs(row['scene_end']-(cursor+len(pcm))/RATE)>1/RATE:raise ValueError('PCM and scene bounds disagree')
  row['duration']=len(pcm)/RATE
  pieces.append(pcm);scenes.append(row)
  proof.append(dict(scene=index+1,kind='current_scene9_unit' if index==8 else 'published_decoded_scene',
   source_samples=[start,end],output_samples=[cursor,cursor+len(pcm)],pcm_sha256=hashlib.sha256(pcm.astype('<i2',copy=False).tobytes()).hexdigest()))
  cursor+=len(pcm)
 return np.concatenate(pieces),scenes,proof

def _unit_inputs(renderer,lesson,old,language,voice,runtime):
 code=renderer.LANGUAGES[language][0];dialect=renderer.narration_dialect(language,code,voice);speed=old['speed']
 if language=='en' or language not in renderer.LANGUAGES or voice not in renderer.LANGUAGES[language][1] or old.get('sample_rate')!=RATE:raise ValueError('Only original non-English voice inputs')
 plans=renderer.prepare_scene_plans(lesson,language,dialect,speed,voice=voice)
 previous=old['render_inputs']['scenes']
 if len(plans)!=len(previous) or any([u['speech_text'] for u in p['sentences']]!=previous[i]['speech_text'] for i,p in enumerate(plans) if i!=8):raise ValueError('Unchanged spoken text changed outside approved scene9')
 plan=plans[8]
 if len(plan['sentences'])!=1 or plan['effective_speed']!=old['scenes'][8]['effective_speed']:raise ValueError('Keep normal whole-scene non-English unit and original speed')
 inputs=dict(schema=1,speech_text=plan['sentences'][0]['speech_text'],speed=plan['effective_speed'],language=language,lang_code=code,dialect=dialect,voice=voice,sample_rate=RATE,synthesis_runtime=runtime)
 return plan,inputs

def _replacement(renderer,plan,raw,phonemes,np):
 speech,segments=renderer.assemble_sentence_audio([raw],np)
 peak=float(np.max(np.abs(speech)))
 if peak>.98:speech=speech*(.98/peak)  # changed scene only, as normal renderer
 hold=round(plan['hold']*RATE);full=np.concatenate([speech,np.zeros(hold,dtype=np.float32)])
 pcm=np.clip(np.floor(full*32768),-32768,32767).astype(np.int16)
 sentence=dict(segments[0]);sentence=dict(sentence=1,speech_start=sentence['clip_start'],speech_end=sentence['clip_end'],audible_start=sentence['audible_start'],audible_end=sentence['audible_end'],duration=sentence['duration'],gap_from_previous=None,trimmed_lead=sentence['trimmed_lead'],trimmed_tail=sentence['trimmed_tail'],**plan['sentences'][0],phonemes=phonemes)
 row=dict(scene=9,speech_start=0.0,speech_end=len(speech)/RATE,scene_end=len(pcm)/RATE,duration=len(pcm)/RATE,text=plan['display_text'],speech_text=plan['sentences'][0]['speech_text'],phonemes=phonemes,authored_hold_after=plan['authored_hold'],hold_after=hold/RATE,effective_speed=plan['effective_speed'],sentences=[sentence])
 return pcm,row

def mastering_contract(renderer,language,voice):
 config=renderer.mastering_config(LESSON,language,voice)
 selected=config['filters'][0]
 if selected!='loudnorm=I=-16:TP=-3:LRA=11':raise ValueError('Use only the approved first normal mastering filter')
 return dict(configuration=deepcopy(config),selected_filter=selected,sample_rate=RATE,channels=1)

def master_changed(pcm,unitdir,renderer,language,voice,np):
 from narration_audio import ffmpeg_runtime_identity
 contract=mastering_contract(renderer,language,voice)
 source,target=unitdir/'scene9-unmastered.s16le',unitdir/'scene9-mastered.s16le'
 if any(os.path.lexists(p) for p in (source,target)):raise FileExistsError('Preserve earlier changed-scene mastering')
 source.write_bytes(pcm.astype('<i2',copy=False).tobytes())
 subprocess.run(['ffmpeg','-nostdin','-n','-v','error','-threads','2','-f','s16le','-ar',str(RATE),'-ac','1','-i',str(source),'-af',contract['selected_filter'],'-ar',str(RATE),'-ac','1','-c:a','pcm_s16le','-f','s16le',str(target)],check=True)
 if target.stat().st_size!=len(pcm)*2:raise ValueError('Changed-scene mastering changed sample count; preserve output for review')
 contract.update(tool=ffmpeg_runtime_identity(),assembled_pcm_sha256=digest(source),mastered_pcm_sha256=digest(target),samples=len(pcm))
 return np.frombuffer(target.read_bytes(),dtype='<i2'),contract

def timing_metadata(lesson,old,inputs,scenes,samples,audio):
 return dict(schema=1,language=inputs['language'],voice=inputs['voice'],lang_code=inputs['lang_code'],dialect=inputs['dialect'],speed=old['speed'],sample_rate=RATE,total_duration=samples/RATE,media_sha256=digest(audio),media_bytes=audio.stat().st_size,render_inputs=dict(kind='mixed_published_scene_audio',lesson=LESSON,language=inputs['language'],voice=inputs['voice'],scenes=[dict(narration=s['narration']) for s in lesson['scenes']]),scenes=scenes,mixed_published_scene_audio=True)

def render(stage,baseline):
 """Owning capped activity launcher supplies normal model/cache/offline settings."""
 import numpy as np
 import torch
 from kokoro import KPipeline,KModel
 sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'));import render_all_voices as renderer
 from sentence_audio_cache import cached_sentence
 from verify_audio_release import _decode_track,_probe_track,DECODED_DURATION_TOLERANCE_SECONDS
 from narration_audio import ffmpeg_runtime_identity
 stage=Path(stage).resolve();source,catalogs=_source(stage,baseline);root=stage/'production'/LESSON
 work=root/WORK;record=root/RECORD
 if os.path.lexists(work) or os.path.lexists(record):raise FileExistsError('Preserve failed/previous mixed audio outputs')
 english=root/'audio/en/af_heart'
 if not english.with_suffix('.m4a').is_file() or not english.with_suffix('.json').is_file():raise ValueError('Normal English cache renderer must finish first')
 if any((root/'audio'/t['language']/(t['voice']+suffix)).exists() for t in source['tracks'] if t['language']!='en' for suffix in ('.m4a','.json')):raise FileExistsError('Mixed outputs must be fresh; preserve old attempts')
 if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('HF_HUB_OFFLINE')!='1'
     or Path(os.environ['SPACR_TUTORIAL_MODEL_CACHE']).resolve()!=MODELS or Path(os.environ['SPACR_TUTORIAL_SENTENCE_CACHE']).resolve()!=CACHE):raise ValueError('Use reviewed CPU/offline models and unchanged shared cache')
 cache=Path(os.environ['SPACR_TUTORIAL_SENTENCE_CACHE']);work.mkdir();results=[];pipelines={};model=None
 for track in source['tracks']:
  language,voice=track['language'],track['voice']
  if language=='en':continue
  baseline=Path(source['baseline_root']);old=read(baseline/track['timing']);unitdir=work/language/voice;unitdir.mkdir(parents=True)
  _,_,errors=_probe_track(baseline/track['audio'])
  if errors:raise ValueError('Published AAC format differs: '+str(errors))
  frames,duration,data,errors=_decode_track(baseline/track['audio'])
  if errors or abs(duration-old['total_duration'])>DECODED_DURATION_TOLERANCE_SECONDS:raise ValueError('Published AAC decode/duration failed: '+str(errors))
  source_pcm=np.frombuffer(data,dtype='<i2');(unitdir/'source.s16le').write_bytes(data)
  runtime=renderer.synthesis_runtime_identity(renderer.SNAPSHOT,voice,'cpu')
  tensor=torch.load(renderer.SNAPSHOT/'voices'/str(voice+'.pt'),weights_only=True,map_location='cpu')
  runtime['voice_pack']['loaded_tensor_sha256']=hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()
  if runtime['voice_pack']['loaded_tensor_sha256']!=runtime['voice_pack']['tensor_sha256']:raise ValueError('Loaded voice differs')
  lesson=catalogs['lessons_'+language+'.json'];plan,inputs=_unit_inputs(renderer,lesson,old,language,voice,runtime)
  def synthesize():
   nonlocal model
   if model is None:model=KModel(repo_id='hexgrad/Kokoro-82M',model=str(renderer.SNAPSHOT/'kokoro-v1_0.pth'),config=str(renderer.SNAPSHOT/'config.json')).to('cpu').eval()
   code=inputs['lang_code']
   if code not in pipelines:pipelines[code]=KPipeline(lang_code=code,repo_id='hexgrad/Kokoro-82M',model=model,device='cpu')
   pipeline=pipelines[code]
   loaded=pipeline.load_voice(str(renderer.SNAPSHOT/'voices'/str(voice+'.pt')))
   if hashlib.sha256(loaded.detach().cpu().contiguous().numpy().tobytes()).hexdigest()!=runtime['voice_pack']['tensor_sha256']:raise ValueError('Normal pipeline loaded another voice')
   chunks=[];phonemes=[]
   for result in pipeline(inputs['speech_text'],voice=loaded,speed=inputs['speed']):
    if result.audio is not None:chunks.append(result.audio.detach().cpu().numpy().astype(np.float32))
    if result.phonemes:phonemes.append(str(result.phonemes))
   if not chunks:raise ValueError('Empty reviewed scene9 synthesis')
   return np.concatenate(chunks),' '.join(phonemes)
  raw,phonemes,reused=cached_sentence(cache,inputs,synthesize,np)
  replacement,row=_replacement(renderer,plan,raw,phonemes,np)
  replacement,mastering=master_changed(replacement,unitdir,renderer,language,voice,np)
  mastering['raw_unit_float32_sha256']=hashlib.sha256(raw.astype('<f4',copy=False).tobytes()).hexdigest()
  joined,scenes,intervals=splice_pcm(source_pcm,old['scenes'],replacement,row,np)
  pcm=unitdir/'joined.s16le';pcm.write_bytes(joined.astype('<i2',copy=False).tobytes())
  audio=root/'audio'/language/str(voice+'.m4a');audio.parent.mkdir(parents=True,exist_ok=True)
  # No whole-track loudnorm/limiter: unchanged PCM must reach the codec untouched.
  subprocess.run(['ffmpeg','-nostdin','-n','-v','error','-threads','2','-f','s16le','-ar',str(RATE),'-ac','1','-i',str(pcm),'-c:a','aac','-b:a','48k','-movflags','+faststart',str(audio)],check=True)
  peak=renderer.true_peak_dbfs(audio)
  if peak is None or peak>renderer.MAX_RELEASE_TRUE_PEAK_DBFS:raise ValueError('Codec-only mixed output exceeds normal true-peak gate; no global reprocessing allowed')
  metadata=timing_metadata(lesson,old,inputs,scenes,len(joined),audio)
  write(audio.with_suffix('.json'),metadata)
  results.append(dict(**track,original_runtime=old['render_inputs']['synthesis_runtime'],unit_inputs=inputs,unit_cache_reused=reused,phonemes=phonemes,scene9_mastering=mastering,source_pcm_sha256=digest(unitdir/'source.s16le'),source_decoded_samples=frames,source_decoded_duration=duration,decoder=ffmpeg_runtime_identity(),joined_pcm_sha256=digest(pcm),intervals=intervals,output_audio_sha256=digest(audio),output_timing_sha256=digest(audio.with_suffix('.json')),decoded_true_peak_dbfs=peak))
 if _source(stage,baseline)!=(source,catalogs):raise ValueError('Frozen source/review changed during mixed audio')
 write(record,dict(schema=1,kind='mixed_published_scene_audio',source=source,producer_sha256=digest(__file__),english=dict(audio_sha256=digest(english.with_suffix('.m4a')),timing_sha256=digest(english.with_suffix('.json'))),tracks=results))
 return record

def checked_tracks(stage,baseline,lesson,catalogs,*,decode=True):
 """No TTS/source AAC re-decode. Reconstruct exact join from retained PCM+cache."""
 import numpy as np
 sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'));import render_all_voices as renderer
 import verify_audio_release as audio
 from sentence_audio_cache import cached_sentence
 stage=Path(stage).resolve();source,current=_source(stage,baseline);root=stage/'production'/LESSON;record=read(regular(root,RECORD))
 if (lesson!=current['lessons_en.json'] or any(next(l for l in catalogs[n]['lessons'] if l['id']==LESSON)!=value for n,value in current.items())
     or set(record)!={'schema','kind','source','producer_sha256','english','tracks'} or record['schema']!=1 or record['kind']!='mixed_published_scene_audio' or record['source']!=source or record['producer_sha256']!=digest(__file__)):raise ValueError('Explicit reviewed mixed provenance differs')
 expected={(t['language'],t['voice']):t for t in source['tracks'] if t['language']!='en'}
 if len(record['tracks'])!=26 or {(t['language'],t['voice']) for t in record['tracks']}!=set(expected):raise ValueError('Exact26 mixed tracks required')
 inventory={f'{t["language"]}/{t["voice"]}{s}' for t in source['tracks'] for s in ('.m4a','.json')}
 if {p.relative_to(root/'audio').as_posix() for p in (root/'audio').rglob('*') if p.is_file()}!=inventory:raise ValueError('Audio directory must remain exactly54 normal files')
 checked=[]
 for original in source['tracks']:
  language,voice=original['language'],original['voice'];path=regular(root,Path('audio')/language/str(voice+'.m4a'));timing_path=regular(root,Path('audio')/language/str(voice+'.json'));timing=read(timing_path)
  if language=='en':
   if record['english']!=dict(audio_sha256=digest(path),timing_sha256=digest(timing_path)):raise ValueError('Normal English output changed')
   if decode:
    code='a';dialect=renderer.narration_dialect('en',code,voice);speed=renderer.resolve_voice_speed(voice)
    spec=audio.TrackSpec(path,current['lessons_en.json'],'en',code,voice,dialect,speed,renderer.prepare_scene_plans(current['lessons_en.json'],'en',dialect,speed,voice=voice))
    errors=audio.check_track(spec)
  else:
   row=next(t for t in record['tracks'] if (t['language'],t['voice'])==(language,voice))
   if any(row.get(k)!=v for k,v in original.items()):raise ValueError('Original AAC/JSON binding changed')
   old=read(Path(source['baseline_root'])/original['timing']);unitdir=Path(WORK)/language/voice
   pcm_path=regular(root,unitdir/'source.s16le');joined_path=regular(root,unitdir/'joined.s16le')
   if type(row.get('decoded_true_peak_dbfs')) not in (int,float) or not math.isfinite(row['decoded_true_peak_dbfs']) or row['decoded_true_peak_dbfs']>renderer.MAX_RELEASE_TRUE_PEAK_DBFS:raise ValueError('Normal decoded peak gate not satisfied')
   if digest(pcm_path)!=row['source_pcm_sha256'] or digest(joined_path)!=row['joined_pcm_sha256'] or row['original_runtime']!=old['render_inputs']['synthesis_runtime']:raise ValueError('Retained decoded PCM/runtime changed')
   if pcm_path.stat().st_size!=row['source_decoded_samples']*2 or abs(row['source_decoded_duration']-row['source_decoded_samples']/RATE)>1e-9:raise ValueError('Retained decode sample proof changed')
   runtime=renderer.synthesis_runtime_identity(renderer.SNAPSHOT,voice,'cpu');runtime['voice_pack']['loaded_tensor_sha256']=runtime['voice_pack']['tensor_sha256']
   plan,inputs=_unit_inputs(renderer,current['lessons_'+language+'.json'],old,language,voice,runtime)
   if inputs!=row['unit_inputs']:raise ValueError('Current scene9 synthesis inputs changed')
   def unavailable():raise ValueError('Current scene9 cache unit unavailable; checker never synthesizes')
   raw,phonemes,_=cached_sentence(Path(os.environ['SPACR_TUTORIAL_SENTENCE_CACHE']),inputs,unavailable,np)
   replacement,newscene=_replacement(renderer,plan,raw,phonemes,np)
   from narration_audio import ffmpeg_runtime_identity
   assembled=regular(root,unitdir/'scene9-unmastered.s16le');mastered=regular(root,unitdir/'scene9-mastered.s16le')
   expected_mastering=dict(**mastering_contract(renderer,language,voice),tool=ffmpeg_runtime_identity(),raw_unit_float32_sha256=hashlib.sha256(raw.astype('<f4',copy=False).tobytes()).hexdigest(),assembled_pcm_sha256=hashlib.sha256(replacement.astype('<i2',copy=False).tobytes()).hexdigest(),mastered_pcm_sha256=digest(mastered),samples=len(replacement))
   if (row.get('scene9_mastering')!=expected_mastering or assembled.read_bytes()!=replacement.astype('<i2',copy=False).tobytes() or mastered.stat().st_size!=len(replacement)*2):raise ValueError('Changed-scene mastering source/configuration/PCM/count changed')
   replacement=np.frombuffer(mastered.read_bytes(),dtype='<i2')
   joined,scenes,intervals=splice_pcm(np.frombuffer(pcm_path.read_bytes(),dtype='<i2'),old['scenes'],replacement,newscene,np)
   if (joined.tobytes()!=joined_path.read_bytes() or row['intervals']!=intervals or row['phonemes']!=phonemes or timing!=timing_metadata(current['lessons_'+language+'.json'],old,inputs,scenes,len(joined),path) or row['output_audio_sha256']!=digest(path) or row['output_timing_sha256']!=digest(timing_path)
       or timing.get('render_fingerprint') is not None or timing.get('mixed_published_scene_audio') is not True):raise ValueError('Mixed output PCM/timing/audio changed')
   if decode:errors=audio.check_track(path)
  if decode and errors:raise ValueError('Normal AAC validation failed: '+str(errors))
  checked.append(dict(language=language,voice=voice,audio_sha256=digest(path),timing_sha256=digest(timing_path)))
 return deepcopy(current['lessons_en.json']['narration_voices']),checked

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--render',action='store_true',required=True)
 a=p.parse_args();print(render(a.stage,a.baseline))
if __name__=='__main__':main()
