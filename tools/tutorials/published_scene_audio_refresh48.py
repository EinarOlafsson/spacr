"""Explicit Tutorial48 unit refresh; original decoded PCM outside the approved unit is retained.

No builder, catalog promotion or picture editing is performed here. This is mixed
provenance, not a current whole-track TTS or listening claim.
"""
from copy import deepcopy
from pathlib import Path
import argparse,hashlib,json,math,os,subprocess,sys
from stage_lesson import REPO,read,write
from check_completed_matrix import digest
import published_scene_audio_refresh as common
LESSON='48_hit_list'
RECORD='published-scene-audio-refresh.json'
WORK='published-scene-audio48-work'
KIND='mixed_published_sentence_scene_audio48'
RATE=common.RATE
ORIGINAL_48_IDENTITY='659e3fa634ad6d4da0c40ce81733cbbc445e89af885fb9e350fa17308c645ebf'
CURRENT_ENGLISH_SHA256='98a0eaaf0bb38a6b0d2f7bead3b40748f9b4f8b544e9192492f538f27a5529ec'
REVIEW_CONTENT_SHA256={'48_hit_list.ja.json': 'd267c25193fdcfd76b2e862b855339c14ddec9f87f7f4052bf1ca15928246f91', '48_hit_list.fr.json': '04922768a530171ada5b256a8d0d2dbdc9b2dcc9a3a53fa0e2220a40462fcc62', '48_hit_list.da.json': 'c81722422224d60d98e065eaa40e7a73a2a06fa4a700e1e560b7759e71add752', '48_hit_list.es.json': '5a6820a318fa77d37937fd0b6019e02dd3cdebb9db6b3fa4acf197d791fc8296', '48_hit_list.pt-BR.json': 'a6bc9db82de47cbf0422d2a6b576ade58ba78426cb2ae64c359eb402be7e13f4', '48_hit_list.it.json': 'a8eae3010caf065b1ea2a5df38e0b0652a3424f6d6732f1693d5c6208f410b6e', '48_hit_list.zh-CN.json': 'd23ab1d6fbe4df20fa19154bdb95c45d29ef9ab6aaf4d806e2461749f288ba1d', '48_hit_list.is.json': 'ee3d763f617e6e12365b5e08907df8f54f7db606d7a4df08dc6537df86b6029b', '48_hit_list.de.json': 'b0d9a03fbcd35ae1fb21285c8309ea0da1421178f9cd3b02d5ab445ec3533507', '48_hit_list.nb.json': '4a23315827febf680a9019a89742c1649425b7f1d77b84381123de5a6bb47a79', '48_hit_list.ko.json': 'b708ff91a2c64742ec7b9109e2f436b280eceef3192c4fdf59ed5f622edf9af5', '48_hit_list.sv.json': '4781dc23c0037370d23b6889c17897a4d604800072112c91ce29edcef029aa76', '48_hit_list.hi.json': 'f343ea35f27744eb013a7a8f720dafced0b11545e8d5741a4277f8b45644971c'}
OLD_EN='Extract the linked example and activate your spaCR environment. Run python hit_list_companion.py. In the companion window, click Browse and select the extracted results folder.'
NEW_EN='To use the fallback, extract the linked example and activate your spaCR environment. Run python hit_list_companion.py. In the companion window, click Browse and select the extracted results folder.'

def source_scope(before,after):
 """Only prerequisite and scene2 spoken/display text; every other object is exact."""
 a,b=deepcopy(before),deepcopy(after)
 for value in (a,b):
  if value.get('id')!=LESSON or len(value.get('scenes',[]))!=11 or value['scenes'][1].get('visual')!='01c_explicit_companion_not_native_route':raise ValueError('Exact indexed11-scene Hit List source required')
  if not isinstance(value.get('prerequisite'),str) or not value['prerequisite'].strip():raise ValueError('Explicit prerequisite required')
  value.pop('prerequisite')
  for key in ('narration','speech_text'):value['scenes'][1].pop(key,None)
 if a!=b:raise ValueError('Unapproved source field or other scene changed')
 if before.get('narration_voices')!=after.get('narration_voices'):raise ValueError('Voice inventory changed')

def require_original_identity(catalogs,media):
 if common.identity(dict(catalogs=catalogs,media=sorted(media,key=lambda r:r['path'])))!=ORIGINAL_48_IDENTITY:raise ValueError('Original48 source must match actual C19 media/catalogs')


def _source(stage,baseline):
 from audit_staged_catalogs import CATALOGS
 from build_audio_patch_candidate import require_published_script_baseline,legacy_visual_source
 from retain_narration import catalog_lesson,require_timing
 from apply_translation_review import translated_lesson
 stage,baseline=Path(stage),Path(baseline)
 if stage.is_symlink() or baseline.is_symlink():raise ValueError('Independent source roots required')
 stage,baseline=stage.resolve(),baseline.resolve()
 if stage==baseline or stage.is_relative_to(baseline) or baseline.is_relative_to(stage):raise ValueError('Independent stage required')
 manifest=read(baseline/'release-manifest.json');require_published_script_baseline(baseline,manifest)
 if manifest.get('refreshed_lessons')!=['06_api']:raise ValueError('Actual post06 published baseline required, never an intermediate candidate')
 records={r['path']:r for r in manifest['files']}
 if len(records)!=len(manifest['files']):raise ValueError('Duplicate source paths')
 def checked(relative):
  p=common.regular(baseline,relative);r=records.get(relative)
  if not r or (p.stat().st_size,digest(p))!=(r['bytes'],r['sha256']):raise ValueError('Published source changed: '+relative)
  return p
 current={};original={};reviews={}
 for name in CATALOGS:
  old=read(checked('web/catalog/'+name));new=read(common.regular(stage,'catalog/'+name))
  ids=[l['id'] for l in old['lessons']]
  if ids!=[l['id'] for l in new['lessons']] or len(ids)!=len(set(ids)) or ids.count(LESSON)!=1:raise ValueError('Catalog identity/order changed')
  before=catalog_lesson(baseline/'web/catalog'/name,LESSON);after=catalog_lesson(stage/'catalog'/name,LESSON);source_scope(before,after)
  expected=deepcopy(old);expected['lessons'][ids.index(LESSON)]=deepcopy(after)
  if expected!=new:raise ValueError('Unselected catalog object changed')
  if name=='lessons_en.json':
   if before['scenes'][1]['narration']!=OLD_EN or after['scenes'][1]['narration']!=NEW_EN:raise ValueError('Only approved English first-sentence change')
  else:
   language=name.split('_',1)[1].removesuffix('.json');p=common.regular(stage,f'production/{LESSON}/review.{language}.json');review=read(p)
   if p.read_bytes()!=(REPO/f'tools/tutorials/lessons/reviews/{LESSON}.{language}.json').read_bytes():raise ValueError('Use exact adopted full review')
   if common.identity({k:v for k,v in review.items() if k!='review'})!=REVIEW_CONTENT_SHA256.get(f'{LESSON}.{language}.json'):raise ValueError('Approved localized sentence/prerequisite changed')
   if not review.get('review') or translated_lesson(review,read(stage/'catalog/lessons_en.json')['lessons'][ids.index(LESSON)],before)!=after:raise ValueError('Full source-bound translated object differs')
   reviews[language]=digest(p)
  original[name],current[name]=before,after
 english=current['lessons_en.json'];canonical=read(REPO/f'tools/tutorials/lessons/{LESSON}.json')
 if common.identity(canonical)!=CURRENT_ENGLISH_SHA256 or {k:v for k,v in english.items() if k!='narration_voices'}!=canonical or read(common.regular(stage,f'production/{LESSON}/lesson.en.json'))!=canonical:raise ValueError('Exact approved canonical/staged English required')
 media=[r for r in manifest['files'] if r['path'].startswith('media_host/'+LESSON+'/') or r['path']==f'web/production/{LESSON}/poster.jpg']
 require_original_identity(original,media)
 for row in media:checked(row['path'])
 voices=english['narration_voices']
 if voices!=original['lessons_en.json']['narration_voices'] or voices.get('en')!=['af_heart'] or len(voices)!=8 or sum(map(len,voices.values()))!=27 or any(not v or len(v)!=len(set(v)) for v in voices.values()):raise ValueError('Exact27 original offerings required')
 clock=legacy_visual_source(baseline,LESSON,original['lessons_en.json']);tracks=[];prefix=f'media_host/{LESSON}/audio/'
 expected={prefix+f'{l}/{v}{s}' for l,vs in voices.items() for v in vs for s in ('.m4a','.json')}
 if {p for p in records if p.startswith(prefix)}!=expected:raise ValueError('Exact54 source files required')
 for language,offered in voices.items():
  for voice in offered:
   stem=prefix+f'{language}/{voice}';a,t=checked(stem+'.m4a'),checked(stem+'.json');old=read(t)
   require_timing(old,original['lessons_'+language+'.json'],language,voice,digest(a))
   tracks.append(dict(language=language,voice=voice,audio=stem+'.m4a',timing=stem+'.json',audio_sha256=digest(a),timing_sha256=digest(t)))
 return dict(schema=1,kind=KIND,lesson=LESSON,baseline_root=str(baseline),baseline_manifest_sha256=digest(baseline/'release-manifest.json'),publication_receipt_sha256=digest(baseline/'publication-receipt.json'),original_48_identity=ORIGINAL_48_IDENTITY,original_visual_clock=dict(path=clock.relative_to(baseline).as_posix(),sha256=digest(clock)),catalogs={n:common.identity(v) for n,v in current.items()},reviews=reviews,tracks=tracks,fresh_capture=False,current_whole_track_tts_claimed=False,human_listening_review=False),current

def shifted(row,offset):
 row=deepcopy(row)
 for key in ('speech_start','speech_end','scene_end','audible_start','audible_end'):
  if key in row:row[key]+=offset
 for sentence in row.get('sentences',[]):
  for key in ('speech_start','speech_end','audible_start','audible_end'):
   if key in sentence:sentence[key]+=offset
 return row

def splice_pcm(source,old_scenes,replacement,new_unit,language,current_text,np):
 """Retain exact EN prefix/suffix (including the next gap), or nonEN other scenes."""
 if source.dtype!=np.int16 or source.ndim!=1 or replacement.dtype!=np.int16 or replacement.ndim!=1 or not len(replacement):raise ValueError('Nonempty mono s16 PCM required')
 if len(old_scenes)!=11 or language not in ('en','es','fr','hi','it','ja','pt-BR','zh-CN'):raise ValueError('Exact48 scene/language contract required')
 previous=0
 for i,scene in enumerate(old_scenes,1):
  values=[scene.get(k) for k in ('speech_start','speech_end','scene_end')]
  if type(scene.get('scene')) is not int or scene['scene']!=i or any(type(v) not in (int,float) or not math.isfinite(v) for v in values) or not values[0]<values[1]<=values[2]:raise ValueError('Invalid original scene bounds')
  start,stop=round(values[0]*RATE),round(values[2]*RATE)
  if start!=previous or stop>len(source) or stop<=start:raise ValueError('Noncontiguous source PCM bounds')
  previous=stop
 # Only encoder padding beyond the declared total is omitted, as in19; every declared scene sample is retained outside replacement.
 old=old_scenes[1];units=old.get('sentences',[])
 if len(units)!=(3 if language=='en' else 1):raise ValueError('Original scene2 unit structure changed')
 if language=='en':
  unit=units[0];start,stop=round(unit['speech_start']*RATE),round(unit['speech_end']*RATE)
  if start!=round(old['speech_start']*RATE) or not start<stop<=round(units[1]['speech_start']*RATE):raise ValueError('First unit does not fit original scene2')
 else:start,stop=round(old['speech_start']*RATE),round(old['scene_end']*RATE)
 if new_unit.get('speech_start')!=0 or abs(new_unit.get('scene_end',-1)-len(replacement)/RATE)>1/RATE:raise ValueError('Replacement PCM/timing bounds differ')
 delta=(len(replacement)-(stop-start))/RATE;joined=np.concatenate([source[:start],replacement,source[stop:previous]])
 scenes=deepcopy(old_scenes)
 if language=='en':
  replacement_sentence=shifted(new_unit['sentences'][0],start/RATE);replacement_sentence['sentence']=1
  scenes[1]['sentences']=[replacement_sentence]+[shifted(u,delta) for u in units[1:]]
  for key in ('speech_end','scene_end'):scenes[1][key]+=delta
  scenes[1]['duration']+=delta;scenes[1]['text']=current_text
  scenes[1]['speech_text']=' '.join(u['speech_text'] for u in scenes[1]['sentences'])
  if 'phonemes' in scenes[1]:scenes[1]['phonemes']=' '.join(u.get('phonemes','') for u in scenes[1]['sentences'])
 else:
  scenes[1]=shifted(new_unit,start/RATE);scenes[1]['scene']=2
 for i in range(2,11):scenes[i]=shifted(scenes[i],delta)
 proof=[]
 for kind,a,b,x,y in [('published_decoded_prefix',0,start,0,start),('current_en_sentence1' if language=='en' else 'current_nonenglish_scene2',start,stop,start,start+len(replacement)),('published_decoded_suffix',stop,previous,start+len(replacement),len(joined))]:
  proof.append(dict(kind=kind,source_samples=[a,b],output_samples=[x,y],pcm_sha256=hashlib.sha256(joined[x:y].astype('<i2',copy=False).tobytes()).hexdigest()))
 return joined,scenes,proof

def unit_inputs(renderer,lesson,old,language,voice,runtime):
 if old.get('sample_rate')!=RATE or language not in renderer.LANGUAGES or voice not in renderer.LANGUAGES[language][1]:raise ValueError('Original unit voice/rate required')
 code=renderer.LANGUAGES[language][0];dialect=renderer.narration_dialect(language,code,voice);speed=old['speed']
 plans=renderer.prepare_scene_plans(lesson,language,dialect,speed,voice=voice);previous=old['render_inputs']['scenes']
 if len(plans)!=11 or len(previous)!=11:raise ValueError('Exact11 input plans required')
 for i,plan in enumerate(plans):
  before=previous[i]['speech_text'];after=[u['speech_text'] for u in plan['sentences']]
  if i!=1 and before!=after or i==1 and language=='en' and (len(before)!=3 or len(after)!=3 or before[1:]!=after[1:]):raise ValueError('Unchanged synthesis unit text changed')
 plan=deepcopy(plans[1])
 if len(plan['sentences'])!=(3 if language=='en' else 1) or plan['effective_speed']!=old['scenes'][1]['effective_speed']:raise ValueError('Original unit segmentation/speed changed')
 if language=='en':plan.update(sentences=[plan['sentences'][0]],hold=0,authored_hold=0)
 inputs=dict(schema=1,speech_text=plan['sentences'][0]['speech_text'],speed=plan['effective_speed'],language=language,lang_code=code,dialect=dialect,voice=voice,sample_rate=RATE,synthesis_runtime=runtime)
 return plan,inputs

def replacement(renderer,plan,raw,phonemes,np):
 pcm,row=common._replacement(renderer,plan,raw,phonemes,np);row['scene']=2
 return pcm,row

def mastering_contract(renderer,language,voice):
 config=renderer.mastering_config(LESSON,language,voice);selected=config['filters'][0]
 if selected!='loudnorm=I=-16:TP=-3:LRA=11':raise ValueError('Only normal first mastering filter for replacement')
 return dict(configuration=deepcopy(config),selected_filter=selected,sample_rate=RATE,channels=1)

def master_changed(pcm,unitdir,renderer,language,voice,np):
 from narration_audio import ffmpeg_runtime_identity
 contract=mastering_contract(renderer,language,voice);source,target=unitdir/'replacement-unmastered.s16le',unitdir/'replacement-mastered.s16le'
 if any(os.path.lexists(p) for p in (source,target)):raise FileExistsError('Preserve mastering outputs')
 source.write_bytes(pcm.astype('<i2',copy=False).tobytes())
 subprocess.run(['ffmpeg','-nostdin','-n','-v','error','-threads','2','-f','s16le','-ar',str(RATE),'-ac','1','-i',str(source),'-af',contract['selected_filter'],'-ar',str(RATE),'-ac','1','-c:a','pcm_s16le','-f','s16le',str(target)],check=True)
 if target.stat().st_size!=len(pcm)*2:raise ValueError('Replacement mastering changed sample count')
 contract.update(tool=ffmpeg_runtime_identity(),assembled_pcm_sha256=digest(source),mastered_pcm_sha256=digest(target),samples=len(pcm))
 return np.frombuffer(target.read_bytes(),dtype='<i2'),contract

def timing_metadata(lesson,old,inputs,scenes,samples,audio):
 return dict(schema=1,language=inputs['language'],voice=inputs['voice'],lang_code=inputs['lang_code'],dialect=inputs['dialect'],speed=old['speed'],sample_rate=RATE,total_duration=samples/RATE,media_sha256=digest(audio),media_bytes=audio.stat().st_size,render_inputs=dict(kind=KIND,lesson=LESSON,language=inputs['language'],voice=inputs['voice'],scenes=[dict(narration=s['narration']) for s in lesson['scenes']]),scenes=scenes,mixed_published_scene_audio=True)

def render(stage,baseline):
 """Explicit48 only. Caller owns normal CPU/activity/offline runtime."""
 import numpy as np
 import torch
 from kokoro import KPipeline,KModel
 sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'));import render_all_voices as renderer
 from sentence_audio_cache import cached_sentence
 from verify_audio_release import _decode_track,_probe_track,DECODED_DURATION_TOLERANCE_SECONDS
 from narration_audio import ffmpeg_runtime_identity
 stage=Path(stage).resolve();source,catalogs=_source(stage,baseline);root=stage/'production'/LESSON;work=root/WORK
 if any(os.path.lexists(p) for p in (work,root/RECORD)) or any((root/'audio').rglob('*.m4a')) or any((root/'audio').rglob('*.json')):raise FileExistsError('Fresh outputs required; preserve previous attempts')
 if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('HF_HUB_OFFLINE')!='1' or Path(os.environ['SPACR_TUTORIAL_MODEL_CACHE']).resolve()!=common.MODELS or Path(os.environ['SPACR_TUTORIAL_SENTENCE_CACHE']).resolve()!=common.CACHE):raise ValueError('Normal CPU/offline model/shared-cache runtime required')
 work.mkdir();results=[];pipelines={};model=None
 for track in source['tracks']:
  language,voice=track['language'],track['voice'];old=read(Path(source['baseline_root'])/track['timing']);unitdir=work/language/voice;unitdir.mkdir(parents=True)
  _,_,errors=_probe_track(Path(source['baseline_root'])/track['audio'])
  if errors:raise ValueError('Original format failed: '+str(errors))
  frames,duration,data,errors=_decode_track(Path(source['baseline_root'])/track['audio'])
  if errors or abs(duration-old['total_duration'])>DECODED_DURATION_TOLERANCE_SECONDS:raise ValueError('Original decode/duration failed: '+str(errors))
  source_pcm=np.frombuffer(data,dtype='<i2');(unitdir/'source.s16le').write_bytes(data)
  runtime=renderer.synthesis_runtime_identity(renderer.SNAPSHOT,voice,'cpu');tensor=torch.load(renderer.SNAPSHOT/'voices'/str(voice+'.pt'),weights_only=True,map_location='cpu')
  runtime['voice_pack']['loaded_tensor_sha256']=hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()
  if runtime['voice_pack']['loaded_tensor_sha256']!=runtime['voice_pack']['tensor_sha256']:raise ValueError('Loaded voice tensor changed')
  lesson=catalogs['lessons_'+language+'.json'];plan,inputs=unit_inputs(renderer,lesson,old,language,voice,runtime)
  def synthesize():
   nonlocal model
   if model is None:model=KModel(repo_id='hexgrad/Kokoro-82M',model=str(renderer.SNAPSHOT/'kokoro-v1_0.pth'),config=str(renderer.SNAPSHOT/'config.json')).to('cpu').eval()
   if inputs['lang_code'] not in pipelines:pipelines[inputs['lang_code']]=KPipeline(lang_code=inputs['lang_code'],repo_id='hexgrad/Kokoro-82M',model=model,device='cpu')
   pipeline=pipelines[inputs['lang_code']];loaded=pipeline.load_voice(str(renderer.SNAPSHOT/'voices'/str(voice+'.pt')))
   if hashlib.sha256(loaded.detach().cpu().contiguous().numpy().tobytes()).hexdigest()!=runtime['voice_pack']['tensor_sha256']:raise ValueError('Pipeline loaded wrong voice')
   chunks=[];phonemes=[]
   for result in pipeline(inputs['speech_text'],voice=loaded,speed=inputs['speed']):
    if result.audio is not None:chunks.append(result.audio.detach().cpu().numpy().astype(np.float32))
    if result.phonemes:phonemes.append(str(result.phonemes))
   if not chunks:raise ValueError('Empty approved replacement synthesis')
   return np.concatenate(chunks),' '.join(phonemes)
  raw,phonemes,reused=cached_sentence(common.CACHE,inputs,synthesize,np)
  changed,newunit=replacement(renderer,plan,raw,phonemes,np);changed,mastering=master_changed(changed,unitdir,renderer,language,voice,np)
  mastering['raw_unit_float32_sha256']=hashlib.sha256(raw.astype('<f4',copy=False).tobytes()).hexdigest()
  joined,scenes,intervals=splice_pcm(source_pcm,old['scenes'],changed,newunit,language,lesson['scenes'][1]['narration'],np)
  pcm=unitdir/'joined.s16le';pcm.write_bytes(joined.astype('<i2',copy=False).tobytes());audio=root/'audio'/language/(voice+'.m4a');audio.parent.mkdir(parents=True,exist_ok=True)
  subprocess.run(['ffmpeg','-nostdin','-n','-v','error','-threads','2','-f','s16le','-ar',str(RATE),'-ac','1','-i',str(pcm),'-c:a','aac','-b:a','48k','-movflags','+faststart',str(audio)],check=True)
  peak=renderer.true_peak_dbfs(audio)
  if peak is None or peak>renderer.MAX_RELEASE_TRUE_PEAK_DBFS:raise ValueError('Final codec true peak failed; never globally reprocess retained PCM')
  write(audio.with_suffix('.json'),timing_metadata(lesson,old,inputs,scenes,len(joined),audio))
  results.append(dict(**track,original_runtime=old['render_inputs']['synthesis_runtime'],unit_inputs=inputs,unit_cache_reused=reused,phonemes=phonemes,replacement_mastering=mastering,source_pcm_sha256=digest(unitdir/'source.s16le'),source_decoded_samples=frames,source_decoded_duration=duration,source_declared_samples=round(old['scenes'][-1]['scene_end']*RATE),decoder=ffmpeg_runtime_identity(),joined_pcm_sha256=digest(pcm),intervals=intervals,output_audio_sha256=digest(audio),output_timing_sha256=digest(audio.with_suffix('.json')),decoded_true_peak_dbfs=peak))
 if _source(stage,baseline)!=(source,catalogs):raise ValueError('Frozen source/review changed')
 write(root/RECORD,dict(schema=1,kind=KIND,source=source,producer_sha256=digest(__file__),common_helper_sha256=digest(common.__file__),tracks=results))
 return root/RECORD

def checked_tracks(stage,baseline,lesson,catalogs,*,decode=True):
 """No source AAC decode, re-mastering or synthesis; verify saved join/cache identities."""
 import numpy as np
 sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'));import render_all_voices as renderer
 import verify_audio_release as audio
 from sentence_audio_cache import cached_sentence
 from narration_audio import ffmpeg_runtime_identity
 source,current=_source(stage,baseline);root=Path(stage)/'production'/LESSON;record=read(common.regular(root,RECORD))
 if lesson!=current['lessons_en.json'] or any(next(l for l in catalogs[n]['lessons'] if l['id']==LESSON)!=v for n,v in current.items()):raise ValueError('Current selected catalog changed')
 if set(record)!={'schema','kind','source','producer_sha256','common_helper_sha256','tracks'} or record['schema']!=1 or record['kind']!=KIND or record['source']!=source or record['producer_sha256']!=digest(__file__) or record['common_helper_sha256']!=digest(common.__file__):raise ValueError('Explicit48 mixed record changed')
 expected={(t['language'],t['voice']):t for t in source['tracks']}
 if len(record['tracks'])!=27 or len(expected)!=27 or {(t['language'],t['voice']) for t in record['tracks']}!=set(expected):raise ValueError('Exact27 mixed tracks required')
 inventory={f'{t["language"]}/{t["voice"]}{s}' for t in source['tracks'] for s in ('.m4a','.json')}
 if {p.relative_to(root/'audio').as_posix() for p in (root/'audio').rglob('*') if p.is_file()}!=inventory:raise ValueError('Only54 normal audio files required')
 checked=[]
 for original in source['tracks']:
  language,voice=original['language'],original['voice'];row=next(t for t in record['tracks'] if (t['language'],t['voice'])==(language,voice));path=common.regular(root,Path('audio')/language/(voice+'.m4a'));timing=read(common.regular(root,Path('audio')/language/(voice+'.json')));unitdir=Path(WORK)/language/voice
  if any(row.get(k)!=v for k,v in original.items()):raise ValueError('Original published track binding changed')
  old=read(Path(source['baseline_root'])/original['timing']);src=common.regular(root,unitdir/'source.s16le');joinedpath=common.regular(root,unitdir/'joined.s16le')
  if row['original_runtime']!=old['render_inputs']['synthesis_runtime'] or digest(src)!=row['source_pcm_sha256'] or digest(joinedpath)!=row['joined_pcm_sha256'] or src.stat().st_size!=row['source_decoded_samples']*2 or abs(row['source_decoded_duration']-row['source_decoded_samples']/RATE)>1e-9 or row['source_declared_samples']!=round(old['scenes'][-1]['scene_end']*RATE) or row['decoder']!=ffmpeg_runtime_identity():raise ValueError('Original retained PCM/runtime proof changed')
  if type(row.get('decoded_true_peak_dbfs')) not in (int,float) or not math.isfinite(row['decoded_true_peak_dbfs']) or row['decoded_true_peak_dbfs']>renderer.MAX_RELEASE_TRUE_PEAK_DBFS:raise ValueError('Final true peak failed')
  runtime=renderer.synthesis_runtime_identity(renderer.SNAPSHOT,voice,'cpu');runtime['voice_pack']['loaded_tensor_sha256']=runtime['voice_pack']['tensor_sha256'];plan,inputs=unit_inputs(renderer,current['lessons_'+language+'.json'],old,language,voice,runtime)
  if inputs!=row['unit_inputs']:raise ValueError('Approved unit inputs changed')
  def unavailable():raise ValueError('Replacement cache missing; checker never synthesizes')
  raw,phonemes,_=cached_sentence(Path(os.environ['SPACR_TUTORIAL_SENTENCE_CACHE']),inputs,unavailable,np);new,newunit=replacement(renderer,plan,raw,phonemes,np)
  assembled=common.regular(root,unitdir/'replacement-unmastered.s16le');mastered=common.regular(root,unitdir/'replacement-mastered.s16le')
  config=dict(**mastering_contract(renderer,language,voice),tool=ffmpeg_runtime_identity(),assembled_pcm_sha256=hashlib.sha256(new.astype('<i2',copy=False).tobytes()).hexdigest(),mastered_pcm_sha256=digest(mastered),samples=len(new),raw_unit_float32_sha256=hashlib.sha256(raw.astype('<f4',copy=False).tobytes()).hexdigest())
  if row['replacement_mastering']!=config or assembled.read_bytes()!=new.astype('<i2',copy=False).tobytes() or mastered.stat().st_size!=len(new)*2:raise ValueError('Replacement-only mastering identity/count changed')
  joined,scenes,intervals=splice_pcm(np.frombuffer(src.read_bytes(),dtype='<i2'),old['scenes'],np.frombuffer(mastered.read_bytes(),dtype='<i2'),newunit,language,current['lessons_'+language+'.json']['scenes'][1]['narration'],np)
  if joined.tobytes()!=joinedpath.read_bytes() or intervals!=row['intervals'] or phonemes!=row['phonemes'] or timing!=timing_metadata(current['lessons_'+language+'.json'],old,inputs,scenes,len(joined),path) or row['output_audio_sha256']!=digest(path) or row['output_timing_sha256']!=digest(path.with_suffix('.json')):raise ValueError('Exact current unit/retained sample join/timing changed')
  if timing.get('render_fingerprint') is not None or timing.get('mixed_published_scene_audio') is not True:raise ValueError('No current-whole-track fingerprint allowed')
  if decode:
   errors=audio.check_track(path)
   if errors:raise ValueError('Normal final AAC check failed: '+str(errors))
  checked.append(dict(language=language,voice=voice,audio_sha256=digest(path),timing_sha256=digest(path.with_suffix('.json'))))
 return deepcopy(current['lessons_en.json']['narration_voices']),checked

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--render',action='store_true',required=True);a=p.parse_args();print(render(a.stage,a.baseline))
if __name__=='__main__':main()
