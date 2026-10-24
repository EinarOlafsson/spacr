"""Synthetic PCM/metadata fixtures are not real audio or publication evidence."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'authoring/tools')]
import published_scene_audio_refresh as mixed
from stage_lesson import write

def old_scenes(count=11):
 return [dict(scene=i+1,speech_start=i*.02,speech_end=i*.02+.015,scene_end=(i+1)*.02,duration=.02,
  text='Old chapter '+str(i+1),effective_speed=1.0,sentences=[dict(sentence=1,speech_start=i*.02,
   speech_end=i*.02+.015,audible_start=i*.02+.002,audible_end=i*.02+.013,duration=.015)]) for i in range(count)]

def replacement():
 return dict(scene=9,speech_start=0.0,speech_end=.005,scene_end=.01,duration=.01,text='Reviewed chapter9',
  sentences=[dict(sentence=1,speech_start=0.0,speech_end=.005,audible_start=.001,audible_end=.004,duration=.005)])

def test_exact_pcm_splice_preserves_all_other_samples_and_shifts_later_cues():
 source=np.arange(11*480,dtype=np.int16);before=source.copy();new=np.full(240,-1234,dtype=np.int16)
 scenes=old_scenes();old=deepcopy(scenes)
 joined,current,proof=mixed.splice_pcm(source,scenes,new,replacement(),np)
 assert np.array_equal(joined,np.concatenate([source[:8*480],new,source[9*480:]]))
 assert np.array_equal(source,before) and scenes==old
 assert current[9]['speech_start']==pytest.approx(old[9]['speech_start']-.01)
 assert current[9]['sentences'][0]['audible_start']==pytest.approx(old[9]['sentences'][0]['audible_start']-.01)
 for i,row in enumerate(proof):
  if i==8:assert row['kind']=='current_scene9_unit'
  else:
   a,b=row['source_samples'];x,y=row['output_samples']
   assert joined[x:y].tobytes()==source[a:b].tobytes()
   assert row['pcm_sha256']==hashlib.sha256(source[a:b].astype('<i2').tobytes()).hexdigest()

@pytest.mark.parametrize('fault',['float_source','multichannel','short_source','gap','overlap','nan','missing9','empty_changed','changed_bounds'])
def test_splice_rejects_unknown_or_nonexact_pcm_ranges(fault):
 source=np.arange(11*480,dtype=np.int16);scenes=old_scenes();new=np.zeros(240,dtype=np.int16);row=replacement()
 if fault=='float_source':source=source.astype(np.float32)
 elif fault=='multichannel':source=source.reshape(-1,2)
 elif fault=='short_source':source=source[:100]
 elif fault=='gap':scenes[1]['speech_start']+=1/mixed.RATE
 elif fault=='overlap':scenes[1]['speech_start']-=1/mixed.RATE
 elif fault=='nan':scenes[1]['scene_end']=float('nan')
 elif fault=='missing9':scenes=scenes[:8]
 elif fault=='empty_changed':new=new[:0]
 else:row['scene_end']=.02
 with pytest.raises(ValueError):mixed.splice_pcm(source,scenes,new,row,np)

@pytest.fixture
def checked_fixture(tmp_path,monkeypatch):
 import render_all_voices as renderer
 import sentence_audio_cache as cache
 import verify_audio_release as audio
 import narration_audio
 monkeypatch.setattr(narration_audio,'ffmpeg_runtime_identity',lambda:dict(unit='decoder'))
 stage=tmp_path/'stage';baseline=tmp_path/'baseline';root=stage/'production'/mixed.LESSON
 voices={'en':['af_heart'],'es':['ef_dora','em_alex','em_santa'],'fr':['ff_siwis'],
  'hi':['hf_alpha','hf_beta','hm_omega','hm_psi'],'it':['if_sara','im_nicola'],
  'ja':['jf_alpha','jf_gongitsune','jf_nezumi','jf_tebukuro','jm_kumo'],
  'pt-BR':['pf_dora','pm_alex','pm_santa'],'zh-CN':['zf_xiaobei','zf_xiaoni','zf_xiaoxiao','zf_xiaoyi','zm_yunjian','zm_yunxi','zm_yunxia','zm_yunyang']}
 assert sum(map(len,voices.values()))==27
 lesson=dict(id=mixed.LESSON,narration_voices=voices,scenes=[dict(narration='Chapter '+str(i),visual='view'+str(i)) for i in range(11)])
 current={'lessons_'+lang+'.json':deepcopy(lesson) for lang in voices}
 catalogs={name:dict(lessons=[value]) for name,value in current.items()}
 source=dict(schema=1,kind='mixed_published_scene_audio',lesson=mixed.LESSON,baseline_root=str(baseline),tracks=[])
 runtime={'voice_pack':{'tensor_sha256':'a'*64,'loaded_tensor_sha256':'a'*64}}
 raw=np.full(4000,.1,dtype=np.float32)
 def plans(lesson,language,dialect,speed,voice):
  return [dict(display_text=s['narration'],sentences=[dict(text=s['narration'],speech_text=s['narration'])],
   effective_speed=1.0,hold=.52,authored_hold=.7) for s in lesson['scenes']]
 monkeypatch.setattr(renderer,'prepare_scene_plans',plans)
 monkeypatch.setattr(renderer,'synthesis_runtime_identity',lambda *a:deepcopy(runtime))
 monkeypatch.setattr(cache,'cached_sentence',lambda directory,inputs,synthesize,np:(raw.copy(),'unit phonemes',True))
 monkeypatch.setenv('SPACR_TUTORIAL_SENTENCE_CACHE',str(tmp_path/'cache'))
 records=[]
 for language,offered in voices.items():
  for voice in offered:
   stem=f'media_host/{mixed.LESSON}/audio/{language}/{voice}'
   original=dict(language=language,voice=voice,audio=stem+'.m4a',timing=stem+'.json',audio_sha256='a'*64,timing_sha256='b'*64)
   source['tracks'].append(original)
   path=root/'audio'/language/(voice+'.m4a');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'unit fake AAC')
   if language=='en':write(path.with_suffix('.json'),dict(unit_normal_English=True));continue
   old=dict(speed=1.0,sample_rate=mixed.RATE,scenes=old_scenes(),render_inputs=dict(synthesis_runtime={'historical':'runtime'},scenes=[dict(speech_text=[s['narration']]) for s in lesson['scenes']]))
   write(baseline/original['timing'],old)
   plan,inputs=mixed._unit_inputs(renderer,current['lessons_'+language+'.json'],old,language,voice,deepcopy(runtime))
   changed,row=mixed._replacement(renderer,plan,raw,'unit phonemes',np)
   unmastered=changed.copy();changed=(changed//2).astype(np.int16)  # synthetic local mastering output, not media proof
   pcm=np.arange(11*480,dtype=np.int16);joined,scenes,intervals=mixed.splice_pcm(pcm,old['scenes'],changed,row,np)
   work=root/mixed.WORK/language/voice;work.mkdir(parents=True);(work/'source.s16le').write_bytes(pcm.tobytes());(work/'joined.s16le').write_bytes(joined.tobytes());(work/'scene9-unmastered.s16le').write_bytes(unmastered.tobytes());(work/'scene9-mastered.s16le').write_bytes(changed.tobytes())
   write(path.with_suffix('.json'),mixed.timing_metadata(current['lessons_'+language+'.json'],old,inputs,scenes,len(joined),path))
   mastering=dict(**mixed.mastering_contract(renderer,language,voice),tool=dict(unit='decoder'),raw_unit_float32_sha256=hashlib.sha256(raw.astype('<f4').tobytes()).hexdigest(),assembled_pcm_sha256=mixed.digest(work/'scene9-unmastered.s16le'),mastered_pcm_sha256=mixed.digest(work/'scene9-mastered.s16le'),samples=len(changed))
   records.append(dict(**original,original_runtime=old['render_inputs']['synthesis_runtime'],unit_inputs=inputs,unit_cache_reused=True,phonemes='unit phonemes',scene9_mastering=mastering,source_pcm_sha256=mixed.digest(work/'source.s16le'),source_decoded_samples=len(pcm),source_decoded_duration=len(pcm)/mixed.RATE,decoder={'unit':'decoder'},joined_pcm_sha256=mixed.digest(work/'joined.s16le'),intervals=intervals,output_audio_sha256=mixed.digest(path),output_timing_sha256=mixed.digest(path.with_suffix('.json')),decoded_true_peak_dbfs=-3))
 en=root/'audio/en/af_heart.m4a'
 record=dict(schema=1,kind='mixed_published_scene_audio',source=source,producer_sha256=mixed.digest(mixed.__file__),english=dict(audio_sha256=mixed.digest(en),timing_sha256=mixed.digest(en.with_suffix('.json'))),tracks=records)
 write(root/mixed.RECORD,record)
 frozen_source,frozen_current=deepcopy(source),deepcopy(current)
 monkeypatch.setattr(mixed,'_source',lambda *a:(deepcopy(frozen_source),deepcopy(frozen_current)))
 calls=[];monkeypatch.setattr(audio,'check_track',lambda spec:(calls.append(spec) or []))
 monkeypatch.setattr(audio,'_decode_track',lambda *a:pytest.fail('Checker must not re-decode original AAC'))
 return stage,baseline,lesson,catalogs,root,record,calls

def test_checker_accepts_only_explicit27_and_runs_normal_output_checks_once(checked_fixture):
 stage,baseline,lesson,catalogs,root,record,calls=checked_fixture
 declared,tracks=mixed.checked_tracks(stage,baseline,lesson,catalogs)
 assert declared==lesson['narration_voices'] and len(tracks)==len(calls)==27
 assert len(list((root/'audio').rglob('*.*')))==54
 assert record['kind']=='mixed_published_scene_audio'

@pytest.mark.parametrize('fault',['record_missing','record_kind','source_binding','producer','voice_missing','english','source_pcm','joined_pcm','unit_inputs','timing','audio','extra_audio','wholetrack_fingerprint','peak','master_config','mastered_pcm','mastered_count','raw_unit_hash'])
def test_explicit_mixed_checker_rejects_tampering_without_fresh_fallback(checked_fixture,fault):
 stage,baseline,lesson,catalogs,root,record,calls=checked_fixture
 row=record['tracks'][0];lang,voice=row['language'],row['voice'];audio=root/'audio'/lang/(voice+'.m4a');work=root/mixed.WORK/lang/voice
 if fault=='record_missing':(root/mixed.RECORD).rename(root/'preserved-record.json')
 elif fault=='record_kind':record['kind']='ordinary_fresh_tts'
 elif fault=='source_binding':record['source']['baseline_root']='different'
 elif fault=='producer':record['producer_sha256']='c'*64
 elif fault=='voice_missing':record['tracks'].pop()
 elif fault=='english':(root/'audio/en/af_heart.m4a').write_bytes(b'changed')
 elif fault=='source_pcm':(work/'source.s16le').write_bytes(b'changed')
 elif fault=='joined_pcm':(work/'joined.s16le').write_bytes(b'changed')
 elif fault=='unit_inputs':row['unit_inputs']['speed']=.96
 elif fault=='timing':value=json.loads(audio.with_suffix('.json').read_text());value['scenes'][0]['text']='changed';write(audio.with_suffix('.json'),value)
 elif fault=='audio':audio.write_bytes(b'changed')
 elif fault=='extra_audio':(root/'audio/en/unknown.m4a').write_bytes(b'extra')
 elif fault=='wholetrack_fingerprint':value=json.loads(audio.with_suffix('.json').read_text());value['render_fingerprint']='fake';write(audio.with_suffix('.json'),value)
 elif fault=='peak':row['decoded_true_peak_dbfs']=0
 elif fault=='master_config':row['scene9_mastering']['selected_filter']='unreviewed filter'
 elif fault=='mastered_pcm':(work/'scene9-mastered.s16le').write_bytes(b'changed')
 elif fault=='mastered_count':row['scene9_mastering']['samples']+=1
 else:row['scene9_mastering']['raw_unit_float32_sha256']='c'*64
 if fault!='record_missing':write(root/mixed.RECORD,record)
 with pytest.raises((ValueError,FileNotFoundError)):mixed.checked_tracks(stage,baseline,lesson,catalogs,decode=False)
 assert not calls

@pytest.mark.parametrize('mode',['fresh','mixed19','mixed_other'])
def test_builder_dispatch_is_explicit_and_default_fresh_path_unchanged(tmp_path,monkeypatch,mode):
 import build_audio_patch_candidate as builder
 import build_appended_candidate as normal
 identity='19_train_cellpose' if mode!='mixed_other' else '79_module_inputs_outputs'
 stage=tmp_path/'stage';root=stage/'production'/identity;root.mkdir(parents=True)
 if mode!='fresh':write(root/mixed.RECORD,dict(unit=True))
 catalogs={'lessons_en.json':dict(lessons=[dict(id=identity)])}
 tracks=[dict(language='en',voice='af_heart',audio_sha256='a',timing_sha256='b')]
 planned=[dict(lesson=identity,**tracks[0])];calls=[]
 monkeypatch.setattr(normal,'verify_tracks',lambda *a:(calls.append('fresh') or ({'en':['af_heart']},tracks)))
 def checked(*args):
  assert args[1]==tmp_path/'baseline';calls.append('mixed');return {'en':['af_heart']},tracks
 monkeypatch.setattr(mixed,'checked_tracks',checked)
 catalogs['lessons_en.json']['lessons'][0]['narration_voices']={'en':['af_heart']}
 if mode=='mixed_other':
  with pytest.raises(ValueError):builder.verify_script_tracks(stage,[identity],catalogs,planned,baseline=tmp_path/'baseline')
  assert not calls
 else:
  builder.verify_script_tracks(stage,[identity],catalogs,planned,baseline=tmp_path/'baseline')
  assert calls==['fresh' if mode=='fresh' else 'mixed']


@pytest.fixture
def actual_source_fixture(tmp_path,monkeypatch):
 """Exercise real _source; only remote publication/legacy-clock boundaries stubbed."""
 from audit_staged_catalogs import CATALOGS
 from apply_translation_review import translated_lesson
 from check_completed_matrix import voice_matrix
 from stage_lesson import REPO
 import build_audio_patch_candidate as builder
 base,stage,repo=tmp_path/'baseline',tmp_path/'stage',tmp_path/'repo'
 voices=voice_matrix(REPO/'tools/tutorials/authoring/tools/render_all_voices.py');voices['en']=['af_heart']
 old=dict(id=mixed.LESSON,title='Unit lesson',description='Unit source',objectives=['Inspect'],prerequisite='Example',
  narration_voices=voices,scenes=[dict(visual='view'+str(i),narration='Feature details.') for i in range(11)])
 old['scenes'][8]=dict(visual='train_run_console',narration=mixed.OLD_EN)
 new=deepcopy(old);new['scenes'][8]['narration']=mixed.NEW_EN
 canonical={k:v for k,v in new.items() if k!='narration_voices'}
 write(repo/f'tools/tutorials/lessons/{mixed.LESSON}.json',canonical)
 write(stage/f'production/{mixed.LESSON}/lesson.en.json',canonical)
 def review(language,english,texts):
  return dict(lesson=mixed.LESSON,language=language,english_sha256=mixed.identity(english),
   title=english['title'],description=english['description'],objectives=english['objectives'],prerequisite=english['prerequisite'],
   scenes=texts,review=dict(editorial_review=True,listening_review=False,native_speaker_signoff=False))
 originals={}
 for name in CATALOGS:
  language=name.split('_',1)[1].removesuffix('.json')
  if language=='en':before,after=deepcopy(old),deepcopy(new)
  else:
   texts=['Feature details.' for _ in range(11)];texts[8]='Previous example. Removed timing claim.'
   before=translated_lesson(review(language,old,texts),old,{})
   texts[8]='Previous example.';approved=review(language,new,texts)
   after=translated_lesson(approved,new,before)
   write(stage/f'production/{mixed.LESSON}/review.{language}.json',approved)
   write(repo/f'tools/tutorials/lessons/reviews/{mixed.LESSON}.{language}.json',approved)
  originals[language]=before
  value=dict(schema=1,lessons=[before,dict(id='unrelated',title='Must survive')])
  write(base/'web/catalog'/name,value)
  value['lessons'][0]=after;write(stage/'catalog'/name,value)
 for language,offered in voices.items():
  for voice in offered:
   path=base/f'media_host/{mixed.LESSON}/audio/{language}/{voice}.m4a';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'unit-not-real-audio')
   write(path.with_suffix('.json'),dict(language=language,voice=voice,media_sha256=mixed.digest(path),render_inputs=dict(lesson=mixed.LESSON,language=language,voice=voice,scenes=[dict(narration=s['narration']) for s in originals[language]['scenes']])))
 files=[dict(path=p.relative_to(base).as_posix(),bytes=p.stat().st_size,sha256=mixed.digest(p)) for p in sorted(base.rglob('*')) if p.is_file()]
 write(base/'release-manifest.json',dict(files=files));write(base/'publication-receipt.json',dict(unit_receipt_boundary=True))
 monkeypatch.setattr(builder,'require_published_script_baseline',lambda *a:None)
 monkeypatch.setattr(builder,'legacy_visual_source',lambda *a:base/f'media_host/{mixed.LESSON}/audio/en/af_heart.json')
 monkeypatch.setattr(mixed,'REPO',repo)
 return base,stage,repo


@pytest.mark.parametrize('fault',[None,'canonical','review_bytes','review_binding','unselected_scene','unselected_object','offerings','source_audio'])
def test_actual_source_requires_exact_canonical_reviews_catalogs_and_original_audio(actual_source_fixture,fault):
 base,stage,repo=actual_source_fixture
 def modify(path,change):
  value=json.loads(path.read_text());change(value);write(path,value)
 if fault=='canonical':modify(repo/f'tools/tutorials/lessons/{mixed.LESSON}.json',lambda v:v['scenes'][0].update(narration='Changed'))
 elif fault=='review_bytes':modify(stage/f'production/{mixed.LESSON}/review.es.json',lambda v:v.update(title='Different'))
 elif fault=='review_binding':
  for path in [stage/f'production/{mixed.LESSON}/review.es.json',repo/f'tools/tutorials/lessons/reviews/{mixed.LESSON}.es.json']:
   modify(path,lambda v:v.update(english_sha256='c'*64))
 elif fault=='unselected_scene':modify(stage/'catalog/lessons_en.json',lambda v:v['lessons'][0]['scenes'][0].update(narration='Changed'))
 elif fault=='unselected_object':modify(stage/'catalog/lessons_en.json',lambda v:v['lessons'][1].update(title='Changed'))
 elif fault=='offerings':modify(stage/'catalog/lessons_es.json',lambda v:v['lessons'][0]['narration_voices']['en'].append('af_bella'))
 elif fault=='source_audio':(base/f'media_host/{mixed.LESSON}/audio/es/ef_dora.m4a').write_bytes(b'changed')
 if fault is None:
  source,catalogs=mixed._source(stage,base)
  assert len(source['tracks'])==27 and len(source['reviews'])==13
  assert source['current_whole_track_tts_claimed'] is False
 else:
  with pytest.raises(ValueError):mixed._source(stage,base)


@pytest.mark.parametrize('changed_count',[False,True])
def test_scene9_only_mastering_uses_first_normal_filter_and_preserves_sample_count(tmp_path,monkeypatch,changed_count):
 import render_all_voices as renderer
 import narration_audio
 monkeypatch.setattr(narration_audio,'ffmpeg_runtime_identity',lambda:dict(unit='ffmpeg'))
 pcm=np.arange(400,dtype=np.int16);before=pcm.copy();calls=[]
 def filter_once(argv,**kwargs):
  calls.append(argv)
  assert argv[argv.index('-af')+1]==renderer.mastering_config(mixed.LESSON,'es','ef_dora')['filters'][0]
  data=pcm[:-1] if changed_count else (pcm//2).astype(np.int16)
  Path(argv[-1]).write_bytes(data.tobytes())
 monkeypatch.setattr(mixed.subprocess,'run',filter_once)
 if changed_count:
  with pytest.raises(ValueError,match='sample count'):mixed.master_changed(pcm,tmp_path,renderer,'es','ef_dora',np)
 else:
  mastered,contract=mixed.master_changed(pcm,tmp_path,renderer,'es','ef_dora',np)
  assert np.array_equal(mastered,pcm//2) and contract['samples']==len(pcm)
  assert contract['assembled_pcm_sha256']==hashlib.sha256(pcm.tobytes()).hexdigest()
 assert np.array_equal(before,pcm) and len(calls)==1
 assert (tmp_path/'scene9-unmastered.s16le').read_bytes()==pcm.tobytes()
 assert (tmp_path/'scene9-mastered.s16le').is_file()
 with pytest.raises(FileExistsError):mixed.master_changed(pcm,tmp_path,renderer,'es','ef_dora',np)
