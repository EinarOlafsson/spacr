"""Synthetic PCM and metadata contracts; no real synthesis/decode/publication evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys
import numpy as np
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'authoring/tools')]
import published_scene_audio_refresh48 as mixed
import published_scene_audio_refresh as original19

def source_fixture():
 scenes=[]
 for i in range(11):
  start=i*.04
  units=[dict(sentence=1,text='Old unit',speech_text='Old unit',phonemes='old',speech_start=start,speech_end=start+.032,audible_start=start+.001,audible_end=start+.030,duration=.032)]
  if i==1:
   units=[dict(sentence=j+1,text=f'Old sentence{j}',speech_text=f'Old sentence{j}',phonemes=f'old{j}',speech_start=start+j*.012,speech_end=start+j*.012+.008,audible_start=start+j*.012+.001,audible_end=start+j*.012+.007,duration=.008,gap_from_previous=None if j==0 else .006) for j in range(3)]
  scenes.append(dict(scene=i+1,text=f'Old scene{i}',speech_text='Old unit',speech_start=start,speech_end=start+.032,scene_end=(i+1)*.04,duration=.04,effective_speed=1.0,sentences=units))
 return np.arange(11*960,dtype=np.int16),scenes

def changed_unit(seconds=.012):
 return dict(scene=2,text='Reviewed scene2',speech_text='New first sentence',phonemes='new',speech_start=0.,speech_end=seconds,scene_end=seconds,duration=seconds,effective_speed=1.0,sentences=[dict(sentence=1,text='New first sentence',speech_text='New first sentence',phonemes='new',speech_start=0.,speech_end=seconds,audible_start=.001,audible_end=seconds-.001,duration=seconds,gap_from_previous=None)])

def test_english_replacement_preserves_exact_prefix_following_gap_and_two_sentences():
 source,scenes=source_fixture();before=source.copy();old=deepcopy(scenes);row=changed_unit();pcm=np.full(288,-123,np.int16)
 joined,current,proof=mixed.splice_pcm(source,scenes,pcm,row,'en','Current paragraph',np)
 a,b=960,1152;delta=(len(pcm)-(b-a))/mixed.RATE
 assert joined.tobytes()==np.concatenate([source[:a],pcm,source[b:]]).tobytes()
 # This source slice begins with the ORIGINAL internal gap and contains both unchanged sentence clips.
 assert joined[a+len(pcm):a+len(pcm)+384].tobytes()==source[b:b+384].tobytes()
 assert current[0]==old[0] and current[1]['text']=='Current paragraph'
 for j in (1,2):
  expected=deepcopy(old[1]['sentences'][j])
  for k in ('speech_start','speech_end','audible_start','audible_end'):expected[k]+=delta
  assert current[1]['sentences'][j]==expected
 assert current[1]['sentences'][1]['speech_start']-current[1]['sentences'][0]['speech_end']==pytest.approx(old[1]['sentences'][1]['speech_start']-old[1]['sentences'][0]['speech_end'])
 for i in range(2,11):assert current[i]==mixed.shifted(old[i],delta)
 assert np.array_equal(source,before) and scenes==old
 for p in proof:
  x,y=p['output_samples'];assert p['pcm_sha256']==hashlib.sha256(joined[x:y].tobytes()).hexdigest()
 assert proof[0]['source_samples']==[0,a] and proof[2]['source_samples']==[b,len(source)]


def test_all26_nonenglish_offerings_preserve_every_other_scene_sample_and_caption():
 voices={'es':['ef_dora','em_alex','em_santa'],'fr':['ff_siwis'],'hi':['hf_alpha','hf_beta','hm_omega','hm_psi'],'it':['if_sara','im_nicola'],'ja':['jf_alpha','jf_gongitsune','jf_nezumi','jf_tebukuro','jm_kumo'],'pt-BR':['pf_dora','pm_alex','pm_santa'],'zh-CN':['zf_xiaobei','zf_xiaoni','zf_xiaoxiao','zf_xiaoyi','zm_yunjian','zm_yunxi','zm_yunxia','zm_yunyang']}
 source,scenes=source_fixture();scenes[1]['sentences']=[deepcopy(scenes[0]['sentences'][0])];scenes[1]['sentences'][0]=mixed.shifted(scenes[1]['sentences'][0],.04)
 before=deepcopy(scenes);pcm=np.full(480,321,np.int16);seen=[]
 for language,offered in voices.items():
  for voice in offered:
   joined,current,proof=mixed.splice_pcm(source,scenes,pcm,changed_unit(.02),language,'Reviewed scene2',np)
   assert joined.tobytes()==np.concatenate([source[:960],pcm,source[1920:]]).tobytes()
   for i in (0,*range(2,11)):
    row=proof[0] if i==0 else proof[2];a,b=round(before[i]['speech_start']*mixed.RATE),round(before[i]['scene_end']*mixed.RATE);offset=row['output_samples'][0]-row['source_samples'][0]
    assert joined[a+offset:b+offset].tobytes()==source[a:b].tobytes()
    assert current[i]==mixed.shifted(before[i],0 if i==0 else -.02)
   seen.append((language,voice))
 assert len(seen)==26 and scenes==before

@pytest.mark.parametrize('fault',['float','gap','bad_index','unit_count','changed_duration','unsupported_language'])
def test_splice_rejects_invalid_source_and_replacement_bounds(fault):
 source,scenes=source_fixture();pcm=np.zeros(288,np.int16);unit=changed_unit();language='en'
 if fault=='float':source=source.astype(np.float32)
 elif fault=='gap':scenes[4]['speech_start']+=1/mixed.RATE
 elif fault=='bad_index':scenes[1]['scene']=True
 elif fault=='unit_count':scenes[1]['sentences'].pop()
 elif fault=='changed_duration':unit['scene_end']+=.01
 else:language='de'
 with pytest.raises(ValueError):mixed.splice_pcm(source,scenes,pcm,unit,language,'Reviewed',np)


def lesson_fixture():
 return dict(id=mixed.LESSON,title='Hit List',prerequisite='Old prerequisite',narration_voices={'en':['af_heart']},scenes=[dict(visual='01c_explicit_companion_not_native_route' if i==1 else f'view{i}',narration=f'Old paragraph{i}',hold_after=.7) for i in range(11)])

def test_scope_allows_only_selected_sentence_fields_and_prerequisite_without_mutation():
 old=lesson_fixture();current=deepcopy(old);current['prerequisite']='Conditional prerequisite';current['scenes'][1].update(narration='Approved paragraph',speech_text='Approved spoken text');before=deepcopy((old,current))
 mixed.source_scope(old,current);assert (old,current)==before

@pytest.mark.parametrize('fault',['other_caption','visual','hold','title','voice','missing_prerequisite','missing_scene'])
def test_scope_rejects_unapproved_text_metadata_and_route_changes(fault):
 old=lesson_fixture();current=deepcopy(old)
 if fault=='other_caption':current['scenes'][2]['narration']='Changed'
 elif fault=='visual':current['scenes'][1]['visual']='Different'
 elif fault=='hold':current['scenes'][1]['hold_after']=1
 elif fault=='title':current['title']='Different'
 elif fault=='voice':current['narration_voices']['en'].append('am_adam')
 elif fault=='missing_prerequisite':current.pop('prerequisite')
 else:current['scenes'].pop()
 with pytest.raises(ValueError):mixed.source_scope(old,current)


def test_source_rejects_an_unpublished_intermediate_before_reading_current_stage(tmp_path,monkeypatch):
 import build_audio_patch_candidate as builder
 from stage_lesson import write
 base=tmp_path/'private-candidate';write(base/'release-manifest.json',dict(refreshed_lessons=['06_api']))
 called=[]
 def reject(*args):called.append(args);raise ValueError('No normal publication receipt')
 monkeypatch.setattr(builder,'require_published_script_baseline',reject)
 with pytest.raises(ValueError,match='publication receipt'):mixed._source(tmp_path/'stage',base)
 assert len(called)==1


def test_source_rejects_wrong_delivery_even_when_normal_baseline_guard_passes(tmp_path,monkeypatch):
 import build_audio_patch_candidate as builder
 from stage_lesson import write
 base=tmp_path/'published';write(base/'release-manifest.json',dict(refreshed_lessons=['19_train_cellpose']))
 monkeypatch.setattr(builder,'require_published_script_baseline',lambda *a:None)
 with pytest.raises(ValueError,match='post06'):mixed._source(tmp_path/'stage',base)


def test_input_plan_keeps_en_sentence2_and3_exact_and_uses_original_speed():
 lesson=lesson_fixture();source,scenes=source_fixture();old=dict(sample_rate=mixed.RATE,speed=1.,scenes=scenes,render_inputs=dict(scenes=[]))
 plans=[]
 for i in range(11):
  units=[dict(text=f'Old{i}',speech_text=f'Old{i}')]
  if i==1:units=[dict(text=f'Sentence{j}',speech_text=f'Sentence{j}') for j in range(3)]
  plans.append(dict(sentences=units,display_text=f'Paragraph{i}',hold=.52,authored_hold=.7,effective_speed=1.))
  old['render_inputs']['scenes'].append(dict(speech_text=[u['speech_text'] for u in units]))
 renderer=SimpleNamespace(LANGUAGES={'en':('a',['af_heart'])},narration_dialect=lambda *a:'en-us',prepare_scene_plans=lambda *a,**k:deepcopy(plans))
 plans[1]['sentences'][0]['speech_text']='Approved first sentence'
 plan,inputs=mixed.unit_inputs(renderer,lesson,old,'en','af_heart',{'cpu':'runtime'})
 assert len(plan['sentences'])==1 and plan['hold']==0 and inputs['speed']==1. and inputs['speech_text']=='Approved first sentence'
 plans[1]['sentences'][1]['speech_text']='Unapproved suffix'
 with pytest.raises(ValueError,match='Unchanged'):mixed.unit_inputs(renderer,lesson,old,'en','af_heart',{})


def test_19_default_and_splice_contract_remain_unchanged(monkeypatch):
 calls=[];monkeypatch.setattr(sys,'argv',['mixed','--stage','stage','--baseline','baseline','--render']);monkeypatch.setattr(original19,'render',lambda stage,baseline:calls.append((stage,baseline)) or 'normal19')
 original19.main();assert calls==[(Path('stage'),Path('baseline'))] and original19.LESSON=='19_train_cellpose'
 source,scenes=source_fixture();replacement=np.zeros(240,np.int16);row=changed_unit(.01);row['scene']=9
 joined,current,_=original19.splice_pcm(source,scenes,replacement,row,np)
 assert joined.tobytes()==np.concatenate([source[:8*960],replacement,source[9*960:]]).tobytes() and current[8]['scene']==9


def test_original_media_and_catalog_binding_rejects_changed_source_even_with_same_order(monkeypatch):
 catalogs={'lessons_en.json':lesson_fixture()};media=[dict(path='media_host/48_hit_list/audio/en/af_heart.m4a',sha256='a'*64,bytes=123)]
 monkeypatch.setattr(mixed,'ORIGINAL_48_IDENTITY',original19.identity(dict(catalogs=catalogs,media=media)))
 mixed.require_original_identity(catalogs,media)
 for mutated in ('caption','audio_digest','audio_size'):
  c,m=deepcopy(catalogs),deepcopy(media)
  if mutated=='caption':c['lessons_en.json']['scenes'][3]['narration']='Different accepted facts'
  elif mutated=='audio_digest':m[0]['sha256']='b'*64
  else:m[0]['bytes']+=1
  with pytest.raises(ValueError,match='C19'):mixed.require_original_identity(c,m)


def test_mastering_is_replacement_only_normal_first_filter_and_never_global():
 config=dict(filters=['loudnorm=I=-16:TP=-3:LRA=11','untouched normal later filter'],label='fixture')
 renderer=SimpleNamespace(mastering_config=lambda lesson,language,voice:deepcopy(config))
 value=mixed.mastering_contract(renderer,'en','af_heart')
 assert value['selected_filter']==config['filters'][0] and value['configuration']==config
 config['filters'][0]='volume=2'
 with pytest.raises(ValueError,match='first mastering'):mixed.mastering_contract(renderer,'es','ef_dora')


def test_explicit_48_cli_does_not_call_19(monkeypatch):
 calls=[];monkeypatch.setattr(sys,'argv',['mixed48','--stage','stage','--baseline','baseline','--render'])
 monkeypatch.setattr(mixed,'render',lambda stage,baseline:calls.append((stage,baseline)) or 'mixed48')
 monkeypatch.setattr(original19,'render',lambda *a:pytest.fail('48 must not call19 render'))
 mixed.main();assert calls==[(Path('stage'),Path('baseline'))]
