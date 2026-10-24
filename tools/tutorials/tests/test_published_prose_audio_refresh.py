"""Synthetic source-coordinate/provenance cases; no models or media decoding."""
from pathlib import Path
from copy import deepcopy
import builtins,hashlib,json,sys,types
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import published_prose_audio_refresh as m


def fixture(monkeypatch):
 monkeypatch.setattr(m,'RATE',100)
 scenes=[];plans=[]
 for i in range(3):
  units=[]
  for j in range(3):
   a=i*40+j*12;b=a+8
   units.append(dict(sentence=j+1,speech_start=a/100,speech_end=b/100,audible_start=(a+1)/100,audible_end=(b-1)/100,duration=.08,text=f'{i}-{j}',speech_text=f'{i}-{j}',phonemes=f'p{i}{j}',gap_from_previous=None if j==0 else .06))
  text=' '.join(u['text'] for u in units)
  scenes.append(dict(scene=i+1,speech_start=i*.4,speech_end=(i*40+32)/100,scene_end=(i*40+40)/100,duration=.4,text=text,speech_text=text,sentences=units,effective_speed=1,hold_after=.08))
  plans.append(dict(display_text=text,effective_speed=1,sentences=[dict(text=u['text'],speech_text=u['speech_text']) for u in units]))
 plans[1]['display_text']='new b new';plans[1]['sentences'][0]=dict(text='newA',speech_text='newA');plans[1]['sentences'][2]=dict(text='newC',speech_text='newC')
 return plans,dict(scenes=scenes),np.arange(-60,60,dtype='<i2')


def row(length,text='new',start=0):
 return dict(sentence=0,speech_start=start/100,speech_end=(start+length)/100,audible_start=(start+1)/100,audible_end=(start+length-1)/100,duration=length/100,text=text,speech_text=text,phonemes='new',gap_from_previous=None)


def test_multi_span_preserves_whole_scenes_gaps_holds_and_anchored_speech(monkeypatch):
 plans,old,source=fixture(monkeypatch);scope=m.plan_track(plans,old,'en')
 assert [op['source_samples'] for op in scope['operations']]==[[40,48],[64,72]]
 replacements=[np.arange(5,dtype='<i2')+500,np.arange(7,dtype='<i2')+700]
 joined,proof=m.splice(source,scope['operations'],replacements,120,np)
 assert joined[:40].tobytes()==source[:40].tobytes()
 assert joined[45:61].tobytes()==source[48:64].tobytes() # both neighboring gaps + oldunit2
 assert joined[68:].tobytes()==source[72:].tobytes() # original scene2 hold + all scene3
 assert len(joined)==116 and source.tobytes()==np.arange(-60,60,dtype='<i2').tobytes()
 for span in proof:
  a,b=span['output_samples'];assert hashlib.sha256(joined[a:b].tobytes()).hexdigest()==span['pcm_sha256']
  if span['kind']=='published_source_complement':x,y=span['source_samples'];assert joined[a:b].tobytes()==source[x:y].tobytes()
 scenes=m.current_scenes(plans,old,scope,replacements,[[row(5,'newA')],[row(7,'newC')]])
 assert scenes[0]==old['scenes'][0]
 assert scenes[1]['sentences'][1]['speech_start']==pytest.approx(.49)
 assert scenes[1]['sentences'][1]['speech_end']==pytest.approx(.57)
 assert scenes[2]['speech_start']==pytest.approx(.76)
 assert scenes[2]['sentences'][0]['speech_start']==pytest.approx(.76)
 assert scenes[1]['speech_end']==pytest.approx(.68) and scenes[1]['scene_end']==pytest.approx(.76)


@pytest.mark.parametrize('kind',['ambiguous_old','ambiguous_new','reorder','speed','scene_index','bounds','empty_new'])
def test_ambiguous_or_unapproved_unit_plan_fails_before_synthesis(monkeypatch,kind):
 plans,old,source=fixture(monkeypatch)
 if kind=='ambiguous_old':old['scenes'][1]['sentences'][0]['speech_text']='1-1'
 if kind=='ambiguous_new':plans[1]['sentences'][0]['speech_text']='1-1'
 if kind=='reorder':plans[1]['sentences']=[dict(text='1-2',speech_text='1-2'),dict(text='1-0',speech_text='1-0')]
 if kind=='speed':plans[1]['effective_speed']=1.1
 if kind=='scene_index':old['scenes'][1]['scene']=True
 if kind=='bounds':old['scenes'][1]['sentences'][1]['speech_end']=.9
 if kind=='empty_new':plans[1]['sentences']=[]
 with pytest.raises(ValueError):m.plan_track(plans,old,'en')


def test_unchanged_scene_never_requires_current_cache_or_pronunciation_identity(monkeypatch):
 plans,old,source=fixture(monkeypatch)
 for p,s in zip(plans,old['scenes']):p['display_text']=s['text'];p['sentences'][0]['speech_text']='new global pronunciation must not trigger oldTTS'
 scope=m.plan_track(plans,old,'en');assert scope['operations']==[]
 assert m.current_scenes(plans,old,scope,[],[])==old['scenes']


def test_changed_scene_unchanged_display_alias_drift_rejects_before_any_synthesis(monkeypatch):
 plans,old,source=fixture(monkeypatch);original=deepcopy(old)
 scope=m.plan_track(plans,old,'en')
 assert scope['scenes'][1]['anchors']==[[1,1]]
 assert all(not (op['old_units'][0]<=1<op['old_units'][1]) for op in scope['operations'])
 plans[1]['sentences'][1]['speech_text']='new global alias for the unchanged display unit'
 with pytest.raises(ValueError,match='Unchanged display unit requires exact published spoken anchor'):
  m.plan_track(plans,old,'en')
 assert old==original and old['scenes'][1]['sentences'][1]['speech_text']=='1-1'


@pytest.mark.parametrize('position',['begin','between','end'])
def test_insertions_copy_all_original_samples_and_keep_old_unit_endpoints(monkeypatch,position):
 plans,old,source=fixture(monkeypatch);p=plans[1];p['sentences']=[dict(text=u['text'],speech_text=u['speech_text']) for u in old['scenes'][1]['sentences']];j={'begin':0,'between':1,'end':3}[position];p['sentences'].insert(j,dict(text='inserted',speech_text='inserted'));scope=m.plan_track(plans,old,'en');assert len(scope['operations'])==1;op=scope['operations'][0];assert op['source_samples'][0]==op['source_samples'][1]
 prefix=12 if position=='end' else 0;replacement=np.arange(17,dtype='<i2')+500
 joined,proof=m.splice(source,[op],[replacement],120,np);at=op['source_samples'][0]
 assert joined[:at].tobytes()==source[:at].tobytes() and joined[at+17:].tobytes()==source[at:].tobytes()
 scenes=m.current_scenes(plans,old,scope,[replacement],[[row(5,'inserted',prefix)]])
 assert scenes[1]['sentences'][j]['speech_start']==pytest.approx((at+prefix)/100)
 if position=='end':assert scenes[1]['sentences'][j-1]['speech_end']==pytest.approx(.72)
 else:assert scenes[1]['sentences'][j+1]['speech_start']==pytest.approx((at+17)/100)
 assert scenes[1]['scene_end']==pytest.approx(.97)


def test_deletion_never_synthesizes_and_preserves_orphan_gap_as_hold(monkeypatch):
 plans,old,source=fixture(monkeypatch);plans[1]['sentences']=[dict(text=u['text'],speech_text=u['speech_text']) for u in old['scenes'][1]['sentences'][:2]];scope=m.plan_track(plans,old,'en');assert len(scope['operations'])==1 and scope['operations'][0]['new_units']==[2,2]
 source[60:64]=0;extensions=m.prove_extensions(source,old,scope,np)
 joined,proof=m.splice(source,scope['operations'],[np.zeros(0,dtype='<i2')],120,np)
 assert joined[:64].tobytes()==source[:64].tobytes() and joined[64:].tobytes()==source[72:].tobytes()
 scenes=m.current_scenes(plans,old,scope,[np.zeros(0,dtype='<i2')],[[]],extensions);assert scenes[1]['speech_end']==pytest.approx(.64) and scenes[1]['scene_end']==pytest.approx(.72) and scenes[1]['hold_after']==pytest.approx(.08)
 assert scenes[1]['sentences'][-1]['audible_end']==pytest.approx(.59) and extensions[0]['source_samples']==[60,64]
 cores=m.retained_proofs(source,joined,old,scope,[np.zeros(0,dtype='<i2')],np);assert next(r for r in cores if r['scene']==2 and r['new_sentence']==2)['source_samples']==[52,60]


def test_nonenglish_replaces_only_changed_paragraph_preserving_original_hold(monkeypatch):
 plans,old,source=fixture(monkeypatch);s=old['scenes'][1];u=deepcopy(s['sentences'][0]);u.update(speech_end=.72,audible_end=.71,speech_text='old paragraph',text='old paragraph');s['sentences']=[u];plans[1]['sentences']=[dict(text='new paragraph',speech_text='new paragraph')]
 scope=m.plan_track(plans,old,'es');assert [op['source_samples'] for op in scope['operations']]==[[40,72]]
 joined,proof=m.splice(source,scope['operations'],[np.arange(5,dtype='<i2')+600],120,np)
 assert joined[:40].tobytes()==source[:40].tobytes() and joined[45:].tobytes()==source[72:].tobytes()


@pytest.mark.parametrize('span',[[30,20],[-1,2],[0,121]])
def test_invalid_source_span_rejected(monkeypatch,span):
 _,_,source=fixture(monkeypatch)
 with pytest.raises(ValueError):m.splice(source,[dict(scene=1,source_samples=span)],[np.zeros(2,dtype='<i2')],120,np)


def test_source_rejection_precedes_model_decoder_cache_and_output(monkeypatch,tmp_path):
 def reject(*args):raise ValueError('foreign published source')
 monkeypatch.setattr(m,'sources',reject);original=builtins.__import__;seen=[]
 def guarded(name,*args,**kwargs):
  if name in ('render_all_voices','kokoro','torch','verify_audio_release','sentence_audio_cache'):seen.append(name);raise AssertionError('expensive path before guard')
  return original(name,*args,**kwargs)
 monkeypatch.setattr(builtins,'__import__',guarded)
 with pytest.raises(ValueError,match='foreign'):m.render(tmp_path/'stage',tmp_path/'base',['20_cellpose_masks'])
 assert seen==[] and not (tmp_path/'stage').exists()


def test_unknown_or_held77_is_not_an_engine_route():
 for values in ([],['77_motility'],['20_cellpose_masks','20_cellpose_masks'],['foreign']):
  with pytest.raises(ValueError):m.selection(values)
 with pytest.raises(ValueError,match='exactly one'):m.render_selection(['20_cellpose_masks','71_investigate_hit'])
 assert m.render_selection(['20_cellpose_masks'])==['20_cellpose_masks']
 assert set(m.POLICIES)=={'20_cellpose_masks','71_investigate_hit','74_import_images','78_spacr_screens','80_image_analysis_pathways','81_sequencing_pathways','86_alpha_organism_modules','87_alpha_features'}
 assert {k for k,p in m.POLICIES.items() if p['visual_kind']=='native'}=={'20_cellpose_masks','78_spacr_screens','80_image_analysis_pathways'}


def test_output_symlink_escape_and_overwrite_rejected(tmp_path):
 stage=tmp_path/'stage';stage.mkdir();outside=tmp_path/'outside';outside.mkdir();(stage/'audio').symlink_to(outside,target_is_directory=True)
 with pytest.raises(ValueError):m.fresh_output(stage,stage/'audio/en/file.m4a')
 with pytest.raises(ValueError):m.fresh_output(stage,outside/'file.m4a')
 p=stage/'old';p.write_bytes(b'keep')
 with pytest.raises(FileExistsError):m.fresh_output(stage,p)
 assert p.read_bytes()==b'keep'


def test_original19_producer_unchanged():
 assert m.digest(m.common.__file__)=='38f9f2030c93901cf32f91df721c844e7fb693ebba619b13eea48c6fc177fdeb'


@pytest.mark.parametrize('part',['catalog','audio_hash','audio_bytes'])
def test_immutable_original_projection_rejects_foreign_source(part):
 catalogs={'lessons_en.json':{'id':'lesson','scenes':[{'narration':'Original'}]}};media=[dict(path='media_host/lesson/audio/en/a.m4a',sha256='original',bytes=20)];policy={'source_identity':m.common.identity(dict(catalogs=catalogs,media=media))}
 m.require_original(policy,catalogs,media)
 if part=='catalog':catalogs['lessons_en.json']['scenes'][0]['narration']='Different'
 if part=='audio_hash':media[0]['sha256']='different'
 if part=='audio_bytes':media[0]['bytes']=21
 with pytest.raises(ValueError,match='Foreign original'):m.require_original(policy,catalogs,media)


def test_leading_deletion_extends_only_proven_quiet_metadata_not_the_old_spoken_core(monkeypatch):
 plans,old,source=fixture(monkeypatch);plans[1]['sentences']=[dict(text=u['text'],speech_text=u['speech_text']) for u in old['scenes'][1]['sentences'][1:]];scope=m.plan_track(plans,old,'en')
 source[40:48]=30000;source[52:60]=28000;source[48:52]=4 # quiet codec-level residue, not allzero
 extensions=m.prove_extensions(source,old,scope,np);assert extensions[0]['maximum_frame_peak']>0
 empty=np.zeros(0,dtype='<i2');joined,proof=m.splice(source,scope['operations'],[empty],120,np)
 with pytest.raises(ValueError,match='Metadata extension'):m.current_scenes(plans,old,scope,[empty],[[]])
 scenes=m.current_scenes(plans,old,scope,[empty],[[]],extensions)
 assert scenes[1]['speech_start']==scenes[1]['sentences'][0]['speech_start']==pytest.approx(.4)
 assert scenes[1]['sentences'][0]['audible_start']==pytest.approx(.45)
 assert scenes[1]['sentences'][0]['audible_end']==pytest.approx(.51)
 cores=m.retained_proofs(source,joined,old,scope,[empty],np);core=next(r for r in cores if r['scene']==2 and r['new_sentence']==1)
 assert core['source_samples']==[52,60] and core['output_samples']==[44,52]
 assert joined[44:52].tobytes()==source[52:60].tobytes()
 assert scenes[0]==old['scenes'][0]


def test_loud_or_speech_overlapping_extension_is_unproven_and_rejected(monkeypatch):
 plans,old,source=fixture(monkeypatch);plans[1]['sentences']=[dict(text=u['text'],speech_text=u['speech_text']) for u in old['scenes'][1]['sentences'][1:]];scope=m.plan_track(plans,old,'en');source[48:52]=30000
 with pytest.raises(ValueError,match='Unproven silent gap'):m.prove_extensions(source,old,scope,np)
 source[48:52]=0;old['scenes'][1]['sentences'][0]['audible_end']=.5
 with pytest.raises(ValueError,match='overlaps'):m.prove_extensions(source,old,scope,np)


def test_new_gap_over1s_requires_reviewed_silence_plan_not_threshold_change():
 scenes=[dict(scene_end=3.,sentences=[dict(audible_start=.1,audible_end=.5),dict(audible_start=1.7,audible_end=2.9)])]
 with pytest.raises(ValueError,match='exceeds1s'):m.declared_sound_gaps(scenes)
 scenes[0]['sentences'][1]['audible_start']=1.3
 assert m.declared_sound_gaps(scenes)[0]['seconds']==pytest.approx(.8)
