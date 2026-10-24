"""Explicit04 revision contracts; synthetic unit sources are never capture proof."""
from copy import deepcopy
from pathlib import Path
import json,sys
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'authoring/tools')]
import published_native_mix as mix
import render_platform_maintenance_references as cards

@pytest.fixture
def maintenance_plan():
 original=dict(lesson=cards.LESSON,kind='native_visual_timing',fps=30,native_live_video=True,
  playback_speed=1,looped_or_stretched=False,total_duration=10237/30,coverage=dict(accepted=True,gaps=[]),
  scenes=[dict(scene=i,visual=v,speech_start=a/30,speech_end=b/30,scene_end=b/30,duration=(b-a)/30,native_live_video=i in (8,9)) for i,(v,(a,b)) in enumerate(zip(cards.VISUALS,cards.BOUNDS),1)])
 english=dict(id=cards.LESSON,scenes=[dict(visual=v,narration='Feature '+v) for v in cards.VISUALS])
 tracks=[dict(scenes=[dict(speech_start=i*2,speech_end=i*2+1.48,duration=2) for i in range(14)]) for _ in range(50)]
 references={v:dict(visual=v,kind='authored_reference_card',reference_revision=cards.REVISION,narration_sha256=mix.identity(english['scenes'][cards.VISUALS.index(v)]['narration'])) for v in cards.CARDS}
 bindings=dict(master=dict(sha256=cards.MASTER_SHA),timing=dict(sha256=cards.TIMING_SHA))
 edit=dict(schema=1,lesson=cards.LESSON,master_sha256=cards.MASTER_SHA,visual_timing_sha256=cards.TIMING_SHA,scenes=[])
 fences={v:11.3 if i==8 else 1.0 if i==9 else 0 for i,v in enumerate(cards.VISUALS,1)}
 return english,original,tracks,references,edit,bindings,fences

def test_explicit_maintenance_preserves_all_clocks_four_complete_intervals_and_two_native_flags(maintenance_plan):
 before=deepcopy(maintenance_plan);rows=mix.plan_mix(*maintenance_plan,reference_revision=cards.REVISION)
 assert maintenance_plan==before
 assert [i for i,r in enumerate(rows,1) if 'reference_card' not in r]==[7,8,9,10]
 assert [i for i,r in enumerate(rows,1) if r['native_live_video']]==[8,9]
 for row,source in zip(rows,before[1]['scenes']):
  assert all(row[k]==source[k] for k in ('speech_start','speech_end','scene_end','duration'))
  assert row['edited'] is False
  if 'reference_card' not in row:
   assert row['source_keep_frames']==[[0,round(source['duration']*30)]] and row['final_frame_hold_frames']==0
 assert rows[9]['visual']=='05_doctor' and 'reference_card' not in rows[9]

@pytest.mark.parametrize('fault',['implicit','unknown_revision','wrong_lesson','missing_card','extra_card','card_revision','native_flag','source_index','source_clock','master','timing','missing_fence','fence_change','source_edit','missing_voice','narration','card_overflow','retained_overflow'])
def test_maintenance_cannot_relax_original_source_or_revision_contract(maintenance_plan,fault):
 a=deepcopy(maintenance_plan);english,source,tracks,refs,edit,bindings,fences=a;revision=cards.REVISION
 if fault=='implicit':revision=None
 elif fault=='unknown_revision':revision='unreviewed'
 elif fault=='wrong_lesson':english['id']='05_home'
 elif fault=='missing_card':refs.pop(next(iter(refs)))
 elif fault=='extra_card':refs['05_doctor']=deepcopy(next(iter(refs.values())))
 elif fault=='card_revision':next(iter(refs.values()))['reference_revision']='other'
 elif fault=='native_flag':source['scenes'][4]['native_live_video']=True
 elif fault=='source_index':source['scenes'][0]['scene']=2
 elif fault=='source_clock':source['scenes'][0]['scene_end']+=1/30
 elif fault=='master':bindings['master']['sha256']=edit['master_sha256']='c'*64
 elif fault=='timing':bindings['timing']['sha256']=edit['visual_timing_sha256']='c'*64
 elif fault=='missing_fence':fences.pop(next(iter(fences)))
 elif fault=='fence_change':fences[cards.VISUALS[7]]=13
 elif fault=='source_edit':edit['scenes']=[dict(visual='05_doctor')]
 elif fault=='missing_voice':tracks.pop()
 elif fault=='narration':next(iter(refs.values()))['narration_sha256']='c'*64
 else:
  index=0 if fault=='card_overflow' else 7
  tracks[0]['scenes'][index].update(speech_end=10000,duration=10000)
 with pytest.raises(ValueError):mix.plan_mix(*a,reference_revision=revision)

def test_doctor_only_default_still_uses_its_original_contract(maintenance_plan):
 from render_platform_doctor_reference_card import MASTER_SHA,CARD
 a=deepcopy(maintenance_plan);a[3].clear();a[3][CARD]=dict(visual=CARD,kind='authored_reference_card',narration_sha256=mix.identity(a[0]['scenes'][9]['narration']))
 a[5]['master']['sha256']=a[4]['master_sha256']=MASTER_SHA
 assert mix.plan_mix(*a)==mix.plan_mix(*a,reference_revision=None)
 assert [r['visual'] for r in mix.plan_mix(*a) if 'reference_card' in r]==[CARD]

@pytest.fixture
def result_sources(tmp_path,monkeypatch):
 folder=tmp_path/'saved';folder.mkdir()
 for v in cards.VISUALS[4:6]:(folder/(v+'.png')).write_bytes(('unit-only original '+v).encode())
 hashes={v+'.png':cards.digest(folder/(v+'.png')) for v in cards.VISUALS[4:6]}
 profile=dict(requested_backend='cpu',active_backend='cpu',torch_version='2.14.1+cpu',cuda_available=False,detected_accelerator='nvidia')
 commands=[dict(scene=cards.VISUALS[4],accepted=True,returncode=0,output=json.dumps(profile)),dict(scene=cards.VISUALS[5],accepted=True,returncode=0,output='spaCR: 1.5.1.3\nQt: 6.11.2\n')]
 proof=dict(completed_capture=True,verification_recorded_live=True,application_source_modified=False,installed_identity=dict(version='1.5.1.3'))
 accepted=dict(completed_capture=True,original_evidence_preserved=True,frame_origins={v:dict(image_sha256=hashes[v+'.png']) for v in cards.VISUALS[4:6]})
 for name,value in [('verification-commands.json',commands),('verification-provenance.json',proof),('accepted-provenance.json',accepted)]:
  (folder/name).write_text(json.dumps(value))
 monkeypatch.setattr(cards,'SOURCES',{name:cards.digest(folder/name) for name in cards.SOURCES})
 return folder

@pytest.mark.parametrize('fault',[None,'bytes','profile','version','verification','linkage'])
def test_actual_result_contract_rejects_tampering_and_rebound_wrong_facts(result_sources,fault):
 folder=result_sources
 if fault=='bytes':(folder/'verification-commands.json').write_text('tampered')
 elif fault:
  name='verification-commands.json' if fault in ('profile','version') else 'verification-provenance.json' if fault=='verification' else 'accepted-provenance.json'
  value=cards.read(folder/name)
  if fault=='profile':profile=json.loads(value[0]['output']);profile['active_backend']='cuda';value[0]['output']=json.dumps(profile)
  elif fault=='version':value[1]['output']='spaCR: current\nQt: 6.11.2\n'
  elif fault=='verification':value['verification_recorded_live']=False
  else:value['frame_origins'][cards.VISUALS[4]]['image_sha256']='c'*64
  (folder/name).write_text(json.dumps(value));cards.SOURCES[name]=cards.digest(folder/name)
 if fault is None:assert cards.checked_results(folder)[0]['active_backend']=='cpu'
 else:
  with pytest.raises(ValueError):cards.checked_results(folder)

def test_feature_definitions_preserve_commands_and_label_original_examples():
 from stage_lesson import REPO
 import render_installer_instruction_cards as normal
 before={v:(title,rows,note) for v,title,rows,note in normal.installer_cards('1.5.1.3')};after=cards.definitions(REPO)
 assert tuple(after)==cards.CARDS
 mapping={cards.VISUALS[0]:'18_download_desktop',**{v:v for v in cards.CARDS if v not in cards.VISUALS[:1]+cards.VISUALS[4:6]}}
 for visual,original in mapping.items():assert [command for label,command in after[visual][1]]==[command for label,command in before[original][1]]
 assert after[cards.VISUALS[4]][1][0][0]=='Example installation profile'
 assert 'spaCR: 1.5.1.3' in [value for label,value in after[cards.VISUALS[5]][1]]
 text=json.dumps(after)
 assert not any(word in text.lower() for word in ['not a windows recording','not a macos recording','our isolated run','not executed','does not claim'])

def test_normal_ten_card_generation_binds_pixels_text_definitions_and_original_results(result_sources,maintenance_plan,tmp_path,monkeypatch):
 from stage_lesson import REPO
 import published_native_reuse as reuse
 english,clock,*_=maintenance_plan;baseline=tmp_path/'baseline';baseline.mkdir();source=baseline/'clock.json';source.write_text(json.dumps(clock))
 monkeypatch.setattr(cards,'TIMING_SHA',cards.digest(source))
 bindings=dict(master=dict(sha256=cards.MASTER_SHA),timing=dict(sha256=cards.TIMING_SHA,path='clock.json'))
 monkeypatch.setattr(reuse,'_published_inputs',lambda *a:(baseline,clock,bindings,{}))
 verification=tmp_path/'verification';accepted=tmp_path/'accepted';verification.mkdir();accepted.mkdir()
 for incoming,name in [('verification-commands.json','commands.json'),('verification-provenance.json','provenance.json')]:
  (verification/name).write_bytes((result_sources/incoming).read_bytes())
 (accepted/'provenance.json').write_bytes((result_sources/'accepted-provenance.json').read_bytes())
 for v in cards.VISUALS[4:6]:(accepted/(v+'.png')).write_bytes((result_sources/(v+'.png')).read_bytes())
 folder=tmp_path/'references';cards.render(REPO,english,baseline,verification,accepted,folder)
 assert tuple(cards.validate_cards(folder,english,REPO))==cards.CARDS
 with pytest.raises(FileExistsError):cards.render(REPO,english,baseline,verification,accepted,folder)
 contract=folder/'cards.json';saved=contract.read_bytes();value=cards.read(contract);value['reference_revision']='unknown';contract.write_text(json.dumps(value))
 with pytest.raises(ValueError):cards.validate_cards(folder,english,REPO)
 contract.write_bytes(saved)
 changed=deepcopy(english);changed['scenes'][4]['narration']+=' changed'
 with pytest.raises(ValueError):cards.validate_cards(folder,changed,REPO)
 previous=cards.definitions
 def altered(repo):
  rows=deepcopy(previous(repo));key=next(iter(rows));title,lines,note=rows[key];rows[key]=(title+' changed',lines,note);return rows
 with monkeypatch.context() as local:
  local.setattr(cards,'definitions',altered)
  with pytest.raises(ValueError):cards.validate_cards(folder,english,REPO)
 path=folder/(cards.CARDS[0]+'.png');path.write_bytes(path.read_bytes()+b'changed')
 with pytest.raises(ValueError):cards.validate_cards(folder,english,REPO)
