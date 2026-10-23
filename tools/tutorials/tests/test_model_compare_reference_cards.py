"""Exact published arrangement and saved-output reference validation; no capture or inference."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,sys
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'authoring/tools')]
import published_native_mix as mix
import render_model_compare_reference_cards as cards

@pytest.fixture
def arrangement():
 english=dict(id='21_model_compare',scenes=[dict(visual=v,narration='Use '+v) for v in cards.VISUALS])
 original=dict(scenes=[dict(scene=i+1,visual=v,speech_start=i*3,scene_end=(i+1)*3,duration=3,native_live_video=i<2) for i,v in enumerate(cards.VISUALS)])
 tracks=[dict(scenes=[dict(speech_start=i*2,speech_end=i*2+1.48,duration=2) for i in range(6)]) for _ in range(50)]
 references={v:dict(visual=v,kind='authored_reference_card',narration_sha256=mix.identity(english['scenes'][i]['narration'])) for i,v in enumerate(cards.VISUALS) if v in cards.CARDS}
 bindings=dict(master=dict(sha256='a'*64),timing=dict(sha256='b'*64))
 edit=dict(schema=1,lesson=english['id'],master_sha256='a'*64,visual_timing_sha256='b'*64,scenes=[])
 return english,original,tracks,references,edit,bindings,{}

def test_three_references_preserve_two_native_and_four_held_scenes_and_original_inputs(arrangement):
 snapshot=deepcopy(arrangement);rows=mix.plan_mix(*arrangement)
 assert [r['visual'] for r in rows]==list(cards.VISUALS)
 assert [r['native_live_video'] for r in rows]==[True,True,False,False,False,False]
 assert [i+1 for i,r in enumerate(rows) if 'reference_card' in r]==[3,4,6]
 assert [(r['source_start_frame'],r['source_keep_frames']) for r in rows if 'reference_card' not in r]==[(0,[[0,90]]),(90,[[0,90]]),(360,[[0,90]])]
 assert arrangement==snapshot

@pytest.mark.parametrize('fault',['source_order','authored_order','native_flag','integer_flag','extra_card','missing_card','edit_native','voice_missing'])
def test_model_compare_refresh_rejects_scope_or_identity_changes(arrangement,fault):
 a=deepcopy(arrangement);english,original,tracks,references,edit,bindings,fences=a
 if fault=='source_order':original['scenes'][0],original['scenes'][1]=original['scenes'][1],original['scenes'][0]
 elif fault=='authored_order':english['scenes'][2],english['scenes'][3]=english['scenes'][3],english['scenes'][2]
 elif fault=='native_flag':original['scenes'][4]['native_live_video']=True
 elif fault=='integer_flag':original['scenes'][0]['native_live_video']=1
 elif fault=='extra_card':references[cards.VISUALS[4]]=deepcopy(next(iter(references.values())))
 elif fault=='missing_card':references.pop(cards.CARDS[0])
 elif fault=='edit_native':edit['scenes']=[dict(visual=cards.VISUALS[0])]
 else:tracks.pop()
 with pytest.raises(ValueError):mix.plan_mix(*a)

@pytest.fixture
def saved_results(tmp_path,monkeypatch):
 folder=tmp_path/'results';folder.mkdir()
 csv='field,n_objects_a,n_objects_b,n_matched,mean_matched_iou,ari\nexample,94,95,91,0.96,0.961\n'
 (folder/'comparison.csv').write_text(csv);(folder/'comparison.png').write_bytes(b'test saved figure')
 expected={n:cards.digest(folder/n) for n in ('comparison.csv','comparison.png')}
 report=dict(accepted=True,source_unchanged=True,accuracy_validated=False,different_model_weights_compared=False,gui_workflow_completed=False,inference_performed=False,artifacts=expected,sources={n:'a'*64 for n in ('image.tif','batch.tif','preview.npy')},comparison=dict(n_objects_a=94,n_objects_b=95,n_matched=91,mean_matched_iou=.96,ari=.961))
 (folder/'run.json').write_text(json.dumps(report));expected={**expected,'run.json':cards.digest(folder/'run.json')}
 monkeypatch.setattr(cards,'RESULTS',expected)
 return folder,report

def test_reference_numbers_come_from_saved_csv_and_report(saved_results):
 folder,report=saved_results;comparison=cards.checked_results(folder)
 assert comparison==report['comparison']
 assert 'A 94' in cards.definitions(comparison)[cards.CARDS[2]][1][1]

@pytest.mark.parametrize('fault',['csv_bytes','report_bytes','missing_figure','symlink','csv_report_disagree'])
def test_reference_sources_fail_closed_without_new_comparison(saved_results,monkeypatch,fault):
 folder,report=saved_results
 if fault=='csv_bytes':(folder/'comparison.csv').write_text('changed')
 elif fault=='report_bytes':(folder/'run.json').write_text('{}')
 elif fault=='missing_figure':(folder/'comparison.png').rename(folder/'preserved-figure.png')
 elif fault=='symlink':
  original=folder/'comparison.csv';original.rename(folder/'preserved.csv');original.symlink_to('preserved.csv')
 else:
  path=folder/'comparison.csv';path.write_text(path.read_text().replace(',94,',',93,'));report['artifacts']['comparison.csv']=cards.digest(path)
  (folder/'run.json').write_text(json.dumps(report));monkeypatch.setattr(cards,'RESULTS',{n:cards.digest(folder/n) for n in cards.RESULTS})
 with pytest.raises(ValueError):cards.checked_results(folder)
