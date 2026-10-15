from pathlib import Path
import copy,gzip,hashlib,json,shutil,sys
root=Path.cwd();scratch=Path(__file__).parent;sys.path.insert(0,str(root/'tools'))
import build_documentation_i18n as a;import build_i18n_catalogs as r
sha=lambda x:hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
plan=json.loads((scratch/'sources.json').read_text());repairs=json.loads((scratch/'korean-repairs.json').read_text());paths={};normalizations=[];retirements=[]
for path,expected in plan['source_files'].items():assert sha((root/path).read_bytes())==expected
for lang,data in plan['inputs'].items():
 for packet in data['packets'][:2]:
  original=root/packet['path'];raw=original.read_bytes();assert sha(raw)==packet['sha256'];payload=json.loads(raw);before=copy.deepcopy(payload)
  initial='561_own_dino_classifier_backend_' in packet['path']
  name='2026-10-15-f561-own-classifier-backend-home-b.json' if initial else '2026-10-15-f561-classifier-application-backend-home-b.json'
  dest=root/'docs/i18n/reviewed/api'/lang/name;assert not dest.exists()
  for index,record in enumerate(payload['records']):
   label=record['label']
   if initial and label=='spacr.torch_artifacts.load_model_artifact#2':
    record.update(retired=True,retired_date='2026-10-15',retired_reason='The saved-classifier adapter accepts restricted seekable binary streams; exact replacement is in 2026-10-15-f561-classifier-application-backend-home-b.json. Original source/context/translation retained.')
    retirements.append({'original_packet':packet['path'],'destination':str(dest.relative_to(root)),'record_index':index,'before':before['records'][index],'after':copy.deepcopy(record)})
    continue
   context=a._api_translation_source(record['source'])
   if context!=record['context']:
    normalizations.append({'original_packet':packet['path'],'destination':str(dest.relative_to(root)),'record_index':index,'source':record['source'],'source_sha256':record['source_sha256'],'translation':record['translation'],'before':record['context'],'after':context})
    record['context']=context
   assert record['source']==before['records'][index]['source'] and record['translation']==before['records'][index]['translation']
  payload['source_packet_provenance']={'path':packet['path'],'sha256':packet['sha256'],'operation':'Copy original reviewed drafts; normalize exact current contexts only; retain superseded stream paragraph as retired historical evidence.'}
  dest.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n');paths[str(dest.relative_to(root))]=sha(dest.read_bytes())
assert len(normalizations)==36 and len(retirements)==9
for row in plan['remaining_active_prior_review_retirements']:
 p=root/row['path'];raw=p.read_bytes();assert sha(raw)==row['file_sha256'];payload=json.loads(raw);original=copy.deepcopy(payload)
 i=row['record_index'];assert payload['records'][i]==row['record'];payload['records'][i].update(retired=True,retired_date='2026-10-15',retired_reason='The same loader parameter moved from block3 to4. Its exact source has a reviewed Korean markup/semantic repair in 2026-10-15-f561-loader-ko-shift-repairs-home-b.json; original source/context/target retained.')
 assert all(b==v for j,(b,v) in enumerate(zip(original['records'],payload['records'])) if j!=i)
 p.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n');retirements.append({'path':row['path'],'before_file_sha256':sha(raw),'after_file_sha256':sha(p.read_bytes()),'record_index':i,'before':row['record'],'after':copy.deepcopy(payload['records'][i])});paths[row['path']]=sha(p.read_bytes())
records=[]
for row in repairs:
 records.append({'label':row['label'],'source':row['source'],'source_sha256':row['source_sha256'],'context':row['context'],'previous_same_source_indexes':[int(row['old_label'].rpartition('#')[2])],'translation':row['new_target'],'review_basis':'Exact-source technical repair and independent coordinator semantic review; restores real RST rendering plus documented loader defaults/legacy exceptions. Original catalog bytes/source/old index retained in bound provenance; no native-speaker claim.'})
p=root/'docs/i18n/reviewed/api/ko/2026-10-15-f561-loader-ko-shift-repairs-home-b.json';assert not p.exists();p.write_text(json.dumps({'schema':1,'language':'ko','records':records},ensure_ascii=False,indent=2)+'\n');paths[str(p.relative_to(root))]=sha(p.read_bytes())
def refuse(*args,**kw):raise AssertionError('No canonical collection/model/catalog writer allowed during input packaging')
r.canonical_sources=refuse;a.public_docstrings=refuse;a._translate_blocks=refuse
loaded={}
for lang,data in plan['inputs'].items():
 review=a.reviewed_api_block_translations(plan['source_docs'],lang)
 for label,row in data['active_records'].items():assert review[row['source']]==row['translation'],(lang,label)
 if lang=='ko':
  for row in repairs:assert review[row['source']]==row['new_target']
 loaded[lang]={'normal_selected_review_loader':'PASS','authored_current_records':26,'approved_historical_repairs':4 if lang=='ko' else 0}
for packet in [p for d in plan['inputs'].values() for p in d['packets']]:assert sha((root/packet['path']).read_bytes())==packet['sha256']
out=root/'features/data/561_combined_api_inputs_2026-10-15';out.mkdir(parents=True,exist_ok=True)
for name in ['inventory.py','sources.json','visibility-proof.json','preflight.py','preflight.json','preflight.log','korean-repairs.py','korean-repairs.json','independent-korean-repairs-peer.json','package-inputs.py']:
 shutil.copyfile(scratch/name,out/name)
peer=Path('/mnt/wd4tb/scratch/home-b-f561-classifier-apply-ui-20261015/independent-final-target-peer.json');shutil.copyfile(peer,out/'independent-final-ui243-target-peer.json')
(out/'context-normalizations.json').write_text(json.dumps(normalizations,ensure_ascii=False,indent=2)+'\n');(out/'retirement-provenance.json').write_text(json.dumps(retirements,ensure_ascii=False,indent=2)+'\n')
receipt={'source_files':plan['source_files'],'selected_symbols':23,'new_symbols':19,'current_authored_records':234,'korean_historical_repairs':4,'current_selected_review_records':238,'backend_context_only_normalizations':36,'backend_stream_drafts_retired':9,'active_historical_shift_record_retired':1,'previous_ui_records_retired_in_0b55':18,'other_shifted_targets_to_preserve':41,'new_review_files_and_exact_retired_file':paths,'normal_selected_review_loader':loaded,'all_original_backend_draft_packets_unchanged':True,'all_previous_ui_input_packets_unchanged':True,'canonical_collector_called':False,'catalog_writer_called':False,'generator_gate':'Wait for coordinator final source/baseline/Original writer release. Normal source currentness/scoped translation and Help outputs still pending.'}
(out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'source_bound_current_records':234,'korean_repairs':4,'normal_contexts':36,'retirements':len(retirements),'normal_loaders':len(loaded),'evidence':str(out)}))
