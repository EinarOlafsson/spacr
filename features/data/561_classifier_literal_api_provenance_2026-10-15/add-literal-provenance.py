from pathlib import Path
import gzip,hashlib,json,sys
root=Path.cwd();scratch=Path(__file__).parent;sys.path.insert(0,str(root/'tools'))
import build_documentation_i18n as b
sha=lambda v:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
plan=json.loads((scratch/'run-plan.json').read_text());en=json.loads(gzip.decompress((scratch/'published-en.json.gz').read_bytes()));key='spacr.torch_artifacts.load_model_artifact';label=key+'#8';source=b.translatable_blocks(plan['source_docs'][key])[0][8]
assert source=='``(model, metadata)``.' and b._api_translation_source(source)==source
out=root/'features/data/561_classifier_literal_api_provenance_2026-10-15';out.mkdir(parents=True,exist_ok=True)
for name in ['generation.log','generation.time','generation-stats.json','run-plan.json']:(out/('initial-'+name+'.gz')).write_bytes(gzip.compress((scratch/name).read_bytes(),mtime=0))
rows=[]
for lang in b.MODEL_SPECS:
 old=json.loads((b.API_DIR/f'{lang}.json').read_text());entry=old['symbols'][key];assert entry['source_sha256']==en['symbols'][key]['source_sha256'];oldsrc=b.translatable_blocks(en['symbols'][key]['text'])[0][7];oldtarget=b.translatable_blocks(entry['text'])[0][7]
 assert oldsrc==oldtarget==source
 assert b._reviewed_api_block_valid(source,source,lang)
 row={'label':label,'source':source,'context':source,'source_sha256':sha(source),'translation':source,'review_basis':'Exact unchanged tuple return schema, including identifiers and punctuation; source-proven prior catalog target is byte-identical. Code literal has no prose to translate. Independent coordinator technical approval; no identity allowlist or gate changes.','previous_label':key+'#7','previous_target_sha256':sha(oldtarget),'baseline_catalog_sha256':plan['baseline_catalogs'][lang]}
 p=root/'docs/i18n/reviewed/api'/lang/'2026-10-15-f561-loader-literal-provenance-home-b.json'
 p.write_text(json.dumps({'schema':1,'language':lang,'review_kind':'Exact-source technical identity review for code-only return schema; no native-speaker claim','records':[row]},ensure_ascii=False,indent=2)+'\n')
 assert b.reviewed_api_block_translations(plan['source_docs'],lang)[source]==source
 rows.append({'language':lang,'file':str(p.relative_to(root)),'sha256':sha(p.read_bytes()),'source':source,'context':source,'old_label':key+'#7','new_label':label,'old_target':oldtarget,'new_target':source,'old_entry_source_sha256':entry['source_sha256']})
receipt={'status':'nine exact code-literal identity records pass normal source/context review loaders','source_files':plan['source_files'],'records':rows,'initial_refusal':{'before_any_catalog_write':True,'prepared_locales':[],'written_locales':[],'decoder_guard_requests':1,'actual_model_decodes':0,'runtime_collector_requests':0,'full_api_collector_requests':0},'changed_target_bytes':0,'policy_changes':0}
(out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');(out/'add-literal-provenance.py').write_bytes(Path(__file__).read_bytes());print('PASS nine normal identity review loaders; prior target bytes exact')
