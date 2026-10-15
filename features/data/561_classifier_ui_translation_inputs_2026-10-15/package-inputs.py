from pathlib import Path
import ast,copy,gzip,hashlib,json,shutil,subprocess,sys
root=Path.cwd();scratch=Path('/mnt/wd4tb/scratch/home-b-f561-classifier-ui-20261015');sys.path.insert(0,str(root/'tools'));import build_documentation_i18n as a;import build_i18n_catalogs as r
sha=lambda x:hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
plan=json.loads((scratch/'translation-obligations.json').read_text());assert sha((root/'spacr/qt/screens/embeddings.py').read_bytes())==plan['source_sha256']
check=json.loads((scratch/'input-check.json').read_text());assert not check['failures']
targets={}
for name in ['west','middle','asia']:
 f=scratch/f'targets/{name}.json';assert sha(f.read_bytes())==check['target_files'][name];targets.update(json.loads(f.read_text()))
changes=json.loads((root/'features/data/561_classifier_ui_2026-10-15/api-dependencies.json').read_text())['changes'];docs={x['symbol']:x['after'] for x in changes};key='spacr.qt.screens.embeddings.EmbeddingsScreen.closeEvent';docs[key]=docs[key].replace('this closed screen','this closed GUI view')
for row in plan['api']:
 symbol,_,idx=row['label'].rpartition('#');assert a.translatable_blocks(docs[symbol])[0][int(idx)]==row['source'];assert a._api_translation_source(row['source'])==row['context']
paths={};runtime_name='2026-10-15-f561-classifier-ui-runtime-home-b.json';api_name='2026-10-15-f561-classifier-ui-api-home-b.json'
for lang,parts in targets.items():
 runtime=[];api=[]
 for idx,target in sorted(parts['runtime'].items(),key=lambda x:int(x[0])):
  row=plan['runtime'][int(idx)];source=row['source'];record={'table':'ui','key':source,'source':source,'source_sha256':sha(source),'translation':target};assert set(record)=={'table','key','source','source_sha256','translation'};runtime.append(record)
 for idx,target in sorted(parts['api'].items(),key=lambda x:int(x[0])):
  row=plan['api'][int(idx)];api.append({'label':row['label'],'source':row['source'],'context':row['context'],'source_sha256':row['source_sha256'],'previous_same_source_indexes':[],'translation':target,'review_basis':'Exact-source technical translation authoring with separate coordinator semantic review. Raw crop/channel/label-group contracts, worker cancellation/lifecycle and protected literals retained; normal source/context and actual inline markup gates passed. No native-speaker claim.'})
 assert len(runtime)==34 and len(api)==10
 for domain,filename,records in [('runtime',runtime_name,runtime),('api',api_name,api)]:
  f=root/'docs/i18n/reviewed'/domain/lang/filename;assert not f.exists(),f
  payload={'schema':1,'language':lang,'records':records}
  if domain=='api':payload.update(review_kind='Exact-source technical translation with independent coordinator semantic review; no native-speaker signoff',note='F561 classifier training controls source09e53ecd80 with doc-context clarificationac47b32b60. Ten changed/new blocks in eight UI-owned symbols; six are new enabled nested helpers. Unchanged closeEvent#2 and all backend81 API records remain unchanged. Catalog generation is separate.')
  f.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n');paths[str(f.relative_to(root))]=sha(f.read_bytes())
retired=[];byfile={}
for row in plan['precise_old_api_records_requiring_retirement']:byfile.setdefault(row['path'],[]).append(row)
for name,rows in byfile.items():
 f=root/name;data=f.read_bytes();assert sha(data)==rows[0]['file_sha256'];payload=json.loads(data);before=copy.deepcopy(payload)
 for row in rows:
  index=row['record_index'];assert payload['records'][index]==row['record'];payload['records'][index].update(retired=True,retired_date='2026-10-15',retired_reason=f'F561 classifier training extends busy/close ownership to the classifier runner. Replacement exact-source review is {api_name}; original source, context and translation are retained as historical evidence.')
  retired.append({'path':name,'record_index':index,'before':row['record'],'after':payload['records'][index],'before_file_sha256':sha(data)})
 indices={x['record_index'] for x in rows}
 assert all(b==v for i,(b,v) in enumerate(zip(before['records'],payload['records'])) if i not in indices)
 f.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 for row in retired:
  if row['path']==name:row['after_file_sha256']=sha(f.read_bytes())
assert len(retired)==36
for row in plan['preserve_api_records']:
 now=json.loads((root/row['path']).read_text())['records'][row['record_index']];assert now==row['record']
def refuse(*args,**kwargs):raise AssertionError('Input packaging may not collect canonical/public sources or generate catalogs')
r.canonical_sources=refuse;a.public_docstrings=refuse
loader={}
for lang,parts in targets.items():
 reviewed=a.reviewed_api_block_translations(docs,lang)
 for idx,target in parts['api'].items():assert reviewed[plan['api'][int(idx)]['source']]==target
 keep=plan['preserve_api_records'];row=next(x for x in keep if f'/{lang}/' in x['path']);assert reviewed[row['record']['source']]==row['record']['translation']
 loader[lang]={'normal_scoped_api_loader':'PASS','new_or_changed_reviewed_blocks':10,'unchanged_block_preserved':1}
out=root/'features/data/561_classifier_ui_translation_inputs_2026-10-15';out.mkdir(parents=True,exist_ok=True)
for name in ['input-check.json','translation-obligations.json','context-target-followup.json','check-targets.py','context-followup.py','package-inputs.py']:
 shutil.copyfile(scratch/name,out/name)
for name in ['west','middle','asia']:(out/f'{name}-targets.json.gz').write_bytes(gzip.compress((scratch/f'targets/{name}.json').read_bytes(),mtime=0))
(out/'scoped-docstrings.json').write_text(json.dumps(docs,ensure_ascii=False,indent=2)+'\n')
(out/'retirement-provenance.json').write_text(json.dumps(retired,ensure_ascii=False,indent=2)+'\n')
receipt={'source_commit':plan['source_commit'],'source_sha256':plan['source_sha256'],'new_review_files':paths,'records':{'runtime':306,'api':90,'authored_runtime':279,'exact_existing_runtime_reuse':27,'retired_api':36,'unchanged_closeEvent_api_preserved':9},'normal_scoped_api_review_loader':loader,'pure_gate_counts':check['counts'],'canonical_collection':False,'catalog_writes':False,'backend_api_scope':'Original Hard81 records untouched','generation_gate':'Await final independent source/target input binding and integration; normal runtime/API source-currency, scoped writers and full audits remain required.'}
(out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'review_files':len(paths),'runtime_records':306,'api_records':90,'retired_records':len(retired),'normal_scoped_loader_locales':len(loader),'evidence':str(out)}))
