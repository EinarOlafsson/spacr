from pathlib import Path
import ast,gzip,hashlib,json,subprocess,sys
root=Path.cwd();scratch=Path(__file__).parent
sys.path.insert(0,str(root/'tools'));import build_documentation_i18n as a;import nested_helper_docs as n;import build_i18n_catalogs as r
sha=lambda x:hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def reject(*a,**kw):raise AssertionError('Inventory cannot collect all sources or write artifacts')
a.public_docstrings=reject;r.canonical_sources=reject
files=['spacr/embeddings.py','spacr/torch_artifacts.py','spacr/qt/screens/embeddings.py']
snapshot=scratch/'source-snapshot'; hashes={};all_docs={}
for rel in files:
 p=root/rel;raw=p.read_bytes();hashes[rel]=sha(raw);q=snapshot/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw)
 module=rel[:-3].replace('/','.');tree=ast.parse(raw)
 if a._clean_doc(tree):all_docs[module]=a._clean_doc(tree)
 for node in a._module_scope_nodes(tree.body):
  if not isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):continue
  key=module+'.'+node.name
  visible=not node.name.startswith('_') if isinstance(node,ast.ClassDef) else a._is_visible_function_name(node.name,module_is_package=False)
  if not visible or not a._is_rendered_autoapi_entry(q,module,key):continue
  all_docs[key]=a._autoapi_class_doc(node) if isinstance(node,ast.ClassDef) else a._clean_doc(node)
  if isinstance(node,ast.ClassDef):
   for child in a._module_scope_nodes(node.body):
    if not isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
    ck=key+'.'+child.name
    if a._is_visible_function_name(child.name,module_is_package=False) and a._is_rendered_autoapi_entry(q,module,ck):all_docs[ck]=a._clean_doc(child)
mods={rel[:-3].replace('/','.') for rel in files}&set(n.ENABLED_MODULES)
for row in n.entries(n.inventory(snapshot,ignore_patterns=a.AUTOAPI_IGNORE),modules=mods):all_docs[row.qualified_key]=row.docstring
old81=root/'features/data/561_own_dino_classifier_backend_2026-10-15/api-drafts'
new27=root/'features/data/561_classifier_application_2026-10-15/api-inputs/drafts'
ui90='2026-10-15-f561-classifier-ui-api-home-b.json';ui63='2026-10-15-f561-classifier-apply-ui-api-home-b.json'
source_records={};inputs={};superseded=[];context_issues=[]
for lang in r.MODEL_SPECS:
 latest={};packets=[]
 for path in [old81/f'{lang}.json',new27/f'{lang}.json',root/'docs/i18n/reviewed/api'/lang/ui90,root/'docs/i18n/reviewed/api'/lang/ui63]:
  payload=json.loads(path.read_text());assert payload['language']==lang
  packets.append({'path':str(path.relative_to(root)),'sha256':sha(path.read_bytes()),'records':len(payload['records'])})
  for index,record in enumerate(payload['records']):
   row={**record,'input_path':str(path.relative_to(root)),'input_index':index}
   if row.get('retired'):
    superseded.append({'language':lang,'record':row,'reason':'Already retired by UI followup input'});continue
   label=row['label']
   if label in latest:
    superseded.append({'language':lang,'record':latest[label],'reason':'Backend saved-stream paragraph superseded by actual adapter input'})
   latest[label]=row
 assert len(latest)==26,(lang,len(latest))
 for label,row in latest.items():
  key,_,i=label.rpartition('#');source=a.translatable_blocks(all_docs[key])[0][int(i)]
  assert row['source']==source and row['source_sha256']==sha(source),(lang,label,'source')
  if row['context']!=a._api_translation_source(source):context_issues.append({'language':lang,'label':label,'path':row['input_path'],'before':row['context'],'after':a._api_translation_source(source),'source':source,'translation':row['translation']})
  if label in source_records:assert source_records[label]['source']==source
  else:source_records[label]={'label':label,'source':source,'source_sha256':sha(source),'context':row['context'],'context_sha256':sha(row['context'])}
 inputs[lang]={'packets':packets,'active_records':latest}
keys={label.rpartition('#')[0] for label in source_records};selected={k:all_docs[k] for k in sorted(keys)}
en_path=a.API_DIR/'en.json';en_raw=en_path.read_bytes();en=json.loads(en_raw);baseline=en['symbols']
new_symbols=sorted(keys-baseline.keys());existing=sorted(keys&baseline.keys());unchanged=[];shifts=[];uncovered=[]
for key,doc in selected.items():
 new=a.translatable_blocks(doc)[0];old=a.translatable_blocks(baseline[key]['text'])[0] if key in baseline else []
 for i,source in enumerate(new):
  label=f'{key}#{i}'
  if label in source_records:continue
  if i<len(old) and old[i]==source:unchanged.append(label)
  elif source in old:shifts.append({'label':label,'old_label':f'{key}#{old.index(source)}','source':source,'source_sha256':sha(source)})
  else:uncovered.append({'label':label,'source':source})
assert not uncovered,uncovered
stale=[]
for lang in r.MODEL_SPECS:
 for path in sorted((root/'docs/i18n/reviewed/api'/lang).glob('*.json')):
  raw=path.read_bytes();payload=json.loads(raw)
  for i,row in enumerate(payload.get('records',[])):
   if row.get('retired'):continue
   label=row.get('label','');key,_,pos=label.rpartition('#')
   if key not in keys:continue
   blocks=a.translatable_blocks(selected[key])[0]
   if not pos.isdigit() or int(pos)>=len(blocks) or row['source']!=blocks[int(pos)] or row['context']!=a._api_translation_source(blocks[int(pos)]):
    stale.append({'language':lang,'path':str(path.relative_to(root)),'file_sha256':sha(raw),'record_index':i,'record':row})
proof={'source_files':hashes,'source_docs':selected,'eligible_symbols':sorted(keys),'new_symbols':new_symbols,'normal_helpers_used':['_is_visible_function_name','_is_rendered_autoapi_entry','_clean_doc','_autoapi_class_doc','translatable_blocks','nested_helper_docs.inventory','nested_helper_docs.entries'],'full_collector_called':False,'modules_in_nested_inventory':sorted(mods)}
(scratch/'visibility-proof.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
plan={'source_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'context_normalization_required':context_issues,'status':'Prepared inventory only; do not write until coordinator supplies final integrated baseline and closes Original writer window.','source_files':hashes,'source_docs':selected,'new_symbols':new_symbols,'expected_symbol_count':len(keys),'expected_changed_block_count':len(source_records),'changed_blocks':list(source_records.values()),'unchanged_same_index_blocks':unchanged,'source_identical_shifted_blocks':shifts,'active_backend_records':99,'active_ui_records':135,'active_total_records':234,'input_record_history':{'old_backend':81,'incremental_backend':27,'training_ui':90,'application_ui':63,'superseded_backend':9,'superseded_ui':18},'superseded_draft_records':superseded,'remaining_active_prior_review_retirements':stale,'inputs':inputs,'visibility_proof':{'path':str(scratch/'visibility-proof.json'),'sha256':sha((scratch/'visibility-proof.json').read_bytes())},'private_english_baseline':{'path':str(en_path.relative_to(root)),'sha256':sha(en_raw),'note':'Preparation tree snapshot only; production generation must snapshot final coordinated baseline again without full collection.'},'normal_generation':['Package original81 with exact stream9 retired; add incremental27 unchanged; no new target authorship.','Retire only declared active old rows for changed/moved owned labels; provenance exact. Validate all selected source/context records through normal reviewed_api_block_translations.','Translate selected final23 bodies with normal _translate_api_documents(reuse_history=True), refuse decoder and full runtime/API collectors. Prepare all9 locales before writing.','Normal write_language(preserve_other_symbols=True) for9 locales; normal _write_english_api_manifest with selected replacements in exact final existing English baseline. Preserve all unrelated records/metadata and required source history.','Normal build_help_search_index.py on final integrated source; actual Help consumer acceptance for selected/new symbols after source/hash binding. No direct generated edits.'],'commands':{'api':'flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/home-b-f561-combined-api-20261015/generate.py','help':'CUDA_VISIBLE_DEVICES= QT_QPA_PLATFORM=offscreen PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages:$PWD flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python tools/build_help_search_index.py'},'producer_script_status':'To adapt from committed normal F370 producer after coordinator approves final snapshot and input packaging; not yet executed.'}
(scratch/'sources.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'selected_symbols':len(keys),'new_symbols':len(new_symbols),'changed_blocks':len(source_records),'unchanged_blocks':len(unchanged),'shifted_blocks':len(shifts),'active_records':234,'retire_remaining':len(stale),'context_normalization_required':len(context_issues),'source_files':hashes,'plan':str(scratch/'sources.json')},indent=2))
