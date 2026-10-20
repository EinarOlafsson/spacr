from pathlib import Path
import ast,gzip,hashlib,json,re,subprocess,sys,tempfile
r=Path.cwd();s=Path('/mnt/wd4tb/scratch/n94-current-contract-audit-20261015');old='8fc2aa5601847ccd8f7cff287c36c6a9ee00ba53';sha=lambda b:hashlib.sha256(b).hexdigest()
def nodes(text):
 out={}
 def walk(body,p=''):
  for n in body:
   if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):out[p+n.name]=n;walk(n.body,p+n.name+'.')
 walk(ast.parse(text).body);return out
scope={'spacr/classifier_evaluation.py':['SplitReport','normalize_split_level','split_columns_for','split_group_values','grouped_split'],'spacr/ml.py':['ml_analysis'],'spacr/io.py':['generate_dataset_from_lists'],'spacr/utils.py':['augment_classes'],'spacr/surrogate.py':['fit_surrogate'],'spacr/active_learning.py':['retrain_round','_write_round_card'],'spacr/annotation_umap_qc.py':['fit_on_controls'],'spacr/measure.py':['_phases_by_torch','_phases_by_xgboost'],'spacr/qt/screens/annotate.py':['_RetrainWorker.run']};proof={}
for p,names in scope.items():
 current=(r/p).read_bytes();before=subprocess.check_output(['git','show',old+':'+p]);a=nodes(current);b=nodes(before);proof[p]={'source_sha256':sha(current),'hosted_sha256':sha(before),'symbols':{}}
 for name in names:
  dump=ast.dump(a[name]);prior=ast.dump(b[name]);proof[p]['symbols'][name]={'AST_sha256':sha(dump.encode()),'hosted_AST_sha256':sha(prior.encode()),'identical_hosted':dump==prior,'calls':sorted({ast.unparse(n.func) for n in ast.walk(a[name]) if isinstance(n,ast.Call)})}
assert all(v['identical_hosted'] for p,x in proof.items() for v in x['symbols'].values())
files=['tests/test_grouped_splitting.py','tests/test_every_split_goes_through_the_shared_splitter.py','tests/test_cell_cycle_torch_grouped_holdout.py'];testproof={};log=Path('/mnt/wd4tb/scratch/ci-38699600639-current-triage-20261015/failed.log');passes={p:{} for p in files};fails=[]
with log.open(errors='replace') as f:
 for line in f:
  for p in files:
   if p not in line:continue
   match=re.search(r'(?:PASSED|FAILED) ('+re.escape(p)+r'::\S+)',line)
   if not match:continue
   node=match.group(1)
   if 'PASSED '+node in line:passes[p][node]=passes[p].get(node,0)+1
   else:fails.append(node)
for p in files:
 raw=(r/p).read_bytes();before=subprocess.check_output(['git','show',old+':'+p]);assert raw==before,p
 testproof[p]={'source_sha256':sha(raw),'byte_identical_hosted':True,'assertions':len([n for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.Assert)]),'hosted_unique_PASS_nodes':len(passes[p]),'hosted_PASS_nodes':passes[p]}
assert not fails,fails
calls=[]
for p in (r/'spacr').rglob('*.py'):
 for n in ast.walk(ast.parse(p.read_text())):
  if isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id=='train_test_split') or (isinstance(n.func,ast.Attribute) and n.func.attr=='train_test_split')):calls.append({'file':str(p.relative_to(r)),'line':n.lineno})
assert len(calls)==1 and calls[0]['file']=='spacr/classifier_evaluation.py'
source='The requested acquisition groups cannot place every class in both training and held-out data.';measure=nodes((r/'spacr/measure.py').read_text());assert all(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='tr' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value==source for n in ast.walk(measure[k])) for k in ['_phases_by_torch','_phases_by_xgboost'])
sys.path.insert(0,str(r/'tools'));import build_i18n_catalogs as normal
canonical={'setting_labels':{},'setting_tooltips':{},'ui':{source}};langs=['sv','de','es','zh_CN','fr','pt','hi','ko','is'];locale={}
with tempfile.TemporaryDirectory(dir=s,prefix='exact-one-caption-normal-inputs-') as d:
 normal.REVIEWED_RUNTIME_DIR=Path(d);normal.canonical_sources=lambda:canonical
 for lang in langs:
  p=r/f'docs/i18n/reviewed/runtime/{lang}/2026-10-14-cell-cycle-grouped-evaluation.json';payload=json.loads(p.read_text());rows=[x for x in payload['records'] if x['source']==source];assert len(rows)==1
  out=Path(d)/lang;out.mkdir();(out/'owned.json').write_text(json.dumps({'schema':1,'language':lang,'records':rows},ensure_ascii=False));normal.reviewed_runtime_translations.cache_clear();reviewed=normal.reviewed_runtime_translations(lang)
  tree=ast.parse((r/f'spacr/qt/i18n_catalogs/{lang}.py').read_text());tables={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'UI','SOURCE_HASHES'}};target=tables['UI'][source];assert tables['SOURCE_HASHES'][('UI',source)]==sha(source.encode());assert reviewed[source]==target;assert normal._translation_candidate_valid(source,target,lang,force=normal._looks_translatable(source))
  locale[lang]={'target':target,'target_sha256':sha(target.encode()),'review_file':str(p.relative_to(r)),'review_sha256':sha(p.read_bytes()),'exact_normal_scoped_review_and_candidate_PASS':True}
ownertext=(r/'tools/build_i18n_catalogs.py').read_text();assert '"_phases_by_xgboost", "_phases_by_torch", "_wound_plane"' in ownertext
receipt={'scope':'Read-only original N94 clauses/current source/history and existing test/runtime evidence; no tests,fullcollectors,generators,models,training,network,GPU or source changes.','base_private':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'hosted_source':old,'item_sha256':sha((r/'features/new/94_train_test_splits_must_group_by_well.txt').read_bytes()),'owning_source':proof,'existing_test_source_and_hosted_PASS':testproof,'hosted_failures_exact_test_files':fails,'only_actual_train_test_split_call':calls,'current_diagnostic_owner_tool_sha256':sha((r/'tools/build_i18n_catalogs.py').read_bytes()),'stale_extractor_request_now_resolved':'Actual measure.py owner allowlist includes BOTH _phases_by_torch/_phases_by_xgboost; same exact literal in both functions.','one_runtime_source':source,'runtime_source_sha256':sha(source.encode()),'nine_normal_scoped_review_and_compiled_checks':locale,'original_realdata_or_model_training_required_clause':False,'limits':'No real model training/GPU/biological phase efficacy certified. Those are separate F535 boundaries, absent from original N94 split-policy contract. Historical direct API UMAP no-groups explicit cell behavior retained as dated accepted note/tests; no new fallback introduced. Legacy gui_elements.py no longer exists; actual GUI retrain worker routes shared active_learning.retrain_round. No blanket inference all future split code will remain correct.'}
(s/'source-test-runtime-proof.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'all_selected_symbols_identical_hosted':True,'test_files':{p:v['hosted_unique_PASS_nodes'] for p,v in testproof.items()},'current_direct_split_calls':calls,'runtime_locale_checks':len(locale),'pending_owner_request_resolved':True}),flush=True)
