import ast,collections,gc,hashlib,json,pathlib,re,subprocess
repo=pathlib.Path('/mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015')
out=pathlib.Path('/mnt/wd4tb/scratch/n511-regionprops-closeout-20261015')
rev='20a4313bcd9d58bc9267c1b80d3b51c02206635d';host='8fc2aa5601847ccd8f7cff287c36c6a9ee00ba53'
langs=['en','sv','de','es','zh_CN','pt','hi','ko','is','fr']; retired=[f'{obj}_{bound}' for obj in ['cell','nucleus','pathogen'] for bound in ['min_area','max_area','min_intensity','max_intensity']]
def get(r,p):return subprocess.check_output(['git','show',f'{r}:{p}'],cwd=repo)
def sha(b):return hashlib.sha256(b).hexdigest()
def literal_tables(raw,names):
 tree=ast.parse(raw);result={}
 for n in tree.body:
  targets=n.targets if isinstance(n,ast.Assign) else [n.target] if isinstance(n,ast.AnnAssign) else []
  for target in targets:
   if isinstance(target,ast.Name) and target.id in names:result[target.id]=ast.literal_eval(n.value)
 return result
perkey={k:{'catalogs':{}} for k in retired};catalogs={}
for lang in langs:
 p=f'spacr/qt/i18n_catalogs/{lang}.py'; raw=get(rev,p); tables=literal_tables(raw,{'SETTING_LABELS','SETTING_TOOLTIPS','SOURCE_HASHES'})
 for k in retired:
  state={'setting_labels_present':k in tables['SETTING_LABELS'],'setting_tooltips_present':k in tables['SETTING_TOOLTIPS'],'label_source_hash_present':('SETTING_LABELS',k) in tables['SOURCE_HASHES'],'tooltip_source_hash_present':('SETTING_TOOLTIPS',k) in tables['SOURCE_HASHES']}
  assert not any(state.values());perkey[k]['catalogs'][lang]=state
 if lang=='en':english={t:tables[t]['object_filters'] for t in ['SETTING_LABELS','SETTING_TOOLTIPS']}
 for t in ['SETTING_LABELS','SETTING_TOOLTIPS']:
  assert tables['SOURCE_HASHES'][(t,'object_filters')]==sha(english[t].encode())
  assert tables[t]['object_filters']
 catalogs[lang]={'path':p,'sha256':sha(raw),'retired12_all_absent':True,'retained_object_filters_current_source_hashes_match_English':True}
 del tables;gc.collect()
settingsraw=get(rev,'spacr/settings.py');st=ast.parse(settingsraw)
for table in ['expected_types','tooltips','categories']:
 vals=[n.value for n in st.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==table for t in n.targets)]
 assert len(vals)==1
 seen={n.value for n in ast.walk(vals[0]) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
 for k in retired:assert k not in seen;perkey[k][table+'_literal_registry_absent']=True
 assert 'object_filters' in seen
reviewfiles={}
for lang in langs[1:]:
 for name in ['2026-09-15-mask-mean-bounds.json','2026-09-25-runtime-debt-second-pass-setting-help.json']:
  p=f'docs/i18n/reviewed/runtime/{lang}/{name}';raw=get(rev,p);d=json.loads(raw)
  matches=[r for r in d['records'] if r.get('key') in retired and not r.get('retired')]
  assert not matches
  reviewfiles[p]={'sha256':sha(raw),'active_retired12_records':0,'retained_records':len(d['records'])}
  for k in retired:perkey[k].setdefault('named_owner_review_files_without_active_record',[]).append(p)
def fdefs(r,p):
 t=ast.parse(get(r,p));result={}
 def walk(n,prefix=''):
  for node in getattr(n,'body',[]):
   if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
    name=prefix+node.name;result[name]=ast.dump(node,include_attributes=False);walk(node,name+'.')
 walk(t);return result
scopes={'spacr/qt/mask_engine.py':['_filter_catalogue','filter_properties','normalise_filters','legacy_filters','settings_filters','filter_removals','apply_filters','filter_report','filter_objects','object_filter_area_floor','parse_object_filters'],'spacr/qt/screens/make_masks.py':['ObjectFilterList','MakeMasksScreen.apply_object_filter','MakeMasksScreen._filter_base'],'spacr/settings.py':['_fold_object_bounds'],'spacr/utils.py':['_filter_objects']}
sourceproof={}
for p,names in scopes.items():
 a,b=fdefs(rev,p),fdefs(host,p);rows={}
 for name in names:
  assert name in a and name in b,name
  rows[name]=a[name]==b[name]
  assert rows[name],name
 sourceproof[p]={'sha256':sha(get(rev,p)),'hosted_owning_symbols_ast_unchanged':rows}
selected=[];counts=collections.defaultdict(collections.Counter)
log=pathlib.Path('/mnt/wd4tb/scratch/ci-38699600639-current-triage-20261015/failed.log')
for i,line in enumerate(log.open(errors='replace'),1):
 m=re.search(r'(PASSED|FAILED|SKIPPED) (tests/(?:qt/)?test_(?:make_masks_filters_are_any_regionprop|object_filters_are_any_regionprop)\.py)::([^\s]+)',line)
 if m:
  job=line.split('\t')[0];counts[(job,m[2])][m[1]]+=1;selected.append({'line':i,'job':job,'state':m[1],'nodeid':m[2]+'::'+m[3],'raw':line.rstrip()})
assert selected and not any(x['state']!='PASSED' for x in selected)
for p in ['tests/qt/test_make_masks_filters_are_any_regionprop.py','tests/test_object_filters_are_any_regionprop.py']:
 assert get(host,p)==get(rev,p)
 sourceproof[p]={'sha256':sha(get(rev,p)),'whole_file_identical_to_hosted':True}
(out/'hosted-owning-results.json').write_text(json.dumps(selected,indent=2)+'\n')
(out/'twelve-key-current-bindings.json').write_text(json.dumps(perkey,indent=2)+'\n')
receipt={'item':'N511','revision':rev,'decision':'Whole original regionprops/filter migration contract is supported; remaining12catalog-row LEFT is stale since2026-09-26. Propose source-bound dated closeout, root review required.','retired_keys':retired,'catalogs':catalogs,'named_owner_review_records':reviewfiles,'sourceproof':sourceproof,'hosted_evidence':{'run':38699600639,'source':host,'overall_run':'FAILED overall; only literal selected owning results reused','groups':[{'job':j,'file':f,'counts':dict(c)} for (j,f),c in counts.items()]},'scope':'Scalar regionprops available for actual mask/intensity context; opt-in add/remove rows, exact mask provenance, shared MakeMasks/generation filtering, old12bound load migration with explicit-list precedence and invalid-value refusal. Non-scalar properties remain intentionally excluded as documented.','missing_software_or_localization_at_original_contract':[],'not_in_scope':['Organelle min-area/intensity family explicitly retained as segmentation settings by maintainer choice; do not retire them','No native-speaker/scientific/fullCI guarantee; no new tests/collector/model/generator or catalog/review mutations'],'provenance':{'en_retirement':'f303cc63c13828f5dc26af0ebe9ddd3463647bb7 normal EN regeneration2026-09-26','nine_retirement_commits':['e00bcfc0cb','6b55dbfa10','f7464c5d15','f2795a6657','1a543ce282','b36b381799','b092fc230f','bcc7dcd4a4','cc369ff695'],'current_normal_consumer_tool':'tools/build_setting_consumer_map.py explicitly dates twelve-key catalog retirement2026-09-26 and CATALOG_ONLY_UNREAD=()'},'method':'Read immutable git objects with stdlib AST/literal metadata only; no canonical full collector, import/application/runtime/test/model call. Current catalogs+source hashmaps inspect each named key, preserve every artifact byte/history.'}
(out/'whole-item-review.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'receipt':str(out/'whole-item-review.json'),'sha256':sha((out/'whole-item-review.json').read_bytes()),'groups':receipt['hosted_evidence']['groups'],'decision':receipt['decision']},indent=2))
