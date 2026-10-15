from pathlib import Path
import ast,hashlib,json,sys,tempfile
root=Path.cwd();scratch=Path('/mnt/wd4tb/scratch/n468-current-contract-audit-20261015');sys.path.insert(0,str(root/'tools'))
import build_i18n_catalogs as normal
from spacr.object_roles import setting_label
sha=lambda b:hashlib.sha256(b).hexdigest();history=json.loads((scratch/'historical59.json').read_text());old=history['delta'];langs=['sv','de','es','zh_CN','fr','pt','hi','ko','is']
def tables(path):
 out={}
 for n in ast.parse(path.read_text()).body:
  if not isinstance(n,ast.Assign) or not isinstance(n.targets[0],ast.Name):continue
  k=n.targets[0].id
  if k in ['SETTING_LABELS','SETTING_TOOLTIPS','UI','SOURCE_HASHES']:out[k]=ast.literal_eval(n.value)
  if k=='UI_SOURCES':out[k]=set(ast.literal_eval(n.value.args[0]))
 return out
en=tables(root/'spacr/qt/i18n_catalogs/en.py');sourcepath=root/'spacr/qt/widgets/plaque_preview.py';tree=ast.parse(sourcepath.read_text());constants={}
for n in tree.body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):
  try:constants[n.targets[0].id]=ast.literal_eval(n.value)
  except (ValueError,TypeError):pass
trsources={}
for n in ast.walk(tree):
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='tr' and n.args:
  value=normal._literal(n.args[0])
  if isinstance(n.args[0],ast.Name) and n.args[0].id in constants and isinstance(constants[n.args[0].id],str):value=constants[n.args[0].id]
  if value:trsources.setdefault(value,[]).append(n.lineno)
separate=next(x for x in old['UI_SOURCES'] if x.startswith('Show or hide the plaque preview above the console.'))
ui=set(old['UI_SOURCES'])-{separate};assert len(ui)==46
live=ui&en['UI_SOURCES'];retired=ui-en['UI_SOURCES'];assert len(live)==37 and len(retired)==9
replacement={}
for source in retired:
 if source.startswith("'bundled'"):target=constants['BUNDLED_NOTE']
 elif source=='Sizes':target='Inference sizes'
 elif source=='Read text':target='Read the text around each well'
 elif source.startswith('The install failed:'):target='Installing {name} failed. Nothing was left half-built.'
 elif source.startswith('Legend saved to'):target='Conditions proposed again with the new text settings.'
 elif source.startswith('Figure mode needs'):target='Figure mode needs {names}. Press Install to put them in an environment of their own, so spaCR\'s own packages are not changed. Plaque mode works without them.'
 elif source=='the papers extra':target='Plaque figure reader'
 elif source.startswith('Run pip install'):target="Install the plaque figure reader (YOLO and RapidOCR) into an environment of its own under ~/.spacr/backends; spaCR's own packages are not changed."
 elif source.startswith('{name}: {regions}'):target='{name}: {regions} plaque wells found, {words} words read. Click a well, then Plaque preview, or Find plaques in all wells.'
 else:raise AssertionError(source)
 assert target in trsources and target in en['UI_SOURCES'],target
 replacement[source]={'current':target,'current_tr_source_lines':trsources[target],'reason':'Removed old source; actual current live mode/settings/isolated-install/legend/selected-well path uses this source. No old caption restored.'}
labels={k:en['SETTING_LABELS'][k] for k in old['SETTING_LABELS']};assert all(setting_label(k)==v for k,v in labels.items())
st=ast.parse((root/'spacr/qt/screens/settings_model.py').read_text());stripnode=next(n for n in st.body if isinstance(n,ast.FunctionDef) and n.name=='_strip_type_prefix');namespace={};exec(compile(ast.Module(body=[stripnode],type_ignores=[]),'<exact-current-normal-strip-helper>','exec'),namespace)
settings=ast.parse((root/'spacr/settings.py').read_text());tooltips={k:en['SETTING_TOOLTIPS'][k] for k in old['SETTING_TOOLTIPS']}
for key,value in tooltips.items():
 values=[ast.literal_eval(v) for n in ast.walk(settings) if isinstance(n,ast.Dict) for k,v in zip(n.keys,n.values) if isinstance(k,ast.Constant) and k.value==key and isinstance(v,ast.Constant) and isinstance(v.value,str)]
 assert any(' '.join(namespace['_strip_type_prefix'](v).split())==value for v in values),key
ui_current=live|{v['current'] for v in replacement.values()}|{separate};canonical={'setting_labels':labels,'setting_tooltips':tooltips,'ui':ui_current}
sourceids=[('setting_labels',k,v) for k,v in labels.items()]+[('setting_tooltips',k,v) for k,v in tooltips.items()]+[('ui',v,v) for v in sorted(ui_current)]
assert len(sourceids)==60
results={};issues=[]
with tempfile.TemporaryDirectory(dir=scratch,prefix='normal-runtime-owned-inputs-') as directory:
 normal.REVIEWED_RUNTIME_DIR=Path(directory);normal.canonical_sources=lambda:canonical
 for lang in langs:
  rows=[];inputpins=[]
  for path in sorted((root/'docs/i18n/reviewed/runtime'/lang).glob('*.json')):
   payload=json.loads(path.read_text());selected=[row for row in payload['records'] if row['table'] in canonical and ((isinstance(canonical[row['table']],dict) and canonical[row['table']].get(row['key'])==row['source']) or (row['table']=='ui' and row['source'] in ui_current))]
   if selected:rows.extend(selected);inputpins.append({'path':str(path.relative_to(root)),'sha256':sha(path.read_bytes()),'selected':len(selected)})
  q=Path(directory)/lang;q.mkdir();(q/'owned.json').write_text(json.dumps({'schema':1,'language':lang,'records':rows},ensure_ascii=False));normal.reviewed_runtime_translations.cache_clear();reviewed=normal.reviewed_runtime_translations(lang)
  compiled=tables(root/f'spacr/qt/i18n_catalogs/{lang}.py');checks=[]
  for table,key,source in sourceids:
   column={'ui':'UI','setting_labels':'SETTING_LABELS','setting_tooltips':'SETTING_TOOLTIPS'}[table];target=compiled[column].get(key);assert target is not None,(lang,table,key)
   assert compiled['SOURCE_HASHES'][(column,key)]==sha(source.encode()),(lang,table,key)
   record={'table':table,'key':key,'source_sha256':sha(source.encode()),'target_sha256':sha(target.encode()),'exact_review':reviewed.get(source)==target,'normal_candidate_valid':normal._translation_candidate_valid(source,target,lang,force=normal._looks_translatable(source)),'target':target};checks.append(record)
   if not record['normal_candidate_valid']:issues.append({'language':lang,**record})
  results[lang]={'normal_scoped_review_input_pins':inputpins,'selected_review_records':len(rows),'current_compiled_checks':checks}
receipt={'original_debt_reconstruction':{'English_delta_commit':history['commit'],'actual_raw_delta':{'setting_labels':6,'setting_tooltips':7,'UI':47},'already_reviewed_separate_preview_description':separate,'owed59':'6labels+7tooltip(new6+changedplaque_model)+46UI; previewdescription separately already reviewed per datednote, explains raw60delta.','still_current_UI37':sorted(live),'old_UIretired9_with_actual_current_tr_replacements':replacement,'current_plaque_model_tooltip':tooltips['plaque_model'],'setting_current_source_binding':'Actual current setting_label() for6keys and exact current _strip_type_prefix AST execution +whitespace normalization for7tooltips match compiledEnglish sources; no generic source admission.'},'sourcepins':{str(p.relative_to(root)):sha(p.read_bytes()) for p in [sourcepath,root/'spacr/settings.py',root/'spacr/object_roles.py',root/'spacr/qt/screens/settings_model.py',root/'tools/build_i18n_catalogs.py']},'selected_current_checks_perlocale':len(sourceids),'scoped_normal_sources':'Only60sourceidentities: historical59debt current13settings+37liveUI+9replacementUI plus separately reviewedpreviewdescription. Ordinary reviewloader gates on exact scoped current sourcebindings; NOT globalcanonical/audit.','languages':results,'issues':issues,'live_legacy_UI_absent_owning_literal_tr':sorted(live-set(trsources)),'all_current_literal_tr_sources_count':len(trsources),'scope':'Read-only source/compiled/normalreview/candidate gates; no collectors,generators,tests,models,data/network/GPU or source/item/status edits.'}
(scratch/'runtime59-proof.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'current_checks_perlocale':60,'historical_debt':59,'candidate_issues':len(issues),'legacy_live_but_not_local_tr':receipt['live_legacy_UI_absent_owning_literal_tr']}),flush=True)
