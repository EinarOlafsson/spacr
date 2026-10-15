from pathlib import Path
import ast,hashlib,json,sys,tempfile,subprocess
r=Path.cwd();s=Path('/mnt/wd4tb/scratch/n702-picker-reference-ui-20261015');sys.path.insert(0,str(r/'tools'));import build_i18n_catalogs as runtime
runtime.canonical_sources=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('No fullcanonical collector'))
import build_documentation_i18n as api;import nested_helper_docs as nested
api.public_docstrings=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('No fullAPI collector'))
paths=['spacr/qt/widgets/model_zoo_picker.py','spacr/qt/screens/app_screen.py'];sha=lambda b:hashlib.sha256(b).hexdigest();texts={p:{'before':subprocess.check_output(['git','show','HEAD:'+p],text=True),'after':(r/p).read_text()} for p in paths}
def docs(p,text):
 path=r/p;module=p[:-3].replace('/','.');tree=ast.parse(text);out={module:api._clean_doc(tree)}
 for n in api._module_scope_nodes(tree.body):
  if not isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):continue
  visible=not n.name.startswith('_') if isinstance(n,ast.ClassDef) else api._is_visible_function_name(n.name,module_is_package=False)
  key=module+'.'+n.name
  if not visible or not api._is_rendered_autoapi_entry(path,module,key):continue
  d=api._autoapi_class_doc(n) if isinstance(n,ast.ClassDef) else api._clean_doc(n)
  if d:out[key]=d
  if isinstance(n,ast.ClassDef):
   out.update({k:v for k,v in api._additional_assignment_docs(n.body,key).items() if api._is_rendered_autoapi_entry(path,module,k)})
   for child in api._module_scope_nodes(n.body):
    if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)) and api._is_visible_function_name(child.name,module_is_package=False):
     ck=key+'.'+child.name;cd=api._clean_doc(child)
     if cd and api._is_rendered_autoapi_entry(path,module,ck):out[ck]=cd
 out.update({k:v for k,v in api._additional_assignment_docs(tree.body,module).items() if api._is_rendered_autoapi_entry(path,module,k)})
 return out
public={}
for p,x in texts.items():
 before,after=docs(p,x['before']),docs(p,x['after']);assert before==after,p;public[p]={'normal_rendered_public_docs_unchanged':True,'count':len(after),'docs_sha256':{k:sha(v.encode()) for k,v in after.items()}}
versions={}
for version in ['before','after']:
 with tempfile.TemporaryDirectory(dir=s,prefix='normal-nested-'+version+'-') as d:
  snap=Path(d)
  for p,x in texts.items():q=snap/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(x[version])
  inventory=nested.inventory(snap,ignore_patterns=api.AUTOAPI_IGNORE);modules={e.module for e in inventory if not e.ignored_by}&set(nested.ENABLED_MODULES);versions[version]={e.qualified_key:e.docstring for e in nested.entries(inventory,modules=modules)}
assert versions['before']==versions['after']
app='spacr/qt/screens/app_screen.py';old,new=[ast.parse(texts[app][k]) for k in ['before','after']]
for tree in [old,new]:
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AppScreen');cls.body=[n for n in cls.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name!='_choose_a_model_for']
assert ast.dump(old)==ast.dump(new)
p='spacr/qt/widgets/model_zoo_picker.py';old,new=[ast.parse(texts[p][k]) for k in ['before','after']];new.body=[n for n in new.body if not isinstance(n,ast.ClassDef) or n.name!='_ModelReferenceWorker'];allowed={'__init__','_local_path_on_disk','_selection_changed','_download_selected','_finish_download','_on_download_finished','_accept_selected','_entry_identity','_start_reference_check','_stop_reference_check','_reference_ready','_reference_failed','_reference_cancelled','closeEvent','reject','done'}
for tree in [old,new]:
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelZooPicker');cls.body=[n for n in cls.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name not in allowed]
assert ast.dump(old)==ast.dump(new)
def trset(text):
 return {n.args[0].value for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='tr' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str)}
added=set().union(*(trset(x['after']) for x in texts.values()))-set().union(*(trset(x['before']) for x in texts.values()));en=ast.parse((r/'spacr/qt/i18n_catalogs/en.py').read_text());known=set()
for n in en.body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='UI_SOURCES':known=set(ast.literal_eval(n.value.args[0]))
newkeys=added-known;assert newkeys=={'Checking the selected model file…'},newkeys
receipt={'scope':'Normal scopedpublic visibility/class-constructor/member/assignmentdoc +normalnestedtwo-fileinventory; nofullcanonical/publiccollector, model, tests orwriters.','public_docs':public,'normal_nested_entries_unchanged':len(versions['after']),'normal_nested_source_doc_pins':{k:sha(v.encode()) for k,v in versions['after'].items()},'new_API_targets':0,'source_boundary':'Outside AppScreen._choose_a_model_for exactlysameAST; outside ModelZooPicker exactallowedmethods+oneprivateverificationworker exactlysameAST, includesunchanged choose_model signature/body and model_chosen Signal(str).','allowed_picker_methods':sorted(allowed),'new_owning_tr_literals':sorted(added),'actually_new_canonical_UI_keys':sorted(newkeys),'source_sha256':{p:sha(x['after'].encode()) for p,x in texts.items()},'backend_sha256':sha((r/'spacr/model_zoo.py').read_bytes())};(s/'source-api-proof.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'newAPI':0,'nestedunchanged':len(versions['after']),'newUIkeys':sorted(newkeys),'sourceboundaryPASS':True}),flush=True)
