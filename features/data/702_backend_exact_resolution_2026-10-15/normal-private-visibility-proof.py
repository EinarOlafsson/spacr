from pathlib import Path
import ast,hashlib,json,sys,tempfile,subprocess
r=Path.cwd();s=Path('/mnt/wd4tb/scratch/n702-backend-resolution-context-20261015');sys.path.insert(0,str(r/'tools'));import build_i18n_catalogs as runtime
runtime.canonical_sources=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('No fullcanonical collector'))
import build_documentation_i18n as api;import nested_helper_docs as nested
api.public_docstrings=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('No fullAPI collector'))
paths=['spacr/_segmentation_backends.py','spacr/_mask_workers.py'];sha=lambda b:hashlib.sha256(b).hexdigest();texts={p:{'before':subprocess.check_output(['git','show','HEAD:'+p],text=True),'after':(r/p).read_text()} for p in paths}
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

allowed={'spacr/_segmentation_backends.py':{'_cellpose3_model','_cellpose_dino_model','_prefixed_model'},'spacr/_mask_workers.py':{'_resolved_backend_model'}}
visibility={}
for p,names in allowed.items():
 old,new=[ast.parse(texts[p][k]) for k in ['before','after']]
 module=p[:-3].replace('/','.')
 visibility[p]={name:{'normal_public_name':api._is_visible_function_name(name,module_is_package=False),'normal_rendered_entry':api._is_rendered_autoapi_entry(r/p,module,module+'.'+name)} for name in names}
 for name,status in visibility[p].items():assert not(status['normal_public_name'] and status['normal_rendered_entry']),name
 for t in (old,new):t.body=[n for n in t.body if not isinstance(n,ast.FunctionDef) or n.name not in names]
 assert ast.dump(old)==ast.dump(new),p
 def trs(t):
  return {n.args[0].value for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='tr' and n.args and isinstance(n.args[0],ast.Constant)}
 assert trs(ast.parse(texts[p]['before']))==trs(ast.parse(texts[p]['after']))
receipt={'scope':'Actual ordinary scoped public/nested metadata and visibility predicates for two private modules; no fullcanonical/fullAPI collector/model/test/writer.','public_docs':public,'normal_nested_entries_unchanged':len(versions['after']),'actual_changed_owner_visibility':visibility,'new_API_targets':0,'new_runtime_targets':0,'source_boundary':'Only exact three backend helper functions and _mask_workers._resolved_backend_model changed; _prepare_mask_model/remainingAST unchanged from actual acceptedHard prerequisite.','source_sha256':{p:sha(v['after'].encode()) for p,v in texts.items()}}
(s/'normal-private-visibility-proof.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'API':0,'runtime':0,'nested':len(versions['after']),'visibility':visibility}))
