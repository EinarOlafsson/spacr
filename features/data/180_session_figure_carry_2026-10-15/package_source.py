import ast, copy, gzip, hashlib, json, shutil, subprocess, sys, tempfile
from pathlib import Path
root=Path.cwd(); scratch=Path('/mnt/wd4tb/scratch/home-b-n180-figure-carry-20261015')
sys.path.insert(0,str(root/'tools'))
import build_documentation_i18n as b
import nested_helper_docs as nested
from api_visibility import public_symbol
out=root/'features/data/180_session_figure_carry_2026-10-15';out.mkdir(exist_ok=True)
files=['spacr/qt/widgets/figure_queue.py','spacr/qt/screens/app_screen.py']
base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
sha=lambda x:hashlib.sha256(x).hexdigest()
def render(path,raw):
 tree=ast.parse(raw);module=path.as_posix()[:-3].replace('/','.');docs={}
 doc=b._clean_doc(tree)
 if doc:docs[module]=doc
 for node in b._module_scope_nodes(tree.body):
  if not isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):continue
  visible=not node.name.startswith('_') if isinstance(node,ast.ClassDef) else b._is_visible_function_name(node.name,module_is_package=False)
  key=module+'.'+node.name
  if not visible or not b._is_rendered_autoapi_entry(root/path,module,key):continue
  b._merge_source_doc(docs,key,b._autoapi_class_doc(node) if isinstance(node,ast.ClassDef) else b._clean_doc(node))
  if isinstance(node,ast.ClassDef):
   for child in b._module_scope_nodes(node.body):
    if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)) and b._is_visible_function_name(child.name,module_is_package=False):
     key2=key+'.'+child.name
     if b._is_rendered_autoapi_entry(root/path,module,key2):b._merge_source_doc(docs,key2,b._clean_doc(child))
 return docs
before={};after={};pins={};nested_sets=[]
for version in ('before','after'):
 snapshot=scratch/('source-'+version)
 for name in files:
  raw=subprocess.check_output(['git','show',f'{base}:{name}']) if version=='before' else (root/name).read_bytes()
  p=snapshot/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
  (before if version=='before' else after).update(render(Path(name),raw))
  pins.setdefault(name,{})[version]=sha(raw)
 definitions=nested.inventory(snapshot,ignore_patterns=b.AUTOAPI_IGNORE)
 modules={d.module for d in definitions}&set(nested.ENABLED_MODULES)
 entries=nested.entries(definitions,modules=modules)
 nested_sets.append({x.qualified_key:x.docstring for x in entries})
assert nested_sets[0]==nested_sets[1]
new={k:v for k,v in after.items() if k not in before}
assert set(new)=={'spacr.qt.widgets.figure_queue.FigureQueue.workspace_state','spacr.qt.widgets.figure_queue.FigureQueue.apply_workspace_state'}
assert all(after[k]==v for k,v in before.items())
blocks=[]
for key,body in new.items():
 for i,source in enumerate(b.translatable_blocks(body)[0]):
  context=b._api_translation_source(source)
  blocks.append({'label':f'{key}#{i}','source':source,'source_sha256':b._source_hash(source),'context':context,'context_sha256':b._source_hash(context)})
private=['spacr.qt.widgets.figure_queue.FigureQueue._workspace_restored','spacr.qt.screens.app_screen.AppScreen._on_workspace_figures_restored']
assert not any(public_symbol(x) for x in private)
caption='Saved figure files exceed the workspace copy limit.'
source_node=ast.parse((root/files[0]).read_text())
assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='tr' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value==caption for n in ast.walk(source_node))
manifest={'schema':1,'source_base':base,'source_files':{n:p['after'] for n,p in pins.items()},'source_docs':new,'changed_blocks':blocks,'new_symbols':list(new),'prior_direct_bodies_unchanged':len(before),'nested_entries_unchanged':len(nested_sets[0]),'normal_helpers_used':['_is_visible_function_name','_is_rendered_autoapi_entry','_clean_doc','_autoapi_class_doc','translatable_blocks','_api_translation_source','nested_helper_docs.inventory/entries on exact two-file snapshots'],'private_declarations_excluded':private,'runtime_arrivals':[{'table':'ui','key':caption,'source':caption,'source_sha256':sha(caption.encode())}],'limits':'Exact two-module source visibility only; no global collector, generator, complete source-currentness or publication claim.'}
(out/'api-runtime-obligations.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');(scratch/'sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
for name in ['before.log','after.log','final.log']:(out/(name+'.gz')).write_bytes(gzip.compress((scratch/name).read_bytes(),mtime=0))
(out/'package_source.py').write_bytes(Path(__file__).read_bytes())
receipt={'schema':1,'feature':'N180','scope':'Real session-only FigureQueue raster/PDF persistence and AppScreen contributor/visible already-built-grid restoration. Not whole-item closure.','base':base,'source_hashes':pins,'baseline':'2 FAIL in 5.72s: real AppScreen lacks queue provider in both reference/copy.','intermediate':'14 PASS/1 FAIL in 13.50s: missing-file fixture collected a path before creation; production inventory intentionally lists existing files. Fixture corrected to create/collect/delete.','final':'18 PASS in 15.82s; wall18.38s;1054912KiB peak; cap2G/thread1/offscreen/CUDAhidden/private XDG/private lane lock; host available99576MiB before run.','new_cases':15,'existing_cases':3,'tests_sha256':sha((root/'tests/qt/test_figure_queue_workspace_180.py').read_bytes()),'semantics':['Ordered previews, titles, selected index and run groups; PNG/PDF only, no live objects/pickle/recompute.','Mandatory carry only for real files directly inside queue-owned temporary directory, no symlinks/external originals, at most current workspace.default_copy_limit_mb per file. Larger assets stay inventoried references.','Restore stages available valid PNG/PDF into new private queue storage before replacing current state; source and bundle never moved or changed; malformed/invalid format/over-budget source refuses without replacing current queue.','Missing previews retain slot/title and standard unavailable display; normal core report gives missing-file status.','Actual built grid refreshed, card revealed; deferred regression results/grid remain unbuilt and reported as skipped.','No workspace core/grid/alpha/runtime catalog edits. One new translated budget diagnostic and two ordinary public API methods need normal closure.'],'failures_preserved':True,'no_claims':['No new whole-suite/native/scientific acceptance','No API/runtime catalog generation','No main/release/publication change'],'artifact_sha256':{p.name:sha(p.read_bytes()) for p in out.iterdir() if p.is_file()}}
(out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'new_symbols':len(new),'api_blocks':len(blocks),'nested_unchanged':len(nested_sets[0]),'prior_direct_unchanged':len(before),'runtime':1}))
