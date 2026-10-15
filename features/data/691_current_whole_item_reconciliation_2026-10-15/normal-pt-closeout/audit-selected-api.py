from pathlib import Path
import ast,hashlib,json,sys,tempfile
root=Path.cwd();scratch=Path('/mnt/wd4tb/scratch/n691-current-whole-item-audit-20261015');sys.path.insert(0,str(root/'tools'))
import build_i18n_catalogs as runtime
original=runtime.canonical_sources;actual_snapshot=None;collection_calls=0
mode=sys.argv[1];sha=lambda b:hashlib.sha256(b).hexdigest()
sourcefiles=sorted((root/'spacr').rglob('*.py'))+sorted((root/'tools').rglob('*.py'))
pins={str(p.relative_to(root)):sha(p.read_bytes()) for p in sourcefiles}
def canonical():
 global actual_snapshot,collection_calls
 if mode=='guarded':raise RuntimeError('N691_GUARDED_API_AUDIT_REQUIRES_REAL_RUNTIME_CONTEXT')
 if actual_snapshot is None:
  collection_calls+=1;actual_snapshot=original();print('ACTUAL normal runtime context collected once',flush=True)
 return actual_snapshot
runtime.canonical_sources=canonical
import build_documentation_i18n as api
api.public_docstrings=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('No full API collector'))
key='spacr.model_zoo.mitonet_weights';tree=ast.parse((root/'spacr/model_zoo.py').read_text());node=next(n for n in api._module_scope_nodes(tree.body) if isinstance(n,ast.FunctionDef) and n.name=='mitonet_weights');docs={key:api._clean_doc(node)}
api_before={str(p.relative_to(root)):sha(p.read_bytes()) for p in api.API_DIR.glob('*.json')};runtime_before={str(p.relative_to(root)):sha(p.read_bytes()) for p in (root/'spacr/qt/i18n_catalogs').glob('*.py')}
with tempfile.TemporaryDirectory(dir=scratch,prefix='exact-one-symbol-api-audit-') as d:
 for lang in ['en','pt']:
  src=api.API_DIR/f'{lang}.json';payload=json.loads(src.read_text());assert key in payload['symbols'];payload['symbols']={key:payload['symbols'][key]};(Path(d)/src.name).write_text(json.dumps(payload,ensure_ascii=False))
 api.API_DIR=Path(d)
 result=api.audit(docs,['pt'])
 assert result==0,result
if actual_snapshot is not None:
 fresh=original();assert fresh==actual_snapshot;collection_calls+=1
for path,h in pins.items():assert sha((root/path).read_bytes())==h,path
for path,h in api_before.items():assert sha((root/path).read_bytes())==h,path
for path,h in runtime_before.items():assert sha((root/path).read_bytes())==h,path
receipt={'scope':'Ordinary api.audit on actual one-symbol EN+PT snapshot with original record/header values; SUBSET only, no full API audit. No installed catalogs mutated by audit.','mode':mode,'result':result,'actual_runtime_canonical_calls':collection_calls,'real_context_recollected_equal':actual_snapshot is not None,'sourcefiles_current_pins':pins,'all_API_artifacts_unchanged':api_before,'all_runtime_artifacts_unchanged':runtime_before}
(scratch/f'normal-selected-audit-{mode}.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print('N691 actual selected one-symbol ordinary API audit PASS',flush=True)
