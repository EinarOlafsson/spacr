import ast,gzip,hashlib,json,subprocess,sys
from pathlib import Path
root=Path('/mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015');scratch=Path('/mnt/wd4tb/scratch/n703-cache-cleanup-gui-20261015')
sys.path[:0]=[str(root/'tools'),str(root)]
import build_documentation_i18n as b
from api_visibility import public_symbol
import nested_helper_docs as nested

def no_full_collection(*_args, **_kwargs):
 raise AssertionError('This bounded visibility proof must not run a full inventory')
b.public_docstrings=no_full_collection
owner=ast.parse(Path('/mnt/wd4tb/scratch/home-b-n703-cleanup-api-20261015/visibility.py').read_text())
fn=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='render')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<unchanged owner normal scoped renderer>','exec'))
files=['spacr/install_cleanup.py','spacr/qt/install_consent.py']
before={};after={};pins={}
for name in files:
 original=subprocess.check_output(['git','show','b4e09b640^:'+name],cwd=root)
 raw=(root/name).read_bytes();pins[name]=hashlib.sha256(raw).hexdigest()
 before.update(render(Path(name),original));after.update(render(Path(name),raw))
 for folder,payload in [('visibility-before',original),('visibility-final',raw)]:
  target=scratch/folder/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
assert all(after.get(key)==value for key,value in before.items())
new={key:value for key,value in after.items() if key not in before}
assert set(new)=={'spacr.install_cleanup.preview_optional_cleanup','spacr.install_cleanup.apply_optional_cleanup'}
old_defs=nested.inventory(scratch/'visibility-before',ignore_patterns=b.AUTOAPI_IGNORE)
new_defs=nested.inventory(scratch/'visibility-final',ignore_patterns=b.AUTOAPI_IGNORE)
old_entries={e.qualified_key:e.docstring for e in nested.entries(old_defs,modules={'spacr.install_cleanup'})}
new_entries={e.qualified_key:e.docstring for e in nested.entries(new_defs,modules={'spacr.install_cleanup'})}
assert all(new_entries.get(key)==value for key,value in old_entries.items())
arrivals={key:value for key,value in new_entries.items() if key not in old_entries}
assert set(arrivals)=={'spacr.install_cleanup.preview_optional_cleanup.normalized','spacr.install_cleanup.preview_optional_cleanup.retained_path'}
private={key:public_symbol(key) for key in ['spacr.qt.install_consent._InstallerCleanupDialog','spacr.qt.install_consent._InstallerCleanupDialog.choices','spacr.qt.install_consent._choose_optional_cleanup','spacr.qt.install_consent._cleanup_dialog_rows']}
assert not any(private.values())
prior=ast.parse(subprocess.check_output(['git','show','bf3a0e7ff:spacr/install_cleanup.py'],cwd=root));current=ast.parse((root/'spacr/install_cleanup.py').read_bytes())
def erase_docs(tree):
 for node in ast.walk(tree):
  if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
 return ast.dump(tree,include_attributes=False)
assert erase_docs(prior)==erase_docs(current)
record={'source_sha256':pins,'direct_public_arrivals':new,'new_enabled_nested_arrivals':arrivals,'unchanged_direct_bodies':len(before),'unchanged_nested_bodies':len(old_entries),'new_nested_definitions':len(new_defs)-len(old_defs),'private_visibility':private,'normal_enabled_entries':'PASS on exact two-module snapshot','only_docstrings_changed_since_bf3a':True,'boundary':'Known-small normal owner renderer/public_symbol and actual nested.inventory/entries on two-source snapshots only. Full collector actively refused; no source execution/model/network/catalog generation/native/human claim.'}
(scratch/'final-visibility-proof.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print('PASS2public +2enabled nested arrivals;',len(before),'old direct and',len(old_entries),'old nested bodies unchanged;4private GUI candidates excluded; doc-only executable AST identity PASS')
