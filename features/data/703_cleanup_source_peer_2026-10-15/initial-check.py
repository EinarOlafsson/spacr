import ast,hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path('tools').resolve()))
import build_documentation_i18n as b
from api_visibility import public_symbol
import nested_helper_docs as nested
root=Path('/mnt/wd4tb/scratch/home-b-n703-cleanup-source-peer-20261015')
c=subprocess.check_output(['git','rev-parse','b4e09b640'],text=True).strip()
paths={'spacr/install_cleanup.py':'06b4fd48abb45b9691a0721d4c08b3096d2fd89032ed6913ab66873a5022b32d','spacr/qt/install_consent.py':'0885af8ba73fbd3b2376b25f78077a01a708bebc587be86ac6c7a65ad822c77a'}
owner=ast.parse(Path('/mnt/wd4tb/scratch/home-b-n703-cleanup-api-20261015/visibility.py').read_text())
fn=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='render')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<normal scoped renderer>','exec'))
before={};after={}
for path,pin in paths.items():
 raw=subprocess.check_output(['git','show',f'{c}:{path}'])
 assert hashlib.sha256(raw).hexdigest()==pin
 old=subprocess.check_output(['git','show',f'{c}^:{path}'])
 before.update(render(Path(path),old));after.update(render(Path(path),raw))
 p=root/'source'/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
assert len(before)==16
assert all(after.get(k)==v for k,v in before.items())
new={k:v for k,v in after.items() if k not in before}
assert set(new)=={'spacr.install_cleanup.preview_optional_cleanup','spacr.install_cleanup.apply_optional_cleanup'}
private={k:public_symbol(k) for k in ['spacr.qt.install_consent._InstallerCleanupDialog','spacr.qt.install_consent._InstallerCleanupDialog.choices','spacr.qt.install_consent._choose_optional_cleanup','spacr.qt.install_consent._cleanup_dialog_rows']}
assert not any(private.values())
definitions=nested.inventory(root/'source',ignore_patterns=b.AUTOAPI_IGNORE)
undocumented=[d.qualified_key for d in definitions if not d.docstring]
assert set(undocumented)=={'spacr.install_cleanup.preview_optional_cleanup.'+n for n in ['identity','open_directory','normalized','retained_path']},undocumented
assert 'spacr.install_cleanup' in nested.ENABLED_MODULES
try:
 nested.entries(definitions,modules={'spacr.install_cleanup'})
except ValueError as e:
 failure=str(e)
else:raise AssertionError('Expected normal enabled-helper gate to reject undocumented helpers')
r={'source_commit':c,'source_files':paths,'old_rendered_bodies_unchanged':len(before),'new_public_symbols':new,'private_visibility':private,'scoped_normal_nested_inventory_count':len(definitions),'undocumented_enabled_helpers':undocumented,'normal_entries_result':'FAIL as expected','normal_entries_error':failure,'boundary':'Two exact source-file snapshot only, using actual normal direct-render/helper functions. No full inventory, app imports, tests, generation, mutation of source-owner tree or validation bypass.'}
(root/'visibility-proof.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
