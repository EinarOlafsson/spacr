from pathlib import Path
import subprocess,json,hashlib,ast
root=Path('/mnt/wd4tb/spacr-worktrees/codex-f537-parent-identity-20261015')
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0',*args],cwd=root)
def tree(pin):return {entry.split(b'\t',1)[1].decode():entry.split(b'\t',1)[0].split()[-1].decode() for entry in git('ls-tree','-rz',pin).split(b'\0') if entry}
public=git('rev-parse','origin/nightly').decode().strip();tested='d3cb20691';baseline='2252f29a2';own=set(git('diff','--name-only',baseline,tested).decode().splitlines());incoming=tree(public);actual=tree('HEAD');preserved=[p for p in incoming if p not in own];assert all(incoming[p]==actual[p] for p in preserved),'Unrelated public file differs'
for path in ['spacr/qt/i18n_catalogs','tools','tests/data/release_contracts/runtime_caption_arrivals_2026_10_08.json.gz']:assert git('rev-parse','HEAD:'+path)==git('rev-parse',tested+':'+path),path
normal=json.loads((root/'features/data/537_540_runtime_closure_2026-10-15/runtime-receipt.json').read_text())
for path,expected in normal['source_file_sha256_unchanged'].items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected,path
def tr_calls(source):
 return sorted(ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id=='tr') or (isinstance(n.func,ast.Attribute) and n.func.attr=='tr')))
sources=[]
for path in git('diff','--name-only',baseline,public,'--','spacr','tools').decode().splitlines():
 if path.endswith('.py'):
  assert tr_calls(git('show',baseline+':'+path).decode())==tr_calls(git('show',public+':'+path).decode()),path;sources.append(path)
assert git('rev-parse',baseline+':tools')==git('rev-parse',public+':tools')
assert git('rev-parse',baseline+':spacr/qt/i18n_catalogs')==git('rev-parse',public+':spacr/qt/i18n_catalogs')
receipt={'tested_coherent_closure':tested,'incoming_public':public,'unrelated_public_files_exact':len(preserved),'incoming_changed_runtime_source_tr_calls_unchanged':sources,'incoming_tools_runtime_catalogs_unchanged':True,'actual_generated_catalogs_arrivals_exact_tested_closure':True,'actual_source_hashes_exact':normal['source_file_sha256_unchanged'],'checks':'F53736 owning; F54022 unchanged parser plus final9 affected classifier checks; actual18 localized callbacks zero skips; normal full-nine audit and arrival closure. None repeated after foreign body/API merge.','limits':'No final global CI/release, biological accuracy/native inference or whole F537/F540 acceptance claim. Actual full-nine audit belongs to tested closure; no new collector after caption-neutral foreign merge.','validated_tree_ids':{p:git('rev-parse','HEAD:'+p).decode().strip() for p in ['spacr','tools','docs','tests']},'artifact_sha256':normal['source_file_sha256_unchanged']}
(root/'features/data/537_540_runtime_closure_2026-10-15/final-publication-proof.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'public':public,'preserved':len(preserved),'runtime_sources':sources}))
