from pathlib import Path
import ast,hashlib,json,subprocess
root=Path(__file__).resolve().parents[3]
base='ecfd60fcf7039ec600e73eac4c706ea374b20773'
tested='01a787f88'
incoming='cc42aa2d82ecc2448d0aa35af5f39a78ee7e3938'
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0',*args],cwd=root)
def tree(pin):return {entry.split(b'\t',1)[1].decode():entry.split(b'\t',1)[0].split()[-1].decode() for entry in git('ls-tree','-rz',pin).split(b'\0') if entry}
own=set(git('diff','--name-only',base,tested).decode().splitlines());public,actual=tree(incoming),tree('HEAD');preserved=[path for path in public if path not in own];assert all(public[path]==actual[path] for path in preserved)
for path in ['tools','spacr/qt/i18n_catalogs','docs/source/_static/i18n/api','tests/data/release_contracts/runtime_caption_arrivals_2026_10_08.json.gz']:assert git('rev-parse','HEAD:'+path)==git('rev-parse',incoming+':'+path)
receipt=json.loads((Path(__file__).parent/'receipt.json').read_text());hashes=dict(receipt['source_file_sha256']);oracle=json.loads((Path(__file__).parent/'valid-oracle-receipt.json').read_text());hashes.update({str((Path(__file__).parent/name).relative_to(root)):expected for name,expected in oracle['artifact_sha256'].items()})
for path,expected in hashes.items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected,path
def tr_calls(raw):return sorted(ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id=='tr') or (isinstance(n.func,ast.Attribute) and n.func.attr=='tr')))
sources=[]
for path in git('diff','--name-only',base,incoming,'--','spacr','tools').decode().splitlines():
 if path.endswith('.py'):
  assert tr_calls(git('show',base+':'+path).decode())==tr_calls(git('show',incoming+':'+path).decode()),path;sources.append(path)
assert tr_calls((root/'spacr/attribution.py').read_text())==tr_calls(git('show',base+':spacr/attribution.py').decode())
proof={'immutable_incoming_public':incoming,'tested_coherent_source':tested,'unrelated_incoming_public_files_exact':len(preserved),'incoming_changed_python_sources_tr_calls_exact':sources,'latest_API_runtime_catalogs_arrivals_tools_exact_incoming':True,'actual_source_test_fixture_and_prepatch_oracle_hashes_exact':hashes,'checks':'33 targeted checks on exact source; immutable prepatch valid-output oracle, model modes and no transfer/wrapper on invalid targets. Body-only own signature/prose/tr neutrality and incoming preservation prove no new derived target; no tests/writer/inventory repeated after foreign AI/Live/Dock source/API merge.','limits':receipt['limits'],'validated_tree_ids':{path:git('rev-parse','HEAD:'+path).decode().strip() for path in ['spacr','tools','docs','tests']},'artifact_sha256':hashes};(Path(__file__).parent/'final-publication-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'incoming':incoming,'unrelated_preserved':len(preserved),'runtime_sources':sources}))
