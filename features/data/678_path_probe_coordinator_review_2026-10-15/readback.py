from pathlib import Path
import ast,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools'))
import build_documentation_i18n as b
BASE='08e2519d5';ARTIFACT='b7077bed768e90fb225083d2df6ac936175282f1'
keys={'spacr.qt.path_probe','spacr.qt.path_probe.exists','spacr.qt.path_probe.isdir','spacr.qt.path_probe.forget','spacr.qt.path_probe.prime'}
def show(ref,p):return subprocess.check_output(['git','show',ref+':'+str(p)],cwd=ROOT)
source=ROOT/'spacr/qt/path_probe.py';assert hashlib.sha256(source.read_bytes()).hexdigest()=='7a617ca9623c977c14f00801a78290d8a9bff0667d8217c2c7772022050178e4'
tree=ast.parse(source.read_text());docs={'spacr.qt.path_probe':b._clean_doc(tree)}
for n in tree.body:
 if isinstance(n,ast.FunctionDef) and 'spacr.qt.path_probe.'+n.name in keys:docs['spacr.qt.path_probe.'+n.name]=b._clean_doc(n)
assert set(docs)==keys
proof={}
for p in (ROOT/'docs/source/_static/i18n/api').glob('*.json'):
 rel=p.relative_to(ROOT);old=json.loads(show(BASE,rel));want=json.loads(show(ARTIFACT,rel));actual=json.loads(p.read_bytes())
 assert set(old['symbols'])==set(actual['symbols'])
 assert {k:v for k,v in actual.items() if k!='symbols'}=={k:v for k,v in old.items() if k!='symbols'}
 assert {k:v for k,v in actual['symbols'].items() if k not in keys}=={k:v for k,v in old['symbols'].items() if k not in keys}
 assert {k:actual['symbols'][k] for k in keys}=={k:want['symbols'][k] for k in keys}
 if p.stem=='en':assert all(actual['symbols'][k]['text']==v for k,v in docs.items())
 proof[p.stem]={'owned5_exact_artifact':True,'unrelated_exact_public_records':len(old['symbols'])-5,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
history=Path('docs/i18n/api_source_history.json');assert (ROOT/history).read_bytes()==show(BASE,history)
result={'published_baseline':BASE,'normal_artifact':ARTIFACT,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'all10catalogs':proof,'source_history_exact':True,'English_owned_exact_current_AST':True,'limitations':'Scoped readback only; no repeated generator, runtime collection or whole-package acceptance.'}
(ROOT/'features/data/678_path_probe_coordinator_review_2026-10-15/catalog-readback.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS exact5owned entries/all10catalogs, allunrelated public records and history preserved')
