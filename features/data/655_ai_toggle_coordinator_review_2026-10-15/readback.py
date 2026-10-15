from pathlib import Path
import ast,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools'))
import build_documentation_i18n as b
BASE='23f694a67';ARTIFACT='c1f87e27c'
import gzip
ledger=json.loads(gzip.decompress((ROOT/'features/data/655_ai_toggle_api_reviews_2026-10-15/sources.json.gz').read_bytes()))
docs=ledger['source_docs'];keys=set(docs);assert len(keys)==4
for name,h in ledger['source_files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
def show(ref,p):return subprocess.check_output(['git','show',ref+':'+str(p)],cwd=ROOT)
proof={}
for p in (ROOT/'docs/source/_static/i18n/api').glob('*.json'):
 rel=p.relative_to(ROOT);old=json.loads(show(BASE,rel));want=json.loads(show(ARTIFACT,rel));actual=json.loads(p.read_bytes())
 assert not keys & set(old['symbols']);assert set(actual['symbols'])==set(old['symbols'])|keys
 assert {k:v for k,v in actual.items() if k!='symbols'}=={k:v for k,v in old.items() if k!='symbols'}
 assert {k:v for k,v in actual['symbols'].items() if k not in keys}=={k:v for k,v in old['symbols'].items() if k not in keys}
 assert {k:actual['symbols'][k] for k in keys}=={k:want['symbols'][k] for k in keys}
 if p.stem=='en':assert all(actual['symbols'][k]['text']==v for k,v in docs.items())
 proof[p.stem]={'owned4_exact_artifact':True,'unrelated_exact_public_records':len(old['symbols']),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
history=Path('docs/i18n/api_source_history.json');assert (ROOT/history).read_bytes()==show(BASE,history)
result={'published_baseline':BASE,'normal_artifact':ARTIFACT,'source_sha256':ledger['source_files'],'all10catalogs':proof,'source_history_exact':True,'English_owned_exact_current_AST':True,'limitations':'Scoped readback only; no repeated generator, runtime collection or whole-package acceptance.'}
(ROOT/'features/data/655_ai_toggle_coordinator_review_2026-10-15/catalog-readback.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS exact4owned entries/all10catalogs, allunrelated public records and history preserved')
