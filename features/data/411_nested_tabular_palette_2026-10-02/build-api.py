"""Use normal reviewed API repair/writer and strict full audit for two new nested helpers."""
import argparse,functools,hashlib,json,sys,time,threading
from pathlib import Path
root=Path.cwd();p=Path(__file__).parent
sys.path.insert(0,str(root/'tools'))
import build_i18n_catalogs as runtime
import build_documentation_i18n as builder
runtime.SECONDARY_MODEL_FOLDER='/tmp/spacr-translation-no-7b-fallback'
normalize=functools.lru_cache(maxsize=65536)(runtime._simplify_chinese_prose)
runtime._simplify_chinese_prose=normalize
if hasattr(builder,'_simplify_chinese_prose'):builder._simplify_chinese_prose=normalize
builder._api_translation_source=functools.lru_cache(maxsize=65536)(builder._api_translation_source)
state={'stage':'extract'};finished=threading.Event();started=time.monotonic()
def heartbeat():
 while not finished.wait(30):print('HEARTBEAT',round(time.monotonic()-started,1),state['stage'],flush=True)
threading.Thread(target=heartbeat,daemon=True).start()
docs=builder.public_docstrings();keys=set(json.loads((p/'source-delta.json').read_text())['added']);subset={key:docs[key] for key in keys}
old_en=json.loads((builder.API_DIR/'en.json').read_text())['symbols']
assert set(docs)-set(old_en)==keys and set(old_en)<=set(docs)
assert all(docs[k]==v['text'] for k,v in old_en.items())
args=argparse.Namespace(device='cpu',threads=1,batch_size=8,beams=4,force=False)
proof={}
try:
 for language in builder.MODEL_SPECS:
  path=builder.API_DIR/f'{language}.json';before=json.loads(path.read_text());state['stage']=language+' reviewed repair'
  repaired=builder.repair_api_translations(subset,language,builder.default_model_root(),args)
  assert set(repaired)==keys
  translations={k:v['text'] for k,v in before['symbols'].items()};translations.update(repaired)
  state['stage']=language+' normal write';builder.write_language(docs,language,translations)
  after=json.loads(path.read_text());assert set(after['symbols'])-set(before['symbols'])==keys
  assert all(after['symbols'][k]==v for k,v in before['symbols'].items())
  assert {k:v for k,v in before.items() if k!='symbols'}=={k:v for k,v in after.items() if k!='symbols'}
  proof[language]={'added':sorted(keys),'removed':[],'changed_existing':[],'unchanged_symbols':len(before['symbols']),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
  print(language,'PASS +2; all12036 old records unchanged',flush=True)
 builder._write_english_api_manifest(docs)
 after_en=json.loads((builder.API_DIR/'en.json').read_text())['symbols'];assert all(after_en[k]==v for k,v in old_en.items())
 (p/'api-catalog-diff.json').write_text(json.dumps(proof,indent=2)+'\n')
 state['stage']='full nine-locale strict API/readme audit'
 status=builder.audit(docs,tuple(builder.MODEL_SPECS))
 (p/'api-audit-result.json').write_text(json.dumps({'status':status,'symbols':len(docs),'languages':list(builder.MODEL_SPECS),'elapsed_seconds':round(time.monotonic()-started,2)},indent=2)+'\n')
finally:finished.set()
sys.exit(status)
