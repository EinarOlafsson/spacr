import json,sys,subprocess,hashlib
from pathlib import Path
root=Path('/mnt/wd4tb/spacr-worktrees/codex-n703-debian-adapter-20261015')
scratch=Path('/mnt/wd4tb/scratch/n703-coalesced-runtime-20261015')
sys.path[:0]=[str(root/'tools'),str(root)]
import build_i18n_catalogs as b
pins={}
for name in ['spacr/install_cleanup.py','spacr/qt/app.py','spacr/qt/install_consent.py','tools/build_i18n_catalogs.py','tools/i18n_caption_arrivals.py']:
 original=subprocess.check_output(['git','show','1eabf4b49:'+name],cwd=root)
 pins[name]=hashlib.sha256(original).hexdigest()
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==pins[name],name
sources=json.loads((scratch/'canonical-snapshot.json').read_text())
b.canonical_sources=lambda:sources
def refuse(*args,**kwargs):raise AssertionError('No decoding is authorized in review-only preflight')
b._translate_batches=refuse
result={'source_pins':pins,'locales':{}}
for lang in b.MODEL_SPECS:
 try:
  b.reviewed_runtime_translations(lang)
  result['locales'][lang]={'ok':True}
 except ValueError as error:
  result['locales'][lang]={'ok':False,'errors':str(error)}
 print(lang+':'+json.dumps(result['locales'][lang],ensure_ascii=False),flush=True)
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==value for name,value in pins.items())
(scratch/'all9-preflight-receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
raise SystemExit(0 if all(row['ok'] for row in result['locales'].values()) else 1)
