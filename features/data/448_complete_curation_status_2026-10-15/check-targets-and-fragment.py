from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys

root=Path.cwd();scratch=Path('/mnt/wd4tb/scratch/n448-complete-status-locales-20261015');sys.path.insert(0,str(root/'tools'))
import build_i18n_catalogs as normal
normal.canonical_sources=lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('No global collector'))
owner=json.loads((scratch/'normal-literal-owner.json').read_text());captions=set(owner['captions']);results={}
import tempfile
import shutil
with tempfile.TemporaryDirectory(dir=scratch,prefix='scoped-review-inputs-') as directory:
 normal.REVIEWED_RUNTIME_DIR=Path(directory)
 normal.canonical_sources=lambda: {'ui':set(captions)}
 for p in sorted((scratch/'runtime-drafts').glob('*.json')):
  d=json.loads(p.read_text());lang=d['language'];(Path(directory)/lang).mkdir()
  shutil.copy2(p,Path(directory)/lang/'scoped.json')
  for r in d['records']:
   assert r['source'] in captions
   assert r['key']==r['source']
   assert hashlib.sha256(r['source'].encode()).hexdigest()==r['source_sha256']
  normal.reviewed_runtime_translations.cache_clear()
  accepted=normal.reviewed_runtime_translations(lang)
  assert len(accepted)==8
  results[lang]=[{'source_sha256':r['source_sha256'],'context_unchanged':True,'reasons':[]} for r in d['records']]
(scratch/'normal-target-gates.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
fragment='no unjudged fields are left in this queue';matches=subprocess.run(['rg','-l','-F',fragment,'spacr','tools','--glob','*.py','--glob','!spacr/qt/i18n_catalogs/**'],capture_output=True,text=True,check=False)
sourcepaths=matches.stdout.splitlines();uses=[]
for raw in sourcepaths:
 path=Path(raw);tree=ast.parse(path.read_text());exact=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Constant) and n.value==fragment]
 uses.append({'path':raw,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'exact_literal_lines':exact})
assert not any(row['exact_literal_lines'] for row in uses)
retirement={}
for lang in results:
 path=root/'docs/i18n/reviewed/runtime'/lang/'2026-10-13-701-hide-judged-fields.json';payload=json.loads(path.read_text());rows=[r for r in payload['records'] if r['source']==fragment];assert len(rows)==1
 retirement[lang]={'path':str(path.relative_to(root)),'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'exact_old_row':rows[0],'other_rows_to_preserve':len(payload['records'])-1}
(scratch/'old-fragment-proposed-retirement.json').write_text(json.dumps({'source_scan':'rg complete exact phrase over non-generated spacr/tools Python sources, then actual AST exact-literal check','matched_source_files':uses,'remaining_exact_literals':0,'scope':'New complete status templates contain phrase as part of fullsentence; no remaining exact literal source. Normal literal extractor replacement is sourcebound. Final global canonical absence must be confirmed by the upcoming ordinary writer before retiring inputs.','retirement':'Conditional exact9row history-preserving retirement ONLY; archive original full files, keep every other reviewed row. No current review files modified here.','records':retirement},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({lang:[row for row in rows if not row['context_unchanged'] or row['reasons']] for lang,rows in results.items()},ensure_ascii=False))
