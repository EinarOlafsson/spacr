from pathlib import Path
import ast, hashlib, json, shutil, subprocess, sys, tempfile
root=Path.cwd(); scratch=Path('/mnt/wd4tb/scratch/n691-current-whole-item-audit-20261015');sys.path.insert(0,str(root/'tools'))
import build_i18n_catalogs as normal
normal.canonical_sources=lambda *a,**k: (_ for _ in ()).throw(AssertionError('Global runtime collector forbidden'))
import build_documentation_i18n as api
import nested_helper_docs as nested
sha=lambda b:hashlib.sha256(b).hexdigest()
langs=['sv','de','es','zh_CN','fr','pt','hi','ko','is']
paths=['spacr/model_zoo.py','spacr/qt/screens/make_masks.py','spacr/qt/widgets/model_zoo_picker.py']
trees={p:ast.parse((root/p).read_text()) for p in paths}
source_calls={}
for p,t in trees.items():
 for n in ast.walk(t):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='tr' and n.args:
   s=normal._literal(n.args[0])
   if s and ('MitoNet' in s or s=='Third-party' or s.startswith('A third-party model, listed')):source_calls[s]={'path':p,'line':n.lineno,'source_sha256':sha(s.encode())}
assert len(source_calls)==6,source_calls
runtime={}
with tempfile.TemporaryDirectory(dir=scratch,prefix='exact-six-reviews-') as d:
 normal.REVIEWED_RUNTIME_DIR=Path(d)
 for lang in langs:
  p=root/'docs/i18n/reviewed/runtime'/lang/'2026-10-11-n690-n691-spatial-arrivals.json'; payload=json.loads(p.read_text()); rows=[r for r in payload['records'] if r['table']=='ui' and r['source'] in source_calls]; assert len(rows)==6
  normal.canonical_sources=lambda:{'ui':set(source_calls)}
  q=Path(d)/lang/'six.json';q.parent.mkdir();q.write_text(json.dumps({'schema':1,'language':lang,'records':rows}))
  normal.reviewed_runtime_translations.cache_clear(); accepted=normal.reviewed_runtime_translations(lang);assert len(accepted)==6
  compiled={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse((root/f'spacr/qt/i18n_catalogs/{lang}.py').read_text()).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['UI','SOURCE_HASHES']}
  exact=[]
  for r in rows:
   assert compiled['UI'][r['key']]==r['translation']; assert compiled['SOURCE_HASHES'][('UI',r['key'])]==r['source_sha256']
   exact.append({'source':r['source'],'source_sha256':r['source_sha256'],'translation':r['translation'],'translation_sha256':sha(r['translation'].encode()),'ordinary_scoped_review_gates':True,'compiled_exact':True})
  runtime[lang]={'review_file':str(p.relative_to(root)),'review_file_sha256':sha(p.read_bytes()),'records':exact}
(scratch/'runtime.json').write_text(json.dumps({'scope':'Six exact current literal source owners, ordinary scoped review loader with isolated six-row review snapshot; no global collector or source-admission claim beyond actual unchanged literal calls. Compiled UI and SOURCE_HASHES exact.','sources':source_calls,'languages':runtime},ensure_ascii=False,indent=2)+'\n')
print('Six runtime sources × nine exact normal reviewed+compiled targets PASS',flush=True)
