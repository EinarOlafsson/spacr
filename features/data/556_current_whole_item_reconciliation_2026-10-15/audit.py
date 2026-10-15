from pathlib import Path
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

root=Path.cwd();scratch=Path('/mnt/wd4tb/scratch/f556-current-whole-item-audit-20261015')
sys.path.insert(0,str(root/'tools'))
import build_i18n_catalogs as normal
from spacr.qt.i18n_catalogs import ui_text
from spacr.qt import i18n
source=root/'spacr/qt/screens/make_masks.py';text=source.read_text();tree=ast.parse(text)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_Sam2SeedDialog')
class_text=ast.get_source_segment(text,cls)
publicraw=subprocess.check_output(['git','show','a559ba334b258a83be33e8d292daee1f1e9daec5:spacr/qt/screens/make_masks.py'],text=True)
publiccls=next(n for n in ast.parse(publicraw).body if isinstance(n,ast.ClassDef) and n.name=='_Sam2SeedDialog')
assert class_text==ast.get_source_segment(publicraw,publiccls)
trliterals={normal._literal(n.args[0]) for n in ast.walk(cls) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='tr' and n.args}
languages=['sv','de','es','zh_CN','fr','pt','hi','ko','is'];results={}
with tempfile.TemporaryDirectory(dir=scratch,prefix='scoped-reviews-') as directory:
 normal.REVIEWED_RUNTIME_DIR=Path(directory)
 for lang in languages:
  path=root/'docs/i18n/reviewed/runtime'/lang/'2026-10-04-runtime-debt-thirteenth-pass-sam2-save.json';payload=json.loads(path.read_text());assert len(payload['records'])==4
  records=payload['records'];assert all(row['source'] in trliterals for row in records)
  normal.canonical_sources=lambda: {'ui':{row['source'] for row in records}}
  (Path(directory)/lang).mkdir();shutil.copy2(path,Path(directory)/lang/'exact-four.json')
  normal.reviewed_runtime_translations.cache_clear();reviewed=normal.reviewed_runtime_translations(lang)
  assert len(reviewed)==4
  exact=[]
  for row in records:
   assert ui_text(row['source'],lang)==row['translation'],(lang,row['source'])
   exact.append({'source_sha256':row['source_sha256'],'target_sha256':hashlib.sha256(row['translation'].encode()).hexdigest(),'compiled_exact':True,'normal_review_gates':True})
  class_captions=[{'source':caption,'actual_runtime_target':i18n.tr(caption,language=lang),'has_translation':i18n.has_translation(caption,language=lang)} for caption in sorted(trliterals) if caption]
  results[lang]={'review_path':str(path.relative_to(root)),'review_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'exact4':exact,'all_class_tr_captions':class_captions}
logs={}
for name in ['gpu_u373.log','gpu_hela.log','eval/trackastra.out']:
 path=Path('/mnt/wd4tb/scratch/556')/name;logs[str(path)]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'text':path.read_text()}
receipt={'public_base':'a559ba334b258a83be33e8d292daee1f1e9daec5','dialog_class_sha256':hashlib.sha256(class_text.encode()).hexdigest(),'dialog_class_exact_public_source':True,'fourcaption_result':'All36 exact current source-bound reviewed targets pass actual normal review-loader gates and equal compiled runtime ui_text; class compact/static tr targets also inventoried. Scoped metadata only, no global collector/model/test replay.','languages':results,'historical_donewhen_rawlogs':logs,'scope_limits':'Historical realmovie CTC comparison preserved, not rerun or newly extended; no tutorial requirement in original request, Timeflows export explicitly superseded per item. Whole-item decision remains root review.'}
(scratch/'audit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'normal36_exact_compiled':'PASS','class_missing_translations':{lang:[r['source'] for r in row['all_class_tr_captions'] if not r['has_translation']] for lang,row in results.items()}},ensure_ascii=False))
