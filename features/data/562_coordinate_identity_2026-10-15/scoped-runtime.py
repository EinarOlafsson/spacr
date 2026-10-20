import importlib.util,json,sys,hashlib,subprocess,types
from pathlib import Path
root=Path('/mnt/wd4tb/spacr-worktrees/codex-f562-coordinate-identity-20261015')
sys.path[:0]=[str(root/'tools'),str(root)]
import build_i18n_catalogs as b
original_canonical=b.canonical_sources
bound_files=['spacr/embeddings.py','tools/build_i18n_catalogs.py','spacr/qt/i18n.py','tests/test_mil_coordinate_identity_f562.py']
source_file_hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in bound_files}
sources=original_canonical();expected=b._source_hashes(sources)
b.canonical_sources=lambda:sources
tables={'SETTING_LABELS':'setting_labels','SETTING_TOOLTIPS':'setting_tooltips','MODULE_SUMMARIES':'module_summaries','CATEGORY_HELP':'categories','UI':'ui'}
prepared={};receipt={'mode':'normal exact review lookup and write_language; preserve every unrelated source-current target','locales':{}}
for language in b.MODEL_SPECS:
 owned={}
 for name in ['2026-10-15-mil-coordinate-identity.json']:
  for row in json.loads((root/'docs/i18n/reviewed/runtime'/language/name).read_text())['records']:
   owned[row['source']]=row['translation']
 reviewed=b.reviewed_runtime_translations(language)
 assert all(reviewed[k]==v for k,v in owned.items()),language
 old=types.ModuleType('committed_'+language)
 baseline=subprocess.check_output(['git','show','HEAD:spacr/qt/i18n_catalogs/'+language+'.py'],cwd=root,text=True)
 exec(compile(baseline,'committed_catalog_'+language,'exec'),old.__dict__)
 translations={};untouched={};missing=[]
 for table,section in tables.items():
  values=sources[section];pairs=values.items() if isinstance(values,dict) else ((s,s) for s in values)
  for key,source in pairs:
   ident=(table,str(key))
   if source in owned:continue
   previous=getattr(old,table)
   if key not in previous or old.SOURCE_HASHES.get(ident)!=expected[ident]:missing.append(ident);continue
   value=previous[key]
   assert source not in translations or translations[source]==value,(language,source)
   translations[source]=value;untouched[ident]=value
 assert not missing,(language,'unrelated targets not source-current',missing)
 for source,target in owned.items():
  assert b._translation_candidate_valid(source,target,language,force=False),(language,source)
 translations.update(owned);prepared[language]=(translations,untouched,owned)
 print(language+': normal review and source-current preflight passed',flush=True)
if '--write' in sys.argv:
 assert original_canonical()==sources,'Source changed during generation'
 b.write_english(sources)
 for language,(translations,untouched,owned) in prepared.items():
  path=b.write_language(language,sources,translations)
  spec=importlib.util.spec_from_file_location('new_'+language,path);new=importlib.util.module_from_spec(spec);spec.loader.exec_module(new)
  assert all(getattr(new,t)[k]==v for (t,k),v in untouched.items()),language
  assert all(new.UI[k]==v for k,v in owned.items()),language
  receipt['locales'][language]={'owned_reviewed_rows':len(owned),'unrelated_preserved_rows':len(untouched)}
 audit_exit=b.audit(sources,tuple(b.MODEL_SPECS))
 assert audit_exit==0
 import i18n_caption_arrivals as arrivals
 receipt['normal_full_nine_runtime_audit_exit']=audit_exit
 receipt['normal_arrivals']=arrivals.refresh_arrival_contract()
 assert original_canonical()==sources,'Source changed during final normal audit/arrival'
 assert source_file_hashes=={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in bound_files},'Source file body changed'
 receipt['source_file_sha256_unchanged']=source_file_hashes
 Path('/mnt/wd4tb/scratch/f562-coordinate-identity-20261015/runtime-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
