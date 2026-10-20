import importlib.util,json,sys,hashlib,subprocess,types
from pathlib import Path
root=Path('/mnt/wd4tb/spacr-worktrees/codex-f564-pretrained-reuse-20261015')
sys.path[:0]=[str(root/'tools'),str(root)]
import build_i18n_catalogs as b
original_canonical=b.canonical_sources
review_filenames=['2026-10-15-counterfactual-pinned-reuse.json']
bound_files=['spacr/attribution.py','spacr/deep_spacr.py','spacr/settings.py','spacr/qt/screens/settings_model.py','spacr/object_roles.py','tools/build_i18n_catalogs.py','tools/build_documentation_i18n.py','tools/nested_helper_docs.py','tests/test_f564_pinned_diffusion_reuse.py','tests/test_f564_checkpoint_file_boundaries.py','tests/test_f564_pinned_diffusion_locales.py','tests/qt/test_f564_checkpoint_controls.py']
bound_files += ['docs/i18n/reviewed/runtime/'+language+'/'+name for language in b.MODEL_SPECS for name in review_filenames]
source_file_hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in bound_files}
assert source_file_hashes['spacr/attribution.py']=='108d9bd05a7fbfa09d73ec5cd7c18062c772c17cc97c46a10410e10f39e3804c'
sources=original_canonical();expected=b._source_hashes(sources)
packet=json.loads(Path('/mnt/wd4tb/scratch/f564-pretrained-backbone-plan-20261015/runtime-five-source-inputs.json').read_text())
owned_sources={record['source'] for record in packet['sources']}
for row in packet['sources']:
 table=sources[row['table']]
 actual=table.get(row['key']) if isinstance(table,dict) else row['key'] if row['key'] in table else None
 assert actual==row['source'],row
b.canonical_sources=lambda:sources
tables={'SETTING_LABELS':'setting_labels','SETTING_TOOLTIPS':'setting_tooltips','MODULE_SUMMARIES':'module_summaries','CATEGORY_HELP':'categories','UI':'ui'}
section_table={v:k for k,v in tables.items()}
prepared={};receipt={'mode':'normal exact review lookup/write_language/audit/arrival, preserve every unrelated source-current target','locales':{},'owned_sources':5,'api_disposition':'Normal scoped nested21 and public/module-doc proof unchanged against current public54a, UI owner zeroAPI receipt; no invented target'}
receipt['input_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
receipt['public_baseline']='d05aa3058a18abad52e59d15b829e2b45cf28b0d'
receipt['driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
for language in b.MODEL_SPECS:
 rows=json.loads((root/'docs/i18n/reviewed/runtime'/language/review_filenames[0]).read_text())['records']
 owned={row['source']:row['translation'] for row in rows}
 assert set(owned)==owned_sources and len(rows)==5,language
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
 for source,target in owned.items():assert b._translation_candidate_valid(source,target,language,force=False),(language,source)
 translations.update(owned);prepared[language]=(translations,untouched,rows)
 print(language+': normal review and source-current preflight passed',flush=True)
if '--write' in sys.argv:
 b.write_english(sources)
 for language,(translations,untouched,rows) in prepared.items():
  path=b.write_language(language,sources,translations)
  spec=importlib.util.spec_from_file_location('new_'+language,path);new=importlib.util.module_from_spec(spec);spec.loader.exec_module(new)
  assert all(getattr(new,t)[k]==v for (t,k),v in untouched.items()),language
  assert all(getattr(new,section_table[row['table']])[row['key']]==row['translation'] for row in rows),language
  receipt['locales'][language]={'owned_reviewed_rows':len(rows),'unrelated_preserved_rows':len(untouched)}
 audit_exit=b.audit(sources,tuple(b.MODEL_SPECS));assert audit_exit==0
 import i18n_caption_arrivals as arrivals
 receipt['normal_full_nine_runtime_audit_exit']=audit_exit
 receipt['normal_arrivals']=arrivals.refresh_arrival_contract()
 assert original_canonical()==sources,'Source changed during final normal audit/arrival'
 assert source_file_hashes=={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in bound_files},'Source file body changed'
 receipt['source_file_sha256_unchanged']=source_file_hashes
 Path('/mnt/wd4tb/scratch/f564-pretrained-backbone-plan-20261015/runtime45-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
