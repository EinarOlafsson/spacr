import importlib.util,json,sys,hashlib,subprocess,types
from pathlib import Path
root=Path('/mnt/wd4tb/spacr-worktrees/codex-n703-debian-adapter-20261015')
scratch=Path('/mnt/wd4tb/scratch/n703-coalesced-runtime-20261015')
scratch.mkdir(exist_ok=True)
sys.path[:0]=[str(root/'tools'),str(root)]
import build_i18n_catalogs as b
baseline_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
source_names=['spacr/install_cleanup.py','spacr/qt/app.py','spacr/qt/install_consent.py','tools/build_i18n_catalogs.py','tools/i18n_caption_arrivals.py']
source_pins={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in source_names}
def refuse_decode(*args,**kwargs):
 raise AssertionError('This reviewed-only writer must never decode model translations')
b._translate_batches=refuse_decode
original_canonical=b.canonical_sources
sources=original_canonical();expected=b._source_hashes(sources)
(scratch/'canonical-snapshot.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')
b.canonical_sources=lambda:sources
tables={'SETTING_LABELS':'setting_labels','SETTING_TOOLTIPS':'setting_tooltips','MODULE_SUMMARIES':'module_summaries','CATEGORY_HELP':'categories','UI':'ui'}
prepared={};receipt={'mode':'Normal exact reviewed-runtime lookup/write_language, all unrelated source-current targets preserved; full nine-locale audit','locales':{}}
receipt['baseline_head']=baseline_head
receipt['canonical_counts']={key:len(value) for key,value in sources.items()}
receipt['review_files']={}
print(json.dumps({'baseline_head':baseline_head,'canonical_counts':receipt['canonical_counts']}),flush=True)
for language in b.MODEL_SPECS:
 owned={}
 review_pins={}
 for filename,number in [('2026-10-15-installer-cleanup-preview.json',22),('2026-10-15-installer-cleanup-action.json',6),('2026-10-15-debian-update-transaction.json',37)]:
  path=root/'docs/i18n/reviewed/runtime'/language/filename
  rows=json.loads(path.read_text())['records'];assert len(rows)==number,(language,filename,len(rows))
  review_pins[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
  for row in rows:
   assert row['source'] not in owned,(language,'duplicate owned source',row['source'])
   owned[row['source']]=row['translation']
 receipt['review_files'].update(review_pins)
 assert len(owned)==65
 missing_owned=sorted(set(owned)-set(sources['ui']))
 if missing_owned:
  from spacr.qt.i18n import _ROWS,_TERM_ROWS
  (scratch/'missing-owned.json').write_text(json.dumps({'missing':missing_owned,'compact_rows':[x for x in missing_owned if x in _ROWS],'compact_terms':[x for x in missing_owned if x in _TERM_ROWS]},indent=2)+'\n')
 assert not missing_owned,(language,'source extractor misses owned captions',missing_owned)
 reviewed=b.reviewed_runtime_translations(language)
 assert all(reviewed[k]==v for k,v in owned.items()),language
 old=types.ModuleType('committed_'+language)
 baseline=subprocess.check_output(['git','show',baseline_head+':spacr/qt/i18n_catalogs/'+language+'.py'],cwd=root,text=True)
 exec(compile(baseline,'committed_catalog_'+language,'exec'),old.__dict__)
 translations={};untouched={};missing=[]
 existing_owned=set(owned)&set(old.UI)
 receipt['locales'][language]={'baseline_ui_count':len(old.UI),'owned_existing':len(existing_owned),'owned_new':len(set(owned)-existing_owned),'baseline_all_rows':sum(len(getattr(old,t)) for t in tables)}
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
  assert b._translation_candidate_valid(source,target,language,force=False),(language,source,target)
 translations.update(owned);prepared[language]=(translations,untouched,owned,old)
 print(language+': normal review and source-current preflight passed',flush=True)
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in source_pins.items())
b.write_english(sources)
for language,(translations,untouched,owned,old) in prepared.items():
 path=b.write_language(language,sources,translations)
 spec=importlib.util.spec_from_file_location('new_'+language,path);new=importlib.util.module_from_spec(spec);spec.loader.exec_module(new)
 assert all(getattr(new,t)[k]==v for (t,k),v in untouched.items()),language
 assert all(new.UI[k]==v for k,v in owned.items()),language
 assert all(new.SOURCE_HASHES[ident]==old.SOURCE_HASHES[ident] for ident in untouched),language
 for name in ['MODEL','LICENSE','SECONDARY_MODEL','SECONDARY_LICENSE','NORMALIZER']:
  assert getattr(new,name,None)==getattr(old,name,None),(language,'metadata changed',name)
 receipt['locales'][language].update({'owned_reviewed_rows':len(owned),'unrelated_preserved_rows':len(untouched)})
 print(language+':65owned exact; foreign targets/hashes/catalog metadata preserved',flush=True)
result=b.audit(sources,list(b.MODEL_SPECS));assert result==0,result
assert original_canonical()==sources,'Fresh original canonical snapshot differs after writes/audit'
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in source_pins.items()),'Source/tool changed during writer'
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in receipt['review_files'].items()),'Reviewed input changed during writer'
receipt['source_sha256']=source_pins
receipt['full_nine_locale_audit']=result
(scratch/'runtime-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
