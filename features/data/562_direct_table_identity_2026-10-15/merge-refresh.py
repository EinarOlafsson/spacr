import hashlib,json,sys,subprocess,types
from pathlib import Path
root=Path('/mnt/wd4tb/spacr-worktrees/codex-f562-direct-table-identity-20261015');sys.path[:0]=[str(root/'tools'),str(root)]
import build_i18n_catalogs as b
import i18n_caption_arrivals as a
original=b.canonical_sources;sources=original();expected=b._source_hashes(sources)
tables={'SETTING_LABELS':'setting_labels','SETTING_TOOLTIPS':'setting_tooltips','MODULE_SUMMARIES':'module_summaries','CATEGORY_HELP':'categories','UI':'ui'}
receipt={'normal_operation':'write_english, audit(sources,()), normal refresh_arrival_contract; source snapshot only caches repeated canonical lookups and original rechecked before/after','incoming':subprocess.check_output(['git','rev-parse','MERGE_HEAD'],cwd=root,text=True).strip(),'locale_proof':{}}
def committed(ref,language):
 m=types.ModuleType(ref+language);raw=subprocess.check_output(['git','show',ref+':spacr/qt/i18n_catalogs/'+language+'.py'],cwd=root,text=True);exec(compile(raw,ref+language,'exec'),m.__dict__);return m
for language in b.MODEL_SPECS:
 incoming=committed('MERGE_HEAD',language);previous=committed('HEAD',language);merged=types.ModuleType('merged'+language);exec(compile((root/'spacr/qt/i18n_catalogs'/(language+'.py')).read_text(),'merged'+language,'exec'),merged.__dict__)
 row=json.loads((root/'docs/i18n/reviewed/runtime'/language/'2026-10-15-mil-direct-well-identity.json').read_text())['records'][0];caption=row['source'];preserved=0
 assert merged.UI[caption]==row['translation'] and merged.SOURCE_HASHES['UI',caption]==row['source_sha256']==hashlib.sha256(caption.encode()).hexdigest()
 for table,section in tables.items():
  values=sources[section];pairs=values.items() if isinstance(values,dict) else ((x,x) for x in values)
  for key,source in pairs:
   identity=(table,str(key))
   if table=='UI' and source==caption:continue
   baseline=incoming if incoming.SOURCE_HASHES.get(identity)==expected[identity] else previous if previous.SOURCE_HASHES.get(identity)==expected[identity] else None
   assert baseline is not None,(language,identity,'no current unrelated target')
   assert getattr(merged,table)[key]==getattr(baseline,table)[key] and merged.SOURCE_HASHES.get(identity)==expected[identity],(language,identity,'unrelated target drift')
   preserved+=1
 receipt['locale_proof'][language]={'own_exact_reviewed_rows':1,'unrelated_current_targets_preserved':preserved}
 print(language+': exact owned and unrelated incoming current targets preserved',flush=True)
assert original()==sources
b.write_english(sources);assert b.audit(sources,())==0
b.canonical_sources=lambda:sources
assert original()==sources
receipt['normal_arrivals']=a.refresh_arrival_contract()
assert original()==sources
Path('/mnt/wd4tb/scratch/f562-direct-table-identity-20261015/merge-refresh-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['normal_arrivals']),flush=True)
