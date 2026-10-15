import importlib.util,json,sys,hashlib,subprocess,types
from pathlib import Path
root=Path('/mnt/wd4tb/spacr-worktrees/codex-scientific-backends-integration-20261014')
sys.path[:0]=[str(root/'tools'),str(root)]
def committed(ref,language):
    module=types.ModuleType(ref+'_'+language)
    raw=subprocess.check_output(['git','show',ref+':spacr/qt/i18n_catalogs/'+language+'.py'],cwd=root,text=True)
    exec(compile(raw,ref+'_'+language,'exec'),module.__dict__)
    return module
sys.modules['spacr.qt.i18n_catalogs.en']=committed('HEAD','en')
import build_i18n_catalogs as b
sources=b.canonical_sources();expected=b._source_hashes(sources)
tables={'SETTING_LABELS':'setting_labels','SETTING_TOOLTIPS':'setting_tooltips','MODULE_SUMMARIES':'module_summaries','CATEGORY_HELP':'categories','UI':'ui'}
receipt={'normal_operation':'write_english then audit(sources,()); non-English artifacts only ordinary Git merge, no hand edits','locale_proof':{}}
for language in b.MODEL_SPECS:
    owned={}
    for name in ['2026-10-15-annotated-crop-integrity.json','2026-10-14-wound-plane-index.json']:
        for row in json.loads((root/'docs/i18n/reviewed/runtime'/language/name).read_text())['records']:
            assert row['source_sha256']==hashlib.sha256(row['source'].encode()).hexdigest()
            owned[row['source']]=row['translation']
    incoming=committed('MERGE_HEAD',language);previous=committed('HEAD',language)
    merged=types.ModuleType('merged_'+language)
    exec(compile((root/'spacr/qt/i18n_catalogs'/ (language+'.py')).read_text(),'merged_'+language,'exec'),merged.__dict__)
    preserved=0;foreign=[]
    for table,section in tables.items():
        values=sources[section];pairs=values.items() if isinstance(values,dict) else ((x,x) for x in values)
        for key,source in pairs:
            identity=(table,str(key));current=getattr(merged,table)
            if source in owned:
                assert current[key]==owned[source] and merged.SOURCE_HASHES.get(identity)==expected[identity],(language,identity)
                continue
            baseline=incoming if incoming.SOURCE_HASHES.get(identity)==expected[identity] else previous if previous.SOURCE_HASHES.get(identity)==expected[identity] else None
            if baseline is None:
                foreign.append(identity);continue
            assert current[key]==getattr(baseline,table)[key] and merged.SOURCE_HASHES.get(identity)==expected[identity],(language,identity,'unrelated target changed')
            preserved+=1
    receipt['locale_proof'][language]={'owned_exact_reviewed_rows':len(owned),'unrelated_source_current_targets_preserved':preserved,'foreign_missing_or_stale_targets':foreign}
    print(language+': merged owned and unrelated current targets preserved',flush=True)
b.write_english(sources)
assert b.audit(sources,())==0
receipt['incoming_commit']=subprocess.check_output(['git','rev-parse','MERGE_HEAD'],cwd=root,text=True).strip()
Path('/mnt/wd4tb/scratch/f470-f536-integration-20261015/english-merge3-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
