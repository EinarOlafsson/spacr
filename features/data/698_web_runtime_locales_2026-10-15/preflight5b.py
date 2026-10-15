import sys,json
from pathlib import Path
root=Path('/mnt/wd4tb/spacr-worktrees/codex-n698-public-backend-integration-20261015');sys.path[:0]=[str(root/'tools'),str(root)]
import build_i18n_catalogs as b
original=b.canonical_sources;sources=original();b.canonical_sources=lambda:sources
result={}
for language in ['sv','de','es','zh_CN','fr']:
 records=json.loads((root/'docs/i18n/reviewed/runtime'/language/'2026-10-15-web-mask-generator.json').read_text())['records'];failures=[]
 try:reviewed=b.reviewed_runtime_translations(language)
 except ValueError as error:
  result[language]={'records':len(records),'failures':[{'review_error':str(error)}]};print(language+': normal review refused '+str(error),flush=True);continue
 for row in records:
  assert reviewed[row['source']]==row['translation']
  reasons=b._translation_rejection_reasons(row['source'],row['translation'],language,force=False)
  if reasons:failures.append({'source':row['source'],'translation':row['translation'],'reasons':sorted(reasons)})
 result[language]={'records':len(records),'failures':failures};print(language+': '+str(len(records)-len(failures))+'/178 normal candidate gates',flush=True)
assert original()==sources,'Canonical sources changed during preflight'
Path('/mnt/wd4tb/scratch/n698-web-localization-20261015/preflight5b-receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
raise SystemExit(int(any(row['failures'] for row in result.values())))
