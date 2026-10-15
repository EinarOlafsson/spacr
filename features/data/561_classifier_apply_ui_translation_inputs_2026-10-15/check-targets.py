from pathlib import Path
import json,sys,hashlib
root=Path.cwd();scratch=Path('/mnt/wd4tb/scratch/home-b-f561-classifier-apply-ui-20261015');sys.path.insert(0,str(root/'tools'))
import build_i18n_catalogs as r
import build_documentation_i18n as a
import build_guide_i18n as g
plan=json.loads((scratch/'translation-obligations.json').read_text());targets={}
for name in ['west','middle','asia']:targets.update(json.loads((scratch/f'targets/{name}.json').read_text()))
def refuse(*args,**kwargs):raise AssertionError('No canonical/public collection permitted in input-only checks')
r.canonical_sources=refuse;a.public_docstrings=refuse
failures=[];counts={'runtime':0,'api_source_context':0,'api_inline':0};sha=lambda x:hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
for lang,parts in targets.items():
 assert set(map(int,parts['runtime']))=={i for i,x in enumerate(plan['runtime']) if x['predicted_disposition']=='new_ui_source_requires_nine_targets'}
 assert len(parts['api'])==len(plan['api'])
 r._REVIEWED_RUNTIME_LOADING.add(lang)
 try:
  for idx,target in parts['runtime'].items():
   row=plan['runtime'][int(idx)];source=row['source'];assert sha(source)==row['source_sha256']
   normalized=r._contextualize(target,lang,source)
   reasons=sorted(r._translation_rejection_reasons(source,target,lang,force=r._looks_translatable(source)))
   if reasons or normalized!=target:failures.append({'language':lang,'domain':'runtime','index':idx,'source':source,'target':target,'contextualized':normalized,'reasons':reasons})
   counts['runtime']+=1
  for idx,target in enumerate(parts['api']):
   row=plan['api'][int(idx)];assert sha(row['source'])==row['source_sha256'];assert a._api_translation_source(row['source'])==row['context']
   goodsource=a._reviewed_api_block_valid(row['source'],target,lang);goodcontext=a._reviewed_api_block_valid(row['context'],target,lang);markup=list(g.inline_markup_problems(target));counts['api_source_context']+=2;counts['api_inline']+=1
   if not goodsource or not goodcontext or markup:failures.append({'language':lang,'domain':'api','index':idx,'source':row['source'],'target':target,'source_valid':goodsource,'context_valid':goodcontext,'markup':markup})
 finally:r._REVIEWED_RUNTIME_LOADING.discard(lang)
receipt={'counts':counts,'failures':failures,'scope':'Normal pure candidate/context, source/context API review gates and actual Docutils inline markup. Standard runtime review-loading recursion guard only; canonical/public collection explicitly refused. No normal full review loader or writer executed.','target_files':{name:sha((scratch/f'targets/{name}.json').read_bytes()) for name in ['west','middle','asia']}}
(scratch/'input-check.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,ensure_ascii=False,indent=2))
raise SystemExit(bool(failures))
