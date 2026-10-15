from pathlib import Path
import hashlib,json,sys
root=Path.cwd();scratch=Path(__file__).parent;sys.path.insert(0,str(root/'tools'))
import build_documentation_i18n as a;import build_i18n_catalogs as r;import build_guide_i18n as g
plan=json.loads((scratch/'sources.json').read_text());sha=lambda s:hashlib.sha256(s.encode() if isinstance(s,str) else s).hexdigest()
def refuse(*args,**kw):raise AssertionError('No collector permitted in preparation')
r.canonical_sources=refuse;a.public_docstrings=refuse
issues=[];checked=0;shifted=[]
for lang,packet in plan['inputs'].items():
 r._REVIEWED_RUNTIME_LOADING.add(lang)
 try:
  for label,row in packet['active_records'].items():
   context=a._api_translation_source(row['source']);target=row['translation'];flags={'source':a._reviewed_api_block_valid(row['source'],target,lang),'context':a._reviewed_api_block_valid(context,target,lang),'markup':list(g.inline_markup_problems(target))}
   checked+=1
   if not flags['source'] or not flags['context'] or flags['markup']:issues.append({'language':lang,'label':label,'source':row['source'],'context':context,'target':target,'flags':flags})
  en=json.loads((a.API_DIR/'en.json').read_text());catalog=json.loads((a.API_DIR/f'{lang}.json').read_text())
  for row in plan['source_identical_shifted_blocks']:
   key,_,oldpos=row['old_label'].rpartition('#');record=catalog['symbols'][key];source_record=en['symbols'][key]
   assert record['source_sha256']==source_record['source_sha256']
   source_blocks=a.translatable_blocks(source_record['text'])[0];target_blocks=a.translatable_blocks(record['text'])[0];assert len(source_blocks)==len(target_blocks)
   assert source_blocks[int(oldpos)]==row['source'];target=target_blocks[int(oldpos)];context=a._api_translation_source(row['source'])
   flags={'source':a._reviewed_api_block_valid(row['source'],target,lang),'context':a._reviewed_api_block_valid(context,target,lang),'markup':list(g.inline_markup_problems(target))}
   shifted.append({'language':lang,**row,'target':target,'target_sha256':sha(target),'flags':flags,'catalog_sha256':sha((a.API_DIR/f'{lang}.json').read_bytes())})
   if not flags['source'] or not flags['context'] or flags['markup']:issues.append({'language':lang,'label':row['label'],'kind':'historical_shift','source':row['source'],'context':context,'target':target,'flags':flags})
 finally:r._REVIEWED_RUNTIME_LOADING.discard(lang)
receipt={'reviewed_current_records_checked':checked,'shifted_historical_records_checked':len(shifted),'issues':issues,'historical_shift_provenance':shifted,'normalization_count':len(plan['context_normalization_required']),'scope':'Normal pure source/context/inline gates, exact old English/locale layouts, no decoder/collector/writer or repository input changes.'}
(scratch/'preflight.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'current':checked,'shifted':len(shifted),'issues':issues},ensure_ascii=False,indent=2));raise SystemExit(bool(issues))
