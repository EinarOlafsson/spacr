from pathlib import Path
import hashlib,importlib.util,json,sys
p=Path('/mnt/wd4tb/scratch/ci-providers-api-label-diagnosis-20261015')
root=Path('/mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015')
d=json.loads((p/'diagnosis.json').read_text())
path=root/'tools/build_documentation_i18n.py'
assert hashlib.sha256(path.read_bytes()).hexdigest()==d['normal_tool_sha256']
assert hashlib.sha256((root/'spacr/qt/ai/providers.py').read_bytes()).hexdigest()==d['provider_file_sha256']
sys.path.insert(0,str(root/'tools'))
spec=importlib.util.spec_from_file_location('provider_review_block_validator',path)
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
failures=[];locales={}
for lang in ('de','es','fr','sv','pt','is','zh_CN','ko','hi'):
 f=p/'ci-drafts'/lang/'2026-10-15-n700-provider-utilities.json';payload=json.loads(f.read_text());seen={};labels=set()
 assert payload['schema']==1 and payload['language']==lang
 assert len(payload['records'])==43
 for r in payload['records']:
  assert r['label'] not in labels;labels.add(r['label'])
  symbol,_,raw=r['label'].rpartition('#');blocks,_=tool.translatable_blocks(d['public_provider_current_sources'][symbol]);assert blocks[int(raw)]==r['source']
  assert tool._source_hash(r['source'])==r['source_sha256']
  assert tool._api_translation_source(r['source'])==r['context']
  assert seen.setdefault(r['source'],r['translation'])==r['translation']
  for field in ('source','context'):
   if not tool._reviewed_api_block_valid(r[field],r['translation'],lang):
    failures.append({'language':lang,'label':r['label'],'field':field,'source':r[field],'target':r['translation']})
 locales[lang]={'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'records':len(labels),'unique_sources':len(seen)}
result={'normal_tool_sha256':d['normal_tool_sha256'],'provider_source_sha256':d['provider_file_sha256'],'locales':locales,'records':sum(x['records'] for x in locales.values()),'failures':failures,'scope':'Exact ordinary reviewed block validator only; no collector, public inventory, catalog writer or full audit.'}
(p/'ci-drafts-normal-block-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'records':result['records'],'failures':failures},ensure_ascii=False,indent=2))
assert not failures
