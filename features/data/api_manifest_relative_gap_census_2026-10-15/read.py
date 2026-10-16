from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,subprocess,time
root=Path('/mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015');out=Path('/mnt/wd4tb/scratch/api-manifest-relative-gap-census-20261015');base=root/'docs/source/_static/i18n/api';start=time.time();langs=('de','es','fr','sv','pt','is','zh_CN','ko','hi');paths={l:base/(l+'.json') for l in ('en',)+langs};pin=lambda:{l:hashlib.sha256(p.read_bytes()).hexdigest() for l,p in paths.items()};before=pin();en=json.loads(paths['en'].read_text())['symbols'];pending=set(json.loads(Path('/mnt/wd4tb/scratch/ci-providers-api-label-diagnosis-20261015/author-blocks.json').read_text()));assert len(pending)==26
tracked=subprocess.check_output(['git','-C',str(root),'ls-files','spacr'],text=True).splitlines();modules={p.removesuffix('/__init__.py').removesuffix('.py').replace('/','.'):p for p in tracked if p.endswith('.py')}
def group(symbol):
 parts=symbol.split('.')
 for i in range(len(parts),0,-1):
  prefix='.'.join(parts[:i])
  if prefix in modules:return modules[prefix]
 return '[unmapped-source-path]'
metadata_fields=('source_sha256','source_blocks_sha256','translation_source_blocks_sha256');result={};aggregate=defaultdict(lambda:{'languages':set(),'symbols':set(),'missing':0,'stale':0});context_en=sum('translation_source_blocks_sha256' in r for r in en.values());unique={}
for lang in langs:
 locale=json.loads(paths[lang].read_text())['symbols'];rows=[];extras=sorted(set(locale)-set(en));context=Counter()
 for symbol,e in en.items():
  r=locale.get(symbol);diff=[]
  if r is None:state='missing'
  else:
   for field in metadata_fields:
    if field in e and e.get(field)!=r.get(field):diff.append(field)
   if 'translation_source_blocks_sha256' not in r:context['missing_field']+=1
   elif not isinstance(r['translation_source_blocks_sha256'],list):context['invalid_type']+=1
   elif len(r['translation_source_blocks_sha256'])!=len(r.get('source_blocks_sha256',[])):context['length_differs_from_own_source_blocks']+=1
   else:context['present_list_length_matches_own_source_blocks']+=1
   state='stale' if diff else 'same_en_metadata'
  if state=='same_en_metadata':continue
  path=group(symbol);row={'symbol':symbol,'grouped_source_path':path,'status_relative_to_en':state,'pending_provider_writer':symbol in pending,'differing_fields':diff,'en_source_sha256':e.get('source_sha256'),'locale_source_sha256':r.get('source_sha256') if r else None,'en_source_block_count':len(e.get('source_blocks_sha256',[])),'locale_source_block_count':len(r.get('source_blocks_sha256',[])) if r else None,'locale_context_field_present':r is not None and 'translation_source_blocks_sha256' in r};rows.append(row)
  if symbol not in pending:
   aggregate[path]['languages'].add(lang);aggregate[path]['symbols'].add(symbol);aggregate[path][state]+=1;unique[symbol]=path
 result[lang]={'symbol_count':len(locale),'missing':sum(r['status_relative_to_en']=='missing' for r in rows),'stale':sum(r['status_relative_to_en']=='stale' for r in rows),'provider_pending_rows':sum(r['pending_provider_writer'] for r in rows),'other_gap_rows':sum(not r['pending_provider_writer'] for r in rows),'extra_names_absent_from_en':extras,'context_metadata_shape':dict(context),'gaps':rows}
after=pin();assert before==after,'Manifest bytes changed during read-only census'
groups=[{'source_path':path,'unique_symbol_count':len(v['symbols']),'language_count':len(v['languages']),'missing_locale_rows':v['missing'],'stale_locale_rows':v['stale'],'symbols':sorted(v['symbols'])} for path,v in aggregate.items()];groups.sort(key=lambda r:(-r['unique_symbol_count'],r['source_path']))
receipt={'basis':'EXISTING_EN_MANIFEST_RELATIVE_METADATA_ONLY','head':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),'input_paths':{l:str(p.relative_to(root)) for l,p in paths.items()},'input_sha256_before':before,'input_sha256_after':after,'english_symbol_count':len(en),'english_records_with_context_metadata':context_en,'context_limit':'No context currentness inferred where English lacks context hashes. Locale field presence/type/own-source-block count only; no contextualizer/source inventory invoked.','provider_pending_symbols':sorted(pending),'other_unique_gap_count':len(unique),'other_groups':groups,'languages':result,'seconds':time.time()-start,'limits':['English is existing reference, not proven current source inventory outside53 providers','Missing/stale metadata is not a fresh runtime/API/source implementation claim','No collector/public_docstrings/review loader/audit/model/generation/network/test executed']};(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');summary={k:receipt[k] for k in ('basis','head','english_symbol_count','english_records_with_context_metadata','other_unique_gap_count','seconds')};summary['languages']={l:{k:v[k] for k in ('missing','stale','provider_pending_rows','other_gap_rows','context_metadata_shape')} for l,v in result.items()};summary['other_groups']=groups;print(json.dumps(summary,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
