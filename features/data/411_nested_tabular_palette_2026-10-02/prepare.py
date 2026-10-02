import json,hashlib,sys,subprocess
from pathlib import Path
root=Path.cwd();p=Path(__file__).parent;sys.path.insert(0,str(root/'tools'))
import build_documentation_i18n as api
import nested_helper_docs as helpers
bundle=json.loads((p/'drafts.json').read_text());docs=api.public_docstrings();old=json.loads((api.API_DIR/'en.json').read_text())['symbols'];keys={x['key'] for x in bundle['sources']}
delta={'before':len(old),'after':len(docs),'added':sorted(set(docs)-set(old)),'removed':sorted(set(old)-set(docs)),'changed':[k for k in old if k in docs and old[k]['text']!=docs[k]]};assert delta=={'before':12036,'after':12038,'added':sorted(keys),'removed':[],'changed':[]},delta
peer={'reviewer':'root Codex AI','decision':'All18targets accepted: writing each worksheet, exact pending literal and visibility fallback preserved.','scope':'Independent AI technical peer review; no native-speaker signoff.','draft_sha256':hashlib.sha256((p/'drafts.json').read_bytes()).hexdigest()};(p/'peer-review.json').write_text(json.dumps(peer,indent=2)+'\n')
proof=[]
for i,item in enumerate(bundle['sources']):
 blocks,_=api.translatable_blocks(docs[item['key']]);assert blocks==[item['source']]
 uses=[key for key,doc in docs.items() if item['source'] in api.translatable_blocks(doc)[0]]
 proof.append({'key':item['key'],'source':item['source'],'all_exact_source_users':uses})
 for lang,targets in bundle['translations'].items():
  source=item['source'];target=targets[i];normalized=api._contextualize(target,lang,source)
  assert normalized==target,(lang,i,target,normalized)
  assert api._reviewed_api_block_valid(source,target,lang),(lang,i,target)
for lang,targets in bundle['translations'].items():
 records=[{'label':item['key']+'#0','source':item['source'],'source_sha256':api._source_hash(item['source']),'context':api._api_translation_source(item['source']),'translation':targets[i],'review_basis':peer['scope']} for i,item in enumerate(bundle['sources'])]
 dest=api.REVIEWED_API_DIR/lang/'2026-10-02-nested-tabular-palette.json';dest.write_text(json.dumps({'schema':1,'language':lang,'review_kind':'Direct Codex AI translation with independent Codex AI peer review; no human/native-speaker signoff.','records':records},ensure_ascii=False,indent=2)+'\n')
(p/'source-delta.json').write_text(json.dumps(delta,indent=2)+'\n');(p/'shared-source-uses.json').write_text(json.dumps(proof,indent=2)+'\n')
(p/'inventory.json').write_text(json.dumps(helpers.report(helpers.inventory(root,ignore_patterns=api.AUTOAPI_IGNORE),api.translatable_blocks),indent=2)+'\n')
print('PASS',delta,flush=True)
