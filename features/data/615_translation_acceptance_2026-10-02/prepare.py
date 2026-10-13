import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tools'))
import build_documentation_i18n as api
import build_guide_i18n as guide
p=Path(__file__).parent
bundle=json.loads((p/'api-drafts.json').read_text());label=bundle['label'];key=label.rsplit('#',1)[0]
docs=api.public_docstrings();old=json.loads((api.API_DIR/'en.json').read_text())['symbols']
delta={'old_count':len(old),'new_count':len(docs),'added':sorted(set(docs)-set(old)),'removed':sorted(set(old)-set(docs)),'changed':[k for k in old if k in docs and old[k]['text']!=docs[k]]}
assert delta=={'old_count':12035,'new_count':12036,'added':[key],'removed':[],'changed':[]},delta
blocks,_=api.translatable_blocks(docs[key]);assert blocks==[bundle['source']],blocks
(p/'api-source-delta.json').write_text(json.dumps(delta,indent=2)+'\n')
peer={'reviewer':'Codex AI root','method':'Independent AI technical peer review, accepted all nine exact drafts without correction; no human/native-speaker claim.','draft_sha256':hashlib.sha256((p/'api-drafts.json').read_bytes()).hexdigest(),'scope':'Read-only preview of the existing plate-calibration plan in Measure.','accepted':list(bundle['targets'])}
(p/'api-peer-review.json').write_text(json.dumps(peer,indent=2)+'\n')
for language,target in bundle['targets'].items():
 source=blocks[0];target_context=api._contextualize(target,language,source)
 assert target_context==target,(language,target,target_context)
 assert api._reviewed_api_block_valid(source,target,language),(language,target)
 payload={'schema':1,'language':language,'review_kind':'Direct Codex AI technical translation; independently peer-reviewed by root Codex AI; no human/native-speaker signoff.','records':[{'label':label,'source':source,'source_sha256':api._source_hash(source),'context':api._api_translation_source(source),'translation':target,'review_basis':peer['method']}]}
 (api.REVIEWED_API_DIR/language/'2026-10-02-calibration-preview-module.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
report=guide.audit(p/'gettext',guide.LANGUAGES)
(p/'guide-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
for lang,data in report['languages'].items():
 print(lang,{k:v for k,v in data.items() if k in ('total','translated','coverage','stale','invalid','label_missing')},flush=True)
 assert data['translated']==data['total'] and not data['stale'] and not data['invalid'] and not data['label_missing'],lang
print('PASS +1 API module; nine accepted reviewed records; complete current guide source coverage',flush=True)
