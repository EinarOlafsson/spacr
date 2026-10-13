import sys,json,hashlib,io,subprocess
from pathlib import Path
from docutils.core import publish_doctree
root=Path('/tmp/spacr-implementation-20261001/suggest-capture');p=Path(__file__).parent
sys.path.insert(0,str(root/'tools'));import build_guide_i18n as g
sha=lambda x:hashlib.sha256(x).hexdigest()
proof={}
for lang in g.LANGUAGES:
 w=p/f'{lang}-worklist.json';d=p/f'{lang}-draft.json';r=p/f'{lang}-reviewed.json';r=r if r.exists() else d
 rows=json.loads(w.read_text());targets=json.loads(r.read_text())
 for i,row in enumerate(rows):
  value=targets[str(i)];problems=g.message_problems(row['msgid'],value,{**g.load_glossary(lang),**row['ui']});assert not problems,(lang,i,problems)
  warning=io.StringIO();publish_doctree(value,settings_overrides={'warning_stream':warning,'halt_level':6,'report_level':2});assert not warning.getvalue(),(lang,i,warning.getvalue())
 po=root/f'docs/i18n/guides/{lang}/LC_MESSAGES/measure_live.po'
 old_bytes=subprocess.check_output(['git','show',f'HEAD:{po.relative_to(root)}'],cwd=root)
 old_path=p/f'{lang}-previous.po';old_path.write_bytes(old_bytes);old=g.read_catalog(old_path)
 # The merged current file already retains all old translated entries; compare their values after import.
 existing={m.id:m.string for m in old if m.id and m.string and m.id not in {x['msgid'] for x in rows}}
 applied,rejected=g.import_worklist(lang,w,r,reviewer='codex');assert applied==6 and not rejected,(lang,applied,rejected)
 final=g.read_catalog(po);assert all(final.get(k).string==v for k,v in existing.items())
 for i,row in enumerate(rows):assert final.get(row['msgid']).string==targets[str(i)]
 record={'schema':1,'language':lang,'method':'Direct Codex AI technical translation with independent Codex AI technical peer review; no human or native-speaker signoff. Standard guide import and gates unchanged.','draft_author':'root' if lang=='sv' else ('merge_interfaces' if lang in ('de','es','pt','fr') else 'mask_editor' if lang in ('ko','is') else 'shared_ui'),'independent_reviewer':'shared_ui' if lang in ('sv','ko','is') else 'root','draft_sha256':sha(d.read_bytes()),'reviewed_sha256':sha(r.read_bytes()),'source_files':{'docs/source/measure_live.rst':sha((root/'docs/source/measure_live.rst').read_bytes())},'runtime_catalog_sha256':sha((root/f'spacr/qt/i18n_catalogs/{lang}.py').read_bytes()),'records':[{'index':i,'domain':row['domain'],'source':row['msgid'],'source_sha256':sha(row['msgid'].encode()),'original_draft':json.loads(d.read_text())[str(i)],'target':targets[str(i)],'ui':row['ui']} for i,row in enumerate(rows)]}
 dest=root/f'docs/i18n/reviewed/guides/{lang}/2026-10-02-measure-preview-export-calibration.json';dest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 proof[lang]={'applied':applied,'existing_translations_preserved':len(existing),'po_sha256':sha(po.read_bytes()),'review_sha256':sha(dest.read_bytes())}
(p/'import-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
audit=g.audit(p/'gettext',g.LANGUAGES);(p/'guide-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
for lang,v in audit['languages'].items():
 assert v['total']==v['translated'] and not v['stale'] and not v['invalid'] and not v['label_missing'],(lang,v)
print({lang:{k:v[k] for k in ('total','translated','stale')} for lang,v in audit['languages'].items()})
