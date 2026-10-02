import hashlib,io,json,sys
from pathlib import Path
from babel.messages.mofile import write_mo
from babel.messages.pofile import read_po
r=Path.cwd();p=Path(__file__).parent;sys.path.insert(0,str(r/'tools'));import build_guide_i18n as g
proof=[]
for lang in g.LANGUAGES:
 cat=g.read_catalog(g.LOCALE_DIR/lang/'LC_MESSAGES/plate_barcode_linkage.po');old=read_po(io.StringIO((p/f'{lang}-before.po').read_text()));evidence=json.loads((r/f'docs/i18n/reviewed/guides/{lang}/2026-10-02-convert-barcode-linkage.json').read_text());changed={row['source']for row in evidence['records']};preserved=0
 for m in old:
  if not m.id:continue
  now=cat.get(m.id);assert now is not None and now.string==m.string and now.user_comments==m.user_comments and now.fuzzy==m.fuzzy,(lang,m.id);preserved+=1
 assert set(m.id for m in cat if m.id)-set(m.id for m in old if m.id)==changed
 for row in evidence['records']:
  m=cat.get(row['source']);assert m.string==row['reviewed_translation'] and not m.fuzzy
  assert 'AI technical review (Codex), no native-speaker signoff.'in m.user_comments
  assert hashlib.sha256(row['source'].encode()).hexdigest()==row['source_sha256']
  assert evidence['source_files']['docs/source/plate_barcode_linkage.rst']==hashlib.sha256((r/'docs/source/plate_barcode_linkage.rst').read_bytes()).hexdigest()
  for source,target in row['ui_bindings'].items():assert g.runtime_ui_name(source,lang)==target
 output=io.BytesIO();write_mo(output,cat);assert output.tell()>0
 proof.append({'language':lang,'new_messages':len(changed),'existing_messages_and_comments_unchanged':preserved,'compiled_mo_bytes':output.tell()})
report=g.audit(p/'selected-pot',g.LANGUAGES)
for lang,row in report['languages'].items():assert row['translated']==row['total'] and row['stale']==0 and not row['invalid'] and not row['label_missing'],(lang,row)
(p/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');(p/'preservation-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2));print('PASS all9 source-current completepage guide audits')
