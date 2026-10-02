import hashlib,io,json,sys
from pathlib import Path
from docutils.core import publish_doctree
r=Path.cwd();p=Path(__file__).parent;sys.path.insert(0,str(r/'tools'));import build_guide_i18n as g
plans=[]
for lang in g.LANGUAGES:
 rows=json.loads((p/f'{lang}-worklist.json').read_text());draft=json.loads((p/f'{lang}-draft.json').read_text());reviewed=p/f'{lang}-reviewed.json'
 if not reviewed.exists():reviewed.write_text(json.dumps(draft,ensure_ascii=False,indent=2)+'\n')
 targets=json.loads(reviewed.read_text());glossary_path=g.GLOSSARY_DIR/f'{lang}.json';glossary=json.loads(glossary_path.read_text());bindings={k:v for row in rows for k,v in row['ui'].items()};added={k:v for k,v in bindings.items()if k not in glossary['terms']}
 assert set(added)=={'Convert','Plate barcode linkage (Alpha)'}
 for key,value in bindings.items():assert g.runtime_ui_name(key,lang)==value;assert key not in glossary['terms'] or glossary['terms'][key]==value
 glossary['terms'].update(added)
 records=[]
 for row in rows:
  source=row['msgid'];target=targets[str(row['index'])];assert not g.message_problems(source,target,glossary['terms']);stream=io.StringIO();publish_doctree(target,settings_overrides={'warning_stream':stream,'halt_level':6,'report_level':2});assert not stream.getvalue(),(lang,row['index'],stream.getvalue())
  records.append({'index':row['index'],'domain':row['domain'],'source':source,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'original_ai_draft':draft[str(row['index'])],'reviewed_translation':target,'ui_bindings':row['ui']})
 peer=p/('sv-peer-review.json'if lang=='sv' else 'root-peer-review-shared-four.json'if lang in ['zh_CN','hi','ko','is'] else 'root-peer-review-merge-four.json');assert peer.exists(),peer
 evidence={'schema':1,'language':lang,'method':'Direct Codex AI technical translation with independent Codex AI peer review; no local-model draft or human/native-speaker signoff. Normal guide literal, role, UI and source-bound checks unchanged.','draft_author':'root'if lang=='sv' else 'shared_ui'if lang in ['zh_CN','hi','ko','is'] else 'merge_interfaces','independent_technical_reviewer':'shared_ui'if lang=='sv' else 'root','draft_sha256':hashlib.sha256((p/f'{lang}-draft.json').read_bytes()).hexdigest(),'reviewed_sha256':hashlib.sha256(reviewed.read_bytes()).hexdigest(),'peer_review':json.loads(peer.read_text()),'peer_review_sha256':hashlib.sha256(peer.read_bytes()).hexdigest(),'source_files':{'docs/source/plate_barcode_linkage.rst':hashlib.sha256((r/'docs/source/plate_barcode_linkage.rst').read_bytes()).hexdigest()},'runtime_catalog_sha256':hashlib.sha256((r/f'spacr/qt/i18n_catalogs/{lang}.py').read_bytes()).hexdigest(),'derivative':'Korean closing-bold boundary escapes only, displayed meaning unchanged'if lang=='ko' else 'Independent root review clarifies original source plate naming' if lang in ['es','pt'] else 'Accepted original draft unchanged','records':records}
 plans.append((lang,glossary_path,glossary,evidence))
for lang,path,glossary,evidence in plans:
 path.write_text(json.dumps(glossary,ensure_ascii=False,indent=1)+'\n')
 count,rejected=g.import_worklist(lang,p/f'{lang}-worklist.json',p/f'{lang}-reviewed.json',reviewer='codex');assert count==4 and not rejected,(lang,count,rejected)
 (r/f'docs/i18n/reviewed/guides/{lang}/2026-10-02-convert-barcode-linkage.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n');print(lang,count,'imported',flush=True)
