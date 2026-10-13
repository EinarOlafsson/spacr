import hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tools'))
import build_guide_i18n as g
p=Path(__file__).parent;narrow=p/'measure-pot';narrow.mkdir(exist_ok=True)
shutil.copy2(p/'gettext/measure_live.pot',narrow/'measure_live.pot')
proof={}
for lang in g.LANGUAGES:
 path=g.GLOSSARY_DIR/f'{lang}.json';payload=json.loads(path.read_text());before=dict(payload['terms'])
 summary=g.update_language(lang,narrow);g.export_worklist(lang,p/f'{lang}-worklist.json',domains=['measure_live'])
 rows=json.loads((p/f'{lang}-worklist.json').read_text());assert len(rows)==6,(lang,len(rows))
 for row in rows:
  row['ui']={name:g.runtime_ui_name(name,lang) or name for name in g.ui_names(row['msgid'])}
  for name,target in row['ui'].items():
   if name in before:assert before[name]==target,(lang,name,before[name],target)
   elif target!=name:payload['terms'][name]=target
 assert all(payload['terms'][key]==value for key,value in before.items())
 path.write_text(json.dumps(payload,ensure_ascii=False,indent=1)+'\n')
 (p/f'{lang}-worklist.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
 proof[lang]={'new_messages':len(rows),'source_file_sha256':hashlib.sha256(Path('docs/source/measure_live.rst').read_bytes()).hexdigest(),'glossary_additions':{k:v for k,v in payload['terms'].items() if k not in before},'historical_glossary_entries_changed':0}
(p/'worklist-proof.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n');print(json.dumps(proof,ensure_ascii=False,indent=2))
