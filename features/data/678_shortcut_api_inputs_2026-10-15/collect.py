import hashlib,json,sys
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root/'tools'))
import build_documentation_i18n as api
all_docs=api.public_docstrings()
ledger=json.loads((root/'features/data/678_shortcuts_prose_2026-10-15/receipt.json').read_text())
changed=['spacr.qt.shortcuts'+('' if r['symbol']=='<module>' else '.'+r['symbol']) for r in ledger['docstring_ledger'] if r['disposition']=='revised']
english=json.loads((root/'docs/source/_static/i18n/api/en.json').read_text())['symbols']
rows=[];visible=[]
for symbol in changed:
 if symbol not in all_docs:continue
 visible.append(symbol)
 blocks,_=api.translatable_blocks(all_docs[symbol]);old,_=api.translatable_blocks(english.get(symbol,{}).get('text',''))
 for index,source in enumerate(blocks):
  if index<len(old) and source==old[index]:continue
  rows.append({'label':f'{symbol}#{index}','source':source,'context':api._api_translation_source(source),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'unchanged_source_at_old_indexes':[i for i,x in enumerate(old) if x==source]})
retire=[]
for p in sorted((root/'docs/i18n/reviewed/api').glob('*/*.json')):
 payload=json.loads(p.read_text())
 for index,record in enumerate(payload.get('records',[])):
  if record.get('retired'):continue
  label=record.get('label','');symbol,_,position=label.rpartition('#')
  if symbol not in visible:continue
  blocks,_=api.translatable_blocks(all_docs[symbol]);source=record.get('source','')
  if position.isdigit() and int(position)<len(blocks) and blocks[int(position)]==source:continue
  retire.append({'path':str(p.relative_to(root)),'record_index':index,'label':label,'source_sha256':record.get('source_sha256'),'source':source,'action':'rebind' if source in blocks else 'retire','new_indexes':[i for i,x in enumerate(blocks) if x==source]})
payload={'schema':1,'source_commit':'8a20fded961a93cf0b1368c39cbf84fd8db02cc8','source_sha256':hashlib.sha256((root/'spacr/qt/shortcuts.py').read_bytes()).hexdigest(),'collector_sha256':hashlib.sha256((root/'tools/build_documentation_i18n.py').read_bytes()).hexdigest(),'intended_changed_docstrings':changed,'rendered_symbols':visible,'not_rendered':[s for s in changed if s not in visible],'blocks':rows,'superseded_rows':retire}
Path('/mnt/wd4tb/scratch/n678-shortcut-api-inputs-20261015/collected.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print(f'{len(visible)} rendered symbols;{len(rows)} changed/index-moved blocks;{len(retire)} superseded existing rows')
for r in rows:print(r['label']+'\n'+r['source']+'\n')
