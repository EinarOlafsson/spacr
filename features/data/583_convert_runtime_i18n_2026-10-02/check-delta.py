import json,subprocess,hashlib
from pathlib import Path
p=Path(__file__).parent
sources=json.loads((p/'sources.json').read_text());added=set(json.loads((p/'final-source-validation.json').read_text())['added_ui']);removed=set();out={}
for lang in ['en','sv','de','es','fr','pt','zh_CN','hi','ko','is']:
 path=Path('spacr/qt/i18n_catalogs')/f'{lang}.py'
 old={};new={}
 exec(subprocess.check_output(['git','show','HEAD:'+str(path)],text=True),old)
 exec(path.read_text(),new)
 tables=[k for k,v in old.items() if not k.startswith('__') and isinstance(v,(dict,set,frozenset))]
 delta={}
 for table in tables:
  before,after=old[table],new[table]
  if before==after: continue
  deleted=set(before)-set(after);inserted=set(after)-set(before)
  changed={k for k in set(before)&set(after) if isinstance(before,dict) and before[k]!=after[k]}
  allowed_added={( 'UI',s) for s in added} if table=='SOURCE_HASHES' else added
  allowed_removed={( 'UI',s) for s in removed} if table=='SOURCE_HASHES' else removed
  assert not changed,(lang,table,'changed existing values',changed)
  assert inserted==allowed_added,(lang,table,'unexpected additions',inserted-allowed_added,allowed_added-inserted)
  assert deleted==allowed_removed,(lang,table,'unexpected removals',deleted-allowed_removed,allowed_removed-deleted)
  delta[table]={'added':len(inserted),'removed':len(deleted),'changed_existing':len(changed)}
 if lang!='en':
  targets=json.loads((p/f'{lang}-final.json').read_text())
  assert all(new['UI'][s]==v for s,v in targets.items()); assert new['UI']['plate1=BC001; plate2=BC002']=='plate1=BC001; plate2=BC002'
 out[lang]={'delta':delta,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(p/'catalog-diff-proof.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
