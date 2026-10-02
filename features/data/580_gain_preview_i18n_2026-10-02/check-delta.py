import hashlib,json,subprocess
from pathlib import Path
p=Path(__file__).parent;sources=set(json.loads((p/'sources.json').read_text()));out={}
for lang in ['en','sv','de','es','zh_CN','pt','hi','ko','is','fr']:
 path=Path('spacr/qt/i18n_catalogs')/f'{lang}.py';before={};after={}
 exec(subprocess.check_output(['git','show','HEAD:'+str(path)],text=True),before);exec(path.read_text(),after)
 delta={}
 for table,old in before.items():
  if table.startswith('__') or not isinstance(old,(dict,set,frozenset)):continue
  new=after[table]
  if old==new:continue
  added=set(new)-set(old);removed=set(old)-set(new)
  changed={k for k in set(old)&set(new) if isinstance(old,dict) and old[k]!=new[k]}
  expected={('UI',s) for s in sources} if table=='SOURCE_HASHES' else sources
  assert added==expected,(lang,table,added);assert not removed and not changed,(lang,table,removed,changed)
  delta[table]={'added':len(added),'removed':0,'changed_existing':0}
 assert delta,lang
 if lang!='en':
  targets=json.loads((p/f'{lang}-final.json').read_text())
  for s,t in targets.items():assert after['UI'][s]==t,(lang,s)
 out[lang]={'delta':delta,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(p/'catalog-diff-proof.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS exact20source additions,180targets,zero old changes across10catalogs')
