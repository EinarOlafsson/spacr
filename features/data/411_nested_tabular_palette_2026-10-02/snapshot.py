import ast,hashlib,json,subprocess,sys
from pathlib import Path
root=Path.cwd();p=Path(__file__).parent;mode=sys.argv[1];sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
paths=[]
for directory in ('spacr','tools','docs/source'):
 for suffix in ('*.py','*.rst'):
  paths.extend(f for f in (root/directory).rglob(suffix) if not any(x in f.parts for x in ('api','__pycache__')))
paths+=list((root/'docs/source/_static/i18n/api').glob('*.json'))
current={str(f.relative_to(root)):sha(f) for f in sorted(set(paths))}
if mode=='before':
 (p/'build-source-snapshot.json').write_text(json.dumps({'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'files':current},indent=2)+'\n')
else:
 prior=json.loads((p/'build-source-snapshot.json').read_text())['files'];assert current==prior
 (p/'source-stability.json').write_text(json.dumps({'unchanged_source_files':len(current),'source_snapshot_sha256':sha(p/'build-source-snapshot.json')},indent=2)+'\n')
# Help consumes the same new symbols; all other generated relationships stay exact.
path='spacr/qt/help_api_index.py'
def values(text):return {n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(text).body if isinstance(n,ast.Assign)}
a=values(subprocess.check_output(['git','show','HEAD:'+path],text=True));b=values((root/path).read_text());old=dict(a['API_ENTRIES']);new=dict(b['API_ENTRIES']);keys=set(json.loads((p/'source-delta.json').read_text())['added'])
assert set(new)-set(old)==keys and all(new[k]==v for k,v in old.items())
assert a['SETTING_CONSUMERS']==b['SETTING_CONSUMERS'] and a['PREFERENCE_ENTRIES']==b['PREFERENCE_ENTRIES']
(p/'help-delta.json').write_text(json.dumps({'added':sorted(keys),'changed_existing':[],'unchanged_setting_consumers':True,'unchanged_preferences':True,'sha256':sha(root/path)},indent=2)+'\n')
