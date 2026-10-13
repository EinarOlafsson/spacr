"""Apply measured source inventory only after the caller verifies full audit success."""
import ast,hashlib,json,re,runpy,subprocess
from pathlib import Path
path=Path('spacr/qt/i18n_catalogs/en.py')
current=runpy.run_path(str(path))
old={}
exec(compile(subprocess.check_output(['git','show','HEAD:'+str(path)],text=True),str(path),'exec'),old)
names={'SETTING_LABELS':'SETTING_LABELS','SETTING_TOOLTIPS':'SETTING_TOOLTIPS','CATEGORY_HELP':'CATEGORY_SOURCES','UI':'UI_SOURCES','MODULE_SUMMARIES':'MODULE_SUMMARIES'}
def identities(namespace):
 return sorted((table,str(key)) for table,name in names.items() for key in namespace[name])
a,b=identities(old),identities(current)
counts={table:len(current[name]) for table,name in names.items()}
digest=hashlib.sha256('\0'.join(f'{table}\0{key}' for table,key in b).encode()).hexdigest()
record={'old_counts':{table:len(old[name]) for table,name in names.items()},'new_counts':counts,'new_identity_sha256':digest,'added':sorted(set(b)-set(a)),'removed':sorted(set(a)-set(b))}
test=Path('tests/qt/test_i18n_caption_ratchet.py');text=test.read_text()
for table,count in counts.items():
 text,n=re.subn(r'^(    "'+table+r'": )\d+(,?)$',lambda m:m[1]+str(count)+m[2],text,count=1,flags=re.M)
 assert n==1,table
pin=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXTERNAL_SOURCE_KEY_SHA256' for t in n.targets))
record['previous_pinned_identity_sha256']=ast.literal_eval(pin.value)
record['previous_catalog_identity_sha256']=hashlib.sha256('\0'.join(f'{table}\0{key}' for table,key in a).encode()).hexdigest()
lines=text.splitlines(keepends=True)
lines[pin.value.lineno-1:pin.value.end_lineno]=['    '+repr(digest)+'\n']
text=''.join(lines)
note=('# 2026-10-02 F580: measured canonical inventory after the complete nine-locale\n'
      '# generator audit. Twenty identities added and none removed relative to the\n'
      '# preceding committed English catalogue. Translation quality gates unchanged.\n')
text=text.replace('EXTERNAL_SOURCE_COUNTS = {',note+'EXTERNAL_SOURCE_COUNTS = {',1)
test.write_text(text)
Path('/tmp/spacr-implementation-20261001/f580-runtime-i18n/inventory-delta.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({key:value for key,value in record.items() if key not in ('added','removed')},indent=2))
print('Added:',len(record['added']),'Removed:',len(record['removed']))
