import concurrent.futures,datetime,hashlib,json,pathlib,re,subprocess,sys,urllib.request
source_revision,revision,output=sys.argv[1:]
assert re.fullmatch('[0-9a-f]{40}',source_revision) and re.fullmatch('[0-9a-f]{40}',revision)
root=pathlib.Path('/mnt/wd4tb/spacr-worktrees/codex-home-b-site-20261014')
files=['.github/workflows/website-audit.yml','.github/scripts/website-audit.mjs','build.json','index.html','projects/spacr/index.html','projects/parascape/index.html','projects/timepose/index.html','projects/cellect/index.html','assets/js/fx.js']
def fetch(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'Cache-Control':'no-cache'}),timeout=20) as r:
  data=r.read(2_000_001)
  assert r.status==200 and len(data)<=2_000_000 and r.url==url
  return data

def check(name):
 raw=f'https://raw.githubusercontent.com/EinarOlafsson/EinarOlafsson.github.io/{revision}/{name}'
 expected=fetch(raw);live=None
 if name.startswith('.github/'):
  actual=subprocess.check_output(['git','show',source_revision+':src/'+name],cwd=root)
 else:
  route='/' if name=='index.html' else '/'+name.removesuffix('index.html')
  live='https://einarolafsson.github.io'+route+'?audit_identity='+revision
  actual=fetch(live)
 assert actual==expected,name
 if name=='assets/js/fx.js': assert actual==subprocess.check_output(['git','show',source_revision+':src/'+name],cwd=root)
 return {'path':name,'immutable_url':raw,'live_url':live,'status':200,'bytes':len(actual),'sha256':hashlib.sha256(actual).hexdigest(),'exact':True}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(check,files))
receipt={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_revision':source_revision,'public_revision':revision,'rows':rows,'audit_dispatched':False}
pathlib.Path(output).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'pass':True,'files':len(rows),'public_revision':revision,'receipt':output}))
