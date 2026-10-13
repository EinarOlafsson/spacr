"""Verify the actual strict build and its real language-selector bindings."""
import hashlib,importlib.util,json,threading,zlib
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from bs4 import BeautifulSoup
stage=Path(__file__).parent;root=Path.cwd();out=stage/'full/html'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
draft=json.loads((stage/'drafts.json').read_text());sources=draft['sources'];languages=list(draft['translations'])
keys=[s['key'] for s in sources];modules=['spacr.tabular','spacr.qt.command_palette']
inv=zlib.decompress((out/'objects.inv').read_bytes().split(b'\n',4)[4]).decode().splitlines()
records={};catalogs={};catalog_root=root/'docs/source/_static/i18n/api'
version_hash=hashlib.sha256()
for f in sorted(catalog_root.glob('*.json')):
 assert (out/'_static/i18n/api'/f.name).read_bytes()==f.read_bytes()
 version_hash.update(f.name.encode());version_hash.update(f.read_bytes());catalogs[f.stem]=sha(f)
version=version_hash.hexdigest()[:16]
for key,module in zip(keys,modules):
 path=out/'api'/Path(*module.split('.'))/'index.html';html=path.read_text();page=BeautifulSoup(html,'html.parser')
 assert len(page.find_all(id=key))==1,key
 assert page.find(id=key.rsplit('.',1)[0]) is None,key
 matches=[line for line in inv if line.startswith(key+' ')];assert len(matches)==1 and matches[0].split()[1]=='py:function'
 scripts=[n for n in page.find_all('script') if 'api_i18n.js' in n.get('src','')];assert len(scripts)==1
 assert scripts[0]['data-api-language']=='all' and scripts[0]['data-api-catalog-version']==version
 records[key]={'path':str(path.relative_to(out)),'sha256':sha(path),'anchor_count':1,'hidden_parent_absent':True,'inventory_line':matches[0]}
# Serve the unmodified build except an in-memory test observer; static assets and
# locale JSON are the exact output of the strict build, never synthetic catalogs.
spec=importlib.util.spec_from_file_location('frontend',root/'tests/test_api_i18n_frontend.py');frontend=importlib.util.module_from_spec(spec);spec.loader.exec_module(frontend)
current={}
class Handler(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  path=self.path.split('?',1)[0]
  if path==current.get('path'):
   payload=current['html'].encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
  else:super().do_GET()
server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(out)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
try:
 for i,key in enumerate(keys):
  expected={lang:draft['translations'][lang][i].replace('``','') for lang in languages}
  harness='''<script>window.addEventListener('load',()=>{const key=KEY, expected=EXPECTED, langs=Object.keys(expected);const seen=[];let n=0,polls=0;const timer=setInterval(()=>{const select=document.querySelector('.spacr-api-language select');if(!select){if(++polls>250){document.body.dataset.sliceResult='missing-selector';clearInterval(timer);}return;}if(n===langs.length){document.body.dataset.sliceResult='pass';document.body.dataset.sliceLanguages=seen.join(',');clearInterval(timer);return;}const lang=langs[n];if(select.value!==lang){select.value=lang;select.dispatchEvent(new Event('change'));}const signature=document.getElementById(key);const panel=signature?.parentElement.querySelector('.spacr-api-translation');const norm=x=>x.replace(/\\s+/g,' ').replace(/pending \\./g,'pending.').trim();if(panel&&panel.lang===lang.replace('_','-')&&norm(panel.textContent).includes(norm(expected[lang]))){seen.push(lang);n++;polls=0;}else if(++polls>250){document.body.dataset.sliceResult='failure:'+lang;clearInterval(timer);}},50);});</script>'''.replace('KEY',json.dumps(key)).replace('EXPECTED',json.dumps(expected,ensure_ascii=False))
  relative=records[key]['path'];current.update(path='/'+relative,html=(out/relative).read_text().replace('</body>',harness+'</body>'))
  dom=frontend._dump_dom(f'http://127.0.0.1:{server.server_port}/{relative}',budget=40000,wall_timeout=120)
  page=BeautifulSoup(dom,'html.parser');assert page.body.get('data-slice-result')=='pass',(key,page.body.get('data-slice-result'))
  assert page.body.get('data-slice-languages').split(',')==languages
  records[key]['browser_rendered_languages']=languages
finally:server.shutdown();server.server_close();thread.join()
(stage/'rendered-proof.json').write_text(json.dumps({'local_build_only':True,'api_catalog_version':version,'exact_copied_catalog_hashes':catalogs,'helpers':records,'html_pages':len(list(out.rglob('*.html')))},indent=2)+'\n')
print('PASS: two real module pages, hidden parents, inventory anchors, 18 actual browser translations')
