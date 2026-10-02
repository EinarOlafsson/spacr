import json,hashlib
from html.parser import HTMLParser
from pathlib import Path
from bs4 import BeautifulSoup
p=Path(__file__).parent;root=Path('/tmp/spacr-implementation-20261001/suggest-capture');out=p/'full/html';sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
proof={}
for lang in ('sv','de','es','pt','fr','is','zh_CN','ko','hi'):
 page=out/lang/'measure_live.html';s=BeautifulSoup(page.read_text(),'html.parser');text=s.get_text(' ',strip=True)
 work=json.loads((p/f'{lang}-worklist.json').read_text());targets=json.loads((p/f'{lang}-draft.json').read_text())
 headings=[x.get_text(' ',strip=True).replace('¶','').strip() for x in s.select('h1,h2,h3')]
 assert targets['0'] in headings and targets['3'] in headings,(lang,headings)
 for row in work:
  for label in row['ui'].values():assert label in text,(lang,label)
 for code in ('unmix=True','unmix_controls','provenance.json','intensity_calibration','merged'):assert code in text,(lang,code)
 assert not s.select('.untranslated'),(lang,'untranslated fallback')
 proof[lang]={'page_sha256':sha(page),'all_new_headings_present':True,'all_literal_and_ui_bindings_present':True,'untranslated_elements':0,'strict_build_exit_code':0}
api={}
for f in (root/'docs/source/_static/i18n/api').glob('*.json'):
 target=out/'_static/i18n/api'/f.name
 assert target.exists() and target.read_bytes()==f.read_bytes(),f
 api[f.name]=sha(f)
class Scripts(HTMLParser):
 def __init__(self):super().__init__();self.api=[]
 def handle_starttag(self,tag,attrs):
  attrs=dict(attrs)
  if tag=='script' and 'api_i18n.js' in attrs.get('src',''):self.api.append(attrs)
hasher=hashlib.sha256()
for f in sorted((root/'docs/source/_static/i18n/api').glob('*.json')):
 hasher.update(f.name.encode());hasher.update(f.read_bytes())
version=hasher.hexdigest()[:16];api_pages=list((out/'api').rglob('*.html'));assert len(api_pages)==635
for page in api_pages:
 parser=Scripts();parser.feed(page.read_text());assert len(parser.api)==1,page
 assert parser.api[0]['data-api-language']=='all',page
 assert parser.api[0]['data-api-catalog-version']==version,page
 assert set(parser.api[0]['data-guide-languages'].split())==set(proof),page
assert (out/'_static/api_i18n.js').read_bytes()==(root/'docs/source/_static/api_i18n.js').read_bytes()
final=json.loads((p/'final-source-snapshot.json').read_text());assert all(sha(root/f)==h for f,h in final['files'].items())
manifest={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*.html'))}
(p/'html-hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
(p/'rendered-proof.json').write_text(json.dumps({'local_only':True,'api_publication_mode':'all','api_pages_with_exact_mode_and_catalog_version':len(api_pages),'api_catalog_version':version,'languages':proof,'html_pages_total':len(manifest),'english_html_pages':len([f for f in manifest if f.split('/')[0] not in proof]),'api_catalogs_exact':api,'final_source_files_unchanged':len(final['files'])},indent=2)+'\n');print('Rendered guide labels/literals, no fallback, exact static API catalogs and final source fingerprints PASS',len(manifest))
