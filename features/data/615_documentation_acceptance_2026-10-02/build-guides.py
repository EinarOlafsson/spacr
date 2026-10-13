import sys,json,time,hashlib
from pathlib import Path
root=Path('/tmp/spacr-implementation-20261001/suggest-capture');p=Path(__file__).parent
sys.path.insert(0,str(root/'tools'));import build_guide_i18n as g
results={}
for lang in g.LANGUAGES:
 start=time.monotonic();output=p/'full/html'/lang
 status=g.build(lang,output,p/'full/html/objects.inv',p/'guide-doctrees'/lang,warnings_are_errors=True)
 results[lang]={'exit_code':status,'elapsed_seconds':round(time.monotonic()-start,2),'html_pages':len(list(output.rglob('*.html'))),'measure_page_sha256':hashlib.sha256((output/'measure_live.html').read_bytes()).hexdigest() if(output/'measure_live.html').exists()else None}
 (p/'guide-builds.json').write_text(json.dumps(results,indent=2)+'\n');print(lang,results[lang],flush=True)
 assert status==0,(lang,status)
