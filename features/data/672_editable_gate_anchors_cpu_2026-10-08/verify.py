import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--git',action='store_true');args=p.parse_args();root=Path(__file__).resolve().parent;m=json.loads((root/'manifest.json').read_text());r=json.loads((root/'receipt.json').read_text())
for item in m['payloads']:
    data=(root/item['path']).read_bytes();assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],item['path']
for path,digest in r['bindings'].items():
    data=subprocess.check_output(['git','show','HEAD:'+path],cwd=root) if args.git else (Path.cwd()/path).read_bytes();assert hashlib.sha256(data).hexdigest()==digest,path
assert r['painter_ast']['before']==r['painter_ast']['after']
print(f"Verified {len(m['payloads'])} payloads and {len(r['bindings'])} current bindings; painter AST unchanged")
