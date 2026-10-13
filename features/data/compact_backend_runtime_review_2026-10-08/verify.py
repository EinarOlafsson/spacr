from pathlib import Path
import gzip,hashlib,json,subprocess,sys
root=Path(sys.argv[1]);packet=root/sys.argv[2]
receipt=json.loads((packet/'receipt.json').read_text())
sha=lambda data:hashlib.sha256(data).hexdigest()
for row in receipt['payloads']:
 raw=(packet/row['path']).read_bytes()
 assert sha(raw)==row['gzip_sha256'] and len(raw)==row['gzip_bytes'],row['path']
 plain=gzip.decompress(raw)
 assert sha(plain)==row['raw_sha256'] and len(plain)==row['raw_bytes'],row['path']
bindings=receipt.get('frozen_all_bindings_sha256',receipt.get('artifact_bindings_sha256',{}))
for path,digest in bindings.items():assert sha((root/path).read_bytes())==digest,path
for path,digest in receipt['frozen_app_python_sha256'].items():
 assert sha((root/path).read_bytes())==digest,path
 assert sha(subprocess.check_output(['git','-C',str(root),'show',receipt['source_commit']+':'+path]))==digest,path
print(json.dumps({'verified_payloads':len(receipt['payloads']),'verified_bindings':len(bindings),'verified_git_application_sources':len(receipt['frozen_app_python_sha256']),'receipt_sha256':sha((packet/'receipt.json').read_bytes())}))
