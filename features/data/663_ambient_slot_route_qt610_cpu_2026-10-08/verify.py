import argparse,gzip,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--git',action='store_true');p.add_argument('--frozen',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parent;repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=root,text=True).strip());relative=root.relative_to(repo)
def read(path):
    local=root/path
    if local.exists():return local.read_bytes()
    if args.git:return subprocess.check_output(['git','show','HEAD:'+str(relative/path)],cwd=repo)
    raise FileNotFoundError(local)
m=json.loads(read('manifest.json'));r=json.loads(read('receipt.json'));probe=json.loads(read('probe-receipt.json'))
for row in m['payloads']:
    data=read(row['path']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],row['path']
for path,digest in r['current_bindings'].items():
    assert hashlib.sha256(gzip.decompress(read('source/'+path+'.gz'))).hexdigest()==digest,path
    if not args.frozen:
        data=subprocess.check_output(['git','show','HEAD:'+path],cwd=repo) if args.git else (repo/path).read_bytes();assert hashlib.sha256(data).hexdigest()==digest,path
assert probe['Qt']=='6.10.0' and not probe['exceptions'] and not probe['capture_limits']['qt_capture_truncated']
assert probe['alive_producers_after']==0 and probe['remaining_ambient_widgets']==0 and probe['window_native_destroyed']
assert all(w['tick_index']==33 and w['meta_class']=='AmbientWidget' for row in probe['records'] for w in row['ambient'])
print('Verified',len(m['payloads']),'payloads,',len(r['current_bindings']),'frozen source bindings;', 'current bindings checked' if not args.frozen else 'historical checkpoint only')
