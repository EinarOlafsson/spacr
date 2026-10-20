from pathlib import Path
import ast, collections, hashlib, json, os, stat, subprocess, tarfile, time, traceback, zipfile
ROOT=Path('/mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015')
OUT=Path('/mnt/wd4tb/scratch/candidate-1514-b51fb194bb-20261015/ci-artifact-verification')
DIST=OUT.parent/'artifacts'
SOURCE='b51fb194bb183990ab25c981d70e3030063bf9ac'
EXPECTED={'spacr-1.5.1.4-py3-none-any.whl':'0be5f07cf06de0c5d0273267b9da2036378a8f3ad081596b556f71270fdf84ac','spacr-1.5.1.4.tar.gz':'b552e71a09f5448ec1e7f53a8c603ed45d3c3fdda39e5dc6db58a15f5b1936db'}
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def digest(stream):
 h=hashlib.sha256()
 while chunk:=stream.read(1048576):h.update(chunk)
 return h.hexdigest()
def file_digest(path):
 with path.open('rb') as f:return digest(f)
def hashes():return {name:file_digest(DIST/name) for name in EXPECTED}
report={'source_pin':SOURCE,'source_tree':str(ROOT),'head_at_start':git('rev-parse','HEAD').decode().strip(),'started_epoch':time.time(),'scope':'Exact two normal archive checks plus separately bounded required regular-runtime Git-blob byte readback; no installation or runtime imports.','artifact_paths':{name:str(DIST/name) for name in EXPECTED},'checks':[],'excluded_parity_scope':'Not all repository/source bytes, all archive extras, installed execution, entry-point functionality, native/scientific performance, or current hosted CI.'}
try:
 os.chdir(ROOT)
 report['artifacts_before']=hashes();assert report['artifacts_before']==EXPECTED
 assert git('diff','--name-only',SOURCE,'HEAD','--','spacr','setup.py','pyproject.toml','MANIFEST.in').strip()==b''
 source=git('show',SOURCE+':.github/workflows/compat-matrix.yml').decode()
 report['workflow_sha256']=hashlib.sha256(source.encode()).hexdigest()
 lines=source.splitlines(keepends=True)
 def body(name):
  start=next(i for i,l in enumerate(lines) if l.strip()=='- name: '+name)
  begin=next(i for i in range(start,len(lines)) if lines[i].strip()=="python - <<'PY'")+1
  end=next(i for i in range(begin,len(lines)) if lines[i].strip()=='PY')
  return ''.join(l[10:] for l in lines[begin:end])
 normal=body('Audit built artifact inventories')
 assert normal.count('Path("dist")')==2
 adapted=normal.replace('Path("dist")','Path('+repr(str(DIST))+')')
 assert adapted.replace('Path('+repr(str(DIST))+')','Path("dist")')==normal
 (OUT/'inventory-original.py').write_text(normal);(OUT/'inventory-dist-adapted.py').write_text(adapted)
 scope={};exec(compile(adapted,'compat-matrix:Audit built artifact inventories','exec'),scope)
 report['checks'].append({'name':'normal required/forbidden archive inventories','status':'PASS','normal_sha256':hashlib.sha256(normal.encode()).hexdigest(),'adapted_sha256':hashlib.sha256(adapted.encode()).hexdigest(),'adaptation':'Only two dist directory literals replaced with exact artifact directory.'})
 metadata=body('Assert the built metadata carries Requires-Python')
 assert metadata.count('"dist/*.whl"')==1
 md_adapted=metadata.replace('"dist/*.whl"',repr(str(DIST/'*.whl')))
 assert md_adapted.replace(repr(str(DIST/'*.whl')),'"dist/*.whl"')==metadata
 (OUT/'metadata-original.py').write_text(metadata);(OUT/'metadata-dist-adapted.py').write_text(md_adapted)
 exec(compile(md_adapted,'compat-matrix:Assert the built metadata carries Requires-Python','exec'),{})
 report['checks'].append({'name':'normal Requires-Python and OS classifier archive metadata','status':'PASS','normal_sha256':hashlib.sha256(metadata.encode()).hexdigest(),'adapted_sha256':hashlib.sha256(md_adapted.encode()).hexdigest(),'adaptation':'Only dist wheel glob replaced with exact artifact glob.'})
 tracked={}
 for row in git('ls-tree','-r','-z',SOURCE,'spacr').split(b'\0'):
  if not row:continue
  meta,path=row.split(b'\t',1);mode,kind,oid=meta.decode().split();tracked[path.decode()]={'mode':mode,'kind':kind,'object':oid}
 assert set(scope['tracked'])==set(tracked)
 required=scope['required'];regular=sorted(p for p in required if tracked[p]['mode'] in ('100644','100755'))
 symlinks=sorted(p for p in required if tracked[p]['mode']=='120000')
 unknown=sorted(p for p in required if p not in regular and p not in symlinks);assert not unknown
 report['required_member_count']=len(required);report['regular_runtime_count']=len(regular);report['required_symlink_paths_excluded_from_regular_parity']=symlinks
 report['tracked_paths_outside_required_scope']=len(set(tracked)-required)
 comparisons=[]
 proc=subprocess.Popen(['git','-C',str(ROOT),'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
 try:
  with zipfile.ZipFile(DIST/'spacr-1.5.1.4-py3-none-any.whl') as wheel,tarfile.open(DIST/'spacr-1.5.1.4.tar.gz') as sdist:
   w=collections.defaultdict(list);t=collections.defaultdict(list)
   for member in wheel.infolist():w[member.filename].append(member)
   for member in sdist.getmembers():
    if '/' in member.name:t[member.name.split('/',1)[1]].append(member)
   for path in regular:
    row=tracked[path];proc.stdin.write((row['object']+'\n').encode());proc.stdin.flush()
    header=proc.stdout.readline().decode().split();assert header[0]==row['object'] and header[1]=='blob'
    size=int(header[2]);remaining=size;h=hashlib.sha256()
    while remaining:
     chunk=proc.stdout.read(min(remaining,1048576));assert chunk;h.update(chunk);remaining-=len(chunk)
    assert proc.stdout.read(1)==b'\n';expected=h.hexdigest()
    assert len(w[path])==1 and len(t[path])==1,(path,'duplicate/missing member')
    wm=w[path][0];tm=t[path][0]
    assert not wm.is_dir() and not stat.S_ISLNK(wm.external_attr>>16),(path,'wheel symlink/directory')
    assert tm.isfile() and not tm.issym() and not tm.islnk(),(path,'sdist is not regular')
    with wheel.open(wm) as f:wd=digest(f)
    with sdist.extractfile(tm) as f:td=digest(f)
    assert wd==td==expected,(path,'runtime bytes differ from frozen Git blob')
    comparisons.append({'path':path,'git_blob':row['object'],'bytes':size,'sha256':expected,'wheel_matches':True,'sdist_matches':True})
 finally:
  proc.stdin.close();assert proc.wait(timeout=10)==0
 (OUT/'required-regular-runtime-digests.json').write_text(json.dumps(comparisons,indent=2)+'\n')
 report['checks'].append({'name':'required tracked REGULAR runtime bytes against frozen Git blobs, both archives','status':'PASS','files':len(comparisons),'bytes_per_format':sum(r['bytes'] for r in comparisons),'digest_receipt_sha256':file_digest(OUT/'required-regular-runtime-digests.json')})
 report['status']='PASS'
except BaseException:
 report['status']='FAIL';report['error']=traceback.format_exc();traceback.print_exc()
finally:
 report['artifacts_after']=hashes();report['artifacts_unchanged']=report['artifacts_after']==report.get('artifacts_before')==EXPECTED
 report['ended_epoch']=time.time();report['duration_seconds']=report['ended_epoch']-report['started_epoch']
 if not report['artifacts_unchanged']:report['status']='FAIL'
 (OUT/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'status':report['status'],'checks':report['checks'],'artifacts_unchanged':report['artifacts_unchanged'],'seconds':report['duration_seconds']},indent=2))
raise SystemExit(0 if report['status']=='PASS' else 1)
