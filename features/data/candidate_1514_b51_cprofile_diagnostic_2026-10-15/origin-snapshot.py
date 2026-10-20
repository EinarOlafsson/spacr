from pathlib import Path
import hashlib,json,sys
ROOT=Path('/mnt/wd4tb/scratch/candidate-1514-b51fb194bb-20261015');TASK=ROOT/'installed-cprofile-diagnostic';SITE=ROOT/'smoke-preflight/venv/lib/python3.12/site-packages'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while chunk:=f.read(1048576):h.update(chunk)
 return h.hexdigest()
pins={str(p.relative_to(SITE)):sha(p) for p in sorted((SITE/'spacr').rglob('*')) if p.is_file() and '__pycache__' not in p.parts and not p.is_symlink()}
symlinks=[str(p.relative_to(SITE)) for p in (SITE/'spacr').rglob('*') if p.is_symlink()]
md=SITE/'spacr-1.5.1.4.dist-info';metadata={p.name:sha(p) for p in [md/'METADATA',md/'direct_url.json',md/'RECORD',md/'entry_points.txt']}
origin=json.loads((md/'direct_url.json').read_text());assert origin['archive_info']['hashes']['sha256']=='b552e71a09f5448ec1e7f53a8c603ed45d3c3fdda39e5dc6db58a15f5b1936db'
artifacts={p.name:sha(p) for p in sorted((ROOT/'artifacts').iterdir()) if p.name.endswith(('.whl','.tar.gz'))}
assert artifacts=={'spacr-1.5.1.4-py3-none-any.whl':'0be5f07cf06de0c5d0273267b9da2036378a8f3ad081596b556f71270fdf84ac','spacr-1.5.1.4.tar.gz':'b552e71a09f5448ec1e7f53a8c603ed45d3c3fdda39e5dc6db58a15f5b1936db'}
r={'source':'b51fb194bb183990ab25c981d70e3030063bf9ac','site_packages':str(SITE),'installed_package':str(SITE/'spacr/__init__.py'),'driver_sha256':sha(TASK/'spacr_startup_benchmark.py'),'direct_url':origin,'distribution_metadata':metadata,'artifacts':artifacts,'installed_regular_nonpycache_files':pins,'installed_symlinks':symlinks}
(TASK/(sys.argv[1]+'.json')).write_text(json.dumps(r,indent=2)+'\n')
if sys.argv[1]=='origin-after':
 before=json.loads((TASK/'origin-before.json').read_text());assert r==before,'Installed origin/content or artifact source changed during diagnostic'
print(sys.argv[1],len(pins),'regular installed files;',len(symlinks),'symlinks')
