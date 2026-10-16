#!/usr/bin/env python3
"""06 link-only downstream source guard; no capture or current audio claim."""
from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'tools/tutorials'),str(ROOT/'tools/tutorials/authoring/tools')]
sys.meta_path[:]=[f for f in sys.meta_path if '__editable__' not in (getattr(f,'__module__','') or type(f).__module__)]
import os,subprocess,hashlib,json,runpy
K='5430c0ceabbd9cf99a4f40ea3030eefa89c2bd87';P='8ab0b4f421baf9e6318c9f9f0d08ccd73241113b'
OLD=b'a48e16fce2492866cb7894a42976f612efa1239b793128a8ccfce2ef2d62fbc1';NEW=b'5182c3efe47b1f6fa83911262a5b2b4eea6d619af5ea77044974276de96774f1'
LANGS=['da','de','es','fr','hi','is','it','ja','ko','nb','pt-BR','sv','zh-CN']
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def source_guard():
 assert Path.cwd().resolve()==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert os.environ['SOURCE_COMMIT']==K==git('rev-parse','HEAD').decode().strip()
 assert git('rev-parse',K+'^').decode().strip()==P
 paths=sorted('tools/tutorials/lessons/reviews/06_api.'+lang+'.json' for lang in LANGS)
 assert git('diff','--name-only',P,K).decode().splitlines()==paths
 for rel in paths:
  before=git('show',P+':'+rel);after=git('show',K+':'+rel)
  assert before.count(OLD)==1 and after==before.replace(OLD,NEW,1) and (ROOT/rel).read_bytes()==after
 for rel in ['spacr','tools/tutorials/lessons/06_api.json']:
  subprocess.run(['git','diff','--quiet',P,K,'--',rel],cwd=ROOT,check=True)
 dirty=git('status','--porcelain','--','spacr','tools/tutorials','tests',':(exclude)tools/tutorials/release_candidate').strip()
 assert not dirty, 'Relevant source dirty; generated docs/checkpoint writes alone allowed'
 e=ROOT/'tools/tutorials/lessons/06_api.json'
 assert e.read_bytes()==git('show',K+':'+e.relative_to(ROOT).as_posix())
 assert hashlib.sha256(json.dumps(json.loads(e.read_text()),sort_keys=True,ensure_ascii=False).encode()).hexdigest()==NEW.decode()
 assert Path(os.environ['STAGE']).resolve()==Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-06-links-20261015')
 assert Path(os.environ['SPACR_TUTORIAL_WORKSPACE']).resolve()==Path(os.environ['STAGE']).resolve()
 b=Path(os.environ['BASELINE_CANDIDATE']);assert str(b)=='/media/carruthers/mnt3/claude/n676_stages/host/stage-05-selective-20261015/release-candidate-append-ln3oto4l'
 assert hashlib.sha256((b/'release-manifest.json').read_bytes()).hexdigest()=='ced803ebdd5e93a6ada3fead48a3c6595afd77ef05d104dc0bd43e87a210bd2a'
source_guard()
script=Path(sys.argv[1]).resolve()
allowed={ROOT/'tools/tutorials'/n for n in ['verify_release_candidate.py','check_placeholder_mutations.py','checkpoint_release_candidate.py','publish_release_candidate.py']}
assert script in allowed, 'Only explicit06 publication dispatch tools'
if script.name=='publish_release_candidate.py':assert sys.argv[2] in {'resume-receipt','pages','record'}, 'No06 upload/new branch'
sys.argv=[str(script),*sys.argv[2:]]
try:runpy.run_path(str(script),run_name='__main__')
finally:source_guard()
