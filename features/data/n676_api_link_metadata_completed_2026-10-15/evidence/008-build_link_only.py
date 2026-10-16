from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'tools/tutorials'),str(ROOT/'tools/tutorials/authoring/tools')]
sys.meta_path[:]=[f for f in sys.meta_path if '__editable__' not in (getattr(f,'__module__','') or type(f).__module__)]
import os,subprocess,hashlib,json
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
SOURCE='5430c0ceabbd9cf99a4f40ea3030eefa89c2bd87'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==SOURCE
subprocess.run(['git','diff','--quiet','8ab0b4f421baf9e6318c9f9f0d08ccd73241113b',SOURCE,'--','spacr','tools/tutorials/lessons/06_api.json'],cwd=ROOT,check=True)
from apply_translation_review import translated_lesson
english=json.loads((ROOT/'tools/tutorials/lessons/06_api.json').read_text())
for p in sorted((ROOT/'tools/tutorials/lessons/reviews').glob('06_api.*.json')):
 translated_lesson(json.loads(p.read_text()),english,{})
BASE=Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-05-selective-20261015/release-candidate-append-ln3oto4l')
assert hashlib.sha256((BASE/'release-manifest.json').read_bytes()).hexdigest()=='ced803ebdd5e93a6ada3fead48a3c6595afd77ef05d104dc0bd43e87a210bd2a'
STAGE=Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-06-links-20261015')
STAGE.mkdir(exist_ok=False)
from build_appended_candidate import build
candidate=build(STAGE,BASE,[],link_ids=['06_api'])
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==SOURCE
print('LINK_ONLY_CANDIDATE',candidate,flush=True)
