#!/usr/bin/env python3
"""Execute root-selected selective02 tools; no whole-capture/science restamping."""
from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'tools/tutorials'),str(ROOT/'tools/tutorials/authoring/tools')]
sys.meta_path[:]=[f for f in sys.meta_path if '__editable__' not in (getattr(f,'__module__','') or type(f).__module__)]
import hashlib,json,os,subprocess,runpy
W=Path(__file__).resolve().parent
B=Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-23-r60-20261015/release-candidate-append-7ew0avek')
S=Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-02-selective-r2-20261015')
SOURCE_BEFORE='972ac89c91db8b3842cd75bb30991b8b1af35898'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def guard():
 assert Path.cwd().resolve()==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert git('rev-parse','HEAD')==os.environ['SOURCE_COMMIT']
 assert Path(os.environ['STAGE']).resolve()==S and Path(os.environ['BASELINE_CANDIDATE']).resolve()==B
 subprocess.run(['git','diff','--quiet',SOURCE_BEFORE,os.environ['SOURCE_COMMIT'],'--','spacr','tools/tutorials/lessons/02_install_spacr.json','tools/tutorials/lessons/reviews/02_install_spacr.*.json'],cwd=ROOT,check=True)
 assert not git('status','--porcelain','--','spacr','tools/tutorials','tests',':(exclude)tools/tutorials/release_candidate')
 assert sha(ROOT/'tools/tutorials/lessons/02_install_spacr.json')=='21c4a3dacc742874bf86dafe61ded87f22cfd4df8ea171a56220b80570a9f85d'
 assert sha(B/'release-manifest.json')=='8065f31b9a758cd0a823efbeb8e631cbc740c1c88ca0ea21d7a41e08eab2dcd4'
 decision=W/'root-component-decision.json';assert sha(decision)=='bdadcce6515c7ed15348afdb01c6e07e2a745280e6bcd7ddc35b24ea5c701979'
 d=json.loads(decision.read_text());assert d['lesson']=='02_install_spacr' and d['status']=='targeted_video_repair_authorized_not_completed'
 assert d['repair']['scene']==7 and d['repair']['first_frame']==4420 and d['repair']['frame_count']==692
 for b in d['component_assessments'].values():assert sha(b['path'])==b['sha256'] and Path(b['path']).stat().st_size==b['bytes']
 card=W/'scene07-card-visual-acceptance.json';assert sha(card)=='f99db28091367731527ba42085ba6a77d57123eca020890cc2d2abbef3fb5198'
 a=json.loads(card.read_text());assert a['accepted'] is True and a['terminal_output_fabricated'] is False and a['installation_or_environment_commands_executed'] is False
 for k in ['image','generator_provenance','render_log','generator_source']:assert sha(a[k]['path'])==a[k]['sha256'] and Path(a[k]['path']).stat().st_size==a[k]['bytes']
 for p in sorted((B/'web/catalog').glob('*.json')):
  assert (S/'catalog'/p.name).read_bytes()==p.read_bytes(), 'Retain all14catalog files and lesson objects exactly'
guard()
script=Path(sys.argv[1]).resolve()
assert script.is_relative_to(ROOT/'tools/tutorials') or script.parent==W
sys.argv=[str(script),*sys.argv[2:]]
try:runpy.run_path(str(script),run_name='__main__')
finally:guard()
