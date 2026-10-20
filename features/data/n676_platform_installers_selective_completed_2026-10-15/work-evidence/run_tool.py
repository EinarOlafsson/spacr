#!/usr/bin/env python3
"""Execute root-selected selective04 tools; no whole-capture/science restamping."""
from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'tools/tutorials'),str(ROOT/'tools/tutorials/authoring/tools')]
sys.meta_path[:]=[f for f in sys.meta_path if '__editable__' not in (getattr(f,'__module__','') or type(f).__module__)]
import hashlib,json,os,subprocess,runpy
W=Path(__file__).resolve().parent
B=Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-02-selective-r2-20261015/release-candidate-append-lxmwr53e')
S=Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-04-selective-20261015')
SOURCE_BEFORE='50da1bd9bfd59f9042019a1e20b602fcfae86c24'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def guard():
 assert Path.cwd().resolve()==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert git('rev-parse','HEAD')==os.environ['SOURCE_COMMIT']
 assert Path(os.environ['STAGE']).resolve()==S and Path(os.environ['BASELINE_CANDIDATE']).resolve()==B
 subprocess.run(['git','diff','--quiet',SOURCE_BEFORE,os.environ['SOURCE_COMMIT'],'--','spacr','tools/tutorials/lessons/04_platform_installers.json','tools/tutorials/lessons/reviews/04_platform_installers.*.json'],cwd=ROOT,check=True)
 assert not git('status','--porcelain','--','spacr','tools/tutorials','tests',':(exclude)tools/tutorials/release_candidate')
 assert sha(B/'release-manifest.json')=='98d88a909b3eeaf14f829297d965461e62a07dff7dfd25ce50d2c2bc15a25439'
 decision=W/'root-component-decision-v2.json';assert sha(decision)=='e6cd05d93c47999c2cdfdd7dadbd4ed9f7f64941fa8ca106845b3febd0d2456c'
 d=json.loads(decision.read_text());assert d['lesson']=='04_platform_installers' and d['status']=='targeted_video_repair_authorized_not_completed' and d['replacement_scenes']==[9,10]
 policy=W/'root-native-duration-policy.json';assert sha(policy)=='ff59b6019f8a39aebc8013c7227decc395a67202ee86d31154485e03e85abd9b'
 ap=Path(os.environ['ACCEPTED_NATIVE_EVIDENCE']);assert sha(ap)==os.environ['ACCEPTED_NATIVE_SHA256']
 capture_commit='b929b7dc53465132322c3ca0b866f567de773b9f'
 subprocess.run(['git','merge-base','--is-ancestor',capture_commit,os.environ['SOURCE_COMMIT']],cwd=ROOT,check=True)
 changed=set(git('diff','--name-only',capture_commit,os.environ['SOURCE_COMMIT']).splitlines())
 assert changed=={'tools/tutorials/authoring/web/app_v2.js','tools/tutorials/tests/test_player_preserve_visual_intervals.py'}, 'Exact two-path playback follow-up only'
 a=json.loads(ap.read_text());assert a['accepted'] is True and a['source_commit']==capture_commit and a['actual_native_frames']>=840 and a['source_scene']=='13_current_nightly_home'
 for key in ['video','capture_receipt','gate','host_run','provenance','profile','cleanup','frames_manifest','poster','exact_source']:
  b=a[key];p=Path(b['path']);assert p.stat().st_size==b['bytes'] and sha(p)==b['sha256']
 card=W/'scene10-doctor-card-visual-acceptance.json';assert sha(card)=='0c1efb1d98a00577558ba114c104cb19fddaadfc8c785372ee49d77d5577aee2'
 a=json.loads(card.read_text());assert a['accepted'] is True and a['terminal_output_fabricated'] is False and a['installation_or_environment_commands_executed'] is False
 for k in ['image','generator_provenance','generator_source']:
  b=a[k];assert sha(b['path'])==b['sha256'] and Path(b['path']).stat().st_size==b['bytes']
 for p in sorted((B/'web/catalog').glob('*.json')):
  assert (S/'catalog'/p.name).read_bytes()==p.read_bytes(), 'Retain all14catalog files and lesson objects exactly'
guard()
script=Path(sys.argv[1]).resolve()
assert script.is_relative_to(ROOT/'tools/tutorials') or script.parent==W
if script.name in {'browser_matrix.py','verify_staged_lesson.py'}:
 from retained_visual_refresh import checked_retained_visual_timing
 checked_retained_visual_timing(S,'04_platform_installers',B)
sys.argv=[str(script),*sys.argv[2:]]
try:runpy.run_path(str(script),run_name='__main__')
finally:guard()
