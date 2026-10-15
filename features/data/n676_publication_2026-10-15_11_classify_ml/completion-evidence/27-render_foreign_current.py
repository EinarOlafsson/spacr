#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'tools/tutorials'),str(ROOT/'tools/tutorials/authoring/tools')]
import os,subprocess
from render_all_voices import LANGUAGES
assert os.environ.get('TRANSLATIONS_APPROVED')=='1'
assert os.environ.get('PARALLEL_VOICE_RAM_LEASE') and float(os.environ['PARALLEL_VOICE_RAM_GB'])>=36
voices=[v for lang,(_,items) in LANGUAGES.items() if lang!='en' for v in items]
assert len(voices)==len(set(voices))==26
children=[]
try:
 for i in range(4):
  cmd=[str(Path(__file__).with_name('tool.sh')),f'voices-foreign-current-partition{i+1}','tools/tutorials/authoring/tools/render_all_voices.py','--lessons','11','--languages','all','--voices',','.join(voices[i::4]),'--threads','2','--device','cpu','--force']
  children.append(subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,TOOL_CAP='8G')))
  print('partition',i+1,'PID',children[-1].pid,flush=True)
finally:
 statuses=[p.wait() for p in children]
print('Foreign current runtime terminal statuses',statuses,flush=True)
raise SystemExit(0 if statuses==[0,0,0,0] else 1)
