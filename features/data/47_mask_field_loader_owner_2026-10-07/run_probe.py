import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root=Path('/mnt/wd4tb/spacr-worktrees/codex-mask-load-drain-20261007')
scratch=Path('/mnt/wd4tb/scratch/mask-load-drain-20261007')
phase=sys.argv[1]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QT_QPA_PLATFORM='offscreen',
         OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(root),
         XDG_CONFIG_HOME=str(scratch/(phase+'-config')))
start=time.monotonic()
r=subprocess.run([sys.executable,'-X','faulthandler',str(scratch/'native_probe.py')],
                 cwd=root,env=env,capture_output=True,text=True,timeout=60)
(scratch/(phase+'.log')).write_text(r.stdout+r.stderr)
receipt={'phase':phase,'exit_code':r.returncode,'seconds':time.monotonic()-start,
         'source_sha256':hashlib.sha256((root/'spacr/qt/screens/make_masks.py').read_bytes()).hexdigest(),
         'core_limit_bytes':0,'existing_timeout_ms':5000,
         'scope':'Controlled I/O wait in real _MaskLoadWorker; actual screen close/native destruction. Separate from small-image Shiboken SIGSEGV.'}
(scratch/(phase+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
