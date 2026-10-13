import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/wd4tb/spacr-worktrees/codex-mask-load-drain-20261007')
scratch=Path('/mnt/wd4tb/scratch/annotate-worker-drain-20261007')
phase=sys.argv[1]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QT_QPA_PLATFORM='offscreen',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(root),XDG_CONFIG_HOME=str(scratch/(phase+'-config')))
start=time.monotonic()
r=subprocess.run([sys.executable,'-X','faulthandler',str(scratch/'native_probe.py')],cwd=root,env=env,capture_output=True,text=True,timeout=60)
(scratch/(phase+'.log')).write_text(r.stdout+r.stderr)
receipt={'phase':phase,'exit_code':r.returncode,'seconds':time.monotonic()-start,'source_sha256':hashlib.sha256((root/'spacr/qt/screens/annotate.py').read_bytes()).hexdigest(),'existing_timeout_ms':15000,'core_limit_bytes':0,'scope':'Actual parented retrain QThread and screen close, CPU Event substitutes model work; separate from small-image Shiboken crash.'}
(scratch/(phase+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
