from pathlib import Path
import fcntl, os, subprocess, sys, datetime, tempfile
R=Path('/media/carruthers/mnt3/claude/spacr-player-readiness-port-20261015')
W=Path(__file__).resolve().parent
private=Path(tempfile.mkdtemp(prefix='spacr-pr-',dir='/tmp'))
for name in ['home','config','cache','tmp']: (private/name).mkdir(exist_ok=True)
lease=Path('/media/carruthers/mnt3/claude/ram_leases.md')
line=f'codex-player-readiness-port-20261015 | independent portable player11 CPU browser tests | 4 GB | owner_pid={os.getpid()} | until exit'
with lease.open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+line+'\n')
env=dict(os.environ, HOME=str(private/'home'), XDG_CONFIG_HOME=str(private/'config'), XDG_CACHE_HOME=str(private/'cache'), TMPDIR=str(private/'tmp'), CUDA_VISIBLE_DEVICES='', QT_QPA_PLATFORM='offscreen', QT_OPENGL='software', LIBGL_ALWAYS_SOFTWARE='1', MESA_LOADER_DRIVER_OVERRIDE='swrast', GALLIUM_DRIVER='llvmpipe', __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json', __GLX_VENDOR_LIBRARY_NAME='mesa', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
command=[str(R/'tools/run_capped.sh'),'4G','/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/render-py312/bin/python',str(W/'run_pytest.py'),'-q','-s','tools/tutorials/tests/test_player_audio_readiness.py','tools/tutorials/tests/test_player_complete_native_clips.py','--junitxml='+str(W/'browser-tests-r2.xml')]
try:
 with (W/'browser-tests-r2.log').open('x') as log:
  log.write('Started '+datetime.datetime.now().astimezone().isoformat()+'\n'+repr(command)+'\n');log.flush()
  run=subprocess.run(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
  log.write('\nTerminal exit status: '+str(run.returncode)+'\n');log.flush()
 print('Terminal exit status:',run.returncode,flush=True)
 sys.exit(run.returncode)
finally:
 with lease.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);lines=f.readlines();f.seek(0);f.writelines(x for x in lines if x.strip()!=line);f.truncate()
