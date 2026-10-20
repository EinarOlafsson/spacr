from pathlib import Path
import os,sys,subprocess,fcntl
R=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014');sys.path[:0]=[str(R),str(R/'tools')]
W=Path('/media/carruthers/mnt3/claude/n676_stages/work/lesson04-selective-20261015');ledger=W.parents[2]/'ram_leases.md'
ledger=Path('/media/carruthers/mnt3/claude/ram_leases.md')
lease_name=f'codex-lesson04-retained-mixed-tests | focused static pipeline guards | 4 GB | owner_pid={os.getpid()} | removed on exit'
with ledger.open('a+') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+lease_name+'\n')
try:
 env=os.environ.copy();profile=W/'test-profile';profile.mkdir(exist_ok=True);env.update(HOME=str(profile),TMPDIR=str(profile),CUDA_VISIBLE_DEVICES='',QT_QPA_PLATFORM='offscreen',LIBGL_ALWAYS_SOFTWARE='1',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',GALLIUM_DRIVER='llvmpipe',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json',__GLX_VENDOR_LIBRARY_NAME='mesa',PYTHONDONTWRITEBYTECODE='1')
 code="import sys;sys.path[:0]=["+repr(str(R))+","+repr(str(R/'tools'))+"];import pytest;raise SystemExit(pytest.main(sys.argv[1:]))"
 args=['/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/render-py312/bin/python','-c',code,'-q','tools/tutorials/tests/test_retained_visual_refresh.py','tools/tutorials/tests/test_retained_mixed_visual_refresh.py','tools/tutorials/tests/test_retained_pipeline_sources.py','tools/tutorials/tests/test_completed_matrix.py','tools/tutorials/tests/test_appended_candidate.py','tools/tutorials/tests/test_retained_narration.py','tools/tutorials/tests/test_native_live_timing.py','tools/tutorials/tests/test_installation_merge.py']
 with (W/'retained-mixed-tests-r1.log').open('w') as log:
  result=subprocess.run([str(R/'tools/run_capped.sh'),'4G',*args],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT);log.write(f'\nTerminal exit status: {result.returncode}\n')
 print('Terminal exit status:',result.returncode,flush=True);raise SystemExit(result.returncode)
finally:
 with ledger.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);lines=f.readlines();f.seek(0);f.writelines(x for x in lines if x.strip()!=lease_name);f.truncate()
