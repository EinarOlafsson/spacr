from pathlib import Path
import datetime, hashlib, json, os, resource, signal, subprocess, sys
root=Path.cwd()
scratch=Path('/mnt/wd4tb/scratch/native-qt612-parent-race-hostthreads-20261008')
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
paths=['spacr/qt/bridge.py','spacr/qt/screens/make_masks.py','spacr/qt/widgets/primary_mask_selector.py','tests/conftest.py','tests/qt/conftest.py','tests/qt/test_live_preview_remove_background.py','tests/qt/test_make_masks_parent_source_race_guards.py']
bindings={}
for path in paths:
    raw=(root/path).read_bytes()
    prior=subprocess.check_output(['git','show','0b8c2c4120a0ede4de568d484ea3fc9d517f1de3:'+path])
    assert raw==prior,path
    bindings[path]=hashlib.sha256(raw).hexdigest()
nodes=['tests/qt/test_live_preview_remove_background.py','tests/qt/test_make_masks_parent_source_race_guards.py::test_puncta_requires_an_explicit_valid_parent','tests/qt/test_make_masks_parent_source_race_guards.py::test_native_sources_changed_during_detection_cannot_publish_a_result[image]','tests/qt/test_make_masks_parent_source_race_guards.py::test_native_sources_changed_during_detection_cannot_publish_a_result[parent]']
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONFAULTHANDLER='1',CUDA_VISIBLE_DEVICES='',SPACR_DEVICE='cpu',QT_QPA_PLATFORM='offscreen',MPLBACKEND='Agg',PYTHONPATH='.:/mnt/wd4tb/scratch/ci-7a-root-20261008/qt612-overlay:/mnt/wd4tb/scratch/ci-55bad-20261008/pytest842-overlay',COVERAGE_FILE=str(scratch/'.coverage'))
command=['/usr/bin/gdb','-nx','-nh','-batch','-x',str(scratch/'gdb.commands'),'--args',sys.executable,'-m','pytest','-vv','--tb=short','-p','no:randomly',*nodes,'--cov=spacr','--cov-branch','--cov-report=']
record={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'argv':command,'cwd':str(root),'environment':{key:env[key] for key in ['CUDA_VISIBLE_DEVICES','SPACR_DEVICE','QT_QPA_PLATFORM','MPLBACKEND','PYTHONPATH','COVERAGE_FILE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONFAULTHANDLER']},'source_sha256':bindings,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'core_limit':list(resource.getrlimit(resource.RLIMIT_CORE)),'scope':'One bounded original-order 12-node native replay on newly hosted Qt6.12.0; no complete long-order or xdist reproduction, no native causal acceptance.'}
with (scratch/'gdb.log').open('wb') as output:
    process=subprocess.Popen(command,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    record['gdb_pid']=process.pid
    (scratch/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    try:record['gdb_exit_code']=process.wait(timeout=300)
    except subprocess.TimeoutExpired:
        record['timed_out']=True
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait(timeout=15)
    record['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (scratch/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
