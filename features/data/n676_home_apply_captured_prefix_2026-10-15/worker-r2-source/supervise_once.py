"""Owned diagnostic supervisor; preparation never invokes this module."""
from pathlib import Path
import os, sys, subprocess, time, signal, json, socket, tempfile
R=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path.insert(0,str(R))
W=Path(__file__).resolve().parent
S=Path('/media/carruthers/mnt3/claude/n676_stages/host/lesson05-worker-apply-r2')
assert not S.exists()
revision=os.environ['SPACR_FINAL_CAPTURE_REVISION']
assert revision=='87ffcb60cab0eca8772cb1ffaccc12f3f4f48be9'
started=time.monotonic_ns()
phase_deadline=started+300_000_000_000
total_deadline=started+330_000_000_000
env=dict(os.environ,SPACR_TUTORIAL_DIAGNOSTIC_BEGIN_NS=str(started),SPACR_TUTORIAL_APPLY_NATIVE_PHASE_DEADLINE_NS=str(phase_deadline),SPACR_TUTORIAL_APPLY_TOTAL_DEADLINE_NS=str(total_deadline))
interrupted=[]
for sig in (signal.SIGTERM,signal.SIGINT):
    signal.signal(sig,lambda signum,frame:interrupted.append(signum))

def wait_until(proc,deadline,stop_on_interrupt=False):
    while proc.poll() is None and time.monotonic_ns()<deadline:
        if stop_on_interrupt and interrupted:return None
        remaining=(deadline-time.monotonic_ns())/1e9
        if remaining>0:time.sleep(min(.01,remaining))
    return proc.poll()

def own_group_signal(proc,sig):
    if proc.poll() is None:
        try:os.killpg(proc.pid,sig)
        except ProcessLookupError:pass

report={'schema':1,'source_commit':revision,'execution_timeout_seconds':300,'selected_scene':'13h_preferences_apply','cleanup_grace_seconds':30,'native_term_grace_seconds':5,'total_max_seconds':330,'recording':True}
command=[sys.executable,str(W/'capture_job.py'),'--stage',str(S),'--revision',revision]
proc=None;cleanup=None;cleanup_log=None
try:
    proc=subprocess.Popen(command,cwd=R,env=env,start_new_session=True)
    code=wait_until(proc,phase_deadline,True)
    grace_started=time.monotonic_ns()
    cleanup_deadline=min(total_deadline,grace_started+30_000_000_000)
    report['native_phase_elapsed_ns']=grace_started-started
    report['within_300_second_execution']=grace_started<=phase_deadline
    report['native_timed_out']=code is None and not interrupted
    report['timeout_scope']='outer' if time.monotonic_ns()>=phase_deadline else 'interrupt' if code is None else None
    report['interrupted_signals']=list(interrupted)
    report['native_term_sent']=code is None
    if code is None:
        own_group_signal(proc,signal.SIGTERM)
        code=wait_until(proc,min(grace_started+5_000_000_000,cleanup_deadline-1_000_000_000))
    report['native_kill_sent']=code is None
    if code is None:
        own_group_signal(proc,signal.SIGKILL)
        code=wait_until(proc,min(time.monotonic_ns()+200_000_000,cleanup_deadline-1_000_000_000))
    report['native_exit_code']=code
    report['native_process_reaped']=code is not None
    remaining=cleanup_deadline-time.monotonic_ns()
    cleanup_code=None
    report['marked_cleanup_attempted']=remaining>1_000_000_000
    if remaining>1_000_000_000:
        if S.is_dir():
            cleanup_log=(S/'logs/cleanup-private.log').open('x')
        cleanup=subprocess.Popen([sys.executable,str(W/'capture_job.py'),'--stage',str(S),'--cleanup-only'],cwd=R,env=env,start_new_session=True,stdout=cleanup_log,stderr=subprocess.STDOUT)
        cleanup_code=wait_until(cleanup,cleanup_deadline-1_000_000_000)
        report['cleanup_kill_sent']=cleanup_code is None
        if cleanup_code is None:
            own_group_signal(cleanup,signal.SIGKILL)
            cleanup_code=wait_until(cleanup,cleanup_deadline-500_000_000)
    report['marked_cleanup_exit_code']=cleanup_code
    report['cleanup_grace_elapsed_ns']=time.monotonic_ns()-grace_started
    owner_cleanup=S/'owner-cleanup.json'
    receipt=json.loads(owner_cleanup.read_text()) if owner_cleanup.is_file() else None
    report['marked_cleanup_receipt_present']=receipt is not None
    report['survivors_after_cleanup']=receipt.get('survivors_after_cleanup') if receipt else None
    report['cleanup_complete']=cleanup_code==0 and receipt is not None and receipt.get('survivors_after_cleanup')==[] and code is not None
except BaseException as exc:
    report['supervisor_error']=type(exc).__name__
    report['cleanup_complete']=False
finally:
    for child in (proc,cleanup):
        if child is not None and child.poll() is None:own_group_signal(child,signal.SIGKILL)
    if cleanup_log is not None:cleanup_log.close()
    report['total_elapsed_ns']=time.monotonic_ns()-started
    report['within_330_second_total']=report['total_elapsed_ns']<=330_000_000_000
    report['within_30_second_cleanup_grace']=report.get('cleanup_grace_elapsed_ns',31_000_000_000)<=30_000_000_000
    report['completed_capture_workflow']=report.get('native_exit_code') in (0,2) and report.get('cleanup_complete') is True and report['within_330_second_total'] and report['within_30_second_cleanup_grace'] and report.get('within_300_second_execution') is True and not report.get('native_term_sent') and not interrupted
    report['limits']='Deadlines are enforced with monotonic finite polls and owned process groups. Actual elapsed and cleanup receipt determine compliance; scheduling overrun or unknown survivors is never claimed bounded successful cleanup.'
    if S.is_dir():(S/'supervisor-result.json').write_text(json.dumps(report,indent=2)+'\n')
    else:(W/'supervisor-no-stage-result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report.get(k) for k in ['native_exit_code','marked_cleanup_exit_code','cleanup_complete','total_elapsed_ns','within_330_second_total','completed_capture_workflow']}),flush=True)
raise SystemExit(report['native_exit_code'] if report['completed_capture_workflow'] else 1)
