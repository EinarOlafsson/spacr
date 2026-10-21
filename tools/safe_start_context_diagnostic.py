"""One diagnostic replay of immutable83ed Coverage9 batch5; never a full CI gate."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

COLLECTOR_ERROR_MARKER = b'SPACR_SAFE_COLLECTOR_ERROR:'


def collector_integrity(children, pytest_log):
    """Fail the diagnostic separately when observation was incomplete."""
    issues = []
    paths = [pytest_log, *children.glob('result-*/stdout.bin'),
             *children.glob('result-*/stderr.bin')]
    for path in paths:
        if not path.exists():
            continue
        with path.open('rb') as handle:
            previous = b''
            while chunk := handle.read(65536):
                if COLLECTOR_ERROR_MARKER in previous + chunk:
                    issues.append('Collector error marker in ' + path.name)
                    break
                previous = chunk[-len(COLLECTOR_ERROR_MARKER):]
    for path in children.glob('**/collector-integrity-*.json'):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get('errors'):
            issues.append('Collector integrity errors in ' + path.name)
    for phase in ('broken', 'safe', 'repaired'):
        result = children / ('result-' + phase)
        if not all((result / name).is_file() for name in (
                'invocation.json', 'result.json', 'stdout.bin', 'stderr.bin')):
            issues.append('Missing original child result evidence for ' + phase)
        phase_dirs = list(children.glob(phase + '-pid-*'))
        if len(phase_dirs) != 1:
            issues.append('Missing or duplicate child phase directory for ' + phase)
            continue
        journal = phase_dirs[0] / 'phases.jsonl'
        faults = phase_dirs[0] / 'periodic-stacks.log'
        if not journal.is_file() or not faults.is_file():
            issues.append('Missing child phase/stack evidence for ' + phase)
            continue
        boot = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()
                if json.loads(line).get('event') == 'child_boot']
        if len(boot) != 1 or not boot[0].get('coverage_active'):
            issues.append('Missing covered child boot receipt for ' + phase)
    return issues


parser=argparse.ArgumentParser()
parser.add_argument('mode',choices=['bind','run'])
parser.add_argument('--context',type=Path,required=True)
parser.add_argument('--evidence',type=Path,required=True)
args=parser.parse_args()
context=json.loads(args.context.read_text(encoding="utf-8"))
evidence=args.evidence.resolve();evidence.mkdir(parents=True,exist_ok=True)
root=Path.cwd().resolve(); workflow_sha=os.environ.get('GITHUB_SHA')
def guard():
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert head==context['source_sha']
    subprocess.run(['git','diff','--exit-code','HEAD','--','spacr','tests','tools','.coveragerc'],check=True,stdout=subprocess.DEVNULL)
    for name,expected in context['source_bindings'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
    batch=context['batch5_files'];assert len(batch)==32
    assert hashlib.sha256(('\n'.join(batch)+'\n').encode()).hexdigest()==context['batch5_sha256']
    return {'head':head,'source_bindings':len(context['source_bindings']),'batch5_sha256':context['batch5_sha256'],'original_child_timeout':300}
if args.mode=='bind':
    import importlib.metadata as metadata
    from PySide6.QtCore import qVersion
    required={'torch':'2.14.1+cpu','torchvision':'0.29.1+cpu','PySide6':'6.12.0','pytest':'8.4.2','pytest-qt':'4.5.0','pytest-xdist':'3.8.0','pytest-timeout':'2.4.0','pytest-cov':'7.1.0','coverage':'7.16.2','hypothesis':'6.169.1','anyio':'4.15.1','platformdirs':'4.13.0','pluggy':'1.6.0'}
    actual={name:metadata.version(name) for name in required}
    assert actual==required and qVersion()=='6.12.0'
    assert sys.version_info[:3]==(3,12,15)
    payload={'source':guard(),'workflow_sha':workflow_sha,'runtime':actual,'qt':qVersion(),'python':sys.version,'diagnostic_only':True,'full_ci_acceptance':False}
    (evidence/'source-and-runtime.json').write_text(json.dumps(payload,indent=2)+'\n', encoding="utf-8")
    sys.exit(0)
assert (evidence/'source-and-runtime.json').exists()
before=guard(); env=os.environ.copy();env['GITHUB_SHA']=context['source_sha'];env['COVERAGE_FILE']=str(evidence/'coverage'/'.coverage');env['SPACR_SAFE_DIAGNOSTIC_DIR']=str(evidence/'children');env['SPACR_SAFE_ORIGINAL_LAUNCHER_SHA']=context['original_launcher_sha256'];env['SPACR_HOME']=str(evidence/'app-home');env['XDG_CONFIG_HOME']=str(evidence/'config');env['XDG_CACHE_HOME']=str(evidence/'cache');env['SPACR_LOG_DIR']=str(evidence/'logs');env['SPACR_TEST_COUNT_DIR']=str(evidence/'counts');env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1';env['PYTEST_ADDOPTS']='';env['PYTHONDONTWRITEBYTECODE']='1'
for key in ['SPACR_NO_GL','SPACR_NO_BACKDROP','SPACR_TIMING','SPACR_WATCH_GUI_STALLS','SPACR_BENCHMARK_JSON']:
    env.pop(key,None)
for name in ['coverage','children','app-home','config','cache','logs','counts','pytest-temp']:(evidence/name).mkdir(exist_ok=True)
command=[sys.executable,'-m','pytest',*context['batch5_files'],'-m','not gui','-n','2','--dist=loadfile','--cov=spacr','--cov-branch','--cov-report=','--cov-config=.coveragerc','--timeout=600','--timeout-method=thread','--max-worker-restart=0','-o','faulthandler_timeout=900','-o',f'cache_dir={evidence}/pytest-cache','--basetemp',str(evidence/'pytest-temp'),'--junitxml',str(evidence/'junit.xml'),'-v','--tb=short','-rA']
for plugin in ['_hypothesis_pytestplugin','platformdirs.pytest_plugin','pytestqt.plugin','anyio.pytest_plugin','xdist.plugin','xdist.looponfail','pytest_timeout','pytest_cov.plugin','tools.pytest_plugins.qt_serial_rss_journal','safe_child_evidence']:
    command.extend(['-p',plugin])
(evidence/'launch.json').write_text(json.dumps({'source_sha':context['source_sha'],'workflow_sha':workflow_sha,'command':command,'original_batch':{'coverage_shard':9,'batch':5,'files':32},'retries':0,'full_ci_acceptance':False},indent=2)+'\n', encoding="utf-8")
with (evidence/'pytest.log').open('w') as out:
    process=subprocess.Popen(command,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    timed_out=False
    try:original_exit=process.wait(timeout=2700)
    except subprocess.TimeoutExpired:
        import signal
        timed_out=True
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=30)
        original_exit=process.returncode
integrity=[]
try:after=guard()
except Exception as exc:after={'guard_error':str(exc)};integrity.append('Source guard failed after replay.')
try:
    cases=list(ET.parse(evidence/'junit.xml').getroot().iter('testcase'))
    if len(cases)!=context['original_expected_case_count']:integrity.append('Original580-case batch count differs.')
    target=[c for c in cases if c.attrib.get('classname','').endswith('test_safespacr_gets_in_when_a_preference_breaks_it') and c.attrib.get('name') in context['original_fixture_nodes']]
    if len(target)!=7 or {c.attrib.get('name') for c in target}!=set(context['original_fixture_nodes']):integrity.append('Original seven fixture nodes missing/duplicated.')
    if any(c.find('skipped') is not None for c in target):integrity.append('Original fixture node skipped.')
except Exception as exc:cases=[];target=[];integrity.append('Missing/unreadable original JUnit: '+type(exc).__name__)
launchers=list((evidence/'pytest-temp').glob('**/launch_once.py'))
if len(launchers)!=1 or any(hashlib.sha256(p.read_bytes()).hexdigest()!=context['original_launcher_sha256'] for p in launchers):integrity.append('Original launcher absent/changed/duplicated.')
try:
    integrity.extend(collector_integrity(evidence/'children', evidence/'pytest.log'))
except Exception as exc:
    integrity.append('Unreadable collector integrity evidence: ' + type(exc).__name__)
status=original_exit if original_exit else (2 if integrity or timed_out else 0)
receipt={'source_before':before,'source_after':after,'workflow_sha':workflow_sha,'original_pytest_exit':original_exit,'diagnostic_exit':status,'batch_timeout':timed_out,'integrity_issues':integrity,'case_count':len(cases),'target_case_count':len(target),'target_outcomes':[{'name':c.attrib.get('name'),'failed':c.find('failure') is not None,'error':c.find('error') is not None,'skipped':c.find('skipped') is not None} for c in target],'full_ci_acceptance':False,'diagnostic_only':True}
(evidence/'terminal.json').write_text(json.dumps(receipt,indent=2)+'\n', encoding="utf-8");print(json.dumps(receipt),flush=True);sys.exit(status)
