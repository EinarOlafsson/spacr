"""Only the explicit scoped entry, native terminal job ownership and namespace seams."""
import importlib
import json
from pathlib import Path
import tempfile
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import verify_explain_cv_capture as entry
from verify_explain_cv_scoped import terminal_job, real
from surrogate_source_evidence import partial_scope


@pytest.fixture
def job():
    # Preserve fixtures for inspection instead of deleting source-like artifacts.
    root=Path(tempfile.mkdtemp(prefix='explain-scoped-entry-'))
    queue=root/'t70/queue';queue.mkdir(parents=True)
    stage=root/'actual';stage.mkdir()
    (queue/'c09.job.done').write_text(f'/normal/live18.sh {stage} -- --module explain_cv --record-scenes scene1,scene2\n')
    (queue/'c09.rc').write_text('2\n')
    (queue/'c09.out').write_text(f'capture stage {stage} -> {stage}\nend 2026-10-29T03:33:14-04:00 rc=2\n')
    return stage,queue


def test_normal_entry_keeps_default_and_dispatches_only_explicit_scope(monkeypatch,capsys):
    monkeypatch.setattr(entry,'verify',lambda path:{'default':str(path)})
    entry.main(['/actual/capture'])
    assert json.loads(capsys.readouterr().out)=={'default':'/actual/capture'}
    scoped=importlib.import_module('verify_explain_cv_scoped')
    def check(stage,queue,allow,stem):
        assert (str(stage),str(queue),allow,stem)==('/actual','/queue',True,'c09')
        return dict(scope_passed=True,raw_scientific_accepted=False,lesson_accepted=False,publication_eligible=False)
    monkeypatch.setattr(scoped,'verify',check)
    entry.main(['--stage','/actual','--queue','/queue','--job-stem','c09','--allow-terminal-gate-partial'])
    result=json.loads(capsys.readouterr().out)
    assert result['scope_passed'] and not any(result[k] for k in ('raw_scientific_accepted','lesson_accepted','publication_eligible'))


@pytest.mark.parametrize('args',[
    ['--stage','/actual'],
    ['--stage','/actual','--queue','/queue','--job-stem','c09'],
    ['/capture','--stage','/actual','--queue','/queue','--job-stem','c09','--allow-terminal-gate-partial'],
    ['--stage','/actual','--queue','/queue','--job-stem','a01','--allow-terminal-gate-partial'],
])
def test_incomplete_or_old_scoped_cli_is_rejected(args):
    with pytest.raises(SystemExit):entry.main(args)


def test_actual_first_and_retry_forms_bind_declared_scene_scope(job):
    stage,queue=job
    assert terminal_job(stage,queue,'c09')==('2',['scene1','scene2'])
    capture=stage/'captures/explain_cv';capture.mkdir(parents=True)
    (capture/'host_gate_summary.json').write_text(json.dumps({'failed':['scene2']}))
    retry=Path(str(stage)+'-b');retry.mkdir()
    (queue/'d09.job.done').write_text(f'/normal/retry_failed.sh {queue}/c09.job.done\n')
    (queue/'d09.rc').write_text('2\n')
    (queue/'d09.out').write_text(f're-recording scene2 from {stage}\ncapture stage {retry} -> {retry}\nend now rc=2\n')
    assert terminal_job(retry,queue,'d09')==('2',['scene2'])
    (queue/'d09.job.done').write_text(f'/normal/retry_failed.sh {queue}/other.job.done\n')
    with pytest.raises(ValueError,match='does not own'):terminal_job(retry,queue,'d09')


@pytest.mark.parametrize('case',['wrong_stage','wrong_log','wrong_rc','running','fatal','deadline','wrong_module'])
def test_terminal_ownership_and_fatal_jobs_fail_closed(job,case):
    stage,queue=job
    if case=='wrong_stage':stage=stage.parent
    elif case=='wrong_log':(queue/'c09.out').write_text('capture stage /other -> /other\nend now rc=2\n')
    elif case=='wrong_rc':(queue/'c09.rc').write_text('139\n')
    elif case=='running':(queue/'c09.job.running').write_text('live')
    elif case in ('fatal','deadline'):
        with (queue/'c09.out').open('a') as f:f.write('Fatal Python error: Segmentation fault' if case=='fatal' else 'sending signal INT to command')
    else:
        path=queue/'c09.job.done';path.write_text(path.read_text().replace('explain_cv','other'))
    with pytest.raises(ValueError):terminal_job(stage,queue,'c09')


def test_namespace_mapping_cannot_escape_stage(job):
    stage,_=job
    assert real(stage,'/tmp/spacr-tutorial-current/work/table.csv')==stage/'work/table.csv'
    with pytest.raises(ValueError):real(stage,'/tmp/spacr-tutorial-current/../foreign.csv')
    outside=stage.parent/'outside';outside.mkdir();(stage/'escape').symlink_to(outside)
    with pytest.raises(ValueError):real(stage,str(stage/'escape/table.csv'))


def test_partial_entry_cannot_validate_incomplete_callbacks():
    with pytest.raises(ValueError,match='did not complete'):
        partial_scope({'completed_capture':False,'status':'gate_failed'},
                      {'computational_run_completed':False},'2',True)
