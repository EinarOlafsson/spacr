"""Issue 194: expected CLI refusals keep useful details without automatic filing."""

from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import replace
import json
import logging
import re
import threading
import zipfile

import pytest

from spacr import cli, crashreport as cr, logging_util, validate
from spacr.errors import ConfigurationError
from spacr.validate import ERROR, Problem
from tests.test_cli import _clean_cli_state, fake_pipeline


REFUSAL = "pre-flight found %d error%s; refusing to start."
ADVICE = "Fix them, or pass --force to run anyway (or --no-preflight to skip the check)."


@pytest.fixture
def isolated_reports(tmp_path, monkeypatch):
    """Keep report state local and prevent real network or hardware diagnostics."""
    monkeypatch.setattr(cr, '_reports_dir', lambda: tmp_path / 'reports')
    monkeypatch.setattr(cr, '_ERROR_DISPATCH_LOCAL', threading.local())
    monkeypatch.setattr(cr, '_ERROR_POLICY_CACHE', {'mode': 'all'})
    monkeypatch.setattr(cr, '_gpu_class', lambda: 'none')
    monkeypatch.setattr(cr, '_distribution_version', lambda name: 'test')
    monkeypatch.setattr(logging_util, '_LOG_PATH', tmp_path / 'logs' / 'spacr.log')

    def no_transport(*args, **kwargs):
        """Fail closed if a test reaches an unmocked report or AI transport."""
        raise AssertionError('A report test attempted an unmocked transport')

    monkeypatch.setattr(cr, '_post_json', no_transport)
    monkeypatch.setattr(cr, '_open_url', no_transport)


def _problems():
    """Supply two distinct settings failures with private messages and recovery text."""
    return [
        Problem(ERROR, 'src', 'Input folder is missing: /srv/private_plate/fields',
                'Choose an existing folder instead of /srv/private_plate/fields'),
        Problem(ERROR, 'cell_channel', 'A channel is required; token=sk-abcdefghijklmnopqrstuv',
                r'Select a valid channel after checking C:\Users\private_person\secret.tif'),
    ]


def _record(message, args=(), *, logger='spacr.cli', exception=None):
    """Create a genuine logging record without any synthetic validation metadata."""
    info = None if exception is None else (type(exception), exception, exception.__traceback__)
    return logging.LogRecord(logger, logging.ERROR, __file__, 1, message, args, info)


def test_explicit_report_keeps_both_scrubbed_problems_and_fixes(isolated_reports, monkeypatch):
    """The yielded context supports a manual submission without retaining private inputs."""
    problems = _problems()
    original = deepcopy(problems)
    posted = []

    def post(route, payload, timeout):
        """Emulate the report endpoint and retain its already-scrubbed request."""
        posted.append((route, deepcopy(payload)))
        return 200, {'issue_number': 194, 'issue_url': 'https://example.invalid/issues/194'}

    monkeypatch.setattr(cr, '_post_json', post)
    monkeypatch.setattr(cr, '_sending_allowed', lambda: True)
    monkeypatch.setattr(cr, '_reports_paused_until', lambda: None)
    with cr._validation_error_context('mask', problems, manual_only=True) as context:
        assert cr._ERROR_DISPATCH_LOCAL.validation_context is not context
    summary = cr.public_summary(
        kind=cr.KIND_USER_ERROR, module=context['module'], screen=context['screen'],
        message=context['message'], error_text=context['error_text'])
    outcome = cr.submit_public_summary(summary)

    assert outcome['status'] == cr.REPORT_SENT
    assert len(posted) == 1 and posted[0][0] == '/report'
    payload = posted[0][1]
    assert payload['kind'] == cr.KIND_USER_ERROR
    for field in ('message', 'error_text'):
        for text in ('Input folder is missing', 'Choose an existing folder',
                     'A channel is required', 'Select a valid channel'):
            assert text in payload[field]
    serialized = json.dumps([context, payload])
    for secret in ('private_plate', 'private_person', 'secret.tif',
                   'sk-abcdefghijklmnopqrstuv', '/srv/', 'C:\\Users'):
        assert secret not in serialized
    assert problems == original
    assert cr.queued_summaries() == []


@pytest.mark.parametrize('mode', ['all', 'all_code', 'send_all', 'faults'])
def test_expected_plain_refusal_logs_never_allocate_or_dispatch(isolated_reports, monkeypatch, mode):
    """Both CLI lines stop before automatic state while unrelated errors remain visible."""
    monkeypatch.setattr(cr, '_ERROR_POLICY_CACHE', {'mode': mode})
    occurrences = []
    dispatched = []

    def occurrence(exception=None):
        """Observe event allocation before the mocked automatic dispatcher."""
        occurrences.append(exception)
        return f'test-event-{len(occurrences)}'

    def dispatch(text, **kwargs):
        """Retain automatic routing requests without starting workers or making requests."""
        dispatched.append((text, kwargs))

    monkeypatch.setattr(cr, '_new_error_occurrence', occurrence)
    monkeypatch.setattr(cr, '_dispatch_error', dispatch)
    refusal, advice = _record(REFUSAL, (2, 's')), _record(ADVICE)
    with cr._validation_error_context('mask', _problems(), manual_only=True):
        cr._capture_error_record(refusal)
        cr._capture_error_record(advice)
        assert occurrences == dispatched == []
        assert not hasattr(refusal, '_spacr_error_occurrence')
        assert not hasattr(advice, '_spacr_error_occurrence')
        cr._capture_error_record(_record('Independent CLI failure'))
        cr._capture_error_record(_record(REFUSAL, (2, 's'), logger='spacr.worker'))
    cr._capture_error_record(_record(REFUSAL, (2, 's')))

    assert len(occurrences) == len(dispatched) == 3
    assert [row[0] for row in dispatched] == [
        'Independent CLI failure', 'pre-flight found 2 errors; refusing to start.',
        'pre-flight found 2 errors; refusing to start.']
    assert all(row[1]['context'] is None for row in dispatched)


@pytest.mark.parametrize('exception_type', [RuntimeError, ConfigurationError])
def test_real_exception_records_are_not_suppressed_or_relabelled(
        isolated_reports, monkeypatch, exception_type):
    """Even an identical CLI template cannot hide an attached real exception."""
    exception = exception_type('A real failure while checking input')
    dispatched = []

    def dispatch(text, **kwargs):
        """Capture the exception and absence of unrelated validation context."""
        dispatched.append((text, kwargs))

    monkeypatch.setattr(cr, '_dispatch_error', dispatch)
    monkeypatch.setattr(cr, '_new_error_occurrence', lambda value=None: 'exception-event')
    with cr._validation_error_context('mask', _problems(), manual_only=True):
        cr._capture_error_record(_record(REFUSAL, (2, 's'), exception=exception))
    assert len(dispatched) == 1
    assert dispatched[0][1]['exception'] is exception
    assert dispatched[0][1]['context'] is None
    expected = cr.KIND_USER_ERROR if exception_type is ConfigurationError else cr.KIND_FAULT
    assert cr.classify_failure(exception) == expected
    assert cr.mode_sends(expected, 'all')


def test_default_context_keeps_existing_gui_message_and_restores_nested_state(isolated_reports):
    """The new private option leaves the existing GUI summary and lifetime unchanged."""
    sentinel = {'message': 'existing outer context'}
    cr._ERROR_DISPATCH_LOCAL.validation_context = sentinel
    problem = Problem(ERROR, 'cell_channel', 'Choose a channel', 'Select channel 1')
    with cr._validation_error_context('mask', [problem]) as normal:
        assert normal == {
            'module': 'mask', 'screen': 'Mask',
            'message': 'The run did not start: 1 settings problem.\n- cell_channel: Choose a channel',
            'error_text': 'The run did not start: 1 settings problem.\n- cell_channel: Choose a channel',
        }
        outer = cr._ERROR_DISPATCH_LOCAL.validation_context
        with pytest.raises(ValueError, match='leave the nested context'):
            with cr._validation_error_context('measure', _problems(), manual_only=True) as inner:
                inner['message'] = 'Caller edits cannot change the ambient context'
                assert cr._ERROR_DISPATCH_LOCAL.validation_context['message'] != inner['message']
                raise ValueError('leave the nested context')
        assert cr._ERROR_DISPATCH_LOCAL.validation_context is outer
    assert cr._ERROR_DISPATCH_LOCAL.validation_context is sentinel


def test_manual_context_is_bounded_scrubbed_and_contains_only_safe_values(isolated_reports):
    """Large diagnostics cannot yield unlimited text or retain raw Problem objects."""
    problems = [Problem(ERROR, 'src', 'Missing /srv/private_plate/fields ' + 'detail ' * 400,
                        'Choose another source /srv/private_plate/fields ' + 'action ' * 400)]
    with cr._validation_error_context('/srv/private_plate/run', problems, manual_only=True) as context:
        assert set(context) == {'module', 'screen', 'message', 'error_text', 'manual_only'}
        assert context['manual_only'] is True
        assert len(context['module']) <= 80
        assert len(context['screen']) <= cr.MAX_SCREEN_CHARS
        assert len(context['message']) <= cr.MAX_USER_ERROR_MESSAGE_CHARS
        assert len(context['error_text']) <= cr.MAX_ERROR_TEXT_CHARS
        assert all(isinstance(value, (str, bool)) for value in context.values())
        assert 'private_plate' not in json.dumps(context)


def test_another_thread_does_not_inherit_the_manual_refusal_context(isolated_reports, monkeypatch):
    """A worker logging the same text remains independent of the CLI thread scope."""
    dispatched = []

    def dispatch(text, **kwargs):
        """Capture worker routing without starting automatic reporting."""
        dispatched.append((text, kwargs))

    def emit():
        """Write a refusal-shaped message from a separate execution thread."""
        cr._capture_error_record(_record(REFUSAL, (2, 's')))

    monkeypatch.setattr(cr, '_dispatch_error', dispatch)
    monkeypatch.setattr(cr, '_new_error_occurrence', lambda value=None: 'worker-event')
    with cr._validation_error_context('mask', _problems(), manual_only=True):
        worker = threading.Thread(target=emit)
        worker.start()
        worker.join(5)
        assert not worker.is_alive()
    assert len(dispatched) == 1 and dispatched[0][1]['context'] is None


def test_cli_refusal_reaches_an_explicit_bundle_without_automatic_reporting(
        isolated_reports, tmp_path, monkeypatch, capsys, fake_pipeline):
    """The actual CLI log is collected by the existing manual crashreport ZIP command."""
    settings_path = tmp_path / 'settings.csv'
    with settings_path.open('w', newline='', encoding='utf-8') as stream:
        csv.writer(stream).writerows([('Key', 'Value'), ('src', str(tmp_path))])
    problems = _problems()
    dispatched = []
    monkeypatch.setitem(cli.MODULES, '_fake', replace(cli.MODULES['_fake'], validate_key='mask'))

    def checked(settings, app_key):
        """Supply fixed findings without replacing report formatting or refusal behavior."""
        return problems

    def dispatch(text, **kwargs):
        """Observe automatic dispatch without launching network or AI work."""
        dispatched.append((text, kwargs))

    monkeypatch.setattr(validate, 'validate_settings', checked)
    monkeypatch.setattr(cr, '_dispatch_error', dispatch)
    monkeypatch.setattr(cr, '_install_native_error_capture', lambda *args, **kwargs: None)
    handlers = list(logging.getLogger().handlers)
    root_level = logging.getLogger().level
    package_level = logging.getLogger('spacr').level
    result = cli.main(['_fake', '--settings', str(settings_path)])
    output = capsys.readouterr().out
    assert result == cli.EXIT_USAGE
    assert fake_pipeline.calls == []
    for problem in problems:
        assert problem.message in output and problem.fix in output
    assert 'pre-flight found 2 errors; refusing to start.' in output
    assert dispatched == []
    assert logging.getLogger().handlers == handlers
    assert logging.getLogger().level == root_level
    assert logging.getLogger('spacr').level == package_level
    retained = logging_util.log_path().read_text(encoding='utf-8')
    assert re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*\[WARNING\] spacr\.crashreport\.mask:', retained)
    assert retained.count('The run did not start: 2 settings problems.') == 1
    for text in ('Input folder is missing', 'Choose an existing folder',
                 'A channel is required', 'Select a valid channel'):
        assert text in retained
    for secret in ('private_plate', 'private_person', 'secret.tif',
                   'sk-abcdefghijklmnopqrstuv', '/srv/', 'C:\\Users'):
        assert secret not in retained

    def no_doctor(*args):
        """Skip unrelated live dependency and environment diagnostics for the bundle test."""
        return None

    monkeypatch.setattr(cr, '_doctor_sections', no_doctor)
    monkeypatch.setattr(cr, '_package_versions', lambda: {})
    monkeypatch.setattr(cr, '_gpu_facts', lambda: {})
    monkeypatch.setattr(cr, 'find_last_run_id', lambda problems=None: '')
    destination = tmp_path / 'manual-report.zip'
    assert cr.main(['--output', str(destination)]) == 0
    with zipfile.ZipFile(destination) as archive:
        assert archive.read('spacr.log').decode('utf-8') == retained
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['sections']['spacr.log']['status'] == 'ok'
    assert dispatched == []


def test_an_unwritable_refusal_log_does_not_replace_the_cli_exit(
        isolated_reports, tmp_path, monkeypatch, capsys, fake_pipeline):
    """A directory at the log-file path prevents persistence but leaves refusal behavior intact."""
    blocked = tmp_path / 'not-a-log-file'
    blocked.mkdir()
    monkeypatch.setattr(logging_util, '_LOG_PATH', blocked)
    settings_path = tmp_path / 'settings.csv'
    with settings_path.open('w', newline='', encoding='utf-8') as stream:
        csv.writer(stream).writerows([('Key', 'Value'), ('src', str(tmp_path))])
    dispatched = []

    def checked(settings, app_key):
        """Provide one real Problem object without changing the refusal path."""
        return [Problem(ERROR, 'src', 'Input is unavailable', 'Choose another input')]

    def dispatch(text, **kwargs):
        """Observe any accidental automatic reporting without network activity."""
        dispatched.append((text, kwargs))

    monkeypatch.setattr(validate, 'validate_settings', checked)
    monkeypatch.setattr(cr, '_dispatch_error', dispatch)
    monkeypatch.setattr(cr, '_install_native_error_capture', lambda *args, **kwargs: None)
    assert cli.main(['_fake', '--settings', str(settings_path)]) == cli.EXIT_USAGE
    output = capsys.readouterr().out
    assert 'Input is unavailable' in output and 'Choose another input' in output
    assert 'pre-flight found 1 error; refusing to start.' in output
    assert fake_pipeline.calls == dispatched == []
    assert blocked.is_dir()
