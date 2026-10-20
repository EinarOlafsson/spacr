"""Capture child evidence without changing its original call or outcome."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

_original = subprocess.run
_errors = []
_MARKER = 'SPACR_SAFE_COLLECTOR_ERROR:'


def _error(operation, exc):
    """Retain a bounded integrity error without raising into the test."""
    try:
        record = {'operation': operation, 'error_type': type(exc).__name__,
                  'pid': os.getpid()}
        if len(_errors) < 100:
            _errors.append(record)
        sys.__stderr__.write(_MARKER + json.dumps(record) + '\n')
        sys.__stderr__.flush()
    except Exception:
        pass


def _capture(operation, action):
    try:
        return action()
    except Exception as exc:
        _error(operation, exc)
        return None


def _save_bytes(path, data):
    if data is None:
        data = b''
    elif isinstance(data, str):
        data = data.encode('utf-8', errors='surrogatepass')
    path.write_bytes(data)


def _prepare(command, kwargs):
    digest = hashlib.sha256(Path(str(command[1])).read_bytes()).hexdigest()
    if digest != os.environ['SPACR_SAFE_ORIGINAL_LAUNCHER_SHA']:
        raise ValueError('Original launcher identity differs.')
    if kwargs.get('timeout') != 300:
        raise ValueError('Original child timeout differs.')
    folder = Path(os.environ['SPACR_SAFE_DIAGNOSTIC_DIR']) / (
        'result-' + Path(str(command[4])).stem)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'invocation.json').write_text(json.dumps({
        'phase': str(command[3]), 'name': Path(str(command[4])).stem,
        'original_timeout': kwargs['timeout'], 'launcher': str(command[1]),
        'source': str(command[2]), 'monotonic_start': time.monotonic()}, indent=2) + '\n')
    return folder


def _save_result(folder, result, timeout=False):
    if folder is None:
        return
    _capture('stdout', lambda: _save_bytes(folder / 'stdout.bin', result.stdout))
    _capture('stderr', lambda: _save_bytes(folder / 'stderr.bin', result.stderr))
    receipt = {'timeout_expired': timeout, 'monotonic_end': time.monotonic()}
    if timeout:
        receipt.update(original_timeout=result.timeout, exception_unchanged=True)
    else:
        receipt.update(returncode=result.returncode, result_unchanged=True)
    _capture('result', lambda: (folder / 'result.json').write_text(
        json.dumps(receipt, indent=2) + '\n'))


def _run(*args, **kwargs):
    command = args[0] if args else kwargs.get('args')
    if not isinstance(command, (list, tuple)) or len(command) != 7 or Path(str(command[1])).name != 'launch_once.py' or str(command[3]) not in {'normal', 'safe'} or list(map(str, command[5:])) != ['blobs', 'data_art_impulse_lens']:
        return _original(*args, **kwargs)
    folder = _capture('prepare', lambda: _prepare(command, kwargs))
    try:
        result = _original(*args, **kwargs)
    except subprocess.TimeoutExpired as exc:
        _capture('timeout-evidence', lambda exc=exc: _save_result(folder, exc, timeout=True))
        raise
    else:
        _capture('completed-evidence', lambda: _save_result(folder, result))
        return result


def pytest_configure(config):
    subprocess.run = _run


def pytest_sessionfinish(session, exitstatus):
    if _errors:
        def write_errors():
            base = Path(os.environ['SPACR_SAFE_DIAGNOSTIC_DIR'])
            base.mkdir(parents=True, exist_ok=True)
            (base / f'collector-integrity-parent-{os.getpid()}.json').write_text(
                json.dumps({'errors': _errors}, indent=2) + '\n')
        _capture('integrity-receipt', write_errors)


def pytest_unconfigure(config):
    subprocess.run = _original
