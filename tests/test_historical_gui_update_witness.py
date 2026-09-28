"""Audit-witness regressions only; these tests do not establish native acceptance.

The Windows strings follow CPython's real subprocess audit serialization. Pure
observer tests load the actual functions without entering installed-state or GUI
modes. One isolated smoke child observes a real harmless subprocess execution.
"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


TOOL = Path(__file__).resolve().parents[1] / 'tools/accept_historical_upgrade.py'
PYTHON = r'C:\Users\Runner Admin\AppData\Local\SpaCR\venv\Scripts\python.exe'
UV = r'C:\Users\Runner Admin\AppData\Local\SpaCR\bootstrap\uv.exe'
PIP_COMMAND = [PYTHON, '-m', 'pip', 'install', '--upgrade', 'spacr']
UV_COMMAND = [UV, 'pip', 'install', '--upgrade', '--python', PYTHON, 'spacr']
WINDOWS_COMMANDS = (
    (PIP_COMMAND, f'"{PYTHON}" -m pip install --upgrade spacr'),
    (UV_COMMAND, f'"{UV}" pip install --upgrade --python "{PYTHON}" spacr'),
)


def _load_witness(expected):
    """Load the real observer functions with expected argv and return their globals."""
    tree = ast.parse(TOOL.read_text(encoding='utf-8'), filename=str(TOOL))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name in ('_observed_update_command', 'observe')]
    assert any(node.name == 'observe' for node in functions), 'Expected the actual installed observer'
    namespace = dict(os=os, subprocess=subprocess, expected_update_command=expected, commands=[])
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(TOOL), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize(('expected', 'audited'), WINDOWS_COMMANDS)
def test_windows_string_audit_records_exact_installed_update_command(expected, audited):
    """Both real missing-pip and repaired-uv routes survive Windows string auditing."""
    assert audited == subprocess.list2cmdline(expected)
    witness = _load_witness(expected)
    witness['observe']('subprocess.Popen', (None, audited, None, None))
    assert witness['commands'] == [expected]


@pytest.mark.parametrize(('expected', 'audited'), WINDOWS_COMMANDS)
@pytest.mark.parametrize('container', (list, tuple))
@pytest.mark.parametrize('encoded', (False, True))
def test_sequence_audit_keeps_existing_exact_command_behavior(expected, audited, container, encoded):
    """POSIX list/tuple events retain their observed arguments and byte decoding."""
    values = [os.fsencode(value) for value in expected] if encoded else expected
    witness = _load_witness(expected)
    witness['observe']('subprocess.Popen', (expected[0], container(values), None, None))
    assert witness['commands'] == [expected]


@pytest.mark.parametrize(('expected', 'audited'), WINDOWS_COMMANDS)
@pytest.mark.parametrize('change', ('executable', 'extra_flag', 'different_package', 'noncanonical_quotes'))
def test_unexpected_windows_update_string_remains_rejectable(expected, audited, change):
    """A relevant altered command is retained raw, never relabeled as expected argv."""
    altered = list(expected)
    if change == 'executable':
        altered[0] = r'C:\Other runtime\python.exe'
        unexpected = subprocess.list2cmdline(altered)
    elif change == 'extra_flag':
        unexpected = subprocess.list2cmdline(altered + ['--extra-index-url', 'https://example.invalid/'])
    elif change == 'different_package':
        altered[-1] = 'spacr==1.5.0.6'
        unexpected = subprocess.list2cmdline(altered)
    else:
        unexpected = audited.removesuffix('spacr') + '"spacr"'
    witness = _load_witness(expected)
    witness['observe']('subprocess.Popen', (None, unexpected, None, None))
    assert witness['commands'] == [[unexpected]]
    assert witness['commands'] != [expected], 'The existing exact-command guard must reject this'


@pytest.mark.parametrize(('expected', 'audited'), WINDOWS_COMMANDS)
def test_additional_windows_update_child_is_not_hidden(expected, audited):
    """An unexpected second update child still makes the one-child guard reject."""
    unexpected = audited + ' --no-deps'
    witness = _load_witness(expected)
    witness['observe']('subprocess.Popen', (None, audited, None, None))
    witness['observe']('subprocess.Popen', (None, unexpected, None, None))
    assert witness['commands'] == [expected, [unexpected]]
    assert len(witness['commands']) != 1


@pytest.mark.parametrize(('event', 'argv'), (
    ('unrelated.audit.event', None),
    ('subprocess.Popen', 'git --version'),
    ('subprocess.Popen', ['python', '-c', 'pass']),
))
def test_unrelated_audit_events_do_not_invent_update_children(event, argv):
    """Events without an actual update command leave the child witness empty."""
    witness = _load_witness(PIP_COMMAND)
    witness['observe'](event, (None, argv, None, None))
    assert witness['commands'] == []


def test_real_subprocess_audit_is_observed_without_replacing_execution():
    """An isolated real harmless child confirms the actual audit hook is wired."""
    script = r'''
import ast, json, os, pathlib, subprocess, sys
tool = pathlib.Path(sys.argv[1])
tree = ast.parse(tool.read_text(encoding='utf-8'), filename=str(tool))
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
    and node.name in ('_observed_update_command', 'observe')]
expected = [sys.executable, '-c', 'pass', 'install', '--upgrade', 'spacr']
namespace = dict(os=os, subprocess=subprocess, expected_update_command=expected, commands=[])
exec(compile(ast.Module(body=functions, type_ignores=[]), str(tool), 'exec'), namespace)
sys.addaudithook(namespace['observe'])
completed = subprocess.run(expected, check=True, capture_output=True, timeout=10)
assert completed.returncode == 0 and completed.stdout == b'' and completed.stderr == b''
assert namespace['commands'] == [expected], namespace['commands']
print(json.dumps(namespace['commands']))
'''
    completed = subprocess.run([sys.executable, '-I', '-c', script, str(TOOL)],
        check=True, capture_output=True, text=True, timeout=20)
    assert json.loads(completed.stdout) == [[sys.executable, '-c', 'pass', 'install', '--upgrade', 'spacr']]
