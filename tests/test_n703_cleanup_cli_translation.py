"""New cleanup command help/errors use current reviews without requiring Qt in a helper."""
import ast
import builtins
import json
from pathlib import Path

import pytest

from spacr import install_cleanup as ic

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'])
def test_actual_cli_help_and_error_templates_use_exact_reviewed_locale(monkeypatch, capsys, language):
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    records = json.loads((ROOT / 'docs/i18n/reviewed/runtime' / language /
                          '2026-10-14-n703-uninstall-cli.json').read_text())['records']
    assert len(records) == 9
    expected = {row['source']: row['translation'] for row in records}
    for source, target in expected.items():
        assert ic.tr(source, error='diagnostic-703') == target.format(error='diagnostic-703')
        assert source != target
    commands = {
        'remove': ['remove only this recognized installation, preserving its data',
                   'require an interpreter outside the selected root'],
        'recover-runtime': ['previous desktop process that must have exited'],
        None: ['restore one unchanged retained private POSIX runtime',
               'copy the private managed Python outside this installation'],
    }
    for command, sources in commands.items():
        with pytest.raises(SystemExit) as stopped:
            ic._main(['--help'] if command is None else [command, '--help'])
        assert stopped.value.code == 0
        help_text = ' '.join(capsys.readouterr().out.split())
        for source in sources:
            assert ' '.join(expected[source].split()) in help_text


def test_extracted_helper_still_formats_message_when_application_cannot_import(monkeypatch):
    original = builtins.__import__
    def without_application(name, *args, **kwargs):
        if name == 'spacr.qt.i18n':
            raise ImportError('No application or Qt in the private standalone helper')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_application)
    assert ic.tr('Runtime recovery stopped: {error}', error='retained') == 'Runtime recovery stopped: retained'


def test_each_added_cli_caption_is_a_literal_translation_call():
    tree = ast.parse(Path(ic.__file__).read_text())
    calls = {node.args[0].value for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == 'tr' and node.args and isinstance(node.args[0], ast.Constant)}
    records = json.loads((ROOT / 'docs/i18n/reviewed/runtime/sv/2026-10-14-n703-uninstall-cli.json').read_text())['records']
    assert {row['source'] for row in records} <= calls
