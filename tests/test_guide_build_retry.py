"""Strict translated guides retry an inventory outage without waiving defects."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import zlib

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import build_guide_i18n as guide


INVENTORY_WARNING = (
    'WARNING: failed to reach any of the inventories with the following issues:\n'
    "intersphinx inventory 'https://example.test/objects.inv' not fetchable\n")


@pytest.mark.parametrize('outcomes, attempts, expected', [
    ([(0, '')], 1, 0),
    ([(1, INVENTORY_WARNING), (0, '')], 2, 0),
    ([(1, INVENTORY_WARNING)] * 3, 3, 1),
    ([(1, 'WARNING: unknown document: missing\n')], 1, 1),
    ([(1, INVENTORY_WARNING + 'WARNING: unknown document: missing\n')], 1, 1),
    ([(2, INVENTORY_WARNING)], 1, 2),
    ([(-11, INVENTORY_WARNING)], 1, -11),
    ([(1, INVENTORY_WARNING + 'ERROR: broken configuration\n')], 1, 1),
])
def test_only_inventory_failure_retries_with_all_strict_flags(
        monkeypatch, tmp_path, capsys, outcomes, attempts, expected):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        code, output = outcomes[len(calls) - 1]
        return SimpleNamespace(returncode=code, stdout=output)

    monkeypatch.setattr(guide.subprocess, 'run', run)
    result = guide.build('sv', tmp_path / 'sv', tmp_path / 'objects.inv')
    assert result == expected and len(calls) == attempts
    for command, options in calls:
        assert '-W' in command and '--keep-going' in command
        assert 'language=sv' in command and '-E' in command
        assert options['env']['SPACR_DOCS_GUIDES_ONLY'] == '1'
        assert options['env']['SPACR_DOCS_ENGLISH_INVENTORY'] == str(tmp_path / 'objects.inv')
    output = capsys.readouterr().out
    assert output.count('retrying unavailable external API inventory') == attempts - 1


def test_a_real_sphinx_guide_recovers_from_one_local_inventory_outage(
        tmp_path, monkeypatch, capsys):
    pytest.importorskip('sphinx')
    inventory = (
        b'# Sphinx inventory version 2\n# Project: demo\n# Version: 1\n'
        b'# The remainder of this file is compressed using zlib.\n'
        + zlib.compress(b'example py:class 1 example.html#example -\n'))
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if len(requests) == 1:
                self.send_error(503, 'temporary inventory outage')
                return
            self.send_response(200)
            self.send_header('Content-Length', str(len(inventory)))
            self.end_headers()
            self.wfile.write(inventory)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = tmp_path / 'source'
    source.mkdir()
    url = f'http://127.0.0.1:{server.server_port}'
    (source / 'conf.py').write_text(
        "extensions = ['sphinx.ext.intersphinx']\n"
        f"intersphinx_mapping = {{'demo': ({url!r}, None)}}\n"
        "intersphinx_timeout = 1\nmaster_doc = 'index'\n", encoding='utf-8')
    (source / 'index.rst').write_text(
        'Example\n=======\n\n:external+demo:py:class:`example`\n', encoding='utf-8')
    monkeypatch.setattr(guide, 'ROOT', tmp_path)
    monkeypatch.setattr(guide, 'SOURCE_DIR', source)
    try:
        result = guide.build('sv', tmp_path / 'html', tmp_path / 'objects.inv')
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result == 0
    assert requests == ['/objects.inv', '/objects.inv']
    assert (tmp_path / 'html/index.html').is_file()
    assert 'retrying unavailable external API inventory' in capsys.readouterr().out


@pytest.mark.parametrize('second, expected', [(SimpleNamespace(returncode=0, stdout=''), 0),
    (SimpleNamespace(returncode=1, stdout="WARNING: inventory for external cross-reference not found: 'unknown'\n"), 1)])
def test_dependent_inventory_warning_retries_but_a_remaining_bad_reference_fails(
        monkeypatch, tmp_path, second, expected):
    responses = [SimpleNamespace(returncode=1, stdout=INVENTORY_WARNING
        + "WARNING: inventory for external cross-reference not found: 'demo'\n"), second]
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return responses[len(calls) - 1]

    monkeypatch.setattr(guide.subprocess, 'run', run)
    assert guide.build('sv', tmp_path / 'sv', tmp_path / 'objects.inv') == expected
    assert len(calls) == 2
    assert all('-W' in command and '--keep-going' in command for command in calls)
