import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import build_documentation_i18n as api
import check_reviewed_api_evidence as reporter


def prepare(monkeypatch, tmp_path, source, index=0):
    monkeypatch.setattr(reporter, 'ROOT', tmp_path)
    monkeypatch.setattr(api, 'public_docstrings', lambda: {'spacr.example': source})
    root = tmp_path / 'docs/i18n/reviewed/api'
    record = {'label': f'spacr.example#{index}', 'source': 'Original paragraph.',
              'source_sha256': api._source_hash('Original paragraph.')}
    active = root / 'sv/current.json'
    active.parent.mkdir(parents=True)
    active.write_text(json.dumps({'records': [record]}))
    archived = root / '_archive/sv/previous/historical.json'
    archived.parent.mkdir(parents=True)
    archived.write_text(json.dumps({'records': [{'label': 'spacr.example#0',
                                              'source': 'Archived obsolete paragraph.',
                                              'source_sha256': 'obsolete'}]}))


def test_archived_reviews_do_not_override_current_evidence(monkeypatch, tmp_path, capsys):
    prepare(monkeypatch, tmp_path, 'Original paragraph.')
    assert reporter.main() == 0
    assert 'still matches its docstring' in capsys.readouterr().out


def test_active_moved_review_still_fails_with_rebind_location(monkeypatch, tmp_path, capsys):
    prepare(monkeypatch, tmp_path, 'New paragraph.\n\nOriginal paragraph.')
    assert reporter.main() == 1
    output = capsys.readouterr().out
    assert 'MOVED' in output and 're-bind to #1' in output
    assert 'Archived obsolete paragraph.' not in output


def test_active_rewritten_review_still_fails(monkeypatch, tmp_path, capsys):
    prepare(monkeypatch, tmp_path, 'Rewritten paragraph.')
    assert reporter.main() == 1
    assert 'STALE' in capsys.readouterr().out
