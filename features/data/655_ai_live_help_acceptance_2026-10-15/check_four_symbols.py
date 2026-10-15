import hashlib
import json
import subprocess
import sys
from pathlib import Path

from spacr.qt.help_api_index import API_ENTRIES
from spacr.qt.help_index import build_index, search
from spacr.qt.help_search import api_url, local_docstring

root = Path.cwd()
expected_commit, expected_index_sha = sys.argv[1:3]
actual_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert actual_commit == expected_commit, (actual_commit, expected_commit)
module = 'spacr.qt.widgets.ai_toggle_label'
keys = [module + '.AiToggleLabel.' + name for name in ('focusInEvent', 'focusOutEvent', 'keyPressEvent', 'paintEvent')]
index_hash = hashlib.sha256((root / 'spacr/qt/help_api_index.py').read_bytes()).hexdigest()
assert index_hash == expected_index_sha, (index_hash, expected_index_sha)
source_hash = hashlib.sha256((root / 'spacr/qt/widgets/ai_toggle_label.py').read_bytes()).hexdigest()
assert source_hash == 'f4fa6c499fcae7d519d6df6afd7fdcedcd0b2d9a7254066d50f711d867e2bc46'
manifest = json.loads((root / 'docs/source/_static/i18n/api/en.json').read_text())['symbols']
checks = []
for key in keys:
    assert module not in sys.modules, module
    matches = [summary for symbol, summary in API_ENTRIES if symbol == key]
    assert len(matches) == 1, (key, matches)
    summary = matches[0]
    local = local_docstring(key)
    assert local and local.startswith(summary), (key, summary, local)
    assert manifest[key]['text'].startswith(summary), key
    assert module not in sys.modules, module
    checks.append({'symbol': key, 'summary': summary, 'local_doc_sha256': hashlib.sha256(local.encode()).hexdigest()})
for check in checks:
    key = check['symbol']
    suffix = '/' + module.replace('.', '/') + '/index.html'
    english = api_url(key, 'en')
    korean = api_url(key, 'ko')
    assert english.endswith(suffix + '#' + key), (key, english)
    assert korean.endswith(suffix + '?lang=ko#' + key), (key, korean)
    check.update({'english_url': english, 'korean_url': korean})
rows = build_index()
queries = []
for key in keys:
    actual_rows = [row for row in rows if row.kind == 'api' and row.title == key]
    assert len(actual_rows) == 1, (key, len(actual_rows))
    assert actual_rows[0].payload['symbol'] == key
    for query in (key, key.removeprefix(module + '.')):
        found = search(rows, query, limit=12)
        assert found and found[0].title == key, (query, [row.title for row in found])
        queries.append({'query': query, 'first_result': found[0].title, 'result_count': len(found)})
print(json.dumps({'result': 'PASS', 'exact_tested_commit': actual_commit, 'checked_symbols': checks, 'production_search_queries': queries, 'production_index_row_count': len(rows), 'help_index_sha256': index_hash, 'widget_source_sha256': source_hash, 'scope': 'Four generated arrivals, English summaries, AST-only local prose without target import, English/Korean method URLs and eight full/qualified queries against the real production index. Consumer acceptance only; no generator, helper census, source review, network or rendered-HTML claim.'}, indent=2))
