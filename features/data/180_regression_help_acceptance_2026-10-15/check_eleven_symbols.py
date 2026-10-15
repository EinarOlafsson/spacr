import hashlib
import json
import sys
from pathlib import Path

from spacr.qt.help_api_index import API_ENTRIES
from spacr.qt.help_index import HelpEntry, search
from spacr.qt.help_search import api_url, local_docstring

root = Path.cwd()
evidence = root / 'features/data/180_regression_coordinator_review_2026-10-15'
readback = json.loads((evidence / 'catalog-readback.json').read_text())
generated = json.loads((evidence / 'help-generation.json').read_text())
keys = [key for batch in readback['batches'].values() for key in batch['symbols']]
assert len(keys) == len(set(keys)) == 11
index_hash = hashlib.sha256((root / 'spacr/qt/help_api_index.py').read_bytes()).hexdigest()
assert index_hash == generated['output_sha256']
manifest = json.loads((root / 'docs/source/_static/i18n/api/en.json').read_text())['symbols']
rows = []
checks = []
modules = ('spacr.qt.job_runner', 'spacr.qt.widgets.regression_results')
for key in keys:
    module = next(module for module in modules if key == module or key.startswith(module + '.'))
    assert module not in sys.modules, module
    matches = [summary for symbol, summary in API_ENTRIES if symbol == key]
    assert len(matches) == 1, (key, matches)
    summary = matches[0]
    local = local_docstring(key)
    assert local and local.startswith(summary), (key, summary, local)
    assert manifest[key]['text'].startswith(summary), key
    suffix = '/' + module.replace('.', '/') + '/index.html'
    anchor = '' if key == module else '#' + key
    english = api_url(key, 'en')
    korean = api_url(key, 'ko')
    assert english.endswith(suffix + anchor), (key, english)
    assert korean.endswith(suffix + '?lang=ko' + anchor), (key, korean)
    assert module not in sys.modules, module
    rows.append(HelpEntry(kind='api', title=key, subtitle='API reference', description=summary, payload={'symbol': key}))
    checks.append({'symbol': key, 'summary': summary, 'local_doc_sha256': hashlib.sha256(local.encode()).hexdigest(), 'english_url': english, 'korean_url': korean})
for row in rows:
    found = search(rows, row.title, limit=12)
    assert found and found[0].title == row.title, row.title
print(json.dumps({'result': 'PASS', 'checked_symbols': checks, 'help_index_sha256': index_hash, 'target_modules_imported': [module for module in modules if module in sys.modules], 'scope': 'Eleven specific generated entries, English summaries, AST-only local prose, module/method en+ko URLs and exact-name ranking. No full generator, helper census, API audit, network or rendered-HTML claim.'}, indent=2))
