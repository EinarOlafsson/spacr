import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
pin = 'a59e1f06f08c42cde033dd52f0194c2798e96f28'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == pin
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'tools'))
from spacr.qt.help_api_index import API_ENTRIES
from spacr.qt.help_index import api_entries, search
from spacr.qt.help_search import api_url, local_docstring
from build_help_search_index import _summary

module = 'spacr.qt.path_probe'
assert module not in sys.modules
generation = json.loads((root / 'features/data/678_path_probe_coordinator_review_2026-10-15/help-generation.json').read_text())
keys = generation['owned5_exact_English_summaries']
assert len(keys) == len(set(keys)) == 5
index_hash = hashlib.sha256((root / 'spacr/qt/help_api_index.py').read_bytes()).hexdigest()
assert index_hash == generation['output_sha256']
source_path = root / 'spacr/qt/path_probe.py'
source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
assert source_hash == '7a617ca9623c977c14f00801a78290d8a9bff0667d8217c2c7772022050178e4'
manifest = json.loads((root / 'docs/source/_static/i18n/api/en.json').read_text())['symbols']
rows = api_entries()
assert len(rows) >= len(API_ENTRIES)
checked = []
for key in keys:
    summaries = [summary for symbol, summary in API_ENTRIES if symbol == key]
    assert len(summaries) == 1, (key, summaries)
    summary = summaries[0]
    actual_rows = [row for row in rows if row.title == key]
    assert len(actual_rows) == 1 and actual_rows[0].payload['symbol'] == key
    assert actual_rows[0].kind == 'api' and actual_rows[0].description.startswith(summary)
    local = local_docstring(key)
    assert local and _summary(local) == summary == _summary(manifest[key]['text'])
    assert ' '.join(local.split()) == ' '.join(manifest[key]['text'].split()), key
    assert module not in sys.modules, 'Offline source reader imported its target'
    queries = (key, 'path_probe' if key == module else 'path_probe ' + key.rsplit('.', 1)[-1])
    ranks = []
    for query in queries:
        found = search(rows, query, limit=12)
        assert found and found[0].title == key, (key, query, [r.title for r in found])
        ranks.append({'query': query, 'rank': 1, 'top_three': [r.title for r in found[:3]]})
    checked.append({'symbol': key, 'summary': summary,
                    'local_doc_sha256': hashlib.sha256(local.encode()).hexdigest(),
                    'production_provider_rows': len(actual_rows), 'rankings': ranks})

before_url_target_imported = module in sys.modules
for row in checked:
    key = row['symbol']
    suffix = '/spacr/qt/path_probe/index.html'
    fragment = '' if key == module else '#' + key
    english = api_url(key, 'en')
    korean = api_url(key, 'ko')
    assert english.endswith(suffix + fragment), (key, english)
    assert korean.endswith(suffix + '?lang=ko' + fragment), (key, korean)
    row.update(english_url=english, korean_url=korean)
after_url_target_imported = module in sys.modules
if after_url_target_imported:
    assert sys.modules[module]._started is False, 'URL construction started path probe workers'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == pin
print(json.dumps({'result': 'PASS', 'source_pin': pin, 'checked_symbols': checked,
                  'help_index_sha256': index_hash, 'source_file_sha256': source_hash,
                  'actual_production_api_rows': len(rows), 'exact_ranking_queries': 10,
                  'target_imported_during_offline_source_read': before_url_target_imported,
                  'target_imported_after_url_dependency_loading': after_url_target_imported,
                  'path_probe_workers_started': False,
                  'scope': 'Five actual generated/production API rows, current full AST-read local prose and English summaries, module/function en+ko URL construction, and ten actual exact/qualified query rankings across the production API provider. No generator, canonical collector, network, Sphinx or rendered/native UI claim. api_url may load path_probe through existing settings_model/file_list imports; the offline prose reader does not.'}, indent=2))
