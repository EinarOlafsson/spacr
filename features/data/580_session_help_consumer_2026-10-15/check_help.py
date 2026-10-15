"""Consume the normally generated F580 Help module entry and setting links."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
key = 'spacr.qt.widgets.calibration_preview'
source = "Read-only preview of Measure's plate and acquisition-session calibration plans."
source_path = ROOT / 'spacr/qt/widgets/calibration_preview.py'
assert sha(source_path.read_bytes()) == '699605f2d1e8b4d8d1d2d0c42a9a79373a3e95bdddd5055f456a3c8c8267b94d'

def assignments(raw):
    return {node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(raw).body if isinstance(node, ast.Assign)
            and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}

before = assignments((OUT / 'help-before.py').read_bytes())
after_bytes = (ROOT / 'spacr/qt/help_api_index.py').read_bytes()
after = assignments(after_bytes)
old_api = dict(before['API_ENTRIES'])
new_api = dict(after['API_ENTRIES'])
assert set(old_api) == set(new_api)
assert new_api[key] == source
assert {k: v for k, v in old_api.items() if k != key} == {k: v for k, v in new_api.items() if k != key}
assert before['PREFERENCE_ENTRIES'] == after['PREFERENCE_ENTRIES']
keys = ('intensity_calibration_sessions', 'intensity_calibration_session_column',
        'intensity_calibration_beads', 'intensity_calibration_reference_session')
changed_consumers = {k: {'before': before['SETTING_CONSUMERS'].get(k), 'after': after['SETTING_CONSUMERS'].get(k)}
    for k in set(before['SETTING_CONSUMERS']) | set(after['SETTING_CONSUMERS'])
    if before['SETTING_CONSUMERS'].get(k) != after['SETTING_CONSUMERS'].get(k)}
assert set(changed_consumers) == set(keys), changed_consumers
for name in keys:
    assert after['SETTING_CONSUMERS'][name] == (('spacr.settings', 'get_measure_crop_settings'),)

from spacr.qt.help_search import api_url, local_docstring
assert key not in sys.modules
local = local_docstring(key)
assert local == source
assert key not in sys.modules
english = api_url(key, 'en')
korean = api_url(key, 'ko')
suffix = '/spacr/qt/widgets/calibration_preview/index.html'
assert english.endswith(suffix), english
assert korean.endswith(suffix + '?lang=ko'), korean

from spacr.qt.help_index import build_index, search
rows = build_index()
entries = [row for row in rows if row.kind == 'api' and row.title == key]
assert len(entries) == 1 and entries[0].payload['symbol'] == key
queries = []
for query in (key, 'calibration_preview'):
    found = search(rows, query, limit=12)
    assert found and found[0].title == key, (query, [row.title for row in found])
    queries.append({'query': query, 'first_result': found[0].title})
manifest = json.loads((ROOT / 'docs/source/_static/i18n/api/en.json').read_text())
assert manifest['symbols'][key]['text'] == source
result = {'status': 'PASS', 'consumer_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'module': key, 'summary': source, 'local_docstring': local,
    'english_url': english, 'korean_url': korean, 'queries': queries,
    'api_entry_count': len(new_api), 'unrelated_api_summaries_preserved': len(new_api) - 1,
    'preference_rows_preserved': len(after['PREFERENCE_ENTRIES']), 'changed_setting_consumers': changed_consumers,
    'help_before_sha256': sha((OUT / 'help-before.py').read_bytes()), 'help_after_sha256': sha(after_bytes),
    'source_sha256': sha(source_path.read_bytes()),
    'scope': 'Production search, exact module summary, AST-only local prose before URL/index calls, and English/Korean module routes. No rendered-page, network or native-user acceptance.'}
(OUT / 'help-consumer.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
