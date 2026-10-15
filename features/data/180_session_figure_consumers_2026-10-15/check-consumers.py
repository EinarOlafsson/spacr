"""Run only after coherent N180 catalogs and normal Help generation are frozen."""
import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--expected-commit', required=True)
parser.add_argument('--expected-help-sha256', required=True)
parser.add_argument('--before-help', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
sys.path.insert(0, str(root))
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
assert head == args.expected_commit, (head, args.expected_commit)
hash_bytes = lambda value: hashlib.sha256(value).hexdigest()
help_path = root / 'spacr/qt/help_api_index.py'
assert hash_bytes(help_path.read_bytes()) == args.expected_help_sha256
packet = root / 'features/data/180_session_figure_inputs_2026-10-15'
manifest = json.loads((packet / 'sources.json').read_text())
for filename, expected in manifest['source_files'].items():
    assert hash_bytes((root / filename).read_bytes()) == expected, filename
module = 'spacr.qt.widgets.figure_queue'
symbols = manifest['new_symbols']
assert len(symbols) == 2
languages = ('sv', 'de', 'es', 'fr', 'pt', 'is', 'zh_CN', 'hi', 'ko')

def index_entries(path):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'API_ENTRIES' for t in node.targets):
            return dict(ast.literal_eval(node.value))
    raise AssertionError(('API_ENTRIES missing', str(path)))

before = index_entries(args.before_help)
from spacr.qt.help_api_index import API_ENTRIES
from spacr.qt.help_search import api_url, local_docstring

after = dict(API_ENTRIES)
assert len(after) == len(API_ENTRIES)
assert set(after) == set(before) | set(symbols)
assert all(after[key] == value for key, value in before.items())
assert not set(before).intersection(symbols)
english = json.loads((root / 'docs/source/_static/i18n/api/en.json').read_text())['symbols']
local_checks = []
assert module not in sys.modules, 'No-import boundary starts before production URL/search helpers'
for symbol in symbols:
    local = local_docstring(symbol)
    assert local == manifest['source_docs'][symbol], symbol
    assert module not in sys.modules, symbol
    assert english[symbol]['text'] == local, symbol
    assert local.startswith(after[symbol]), symbol
    local_checks.append({'symbol': symbol, 'summary': after[symbol], 'local_sha256': hash_bytes(local.encode())})

from spacr.qt.i18n import tr

caption_checks = []
api_checks = []
for language in languages:
    runtime_file = root / f'docs/i18n/reviewed/runtime/{language}/2026-10-15-n180-session-figures.json'
    records = json.loads(runtime_file.read_text())['records']
    assert len(records) == 1
    record = records[0]
    assert record['source_sha256'] == hash_bytes(record['source'].encode())
    actual = tr(record['source'], language=language)
    assert actual == record['translation'], (language, actual, record['translation'])
    caption_checks.append({'language': language, 'source_sha256': record['source_sha256'], 'target_sha256': hash_bytes(actual.encode())})
    review = json.loads((root / f'docs/i18n/reviewed/api/{language}/2026-10-15-n180-session-figures.json').read_text())['records']
    catalog = json.loads((root / f'docs/source/_static/i18n/api/{language}.json').read_text())['symbols']
    assert len(review) == 7
    for symbol in symbols:
        chosen = [row for row in review if row['label'].rsplit('#', 1)[0] == symbol]
        value = catalog[symbol]
        assert value['source_sha256'] == english[symbol]['source_sha256']
        assert value['source_blocks_sha256'] == [row['source_sha256'] for row in chosen]
        assert value['translation_source_blocks_sha256'] == [row['context_sha256'] for row in chosen]
        normalized = ' '.join(value['text'].split())
        for row in chosen:
            assert ' '.join(row['translation'].split()) in normalized, (language, row['label'])
        api_checks.append({'language': language, 'symbol': symbol, 'text_sha256': hash_bytes(value['text'].encode())})

for row in local_checks:
    symbol = row['symbol']
    suffix = '/' + module.replace('.', '/') + '/index.html'
    en_url, ko_url = api_url(symbol, 'en'), api_url(symbol, 'ko')
    assert en_url.endswith(suffix + '#' + symbol), en_url
    assert ko_url.endswith(suffix + '?lang=ko#' + symbol), ko_url
    row.update({'english_url': en_url, 'korean_url': ko_url})

from spacr.qt.help_index import build_index, search

entries = build_index()
queries = []
for symbol in symbols:
    matches = [entry for entry in entries if entry.kind == 'api' and entry.title == symbol]
    assert len(matches) == 1 and matches[0].payload['symbol'] == symbol, symbol
    for query in (symbol, symbol.removeprefix(module + '.')):
        found = search(entries, query, limit=12)
        assert found and found[0].title == symbol, (query, [entry.title for entry in found])
        queries.append({'query': query, 'first_result': found[0].title})
print(json.dumps({'result': 'PASS', 'commit': head, 'help_sha256': args.expected_help_sha256, 'prior_help_entries_preserved': len(before), 'added_help_entries': 2, 'caption_checks': caption_checks, 'selected_localized_api_records': api_checks, 'local_api_and_routes': local_checks, 'production_search_queries': queries, 'scope': 'Nine real compiled-runtime tr lookups, eighteen selected API catalog records, two AST-only local prose lookups before legitimate production URL/search imports, English/Korean anchors and four production searches. No dialog/openers/network, generator, collector or source-test replay.'}, indent=2))
