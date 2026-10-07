import gzip
import hashlib
import json
import subprocess
from pathlib import Path

root = Path('/mnt/wd4tb/spacr-worktrees/codex-cpu-final-20261006')
scratch = Path('/mnt/wd4tb/scratch/qt-exclude-owner-20261007')
out = root / 'features/data/47_exclude_settings_owner_2026-10-07'
out.mkdir(parents=True, exist_ok=True)
parent = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
test = 'tests/qt/test_exclude_multi_column.py'
assert subprocess.check_output(['git', 'show', f'{parent}:{test}'], cwd=root) == (scratch / 'before.py').read_bytes()
records = {}
for name in ('before', 'after'):
 log = (scratch / f'{name}.log').read_text()
 assert '21 passed' in log and 'failed' not in log
 rows = [json.loads(line) for line in (scratch / f'{name}.jsonl').read_text().splitlines()]
 assert rows[-1]['exitstatus'] == 0
 records[name] = [row for row in rows if row['event'] == 'file_end']
 for suffix in ('jsonl', 'log'):
  source = scratch / f'{name}.{suffix}'
  (out / f'{name}.{suffix}.gz').write_bytes(gzip.compress(source.read_bytes(), mtime=0))
(out / 'before.py.gz').write_bytes(gzip.compress((scratch / 'before.py').read_bytes(), mtime=0))
paths = [test, 'tests/qt/test_field_fade.py', 'spacr/qt/screens/settings_model.py', 'tests/conftest.py', 'tools/pytest_plugins/qt_serial_rss_journal.py']
receipt = {
 'source_parent': parent,
 'test_before_sha256': hashlib.sha256((scratch / 'before.py').read_bytes()).hexdigest(),
 'source_sha256': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
 'passed_before': 21, 'passed_after': 21, 'seconds_before': 5.74, 'seconds_after': 5.98,
 'hard_memory_cap': '4G', 'CUDA_VISIBLE_DEVICES': '', 'QT_QPA_PLATFORM': 'offscreen',
 'selection': [test, 'tests/qt/test_field_fade.py::test_the_field_owner_outlives_its_native_painter'],
 'file_end_records': records,
 'scope': 'Matched bounded file and sentinel. Qt owners retire temporary controls; no production, assertion, collection-order, forced-GC or memory-limit change. Approximately0.3MiB RSS difference is not meaningful whole-suite savings or native-crash acceptance.'
}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(out / 'archive.py').write_bytes(Path(__file__).read_bytes())
(out / 'manifest.json').write_text(json.dumps({'files': {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.iterdir()) if p.name != 'manifest.json'}}, indent=2) + '\n')
print(json.dumps({'archive': str(out), 'files': len(list(out.iterdir())), 'passed':21}))
