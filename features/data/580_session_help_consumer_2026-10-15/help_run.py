import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
out = Path(__file__).resolve().parent
before = (root / 'spacr/qt/help_api_index.py').read_bytes()
(out / 'help-before.py').write_bytes(before)
expected = json.loads((root / 'features/data/580_generated_setting_links_2026-10-15/receipt.json').read_text())
for path, digest in expected['tracked_outputs_sha256'].items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
subprocess.run([sys.executable, 'tools/build_help_search_index.py'], check=True)
print('Normal Help generator terminal PASS; running exact module consumer.', flush=True)
subprocess.run([sys.executable, str(out / 'check_help.py')], check=True)
