"""The publication shell retries unavailable inventories without waiving errors."""
import os
from pathlib import Path
import subprocess

import pytest
import yaml


@pytest.mark.parametrize('mode,attempts,success', [
    ('transient', 2, True), ('unavailable', 3, False),
    ('invalid_reference', 1, False), ('missing_output', 1, False),
])
def test_strict_build_retries_only_inventory_fetch_failure(tmp_path, mode, attempts, success):
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / '.github/workflows/docs.yml').read_text())
    script = next(step['run'] for step in workflow['jobs']['build']['steps']
                  if step.get('name') == 'Build docs')
    script = script.replace('/tmp/sphinx', str(tmp_path / 'sphinx'))
    (tmp_path / 'tools').mkdir()
    capped = tmp_path / 'tools/run_capped.sh'
    capped.write_text('#!/bin/sh\ncase "$*" in *version) echo 1.5.0.9 ;; esac\n')
    capped.chmod(0o755)
    binary = tmp_path / 'bin'
    binary.mkdir()
    sphinx = binary / 'sphinx-build'
    sphinx.write_text('''#!/bin/sh
count=0
if [ -f "$COUNT_FILE" ]; then count=$(cat "$COUNT_FILE"); fi
count=$((count + 1))
printf '%s' "$count" > "$COUNT_FILE"
if [ "$MODE" = unavailable ] || { [ "$MODE" = transient ] && [ "$count" -eq 1 ]; }; then
  echo 'WARNING: failed to reach any of the inventories with the following issues:'
  echo "intersphinx inventory 'https://example.test/objects.inv' not fetchable"
  exit 1
fi
if [ "$MODE" = invalid_reference ]; then
  echo 'WARNING: unresolved internal API reference'
  exit 1
fi
if [ "$MODE" != missing_output ]; then
  mkdir -p docs/_build/html
  touch docs/_build/html/index.html
fi
exit 0
''')
    sphinx.chmod(0o755)
    result = subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', script],
                            cwd=tmp_path, capture_output=True, text=True,
                            env={**os.environ, 'PATH': str(binary) + os.pathsep + os.environ['PATH'],
                                 'DOCS_BRANCH': 'nightly', 'MODE': mode,
                                 'COUNT_FILE': str(tmp_path / 'attempts')})
    assert int((tmp_path / 'attempts').read_text()) == attempts, result.stdout
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    assert (tmp_path / 'docs/_build/html/.nojekyll').exists() is success
