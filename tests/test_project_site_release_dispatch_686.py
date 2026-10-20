"""Release-triggered site refresh waits for final published facts and fails visibly."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def job():
    workflow = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text(encoding="utf-8"))
    return workflow['jobs']['refresh-project-site']


def test_refresh_is_post_publication_and_does_not_gate_the_release(job):
    prerequisites = {'bump', 'verify-pypi', 'github-release', 'release-metadata'}
    assert set(job['needs']) == prerequisites
    for name in prerequisites:
        assert f"needs.{name}.result == 'success'" in job['if']
    assert "needs.bump.outputs.release_required == 'true'" in job['if']
    assert job['continue-on-error'] is True
    assert job['permissions'] == {'contents': 'read'}
    assert job['timeout-minutes'] == 5
    assert job['steps'][0]['env']['GH_TOKEN'] == '${{ secrets.SITE_DISPATCH_TOKEN }}'


@pytest.mark.parametrize('token,status', [('', 0), ('test-token', 0), ('test-token', 1)])
def test_actual_step_skips_missing_credentials_and_sends_one_release_event(
        job, tmp_path, token, status):
    executable = tmp_path / 'gh'
    executable.write_text(f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
Path(os.environ['DISPATCH_LOG']).write_text(json.dumps(sys.argv[1:]))
sys.exit(int(os.environ['DISPATCH_STATUS']))
''')
    executable.chmod(0o755)
    summary = tmp_path / 'summary'
    request = tmp_path / 'request.json'
    environment = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'],
                   'GH_TOKEN': token, 'VERSION': '1.5.1.5', 'GITHUB_STEP_SUMMARY': str(summary),
                   'DISPATCH_LOG': str(request), 'DISPATCH_STATUS': str(status)}
    result = subprocess.run(['bash', '-e', '-c', job['steps'][0]['run']],
                            env=environment, capture_output=True, text=True, timeout=10)
    assert token == '' or token not in result.stdout + result.stderr
    if not token:
        assert result.returncode == 0
        assert not request.exists()
        assert 'SITE_DISPATCH_TOKEN' in summary.read_text(encoding="utf-8")
    else:
        assert result.returncode == status
        assert json.loads(request.read_text(encoding="utf-8")) == [
            'api', '--method', 'POST', 'repos/EinarOlafsson/site-src/dispatches',
            '-f', 'event_type=spacr-release', '-f', 'client_payload[version]=1.5.1.5',
            '-f', 'client_payload[tag]=v1.5.1.5']
        if status:
            assert not summary.exists()
        else:
            assert 'requested rebuild' in summary.read_text(encoding="utf-8")
            assert 'publication is verified by the site workflow' in summary.read_text(encoding="utf-8")
