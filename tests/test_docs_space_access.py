"""A public Space read must not be mistaken for CI token write access."""
import hashlib
import base64
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import subprocess

import huggingface_hub
from huggingface_hub.hf_api import RepoFile, RepoFolder
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import publish_docs_channels as publisher


@pytest.fixture
def service(tmp_path, monkeypatch):
    raw = b'---\nsdk: static\n---\nDocumentation\n'
    readme = tmp_path / 'README.md'
    readme.write_bytes(raw)
    oid = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
    entries = [RepoFile(path='README.md', size=len(raw), oid=oid),
               RepoFile(path='index.html', size=4, oid='b' * 40),
               RepoFolder(path='api', oid='c' * 40)]
    calls = []

    class Api:
        denied = False
        changed = False
        revision = 'a' * 40
        sdk = 'static'
        token = 'private-test-sentinel'
        endpoint = 'https://huggingface.co'

        def space_info(self, space):
            calls.append(('info', space))
            return SimpleNamespace(sha=self.revision, sdk=self.sdk)

        def list_repo_tree(self, space, **kwargs):
            calls.append(('tree', space, kwargs))
            if self.changed and kwargs['revision'] == 'd' * 40:
                return entries + [RepoFile(path='unexpected', size=0, oid='e' * 40)]
            return entries

        def create_commit(self, **kwargs):
            pytest.fail('SDK can skip unchanged files; write proof must reach actual commit POST')

    api = Api()
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: api)

    class Response(io.BytesIO):
        status = 200

    class Opener:
        def open(self, request, *, timeout):
            calls.append(('commit', request))
            assert request.full_url == 'https://huggingface.co/api/spaces/einarolafsson/spacr-docs-nightly-v2/commit/main'
            assert request.get_header('Authorization') == 'Bearer private-test-sentinel'
            assert request.get_header('Content-type') == 'application/x-ndjson'
            assert timeout == 30 and request.get_method() == 'POST'
            header, operation = map(json.loads, request.data.splitlines())
            assert header['key'] == 'header' and header['value']['parentCommit'] == 'a' * 40
            assert operation['key'] == 'file'
            assert operation['value']['path'] == 'README.md'
            assert operation['value']['encoding'] == 'base64'
            assert base64.b64decode(operation['value']['content']) == raw
            if api.denied:
                raise PermissionError('Public read succeeded; token cannot write')
            return Response(json.dumps({'commitOid': 'd' * 40}).encode())

    def opener(handler):
        assert handler.redirect_request(None, None, 302, None, None, 'https://other.invalid') is None
        return Opener()

    monkeypatch.setattr(publisher, 'build_opener', opener)

    def download(**kwargs):
        calls.append(('download', kwargs))
        assert kwargs == {'repo_id': 'einarolafsson/spacr-docs-nightly-v2',
                          'repo_type': 'space', 'filename': 'README.md',
                          'revision': 'a' * 40}
        return str(readme)

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    return api, calls, readme, raw


def test_write_access_uses_only_existing_pinned_readme_and_checks_unchanged_tree(service):
    api, calls, readme, raw = service
    result = publisher.verify_space_access('einarolafsson/spacr-docs-nightly-v2')
    assert result['revision'] == 'd' * 40
    assert result['parent_revision'] == 'a' * 40
    assert result['tree_unchanged'] is True and result['files_verified'] == 2
    assert result['readme_sha256'] == hashlib.sha256(raw).hexdigest()
    assert [name for name, *_ in calls] == ['info', 'tree', 'download', 'commit', 'tree']


def test_public_read_does_not_pass_when_existing_ci_token_cannot_write(service):
    api, calls, *_ = service
    api.denied = True
    with pytest.raises(PermissionError, match='cannot write'):
        publisher.verify_space_access('einarolafsson/spacr-docs-nightly-v2')
    assert calls[-1][0] == 'commit'


def test_changed_resulting_tree_is_not_reported_as_verified_access(service):
    api, *_ = service
    api.changed = True
    with pytest.raises(ValueError, match='tree changed'):
        publisher.verify_space_access('einarolafsson/spacr-docs-nightly-v2')


@pytest.mark.parametrize('change', ['cache-bytes', 'mutable-parent', 'non-static'])
def test_invalid_parent_or_readme_refuses_before_commit(service, change):
    api, calls, readme, _ = service
    if change == 'cache-bytes':
        readme.write_bytes(b'changed cache')
    elif change == 'mutable-parent':
        api.revision = 'main'
    else:
        api.sdk = 'gradio'
    with pytest.raises(ValueError):
        publisher.verify_space_access('einarolafsson/spacr-docs-nightly-v2')
    assert not any(call[0] == 'commit' for call in calls)


def test_cli_receipt_and_failure_never_echo_credential_exception_text(service, capsys):
    api, *_ = service
    assert publisher.main(['verify-space-access', '--space',
                           'einarolafsson/spacr-docs-nightly-v2']) == 0
    assert json.loads(capsys.readouterr().out)['tree_unchanged'] is True
    api.denied = True
    assert publisher.main(['verify-space-access', '--space',
                           'einarolafsson/spacr-docs-nightly-v2']) == 1
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err)['error'] == 'PermissionError'
    assert 'Public read succeeded' not in output.err


@pytest.mark.parametrize('target', ['other/private', 'einarolafsson/models',
                                   'https://huggingface.co/owner/docs'])
def test_non_documentation_target_is_refused_before_service_access(service, target):
    _, calls, *_ = service
    with pytest.raises(ValueError, match='documentation Space'):
        publisher.verify_space_access(target)
    assert calls == []


def test_actual_workflow_isolates_access_check_from_all_build_and_deploy_jobs():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / '.github/workflows/docs.yml').read_text(encoding="utf-8"))
    jobs = workflow['jobs']
    probe = jobs['verify-host-access']
    assert probe['if'] == "github.event_name == 'workflow_dispatch' && inputs.mode == 'verify-host-access'"
    assert jobs['resolve']['if'] == "github.event_name != 'workflow_dispatch' || inputs.mode == 'publish'"
    assert jobs['build']['needs'] == 'resolve'
    for name in ['deploy-pages', 'deploy-space']:
        assert {'resolve', 'build'} <= set(jobs[name]['needs'])
    assert probe['env']['HF_TOKEN'] == '${{ secrets.HF_DOCS_TOKEN }}'
    assert probe['concurrency']['group'] == jobs['deploy-space']['concurrency']['group']
    assert probe['timeout-minutes'] == 10
    scripts = '\n'.join(step.get('run', '') for step in probe['steps'])
    assert 'verify-space-access' in scripts
    assert not any(command in scripts for command in ['sphinx', 'upload-nightly', 'assemble-', 'create_repo'])
    assert 'HF_TOKEN' not in probe['steps'][-1]['run']


def test_actual_workflow_does_not_mask_failed_access_check_behind_tee(tmp_path):
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / '.github/workflows/docs.yml').read_text(encoding="utf-8"))
    probe = workflow['jobs']['verify-host-access']
    step = next(s for s in probe['steps'] if 'verify-space-access' in s.get('run', ''))
    tools = tmp_path / 'tools'
    tools.mkdir()
    runner = tools / 'run_capped.sh'
    runner.write_text('#!/bin/bash\nprintf \'{"error":"PermissionError"}\\n\' >&2\nexit 7\n')
    runner.chmod(0o755)
    result = subprocess.run(['bash', '-c', step['run']], cwd=tmp_path,
                            env=dict(os.environ, HF_TOKEN='private-test-sentinel',
                                     ACCESS_SPACE='einarolafsson/spacr-docs-nightly-v2'),
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 7
    assert 'PermissionError' in result.stderr
    assert 'private-test-sentinel' not in result.stdout + result.stderr


@pytest.mark.parametrize("operation", ["metadata", "tree", "readme", "commit", "readback"])
def test_denied_boundary_preserves_exception_and_reports_only_safe_operation(
        service, monkeypatch, capsys, operation):
    from urllib.error import HTTPError
    api, _, *_ = service
    failure = HTTPError("https://invalid/?token=sensitive-credential", 403,
                        "private server message sensitive-credential", {}, None)
    # Python 3.9 leaves the delegated file unset when HTTPError has no body.
    failure.__dict__.pop("file", None)

    def fail(*args, **kwargs):
        raise failure

    if operation == "metadata":
        monkeypatch.setattr(api, "space_info", fail)
    elif operation == "readme":
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)
    elif operation == "commit":
        monkeypatch.setattr(publisher, "build_opener", lambda *args: SimpleNamespace(open=fail))
    else:
        original = api.list_repo_tree
        def entries(*args, **kwargs):
            revision = kwargs["revision"]
            if operation == "tree" or revision == "d" * 40:
                raise failure
            yield from original(*args, **kwargs)
        monkeypatch.setattr(api, "list_repo_tree", entries)

    with pytest.raises(HTTPError) as caught:
        publisher.verify_space_access("einarolafsson/spacr-docs-nightly-v2")
    assert caught.value is failure and caught.value.code == 403
    assert failure.spacr_space_access_operation == operation
    assert publisher.main(["verify-space-access", "--space",
                           "einarolafsson/spacr-docs-nightly-v2"]) == 1
    output = capsys.readouterr()
    assert not output.out and "sensitive-credential" not in output.err
    assert json.loads(output.err) == {"space": "einarolafsson/spacr-docs-nightly-v2",
                                      "error": "HTTPError", "status": 403,
                                      "operation": operation}


@pytest.mark.parametrize("message, reason", [
    ("Storage limit reached for this space (Max: 1 GB).", "storage-limit"),
    ("You don't have the rights to create a commit on this repo", "not-authorized"),
    ("private server message sensitive-credential", None),
])
def test_commit_refusal_reports_a_fixed_reason_without_server_text(
        service, monkeypatch, capsys, message, reason):
    from urllib.error import HTTPError
    failure = HTTPError("https://invalid/?token=sensitive-credential", 403, "Forbidden",
                        {"X-Error-Message": message + " sensitive-credential"}, None)
    # Preserve the bodyless Python 3.9 case on newer interpreters too.
    failure.__dict__.pop("file", None)

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(publisher, "build_opener", lambda *args: SimpleNamespace(open=fail))
    assert publisher.main(["verify-space-access", "--space",
                           "einarolafsson/spacr-docs-nightly-v2"]) == 1
    output = capsys.readouterr()
    assert "sensitive-credential" not in output.err
    assert message not in output.err
    report = json.loads(output.err)
    assert report["operation"] == "commit" and report["status"] == 403
    assert report.get("reason") == reason


def test_upload_storage_refusal_names_the_cause_and_still_fails(monkeypatch, capsys, tmp_path):
    class HfHubHTTPError(Exception):
        response = SimpleNamespace(status_code=403, headers={},
                                   text="Storage limit reached for this space (Max: 1 GB).")

    failure = HfHubHTTPError("403 Forbidden")

    def fail(site, space):
        raise failure

    monkeypatch.setattr(publisher, "upload_nightly", fail)
    with pytest.raises(HfHubHTTPError):
        publisher.main(["upload-nightly", "--site", str(tmp_path), "--space",
                        "einarolafsson/spacr-docs-nightly"])
    assert "storage limit is reached" in capsys.readouterr().err
