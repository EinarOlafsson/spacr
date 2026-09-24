"""Packaging control flow with fake Apple tools; no native signing claim."""
import base64
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'spacr_macos_signing', ROOT / 'packaging/online/macos_signing.py')
signing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(signing)


def test_apple_version_encodes_fourth_component_and_preserves_order():
    assert signing.bundle_versions('1.5.0.9') == ('1.5.0', '105.0.9')
    assert signing.bundle_versions('1.5.1.0') == ('1.5.1', '105.1.0')
    versions = ['1.5.0', '1.5.0.9', '1.5.1.0', '1.6.0.0', '2.0.0.0']
    builds = [tuple(map(int, signing.bundle_versions(v)[1].split('.'))) for v in versions]
    assert builds == sorted(set(builds))


@pytest.mark.parametrize('version', ['1.5', '1.5.0rc1', '1.5.0.0.1', '1.100.0', '-1.2.3'])
def test_ambiguous_bundle_versions_are_refused(version):
    with pytest.raises(ValueError):
        signing.bundle_versions(version)


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    for name in (*signing.CI_KEYS, *signing.BUILD_KEYS, 'SPACR_MACOS_SIGNING_REQUIRED',
                 'SPACR_SIGNING_KEYCHAIN', 'SPACR_SIGNING_DIRECTORY'):
        monkeypatch.delenv(name, raising=False)


def configure(monkeypatch):
    monkeypatch.setenv('CODESIGN_IDENTITY', 'Developer ID Application: Test (TEAMID)')
    monkeypatch.setenv('PRODUCTSIGN_IDENTITY', 'Developer ID Installer: Test (TEAMID)')
    monkeypatch.setenv('SPACR_NOTARY_PROFILE', 'local-profile')
    monkeypatch.setenv('SPACR_SIGNING_KEYCHAIN', '/private/keychain')


@pytest.mark.parametrize('key', signing.BUILD_KEYS)
def test_partial_signing_is_refused_before_tools(monkeypatch, key):
    monkeypatch.setenv(key, 'provided')
    with pytest.raises(ValueError, match='Signing requires'):
        signing.sign_app('app')


def test_required_cannot_fall_back_to_unsigned(monkeypatch):
    monkeypatch.setenv('SPACR_MACOS_SIGNING_REQUIRED', '1')
    with pytest.raises(ValueError):
        signing.signing_config()
    with pytest.raises(ValueError):
        signing.prepare_ci()


def test_unsigned_receipt_cannot_claim_notarization(tmp_path, monkeypatch):
    monkeypatch.setattr(signing, '_run', lambda *a, **k: pytest.fail('no Apple request'))
    package = tmp_path / 'app.pkg'
    package.write_bytes(b'original')
    record = signing.sign_package(package)
    assert record['status'] == 'unsigned'
    assert record['stapled'] is record['gatekeeper_verified'] is False
    assert package.read_bytes() == b'original'


@pytest.mark.parametrize('identity', ['-', 'Apple Development: Test', 'Developer ID Installer: Test'])
def test_app_needs_application_certificate(monkeypatch, identity):
    configure(monkeypatch)
    monkeypatch.setenv('CODESIGN_IDENTITY', identity)
    with pytest.raises(ValueError, match='Developer ID Application'):
        signing.signing_config()


def test_sign_app_uses_runtime_timestamp_and_strict_verification(monkeypatch):
    configure(monkeypatch)
    calls = []
    monkeypatch.setattr(signing, '_run', lambda args: calls.append(args))
    signing.sign_app('/tmp/spaCR.app')
    assert '--timestamp' in calls[0]
    assert calls[0][calls[0].index('--options') + 1] == 'runtime'
    assert calls[0][calls[0].index('--keychain') + 1] == '/private/keychain'
    assert '--deep' not in calls[0]
    assert calls[1] == ['codesign', '--verify', '--deep', '--strict', '/tmp/spaCR.app']


@pytest.mark.parametrize('failure', [None, 'productsign', 'pkgutil', 'submit',
                                    'rejected', 'malformed', 'staple', 'validate', 'spctl'])
def test_no_signed_artifact_is_published_until_every_check_passes(
        monkeypatch, tmp_path, failure):
    configure(monkeypatch)
    package = tmp_path / 'spaCR.pkg'
    package.write_bytes(b'unsigned')
    receipt = package.with_suffix('.pkg.notarization.json')
    receipt.write_text('stale old acceptance')
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        step = args[2] if args[0] == 'xcrun' else args[0]
        if step == failure:
            raise RuntimeError('rejected stage')
        if step == 'productsign':
            Path(args[-1]).write_bytes(b'signed and stapled')
        if step == 'submit':
            if failure == 'malformed':
                return 'not-json'
            return json.dumps({'status': 'Invalid' if failure == 'rejected' else 'Accepted',
                               'id': 'submission-123'})
        return ''
    monkeypatch.setattr(signing, '_run', run)
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            signing.sign_package(package)
        assert package.read_bytes() == b'unsigned'
        assert not receipt.exists()
    else:
        record = signing.sign_package(package)
        assert record['status'] == 'Accepted'
        assert record['submission_id'] == 'submission-123'
        assert record['stapled'] and record['gatekeeper_verified']
        assert package.read_bytes() == b'signed and stapled'
        assert json.loads(receipt.read_text()) == record
        assert [a[2] if a[0] == 'xcrun' else a[0] for a in calls] == [
            'productsign', 'pkgutil', 'submit', 'staple', 'validate', 'spctl']
    assert not list(tmp_path.glob('.spacr-notary-*'))


def test_credentials_never_appear_in_timeout_or_tool_failure(monkeypatch):
    def timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 1, output='a secret')
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(RuntimeError) as error:
        signing._run(['security', '-p', 'very-secret'])
    assert 'very-secret' not in str(error.value)
    assert 'a secret' not in str(error.value)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(
        a[0], 1, 'very-secret', 'a secret'))
    with pytest.raises(RuntimeError) as error:
        signing._run(['security', '-p', 'very-secret'])
    assert str(error.value) == 'security failed (exit 1)'


@pytest.mark.parametrize('fail_import', [False, True])
def test_ci_keychain_cleanup_restores_search_path_even_after_failed_import(
        monkeypatch, tmp_path, fail_import):
    for name in signing.CI_KEYS:
        monkeypatch.setenv(name, 'private-value')
    monkeypatch.setenv('MACOS_CERTIFICATES_BASE64', base64.b64encode(b'private-cert').decode())
    environment = tmp_path / 'github-env'
    monkeypatch.setenv('GITHUB_ENV', str(environment))
    monkeypatch.setenv('RUNNER_TEMP', str(tmp_path))
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        if args == ['security', 'list-keychains', '-d', 'user']:
            return '"/Users/test/login.keychain-db" "system.keychain"'
        if args[1] == 'create-keychain':
            Path(args[-1]).touch()
        if args[1] == 'import':
            path = Path(args[2])
            assert path.read_bytes() == b'private-cert'
            assert path.stat().st_mode & 0o777 == 0o600
            if fail_import:
                raise RuntimeError('cannot import')
        return ''
    monkeypatch.setattr(signing, '_run', run)
    if fail_import:
        with pytest.raises(RuntimeError, match='cannot import'):
            signing.prepare_ci()
    else:
        signing.prepare_ci()
    exports = dict(line.split('=', 1) for line in environment.read_text().splitlines())
    folder = Path(exports['SPACR_SIGNING_DIRECTORY'])
    assert not (folder / 'certificates.p12').exists()
    assert not any('private-cert' in v for v in exports.values())
    if fail_import:
        assert 'SPACR_MACOS_SIGNING_REQUIRED' not in exports
    else:
        assert exports['SPACR_MACOS_SIGNING_REQUIRED'] == '1'
        assert 'MACOS_APP_PASSWORD' not in exports
    monkeypatch.setenv('SPACR_SIGNING_DIRECTORY', str(folder))
    signing.cleanup_ci()
    assert calls[-2] == ['security', 'list-keychains', '-d', 'user', '-s',
                         '/Users/test/login.keychain-db', 'system.keychain']
    assert calls[-1] == ['security', 'delete-keychain', str(folder / 'build.keychain-db')]
    assert not folder.exists()


def test_cleanup_cannot_delete_unrelated_directory(monkeypatch, tmp_path):
    monkeypatch.setenv('RUNNER_TEMP', str(tmp_path))
    monkeypatch.setenv('SPACR_SIGNING_DIRECTORY', str(tmp_path))
    with pytest.raises(ValueError):
        signing.cleanup_ci()
    assert tmp_path.exists()


@pytest.fixture
def launcher(tmp_path):
    compiler = shutil.which('cc')
    if not compiler:
        pytest.skip('host C compiler unavailable')
    binary = tmp_path / 'launcher'
    subprocess.run([compiler, '-Wall', '-Wextra', '-Werror',
                    str(ROOT / 'packaging/online/macos_launcher.c'), '-o', str(binary)], check=True)
    return binary


def test_native_launcher_executes_private_python_with_exact_arguments(launcher, tmp_path):
    home = tmp_path / 'a home with spaces'
    python = home / 'Library/Application Support/spaCR/venv/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\nexit 7\n')
    python.chmod(0o755)
    args = ['--argument', 'a b', '$(touch forbidden-file)', '日本語']
    result = subprocess.run([str(launcher), *args], env={**os.environ, 'HOME': str(home)},
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 7
    assert result.stdout.splitlines() == ['-m', 'spacr.qt', *args]
    assert not (tmp_path / 'forbidden-file').exists()


def test_launcher_refuses_truncated_runtime_path(launcher):
    result = subprocess.run([str(launcher)], env={**os.environ, 'HOME': '/' + 'x' * 10000},
                            text=True, capture_output=True)
    assert result.returncode == 1
    assert 'too long' in result.stderr
