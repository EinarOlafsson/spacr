"""Developer ID signing and notarization for the small online macOS package.

No credentials means an explicitly unsigned development build. Partial settings,
or missing credentials when signing is required, fail before packaging. A signed
artifact replaces the input only after notarization, stapling and Gatekeeper pass.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile


CI_KEYS = ('MACOS_CERTIFICATES_BASE64', 'MACOS_CERTIFICATE_PASSWORD',
           'MACOS_APPLICATION_IDENTITY', 'MACOS_INSTALLER_IDENTITY',
           'MACOS_APPLE_ID', 'MACOS_TEAM_ID', 'MACOS_APP_PASSWORD')
BUILD_KEYS = ('CODESIGN_IDENTITY', 'PRODUCTSIGN_IDENTITY', 'SPACR_NOTARY_PROFILE')


def bundle_versions(version):
    """Encode spaCR's fourth component in an ordered three-part Apple build ID."""
    if not re.fullmatch(r'[0-9]+(?:\.[0-9]+){2,3}', version):
        raise ValueError('Expected a three- or four-component numeric spaCR version')
    parts = [int(part) for part in version.split('.')]
    if any(part > 99 for part in parts):
        raise ValueError('Apple build encoding requires each spaCR component below 100')
    major, minor, patch = parts[:3]
    revision = parts[3] if len(parts) == 4 else 0
    return f'{major}.{minor}.{patch}', f'{100 * major + minor}.{patch}.{revision}'


def _run(arguments, *, timeout=120):
    """Execute an Apple tool without echoing credential-bearing arguments."""
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f'{Path(arguments[0]).name} timed out') from None
    if result.returncode:
        raise RuntimeError(f'{Path(arguments[0]).name} failed (exit {result.returncode})')
    return result.stdout


def _required():
    value = os.environ.get('SPACR_MACOS_SIGNING_REQUIRED', '0')
    if value not in ('0', '1'):
        raise ValueError('SPACR_MACOS_SIGNING_REQUIRED must be 0 or 1')
    return value == '1'


def _export(values):
    """Make nonsecret build paths and identity names available to later CI steps."""
    destination = Path(os.environ['GITHUB_ENV'])
    for key, value in values.items():
        if '\n' in value or '\r' in value:
            raise ValueError(f'{key} must be a single line')
    with destination.open('a') as stream:
        for key, value in values.items():
            stream.write(f'{key}={value}\n')


def prepare_ci():
    """Import both Developer ID certificates into a disposable runner keychain."""
    supplied = [key for key in CI_KEYS if os.environ.get(key)]
    if not supplied and not _required():
        print('::warning::No Apple signing credentials: building an unsigned development package.')
        return
    missing = [key for key in CI_KEYS if not os.environ.get(key)]
    if missing:
        raise ValueError('Incomplete Apple signing configuration: ' + ', '.join(missing))
    directory = Path(tempfile.mkdtemp(prefix='spacr-signing-', dir=os.environ['RUNNER_TEMP']))
    _export({'SPACR_SIGNING_DIRECTORY': str(directory)})
    keychain = directory / 'build.keychain-db'
    previous = shlex.split(_run(['security', 'list-keychains', '-d', 'user']))
    state = {'previous_keychains': previous, 'keychain': str(keychain)}
    (directory / 'state.json').write_text(json.dumps(state))
    password = secrets.token_urlsafe(32)
    certificate = directory / 'certificates.p12'
    try:
        with certificate.open('xb') as stream:
            os.chmod(certificate, 0o600)
            stream.write(base64.b64decode(os.environ['MACOS_CERTIFICATES_BASE64'], validate=True))
        _run(['security', 'create-keychain', '-p', password, str(keychain)])
        _run(['security', 'set-keychain-settings', '-lut', '7200', str(keychain)])
        _run(['security', 'unlock-keychain', '-p', password, str(keychain)])
        _run(['security', 'import', str(certificate), '-P', os.environ['MACOS_CERTIFICATE_PASSWORD'],
              '-k', str(keychain), '-T', '/usr/bin/codesign', '-T', '/usr/bin/productsign'])
        _run(['security', 'set-key-partition-list', '-S', 'apple-tool:,apple:',
              '-k', password, str(keychain)])
        _run(['security', 'list-keychains', '-d', 'user', '-s', str(keychain), *previous])
        _run(['xcrun', 'notarytool', 'store-credentials', 'spacr-notary',
              '--apple-id', os.environ['MACOS_APPLE_ID'],
              '--team-id', os.environ['MACOS_TEAM_ID'],
              '--password', os.environ['MACOS_APP_PASSWORD'], '--keychain', str(keychain)])
        _export({'CODESIGN_IDENTITY': os.environ['MACOS_APPLICATION_IDENTITY'],
                 'PRODUCTSIGN_IDENTITY': os.environ['MACOS_INSTALLER_IDENTITY'],
                 'SPACR_NOTARY_PROFILE': 'spacr-notary',
                 'SPACR_SIGNING_KEYCHAIN': str(keychain),
                 'SPACR_MACOS_SIGNING_REQUIRED': '1'})
    finally:
        certificate.unlink(missing_ok=True)


def cleanup_ci():
    """Restore the runner's keychain search list and remove only this job's files."""
    location = os.environ.get('SPACR_SIGNING_DIRECTORY')
    if not location:
        return
    directory = Path(location).resolve()
    root = Path(os.environ['RUNNER_TEMP']).resolve()
    if directory.parent != root or not directory.name.startswith('spacr-signing-'):
        raise ValueError('Refusing to clean an unexpected signing directory')
    state_path = directory / 'state.json'
    if not state_path.exists():
        shutil.rmtree(directory)
        return
    state = json.loads(state_path.read_text())
    keychain = directory / 'build.keychain-db'
    try:
        _run(['security', 'list-keychains', '-d', 'user', '-s', *state['previous_keychains']])
    finally:
        try:
            if keychain.exists():
                _run(['security', 'delete-keychain', str(keychain)])
        finally:
            shutil.rmtree(directory)


def signing_config():
    """Validate the complete signing contract before touching any artifact."""
    values = {key: os.environ.get(key, '') for key in BUILD_KEYS}
    if not any(values.values()) and not _required():
        return None
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise ValueError('Signing requires ' + ', '.join(missing))
    for key, prefix in (('CODESIGN_IDENTITY', 'Developer ID Application: '),
                        ('PRODUCTSIGN_IDENTITY', 'Developer ID Installer: ')):
        if not values[key].startswith(prefix):
            raise ValueError(f'{key} must name a {prefix.strip()} certificate')
    return values


def _keychain_arguments():
    path = os.environ.get('SPACR_SIGNING_KEYCHAIN')
    return ['--keychain', path] if path else []


def sign_app(path):
    """Sign the native launcher bundle and verify its sealed resources."""
    config = signing_config()
    arguments = ['codesign', '--force', '--sign', config['CODESIGN_IDENTITY'] if config else '-']
    if config:
        arguments += ['--timestamp', '--options', 'runtime', *_keychain_arguments()]
    _run([*arguments, str(path)])
    _run(['codesign', '--verify', '--deep', '--strict', str(path)])


def sign_package(path):
    """Publish a signed package only after Apple's service and Gatekeeper accept it."""
    config = signing_config()
    path = Path(path)
    record = {'package': path.name, 'status': 'unsigned', 'stapled': False,
              'gatekeeper_verified': False}
    receipt = path.with_suffix(path.suffix + '.notarization.json')
    receipt.unlink(missing_ok=True)
    if config:
        with tempfile.TemporaryDirectory(prefix='.spacr-notary-', dir=path.parent) as staging:
            signed = Path(staging) / path.name
            _run(['productsign', '--sign', config['PRODUCTSIGN_IDENTITY'], '--timestamp',
                  *_keychain_arguments(), str(path), str(signed)])
            _run(['pkgutil', '--check-signature', str(signed)])
            output = _run(['xcrun', 'notarytool', 'submit', str(signed),
                           '--keychain-profile', config['SPACR_NOTARY_PROFILE'],
                           *_keychain_arguments(), '--wait', '--timeout', '20m',
                           '--output-format', 'json'], timeout=1260)
            result = json.loads(output)
            if result.get('status') != 'Accepted' or not result.get('id'):
                raise RuntimeError('Apple notarization was not accepted; no signed package published')
            _run(['xcrun', 'stapler', 'staple', str(signed)])
            _run(['xcrun', 'stapler', 'validate', str(signed)])
            _run(['spctl', '--assess', '--type', 'install', '--verbose=2', str(signed)])
            record.update(status='Accepted', submission_id=result['id'], stapled=True,
                          gatekeeper_verified=True)
            os.replace(signed, path)
    record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare-ci', 'cleanup-ci', 'validate', 'bundle-versions',
                                          'sign-app', 'sign-package'))
    parser.add_argument('path', nargs='?')
    args = parser.parse_args()
    if sys.platform != 'darwin':
        parser.error('Apple signing tools require macOS')
    if args.action in ('sign-app', 'sign-package', 'bundle-versions') and not args.path:
        parser.error('this action requires an artifact path')
    actions = {'prepare-ci': prepare_ci, 'cleanup-ci': cleanup_ci, 'validate': signing_config,
               'bundle-versions': lambda: print(*bundle_versions(args.path)),
               'sign-app': lambda: sign_app(args.path),
               'sign-package': lambda: sign_package(args.path)}
    try:
        actions[args.action]()
    except Exception as error:
        print(f'macOS signing failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
