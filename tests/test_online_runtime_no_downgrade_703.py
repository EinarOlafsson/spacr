"""External online replacement keeps a newer runtime and repairs broken imports."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

import pytest

from spacr import install_cleanup as cleanup


HELPER = os.environ.get('SPACR_VERIFY_HELPER', cleanup.__file__)


def _runtime(root, version, *, broken_import=False, with_packaging=False):
    venv.EnvBuilder(with_pip=False).create(root)
    python = root / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    site = Path(subprocess.check_output(
        [str(python), '-I', '-B', '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'],
        text=True, timeout=15).strip())
    if with_packaging:
        packaging = Path(importlib.util.find_spec('packaging').origin).parent
        shutil.copytree(packaging, site / 'packaging', ignore=shutil.ignore_patterns('__pycache__'))
    package = site / 'spacr'
    package.mkdir()
    (package / '__init__.py').write_text(
        'raise ImportError("unavailable optional GUI dependency")\n' if broken_import else
        'from ._version import __version__\n')
    (package / '_version.py').write_text(f'__version__ = {version!r}\n')
    info = site / f'spacr-{version}.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: spacr\nVersion: {version}\n')
    retained = root / 'private cache'
    retained.mkdir()
    (retained / 'weights.bin').write_bytes(b'original model bytes')
    return python, package, info


def _snapshot(root):
    return {str(path.relative_to(root)): ('link', os.readlink(path)) if path.is_symlink()
            else ('file', hashlib.sha256(path.read_bytes()).hexdigest())
            for path in root.rglob('*') if '__pycache__' not in path.parts
            and (path.is_file() or path.is_symlink())}


def _verify(python, old, requested='1.5.1.5', *, minimum=False):
    command = [sys.executable, '-I', '-S', '-B', HELPER, 'verify-runtime', '--python', str(python),
               '--retiring-prefix', str(old), '--version', requested]
    if minimum:
        command.append('--minimum')
    return subprocess.run(command, text=True, capture_output=True, timeout=45)


@pytest.mark.parametrize('old_version,broken', [
    (None, False), ('1.5.1.4', False), ('1.5.1.5', False),
    ('1.5.1.4', True), ('1.5.1.5', True),
])
def test_fresh_newer_same_version_and_broken_import_repairs(tmp_path, old_version, broken):
    old = tmp_path / 'previous runtime'
    if old_version:
        _runtime(old, old_version, broken_import=broken)
    staged = tmp_path / 'staged runtime'
    python, _, _ = _runtime(staged, '1.5.1.5', with_packaging=True)
    before = _snapshot(tmp_path)
    result = _verify(python, old)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    assert receipt['verified'] is True
    assert receipt['version'] == '1.5.1.5'
    assert receipt['previous_version'] == old_version
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize('broken,minimum', [(False, False), (True, False), (True, True)])
def test_older_staged_runtime_cannot_replace_newer_even_with_broken_import(tmp_path, broken, minimum):
    old = tmp_path / 'previous runtime'
    _runtime(old, '1.5.1.6', broken_import=broken)
    python, _, _ = _runtime(tmp_path / 'staged runtime', '1.5.1.5', with_packaging=True)
    before = _snapshot(tmp_path)
    result = _verify(python, old, minimum=minimum)
    assert result.returncode != 0
    assert json.loads(result.stdout) == {
        'verified': False, 'reason': 'runtime_downgrade_refused',
        'previous_version': '1.5.1.6', 'staged_version': '1.5.1.5'}
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize('damage', ['missing-python', 'missing-metadata', 'different-code',
                                    'duplicate-owner', 'editable', 'malformed-metadata', 'shadow', 'file'])
def test_untrusted_previous_identity_is_refused_without_modifying_either_runtime(tmp_path, damage):
    old = tmp_path / 'previous runtime'
    if damage == 'file':
        old.write_text('unknown previous runtime')
    else:
        previous_python, package, info = _runtime(old, '1.5.1.4', broken_import=True)
        if damage == 'missing-python':
            previous_python.unlink()
        elif damage == 'missing-metadata':
            shutil.rmtree(info)
        elif damage == 'different-code':
            (package / '_version.py').write_text('__version__ = "1.5.1.6"\n')
        elif damage == 'duplicate-owner':
            other = info.with_name('spacr_nightly-1.5.1.6.dist-info')
            other.mkdir()
            (other / 'METADATA').write_text('Metadata-Version: 2.1\nName: spacr-nightly\nVersion: 1.5.1.6\n')
        elif damage == 'editable':
            (info / 'direct_url.json').write_text('{"dir_info":{"editable":true}}')
        elif damage == 'shadow':
            shadow = tmp_path / 'foreign code'
            (shadow / 'spacr').mkdir(parents=True)
            (shadow / 'spacr/__init__.py').write_text('raise RuntimeError("never execute this")\n')
            (package.parent / 'foreign.pth').write_text(
                f'import sys; sys.path.insert(0, {str(shadow)!r})\n')
        elif damage == 'malformed-metadata':
            (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: spacr\n')
    python, _, _ = _runtime(tmp_path / 'staged runtime', '1.5.1.5', with_packaging=True)
    before = _snapshot(tmp_path)
    result = _verify(python, old)
    assert result.returncode == 2, result.stdout + result.stderr
    assert json.loads(result.stdout) == {'verified': False, 'reason': 'previous_runtime_identity_unverified'}
    assert _snapshot(tmp_path) == before


def test_explicit_request_still_cannot_accept_a_different_staged_version(tmp_path):
    old = tmp_path / 'previous runtime'
    _runtime(old, '1.5.1.4', broken_import=True)
    python, _, _ = _runtime(tmp_path / 'staged runtime', '1.5.1.6', with_packaging=True)
    before = _snapshot(tmp_path)
    result = _verify(python, old)
    assert result.returncode != 0
    assert 'Requested spaCR version differs' in result.stdout + result.stderr
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize('previous,staged,refused', [
    ('1.5.1.5rc1', '1.5.1.5', False), ('1.5.1.5', '1.5.1.5rc1', True),
    ('1.5.1.9', '1.5.1.10', False), ('1.5.1.10', '1.5.1.9', True),
])
def test_release_and_prerelease_order_uses_pep440_instead_of_text(tmp_path, previous, staged, refused):
    old = tmp_path / 'previous runtime'
    _runtime(old, previous, broken_import=True)
    python, _, _ = _runtime(tmp_path / 'staged runtime', staged, with_packaging=True)
    before = _snapshot(tmp_path)
    result = _verify(python, old, requested=staged)
    assert (result.returncode != 0) is refused, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    if refused:
        assert receipt['reason'] == 'runtime_downgrade_refused'
    else:
        assert receipt['verified'] is True and receipt['previous_version'] == previous
    assert _snapshot(tmp_path) == before


def test_unknown_previous_version_is_not_compared_lexically(tmp_path):
    old = tmp_path / 'previous runtime'
    _runtime(old, 'unknown-release', broken_import=True)
    python, _, _ = _runtime(tmp_path / 'staged runtime', '1.5.1.5', with_packaging=True)
    before = _snapshot(tmp_path)
    result = _verify(python, old)
    assert result.returncode == 2, result.stdout + result.stderr
    assert json.loads(result.stdout) == {'verified': False, 'reason': 'previous_runtime_version_invalid'}
    assert _snapshot(tmp_path) == before
