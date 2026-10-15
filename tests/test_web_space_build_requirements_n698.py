"""Managed requirements must resolve before the Space source COPY layer."""
from __future__ import annotations

import csv
from email.parser import Parser
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit
import zipfile

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]
PIN = '66ac681d3040aab1823aed0bf16b4ed5c89589f7'


def test_managed_project_requirement_uses_the_immutable_public_source_and_extra():
    rows = (ROOT / 'tools/web_mask_generator/space/requirements.txt').read_text().splitlines()
    project = Requirement(rows[0])
    assert project.name == 'spacr' and project.extras == {'plaque'}
    assert project.url == 'git+https://huggingface.co/spaces/einarolafsson/spacr-mask-generator@' + PIN
    parsed = urlsplit(project.url.removeprefix('git+'))
    assert parsed.scheme == 'https' and parsed.hostname == 'huggingface.co'
    assert parsed.username is None and parsed.password is None
    assert parsed.query == parsed.fragment == ''
    assert rows[1:] == ['huggingface-hub==0.36.2', 'torch==2.13.0',
                        'torchvision==0.28.0', 'pypdfium2==5.13.0']
    readme = (ROOT / 'tools/web_mask_generator/space/README.md').read_text()
    assert PIN in readme and '`git lfs install`' in readme


def _dependency_wheel(destination):
    name = 'managed_plaque_dependency'
    info = name + '-1.0.dist-info/'
    members = {name + '.py': b'VALUE = "resolved-plaque-extra"\n',
               info + 'METADATA': b'Metadata-Version: 2.1\nName: managed-plaque-dependency\nVersion: 1.0\n',
               info + 'WHEEL': b'Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n'}
    rows = [[path, '', str(len(data))] for path, data in members.items()]
    rows.append([info + 'RECORD', '', ''])
    record = io.StringIO(); csv.writer(record, lineterminator='\n').writerows(rows)
    members[info + 'RECORD'] = record.getvalue().encode()
    path = destination / (name + '-1.0-py3-none-any.whl')
    with zipfile.ZipFile(path, 'w') as archive:
        for entry, data in members.items():
            archive.writestr(entry, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_actual_offline_pip_resolves_immutable_git_extra_before_source_copy(tmp_path):
    repo = tmp_path / 'public-source-fixture'; repo.mkdir()
    (repo / 'setup.py').write_text("from setuptools import setup\nsetup(name='managed-build-fixture',version='1.0',py_modules=['fixture_model'],extras_require={'plaque':['managed-plaque-dependency==1.0']})\n")
    (repo / 'fixture_model.py').write_text("SOURCE = 'immutable-first-revision'\n")
    (repo / 'pyproject.toml').write_text('[build-system]\nrequires=["setuptools", "wheel"]\nbuild-backend="setuptools.build_meta"\n')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.STDOUT, text=True, timeout=15).strip()
    git('init', '-q'); git('add', '.')
    git('-c', 'user.name=Disposable Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'Initial immutable source')
    commit = git('rev-parse', 'HEAD')
    (repo / 'fixture_model.py').write_text("SOURCE = 'wrong-later-revision'\n")
    git('add', '.'); git('-c', 'user.name=Disposable Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'Later version must not leak')
    wheelhouse = tmp_path / 'wheelhouse'; wheelhouse.mkdir(); digest = _dependency_wheel(wheelhouse)
    empty = tmp_path / 'managed-before-copy'; empty.mkdir()
    outputs = tmp_path / 'built-wheels'
    environment = dict(os.environ, PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_INPUT='1')
    command = [sys.executable, '-m', 'pip', 'wheel', '--no-index', '--no-cache-dir', '--no-build-isolation', '--find-links', str(wheelhouse), '--wheel-dir', str(outputs)]
    local = subprocess.run(command + ['.[plaque]'], cwd=empty, env=environment, capture_output=True, text=True, timeout=25)
    assert local.returncode != 0 and 'Neither' in local.stderr
    request = 'managed-build-fixture[plaque] @ git+' + repo.as_uri() + '@' + commit
    actual = subprocess.run(command + [request], cwd=empty, env=environment, capture_output=True, text=True, timeout=40)
    assert actual.returncode == 0, actual.stdout + actual.stderr
    assert not (empty / 'setup.py').exists() and not (empty / 'pyproject.toml').exists()
    dependency = outputs / 'managed_plaque_dependency-1.0-py3-none-any.whl'
    assert dependency.is_file() and hashlib.sha256(dependency.read_bytes()).hexdigest() == digest
    with zipfile.ZipFile(outputs / 'managed_build_fixture-1.0-py3-none-any.whl') as archive:
        assert archive.read('fixture_model.py') == b"SOURCE = 'immutable-first-revision'\n"
        metadata = Parser().parsestr(archive.read('managed_build_fixture-1.0.dist-info/METADATA').decode())
        assert metadata['Provides-Extra'] == 'plaque'
        assert any(Requirement(value).name == 'managed-plaque-dependency'
                   for value in metadata.get_all('Requires-Dist'))
