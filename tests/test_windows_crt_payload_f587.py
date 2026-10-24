"""Pinned CRT preparation and private staged-runtime write boundaries."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import ctypes
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).parents[1] / 'packaging/online/windows_crt.py'


@pytest.fixture
def crt():
    spec = importlib.util.spec_from_file_location('windows_crt_f587', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metadata_pins_the_verified_official_release(crt):
    data = crt.metadata()
    assert data['artifact']['size'] == 18558944
    assert data['artifact']['sha256'] == '8995548dfffcde7c49987029c764355612ba6850ee09a7b6f0fddc85bdc5c280'
    assert len(data['members']) == 12
    assert data['members']['vcruntime140_threads.dll'] == 'a6222020b500a9a86b36e040c2dbd0e459716db1bf2810a11cd7512ea9b8d89b'


def test_metadata_requires_an_object(crt, tmp_path):
    path = tmp_path / 'metadata.json'
    path.write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='must be an object'):
        crt.metadata(path)


@pytest.mark.parametrize('change', ['release', 'artifact', 'member', 'escape'])
def test_metadata_rejects_changed_release_and_escaping_members(crt, tmp_path, change):
    data = crt.metadata()
    if change == 'release':
        data['version'] = '14.51.0.0'
    elif change == 'artifact':
        data['artifact']['url'] = 'https://example.invalid/runtime.exe'
    elif change == 'member':
        del data['members']['vcruntime140_threads.dll']
    else:
        data['members']['../vcruntime140_threads.dll'] = data['members'].pop('vcruntime140_threads.dll')
    path = tmp_path / 'metadata.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='pinned x64 release'):
        crt.metadata(path)


@pytest.mark.parametrize('size', [4, 18558944])
def test_bad_artifact_fails_before_extraction_or_output_creation(crt, tmp_path, monkeypatch, size):
    path = tmp_path / 'wrong.exe'
    with path.open('wb') as stream:
        stream.truncate(size)
    monkeypatch.setattr(crt, '_expand', lambda *args: pytest.fail('unverified artifact reached expand'))
    output = tmp_path / 'crt'
    with pytest.raises(ValueError, match='artifact size or SHA256'):
        crt.prepare(path, output)
    assert not output.exists()


def _image(machine=0x8664):
    image = bytearray(96)
    image[:2] = b'MZ'
    struct.pack_into('<I', image, 0x3c, 64)
    image[64:68] = b'PE\0\0'
    struct.pack_into('<H', image, 68, machine)
    return bytes(image)


@pytest.fixture
def staged(crt, tmp_path, monkeypatch):
    payload = tmp_path / 'prepared'
    payload.mkdir(mode=0o700)
    image = _image()
    data = crt.metadata()
    data['members'] = dict.fromkeys(crt.MEMBERS, hashlib.sha256(image).hexdigest())
    for name in data['members']:
        (payload / name).write_bytes(image)
    monkeypatch.setattr(crt, 'metadata', lambda: data)
    runtime = tmp_path / 'stage'
    runtime.mkdir(mode=0o700)
    (runtime / 'pyvenv.cfg').write_text('include-system-site-packages = false\n', encoding='utf-8')
    destination = runtime / 'Lib/site-packages/torch/lib'
    destination.mkdir(parents=True, mode=0o700)
    return payload, destination


def test_install_preserves_other_torch_files_and_matching_crt_bytes(crt, staged):
    payload, destination = staged
    keep = destination / 'torch_cpu.dll'
    keep.write_bytes(b'existing wheel bytes')
    existing = destination / 'vcruntime140.dll'
    existing.write_bytes((payload / existing.name).read_bytes())
    before = existing.stat().st_mtime_ns
    crt.install(payload, destination)
    assert keep.read_bytes() == b'existing wheel bytes'
    assert existing.stat().st_mtime_ns == before
    assert all((destination / name).read_bytes() == (payload / name).read_bytes() for name in crt.MEMBERS)


def test_unknown_existing_crt_refuses_every_write(crt, staged):
    payload, destination = staged
    blocker = destination / 'vcruntime140_threads.dll'
    blocker.write_bytes(b'foreign runtime bytes')
    with pytest.raises(ValueError, match='unknown existing CRT'):
        crt.install(payload, destination)
    assert list(destination.iterdir()) == [blocker]
    assert blocker.read_bytes() == b'foreign runtime bytes'


@pytest.mark.parametrize('problem', ['tamper', 'extra', 'wrong_arch', 'hardlink'])
def test_untrusted_prepared_members_never_reach_destination(crt, staged, problem, tmp_path):
    payload, destination = staged
    if problem == 'extra':
        (payload / 'foreign.dll').write_bytes(b'unexpected')
    elif problem == 'wrong_arch':
        wrong = _image(0x14c)
        for name in crt.MEMBERS:
            (payload / name).write_bytes(wrong)
            crt.metadata()['members'][name] = hashlib.sha256(wrong).hexdigest()
    elif problem == 'hardlink':
        member = payload / 'vcruntime140_threads.dll'
        os.link(member, tmp_path / 'linked-elsewhere.dll')
    else:
        (payload / 'vcruntime140_threads.dll').write_bytes(b'changed')
    with pytest.raises(ValueError):
        crt.install(payload, destination)
    assert not list(destination.iterdir())


def test_destination_must_be_a_dedicated_staged_runtime(crt, staged):
    payload, destination = staged
    (destination.parents[3] / 'pyvenv.cfg').write_text('include-system-site-packages = true\n', encoding='utf-8')
    with pytest.raises(ValueError, match='dedicated virtual environment'):
        crt.install(payload, destination)
    assert not list(destination.iterdir())


@pytest.mark.parametrize('settings', [
    '# include-system-site-packages = false\ninclude-system-site-packages = true\n',
    'include-system-site-packages = true\ninclude-system-site-packages = false\n',
    'include-system-site-packages = false\ninclude-system-site-packages = false\n',
    'home = C:/Python\n',
    'include-system-site-packages = false\n' + ' ' * 16384,
])
def test_verify_stage_rejects_ambiguous_or_missing_dedicated_setting(crt, staged, settings):
    _, destination = staged
    runtime = destination.parents[3]
    (runtime / 'pyvenv.cfg').write_text(settings, encoding='utf-8')
    with pytest.raises(ValueError, match='dedicated virtual environment'):
        crt.verify_stage(runtime)


def test_verify_stage_before_packages_is_read_only_and_reports_only_its_scope(crt, tmp_path, capsys):
    runtime = tmp_path / 'runtime'
    runtime.mkdir(mode=0o700)
    config = runtime / 'pyvenv.cfg'
    config.write_text('home = C:/Python\ninclude-system-site-packages = false\n', encoding='utf-8')
    before = config.read_bytes()
    assert crt.main(['verify-stage', '--runtime', str(runtime)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == {'verified': True, 'scope': 'private-dedicated-runtime', 'directory': str(runtime)}
    assert list(runtime.iterdir()) == [config]
    assert config.read_bytes() == before


@pytest.mark.parametrize('sid,propagation,rights,accepted', [
    ('S-1-3-0', 2, 0x10000000, True),
    ('S-1-3-0', 0, 0x10000000, False),
    ('S-1-5-32-545', 2, 0x10000000, False),
    ('S-1-1-0', 2, 0x40000000, False),
    ('S-1-5-32-545', 0, 2, False),
    ('S-1-5-32-545', 0, 4, False),
])
def test_private_acl_creator_owner_placeholder_is_not_a_foreign_writer_exception(
        crt, monkeypatch, sid, propagation, rights, accepted):
    user = 'S-1-5-21-123-456-789-1001'
    evidence = {'user': user, 'owner': user, 'has_dacl': True, 'rules': [
        {'type': 0, 'sid': user, 'rights': 0x1f01ff, 'propagation': 0},
        {'type': 0, 'sid': 'S-1-3-0', 'rights': 0x10000000, 'propagation': 2},
        {'type': 0, 'sid': sid, 'rights': rights, 'propagation': propagation}]}
    monkeypatch.setattr(crt, 'os', SimpleNamespace(name='nt'))
    monkeypatch.setattr(crt, '_read_acl', lambda path: evidence)
    if accepted:
        crt._private('owned directory')
    else:
        with pytest.raises(ValueError, match='another principal write'):
            crt._private('untrusted directory')


@pytest.mark.skipif(os.name != 'nt', reason='native Windows DACL inheritance')
@pytest.mark.parametrize('foreign_rights,inherit_only', [(0, False), (2, False), (4, False), (2, True)])
def test_native_stage_acl_creator_owner_and_actual_users_write_boundaries(
        crt, tmp_path, foreign_rights, inherit_only):
    root = tmp_path / 'new-owned-acl-fixture'
    root.mkdir()
    user = crt._read_acl(root)['user']
    descriptor = 'O:' + user + 'D:P'
    for sid in (user, 'S-1-5-18', 'S-1-5-32-544'):
        descriptor += '(A;OICI;FA;;;' + sid + ')'
    descriptor += '(A;OICIIO;GA;;;CO)'
    if foreign_rights:
        descriptor += '(A;OICI' + ('IO' if inherit_only else '') + ';' + hex(foreign_rights) + ';;;BU)'
    security = ctypes.WinDLL('advapi32', use_last_error=True)
    convert = security.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    convert.restype = ctypes.c_int32
    set_security = security.SetFileSecurityW
    set_security.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_void_p]
    set_security.restype = ctypes.c_int32
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    pointer = ctypes.c_void_p()
    if not convert(descriptor, 1, ctypes.byref(pointer), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not set_security(str(root), 0x80000005, pointer):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.LocalFree(pointer)
    runtime = root / 'runtime'
    runtime.mkdir()
    (runtime / 'pyvenv.cfg').write_text('include-system-site-packages = false\n', encoding='utf-8')
    evidence = crt._read_acl(runtime)
    assert any(rule['sid'] == 'S-1-3-0' and rule['propagation'] & 2 for rule in evidence['rules'])
    if foreign_rights:
        assert any(rule['sid'] == 'S-1-5-32-545' and rule['rights'] & foreign_rights
                   for rule in evidence['rules'])
        with pytest.raises(ValueError, match='another principal write'):
            crt.verify_stage(runtime)
    else:
        assert crt.verify_stage(runtime) == runtime
        (runtime / 'Lib/site-packages/torch/lib').mkdir(parents=True)
        assert crt.verify_stage(runtime) == runtime


def test_protected_destination_is_refused(crt, tmp_path, monkeypatch):
    monkeypatch.setenv('SystemRoot', str(tmp_path))
    with pytest.raises(ValueError, match='system installation directory'):
        crt._path(tmp_path / 'new-runtime', writing=True)


@pytest.mark.parametrize('owner', ['S-1-5-21-123-456-789-1001', 'S-1-5-32-544'])
def test_private_acl_accepts_current_user_or_builtin_administrators_owner(crt, monkeypatch, owner):
    user = 'S-1-5-21-123-456-789-1001'
    evidence = {'user': user, 'owner': owner, 'has_dacl': True, 'rules': [
        {'type': 0, 'sid': sid, 'rights': 0x1f01ff}
        for sid in (user, 'S-1-5-32-544', 'S-1-5-18', 'S-1-3-4')]}
    monkeypatch.setattr(crt, 'os', SimpleNamespace(name='nt'))
    monkeypatch.setattr(crt, '_read_acl', lambda path: evidence)
    crt._private('owned directory')
    evidence['has_dacl'] = False
    with pytest.raises(ValueError, match='private current-user or Administrators-owned ACL'):
        crt._private('owned directory')


@pytest.mark.parametrize('owner,has_dacl,rules', [
    ('S-1-5-21-123-456-789-9999', True, [{'type': 0, 'sid': 'S-1-5-32-544', 'rights': 0x1f01ff}]),
    ('S-1-5-18', True, [{'type': 0, 'sid': 'S-1-5-32-544', 'rights': 0x1f01ff}]),
    ('S-1-3-4', True, [{'type': 0, 'sid': 'S-1-5-32-544', 'rights': 0x1f01ff}]),
    ('S-1-5-32-544', False, [{'type': 0, 'sid': 'S-1-5-32-544', 'rights': 0x1f01ff}]),
    ('S-1-5-32-544', True, []),
    ('S-1-5-32-544', True, [{'type': 0, 'sid': 'S-1-1-0', 'rights': 0x40000000}]),
    ('S-1-5-21-123-456-789-1001', True, [{'type': 0, 'sid': 'S-1-5-32-545', 'rights': 0x40000000}]),
])
def test_private_acl_keeps_foreign_owner_writer_and_missing_dacl_refusals(
        crt, monkeypatch, owner, has_dacl, rules):
    evidence = {'user': 'S-1-5-21-123-456-789-1001', 'owner': owner,
                'has_dacl': has_dacl, 'rules': rules}
    monkeypatch.setattr(crt, 'os', SimpleNamespace(name='nt'))
    monkeypatch.setattr(crt, '_read_acl', lambda path: evidence)
    with pytest.raises(ValueError, match='ACL|another principal write'):
        crt._private('untrusted directory')


def test_cabinet_traversal_is_rejected_before_expand(crt, tmp_path):
    name = b'..\\escape.dll\0'
    header = bytearray(36)
    header[:8] = b'MSCF\0\0\0\0'
    struct.pack_into('<I', header, 8, 36 + 16 + len(name))
    struct.pack_into('<I', header, 16, 36)
    header[24:26] = b'\x03\x01'
    struct.pack_into('<H', header, 28, 1)
    cab = tmp_path / 'escape.cab'
    cab.write_bytes(header + bytes(16) + name)
    with pytest.raises(ValueError, match='escaping'):
        crt._cab_members(cab)


def test_signed_cabinet_trailer_does_not_expand_file_table_bounds(crt, tmp_path):
    name = b'vcruntime140.dll_amd64\0'
    header = bytearray(36)
    header[:8] = b'MSCF\0\0\0\0'
    size = 36 + 16 + len(name)
    struct.pack_into('<I', header, 8, size)
    struct.pack_into('<I', header, 16, 36)
    header[24:26] = b'\x03\x01'
    struct.pack_into('<H', header, 28, 1)
    cab = tmp_path / 'signed.cab'
    cab.write_bytes(header + bytes(16) + name + b'certificate trailer')
    assert crt._cab_members(cab) == {'vcruntime140.dll_amd64'}
    struct.pack_into('<I', header, 8, size - 4)
    cab.write_bytes(header + bytes(16) + name + b'certificate trailer')
    with pytest.raises(ValueError, match='declared bounds'):
        crt._cab_members(cab)
