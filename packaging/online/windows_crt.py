"""Prepare pinned Microsoft CRT bytes and install them in a private staged runtime.

This helper never executes the redistributable or imports the installed app.
The release artifact and member hashes identify the Microsoft-signed bytes;
verification does not require network access on an air-gapped target.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET

VERSION = '14.50.35719.0'
ARTIFACT_HASH = '8995548dfffcde7c49987029c764355612ba6850ee09a7b6f0fddc85bdc5c280'
ARTIFACT_SIZE = 18558944
MEMBERS = frozenset((
    'concrt140.dll', 'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
    'msvcp140_atomic_wait.dll', 'msvcp140_codecvt_ids.dll', 'vcamp140.dll',
    'vccorlib140.dll', 'vcomp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll',
    'vcruntime140_threads.dll',
))


def metadata(path=None):
    """Read the release's trusted companion manifest with strict shape checks."""
    path = Path(path) if path is not None else Path(__file__).with_suffix('.json')
    if path.stat().st_size > 16384:
        raise ValueError('CRT metadata is oversized')
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('CRT metadata must be an object')
    artifact = data.get('artifact', {})
    members = data.get('members', {})
    if (data.get('schema_version') != 1 or data.get('version') != VERSION
            or data.get('vendor') != 'Microsoft Corporation' or data.get('architecture') != 'x64'
            or data.get('license_url') != 'https://aka.ms/VCRedistLicense'
            or artifact != {'filename': 'VC_redist.x64.exe',
                            'url': 'https://aka.ms/vs/18/release/14.50.35719/VC_redist.x64.exe',
                            'sha256': ARTIFACT_HASH, 'size': ARTIFACT_SIZE}
            or not isinstance(members, dict) or set(members) != MEMBERS
            or any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)
                   for value in members.values())
            or data.get('cabinet') != {
                'file_path': 'packages\\vcRuntimeMinimum_amd64\\cab1.cab',
                'sha256': 'c346d91193246b18301cce532f2ae784fb377cb068d77bd22edec238e5746aa7'}):
        raise ValueError('CRT metadata does not describe the pinned x64 release')
    return data


def _native(path):
    value = os.path.abspath(os.fspath(path))
    if os.name == 'nt' and not value.startswith('\\\\?\\'):
        return '\\\\?\\' + value
    return value


def _path(path, *, writing=False):
    value = os.fspath(path)
    if os.name == 'nt':
        if value.startswith('\\\\?\\') and re.match(r'^[A-Za-z]:\\', value[4:]):
            value = value[4:]
        if value.startswith('\\\\') or ':' in value[2:]:
            raise ValueError('CRT paths must be ordinary local filesystem paths')
    result = Path(os.path.abspath(value))
    for ancestor in reversed((result, *result.parents)):
        try:
            details = os.lstat(_native(ancestor))
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(details.st_mode) or getattr(details, 'st_file_attributes', 0) & 0x400:
            raise ValueError('CRT path contains a link or reparse point: ' + str(ancestor))
    if writing:
        protected = [os.environ.get(name) for name in
                     ('SystemRoot', 'WINDIR', 'ProgramFiles', 'ProgramFiles(x86)', 'ProgramData')]
        for folder in filter(None, protected):
            base = os.path.normcase(os.path.abspath(folder))
            try:
                shared = os.path.commonpath((base, os.path.normcase(str(result))))
            except ValueError:
                continue
            if shared == base:
                raise ValueError('CRT writes cannot target a system installation directory')
    return result


def _directory(path, *, writing=False):
    path = _path(path, writing=writing)
    if not stat.S_ISDIR(os.lstat(_native(path)).st_mode):
        raise ValueError('CRT path is not an existing directory: ' + str(path))
    _private(path)
    return path


def _file(path):
    path = _path(path)
    details = os.lstat(_native(path))
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise ValueError('CRT input is not an ordinary unlinked file: ' + str(path))
    return path


def _digest(path):
    digest = hashlib.sha256()
    with open(_native(_file(path)), 'rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _pe_x64(path):
    with open(_native(_file(path)), 'rb') as stream:
        if stream.read(2) != b'MZ':
            raise ValueError('CRT member has no DOS image header')
        stream.seek(0x3c)
        offset = struct.unpack('<I', stream.read(4))[0]
        if not 64 <= offset <= 1048576:
            raise ValueError('CRT member PE header is outside its bounded header')
        stream.seek(offset)
        if stream.read(4) != b'PE\0\0' or stream.read(2) != b'\x64\x86':
            raise ValueError('CRT member is not a native x64 PE image')


def verify_artifact(path, data=None):
    data = metadata() if data is None else data
    path = _file(path)
    if os.stat(_native(path)).st_size != data['artifact']['size'] or _digest(path) != data['artifact']['sha256']:
        raise ValueError('Microsoft CRT artifact size or SHA256 does not match the pinned release')
    return path


def _verified_members(folder, data):
    folder = _directory(folder)
    if set(os.listdir(_native(folder))) != MEMBERS:
        raise ValueError('Prepared CRT directory must contain exactly the twelve pinned DLLs')
    for name, expected in data['members'].items():
        path = _file(folder / name)
        if not 1 <= os.stat(_native(path)).st_size <= 2097152 or _digest(path) != expected:
            raise ValueError('CRT member SHA256 does not match: ' + name)
        _pe_x64(path)
    return folder


def _cab_members(path):
    with open(_native(_file(path)), 'rb') as stream:
        header = stream.read(36)
        if len(header) != 36 or header[:8] != b'MSCF\0\0\0\0' or header[24:26] != b'\x03\x01':
            raise ValueError('Invalid bounded Microsoft cabinet header')
        size, start = struct.unpack_from('<I', header, 8)[0], struct.unpack_from('<I', header, 16)[0]
        count = struct.unpack_from('<H', header, 28)[0]
        if not size <= os.stat(_native(path)).st_size <= ARTIFACT_SIZE or not 0 < count <= 64 or not 36 <= start < size:
            raise ValueError('Microsoft cabinet has invalid file-table bounds')
        stream.seek(start)
        names = []
        for _ in range(count):
            if len(stream.read(16)) != 16:
                raise ValueError('Truncated cabinet file descriptor')
            name = bytearray()
            for _ in range(256):
                byte = stream.read(1)
                if byte == b'\0':
                    break
                if not byte:
                    raise ValueError('Truncated cabinet filename')
                name.extend(byte)
            else:
                raise ValueError('Oversized cabinet filename')
            names.append(name.decode('ascii'))
            if stream.tell() > size:
                raise ValueError('Cabinet file table exceeds its declared bounds')
        if len(set(names)) != len(names) or any(not re.fullmatch(r'[A-Za-z0-9_.]+', name) or name in ('.', '..') for name in names):
            raise ValueError('Cabinet contains linked, escaping or duplicate filenames')
        return set(names)


def _expand(cabinet, destination):
    if os.name != 'nt':
        raise ValueError('CRT cabinet preparation requires native Windows expand.exe')
    buffer = ctypes.create_unicode_buffer(32768)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    if not kernel.GetSystemDirectoryW(buffer, len(buffer)):
        raise ctypes.WinError(ctypes.get_last_error())
    executable = os.path.join(buffer.value, 'expand.exe')
    cabinet, destination = _file(cabinet), _directory(destination, writing=True)
    base = _directory(os.path.commonpath((str(cabinet.parent), str(destination))), writing=True)
    result = subprocess.run([executable, '-F:*', os.path.relpath(cabinet, base),
                             os.path.relpath(destination, base)], cwd=_native(base),
                            capture_output=True, timeout=45, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise ValueError('Windows expand.exe rejected the verified CRT cabinet: ' +
                         (result.stdout + result.stderr).decode('utf-8', errors='replace')[-4096:])


def _xml(path):
    if os.stat(_native(_file(path))).st_size > 1048576:
        raise ValueError('Oversized CRT manifest')
    with open(_native(path), 'rb') as stream:
        raw = stream.read(1048577)
    if b'<!DOCTYPE' in raw.upper():
        raise ValueError('DTD is not allowed in CRT metadata')
    return ET.fromstring(raw)


def prepare(artifact, output):
    data = metadata()
    artifact = verify_artifact(artifact, data)
    output = _path(output, writing=True)
    parent = _directory(output.parent, writing=True)
    if os.path.lexists(_native(output)):
        raise ValueError('CRT preparation output already exists')
    with tempfile.TemporaryDirectory(prefix='spacr-windows-crt-') as temporary:
        work = _directory(temporary, writing=True)
        if os.name == 'nt' and len(str(work)) > 180:
            raise ValueError('Windows cabinet expansion needs a private short temporary directory')
        cabinets = []
        with open(_native(artifact), 'rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as image:
            cursor = 0
            while True:
                cursor = image.find(b'MSCF\0\0\0\0', cursor)
                if cursor < 0:
                    break
                if cursor + 36 > len(image):
                    raise ValueError('Truncated embedded cabinet')
                size = struct.unpack_from('<I', image, cursor + 8)[0]
                if not 36 <= size <= ARTIFACT_SIZE or cursor + size > len(image) or len(cabinets) >= 2:
                    raise ValueError('Invalid embedded cabinet bounds')
                path = work / ('outer' + str(len(cabinets)) + '.cab')
                with open(_native(path), 'xb') as target:
                    for offset in range(cursor, cursor + size, 1048576):
                        target.write(image[offset:min(offset + 1048576, cursor + size)])
                members = _cab_members(path)
                if any(not re.fullmatch(r'(?:0|[ua][0-9]+)', name) for name in members):
                    raise ValueError('Unexpected outer cabinet member name')
                folder = work / ('outer' + str(len(cabinets)))
                os.mkdir(_native(folder), mode=0o700)
                _expand(path, folder)
                if set(os.listdir(_native(folder))) != members:
                    raise ValueError('Expanded cabinet differs from its file table')
                cabinets.append(folder)
                cursor += size
        if len(cabinets) != 2:
            raise ValueError('The pinned CRT release must contain two cabinets')
        manifest = _xml(cabinets[0] / '0')
        registrations = [node for node in manifest.iter() if node.tag.rsplit('}', 1)[-1] == 'Registration']
        if len(registrations) != 1 or registrations[0].get('Version') != VERSION:
            raise ValueError('CRT manifest version does not match the release')
        selected = [node for node in manifest.iter() if node.tag.rsplit('}', 1)[-1] == 'Payload'
                    and node.get('FilePath') == data['cabinet']['file_path']]
        if len(selected) != 1 or not re.fullmatch(r'a[0-9]+', selected[0].get('SourcePath', '')):
            raise ValueError('CRT manifest has no unique x64 minimum cabinet')
        inner = cabinets[1] / selected[0].get('SourcePath')
        if _digest(inner) != data['cabinet']['sha256']:
            raise ValueError('CRT x64 minimum cabinet SHA256 does not match')
        expected = {name[:-4] + '.dll_amd64' for name in MEMBERS}
        if _cab_members(inner) != expected:
            raise ValueError('CRT cabinet must contain exactly the twelve x64 release DLLs')
        payload = work / 'crt'
        os.mkdir(_native(payload), mode=0o700)
        _expand(inner, payload)
        if set(os.listdir(_native(payload))) != expected:
            raise ValueError('Expanded CRT cabinet has unexpected members')
        for name in expected:
            os.rename(_native(payload / name), _native(payload / (name[:-10] + '.dll')))
        _verified_members(payload, data)
        _directory(parent, writing=True)
        if os.path.lexists(_native(output)):
            raise ValueError('CRT output appeared during preparation')
        with tempfile.TemporaryDirectory(prefix='crt-publish-', dir=_native(parent)) as pending:
            publication = Path(pending) / 'payload'
            os.mkdir(_native(publication), mode=0o700)
            for name in MEMBERS:
                with open(_native(payload / name), 'rb') as original, open(_native(publication / name), 'xb') as target:
                    shutil.copyfileobj(original, target, 1048576)
            _verified_members(publication, data)
            if os.path.lexists(_native(output)):
                raise ValueError('CRT output appeared during publication')
            os.rename(_native(publication), _native(output))
    return output


def verify_stage(runtime):
    """Verify existing private venv directories before installing any packages."""
    runtime = _directory(runtime, writing=True)
    config = _file(runtime / 'pyvenv.cfg')
    with open(_native(config), encoding='utf-8') as stream:
        settings = stream.read(16385)
    values = [value.strip().lower() for line in settings.splitlines()
              if not line.lstrip().startswith(('#', ';')) and '=' in line
              for key, value in [line.split('=', 1)]
              if key.strip().lower() == 'include-system-site-packages']
    if len(settings) > 16384 or values != ['false']:
        raise ValueError('CRT destination must belong to a dedicated virtual environment')
    descendant = runtime
    for name in ('Lib', 'site-packages', 'torch', 'lib'):
        descendant = descendant / name
        if not os.path.lexists(_native(descendant)):
            break
        _directory(descendant, writing=True)
    return runtime


def install(source, destination):
    data = metadata()
    source = _verified_members(source, data)
    destination = _directory(destination, writing=True)
    if tuple(part.lower() for part in destination.parts[-4:]) != ('lib', 'site-packages', 'torch', 'lib'):
        raise ValueError('CRT destination must be the staged runtime torch/lib directory')
    verify_stage(destination.parents[3])
    missing = []
    for name, expected in data['members'].items():
        target = destination / name
        if os.path.lexists(_native(target)):
            if _digest(target) != expected:
                raise ValueError('Refusing to overwrite an unknown existing CRT file: ' + name)
        else:
            missing.append(name)
    for name in missing:
        _directory(destination, writing=True)
        with open(_native(source / name), 'rb') as original, open(_native(destination / name), 'xb') as target:
            shutil.copyfileobj(original, target, 1048576)
        if _digest(destination / name) != data['members'][name]:
            raise ValueError('Installed CRT member failed its SHA256 check: ' + name)
    return destination


def _private(path):
    if os.name != 'nt':
        details = os.stat(_native(path))
        if details.st_uid != os.getuid() or details.st_mode & 0o022:
            raise ValueError('CRT work and destination directories must be private')
        return
    evidence = _read_acl(path)
    allowed = {evidence['user'], 'S-1-5-18', 'S-1-5-32-544', 'S-1-3-4'}
    owners = {evidence['user'], 'S-1-5-32-544'}
    if evidence['owner'] not in owners or not evidence['has_dacl'] or not evidence['rules']:
        raise ValueError('CRT directory must have a private current-user or Administrators-owned ACL: ' + str(path))
    for rule in evidence['rules']:
        if rule['sid'] == 'S-1-3-0' and rule.get('propagation', 0) & 2:
            continue
        if rule['type'] == 0 and rule['rights'] & 0x500D0156 and rule['sid'] not in allowed:
            raise ValueError('CRT directory grants another principal write access: ' + str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    first = commands.add_parser('prepare')
    first.add_argument('--artifact', required=True)
    first.add_argument('--output', required=True)
    second = commands.add_parser('install')
    second.add_argument('--source', required=True)
    second.add_argument('--destination', required=True)
    stage = commands.add_parser('verify-stage')
    stage.add_argument('--runtime', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'verify-stage':
            result = verify_stage(args.runtime)
        elif args.command == 'prepare':
            result = prepare(args.artifact, args.output)
        else:
            result = install(args.source, args.destination)
    except (OSError, ValueError, ET.ParseError, struct.error, subprocess.TimeoutExpired) as error:
        parser.exit(1, 'Windows CRT preparation failed: ' + str(error) + '\n')
    if args.command == 'verify-stage':
        print(json.dumps({'verified': True, 'scope': 'private-dedicated-runtime', 'directory': str(result)}))
    else:
        print(json.dumps({'verified': True, 'version': VERSION, 'directory': str(result), 'members': len(MEMBERS)}))
    return 0


def _read_acl(path):
    """Read local fixed-volume ownership and ordinary DACL entries through native Windows APIs."""
    import ctypes

    if os.name != "nt":
        raise ValueError('CRT directory ACL cannot be verified')
    pointer = ctypes.c_void_p
    dword = ctypes.c_uint32
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    volume = kernel.GetVolumePathNameW
    volume.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, dword]
    volume.restype = ctypes.c_int32
    drive_type = kernel.GetDriveTypeW
    drive_type.argtypes = [ctypes.c_wchar_p]
    drive_type.restype = dword
    absolute = os.path.abspath(path)
    if absolute.startswith("\\\\?\\") and absolute[5:7] == ":\\":
        absolute = absolute[4:]
    if absolute.startswith("\\\\") or drive_type(os.path.splitdrive(absolute)[0] + os.sep) != 3:
        raise ValueError('CRT directory ACL cannot be verified')
    root = ctypes.create_unicode_buffer(32768)
    if not volume(_native(absolute), root, len(root)):
        raise OSError(ctypes.get_last_error(), os.strerror(errno.EACCES), path)
    volume_root = root.value
    if volume_root.startswith("\\\\?\\") and volume_root[5:7] == ":\\":
        volume_root = volume_root[4:]
    if volume_root.startswith("\\\\") or drive_type(volume_root) != 3:
        raise ValueError('CRT directory ACL cannot be verified')
    read_security = security.GetNamedSecurityInfoW
    read_security.argtypes = [ctypes.c_wchar_p, dword, dword,
                              ctypes.POINTER(pointer), ctypes.POINTER(pointer),
                              ctypes.POINTER(pointer), ctypes.POINTER(pointer), ctypes.POINTER(pointer)]
    read_security.restype = dword
    get_ace = security.GetAce
    get_ace.argtypes = [pointer, dword, ctypes.POINTER(pointer)]
    get_ace.restype = ctypes.c_int32
    sid_text = security.ConvertSidToStringSidW
    sid_text.argtypes = [pointer, ctypes.POINTER(ctypes.c_wchar_p)]
    sid_text.restype = ctypes.c_int32
    open_token = security.OpenProcessToken
    open_token.argtypes = [pointer, dword, ctypes.POINTER(pointer)]
    open_token.restype = ctypes.c_int32
    token_info = security.GetTokenInformation
    token_info.argtypes = [pointer, dword, pointer, dword, ctypes.POINTER(dword)]
    token_info.restype = ctypes.c_int32
    current_process = kernel.GetCurrentProcess
    current_process.argtypes = []
    current_process.restype = pointer
    free = kernel.LocalFree
    free.argtypes = [pointer]
    free.restype = pointer
    close = kernel.CloseHandle
    close.argtypes = [pointer]
    close.restype = ctypes.c_int32
    owner, dacl, descriptor, token = pointer(), pointer(), pointer(), pointer()
    code = read_security(_native(absolute), 1, 5, ctypes.byref(owner), None,
                         ctypes.byref(dacl), None, ctypes.byref(descriptor))
    if code:
        raise OSError(code, os.strerror(errno.EACCES), path)
    try:
        if not open_token(current_process(), 8, ctypes.byref(token)):
            raise OSError(ctypes.get_last_error(), os.strerror(errno.EACCES), path)
        needed = dword()
        token_info(token, 1, None, 0, ctypes.byref(needed))
        if not ctypes.sizeof(pointer) <= needed.value <= 1 << 20:
            raise ValueError('CRT directory ACL cannot be verified')
        user_buffer = ctypes.create_string_buffer(needed.value)
        if not token_info(token, 1, user_buffer, needed, ctypes.byref(needed)):
            raise OSError(ctypes.get_last_error(), os.strerror(errno.EACCES), path)
        user_pointer = ctypes.cast(user_buffer, ctypes.POINTER(pointer)).contents
        owner_text, user_text = ctypes.c_wchar_p(), ctypes.c_wchar_p()
        try:
            if not sid_text(owner, ctypes.byref(owner_text)) or not sid_text(user_pointer, ctypes.byref(user_text)):
                raise OSError(ctypes.get_last_error(), os.strerror(errno.EACCES), path)
            result = {"owner": owner_text.value, "user": user_text.value, "has_dacl": bool(dacl), "rules": []}
        finally:
            if owner_text:
                free(ctypes.cast(owner_text, pointer))
            if user_text:
                free(ctypes.cast(user_text, pointer))
        if dacl:
            header = ctypes.string_at(dacl, 8)
            size = int.from_bytes(header[2:4], "little")
            count = int.from_bytes(header[4:6], "little")
            if size < 8 or count > (size - 8) // 12:
                raise ValueError('CRT directory ACL cannot be verified')
            for index in range(count):
                ace = pointer()
                if not get_ace(dacl, index, ctypes.byref(ace)):
                    raise OSError(ctypes.get_last_error(), os.strerror(errno.EACCES), path)
                entry = ctypes.string_at(ace, 8)
                length = int.from_bytes(entry[2:4], "little")
                if (entry[0] not in {0, 1} or entry[1] & ~0x1f or length < 12
                        or ace.value < dacl.value + 8 or ace.value + length > dacl.value + size):
                    raise ValueError('CRT directory ACL cannot be verified')
                text = ctypes.c_wchar_p()
                try:
                    if not sid_text(pointer(ace.value + 8), ctypes.byref(text)):
                        raise OSError(ctypes.get_last_error(), os.strerror(errno.EACCES), path)
                    result["rules"].append({"sid": text.value, "type": entry[0],
                                            "rights": int.from_bytes(entry[4:8], "little"),
                                            "propagation": (1 if entry[1] & 4 else 0) | (2 if entry[1] & 8 else 0)})
                finally:
                    if text:
                        free(ctypes.cast(text, pointer))
        return result
    finally:
        if token:
            close(token)
        if descriptor:
            free(descriptor)



if __name__ == '__main__':
    raise SystemExit(main())
