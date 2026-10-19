"""Find spaCR installations and run the appropriate update or removal procedure.

The online installers stage and validate the new runtime before retiring old
application copies. They preserve the current installation during staging and
restore its previous runtime if the replacement fails verification. The update
helper verifies the desktop interpreter and imported application before retiring
other copies or requesting a restart. :func:`run_update_sequence` remains the
low-level removal procedure used by explicit cleanup commands.

Supported macOS online installations and dedicated POSIX or Windows pip
environments use their existing package tool to stage and verify an updated
sibling runtime before requesting Restart. Activation requires approved
shutdown. Bootstrap files, launchers, unchanged packages and the previous
runtime are retained. Shared and Conda environments require their own
replacement adapter.

A supported frozen macOS, Windows or Debian installation stages its verified
replacement while the desktop remains open. Protected installations obtain
graphical administrator authorization before requesting Restart. Activation
requires approved shutdown and actual process exit; recovery data remains
until the updated desktop confirms its identity and main window.

Ordinary removal preserves preferences and user data. The confirmed GUI
uninstall instead previews current or all detected installations and separate
cleanup categories, selected by default with individual opt-outs. Protected
removal uses only a validated installed helper with the required protocol;
authorization, preparation and approved desktop exit precede deletion.
Unrelated projects, unverified data and shared environments are preserved.

A pip or conda environment the user made is listed but never deleted; the
spaCR package is uninstalled from it only when the caller says it was ticked.
An editable source checkout is reported and left alone.

The module uses only the standard library and imports nothing from spaCR, so
an installer can run it with its bootstrapped Python before any spaCR exists,
and a helper process can run a copy of it after the running spaCR has closed::

    python -I install_cleanup.py find [--json]
    python -I install_cleanup.py remove [--keep PATH] [--sudo] [--purge [--yes]]
    python -I install_cleanup.py purge [--yes] [--dry-run]
    python -I install_cleanup.py run-plan PLAN.json

``purge`` is the opt-in clean uninstall: it also deletes spaCR's caches,
backend environments and user data, after listing them and asking. It never
runs on its own; ``remove --purge`` runs it after the removal succeeded.
"""
from __future__ import annotations

import argparse
import errno
import glob
import hashlib
import json
import os
import plistlib
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

_OLD_NAME = "Spa" + "CR"
_NAME = "spaCR"
_NAMES = (_NAME, _OLD_NAME)

_UNINSTALL_KEY = "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\spaCR"
_SOFTWARE_KEY = "Software\\spaCR"
_SHELL_FOLDERS_KEY = (
    "Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\User Shell Folders")
_DEB_PACKAGES = ("python3-spacr", "spacr")
_DISTRIBUTIONS = ("spacr", "spacr_nightly")
_RELEASE_DOWNLOAD = "https://github.com/EinarOlafsson/spacr/releases/download"
_ASSET_SUFFIX = {
    "windows": "Windows-Online-Setup.exe",
    "linux": "Linux-x86_64-Online.run",
    "macos": "macOS-Universal-Online.pkg",
}
_WAIT_SECONDS = 600.0
_FROZEN_PREPARATION_SECONDS = 3600.0
#: An update preparing without any sign of progress for this long has stalled.
_PREPARATION_STALL_SECONDS = 900.0
#: However much progress it reports, an update plan expires this long after its deadline.
_PROGRESS_CAP_SECONDS = 24 * 3600.0
_RMTREE_TAKES_ONEXC = sys.version_info >= (3, 12)
_PRIVILEGED_DEBIAN_HELPER = False
_MACOS_FROZEN_PHASE_COMMANDS = ("apply-macos-root", "uninstall-macos-root")


@dataclass(frozen=True)
class InstallRecord:
    """One spaCR installation found on this computer.

    :param kind: ``"installer"`` for a copy a spaCR installer made,
        ``"environment"`` for a pip or conda environment the user made, or
        ``"checkout"`` for an editable source checkout.
    :param layout: which layout it is, for example ``"windows-online"``,
        ``"linux-online"``, ``"macos-app"`` or ``"conda"``.
    :param platform: ``"windows"``, ``"macos"`` or ``"linux"``.
    :param root: the directory the installation lives in.
    :param version: the spaCR version, when it can be read.
    :param launchers: command launchers that start this copy.
    :param shortcuts: desktop and Start Menu shortcut files.
    :param menu_entries: application-menu entries: ``.desktop`` files and
        Start Menu folders.
    :param registrations: uninstall registrations: ``registry:`` keys,
        ``deb:`` packages, and application bundles.
    :param python: the environment's interpreter, used to uninstall spaCR.
    :param running: whether the current process runs from this copy.
    :param needs_admin: whether removal needs administrator rights.
    :param notes: other facts worth showing, such as a missing uninstaller.
    """

    kind: str
    layout: str
    platform: str
    root: str
    version: Optional[str] = None
    launchers: Tuple[str, ...] = ()
    shortcuts: Tuple[str, ...] = ()
    menu_entries: Tuple[str, ...] = ()
    registrations: Tuple[str, ...] = ()
    python: Optional[str] = None
    running: bool = False
    needs_admin: bool = False
    notes: Tuple[str, ...] = ()


@dataclass
class RemovalReport:
    """What :func:`remove_install` did with one installation.

    :param record: the installation this report is about.
    :param removed: every path, registry entry and package that was removed.
    :param failed: ``(item, reason)`` for everything that could not be removed.
    :param skipped: ``(item, reason)`` for everything left in place on purpose.
    """

    record: InstallRecord
    removed: List[str] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether nothing failed."""
        return not self.failed



class _WindowsRegistry:
    """``HKEY_CURRENT_USER`` read and written through :mod:`winreg`."""

    def _open(self, key: str, write: bool = False):
        """Open ``key`` under HKCU, or return ``None`` when it is absent.

        :param key: path below HKCU.
        :param write: open for modification rather than reading.
        """
        import winreg
        access = winreg.KEY_ALL_ACCESS if write else winreg.KEY_READ
        try:
            return winreg.OpenKey(winreg.HKEY_CURRENT_USER, key, 0, access)
        except OSError:
            return None

    def values(self, key: str) -> Dict[str, str]:
        """Return the named values of ``key``; empty when it does not exist.

        :param key: path below HKCU.
        """
        import winreg
        handle = self._open(key)
        found: Dict[str, str] = {}
        if handle is None:
            return found
        with handle:
            index = 0
            while True:
                try:
                    name, value, _kind = winreg.EnumValue(handle, index)
                except OSError:
                    break
                found[str(name)] = str(value)
                index += 1
        return found

    def subkeys(self, key: str) -> List[str]:
        """Return the names of the keys directly below ``key``.

        :param key: path below HKCU.
        """
        import winreg
        handle = self._open(key)
        names: List[str] = []
        if handle is None:
            return names
        with handle:
            index = 0
            while True:
                try:
                    names.append(winreg.EnumKey(handle, index))
                except OSError:
                    break
                index += 1
        return names

    def delete_value(self, key: str, name: str) -> None:
        """Delete one named value.

        :param key: path below HKCU.
        :param name: the value to delete.
        """
        import winreg
        handle = self._open(key, write=True)
        if handle is None:
            return
        with handle:
            winreg.DeleteValue(handle, name)

    def delete_key(self, key: str) -> None:
        """Delete a key that has no subkeys.

        :param key: path below HKCU.
        """
        import winreg
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key)


class _SystemPackages:
    """The Debian package manager, queried and asked to remove a package."""

    def __init__(self, runner, which=None, geteuid=None):
        """Keep the command runner.

        :param runner: callable taking an argv and returning
            ``(exit_code, output)``.
        :param which: finds a command on ``PATH``; :func:`shutil.which` when
            ``None``.
        :param geteuid: returns the effective user id; :func:`os.geteuid`
            when ``None``, and no check where the platform has none.
        """
        self._runner = runner
        self._which = which or shutil.which
        self._geteuid = geteuid or getattr(os, "geteuid", None)

    def version(self, name: str) -> Optional[str]:
        """Return the installed version of package ``name``, or ``None``.

        :param name: Debian package name.
        """
        if not self._which("dpkg-query"):
            return None
        code, output = self._runner(
            ["dpkg-query", "-W", "-f=${Version}", name])
        text = (output or "").strip()
        return text if code == 0 and text and " " not in text else None

    def owns(self, name: str, path: str) -> bool:
        """Whether dpkg lists this exact path as a file owned by the package.

        :param name: Debian package name.
        :param path: exact installed file path whose ownership is required.
        """
        if not self._which("dpkg-query"):
            return False
        code, output = self._runner(["dpkg-query", "-L", name])
        return code == 0 and os.path.normpath(path) in {
            os.path.normpath(line) for line in (output or "").splitlines()
        }

    def remove(self, name: str, sudo: bool) -> Tuple[bool, str]:
        """Remove package ``name``; say why when it could not be done.

        :param name: Debian package name.
        :param sudo: whether the caller allows an interactive ``sudo``.
        :returns: ``(removed, reason)``.
        """
        argv = ["dpkg", "-r", name]
        if self._geteuid is not None and self._geteuid() != 0:
            if not sudo:
                return False, _needs_admin(f"sudo dpkg -r {name}")
            argv = ["sudo"] + argv
        code, output = self._runner(argv)
        if code == 0:
            return True, ""
        lines = (output or "").strip().splitlines()
        return False, lines[-1] if lines else f"exit code {code}"


def _run(argv: Sequence[str], timeout: float = 1800.0) -> Tuple[int, str]:
    """Run one command and capture what it said.

    :param argv: the command.
    :param timeout: seconds before it is given up on.
    :returns: ``(exit_code, output)``.
    """
    try:
        done = subprocess.run([str(part) for part in argv],
                              capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        return 127, f"could not run {argv[0]}: {exc}"
    except subprocess.TimeoutExpired:
        return 124, f"{argv[0]} did not finish within {int(timeout)} seconds"
    return done.returncode, (done.stdout or "") + (done.stderr or "")


class _Machine:
    """Where this computer keeps things, read through one replaceable seam.

    Tests build one over a temporary directory, with a fake registry and a
    fake package manager, so nothing outside the sandbox is ever looked at
    or touched.
    """

    def __init__(self, platform: Optional[str] = None, environ=None,
                 fs_root: str = "/", registry=None, packages=None,
                 runner=None, running_prefix=None, executable=None,
                 sudo: bool = False, which=None):
        """Describe a computer.

        :param platform: ``"windows"``, ``"macos"`` or ``"linux"``; the
            current platform when ``None``.
        :param environ: environment variables; :data:`os.environ` when
            ``None``.
        :param fs_root: directory standing in for ``/`` for absolute system
            paths such as ``/Applications``.
        :param registry: HKCU adapter; the real registry on Windows.
        :param packages: Debian package adapter; ``dpkg`` on a real Linux.
        :param runner: command runner returning ``(exit_code, output)``.
        :param running_prefix: ``sys.prefix`` of the running spaCR; the
            current one when ``None`` on a real computer.
        :param executable: interpreter of the running process.
        :param sudo: whether removal may ask for ``sudo`` interactively.
        :param which: finds a command on ``PATH``; :func:`shutil.which` when
            ``None``. Consulted only on a real computer.
        """
        if platform is None:
            platform = ("windows" if os.name == "nt" else
                        "macos" if sys.platform == "darwin" else "linux")
        self.platform = platform
        self.real = environ is None and fs_root in ("/", "")
        self.environ = dict(os.environ if environ is None else environ)
        self.fs_root = os.path.abspath(fs_root or "/")
        self.runner = runner or _run
        self.which = which or shutil.which
        if registry is None and self.real and platform == "windows":
            registry = _WindowsRegistry()
        self.registry = registry
        if packages is None and self.real and platform == "linux":
            packages = _SystemPackages(self.runner, self.which)
        self.packages = packages
        if running_prefix is None and self.real:
            running_prefix = sys.prefix
        self.running_prefix = running_prefix
        self.executable = executable or sys.executable
        self.sudo = bool(sudo)

    def env(self, name: str) -> str:
        """Return an environment variable, or ``""``.

        :param name: variable name, looked up exactly and then upper-cased.
        """
        value = self.environ.get(name)
        if value is None:
            value = self.environ.get(name.upper(), "")
        return str(value or "")

    @property
    def home(self) -> str:
        """The user's home directory."""
        name = "USERPROFILE" if self.platform == "windows" else "HOME"
        value = self.env(name) or self.env("HOME")
        if not value and self.real:
            value = os.path.expanduser("~")
        return value

    def path(self, absolute: str) -> str:
        """Map an absolute system path into :attr:`fs_root`.

        :param absolute: a POSIX path such as ``/Applications``.
        """
        if self.fs_root == os.path.abspath("/"):
            return absolute
        return os.path.join(self.fs_root, absolute.lstrip("/"))

    def unmapped(self, path: str) -> str:
        """Return ``path`` as the real computer would spell it.

        :param path: a path produced by :meth:`path`.
        """
        if self.fs_root == os.path.abspath("/"):
            return path
        if _inside(path, self.fs_root):
            return "/" + os.path.relpath(path, self.fs_root).replace(os.sep, "/")
        return path

    def xdg(self, name: str, fallback: str) -> str:
        """Return an XDG base directory.

        :param name: the variable, for example ``XDG_DATA_HOME``.
        :param fallback: the path below home used when it is unset.
        """
        return self.env(name) or os.path.join(self.home, fallback)



def _norm(path: str) -> str:
    """Return a comparable spelling of ``path``: absolute and lower case.

    :param path: any path.
    """
    return os.path.normpath(os.path.abspath(str(path))).lower()


def _inside(path: str, parent: str) -> bool:
    """Whether ``path`` is ``parent`` or below it, ignoring case.

    :param path: the candidate.
    :param parent: the directory.
    """
    p, q = _norm(path), _norm(parent)
    return p == q or p.startswith(q.rstrip(os.sep) + os.sep)


def _exists(path: str) -> bool:
    """Whether ``path`` exists, counting a dangling symbolic link.

    :param path: any path.
    """
    return os.path.lexists(path)


def _identity(path: str):
    """Return what makes two spellings of one directory the same directory.

    On a case-insensitive file system the old capitalised folder name and
    ``spaCR`` are one folder, and the device and inode say so where a string
    comparison cannot.

    :param path: an existing path.
    """
    try:
        info = os.stat(path)
        return (info.st_dev, info.st_ino)
    except OSError:
        return ("missing", _norm(path))


def _mentions(path: str, needles: Iterable[str]) -> bool:
    """Whether a file, shortcut or link names any of ``needles``.

    Windows shortcuts store their target as text in either an 8-bit or a
    UTF-16 encoding, so both readings are searched.

    :param path: a file or symbolic link.
    :param needles: strings to look for, compared without case.
    """
    wanted = [w for w in (str(n).lower().rstrip("/\\") for n in needles if n) if w]
    if not wanted:
        return False
    texts = []
    try:
        if os.path.islink(path):
            texts.append(os.readlink(path))
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                raw = handle.read(1 << 20)
            texts.append(raw.decode("latin-1"))
            texts.append(raw.decode("utf-16-le", errors="ignore"))
    except OSError:
        return False
    for text in texts:
        low = text.lower().replace("\\\\", "\\")
        if any(_names_whole_path(low, needle) for needle in wanted):
            return True
    return False


def _names_whole_path(text: str, path: str) -> bool:
    """Whether ``text`` names ``path`` itself, not a longer name it begins.

    ``spacr`` begins ``spacr-dev``, so a launcher for a folder beside an old
    copy must not count as the old copy's. A name ends at a path separator,
    a quote, a control character, or the end of the text.

    :param text: the text to search, lower case.
    :param path: the path to look for, lower case.
    """
    start = text.find(path)
    while start >= 0:
        end = start + len(path)
        if end == len(text) or text[end] in "/\\\"'" or text[end] < " ":
            return True
        start = text.find(path, start + 1)
    return False


def _spellings(path: str) -> List[str]:
    """Return ``path`` in both separator styles, for matching inside files.

    :param path: a path.
    """
    text = str(path)
    return sorted({text, text.replace("\\", "/"), text.replace("/", "\\")})


def _dedupe(paths: Iterable[str]) -> List[str]:
    """Return the existing ``paths`` once each, in order.

    :param paths: candidate paths, possibly different spellings of one.
    """
    seen, out = set(), []
    for path in paths:
        if not path or not _exists(path):
            continue
        key = _identity(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _needs_admin(command: str = "") -> str:
    """Return the reason text for a removal that needs administrator rights.

    :param command: the command a user can run instead, if there is one.
    """
    reason = "needs administrator rights"
    return f"{reason}; run: {command}" if command else (
        f"{reason}; delete it as an administrator")



def _site_packages(prefix: str) -> List[str]:
    """Return every ``site-packages`` directory of an environment.

    :param prefix: the environment's prefix.
    """
    patterns = (
        os.path.join(prefix, "lib", "python3*", "site-packages"),
        os.path.join(prefix, "Lib", "site-packages"),
        os.path.join(prefix, "lib", "site-packages"),
    )
    found: List[str] = []
    for pattern in patterns:
        found.extend(sorted(glob.glob(pattern)))
    return _dedupe(found)


def _distributions(prefix: str) -> List[str]:
    """Return the spaCR ``.dist-info`` directories of an environment.

    :param prefix: the environment's prefix.
    """
    found: List[str] = []
    for site in _site_packages(prefix):
        for name in _DISTRIBUTIONS:
            found.extend(sorted(glob.glob(os.path.join(site, f"{name}-*.dist-info"))))
    return found


def _dist_version(dist_info: str) -> Optional[str]:
    """Read the version of one ``.dist-info`` directory.

    :param dist_info: the directory.
    """
    try:
        with open(os.path.join(dist_info, "METADATA"), encoding="utf-8",
                  errors="replace") as handle:
            for line in handle:
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip() or None
                if not line.strip():
                    break
    except OSError:
        pass
    name = os.path.basename(dist_info)[:-len(".dist-info")]
    return name.split("-", 1)[1] if "-" in name else None


def _dist_name(dist_info: str) -> str:
    """Return the distribution name pip knows a ``.dist-info`` by.

    :param dist_info: the directory.
    """
    return os.path.basename(dist_info).split("-", 1)[0].replace("_", "-")


def _editable_source(dist_info: str) -> Optional[str]:
    """Return the checkout an editable install points at, or ``None``.

    :param dist_info: the directory.
    """
    try:
        with open(os.path.join(dist_info, "direct_url.json"),
                  encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return None
    if not (record.get("dir_info") or {}).get("editable"):
        return None
    url = str(record.get("url") or "")
    if url.startswith("file://"):
        from urllib.parse import unquote, urlparse
        return unquote(urlparse(url).path) or url
    return url or None


def _env_version(prefix: str) -> Optional[str]:
    """Return the spaCR version installed in an environment, if any.

    :param prefix: the environment's prefix.
    """
    for dist in _distributions(prefix):
        version = _dist_version(dist)
        if version:
            return version
    return None


def _env_python(prefix: str) -> Optional[str]:
    """Return an environment's interpreter.

    :param prefix: the environment's prefix.
    """
    for relative in (("bin", "python"), ("bin", "python3"), ("python.exe",),
                     ("Scripts", "python.exe")):
        candidate = os.path.join(prefix, *relative)
        if os.path.isfile(candidate):
            return candidate
    return None



def find_old_installs(*, system=None) -> List[InstallRecord]:
    """Find every spaCR installation on this computer.

    Installer-made copies come first, then environments the user made, then
    editable source checkouts. Every layout any released installer used is
    looked for: the Windows online and offline installers, the Linux and
    macOS online installers, the macOS application bundle, and the Debian
    package.

    :param system: the computer to look at; ``None`` means this one. Tests
        pass a stand-in over a temporary directory.
    :returns: one :class:`InstallRecord` per installation.
    """
    machine = system or _Machine()
    records: List[InstallRecord] = []
    if machine.platform == "windows":
        records.extend(_find_windows_online(machine))
        records.extend(_find_windows_offline(machine))
    else:
        if machine.platform == "macos":
            records.extend(_find_macos_apps(machine))
        records.extend(_find_unix_online(machine, records))
        if machine.platform == "linux":
            records.extend(_find_deb(machine))
    records.extend(_find_environments(machine, records))
    return records


def _running(machine: _Machine, root: str) -> bool:
    """Whether the running spaCR lives inside ``root``.

    :param machine: the computer.
    :param root: an installation directory.
    """
    prefix = machine.running_prefix
    if bool(prefix) and _inside(prefix, root):
        return True
    return (bool(getattr(sys, "frozen", False))
            and os.path.realpath(machine.executable) == os.path.realpath(sys.executable)
            and _inside(machine.executable, root))


def _windows_start_menu(machine: _Machine) -> str:
    """Return the per-user Start Menu ``Programs`` folder.

    :param machine: the computer.
    """
    return os.path.join(machine.env("APPDATA"), "Microsoft", "Windows",
                        "Start Menu", "Programs")


def _windows_desktops(machine: _Machine) -> List[str]:
    """Return the folders the Desktop may be, including a redirected one.

    :param machine: the computer.
    """
    folders = [os.path.join(machine.home, "Desktop"),
               os.path.join(machine.home, "OneDrive", "Desktop")]
    if machine.registry is not None:
        value = machine.registry.values(_SHELL_FOLDERS_KEY).get("Desktop", "")
        if value:
            for name in ("USERPROFILE", "HOMEDRIVE", "HOMEPATH", "OneDrive"):
                value = value.replace(f"%{name}%", machine.env(name))
            folders.insert(0, value)
    return folders


def _find_windows_online(machine: _Machine) -> List[InstallRecord]:
    """Find copies made by the Windows online installer.

    :param machine: the computer.
    """
    registry = machine.registry
    uninstall = registry.values(_UNINSTALL_KEY) if registry else {}
    software = registry.values(_SOFTWARE_KEY) if registry else {}
    local = machine.env("LOCALAPPDATA")
    named = [uninstall.get("InstallLocation", ""), software.get("InstallRoot", "")]
    defaults = [os.path.join(local, name) for name in _NAMES] if local else []
    markers = ("venv", "Uninstall.exe", "launch_spacr.pyw", "spacr.cmd",
               os.path.join("bootstrap", "uv.exe"))
    roots = [root for root in _dedupe(named + defaults)
             if os.path.isdir(root)
             and any(_exists(os.path.join(root, m)) for m in markers)]
    if not roots and (uninstall or software.get("InstallRoot")):
        gone = next((p for p in named if p), defaults[0] if defaults else "")
        roots = [gone] if gone else []
    menu = _windows_start_menu(machine)
    desktops = _windows_desktops(machine)
    records = []
    for root in roots:
        needles = _spellings(root)
        shortcuts = [link for link in _dedupe(
            os.path.join(folder, f"{name}.lnk")
            for folder in desktops for name in _NAMES)
            if _mentions(link, needles)]
        folders = []
        for folder in _dedupe(os.path.join(menu, name) for name in _NAMES):
            if not os.path.isdir(folder):
                continue
            links = glob.glob(os.path.join(folder, "*.lnk"))
            if not links or any(_mentions(link, needles) for link in links):
                folders.append(folder)
        registrations = []
        location = uninstall.get("InstallLocation", "")
        if uninstall and (not location or _norm(location) == _norm(root)):
            registrations.append(f"registry:HKCU\\{_UNINSTALL_KEY}")
        install_root = software.get("InstallRoot", "")
        if install_root and _norm(install_root) == _norm(root):
            registrations.append(f"registry-value:HKCU\\{_SOFTWARE_KEY}\\InstallRoot")
        venv = os.path.join(root, "venv")
        notes = []
        if not os.path.isdir(root):
            notes.append("its folder is already gone")
        elif not _exists(os.path.join(root, "Uninstall.exe")):
            notes.append("its own uninstaller is missing")
        records.append(InstallRecord(
            kind="installer", layout="windows-online", platform="windows",
            root=root,
            version=_env_version(venv) or uninstall.get("DisplayVersion") or None,
            launchers=tuple(p for p in (os.path.join(root, "launch_spacr.pyw"),
                                        os.path.join(root, "spacr.cmd"))
                            if _exists(p)),
            shortcuts=tuple(shortcuts), menu_entries=tuple(folders),
            registrations=tuple(registrations),
            python=_env_python(venv), running=_running(machine, root),
            notes=tuple(notes)))
    return records


def _find_windows_offline(machine: _Machine) -> List[InstallRecord]:
    """Find copies made by the offline Windows installer in Program Files.

    :param machine: the computer.
    """
    bases = [machine.env("ProgramW6432"), machine.env("PROGRAMFILES")]
    roots = [root for root in _dedupe(
        os.path.join(base, name) for base in bases if base for name in _NAMES)
        if any(_exists(os.path.join(root, m))
               for m in ("spacr.exe", "Uninstall.exe", "_internal"))]
    menu = _windows_start_menu(machine)
    menus = [menu]
    if machine.real and os.name == "nt":
        menus.append(_windows_known_folder(0x17))
    records = []
    for root in roots:
        needles = _spellings(root)
        shortcuts = [link for link in _dedupe(
            os.path.join(folder, f"{name}.lnk") for folder in menus for name in _NAMES)
            if _mentions(link, needles)]
        notes = () if _exists(os.path.join(root, "Uninstall.exe")) else (
            "its own uninstaller is missing",)
        records.append(InstallRecord(
            kind="installer", layout="windows-offline", platform="windows",
            root=root, launchers=tuple(
                p for p in (os.path.join(root, "spacr.exe"),) if _exists(p)),
            shortcuts=tuple(shortcuts), running=_running(machine, root),
            needs_admin=True, notes=notes))
    return records


def _plist_version(app: str) -> Optional[str]:
    """Read ``CFBundleShortVersionString`` from an application bundle.

    :param app: the ``.app`` directory.
    """
    try:
        with open(os.path.join(app, "Contents", "Info.plist"), "rb") as handle:
            info = plistlib.load(handle)
    except Exception:                                        # noqa: BLE001
        return None
    return str(info.get("SPACRPackageVersion") or info.get("CFBundleShortVersionString") or "") or None


def _find_macos_apps(machine: _Machine) -> List[InstallRecord]:
    """Find the macOS application bundle and the files its package installed.

    :param machine: the computer.
    """
    apps = _dedupe(machine.path(f"/Applications/{name}.app") for name in _NAMES)
    current_app = ""
    if (getattr(sys, "frozen", False)
            and os.path.realpath(machine.executable) == os.path.realpath(sys.executable)):
        executable_folder = os.path.dirname(os.path.realpath(machine.executable))
        contents = os.path.dirname(executable_folder)
        candidate = os.path.dirname(contents)
        if (os.path.basename(executable_folder) == "MacOS"
                and os.path.basename(contents) == "Contents" and candidate.endswith(".app")):
            current_app = candidate
            if current_app not in apps:
                apps.append(current_app)
    supports = _dedupe(machine.path(f"/Library/Application Support/{name}")
                       for name in _NAMES)
    command = machine.path("/usr/local/bin/spacr")
    records = []
    for index, app in enumerate(apps):
        support = supports[0] if supports and index == 0 and app != current_app else ""
        needles = [os.path.basename(app) + "/contents/macos",
                   machine.unmapped(app)]
        launchers = [command] if _exists(command) and _mentions(command, needles) else []
        notes = []
        if support and not _exists(os.path.join(support, "uninstall-spacr.sh")):
            notes.append("its own uninstaller is missing")
        records.append(InstallRecord(
            kind="installer", layout="macos-app", platform="macos",
            root=support or app, version=_plist_version(app),
            launchers=tuple(launchers), registrations=(app,),
            running=_running(machine, app) or bool(
                support and _running(machine, support)),
            needs_admin=True, notes=tuple(notes)))
    if supports and not apps:
        records.append(InstallRecord(
            kind="installer", layout="macos-app", platform="macos",
            root=supports[0], running=_running(machine, supports[0]),
            needs_admin=True, notes=("its application bundle is already gone",)))
    return records


def _uninstall_script_paths(root: str) -> List[str]:
    """Return the files an online installer's ``uninstall-spacr.sh`` names.

    :param root: the installation directory.
    """
    paths = []
    try:
        with open(os.path.join(root, "uninstall-spacr.sh"), encoding="utf-8",
                  errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("rm -f "):
                    paths.append(line[len("rm -f "):].strip().strip('"'))
                elif line.startswith("# SPACR_UNINSTALL_PATH "):
                    paths.append(line[len("# SPACR_UNINSTALL_PATH "):])
    except OSError:
        pass
    return paths


def _find_unix_online(machine: _Machine,
                      claimed: Sequence[InstallRecord] = ()) -> List[InstallRecord]:
    """Find copies made by the Linux and macOS online installer.

    :param machine: the computer.
    :param claimed: copies already found, whose folders are not listed twice.
    """
    data = machine.xdg("XDG_DATA_HOME", os.path.join(".local", "share"))
    bin_dirs = [machine.env("XDG_BIN_HOME"), os.path.join(machine.home, ".local", "bin")]
    if machine.platform == "macos":
        bin_dirs.append(machine.path("/usr/local/bin"))
    launchers_seen = [os.path.join(d, "spacr") for d in bin_dirs if d]
    candidates = [os.path.join(data, "spacr"),
                  os.path.join(machine.home, ".local", "share", "spacr")]
    layout = "linux-online"
    if machine.platform == "macos":
        layout = "macos-online"
        support = os.path.join(machine.home, "Library", "Application Support")
        candidates += [os.path.join(support, name) for name in _NAMES]
        candidates += [machine.path(f"/Library/Application Support/{name}")
                       for name in _NAMES]
    named_by_launchers = []
    for launcher in launchers_seen:
        if os.path.isfile(launcher) and not os.path.islink(launcher):
            try:
                with open(launcher, encoding="utf-8", errors="replace") as handle:
                    text = handle.read(4096)
            except OSError:
                continue
            marker = "/venv/bin/python"
            if marker in text:
                start = text.rfind('"', 0, text.index(marker)) + 1
                named_by_launchers.append(text[start:text.index(marker)])
    markers = (os.path.join("venv", "bin", "python"), os.path.join("bootstrap", "uv"),
               "uninstall-spacr.sh")
    installer_only = markers[1:]
    taken = {_identity(r.root) for r in claimed if _exists(r.root)}
    roots = [root for root in _dedupe(candidates + named_by_launchers)
             if os.path.isdir(root) and _identity(root) not in taken
             and any(_exists(os.path.join(root, m))
                     for m in (markers if root in candidates else installer_only))]
    apps_dir = os.path.join(data, "applications")
    records = []
    for root in roots:
        needles = _spellings(root)
        named = _uninstall_script_paths(root)
        launchers = [p for p in _dedupe(launchers_seen + [
            p for p in named if os.path.basename(p) == "spacr"])
            if _mentions(p, needles)]
        desktop = [p for p in _dedupe(
            [os.path.join(apps_dir, "spacr.desktop"),
             os.path.join(apps_dir, "io.github.olafssonlab.spacr.desktop")]
            + [p for p in named if p.endswith(".desktop")])
            if _mentions(p, needles)]
        notes = []
        if machine.platform == "linux" and not _exists(
                os.path.join(root, "uninstall-spacr.sh")):
            notes.append("its own uninstaller is missing")
        venv = os.path.join(root, "venv")
        is_runtime = "application support" in _norm(root)
        records.append(InstallRecord(
            kind="installer", layout=("macos-runtime" if is_runtime else layout),
            platform=machine.platform, root=root, version=_env_version(venv),
            launchers=tuple(launchers), menu_entries=tuple(desktop),
            python=_env_python(venv), running=_running(machine, root),
            needs_admin=root.startswith(machine.path("/Library")),
            notes=tuple(notes)))
    return records


def _find_deb(machine: _Machine) -> List[InstallRecord]:
    """Find the Debian package.

    :param machine: the computer.
    """
    if machine.packages is None:
        return []
    records = []
    for name in _DEB_PACKAGES:
        version = machine.packages.version(name)
        if not version:
            continue
        frozen_root = machine.path("/opt/spacr")
        frozen_binary = os.path.join(frozen_root, "spacr")
        owns = getattr(machine.packages, "owns", None)
        root = (frozen_root if name == "spacr"
                and os.path.isfile(frozen_binary)
                and callable(owns) and owns(name, frozen_binary)
                else machine.path("/usr/lib/python3/dist-packages/spacr"))
        launcher = machine.path("/usr/bin/spacr")
        records.append(InstallRecord(
            kind="installer", layout="linux-deb", platform="linux", root=root,
            version=version,
            launchers=(launcher,) if _exists(launcher) else (),
            registrations=(f"deb:{name}",),
            running=_running(machine, root), needs_admin=True))
    return records


def _conda_environments(machine: _Machine) -> List[str]:
    """Return every conda environment this user is known to have.

    :param machine: the computer.
    """
    home = machine.home
    prefixes: List[str] = []
    try:
        with open(os.path.join(home, ".conda", "environments.txt"),
                  encoding="utf-8", errors="replace") as handle:
            prefixes.extend(line.strip() for line in handle if line.strip())
    except OSError:
        pass
    bases = [os.path.join(home, name) for name in (
        "anaconda3", "miniconda3", "miniforge3", "mambaforge", "micromamba")]
    conda_exe = machine.env("CONDA_EXE")
    if conda_exe:
        bases.append(os.path.dirname(os.path.dirname(conda_exe)))
    if machine.platform != "windows":
        bases += [machine.path("/opt/conda"), machine.path("/opt/anaconda3")]
    else:
        bases += [os.path.join(machine.env("LOCALAPPDATA"), name)
                  for name in ("anaconda3", "miniconda3")]
    for base in bases:
        prefixes.append(base)
        prefixes.extend(sorted(glob.glob(os.path.join(base, "envs", "*"))))
    return prefixes


def _find_environments(machine: _Machine,
                       installers: Sequence[InstallRecord]) -> List[InstallRecord]:
    """Find environments the user made, and editable checkouts, holding spaCR.

    :param machine: the computer.
    :param installers: installer-made copies already found; their private
        environments are not listed again.
    """
    home = machine.home
    prefixes = [(p, "conda") for p in _conda_environments(machine)]
    for folder in (".virtualenvs", ".venvs", "venvs", "Envs",
                   os.path.join(".pyenv", "versions"),
                   os.path.join(".local", "pipx", "venvs"),
                   os.path.join(".local", "share", "pipx", "venvs")):
        prefixes.extend((p, "venv") for p in sorted(
            glob.glob(os.path.join(home, folder, "*"))))
    if machine.running_prefix:
        prefixes.append((machine.running_prefix, "venv"))
    user_site = [(p, "user-site") for p in sorted(
        glob.glob(os.path.join(home, ".local", "lib", "python3*")))]
    seen = set()
    records: List[InstallRecord] = []
    checkouts = []
    for prefix, layout in prefixes + user_site:
        if not prefix or not os.path.isdir(prefix):
            continue
        if any(_inside(prefix, r.root) for r in installers):
            continue
        if layout == "user-site":
            dists = []
            for name in _DISTRIBUTIONS:
                dists.extend(glob.glob(os.path.join(
                    prefix, "site-packages", f"{name}-*.dist-info")))
        else:
            dists = _distributions(prefix)
        key = _identity(prefix)
        if not dists or key in seen:
            continue
        seen.add(key)
        if os.path.exists(os.path.join(prefix, "conda-meta")):
            layout = "conda"
        running = bool(machine.running_prefix) and \
            _norm(prefix) == _norm(machine.running_prefix)
        python = None
        if layout == "user-site":
            python = machine.which(os.path.basename(prefix)) if machine.real else None
        else:
            python = _env_python(prefix)
        for dist in dists:
            source = _editable_source(dist)
            if source:
                checkouts.append(InstallRecord(
                    kind="checkout", layout="editable", platform=machine.platform,
                    root=source, version=_dist_version(dist), python=python,
                    running=running, notes=(f"installed in {prefix}",)))
                break
        else:
            records.append(InstallRecord(
                kind="environment", layout=layout, platform=machine.platform,
                root=prefix, version=_dist_version(dists[0]), python=python,
                running=running))
    return records + checkouts



def _user_data(machine: _Machine) -> List[str]:
    """Return preferences and user-data locations removal must never touch.

    :param machine: the computer.
    """
    home = machine.home
    cache = machine.xdg("XDG_CACHE_HOME", ".cache")
    backends = machine.env("SPACR_BACKENDS_DIR").strip()
    if backends == "~" or backends.startswith("~" + os.sep):
        backends = home + backends[1:]
    models = machine.env("SPACR_MODEL_CACHE").strip()
    if models == "~" or models.startswith("~" + os.sep):
        models = home + models[1:]
    paths = [
        machine.env("SPACR_HOME"),
        os.path.expanduser(backends) if backends else "",
        os.path.expanduser(models) if models else "",
        machine.env("SPACR_NEWS_CACHE"),
        machine.env("SPACR_SPACE_CACHE"),
        machine.env("SPACR_SOUND_CACHE").strip(),
        machine.env("SPACR_ICON_CACHE"),
        machine.env("MPLCONFIGDIR"),
        os.path.join(home, ".spacr"),
        os.path.join(cache, "spacr"),
        os.path.join(machine.xdg("XDG_STATE_HOME", os.path.join(".local", "state")), "spacr"),
        os.path.join(home, "spacr-demos"),
        os.path.join(home, "spacr-tutorials"),
        machine.env("HF_HOME") or os.path.join(cache, "huggingface"),
        os.path.join(machine.xdg("XDG_CONFIG_HOME", ".config"), "spacr"),
        os.path.join(home, "Library", "Preferences", "com.spacr.qt.plist"),
        machine.env("TORCH_HOME"),
        machine.env("CELLPOSE_LOCAL_MODELS_PATH"),
    ] + _relocated_caches(machine)
    paths = [path for path in paths if path]
    return list(dict.fromkeys(paths + [os.path.realpath(path) for path in paths]))


def _never_delete(machine: _Machine) -> List[str]:
    """Return shared folders that removal may empty entries from, never delete.

    :param machine: the computer.
    """
    home = machine.home
    data = machine.xdg("XDG_DATA_HOME", os.path.join(".local", "share"))
    folders = [
        machine.fs_root, home, data, os.path.join(data, "applications"),
        os.path.join(home, ".local"), os.path.join(home, ".local", "bin"),
        os.path.join(home, ".local", "share"),
        os.path.join(home, "Library"),
        os.path.join(home, "Library", "Application Support"),
        machine.env("LOCALAPPDATA"), machine.env("APPDATA"),
        machine.env("PROGRAMFILES"), machine.env("ProgramW6432"),
        _windows_start_menu(machine) if machine.env("APPDATA") else "",
    ] + _windows_desktops(machine) + [machine.path(p) for p in (
        "/", "/Applications", "/Library", "/Library/Application Support",
        "/usr", "/usr/bin", "/usr/local", "/usr/local/bin", "/opt")]
    return [f for f in folders if f]


def _refusal(path: str, machine: _Machine, record: InstallRecord) -> Optional[str]:
    """Return why ``path`` must not be deleted, or ``None`` when it may be.

    :param path: a path about to be deleted.
    :param machine: the computer.
    :param record: the installation it belongs to.
    """
    target = _norm(path)
    if any(target == _norm(f) for f in _never_delete(machine)):
        return ("refused: a shared folder, not a spaCR installation; "
                "remove spaCR from it by hand")
    for data in _user_data(machine):
        if _inside(path, data):
            return "kept: preferences and user data"
    if not machine.real:
        allowed = [machine.fs_root, machine.home, machine.env("LOCALAPPDATA"),
                   machine.env("APPDATA"), machine.env("PROGRAMFILES"),
                   machine.env("ProgramW6432")]
        if not any(a and _inside(path, a) for a in allowed):
            return "refused: outside the computer being cleaned"
    name = os.path.basename(target.rstrip(os.sep))
    if "spacr" in name:
        return None
    if _norm(path) == _norm(record.root) and any(
            _exists(os.path.join(path, m)) for m in (
                "venv", "Uninstall.exe", "uninstall-spacr.sh", "spacr.exe")):
        return None
    return ("refused: not recognisably a spaCR installation; "
            "delete it by hand if it is one")


def _delete(path: str, keep: Sequence[str], report: RemovalReport,
            reason_on_denied: str) -> None:
    """Delete one file, link or directory, sparing anything in ``keep``.

    :param path: what to delete.
    :param keep: paths that must survive, even inside ``path``.
    :param report: where the outcome is recorded.
    :param reason_on_denied: the reason given when permission is refused.
    """
    if not _exists(path):
        return
    if any(_inside(path, k) for k in keep):
        report.skipped.append((path, "kept: part of the new installation"))
        return
    inner_keep = [k for k in keep if _inside(k, path)]
    try:
        if os.path.islink(path) or not os.path.isdir(path):
            os.unlink(path)
        elif inner_keep:
            for child in sorted(os.listdir(path)):
                _delete(os.path.join(path, child), keep, report, reason_on_denied)
            return
        else:
            errors: List[Tuple[str, BaseException]] = []

            def _writable_then_retry(func, failed_path, exc_info):
                """Clear a read-only flag and retry once, recording failures.

                :param func: the :mod:`os` function that failed.
                :param failed_path: the path it failed on.
                :param exc_info: the error, as :func:`shutil.rmtree` passes it.
                """
                try:
                    os.chmod(failed_path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
                    func(failed_path)
                except OSError as exc:
                    errors.append((failed_path, exc))
                except TypeError:
                    errors.append((failed_path, exc_info if isinstance(
                        exc_info, BaseException) else exc_info[1]))

            if _RMTREE_TAKES_ONEXC:
                shutil.rmtree(path, onexc=_writable_then_retry)
            else:
                shutil.rmtree(path, onerror=_writable_then_retry)
            if errors or _exists(path):
                failed_path, exc = errors[0] if errors else (path, OSError("still present"))
                raise _Denied(failed_path, exc)
    except _Denied as denied:
        report.failed.append((path, _reason(denied.exc, reason_on_denied)))
        return
    except OSError as exc:
        report.failed.append((path, _reason(exc, reason_on_denied)))
        return
    report.removed.append(path)


_CACHE_LOCATIONS_FILE = "cache_locations.json"
_SETTINGS_REGISTRY_KEY = "Software\\spacr\\qt"


def _relocated_caches(machine: _Machine) -> List[str]:
    """Return cache folders spaCR moved out of its home folder on request.

    They are read from ``cache_locations.json`` in ``SPACR_HOME`` or
    ``~/.spacr``. Only a folder whose
    own name starts with ``spacr-`` counts, which is the name spaCR gives
    every folder it relocates, so a hand-edited entry naming a shared folder
    is never returned.

    :param machine: the computer.
    """
    home = machine.env("SPACR_HOME") or os.path.join(machine.home, ".spacr")
    path = os.path.join(home, _CACHE_LOCATIONS_FILE)
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return []
    found = []
    for value in (data.values() if isinstance(data, dict) else ()):
        folder = str(value or "")
        name = os.path.basename(os.path.normpath(folder)).lower() if folder else ""
        if name.startswith("spacr-"):
            found.append(folder)
    return found


def _purge_targets(machine: _Machine) -> List[str]:
    """Return the caches, backends and user data an opt-in purge deletes.

    The list is spaCR's own home folder (logs, runs, models, backend
    environments, news) or the folder ``SPACR_HOME`` names, its cache, state
    and configuration folders, the demo and tutorial folders, the macOS
    preferences file, a backends folder named by ``SPACR_BACKENDS_DIR`` and
    every cache spaCR relocated. Shared
    caches such as Hugging Face's and Torch's own folders, shared system
    folders, anything that is not named after spaCR and the folder of the
    running interpreter are never included. Only existing paths come back.

    :param machine: the computer.
    """
    home = machine.home
    cache = machine.xdg("XDG_CACHE_HOME", ".cache")
    candidates = [
        os.path.join(home, ".spacr"),
        os.path.join(cache, "spacr"),
        os.path.join(machine.xdg("XDG_STATE_HOME", os.path.join(".local", "state")), "spacr"),
        os.path.join(machine.xdg("XDG_CONFIG_HOME", ".config"), "spacr"),
        os.path.join(home, "spacr-demos"),
        os.path.join(home, "spacr-tutorials"),
        os.path.join(home, "Library", "Preferences", "com.spacr.qt.plist"),
        os.path.join(home, "Library", "Caches", "spacr"),
        machine.env("SPACR_HOME"),
        machine.env("SPACR_BACKENDS_DIR"),
    ] + _relocated_caches(machine)
    shared = {_norm(f) for f in _never_delete(machine)}
    running = machine.running_prefix
    chosen: List[str] = []
    for path in _dedupe(candidates):
        name = os.path.basename(os.path.normpath(path)).lower()
        if _norm(path) in shared or "spacr" not in name:
            continue
        if running and _inside(running, path):
            continue
        if any(_inside(path, kept) for kept in chosen):
            continue
        chosen = [kept for kept in chosen if not _inside(kept, path)] + [path]
    return chosen


def _delete_registry_tree(registry, key: str, report: RemovalReport) -> None:
    """Delete a registry key and every key below it.

    :param registry: the HKCU adapter.
    :param key: path below HKCU.
    :param report: where the outcome is recorded.
    """
    try:
        for child in registry.subkeys(key):
            _delete_registry_tree(registry, key + "\\" + child, report)
        registry.delete_key(key)
    except OSError as exc:
        report.failed.append(("registry:" + key, _reason(exc, "access denied")))
        return
    report.removed.append("registry:" + key)


def _purge(machine: _Machine, targets: Optional[Sequence[str]] = None) -> RemovalReport:
    """Delete what :func:`_purge_targets` lists and spaCR's registry settings.

    :param machine: the computer.
    :param targets: the paths to delete; :func:`_purge_targets` when ``None``.
    :returns: a report whose ``record`` is ``None``.
    """
    report = RemovalReport(record=None)
    keep = [machine.running_prefix] if machine.running_prefix else []
    for path in (_purge_targets(machine) if targets is None else targets):
        _delete(path, keep, report, "permission denied; delete it by hand")
    registry = machine.registry
    if registry is not None and (registry.values(_SETTINGS_REGISTRY_KEY)
                                 or registry.subkeys(_SETTINGS_REGISTRY_KEY)):
        _delete_registry_tree(registry, _SETTINGS_REGISTRY_KEY, report)
    return report


def _ask_on_terminal(prompt: str) -> Optional[str]:
    """Ask ``prompt`` on an interactive terminal, or return ``None``.

    :param prompt: the question.
    """
    try:
        if not (sys.stdin and sys.stdin.isatty()):
            return None
        return input(prompt)
    except (EOFError, OSError):
        return None


def _purge_command(machine: _Machine, *, yes: bool = False, dry_run: bool = False,
                   ask: Callable[[str], Optional[str]] = _ask_on_terminal) -> int:
    """List what a purge deletes, ask, and delete it.

    Nothing is deleted unless ``yes`` is given or the person types
    ``purge`` at the prompt. Without a terminal to ask on, and without
    ``yes``, nothing is deleted.

    :param machine: the computer.
    :param yes: delete without asking.
    :param dry_run: list only.
    :param ask: asks a question and returns the answer, or ``None``.
    :returns: ``0`` when done or nothing was there, ``3`` when the purge
        was not confirmed, ``1`` when something could not be deleted.
    """
    targets = _purge_targets(machine)
    registry = machine.registry
    has_settings = registry is not None and bool(
        registry.values(_SETTINGS_REGISTRY_KEY)
        or registry.subkeys(_SETTINGS_REGISTRY_KEY))
    if not targets and not has_settings:
        print("No spaCR caches, backends or user data were found.")
        return 0
    print("A purge deletes spaCR's caches, backends and user data:")
    for path in targets:
        print(f"  {path}")
    if has_settings:
        print(f"  registry: HKCU\\{_SETTINGS_REGISTRY_KEY}")
    if dry_run:
        return 0
    if not yes:
        answer = ask("Type purge to delete these, anything else to keep them: ")
        if answer is None:
            print("Nothing was deleted: pass --yes to purge without a terminal.")
            return 3
        if answer.strip().lower() != "purge":
            print("Nothing was deleted.")
            return 3
    report = _purge(machine, targets)
    for item in report.removed:
        print(f"  removed: {item}")
    for item, why in report.failed:
        print(f"  could not remove: {item} ({why})")
    return 0 if report.ok else 1


class _Denied(Exception):
    """A directory tree that could not be removed completely."""

    def __init__(self, path: str, exc: BaseException):
        """Keep the first path that failed and why.

        :param path: the path that could not be removed.
        :param exc: the error.
        """
        super().__init__(path)
        self.path = path
        self.exc = exc


def _reason(exc: BaseException, reason_on_denied: str) -> str:
    """Turn an operating-system error into a reason a person can act on.

    :param exc: the error.
    :param reason_on_denied: the reason for a refused permission.
    """
    if isinstance(exc, PermissionError):
        return reason_on_denied
    return str(getattr(exc, "strerror", "") or exc)


_LEASE_OWNER = "owner"
_LEASE_ROLLBACK = "rollback"


def _process_started(pid):
    """A start stamp for ``pid`` that no clock step, locale or time zone changes; None when it cannot be read.

    Linux reads the start tick from ``/proc/<pid>/stat``; other POSIX systems read ``ps -o lstart=`` in the C
    locale and UTC; Windows reads the creation FILETIME and answers "" when no such process exists. A missing
    POSIX process is decided by :func:`_process_gone`, never by an unreadable stamp.
    """
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return "" if ctypes.get_last_error() == 87 else None
        try:
            status = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(status)):
                return None
            if status.value != 259:
                return ""
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, *[ctypes.byref(item) for item in times]):
                return None
            return "filetime:" + str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
        finally:
            kernel.CloseHandle(handle)
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if sys.platform.startswith("linux"):
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as handle:
                ticks = handle.read().rsplit(")", 1)[1].split()[19]
        except (OSError, IndexError):
            return None
        return "ticks:" + ticks if ticks.isdigit() else None
    try:
        done = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True,
                              timeout=10, env=dict(os.environ, LC_ALL="C", TZ="UTC0"))
    except (OSError, subprocess.SubprocessError):
        return None
    text = " ".join(done.stdout.split())
    return "lstart:" + text if done.returncode == 0 and text else None


def _process_gone(pid):
    """True only when the system says no process has ``pid``; a process it will not signal counts as present."""
    if os.name == "nt":
        return _process_started(pid) == ""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except (OSError, ValueError):
        return False
    return False


def _pid_namespace():
    """The pid namespace this process sees on Linux, "unknown" when it cannot be read, and "" elsewhere."""
    if not sys.platform.startswith("linux"):
        return ""
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return "unknown"


def _process_group_live_member(group):
    """A process still in ``group`` owned by this user, other than this process's ancestors and descendants.

    Children of a killed installer keep its process group after the installer itself is gone. Returns that
    member's pid, "unknown" when the process table cannot be read, or "" when the group is empty.
    """
    try:
        done = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,pgid=,uid="], capture_output=True, text=True,
                              timeout=10, env=dict(os.environ, LC_ALL="C"))
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    rows = [line.split() for line in done.stdout.splitlines()]
    rows = [row for row in rows if len(row) == 4 and all(item.isdigit() for item in row)]
    if done.returncode != 0 or not rows:
        return "unknown"
    parent = {row[0]: row[1] for row in rows}
    me = str(os.getpid())
    lineage, item = set(), me
    while item and item not in lineage and item != "0":
        lineage.add(item)
        item = parent.get(item, "")
    uid = str(os.geteuid())
    for pid, _, pgid, owner in rows:
        if pgid != str(group) or owner != uid or pid in lineage:
            continue
        ancestor, seen = parent.get(pid, ""), set()
        while ancestor and ancestor not in ("0", "1", me) and ancestor not in seen:
            seen.add(ancestor)
            ancestor = parent.get(ancestor, "")
        if ancestor != me:
            return pid
    return ""


def _windows_children(pid):
    """The live children of ``pid`` as (pid, start stamp) pairs, or None when the process table cannot be read.

    Windows keeps a dead parent's id in its children's records, so a killed installer's pip or uv still names
    it. A child whose start cannot be read makes the answer unknown.
    """
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        """PROCESSENTRY32W."""

        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_wchar * 260)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snapshot = kernel.CreateToolhelp32Snapshot(0x2, 0)
    if snapshot in (None, 0, ctypes.c_void_p(-1).value):
        return None
    try:
        entry = Entry()
        entry.dwSize = ctypes.sizeof(Entry)
        children = []
        more = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        if not more:
            return None
        while more:
            if entry.th32ParentProcessID == int(pid) and entry.th32ProcessID != int(pid):
                children.append(int(entry.th32ProcessID))
            more = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        if ctypes.get_last_error() != 18:
            return None
    finally:
        kernel.CloseHandle(snapshot)
    found = []
    for child in children:
        stamp = _process_started(child)
        if stamp is None:
            return None
        if stamp:
            found.append((child, stamp))
    return found


def _owner_children_live(pid, started):
    """Whether a process the lease's owner started still runs: True, False, or None when that is unknown.

    Only a child created at or after the owner's own start counts; a child of an earlier process that held the
    same id is older than the owner. Stamps of another kind make the answer unknown.
    """
    children = _windows_children(pid)
    if children is None:
        return None
    kind, _, value = started.partition(":")
    if kind != "filetime" or not value.isdigit():
        return None
    for _child, stamp in children:
        child_kind, _, child_value = stamp.partition(":")
        if child_kind != kind or not child_value.isdigit():
            return None
        if int(child_value) >= int(value):
            return True
    return False


def _lease_host():
    """This computer's name as the installers record it: COMPUTERNAME on Windows, ``uname -n`` elsewhere."""
    import platform

    return os.environ.get("COMPUTERNAME", "") if os.name == "nt" else platform.node()


_OWN_LEASE_RECORD = {}


def _lease_owner_text():
    """The owner record a new lease carries, measured once per process.

    Process id, start stamp and host everywhere, plus the pid namespace on POSIX, so a lease written inside a
    container that shares this host name is never judged by pids it cannot see. Windows records tree=1: the
    lease stays live while a process the owner started still runs, since Windows does not end a killed
    installer's pip or uv with it.
    """
    pid = os.getpid()
    if pid not in _OWN_LEASE_RECORD:
        text = f"pid={pid}\nstarted={_process_started(pid) or ''}\nhost={_lease_host()}\n"
        if os.name == "nt":
            text += "tree=1\n"
        else:
            text += f"pidns={_pid_namespace()}\n"
            try:
                text += f"pgid={os.getpgrp()}\n"
            except OSError:
                pass
        _OWN_LEASE_RECORD[pid] = text
    return _OWN_LEASE_RECORD[pid]


def _lease_fields(text):
    """The key=value lines of a lease record."""
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def _lease_file_in_use(path):
    """On Windows, whether another handle holds the lease file open; POSIX has no such lock and answers False."""
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                   wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    handle = kernel.CreateFileW(path, 0x80000000, 0, None, 3, 0x80, None)
    if handle in (None, 0, ctypes.c_void_p(-1).value):
        return True
    kernel.CloseHandle(handle)
    return False


def _lease_owner_dead(path):
    """True only when a lease's recorded owner is provably gone and the lease holds nothing else.

    The record must be this user's, written in this host's format with this host name and pid namespace, and
    its process must be absent or carry another start stamp of the same kind. A rollback marker, a journal,
    any unknown file, an open lease file, another user's or another host's or container's lease, an unreadable
    process, or a lease from a version that recorded no owner is never treated as abandoned.
    """
    try:
        if os.path.islink(path):
            return False
        if os.path.isdir(path):
            if os.listdir(path) != [_LEASE_OWNER]:
                return False
            record = os.path.join(path, _LEASE_OWNER)
            if os.path.islink(record) or not os.path.isfile(record):
                return False
        elif os.path.isfile(path):
            record = path
            if _lease_file_in_use(path):
                return False
        else:
            return False
        if os.name != "nt" and any(os.lstat(item).st_uid != os.geteuid() for item in (path, record)):
            return False
        with open(record, encoding="utf-8") as handle:
            text = handle.read(4096)
    except (OSError, UnicodeDecodeError):
        return False
    fields = _lease_fields(text)
    expected = _lease_fields(_lease_owner_text())
    expected.pop("pgid", None)
    expected.pop("tree", None)
    group = fields.pop("pgid", None)
    tree = fields.pop("tree", None)
    if (set(fields) != set(expected) or not fields["started"] or fields["host"] != expected["host"]
            or fields.get("pidns") != expected.get("pidns") or expected.get("pidns") == "unknown"):
        return False
    try:
        pid = int(fields["pid"])
    except ValueError:
        return False
    if pid <= 0 or pid == os.getpid():
        return False
    if not _process_gone(pid):
        now = _process_started(pid)
        if not now or now.partition(":")[0] != fields["started"].partition(":")[0] or now == fields["started"]:
            return False
    if tree is not None:
        if tree != "1" or group is not None:
            return False
        return _owner_children_live(pid, fields["started"]) is False
    if group is None:
        return True
    if not group.isdigit() or int(group) <= 0:
        return False
    return _process_group_live_member(group) == ""


_LEASE_KNOWN_FIELDS = frozenset({"pid", "started", "host", "pidns", "pgid", "tree", "rollback"})


def _lease_held_message(path):
    """A refusal naming the lease and, when recorded, its owning process.

    A lease with fields a later installer added, whose pid is gone, names no holder: this version cannot check
    it, and the dead pid holds nothing.
    """
    owner = os.path.join(path, _LEASE_OWNER) if os.path.isdir(path) else path
    try:
        with open(owner, encoding="utf-8") as handle:
            fields = _lease_fields(handle.read(4096))
    except (OSError, UnicodeDecodeError):
        fields = {}
    pid = fields.get("pid")
    if pid and set(fields) - _LEASE_KNOWN_FIELDS and pid.isdigit() and _process_gone(int(pid)):
        pid = None
    if pid:
        return tr("Another spaCR installation holds {path} (process {pid}). Wait for it to finish; "
                  "if it has stopped and left a rollback, follow the installer log first.", path=path, pid=pid)
    return tr("Another spaCR installation holds {path}. Wait for it to finish, or remove that folder once "
              "no spaCR installer is running.", path=path)


@contextmanager
def _lease_takeover(path):
    """Let one installer at a time replace an abandoned lease; a second one judging the same lease refuses.

    The guard is a sibling directory stamped with its owner. Under it the caller judges the lease again, so a
    lease another installer has just taken over is never replaced.
    """
    guard = path + ".takeover"
    try:
        os.mkdir(guard, 0o700)
    except FileExistsError:
        raise FileExistsError(errno.EEXIST, _lease_held_message(guard), guard) from None
    try:
        with open(os.path.join(guard, _LEASE_OWNER), "x", encoding="utf-8") as handle:
            handle.write(_lease_owner_text())
        yield
    finally:
        try:
            _release_lease_dir(guard)
        except OSError:
            pass


def _take_lease_dir(path):
    """Create a directory lease stamped with its owner, replacing only a provably abandoned one."""
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        if not _lease_owner_dead(path):
            raise FileExistsError(errno.EEXIST, _lease_held_message(path), path) from None
        with _lease_takeover(path):
            if not _lease_owner_dead(path):
                raise FileExistsError(errno.EEXIST, _lease_held_message(path), path) from None
            os.unlink(os.path.join(path, _LEASE_OWNER))
            os.rmdir(path)
            os.mkdir(path, 0o700)
    descriptor = os.open(os.path.join(path, _LEASE_OWNER), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(_lease_owner_text())
    return os.stat(path, follow_symlinks=False)


def _take_lease_file(path):
    """Create a Windows file lease stamped with its owner and kept open, replacing only an abandoned one."""
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if not _lease_owner_dead(path):
            raise
        with _lease_takeover(path):
            if not _lease_owner_dead(path):
                raise
            os.unlink(path)
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, _lease_owner_text().encode("utf-8"))
    except OSError:
        os.close(descriptor)
        raise
    return descriptor


def _release_lease_dir(path):
    """Remove a directory lease this process owns: its owner record, then the directory."""
    try:
        os.unlink(os.path.join(path, _LEASE_OWNER))
    except FileNotFoundError:
        pass
    os.rmdir(path)


def _retain_lease_dir(path, reason):
    """Mark a lease that must outlive its owner because a rollback still needs resolving."""
    try:
        with open(os.path.join(path, _LEASE_ROLLBACK), "w", encoding="utf-8") as handle:
            handle.write(str(reason) + "\n")
    except OSError:
        pass


class _RemovalLeaseUnavailable(OSError):
    """Removal was refused before mutation because its transaction leases could not be acquired."""


@contextmanager
def _removal_transaction_lease(record, machine):
    """Hold the installer and private-runtime leases while removing one online installation or selected environment."""
    root = os.path.realpath(record.root)
    if record.kind == "environment":
        paths = [os.path.join(os.path.dirname(root), "." + os.path.basename(root) + ".spacr-update-lock")]
    else:
        paths = [os.path.join(os.path.dirname(root), "." + os.path.basename(root) + ".spacr-installer-lock")]
        if os.path.isdir(root):
            paths.append(os.path.join(root, ".venv.spacr-update-lock"))
    owned = []
    completed = False
    result = {"remove_empty_root": False}
    linked_root = os.path.islink(record.root)
    root_identity = _identity(root)
    try:
        for path in paths:
            descriptor = None
            try:
                if machine.platform == "windows":
                    descriptor = _take_lease_file(path)
                    identity = os.fstat(descriptor)
                else:
                    identity = _take_lease_dir(path)
            except OSError as error:
                if descriptor is not None:
                    os.close(descriptor)
                raise _RemovalLeaseUnavailable(path) from error
            owned.append((path, descriptor, identity.st_dev, identity.st_ino))
        yield result
        completed = True
    finally:
        for path, descriptor, device, inode in reversed(owned):
            try:
                current = os.lstat(path)
                matches = (current.st_dev, current.st_ino) == (device, inode)
                if descriptor is not None:
                    os.close(descriptor)
                    descriptor = None
                    if matches:
                        os.unlink(path)
                elif matches:
                    _release_lease_dir(path)
            except OSError:
                pass
            finally:
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
        protected = (any(_norm(root) == _norm(path) for path in _never_delete(machine))
                     or any(_inside(root, path) for path in _user_data(machine)))
        if (completed and result["remove_empty_root"] and record.kind == "installer"
                and not linked_root and not protected and _identity(root) == root_identity):
            try:
                os.rmdir(root)
            except OSError:
                pass


def remove_install(record: InstallRecord, *, ticked: bool = False,
                   keep: Sequence[str] = (), system=None) -> RemovalReport:
    """Remove one installation, without running its own uninstaller.

    An installer-made copy's application code, launchers, shortcuts, menu entries
    and uninstall registrations are removed. Known caches, models, test data,
    configured user locations and non-application entries in online runtime
    roots are kept. Its root can remain to hold those files. A Debian package is
    removed through the package manager, and reported as needing
    administrator rights when that is what stops it. From an environment the
    user made, only the spaCR package is uninstalled, and only when
    ``ticked``; the environment itself stays. A source checkout is skipped.

    The running copy is not removed here, because on Windows an installation
    cannot delete itself while it runs. :func:`start_update_helper` runs the
    appropriate update procedure after spaCR has closed.

    :param record: an installation from :func:`find_old_installs`.
    :param ticked: whether the user ticked this environment for removal.
    :param keep: paths to leave in place, such as the log of the install
        that is about to start.
    :param system: the computer; ``None`` means this one.
    :returns: what was removed, what could not be, and why.
    """
    machine = system or _Machine()
    online = record.kind == "installer" and record.layout in {
        "linux-online", "macos-online", "macos-runtime", "windows-online"}
    selected_environment = record.kind == "environment" and ticked
    kept = any(_norm(path) == _norm(record.root) for path in keep)
    if (online or selected_environment) and not record.running and not kept:
        try:
            with _removal_transaction_lease(record, machine) as lease:
                lease["remove_empty_root"] = _refusal(record.root, machine, record) is None
                return _remove_install_unleased(record, ticked=ticked, keep=keep, system=machine)
        except _RemovalLeaseUnavailable:
            report = RemovalReport(record)
            report.failed.append((record.root, tr(
                "This installation could not be locked for removal. Nothing was removed.")))
            return report
    return _remove_install_unleased(record, ticked=ticked, keep=keep, system=machine)


def _remove_install_unleased(record: InstallRecord, *, ticked: bool = False,
                   keep: Sequence[str] = (), system=None) -> RemovalReport:
    """Apply the existing removal procedure after the caller has protected any applicable transaction roots."""
    machine = system or _Machine()
    report = RemovalReport(record)
    if record.kind == "checkout":
        report.skipped.append((record.root, "source checkout; update it with git pull"))
        return report
    if record.running:
        report.skipped.append((record.root, "in use by the running spaCR"))
        if record.kind == "installer":
            report.failed.append((record.root, "in use; it is removed after spaCR closes"))
        return report
    if record.kind == "environment":
        if not ticked:
            report.skipped.append((record.root, "your environment; not ticked"))
            return report
        _uninstall_from_environment(record, machine, report)
        return report
    if any(_norm(path) == _norm(record.root) for path in keep):
        report.skipped.append((record.root, "kept: part of the new installation"))
        return report
    preserved = list(keep) + _user_data(machine)
    physical_root = os.path.realpath(record.root)
    preserved.extend(os.path.join(record.root, os.path.relpath(path, physical_root))
                     for path in tuple(preserved) if _inside(path, physical_root))
    preserved.extend(path for name in (
        "cache", "test_data", "models", "outputs", "projects", "runs")
        if _exists(path := os.path.join(record.root, name)))
    if os.path.isdir(record.root) and not record.root.endswith(".app"):
        application_entries = {
            "venv", "python", "bootstrap", "_internal", "spacr", "spacr_drivers",
            "bin", "lib", "lib64", "scripts", "uninstall.exe", "spacr.exe",
            "launch_spacr.pyw", "spacr.cmd", "spacr.ico", "install.log",
            "install-profile.json", "uninstall-spacr.sh", "runtime-identity.json",
            "install-online.sh", "install-for-user.sh", "installer_messages.sh", "generated",
            "offline_mask_check.py", "offline-bundle.json", "nsis-bootstrap-status.txt"}
        try:
            entries = os.listdir(record.root)
        except OSError as exc:
            report.failed.append((record.root, _reason(exc, _needs_admin() if record.needs_admin
                                                     else "in use or not permitted; close anything using it and try again")))
            return report
        preserved.extend(os.path.join(record.root, name) for name in entries
                         if name.lower() not in application_entries
                         and not name.startswith((".venv-staging-", ".venv-previous")))
    denied = _needs_admin() if record.needs_admin else (
        "in use or not permitted; close anything using it and try again")
    packaged = any(r.startswith("deb:") for r in record.registrations)
    paths = [] if packaged else [record.root, *record.launchers, *record.shortcuts,
             *record.menu_entries,
             *(r for r in record.registrations
               if not r.startswith(("registry:", "registry-value:", "deb:")))]
    for path in paths:
        if not _exists(path):
            continue
        refusal = _refusal(path, machine, record)
        if refusal:
            (report.skipped if refusal.startswith("kept") else report.failed).append(
                (path, refusal))
            continue
        _delete(path, preserved, report, denied)
    for registration in record.registrations:
        if registration.startswith("registry"):
            _remove_registration(registration, machine, report)
        elif registration.startswith("deb:"):
            name = registration[len("deb:"):]
            if machine.packages is None:
                report.failed.append((registration, "no package manager to ask"))
                continue
            removed, why = machine.packages.remove(name, machine.sudo)
            if removed:
                report.removed.append(registration)
            else:
                report.failed.append((registration, why or _needs_admin()))
    return report


def _remove_registration(token: str, machine: _Machine,
                         report: RemovalReport) -> None:
    """Delete a Windows registry entry an installer wrote, sparing preferences.

    The installer's ``Software`` key is the same key as the one the settings
    live in, because registry names ignore case. So only the ``InstallRoot``
    value is deleted there, and the key only when nothing else is left in it.

    :param token: a ``registry:`` key or a ``registry-value:`` value.
    :param machine: the computer.
    :param report: where the outcome is recorded.
    """
    registry = machine.registry
    if registry is None:
        report.failed.append((token, "no registry to change"))
        return
    kind, _, where = token.partition(":")
    where = where.split("\\", 1)[1] if where.upper().startswith("HKCU\\") else where
    try:
        if kind == "registry-value":
            key, _, name = where.rpartition("\\")
            if name in registry.values(key):
                registry.delete_value(key, name)
        else:
            key = where
            for name in list(registry.values(key)):
                registry.delete_value(key, name)
        if not registry.values(key) and not registry.subkeys(key):
            registry.delete_key(key)
    except OSError as exc:
        report.failed.append((token, _reason(
            exc, "not permitted; delete this registry entry by hand")))
        return
    report.removed.append(token)


def _uninstall_from_environment(record: InstallRecord, machine: _Machine,
                                report: RemovalReport) -> None:
    """Uninstall the spaCR package from a ticked environment, and nothing else.

    :param record: an environment the user made.
    :param machine: the computer.
    :param report: where the outcome is recorded.
    """
    dists = _distributions(record.root) or [dist for name in _DISTRIBUTIONS
        for dist in glob.glob(os.path.join(record.root, "site-packages", f"{name}-*.dist-info"))]
    names = sorted({_dist_name(d) for d in dists}) or ["spacr"]
    python = record.python
    if not python:
        report.failed.append((record.root, "no Python found in it; run pip "
                              "uninstall spacr in that environment"))
        return
    has_pip = any(os.path.isdir(os.path.join(site, "pip"))
                  for site in _site_packages(record.root))
    uv = machine.which("uv") if machine.real else None
    if has_pip or not uv:
        argv = [python, "-m", "pip", "uninstall", "-y", *names]
    else:
        argv = [uv, "pip", "uninstall", "--python", python, *names]
    code, output = machine.runner(argv)
    item = f"{' '.join(names)} in {record.root}"
    if code == 0:
        report.removed.append(item)
    else:
        last = (output or "").strip().splitlines()
        report.failed.append((item, last[-1] if last else f"exit code {code}"))


_UNINSTALL_CATEGORIES = ("caches", "models", "logs", "settings", "history",
                         "temporary", "downloads", "other")


def _uninstall_path_state(path):
    """Bind an existing path to its unlinked directory ancestry without reading file contents."""
    path = os.path.abspath(path)
    cursor = path
    identity = []
    while True:
        details = os.lstat(cursor)
        if (stat.S_ISLNK(details.st_mode)
                or getattr(details, "st_file_attributes", 0) & 0x400):
            raise ValueError("linked_path")
        identity.append([cursor, details.st_dev,
                         None if cursor == path and stat.S_ISREG(details.st_mode) else details.st_ino,
                         stat.S_IFMT(details.st_mode)])
        parent = os.path.dirname(cursor)
        if parent == cursor:
            return identity
        cursor = parent


def _uninstall_data_candidates(machine):
    """List known spaCR data locations by independent uninstall category, without broad home deletion."""
    home = machine.home

    def expand(value):
        """Expand the configured user's leading home marker."""
        value = value.strip()
        if value == "~" or value.startswith(("~/", "~\\")):
            value = home + value[1:]
        return os.path.abspath(value)

    app_home = expand(machine.env("SPACR_HOME").strip() or os.path.join(home, ".spacr"))
    children = {
        "caches": ("cache", "news", "backgrounds", "sounds", "icons"),
        "models": ("models", "backends", "community_models.json"),
        "logs": ("logs", "anonymous_reports"),
        "settings": ("settings", "network.json", "notification_secrets.json", "cache_locations.json"),
        "history": ("runs", "history", "restart_state.json", "last_session.json", "queue.json"),
        "temporary": ("tmp", "temp"),
        "downloads": ("downloads",),
        "other": ("plugins", "recipes", "macros", "blinding", "analysis_locks", "features"),
    }
    candidates = [(category, os.path.join(app_home, name), True)
                  for category, names in children.items() for name in names]
    candidates.extend((category, path, True) for category, path in (
        ("caches", os.path.join(machine.xdg("XDG_CACHE_HOME", ".cache"), "spacr")),
        ("settings", os.path.join(machine.xdg("XDG_CONFIG_HOME", ".config"), "spacr")),
        ("settings", os.path.join(home, "Library", "Preferences", "com.spacr.qt.plist")),
        ("history", os.path.join(machine.xdg("XDG_STATE_HOME", os.path.join(".local", "state")), "spacr")),
        ("downloads", os.path.join(home, "spacr-demos")),
        ("downloads", os.path.join(home, "spacr-tutorials")),
    ))
    variables = {
        "SPACR_MODEL_CACHE": "models", "SPACR_BACKENDS_DIR": "models",
        "SPACR_LOG_DIR": "logs", "SPACR_PLUGIN_HOME": "other",
        "SPACR_NEWS_CACHE": "caches", "SPACR_SPACE_CACHE": "caches",
        "SPACR_SOUND_CACHE": "caches", "SPACR_ICON_CACHE": "caches",
        "MPLCONFIGDIR": "caches", "HF_HOME": "models", "TORCH_HOME": "models",
        "CELLPOSE_LOCAL_MODELS_PATH": "models",
    }
    relocated = {}
    try:
        with open(os.path.join(app_home, _CACHE_LOCATIONS_FILE), encoding="utf-8") as stream:
            locations = json.load(stream)
        for variable, key in (("SPACR_MODEL_CACHE", "models"), ("SPACR_BACKENDS_DIR", "backends"),
                              ("SPACR_NEWS_CACHE", "news")):
            path = locations.get(variable, "") if isinstance(locations, dict) else ""
            if isinstance(path, str) and os.path.basename(path) == "spacr-" + key:
                relocated[_norm(path)] = (variables[variable], os.path.abspath(path))
    except (OSError, ValueError, RecursionError):
        pass
    for variable, category in variables.items():
        raw = machine.env(variable)
        if not raw.strip():
            continue
        path = expand(raw)
        owned = (_inside(path, app_home) and _norm(path) != _norm(app_home)
                 or _norm(path) in relocated and variable not in {"HF_HOME", "TORCH_HOME", "CELLPOSE_LOCAL_MODELS_PATH"})
        candidates.append((category, path, owned))
    for category, path in relocated.values():
        candidates.append((category, path, True))
    shared = {_norm(path) for path in _never_delete(machine)}
    shared_models = [expand(machine.env(variable))
                     for variable in ("HF_HOME", "TORCH_HOME", "CELLPOSE_LOCAL_MODELS_PATH")
                     if machine.env(variable).strip()
                     and not _inside(expand(machine.env(variable)), app_home)]
    unique = {}
    for category, path, owned in candidates:
        if not _exists(path):
            continue
        path = os.path.abspath(path)
        if _norm(path) in shared or (machine.running_prefix and _inside(machine.running_prefix, path)):
            owned = False
        if any(_inside(path, folder) or _inside(folder, path) for folder in shared_models):
            owned = False
        unique[path] = (category, path, owned)
    result = []
    pending = list(unique.values())
    while pending:
        category, path, owned = pending.pop(0)
        nested = [other for other in unique if other != path and _inside(other, path)]
        if nested and owned and os.path.isdir(path) and not os.path.islink(path):
            try:
                entries = sorted(os.listdir(path))
            except OSError:
                result.append((category, path, False))
                continue
            for name in entries:
                child = os.path.join(path, name)
                if child not in unique:
                    pending.append((category, child, owned))
        else:
            result.append((category, path, owned))
    return sorted(set(result), key=lambda row: (row[0], row[1]))


def _uninstall_preview(operation_id, *, system=None):
    """Discover exact current/all installation scopes and separately selectable spaCR data paths."""
    if not isinstance(operation_id, str) or not operation_id.strip():
        raise ValueError("operation_id")
    machine = system or _Machine()
    records = find_old_installs(system=machine)
    current = {record.root for record in records if record.running
               or (machine.real and record.kind == "installer" and _inside(__file__, record.root))}
    mac_apps = [record for record in records if record.layout == "macos-app"]
    if (len(mac_apps) == 1 and os.path.isfile(os.path.join(mac_apps[0].root, "install-for-user.sh"))
            and any(record.root in current and record.layout in {"macos-runtime", "macos-online"}
                                  and os.path.basename(record.root).lower() == "spacr"
                                  for record in records)):
        current.add(mac_apps[0].root)
    installations = []
    for record in records:
        data = json.loads(json.dumps(asdict(record)))
        key = hashlib.sha256((record.kind + "\0" + _norm(record.root)).encode()).hexdigest()
        try:
            identity = _uninstall_path_state(record.root)
            reason = "checkout" if record.kind == "checkout" else ""
        except (OSError, ValueError):
            identity, reason = [], "unverified"
        sites = [os.path.join(record.root, "site-packages")] if record.layout == "user-site" else _site_packages(record.root)
        paths = ([record.root] if record.kind != "environment" else
                 [path for site in sites for name in _DISTRIBUTIONS
                  for path in glob.glob(os.path.join(site, f"{name}-*.dist-info"))]
                 + [os.path.join(site, "spacr") for site in sites if os.path.isdir(os.path.join(site, "spacr"))])
        paths += list(record.launchers + record.shortcuts + record.menu_entries + record.registrations)
        installations.append({"key": key, "record": data, "paths": list(dict.fromkeys(paths)),
                              "identity": identity, "current": record.root in current,
                              "selectable": not reason, "reason": reason})
    cleanup = []
    for category, path, owned in _uninstall_data_candidates(machine):
        try:
            identity = _uninstall_path_state(path)
            reason = "" if owned else "unverified"
        except (OSError, ValueError):
            identity, reason = [], "unverified"
        cleanup.append({"category": category, "path": path, "identity": identity,
                        "selectable": not reason, "reason": reason})
    listed = {row["path"] for row in cleanup}
    for record in records:
        if record.kind != "installer":
            continue
        for path in _runtime_transaction_folders(record.root):
            if path in listed:
                continue
            try:
                identity = _uninstall_path_state(path)
                reason = ""
            except (OSError, ValueError):
                identity, reason = [], "unverified"
            cleanup.append({"category": "temporary", "path": path, "identity": identity,
                            "selectable": not reason, "reason": reason})
            listed.add(path)
    registry = machine.registry
    if registry is not None and (registry.values(_SETTINGS_REGISTRY_KEY) or registry.subkeys(_SETTINGS_REGISTRY_KEY)):
        cleanup.append({"category": "settings", "path": "registry:" + _SETTINGS_REGISTRY_KEY,
                        "identity": [], "selectable": True, "reason": ""})
    payload = {"schema": 1, "operation_id": operation_id, "installations": installations,
               "cleanup": cleanup, "categories": list(_UNINSTALL_CATEGORIES)}
    payload["snapshot_id"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


_UNINSTALL_SCOPES = ("current", "older", "all", "custom")


def _uninstall_scope_keys(preview, scope):
    """The selectable installation keys a named current/older/all scope covers, in preview order."""
    return [row["key"] for row in preview["installations"]
            if row["selectable"] and (scope == "all" or (row["current"] if scope == "current" else not row["current"]))]


def _uninstall_selection(preview, *, scope="current", categories=None, installations=None):
    """Freeze an explicit current/older/all choice, or ticked installations.

    Data categories default to selected only when the running installation is removed too; a choice that keeps
    it running keeps its shared settings, models and history unless categories are named.
    """
    if scope not in _UNINSTALL_SCOPES:
        raise ValueError("scope")
    chosen = None if categories is None else list(categories)
    if chosen is not None and (len(set(chosen)) != len(chosen)
                               or any(key not in _UNINSTALL_CATEGORIES for key in chosen)):
        raise ValueError("categories")
    if scope == "custom":
        ticked = list(installations or ())
        selectable = [row["key"] for row in preview["installations"] if row["selectable"]]
        if len(set(ticked)) != len(ticked) or any(key not in selectable for key in ticked):
            raise ValueError("installations")
        keys = [key for key in selectable if key in ticked]
    elif installations is not None:
        raise ValueError("installations")
    else:
        keys = _uninstall_scope_keys(preview, scope)
    if not keys:
        raise ValueError(tr("No removable spaCR installation was found for this scope."))
    if chosen is None:
        chosen = [] if _uninstall_keeps_running(preview, {"installations": keys}) else list(_UNINSTALL_CATEGORIES)
    return {"operation_id": preview["operation_id"], "snapshot_id": preview["snapshot_id"],
            "scope": scope, "installations": keys, "categories": chosen}


def _uninstall_keeps_running(preview, selection):
    """True when no selected installation is the one running, so spaCR reopens after removal."""
    current = {row["key"] for row in preview["installations"] if row["current"]}
    return bool(current) and not current.intersection(selection["installations"])


def _validate_uninstall_selection(preview, selection, machine, *, after_shutdown=False):
    """Re-discover selected scope and reject changed roots before any deletion or package operation."""
    expected = _uninstall_selection(
        preview, scope=selection.get("scope"), categories=selection.get("categories"),
        installations=selection.get("installations") if selection.get("scope") == "custom" else None)
    if expected != selection:
        raise ValueError(tr("The uninstall selection changed. Review the paths again."))
    current = _uninstall_preview(preview["operation_id"], system=machine)
    available = {row["key"]: row for row in current["installations"]}
    selected = []
    for row in preview["installations"]:
        if row["key"] not in selection["installations"]:
            continue
        fresh = available.get(row["key"])
        if fresh is None:
            raise ValueError(tr("The installation changed. Review the uninstall paths again."))
        before, now = dict(row["record"]), dict(fresh["record"])
        before.pop("running", None)
        now.pop("running", None)
        if (before != now or row["identity"] != fresh["identity"]
                or row["paths"] != fresh["paths"] or not fresh["selectable"]):
            raise ValueError(tr("The installation changed. Review the uninstall paths again."))
        if fresh["record"]["running"] and not after_shutdown:
            raise ValueError(tr("spaCR must close before this installation can be removed."))
        selected.append(_record_from_json(dict(fresh["record"], running=False)))
    fresh_data = {(row["category"], row["path"]): row for row in current["cleanup"]}
    cleanup = []
    for row in preview["cleanup"]:
        if not row["selectable"] or row["category"] not in selection["categories"]:
            continue
        fresh = fresh_data.get((row["category"], row["path"]))
        if fresh != row:
            raise ValueError(tr("The cleanup paths changed. Review the uninstall paths again."))
        cleanup.append(row)
    return selected, cleanup


def _apply_uninstall_selection(preview, selection, *, system=None, after_shutdown=False):
    """Remove only a revalidated confirmed scope, retaining shared environments and opted-out data."""
    machine = system or _Machine()
    records, cleanup = _validate_uninstall_selection(preview, selection, machine,
                                                     after_shutdown=after_shutdown)
    return _apply_uninstall_records(preview, selection, records, cleanup, machine)


def _apply_uninstall_records(preview, selection, records, cleanup, machine):
    """Apply the confirmed user-owned remainder after any native software removal."""
    for row in cleanup:
        if not row["path"].startswith("registry:") and _uninstall_path_state(row["path"]) != row["identity"]:
            raise ValueError(tr("The cleanup paths changed. Review the uninstall paths again."))
    protected = [row["path"] for row in preview["cleanup"] if not row["path"].startswith("registry:")]
    reports = []
    for record in records:
        report = remove_install(record, ticked=True, keep=protected, system=machine)
        reports.append(report)
        if not report.ok:
            return {"ok": False, "reports": [asdict(item) for item in reports], "cleanup": None}
    data = RemovalReport(record=None)
    keep = [row["path"] for row in preview["cleanup"]
            if (not row["selectable"] or row["category"] not in selection["categories"])
            and not row["path"].startswith("registry:")]
    for row in cleanup:
        if row["path"].startswith("registry:"):
            _delete_registry_tree(machine.registry, _SETTINGS_REGISTRY_KEY, data)
        else:
            _delete(row["path"], keep, data, tr("Permission denied. These files were kept."))
    return {"ok": data.ok, "reports": [asdict(item) for item in reports], "cleanup": asdict(data)}


def _desktop_processes():
    """Process ids of running spaCR desktops, found by their module argument."""
    try:
        done = subprocess.run(["pgrep", "-f", "spacr[.]qt"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(pid) for pid in done.stdout.split() if pid.isdigit() and int(pid) != os.getpid()]


_INSTALLER_ADMIN_PATHS = ("/Applications/spaCR.app", "/Library/Application Support/spaCR", "/usr/local/bin/spacr")


def _installer_uninstall_apply(preview, categories, *, system=None):
    """Remove every detected installation from the installer window, leaving protected app parts to its admin step.

    The installer has no running desktop to hand over to, so a running spaCR refuses the removal rather than
    being closed. Records needing administrator rights are reported, never removed here.
    """
    machine = system or _Machine(running_prefix="")
    if system is None and _desktop_processes():
        raise ValueError(tr("Quit spaCR before uninstalling it from the installer."))
    selection = _uninstall_selection(preview, scope="all", categories=categories)
    records, cleanup = _validate_uninstall_selection(preview, selection, machine, after_shutdown=False)
    protected = [record for record in records if record.needs_admin or record.layout == "macos-app"]
    owned = [record for record in records if record not in protected]
    outcome = _apply_uninstall_records(preview, selection, owned, cleanup, machine)
    outcome["admin"] = [path for path in _INSTALLER_ADMIN_PATHS if os.path.lexists(path)] if protected else []
    return outcome


def _uninstall_binding(preview, selection):
    """Bind the exact displayed scope and selected categories into the existing shutdown lease."""
    return hashlib.sha256(json.dumps([preview, selection], sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _macos_native_uninstall_adapter(preview, records, machine):
    """Select the installed native helper only for its fixed package-owned macOS paths."""
    if machine.platform != "macos":
        return None
    app = machine.path("/Applications/spaCR.app")
    support = machine.path("/Library/Application Support/spaCR")
    eligible = [record for record in records if record.layout == "macos-app"
                and record.root in {app, support} and any(
                    _exists(path) and (not os.access(os.path.dirname(path), os.W_OK | os.X_OK)
                        or os.path.isdir(path) and (not os.access(path, os.W_OK | os.X_OK)
                            or machine.real and os.lstat(path).st_uid != os.geteuid()))
                    for path in [record.root, *record.registrations, *record.launchers])]
    if not eligible:
        return None
    helper = os.path.join(app, "Contents", "MacOS", "install-spacr-gui")
    capabilities = os.path.join(app, "Contents", "Resources", "uninstall-capabilities.json")
    for leaf in (helper, capabilities):
        current = leaf
        while _inside(current, app):
            metadata = os.lstat(current)
            if (stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != (0 if machine.real else os.getuid())
                    or stat.S_IMODE(metadata.st_mode) & 0o022
                    or not (stat.S_ISREG(metadata.st_mode) if current == leaf else stat.S_ISDIR(metadata.st_mode))):
                raise PermissionError(tr("Uninstall preparation stopped: {error}", error=leaf))
            current = os.path.dirname(current)
    descriptor = os.open(capabilities, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, encoding="utf-8") as stream:
        metadata = os.fstat(stream.fileno())
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != (0 if machine.real else os.getuid())
                or stat.S_IMODE(metadata.st_mode) & 0o022 or metadata.st_size > 128
                or stream.read(129).strip() != '{"native_uninstall_schema":1}'):
            raise ValueError(tr("Uninstall preparation stopped: {error}", error=capabilities))
    if os.path.realpath(helper) != helper:
        raise PermissionError(tr("Uninstall preparation stopped: {error}", error=helper))
    targets, launchers = set(), set()
    for record in eligible:
        targets.update([record.root, *record.registrations])
        launchers.update(record.launchers)
    command_link = machine.path("/usr/local/bin/spacr")
    if (app not in targets or not targets <= {app, support}
            or not launchers <= {command_link}
            or any(record.shortcuts or record.menu_entries for record in eligible)):
        raise ValueError(tr("The installation changed. Review the uninstall paths again."))
    if launchers and (not os.path.islink(command_link)
                      or os.readlink(command_link) != os.path.join(app, "Contents", "MacOS", "spaCR")):
        raise ValueError(tr("The installation changed. Review the uninstall paths again."))
    return {"adapter": "macos-online-native", "helper": helper,
            "roots": [record.root for record in eligible],
            "desktop_executable": machine.executable,
            "request": {"launcher": bool(launchers), "targets": [
                {"path": path, "identity": list(_identity(path)),
                 "keep": [row["path"] for row in preview["cleanup"]
                          if not row["path"].startswith("registry:") and _inside(row["path"], path)]}
                for path in sorted(targets)]}}


def _macos_native_uninstall_command(adapter, endpoint, pid):
    """Request graphical authorization for one installed native helper with fixed arguments."""
    import shlex

    arguments = [adapter["helper"], "--spacr-uninstall-native", endpoint, str(pid),
                 adapter["desktop_executable"], json.dumps(adapter["request"], separators=(",", ":"))]
    command = "exec " + shlex.join(arguments)
    literal = command.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
    return ["/usr/bin/osascript", "-e",
            f'do shell script "{literal}" with administrator privileges without altering line endings']


def _native_uninstall_adapter(preview, records, machine):
    """Bind each supported protected installation to its installed administrator helper."""
    if machine.platform == "windows" and machine.real and not _windows_process_elevated():
        return _windows_native_uninstall_adapter(preview, records, machine)
    if machine.platform == "macos":
        frozen = _macos_frozen_uninstall_adapter(preview, records, machine)
        if frozen is not None:
            return frozen
    if machine.platform != "linux":
        return _macos_native_uninstall_adapter(preview, records, machine)
    selected = [record for record in records if record.layout == "linux-deb" and record.needs_admin]
    if not selected or not machine.real or os.geteuid() == 0:
        return None
    if (len(selected) != 1 or selected[0].root != "/opt/spacr"
            or selected[0].registrations != ("deb:spacr",)
            or any(_inside(row["path"], "/opt/spacr") for row in preview["cleanup"]
                   if not row["path"].startswith("registry:"))):
        raise ValueError(tr("Uninstall preparation stopped: {error}", error="/opt/spacr"))
    helper = _debian_check_package_owner(selected[0].version, run=_run)
    marker = "/opt/spacr/uninstall-protocol.json"
    details = os.lstat(marker)
    if (not stat.S_ISREG(details.st_mode) or details.st_uid != 0
            or stat.S_IMODE(details.st_mode) & 0o022 or details.st_size > 1024):
        raise ValueError(tr("Uninstall preparation stopped: {error}", error=marker))
    with open(marker, encoding="utf-8") as stream:
        if json.load(stream) != {"schema": 1, "adapter": "linux-deb-uninstall-v1"}:
            raise ValueError(tr("Uninstall preparation stopped: {error}", error=marker))
    if not os.path.isfile("/usr/bin/pkexec") or not os.access("/usr/bin/pkexec", os.X_OK):
        raise PermissionError(tr("graphical administrator authorization is unavailable"))
    return {"adapter": "linux-deb-native", "helper": helper, "roots": [selected[0].root],
            "desktop_executable": machine.executable, "desktop_uid": os.getuid(),
            "previous_version": selected[0].version}


def _native_uninstall_command(adapter, endpoint, pid):
    """Build one graphical authorization request without elevating user-owned code."""
    if adapter["adapter"] == "macos-online-native":
        return _macos_native_uninstall_command(adapter, endpoint, pid)
    if adapter["adapter"] == "macos-frozen-native":
        return _macos_frozen_authorized_command("uninstall-macos-root", [
            "--phase-channel", endpoint, "--desktop-pid", str(pid),
            "--desktop-executable", adapter["desktop_executable"],
            "--previous-version", adapter["previous_version"],
            "--app-identity", ":".join(str(value) for value in adapter["identity"]),
            "--keep-json", json.dumps(adapter["keep"], separators=(",", ":"))])
    if adapter["adapter"] != "linux-deb-native":
        raise ValueError("unsupported_native_uninstall")
    return ["/usr/bin/pkexec", adapter["helper"], "uninstall-debian-root", "--phase-channel", endpoint,
            "--desktop-pid", str(pid), "--desktop-executable", adapter["desktop_executable"],
            "--desktop-uid", str(adapter["desktop_uid"]), "--previous-version", adapter["previous_version"]]


def _uninstall_debian_root(endpoint, desktop_pid, desktop_executable, desktop_uid, previous_version):
    """Remove only the trusted spaCR Debian package after authorization and the desktop's exit."""
    _debian_authenticated_user(desktop_uid)
    if _desktop_process_identity(desktop_pid) != (desktop_uid, os.path.realpath(desktop_executable)):
        raise ValueError(tr("The running desktop could not be identified."))
    helper = _debian_check_package_owner(previous_version)
    if not getattr(sys, "frozen", False) or os.path.realpath(sys.executable) != helper:
        raise ValueError(tr("Uninstall preparation stopped: {error}", error=helper))
    with _UpdatePhaseClient(endpoint, desktop_pid, desktop_executable) as channel:
        with _debian_frontend_lock():
            code, output = _debian_root_runner(["/usr/bin/dpkg", "--no-act", "--remove", "spacr"])
        if code:
            raise RuntimeError(tr("Uninstall preparation stopped: {error}", error=output[-2000:]))
        channel._send("prepared")
        decision = channel._receive({"activate", "cancel"}, timeout=_WAIT_SECONDS)
        if decision["phase"] != "activate":
            return 3
        if not _wait_for_exit(desktop_pid, timeout=1):
            raise RuntimeError(tr("spaCR must close before this installation can be removed."))
        with _debian_frontend_lock():
            _debian_check_package_owner(previous_version)
            code, output = _debian_root_runner(["/usr/bin/dpkg", "--remove", "spacr"])
        if code:
            raise RuntimeError(tr("Uninstall preparation stopped: {error}", error=output[-2000:]))
        channel._send("complete")
    return 0


def _macos_frozen_uninstall_adapter(preview, records, machine):
    """Select the signed frozen bundle's installed helper without elevating user-owned code."""
    app = machine.path("/Applications/spaCR.app")
    selected = [record for record in records if record.layout == "macos-app" and record.root == app]
    if not selected or not machine.real or not os.path.isfile(os.path.join(app, "Contents", "Frameworks", "spacr-update-helper")):
        return None
    required = [os.path.dirname(app), app, os.path.join(app, "Contents"),
                os.path.join(app, "Contents", "MacOS"), os.path.join(app, "Contents", "Frameworks")]
    if os.lstat(app).st_uid == os.geteuid() and all(os.access(path, os.W_OK | os.X_OK) for path in required):
        return None
    if len(selected) != 1 or selected[0].registrations != (app,) or selected[0].launchers:
        raise ValueError(tr("The installation changed. Review the uninstall paths again."))
    _macos_frozen_trusted_helper()
    info = _macos_bundle_identity(app, selected[0].version)
    return {"adapter": "macos-frozen-native", "roots": [app], "identity": list(_identity(app)),
            "previous_version": info["SPACRPackageVersion"], "desktop_executable": machine.executable,
            "keep": [row["path"] for row in preview["cleanup"]
                     if not row["path"].startswith("registry:") and _inside(row["path"], app)]}


def _macos_uninstall_tree(path, identity, keep, *, apply=False, owner=0):
    """Traverse bound directories by descriptor, preserving selected paths and never following links."""
    def visit(parent, name, absolute):
        """Remove one verified child through its parent descriptor while preserving selected paths."""
        if any(_inside(absolute, saved) for saved in keep):
            return False
        before = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if stat.S_ISDIR(before.st_mode):
            if before.st_uid != owner or stat.S_IMODE(before.st_mode) & 0o022:
                raise PermissionError(tr("Uninstall preparation stopped: {error}", error=absolute))
            descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            try:
                actual = os.fstat(descriptor)
                if (actual.st_dev, actual.st_ino) != (before.st_dev, before.st_ino):
                    raise ValueError(tr("The installation changed. Review the uninstall paths again."))
                removed = True
                for child in sorted(os.listdir(descriptor)):
                    removed = visit(descriptor, child, os.path.join(absolute, child)) and removed
            finally:
                os.close(descriptor)
            if not removed:
                return False
        elif not (stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode)):
            raise ValueError(tr("Uninstall preparation stopped: {error}", error=absolute))
        if apply:
            actual = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if (actual.st_dev, actual.st_ino, stat.S_IFMT(actual.st_mode)) != (
                    before.st_dev, before.st_ino, stat.S_IFMT(before.st_mode)):
                raise ValueError(tr("The installation changed. Review the uninstall paths again."))
            (os.rmdir if stat.S_ISDIR(before.st_mode) else os.unlink)(name, dir_fd=parent)
        return True

    if (not isinstance(keep, list) or len(keep) > 256
            or any(not isinstance(saved, str) or os.path.abspath(saved) != saved or not _inside(saved, path)
                   for saved in keep) or tuple(_identity(path)) != tuple(identity)
            or os.path.realpath(path) != path):
        raise ValueError(tr("The installation changed. Review the uninstall paths again."))
    parent = os.open(os.path.dirname(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        actual = os.stat(os.path.basename(path), dir_fd=parent, follow_symlinks=False)
        if (actual.st_dev, actual.st_ino) != tuple(identity) or not stat.S_ISDIR(actual.st_mode):
            raise ValueError(tr("The installation changed. Review the uninstall paths again."))
        visit(parent, os.path.basename(path), path)
    finally:
        os.close(parent)


def _uninstall_macos_root(endpoint, desktop_pid, desktop_executable, previous_version, identity, keep):
    """Remove only the fixed signed macOS bundle after native authorization and actual desktop exit."""
    import re

    target = "/Applications/spaCR.app"
    executable = os.path.realpath(sys.executable)
    if (os.geteuid() != 0 or not getattr(sys, "frozen", False)
            or not re.fullmatch(r"/private/var/root/spacr-helper\.[A-Za-z0-9]{8}/helper", executable)):
        raise ValueError(tr("Uninstall preparation stopped: {error}", error="trusted_helper"))
    for path in (os.path.dirname(executable), executable):
        details = os.lstat(path)
        if details.st_uid != 0 or stat.S_IMODE(details.st_mode) & 0o077 or os.path.realpath(path) != path:
            raise ValueError(tr("Uninstall preparation stopped: {error}", error=path))
    owner, desktop = _desktop_process_identity(desktop_pid)
    if owner == 0 or desktop != os.path.realpath(desktop_executable):
        raise ValueError(tr("The running desktop could not be identified."))
    installed = _macos_frozen_trusted_helper()
    if _macos_digest(executable) != _macos_digest(installed):
        raise ValueError(tr("Uninstall preparation stopped: {error}", error="trusted_helper"))
    _macos_bundle_identity(target, previous_version)
    if len(keep) > 32768 or not re.fullmatch(r"[0-9]+:[0-9]+", identity):
        raise ValueError(tr("The installation changed. Review the uninstall paths again."))
    bound = tuple(int(value) for value in identity.split(":"))
    saved = json.loads(keep)
    if not isinstance(saved, list):
        raise ValueError(tr("The installation changed. Review the uninstall paths again."))
    saved += [os.path.join(target, name) for name in os.listdir(target) if name != "Contents"]
    _macos_uninstall_tree(target, bound, saved)
    with _UpdatePhaseClient(endpoint, desktop_pid, desktop_executable) as channel:
        channel._send("prepared")
        if channel._receive({"activate", "cancel"}, timeout=_WAIT_SECONDS)["phase"] != "activate":
            return 3
        if not _wait_for_exit(desktop_pid, timeout=1):
            raise RuntimeError(tr("spaCR must close before this installation can be removed."))
        _macos_frozen_trusted_helper()
        _macos_bundle_identity(target, previous_version)
        _macos_uninstall_tree(target, bound, saved)
        _macos_uninstall_tree(target, bound, saved, apply=True)
        channel._send("complete")
    return 0


def _run_native_uninstall(plan, adapter, handshake, wait=None):
    """Finish native authorization while the desktop remains open, then await its approved exit."""
    if adapter["adapter"] == "windows-frozen-native":
        return _run_windows_native_uninstall(plan, adapter, handshake, wait=wait)
    with _UpdatePhaseServer(plan["workdir"]) as channel:
        command = _native_uninstall_command(adapter, channel.endpoint, plan["pid"])
        with open(os.path.join(plan["workdir"], "native-uninstall.log"), "xb") as output:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            activated = False
            try:
                def stopped():
                    """Check desktop cancellation before observing the authorized helper's completion."""
                    handshake.check()
                    return process.poll() is not None

                channel._receive({"prepared"}, stopped=stopped)
                handshake.ready()
                handshake.wait_for_shutdown(wait=wait)
                channel._send("activate")
                activated = True
                channel._receive({"complete"}, stopped=lambda: process.poll() is not None)
                if process.wait(timeout=10):
                    raise RuntimeError(tr("Uninstall preparation stopped: {error}", error=plan["workdir"]))
            finally:
                if not activated and channel._connection is not None:
                    try:
                        channel._send("cancel")
                    except OSError:
                        pass
                if process.poll() is None and not activated:
                    process.terminate()


def _windows_known_folder(identifier):
    """Read an operating-system folder without trusting an inherited environment variable."""
    import ctypes

    shell = _windows_update_library("shell32")
    function = shell.SHGetFolderPathW
    function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_wchar_p]
    function.restype = ctypes.c_int32
    path = ctypes.create_unicode_buffer(32768)
    if function(None, identifier, None, 0, path) != 0 or not path.value:
        raise OSError(tr("Uninstall preparation stopped: {error}", error="windows_folder"))
    return _installer_metadata_folder(path.value, "windows_uninstall_folder", windows=True)


def _windows_system_owned(path):
    """Require native system or administrator ownership without effective foreign write access."""
    trusted = {"S-1-5-18", "S-1-5-32-544",
               "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464"}
    _installer_metadata_folder(path, "windows_uninstall_path", windows=True)
    details = os.lstat(path)
    if not (stat.S_ISREG(details.st_mode) or stat.S_ISDIR(details.st_mode)):
        raise ValueError(tr("Uninstall preparation stopped: {error}", error=path))
    evidence = _windows_pip_acl(path)
    if (evidence.get("owner") not in trusted or evidence.get("has_dacl") is not True
            or not evidence.get("rules")):
        raise PermissionError(tr("Uninstall preparation stopped: {error}", error=path))
    for rule in evidence["rules"]:
        if (rule.get("type") not in (0, 1) or type(rule.get("rights")) is not int
                or type(rule.get("propagation")) is not int or rule["propagation"] & ~3
                or not isinstance(rule.get("sid"), str)
                or (rule["type"] == 0 and not rule["propagation"] & 2
                    and rule["rights"] & 0x500D0156 and rule["sid"] not in trusted)):
            raise PermissionError(tr("Uninstall preparation stopped: {error}", error=path))


def _windows_uninstall_targets(request):
    """Validate all fixed installed roots and manifest-owned files before permitting any removal."""
    from pathlib import PurePosixPath

    if not isinstance(request, dict) or set(request) != {"targets"} or not 0 < len(request["targets"]) <= 4:
        raise ValueError("invalid_windows_uninstall_scope")
    allowed = {os.path.normcase(os.path.join(_windows_known_folder(identifier), name))
               for identifier in (0x26, 0x2a) for name in _NAMES}
    common_menu = _windows_known_folder(0x17)
    results, seen = [], set()
    for target in request["targets"]:
        root = _installer_metadata_folder(target["root"], "windows_uninstall_root", windows=True)
        if os.path.normcase(root) not in allowed or os.path.normcase(root) in seen:
            raise ValueError("invalid_windows_uninstall_root")
        seen.add(os.path.normcase(root))
        if list(_identity(root)) != target["identity"]:
            raise ValueError(tr("The installation changed. Review the uninstall paths again."))
        keep = target["keep"]
        if (not isinstance(keep, list) or len(keep) > 256
                or any(not isinstance(path, str) or os.path.abspath(path) != path or not _inside(path, root)
                       for path in keep)):
            raise ValueError("invalid_windows_uninstall_keep")
        for path in (os.path.dirname(root), root, os.path.join(root, "_internal")):
            _windows_system_owned(path)
        _windows_uninstall_protocol(root)
        version = _frozen_windows_previous_version(root)
        manifest = _frozen_windows_payload(root, version)
        if version != target["version"] or _macos_digest(os.path.join(root, "spacr-payload.json")) != target["manifest"]:
            raise ValueError(tr("The installation changed. Review the uninstall paths again."))
        files = [os.path.join(root, *PurePosixPath(name).parts) for name in manifest["files"]]
        files.extend(os.path.join(root, name) for name in (
            "Uninstall.exe", "spacr-uninstaller.json", "spacr-payload.json") if _exists(os.path.join(root, name)))
        directories = {root}
        for path in files:
            cursor = os.path.dirname(path)
            while _inside(cursor, root):
                directories.add(cursor)
                if cursor == root:
                    break
                cursor = os.path.dirname(cursor)
            _windows_system_owned(path)
        for path in directories:
            _windows_system_owned(path)
        shortcuts = target["shortcuts"]
        if (not isinstance(shortcuts, list) or any(path not in {
                os.path.join(common_menu, name + ".lnk") for name in _NAMES} for path in shortcuts)):
            raise ValueError("invalid_windows_uninstall_shortcut")
        for path in shortcuts:
            _windows_system_owned(os.path.dirname(path))
            _windows_system_owned(path)
            if not _mentions(path, _spellings(root)):
                raise ValueError("changed_windows_uninstall_shortcut")
        results.append({"root": root, "files": files + shortcuts,
                        "directories": sorted(directories, key=len, reverse=True), "keep": keep})
    return results


def _windows_uninstall_protocol(root):
    """Bind the advertised native protocol to the actual installed uninstaller before launching it."""
    receipt = os.path.join(root, "spacr-uninstaller.json")
    uninstaller = os.path.join(root, "Uninstall.exe")
    for path in (receipt, uninstaller):
        _windows_system_owned(path)
    details = os.lstat(receipt)
    if not stat.S_ISREG(details.st_mode) or details.st_size > 1024:
        raise ValueError("invalid_windows_uninstaller_receipt")
    with open(receipt, encoding="utf-8") as stream:
        bound = json.loads(stream.read(1025))
    expected = {"schema": 1, "adapter": "windows-frozen-uninstall-v1", "sha256": _macos_digest(uninstaller)}
    if bound != expected:
        raise ValueError("changed_windows_uninstaller_receipt")
    return expected["sha256"]


def _windows_native_uninstall_adapter(preview, records, machine):
    """Use the installed native uninstaller only when its declared protocol and payload are intact."""
    selected = [record for record in records if record.layout == "windows-offline" and record.needs_admin]
    if not selected:
        return None
    common_menu = _windows_known_folder(0x17)
    targets, remaining = [], []
    for record in selected:
        root = record.root
        _windows_uninstall_protocol(root)
        marker = os.path.join(root, "_internal", "spacr-uninstall-protocol.json")
        _windows_system_owned(marker)
        if os.path.getsize(marker) > 1024:
            raise ValueError("invalid_windows_uninstall_protocol")
        with open(marker, encoding="utf-8") as stream:
            if json.load(stream) != {"schema": 1, "adapter": "windows-frozen-uninstall-v1"}:
                raise ValueError("invalid_windows_uninstall_protocol")
        targets.append({"root": root, "identity": list(_identity(root)),
                        "version": _frozen_windows_previous_version(root),
                        "manifest": _macos_digest(os.path.join(root, "spacr-payload.json")),
                        "keep": [row["path"] for row in preview["cleanup"]
                                 if not row["path"].startswith("registry:") and _inside(row["path"], root)],
                        "shortcuts": [path for path in record.shortcuts if os.path.dirname(path) == common_menu]})
        remaining.extend(path for path in record.shortcuts if os.path.dirname(path) != common_menu)
    request = {"targets": targets}
    _windows_uninstall_targets(request)
    _windows_system_owned(os.path.join(selected[0].root, "Uninstall.exe"))
    return {"adapter": "windows-frozen-native", "roots": [record.root for record in selected],
            "uninstaller": os.path.join(selected[0].root, "Uninstall.exe"),
            "uninstaller_sha256": _windows_uninstall_protocol(selected[0].root),
            "desktop_executable": machine.executable, "request": request,
            "remaining": [{"path": path, "identity": _uninstall_path_state(path)} for path in remaining]}


def _run_windows_native_uninstall(plan, adapter, handshake, *, wait=None):
    """Use authenticated completion across the native uninstaller's standard outside-root copy."""
    request = os.path.join(plan["workdir"], "native-request.json")
    descriptor = os.open(request, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(adapter["request"], stream, separators=(",", ":"))
    _windows_pip_private_access(request, private_read=True)
    with _UpdatePhaseServer(plan["workdir"]) as channel:
        values = (channel.endpoint, adapter["desktop_executable"], request)
        if any(any(character in value for character in '\x00\r\n"') for value in values):
            raise ValueError("invalid_windows_uninstall_argument")
        arguments = ('/S /SPACRUNINSTALL="' + channel.endpoint + '" /SPACRPID=' + str(plan["pid"])
                     + ' /SPACREXE="' + adapter["desktop_executable"] + '" /SPACRREQUEST="' + request + '"')
        if len(arguments) > 900:
            raise ValueError("windows_uninstall_scope_too_large")
        result = {}

        def authorize():
            """Retain the native uninstaller's bootstrap result until authenticated completion."""
            try:
                result["result"] = _windows_authorized_process(adapter["uninstaller"], arguments, visible=False)
            except BaseException as error:
                result["error"] = error

        def stopped():
            """Treat failed native startup as stopped without confusing bootstrap exit with completion."""
            handshake.check()
            if "error" in result:
                raise result["error"]
            return "result" in result and result["result"][0] != 0

        activated = False
        try:
            with _windows_frozen_installer_pin(adapter["uninstaller"], adapter["uninstaller_sha256"]):
                worker = threading.Thread(target=authorize, name="spacr-uninstall-authorization", daemon=True)
                worker.start()
                channel._receive({"prepared"}, stopped=stopped)
            handshake.ready()
            handshake.wait_for_shutdown(wait=wait)
            channel._send("activate")
            activated = True
            channel._receive({"complete"}, stopped=stopped)
            worker.join(timeout=10)
            if worker.is_alive() or "error" in result or result.get("result", (1, ""))[0]:
                raise RuntimeError(tr("Uninstall preparation stopped: {error}", error=plan["workdir"]))
        finally:
            if not activated and channel._connection is not None:
                try:
                    channel._send("cancel")
                except OSError:
                    pass


def _uninstall_frozen_windows(request, root, endpoint, desktop_pid, desktop_executable):
    """Run only the native uninstaller's hardened outside-root helper against manifest-owned files."""
    if not _windows_process_elevated() or not getattr(sys, "frozen", False):
        raise ValueError("invalid_windows_native_uninstall")
    owner, executable = _desktop_process_identity(desktop_pid)
    if os.path.normcase(executable) != os.path.normcase(os.path.realpath(desktop_executable)):
        raise ValueError(tr("The running desktop could not be identified."))
    _installer_metadata_folder(request, "windows_uninstall_request", windows=True)
    _windows_pip_private_access(request, owner_sid=owner, private_read=True)
    before = os.lstat(request)
    if not stat.S_ISREG(before.st_mode) or before.st_size > 65536:
        raise ValueError("invalid_windows_uninstall_request")
    with open(request, encoding="utf-8") as stream:
        actual = os.fstat(stream.fileno())
        if not stat.S_ISREG(actual.st_mode) or (actual.st_dev, actual.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("changed_windows_uninstall_request")
        request = json.loads(stream.read(65537))
    targets = _windows_uninstall_targets(request)
    if root not in {target["root"] for target in targets} or any(_inside(sys.executable, target["root"]) for target in targets):
        raise ValueError("windows_uninstall_helper_must_be_outside")
    _windows_system_owned(sys.executable)
    _windows_system_owned(os.path.dirname(sys.executable))
    installed = os.path.join(root, "_internal", "spacr-update-helper.exe")
    if _macos_digest(sys.executable) != _macos_digest(installed):
        raise ValueError("windows_uninstall_helper_changed")
    with _UpdatePhaseClient(endpoint, desktop_pid, desktop_executable) as channel:
        channel._send("prepared")
        if channel._receive({"activate", "cancel"}, timeout=_WAIT_SECONDS)["phase"] != "activate":
            return 3
        if not _wait_for_exit(desktop_pid, timeout=1):
            raise RuntimeError(tr("spaCR must close before this installation can be removed."))
        targets = _windows_uninstall_targets(request)
        for target in targets:
            for path in target["files"]:
                if not any(_inside(path, kept) for kept in target["keep"]):
                    os.unlink(path)
            for directory in target["directories"]:
                if any(_inside(directory, kept) for kept in target["keep"]):
                    continue
                try:
                    os.rmdir(directory)
                except OSError as error:
                    if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                        raise
        channel._send("complete")
    return 0


def _start_uninstall_helper(preview, selection, *, pid=None, workdir=None, system=None, spawn=None):
    """Prepare a copied helper outside the removed paths; no removal precedes approved GUI shutdown."""
    machine = system or _Machine()
    records, cleanup = _validate_uninstall_selection(preview, selection, machine, after_shutdown=True)
    native = _native_uninstall_adapter(preview, records, machine)
    native_roots = set(native["roots"]) if native else set()
    denied = []
    for record in records:
        if record.root in native_roots:
            continue
        if (record.needs_admin and machine.real
                and (machine.platform == "windows" and not _windows_process_elevated()
                     or any(token.startswith("deb:") for token in record.registrations)
                     and getattr(os, "geteuid", lambda: 1)() != 0)):
            denied.append(record.root)
        row = next(row for row in preview["installations"] if row["record"]["root"] == record.root)
        paths = ([] if record.kind == "environment" else [record.root]) + row["paths"]
        for path in (path for path in paths if not path.startswith(("deb:", "registry"))):
            if _exists(path) and not os.access(os.path.dirname(path), os.W_OK | os.X_OK):
                denied.append(path)
    for row in cleanup:
        path = row["path"]
        if not path.startswith("registry:") and not os.access(os.path.dirname(path), os.W_OK | os.X_OK):
            denied.append(path)
    if denied:
        raise PermissionError(tr(
            "Administrator access is required for these paths. Nothing was removed and spaCR will stay open: {paths}",
            paths="\n".join(dict.fromkeys(denied))))
    workdir = os.path.abspath(workdir or tempfile.mkdtemp(prefix="spacr-uninstall-"))
    os.makedirs(workdir, mode=0o700, exist_ok=True)
    paths = [record.root for record in records] + [row["path"] for row in cleanup
                                                   if not row["path"].startswith("registry:")]
    if any(_inside(workdir, path) for path in paths):
        raise ValueError(tr("The uninstall helper must be outside every selected path."))
    plan = {"schema": 1, "adapter": "uninstall-v1", "preview": preview, "selection": selection,
            "version": _uninstall_binding(preview, selection),
            "pid": int(os.getpid() if pid is None else pid),
            "records": [row["record"] for row in preview["installations"]],
            "ticked": list(selection["installations"]), "workdir": workdir,
            "running_prefix": machine.running_prefix, "platform": machine.platform,
            "desktop_executable": machine.executable,
            "native": native,
            "relaunch": (([machine.executable] if getattr(sys, "frozen", False)
                          else [machine.executable, "-I", "-m", "spacr.qt"])
                         if _uninstall_keeps_running(preview, selection) else None),
            "handshake": {"schema": 1, "token": os.urandom(32).hex(), "expires": time.time() + _WAIT_SECONDS}}
    handshake_type = _WindowsPipHandshake if machine.platform == "windows" else _FrozenUpdateHandshake
    handshake_type(plan)
    module = os.path.join(workdir, "install_cleanup.py")
    plan_path = os.path.join(workdir, "plan.json")
    running = next((record for record in records if record.kind == "installer"
                    and _inside(machine.executable, record.root)), None)
    if machine.platform == "windows" and running is not None and not getattr(sys, "frozen", False):
        copied = os.path.join(workdir, "helper")
        os.mkdir(copied, mode=0o700)
        prepared = _prepare_windows_uninstall(running.root, copied, machine=machine)
        command = prepared["command"][:3] + ["run-uninstall-plan", plan_path]
        environment = None
    else:
        if not getattr(sys, "frozen", False):
            shutil.copy2(__file__, module)
        original_records = [_record_from_json(row["record"]) for row in preview["installations"]
                            if row["key"] in selection["installations"]]
        command, environment, error = _helper_command(original_records, workdir, module, plan_path, machine)
        if error or not command:
            raise ValueError(tr("Uninstall preparation stopped: {error}", error=error))
        command[-2] = "run-uninstall-plan"
    plan["command"] = command
    descriptor = os.open(plan_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
    (spawn or _spawn_detached)(command, environment, workdir)
    return plan


def _run_uninstall_plan(plan_path, *, system=None, wait=None):
    """Require an intact confirmed plan, explicit shutdown approval and actual exit before uninstalling."""
    descriptor = os.open(plan_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
        details = os.fstat(handle.fileno())
        if not stat.S_ISREG(details.st_mode) or details.st_size > 16 * 1024 * 1024:
            raise ValueError("invalid_uninstall_plan")
        plan = json.load(handle)
    machine = system or _Machine(running_prefix=plan.get("running_prefix"),
                                 executable=plan.get("desktop_executable"))
    if (plan.get("schema") != 1 or plan.get("adapter") != "uninstall-v1"
            or plan.get("platform") != machine.platform
            or os.path.abspath(plan_path) != os.path.join(plan["workdir"], "plan.json")
            or plan.get("version") != _uninstall_binding(plan["preview"], plan["selection"])):
        raise ValueError("invalid_uninstall_binding")
    handshake = (_WindowsPipHandshake if machine.platform == "windows" else _FrozenUpdateHandshake)(plan)
    try:
        records, cleanup = _validate_uninstall_selection(plan["preview"], plan["selection"], machine,
                                                         after_shutdown=True)
        native = _native_uninstall_adapter(plan["preview"], records, machine)
        if native != plan.get("native"):
            raise ValueError(tr("The installation changed. Review the uninstall paths again."))
        if native:
            _run_native_uninstall(plan, native, handshake, wait=wait)
            records = [record for record in records if record.root not in native["roots"]]
            for row in native.get("remaining", []):
                if _uninstall_path_state(row["path"]) != row["identity"]:
                    raise ValueError(tr("The installation changed. Review the uninstall paths again."))
                report = RemovalReport(record=None)
                _delete(row["path"], [], report, tr("Permission denied. These files were kept."))
                if not report.ok:
                    raise PermissionError(tr("Uninstall preparation stopped: {error}", error=row["path"]))
        else:
            handshake.ready()
            handshake.wait_for_shutdown(wait=wait)
        outcome = _apply_uninstall_records(plan["preview"], plan["selection"], records, cleanup, machine)
        destination = os.path.join(plan["workdir"], "result.json")
        with open(destination, "x", encoding="utf-8") as handle:
            json.dump(outcome, handle, indent=2)
        relaunch = plan.get("relaunch")
        if outcome["ok"] and relaunch and os.path.isfile(relaunch[0]):
            _spawn_detached(relaunch, None, os.path.expanduser("~"))
        return 0 if outcome["ok"] else 1
    except Exception as error:
        handshake.failed(error)
        raise



def run_update_sequence(install: Callable[[], object], *,
                        records: Optional[Sequence[InstallRecord]] = None,
                        ticked: Iterable[str] = (), keep: Sequence[str] = (),
                        find=None, remove=None, system=None):
    """Remove old installations; install only if every installer-made copy is gone.

    Every installer-made copy is removed, the spaCR package is uninstalled
    from each ticked environment, and only then is ``install`` called. When an
    installer-made copy could not be removed, ``install`` is not called at
    all, and the reports say what stopped it.

    :param install: zero-argument callable that installs the new version.
    :param records: installations already found; :func:`find_old_installs`
        is called when ``None``.
    :param ticked: roots of the environments the user ticked.
    :param keep: paths removal must leave in place.
    :param find: replaces :func:`find_old_installs`.
    :param remove: replaces :func:`remove_install`.
    :param system: the computer; ``None`` means this one.
    :returns: ``(reports, install_result)``; ``install_result`` is ``None``
        when the install was not run.
    """
    machine = system or _Machine()
    if records is None:
        records = (find or find_old_installs)(system=machine)
    remover = remove or remove_install
    chosen = {_norm(root) for root in ticked}
    reports = [remover(record, ticked=_norm(record.root) in chosen,
                       keep=keep, system=machine) for record in records]
    blocked = [r for r in reports if r.record.kind == "installer" and not r.ok]
    if blocked:
        return reports, None
    return reports, install()


def _format_records(records: Sequence[InstallRecord]) -> List[str]:
    """Return one line per installation, for the console and the install log.

    :param records: installations.
    """
    labels = {"installer": "older copy, will be removed",
              "environment": "your environment, left in place",
              "checkout": "source checkout, skipped"}
    lines = []
    for record in records:
        extra = [record.layout, f"spaCR {record.version or 'unknown version'}"]
        if record.running:
            extra.append("running")
        extra.extend(record.notes)
        lines.append(f"  {record.root}  [{labels.get(record.kind, record.kind)}; "
                     f"{', '.join(extra)}]")
    return lines


def _format_reports(reports: Sequence[RemovalReport]) -> List[str]:
    """Return what removal did, one line per item.

    :param reports: removal reports.
    """
    lines = []
    for report in reports:
        lines.extend(f"  removed: {item}" for item in report.removed)
        lines.extend(f"  could not remove: {item} ({why})" for item, why in report.failed)
        if report.record.kind == "installer":
            lines.extend(f"  left: {item} ({why})" for item, why in report.skipped)
    return lines



def _record_from_json(data: Dict) -> InstallRecord:
    """Rebuild an :class:`InstallRecord` from its JSON form.

    :param data: a mapping written by :func:`dataclasses.asdict`.
    """
    values = dict(data)
    for name in ("launchers", "shortcuts", "menu_entries", "registrations", "notes"):
        values[name] = tuple(values.get(name) or ())
    return InstallRecord(**values)


def _requires_frozen_adapter(record: InstallRecord) -> bool:
    """Identify installer families that cannot use an online replacement recipe.

    :param record: the discovered installation to classify.
    :returns: whether replacement requires a native frozen-family adapter.
    """
    return (
        record.layout in {"windows-offline", "linux-deb"}
        or (record.layout == "macos-app" and record.root.lower().endswith(".app"))
    )



_MACOS_PROTECTED_TRANSACTION = r"""
set -eu
PATH=/usr/bin:/bin:/usr/sbin:/sbin
export PATH
LC_ALL=C
export LC_ALL
umask 077
[ "$(/usr/bin/id -u)" = 0 ] || exit 70
for parent in / /Applications /private /private/var /private/var/root; do
  [ ! -L "$parent" ] && [ -d "$parent" ] || exit 71
  [ "$(cd "$parent" && /bin/pwd -P)" = "$parent" ] || exit 71
  [ "$(/usr/bin/stat -f %u "$parent")" = 0 ] || exit 71
  parent_mode=$(/usr/bin/stat -f %Lp "$parent")
  [ $((0$parent_mode & 0002)) = 0 ] || exit 71
  if [ $((0$parent_mode & 0020)) != 0 ]; then
    parent_group=$(/usr/bin/stat -f %g "$parent")
    [ "$parent_group" = 0 ] || [ "$parent_group" = 80 ] || exit 71
  fi
  if /bin/ls -lde "$parent" | /usr/bin/grep -Eq '^[[:space:]]+[0-9]+:.* allow '; then exit 71; fi
done
target=/Applications/spaCR.app
mounted=0
phase=initial
txn=
private_stage=
old_identity=
new_identity=
receipt_ready=0
phased=${phased:-0}

private_path() {
  [ ! -L "$1" ] && [ -e "$1" ] || return 1
  [ "$(/usr/bin/stat -f %u "$1")" = 0 ] || return 1
  bits=$(/usr/bin/stat -f %Lp "$1")
  [ $((0$bits & 0077)) = 0 ] || return 1
  if /bin/ls -lde "$1" | /usr/bin/grep -Eq '^[[:space:]]+[0-9]+:.* allow '; then return 1; fi
}

identity() {
  [ ! -L "$1" ] && [ -d "$1" ] || return 1
  /usr/bin/stat -f '%d:%i' "$1"
}

readonly_mount() {
  /sbin/mount | /usr/bin/awk -v wanted="$1" '
    index($0, " on " wanted " (") && $0 ~ /[, ]read-only[,)]/ {found=1}
    END {exit !found}'
}

inventory() {
  (
    cd "$1" || exit 1
    /usr/bin/find -s . -exec /bin/sh -c '
      for entry do
        mode=$(/usr/bin/stat -f %Lp "$entry") || exit 1
        if [ -L "$entry" ]; then
          kind=link
          value=$(/usr/bin/readlink -n "$entry" && printf .) || exit 1
          value=${value%.}
        elif [ -f "$entry" ]; then
          kind=file
          value=$(/usr/bin/shasum -a 256 < "$entry") || exit 1
          value=${value%% *}
        elif [ -d "$entry" ]; then
          kind=dir
          value=
        else exit 1; fi
        printf "%s:%s %s %s %s:%s\000" "${#entry}" "$entry" "$kind" "$mode" "${#value}" "$value" || exit 1
      done
    ' sh '{}' + || exit 1
  ) > "$2"
}

check_links() {
  link_root=$1
  /usr/bin/find -s "$link_root" -type l -print > "$private_stage/links"
  while IFS= read -r link; do
    resolved=$link
    hops=0
    while [ -L "$resolved" ]; do
      hops=$((hops + 1))
      [ "$hops" -le 64 ] || return 1
      destination=$(/usr/bin/readlink -n "$resolved" && printf .) || return 1
      destination=${destination%.}
      case "$destination" in /*) resolved=$destination;; *) resolved="$(/usr/bin/dirname "$resolved")/$destination";; esac
      canonical_parent=$(cd "$(/usr/bin/dirname "$resolved")" && /bin/pwd -P) || return 1
      resolved="$canonical_parent/$(/usr/bin/basename "$resolved")"
    done
    if [ -d "$resolved" ]; then resolved=$(cd "$resolved" && /bin/pwd -P) || return 1; fi
    [ -e "$resolved" ] || return 1
    case "$resolved" in "$link_root"/*) ;; *) return 1;; esac
  done < "$private_stage/links"
}

matches_inventory() {
  inventory "$1" "$txn/check.inventory" || return 1
  /usr/bin/cmp "$2" "$txn/check.inventory"
}

verify_bundle() {
  app=$1
  info="$app/Contents/Info.plist"
  [ ! -L "$info" ] && [ -f "$info" ] || return 1
  [ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$info")" = com.einarolafsson.spacr ] || return 1
  [ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$info")" = spacr ] || return 1
  [ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$info")" = "$short_version" ] || return 1
  [ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$info")" = "$build_version" ] || return 1
  [ "$(/usr/libexec/PlistBuddy -c 'Print :SPACRPackageVersion' "$info")" = "$package_version" ] || return 1
  /usr/bin/lipo "$app/Contents/MacOS/spacr" -verify_arch "$(/usr/bin/uname -m)" || return 1
  /usr/bin/codesign --verify --deep --strict "$app" || return 1
  metadata=$(/usr/bin/find "$app" -type f -path "*/spacr-$package_version.dist-info/METADATA") || return 1
  [ -n "$metadata" ] && [ "$(printf '%s\n' "$metadata" | /usr/bin/wc -l | /usr/bin/tr -d ' ')" = 1 ] || return 1
  /usr/bin/grep -Fqx 'Name: spacr' "$metadata" || return 1
  /usr/bin/grep -Fqx "Version: $package_version" "$metadata"
}

write_journal() {
  journal_next=$(/usr/bin/mktemp "$txn/journal.next.XXXXXXXX") || return 1
  printf 'schema=1\nfamily=macos-frozen\nstate=%s\nversion=%s\ndmg_sha256=%s\ntarget=%s\nold_identity=%s\nnew_identity=%s\nprivate_stage=%s\n' \
    "$phase" "$package_version" "$expected_digest" "$target" "$old_identity" "$new_identity" "$private_stage" > "$journal_next" || return 1
  /bin/mv -f "$journal_next" "$txn/journal" || return 1
  /bin/sync
}

restore_transaction() {
  private_path "$txn" && private_path "$txn/old.inventory" && private_path "$txn/source.inventory" || return 1
  current=$(identity "$target" 2>/dev/null || :)
  previous=$(identity "$txn/previous.app" 2>/dev/null || :)
  next=$(identity "$txn/new.app" 2>/dev/null || :)
  failed=$(identity "$txn/failed.app" 2>/dev/null || :)
  if [ "$phase" = committed ]; then
    [ "$current" = "$new_identity" ] && [ "$previous" = "$old_identity" ] || return 1
    matches_inventory "$target" "$txn/source.inventory" && matches_inventory "$txn/previous.app" "$txn/old.inventory" || return 1
    verify_bundle "$target" || return 1
    return 0
  fi
  if [ "$current" = "$old_identity" ]; then
    matches_inventory "$target" "$txn/old.inventory" || return 1
    if [ "$next" = "$new_identity" ]; then
      matches_inventory "$txn/new.app" "$txn/source.inventory" || return 1
    elif [ "$failed" = "$new_identity" ]; then
      matches_inventory "$txn/failed.app" "$txn/source.inventory" || return 1
    else return 1; fi
  else
    [ "$previous" = "$old_identity" ] || return 1
    matches_inventory "$txn/previous.app" "$txn/old.inventory" || return 1
    if [ "$current" = "$new_identity" ]; then
      [ ! -e "$txn/failed.app" ] && [ ! -L "$txn/failed.app" ] || return 1
      matches_inventory "$target" "$txn/source.inventory" || return 1
      phase=recovering
      write_journal || return 1
      /bin/mv -n "$target" "$txn/failed.app" || return 1
      [ "$(identity "$txn/failed.app")" = "$new_identity" ] || return 1
    elif [ -e "$target" ] || [ -L "$target" ]; then return 1
    elif [ "$next" = "$new_identity" ]; then
      matches_inventory "$txn/new.app" "$txn/source.inventory" || return 1
    elif [ "$failed" = "$new_identity" ]; then
      matches_inventory "$txn/failed.app" "$txn/source.inventory" || return 1
    else return 1; fi
    [ ! -e "$target" ] && [ ! -L "$target" ] || return 1
    phase=recovering
    write_journal || return 1
    /bin/mv -n "$txn/previous.app" "$target" || return 1
    [ "$(identity "$target")" = "$old_identity" ] || return 1
    matches_inventory "$target" "$txn/old.inventory" || return 1
  fi
  phase=rolled-back
  write_journal
}

phase_checkpoint() {
  [ "$phased" = 1 ] || return 0
  printf 'SPACR_PHASE_%s\n' "$1"
  answer=
  IFS= read -r answer || answer=retain
  case "$1:$answer" in
    prepared:activate|activated:confirmed) return 0;;
    activated:failed) return 1;;
    activated:*) phase=desktop-unconfirmed; write_journal; exit 79;;
    *) return 1;;
  esac
}

finish() {
  result=$?
  trap - EXIT HUP INT TERM
  set +e
  if [ "$phase" != initial ] && [ "$phase" != committed ] && [ "$phase" != desktop-unconfirmed ] && [ "$phase" != rolled-back ] && [ -n "$old_identity" ] && [ -n "$new_identity" ]; then
    if ! restore_transaction; then
      printf 'Automatic recovery refused changed identities/content; preserved transaction: %s\n' "$txn" >&2
      result=76
    fi
  fi
  detach_status=not-mounted
  if [ "$mounted" = 1 ]; then
    detach_status=detached
    if ! /usr/bin/hdiutil detach "$private_stage/mount" >/dev/null; then
      printf 'Read-only mount retained for recovery: %s\n' "$private_stage/mount" >&2
      detach_status=failed
      result=77
    fi
  fi
  if [ "$receipt_ready" = 1 ]; then
    printf '%s\t%s\t%s\n' "$phase" "$txn" "$detach_status"
    exit 0
  fi
  exit "$result"
}
trap finish EXIT
trap 'exit 130' HUP INT TERM

if [ "$operation" = recover ]; then
  txn=$recovery_transaction
  printf '%s\n' "$txn" | /usr/bin/grep -Eq '^/Applications/\.spacr-update\.[a-zA-Z0-9]{8}$' || exit 78
  [ "$(cd "$txn" && /bin/pwd -P)" = "$txn" ] || exit 78
  private_path "$txn" && private_path "$txn/journal" || exit 78
  [ -f "$txn/journal" ] || exit 78
  seen=
  while IFS='=' read -r key value; do
    case " $seen " in *" $key "*) exit 78;; esac
    seen="$seen $key"
    case "$key" in
      schema) [ "$value" = 1 ] || exit 78;;
      family) [ "$value" = macos-frozen ] || exit 78;;
      state) case "$value" in prepared|old-moved|installed|committed|recovering|rolled-back) journal_phase=$value;; *) exit 78;; esac;;
      version) [ "$value" = "$package_version" ] || exit 78;;
      dmg_sha256) [ "$value" = "$expected_digest" ] || exit 78;;
      target) [ "$value" = "$target" ] || exit 78;;
      old_identity) old_identity=$value;;
      new_identity) new_identity=$value;;
      private_stage) private_stage=$value;;
      *) exit 78;;
    esac
  done < "$txn/journal"
  [ "$(printf '%s\n' "$seen" | /usr/bin/wc -w | /usr/bin/tr -d ' ')" = 9 ] || exit 78
  printf '%s\n%s\n' "$old_identity" "$new_identity" | /usr/bin/grep -Eqv '^[0-9]+:[0-9]+$' && exit 78
  printf '%s\n' "$private_stage" | /usr/bin/grep -Eq '^/private/var/root/spacr-update\.[a-zA-Z0-9]{8}$' || exit 78
  private_path "$private_stage" || exit 78
  [ -d "$private_stage" ] || exit 78
  if readonly_mount "$private_stage/mount"; then mounted=1; fi
  phase=$journal_phase
  restore_transaction || exit 78
  receipt_ready=1
  exit 0
fi
[ "$operation" = replace ] || exit 78
[ "$(identity "$target")" = "$original_identity" ] || exit 72
[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$target/Contents/Info.plist")" = com.einarolafsson.spacr ] || exit 72
private_stage=$(/usr/bin/mktemp -d /private/var/root/spacr-update.XXXXXXXX)
/bin/cp -R -P -X "$download" "$private_stage/image.dmg"
[ ! -L "$private_stage/image.dmg" ] && [ -f "$private_stage/image.dmg" ] || exit 73
actual_digest=$(/usr/bin/shasum -a 256 "$private_stage/image.dmg")
[ "${actual_digest%% *}" = "$expected_digest" ] || exit 73
/bin/mkdir "$private_stage/mount"
mounted=1
/usr/bin/hdiutil attach -readonly -nobrowse -mountpoint "$private_stage/mount" -plist "$private_stage/image.dmg" > "$private_stage/mount.plist"
readonly_mount "$private_stage/mount"
source_app="$private_stage/mount/spaCR.app"
[ ! -L "$source_app" ] && [ -d "$source_app" ] || exit 74
verify_bundle "$source_app"
txn=$(/usr/bin/mktemp -d /Applications/.spacr-update.XXXXXXXX)
private_path "$txn"
/usr/bin/ditto --rsrc --extattr "$source_app" "$txn/new.app"
verify_bundle "$txn/new.app"
inventory "$source_app" "$txn/source.inventory"
inventory "$txn/new.app" "$txn/new.inventory"
/usr/bin/cmp "$txn/source.inventory" "$txn/new.inventory"
check_links "$source_app"
check_links "$txn/new.app"
/bin/chown -R -P root:wheel "$txn/new.app"
/bin/chmod -RN "$txn/new.app"
unsafe_owners=$(/usr/bin/find "$txn/new.app" ! -user root -print)
[ -z "$unsafe_owners" ] || exit 74
unsafe_modes=$(/usr/bin/find "$txn/new.app" ! -type l -perm -002 -print)
[ -z "$unsafe_modes" ] || exit 74
old_identity=$(identity "$target")
[ "$old_identity" = "$original_identity" ] || exit 75
new_identity=$(identity "$txn/new.app")
inventory "$target" "$txn/old.inventory"
phase=prepared
write_journal
phase_checkpoint prepared || exit 79
[ "$(identity "$target")" = "$old_identity" ] || exit 75
matches_inventory "$target" "$txn/old.inventory" || exit 75
/bin/mv -n "$target" "$txn/previous.app"
[ "$(identity "$txn/previous.app")" = "$old_identity" ] || exit 75
phase=old-moved
write_journal
[ ! -e "$target" ] && [ ! -L "$target" ] || exit 75
/bin/mv -n "$txn/new.app" "$target"
[ "$(identity "$target")" = "$new_identity" ] || exit 75
phase=installed
write_journal
verify_bundle "$target"
matches_inventory "$target" "$txn/source.inventory"
matches_inventory "$txn/previous.app" "$txn/old.inventory"
phase_checkpoint activated || exit 79
phase=committed
write_journal
receipt_ready=1
"""


def _macos_protected_script(version, digest, *, image=None, transaction=None,
                            original_identity=None):
    """Build the fixed privileged transaction without entering an authorization boundary."""
    import re
    import shlex

    fields = _macos_version_fields(version)
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("a positive exact release digest is required")
    if (image is None) == (transaction is None):
        raise ValueError("choose exactly one replacement image or recovery transaction")
    if image is not None:
        if (not isinstance(image, str) or not os.path.isabs(image)
                or os.path.basename(image) != f"spaCR-{version}.dmg"
                or any(ord(character) < 32 for character in image)):
            raise ValueError("the replacement must name its exact absolute DMG path")
        if (not isinstance(original_identity, (list, tuple)) or len(original_identity) != 2
                or any(type(value) is not int or value < 0 for value in original_identity)):
            raise ValueError("replacement requires the actual original device and directory inode")
    elif not isinstance(transaction, str) or not re.fullmatch(
            r"/Applications/\.spacr-update\.[A-Za-z0-9]{8}", transaction):
        raise ValueError("recovery requires an exact private standard-Applications transaction")
    values = {"operation": "replace" if image is not None else "recover",
              "package_version": version, "short_version": fields["CFBundleShortVersionString"],
              "build_version": fields["CFBundleVersion"], "expected_digest": digest,
              "download": image or "", "recovery_transaction": transaction or "",
              "original_identity": ":".join(map(str, original_identity or []))}
    return "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items()) + "\n" + _MACOS_PROTECTED_TRANSACTION


def _macos_protected_command(version, digest, *, image=None, transaction=None,
                              original_identity=None):
    """Encode one fixed native administrator transaction without executing user-owned code.

    The protected inventory uses byte lengths under ``LC_ALL=C`` to delimit
    paths and link targets, retaining trailing target newlines and removing only
    readlink's output terminator. Every inventory command propagates failure
    even when a caller uses a conditional or OR-list.

    :param version: exact numeric public package release string.
    :param digest: positive published lowercase SHA-256 of that release's DMG.
    :param image: absolute DMG path for replacement, mutually exclusive with transaction.
    :param transaction: exact protected transaction directory for restart recovery.
    :param original_identity: installed bundle's device/inode pair, required for replacement.
    :returns: quoted OS osascript argument vector; constructing it executes nothing.
    """
    script = _macos_protected_script(version, digest, image=image, transaction=transaction,
                                      original_identity=original_identity)
    literal = script.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return ["/usr/bin/osascript", "-e",
            f'do shell script "{literal}" with administrator privileges without altering line endings']


def _macos_frozen_trusted_helper(*, run=None):
    """Validate the installed root-owned helper and every containing directory."""
    import re

    helper = "/Applications/spaCR.app/Contents/Frameworks/spacr-update-helper"
    invoke = run or _macos_native
    paths = ["/", "/Applications", "/Applications/spaCR.app", "/Applications/spaCR.app/Contents",
             "/Applications/spaCR.app/Contents/Frameworks", helper]
    for path in paths:
        details = os.lstat(path)
        allowed_group = path in {"/", "/Applications"} and details.st_gid in {0, 80}
        if (os.path.realpath(path) != path or details.st_uid != 0
                or not (stat.S_ISREG if path == helper else stat.S_ISDIR)(details.st_mode)
                or stat.S_IMODE(details.st_mode) & (0o002 if allowed_group else 0o022)
                or re.search(r"(?m)^\s+[0-9]+:.* allow ", invoke(["/bin/ls", "-lde", path]))):
            raise ValueError(tr("The installed macOS helper is not protected from modification."))
    invoke(["/usr/bin/codesign", "--verify", "--strict", helper])
    return helper


def _macos_frozen_authorized_command(command, arguments):
    """Authorize only an installed trusted helper copied into a root-private directory."""
    import shlex

    if (command not in {"apply-macos-root", "uninstall-macos-root"}
            or not isinstance(arguments, (list, tuple)) or len(arguments) > 32
            or any(not isinstance(value, str) or len(value) > 32768
                   or any(ord(character) < 32 for character in value) for value in arguments)):
        raise ValueError(tr("The macOS helper request is invalid."))
    if os.geteuid() == 0:
        raise ValueError(tr("The desktop coordinator must run without administrator privileges."))
    helper = _macos_frozen_trusted_helper()
    capability = json.loads(_macos_native([helper, "native-phase-protocol"]))
    if (not isinstance(capability, dict) or capability.get("schema") != 1
            or not isinstance(capability.get("commands"), list)
            or command not in capability["commands"]):
        raise ValueError(tr("The installed macOS helper does not support this operation."))
    script = r'''
set -eu
PATH=/usr/bin:/bin:/usr/sbin:/sbin
export PATH
umask 077
[ "$(/usr/bin/id -u)" = 0 ] || exit 70
helper=/Applications/spaCR.app/Contents/Frameworks/spacr-update-helper
for path in / /Applications /Applications/spaCR.app /Applications/spaCR.app/Contents /Applications/spaCR.app/Contents/Frameworks "$helper" /private /private/var /private/var/root; do
  [ ! -L "$path" ] && [ "$(/usr/bin/stat -f %u "$path")" = 0 ] || exit 71
  mode=$(/usr/bin/stat -f %Lp "$path")
  [ $((0$mode & 0002)) = 0 ] || exit 71
  if [ $((0$mode & 0020)) != 0 ]; then
    [ "$path" = / ] || [ "$path" = /Applications ] || exit 71
    group=$(/usr/bin/stat -f %g "$path")
    [ "$group" = 0 ] || [ "$group" = 80 ] || exit 71
  fi
  if /bin/ls -lde "$path" | /usr/bin/grep -Eq '^[[:space:]]+[0-9]+:.* allow '; then exit 71; fi
done
[ -f "$helper" ] || exit 71
/usr/bin/codesign --verify --strict "$helper"
stage=$(/usr/bin/mktemp -d /private/var/root/spacr-helper.XXXXXXXX)
/bin/cp -pP "$helper" "$stage/helper"
/usr/bin/cmp "$helper" "$stage/helper"
/bin/chmod 700 "$stage/helper"
/usr/bin/codesign --verify --strict "$stage/helper"
trap 'result=$?; if [ "$result" = 0 ]; then /bin/rm -rf "$stage"; fi; exit "$result"' EXIT
'''
    script += '\n/usr/bin/env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/private/var/root TMPDIR="$stage/" PYINSTALLER_RESET_ENVIRONMENT=1 "$stage/helper" '
    script += " ".join(shlex.quote(value) for value in [command, *arguments])
    literal = script.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return ["/usr/bin/osascript", "-e",
            f'do shell script "{literal}" with administrator privileges without altering line endings']


def _macos_phased_transaction(script, channel, version, desktop_pid, *, popen=None):
    """Relay fixed transaction checkpoints over root-private pipes and retain uncertain activation."""
    import selectors

    with tempfile.TemporaryFile(mode="w+b") as errors:
        child = (popen or subprocess.Popen)(["/bin/sh", "-c", script], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=errors, cwd="/", env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "HOME": "/private/var/root", "LC_ALL": "C"})
        phase = "preparing"
        output = bytearray()
        pending = bytearray()
        failure = None
        deadline = time.monotonic() + 10800
        try:
            with selectors.DefaultSelector() as reader:
                reader.register(child.stdout, selectors.EVENT_READ)
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError(tr("The macOS transaction did not finish before its deadline."))
                    if not reader.select(0.2):
                        continue
                    chunk = os.read(child.stdout.fileno(), 8192)
                    if not chunk:
                        break
                    pending.extend(chunk)
                    if len(pending) > 65536:
                        raise ValueError(tr("The macOS helper returned an invalid transaction receipt."))
                    while b"\n" in pending:
                        line, _, remaining = pending.partition(b"\n")
                        pending = bytearray(remaining)
                        if line == b"SPACR_PHASE_prepared" and phase == "preparing":
                            _native_update_ready(channel, version, desktop_pid)
                            child.stdin.write(b"activate\n")
                            child.stdin.flush()
                            phase = "activating"
                        elif line == b"SPACR_PHASE_activated" and phase == "activating":
                            phase = "activated"
                            try:
                                _native_update_activated(channel, version)
                            except _UpdateActivationUnconfirmed:
                                child.stdin.write(b"retain\n")
                                child.stdin.flush()
                                raise
                            except Exception:
                                child.stdin.write(b"failed\n")
                                child.stdin.flush()
                                raise
                            child.stdin.write(b"confirmed\n")
                            child.stdin.flush()
                            phase = "confirmed"
                        else:
                            output.extend(line + b"\n")
                            if len(output) > 65536:
                                raise ValueError(tr("The macOS helper returned an invalid transaction receipt."))
            output.extend(pending)
        except BaseException as error:
            failure = error
        finally:
            try:
                child.stdin.close()
            except OSError:
                pass
            try:
                code = child.wait(timeout=180)
            except subprocess.TimeoutExpired as error:
                raise _UpdateActivationUnconfirmed(tr("The macOS transaction is still running; its recovery files were retained.")) from error
            child.stdout.close()
        if failure is not None:
            raise failure
        if code or phase != "confirmed":
            errors.seek(0)
            detail = errors.read(8192).decode("utf-8", "replace")
            raise RuntimeError(tr("The macOS transaction stopped: {error}", error=detail))
        return output.decode("utf-8")


def _macos_apply_root_transaction(version, digest, image, phase_channel, desktop_pid):
    """Run only the authenticated release transaction from the copied root-owned helper."""
    import re

    executable = os.path.realpath(sys.executable)
    if (os.geteuid() != 0 or not getattr(sys, "frozen", False)
            or not re.fullmatch(r"/private/var/root/spacr-helper\.[A-Za-z0-9]{8}/helper", executable)):
        raise ValueError(tr("The macOS transaction requires its trusted copied helper."))
    for path in (os.path.dirname(executable), executable):
        details = os.lstat(path)
        if details.st_uid != 0 or stat.S_IMODE(details.st_mode) & 0o077 or os.path.realpath(path) != path:
            raise ValueError(tr("The macOS transaction requires its trusted copied helper."))
    owner, desktop = _desktop_process_identity(desktop_pid)
    expected_desktop = "/Applications/spaCR.app/Contents/MacOS/spacr"
    if owner == 0 or desktop != expected_desktop:
        raise ValueError(tr("The running desktop could not be identified."))
    version = _runtime_requested_version(version)
    if _published_digest(_RELEASE_DOWNLOAD + "/v" + version + "/spaCR-" + version + ".dmg") != digest:
        raise ValueError(tr("The macOS release checksum could not be authenticated."))
    target = "/Applications/spaCR.app"
    _macos_frozen_trusted_helper()
    details = os.lstat(target)
    script = "phased=1\n" + _macos_protected_script(version, digest, image=image,
        original_identity=(details.st_dev, details.st_ino))
    with _UpdatePhaseClient(phase_channel, desktop_pid, expected_desktop) as channel:
        return _macos_phased_transaction(script, channel, version, desktop_pid)


def _macos_request_protected_update(version, digest, *, image=None, transaction=None,
                                    expected_inventory=None, run=None):
    """Request OS authorization, then independently verify its receipt at the original UID.

    :param version: exact release expected in the replacement or recovery journal.
    :param digest: published DMG SHA-256 bound to the protected transaction.
    :param image: authenticated DMG path for replacement; omit for recovery.
    :param transaction: protected journal directory for recovery; omit for replacement.
    :param expected_inventory: mounted release inventory, required when replacing.
    :param run: optional native-command runner compatible with _macos_native.
    :returns: verified transaction receipt, including retained backup and journal paths.
    """

    original_uid = os.geteuid()
    if original_uid == 0:
        raise ValueError("a frozen updater must not run as root")
    target = "/Applications/spaCR.app"
    identity = None
    if image is not None:
        if os.path.islink(image) or not os.path.isfile(image) or _macos_digest(image) != digest:
            raise ValueError("the normal-user image does not match the release digest")
        if expected_inventory is None:
            raise ValueError("the actual read-only mounted payload inventory is required")
        if os.path.islink(target) or not os.path.isdir(target):
            raise ValueError("the protected app is not its actual directory")
        details = os.lstat(target)
        identity = [details.st_dev, details.st_ino]
    argv = _macos_protected_command(version, digest, image=image, transaction=transaction,
                                    original_identity=identity)
    receipt = (run or _macos_native)(argv).strip()
    return _macos_protected_receipt(version, digest, receipt, original_uid, image=image,
        transaction=transaction, expected_inventory=expected_inventory, run=run)


def _macos_protected_receipt(version, digest, receipt, original_uid, *, image=None,
                             transaction=None, expected_inventory=None, run=None):
    """Check the fixed privileged receipt and installed bundle from the original desktop account."""
    import re

    target = "/Applications/spaCR.app"
    match = re.fullmatch(r"(committed|rolled-back)\t(/Applications/\.spacr-update\.[A-Za-z0-9]{8})\t(detached|failed|not-mounted)", receipt)
    if match is None:
        raise ValueError("native authorization returned no valid transaction receipt")
    if os.geteuid() != original_uid or original_uid == 0:
        raise RuntimeError("updater identity changed; refusing privileged application handling")
    state, root, detach_status = match.groups()
    if transaction is not None and root != transaction:
        raise ValueError("the recovered transaction differs from the requested journal")
    if image is not None and state != "committed":
        raise ValueError("the native replacement was rolled back")
    if state == "committed":
        _macos_bundle_identity(target, version, run=run)
        if expected_inventory is not None and _macos_tree(target) != expected_inventory:
            raise ValueError("installed protected payload differs from the actual mounted release")
    else:
        _macos_bundle_identity(target, run=run)
    return {"state": state, "version": version, "target": target, "transaction": root,
            "backup": os.path.join(root, "previous.app"), "journal": os.path.join(root, "journal"),
            "original_uid": original_uid, "dmg_sha256": digest,
            "detach": {"status": detach_status},
            "unknown_files": "retained in the complete prior bundle"}


def _macos_native(argv, *, binary=False):
    """Run only a requested native tool in the normal user's sanitized environment.

    :param argv: complete OS-command argument vector, without shell interpretation.
    :param binary: return stdout bytes rather than decoded text when true.
    :returns: captured stdout; nonzero status raises with captured stderr.
    """
    environment = dict(os.environ)
    environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    for key in tuple(environment):
        if key.startswith("DYLD_") or key in {"PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH", "LD_LIBRARY_PATH_ORIG"}:
            environment.pop(key, None)
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True,
                            text=not binary, env=environment, check=False)
    if result.returncode:
        detail = result.stderr.decode("utf-8", "replace") if binary else result.stderr
        raise RuntimeError(f"native macOS command failed ({result.returncode}): {detail}")
    return result.stdout


def _macos_digest(path):
    """Hash payload bytes without importing the application or external packages.

    :param path: file whose contents are to be hashed; callers validate its type.
    :returns: lowercase hexadecimal SHA-256.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _macos_tree(bundle, *, allow_external_links=False):
    """Inventory every file, directory and link without following unknown user content.

    Include the bundle root's mode and propagate traversal errors, so a failed
    directory read cannot produce a partial inventory.

    :param bundle: canonical, actual bundle directory to inventory.
    :param allow_external_links: retain outside link targets only for the prior bundle.
    :returns: relative-path records containing modes, kinds, file hashes or link targets.
    :raises OSError: the complete directory traversal could not be read.
    """
    root = os.path.realpath(bundle)
    if root != os.path.abspath(bundle) or not os.path.isdir(root) or os.path.islink(bundle):
        raise ValueError("a bundle must be an actual canonical directory")

    def refuse_unreadable(error):
        """Propagate a filesystem traversal failure instead of omitting its subtree.

        :param error: operating-system exception reported by os.walk.
        """
        raise error

    result = {".": {"kind": "directory", "mode": stat.S_IMODE(os.lstat(root).st_mode)}}
    for directory, folders, files in os.walk(root, followlinks=False, onerror=refuse_unreadable):
        for name in sorted(set(folders + files)):
            path = os.path.join(directory, name)
            relative = os.path.relpath(path, root)
            details = os.lstat(path)
            record = {"mode": stat.S_IMODE(details.st_mode)}
            if stat.S_ISLNK(details.st_mode):
                target = os.readlink(path)
                if not allow_external_links and os.path.commonpath([root, os.path.realpath(path)]) != root:
                    raise ValueError("the incoming bundle links outside its own payload")
                record.update(kind="symlink", target=target)
            elif stat.S_ISDIR(details.st_mode):
                record.update(kind="directory")
            elif stat.S_ISREG(details.st_mode):
                record.update(kind="file", size=details.st_size, sha256=_macos_digest(path))
            else:
                raise ValueError("a bundle contains an unsupported special file")
            result[relative] = record
    return result


def _macos_version_fields(version):
    """Map bounded package versions to Apple's three-part build field.

    :param version: exact three- or four-component numeric package release; a
        missing fourth component normalizes to zero for the native build field.
    :returns: marketing/build version fields and the exact package-version field.
    """
    import re

    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){2,3}", version):
        raise ValueError("unsupported release version for a native macOS bundle")
    parts = [int(part) for part in version.split(".")]
    parts += [0] * (4 - len(parts))
    major, minor, patch, revision = parts
    if not 1 <= major <= 99 or any(not 0 <= part <= 99 for part in parts[1:]):
        raise ValueError("native bundle version components exceed the supported bounds")
    return {"CFBundleShortVersionString": f"{major}.{minor}.{patch}",
            "CFBundleVersion": f"{major * 100 + minor}.{patch}.{revision}",
            "SPACRPackageVersion": version}


def _macos_frozen_version_check(root, version):
    """Confirm the bundled spaCR release when frozen bundles omit distribution metadata.

    A bundle carrying ``spacr-<version>.dist-info`` must carry exactly one,
    naming the requested release. A frozen bundle without it is identified by
    the exact package version already confirmed in its Info.plist, any bundled
    ``spacr/_version.py`` must agree, and metadata for any other spaCR release
    is refused.

    :param root: canonical application bundle path.
    :param version: exact incoming release.
    :raises ValueError: the bundled release is ambiguous or differs.
    """
    import re

    matches, others, sources = set(), set(), set()
    pattern = re.compile(r"spacr-(.+)\.dist-info", re.IGNORECASE)
    for directory, _, files in os.walk(root, followlinks=False):
        name = os.path.basename(directory)
        found = pattern.fullmatch(name)
        if found and "METADATA" in files:
            target = matches if found.group(1).lower() == version.lower() else others
            target.add(os.path.realpath(os.path.join(directory, "METADATA")))
        if name == "spacr" and "_version.py" in files:
            sources.add(os.path.realpath(os.path.join(directory, "_version.py")))
    if others or len(matches) > 1:
        raise ValueError("the target distribution metadata is absent or ambiguous")
    for path in matches | sources:
        if os.path.commonpath([root, path]) != root:
            raise ValueError("the target distribution metadata escapes its application")
    for metadata in matches:
        with open(metadata, encoding="utf-8") as stream:
            lines = stream.read().splitlines()
        if "Name: spacr" not in lines or f"Version: {version}" not in lines:
            raise ValueError("the frozen distribution version differs from the bundle version")
    for source in sources:
        with open(source, encoding="utf-8") as stream:
            declared = re.findall(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", stream.read(), re.MULTILINE)
        if declared != [version]:
            raise ValueError("the frozen distribution version differs from the bundle version")


def _macos_bundle_identity(bundle, version=None, *, run=None):
    """Check actual native identity, version, executable and distribution metadata.

    :param bundle: canonical application bundle to inspect.
    :param version: exact incoming release; None limits checks to existing identity.
    :param run: optional native-command runner for architecture/signature checks.
    :returns: parsed native Info.plist after the requested checks pass.
    """
    invoke = run or _macos_native
    root = os.path.realpath(bundle)
    if root != os.path.abspath(bundle) or os.path.islink(bundle):
        raise ValueError("application bundle aliases are unsupported")
    with open(os.path.join(root, "Contents", "Info.plist"), "rb") as stream:
        info = plistlib.load(stream)
    if (info.get("CFBundleIdentifier") != "com.einarolafsson.spacr"
            or info.get("CFBundleExecutable") != "spacr"):
        raise ValueError("the application identity does not match the frozen spaCR family")
    if version is not None and any(info.get(key) != expected for key, expected in
                                   _macos_version_fields(version).items()):
        raise ValueError("the actual bundle version differs from the requested release")
    executable = os.path.join(root, "Contents", "MacOS", "spacr")
    if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
        raise ValueError("the native bundle executable is missing or not executable")
    if os.path.commonpath([root, os.path.realpath(executable)]) != root:
        raise ValueError("the bundle executable escapes its application")
    if version is not None:
        _macos_frozen_version_check(root, version)
        architecture = invoke(["/usr/bin/uname", "-m"]).strip()
        if architecture not in {"arm64", "x86_64"}:
            raise ValueError("unsupported native macOS architecture")
        invoke(["/usr/bin/lipo", executable, "-verify_arch", architecture])
        invoke(["/usr/bin/codesign", "--verify", "--deep", "--strict", root])
    return info


def _macos_swap(left, right):
    """Atomically exchange two existing same-filesystem bundle directories on macOS.

    :param left: first existing bundle directory.
    :param right: second existing bundle directory on the same filesystem.
    :raises OSError: native RENAME_SWAP failure.
    """
    import ctypes

    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    exchange = library.renamex_np
    exchange.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    exchange.restype = ctypes.c_int
    if exchange(os.fsencode(left), os.fsencode(right), 0x00000002) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), left, None, right)


def _macos_journal(transaction, data):
    """Durably publish a replacement state before its next atomic filesystem change.

    Each publication uses a fresh private temporary. An interrupted write may
    leave that file behind; future publication preserves it for inspection and
    is not blocked by its name.

    :param transaction: private transaction directory containing the journal.
    :param data: JSON-serializable journal state to publish atomically.
    """
    path = os.path.join(transaction, "journal.json")
    descriptor, temporary = tempfile.mkstemp(prefix="journal-", suffix=".next", dir=transaction)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(transaction, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _macos_replace_user_bundle(target, staged, transaction, version, expected, *, run=None, swap=None):
    """Exchange verified bundles and retain the whole previous tree for rollback or review.

    Committed publication sits outside verification rollback because publication
    may precede a durability error. Both trees then stay in place so recovery can
    follow the actual published journal.

    :param target: installed bundle's canonical path.
    :param staged: verified incoming bundle within the private sibling transaction.
    :param transaction: private same-filesystem transaction directory.
    :param version: exact incoming public release string.
    :param expected: authenticated read-only mounted bundle inventory.
    :param run: optional native-command runner for bundle verification.
    :param swap: optional atomic directory-exchange backend, defaulting to Darwin.
    :returns: committed replacement receipt with whole-old-bundle backup and journal.
    """
    invoke = run or _macos_native
    exchange = swap or _macos_swap
    root = os.path.realpath(transaction)
    if root != os.path.abspath(transaction) or os.path.dirname(root) != os.path.dirname(target):
        raise ValueError("transaction must be a canonical sibling of the installed application")
    if os.path.dirname(staged) != root or os.path.islink(staged):
        raise ValueError("staged bundle must remain inside its private sibling transaction")
    details = os.stat(root)
    if details.st_uid != os.geteuid() or stat.S_IMODE(details.st_mode) & 0o077:
        raise ValueError("the transaction directory must be private to the normal user")
    if os.stat(target).st_dev != os.stat(staged).st_dev:
        raise ValueError("atomic replacement requires the same filesystem")
    _macos_bundle_identity(target, run=invoke)
    _macos_bundle_identity(staged, version, run=invoke)
    old = _macos_tree(target, allow_external_links=True)
    if _macos_tree(staged) != expected:
        raise ValueError("staged bytes differ from the authenticated read-only mounted payload")
    def identity(path):
        """Record the actual directory identity before or after the exchange.

        :param path: bundle directory whose device and inode are recorded.
        :returns: JSON-compatible device/inode pair.
        """
        details = os.lstat(path)
        return [details.st_dev, details.st_ino]
    data = {"schema": 1, "family": "macos-frozen", "target": target, "backup": staged,
            "version": version, "old_identity": identity(target), "new_identity": identity(staged),
            "old_inventory": old, "new_inventory": expected, "state": "prepared"}
    _macos_journal(root, data)
    exchange(target, staged)
    try:
        if identity(target) != data["new_identity"] or identity(staged) != data["old_identity"]:
            raise ValueError("bundle identities changed unexpectedly during replacement")
        if _macos_tree(target) != expected or _macos_tree(staged, allow_external_links=True) != old:
            raise ValueError("bundle content changed during replacement")
        _macos_bundle_identity(target, version, run=invoke)
    except Exception:
        if identity(target) == data["new_identity"] and identity(staged) == data["old_identity"]:
            exchange(target, staged)
            data["state"] = "rolled-back"
            _macos_journal(root, data)
        raise
    data["state"] = "committed"
    _macos_journal(root, data)
    return {"state": "committed", "version": version, "target": target, "backup": staged,
            "journal": os.path.join(root, "journal.json"), "unknown_files": "retained in complete prior bundle"}


class _FrozenUpdateHandshake:
    """Bind frozen preparation and shutdown consent to one private, expiring plan.

    Preparation allows an hour for the multi-gigabyte image and native staging;
    verified readiness then allows ten minutes for consent and orderly shutdown.
    The GUI polls these messages without a network call or blocking process wait.
    """

    def __init__(self, plan):
        """Validate the private message directory and bind every plan input.

        :param plan: the macOS frozen helper plan containing its one-use lease.
        """
        lease = plan["handshake"]
        token = lease["token"]
        self.expires = float(lease["expires"])
        if (lease.get("schema") != 1 or not isinstance(token, str)
                or len(token) != 64 or any(c not in "0123456789abcdef" for c in token)
                or not 0 < self.expires < float("inf")):
            raise ValueError("invalid frozen updater readiness lease")
        requested_root = os.path.abspath(plan["workdir"])
        details = os.lstat(requested_root)
        if (not stat.S_ISDIR(details.st_mode)
                or details.st_uid != os.geteuid() or stat.S_IMODE(details.st_mode) & 0o077):
            raise ValueError("frozen updater readiness requires its private owner directory")
        self.root = os.path.realpath(requested_root)
        resolved = os.lstat(self.root)
        if (resolved.st_dev, resolved.st_ino) != (details.st_dev, details.st_ino):
            raise ValueError("frozen updater readiness directory changed during validation")
        self.pid = int(plan["pid"])
        bound = {key: plan[key] for key in
                 ("adapter", "version", "pid", "records", "ticked", "workdir", "handshake")}
        self.binding = hashlib.sha256(json.dumps(
            bound, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def _read(self, name):
        """Read one bounded owner-only message, rejecting another plan's bytes.

        :param name: internal message filename inside the private helper folder.
        :returns: decoded message, or None before publication.
        """
        try:
            descriptor = os.open(os.path.join(self.root, name),
                                 os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            details = os.fstat(stream.fileno())
            if (not stat.S_ISREG(details.st_mode) or details.st_uid != os.geteuid()
                    or stat.S_IMODE(details.st_mode) & 0o077 or details.st_size > 8192):
                raise ValueError("invalid frozen updater readiness message")
            message = json.loads(stream.read(8193))
        if message.get("binding") != self.binding:
            raise ValueError("frozen updater readiness belongs to a different plan")
        return message

    def _write(self, name, state, error="", expires=None, details=None):
        """Atomically publish a complete message; interruption cannot expose half JSON.

        :param name: internal message filename inside the private helper folder.
        :param state: the protocol state being published.
        :param error: bounded diagnostic for a preparation failure.
        :param expires: a readiness-specific shutdown deadline, or the preparation deadline.
        :param details: optional verified desktop identity carried by a launch acknowledgment.
        """
        descriptor, temporary = tempfile.mkstemp(prefix="readiness-", dir=self.root)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"binding": self.binding, "state": state,
                       "error": str(error)[:2000],
                       "expires": self.expires if expires is None else expires,
                       "details": details}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, os.path.join(self.root, name))

    def progress(self):
        """Record that preparation, or the desktop waiting on its Restart prompt, is still alive.

        Each record moves the stall deadline: preparation times out after
        :data:`_PREPARATION_STALL_SECONDS` without progress, and readiness after
        :data:`_WAIT_SECONDS` without the desktop, rather than on total time.
        """
        self._write("progress.json", "progress", expires=time.time())

    def _last_progress(self):
        """When progress was last recorded, or 0 when never or unreadable."""
        try:
            message = self._read("progress.json")
        except (OSError, ValueError):
            return 0.0
        if message is None or message.get("state") != "progress":
            return 0.0
        try:
            at = float(message["expires"])
        except (KeyError, TypeError, ValueError):
            return 0.0
        return at if 0 < at <= time.time() + 60 else 0.0

    def check(self):
        """Refuse expired or cancelled work, even if ready or approved arrived later.

        The deadline follows progress: the original deadline, or the last progress plus the stall
        allowance, whichever is later, and never past :data:`_PROGRESS_CAP_SECONDS` beyond the plan.
        """
        if self._read("cancelled.json") is not None:
            raise RuntimeError("frozen update cancelled; nothing was changed")
        message = self._read("readiness.json")
        ready = message is not None and message.get("state") == "ready"
        initial = float(message["expires"]) if ready else self.expires
        cap = self.expires + _PROGRESS_CAP_SECONDS
        if not 0 < initial <= cap:
            raise ValueError("invalid frozen updater readiness deadline")
        last = self._last_progress()
        deadline = min(cap, max(initial, last + (_WAIT_SECONDS if ready else _PREPARATION_STALL_SECONDS)))
        if time.time() >= deadline:
            raise RuntimeError("frozen update preparation or shutdown timed out; nothing was changed")

    def status(self):
        """Return the helper's verified readiness or error without waiting."""
        self.check()
        message = self._read("readiness.json")
        if message is not None and message.get("state") not in {"ready", "error"}:
            raise ValueError("invalid frozen updater readiness state")
        return message

    def ready(self):
        """Publish readiness only while the original preparation remains authorized."""
        self.check()
        self._write("readiness.json", "ready", expires=time.time() + _WAIT_SECONDS)

    def failed(self, error):
        """Expose preparation failure while preserving the installed application.

        :param error: the helper's verification or preparation exception.
        """
        self._write("readiness.json", "error", error)

    def cancel(self):
        """Permanently disarm this plan, including any subsequently published readiness."""
        self._write("cancelled.json", "cancelled")

    def approve(self):
        """Authorize replacement after readiness and all GUI shutdown veto checks."""
        message = self.status()
        if message is None or message["state"] != "ready":
            raise RuntimeError("frozen update is not ready for shutdown")
        self._write("approved.json", "approved")

    def wait_for_shutdown(self, wait=None):
        """Require both explicit approval and actual process exit within the lease.

        :param wait: optional short PID-exit probe for deterministic tests.
        """
        while True:
            self.check()
            approved = self._read("approved.json")
            if approved is not None:
                if approved.get("state") != "approved":
                    raise ValueError("invalid frozen updater shutdown approval")
                exited = (wait(self.pid) if wait is not None
                          else _wait_for_exit(self.pid, timeout=0.25))
                if exited:
                    self.check()
                    return
            time.sleep(0.05)


def _process_tree_activity(pid):
    """A total that grows while ``pid`` or a descendant uses the CPU or moves bytes; None when unknown."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        root = psutil.Process(pid)
        members = [root] + root.children(recursive=True)
    except psutil.Error:
        return None
    total = 0.0
    for member in members:
        try:
            times = member.cpu_times()
            total += times.user + times.system
        except (psutil.Error, AttributeError):
            pass
        try:
            counters = member.io_counters()
        except (psutil.Error, AttributeError, NotImplementedError):
            continue
        total += sum(getattr(counters, name, 0) for name in
                     ("read_bytes", "write_bytes", "read_chars", "write_chars", "other_bytes"))
    return total


@contextmanager
def _preparation_heartbeat(handshake, *, interval=30.0, activity=None):
    """Record progress while this helper or its children are doing work, for as long as the block runs.

    A download, a pip resolve or a copy shows up as CPU or byte counts that keep growing; a hung child
    shows none, so its update still times out after the stall allowance. When activity cannot be read
    (no psutil), nothing is recorded and the original deadlines apply.

    :param handshake: the plan's readiness channel.
    :param interval: seconds between activity samples.
    :param activity: replaces :func:`_process_tree_activity` for tests.
    """
    measure = activity or (lambda: _process_tree_activity(os.getpid()))
    stop = threading.Event()

    def beat():
        """Sample activity and record progress whenever it moved."""
        previous = measure()
        while not stop.wait(interval):
            current = measure()
            if current is not None and previous is not None and current != previous:
                try:
                    handshake.progress()
                except (OSError, ValueError):
                    pass
            previous = current

    worker = threading.Thread(target=beat, name="spacr-update-heartbeat", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=interval + 5)


def _run_macos_frozen_update(plan, *, run=None, download=None, wait=None, swap=None):
    """Verify an official read-only DMG and replace a writable app without elevation.

    Cleanup and relaunch failures retain the committed replacement receipt.
    Relaunch requires the new executable to verify itself and reach its main window.

    :param plan: serialized update plan with the running bundle, release, workdir and PID.
    :param run: optional native-command runner, including the captured open request.
    :param download: optional URL-to-file downloader replacing _download.
    :param wait: optional PID-exit waiter returning whether the original app closed.
    :param swap: optional atomic bundle-exchange backend.
    :returns: replacement receipt with distinct detach and relaunch outcomes.
    """
    import re

    invoke = run or _macos_native
    handshake = _FrozenUpdateHandshake(plan) if plan.get("handshake") else None
    if handshake is not None:
        handshake.check()
    original_uid = os.geteuid()
    if original_uid == 0:
        raise ValueError("start the application as its normal user before updating")
    records = [_record_from_json(data) for data in plan["records"]]
    installers = [record for record in records if record.kind == "installer" and record.running]
    if (len(installers) != 1 or not installers[0].running or plan.get("ticked")
            or installers[0].layout != "macos-app" or not installers[0].root.endswith(".app")):
        raise ValueError("mixed or legacy macOS installations require a separately reviewed migration")
    target = installers[0].root
    if os.path.realpath(target) != target or not os.path.isdir(target):
        raise ValueError("the installed app must be an actual canonical directory")
    _macos_bundle_identity(target, run=invoke)
    protected = not os.access(os.path.dirname(target), os.W_OK | os.X_OK)
    if protected and (target != "/Applications/spaCR.app" or handshake is None):
        raise ValueError(tr("Protected macOS updates require the standard application and its desktop handoff."))
    parent_details = os.stat(os.path.dirname(target))
    if parent_details.st_mode & 0o002:
        raise ValueError("an application in a world-writable parent cannot be replaced safely")
    version = str(plan["version"])
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){2,3}", version):
        raise ValueError("unsupported frozen macOS version")
    name = f"spaCR-{version}.dmg"
    image = os.path.join(plan["workdir"], name)
    url = f"{_RELEASE_DOWNLOAD}/v{version}/{name}"
    expected_digest = _published_digest(url)
    if not expected_digest or not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise ValueError("the official DMG has no positive published digest")
    (download or _download)(url, image)
    if _macos_digest(image) != expected_digest:
        raise ValueError("downloaded DMG differs from its published digest")
    mount = tempfile.mkdtemp(prefix="mounted-", dir=os.path.realpath(plan["workdir"]))
    attached = False
    result = None
    try:
        attached = True
        raw_receipt = invoke(["/usr/bin/hdiutil", "attach", "-readonly", "-nobrowse",
                              "-mountpoint", mount, "-plist", image], binary=True)
        receipt = plistlib.loads(raw_receipt)
        mounted = [entry for entry in receipt.get("system-entities", []) if entry.get("mount-point") == mount]
        if len(mounted) != 1 or not os.statvfs(mount).f_flag & os.ST_RDONLY:
            raise ValueError("hdiutil did not provide the requested actual read-only mount")
        source = os.path.join(mount, "spaCR.app")
        _macos_bundle_identity(source, version, run=invoke)
        expected = _macos_tree(source)
        if protected:
            before = os.lstat(target)
            old_inventory = _macos_tree(target, allow_external_links=True)

            def _unchanged():
                """Refuse activation if the installed bundle or authenticated image changed."""
                current = os.lstat(target)
                if ((current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
                        or _macos_tree(target, allow_external_links=True) != old_inventory
                        or _macos_digest(image) != expected_digest):
                    raise ValueError(tr("The installed application changed during staging; activation was stopped."))

            def _authorized(endpoint):
                """Request one native authorization for trusted staging and deferred activation."""
                command = _macos_frozen_authorized_command("apply-macos-root", [
                    "--version", version, "--digest", expected_digest, "--image", image,
                    "--phase-channel", endpoint, "--desktop-pid", str(plan["pid"])])
                return 0, invoke(command)

            def _new_desktop():
                """Verify the activated release and wait for its actual main window."""
                _macos_bundle_identity(target, version, run=invoke)
                if _macos_tree(target) != expected:
                    raise ValueError(tr("The activated macOS application differs from its authenticated release."))
                return _launch_updated_desktop(plan, [os.path.join(target, "Contents", "MacOS", "spacr")],
                    version, target, frozen=True)

            receipt, desktop = _coordinate_native_update(plan, handshake, _authorized,
                check=_unchanged, launch=_new_desktop, wait=wait)
            result = _macos_protected_receipt(version, expected_digest, receipt.strip(), original_uid,
                image=image, expected_inventory=expected, run=invoke)
            result.update(desktop=desktop, relaunch={"status": "ready"})
        else:
            transaction = tempfile.mkdtemp(prefix=".spacr-update-", dir=os.path.dirname(target))
            staged = os.path.join(transaction, "previous-or-new.app")
            invoke(["/usr/bin/ditto", "--rsrc", "--extattr", source, staged])
            if _macos_digest(image) != expected_digest:
                raise ValueError("DMG changed while its payload was staged")
            if handshake is not None:
                _macos_bundle_identity(staged, version, run=invoke)
                if _macos_tree(staged) != expected:
                    raise ValueError("staged bytes differ from the authenticated read-only mounted payload")
                _macos_tree(target, allow_external_links=True)
                handshake.ready()
                handshake.wait_for_shutdown(wait)
            elif not (wait or _wait_for_exit)(int(plan["pid"])):
                raise RuntimeError("spaCR did not close; the installed bundle was not changed")
            result = _macos_replace_user_bundle(target, staged, transaction, version, expected,
                                                run=invoke, swap=swap)
    finally:
        if attached:
            operation_error = sys.exc_info()[1]
            try:
                invoke(["/usr/bin/hdiutil", "detach", mount])
                if result is not None:
                    result["detach"] = {"status": "detached"}
            except Exception as detach_error:
                if result is not None:
                    result["detach"] = {"status": "failed", "mount": mount, "error": str(detach_error)}
                elif operation_error is not None:
                    raise RuntimeError(f"macOS update stopped ({operation_error}); could not detach {mount}: {detach_error}") from operation_error
                else:
                    raise
    result.update(original_uid=original_uid, dmg_sha256=expected_digest)
    return result if protected else _macos_relaunch(result, original_uid, run=invoke, plan=plan)


def _macos_relaunch(result, original_uid, *, run=None, plan=None):
    """Keep verified replacement/recovery state even when normal-user launch fails.

    :param result: verified transaction receipt naming its installed target.
    :param original_uid: normal user that began this replacement or recovery.
    :param run: optional native-command runner for the captured LaunchServices request.
    :param plan: active update plan requiring actual desktop readiness; absent for legacy recovery.
    :returns: the same receipt with a ready, accepted, failed or refused relaunch result.
    """
    invoke = run or _macos_native
    if os.geteuid() != original_uid or original_uid == 0:
        result["relaunch"] = {"status": "refused", "error": "updater identity changed; GUI relaunch requires the original normal user"}
        return result
    try:
        if plan is None:
            invoke(["/usr/bin/open", "-n", result["target"]])
        else:
            executable = os.path.join(result["target"], "Contents", "MacOS", "spacr")
            result["desktop"] = _launch_updated_desktop(
                plan, [executable], result["version"], result["target"], frozen=True)
    except Exception as launch_error:
        result["relaunch"] = {"status": "failed", "error": str(launch_error)}
    else:
        result["relaunch"] = {"status": "ready" if plan is not None else "accepted"}
    return result


def _run_macos_frozen_recovery(transaction, *, protected=False, version=None,
                                digest=None, run=None, swap=None):
    """Recover one explicit frozen-family journal, then request normal-user launch.

    :param transaction: exact original transaction directory, never a scanned guess.
    :param protected: request native authorization for the fixed Applications route.
    :param version: exact release bound to a protected journal; omit for user recovery.
    :param digest: published DMG digest bound to a protected journal; omit for user recovery.
    :param run: optional captured native-tool runner used for verification and relaunch.
    :param swap: optional atomic exchange backend for user-owned recovery only.
    :returns: verified transaction state and independent cleanup/relaunch outcomes.
    """
    if sys.platform != "darwin" or os.geteuid() == 0:
        raise ValueError("macOS frozen recovery requires its native platform and original normal user")
    original_uid = os.geteuid()
    if protected:
        if version is None or digest is None or swap is not None:
            raise ValueError("protected recovery requires its exact version and published digest")
        result = _macos_request_protected_update(
            version, digest, transaction=transaction, run=run)
    else:
        if version is not None or digest is not None:
            raise ValueError("release arguments apply only to protected recovery")
        result = _macos_recover_user_bundle(transaction, run=run, swap=swap)
        _macos_bundle_identity(result["target"], run=run)
        result.update(transaction=os.path.realpath(transaction),
                      journal=os.path.join(os.path.realpath(transaction), "journal.json"),
                      original_uid=original_uid)
    return _macos_relaunch(result, original_uid, run=run)


def _macos_recover_user_bundle(transaction, *, run=None, swap=None):
    """Recover a journaled exchange only when original directory identities and content agree.

    :param transaction: original private sibling directory containing journal.json.
    :param run: optional native-command runner for a committed bundle's identity checks.
    :param swap: optional atomic directory-exchange backend used when restoring.
    :returns: verified committed or newly published rolled-back journal state.
    """
    invoke = run or _macos_native
    exchange = swap or _macos_swap
    root = os.path.realpath(transaction)
    details = os.lstat(root)
    if (root != os.path.abspath(transaction) or details.st_uid != os.geteuid()
            or not stat.S_ISDIR(details.st_mode) or stat.S_IMODE(details.st_mode) & 0o077):
        raise ValueError("recovery requires the original private transaction directory")
    journal = os.path.join(root, "journal.json")
    if os.path.islink(journal):
        raise ValueError("a recovery journal cannot be a symlink")
    with open(journal, encoding="utf-8") as stream:
        data = json.load(stream)
    target, backup = data["target"], data["backup"]
    if (data.get("schema") != 1 or data.get("family") != "macos-frozen"
            or os.path.dirname(target) != os.path.dirname(root) or os.path.dirname(backup) != root
            or os.path.islink(target) or os.path.islink(backup)):
        raise ValueError("recovery paths differ from the original application transaction")

    def identity(path):
        """Return the device and inode of a journaled bundle directory.

        :param path: journaled bundle directory to identify.
        :returns: JSON-compatible device/inode pair.
        """
        details = os.lstat(path)
        return [details.st_dev, details.st_ino]

    old_at_target = identity(target) == data["old_identity"] and identity(backup) == data["new_identity"]
    new_at_target = identity(target) == data["new_identity"] and identity(backup) == data["old_identity"]
    if new_at_target:
        if (_macos_tree(target) != data["new_inventory"]
                or _macos_tree(backup, allow_external_links=True) != data["old_inventory"]):
            raise ValueError("journaled application content changed; manual recovery is required")
        if data["state"] == "committed":
            _macos_bundle_identity(target, data["version"], run=invoke)
            return data
        if data["state"] != "prepared":
            raise ValueError("journal state and actual application identities disagree")
        exchange(target, backup)
    elif old_at_target:
        if (_macos_tree(target, allow_external_links=True) != data["old_inventory"]
                or _macos_tree(backup) != data["new_inventory"] or data["state"] == "committed"):
            raise ValueError("recovery cannot replace changed or committed application content")
    else:
        raise ValueError("one journaled application directory was replaced; preserving all paths")
    data["state"] = "rolled-back"
    _macos_journal(root, data)
    return data

def _reinstall_steps(record: InstallRecord, version: str, workdir: str,
                     machine: _Machine):
    """Return how the new version replaces an installer-made copy.

    :param record: the running installer-made copy.
    :param version: Release selected for a replacement installer. For a macOS
        online upgrade, the minimum acceptable version after an unpinned
        package upgrade.
    :param workdir: the helper's folder, outside every installation.
    :param machine: the computer.
    :returns: ``(fetch, install, relaunch)``: the installer download, the
        command that runs it, and the command that starts the new version.
    """
    if _requires_frozen_adapter(record):
        raise ValueError("this installer family needs a verified frozen replacement adapter")
    suffix = _ASSET_SUFFIX[record.platform]
    name = f"{_NAME}-{version}-{suffix}"
    target = os.path.join(workdir, name)
    fetch = {"url": f"{_RELEASE_DOWNLOAD}/v{version}/{name}", "path": target}
    root = record.root
    if record.platform == "windows":
        install = [target, "/S", f"/D={root}"]
        relaunch = [os.path.join(root, "venv", "Scripts", "pythonw.exe"),
                    "-I", os.path.join(root, "launch_spacr.pyw")]
    elif record.platform == "macos":
        install = ["open", "-W", target]
        relaunch = ["open", "-a", machine.unmapped(
            machine.path(f"/Applications/{_NAME}.app"))]
    else:
        install = ["bash", target, "--install-root", root, "--no-launch",
                   "--skip-system-deps"]
        relaunch = [os.path.join(root, "venv", "bin", "python"), "-I", "-m", "spacr.qt"]
    return fetch, install, relaunch


def _standalone_helper_command(records: Sequence[InstallRecord], workdir: str,
                               plan_path: str, machine: _Machine):
    """Copy the bundled updater outside every removed root without host Python.

    The independently extracted one-file helper contains only this module and
    Python's standard library. Preparing it does not authorize a replacement
    recipe or remove an installation.

    :param records: installations whose roots must survive helper preparation.
    :param workdir: private helper directory outside all installation roots.
    :param plan_path: update plan inside that private directory.
    :param machine: current platform, executable and environment description.
    :returns: helper argv, isolated environment and any preparation error.
    """
    def contains(path, root):
        """Compare canonical paths using the host filesystem's case semantics.

        :param path: candidate nested path.
        :param root: containing directory to check.
        :returns: whether the candidate resolves inside the directory.
        """
        try:
            canonical_path = os.path.normcase(os.path.realpath(path))
            canonical_root = os.path.normcase(os.path.realpath(root))
            return os.path.commonpath([canonical_path, canonical_root]) == canonical_root
        except (OSError, ValueError):
            return False

    bundle = getattr(sys, "_MEIPASS", None)
    if not getattr(sys, "frozen", False) or not bundle:
        return None, None, "a frozen application bundle is required"
    roots = [record.root for record in records if record.kind == "installer"]
    roots.extend(path for record in records if record.kind == "installer"
                 for path in record.registrations if path.lower().endswith(".app"))
    roots.extend([str(bundle), os.path.dirname(machine.executable)])
    executable_folder = os.path.dirname(os.path.realpath(machine.executable))
    contents = os.path.dirname(executable_folder)
    application = os.path.dirname(contents)
    if (os.path.basename(executable_folder) == "MacOS"
            and os.path.basename(contents) == "Contents"
            and application.lower().endswith(".app")):
        roots.append(application)
    destination_root = os.path.realpath(workdir)
    if any(contains(destination_root, root) for root in roots):
        return None, None, "the standalone updater folder is inside an installation"
    canonical_plan = os.path.realpath(plan_path)
    if not contains(canonical_plan, destination_root):
        return None, None, "the update plan must stay inside the standalone updater folder"
    name = "spacr-update-helper" + (".exe" if machine.platform == "windows" else "")
    source = os.path.realpath(os.path.join(str(bundle), name))
    if not contains(source, str(bundle)) or not os.path.isfile(source):
        return None, None, "the bundled standalone updater is missing or outside its bundle"
    target = os.path.join(destination_root, name)
    runtime = os.path.join(destination_root, "runtime")
    try:
        os.makedirs(destination_root, mode=0o700, exist_ok=True)
        if os.name != "nt" and stat.S_IMODE(os.stat(destination_root).st_mode) & 0o077:
            return None, None, "the standalone updater folder must be private to its owner"
        os.mkdir(runtime, mode=0o700)
        with open(source, "rb") as original, open(target, "xb") as copied:
            shutil.copyfileobj(original, copied)
        os.chmod(target, 0o700)
    except OSError as error:
        return None, None, f"could not prepare the standalone updater: {error}"
    environment = dict(machine.environ)
    environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    for variable in ("PYTHONHOME", "PYTHONPATH"):
        environment.pop(variable, None)
    for variable in ("LD_LIBRARY_PATH", "LIBPATH"):
        previous = environment.pop(variable + "_ORIG", None)
        if previous is None:
            environment.pop(variable, None)
        else:
            environment[variable] = previous
    for variable in ("TMPDIR", "TMP", "TEMP"):
        environment[variable] = runtime
    return [target, "run-plan", canonical_plan], environment, None


def _helper_command(records: Sequence[InstallRecord], workdir: str,
                    module: str, plan_path: str, machine: _Machine):
    """Choose a standalone helper or interpreter that survives the removal.

    :param records: every installation in the plan.
    :param workdir: the helper's folder.
    :param module: this module's copy inside ``workdir``.
    :param plan_path: the plan file.
    :param machine: the computer.
    :returns: ``(argv, environment, error)``; ``argv`` is ``None`` when no
        standalone helper or interpreter outside the removed copies exists.
    """
    doomed = [r.root for r in records if r.kind == "installer"]
    tail = ["-I", module, "run-plan", plan_path]
    frozen_executable = (
        bool(getattr(sys, "frozen", False))
        and os.path.realpath(machine.executable) == os.path.realpath(sys.executable)
    )
    if frozen_executable:
        return _standalone_helper_command(records, workdir, plan_path, machine)
    if not frozen_executable and not any(
            _inside(machine.executable, root) for root in doomed):
        return [machine.executable, *tail], None, None
    uv = None
    for record in records:
        if record.kind == "installer" and record.running:
            for name in ("uv.exe", "uv"):
                candidate = os.path.join(record.root, "bootstrap", name)
                if os.path.isfile(candidate):
                    uv = candidate
                    break
    if uv is None and machine.real:
        found = machine.which("uv")
        if found and not any(_inside(found, root) for root in doomed):
            uv = found
    if uv is None:
        return None, None, ("no Python outside the installation could be found "
                            "to finish the update; run the new installer instead")
    copy = os.path.join(workdir, os.path.basename(uv))
    shutil.copy2(uv, copy)
    environment = dict(machine.environ)
    environment["UV_PYTHON_INSTALL_DIR"] = os.path.join(workdir, "python")
    environment["UV_CACHE_DIR"] = os.path.join(workdir, "cache")
    argv = [copy, "run", "--no-project", "--no-config", "--python", "3.12",
            "--managed-python", "--", "python", *tail]
    return argv, environment, None


def _spawn_detached(argv: Sequence[str], env=None, cwd: Optional[str] = None,
                    *, os_name: Optional[str] = None, popen=None, return_process=False):
    """Start a process that keeps running after spaCR exits.

    :param argv: the command.
    :param env: its environment; inherited when ``None``.
    :param cwd: its working folder, which must not be inside an installation.
    :param os_name: :data:`os.name` of the computer; this one when ``None``.
    :param popen: replaces :class:`subprocess.Popen`.
    :param return_process: keep the owned process handle for verified startup and safe failure shutdown.
    :returns: the new process id, or its handle when requested.
    """
    options: Dict[str, object] = {
        "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL, "close_fds": True, "cwd": cwd, "env": env,
    }
    if (os_name or os.name) == "nt":
        options["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0x8)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0x8000000))
    else:
        options["start_new_session"] = True
    process = (popen or subprocess.Popen)([str(a) for a in argv], **options)
    return process if return_process else process.pid


class _UpdateLaunchHandshake(_FrozenUpdateHandshake):
    """A private one-use acknowledgment from the newly launched GUI process."""

    def __init__(self, root, binding):
        """Validate the message location before either side reads a launch request."""
        if (not isinstance(binding, str) or len(binding) != 64
                or any(character not in "0123456789abcdef" for character in binding)):
            raise ValueError(tr("invalid update launch binding"))
        self.root = os.path.abspath(root)
        if os.name == "nt":
            _windows_pip_private_directory(self.root)
        else:
            details = os.lstat(self.root)
            if (not stat.S_ISDIR(details.st_mode) or details.st_uid != os.geteuid()
                    or stat.S_IMODE(details.st_mode) & 0o077):
                raise ValueError(tr("update launch requires a private owner directory"))
            self.root = os.path.realpath(self.root)
            resolved = os.lstat(self.root)
            if (resolved.st_dev, resolved.st_ino) != (details.st_dev, details.st_ino):
                raise ValueError(tr("update launch directory changed during validation"))
        self.binding = binding
        self.expires = time.time() + _WAIT_SECONDS

    def _read(self, name):
        """Use the same bounded message guards as the platform's shutdown handoff."""
        reader = _WindowsPipHandshake._read if os.name == "nt" else super()._read
        return reader(self, name) if os.name == "nt" else reader(name)


def _begin_update_launch():
    """Verify this running executable and imported module before opening its GUI.

    The environment carries only a private message location and one-use binding.
    A normal launch has no update request and does no additional work. It runs
    inside the restarted application, so the application package and
    ``packaging`` are looked up only then; the module itself still runs as a
    standalone helper with the standard library alone.
    """
    request = os.environ.pop("SPACR_UPDATE_LAUNCH", "")
    if not request:
        return None
    root, binding = json.loads(request)
    handshake = _UpdateLaunchHandshake(root, binding)
    try:
        message = handshake._read("launch-request.json")
        if message is None or message.get("state") != "requested" or time.time() >= float(message["expires"]):
            raise ValueError(tr("update launch request expired or missing"))
        if handshake._read("launch-result.json") is not None:
            raise ValueError(tr("update launch request already consumed"))
        expected = message["details"]
        from importlib import import_module

        Version = import_module("packaging.version").Version
        spacr = import_module("spacr")
        shipped = import_module("spacr._version").__version__

        actual = {"version": str(Version(str(spacr.__version__))),
                  "executable": os.path.abspath(sys.executable),
                  "module": os.path.realpath(spacr.__file__),
                  "prefix": os.path.realpath(sys.prefix), "pid": os.getpid()}
        if (Version(actual["version"]) != Version(expected["version"])
                or Version(str(shipped)) != Version(expected["version"])
                or os.path.normcase(actual["executable"]) != os.path.normcase(expected["executable"])
                or not _inside(actual["module"], expected["root"])):
            raise ValueError(tr("restarted executable, module or version differs from the verified update"))
        if expected["frozen"]:
            if not getattr(sys, "frozen", False):
                raise ValueError(tr("the restarted application is not the verified frozen bundle"))
        else:
            from importlib.metadata import distributions

            owners = [dist for dist in distributions()
                      if dist.metadata.get("Name", "").lower().replace("_", "-") in {"spacr", "spacr-nightly"}
                      and os.path.realpath(dist.locate_file("spacr/__init__.py")) == actual["module"]]
            if (len(owners) != 1 or Version(owners[0].version) != Version(expected["version"])
                    or actual["prefix"] != os.path.realpath(expected["root"])
                    or not _inside(str(getattr(owners[0], "_path", "")), expected["root"])):
                raise ValueError(tr("the restarted module is not owned by the verified update"))
        handshake._write("launch-result.json", "starting", details=actual)
        return handshake, actual
    except Exception as error:
        handshake._write("launch-result.json", "error", error)
        raise


def _finish_update_launch(launch):
    """Acknowledge the verified process only after its startup reaches the main window."""
    if launch is not None:
        handshake, actual = launch
        handshake._write("launch-result.json", "ready", details=actual)


class _UpdateLaunchStillRunning(RuntimeError):
    """Keep both runtimes when an unconfirmed desktop cannot be safely stopped."""

    def __init__(self, pid):
        """Bind recovery to the exact desktop that could not be stopped."""
        self.pid = int(pid)
        super().__init__(tr("The updated desktop could not be stopped; both runtimes were retained."))


def _desktop_process_identity(pid):
    """Read a live desktop's native owner and executable without running its code."""
    if isinstance(pid, bool) or not isinstance(pid, int) or not 1 < pid <= 0xFFFFFFFF:
        raise ValueError(tr("The running desktop could not be identified."))
    if os.name == "nt":
        import ctypes

        kernel = _windows_update_library("kernel32")
        security = _windows_update_library("advapi32")
        pointer, dword = ctypes.c_void_p, ctypes.c_uint32
        kernel.OpenProcess.argtypes = [dword, ctypes.c_int32, dword]
        kernel.OpenProcess.restype = pointer
        kernel.CloseHandle.argtypes = [pointer]
        security.OpenProcessToken.argtypes = [pointer, dword, ctypes.POINTER(pointer)]
        security.GetTokenInformation.argtypes = [pointer, dword, pointer, dword, ctypes.POINTER(dword)]
        security.ConvertSidToStringSidW.argtypes = [pointer, ctypes.POINTER(ctypes.c_wchar_p)]
        kernel.QueryFullProcessImageNameW.argtypes = [pointer, dword, ctypes.c_wchar_p, ctypes.POINTER(dword)]
        kernel.LocalFree.argtypes = [pointer]
        process = kernel.OpenProcess(0x1000, False, pid)
        if not process:
            raise OSError(tr("The running desktop could not be identified."))
        token, text = pointer(), ctypes.c_wchar_p()
        try:
            size = dword(32768)
            path = ctypes.create_unicode_buffer(size.value)
            if not kernel.QueryFullProcessImageNameW(process, 0, path, ctypes.byref(size)):
                raise OSError(tr("The running desktop could not be identified."))
            if not security.OpenProcessToken(process, 8, ctypes.byref(token)):
                raise OSError(tr("The running desktop could not be identified."))
            needed = dword()
            security.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
            if not ctypes.sizeof(pointer) <= needed.value <= 1 << 20:
                raise ValueError(tr("The running desktop could not be identified."))
            buffer = ctypes.create_string_buffer(needed.value)
            if not security.GetTokenInformation(token, 1, buffer, needed, ctypes.byref(needed)):
                raise OSError(tr("The running desktop could not be identified."))
            sid = ctypes.cast(buffer, ctypes.POINTER(pointer)).contents
            if not security.ConvertSidToStringSidW(sid, ctypes.byref(text)):
                raise OSError(tr("The running desktop could not be identified."))
            return text.value, os.path.realpath(path.value)
        finally:
            if text:
                kernel.LocalFree(ctypes.cast(text, pointer))
            if token:
                kernel.CloseHandle(token)
            kernel.CloseHandle(process)
    if sys.platform == "darwin":
        import ctypes

        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        library.proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        library.proc_pidpath.restype = ctypes.c_int
        path = ctypes.create_string_buffer(4096)
        if library.proc_pidpath(pid, path, len(path)) <= 0:
            raise OSError(tr("The running desktop could not be identified."))
        code, owner = _run(["/bin/ps", "-p", str(pid), "-o", "uid="], timeout=10)
        if code or not owner.strip().isdigit():
            raise ValueError(tr("The running desktop could not be identified."))
        return int(owner.strip()), os.path.realpath(os.fsdecode(path.value))
    proc = "/proc/" + str(pid)
    return os.stat(proc).st_uid, os.path.realpath(os.readlink(proc + "/exe"))


class _UpdatePhaseChannel:
    """Bounded authenticated local messages; privileged peers accept only fixed phases."""

    def _send(self, phase, **details):
        """Send one numbered, HMAC-signed phase message to the peer.

        :param phase: the phase name the peer expects next.
        :param details: JSON-safe facts that travel with the phase.
        :raises ValueError: when the encoded message exceeds 8192 bytes.
        """
        import hmac

        self._sent += 1
        payload = {"sequence": self._sent, "direction": self._direction,
                   "phase": phase, "details": details}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        message = json.dumps({"payload": payload, "mac": hmac.new(
            self._secret, encoded, hashlib.sha256).hexdigest()}, separators=(",", ":")).encode() + b"\n"
        if len(message) > 8192:
            raise ValueError(tr("The authorized helper returned an invalid operation message."))
        self._connection.sendall(message)

    def _receive(self, expected, *, timeout=_FROZEN_PREPARATION_SECONDS, stopped=None):
        """Wait for the peer's next signed phase message and verify it.

        :param expected: the phase names accepted at this point.
        :param timeout: seconds to wait before giving up.
        :param stopped: optional callable that ends the wait early when true.
        :returns: the verified message payload.
        :raises ValueError: for a forged, out-of-order or oversized message.
        :raises RuntimeError: when the peer disconnects or the wait is stopped.
        :raises TimeoutError: when no message arrives before the deadline.
        """
        import hmac
        import socket

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self._connection is None:
                    self._connection, address = self._listener.accept()
                    self._connection.settimeout(0.2)
                    if address[0] != "127.0.0.1":
                        raise ValueError(tr("The authorized helper returned an invalid operation message."))
                if b"\n" not in self._buffer:
                    chunk = self._connection.recv(8193 - len(self._buffer))
                    if not chunk:
                        raise RuntimeError(tr("The authorized helper disconnected before the operation finished."))
                    self._buffer += chunk
                if len(self._buffer) > 8192:
                    raise ValueError(tr("The authorized helper returned an invalid operation message."))
                if b"\n" in self._buffer:
                    line, self._buffer = self._buffer.split(b"\n", 1)
                    message = json.loads(line)
                    payload = message["payload"]
                    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                    digest = hmac.new(self._secret, encoded, hashlib.sha256).hexdigest()
                    if (not hmac.compare_digest(message["mac"], digest)
                            or payload.get("sequence") != self._received + 1
                            or payload.get("direction") == self._direction
                            or payload.get("direction") not in {"desktop", "installer"}
                            or payload.get("phase") not in expected or not isinstance(payload.get("details"), dict)):
                        raise ValueError(tr("The authorized helper returned an invalid operation message."))
                    self._received += 1
                    return payload
            except socket.timeout:
                pass
            if stopped is not None and stopped():
                raise RuntimeError(tr("The authorized helper stopped before confirming its preparation."))
        raise TimeoutError(tr("The authorized helper did not confirm its preparation before the deadline."))

    def __enter__(self):
        """Use the channel as a context manager that closes its connection."""
        return self

    def __exit__(self, *_args):
        """Close the peer connection, if one was accepted or opened."""
        if self._connection is not None:
            self._connection.close()


class _UpdatePhaseServer(_UpdatePhaseChannel):
    """Keep authorization secrets in a private descriptor, never in process arguments."""

    def __init__(self, workdir):
        """Listen on a loopback port and write its port and secret to a private file.

        :param workdir: folder that receives the private descriptor file.
        """
        import socket

        self._connection, self._buffer = None, b""
        self._sent = self._received = 0
        self._direction, self._secret = "desktop", os.urandom(32)
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self._listener.settimeout(0.2)
        self.endpoint = None
        try:
            descriptor, self.endpoint = tempfile.mkstemp(prefix="phase-", dir=workdir)
            with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                if os.name == "nt":
                    _windows_pip_private_access(self.endpoint, private_read=True)
                stream.write(str(self._listener.getsockname()[1]) + ":" + self._secret.hex())
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            self._listener.close()
            if self.endpoint is not None:
                os.unlink(self.endpoint)
            raise

    def __exit__(self, *args):
        """Close the connection and listener and remove the descriptor file."""
        super().__exit__(*args)
        self._listener.close()
        os.unlink(self.endpoint)


class _UpdatePhaseClient(_UpdatePhaseChannel):
    """Read only the bound desktop user's regular private descriptor before staging."""

    def __init__(self, endpoint, desktop_pid, desktop_executable):
        """Connect to the desktop's phase server after checking who owns it.

        :param endpoint: path of the desktop's private descriptor file.
        :param desktop_pid: process ID of the running desktop.
        :param desktop_executable: executable the desktop must be running.
        """
        import re
        import socket

        owner, executable = _desktop_process_identity(desktop_pid)
        if os.path.normcase(executable) != os.path.normcase(os.path.realpath(desktop_executable)):
            raise ValueError(tr("The running desktop could not be identified."))
        before = os.lstat(endpoint)
        if (not stat.S_ISREG(before.st_mode) or before.st_size > 128
                or getattr(before, "st_file_attributes", 0) & 0x400):
            raise ValueError(tr("The operation authorization descriptor is not private."))
        if os.name == "nt":
            _windows_pip_private_access(endpoint, owner_sid=owner, private_read=True)
        elif before.st_uid != owner or stat.S_IMODE(before.st_mode) & 0o077:
            raise ValueError(tr("The operation authorization descriptor is not private."))
        descriptor = os.open(endpoint, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "r", encoding="ascii") as stream:
            actual = os.fstat(stream.fileno())
            if (not stat.S_ISREG(actual.st_mode) or actual.st_size > 128
                    or (actual.st_dev, actual.st_ino) != (before.st_dev, before.st_ino)
                    or os.name != "nt" and (actual.st_uid != owner or stat.S_IMODE(actual.st_mode) & 0o077)):
                raise ValueError(tr("The operation authorization descriptor is not private."))
            if os.name == "nt":
                _windows_pip_private_access(endpoint, owner_sid=owner, private_read=True)
            endpoint_text = stream.read(129)
        match = re.fullmatch(r"([1-9][0-9]{0,4}):([a-f0-9]{64})", endpoint_text)
        if match is None or int(match[1]) > 65535:
            raise ValueError(tr("The operation authorization descriptor is not private."))
        self._secret = bytes.fromhex(match[2])
        self._connection = socket.create_connection(("127.0.0.1", int(match[1])), timeout=10)
        self._connection.settimeout(0.2)
        self._buffer = b""
        self._sent = self._received = 0
        self._direction = "installer"


class _UpdateActivationUnconfirmed(RuntimeError):
    """Retain both payloads after losing the coordinator during desktop startup."""


def _native_update_ready(channel, version, desktop_pid):
    """Offer activation only after staging; require the old desktop's actual exit."""
    channel._send("prepared", version=version, desktop_pid=desktop_pid)
    decision = channel._receive({"activate", "cancel"}, timeout=_WAIT_SECONDS)
    if decision["phase"] != "activate":
        raise RuntimeError(tr("Update cancelled; the previous application was preserved."))
    if not _wait_for_exit(desktop_pid, timeout=1):
        raise RuntimeError(tr("The running application has not closed; activation was stopped."))


def _native_update_activated(channel, version):
    """Retain recovery until the unelevated new desktop confirms its main window."""
    try:
        channel._send("activated", version=version)
        decision = channel._receive({"confirmed", "failed", "retain"}, timeout=_WAIT_SECONDS)
    except Exception as error:
        raise _UpdateActivationUnconfirmed(tr("Desktop startup was not confirmed; both application versions were retained.")) from error
    if decision["phase"] == "retain":
        raise _UpdateActivationUnconfirmed(tr("Desktop startup was not confirmed; both application versions were retained."))
    if decision["phase"] != "confirmed":
        raise RuntimeError(tr("The new desktop could not open; restoring the previous application."))


def _coordinate_native_update(plan, handshake, operation, *, check, launch, wait=None):
    """Authorize staging while the GUI runs, then bind activation to Restart and startup."""
    with _UpdatePhaseServer(plan["workdir"]) as channel:
        completed = threading.Event()
        result = []

        def _run_authorized():
            """Retain the native worker's completion or error until its owned process finishes."""
            try:
                result.append(operation(channel.endpoint))
            except BaseException as error:
                result.append(error)
            finally:
                completed.set()

        def _stopped():
            """Check desktop cancellation before observing native worker completion."""
            handshake.check()
            return completed.is_set()

        worker = threading.Thread(target=_run_authorized, name="spaCR-authorized-update", daemon=True)
        handshake.check()
        worker.start()
        prepared_received = False
        activated = False
        desktop_stopped = False
        try:
            prepared = channel._receive({"prepared"}, stopped=_stopped)
            if prepared["details"] != {"version": plan["version"], "desktop_pid": plan["pid"]}:
                raise ValueError(tr("The authorized installer returned an invalid update message."))
            prepared_received = True
            check()
            handshake.ready()
            handshake.wait_for_shutdown(wait)
            handshake.check()
            check()
            channel._send("activate")
            changed = channel._receive({"activated"}, stopped=completed.is_set)
            if changed["details"] != {"version": plan["version"]}:
                raise ValueError(tr("The authorized installer returned an invalid update message."))
            activated = True
            try:
                desktop = launch()
            except _UpdateLaunchStillRunning:
                raise
            except BaseException:
                desktop_stopped = True
                raise
            channel._send("confirmed")
            completed.wait()
            if isinstance(result[0], BaseException):
                raise result[0]
            code, output = result[0]
            if code:
                raise RuntimeError(tr("The authorized installer could not complete the update: {error}", error=output))
            return output, desktop
        except BaseException:
            if channel._connection is not None:
                try:
                    channel._send("failed" if desktop_stopped else "retain" if activated else "cancel")
                except (OSError, ValueError):
                    pass
            if not completed.is_set():
                if channel._connection is not None:
                    channel._connection.close()
                channel._listener.close()
                completed.wait()
            if not prepared_received and result and isinstance(result[0], BaseException):
                raise result[0] from None
            raise


def _launched_desktop(process, pid):
    """Whether ``pid`` is the desktop this updater launched.

    On Windows a virtual environment's ``Scripts\\pythonw.exe`` is a launcher that runs
    the interpreter as its direct child, so the acknowledging process is that child;
    it is accepted only while the launcher itself still runs, so its id cannot be reused.
    """
    if pid == process.pid:
        return True
    if (os.name != "nt" or isinstance(pid, bool) or not isinstance(pid, int)
            or process.poll() is not None):
        return False
    children = _windows_children(process.pid)
    return bool(children) and pid in {int(child[0]) for child in children}


def _launch_updated_desktop(plan, argv, version, root, *, frozen=False, spawn=None):
    """Launch the exact updated executable and wait for its verified main window.

    Failure stops the owned child before a runtime transaction may roll back.
    An unkillable child retains the activated runtime and its recovery copy.
    """
    folder = tempfile.mkdtemp(prefix="launch-", dir=plan["workdir"])
    handshake = _UpdateLaunchHandshake(folder, os.urandom(32).hex())
    expected = {"version": str(version), "executable": os.path.abspath(argv[0]),
                "root": os.path.realpath(root), "frozen": bool(frozen)}
    handshake._write("launch-request.json", "requested", details=expected)
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.startswith("DYLD_") or key in {"PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH", "LD_LIBRARY_PATH_ORIG"}:
            environment.pop(key, None)
    environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    environment["SPACR_UPDATE_LAUNCH"] = json.dumps([folder, handshake.binding])
    process = (spawn(argv, environment, plan["workdir"]) if spawn is not None else
               _spawn_detached(argv, environment, plan["workdir"], return_process=True))
    try:
        while time.time() < handshake.expires:
            message = handshake._read("launch-result.json")
            if message is not None:
                if message.get("state") == "error":
                    raise RuntimeError(message["error"])
                if message.get("state") == "ready":
                    actual = message["details"]
                    if (actual.get("version") != expected["version"]
                            or actual.get("executable") != expected["executable"]
                            or not _inside(actual.get("module", ""), expected["root"])
                            or (process is not None and not _launched_desktop(process, actual.get("pid")))):
                        raise RuntimeError(tr("the new desktop acknowledgment does not match its update"))
                    if process is not None and process.poll() is not None:
                        raise RuntimeError(tr("the updated application exited before its main window was ready"))
                    return actual
            if process is not None and process.poll() is not None:
                raise RuntimeError(tr("the updated application exited before its main window was ready"))
            time.sleep(0.05)
        raise RuntimeError(tr("the updated application did not confirm its main window before the deadline"))
    except BaseException:
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            except Exception as error:
                raise _UpdateLaunchStillRunning(process.pid) from error
        raise



def _macos_online_update_record(records, *, system=None, strict=False):
    """The running macOS online environment's record, or None elsewhere.

    A damaged bootstrap is recognized too, so it fails without entering
    cleanup.
    """
    machine = system or _Machine()
    if machine.platform != "macos" or getattr(sys, "frozen", False):
        return None
    candidates = [record for record in records if record.kind == "installer"
                  and record.running and record.platform == "macos"
                  and record.layout in {"macos-runtime", "macos-online"}]
    if strict and len(candidates) != 1:
        raise ValueError("the running macOS online environment is ambiguous or unavailable")
    return candidates[0] if candidates else None


def _macos_online_version(value):
    """A release version as a four-part integer tuple; refuses other text."""
    import re

    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){2,3}", str(value)):
        raise ValueError("unsupported macOS online update version")
    parts = tuple(int(part) for part in str(value).split("."))
    return parts + (0,) * (4 - len(parts))


def _runtime_requested_version(value):
    """Validate canonical PEP440 text before constructing a private package command."""
    import re

    text = str(value).lower()
    pattern = (r"(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?"
               r"(?:\.post[0-9]+)?(?:\.dev[0-9]+)?"
               r"(?:\+[a-z0-9]+(?:[._-][a-z0-9]+)*)?")
    if re.fullmatch(pattern, text, re.IGNORECASE) is None:
        raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
    return text


def _runtime_version_compare(python, first, second, runner=None):
    """Compare releases numerically or use the private interpreter's PEP440 parser."""
    first, second = _runtime_requested_version(first), _runtime_requested_version(second)
    try:
        left, right = _macos_online_version(first), _macos_online_version(second)
    except ValueError:
        probe = ("import json, sys; from packaging.version import Version; "
                 "left, right = map(Version, json.loads(sys.argv[1])); "
                 "print((left > right) - (left < right))")
        command = [python, "-I", "-B", "-c", probe, json.dumps([first, second])]
        if runner is None or runner is _run:
            code, output = _run(command, timeout=30)
        else:
            code, output = runner(command)
        if code or output.strip() not in {"-1", "0", "1"}:
            raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
        return int(output.strip())
    return (left > right) - (left < right)


def _runtime_identity_command(python, version=None, *, prefix=None, minimum=False, previous_version=None):
    """Probe the imported application and its owning wheel in a fresh process.

    :param python: exact interpreter used by the desktop launcher.
    :param version: requested release; None checks identity without a version pin.
    :param prefix: expected Python environment; derived from python when omitted.
    :param minimum: permit a newer version for an existing non-downgrade adapter.
    :param previous_version: verified installed release that the replacement must not downgrade.
    :returns: isolated interpreter argv that emits a verified JSON receipt.
    """
    expected = json.dumps({"python": os.path.abspath(python), "version": version,
                           "prefix": prefix or os.path.dirname(os.path.dirname(python)),
                           "minimum": bool(minimum), "previous_version": previous_version})
    probe = '''import json, os, sys
from pathlib import Path
from importlib.metadata import distribution
from packaging.version import InvalidVersion, Version
import spacr
from spacr._version import __version__ as shipped
expected = json.loads(sys.argv[1])
actual = Path(spacr.__file__).resolve()
owners = []
for name in ('spacr', 'spacr-nightly'):
    try:
        dist = distribution(name)
        if Path(dist.locate_file('spacr/__init__.py')).resolve() == actual:
            owners.append(dist)
    except Exception:
        pass
assert len(owners) == 1, 'Imported spaCR is not owned by exactly one installed distribution'
dist = owners[0]
observed = Version(str(spacr.__version__))
assert observed == Version(str(shipped)) == Version(dist.version), 'Imported code and distribution versions differ'
normalize = lambda path: os.path.normcase(os.path.realpath(path))
assert normalize(sys.executable) == normalize(expected['python']), 'Launcher interpreter differs'
assert normalize(sys.prefix) == normalize(expected['prefix']), 'Launcher environment differs'
assert os.path.commonpath([normalize(expected['prefix']), normalize(actual)]) == normalize(expected['prefix']), 'Imported application is outside the launcher environment'
assert os.path.commonpath([normalize(expected['prefix']), normalize(dist.locate_file(''))]) == normalize(expected['prefix']), 'Application distribution is outside the launcher environment'
metadata_path = getattr(dist, '_path', None)
assert metadata_path is not None and os.path.commonpath([normalize(expected['prefix']), normalize(metadata_path)]) == normalize(expected['prefix']), 'Application distribution is outside the launcher environment'
if expected['previous_version'] is not None:
    try:
        previous = Version(expected['previous_version'])
    except InvalidVersion:
        print(json.dumps({'verified': False, 'reason': 'previous_runtime_version_invalid'}))
        sys.exit(2)
    if observed < previous:
        print(json.dumps({'verified': False, 'reason': 'runtime_downgrade_refused',
                          'previous_version': str(previous), 'staged_version': str(observed)}))
        sys.exit(2)
if expected['version'] is not None:
    requested = Version(str(expected['version']))
    assert observed >= requested if expected['minimum'] else observed == requested, 'Requested spaCR version differs'
print(json.dumps({'verified': True, 'version': str(observed), 'module': str(actual),
                  'python': sys.executable, 'prefix': sys.prefix,
                  'distribution': dist.metadata['Name'], 'requested_version': expected['version'],
                  'previous_version': expected['previous_version']}))
'''
    return [python, "-I", "-c", probe, expected]


def _retiring_runtime_identity(prefix):
    """Read one private installed wheel identity without importing its application dependencies."""
    if not os.path.lexists(prefix):
        return None
    prefix = os.path.abspath(prefix)
    python = os.path.join(prefix, "Scripts", "python.exe") if os.name == "nt" else os.path.join(prefix, "bin", "python")
    probe = '''import ast, json, os, sys
from pathlib import Path
from importlib.metadata import distributions
from importlib.util import find_spec
expected = os.path.normcase(os.path.realpath(sys.argv[1]))
assert os.path.normcase(os.path.realpath(sys.prefix)) == expected, 'Previous environment differs'
assert Path(sys.argv[1]).is_dir(), 'Previous environment is not a directory'
owners = [d for d in distributions() if d.metadata.get('Name', '').lower().replace('_', '-') in ('spacr', 'spacr-nightly')]
assert len(owners) == 1, 'Previous application distribution is ambiguous or missing'
dist = owners[0]
module = Path(dist.locate_file('spacr/__init__.py')).resolve()
assert module.is_file(), 'Previous application files are missing'
spec = find_spec('spacr')
assert spec is not None and spec.origin and Path(spec.origin).resolve() == module, 'Previous application import is shadowed'
assert os.path.commonpath([expected, os.path.normcase(str(module))]) == expected, 'Previous application is outside its environment'
assert os.path.commonpath([expected, os.path.normcase(os.path.realpath(dist.locate_file('')))]) == expected, 'Previous metadata is outside its environment'
metadata_path = getattr(dist, '_path', None)
assert metadata_path is not None and os.path.commonpath([expected, os.path.normcase(os.path.realpath(metadata_path))]) == expected, 'Previous metadata is outside its environment'
direct = json.loads(dist.read_text('direct_url.json') or '{}')
assert not direct.get('dir_info', {}).get('editable'), 'Previous application is editable'
version = dist.version
assert isinstance(version, str) and version.strip(), 'Previous application version is missing'
shipped = module.with_name('_version.py')
if shipped.exists():
    values = [node.value.value for node in ast.parse(shipped.read_text()).body
              if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == '__version__' for target in node.targets)
              and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)]
    assert values == [version], 'Previous code and distribution versions differ'
print(json.dumps({'verified': True, 'version': version, 'prefix': sys.prefix,
                  'module': str(module), 'distribution': dist.metadata['Name']}))
'''
    code, output = _run([python, "-I", "-B", "-c", probe, prefix], timeout=30)
    if code:
        raise ValueError("previous_runtime_identity_unverified")
    try:
        identity = json.loads(output)
        if identity.get("verified") is not True or not isinstance(identity.get("version"), str):
            raise ValueError("previous_runtime_identity_unverified")
    except (ValueError, AttributeError, TypeError) as error:
        raise ValueError("previous_runtime_identity_unverified") from error
    return identity


def _macos_online_commands(record, version, workdir):
    """The install, probe and relaunch commands for a macOS online update."""
    import re
    from importlib import import_module
    from pathlib import Path

    profiles = import_module("spacr.install_profile")
    read_profile = profiles.read_profile

    version = _runtime_requested_version(version)
    python = os.path.join(record.root, "venv", "bin", "python")
    uv = os.path.join(record.root, "bootstrap", "uv")
    if record.python is None or os.path.abspath(record.python) != os.path.abspath(python):
        raise ValueError("the running macOS environment does not match its private Python")
    for path in (uv, python):
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            raise ValueError(f"the existing macOS update executable is unavailable: {path}")
    if _inside(os.path.realpath(workdir), os.path.realpath(record.root)):
        raise ValueError("the updater folder must be outside the installed environment")
    explicit = re.search(r"(?:a|b|rc)[0-9]+|\.dev[0-9]+", version.partition("+")[0])
    package = "spacr==" + version if explicit or "+" in version else "spacr"
    install = [uv, "pip", "install", "--upgrade", "--python", python]
    profile_path = Path(record.root) / "install-profile.json"
    profile = read_profile(profile_path)
    if profile_path.exists() and (profile is None or not isinstance(profile.get("without_wgpu", False), bool)):
        raise ValueError(tr("The saved installation choices could not be read. Nothing was changed."))
    if profile:
        install.extend(["--torch-backend", profile["requested_backend"]])
        if profile.get("without_wgpu") is True:
            try:
                no_wgpu_override = profiles.no_wgpu_override
            except AttributeError as error:
                raise ValueError(tr("The saved choice to install without wgpu could not be preserved.")) from error
            override = no_wgpu_override(profile, Path(workdir) / "no-wgpu-override.txt")
            if override is None:
                raise ValueError(tr("The saved choice to install without wgpu could not be preserved."))
            install.extend(["--override", str(override)])
    install.append(package)
    probe = _runtime_identity_command(python, version, minimum=package == "spacr")
    relaunch = [python, "-I", "-m", "spacr.qt"]
    return install, probe, relaunch


def _runtime_exchange(first, second):
    """Atomically exchange two runtime directories, with no missing-prefix gap."""
    import ctypes

    library = ctypes.CDLL(None, use_errno=True)
    left, right = os.fsencode(first), os.fsencode(second)
    if sys.platform == "darwin":
        operation = library.renamex_np
        operation.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        arguments = (left, right, 2)
    elif sys.platform.startswith("linux"):
        operation = library.renameat2
        operation.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        arguments = (-100, left, -100, right, 2)
    else:
        raise OSError("atomic runtime exchange is not available on this platform")
    operation.restype = ctypes.c_int
    if operation(*arguments) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), first)


def _runtime_staging_allowed(prefix, machine):
    """Refuse shared or linked environments before invoking a package tool."""
    prefix = os.path.realpath(prefix)
    configuration = os.path.join(prefix, "pyvenv.cfg")
    if not os.path.isfile(configuration) or os.path.isdir(os.path.join(prefix, "conda-meta")):
        raise ValueError("this update requires a dedicated virtual environment")
    with open(configuration, encoding="utf-8") as handle:
        config = handle.read().lower().replace(" ", "")
    if "include-system-site-packages=true" in config:
        raise ValueError("shared system packages cannot be updated by runtime replacement")
    if any(_exists(path) and _inside(os.path.realpath(path), prefix)
           for path in _user_data(machine)):
        raise ValueError("configured user data inside the runtime must remain untouched")
    for directory, _folders, files in os.walk(prefix):
        for name in files:
            path = os.path.join(directory, name)
            if name.endswith(".egg-link"):
                raise ValueError("linked editable packages require their package manager")
            if name.endswith(".pth"):
                with open(path, encoding="utf-8", errors="replace") as handle:
                    if prefix in handle.read():
                        raise ValueError("absolute links to the active runtime cannot be staged safely")
            if name == "direct_url.json" and directory.endswith(".dist-info"):
                with open(path, encoding="utf-8") as handle:
                    if json.load(handle).get("dir_info", {}).get("editable"):
                        raise ValueError("editable packages require their source checkout")


def _runtime_recovery_inventory(prefix):
    """Bind runtime content and modes while permitting volatile Python cache directories."""
    return {path: entry for path, entry in _macos_tree(prefix, allow_external_links=True).items()
            if "__pycache__" not in path.split(os.sep)}


def _recover_runtime_update(transaction, pid, *, machine=None, runner=None, wait=None, swap=None):
    """Restore an unchanged retained POSIX runtime after the desktop has stopped."""
    machine = machine or _Machine()
    runner = runner or _run
    exchange = swap or _runtime_exchange
    root = os.path.abspath(transaction)
    details = os.lstat(root)
    if (os.path.realpath(root) != root or not stat.S_ISDIR(details.st_mode)
            or details.st_uid != os.geteuid() or stat.S_IMODE(details.st_mode) & 0o077):
        raise ValueError("recovery requires the original private transaction directory")
    descriptor = os.open(os.path.join(root, "transaction.json"), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, encoding="utf-8") as stream:
        details = os.fstat(stream.fileno())
        if (not stat.S_ISREG(details.st_mode) or details.st_uid != os.geteuid()
                or stat.S_IMODE(details.st_mode) & 0o077 or details.st_size > 64 * 1024 * 1024):
            raise ValueError("recovery requires its unchanged owner-only journal")
        state = json.load(stream)
    prefix, backup = state["prefix"], state["rollback_runtime"]
    if (state.get("schema") != 2 or state.get("family") != "posix-runtime"
            or state.get("state") not in {"committed", "rolling_back", "rolled_back", "launch_unconfirmed"}
            or os.path.dirname(prefix) != os.path.dirname(root)
            or backup != os.path.join(root, "runtime")
            or any(os.path.realpath(path) != path for path in (prefix, backup))
            or _inside(os.path.realpath(sys.prefix), prefix)
            or _inside(os.path.realpath(sys.prefix), backup)):
        raise ValueError("recovery requires a retained update and an outside interpreter")
    for directory in (prefix, backup, os.path.dirname(root)):
        details = os.lstat(directory)
        if (not stat.S_ISDIR(details.st_mode) or details.st_uid != os.geteuid()
                or stat.S_IMODE(details.st_mode) & 0o022):
            raise ValueError("recovery cannot replace shared runtime directories")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
        raise ValueError("recovery requires the previous desktop process identity")
    if state.get("state") == "launch_unconfirmed" and pid != state.get("launch_pid"):
        raise ValueError(tr("recovery requires the unconfirmed desktop process identity"))
    lock = os.path.join(os.path.dirname(prefix), "." + os.path.basename(prefix) + ".spacr-update-lock")
    _take_lease_dir(lock)
    try:
        if not (wait or _wait_for_exit)(pid):
            raise RuntimeError("spaCR did not close; both runtimes remain unchanged")
        for path in (prefix, backup):
            _runtime_staging_allowed(path, machine)
        left, right = os.stat(prefix), os.stat(backup)
        identities = ([left.st_dev, left.st_ino], [right.st_dev, right.st_ino])
        new_at_target = identities == (state["new_identity"], state["old_identity"])
        old_at_target = identities == (state["old_identity"], state["new_identity"])
        if ((state["state"] in {"committed", "launch_unconfirmed"} and not new_at_target)
                or (state["state"] == "rolled_back" and not old_at_target)
                or not (old_at_target or new_at_target)):
            raise ValueError("journaled runtime directories were replaced; preserving both")
        wanted = (state["new_inventory"], state["old_inventory"]) if new_at_target else (state["old_inventory"], state["new_inventory"])
        if (_runtime_recovery_inventory(prefix), _runtime_recovery_inventory(backup)) != wanted:
            raise ValueError("journaled runtime content changed; preserving both")
        old = backup if new_at_target else prefix
        probe = _runtime_identity_command(os.path.join(old, "bin", "python"), prefix=old)
        probe.insert(2, "-B")
        code, output = runner(probe)
        previous_identity = json.loads(output) if code == 0 else {}
        if code or previous_identity.get("verified") is not True:
            raise RuntimeError("retained runtime could not be verified; preserving both")
        if new_at_target:
            state["state"] = "rolling_back"
            _runtime_recovery_journal(root, state)
            exchange(prefix, backup)
            try:
                probe = _runtime_identity_command(os.path.join(prefix, "bin", "python"), prefix=prefix)
                probe.insert(2, "-B")
                code, output = runner(probe)
                restored = json.loads(output) if code == 0 else {}
                if (code or restored.get("verified") is not True
                        or restored.get("version") != previous_identity.get("version")
                        or restored.get("distribution") != previous_identity.get("distribution")):
                    raise RuntimeError("restored runtime could not be verified")
            except BaseException:
                exchange(prefix, backup)
                state["state"] = "committed"
                _runtime_recovery_journal(root, state)
                raise
        state["state"] = "rolled_back"
        _runtime_recovery_journal(root, state)
        return state
    finally:
        _release_lease_dir(lock)


def _runtime_recovery_journal(root, state):
    """Publish an owner-only complete recovery journal without a partial JSON window."""
    descriptor, temporary = tempfile.mkstemp(prefix="transaction-", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, os.path.join(root, "transaction.json"))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


#: Retained runtime transactions, by journal family: (folder prefix, schema, rollback child).
_RUNTIME_TRANSACTIONS = {
    "posix-runtime": (".spacr-runtime-update-", 2, "runtime"),
    "windows-private-pip": (".spacr-windows-update-", 1, "previous"),
    "windows-online-uv": (".spacr-windows-update-", 1, "previous"),
}

#: Journal states of an update that stopped before touching the live runtime: only its staged copy is left.
_ABANDONED_TRANSACTION_STATES = frozenset({"staging_failed", "cancelled"})

#: Journal states of an update still staging; abandoned once its owner is provably gone.
_STAGING_TRANSACTION_STATES = frozenset({"staging", "ready"})


def _transaction_owner(pid=None):
    """The owner a new transaction journal records: this process and its start stamp."""
    pid = os.getpid() if pid is None else pid
    return {"pid": pid, "started": _process_started(pid) or ""}


def _transaction_owner_gone(state):
    """Whether the process that staged a transaction is provably gone.

    The caller holds the runtime lease, which every staging update holds for its whole run, so a journal
    from a version that recorded no owner has none left. A recorded owner counts as gone only when its pid
    no longer exists or carries another start stamp of the same kind.
    """
    owner = state.get("owner")
    if owner is None:
        return True
    if not isinstance(owner, dict) or isinstance(owner.get("pid"), bool) or not isinstance(owner.get("pid"), int):
        return False
    pid, started = owner["pid"], str(owner.get("started") or "")
    if pid <= 0 or pid == os.getpid():
        return False
    if _process_gone(pid):
        return True
    now = _process_started(pid)
    return bool(now and started and now.partition(":")[0] == started.partition(":")[0] and now != started)


def _runtime_transaction_folders(parent):
    """Retained update transaction folders directly inside ``parent``, as plain directories only.

    A link, a reparse point, another user's folder or anything but a directory is never returned.
    """
    prefixes = tuple(prefix for prefix, _schema, _child in _RUNTIME_TRANSACTIONS.values())
    try:
        names = sorted(os.listdir(parent))
    except OSError:
        return []
    found = []
    for name in names:
        if not name.startswith(prefixes):
            continue
        path = os.path.join(parent, name)
        try:
            details = os.lstat(path)
        except OSError:
            continue
        if (stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode)
                or getattr(details, "st_file_attributes", 0) & 0x400
                or (os.name != "nt" and details.st_uid != os.geteuid())):
            continue
        found.append(path)
    return found


def _runtime_transaction_journal(path, prefix, family):
    """The journal of one retained transaction for ``prefix``, or None when it is not exactly that."""
    marker, schema, child = _RUNTIME_TRANSACTIONS[family]
    if not os.path.basename(path).startswith(marker):
        return None
    journal = os.path.join(path, "transaction.json")
    try:
        if os.path.islink(journal):
            return None
        descriptor = os.open(journal, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            details = os.fstat(stream.fileno())
            if (not stat.S_ISREG(details.st_mode) or details.st_size > 64 * 1024 * 1024
                    or (os.name != "nt" and details.st_uid != os.geteuid())):
                return None
            state = json.load(stream)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if (not isinstance(state, dict) or state.get("schema") != schema or state.get("family") != family
            or state.get("prefix") != prefix or state.get("rollback_runtime") != os.path.join(path, child)):
        return None
    return state, details.st_mtime_ns


def _folder_size(path):
    """Bytes under ``path`` counted from its entries themselves, never through a link."""
    total = 0
    for directory, folders, files in os.walk(path, followlinks=False):
        for name in folders + files:
            try:
                total += os.lstat(os.path.join(directory, name)).st_size
            except OSError:
                pass
    return total


def _prune_runtime_transactions(prefix, family, *, live=False, inventory=None):
    """Remove retained update copies of ``prefix`` except the newest committed one, and abandoned stages.

    Every update keeps the runtime it replaced as a rollback copy, about 3 GB, and nothing removed the older
    ones. Only a copy whose journal says "committed" is removed; one rolling back, awaiting launch
    confirmation, unknown or unreadable is kept, as is the newest committed one. A copy whose update
    failed or was cancelled before the live runtime was retired (``staging_failed``, ``cancelled``, about
    2.3 GB of staged runtime) is always removed. With ``live`` the committed copies
    are pruned only when the newest transaction of all is committed and its new runtime is the one now
    installed, by directory identity and content. The caller holds the runtime lease. Nothing outside the
    runtime's own folder is touched and no link is followed.

    :param prefix: the runtime directory the transactions replaced.
    :param family: "posix-runtime", "windows-private-pip" or "windows-online-uv".
    :param live: check the newest transaction against the installed runtime first.
    :param inventory: the content inventory of ``prefix``, computed when needed if omitted.
    :returns: ``(path, bytes)`` for each removed copy.
    """
    prefix = os.path.realpath(prefix)
    parent = os.path.dirname(prefix)
    journals = []
    for path in _runtime_transaction_folders(parent):
        found = _runtime_transaction_journal(path, prefix, family)
        if found is not None:
            journals.append((found[1], path, found[0]))
    journals.sort(key=lambda row: (row[0], row[1]))
    _marker, _schema, child = _RUNTIME_TRANSACTIONS[family]
    try:
        details = os.stat(prefix, follow_symlinks=False)
        live_identity = [details.st_dev, details.st_ino]
    except OSError:
        live_identity = None
    abandoned = [row for row in journals if row[2].get("state") in _ABANDONED_TRANSACTION_STATES
                 and not (family != "posix-runtime" and os.path.lexists(os.path.join(row[1], child)))]
    abandoned += [row for row in journals if row[2].get("state") in _STAGING_TRANSACTION_STATES
                  and live_identity is not None and row[2].get("old_identity") == live_identity
                  and not (family != "posix-runtime" and os.path.lexists(os.path.join(row[1], child)))
                  and _transaction_owner_gone(row[2])]
    committed = [row for row in journals if row[2].get("state") == "committed"]
    candidates = list(abandoned)
    if len(committed) >= 2 and _newest_copy_is_live(prefix, family, journals, live, inventory):
        candidates.extend(committed[:-1])
    return _remove_runtime_copies(parent, candidates)


def _newest_copy_is_live(prefix, family, journals, live, inventory):
    """Whether older committed copies may go: always at the start of an update; after a launch only when
    the newest transaction is committed and its new runtime is the installed one."""
    if live:
        newest_state = journals[-1][2]
        if newest_state.get("state") != "committed":
            return False
        try:
            details = os.stat(prefix, follow_symlinks=False)
        except OSError:
            return False
        if [details.st_dev, details.st_ino] != newest_state.get("new_identity"):
            return False
        if inventory is None:
            measure = _runtime_recovery_inventory if family == "posix-runtime" else _windows_pip_tree
            try:
                inventory = measure(prefix)
            except (OSError, ValueError):
                return False
        if inventory != newest_state.get("new_inventory"):
            return False
    return True


def _remove_runtime_copies(parent, candidates):
    """Remove each candidate transaction folder that is still a plain directory directly in ``parent``."""
    import logging

    log = logging.getLogger("spacr.install_cleanup")
    removed = []
    for _mtime, path, _state in candidates:
        try:
            details = os.lstat(path)
            if (stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode)
                    or os.path.dirname(os.path.realpath(path)) != parent):
                continue
            size = _folder_size(path)
            shutil.rmtree(path)
        except OSError as error:
            log.warning("Could not remove the retained update copy %s: %s", path, error)
            continue
        log.info("Removed the retained update copy %s (%.1f MB)", path, size / 1e6)
        removed.append((path, size))
    if removed:
        report = os.path.join(os.path.expanduser("~"), ".spacr", "logs", "update.log")
        try:
            os.makedirs(os.path.dirname(report), exist_ok=True)
            with open(report, "a", encoding="utf-8") as handle:
                for path, size in removed:
                    handle.write(f"Removed the retained update copy {path} ({size / 1e6:.1f} MB)\n")
        except OSError:
            pass
    return removed


def _prune_runtime_transactions_after_launch(prefix=None):
    """After a launch whose main window is up, keep only the newest retained copy of this runtime.

    Runs off the GUI thread. The runtime lease is taken for the duration; when an update holds it, nothing
    is done.

    :param prefix: the running runtime; this interpreter's own when omitted.
    :returns: ``(path, bytes)`` for each removed copy.
    """
    prefix = os.path.realpath(prefix or sys.prefix)
    parent = os.path.dirname(prefix)
    families = ("windows-private-pip", "windows-online-uv") if os.name == "nt" else ("posix-runtime",)
    folders = _runtime_transaction_folders(parent)
    families = [family for family in families
                if any(_runtime_transaction_journal(path, prefix, family) for path in folders)]
    if not families:
        return []
    lock = os.path.join(parent, "." + os.path.basename(prefix) + ".spacr-update-lock")
    try:
        _take_lease_dir(lock)
    except OSError:
        return []
    try:
        return [row for family in families for row in _prune_runtime_transactions(prefix, family, live=True)]
    finally:
        try:
            _release_lease_dir(lock)
        except OSError:
            pass


def _only_saved_wgpu_omission(code, output):
    """Recognize only uv's sole missing spaCR-to-wgpu dependency diagnostic.

    Unknown output, extra diagnostics and checker failures remain failures.
    Callers must separately require the explicit saved installation choice.
    """
    import re

    if code != 1:
        return False
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if lines and re.fullmatch(r"Using Python [0-9][^\r\n]* environment at: .+", lines[0]):
        lines.pop(0)
    if len(lines) != 3 or not re.fullmatch(r"Checked [0-9]+ packages? in [0-9.]+(?:µs|μs|us|ms|s|m)", lines[0]):
        return False
    if lines[1] != "Found 1 incompatibility":
        return False
    missing = re.fullmatch(r"The package `spacr` requires `([^`]+)`, but it's not installed", lines[2])
    if missing is None:
        return False
    specifier = r"(?:===|~=|==|!=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+"
    return re.fullmatch(r"wgpu\s*(?:\(?\s*" + specifier + r"(?:\s*,\s*" + specifier
                        + r")*\s*\)?)?(?:\s*;\s*[^;@\[\]]+)?", missing.group(1).strip(),
                        re.IGNORECASE) is not None


def _transactional_runtime_update(record, version, install, *, machine, runner=None,
                                  activate=None, before_activate=None, without_wgpu=False):
    """Update a private POSIX runtime through a verified atomic exchange.

    The previous runtime is retained as a rollback copy. All package operations
    target a copied sibling environment; the active prefix is only exchanged
    after the staged application's identity and dependency checks pass.
    """
    runner = runner or _run
    prefix = os.path.realpath(os.path.dirname(os.path.dirname(record.python)))
    _runtime_staging_allowed(prefix, machine)
    parent = os.path.dirname(prefix)
    lock = os.path.join(parent, "." + os.path.basename(prefix) + ".spacr-update-lock")
    _take_lease_dir(lock)
    transaction = None
    exchanged = False
    committed = False
    try:
        current = _retiring_runtime_identity(prefix)
        if current is None:
            raise ValueError("previous_runtime_identity_unverified")
        record = replace(record, version=current["version"])
        if _runtime_version_compare(record.python, record.version, version, runner) > 0:
            raise ValueError("The running spaCR is newer than the requested release. Nothing was changed.")
        _prune_runtime_transactions(prefix, "posix-runtime")
        transaction = tempfile.mkdtemp(prefix=".spacr-runtime-update-", dir=parent)
        staged = os.path.join(transaction, "runtime")
        journal = os.path.join(transaction, "transaction.json")
        state = {"schema": 2, "family": "posix-runtime", "prefix": prefix, "rollback_runtime": staged,
                 "owner": _transaction_owner(),
                 "old_identity": [os.stat(prefix).st_dev, os.stat(prefix).st_ino],
                 "old_inventory": _runtime_recovery_inventory(prefix),
                 "requested_version": str(version), "state": "staging"}

        def note(phase):
            """Persist the current update phase in the transaction journal using an atomic file replacement."""
            state["state"] = phase
            temporary = journal + ".next"
            with open(temporary, "w", encoding="utf-8") as handle:
                os.fchmod(handle.fileno(), 0o600)
                json.dump(state, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, journal)

        note("staging")
        shutil.copytree(prefix, staged, symlinks=True)
        for directory, folders, files in os.walk(staged):
            for name in folders + files:
                path = os.path.join(directory, name)
                if os.path.islink(path):
                    target = os.readlink(path)
                    if os.path.isabs(target) and _inside(target, prefix):
                        original = os.path.join(prefix, os.path.relpath(path, staged))
                        os.unlink(path)
                        os.symlink(os.path.relpath(target, os.path.dirname(original)), path)
        staged_python = os.path.join(staged, "bin", "python")
        command = list(install)
        pip_command = len(command) > 4 and command[1:4] == ["-I", "-m", "pip"]
        if pip_command:
            if os.path.abspath(command[0]) != os.path.abspath(record.python):
                raise ValueError("the package command does not target this private runtime")
            if command[4:6] != ["install", "--upgrade"]:
                raise ValueError("the package command does not target this private runtime")
            command[0] = staged_python
            check_command = [staged_python, "-I", "-m", "pip", "check"]
        else:
            position = command.index("--python") + 1
            if os.path.realpath(command[position]) != os.path.realpath(record.python):
                raise ValueError("the package command does not target this private runtime")
            command[position] = staged_python
            check_command = [command[0], "pip", "check", "--python", staged_python]
            if without_wgpu:
                check_command.extend(["--color", "never"])
        code, output = runner(command)
        if code:
            raise RuntimeError(f"The staged package upgrade failed with exit code {code}: {output}")
        code, checked = runner(check_command)
        if code and not (without_wgpu and not pip_command and _only_saved_wgpu_omission(code, checked)):
            raise RuntimeError(f"Staged package dependencies failed verification: {checked}")
        for directory, _folders, files in os.walk(staged, followlinks=False):
            for name in files:
                path = os.path.join(directory, name)
                previous = state["old_inventory"].get(os.path.relpath(path, staged), {})
                if not name.endswith(".py") or previous.get("kind") != "file" or os.path.islink(path):
                    continue
                details = os.stat(path, follow_symlinks=False)
                old = os.stat(os.path.join(prefix, os.path.relpath(path, staged)), follow_symlinks=False)
                if (details.st_size == previous["size"]
                        and int(details.st_mtime) == int(old.st_mtime)
                        and _macos_digest(path) != previous["sha256"]):
                    os.utime(path, ns=(details.st_atime_ns, (int(old.st_mtime) + 2) * 1_000_000_000),
                             follow_symlinks=False)
                    if int(os.stat(path).st_mtime) == int(old.st_mtime):
                        raise OSError(errno.EIO, os.strerror(errno.EIO), path)
        code, imported = runner([staged_python, "-I", "-c", "import spacr.qt"])
        if code:
            raise RuntimeError(f"The staged application import failed: {imported}")
        exact_version = pip_command or command[-1].startswith("spacr==")
        code, observed = runner(_runtime_identity_command(
            staged_python, version, minimum=not exact_version, previous_version=record.version))
        if code:
            raise RuntimeError("The staged application identity could not be verified: " + observed)
        try:
            identity = json.loads(observed)
        except (ValueError, TypeError) as error:
            raise RuntimeError("The staged application identity could not be verified: " + observed) from error
        if (identity.get("verified") is not True
                or identity.get("requested_version") != str(version)
                or _runtime_version_compare(staged_python, identity["version"], version, runner) < 0
                or (record.version and _runtime_version_compare(
                    staged_python, identity["version"], record.version, runner) < 0)):
            raise RuntimeError("The staged application identity could not be verified: " + observed)
        for name in os.listdir(os.path.join(staged, "bin")):
            script = os.path.join(staged, "bin", name)
            if os.path.islink(script) or not os.path.isfile(script):
                continue
            with open(script, "rb") as handle:
                first_line = handle.readline(8192)
                if not first_line.startswith(b"#!"):
                    continue
                header = first_line + handle.readline(8192) + handle.readline(8192)
                if os.fsencode(staged) not in header:
                    continue
                remainder = handle.read()
            with open(script, "wb") as handle:
                handle.write(header.replace(os.fsencode(staged), os.fsencode(prefix)) + remainder)
        state["new_identity"] = [os.stat(staged).st_dev, os.stat(staged).st_ino]
        note("ready")
        if before_activate is not None:
            before_activate()
            _runtime_staging_allowed(prefix, machine)
            if (_retiring_runtime_identity(prefix) != current
                    or [os.stat(prefix).st_dev, os.stat(prefix).st_ino] != state["old_identity"]
                    or _runtime_recovery_inventory(prefix) != state["old_inventory"]):
                raise RuntimeError("The desktop interpreter could not be identified; no relaunch was requested.")
        _runtime_exchange(prefix, staged)
        exchanged = True
        note("verifying")
        code, final = runner(_runtime_identity_command(record.python, version, minimum=True))
        if code:
            raise RuntimeError("The activated application identity could not be verified: " + final)
        try:
            final_identity = json.loads(final)
        except (ValueError, TypeError) as error:
            raise RuntimeError("The activated application identity could not be verified: " + final) from error
        if final_identity.get("verified") is not True or final_identity.get("version") != identity["version"]:
            raise RuntimeError("The activated application identity could not be verified: " + final)
        code, final_import = runner([record.python, "-I", "-c", "import spacr.qt"])
        if code:
            raise RuntimeError("The activated application import failed: " + final_import)
        state["verified_version"] = identity["version"]
        state["new_inventory"] = _runtime_recovery_inventory(prefix)
        note("committed")
        if activate is not None:
            activate(identity["version"])
        committed = True
        return identity["version"], output + "\nPrevious runtime retained at " + staged
    except _UpdateLaunchStillRunning as error:
        committed = True
        state["launch_pid"] = error.pid
        note("launch_unconfirmed")
        raise
    except BaseException:
        if exchanged:
            try:
                _runtime_exchange(prefix, staged)
                exchanged = False
                note("rolled_back")
            except BaseException as rollback_error:
                raise RuntimeError("Runtime restoration failed; previous runtime retained at " + staged) from rollback_error
        raise
    finally:
        if transaction and not committed and not exchanged:
            shutil.rmtree(transaction, ignore_errors=True)
        if not exchanged or committed:
            _release_lease_dir(lock)
        else:
            _retain_lease_dir(lock, "runtime exchange not committed")


def _run_macos_online_update(plan, machine, *, wait=None, run=None, spawn=None):
    """Carry out an approved macOS online update in its own environment.

    The actual paths are rechecked after shutdown; uv or Python found on PATH
    is never substituted.
    """
    records = [_record_from_json(data) for data in plan["records"]]
    record = _macos_online_update_record(records, system=machine, strict=True)
    if record is None or plan.get("frozen_application"):
        raise ValueError("this update requires a running non-frozen macOS online environment")
    install, probe, relaunch = _macos_online_commands(record, plan["version"], plan["workdir"])
    if plan.get("fetch") is not None or plan.get("install") != install or plan.get("relaunch") != relaunch:
        raise ValueError("the macOS online update commands differ from the approved environment")
    current = _retiring_runtime_identity(os.path.dirname(os.path.dirname(record.python)))
    if current is None:
        raise ValueError("previous_runtime_identity_unverified")
    record = replace(record, version=current["version"])
    if _runtime_version_compare(record.python, record.version, plan["version"], run) > 0:
        raise ValueError("The running spaCR is newer than the requested release. Nothing was changed.")
    handshake = _FrozenUpdateHandshake(plan)
    handshake.check()

    def prepared():
        """Offer Restart only after the replacement passes its staged checks."""
        handshake.ready()
        handshake.wait_for_shutdown(wait)
        if _macos_online_commands(record, plan["version"], plan["workdir"]) != (install, probe, relaunch):
            raise ValueError(tr("The saved installation choices changed while preparing the update. Nothing was changed."))

    return _transactional_runtime_update(
        record, plan["version"], install, machine=machine, runner=run,
        before_activate=prepared, without_wgpu="--override" in install,
        activate=lambda observed: _launch_updated_desktop(
            plan, relaunch, observed, os.path.dirname(os.path.dirname(record.python)), spawn=spawn))


@contextmanager
def _debian_frontend_lock(path="/var/lib/dpkg/lock-frontend"):
    """Hold Debian's frontend lock without waiting behind another installer.

    :param path: System lock, or an isolated test database's existing lock.
    :yields: Nothing while the frontend lock is held. Child dpkg commands must
        receive DPKG_FRONTEND_LOCKED=1 from the privileged runner.
    :raises OSError: The lock is unavailable or its file identity is unsafe.
    """
    import fcntl

    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        details = os.fstat(descriptor)
        if (not stat.S_ISREG(details.st_mode) or details.st_uid != os.geteuid()
                or stat.S_IMODE(details.st_mode) & 0o022):
            raise OSError(tr("unsafe Debian frontend lock"))
        fcntl.lockf(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def _debian_prepare_archive(version, stage, *, fetch=None, published_digest=None):
    """Download a release into private staging with a mandatory published hash.

    :param version: Exact canonical release version to stage.
    :param stage: Existing owner-only directory; the privileged caller owns it.
    :param fetch: Optional download implementation used by offline fixtures.
    :param published_digest: Optional release checksum reader for fixtures.
    :returns: Canonical archive path and its authenticated SHA256.
    :raises ValueError: Staging or published identity cannot be verified.
    """
    version = _runtime_requested_version(version)
    stage = os.path.abspath(stage)
    details = os.lstat(stage)
    if (not stat.S_ISDIR(details.st_mode) or os.path.realpath(stage) != stage
            or details.st_uid != os.geteuid() or stat.S_IMODE(details.st_mode) & 0o077):
        raise ValueError(tr("Debian archives require private trusted staging"))
    name = "spacr_" + version + "_amd64.deb"
    url = _RELEASE_DOWNLOAD + "/v" + version + "/" + name
    expected = (published_digest or _published_digest)(url)
    if (not isinstance(expected, str) or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)):
        raise ValueError(tr("a published SHA256 is required for Debian replacement and rollback"))
    target = os.path.join(stage, name)
    if os.path.lexists(target):
        raise ValueError(tr("Debian staging must not reuse an existing archive"))
    (fetch or _download)(url, target)
    descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as archive:
        actual = os.fstat(archive.fileno())
        if (not stat.S_ISREG(actual.st_mode) or actual.st_uid != os.geteuid()
                or actual.st_nlink != 1 or stat.S_IMODE(actual.st_mode) & 0o022):
            raise ValueError(tr("unsafe Debian archive staging identity"))
        digest = hashlib.sha256()
        for chunk in iter(lambda: archive.read(1 << 20), b""):
            digest.update(chunk)
        if not actual.st_size or digest.hexdigest() != expected:
            raise ValueError(tr("Debian archive does not match its published SHA256"))
    return {"path": target, "sha256": expected, "version": version}


def _debian_package_metadata(archive, version, architecture, *, run=None):
    """Require the exact package identity before a Debian transaction.

    :param archive: Already authenticated archive in private staging.
    :param version: Exact release version expected in the archive.
    :param architecture: Native Debian architecture, currently amd64.
    :param run: Optional command runner used by isolated acceptance fixtures.
    :returns: The verified package identity.
    :raises ValueError: An unsupported or mismatched package was supplied.
    """
    version = _runtime_requested_version(version)
    if architecture != "amd64":
        raise ValueError(tr("unsupported Debian package architecture"))
    code, output = (run or _run)([
        "/usr/bin/dpkg-deb", "--field", os.path.abspath(archive),
        "Package", "Version", "Architecture"])
    fields = {}
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if not separator or key in fields:
            raise ValueError(tr("invalid Debian package metadata"))
        fields[key] = value.strip()
    expected = {"Package": "spacr", "Version": version,
                "Architecture": architecture}
    if code or fields != expected:
        raise ValueError(tr("Debian archive does not match the requested spaCR release"))
    return expected


def _debian_installed_state(*, run=None):
    """Read package state without treating a partial unpack as a healthy install.

    :param run: Optional runner targeting an isolated dpkg database.
    :returns: Status, version and architecture of the spaCR package.
    :raises ValueError: Package identity cannot be read unambiguously.
    """
    code, output = (run or _run)([
        "/usr/bin/dpkg-query", "--show",
        "--showformat=${Status}\\n${Version}\\n${Architecture}\\n", "spacr"])
    fields = output.splitlines()
    if code or len(fields) != 3 or fields[2] != "amd64":
        raise ValueError(tr("spaCR Debian package state is ambiguous"))
    return {"status": fields[0], "version": _runtime_requested_version(fields[1]),
            "architecture": fields[2]}


def _debian_installed_version(*, run=None):
    """Read dpkg's complete installed state rather than a cached release label.

    :param run: Optional command runner for a private package-database fixture.
    :returns: Exact version of a fully configured spaCR package.
    :raises ValueError: The package is missing, unpacked or requires repair.
    """
    state = _debian_installed_state(run=run)
    if state["status"] != "install ok installed":
        raise ValueError(tr("spaCR is not a configured native Debian package"))
    return state["version"]


def _debian_package_transaction(new_archive, rollback_archive, previous_version,
                                requested_version, *, verify, run=None, journal=None, after_activate=None):
    """Install a prepared package and restore its prepared predecessor on failure.

    The privileged caller must first authenticate both archives and retain them
    in root-owned private staging and hold the dpkg frontend lock throughout.
    The verification callback must execute the application as the authenticated
    desktop user, never as root. This function does not authorize privilege,
    accept a user plan, download files, retire other copies or launch the GUI.

    :param new_archive: Authenticated requested package in trusted staging.
    :param rollback_archive: Authenticated package for the installed version.
    :param previous_version: Exact installed version observed before preparation.
    :param requested_version: Exact requested replacement version.
    :param verify: Required runtime verification callback taking a version.
    :param run: Optional command runner for an isolated package database.
    :param journal: Optional durable state recorder; initial recording must
        succeed before mutation. Later recording failures trigger rollback.
    :returns: Committed state only after package and runtime verification.
    :raises RuntimeError: Replacement failed; reports whether rollback verified.
    """
    runner = run or _run
    previous_version = _runtime_requested_version(previous_version)
    requested_version = _runtime_requested_version(requested_version)
    _debian_package_metadata(new_archive, requested_version, "amd64", run=runner)
    _debian_package_metadata(rollback_archive, previous_version, "amd64", run=runner)
    if _debian_installed_version(run=runner) != previous_version:
        raise ValueError(tr("the installed Debian package changed during preparation"))
    code, _output = runner(["/usr/bin/dpkg", "--compare-versions",
                            requested_version, "gt", previous_version])
    if code:
        raise ValueError(tr("the requested Debian release is not newer than the installed version"))
    if journal is not None:
        journal("installing")
    try:
        code, output = runner(["/usr/bin/dpkg", "--install", os.path.abspath(new_archive)])
        if code:
            raise RuntimeError(tr("Debian replacement failed: {error}", error=output))
        if _debian_installed_version(run=runner) != requested_version or verify(requested_version) is not True:
            raise RuntimeError(tr("Debian replacement identity could not be verified"))
        if after_activate is not None:
            after_activate()
        if journal is not None:
            journal("committed")
    except _UpdateActivationUnconfirmed:
        if journal is not None:
            journal("desktop-unconfirmed")
        raise
    except BaseException as failure:
        try:
            state = _debian_installed_state(run=runner)
            if state["version"] not in {previous_version, requested_version}:
                raise RuntimeError(tr("a different Debian package replaced this transaction"))
            code, output = runner(["/usr/bin/dpkg", "--install", os.path.abspath(rollback_archive)])
            if code:
                raise RuntimeError(output)
            if _debian_installed_version(run=runner) != previous_version or verify(previous_version) is not True:
                raise RuntimeError(tr("restored Debian identity could not be verified"))
        except BaseException as rollback_failure:
            if journal is not None:
                try:
                    journal("recovery-required")
                except Exception:
                    pass
            raise RuntimeError(tr("Debian update and rollback failed; both archives are retained for recovery")) from rollback_failure
        if journal is not None:
            try:
                journal("restored")
            except Exception:
                pass
        raise RuntimeError(tr("Debian update failed; the previous package and runtime were restored")) from failure
    return {"state": "committed", "version": requested_version,
            "previous_version": previous_version,
            "rollback_archive": os.path.abspath(rollback_archive)}


def _debian_root_runner(command):
    """Run fixed package-manager commands with a sanitized privileged environment.

    :param command: Argument vector built by the Debian adapter, never a user plan.
    :returns: Exit code and combined command output.
    :raises ValueError: Caller is not root or the executable is not approved.
    """
    if os.geteuid() != 0 or command[0] not in {
            "/usr/bin/dpkg", "/usr/bin/dpkg-deb", "/usr/bin/dpkg-query"}:
        raise ValueError(tr("Debian package commands require the trusted privileged adapter"))
    environment = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8",
                   "LC_ALL": "C.UTF-8", "DEBIAN_FRONTEND": "noninteractive",
                   "DPKG_FRONTEND_LOCKED": "1", "HOME": "/root"}
    result = subprocess.run(command, env=environment, cwd="/", text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, timeout=1800)
    return result.returncode, result.stdout


def _debian_verify_desktop(version, desktop_uid, *, root="/opt/spacr", invoke=None):
    """Verify the installed frozen version after dropping all root credentials.

    :param version: Exact expected application version.
    :param desktop_uid: Authenticated non-root desktop account.
    :param root: Fixed bundle root, overridable only by isolated fixtures.
    :param invoke: Optional subprocess runner for credential-bound fixtures.
    :returns: Whether the installed executable reports exactly the expected version.
    :raises ValueError: Desktop identity or installed executable is untrusted.
    """
    import pwd

    if isinstance(desktop_uid, bool) or not isinstance(desktop_uid, int) or desktop_uid <= 0:
        raise ValueError(tr("Debian verification requires the authenticated desktop user"))
    account = pwd.getpwuid(desktop_uid)
    root = os.path.abspath(root)
    executable = os.path.join(root, "spacr")
    if os.path.realpath(root) != root:
        raise ValueError(tr("the Debian application root must not be redirected"))
    for path in (root, executable):
        details = os.lstat(path)
        expected_type = stat.S_ISDIR if path == root else stat.S_ISREG
        if (not expected_type(details.st_mode) or details.st_uid != 0
                or stat.S_IMODE(details.st_mode) & 0o022):
            raise ValueError(tr("the Debian application is not a trusted system package"))
    profile = tempfile.mkdtemp(prefix="spacr-debian-verify-")
    try:
        os.chown(profile, desktop_uid, account.pw_gid)
        environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                       "HOME": profile, "USER": account.pw_name, "LOGNAME": account.pw_name,
                       "XDG_CONFIG_HOME": os.path.join(profile, ".config"),
                       "XDG_CACHE_HOME": os.path.join(profile, ".cache"),
                       "XDG_STATE_HOME": os.path.join(profile, ".state"),
                       "CUDA_VISIBLE_DEVICES": "", "QT_QPA_PLATFORM": "offscreen",
                       "PYINSTALLER_RESET_ENVIRONMENT": "1"}
        result = (invoke or subprocess.run)(
            [executable, "--version"], env=environment, cwd=profile,
            user=desktop_uid, group=account.pw_gid, extra_groups=[],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=180)
        lines = result.stdout.strip().splitlines()
        return result.returncode == 0 and bool(lines) and lines[-1] == version
    finally:
        shutil.rmtree(profile)


def _debian_update_commands(records, version, machine, *, helper=False):
    """Bind the frozen Debian GUI to one installed package and root helper.

    :param records: Installation discovery results included in the approved plan.
    :param version: Exact requested replacement release.
    :param machine: Actual Linux process and installation description.
    :param helper: The already extracted unprivileged helper validates the package
        while running outside its root; GUI preparation requires the real GUI binary.
    :returns: Fixed graphical-authorization command and desktop relaunch command.
    :raises ValueError: This is not the supported, protocol-marked Debian bundle.
    """
    candidates = [record for record in records if record.running]
    if (machine.platform != "linux" or len(records) != 1 or len(candidates) != 1
            or candidates[0].layout != "linux-deb" or candidates[0].kind != "installer"
            or candidates[0].root != "/opt/spacr" or candidates[0].registrations != ("deb:spacr",)
            or (not helper and os.path.realpath(machine.executable) != "/opt/spacr/spacr")
            or not getattr(sys, "frozen", False) or os.getuid() == 0):
        raise ValueError(tr("this Debian installation has no verified graphical replacement adapter"))
    record = candidates[0]
    if any(_inside(os.path.realpath(path), record.root) for path in _user_data(machine)):
        raise ValueError(tr("configured user data inside the Debian application must remain untouched"))
    helper = _debian_check_package_owner(record.version, run=_run)
    marker = "/opt/spacr/update-protocol.json"
    details = os.lstat(marker)
    if (not stat.S_ISREG(details.st_mode) or details.st_uid != 0
            or stat.S_IMODE(details.st_mode) & 0o022 or details.st_size > 1024):
        raise ValueError(tr("the Debian update protocol marker is not trusted"))
    with open(marker, encoding="utf-8") as stream:
        protocol = json.load(stream)
    if protocol != {"schema": 1, "adapter": "linux-deb-v1"}:
        raise ValueError(tr("the installed Debian updater protocol is unsupported"))
    version = _runtime_requested_version(version)
    code, _output = _run(["/usr/bin/dpkg", "--compare-versions", version, "gt", record.version])
    if code:
        raise ValueError(tr("the requested Debian release is not newer than the installed version"))
    if not os.path.isfile("/usr/bin/pkexec") or not os.access("/usr/bin/pkexec", os.X_OK):
        raise ValueError(tr("graphical administrator authorization is unavailable"))
    return (["/usr/bin/pkexec", helper, "prepare-debian-root", "--version", version,
             "--previous-version", record.version, "--desktop-uid", str(os.getuid())],
            ["/opt/spacr/spacr"])


def _run_debian_update(plan, machine, *, wait=None, run=None, spawn=None, published_digest=None):
    """Authorize and verify a staged package before offering Restart and activation.

    :param plan: Immutable package/desktop identity bound to a readiness lease.
    :param machine: Actual process and installation description.
    :param wait: Optional shutdown observer for isolated fixtures.
    :param run: Optional unprivileged command runner for graphical authorization.
    :param spawn: Optional unprivileged desktop launcher.
    :param published_digest: Optional checksum reader for offline fixtures.
    :returns: Requested version and privileged transaction output.
    :raises ValueError: Plan commands or package identity changed.
    """
    records = [_record_from_json(data) for data in plan["records"]]
    workdir = os.path.realpath(plan["workdir"])
    helper_binary = os.path.join(workdir, "spacr-update-helper")
    if (os.path.realpath(sys.executable) != helper_binary
            or plan.get("command") != [helper_binary, "run-plan", os.path.join(workdir, "plan.json")]):
        raise ValueError("Debian preparation must run in its extracted private helper")
    command, relaunch = _debian_update_commands(records, plan["version"], machine, helper=True)
    if (not plan.get("frozen_application") or plan.get("ticked") or plan.get("fetch") is not None
            or plan.get("install") != command or plan.get("relaunch") != relaunch):
        raise ValueError("the Debian update plan differs from the installed package")
    previous = next(record.version for record in records if record.running)
    for version in (plan["version"], previous):
        url = _RELEASE_DOWNLOAD + "/v" + version + "/spacr_" + version + "_amd64.deb"
        digest = (published_digest or _published_digest)(url)
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError("both replacement and rollback releases require published checksums")
    handshake = _FrozenUpdateHandshake(plan)

    def _check_current():
        """Bind activation to the same installed Debian package and trusted helper."""
        if _debian_update_commands(records, plan["version"], machine, helper=True) != (command, relaunch):
            raise ValueError("the Debian update identity changed after shutdown")

    output, _desktop = _coordinate_native_update(plan, handshake,
        lambda endpoint: (run or (lambda argv: _run(argv, timeout=10800)))(command + [
            "--phase-channel", endpoint, "--desktop-pid", str(plan["pid"])]),
        check=_check_current, wait=wait,
        launch=lambda: _launch_updated_desktop(plan, relaunch, plan["version"], "/opt/spacr", frozen=True, spawn=spawn))
    try:
        receipt = json.loads(output.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise RuntimeError("the privileged Debian helper returned no verified receipt") from error
    if receipt.get("state") != "committed" or receipt.get("version") != plan["version"]:
        raise RuntimeError("the privileged Debian replacement was not verified")
    handshake._write("debian-transaction.json", "committed", details=receipt)
    return plan["version"], output


def _debian_authenticated_user(desktop_uid):
    """Bind privileged work to pkexec's authenticated non-root desktop account.

    :param desktop_uid: Account requested by the unprivileged update adapter.
    :returns: Validated numeric account identifier.
    :raises ValueError: Privilege or the authenticated account does not match.
    """
    import pwd

    if (os.geteuid() != 0 or isinstance(desktop_uid, bool)
            or not isinstance(desktop_uid, int) or desktop_uid <= 0
            or os.environ.get("PKEXEC_UID") != str(desktop_uid)):
        raise ValueError(tr("Debian updates require graphical authorization from the desktop user"))
    pwd.getpwuid(desktop_uid)
    return desktop_uid


def _debian_check_package_owner(previous_version, *, run=None):
    """Bind both executable and privileged helper to the configured spaCR package.

    :param previous_version: Exact package version observed before authorization.
    :param run: Optional command runner used by isolated fixtures.
    :returns: Trusted installed helper path.
    :raises ValueError: Ownership, version or canonical system layout changed.
    """
    runner = run or _debian_root_runner
    helper = "/opt/spacr/_internal/spacr-update-helper"
    if _debian_installed_version(run=runner) != previous_version:
        raise ValueError(tr("the installed Debian package changed before authorization"))
    for path in ("/opt", "/opt/spacr", "/opt/spacr/_internal", "/opt/spacr/spacr", helper):
        details = os.lstat(path)
        if (os.path.realpath(path) != path or details.st_uid != 0
                or stat.S_IMODE(details.st_mode) & 0o022
                or (not stat.S_ISREG(details.st_mode) and not stat.S_ISDIR(details.st_mode))):
            raise ValueError(tr("the Debian package layout is not a trusted system installation"))
    for path in ("/opt/spacr/spacr", helper):
        code, output = runner(["/usr/bin/dpkg-query", "--search", path])
        if code or output.strip() != "spacr: " + path:
            raise ValueError(tr("the Debian executable is not exclusively owned by spaCR"))
    return helper


def _debian_apply_root_transaction(stage, version, previous_version, desktop_uid, *, phase_channel=None, desktop_pid=None):
    """Apply an authenticated update from the root-owned outside-runtime helper.

    :param stage: Private directory containing this independently copied helper.
    :param version: Exact requested release.
    :param previous_version: Exact rollback release.
    :param desktop_uid: pkexec-authenticated desktop account, never root.
    :returns: Verified committed receipt retaining both package archives.
    :raises ValueError: Helper identity, privilege or package ownership is unsafe.
    """
    _debian_authenticated_user(desktop_uid)
    stage = os.path.abspath(stage)
    details = os.lstat(stage)
    if (os.path.dirname(stage) != "/var/lib/spacr-updates" or os.path.realpath(stage) != stage
            or not stat.S_ISDIR(details.st_mode) or details.st_uid != 0
            or stat.S_IMODE(details.st_mode) & 0o077
            or os.path.realpath(sys.executable) != os.path.join(stage, "spacr-update-helper")):
        raise ValueError(tr("Debian replacement must run from its trusted outside-runtime helper"))
    parent_details = os.lstat(os.path.dirname(stage))
    helper_details = os.lstat(sys.executable)
    if (not stat.S_ISDIR(parent_details.st_mode) or parent_details.st_uid != 0
            or stat.S_IMODE(parent_details.st_mode) & 0o077
            or not stat.S_ISREG(helper_details.st_mode) or helper_details.st_uid != 0
            or stat.S_IMODE(helper_details.st_mode) & 0o022):
        raise ValueError(tr("Debian privileged staging identity changed"))
    channel = None
    staged = None
    if phase_channel is not None:
        if _desktop_process_identity(desktop_pid)[0] != desktop_uid:
            raise ValueError(tr("The running desktop could not be identified."))
        channel = _UpdatePhaseClient(phase_channel, desktop_pid, "/opt/spacr/spacr")
    try:
        base = {"schema": 1, "adapter": "linux-deb-v1", "version": version,
                "previous_version": previous_version, "desktop_uid": desktop_uid}
        _debian_write_journal(stage, dict(base, state="preparing"))
        new_archive = _debian_prepare_archive(version, stage)
        rollback = _debian_prepare_archive(previous_version, stage)
        base.update(replacement=new_archive, rollback=rollback)
        if channel is not None:
            _debian_package_metadata(rollback["path"], previous_version, "amd64", run=_debian_root_runner)
            staged = _debian_stage_payload(new_archive["path"], version, desktop_uid)
            base["staged_payload"] = staged
        _debian_write_journal(stage, dict(base, state="prepared"))
        if channel is not None:
            _native_update_ready(channel, version, desktop_pid)
        with _debian_frontend_lock():
            _debian_check_package_owner(previous_version)
            return _debian_package_transaction(
                new_archive["path"], rollback["path"], previous_version, version,
                verify=lambda expected: _debian_verify_desktop(expected, desktop_uid),
                run=_debian_root_runner,
                after_activate=(lambda: _native_update_activated(channel, version)) if channel is not None else None,
                journal=lambda state: _debian_write_journal(stage, dict(base, state=state)))
    finally:
        if channel is not None:
            channel.__exit__()
        if staged is not None:
            shutil.rmtree(staged)


def _debian_stage_payload(archive, version, desktop_uid, *, run=None, verify=None):
    """Unpack and run the new bundle unprivileged while the installed desktop stays intact."""
    runner = run or _debian_root_runner
    _debian_package_metadata(archive, version, "amd64", run=runner)
    parent = os.lstat("/opt")
    if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != 0 or stat.S_IMODE(parent.st_mode) & 0o022):
        raise ValueError(tr("The Debian application is not a trusted system package."))
    staged = tempfile.mkdtemp(prefix=".spacr-staged-", dir="/opt")
    try:
        os.chmod(staged, 0o755)
        code, output = runner(["/usr/bin/dpkg-deb", "--extract", os.path.abspath(archive), staged])
        if code or not (verify or _debian_verify_desktop)(version, desktop_uid,
                root=os.path.join(staged, "opt", "spacr")):
            raise RuntimeError(tr("The staged Debian application could not be verified: {error}", error=output))
        return staged
    except BaseException:
        shutil.rmtree(staged)
        raise


def _debian_write_journal(stage, data):
    """Atomically retain an fsynced transaction record beside both package archives.

    :param stage: Existing private directory owned by the privileged caller.
    :param data: Serializable protocol identity and current transition state.
    :raises OSError: Recording failed; an initial failure must prevent mutation.
    """
    descriptor, temporary = tempfile.mkstemp(prefix=".journal-", dir=stage)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, os.path.join(stage, "journal.json"))
        directory = os.open(stage, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def _debian_prepare_root_transaction(version, previous_version, desktop_uid, *, phase_channel=None, desktop_pid=None):
    """Copy only the trusted installed helper before replacing its package.

    :param version: Exact requested release version.
    :param previous_version: Exact installed release version to retain for rollback.
    :param desktop_uid: pkexec-authenticated desktop account.
    :returns: Exit code and bounded child output from the copied root helper.
    :raises ValueError: Privilege, installed helper or system staging is unsafe.
    """
    _debian_authenticated_user(desktop_uid)
    version = _runtime_requested_version(version)
    previous_version = _runtime_requested_version(previous_version)
    parent = "/var/lib/spacr-updates"
    if not os.path.lexists(parent):
        os.mkdir(parent, 0o700)
    details = os.lstat(parent)
    if (os.path.realpath(parent) != parent or not stat.S_ISDIR(details.st_mode)
            or details.st_uid != 0 or stat.S_IMODE(details.st_mode) & 0o077):
        raise ValueError(tr("Debian updater staging is not a private system directory"))
    with _debian_frontend_lock():
        source = _debian_check_package_owner(previous_version)
        if os.path.realpath(sys.executable) != source:
            raise ValueError(tr("only the installed Debian helper may prepare privileged replacement"))
        stage = tempfile.mkdtemp(prefix="transaction-", dir=parent)
        helper = os.path.join(stage, "spacr-update-helper")
        with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW), "rb") as original:
            destination_fd = os.open(helper, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o700)
            with os.fdopen(destination_fd, "wb") as copied:
                shutil.copyfileobj(original, copied)
                copied.flush()
                os.fsync(copied.fileno())
    environment = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8",
                   "HOME": "/root", "PKEXEC_UID": str(desktop_uid),
                   "PYINSTALLER_RESET_ENVIRONMENT": "1"}
    command = [helper, "apply-debian-root", "--stage", stage, "--version", version,
        "--previous-version", previous_version, "--desktop-uid", str(desktop_uid)]
    if phase_channel is not None:
        command.extend(["--phase-channel", phase_channel, "--desktop-pid", str(desktop_pid)])
    result = subprocess.run(command,
        env=environment, cwd="/", text=True, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=7200)
    return result.returncode, result.stdout


def _windows_pip_acl(path):
    """Read local fixed-volume ownership and ordinary DACL entries through native Windows APIs."""
    import ctypes

    if os.name != "nt":
        raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
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
    if absolute.startswith("\\\\") or drive_type(os.path.splitdrive(absolute)[0] + os.sep) != 3:
        raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
    root = ctypes.create_unicode_buffer(32768)
    if not volume(absolute, root, len(root)):
        raise OSError(ctypes.get_last_error(), os.strerror(errno.EACCES), path)
    if root.value.startswith("\\\\") or drive_type(root.value) != 3:
        raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
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
    code = read_security(os.path.abspath(path), 1, 5, ctypes.byref(owner), None,
                         ctypes.byref(dacl), None, ctypes.byref(descriptor))
    if code:
        raise OSError(code, os.strerror(errno.EACCES), path)
    try:
        if not open_token(current_process(), 8, ctypes.byref(token)):
            raise OSError(ctypes.get_last_error(), os.strerror(errno.EACCES), path)
        needed = dword()
        token_info(token, 1, None, 0, ctypes.byref(needed))
        if not ctypes.sizeof(pointer) <= needed.value <= 1 << 20:
            raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
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
                raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
            for index in range(count):
                ace = pointer()
                if not get_ace(dacl, index, ctypes.byref(ace)):
                    raise OSError(ctypes.get_last_error(), os.strerror(errno.EACCES), path)
                entry = ctypes.string_at(ace, 8)
                length = int.from_bytes(entry[2:4], "little")
                if (entry[0] not in {0, 1} or entry[1] & ~0x1f or length < 12
                        or ace.value < dacl.value + 8 or ace.value + length > dacl.value + size):
                    raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
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


def _windows_pip_tree_acl(path, *, system_owner=False):
    """Validate every descendant with native ACL checks in the background preparation worker."""
    pending = [os.path.abspath(path)]
    count = 0
    while pending:
        current = pending.pop()
        details = os.lstat(current)
        if (stat.S_ISLNK(details.st_mode) or getattr(details, "st_file_attributes", 0) & 0x400
                or not (stat.S_ISDIR(details.st_mode) or stat.S_ISREG(details.st_mode))):
            raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
        _windows_pip_private_access(current, system_owner=system_owner)
        if stat.S_ISDIR(details.st_mode):
            with os.scandir(current) as entries:
                pending.extend(entry.path for entry in entries)
        count += 1
    return {"verified": True, "entries": count}


def _windows_pip_private_directory(path):
    """Require an actual directory owned by this user without foreign write grants."""
    root = _installer_metadata_folder(path, "untrusted_windows_runtime_directory")
    if not os.path.isdir(root):
        raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
    _windows_pip_private_access(root)
    return root


def _windows_pip_private_access(path, *, system_owner=False, owner_sid=None, private_read=False):
    """Require a non-null trusted-owner ACL without foreign file or descendant write grants.

    OWNER RIGHTS (S-1-3-4) grants only to the object's owner, which is already
    required to be trusted; Python 3.12.4 and later give every ``mkdtemp`` and
    ``mkdir(mode=0o700)`` folder on Windows exactly that owner-only ACL.
    """
    evidence = _windows_pip_acl(path)
    user = evidence.get("user") if owner_sid is None else owner_sid
    rules = evidence.get("rules")
    if (not isinstance(user, str) or not user.startswith("S-1-")
            or evidence.get("owner") not in ({user, "S-1-5-18", "S-1-5-32-544"} if system_owner else {user})
            or evidence.get("has_dacl") is not True
            or not isinstance(rules, list) or not rules):
        raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
    permitted = {user, "S-1-5-18", "S-1-5-32-544", "S-1-3-4"}
    write_mask = 0xF01F01FF if private_read else 0x500D0156
    for rule in rules:
        if (not isinstance(rule, dict) or rule.get("type") not in (0, 1)
                or not isinstance(rule.get("sid"), str)
                or type(rule.get("rights")) is not int
                or type(rule.get("propagation")) is not int
                or rule["propagation"] & ~3):
            raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
        if rule["type"] == 0:
            if rule["rights"] & write_mask and rule["sid"] not in permitted:
                raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))



def _windows_pip_commands(record, version, workdir, machine):
    """Bind an ordinary running Windows venv without admitting shared environments."""
    if (machine.platform != "windows" or record.platform != "windows"
            or record.kind != "environment" or not record.running or not record.python):
        raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
    prefix = _windows_pip_private_directory(record.root)
    _windows_pip_private_directory(os.path.dirname(prefix))
    python = os.path.abspath(record.python)
    executable = os.path.abspath(machine.executable)
    scripts = os.path.join(prefix, "Scripts")
    if (os.path.normcase(os.path.realpath(machine.running_prefix or "")) != os.path.normcase(prefix)
            or os.path.normcase(os.path.realpath(os.path.dirname(python))) != os.path.normcase(scripts)
            or os.path.normcase(os.path.realpath(os.path.dirname(executable))) != os.path.normcase(scripts)
            or os.path.basename(python).lower() not in {"python.exe", "pythonw.exe"}
            or os.path.basename(executable).lower() not in {"python.exe", "pythonw.exe"}
            or not os.path.isfile(python) or os.path.islink(python)
            or not os.path.isfile(executable) or os.path.islink(executable)
            or not os.path.isfile(os.path.join(prefix, "pyvenv.cfg"))
            or os.path.isdir(os.path.join(prefix, "conda-meta"))
            or _inside(os.path.realpath(workdir), prefix)):
        raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
    _runtime_staging_allowed(prefix, machine)
    _windows_pip_tree_acl(prefix)
    console = os.path.join(scripts, "python.exe")
    if not os.path.isfile(console):
        raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
    version = _runtime_requested_version(version)
    return [console, "-I", "-m", "pip", "install", "--upgrade", "spacr==" + version], [python, "-I", "-m", "spacr.qt"]


def _windows_pip_helper(record, module, plan_path, machine, *, base=None):
    """Verify the current base interpreter outside the runtime before detaching it.

    :param base: the resolved base interpreter folder; ``sys.base_prefix`` when
        ``None``. A uv-managed runtime names its interpreter through a minor-version
        junction, so its caller resolves and confines it first.
    """
    if machine.platform != "windows":
        raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
    prefix = os.path.realpath(record.root)
    _windows_pip_private_directory(os.path.dirname(module))
    base = _installer_metadata_folder(sys.base_prefix if base is None else base,
                                      "untrusted_windows_base_interpreter", windows=True)
    python = os.path.join(base, "python.exe")
    if (not os.path.isfile(python) or os.path.islink(python)
            or _inside(base, prefix) or _inside(os.path.realpath(python), prefix)):
        raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
    _windows_pip_private_access(base, system_owner=True)
    _windows_pip_private_access(python, system_owner=True)
    _windows_pip_tree_acl(base, system_owner=True)
    _windows_pip_tree_acl(os.path.dirname(module))
    probe = '''import json, os, sys, sysconfig
print(json.dumps({'python':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'stdlib':sysconfig.get_path('stdlib'),'os_name':os.name}))
'''
    code, output = _run([python, "-I", "-B", "-c", probe], timeout=30)
    try:
        identity = json.loads(output)
        normalize = os.path.normcase
        if (code or identity.get("os_name") != "nt"
                or normalize(os.path.realpath(identity["python"])) != normalize(os.path.realpath(python))
                or normalize(os.path.realpath(identity["prefix"])) != normalize(base)
                or normalize(os.path.realpath(identity["base_prefix"])) != normalize(base)
                or _inside(os.path.realpath(identity["stdlib"]), prefix)):
            raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR)) from error
    environment = dict(machine.environ)
    for name in ("PYTHONHOME", "PYTHONPATH"):
        environment.pop(name, None)
    return [python, "-I", "-B", module, "run-plan", plan_path], environment


def _windows_update_library(name):
    """Load a native updater dependency only on the actual Windows host."""
    import ctypes

    if os.name != "nt":
        raise OSError(tr("a native Windows updater is required"))
    return ctypes.WinDLL(name, use_last_error=True)


def _windows_process_elevated():
    """Read the current process token without requesting administrator access."""
    import ctypes

    kernel = _windows_update_library("kernel32")
    security = _windows_update_library("advapi32")
    pointer, dword = ctypes.c_void_p, ctypes.c_uint32
    current = kernel.GetCurrentProcess
    current.argtypes, current.restype = [], pointer
    opened = security.OpenProcessToken
    opened.argtypes = [pointer, dword, ctypes.POINTER(pointer)]
    opened.restype = ctypes.c_int32
    information = security.GetTokenInformation
    information.argtypes = [pointer, dword, pointer, dword, ctypes.POINTER(dword)]
    information.restype = ctypes.c_int32
    close = kernel.CloseHandle
    close.argtypes, close.restype = [pointer], ctypes.c_int32
    token = pointer()
    if not opened(current(), 0x0008, ctypes.byref(token)):
        raise OSError(ctypes.get_last_error(), tr("Windows process token could not be opened"))
    try:
        elevation, size = dword(), dword()
        if not information(token, 20, ctypes.byref(elevation), ctypes.sizeof(elevation), ctypes.byref(size)):
            raise OSError(ctypes.get_last_error(), tr("Windows process elevation could not be read"))
        if size.value != ctypes.sizeof(elevation):
            raise OSError(tr("Windows returned an invalid process elevation"))
        return bool(elevation.value)
    finally:
        close(token)


def _windows_frozen_install(installer, destination, *, timeout=1800, phase_channel=None, desktop_pid=None):
    """Elevate only the pinned NSIS image and retain an unelevated coordinator.

    The caller must keep a checksum-verified installer handle denying writes
    and deletion open throughout this call. A timeout leaves the installer
    running; its caller must retain its pending journal and must not attempt
    another installation, rollback, cleanup or application relaunch.
    """
    import math
    import ntpath

    if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
            or not math.isfinite(timeout) or timeout <= 0 or timeout * 1000 >= 0xFFFFFFFF):
        raise ValueError(tr("invalid Windows installer timeout"))
    installer, destination = os.fspath(installer), os.fspath(destination)
    if any(character in value for value in (installer, destination) for character in '\x00\r\n"'):
        raise ValueError(tr("invalid Windows installer path"))
    if any(not ntpath.isabs(path) or len(ntpath.splitdrive(path)[0]) != 2
           or not ntpath.splitdrive(path)[0][0].isalpha()
           or ntpath.splitdrive(path)[0][1] != ":" for path in (installer, destination)):
        raise ValueError(tr("Windows installer paths must be absolute"))
    parameters = "/S /D=" + destination
    if phase_channel is not None:
        if (not isinstance(desktop_pid, int) or isinstance(desktop_pid, bool) or desktop_pid <= 1
                or any(character in phase_channel for character in '\x00\r\n"')
                or not ntpath.isabs(phase_channel)):
            raise ValueError(tr("invalid Windows installer path"))
        parameters = '/S /SPACRUPDATE="' + phase_channel + '" /SPACRPID=' + str(desktop_pid) + ' /D=' + destination
    return _windows_authorized_process(installer, parameters, timeout=timeout)


def _windows_authorized_process(installer, parameters, *, timeout=1800, visible=True):
    """Run a caller-validated trusted helper with one native authorization and an owned process handle.

    The caller must pin or otherwise protect the executable and construct only
    fixed supported arguments. An uncertain completion retains the process ID
    in TimeoutError; it never authorizes cleanup or another privileged operation.
    """
    import ctypes
    import math
    import ntpath

    if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
            or not math.isfinite(timeout) or timeout <= 0 or timeout * 1000 >= 0xFFFFFFFF):
        raise ValueError(tr("invalid Windows installer timeout"))
    if _windows_process_elevated():
        raise ValueError(tr("the update coordinator must run without administrator privileges"))
    if (not ntpath.isabs(installer) or any(character in installer for character in '\x00\r\n"')
            or any(character in parameters for character in '\x00\r\n')):
        raise ValueError(tr("invalid Windows installer path"))
    kernel = _windows_update_library("kernel32")
    shell = _windows_update_library("shell32")
    pointer, dword = ctypes.c_void_p, ctypes.c_uint32

    class _ShellExecuteInfo(ctypes.Structure):
        """Use the native pointer-width SHELLEXECUTEINFOW process layout."""
        _fields_ = [("cbSize", dword), ("fMask", dword), ("hwnd", pointer),
                    ("lpVerb", ctypes.c_wchar_p), ("lpFile", ctypes.c_wchar_p),
                    ("lpParameters", ctypes.c_wchar_p), ("lpDirectory", ctypes.c_wchar_p),
                    ("nShow", ctypes.c_int32), ("hInstApp", pointer), ("lpIDList", pointer),
                    ("lpClass", ctypes.c_wchar_p), ("hkeyClass", pointer),
                    ("dwHotKey", dword), ("hIcon", pointer), ("hProcess", pointer)]

    execute = shell.ShellExecuteExW
    execute.argtypes, execute.restype = [ctypes.POINTER(_ShellExecuteInfo)], ctypes.c_int32
    wait = kernel.WaitForSingleObject
    wait.argtypes, wait.restype = [pointer, dword], dword
    result = kernel.GetExitCodeProcess
    result.argtypes, result.restype = [pointer, ctypes.POINTER(dword)], ctypes.c_int32
    process_id = kernel.GetProcessId
    process_id.argtypes, process_id.restype = [pointer], dword
    close = kernel.CloseHandle
    close.argtypes, close.restype = [pointer], ctypes.c_int32
    request = _ShellExecuteInfo()
    request.cbSize = ctypes.sizeof(request)
    request.fMask = 0x00000040 | 0x00000100 | 0x00000400
    request.lpVerb = "runas"
    request.lpFile = installer
    request.lpParameters = parameters
    request.lpDirectory = ntpath.dirname(installer)
    request.nShow = 1 if visible else 0
    if not execute(ctypes.byref(request)):
        raise OSError(ctypes.get_last_error(), tr("Windows installer authorization did not complete"))
    if not request.hProcess:
        raise TimeoutError(tr("Windows installer did not return a process handle"))
    try:
        status = wait(request.hProcess, max(1, int(timeout * 1000)))
        if status == 258:
            error = TimeoutError(tr("Windows installer is still running; its pending update must be retained"))
            error.pid = int(process_id(request.hProcess))
            raise error
        if status != 0:
            error = TimeoutError(tr("Windows installer completion is unknown"))
            error.pid = int(process_id(request.hProcess))
            raise error
        code = dword()
        if not result(request.hProcess, ctypes.byref(code)) or code.value == 259:
            error = TimeoutError(tr("Windows installer exit status is unavailable"))
            error.pid = int(process_id(request.hProcess))
            raise error
        return int(code.value), ""
    finally:
        close(request.hProcess)


class _WindowsPipHandshake(_FrozenUpdateHandshake):
    """Retain the authenticated shutdown protocol with native Windows ACL and file-identity guards."""

    def __init__(self, plan):
        """Bind the same expiring plan fields after proving its private Windows message directory."""
        lease = plan["handshake"]
        token = lease["token"]
        self.expires = float(lease["expires"])
        if (lease.get("schema") != 1 or not isinstance(token, str)
                or len(token) != 64 or any(c not in "0123456789abcdef" for c in token)
                or not 0 < self.expires < float("inf")):
            raise ValueError("invalid frozen updater readiness lease")
        self.root = _windows_pip_private_directory(plan["workdir"])
        self.pid = int(plan["pid"])
        bound = {key: plan[key] for key in
                 ("adapter", "version", "pid", "records", "ticked", "workdir", "handshake")}
        self.binding = hashlib.sha256(json.dumps(
            bound, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def _read(self, name):
        """Reject linked, foreign-writable or replaced messages before decoding bounded authenticated JSON.

        Every message is published by an atomic replace, and a relaunched desktop
        replaces ``launch-result.json`` ("starting", then "ready") while the helper
        polls it. A message replaced between the checks and the open is read again
        from the start, a bounded number of times; a file that keeps changing is refused.
        """
        _windows_pip_private_directory(self.root)
        path = os.path.join(self.root, name)
        for _attempt in range(20):
            try:
                before = os.lstat(path)
            except FileNotFoundError:
                return None
            if (not stat.S_ISREG(before.st_mode)
                    or getattr(before, "st_file_attributes", 0) & 0x400 or before.st_size > 8192):
                raise ValueError("invalid frozen updater readiness message")
            _windows_pip_private_access(path)
            try:
                descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            except FileNotFoundError:
                continue
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                details = os.fstat(stream.fileno())
                if (details.st_dev, details.st_ino) != (before.st_dev, before.st_ino):
                    continue
                if not stat.S_ISREG(details.st_mode) or details.st_size > 8192:
                    raise ValueError("invalid frozen updater readiness message")
                message = json.loads(stream.read(8193))
            break
        else:
            raise ValueError("invalid frozen updater readiness message")
        if not isinstance(message, dict) or message.get("binding") != self.binding:
            raise ValueError("frozen updater readiness belongs to a different plan")
        return message


def _windows_pip_scripts(staged, prefix, runner):
    """Regenerate staged-prefix Windows entry points with their final paths and owning records."""
    probe = '''import base64, csv, hashlib, json, os, re, sys, tempfile
from pathlib import Path
from importlib.metadata import distributions
from pip._vendor.distlib.scripts import ScriptMaker, enquote_executable
stage, final = map(Path, sys.argv[1:])
scripts = stage / 'Scripts'
encodings = (str(stage).encode('utf-8'), str(stage).replace('\\\\', '/').encode('utf-8'))
owners = {}
for dist in distributions(path=[str(stage / 'Lib' / 'site-packages')]):
    metadata = Path(dist._path).resolve()
    assert metadata.is_relative_to(stage.resolve()) and metadata.is_dir()
    for entry in dist.entry_points:
        if entry.group not in ('console_scripts', 'gui_scripts'):
            continue
        maker = ScriptMaker(None, str(scripts))
        for name in maker.get_script_filenames(entry.name):
            owners.setdefault((name + '.exe').lower(), []).append((dist, entry))
changed = []
for path in scripts.iterdir():
    assert not path.is_symlink()
    if not path.is_file() or not any(token in path.read_bytes() for token in encodings):
        continue
    candidates = owners.get(path.name.lower(), [])
    assert len(candidates) == 1 and path.suffix.lower() == '.exe', 'Unowned staged-prefix script'
    dist, entry = candidates[0]
    assert re.fullmatch(r'[A-Za-z0-9_.-]+', path.stem), 'Unsafe entry point name'
    record = Path(dist._path) / 'RECORD'
    rows = list(csv.reader(record.open(newline='', encoding='utf-8')))
    selected = [row for row in rows if len(row) == 3 and Path(dist.locate_file(row[0])).resolve() == path.resolve()]
    assert len(selected) == 1, 'Missing or ambiguous entry point ownership'
    maker = ScriptMaker(None, str(scripts))
    maker.variants = {''}
    maker.clobber = True
    maker.executable = enquote_executable(str(final / 'Scripts' / 'python.exe'))
    assert (stage / 'Scripts' / ('pythonw.exe' if entry.group == 'gui_scripts' else 'python.exe')).is_file()
    files = maker.make(path.stem + ' = ' + entry.value, options={'gui': entry.group == 'gui_scripts'})
    assert files == [str(path)] and not any(token in path.read_bytes() for token in encodings)
    payload = path.read_bytes()
    selected[0][1:] = ['sha256=' + base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b'=').decode('ascii'), str(len(payload))]
    descriptor, temporary = tempfile.mkstemp(prefix='RECORD-', dir=record.parent)
    try:
        with os.fdopen(descriptor, 'w', newline='', encoding='utf-8') as stream:
            csv.writer(stream).writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, record)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    changed.append(path.name)
print(json.dumps({'verified': True, 'scripts': changed}))
'''
    code, output = runner([os.path.join(staged, "Scripts", "python.exe"), "-I", "-B", "-c", probe, staged, prefix])
    try:
        receipt = json.loads(output)
        if code or receipt.get("verified") is not True or not isinstance(receipt.get("scripts"), list):
            raise ValueError("windows_entrypoint_verification_failed")
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("windows_entrypoint_verification_failed") from error
    return receipt


def _windows_relocatable_scripts(staged):
    """Prove a uv runtime survives its rename: relocatable, and no file in Scripts names the staging path."""
    configuration = os.path.join(staged, "pyvenv.cfg")
    with open(configuration, encoding="utf-8") as handle:
        settings = {key.strip().lower(): value.strip().lower()
                    for key, separator, value in (line.partition("=") for line in handle) if separator}
    if settings.get("relocatable") != "true":
        raise ValueError("windows_entrypoint_verification_failed")
    tokens = set()
    for spelling in {staged, os.path.realpath(staged)}:
        for form in {spelling, spelling.replace("\\", "/"), spelling.lower(), spelling.replace("\\", "/").lower()}:
            tokens.update({form.encode("utf-8"), form.encode("utf-16-le")})
    checked = 0
    with os.scandir(os.path.join(staged, "Scripts")) as entries:
        for entry in entries:
            details = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(details.st_mode) or getattr(details, "st_file_attributes", 0) & 0x400:
                raise ValueError("windows_entrypoint_verification_failed")
            with open(entry.path, "rb") as stream:
                payload = stream.read()
            if any(token in payload or token in payload.lower() for token in tokens):
                raise ValueError("windows_entrypoint_verification_failed")
            checked += 1
    return {"verified": True, "relocatable": True, "scripts": checked}


def _windows_pip_tree(prefix):
    """Read complete runtime identity and reject every linked or reparse payload before use."""
    _installer_metadata_folder(prefix, "untrusted_windows_runtime_directory", windows=True)
    _windows_pip_tree_acl(prefix)
    return _runtime_recovery_inventory(prefix)


def _windows_pip_transition(path, identity, inventory):
    """Refuse a foreign or modified runtime before any two-rename activation or rollback step."""
    details = os.stat(path, follow_symlinks=False)
    if ([details.st_dev, details.st_ino] != identity or _windows_pip_tree(path) != inventory):
        raise ValueError("windows_runtime_replaced_or_modified")


def _windows_pip_transaction(record, version, install, *, machine, runner=None, activate=None,
                             before_activate=None, uv_tool=False, without_wgpu=False):
    """Stage a Windows venv and retain its original through journaled, non-atomic replacement.

    :param uv_tool: the runtime is a relocatable uv environment made by the online
        installer; ``install`` is its bundled uv command, retargeted to the staged copy.
    :param without_wgpu: the saved installation choice omits wgpu, so uv's sole
        missing-wgpu diagnostic is accepted.
    """
    runner = runner or _run
    prefix = _windows_pip_private_directory(record.root)
    parent = _windows_pip_private_directory(os.path.dirname(prefix))
    lock = os.path.join(parent, "." + os.path.basename(prefix) + ".spacr-update-lock")
    lease = _take_lease_dir(lock)
    root = None
    state = None
    retired = False
    installed = False
    safe_release = True
    try:
        _windows_pip_private_directory(lock)
        _runtime_staging_allowed(prefix, machine)
        old = _retiring_runtime_identity(prefix)
        if old is None:
            raise ValueError("previous_runtime_identity_unverified")
        record = replace(record, version=old["version"])
        if _runtime_version_compare(record.python, record.version, version, runner) > 0:
            raise ValueError("The running spaCR is newer than the requested release. Nothing was changed.")
        _prune_runtime_transactions(prefix, "windows-online-uv" if uv_tool else "windows-private-pip")
        root = tempfile.mkdtemp(prefix=".spacr-windows-update-", dir=parent)
        _windows_pip_private_directory(root)
        staged = os.path.join(root, "staged")
        previous = os.path.join(root, "previous")
        details = os.stat(prefix, follow_symlinks=False)
        state = {"schema": 1, "family": "windows-online-uv" if uv_tool else "windows-private-pip", "prefix": prefix,
                 "rollback_runtime": previous, "staged_runtime": staged, "owner": _transaction_owner(),
                 "requested_version": version, "old_identity": [details.st_dev, details.st_ino],
                 "old_inventory": _windows_pip_tree(prefix), "state": "staging"}
        _runtime_recovery_journal(root, state)
        shutil.copytree(prefix, staged)
        _windows_pip_private_directory(staged)
        python = os.path.join(staged, "Scripts", "python.exe")
        if uv_tool:
            command = list(install)
            position = command.index("--python") + 1 if "--python" in command else 0
            if (not position or command[1:4] != ["pip", "install", "--upgrade"]
                    or "spacr==" + str(version) not in command
                    or os.path.normcase(command[position]) != os.path.normcase(os.path.join(prefix, "Scripts", "python.exe"))):
                raise ValueError("the package command does not target this private runtime")
            command[position] = python
            check_command = [command[0], "pip", "check", "--python", python]
            if "--cache-dir" in command:
                check_command.extend(["--cache-dir", command[command.index("--cache-dir") + 1]])
            if without_wgpu:
                check_command.extend(["--color", "never"])
        else:
            if install != [os.path.join(prefix, "Scripts", "python.exe"), "-I", "-m", "pip", "install", "--upgrade", "spacr==" + str(version)]:
                raise ValueError("the package command does not target this private runtime")
            command = [python, *install[1:]]
            check_command = [python, "-I", "-m", "pip", "check"]
        code, output = runner(command)
        if code:
            raise RuntimeError(output)
        code, checked = runner(check_command)
        if code and not (uv_tool and without_wgpu and _only_saved_wgpu_omission(code, checked)):
            raise RuntimeError(checked)
        _windows_pip_tree(staged)
        for directory, _folders, files in os.walk(staged):
            for name in files:
                path = os.path.join(directory, name)
                relative = os.path.relpath(path, staged)
                prior = state["old_inventory"].get(relative, {})
                if not name.endswith(".py") or prior.get("kind") != "file":
                    continue
                details = os.stat(path, follow_symlinks=False)
                original = os.stat(os.path.join(prefix, relative), follow_symlinks=False)
                if (details.st_size == prior["size"] and int(details.st_mtime) == int(original.st_mtime)
                        and _macos_digest(path) != prior["sha256"]):
                    os.utime(path, ns=(details.st_atime_ns, (int(original.st_mtime) + 2) * 1_000_000_000),
                             **({"follow_symlinks": False} if os.utime in os.supports_follow_symlinks else {}))
                    if int(os.stat(path).st_mtime) == int(original.st_mtime):
                        raise OSError(errno.EIO, os.strerror(errno.EIO), path)
        code, observed = runner(_runtime_identity_command(python, version, prefix=staged, previous_version=record.version))
        try:
            identity = json.loads(observed)
            if (code or identity.get("verified") is not True
                    or identity.get("requested_version") != str(version)
                    or _runtime_version_compare(python, identity["version"], version, runner) != 0
                    or _runtime_version_compare(python, identity["version"], record.version, runner) < 0):
                raise ValueError("windows_staged_identity_unverified")
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            raise ValueError("windows_staged_identity_unverified") from error
        code, checked = runner([python, "-I", "-B", "-c", "import spacr.qt"])
        if code:
            raise RuntimeError(checked)
        state["entrypoints"] = (_windows_relocatable_scripts(staged) if uv_tool
                                else _windows_pip_scripts(staged, prefix, runner))
        details = os.stat(staged, follow_symlinks=False)
        state.update(new_identity=[details.st_dev, details.st_ino], new_inventory=_windows_pip_tree(staged), state="ready")
        _runtime_recovery_journal(root, state)
        if before_activate is not None:
            before_activate()
        _windows_pip_transition(prefix, state["old_identity"], state["old_inventory"])
        _windows_pip_transition(staged, state["new_identity"], state["new_inventory"])
        state["state"] = "retiring"
        _runtime_recovery_journal(root, state)
        os.rename(prefix, previous)
        retired = True
        state["state"] = "retired"
        _runtime_recovery_journal(root, state)
        if os.path.lexists(prefix):
            raise ValueError("windows_runtime_replaced_or_modified")
        os.rename(staged, prefix)
        installed = True
        state["state"] = "verifying"
        _runtime_recovery_journal(root, state)
        _windows_pip_transition(previous, state["old_identity"], state["old_inventory"])
        code, observed = runner(_runtime_identity_command(record.python, version, prefix=prefix, previous_version=record.version))
        final = json.loads(observed)
        if code or final.get("verified") is not True or final.get("version") != identity["version"]:
            raise ValueError("windows_activated_identity_unverified")
        code, checked = runner([record.python, "-I", "-B", "-c", "import spacr.qt"])
        if code:
            raise RuntimeError(checked)
        _windows_pip_transition(prefix, state["new_identity"], state["new_inventory"])
        state.update(state="committed", verified_version=identity["version"])
        _runtime_recovery_journal(root, state)
        if activate is not None:
            activate(identity["version"])
        return identity["version"], output + "\nPrevious runtime retained at " + previous
    except _UpdateLaunchStillRunning as error:
        state["state"] = "launch_unconfirmed"
        state["launch_pid"] = error.pid
        _runtime_recovery_journal(root, state)
        raise
    except BaseException:
        if retired:
            try:
                _windows_pip_transition(previous, state["old_identity"], state["old_inventory"])
                if installed:
                    _windows_pip_transition(prefix, state["new_identity"], state["new_inventory"])
                    if os.path.lexists(staged):
                        raise ValueError("windows_runtime_replaced_or_modified")
                    os.rename(prefix, staged)
                if os.path.lexists(prefix):
                    raise ValueError("windows_runtime_replaced_or_modified")
                os.rename(previous, prefix)
                state["state"] = "rolled_back"
                _runtime_recovery_journal(root, state)
            except BaseException as error:
                safe_release = False
                raise RuntimeError("windows_runtime_rollback_requires_review", root) from error
        if root and state is not None and not retired:
            state["state"] = "staging_failed"
            _runtime_recovery_journal(root, state)
        raise
    finally:
        try:
            current = os.stat(lock, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None and (current.st_dev, current.st_ino) == (lease.st_dev, lease.st_ino):
            if safe_release:
                _release_lease_dir(lock)
            else:
                _retain_lease_dir(lock, "windows runtime rollback requires review")


def _run_windows_pip_update(plan, machine, *, wait=None, run=None, spawn=None):
    """Approve a private Windows runtime update and recheck identity after desktop shutdown."""
    records = [_record_from_json(data) for data in plan["records"]]
    candidates = [record for record in records if record.kind == "environment" and record.running]
    if len(candidates) != 1 or plan.get("frozen_application") or machine.platform != "windows":
        raise ValueError(tr(_PRIVATE_PIP_UPDATE_ERROR))
    record = candidates[0]
    install, relaunch = _windows_pip_commands(record, plan["version"], plan["workdir"], machine)
    if plan.get("fetch") is not None or plan.get("install") != install or plan.get("relaunch") != relaunch:
        raise ValueError("the package command does not target this private runtime")
    runner = run or _run
    current = _retiring_runtime_identity(record.root)
    if current is None:
        raise ValueError("previous_runtime_identity_unverified")
    if _runtime_version_compare(record.python, current["version"], plan["version"], runner) > 0:
        raise ValueError("The running spaCR is newer than the requested release. Nothing was changed.")
    code, output = runner([record.python, "-I", "-B", "-c", "import pip"])
    if code:
        raise RuntimeError(output)
    handshake = _WindowsPipHandshake(plan)
    handshake.check()

    def prepared():
        """Request shutdown only when the private replacement is verified."""
        handshake.ready()
        handshake.wait_for_shutdown(wait)
        _windows_pip_commands(record, plan["version"], plan["workdir"], machine)
        if _retiring_runtime_identity(record.root) != current:
            raise ValueError("previous_runtime_identity_unverified")

    return _windows_pip_transaction(record, plan["version"], install, machine=machine, runner=runner,
                                    before_activate=prepared,
                                    activate=lambda observed: _launch_updated_desktop(
                                        plan, relaunch, observed, record.root, spawn=spawn))


_WINDOWS_ONLINE_RESOLVER_GUARDS = ("numba>=0.60,<1.0", "llvmlite>=0.43,<1.0")


def _windows_online_update_record(records, *, system=None, strict=False):
    """The running Windows online installation's record, or None elsewhere."""
    machine = system or _Machine()
    if machine.platform != "windows" or getattr(sys, "frozen", False):
        return None
    candidates = [record for record in records if record.kind == "installer"
                  and record.running and record.platform == "windows"
                  and record.layout == "windows-online"]
    if strict and len(candidates) != 1:
        raise ValueError("the running Windows online installation is ambiguous or unavailable")
    return candidates[0] if candidates else None


def _windows_online_profiles(prefix):
    """The installer-profile reader, from the runtime being updated when spaCR itself is not importable.

    The update helper runs on the installation's base Python, outside the runtime, where
    ``spacr`` cannot be imported; ``install_profile`` uses only the standard library.
    """
    from importlib import import_module, util

    try:
        return import_module("spacr.install_profile")
    except ModuleNotFoundError:
        source = os.path.join(prefix, "Lib", "site-packages", "spacr", "install_profile.py")
        if not os.path.isfile(source) or os.path.islink(source):
            raise ValueError(tr("The saved installation choices could not be read. Nothing was changed."))
        spec = util.spec_from_file_location("_spacr_runtime_install_profile", source)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _windows_online_commands(record, version, workdir, desktop_prefix, desktop_executable):
    """Bind the online installer's own uv, private runtime and saved choices to one update.

    :param record: the running online installation.
    :param version: the exact release to stage.
    :param workdir: the updater's private folder, outside the installation.
    :param desktop_prefix: ``sys.prefix`` of the desktop that requested the update.
    :param desktop_executable: the interpreter that runs that desktop.
    :returns: the staged uv install command and the relaunch command.
    """
    from pathlib import Path

    refusal = tr("This installation cannot stage a verified update. Nothing was changed.")
    if (record.kind != "installer" or record.layout != "windows-online"
            or record.platform != "windows" or not record.running or not record.python):
        raise ValueError(refusal)
    root = _windows_pip_private_directory(record.root)
    prefix = _windows_pip_private_directory(os.path.join(root, "venv"))
    scripts = os.path.join(prefix, "Scripts")
    python = os.path.join(scripts, "python.exe")
    windowed = os.path.join(scripts, "pythonw.exe")
    uv = os.path.join(root, "bootstrap", "uv.exe")
    executable = os.path.abspath(desktop_executable or "")
    if (_windows_path_key(record.python) != _windows_path_key(python)
            or _windows_path_key(desktop_prefix or "") != _windows_path_key(prefix)
            or _windows_path_key(os.path.dirname(executable)) != _windows_path_key(scripts)
            or os.path.basename(executable).lower() not in {"python.exe", "pythonw.exe"}
            or any(not os.path.isfile(path) or os.path.islink(path) for path in (uv, python, windowed, executable))
            or _inside(os.path.realpath(workdir), root)):
        raise ValueError(refusal)
    profiles = _windows_online_profiles(prefix)
    profile_path = Path(root) / "install-profile.json"
    profile = profiles.read_profile(profile_path)
    if profile_path.exists() and (profile is None or not isinstance(profile.get("without_wgpu", False), bool)):
        raise ValueError(tr("The saved installation choices could not be read. Nothing was changed."))
    version = _runtime_requested_version(version)
    install = [uv, "pip", "install", "--upgrade", "--cache-dir", os.path.join(root, "cache"), "--python", python]
    if profile:
        install.extend(["--torch-backend", profile["requested_backend"]])
        if profile.get("without_wgpu") is True:
            try:
                no_wgpu_override = profiles.no_wgpu_override
            except AttributeError as error:
                raise ValueError(tr("The saved choice to install without wgpu could not be preserved.")) from error
            override = no_wgpu_override(profile, Path(workdir) / "no-wgpu-override.txt")
            if override is None:
                raise ValueError(tr("The saved choice to install without wgpu could not be preserved."))
            install.extend(["--override", str(override)])
    install.extend(["spacr==" + version, *_WINDOWS_ONLINE_RESOLVER_GUARDS])
    return install, [windowed, "-I", "-m", "spacr.qt"]


def _windows_online_helper(record, module, plan_path, machine):
    """Detach the updater on the installation's own uv-managed Python, outside the runtime it replaces."""
    root = os.path.realpath(record.root)
    prefix = os.path.join(root, "venv")
    base = os.path.realpath(sys.base_prefix)
    if (os.path.normcase(os.path.realpath(sys.prefix)) != os.path.normcase(prefix)
            or not _inside(base, os.path.join(root, "python")) or _inside(base, prefix)):
        raise ValueError(tr("This installation cannot stage a verified update. Nothing was changed."))
    return _windows_pip_helper(replace(record, root=prefix), module, plan_path, machine, base=base)


def _remove_runtime_copy(path):
    """Delete a retired runtime copy; a link or junction inside it is removed as a link, never entered."""
    with os.scandir(path) as entries:
        for entry in entries:
            details = entry.stat(follow_symlinks=False)
            attributes = getattr(details, "st_file_attributes", 0)
            if stat.S_ISLNK(details.st_mode) or attributes & 0x400:
                if attributes & 0x10:
                    os.rmdir(entry.path)
                else:
                    os.unlink(entry.path)
            elif stat.S_ISDIR(details.st_mode):
                _remove_runtime_copy(entry.path)
            else:
                if attributes & 0x1:
                    os.chmod(entry.path, stat.S_IWRITE)
                os.unlink(entry.path)
    os.rmdir(path)


def _prune_windows_online_recovery(root):
    """Keep only the newest committed online-runtime recovery copy beside the runtime.

    Only folders this adapter made, holding a committed ``windows-online-uv``
    journal for this runtime, are candidates; links are never followed, and any
    folder that cannot be read is kept.

    :returns: the folders removed.
    """
    root = os.path.realpath(root)
    prefix = os.path.join(root, "venv")
    committed = []
    with os.scandir(root) as entries:
        for entry in entries:
            if not entry.name.startswith(".spacr-windows-update-"):
                continue
            details = entry.stat(follow_symlinks=False)
            if not stat.S_ISDIR(details.st_mode) or getattr(details, "st_file_attributes", 0) & 0x400:
                continue
            journal = os.path.join(entry.path, "transaction.json")
            try:
                with open(journal, encoding="utf-8") as handle:
                    state = json.load(handle)
                stamp = os.stat(journal, follow_symlinks=False).st_mtime_ns
            except (OSError, ValueError):
                continue
            if (isinstance(state, dict) and state.get("family") == "windows-online-uv"
                    and state.get("state") == "committed"
                    and os.path.normcase(str(state.get("prefix", ""))) == os.path.normcase(prefix)):
                committed.append((stamp, entry.path))
    removed = []
    for _stamp, path in sorted(committed)[:-1]:
        try:
            _remove_runtime_copy(path)
            removed.append(path)
        except OSError:
            pass
    return removed


def _windows_path_key(path):
    """Compare Windows paths by their resolved, case-folded spelling."""
    return os.path.normcase(os.path.realpath(path))


def _windows_online_ready(handshake, wait, record, plan, desktop, install, relaunch, runtime, current):
    """Offer Restart only after the staged runtime passes its checks."""
    handshake.ready()
    handshake.wait_for_shutdown(wait)
    if _windows_online_commands(record, plan["version"], plan["workdir"], desktop.get("prefix"),
                                desktop.get("executable")) != (install, relaunch):
        raise ValueError(tr("The saved installation choices changed while preparing the update. Nothing was changed."))
    if _retiring_runtime_identity(runtime.root) != current:
        raise ValueError("previous_runtime_identity_unverified")


def _run_windows_online_update(plan, machine, *, wait=None, run=None, spawn=None):
    """Carry out an approved Windows online update with the installation's own uv.

    The desktop identity recorded when the update was requested is rechecked after
    shutdown; uv or Python found on PATH is never substituted.
    """
    records = [_record_from_json(data) for data in plan["records"]]
    record = _windows_online_update_record(records, system=machine, strict=True)
    desktop = plan.get("desktop") or {}
    if record is None or plan.get("frozen_application"):
        raise ValueError(tr("This installation cannot stage a verified update. Nothing was changed."))
    install, relaunch = _windows_online_commands(record, plan["version"], plan["workdir"],
                                                 desktop.get("prefix"), desktop.get("executable"))
    if plan.get("fetch") is not None or plan.get("install") != install or plan.get("relaunch") != relaunch:
        raise ValueError("the Windows online update commands differ from the approved installation")
    runtime = replace(record, root=os.path.join(os.path.realpath(record.root), "venv"))
    runner = run or _run
    current = _retiring_runtime_identity(runtime.root)
    if current is None:
        raise ValueError("previous_runtime_identity_unverified")
    if _runtime_version_compare(runtime.python, current["version"], plan["version"], runner) > 0:
        raise ValueError("The running spaCR is newer than the requested release. Nothing was changed.")
    handshake = _WindowsPipHandshake(plan)
    handshake.check()
    from functools import partial

    prepared = partial(_windows_online_ready, handshake, wait, record, plan, desktop, install, relaunch,
                       runtime, current)
    observed, output = _windows_pip_transaction(
        runtime, plan["version"], install, machine=machine, runner=runner, before_activate=prepared,
        uv_tool=True, without_wgpu="--override" in install,
        activate=lambda version: _launch_updated_desktop(plan, relaunch, version, runtime.root, spawn=spawn))
    pruned = _prune_windows_online_recovery(record.root)
    if pruned:
        output += "\nRemoved older recovery copies: " + ", ".join(pruned)
    return observed, output


_PRIVATE_PIP_UPDATE_ERROR = "This environment needs its package manager; spaCR cannot replace it safely. Nothing was changed."


def _pip_environment_commands(record, version, workdir, machine):
    """Bind a dedicated owner-only POSIX venv to its isolated pip command."""
    if machine.platform == "windows":
        return _windows_pip_commands(record, version, workdir, machine)
    if (machine.platform not in {"linux", "macos"} or record.kind != "environment"
            or not record.running or not record.python or record.platform != machine.platform):
        raise ValueError(_PRIVATE_PIP_UPDATE_ERROR)
    prefix = os.path.realpath(record.root)
    python = os.path.abspath(record.python)
    if (os.path.abspath(record.root) != prefix
            or os.path.realpath(machine.running_prefix or "") != prefix
            or os.path.abspath(machine.executable) != python
            or os.path.dirname(python) != os.path.join(prefix, "bin")
            or not os.path.isfile(python) or not os.access(python, os.X_OK)
            or not os.path.isfile(os.path.join(prefix, "pyvenv.cfg"))
            or os.path.isdir(os.path.join(prefix, "conda-meta"))):
        raise ValueError(_PRIVATE_PIP_UPDATE_ERROR)
    for directory in (prefix, os.path.dirname(prefix)):
        details = os.lstat(directory)
        if (not stat.S_ISDIR(details.st_mode) or details.st_uid != os.geteuid()
                or stat.S_IMODE(details.st_mode) & 0o022):
            raise ValueError(_PRIVATE_PIP_UPDATE_ERROR)
    if _inside(os.path.realpath(workdir), prefix):
        raise ValueError("the updater folder must be outside the installed environment")
    version = _runtime_requested_version(version)
    if record.version:
        _runtime_requested_version(record.version)
        try:
            stable_order = (_macos_online_version(record.version) > _macos_online_version(version))
        except ValueError:
            stable_order = None
        if stable_order is True:
            raise ValueError("The running spaCR is newer than the requested release. Nothing was changed.")
    command = [python, "-I", "-m", "pip", "install", "--upgrade", "spacr==" + str(version)]
    return command, [python, "-I", "-m", "spacr.qt"]


def _pip_environment_identity(record, runner):
    """Read the original runtime without bytecode writes or trusting stale metadata."""
    probe = _runtime_identity_command(record.python, record.version, prefix=record.root)
    probe.insert(2, "-B")
    code, output = runner(probe)
    try:
        current = json.loads(output)
        verified = (code == 0 and current.get("verified") is True
                    and os.path.realpath(current["prefix"]) == os.path.realpath(record.root)
                    and os.path.realpath(current["python"]) == os.path.realpath(record.python)
                    and isinstance(current.get("version"), str))
        if verified and record.version:
            verified = _runtime_version_compare(record.python, current["version"], record.version, runner) == 0
    except (ValueError, TypeError, KeyError, AttributeError):
        verified = False
    if not verified:
        raise RuntimeError("The desktop interpreter could not be identified; no relaunch was requested.")
    return current


def _run_pip_environment_update(plan, machine, *, wait=None, run=None, spawn=None):
    """Stage a private pip update before requesting approved shutdown."""
    if machine.platform == "windows":
        return _run_windows_pip_update(plan, machine, wait=wait, run=run, spawn=spawn)
    records = [_record_from_json(data) for data in plan["records"]]
    candidates = [record for record in records if record.kind == "environment" and record.running]
    if len(candidates) != 1 or plan.get("frozen_application"):
        raise ValueError(_PRIVATE_PIP_UPDATE_ERROR)
    record = candidates[0]
    install, relaunch = _pip_environment_commands(record, plan["version"], plan["workdir"], machine)
    if plan.get("fetch") is not None or plan.get("install") != install or plan.get("relaunch") != relaunch:
        raise ValueError("the package command does not target this private runtime")
    _runtime_staging_allowed(os.path.realpath(record.root), machine)
    runner = run or _run
    current = _pip_environment_identity(record, runner)
    if _runtime_version_compare(record.python, current["version"], plan["version"], runner) > 0:
        raise ValueError("The running spaCR is newer than the requested release. Nothing was changed.")
    code, output = runner([record.python, "-I", "-B", "-c", "import pip"])
    if code:
        raise RuntimeError(output)
    handshake = _FrozenUpdateHandshake(plan)
    handshake.check()

    def prepared():
        """Keep the running environment until the verified replacement is accepted."""
        handshake.ready()
        handshake.wait_for_shutdown(wait)
        _pip_environment_commands(record, plan["version"], plan["workdir"], machine)
        if _pip_environment_identity(record, runner) != current:
            raise RuntimeError("The desktop interpreter could not be identified; no relaunch was requested.")

    return _transactional_runtime_update(
        record, plan["version"], install, machine=machine, runner=runner,
        before_activate=prepared,
        activate=lambda observed: _launch_updated_desktop(
            plan, relaunch, observed, record.root, spawn=spawn))


def _frozen_windows_walk_error(error):
    """Refuse a partial payload traversal instead of silently omitting files."""
    raise error


def _frozen_windows_payload(root, version, *, retirement=False):
    """Validate the complete, regular-file Windows onedir payload before use."""
    from pathlib import Path, PurePosixPath

    root = Path(root)
    observed = {}
    directories = set()
    for directory, folders, files in os.walk(root, followlinks=False, onerror=_frozen_windows_walk_error):
        if Path(directory) == root:
            folders[:] = [name for name in folders if name == "_internal"]
            files = [name for name in files if name in {"spacr.exe", "spacr-payload.json"}]
        for name in folders + files:
            path = Path(directory) / name
            details = path.lstat()
            if (stat.S_ISLNK(details.st_mode)
                    or getattr(details, "st_file_attributes", 0) & 0x400
                    or not (stat.S_ISDIR(details.st_mode) or stat.S_ISREG(details.st_mode))):
                raise ValueError("frozen_payload_nonregular_path")
            if stat.S_ISDIR(details.st_mode):
                directories.add(path.relative_to(root).as_posix())
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if relative != "spacr-payload.json":
                observed[relative] = _macos_digest(str(path))
    manifest_path = root / "spacr-payload.json"
    if manifest_path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("frozen_payload_manifest_too_large")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    files = manifest.get("files")
    if (manifest.get("schema") != 1 or manifest.get("version") != version
            or not isinstance(files, dict) or not files or files != observed):
        raise ValueError("frozen_payload_manifest_mismatch")
    for name in files:
        path = PurePosixPath(name)
        if (path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts)
                or "\\" in name or ":" in name
                or path.parts[0] not in {"spacr.exe", "_internal"}):
            raise ValueError("frozen_payload_path_outside_application")
    if "spacr.exe" not in files or not any(name.startswith("_internal/") for name in files):
        raise ValueError("frozen_payload_incomplete")
    if retirement:
        expected_directories = {parent.as_posix() for name in files
                                for parent in PurePosixPath(name).parents if parent.as_posix() != "."}
        if directories - expected_directories:
            raise ValueError("frozen_payload_contains_unknown_directories")
    return manifest


def _frozen_windows_identity(root, version, run=None):
    """Check the loaded executable, application, import root and GUI imports."""
    from pathlib import Path

    root = Path(root).resolve()
    with tempfile.TemporaryDirectory(prefix="spacr-frozen-identity-") as folder:
        output = Path(folder) / "identity.json"
        argv = [str(root / "spacr.exe"), "--installer-identity", str(output)]
        code, _text = (run(argv) if run else _run(argv, timeout=60))
        if code or not output.exists():
            raise ValueError("frozen_application_identity_unverified")
        details = output.lstat()
        if (not stat.S_ISREG(details.st_mode)
                or getattr(details, "st_file_attributes", 0) & 0x400):
            raise ValueError("frozen_application_identity_unverified")
        descriptor = os.open(output, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            details = os.fstat(stream.fileno())
            if (not stat.S_ISREG(details.st_mode) or details.st_size > 16384
                    or getattr(details, "st_file_attributes", 0) & 0x400
                    or output.is_symlink()):
                raise ValueError("frozen_application_identity_unverified")
            identity = json.load(stream)
    if (identity.get("schema") != 1 or identity.get("frozen") is not True
            or identity.get("version") != version
            or Path(identity.get("executable", "")).resolve() != root / "spacr.exe"
            or Path(identity.get("bundle_root", "")).resolve() != root
            or Path(identity.get("import_root", "")).resolve() != root / "_internal"
            or not _inside(str(Path(identity.get("module", "")).resolve()), str(root / "_internal"))):
        raise ValueError("frozen_application_identity_mismatch")
    return identity


def _frozen_windows_previous_version(root):
    """Read a legacy bundled distribution without executing its old application."""
    from email.parser import Parser
    from pathlib import Path

    root = Path(root)
    for path in (root / "spacr.exe", root / "_internal"):
        if path.exists() or path.is_symlink():
            details = path.lstat()
            if stat.S_ISLNK(details.st_mode) or getattr(details, "st_file_attributes", 0) & 0x400:
                raise ValueError("previous_frozen_application_identity_unverified")
    if not (root / "spacr.exe").exists():
        if (root / "_internal").exists():
            raise ValueError("previous_frozen_application_identity_unverified")
        return None
    versions = []
    for path in (root / "_internal").glob("spacr*.dist-info/METADATA"):
        for candidate in (path.parent, path):
            details = candidate.lstat()
            if stat.S_ISLNK(details.st_mode) or getattr(details, "st_file_attributes", 0) & 0x400:
                raise ValueError("previous_frozen_application_identity_unverified")
        if not path.is_file() or path.stat().st_size > 1024 * 1024:
            raise ValueError("previous_frozen_application_identity_unverified")
        metadata = Parser().parsestr(path.read_text(encoding="utf-8"))
        if metadata.get("Name", "").lower().replace("_", "-") in {"spacr", "spacr-nightly"}:
            versions.append(metadata.get("Version", ""))
    if len(versions) != 1:
        raise ValueError("previous_frozen_application_identity_unverified")
    return versions[0]


def _frozen_windows_entry_state(path):
    """Bind a moved payload so rollback never deletes newly supplied user files."""
    details = os.lstat(path)
    if stat.S_ISDIR(details.st_mode):
        contents = _macos_tree(str(path))
    elif stat.S_ISREG(details.st_mode):
        contents = _macos_digest(str(path))
    else:
        raise ValueError("frozen_payload_nonregular_path")
    return details.st_dev, details.st_ino, contents


def _install_frozen_windows(source, destination, version, *, run=None,
                            before_activate=None, after_activate=None, preserve_previous=False):
    """Stage, verify and replace only a Windows bundle's application payload.

    Caches, preferences and unrelated install-root files are never part of the
    replacement. A new native uninstaller and its executable-bound capability
    record participate in the same transaction when supplied by the installer;
    otherwise the existing uninstaller remains unchanged. A displaced legacy payload, or one with
    unknown or modified files, remains in private recovery storage. A failed
    activation restores the original payload; an interrupted transaction refuses
    further changes and leaves its recovery directory intact.
    """
    from pathlib import Path

    source, destination = Path(os.path.abspath(source)), Path(os.path.abspath(destination))
    for path in (source, destination):
        for parent in (path, *path.parents):
            if parent.exists() or parent.is_symlink():
                details = parent.lstat()
                if stat.S_ISLNK(details.st_mode) or getattr(details, "st_file_attributes", 0) & 0x400:
                    raise ValueError("frozen_install_reparse_path")
    if (_inside(str(source), str(destination)) or _inside(str(destination), str(source))
            or _inside(os.path.abspath(sys.executable), str(destination))):
        raise ValueError("frozen_install_requires_outside_helper")
    _macos_online_version(version)
    _frozen_windows_payload(source, version)
    names = ("_internal", "spacr.exe", "spacr-payload.json")
    native_names = ("Uninstall.exe", "spacr-uninstaller.json")
    native_state = None
    candidate = source / native_names[0]
    if os.path.lexists(candidate):
        details = candidate.lstat()
        if (not stat.S_ISREG(details.st_mode)
                or getattr(details, "st_file_attributes", 0) & 0x400
                or details.st_size == 0 or os.path.lexists(source / native_names[1])):
            raise ValueError("frozen_uninstaller_candidate_invalid")
        native_state = _frozen_windows_entry_state(candidate)
        names += native_names
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / ("." + destination.name + ".spacr-install-lock")
    lock.mkdir(mode=0o700)
    transaction = None
    moved, activated = [], {}
    complete = False
    restored = True
    try:
        previous = _frozen_windows_previous_version(destination)
        if previous is not None and _macos_online_version(previous) > _macos_online_version(version):
            raise ValueError("frozen_application_downgrade_refused")
        previous_state = _windows_frozen_root_state(str(destination)) if before_activate is not None else None
        if native_state:
            for name in native_names:
                if os.path.lexists(destination / name) and not stat.S_ISREG((destination / name).lstat().st_mode):
                    raise ValueError("frozen_uninstaller_candidate_invalid")
        previous_native = {name: _frozen_windows_entry_state(destination / name)
                           if os.path.lexists(destination / name) else None for name in native_names} if native_state else None
        transaction = Path(tempfile.mkdtemp(prefix=".spacr-frozen-install-", dir=destination.parent))
        staged, backup = transaction / "new", transaction / "previous"
        shutil.copytree(source, staged)
        if native_state is not None:
            if (_frozen_windows_entry_state(candidate) != native_state
                    or _frozen_windows_entry_state(staged / native_names[0])[2] != native_state[2]):
                raise ValueError("frozen_uninstaller_candidate_changed")
            with open(staged / native_names[1], "x", encoding="utf-8") as stream:
                json.dump({"schema": 1, "adapter": "windows-frozen-uninstall-v1",
                           "sha256": native_state[2]}, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            staged_native = {name: _frozen_windows_entry_state(staged / name) for name in native_names}
        _frozen_windows_payload(staged, version)
        _frozen_windows_identity(staged, version, run)
        _frozen_windows_payload(staged, version)
        if before_activate is not None:
            before_activate()
            if _windows_frozen_root_state(str(destination)) != previous_state:
                raise ValueError(tr("The installed application changed during staging; activation was stopped."))
        if previous_native is not None:
            for name, expected in previous_native.items():
                actual = _frozen_windows_entry_state(destination / name) if os.path.lexists(destination / name) else None
                if actual != expected or _frozen_windows_entry_state(staged / name) != staged_native[name]:
                    raise ValueError("frozen_uninstaller_changed_during_staging")
        backup.mkdir()
        destination.mkdir(exist_ok=True)
        old_manifest = None
        try:
            if previous is not None:
                old_manifest = _frozen_windows_payload(destination, previous, retirement=True)
        except (OSError, ValueError, TypeError):
            pass
        with open(transaction / "recovery.json", "x", encoding="utf-8") as stream:
            json.dump({"schema": 1, "destination": str(destination), "version": version,
                       "payload": names}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        for name in names:
            target = destination / name
            if target.exists() or target.is_symlink():
                os.replace(target, backup / name)
                moved.append(name)
        for name in names:
            entry_state = _frozen_windows_entry_state(staged / name)
            os.replace(staged / name, destination / name)
            activated[name] = entry_state
        _frozen_windows_payload(destination, version)
        _frozen_windows_identity(destination, version, run)
        if after_activate is not None:
            after_activate()
        complete = True
        retained = str(backup) if moved else None
        if old_manifest is not None and not preserve_previous:
            try:
                _frozen_windows_payload(backup, previous, retirement=True)
            except (OSError, ValueError, TypeError):
                pass
            else:
                try:
                    shutil.rmtree(backup)
                    retained = None
                except OSError:
                    pass
        return {"ok": True, "version": version, "previous_runtime": retained}
    except _UpdateActivationUnconfirmed:
        restored = False
        raise
    except BaseException:
        for name in reversed(activated):
            try:
                if _frozen_windows_entry_state(destination / name) != activated[name]:
                    restored = False
                    continue
                os.replace(destination / name, transaction / "new" / name)
            except (OSError, ValueError):
                restored = False
        for name in reversed(moved):
            try:
                if not os.path.lexists(destination / name):
                    os.replace(transaction / "previous" / name, destination / name)
                else:
                    restored = False
            except OSError:
                restored = False
        raise
    finally:
        if transaction is not None and restored:
            backup = transaction / "previous"
            if not complete or not backup.exists() or not any(backup.iterdir()):
                shutil.rmtree(transaction, ignore_errors=True)
        if restored:
            lock.rmdir()


def _windows_frozen_root_state(root):
    """Bind the actual install directory and regular payload without following reparse points."""
    root = _installer_metadata_folder(root, "windows_frozen_root", windows=True)
    state = []
    for path, directory in ((os.path.dirname(root), True), (root, True),
                            (os.path.join(root, "_internal"), True),
                            (os.path.join(root, "spacr.exe"), False)):
        details = os.lstat(path)
        if (getattr(details, "st_file_attributes", 0) & 0x400
                or stat.S_ISLNK(details.st_mode)
                or not (stat.S_ISDIR(details.st_mode) if directory else stat.S_ISREG(details.st_mode))):
            raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_root"))
        state.append((details.st_dev, details.st_ino) if directory else
                     (details.st_dev, details.st_ino, details.st_size,
                      details.st_mtime_ns, details.st_ctime_ns))
    return root, state


def _windows_frozen_update_record(records, version, workdir, machine):
    """Select one real frozen Windows installation and refuse an unverified version or downgrade."""
    candidates = [record for record in records if record.running]
    if (machine.platform != "windows" or len(candidates) != 1
            or candidates[0].kind != "installer" or candidates[0].platform != "windows"
            or candidates[0].layout != "windows-offline"):
        raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_layout"))
    record = candidates[0]
    root, _state = _windows_frozen_root_state(record.root)
    if _inside(os.path.realpath(workdir), root) or _inside(root, os.path.realpath(workdir)):
        raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_helper_location"))
    payload = [os.path.join(root, name) for name in ("_internal", "spacr.exe", "spacr-payload.json")]
    if any(_inside(os.path.realpath(path), owned) or _inside(owned, os.path.realpath(path))
           for path in _user_data(machine) for owned in payload):
        raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_user_data_overlap"))
    previous = _frozen_windows_previous_version(root)
    if (previous is None or record.version is not None and previous != record.version
            or _macos_online_version(previous) > _macos_online_version(version)):
        raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_version"))
    return record, previous


@contextmanager
def _windows_frozen_installer_pin(path, expected):
    """Hash one held regular installer while native Windows sharing denies writes and replacement."""
    import re

    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_published_checksum"))
    path = _installer_metadata_folder(path, "windows_frozen_installer", windows=True)
    before = os.lstat(path)
    if (not stat.S_ISREG(before.st_mode) or getattr(before, "st_file_attributes", 0) & 0x400
            or not 0 < before.st_size <= 8 * 1024 ** 3):
        raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_installer"))
    _windows_pip_private_access(path)
    if os.name == "nt":
        import ctypes
        import msvcrt

        kernel = _windows_update_library("kernel32")
        create = kernel.CreateFileW
        create.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                           ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        create.restype = ctypes.c_void_p
        close = kernel.CloseHandle
        close.argtypes = [ctypes.c_void_p]
        close.restype = ctypes.c_int32
        handle = create(path, 0x80000000, 1, None, 3, 0x08200000, None)
        if handle is None or handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        except BaseException:
            close(handle)
            raise
    else:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        details = os.fstat(stream.fileno())
        if (not stat.S_ISREG(details.st_mode)
                or getattr(details, "st_file_attributes", 0) & 0x400
                or (details.st_dev, details.st_ino) != (before.st_dev, before.st_ino)):
            raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_installer_identity"))
        digest = hashlib.sha256()
        total = 0
        while True:
            block = stream.read(min(1024 * 1024, 8 * 1024 ** 3 - total + 1))
            if not block:
                break
            total += len(block)
            if total > 8 * 1024 ** 3:
                raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_installer_size"))
            digest.update(block)
        after = os.fstat(stream.fileno())
        current = os.lstat(path)
        identity = (details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns, details.st_ctime_ns)
        if (identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                or (current.st_dev, current.st_ino) != identity[:2]
                or digest.hexdigest() != expected):
            raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_installer_checksum"))
        yield stream


def _windows_frozen_update_journal(directory, plan, state, **details):
    """Publish only this coordinator's private transaction state, retaining uncertain native outcomes."""
    _windows_pip_private_directory(directory)
    descriptor, temporary = tempfile.mkstemp(prefix=".journal-", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(schema=1, state=state, version=plan["version"],
                           workdir=plan["workdir"], **details), stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, os.path.join(directory, "journal.json"))
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def _run_windows_frozen_update(plan, machine, *, download=None, install=None, run=None, wait=None, spawn=None):
    """Stage a mandatory-phase official updater, then activate and confirm the new desktop unelevated."""
    if (not plan.get("frozen_application") or plan.get("adapter") != "windows-frozen-nsis-v1"
            or _windows_process_elevated()):
        raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_coordinator"))
    handshake = _WindowsPipHandshake(plan)
    records = [_record_from_json(data) for data in plan["records"]]
    record, previous = _windows_frozen_update_record(records, plan["version"], plan["workdir"], machine)
    root, initial = _windows_frozen_root_state(record.root)
    if _inside(os.path.realpath(sys.executable), root):
        raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_outside_helper"))
    name = "spaCR-" + str(plan["version"]) + "-update.exe"
    installer = os.path.join(handshake.root, name)
    url = _RELEASE_DOWNLOAD + "/v" + str(plan["version"]) + "/" + name
    relaunch = [os.path.join(root, "spacr.exe")]
    if (plan.get("fetch") != {"url": url, "path": installer}
            or plan.get("install") != [installer, "/S", "/D=" + root]
            or plan.get("relaunch") != relaunch):
        raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_plan"))
    lock_parent = os.path.join(machine.home, ".spacr", "update-locks")
    _installer_metadata_folder(lock_parent, "windows_frozen_lock", windows=True)
    os.makedirs(lock_parent, mode=0o700, exist_ok=True)
    _windows_pip_private_directory(lock_parent)
    lock = os.path.join(lock_parent, hashlib.sha256(os.path.normcase(root).encode()).hexdigest())
    _take_lease_dir(lock)
    retained = False
    try:
        _windows_pip_private_directory(lock)
        handshake.check()
        expected = _published_digest(url)
        if os.path.lexists(installer):
            raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_existing_stage"))
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_published_checksum"))
        (download or _download)(url, installer)
        with _windows_frozen_installer_pin(installer, expected):
            _windows_frozen_update_record(records, plan["version"], plan["workdir"], machine)
            if _windows_frozen_root_state(root) != (root, initial):
                raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_changed_destination"))
            _windows_frozen_update_journal(lock, plan, "preparing", root=root, previous=previous, installer=installer, sha256=expected)
            retained = True

            def _check_current():
                """Recheck the pinned installer and unchanged Windows destination before activation."""
                if _macos_digest(installer) != expected:
                    raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_installer_checksum"))
                if (_windows_process_elevated() or _windows_frozen_root_state(root) != (root, initial)
                        or _windows_frozen_update_record(records, plan["version"], plan["workdir"], machine)[1] != previous):
                    raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_changed_destination"))

            def _launch_verified():
                """Confirm the activated Windows payload and its unelevated main window."""
                observed_root, observed_state = _windows_frozen_root_state(root)
                if observed_root != root or observed_state[:2] != initial[:2]:
                    raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_changed_destination"))
                _frozen_windows_payload(root, plan["version"])
                identity = _frozen_windows_identity(root, plan["version"], run)
                if _windows_process_elevated():
                    raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_elevated_relaunch"))
                _windows_frozen_update_journal(lock, plan, "verified", root=root, identity=identity)
                return _launch_updated_desktop(plan, relaunch, identity["version"], root, frozen=True, spawn=spawn)

            try:
                output, desktop = _coordinate_native_update(plan, handshake,
                    lambda endpoint: (install or _windows_frozen_install)(
                        installer, root, phase_channel=endpoint, desktop_pid=plan["pid"]),
                    check=_check_current, launch=_launch_verified, wait=wait)
            except Exception as error:
                _windows_frozen_update_journal(lock, plan, "desktop-unconfirmed", root=root,
                    pid=getattr(error, "pid", None), error=str(error))
                if handshake._read("cancelled.json") is not None and _windows_frozen_root_state(root) == (root, initial):
                    retained = False
                raise
            _windows_frozen_update_journal(lock, plan, "desktop-ready", root=root, identity=desktop)
            retained = False
            return desktop["version"], output
    except TimeoutError as error:
        if retained:
            try:
                _windows_frozen_update_journal(lock, plan, "installer-uncertain", root=root,
                                               pid=getattr(error, "pid", None), error=str(error))
            except OSError:
                pass
        raise
    except OSError as error:
        if getattr(error, "winerror", error.errno) == 1223:
            retained = False
            _windows_frozen_update_journal(lock, plan, "authorization-cancelled", root=root)
        raise
    finally:
        if not retained:
            try:
                journal = os.path.join(lock, "journal.json")
                if os.path.isfile(journal) and not os.path.islink(journal):
                    result_path = os.path.join(handshake.root, "windows-update-result-" + handshake.binding + ".json")
                    with open(journal, "rb") as source, open(result_path, "xb") as result:
                        shutil.copyfileobj(source, result)
                        result.flush()
                        os.fsync(result.fileno())
                    os.unlink(journal)
                _release_lease_dir(lock)
            except OSError:
                pass


def start_update_helper(records: Sequence[InstallRecord], version: str, *,
                        ticked: Iterable[str] = (), pid: Optional[int] = None,
                        workdir: Optional[str] = None, system=None,
                        spawn=None) -> Dict:
    """Run an installation update in a process that outlives spaCR.

    Installer-made copies and supported dedicated POSIX or Windows pip
    environments can use this helper. It stages replacements while the current
    application runs and requires authenticated shutdown approval and actual
    process exit before activation.

    For a supported macOS online installation, use the existing bundled
    ``uv`` and private Python to install the requested spaCR release into a
    staged sibling environment. Preserve bootstrap files, the application
    launcher, and packages that need no update. After verified staging and
    approved shutdown, activate the staged environment at the existing runtime
    path and retain the complete previous environment for recovery. Relaunch
    verifies the requested version, executable, imported module and visible
    main window. Failed preparation leaves the running environment unchanged.

    Runtime adapters install and verify a private replacement before publishing
    readiness. The GUI offers Restart and preserves its settings before
    approving shutdown. Relaunch verifies the actual new desktop process and
    its main window. Unsupported layouts refuse before starting an installer.

    :param records: installations from :func:`find_old_installs`.
    :param version: the version to install.
    :param ticked: roots of the environments the user ticked.
    :param pid: the process to wait for; this process when ``None``.
    :param workdir: the helper's folder; a new temporary folder when ``None``.
    :param system: the computer; ``None`` means this one.
    :param spawn: replaces the detached process launcher.
    :returns: the plan as written to ``plan.json``. ``plan["command"]`` is the
        helper's command, or ``None`` with ``plan["error"]`` saying why
        nothing was started.

    Frozen macOS, Windows and protocol-marked Debian installations use their
    verified native replacement adapters. Protected macOS requires an installed
    helper with the current native phase protocol. Windows requires a separately
    published, checksummed update executable. Administrator authorization and
    staging precede Restart; activation follows approved desktop exit. The new
    desktop must confirm its identity and main window before recovery files are
    retired. Unsupported layouts or missing capabilities leave the running
    application unchanged; no ordinary-installer fallback is enabled.
    """
    machine = system or _Machine()
    records = list(records)
    running = next((r for r in records if r.kind == "installer" and r.running), None)
    if running is None:
        running = next((r for r in records if r.kind == "environment" and r.running), None)
    environment = None
    workdir = workdir or tempfile.mkdtemp(prefix="spacr-update-")
    os.makedirs(workdir, mode=0o700, exist_ok=True)
    module = os.path.join(workdir, "install_cleanup.py")
    plan_path = os.path.join(workdir, "plan.json")
    plan: Dict = {
        "schema": 1,
        "steps": ["wait", "fetch", "install", "verify", "retire", "relaunch"],
        "pid": int(pid if pid is not None else os.getpid()),
        "version": str(version),
        "records": [asdict(r) for r in records],
        "ticked": sorted(str(t) for t in ticked),
        "log": os.path.join(machine.home, ".spacr", "logs", "update.log"),
        "workdir": workdir,
        "fetch": None, "install": None, "relaunch": None,
        "command": None, "error": None,
        "frozen_application": bool(getattr(sys, "frozen", False)),
    }
    if running is None:
        plan["error"] = "the running spaCR is not an installer-made copy"
    elif running.kind == "environment":
        plan["adapter"] = "pip-private-windows-v1" if machine.platform == "windows" else "pip-private-posix-v1"
        plan["steps"] = ["wait", "upgrade", "verify", "relaunch"]
        plan["handshake"] = {"schema": 1, "token": os.urandom(32).hex(),
                             "expires": time.time() + _FROZEN_PREPARATION_SECONDS}
        try:
            if plan["frozen_application"] or len([r for r in records if r.kind == "environment" and r.running]) != 1:
                raise ValueError(_PRIVATE_PIP_UPDATE_ERROR)
            plan["install"], plan["relaunch"] = _pip_environment_commands(
                running, str(version), workdir, machine)
            (_WindowsPipHandshake if machine.platform == "windows" else _FrozenUpdateHandshake)(plan)
            with open(__file__, "rb") as source, open(module, "wb") as copy:
                copy.write(source.read())
            if machine.platform == "windows":
                plan["command"], environment = _windows_pip_helper(running, module, plan_path, machine)
            else:
                plan["command"] = [running.python, "-I", module, "run-plan", plan_path]
        except (OSError, ValueError) as error:
            plan["error"] = str(error)
    elif _macos_online_update_record(records, system=machine) is not None:
        plan["adapter"] = "macos-online-uv-v1"
        plan["steps"] = ["wait", "upgrade", "verify", "relaunch"]
        plan["handshake"] = {"schema": 1, "token": os.urandom(32).hex(),
                             "expires": time.time() + _FROZEN_PREPARATION_SECONDS}
        try:
            running = _macos_online_update_record(records, system=machine, strict=True)
            plan["install"], _probe, plan["relaunch"] = _macos_online_commands(
                running, str(version), workdir)
            _FrozenUpdateHandshake(plan)
            with open(__file__, "rb") as source, open(module, "wb") as copy:
                copy.write(source.read())
            plan["command"] = [running.python, "-I", module, "run-plan", plan_path]
        except (OSError, ValueError) as error:
            plan["error"] = f"could not prepare the macOS environment update: {error}. Nothing was removed."
    elif _windows_online_update_record(records, system=machine) is not None:
        plan["adapter"] = "windows-online-uv-v1"
        plan["steps"] = ["wait", "upgrade", "verify", "relaunch"]
        plan["handshake"] = {"schema": 1, "token": os.urandom(32).hex(),
                             "expires": time.time() + _FROZEN_PREPARATION_SECONDS}
        plan["desktop"] = {"prefix": machine.running_prefix, "executable": machine.executable}
        try:
            running = _windows_online_update_record(records, system=machine, strict=True)
            plan["install"], plan["relaunch"] = _windows_online_commands(
                running, str(version), workdir, machine.running_prefix, machine.executable)
            _WindowsPipHandshake(plan)
            with open(__file__, "rb") as source, open(module, "wb") as copy:
                copy.write(source.read())
            plan["command"], environment = _windows_online_helper(running, module, plan_path, machine)
        except (OSError, ValueError) as error:
            plan["error"] = f"could not prepare the Windows environment update: {error}. Nothing was removed."
    elif (plan["frozen_application"] and running.layout == "linux-deb"
          and machine.platform == "linux"):
        plan["adapter"] = "linux-deb-v1"
        plan["steps"] = ["prepare", "wait", "authorize", "install", "verify", "relaunch"]
        plan["handshake"] = {"schema": 1, "token": os.urandom(32).hex(),
                             "expires": time.time() + _FROZEN_PREPARATION_SECONDS}
        try:
            plan["install"], plan["relaunch"] = _debian_update_commands(records, str(version), machine)
            _FrozenUpdateHandshake(plan)
            argv, environment, error = _standalone_helper_command(records, workdir, plan_path, machine)
            plan["command"], plan["error"] = argv, error
        except (OSError, ValueError) as error:
            plan["error"] = str(error)
    elif (plan["frozen_application"] and running.layout == "windows-offline"
          and machine.platform == "windows"):
        plan["adapter"] = "windows-frozen-nsis-v1"
        plan["steps"] = ["prepare", "wait", "authorize", "install", "verify", "relaunch"]
        plan["handshake"] = {"schema": 1, "token": os.urandom(32).hex(),
                             "expires": time.time() + _FROZEN_PREPARATION_SECONDS}
        try:
            running, _previous = _windows_frozen_update_record(records, str(version), workdir, machine)
            if (_windows_process_elevated()
                    or os.path.normcase(os.path.realpath(machine.executable)) != os.path.normcase(os.path.join(os.path.realpath(running.root), "spacr.exe"))):
                raise ValueError(tr("Upgrade unavailable: {error}", error="windows_frozen_coordinator"))
            _WindowsPipHandshake(plan)
            name = "spaCR-" + str(version) + "-update.exe"
            target = os.path.join(os.path.realpath(workdir), name)
            plan["fetch"] = {"url": _RELEASE_DOWNLOAD + "/v" + str(version) + "/" + name, "path": target}
            plan["install"] = [target, "/S", "/D=" + os.path.realpath(running.root)]
            plan["relaunch"] = [os.path.join(os.path.realpath(running.root), "spacr.exe")]
            argv, environment, error = _standalone_helper_command(records, workdir, plan_path, machine)
            plan["command"], plan["error"] = argv, error
        except (OSError, ValueError) as error:
            plan["error"] = str(error)
    elif (plan["frozen_application"] and running.layout == "macos-app"
          and running.root.endswith(".app") and machine.platform == "macos"):
        plan["adapter"] = "macos-frozen-v1"
        plan["handshake"] = {"schema": 1, "token": os.urandom(32).hex(),
                             "expires": time.time() + _FROZEN_PREPARATION_SECONDS}
        argv, environment, error = _standalone_helper_command(records, workdir, plan_path, machine)
        plan["command"], plan["error"] = argv, error
    elif plan["frozen_application"] or _requires_frozen_adapter(running):
        plan["error"] = (
            "this installation requires a verified frozen replacement adapter; "
            "run the matching installer instead. Nothing was removed."
        )
    elif not str(__file__).endswith(".py") or not os.path.isfile(__file__):
        plan["error"] = (
            "standalone updater source is unavailable in this installation; "
            "run the new installer instead. Nothing was removed."
        )
    else:
        plan["error"] = tr("This installation cannot stage a verified update. Nothing was changed.")
    with open(plan_path, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
    if plan["command"]:
        (spawn or _spawn_detached)(plan["command"], environment, workdir)
    return plan


def _wait_for_exit(pid: int, timeout: float = _WAIT_SECONDS, *,
                   os_name: Optional[str] = None, kill=None, sleep=None,
                   clock=None) -> bool:
    """Wait for a process to end.

    :param pid: the process id.
    :param timeout: seconds to wait.
    :param os_name: :data:`os.name` of the computer; this one when ``None``.
    :param kill: replaces :func:`os.kill`, used with signal 0 to ask whether
        the process still exists.
    :param sleep: replaces :func:`time.sleep`.
    :param clock: replaces :func:`time.monotonic`.
    :returns: whether it ended in time.
    """
    clock = clock or time.monotonic
    kill = kill or os.kill
    sleep = sleep or time.sleep
    deadline = clock() + timeout
    if (os_name or os.name) == "nt":
        if not 0 < int(pid) <= 0xFFFFFFFF:
            return False
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int32
        handle = kernel32.OpenProcess(0x00100000, False, int(pid))
        if not handle:
            return kernel32.GetLastError() == 87
        try:
            return kernel32.WaitForSingleObject(handle, int(timeout * 1000)) == 0
        finally:
            kernel32.CloseHandle(handle)
    while clock() < deadline:
        try:
            kill(int(pid), 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        sleep(0.5)
    return False


#: The checksums every release publishes beside its assets, as a file name.
#: `release.yml` has uploaded it since 1.5.0.5; a release without one is
#: handled rather than refused, because an older release must still be
#: installable.
_SUMS_NAME = "SHA256SUMS.txt"


def _published_digest(url: str) -> Optional[str]:
    """The sha256 this release publishes for the asset at ``url``.

    WHAT THIS IS AND IS NOT. The sums file comes from the same server as
    the installer, so it is not a defence against a compromised release --
    an attacker who can replace one can replace the other. It catches what
    actually happens: a truncated or corrupted download, a proxy serving
    something stale, an asset that is not the one the plan named. That is
    worth having before a file is made executable and run.

    :param url: the installer's download address.
    :returns: the expected hex digest, or None when the release publishes
        no sums file or names no line for this asset.
    """
    import urllib.error
    import urllib.request

    base, _, name = url.rpartition("/")
    request = urllib.request.Request(f"{base}/{_SUMS_NAME}",
                                     headers={"User-Agent": "spacr-updater"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            published = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    for line in published.splitlines():
        parts = line.split()
        if len(parts) >= 2 and os.path.basename(parts[-1]) == name:
            return parts[0].strip().lower()
    return None


def _download(url: str, path: str) -> None:
    """Download a release asset over HTTPS and check what arrived.

    THE FILE IS ABOUT TO BE MADE EXECUTABLE AND RUN, so what arrived is
    checked against the digest the release publishes before that happens.
    A mismatch deletes the file and raises, and the caller treats that as a
    failed fetch -- which means nothing is removed and no installer runs,
    because `_run_plan` fetches before it deletes anything.

    Hashed while it is written rather than read back afterwards: the file
    is up to 40 MB and there is no reason to read it twice.

    :param url: the asset's address on GitHub.
    :param path: where to write it.
    :raises OSError: when the download is empty or does not match the
        digest the release publishes for it.
    """
    if not url.startswith(_RELEASE_DOWNLOAD + "/"):
        raise ValueError(f"refusing to download from {url}")
    import urllib.request
    request = urllib.request.Request(url, headers={"User-Agent": "spacr-updater"})
    digest = hashlib.sha256()
    staged = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".spacr-download-",
                                         dir=os.path.dirname(os.path.abspath(path)),
                                         delete=False) as handle:
            staged = handle.name
            with urllib.request.urlopen(request, timeout=120) as response:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
        if os.path.getsize(staged) == 0:
            raise OSError(f"{url} was empty")
        expected = _published_digest(url)
        if expected and digest.hexdigest() != expected:
            raise OSError(
                f"{os.path.basename(path)} does not match the checksum this "
                f"release publishes for it: expected {expected}, got "
                f"{digest.hexdigest()}. Nothing was installed and nothing was "
                f"removed.")
        os.chmod(staged, 0o755)
        os.replace(staged, path)
    finally:
        if staged is not None:
            try:
                os.unlink(staged)
            except FileNotFoundError:
                pass


def _run_plan(plan_path: str, *, wait=None, fetch=None, run=None, remove=None,
              spawn=None, system=None) -> int:
    """Carry out a plan written by :func:`start_update_helper`.

    :param plan_path: the plan file.
    :param wait: replaces the wait for spaCR to exit.
    :param fetch: replaces the installer download.
    :param run: replaces the command runner used for the install.
    :param remove: replaces :func:`remove_install`.
    :param spawn: replaces the launcher used to start the new version.
    :param system: the computer; ``None`` means this one.
    :returns: the helper's exit code; ``0`` when replacement succeeds and relaunch
        is requested. Native LaunchServices acceptance does not confirm GUI startup.
    """
    with open(plan_path, encoding="utf-8") as handle:
        plan = json.load(handle)
    machine = system or _Machine()
    lines: List[str] = [f"spaCR update to {plan.get('version')}, "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')}"]

    def _finish(code: int) -> int:
        """Write the update log and return ``code``.

        :param code: the exit code to return.
        """
        log = plan.get("log")
        if log:
            try:
                os.makedirs(os.path.dirname(log), exist_ok=True)
                with open(log, "a", encoding="utf-8") as out:
                    out.write("\n".join(lines) + "\n\n")
            except OSError:
                pass
        print("\n".join(lines))
        return code

    planned_records = [_record_from_json(data) for data in plan["records"]]
    if plan.get("adapter") in {"macos-online-uv-v1", "windows-online-uv-v1", "pip-private-posix-v1", "pip-private-windows-v1"}:
        try:
            operation = (_run_pip_environment_update if plan["adapter"] in {"pip-private-posix-v1", "pip-private-windows-v1"}
                         else _run_windows_online_update if plan["adapter"] == "windows-online-uv-v1"
                         else _run_macos_online_update)
            channel = (_WindowsPipHandshake if plan["adapter"] in {"pip-private-windows-v1", "windows-online-uv-v1"}
                       else _FrozenUpdateHandshake)(plan)
            with _preparation_heartbeat(channel):
                observed, output = operation(plan, machine, wait=wait, run=run, spawn=spawn)
        except Exception as error:
            lines.append(f"macOS environment update stopped: {error}" if plan["adapter"] == "macos-online-uv-v1"
                         else f"Windows environment update stopped: {error}" if plan["adapter"] == "windows-online-uv-v1"
                         else str(error))
            try:
                (_WindowsPipHandshake if plan["adapter"] in {"pip-private-windows-v1", "windows-online-uv-v1"}
                 else _FrozenUpdateHandshake)(plan).failed(error)
            except Exception:
                lines.append("The update error could not be published to the readiness channel.")
            return _finish(6)
        lines.extend([output, f"Verified spaCR {observed} in the existing environment; relaunch requested."])
        return _finish(0)
    if plan.get("adapter") == "windows-frozen-nsis-v1":
        try:
            with _preparation_heartbeat(_WindowsPipHandshake(plan)):
                observed, output = _run_windows_frozen_update(
                    plan, machine, wait=wait, download=fetch, run=run, spawn=spawn)
            lines.extend([output, json.dumps({"version": observed, "relaunch": "ready"})])
            return _finish(0)
        except Exception as error:
            lines.append(str(error))
            try:
                _WindowsPipHandshake(plan).failed(error)
            except Exception:
                pass
            return _finish(6)
    if plan.get("adapter") == "linux-deb-v1":
        try:
            observed, output = _run_debian_update(plan, machine, wait=wait, run=run, spawn=spawn)
            lines.extend([output, "Verified Debian spaCR " + observed + "; updated desktop ready."])
            return _finish(0)
        except Exception as error:
            if plan.get("handshake"):
                try:
                    _FrozenUpdateHandshake(plan).failed(error)
                except Exception:
                    lines.append("The Debian readiness error could not be published.")
            lines.append("Debian update stopped: " + str(error))
            return _finish(6)
    if plan.get("adapter") == "macos-frozen-v1":
        try:
            if machine.platform != "macos":
                raise ValueError("a macOS frozen update requires its actual native platform")
            if not plan.get("handshake"):
                raise ValueError("the frozen update plan has no readiness handshake")
            receipt = _run_macos_frozen_update(plan, wait=wait)
            lines.append(json.dumps(receipt, sort_keys=True))
            if receipt["relaunch"]["status"] != "ready":
                lines.append("The bundle replacement committed, but its new desktop did not confirm startup; the backup and journal are retained.")
                return _finish(6)
        except Exception as error:
            if plan.get("handshake"):
                try:
                    _FrozenUpdateHandshake(plan).failed(error)
                except Exception:
                    lines.append("The readiness error could not be published; shutdown remains unapproved.")
            lines.append(f"Frozen macOS update stopped: {error}")
            return _finish(6)
        return _finish(0)
    if plan.get("frozen_application") or any(
            record.running and _requires_frozen_adapter(record)
            for record in planned_records):
        lines.append("A verified frozen replacement adapter is required; nothing was changed.")
        return _finish(2)
    if not (wait or _wait_for_exit)(int(plan["pid"])):
        lines.append("spaCR did not close, so nothing was changed.")
        return _finish(3)
    if plan.get("fetch"):
        try:
            (fetch or _download)(plan["fetch"]["url"], plan["fetch"]["path"])
        except Exception as exc:                                 # noqa: BLE001
            lines.append(f"The new installer could not be downloaded ({exc}); "
                         f"nothing was removed.")
            return _finish(4)
    records = [replace(_record_from_json(d), running=False) for d in plan["records"]]
    lines.append("Found:")
    lines.extend(_format_records(records))
    runner = run or _run
    result = runner(plan["install"])
    code = int(result[0] if isinstance(result, tuple) else result)
    if code != 0:
        lines.append(f"The installer failed with exit code {code}.")
        return _finish(code)
    previous = [record for record in planned_records
                if record.kind == "installer" and record.running]
    if len(previous) != 1 or not previous[0].python:
        lines.append("The desktop interpreter could not be identified; no relaunch was requested.")
        return _finish(7)
    active = previous[0]
    identity_code, identity_output = runner(_runtime_identity_command(active.python, plan["version"]))
    try:
        identity = json.loads(identity_output)
        verified = (identity_code == 0 and identity.get("verified") is True
                    and identity.get("requested_version") == plan["version"])
    except (ValueError, TypeError, AttributeError):
        verified = False
    if not verified:
        lines.append("The installed desktop application could not be verified; no relaunch was requested.")
        return _finish(7)
    reports, retirement = run_update_sequence(
        lambda: result, records=[record for record in records
                                if _norm(record.root) != _norm(active.root)],
        ticked=plan.get("ticked") or (), remove=remove, system=machine)
    lines.extend(_format_reports(reports))
    if retirement is None:
        lines.append("The installed application was verified, but an older copy could not be retired.")
        return _finish(5)
    lines.append("Installed.")
    if plan.get("relaunch"):
        (spawn or _spawn_detached)(plan["relaunch"], None, plan.get("workdir"))
    return _finish(0)


def _prepare_windows_uninstall(root, workdir, *, machine=None, base_prefix=None):
    """Copy only this installer's private Windows Python outside the root before removal."""
    machine = machine or _Machine()
    root, workdir = os.path.abspath(root), os.path.abspath(workdir)
    base = os.path.realpath(base_prefix if base_prefix is not None else sys.base_prefix)
    expected = os.path.join(root, "venv", "Scripts", "python.exe")
    if (machine.platform != "windows" or os.path.realpath(root) != root
            or _norm(machine.executable) != _norm(expected)
            or _norm(machine.running_prefix or "") != _norm(os.path.join(root, "venv"))
            or not _inside(base, os.path.join(root, "python"))
            or not os.path.isfile(os.path.join(base, "python.exe"))
            or os.path.realpath(workdir) != workdir or _inside(workdir, root)
            or not os.path.isdir(workdir) or os.listdir(workdir)):
        raise ValueError("uninstall preparation requires this installer's private Python and a fresh outside folder")
    if os.name != "nt":
        details = os.stat(workdir)
        if details.st_uid != os.geteuid() or stat.S_IMODE(details.st_mode) & 0o077:
            raise ValueError("uninstall preparation requires its private owner folder")
    if any(_exists(path) and _inside(os.path.realpath(path), base) for path in _user_data(machine)):
        raise ValueError("user data inside the managed Python must remain untouched")
    junction = getattr(os.path, "isjunction", None)
    for directory, folders, files in os.walk(base, followlinks=False):
        for name in folders + files:
            path = os.path.join(directory, name)
            if os.path.islink(path) or (junction is not None and junction(path)):
                raise ValueError("linked managed Python content cannot be copied for uninstall")
    runtime = os.path.join(workdir, "runtime")
    shutil.copytree(base, runtime)
    module = os.path.join(workdir, "install_cleanup.py")
    shutil.copy2(__file__, module)
    command = [os.path.join(runtime, "python.exe"), "-I", module, "remove",
               "--only-root", root, "--require-outside"]
    return {"schema": 1, "workdir": workdir, "root": root, "command": command}



def _installer_metadata_folder(folder, code, *, windows=None):
    """Resolve native Windows aliases while refusing symbolic or reparse folder components."""
    root = os.path.abspath(folder)
    canonical = os.path.realpath(root)
    if windows is None:
        windows = os.name == "nt"
    if not windows:
        if root != canonical:
            raise ValueError({"code": code, "path": root})
        return root
    cursor = root
    while True:
        try:
            details = os.lstat(cursor)
        except FileNotFoundError:
            pass
        else:
            if (stat.S_ISLNK(details.st_mode)
                    or getattr(details, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                    or not os.path.samefile(cursor, os.path.realpath(cursor))):
                raise ValueError({"code": code, "path": root})
        parent = os.path.dirname(cursor)
        if parent == cursor:
            break
        cursor = parent
    return canonical


def _installer_metadata_snapshot(folder, paths):
    """Copy explicit installer sidecars without following links or changing their sources."""
    root = _installer_metadata_folder(folder, "linked_metadata_folder")
    selected = sorted(set(os.path.join(os.path.realpath(os.path.dirname(os.path.abspath(path))),
                                       os.path.basename(path)) for path in paths))
    if not 0 < len(selected) <= 32 or any(_inside(path, root) for path in selected):
        raise ValueError({"code": "invalid_metadata_selection"})
    os.mkdir(root, 0o700)
    details = os.stat(root)
    state = {"schema": 1, "family": "installer-metadata", "folder": root,
             "identity": [details.st_dev, details.st_ino], "records": []}
    for index, path in enumerate(selected):
        record = {"path": path, "parent": os.path.dirname(path), "kind": "absent"}
        if os.path.lexists(path):
            details = os.lstat(path)
            backup = os.path.join(root, str(index))
            if stat.S_ISLNK(details.st_mode):
                target = os.readlink(path)
                os.symlink(target, backup)
                record.update(kind="link", backup=str(index), target=target)
            elif stat.S_ISREG(details.st_mode):
                shutil.copy2(path, backup)
                record.update(kind="file", backup=str(index), sha256=_macos_digest(backup))
            else:
                raise ValueError({"code": "unsupported_metadata_entry", "path": path})
        state["records"].append(record)
    descriptor, temporary = tempfile.mkstemp(prefix="metadata-", dir=root)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, os.path.join(root, "metadata.json"))
    return state


def _installer_metadata_restore(folder):
    """Validate retained sidecars before atomically restoring their original files or absence."""
    root = _installer_metadata_folder(folder, "untrusted_metadata_folder")
    details = os.lstat(root)
    if (not stat.S_ISDIR(details.st_mode)
            or (os.name != "nt" and (details.st_uid != os.geteuid()
                                     or stat.S_IMODE(details.st_mode) & 0o077))):
        raise ValueError({"code": "untrusted_metadata_folder", "path": root})
    journal_path = os.path.join(root, "metadata.json")
    if os.path.islink(journal_path):
        raise ValueError({"code": "linked_metadata_journal"})
    descriptor = os.open(journal_path, os.O_RDONLY
                         | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, encoding="utf-8") as stream:
        journal = os.fstat(stream.fileno())
        if not stat.S_ISREG(journal.st_mode) or journal.st_size > 16 * 1024 * 1024:
            raise ValueError({"code": "invalid_metadata_journal"})
        state = json.load(stream)
    if (state.get("schema") != 1 or state.get("family") != "installer-metadata"
            or state.get("folder") != root or state.get("identity") != [details.st_dev, details.st_ino]
            or not isinstance(state.get("records"), list) or not 0 < len(state["records"]) <= 32):
        raise ValueError({"code": "invalid_metadata_binding"})
    seen = set()
    for record in state["records"]:
        path = record["path"]
        if (not os.path.isabs(path) or path in seen or _inside(path, root)
                or os.path.dirname(path) != record["parent"]
                or os.path.realpath(os.path.dirname(path)) != record["parent"]):
            raise ValueError({"code": "changed_metadata_parent", "path": path})
        seen.add(path)
        kind = record["kind"]
        if kind == "absent":
            continue
        name = record["backup"]
        if not isinstance(name, str) or not name.isdecimal() or str(int(name)) != name:
            raise ValueError({"code": "invalid_metadata_backup"})
        backup = os.path.join(root, name)
        details = os.lstat(backup)
        if kind == "file":
            if not stat.S_ISREG(details.st_mode) or _macos_digest(backup) != record["sha256"]:
                raise ValueError({"code": "changed_metadata_backup", "path": backup})
        elif kind == "link":
            if not stat.S_ISLNK(details.st_mode) or os.readlink(backup) != record["target"]:
                raise ValueError({"code": "changed_metadata_backup", "path": backup})
        else:
            raise ValueError({"code": "invalid_metadata_kind"})
    for record in state["records"]:
        path = record["path"]
        if record["kind"] == "absent":
            if os.path.lexists(path):
                os.unlink(path)
            continue
        descriptor, temporary = tempfile.mkstemp(prefix=".spacr-metadata-", dir=record["parent"])
        os.close(descriptor)
        try:
            if record["kind"] == "link":
                os.unlink(temporary)
                os.symlink(record["target"], temporary)
            else:
                shutil.copy2(os.path.join(root, record["backup"]), temporary)
            os.replace(temporary, path)
        finally:
            if os.path.lexists(temporary):
                os.unlink(temporary)
    return {"schema": 1, "state": "restored", "folder": root, "paths": sorted(seen)}


def _cleanup_translate(source, **values):
    """Use the application catalog when available while keeping standalone helpers usable."""
    if _PRIVILEGED_DEBIAN_HELPER:
        return str(source).format(**values)
    try:
        from spacr.qt.i18n import tr as application_tr
        return application_tr(source, **values)
    except Exception:
        return str(source).format(**values)


tr = _cleanup_translate


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line used by the installers and by the update helper.

    :param argv: arguments; :data:`sys.argv` when ``None``.
    :returns: the exit code; ``1`` when an installer-made copy was not removed,
        or ``6`` when retained-runtime recovery or its launch request fails.
    """
    global _PRIVILEGED_DEBIAN_HELPER

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["native-phase-protocol"]:
        print(json.dumps({"schema": 1, "commands": list(_MACOS_FROZEN_PHASE_COMMANDS)}))
        return 0
    if arguments and arguments[0] == "apply-macos-root":
        root_parser = argparse.ArgumentParser(prog="spacr-update-helper")
        root_parser.add_argument("command", choices=("apply-macos-root",))
        for name in ("version", "digest", "image", "phase-channel"):
            root_parser.add_argument("--" + name, required=True)
        root_parser.add_argument("--desktop-pid", type=int, required=True)
        root_args = root_parser.parse_args(arguments)
        _PRIVILEGED_DEBIAN_HELPER = True
        try:
            print(_macos_apply_root_transaction(root_args.version, root_args.digest, root_args.image,
                root_args.phase_channel, root_args.desktop_pid), end="")
            return 0
        except Exception as error:
            print(json.dumps({"state": "stopped", "error": str(error)}, sort_keys=True))
            return 6
        finally:
            _PRIVILEGED_DEBIAN_HELPER = False
    if arguments and arguments[0] == "uninstall-macos-root":
        root_parser = argparse.ArgumentParser(prog="spacr-update-helper")
        root_parser.add_argument("command")
        root_parser.add_argument("--phase-channel", required=True)
        root_parser.add_argument("--desktop-pid", type=int, required=True)
        root_parser.add_argument("--desktop-executable", required=True)
        root_parser.add_argument("--previous-version", required=True)
        root_parser.add_argument("--app-identity", required=True)
        root_parser.add_argument("--keep-json", required=True)
        root_args = root_parser.parse_args(arguments)
        _PRIVILEGED_DEBIAN_HELPER = True
        try:
            return _uninstall_macos_root(root_args.phase_channel, root_args.desktop_pid,
                root_args.desktop_executable, root_args.previous_version, root_args.app_identity, root_args.keep_json)
        except Exception as error:
            print(json.dumps({"state": "stopped", "error": str(error)}, sort_keys=True))
            return 6
        finally:
            _PRIVILEGED_DEBIAN_HELPER = False
    if arguments and arguments[0] in {"prepare-debian-root", "apply-debian-root"}:
        root_parser = argparse.ArgumentParser(prog="spacr-update-helper")
        root_parser.add_argument("command", choices=("prepare-debian-root", "apply-debian-root"))
        root_parser.add_argument("--version", required=True)
        root_parser.add_argument("--previous-version", required=True)
        root_parser.add_argument("--desktop-uid", type=int, required=True)
        root_parser.add_argument("--stage")
        root_parser.add_argument("--phase-channel")
        root_parser.add_argument("--desktop-pid", type=int)
        root_args = root_parser.parse_args(arguments)
        _PRIVILEGED_DEBIAN_HELPER = True
        try:
            if (root_args.phase_channel is None) != (root_args.desktop_pid is None):
                raise ValueError(tr("The update authorization descriptor is not private."))
            if root_args.command == "prepare-debian-root":
                code, output = _debian_prepare_root_transaction(
                    root_args.version, root_args.previous_version, root_args.desktop_uid,
                    **({"phase_channel": root_args.phase_channel, "desktop_pid": root_args.desktop_pid}
                       if root_args.phase_channel is not None else {}))
                print(output)
                return code
            if root_args.stage is None:
                raise ValueError("a private Debian transaction directory is required")
            receipt = _debian_apply_root_transaction(
                root_args.stage, root_args.version, root_args.previous_version, root_args.desktop_uid,
                **({"phase_channel": root_args.phase_channel, "desktop_pid": root_args.desktop_pid}
                   if root_args.phase_channel is not None else {}))
            print(json.dumps(receipt, sort_keys=True))
            return 0
        except Exception as error:
            print(json.dumps({"state": "stopped", "error": str(error)}, sort_keys=True))
            return 6
        finally:
            _PRIVILEGED_DEBIAN_HELPER = False
    if arguments and arguments[0] == "uninstall-frozen-windows":
        root_parser = argparse.ArgumentParser(prog="spacr-update-helper")
        root_parser.add_argument("command")
        root_parser.add_argument("--root", required=True)
        root_parser.add_argument("--request", required=True)
        root_parser.add_argument("--desktop-pid", type=int, required=True)
        root_parser.add_argument("--desktop-executable", required=True)
        root_parser.add_argument("--phase-channel", required=True)
        root_args = root_parser.parse_args(arguments)
        _PRIVILEGED_DEBIAN_HELPER = True
        try:
            return _uninstall_frozen_windows(root_args.request, root_args.root,
                root_args.phase_channel, root_args.desktop_pid, root_args.desktop_executable)
        except Exception as error:
            print(json.dumps({"state": "stopped", "error": str(error)}, sort_keys=True))
            return 6
        finally:
            _PRIVILEGED_DEBIAN_HELPER = False
    if arguments and arguments[0] == "uninstall-debian-root":
        root_parser = argparse.ArgumentParser(prog="spacr-update-helper")
        root_parser.add_argument("command")
        root_parser.add_argument("--previous-version", required=True)
        root_parser.add_argument("--desktop-uid", type=int, required=True)
        root_parser.add_argument("--desktop-pid", type=int, required=True)
        root_parser.add_argument("--desktop-executable", required=True)
        root_parser.add_argument("--phase-channel", required=True)
        root_args = root_parser.parse_args(arguments)
        _PRIVILEGED_DEBIAN_HELPER = True
        try:
            return _uninstall_debian_root(root_args.phase_channel, root_args.desktop_pid,
                root_args.desktop_executable, root_args.desktop_uid, root_args.previous_version)
        except Exception as error:
            print(json.dumps({"state": "stopped", "error": str(error)}, sort_keys=True))
            return 6
        finally:
            _PRIVILEGED_DEBIAN_HELPER = False
    parser = argparse.ArgumentParser(
        prog="install_cleanup",
        description="Find, and remove, older spaCR installations.")
    commands = parser.add_subparsers(dest="command")
    frozen_windows = commands.add_parser("install-frozen-windows", help=argparse.SUPPRESS)
    frozen_windows.add_argument("--source", required=True)
    frozen_windows.add_argument("--destination", required=True)
    frozen_windows.add_argument("--version", required=True)
    frozen_windows.add_argument("--phase-channel")
    frozen_windows.add_argument("--desktop-pid", type=int)
    verify = commands.add_parser("verify-runtime", help="verify a desktop interpreter and imported application")
    verify.add_argument("--python", required=True)
    verify.add_argument("--version")
    verify.add_argument("--prefix")
    verify.add_argument("--minimum", action="store_true")
    verify.add_argument("--retiring-prefix")
    find = commands.add_parser("find", help="list every spaCR installation")
    find.add_argument("--json", action="store_true")
    find.add_argument("--root", default="/", help=argparse.SUPPRESS)
    remove = commands.add_parser("remove", help="remove every installer-made copy")
    remove.add_argument("--keep", action="append", default=[])
    remove.add_argument("--only-root", help=tr("remove only this recognized installation, preserving its data"))
    remove.add_argument("--require-outside", action="store_true", help=tr("require an interpreter outside the selected root"))
    remove.add_argument("--sudo", action="store_true")
    remove.add_argument("--root", default="/", help=argparse.SUPPRESS)
    remove.add_argument("--purge", action="store_true",
                        help="also delete caches, backends and user data")
    remove.add_argument("--yes", action="store_true",
                        help="purge without asking")
    purge = commands.add_parser(
        "purge", help="delete spaCR's caches, backends and user data")
    purge.add_argument("--yes", action="store_true", help="do not ask first")
    purge.add_argument("--dry-run", action="store_true", help="list only")
    purge.add_argument("--root", default="/", help=argparse.SUPPRESS)
    plan = commands.add_parser("run-plan", help="finish an in-app update")
    plan.add_argument("plan")
    uninstall_plan = commands.add_parser("run-uninstall-plan", help=tr("finish a confirmed uninstall after spaCR closes"))
    uninstall_plan.add_argument("plan")
    recover = commands.add_parser("recover-macos-frozen", help="recover one retained macOS bundle transaction")
    recover.add_argument("transaction")
    recover.add_argument("--protected", action="store_true")
    recover.add_argument("--version")
    recover.add_argument("--sha256")
    runtime_recover = commands.add_parser("recover-runtime", help=tr("restore one unchanged retained private POSIX runtime"))
    runtime_recover.add_argument("transaction")
    runtime_recover.add_argument("--pid", type=int, required=True, help=tr("previous desktop process that must have exited"))
    prepare_uninstall = commands.add_parser("prepare-windows-uninstall", help=tr("copy the private managed Python outside this installation"))
    prepare_uninstall.add_argument("--install-root", required=True)
    prepare_uninstall.add_argument("--workdir", required=True)
    uninstall_preview = commands.add_parser("uninstall-preview", help=tr("list every spaCR installation and its data for the installer window"))
    uninstall_preview.add_argument("--output", required=True)
    uninstall_apply = commands.add_parser("uninstall-apply", help=tr("remove every listed spaCR installation and the chosen data"))
    uninstall_apply.add_argument("--preview", required=True)
    uninstall_apply.add_argument("--categories", default="")
    metadata = commands.add_parser("installer-metadata", help=tr("preserve or restore installer-owned metadata"))
    metadata.add_argument("action", choices=("snapshot", "restore"))
    metadata.add_argument("--folder", required=True)
    metadata.add_argument("--paths", nargs="*")
    args = parser.parse_args(argv)
    if args.command == "run-uninstall-plan":
        try:
            return _run_uninstall_plan(args.plan)
        except Exception as error:
            print(tr("Uninstall stopped: {error}", error=str(error)))
            return 1
    if args.command == "install-frozen-windows":
        try:
            if args.phase_channel is not None:
                with _UpdatePhaseClient(args.phase_channel, args.desktop_pid,
                        os.path.join(args.destination, "spacr.exe")) as channel:
                    outcome = _install_frozen_windows(args.source, args.destination, args.version,
                        before_activate=lambda: _native_update_ready(channel, args.version, args.desktop_pid),
                        after_activate=lambda: _native_update_activated(channel, args.version), preserve_previous=True)
            else:
                if args.desktop_pid is not None:
                    raise ValueError(tr("The update authorization descriptor is not private."))
                outcome = _install_frozen_windows(args.source, args.destination, args.version)
        except Exception as error:
            print(json.dumps({"ok": False, "error": str(error)}))
            return 1
        print(json.dumps(outcome, sort_keys=True))
        return 0
    if args.command == "uninstall-preview":
        import uuid

        preview = _uninstall_preview(uuid.uuid4().hex, system=_Machine(running_prefix=""))
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(preview, handle, indent=2)
        for row in preview["installations"]:
            record = row["record"]
            action = tr("Remove") if row["selectable"] else tr("Kept")
            print(f"{action}: {record['root']} ({record['version'] or tr('Unknown')})")
        for row in preview["cleanup"]:
            print(f"  [{row['category']}] {row['path']}")
        return 0
    if args.command == "uninstall-apply":
        try:
            with open(args.preview, encoding="utf-8") as handle:
                preview = json.load(handle)
            categories = [key for key in args.categories.split(",") if key]
            outcome = _installer_uninstall_apply(preview, categories)
        except Exception as error:
            print(tr("Uninstall stopped: {error}", error=str(error)))
            return 1
        print(json.dumps(outcome, sort_keys=True))
        return 0 if outcome["ok"] else 1
    if args.command == "installer-metadata":
        try:
            receipt = (_installer_metadata_snapshot(args.folder, args.paths or [])
                       if args.action == "snapshot" else _installer_metadata_restore(args.folder))
        except Exception as error:
            print(json.dumps({"ok": False, "error": str(error)}))
            return 1
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "prepare-windows-uninstall":
        try:
            receipt = _prepare_windows_uninstall(args.install_root, args.workdir)
        except Exception as error:
            print(tr("Uninstall preparation stopped: {error}", error=str(error)))
            return 1
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "recover-runtime":
        try:
            receipt = _recover_runtime_update(args.transaction, args.pid)
        except Exception as error:
            print(tr("Runtime recovery stopped: {error}", error=str(error)))
            return 6
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "verify-runtime":
        if args.retiring_prefix:
            retained = [path for path in _user_data(_Machine()) if _exists(path)
                        and _inside(os.path.realpath(path), os.path.realpath(args.retiring_prefix))]
            if retained:
                print(json.dumps({"verified": False, "reason": "user_data_inside_runtime",
                                  "preserved_paths": retained}))
                return 2
        try:
            previous = _retiring_runtime_identity(args.retiring_prefix) if args.retiring_prefix else None
        except (OSError, ValueError, subprocess.SubprocessError):
            print(json.dumps({"verified": False, "reason": "previous_runtime_identity_unverified"}))
            return 2
        code, output = _run(_runtime_identity_command(
            args.python, args.version, prefix=args.prefix, minimum=args.minimum,
            previous_version=previous["version"] if previous else None), timeout=30)
        print(output.strip())
        return code
    if args.command == "recover-macos-frozen":
        try:
            receipt = _run_macos_frozen_recovery(
                args.transaction, protected=args.protected,
                version=args.version, digest=args.sha256)
        except Exception as error:
            print(f"Frozen macOS recovery stopped: {error}")
            return 6
        print(json.dumps(receipt, sort_keys=True))
        return 0 if receipt["relaunch"]["status"] == "accepted" else 6
    if args.command == "run-plan":
        return _run_plan(args.plan)
    if args.command not in ("find", "remove", "purge"):
        parser.print_help()
        return 2
    if args.root not in ("/", ""):
        machine = _Machine(environ=dict(os.environ), fs_root=args.root,
                           sudo=getattr(args, "sudo", False))
    else:
        machine = _Machine(sudo=getattr(args, "sudo", False))
    if args.command == "purge":
        return _purge_command(machine, yes=args.yes, dry_run=args.dry_run)
    machine.running_prefix = None
    if args.command == "remove" and args.require_outside:
        if (not args.only_root or any(_inside(os.path.realpath(path), os.path.realpath(args.only_root))
                                     for path in (sys.prefix, sys.executable, __file__))):
            print(tr("Uninstall requires a helper and interpreter outside the selected installation."))
            return 1
    records = find_old_installs(system=machine)
    if args.command == "remove" and args.only_root:
        requested = os.path.abspath(args.only_root)
        selected = [record for record in records if record.kind == "installer"
                    and os.path.abspath(record.root) == requested]
        if (args.purge or os.path.realpath(requested) != requested or len(selected) != 1):
            print(tr("Targeted removal requires one recognized canonical installation and cannot purge user data."))
            return 1
        records = selected
    if args.command == "find":
        if args.json:
            print(json.dumps([asdict(r) for r in records], indent=2))
        else:
            print(f"Found {len(records)} spaCR installation(s).")
            print("\n".join(_format_records(records)))
        return 0
    print(f"Finding older spaCR installations: {len(records)} found.")
    for line in _format_records(records):
        print(line)
    reports, _result = run_update_sequence(
        lambda: 0, records=[r for r in records if r.kind == "installer"],
        keep=[os.path.abspath(k) for k in args.keep], system=machine)
    for line in _format_reports(reports):
        print(line)
    if any(not r.ok for r in reports):
        print("An older spaCR could not be removed, so nothing new was installed.")
        return 1
    if args.purge:
        return _purge_command(machine, yes=args.yes)
    return 0


_OPTIONAL_CLEANUP_ACTIONS = set()
_OPTIONAL_CLEANUP_ACTION_LOCK = threading.Lock()


def _optional_cleanup_identity(details):
    """Describe a non-linked cache entry without reading its payload."""
    if stat.S_ISLNK(details.st_mode):
        raise ValueError("link")
    return (details.st_dev, details.st_ino, details.st_mode, details.st_size,
            details.st_mtime_ns, details.st_ctime_ns)


def _optional_cleanup_directory(path):
    """Open each absolute directory component without following a link."""
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("no_nofollow_directory_support")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(os.sep, flags)
    try:
        for component in os.path.abspath(path).split(os.sep):
            if component:
                child = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _optional_cleanup_tree(descriptor, key):
    """Inventory only bounded writer-known entries through held directory descriptors."""
    import re
    from types import MappingProxyType

    patterns = {
        "news": r"releases\.json",
        "backgrounds": r"space-[A-Za-z0-9_-]+-[0-9]+x[0-9]+-s-?[0-9]+-v[0-9]+\.png",
        "icons": r".+-[0-9a-f]{20}\.png",
        "sounds": r"[A-Za-z0-9_-]+-[0-9a-f]{12}/(?:click-[0-3]|hover-[0-3]|run_finished|run_failed|bed)\.wav",
    }
    root_identity = _optional_cleanup_identity(os.fstat(descriptor))
    entries = [("", root_identity)]
    contents = []
    total = 0
    with os.scandir(descriptor) as scan:
        for entry in scan:
            if len(entries) >= 512:
                raise ValueError("budget")
            item = _optional_cleanup_identity(os.stat(entry.name, dir_fd=descriptor, follow_symlinks=False))
            entries.append((entry.name, item))
            if stat.S_ISDIR(item[2]) and key == "sounds":
                if not re.fullmatch(r"[A-Za-z0-9_-]+-[0-9a-f]{12}", entry.name):
                    raise ValueError("layout")
                child_fd = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                try:
                    if _optional_cleanup_identity(os.fstat(child_fd)) != item:
                        raise ValueError("changed")
                    with os.scandir(child_fd) as children:
                        for child in children:
                            if len(entries) >= 512:
                                raise ValueError("budget")
                            relative = entry.name + "/" + child.name
                            details = _optional_cleanup_identity(os.stat(child.name, dir_fd=child_fd, follow_symlinks=False))
                            entries.append((relative, details))
                            if not stat.S_ISREG(details[2]) or not re.fullmatch(patterns[key], relative):
                                raise ValueError("layout")
                            total += details[3]
                            contents.append(MappingProxyType({"name": relative, "size_bytes": details[3]}))
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(item[2]) and re.fullmatch(patterns[key], entry.name):
                total += item[3]
                contents.append(MappingProxyType({"name": entry.name, "size_bytes": item[3]}))
            else:
                raise ValueError("layout")
    if _optional_cleanup_identity(os.fstat(descriptor)) != root_identity:
        raise ValueError("changed")
    contents.sort(key=lambda row: row["name"])
    return root_identity, tuple(sorted(entries)), tuple(contents), total


def preview_optional_cleanup(operation_id, *, system=None, guard=None):
    """Inspect dedicated caches and offer selection only under a live run guard.

    Recognized cache contents are inspected only under the default spaCR home.
    A bounded relocation manifest may also be read from a configured spaCR home
    to display retained external paths. External, symlinked, unknown and
    protected cache contents remain unavailable.
    Inspection is bounded and never follows links. A trusted live application
    guard can enable candidates; a saved selection cannot authorize removal.
    This does not establish that every other application instance is idle.

    :param operation_id: Fresh identity supplied by the current installer or
        update coordinator, independent of saved privacy preferences.
    :param system: Optional machine adapter for isolated filesystem checks.
    :param guard: Optional trusted live context-manager factory receiving derived
        candidate paths and yielding ``None`` only while cleanup is permitted.
    :returns: An immutable operation-bound preview with actual paths, content
        sizes and a snapshot identity. Without a live guard, removal stays
        disabled. No file is changed during inspection.
    :raises ValueError: The operation identity is empty or invalid.
    """
    from types import MappingProxyType

    if not isinstance(operation_id, str) or not operation_id.strip():
        raise ValueError("operation_id")
    machine = system or _Machine()
    default_home = os.path.join(machine.home, ".spacr")
    configured_home = machine.env("SPACR_HOME").strip()
    cache_home = configured_home or default_home
    rows = []
    identities = []
    variables = {"news": "SPACR_NEWS_CACHE", "backgrounds": "SPACR_SPACE_CACHE",
                 "sounds": "SPACR_SOUND_CACHE", "icons": "SPACR_ICON_CACHE"}

    def normalized(value, *, expand=True):
        """Return an absolute display path using the writer's tilde rule.

        :param value: Configured or default path text.
        :param expand: Whether this writer expands a leading user-home marker.
        :returns: Absolute path text without inspecting cache contents.
        """
        if expand and (value == "~" or value.startswith("~/") or value.startswith("~\\")):
            value = machine.home + value[1:]
        elif expand:
            value = os.path.expanduser(value)
        return os.path.abspath(value)

    cache_home = normalized(cache_home)
    locations = {}
    try:
        home_fd = _optional_cleanup_directory(cache_home)
        try:
            descriptor = os.open(_CACHE_LOCATIONS_FILE, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0), dir_fd=home_fd)
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                details = os.fstat(stream.fileno())
                if stat.S_ISREG(details.st_mode) and details.st_size <= 1_048_576:
                    raw = stream.read(1_048_577)
                    data = json.loads(raw) if len(raw) <= 1_048_576 else None
                    if isinstance(data, dict):
                        locations = {key: value for key, value in data.items()
                                     if isinstance(value, str) and value.strip() and "\x00" not in value}
        finally:
            os.close(home_fd)
    except (OSError, ValueError, RecursionError):
        pass

    for key, variable in variables.items():
        override = machine.env(variable) or locations.get(variable, "")
        if key == "sounds":
            override = override.strip()
        path = normalized(override, expand=False) if override else os.path.join(cache_home, key)
        contents = []
        reason, size, selectable = "unverified", None, False
        try:
            if configured_home or override or not os.path.isabs(default_home):
                raise ValueError("external")
            home_fd = _optional_cleanup_directory(default_home)
            try:
                home_identity = _optional_cleanup_identity(os.fstat(home_fd))
                root_fd = os.open(key, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=home_fd)
                try:
                    root_identity, entries, contents_tuple, total = _optional_cleanup_tree(root_fd, key)
                    if _optional_cleanup_identity(os.stat(key, dir_fd=home_fd, follow_symlinks=False)) != root_identity:
                        raise ValueError("changed")
                    contents = list(contents_tuple)
                finally:
                    os.close(root_fd)
                if _optional_cleanup_identity(os.fstat(home_fd)) != home_identity:
                    raise ValueError("changed")
            finally:
                os.close(home_fd)
            identities.append((key, home_identity, entries))
            size, selectable, reason = total, False, "coordinator_unavailable"
        except FileNotFoundError:
            reason = "missing"
            contents = []
        except (OSError, ValueError):
            contents = []
        rows.append(MappingProxyType({"key": key, "path": path, "size_bytes": size,
                                     "contents": tuple(contents), "selectable": selectable,
                                     "reason": reason}))
    cache = normalized(machine.env("XDG_CACHE_HOME").strip() or os.path.join(machine.home, ".cache"), expand=False)

    def retained_path(variable, default):
        """Resolve a retained cache path with external override precedence.

        :param variable: Registered cache environment-variable name.
        :param default: Fallback when neither environment nor relocation supplies it.
        :returns: Normalized display path; its protected contents are not scanned.
        """
        value = machine.env(variable) or locations.get(variable, "")
        return normalized(value.strip() if value else default)

    protected = {
        "models": retained_path("SPACR_MODEL_CACHE", os.path.join(cache_home, "models")),
        "backends": retained_path("SPACR_BACKENDS_DIR", os.path.join(cache_home, "backends")),
        "huggingface": retained_path("HF_HOME", os.path.join(cache, "huggingface")),
        "torch": retained_path("TORCH_HOME", os.path.join(cache, "torch")),
        "cellpose": retained_path("CELLPOSE_LOCAL_MODELS_PATH", os.path.join(machine.home, ".cellpose", "models")),
        "temporary": normalized(machine.env("TMPDIR") or tempfile.gettempdir(), expand=False),
        "data": os.path.join(machine.home, "spacr-demos"),
    }
    for key, path in protected.items():
        rows.append(MappingProxyType({"key": key, "path": path, "size_bytes": None,
                                     "contents": (), "selectable": False, "reason": "protected"}))
    inventory = tuple((row["key"], row["path"], row["size_bytes"],
                       tuple((entry["name"], entry["size_bytes"]) for entry in row["contents"]))
                      for row in rows)
    binding = hashlib.sha256(repr((operation_id, identities, inventory)).encode()).hexdigest()
    candidates = tuple(row["path"] for row in rows if row["reason"] == "coordinator_unavailable")
    available = False
    active = False
    if callable(guard) and candidates:
        try:
            with guard(candidates) as refusal:
                available = refusal is None
                active = not available
        except Exception:
            pass
    if available or active:
        rows = [MappingProxyType(dict(row, selectable=available,
                                      reason="" if available else "active"))
                if row["reason"] == "coordinator_unavailable" else row for row in rows]
    return MappingProxyType({"operation_id": operation_id, "snapshot_id": binding,
                             "folders": tuple(rows), "_snapshots": tuple(identities)})


def _optional_cleanup_remove(key, home_fd, root_fd, expected, path):
    """Detach one selected cache privately and remove only unchanged known entries.

    Failed or changed detached trees remain in their private sibling directory.
    Restoration never overwrites a concurrently created original leaf; this
    boundary retains the detached tree instead of attempting a racy restore.
    """
    import secrets

    name = ".spacr-cleanup-" + secrets.token_hex(16)
    private_fd = None
    private_identity = None
    renamed = False
    removed = 0
    retained = path
    try:
        os.mkdir(name, 0o700, dir_fd=home_fd)
        private_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=home_fd)
        private_identity = _optional_cleanup_identity(os.fstat(private_fd))
        root_identity = _optional_cleanup_identity(os.fstat(root_fd))
        if _optional_cleanup_identity(os.stat(key, dir_fd=home_fd, follow_symlinks=False)) != root_identity:
            raise ValueError("changed")
        os.rename(key, "cache", src_dir_fd=home_fd, dst_dir_fd=private_fd)
        renamed = True
        retained = os.path.join(os.path.dirname(path), name)
        detached = _optional_cleanup_identity(os.stat("cache", dir_fd=private_fd, follow_symlinks=False))
        if detached[:3] != root_identity[:3]:
            raise ValueError("changed")
        _root, actual, _contents, _size = _optional_cleanup_tree(root_fd, key)
        if [(entry, value[:5] if not entry else value) for entry, value in actual] != [
                (entry, value[:5] if not entry else value) for entry, value in expected]:
            raise ValueError("changed")
        os.mkdir(key, stat.S_IMODE(root_identity[2]), dir_fd=home_fd)
        for relative, details in expected:
            if not relative or not stat.S_ISREG(details[2]):
                continue
            directory, separator, leaf = relative.partition("/")
            parent_fd = root_fd
            child_fd = None
            try:
                if separator:
                    child_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
                    expected_directory = dict(expected)[directory]
                    if _optional_cleanup_identity(os.fstat(child_fd))[:3] != expected_directory[:3]:
                        raise ValueError("changed")
                    parent_fd = child_fd
                else:
                    leaf = directory
                current = _optional_cleanup_identity(os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False))
                if current != details:
                    raise ValueError("changed")
                os.unlink(leaf, dir_fd=parent_fd)
                removed += 1
            finally:
                if child_fd is not None:
                    os.close(child_fd)
        for relative, details in reversed(expected):
            if not relative or not stat.S_ISDIR(details[2]):
                continue
            current = _optional_cleanup_identity(os.stat(relative, dir_fd=root_fd, follow_symlinks=False))
            if current[:3] != details[:3]:
                raise ValueError("changed")
            os.rmdir(relative, dir_fd=root_fd)
        current = _optional_cleanup_identity(os.stat("cache", dir_fd=private_fd, follow_symlinks=False))
        if current[:3] != root_identity[:3]:
            raise ValueError("changed")
        os.rmdir("cache", dir_fd=private_fd)
        current = _optional_cleanup_identity(os.stat(name, dir_fd=home_fd, follow_symlinks=False))
        if current[:3] != private_identity[:3]:
            raise ValueError("changed")
        os.rmdir(name, dir_fd=home_fd)
        return {"key": key, "ok": True, "removed": removed}
    except (OSError, ValueError):
        return {"key": key, "ok": False, "reason": "retained",
                "removed": removed, "retained_path": retained}
    finally:
        if private_fd is not None:
            os.close(private_fd)
        if not renamed and private_identity is not None:
            try:
                current = _optional_cleanup_identity(os.stat(name, dir_fd=home_fd, follow_symlinks=False))
                if current[:3] == private_identity[:3]:
                    os.rmdir(name, dir_fd=home_fd)
            except (OSError, ValueError):
                pass


def apply_optional_cleanup(preview, choices, *, system=None, guard=None):
    """Apply a fresh selected-cache action under the application's live run guard.

    Only a trusted Python context-manager factory can enable execution. The
    application's caller must hold its established run-registration lock and
    refuse active work throughout the context. This is an application-local
    safeguard, not proof that every other process is idle. A separate helper
    or saved dictionary cannot grant that authority.

    Every selected snapshot is validated before mutation. Each action identity
    is consumed once in this process. Changed, unknown or locked detached
    entries remain in private staging; no recursive deletion or overwriting
    restoration is performed. Models, data and other protected roots cannot
    be selected. Supported no-follow directory descriptors are required.

    :param preview: Immutable result of :func:`preview_optional_cleanup`.
    :param choices: Current operation ID, snapshot ID and explicitly chosen
        registered cache keys returned by the GUI.
    :param system: Optional isolated machine adapter used for reinspection.
    :param guard: Trusted live context-manager factory receiving derived paths;
        it yields ``None`` only while application-local cleanup is permitted.
    :returns: Action outcome with a refusal reason or removed counts and any private retained paths.
    """
    from collections.abc import Mapping
    from types import MappingProxyType

    if not isinstance(preview, MappingProxyType) or not isinstance(choices, Mapping):
        return {"ok": False, "reason": "invalid_selection"}
    keys = choices.get("keys")
    eligible = {row["key"] for row in preview["folders"]
                if row["selectable"] or row["reason"] == "coordinator_unavailable"}
    if (choices.get("operation_id") != preview["operation_id"]
            or choices.get("snapshot_id") != preview["snapshot_id"]
            or not isinstance(keys, (list, tuple)) or not keys
            or any(not isinstance(key, str) or key not in eligible for key in keys)
            or len(set(keys)) != len(keys)):
        return {"ok": False, "reason": "invalid_selection"}
    if not callable(guard):
        return {"ok": False, "reason": "coordinator_unavailable"}
    paths = {row["key"]: row["path"] for row in preview["folders"]}
    descriptors = []
    home_fd = None
    try:
        with guard(tuple(paths[key] for key in keys)) as refusal:
            if refusal is not None:
                return {"ok": False, "reason": "active"}
            current = preview_optional_cleanup(preview["operation_id"], system=system)
            current_paths = {row["key"]: row["path"] for row in current["folders"]}
            original = {key: (home, entries) for key, home, entries in preview["_snapshots"]}
            snapshots = {key: (home, entries) for key, home, entries in current["_snapshots"]}
            if any(key not in original or key not in snapshots
                   or current_paths[key] != paths[key]
                   or original[key][0][:3] != snapshots[key][0][:3]
                   or original[key][1] != snapshots[key][1] for key in keys):
                return {"ok": False, "reason": "preview_changed"}
            paths = current_paths
            snapshots = {key: entries for key, (_home, entries) in snapshots.items()}
            home_fd = _optional_cleanup_directory(os.path.dirname(paths[keys[0]]))
            for key in keys:
                descriptor = os.open(key, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=home_fd)
                descriptors.append((key, descriptor))
                _identity, entries, _contents, _size = _optional_cleanup_tree(descriptor, key)
                if entries != snapshots[key]:
                    return {"ok": False, "reason": "preview_changed"}
            with _OPTIONAL_CLEANUP_ACTION_LOCK:
                if preview["operation_id"] in _OPTIONAL_CLEANUP_ACTIONS:
                    return {"ok": False, "reason": "operation_used"}
                _OPTIONAL_CLEANUP_ACTIONS.add(preview["operation_id"])
            outcomes = [_optional_cleanup_remove(key, home_fd, descriptor, snapshots[key], paths[key])
                        for key, descriptor in descriptors]
            return {"ok": all(outcome["ok"] for outcome in outcomes),
                    "removed": sum(outcome["removed"] for outcome in outcomes),
                    "folders": outcomes}
    except (OSError, ValueError, TypeError):
        return {"ok": False, "reason": "coordinator_unavailable"}
    finally:
        for _key, descriptor in descriptors:
            os.close(descriptor)
        if home_fd is not None:
            os.close(home_fd)


if __name__ == "__main__":
    raise SystemExit(_main())
