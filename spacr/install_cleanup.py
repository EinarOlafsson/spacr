"""Find spaCR installations and run the appropriate update or removal procedure.

The standard replacement procedure runs three steps in order:

1. find old spaCR files -- :func:`find_old_installs`
2. delete old spaCR files -- :func:`remove_install`
3. install new spaCR -- :func:`run_update_sequence` runs the install only
   after step 2 removed every installer-made copy.

In-app updates use a different procedure for supported macOS installations.
An online installation upgrades spaCR in its existing Python environment
using its bundled ``uv`` executable. It keeps the environment, bootstrap
files, application launcher, and installed packages that need no update.
It verifies the installed spaCR version before restarting the application.

A supported frozen macOS application uses a verified DMG replacement and
retains its complete previous application bundle. These macOS update paths
do not run the standard deletion procedure. Other frozen application
families require a supported replacement adapter before an update can start.

Removal never runs an old version's own uninstaller, so a copy whose
``Uninstall.exe`` or ``uninstall-spacr.sh`` is missing or broken is removed
all the same. Preferences and user data are kept: ``~/.spacr``,
``~/.cache/spacr``, ``~/.local/state/spacr``, ``~/spacr-demos``,
``~/spacr-tutorials``, the Hugging Face cache, and the ``spacr``/``qt``
settings (``~/.config/spacr``, the macOS preferences file, and the
``Software\\spacr`` registry key on Windows).

A pip or conda environment the user made is listed but never deleted; the
spaCR package is uninstalled from it only when the caller says it was ticked.
An editable source checkout is reported and left alone.

The module uses only the standard library and imports nothing from spaCR, so
an installer can run it with its bootstrapped Python before any spaCR exists,
and a helper process can run a copy of it after the running spaCR has closed::

    python -I install_cleanup.py find [--json]
    python -I install_cleanup.py remove [--keep PATH] [--sudo]
    python -I install_cleanup.py run-plan PLAN.json
"""
from __future__ import annotations

import argparse
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
import time
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
_RMTREE_TAKES_ONEXC = sys.version_info >= (3, 12)


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
    records = []
    for root in roots:
        needles = _spellings(root)
        shortcuts = [link for link in _dedupe(
            os.path.join(menu, f"{name}.lnk") for name in _NAMES)
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
    paths = [
        os.path.join(home, ".spacr"),
        os.path.join(cache, "spacr"),
        os.path.join(machine.xdg("XDG_STATE_HOME", os.path.join(".local", "state")), "spacr"),
        os.path.join(home, "spacr-demos"),
        os.path.join(home, "spacr-tutorials"),
        machine.env("HF_HOME") or os.path.join(cache, "huggingface"),
        os.path.join(machine.xdg("XDG_CONFIG_HOME", ".config"), "spacr"),
        os.path.join(home, "Library", "Preferences", "com.spacr.qt.plist"),
    ]
    return [p for p in paths if p]


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
        if _inside(path, data) or _inside(data, path):
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


def remove_install(record: InstallRecord, *, ticked: bool = False,
                   keep: Sequence[str] = (), system=None) -> RemovalReport:
    """Remove one installation, without running its own uninstaller.

    An installer-made copy is removed completely: its folder, launchers,
    shortcuts, menu entries and uninstall registrations. A Debian package is
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
        _delete(path, list(keep), report, denied)
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
    dists = _distributions(record.root) or glob.glob(
        os.path.join(record.root, "site-packages", "spacr*-*.dist-info"))
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



def run_update_sequence(install: Callable[[], object], *,
                        records: Optional[Sequence[InstallRecord]] = None,
                        ticked: Iterable[str] = (), keep: Sequence[str] = (),
                        find=None, remove=None, system=None):
    """Find, then delete, then install -- and never install over an old copy.

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
  /usr/bin/lipo -verify_arch "$(/usr/bin/uname -m)" "$app/Contents/MacOS/spacr" || return 1
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

finish() {
  result=$?
  trap - EXIT HUP INT TERM
  set +e
  if [ "$phase" != initial ] && [ "$phase" != committed ] && [ "$phase" != rolled-back ] && [ -n "$old_identity" ] && [ -n "$new_identity" ]; then
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
phase=committed
write_journal
receipt_ready=1
"""


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
    script = "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items()) + "\n" + _MACOS_PROTECTED_TRANSACTION
    literal = script.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return ["/usr/bin/osascript", "-e",
            f'do shell script "{literal}" with administrator privileges without altering line endings']


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
    import re

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
        invoke(["/usr/bin/lipo", "-verify_arch", architecture, executable])
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
        raise OSError(error, os.strerror(error), left, right)


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

    def _write(self, name, state, error="", expires=None):
        """Atomically publish a complete message; interruption cannot expose half JSON.

        :param name: internal message filename inside the private helper folder.
        :param state: the protocol state being published.
        :param error: bounded diagnostic for a preparation failure.
        :param expires: a readiness-specific shutdown deadline, or the preparation deadline.
        """
        descriptor, temporary = tempfile.mkstemp(prefix="readiness-", dir=self.root)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"binding": self.binding, "state": state,
                       "error": str(error)[:2000],
                       "expires": self.expires if expires is None else expires}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, os.path.join(self.root, name))

    def check(self):
        """Refuse expired or cancelled work, even if ready or approved arrived later."""
        if self._read("cancelled.json") is not None:
            raise RuntimeError("frozen update cancelled; nothing was changed")
        message = self._read("readiness.json")
        deadline = (float(message["expires"]) if message is not None
                    and message.get("state") == "ready" else self.expires)
        if not 0 < deadline <= self.expires + _WAIT_SECONDS:
            raise ValueError("invalid frozen updater readiness deadline")
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


def _run_macos_frozen_update(plan, *, run=None, download=None, wait=None, swap=None):
    """Verify an official read-only DMG and replace a writable app without elevation.

    Cleanup and relaunch failures retain the committed replacement receipt.
    A zero open result proves LaunchServices acceptance, not GUI readiness.

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
    installers = [record for record in records if record.kind == "installer"]
    if (len(installers) != 1 or not installers[0].running or plan.get("ticked")
            or installers[0].layout != "macos-app" or not installers[0].root.endswith(".app")):
        raise ValueError("mixed or legacy macOS installations require a separately reviewed migration")
    target = installers[0].root
    if os.path.realpath(target) != target or not os.path.isdir(target):
        raise ValueError("the installed app must be an actual canonical directory")
    _macos_bundle_identity(target, run=invoke)
    if not os.access(os.path.dirname(target), os.W_OK | os.X_OK):
        raise ValueError("protected Applications replacement requires its reviewed native authorization adapter")
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
    return _macos_relaunch(result, original_uid, run=invoke)


def _macos_relaunch(result, original_uid, *, run=None):
    """Keep verified replacement/recovery state even when normal-user launch fails.

    :param result: verified transaction receipt naming its installed target.
    :param original_uid: normal user that began this replacement or recovery.
    :param run: optional native-command runner for the captured LaunchServices request.
    :returns: the same receipt with an accepted, failed or refused relaunch result.
    """
    invoke = run or _macos_native
    if os.geteuid() != original_uid or original_uid == 0:
        result["relaunch"] = {"status": "refused", "error": "updater identity changed; GUI relaunch requires the original normal user"}
        return result
    try:
        invoke(["/usr/bin/open", "-n", result["target"]])
    except Exception as launch_error:
        result["relaunch"] = {"status": "failed", "error": str(launch_error)}
    else:
        result["relaunch"] = {"status": "accepted"}
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
        """Bind recovery to actual directory inodes, not mutable app names alone.

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
                    os.path.join(root, "launch_spacr.pyw")]
    elif record.platform == "macos":
        install = ["open", "-W", target]
        relaunch = ["open", "-a", machine.unmapped(
            machine.path(f"/Applications/{_NAME}.app"))]
    else:
        install = ["bash", target, "--install-root", root, "--no-launch",
                   "--skip-system-deps"]
        relaunch = [os.path.join(root, "venv", "bin", "python"), "-m", "spacr.qt"]
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
                    *, os_name: Optional[str] = None, popen=None) -> int:
    """Start a process that keeps running after spaCR exits.

    :param argv: the command.
    :param env: its environment; inherited when ``None``.
    :param cwd: its working folder, which must not be inside an installation.
    :param os_name: :data:`os.name` of the computer; this one when ``None``.
    :param popen: replaces :class:`subprocess.Popen`.
    :returns: the new process id.
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
    return (popen or subprocess.Popen)([str(a) for a in argv], **options).pid



def _macos_online_update_record(records, *, system=None, strict=False):
    # Recognize a damaged bootstrap too: it must fail without entering cleanup.
    """The running macOS online environment's record, or None elsewhere."""
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


def _macos_online_commands(record, version, workdir):
    """The install, probe and relaunch commands for a macOS online update."""
    _macos_online_version(version)
    python = os.path.join(record.root, "venv", "bin", "python")
    uv = os.path.join(record.root, "bootstrap", "uv")
    if record.python is None or os.path.abspath(record.python) != os.path.abspath(python):
        raise ValueError("the running macOS environment does not match its private Python")
    for path in (uv, python):
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            raise ValueError(f"the existing macOS update executable is unavailable: {path}")
    if _inside(os.path.realpath(workdir), os.path.realpath(record.root)):
        raise ValueError("the updater folder must be outside the installed environment")
    install = [uv, "pip", "install", "--upgrade", "--python", python, "spacr"]
    probe = [python, "-I", "-c",
             "from importlib.metadata import version; print(version('spacr'))"]
    relaunch = [python, "-m", "spacr.qt"]
    return install, probe, relaunch


def _run_macos_online_update(plan, machine, *, wait=None, run=None, spawn=None):
    """Carry out an approved macOS online update in its own environment."""
    records = [_record_from_json(data) for data in plan["records"]]
    record = _macos_online_update_record(records, system=machine, strict=True)
    if record is None or plan.get("frozen_application"):
        raise ValueError("this update requires a running non-frozen macOS online environment")
    install, probe, relaunch = _macos_online_commands(record, plan["version"], plan["workdir"])
    if plan.get("fetch") is not None or plan.get("install") != install or plan.get("relaunch") != relaunch:
        raise ValueError("the macOS online update commands differ from the approved environment")
    handshake = _FrozenUpdateHandshake(plan)
    handshake.ready()
    handshake.wait_for_shutdown(wait)
    # Recheck the actual paths after shutdown; never substitute PATH's uv/Python.
    _macos_online_commands(record, plan["version"], plan["workdir"])
    runner = run or _run
    code, output = runner(install)
    if code:
        raise RuntimeError(f"The package upgrade failed with exit code {code}: {output}")
    code, observed = runner(probe)
    try:
        observed_version = _macos_online_version(observed.strip())
        offered_version = _macos_online_version(plan["version"])
        previous_version = _macos_online_version(record.version) if record.version else None
        verified = (code == 0 and observed_version >= offered_version
                    and (previous_version is None or observed_version > previous_version))
    except ValueError:
        verified = False
    if not verified:
        raise RuntimeError(
            f"The package command completed, but spaCR {plan['version']} could not be verified "
            f"in the existing environment: {observed.strip()}. No relaunch was requested.")
    (spawn or _spawn_detached)(relaunch, None, plan["workdir"])
    return observed.strip(), output


def start_update_helper(records: Sequence[InstallRecord], version: str, *,
                        ticked: Iterable[str] = (), pid: Optional[int] = None,
                        workdir: Optional[str] = None, system=None,
                        spawn=None) -> Dict:
    """Run an installation update in a process that outlives spaCR.

    The running spaCR must be an installer-made copy. The helper waits for
    this process to exit before changing the installation.

    For a supported macOS online installation, use the existing bundled
    ``uv`` and private Python to upgrade spaCR in the same environment.
    Preserve the environment directory, bootstrap files, application launcher,
    and installed packages that need no update. Restart only after verifying
    that the installed version is at least ``version`` and newer than the
    previous version, when that previous version is known. A failed upgrade
    or version check does not trigger removal or a replacement installer.

    The standard replacement path downloads the new installer, removes older
    installer-made copies, installs the new version, and starts it. If any
    required removal fails, skip installation and write the reason to the log.

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

    A frozen macOS bundle in a user-writable location uses its verified DMG
    replacement adapter and retains its complete prior bundle. Other frozen
    application families return an error before removing anything; the helper
    does not enable an online-installer fallback for these applications.
    """
    machine = system or _Machine()
    records = list(records)
    running = next((r for r in records if r.kind == "installer" and r.running), None)
    environment = None
    workdir = workdir or tempfile.mkdtemp(prefix="spacr-update-")
    os.makedirs(workdir, mode=0o700, exist_ok=True)
    module = os.path.join(workdir, "install_cleanup.py")
    plan_path = os.path.join(workdir, "plan.json")
    plan: Dict = {
        "schema": 1,
        "steps": ["wait", "fetch", "delete", "install", "relaunch"],
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
        try:
            with open(__file__, "rb") as source, open(module, "wb") as copy:
                copy.write(source.read())
        except OSError as error:
            plan["error"] = f"could not prepare the updater: {error}. Nothing was removed."
        else:
            plan["fetch"], plan["install"], plan["relaunch"] = _reinstall_steps(
                running, str(version), workdir, machine)
            argv, environment, error = _helper_command(
                records, workdir, module, plan_path, machine)
            plan["command"], plan["error"] = argv, error
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
        import ctypes
        kernel32 = ctypes.windll.kernel32                    # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x00100000, False, int(pid))
        if not handle:
            return True
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
    import urllib.request

    base, _, name = url.rpartition("/")
    request = urllib.request.Request(f"{base}/{_SUMS_NAME}",
                                     headers={"User-Agent": "spacr-updater"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            published = response.read().decode("utf-8", "replace")
    except Exception:                                        # noqa: BLE001
        return None
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
    with urllib.request.urlopen(request, timeout=120) as response, \
            open(path, "wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
    if os.path.getsize(path) == 0:
        raise OSError(f"{url} was empty")
    expected = _published_digest(url)
    if expected and digest.hexdigest() != expected:
        try:
            os.remove(path)
        except OSError:
            pass
        raise OSError(
            f"{os.path.basename(path)} does not match the checksum this "
            f"release publishes for it: expected {expected}, got "
            f"{digest.hexdigest()}. Nothing was installed and nothing was "
            f"removed.")
    os.chmod(path, 0o755)


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
    if plan.get("adapter") == "macos-online-uv-v1":
        try:
            observed, output = _run_macos_online_update(plan, machine, wait=wait, run=run, spawn=spawn)
        except Exception as error:
            lines.append(f"macOS environment update stopped: {error}")
            try:
                _FrozenUpdateHandshake(plan).failed(error)
            except Exception:
                lines.append("The update error could not be published to the readiness channel.")
            return _finish(6)
        lines.extend([output, f"Verified spaCR {observed} in the existing environment; relaunch requested."])
        return _finish(0)
    if plan.get("adapter") == "macos-frozen-v1":
        try:
            if machine.platform != "macos":
                raise ValueError("a macOS frozen update requires its actual native platform")
            if not plan.get("handshake"):
                raise ValueError("the frozen update plan has no readiness handshake")
            receipt = _run_macos_frozen_update(plan, wait=wait)
            lines.append(json.dumps(receipt, sort_keys=True))
            if receipt["relaunch"]["status"] != "accepted":
                lines.append("The bundle replacement committed, but LaunchServices did not accept its relaunch; the backup and journal are retained.")
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
    reports, result = run_update_sequence(
        lambda: runner(plan["install"]), records=records,
        ticked=plan.get("ticked") or (), remove=remove, system=machine)
    lines.extend(_format_reports(reports))
    if result is None:
        lines.append("The update stopped before installing, because an older "
                     "copy could not be removed.")
        return _finish(5)
    code = int(result[0] if isinstance(result, tuple) else result)
    if code != 0:
        lines.append(f"The installer failed with exit code {code}.")
        return _finish(code)
    lines.append("Installed.")
    if plan.get("relaunch"):
        (spawn or _spawn_detached)(plan["relaunch"], None, plan.get("workdir"))
    return _finish(0)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line used by the installers and by the update helper.

    :param argv: arguments; :data:`sys.argv` when ``None``.
    :returns: the exit code; ``1`` when an installer-made copy was not removed,
        or ``6`` when frozen macOS recovery or its launch request fails.
    """
    parser = argparse.ArgumentParser(
        prog="install_cleanup",
        description="Find, and remove, older spaCR installations.")
    commands = parser.add_subparsers(dest="command")
    find = commands.add_parser("find", help="list every spaCR installation")
    find.add_argument("--json", action="store_true")
    find.add_argument("--root", default="/", help=argparse.SUPPRESS)
    remove = commands.add_parser("remove", help="remove every installer-made copy")
    remove.add_argument("--keep", action="append", default=[])
    remove.add_argument("--sudo", action="store_true")
    remove.add_argument("--root", default="/", help=argparse.SUPPRESS)
    plan = commands.add_parser("run-plan", help="finish an in-app update")
    plan.add_argument("plan")
    recover = commands.add_parser("recover-macos-frozen", help="recover one retained macOS bundle transaction")
    recover.add_argument("transaction")
    recover.add_argument("--protected", action="store_true")
    recover.add_argument("--version")
    recover.add_argument("--sha256")
    args = parser.parse_args(argv)
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
    if args.command not in ("find", "remove"):
        parser.print_help()
        return 2
    if args.root not in ("/", ""):
        machine = _Machine(environ=dict(os.environ), fs_root=args.root,
                           sudo=getattr(args, "sudo", False))
    else:
        machine = _Machine(sudo=getattr(args, "sudo", False))
    machine.running_prefix = None
    records = find_old_installs(system=machine)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
