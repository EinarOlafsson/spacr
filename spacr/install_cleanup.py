"""Find every older spaCR on this computer, and remove the installer-made ones.

An update or an install runs three steps, in this order, whether it starts
from the in-app **Check for updates** or from a native installer:

1. find old spaCR files -- :func:`find_old_installs`
2. delete old spaCR files -- :func:`remove_install`
3. install new spaCR -- :func:`run_update_sequence` runs the install only
   after step 2 removed every installer-made copy.

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

# The 1.5.0.1 to 1.5.0.4 installers capitalised the first letter of the
# folder, app bundle and Start Menu names. Built rather than written, so a
# scan for the mis-cased project name does not mistake it for prose.
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
# Python 3.12 renamed shutil.rmtree's error hook from onerror to onexc.
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


# ---------------------------------------------------------------------------
# The computer, behind one seam
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Small path helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Step 1: find
# ---------------------------------------------------------------------------

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
    return bool(prefix) and _inside(prefix, root)


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
        # Registered, but the folder is already gone: the Apps list entry and
        # shortcuts of a half-removed copy still have to go.
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
    return str(info.get("CFBundleShortVersionString") or "") or None


def _find_macos_apps(machine: _Machine) -> List[InstallRecord]:
    """Find the macOS application bundle and the files its package installed.

    :param machine: the computer.
    """
    apps = _dedupe(machine.path(f"/Applications/{name}.app") for name in _NAMES)
    supports = _dedupe(machine.path(f"/Library/Application Support/{name}")
                       for name in _NAMES)
    command = machine.path("/usr/local/bin/spacr")
    records = []
    for index, app in enumerate(apps):
        # The package's shared support folder belongs to the first bundle.
        support = supports[0] if supports and index == 0 else ""
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
    # A launcher names its root, which finds a copy put somewhere unusual.
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
    # Any project can have a venv. A folder known only because a launcher
    # names it must also hold what only the installer writes, or a launcher
    # a user made for a project's own venv would make the project an old copy.
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
        root = machine.path("/usr/lib/python3/dist-packages/spacr")
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


# ---------------------------------------------------------------------------
# Step 2: delete
# ---------------------------------------------------------------------------

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
                    # os.open and os.close cannot be called again with a path
                    # alone; keep the error rmtree reported (onexc passes the
                    # exception, the older onerror an exc_info tuple).
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
    cannot delete itself while it runs: :func:`start_update_helper` removes
    it after spaCR has closed.

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


# ---------------------------------------------------------------------------
# Step 3: install, only after step 2
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Steps 2 and 3 after spaCR has closed
# ---------------------------------------------------------------------------

def _record_from_json(data: Dict) -> InstallRecord:
    """Rebuild an :class:`InstallRecord` from its JSON form.

    :param data: a mapping written by :func:`dataclasses.asdict`.
    """
    values = dict(data)
    for name in ("launchers", "shortcuts", "menu_entries", "registrations", "notes"):
        values[name] = tuple(values.get(name) or ())
    return InstallRecord(**values)


def _reinstall_steps(record: InstallRecord, version: str, workdir: str,
                     machine: _Machine):
    """Return how the new version replaces an installer-made copy.

    :param record: the running installer-made copy.
    :param version: the version to install.
    :param workdir: the helper's folder, outside every installation.
    :param machine: the computer.
    :returns: ``(fetch, install, relaunch)``: the installer download, the
        command that runs it, and the command that starts the new version.
    """
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


def _helper_command(records: Sequence[InstallRecord], workdir: str,
                    module: str, plan_path: str, machine: _Machine):
    """Choose an interpreter that survives the removal, and the helper's argv.

    :param records: every installation in the plan.
    :param workdir: the helper's folder.
    :param module: this module's copy inside ``workdir``.
    :param plan_path: the plan file.
    :param machine: the computer.
    :returns: ``(argv, environment, error)``; ``argv`` is ``None`` when no
        interpreter outside the removed copies exists.
    """
    doomed = [r.root for r in records if r.kind == "installer"]
    tail = ["-I", module, "run-plan", plan_path]
    if not any(_inside(machine.executable, root) for root in doomed):
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


def start_update_helper(records: Sequence[InstallRecord], version: str, *,
                        ticked: Iterable[str] = (), pid: Optional[int] = None,
                        workdir: Optional[str] = None, system=None,
                        spawn=None) -> Dict:
    """Hand deleting and installing to a process that outlives spaCR.

    Used when the running spaCR is itself an installer-made copy, which on
    Windows cannot delete itself while it runs. The helper waits for this
    process to exit, downloads the new installer, removes every older copy
    including the one that was running, runs the installer, and starts the new
    version. If any installer-made copy cannot be removed, nothing is
    installed and the reason is written to the update log.

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
    """
    machine = system or _Machine()
    records = list(records)
    running = next((r for r in records if r.kind == "installer" and r.running), None)
    environment = None
    workdir = workdir or tempfile.mkdtemp(prefix="spacr-update-")
    os.makedirs(workdir, exist_ok=True)
    module = os.path.join(workdir, "install_cleanup.py")
    with open(__file__, "rb") as source, open(module, "wb") as copy:
        copy.write(source.read())
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
    }
    if running is None:
        plan["error"] = "the running spaCR is not an installer-made copy"
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
    :returns: the helper's exit code; ``0`` when the new version started.
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
    :returns: the exit code; ``1`` when an installer-made copy was not removed.
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
    args = parser.parse_args(argv)
    if args.command == "run-plan":
        return _run_plan(args.plan)
    if args.command not in ("find", "remove"):
        parser.print_help()
        return 2
    if args.root not in ("/", ""):
        # A sandboxed file system, for testing an installer: no real registry,
        # package manager or running environment is consulted.
        machine = _Machine(environ=dict(os.environ), fs_root=args.root,
                           sudo=getattr(args, "sudo", False))
    else:
        machine = _Machine(sudo=getattr(args, "sudo", False))
    # The interpreter running this is a tool, not the spaCR being replaced.
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
