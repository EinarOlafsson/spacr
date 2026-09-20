"""An update removes every older spaCR first (features item 416).

A machine with a 1.5.0.1 desktop install could not be updated. Both entry
points -- the in-app update and the native installers -- now run the same
three steps in the same order: find old spaCR files, delete them, install the
new version, through one finder and one remover (``spacr/install_cleanup.py``).

Every layout a released installer used is rebuilt here from the packaging
scripts at the v1.5.0.1, .4, .5, .6 and .7 tags:

    Windows online   %LOCALAPPDATA%\\<capital S>paCR  (NSIS InstallDir, every tag)
                     Start Menu\\spaCR\\, Desktop\\spaCR.lnk, HKCU Uninstall key,
                     HKCU\\Software\\spaCR InstallRoot; install-profile.json >= .5
    Windows offline  %PROGRAMFILES%\\<S|s>paCR, Start Menu\\<S|s>paCR.lnk
    Linux online     ~/.local/share/spacr, ~/.local/bin/spacr,
                     spacr.desktop (<= .4) or io.github.olafssonlab.spacr.desktop
    macOS package    /Applications/<S|s>paCR.app, /Library/Application Support/
                     <S|s>paCR, /usr/local/bin/spacr link, and the per-user
                     runtime under ~/Library/Application Support
    macOS disk image /Applications/<S|s>paCR.app
    Debian package   python3-spacr

and the finder must list it and the remover must leave nothing behind but
user data -- including a broken 1.5.0.1 whose own uninstaller is missing.

Nothing here touches this computer. Every path is below ``tmp_path``, the
registry and the package manager are fakes, the command runner records instead
of running, and a sandboxed machine refuses to delete outside the sandbox.
"""
from __future__ import annotations

import ast
import ctypes
import dataclasses
import errno
import importlib.util
import io
import json
import mimetypes  # noqa: F401 -- bound before the guard's winreg; see the guard
import os
import plistlib
import runpy
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import types
import urllib.request
import warnings
from pathlib import Path

import pytest

from spacr import install_cleanup as ic

ROOT = Path(__file__).resolve().parents[1]
#: The capitalised spelling the 1.5.0.1 to 1.5.0.4 installers used, built so a
#: scan for the mis-cased project name does not mistake it for prose.
OLD = "Spa" + "CR"
TAGS = ("1.5.0.1", "1.5.0.4", "1.5.0.5", "1.5.0.6", "1.5.0.7")

#: The module's own doors to this computer, kept before the guard below closes
#: them, for the tests that exercise those doors with harmless arguments.
_REAL_RUN = ic._run
_REAL_DELETE = ic._delete
_REAL_SPAWN = ic._spawn_detached

#: POSIX permission failures only happen to a user who is not root.
_PERMISSIONS_BIND = pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0,
    reason="needs POSIX permissions that bind the user running the tests")


def _capitalised(tag):
    return tag in ("1.5.0.1", "1.5.0.4")


def _refuse(what):
    def refuse(*args, **kwargs):
        raise AssertionError(f"a test reached the real {what}; pass a fake")
    return refuse


class _RefusingWinreg(types.ModuleType):
    """A ``winreg`` that fails any test that reaches the registry unfaked."""

    def __init__(self):
        super().__init__("winreg")

    def __getattr__(self, name):
        raise AssertionError(f"a test reached the real registry (winreg.{name})")


@pytest.fixture(autouse=True)
def _nothing_here_reaches_this_computer(tmp_path_factory, monkeypatch):
    """Keep every test in this file away from this computer's spaCR copies.

    The module deletes installations, and the computer running these tests
    may have real ones. A test that built the default computer, or forgot
    ``system=``, would read the real home folder and could delete from it.
    So every location the module reads by default points into a sandbox,
    deleting anything outside pytest's temporary folder fails the test before
    it is touched, and commands, detached processes, downloads and the
    Windows registry fail the test until it passes its own fake.
    """
    base = str(tmp_path_factory.getbasetemp())
    sandbox = tmp_path_factory.mktemp("default-computer")
    for name in ("HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA",
                 "PROGRAMFILES", "ProgramW6432"):
        monkeypatch.setenv(name, str(sandbox / name))
    for name in ("XDG_DATA_HOME", "XDG_BIN_HOME", "XDG_CACHE_HOME",
                 "XDG_STATE_HOME", "XDG_CONFIG_HOME", "HF_HOME", "CONDA_EXE",
                 "HOMEDRIVE", "HOMEPATH", "OneDrive"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(sandbox))

    def delete_inside_the_sandbox(path, keep, report, reason_on_denied):
        if not ic._inside(path, base):
            raise AssertionError(f"a test tried to delete {path}, outside {base}")
        return _REAL_DELETE(path, keep, report, reason_on_denied)

    monkeypatch.setattr(ic, "_delete", delete_inside_the_sandbox)
    monkeypatch.setattr(ic, "_run", _refuse("command runner"))
    monkeypatch.setattr(ic, "_spawn_detached", _refuse("process launcher"))
    monkeypatch.setattr(urllib.request, "urlopen", _refuse("network"))
    monkeypatch.setitem(sys.modules, "winreg", _RefusingWinreg())


class FakeRegistry:
    """HKEY_CURRENT_USER as a dictionary whose key names ignore case."""

    def __init__(self):
        self._keys = {}

    def set(self, key, **values):
        parts = key.split("\\")
        for index in range(1, len(parts) + 1):
            self._keys.setdefault("\\".join(parts[:index]).lower(), {})
        self._keys[key.lower()].update(values)

    def exists(self, key):
        return key.lower() in self._keys

    def values(self, key):
        return dict(self._keys.get(key.lower(), {}))

    def subkeys(self, key):
        prefix = key.lower() + "\\"
        return sorted({name[len(prefix):].split("\\")[0]
                       for name in self._keys if name.startswith(prefix)})

    def delete_value(self, key, name):
        del self._keys[key.lower()][name]

    def delete_key(self, key):
        if self.subkeys(key):
            raise OSError("the key has subkeys")
        del self._keys[key.lower()]


class FakePackages:
    """dpkg, removing the package's files only when it may use sudo."""

    def __init__(self, box):
        self.box = box
        self.installed = {}
        self.calls = []

    def version(self, name):
        return self.installed.get(name)

    def remove(self, name, sudo):
        self.calls.append((name, sudo))
        if not sudo:
            return False, ic._needs_admin(f"sudo dpkg -r {name}")
        self.installed.pop(name, None)
        for path in self.box.deb_files:
            path.unlink()
        return True, ""


class FakeWinreg(types.ModuleType):
    """``winreg`` over a :class:`FakeRegistry`, for the adapter Windows uses."""

    HKEY_CURRENT_USER = "HKEY_CURRENT_USER"
    KEY_READ = 0x20019
    KEY_ALL_ACCESS = 0xF003F

    class Handle:
        def __init__(self, key):
            self.key = key

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    def __init__(self, registry, denied=()):
        super().__init__("winreg")
        self.registry = registry
        self.denied = {key.lower() for key in denied}
        self.opened = []

    def OpenKey(self, root, key, reserved, access):
        assert (root, reserved) == (self.HKEY_CURRENT_USER, 0)
        if not self.registry.exists(key):
            raise FileNotFoundError(errno.ENOENT, "The system cannot find the file specified")
        self.opened.append((key, access))
        return self.Handle(key)

    def EnumValue(self, handle, index):
        items = sorted(self.registry.values(handle.key).items())
        if index >= len(items):
            raise OSError(259, "No more data is available")
        return items[index][0], items[index][1], 1

    def EnumKey(self, handle, index):
        names = self.registry.subkeys(handle.key)
        if index >= len(names):
            raise OSError(259, "No more data is available")
        return names[index]

    def DeleteValue(self, handle, name):
        if handle.key.lower() in self.denied:
            raise PermissionError(errno.EACCES, "Access is denied")
        self.registry.delete_value(handle.key, name)

    def DeleteKey(self, root, key):
        assert root == self.HKEY_CURRENT_USER
        self.registry.delete_key(key)


class Box:
    """A computer inside ``tmp_path``."""

    def __init__(self, base: Path):
        self.base = base
        self.fs = base / "root"
        self.home = base / "home"
        self.local = self.home / "AppData" / "Local"
        self.appdata = self.home / "AppData" / "Roaming"
        self.program_files = self.fs / "Program Files"
        for folder in (self.fs, self.home, self.local, self.appdata,
                       self.program_files):
            folder.mkdir(parents=True, exist_ok=True)
        self.registry = FakeRegistry()
        self.packages = FakePackages(self)
        self.deb_files = []
        self.commands = []
        self.user_files = set()

    def machine(self, platform, running_prefix=None, executable=None,
                sudo=False, which=None):
        environ = {"HOME": str(self.home), "USERPROFILE": str(self.home),
                   "LOCALAPPDATA": str(self.local), "APPDATA": str(self.appdata),
                   "PROGRAMFILES": str(self.program_files)}
        return ic._Machine(
            platform=platform, environ=environ, fs_root=str(self.fs),
            registry=self.registry, packages=self.packages, runner=self.run,
            running_prefix=running_prefix,
            executable=executable or str(self.base / "elsewhere" / "python"),
            sudo=sudo, which=which)

    def run(self, argv):
        self.commands.append([str(part) for part in argv])
        return 0, ""

    def write(self, path, content="x"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def system(self, absolute):
        return self.fs / absolute.lstrip("/")

    def files(self):
        found = set()
        for folder, dirs, names in os.walk(self.base):
            for name in names + [d for d in dirs
                                 if os.path.islink(os.path.join(folder, d))]:
                found.add(os.path.join(folder, name))
        return found

    def add_user_data(self):
        for path in (
            self.home / ".spacr" / "runs" / "r1" / "workspace.json",
            self.home / ".spacr" / "models" / "custom.pth",
            self.home / ".cache" / "spacr" / "example_data" / "plate.tif",
            self.home / ".local" / "state" / "spacr" / "state.json",
            self.home / "spacr-demos" / "demo.txt",
            self.home / "spacr-tutorials" / "lesson.txt",
            self.home / ".cache" / "huggingface" / "hub" / "model.bin",
            self.home / ".config" / "spacr" / "qt.conf",
            self.home / "Library" / "Preferences" / "com.spacr.qt.plist",
        ):
            self.user_files.add(str(self.write(path)))
        # QSettings("spacr", "qt") on Windows: the SAME key as the installer's
        # HKCU\Software\spaCR, because registry names ignore case.
        self.registry.set("Software\\spacr\\qt", theme="dark")


# ---------------------------------------------------------------------------
# The layouts, as the installers at each tag wrote them
# ---------------------------------------------------------------------------

def _dist(site, version):
    info = Path(site) / f"spacr-{version}.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: spacr\nVersion: {version}\n\n",
        encoding="utf-8")
    package = Path(site) / "spacr"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")


def _shortcut(box, path, target):
    return box.write(path, b"L\x00\x00\x00" + str(target).encode("utf-16-le"))


def windows_online(box, tag, broken=False):
    root = box.local / OLD                     # NSIS InstallDir at every tag
    venv = root / "venv"
    box.write(venv / "Scripts" / "python.exe")
    box.write(venv / "Scripts" / "pythonw.exe")
    _dist(venv / "Lib" / "site-packages", tag)
    box.write(root / "python" / "cpython-3.12.11-windows-x86_64-none" / "python.exe")
    box.write(root / "bootstrap" / "uv.exe")
    box.write(root / "cache" / "wheels" / "w.whl")
    box.write(root / "launch_spacr.pyw", "from spacr.qt import run\n")
    box.write(root / "spacr.cmd", f'@echo off\r\n"{venv}\\Scripts\\python.exe" -m spacr.qt\r\n')
    box.write(root / "spacr.ico")
    box.write(root / "install.log", "old log\n")
    if not broken:
        box.write(root / "Uninstall.exe")
    if not _capitalised(tag):
        box.write(root / "install-profile.json", "{}")
    pythonw = venv / "Scripts" / "pythonw.exe"
    menu = box.appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "spaCR"
    _shortcut(box, menu / "spaCR.lnk", pythonw)
    _shortcut(box, menu / "Uninstall spaCR.lnk", root / "Uninstall.exe")
    _shortcut(box, box.home / "Desktop" / "spaCR.lnk", pythonw)
    box.registry.set(ic._UNINSTALL_KEY, DisplayName="spaCR", DisplayVersion=tag,
                     Publisher="Olafsson Lab", InstallLocation=str(root),
                     UninstallString=f'"{root}\\Uninstall.exe"')
    box.registry.set(ic._SOFTWARE_KEY, InstallRoot=str(root))
    return root


def windows_offline(box, tag, broken=False):
    name = OLD if _capitalised(tag) else "spaCR"
    root = box.program_files / name
    box.write(root / "spacr.exe")
    box.write(root / "_internal" / "base_library.zip")
    if not broken:
        box.write(root / "Uninstall.exe")
    menu = box.appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    _shortcut(box, menu / f"{name}.lnk", root / "spacr.exe")
    return root


def linux_online(box, tag, broken=False):
    root = box.home / ".local" / "share" / "spacr"
    venv = root / "venv"
    box.write(venv / "bin" / "python")
    _dist(venv / "lib" / "python3.12" / "site-packages", tag)
    box.write(root / "python" / "cpython-3.12.11-linux-x86_64-gnu" / "bin" / "python3.12")
    box.write(root / "bootstrap" / "uv")
    box.write(root / "cache" / "archive-v0" / "x")
    box.write(root / "install.log", "old log\n")
    box.write(root / ".venv-staging-563392" / "bin" / "python")   # a crashed run
    if not _capitalised(tag):
        box.write(root / "install-profile.json", "{}")
    launcher = box.write(box.home / ".local" / "bin" / "spacr",
                         f'#!/usr/bin/env sh\nexec "{venv}/bin/python" -m spacr.qt "$@"\n')
    applications = box.home / ".local" / "share" / "applications"
    desktop = applications / ("spacr.desktop" if _capitalised(tag)
                              else "io.github.olafssonlab.spacr.desktop")
    box.write(desktop, f"[Desktop Entry]\nType=Application\nName=spaCR\n"
                       f"Exec={launcher}\nIcon={venv}/lib/python3.12/site-packages/"
                       f"spacr/resources/icons/app_icon.png\nTerminal=false\n")
    if not broken:
        box.write(root / "uninstall-spacr.sh",
                  f'#!/usr/bin/env sh\nset -eu\nrm -f "{launcher}"\n'
                  f'rm -f "{desktop}"\nrm -rf "{root}"\n')
    return root


def _info_plist(tag, executable):
    return plistlib.dumps({"CFBundleShortVersionString": tag,
                           "CFBundleExecutable": executable})


def macos_package(box, tag, broken=False):
    name = OLD if _capitalised(tag) else "spaCR"
    app = box.system(f"/Applications/{name}.app")
    box.write(app / "Contents" / "MacOS" / name, "#!/bin/sh\n")
    box.write(app / "Contents" / "Info.plist", _info_plist(tag, name))
    support = box.system(f"/Library/Application Support/{name}")
    box.write(support / "install-online.sh")
    box.write(support / "install-for-user.sh")
    if not _capitalised(tag):
        box.write(support / "installer_messages.sh")
    if not broken:
        box.write(support / "uninstall-spacr.sh")
    link = box.system("/usr/local/bin/spacr")
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(f"/Applications/{name}.app/Contents/MacOS/{name}", link)
    runtime = box.home / "Library" / "Application Support" / name
    box.write(runtime / "venv" / "bin" / "python")
    _dist(runtime / "venv" / "lib" / "python3.12" / "site-packages", tag)
    box.write(runtime / "python" / "x")
    box.write(runtime / "bootstrap" / "uv")
    box.write(runtime / "cache" / "x")
    box.write(runtime / "install.log")
    return support


def macos_disk_image(box, tag, broken=False):
    name = OLD if _capitalised(tag) else "spaCR"
    app = box.system(f"/Applications/{name}.app")
    box.write(app / "Contents" / "MacOS" / "spacr")
    box.write(app / "Contents" / "Info.plist", _info_plist(tag, "spacr"))
    return app


def linux_deb(box, tag, broken=False):
    root = box.system("/usr/lib/python3/dist-packages/spacr")
    box.deb_files = [box.write(root / "__init__.py"),
                     box.write(box.system("/usr/bin/spacr"), "#!/usr/bin/python3\n")]
    box.packages.installed["python3-spacr"] = tag
    return root


LAYOUTS = {
    "windows-online": ("windows", windows_online, "windows-online"),
    "windows-offline": ("windows", windows_offline, "windows-offline"),
    "linux-online": ("linux", linux_online, "linux-online"),
    "macos-package": ("macos", macos_package, "macos-app"),
    "macos-disk-image": ("macos", macos_disk_image, "macos-app"),
    "linux-deb": ("linux", linux_deb, "linux-deb"),
}


def _installers(records):
    return [r for r in records if r.kind == "installer"]


# ---------------------------------------------------------------------------
# Find, then delete: every layout, every tag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag", TAGS)
@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_every_released_layout_is_found_and_removed_down_to_user_data(
        tmp_path, name, tag):
    box = Box(tmp_path)
    box.add_user_data()
    platform, build, layout = LAYOUTS[name]
    root = build(box, tag)
    machine = box.machine(platform, sudo=True)

    installers = _installers(ic.find_old_installs(system=machine))

    assert layout in {r.layout for r in installers}, installers
    assert str(root) in {r.root for r in installers}, installers
    if name != "windows-offline":          # PyInstaller keeps no readable version
        assert tag in {r.version for r in installers}, installers

    reports = [ic.remove_install(r, system=machine) for r in installers]

    assert all(r.ok for r in reports), [r.failed for r in reports]
    assert box.files() == box.user_files
    assert box.commands == [], "no old uninstaller or other command was run"
    if platform == "windows":
        assert not box.registry.exists(ic._UNINSTALL_KEY)
        assert "InstallRoot" not in box.registry.values(ic._SOFTWARE_KEY)
        assert box.registry.values("Software\\spacr\\qt") == {"theme": "dark"}


@pytest.mark.parametrize("name", ["windows-online", "windows-offline",
                                  "linux-online", "macos-package"])
def test_a_1_5_0_1_whose_own_uninstaller_is_missing_is_removed_all_the_same(
        tmp_path, name):
    box = Box(tmp_path)
    box.add_user_data()
    platform, build, _layout = LAYOUTS[name]
    root = build(box, "1.5.0.1", broken=True)
    assert not list(tmp_path.rglob("Uninstall.exe"))
    assert not list(tmp_path.rglob("uninstall-spacr.sh"))
    machine = box.machine(platform)

    [record] = [r for r in ic.find_old_installs(system=machine) if r.root == str(root)]
    assert "its own uninstaller is missing" in record.notes
    for other in _installers(ic.find_old_installs(system=machine)):
        assert ic.remove_install(other, system=machine).ok

    assert box.files() == box.user_files


def test_the_linux_desktop_entry_and_launcher_of_another_spacr_are_not_claimed(
        tmp_path):
    """The maintainer's workstation: a 1.5.0.1 copy, but the menu entry and
    the ``spacr`` command belong to a conda environment."""
    box = Box(tmp_path)
    root = linux_online(box, "1.5.0.1")
    applications = box.home / ".local" / "share" / "applications"
    (applications / "spacr.desktop").unlink()
    command = box.write(box.home / ".local" / "bin" / "spacr",
                        "#!/home/u/miniforge3/envs/spacr/bin/python\n"
                        "from spacr.qt import run\n")
    entry = box.write(applications / "io.github.olafssonlab.spacr.desktop",
                      "[Desktop Entry]\nExec=/home/u/miniforge3/envs/spacr/bin/spacr\n"
                      "Icon=/src/spacr/spacr/resources/icons/app_icon.png\n")
    machine = box.machine("linux")

    [record] = _installers(ic.find_old_installs(system=machine))
    assert record.launchers == () and record.menu_entries == ()
    assert ic.remove_install(record, system=machine).ok

    assert not root.exists()
    assert command.exists() and entry.exists()


@pytest.mark.parametrize(("where", "verdict"), [
    ("home", "refused"), ("applications", "refused"), ("outside", "refused"),
    (".spacr", "kept"), ("demos", "kept"),
])
def test_the_remover_will_not_delete_shared_folders_user_data_or_anything_outside(
        tmp_path, where, verdict):
    box = Box(tmp_path / "box")
    box.add_user_data()
    target = {
        "home": box.home,
        "applications": box.home / ".local" / "share" / "applications",
        "outside": tmp_path / "not-this-computer" / "spacr",
        ".spacr": box.home / ".spacr",
        "demos": box.home / "spacr-demos",
    }[where]
    target.mkdir(parents=True, exist_ok=True)
    marker = box.write(target / "spacr-marker.txt") if where == "outside" else None
    record = ic.InstallRecord(kind="installer", layout="linux-online",
                              platform="linux", root=str(target))

    report = ic.remove_install(record, system=box.machine("linux"))

    assert all(Path(f).exists() for f in box.user_files)
    assert target.is_dir()
    if marker is not None:
        assert marker.exists()
    reasons = [why for _item, why in report.failed + report.skipped]
    assert any(why.startswith(verdict) for why in reasons), reasons


@pytest.mark.parametrize("with_settings", [True, False])
def test_windows_settings_in_the_installers_registry_key_survive(
        tmp_path, with_settings):
    box = Box(tmp_path)
    windows_online(box, "1.5.0.5")
    if with_settings:
        box.registry.set("Software\\spacr\\qt", theme="dark")
    machine = box.machine("windows")

    for record in _installers(ic.find_old_installs(system=machine)):
        assert ic.remove_install(record, system=machine).ok

    assert not box.registry.exists(ic._UNINSTALL_KEY)
    # The key goes only when nothing but the installer's value was in it.
    assert box.registry.exists(ic._SOFTWARE_KEY) is with_settings
    if with_settings:
        assert box.registry.values("Software\\spacr\\qt") == {"theme": "dark"}


def test_a_debian_package_needs_sudo_and_says_so_instead_of_failing_quietly(
        tmp_path):
    box = Box(tmp_path)
    linux_deb(box, "1.5.0.1")
    machine = box.machine("linux", sudo=False)

    [record] = _installers(ic.find_old_installs(system=machine))
    report = ic.remove_install(record, system=machine)

    assert not report.ok
    assert "needs administrator rights" in report.failed[0][1]
    assert "sudo dpkg -r python3-spacr" in report.failed[0][1]
    assert all(path.exists() for path in box.deb_files)
    assert box.packages.calls == [("python3-spacr", False)]


def test_the_log_of_the_new_install_is_kept_while_the_old_copy_goes(tmp_path):
    box = Box(tmp_path)
    box.add_user_data()
    root = linux_online(box, "1.5.0.6")
    machine = box.machine("linux")
    [record] = _installers(ic.find_old_installs(system=machine))

    report = ic.remove_install(record, keep=[str(root / "install.log")], system=machine)

    assert report.ok
    assert box.files() == box.user_files | {str(root / "install.log")}


def test_the_running_copy_is_never_deleted_from_inside_itself(tmp_path):
    box = Box(tmp_path)
    root = linux_online(box, "1.5.0.5")
    machine = box.machine("linux", running_prefix=str(root / "venv"))

    records = ic.find_old_installs(system=machine)
    assert [r.kind for r in records] == ["installer"], (
        "the installer's private venv is part of the copy, not a user environment")
    assert records[0].running

    report = ic.remove_install(records[0], system=machine)

    assert not report.ok
    assert (root / "venv" / "bin" / "python").exists()


# ---------------------------------------------------------------------------
# Environments the user made, and checkouts
# ---------------------------------------------------------------------------

def _conda_env(box, name, version="1.5.0.7", editable_source=None):
    prefix = box.home / "miniforge3" / "envs" / name
    box.write(prefix / "bin" / "python")
    (prefix / "conda-meta").mkdir(parents=True, exist_ok=True)
    site = prefix / "lib" / "python3.12" / "site-packages"
    _dist(site, version)
    (site / "pip").mkdir(exist_ok=True)
    if editable_source is not None:
        (site / f"spacr-{version}.dist-info" / "direct_url.json").write_text(
            json.dumps({"url": Path(editable_source).as_uri(),
                        "dir_info": {"editable": True}}), encoding="utf-8")
    listing = box.home / ".conda" / "environments.txt"
    listing.parent.mkdir(parents=True, exist_ok=True)
    with listing.open("a", encoding="utf-8") as handle:
        handle.write(f"{prefix}\n")
    return prefix


def test_an_environment_you_made_is_only_listed_until_you_tick_it(tmp_path):
    box = Box(tmp_path)
    prefix = _conda_env(box, "analysis")
    machine = box.machine("linux")

    [record] = ic.find_old_installs(system=machine)
    assert (record.kind, record.layout, record.root, record.version) == (
        "environment", "conda", str(prefix), "1.5.0.7")
    assert record.python == str(prefix / "bin" / "python")
    before = box.files()

    unticked = ic.remove_install(record, system=machine)
    assert unticked.ok and unticked.skipped and box.commands == []
    assert box.files() == before

    ticked = ic.remove_install(record, ticked=True, system=machine)
    assert ticked.ok
    assert box.commands == [[str(prefix / "bin" / "python"), "-m", "pip",
                             "uninstall", "-y", "spacr"]]
    assert prefix.is_dir() and box.files() == before, "the environment stays"


def test_only_the_ticked_environment_is_changed_by_an_update(tmp_path):
    box = Box(tmp_path)
    ticked = _conda_env(box, "old-analysis")
    kept = _conda_env(box, "current-work")
    machine = box.machine("linux")
    installs = []

    reports, result = ic.run_update_sequence(
        lambda: installs.append(1) or 0, ticked=[str(ticked)], system=machine)

    assert result == 0 and installs == [1]
    assert box.commands == [[str(ticked / "bin" / "python"), "-m", "pip",
                             "uninstall", "-y", "spacr"]]
    assert not any(str(kept) in part for command in box.commands for part in command)


def test_the_running_environment_is_never_uninstalled_even_when_ticked(tmp_path):
    box = Box(tmp_path)
    prefix = _conda_env(box, "spacr")
    machine = box.machine("linux", running_prefix=str(prefix))
    [record] = ic.find_old_installs(system=machine)
    assert record.running

    report = ic.remove_install(record, ticked=True, system=machine)

    assert box.commands == [] and report.ok and report.skipped


def test_a_source_checkout_is_reported_and_left_alone(tmp_path):
    box = Box(tmp_path)
    checkout = box.home / "src" / "spacr"
    box.write(checkout / "pyproject.toml")
    prefix = _conda_env(box, "dev", editable_source=checkout)
    machine = box.machine("linux")
    [record] = ic.find_old_installs(system=machine)
    assert (record.kind, record.root) == ("checkout", str(checkout))
    assert any(str(prefix) in note for note in record.notes)
    before = box.files()

    report = ic.remove_install(record, ticked=True, system=machine)

    assert report.skipped and box.commands == [] and box.files() == before


# ---------------------------------------------------------------------------
# The order: find, then delete, then install
# ---------------------------------------------------------------------------

def _record(kind, root, running=False):
    return ic.InstallRecord(kind=kind, layout="test", platform="linux",
                            root=str(root), running=running)


def test_the_update_finds_then_deletes_then_installs(tmp_path):
    events = []
    old = _record("installer", tmp_path / "old")
    env = _record("environment", tmp_path / "env")

    def find(system):
        events.append("find")
        return [old, env]

    def remove(record, ticked, keep, system):
        events.append(("delete", record.kind, ticked))
        return ic.RemovalReport(record)

    def install():
        events.append("install")
        return 0, "installed"

    _reports, result = ic.run_update_sequence(
        install, ticked=[env.root], find=find, remove=remove,
        system=Box(tmp_path / "box").machine("linux"))

    assert events == ["find", ("delete", "installer", False),
                      ("delete", "environment", True), "install"]
    assert result == (0, "installed")


def test_nothing_is_installed_when_an_installer_made_copy_was_not_removed(tmp_path):
    events = []
    old = _record("installer", tmp_path / "old")

    def remove(record, ticked, keep, system):
        events.append("delete")
        return ic.RemovalReport(record, failed=[(record.root, "in use")])

    reports, result = ic.run_update_sequence(
        lambda: events.append("install"), records=[old], remove=remove,
        system=Box(tmp_path / "box").machine("linux"))

    assert events == ["delete"] and result is None
    assert reports[0].failed == [(old.root, "in use")]


def test_a_failure_on_your_own_environment_does_not_hold_the_install_back(tmp_path):
    """The rule stops only for installer-made copies; the control for the test above."""
    env = _record("environment", tmp_path / "env")

    def remove(record, ticked, keep, system):
        return ic.RemovalReport(record, failed=[(record.root, "pip failed")])

    _reports, result = ic.run_update_sequence(
        lambda: "installed", records=[env], ticked=[env.root], remove=remove,
        system=Box(tmp_path / "box").machine("linux"))

    assert result == "installed"


def test_an_in_process_update_never_installs_beside_the_running_desktop_copy(tmp_path):
    box = Box(tmp_path)
    root = linux_online(box, "1.5.0.7")
    machine = box.machine("linux", running_prefix=str(root / "venv"))
    installs = []

    reports, result = ic.run_update_sequence(
        lambda: installs.append(1), system=machine)

    assert result is None and installs == []
    assert (root / "venv").is_dir()


# ---------------------------------------------------------------------------
# The helper that finishes an in-app update after spaCR has closed
# ---------------------------------------------------------------------------

def _running_desktop(box, platform="linux"):
    if platform == "windows":
        root = windows_online(box, "1.5.0.5")
        executable = root / "venv" / "Scripts" / "python.exe"
    else:
        root = linux_online(box, "1.5.0.5")
        executable = root / "venv" / "bin" / "python"
    machine = box.machine(platform, running_prefix=str(root / "venv"),
                          executable=str(executable))
    return root, machine, ic.find_old_installs(system=machine)


def test_the_helper_plan_waits_fetches_deletes_installs_and_relaunches(tmp_path):
    box = Box(tmp_path / "box")
    root, machine, records = _running_desktop(box)
    workdir = tmp_path / "helper"
    spawned = []

    plan = ic.start_update_helper(
        records, "9.9.9", pid=4242, workdir=str(workdir), system=machine,
        spawn=lambda argv, env, cwd: spawned.append((argv, env, cwd)))

    assert plan["steps"] == ["wait", "fetch", "delete", "install", "relaunch"]
    assert plan["pid"] == 4242
    assert json.loads((workdir / "plan.json").read_text()) == json.loads(json.dumps(plan))
    assert (workdir / "install_cleanup.py").read_bytes() == Path(ic.__file__).read_bytes()
    assert plan["fetch"]["url"] == (
        "https://github.com/EinarOlafsson/spacr/releases/download/v9.9.9/"
        "spaCR-9.9.9-Linux-x86_64-Online.run")
    assert plan["install"][:2] == ["bash", plan["fetch"]["path"]]
    assert plan["install"][plan["install"].index("--install-root") + 1] == str(root)
    [(argv, env, cwd)] = spawned
    assert argv == plan["command"] and argv[-2:] == ["run-plan", str(workdir / "plan.json")]
    # The copy being removed is the one running, so neither the helper's
    # interpreter nor its working folder may live inside it.
    assert not ic._inside(argv[0], str(root)) and not ic._inside(cwd, str(root))
    assert argv[0] == str(workdir / "uv")
    assert env["UV_PYTHON_INSTALL_DIR"].startswith(str(workdir))


def test_the_helper_uses_the_current_python_when_it_is_not_being_removed(tmp_path):
    box = Box(tmp_path / "box")
    root, machine, records = _running_desktop(box)
    machine.executable = str(tmp_path / "conda" / "bin" / "python")
    spawned = []

    plan = ic.start_update_helper(records, "9.9.9", workdir=str(tmp_path / "h"),
                                  system=machine,
                                  spawn=lambda argv, env, cwd: spawned.append(argv))

    assert plan["command"][0] == machine.executable
    assert spawned == [plan["command"]]


def test_no_helper_starts_without_a_python_that_survives_the_removal(tmp_path):
    box = Box(tmp_path / "box")
    root, machine, records = _running_desktop(box)
    (root / "bootstrap" / "uv").unlink()
    spawned = []

    plan = ic.start_update_helper(records, "9.9.9", workdir=str(tmp_path / "h"),
                                  system=machine,
                                  spawn=lambda *args: spawned.append(args))

    assert plan["command"] is None and "run the new installer" in plan["error"]
    assert spawned == []


def test_a_windows_helper_runs_the_setup_silently_into_the_same_folder(tmp_path):
    box = Box(tmp_path / "box")
    root, machine, records = _running_desktop(box, "windows")

    plan = ic.start_update_helper(records, "9.9.9", workdir=str(tmp_path / "h"),
                                  system=machine, spawn=lambda *args: None)

    assert plan["fetch"]["url"].endswith("/v9.9.9/spaCR-9.9.9-Windows-Online-Setup.exe")
    assert plan["install"] == [plan["fetch"]["path"], "/S", f"/D={root}"]
    assert plan["relaunch"][0].endswith("pythonw.exe")
    assert Path(plan["command"][0]).name == "uv.exe"


def _helper_plan(tmp_path):
    box = Box(tmp_path / "box")
    box.add_user_data()
    root, machine, records = _running_desktop(box)
    ic.start_update_helper(records, "9.9.9", pid=4242,
                           workdir=str(tmp_path / "helper"), system=machine,
                           spawn=lambda *args: None)
    return box, root, machine, str(tmp_path / "helper" / "plan.json")


def test_the_helper_carries_out_its_plan_in_order(tmp_path):
    box, root, machine, plan = _helper_plan(tmp_path)
    events = []

    def remove(record, ticked, keep, system):
        events.append(("delete", record.root, record.running))
        return ic.RemovalReport(record)

    code = ic._run_plan(
        plan, wait=lambda pid: events.append(("wait", pid)) or True,
        fetch=lambda url, path: events.append(("fetch", url)),
        run=lambda argv: events.append(("install", argv[0])) or (0, ""),
        remove=remove, spawn=lambda argv, env, cwd: events.append(("relaunch", argv[0])),
        system=machine)

    assert code == 0
    assert [event[0] for event in events] == ["wait", "fetch", "delete",
                                              "install", "relaunch"]
    assert ("wait", 4242) in events
    # Once spaCR has closed, the copy that was running is removed too.
    assert ("delete", str(root), False) in events
    assert (box.home / ".spacr" / "logs" / "update.log").is_file()


def test_the_helper_removes_the_old_copy_for_real_before_it_installs(tmp_path):
    box, root, machine, plan = _helper_plan(tmp_path)
    seen = []

    code = ic._run_plan(plan, wait=lambda pid: True, fetch=lambda url, path: None,
                        run=lambda argv: seen.append(root.exists()) or (0, ""),
                        spawn=lambda *args: None, system=machine)

    assert code == 0 and seen == [False]
    assert all(Path(f).exists() for f in box.user_files)
    assert not (box.home / ".local" / "bin" / "spacr").exists()


def test_the_helper_installs_nothing_when_an_old_copy_could_not_be_removed(tmp_path):
    box, root, machine, plan = _helper_plan(tmp_path)
    events = []

    code = ic._run_plan(
        plan, wait=lambda pid: True, fetch=lambda url, path: None,
        run=lambda argv: events.append("install") or (0, ""),
        remove=lambda record, ticked, keep, system: ic.RemovalReport(
            record, failed=[(record.root, "in use")]),
        spawn=lambda *args: events.append("relaunch"), system=machine)

    assert code == 5 and events == []
    log = (box.home / ".spacr" / "logs" / "update.log").read_text()
    assert "stopped before installing" in log and "in use" in log


def test_the_helper_deletes_nothing_when_the_new_installer_did_not_download(tmp_path):
    box, root, machine, plan = _helper_plan(tmp_path)
    events = []

    def fail(url, path):
        raise OSError("offline")

    code = ic._run_plan(
        plan, wait=lambda pid: True, fetch=fail,
        run=lambda argv: events.append("install"),
        remove=lambda *args, **kwargs: events.append("delete"), system=machine)

    assert code == 4 and events == [] and root.exists()


def test_the_helper_changes_nothing_when_spacr_does_not_close(tmp_path):
    box, root, machine, plan = _helper_plan(tmp_path)
    events = []

    code = ic._run_plan(plan, wait=lambda pid: False,
                        fetch=lambda url, path: events.append("fetch"),
                        system=machine)

    assert code == 3 and events == [] and root.exists()


# ---------------------------------------------------------------------------
# The command line the installers run, and the installers themselves
# ---------------------------------------------------------------------------

def _clean_environment(**extra):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("XDG_", "CONDA", "SPACR_", "UV_", "PYTHON"))}
    env.update(extra)
    return env


def test_the_command_line_removes_installer_copies_and_leaves_environments(tmp_path):
    box = Box(tmp_path)
    box.add_user_data()
    root = linux_online(box, "1.5.0.1", broken=True)
    prefix = _conda_env(box, "analysis")
    environment_files = {f for f in box.files() if f.startswith(str(prefix))}
    log = root / "install.log"

    done = subprocess.run(
        [sys.executable, "-I", ic.__file__, "remove", "--root", str(box.fs),
         "--keep", str(log)],
        capture_output=True, text=True, timeout=120,
        env=_clean_environment(HOME=str(box.home)))

    assert done.returncode == 0, done.stdout + done.stderr
    assert str(root) in done.stdout
    assert "your environment, left in place" in done.stdout
    remaining = box.files()
    assert remaining == box.user_files | environment_files | {
        str(log), str(box.home / ".conda" / "environments.txt")}


def test_the_module_embeds_in_a_shell_installer_and_needs_only_the_standard_library():
    text = Path(ic.__file__).read_text(encoding="utf-8")
    assert "SPACR_CLEANUP_PY" not in text.splitlines()
    imported = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add(node.module.split(".")[0])
    standard = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
    if standard:
        assert imported <= standard, imported - standard
    assert "spacr" not in imported


def _renderer():
    spec = importlib.util.spec_from_file_location(
        "spacr_installer_render_416", ROOT / "packaging" / "i18n" / "render.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _standalone_unix_installer(tmp_path):
    renderer = _renderer()
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "installer_messages.sh").write_text(
        renderer.render_shell(renderer.catalogs()), encoding="utf-8")
    renderer.OUTPUT_DIR = generated
    installer = tmp_path / "spaCR-9.9.9-Linux-x86_64-Online.run"
    renderer.embed_unix(ROOT / "packaging" / "online" / "install_spacr_unix.sh",
                        installer, "9.9.9")
    return renderer, installer


def test_the_rendered_unix_installer_carries_the_module_verbatim(tmp_path):
    renderer, installer = _standalone_unix_installer(tmp_path)
    text = installer.read_text(encoding="utf-8")
    module = Path(ic.__file__).read_text(encoding="utf-8").rstrip()

    assert module in text
    assert text.count(renderer.CLEANUP_BEGIN) == text.count(renderer.CLEANUP_END) == 1
    assert renderer.embed_cleanup_module(text) == text
    subprocess.run(["bash", "-n", str(installer)], check=True)


def test_the_unix_installer_deletes_old_copies_before_it_installs(tmp_path):
    box = Box(tmp_path / "box")
    box.add_user_data()
    root = linux_online(box, "1.5.0.1", broken=True)
    old_launcher = box.home / ".local" / "bin" / "spacr"
    old_entry = box.home / ".local" / "share" / "applications" / "spacr.desktop"
    _renderer_module, installer = _standalone_unix_installer(tmp_path)
    fake = tmp_path / "fake-bin"
    fake.mkdir()
    calls = tmp_path / "calls.log"
    uv = fake / "uv-binary"
    uv.write_text(textwrap.dedent(f"""\
        #!/bin/sh
        echo "uv $*" >> "{calls}"
        case "$1 $2" in
          "python install") exit 0 ;;
          "python find") echo "{sys.executable}"; exit 0 ;;
          "pip install")
            if [ -e "{root}/venv" ]; then echo "old copy still there" >> "{calls}"
            else echo "old copy gone" >> "{calls}"; fi
            exit 7 ;;
        esac
        if [ "$1" = venv ]; then mkdir -p "$2"; fi
        exit 0
        """), encoding="utf-8")
    uv.chmod(0o755)
    curl = fake / "curl"
    curl.write_text(textwrap.dedent(f"""\
        #!/bin/sh
        out=""
        while [ $# -gt 0 ]; do
          if [ "$1" = --output ]; then out="$2"; fi
          shift
        done
        printf '#!/bin/sh\\nmkdir -p "$UV_UNMANAGED_INSTALL"\\ncp "{uv}" "$UV_UNMANAGED_INSTALL/uv"\\n' > "$out"
        """), encoding="utf-8")
    curl.chmod(0o755)
    (tmp_path / "tmp").mkdir()
    env = _clean_environment(
        HOME=str(box.home), TMPDIR=str(tmp_path / "tmp"),
        PATH=f"{fake}{os.pathsep}{os.environ['PATH']}",
        SPACR_TORCH_BACKEND="cpu", SPACR_CLEANUP_ROOT=str(box.fs))

    done = subprocess.run(
        ["bash", str(installer), "--platform", "linux", "--skip-system-deps",
         "--no-launch", "--package-spec", "spacr[qt]==9.9.9"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, env=env,
        timeout=300)

    assert done.returncode == 7, done.stdout + done.stderr
    log = calls.read_text()
    assert "old copy gone" in log and "old copy still there" not in log
    assert log.index("uv python find") < log.index("uv pip install")
    assert not old_launcher.exists() and not old_entry.exists()
    assert all(Path(f).exists() for f in box.user_files)
    assert f"removed: {old_launcher}" in done.stdout
    assert f"removed: {root / 'venv'}" in done.stdout


def test_the_windows_installer_finds_and_deletes_before_it_installs():
    ps1 = (ROOT / "packaging" / "online" / "install_spacr_windows.ps1").read_text(
        encoding="utf-8")
    remove = ps1.index("$CleanupModule remove")
    assert ps1.index("python find $PythonVersion") < remove
    assert remove < ps1.index("Move-Item -Force $WorkUv $UvExe")
    assert remove < ps1.index("pip install --python $StagePython")
    assert "--keep $LogPath" in ps1
    nsis = (ROOT / "packaging" / "online" / "spacr_online_installer.nsi").read_text(
        encoding="utf-8")
    shipped = 'File "..\\..\\spacr\\install_cleanup.py"'
    assert shipped in nsis and nsis.index(shipped) < nsis.index("nsExec::ExecToLog")
    uninstall = nsis[nsis.index('Section "Uninstall"'):]
    assert 'DeleteRegKey HKCU "Software\\spaCR"' not in uninstall
    assert 'DeleteRegKey /ifempty HKCU "Software\\spaCR"' in uninstall


def test_the_unix_installer_source_orders_find_delete_install():
    sh = (ROOT / "packaging" / "online" / "install_spacr_unix.sh").read_text(
        encoding="utf-8")
    remove = sh.index('"$cleanup_python" -I "$cleanup_module" "${cleanup_args[@]}"')
    assert sh.index('python find "$PYTHON_VERSION"') < remove
    assert remove < sh.index('mv "$work_uv" "$UV_BIN"')
    assert remove < sh.index('"$UV_BIN" pip install')
    assert '--keep "$SPACR_SOURCE_DIR" --keep "/Applications/spaCR.app"' in sh


# ---------------------------------------------------------------------------
# The guard above, proven
# ---------------------------------------------------------------------------

def test_the_default_computer_in_these_tests_is_a_sandbox(tmp_path_factory):
    base = str(tmp_path_factory.getbasetemp())
    machine = ic._Machine()

    assert machine.real
    assert ic._inside(machine.home, base)
    assert ic._inside(machine.xdg("XDG_DATA_HOME", ".local"), base)
    assert ic._inside(machine.env("LOCALAPPDATA"), base)
    assert ic._inside(tempfile.gettempdir(), base)
    # A path that does not exist, so a missing guard could not harm anything.
    probe = os.path.join(os.sep, "no-such-folder-spacr-guard-probe", "spacr")
    with pytest.raises(AssertionError, match="outside"):
        ic._delete(probe, [], ic.RemovalReport(_record("installer", probe)), "x")
    with pytest.raises(AssertionError, match="command runner"):
        machine.runner([sys.executable, "-c", "pass"])
    with pytest.raises(AssertionError, match="real registry"):
        ic._WindowsRegistry().values(ic._SOFTWARE_KEY)


# ---------------------------------------------------------------------------
# What removal must never delete: each of these failed before its fix
# ---------------------------------------------------------------------------

def test_a_launcher_for_your_project_venv_does_not_make_the_project_an_old_copy(
        tmp_path):
    """A ``~/.local/bin/spacr`` a user wrote to start spaCR from a project's
    own ``venv`` named the project as an installer-made copy, and the update
    deleted the whole project folder."""
    box = Box(tmp_path)
    project = box.home / "research"
    box.write(project / "venv" / "bin" / "python")
    _dist(project / "venv" / "lib" / "python3.12" / "site-packages", "1.5.0.7")
    plate = box.write(project / "plates" / "plate1.tif")
    launcher = box.write(box.home / ".local" / "bin" / "spacr",
                         f'#!/bin/sh\nexec "{project}/venv/bin/python" -m spacr.qt "$@"\n')
    machine = box.machine("linux")
    installs = []

    records = ic.find_old_installs(system=machine)
    _reports, result = ic.run_update_sequence(
        lambda: installs.append(1) or 0, records=records, system=machine)

    assert _installers(records) == []
    assert plate.exists() and launcher.exists()
    assert result == 0 and installs == [1]


@pytest.mark.parametrize("evidence", [os.path.join("bootstrap", "uv"),
                                      "uninstall-spacr.sh"])
def test_a_copy_installed_somewhere_unusual_is_still_found_by_its_launcher(
        tmp_path, evidence):
    """The control for the test above: ``--install-root`` anywhere."""
    box = Box(tmp_path)
    root = box.home / "apps" / "imaging"
    box.write(root / "venv" / "bin" / "python")
    _dist(root / "venv" / "lib" / "python3.12" / "site-packages", "1.5.0.6")
    box.write(root / evidence)
    launcher = box.write(box.home / ".local" / "bin" / "spacr",
                         f'#!/usr/bin/env sh\nexec "{root}/venv/bin/python" -m spacr.qt "$@"\n')
    machine = box.machine("linux")

    [record] = _installers(ic.find_old_installs(system=machine))
    assert (record.root, record.version, record.launchers) == (
        str(root), "1.5.0.6", (str(launcher),))
    assert ic.remove_install(record, system=machine).ok

    assert not root.exists() and not launcher.exists()
    assert (box.home / "apps").is_dir()


@pytest.mark.parametrize("platform", ["linux", "windows"])
def test_a_shortcut_for_a_folder_beside_the_old_copy_is_not_claimed(
        tmp_path, platform):
    """``spacr`` begins ``spacr-dev``: a launcher, menu entry or shortcut for a
    folder beside the old copy was claimed by it and deleted with it."""
    box = Box(tmp_path)
    if platform == "linux":
        root = linux_online(box, "1.5.0.5")
        sibling = Path(f"{root}-dev")
        others = [
            box.write(box.home / ".local" / "bin" / "spacr",
                      f'#!/bin/sh\nexec "{sibling}/bin/python" -m spacr.qt "$@"\n'),
            box.write(box.home / ".local" / "share" / "applications"
                      / "io.github.olafssonlab.spacr.desktop",
                      f"[Desktop Entry]\nExec={sibling}/bin/spacr\n"),
        ]
    else:
        root = windows_online(box, "1.5.0.5")
        sibling = Path(f"{root}-portable")
        others = [_shortcut(box, box.home / "Desktop" / "spaCR.lnk",
                            sibling / "spacr.exe")]
    machine = box.machine(platform)

    [record] = [r for r in ic.find_old_installs(system=machine)
                if r.root == str(root)]
    claimed = record.launchers + record.shortcuts + record.menu_entries
    assert not {str(p) for p in others} & set(claimed), claimed
    assert ic.remove_install(record, system=machine).ok

    assert not root.exists()
    assert all(path.exists() for path in others)


def test_a_file_names_the_copy_only_where_the_folder_name_ends(tmp_path):
    root = str(tmp_path / "share" / "spacr")
    entry = tmp_path / "spacr.desktop"
    spellings = ic._spellings(root)

    entry.write_text(f"Exec={root}-dev/bin/spacr\n", encoding="utf-8")
    assert not ic._mentions(str(entry), spellings)
    entry.write_text(f"Exec={root}-dev/bin/spacr\nIcon={root}/venv/icon.png\n",
                     encoding="utf-8")
    assert ic._mentions(str(entry), spellings)
    assert ic._mentions(str(entry), [root + "/"]), "a trailing separator is the same folder"
    entry.write_text(f"Path={root}", encoding="utf-8")
    assert ic._mentions(str(entry), spellings), "the end of the file ends the name"
    assert not ic._mentions(str(entry), [])
    assert not ic._mentions(str(entry), ["", None, "/"])


@_PERMISSIONS_BIND
@pytest.mark.parametrize("onexc", [True, False], ids=["onexc", "onerror"])
def test_a_folder_that_cannot_be_opened_is_reported_not_a_crash(
        tmp_path, monkeypatch, onexc):
    """rmtree's retry called ``os.open(path)`` without its flags; the
    TypeError escaped the remover and ended the update with a traceback."""
    monkeypatch.setattr(ic, "_RMTREE_TAKES_ONEXC", onexc)
    box = Box(tmp_path)
    root = linux_online(box, "1.5.0.7")
    locked = root / "cache" / "made-by-sudo"
    box.write(locked / "wheel.whl")
    locked.chmod(0)
    machine = box.machine("linux")
    installs = []

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)   # onerror, 3.12+
            reports, result = ic.run_update_sequence(
                lambda: installs.append(1), system=machine)
    finally:
        locked.chmod(0o700)

    [report] = reports
    assert report.failed == [(str(root), "in use or not permitted; close anything using it and try again")]
    assert (locked / "wheel.whl").exists()
    assert result is None and installs == []


# ---------------------------------------------------------------------------
# Removal that cannot finish: the install does not start
# ---------------------------------------------------------------------------

@_PERMISSIONS_BIND
@pytest.mark.parametrize(("needs_admin", "reason"), [
    (False, "in use or not permitted; close anything using it and try again"), (True, "needs administrator rights; delete it as an administrator")])
def test_a_folder_you_may_not_change_stops_the_update_and_says_why(
        tmp_path, needs_admin, reason):
    box = Box(tmp_path)
    root = linux_online(box, "1.5.0.7")
    locked = root / "python" / "cpython-3.12.11-linux-x86_64-gnu" / "bin"
    locked.chmod(0o555)
    machine = box.machine("linux")
    [record] = _installers(ic.find_old_installs(system=machine))
    installs = []

    try:
        reports, result = ic.run_update_sequence(
            lambda: installs.append(1),
            records=[dataclasses.replace(record, needs_admin=needs_admin)],
            system=machine)
    finally:
        locked.chmod(0o755)

    assert reports[0].failed == [(str(root), reason)]
    assert (locked / "python3.12").exists()
    assert result is None and installs == []


@pytest.mark.parametrize(("error", "reason"), [
    (PermissionError(errno.EACCES, "The process cannot access the file because "
                     "it is being used by another process"), "in use or not permitted; close anything using it and try again"),
    (OSError(errno.EBUSY, "Device or resource busy"), "Device or resource busy"),
], ids=["in-use", "busy"])
def test_a_launcher_that_cannot_be_deleted_stops_the_update(
        tmp_path, monkeypatch, error, reason):
    box = Box(tmp_path)
    root = linux_online(box, "1.5.0.7")
    launcher = box.home / ".local" / "bin" / "spacr"
    real_unlink = os.unlink

    def unlink(path, *args, **kwargs):
        if str(path) == str(launcher):
            raise error
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    machine = box.machine("linux")
    installs = []

    [report], result = ic.run_update_sequence(lambda: installs.append(1),
                                              system=machine)

    assert report.failed == [(str(launcher), reason)]
    assert launcher.exists() and not root.exists()
    assert result is None and installs == []


def test_a_folder_still_there_after_removal_is_not_reported_removed(
        tmp_path, monkeypatch):
    box = Box(tmp_path)
    root = linux_online(box, "1.5.0.7")
    monkeypatch.setattr(shutil, "rmtree", lambda path, **hook: None)
    machine = box.machine("linux")
    [record] = _installers(ic.find_old_installs(system=machine))

    report = ic.remove_install(record, system=machine)

    assert (str(root), "still present") in report.failed
    assert str(root) not in report.removed


def test_deleting_what_another_process_already_deleted_records_nothing(tmp_path):
    gone = tmp_path / "spacr"
    report = ic.RemovalReport(_record("installer", gone))

    ic._delete(str(gone), [], report, "in use or not permitted; close anything using it and try again")

    assert (report.removed, report.failed, report.skipped) == ([], [], [])


@pytest.mark.parametrize("real", [False, True], ids=["sandbox", "real-computer"])
def test_a_named_path_that_is_not_recognisably_spacr_is_refused(tmp_path, real):
    box = Box(tmp_path)
    root = box.write(box.home / ".local" / "share" / "spacr" / "uninstall-spacr.sh").parent
    other = box.write(box.home / "bin" / "start-imaging", "#!/bin/sh\n")
    record = ic.InstallRecord(kind="installer", layout="linux-online",
                              platform="linux", root=str(root),
                              launchers=(str(other),))
    machine = box.machine("linux")
    machine.real = real
    installs = []

    reports, result = ic.run_update_sequence(
        lambda: installs.append(1), records=[record], system=machine)

    assert reports[0].failed == [
        (str(other), "refused: not recognisably a spaCR installation; "
                    "delete it by hand if it is one")]
    assert other.exists() and not root.exists()
    assert result is None and installs == []


def test_a_copy_registered_in_a_folder_with_another_name_is_removed_by_its_markers(
        tmp_path):
    box = Box(tmp_path)
    root = box.local / "Imaging Suite"
    (root / "venv" / "Lib").mkdir(parents=True)          # half-installed: no python
    box.write(root / "Uninstall.exe")
    box.write(root / "spacr.cmd")
    box.registry.set(ic._UNINSTALL_KEY, DisplayVersion="1.5.0.4",
                     InstallLocation=str(root))
    box.registry.set(ic._SOFTWARE_KEY, InstallRoot=str(root))
    machine = box.machine("windows")

    [record] = _installers(ic.find_old_installs(system=machine))
    assert (record.root, record.version, record.python) == (str(root), "1.5.0.4", None)
    report = ic.remove_install(record, system=machine)

    assert report.ok and not root.exists()
    assert not box.registry.exists(ic._UNINSTALL_KEY)


# ---------------------------------------------------------------------------
# Windows: the registry, the Desktop and the Start Menu
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("with_settings", [True, False])
def test_the_windows_registry_adapter_removes_registrations_and_keeps_settings(
        tmp_path, monkeypatch, with_settings):
    box = Box(tmp_path)
    windows_online(box, "1.5.0.5")
    if with_settings:
        box.registry.set("Software\\spacr\\qt", theme="dark")
    winreg = FakeWinreg(box.registry)
    monkeypatch.setitem(sys.modules, "winreg", winreg)
    machine = box.machine("windows")
    machine.registry = ic._WindowsRegistry()

    [record] = _installers(ic.find_old_installs(system=machine))
    assert record.registrations == (
        f"registry:HKCU\\{ic._UNINSTALL_KEY}",
        f"registry-value:HKCU\\{ic._SOFTWARE_KEY}\\InstallRoot")
    assert not any(access == winreg.KEY_ALL_ACCESS for _key, access in winreg.opened)
    report = ic.remove_install(record, system=machine)

    assert report.ok
    assert not box.registry.exists(ic._UNINSTALL_KEY)
    assert box.registry.exists(ic._SOFTWARE_KEY) is with_settings
    if with_settings:
        assert box.registry.values("Software\\spacr\\qt") == {"theme": "dark"}
    assert (ic._SOFTWARE_KEY, winreg.KEY_ALL_ACCESS) in winreg.opened


def test_the_registry_adapter_reads_a_missing_key_as_empty(monkeypatch):
    monkeypatch.setitem(sys.modules, "winreg", FakeWinreg(FakeRegistry()))
    registry = ic._WindowsRegistry()

    assert registry.values("Software\\missing") == {}
    assert registry.subkeys("Software\\missing") == []
    assert registry.delete_value("Software\\missing", "InstallRoot") is None


def test_a_registry_entry_you_may_not_delete_stops_the_update(tmp_path, monkeypatch):
    box = Box(tmp_path)
    windows_online(box, "1.5.0.7")
    monkeypatch.setitem(sys.modules, "winreg",
                        FakeWinreg(box.registry, denied=[ic._UNINSTALL_KEY]))
    machine = box.machine("windows")
    machine.registry = ic._WindowsRegistry()
    installs = []

    reports, result = ic.run_update_sequence(lambda: installs.append(1),
                                             system=machine)

    [report] = reports
    assert report.failed == [(f"registry:HKCU\\{ic._UNINSTALL_KEY}", "not permitted; delete this registry entry by hand")]
    assert box.registry.exists(ic._UNINSTALL_KEY)
    assert result is None and installs == []


def test_registrations_with_no_registry_to_change_stop_the_update(tmp_path):
    box = Box(tmp_path)
    windows_online(box, "1.5.0.7")
    records = ic.find_old_installs(system=box.machine("windows"))
    machine = box.machine("windows")
    machine.registry = None
    installs = []

    reports, result = ic.run_update_sequence(
        lambda: installs.append(1), records=records, system=machine)

    assert [f for r in reports for f in r.failed] == [
        (f"registry:HKCU\\{ic._UNINSTALL_KEY}", "no registry to change"),
        (f"registry-value:HKCU\\{ic._SOFTWARE_KEY}\\InstallRoot", "no registry to change")]
    assert result is None and installs == []


def test_an_install_root_value_already_gone_is_not_an_error(tmp_path):
    box = Box(tmp_path)
    windows_online(box, "1.5.0.7")
    box.registry.set("Software\\spacr\\qt", theme="dark")
    machine = box.machine("windows")
    [record] = _installers(ic.find_old_installs(system=machine))
    box.registry.delete_value(ic._SOFTWARE_KEY, "InstallRoot")

    report = ic.remove_install(record, system=machine)

    assert report.ok
    assert box.registry.values("Software\\spacr\\qt") == {"theme": "dark"}


def test_a_registered_copy_whose_folder_is_gone_still_loses_its_entry_and_shortcuts(
        tmp_path):
    box = Box(tmp_path)
    root = windows_online(box, "1.5.0.1")
    shutil.rmtree(root)
    machine = box.machine("windows")

    [record] = _installers(ic.find_old_installs(system=machine))
    assert record.root == str(root) and record.version == "1.5.0.1"
    assert "its folder is already gone" in record.notes
    assert record.shortcuts and record.menu_entries and record.python is None
    report = ic.remove_install(record, system=machine)

    assert report.ok
    assert not box.registry.exists(ic._UNINSTALL_KEY)
    assert box.files() == set()


def test_a_registration_naming_another_folder_is_not_this_copys_to_delete(tmp_path):
    box = Box(tmp_path)
    root = windows_online(box, "1.5.0.7")
    elsewhere = str(box.base / "D" / "spaCR")
    box.registry.set(ic._UNINSTALL_KEY, InstallLocation=elsewhere)
    box.registry.set(ic._SOFTWARE_KEY, InstallRoot=elsewhere)
    machine = box.machine("windows")

    [record] = _installers(ic.find_old_installs(system=machine))
    assert record.root == str(root) and record.registrations == ()
    assert ic.remove_install(record, system=machine).ok

    assert not root.exists()
    assert box.registry.values(ic._UNINSTALL_KEY)["InstallLocation"] == elsewhere
    assert box.registry.values(ic._SOFTWARE_KEY)["InstallRoot"] == elsewhere


def test_a_redirected_desktop_is_searched_for_the_old_shortcut(tmp_path):
    box = Box(tmp_path)
    root = windows_online(box, "1.5.0.5")
    redirected = box.home / "OneDrive - Lab" / "Desktop"
    link = _shortcut(box, redirected / "spaCR.lnk",
                     root / "venv" / "Scripts" / "pythonw.exe")
    box.registry.set(ic._SHELL_FOLDERS_KEY,
                     Desktop="%USERPROFILE%/OneDrive - Lab/Desktop")
    machine = box.machine("windows")

    [record] = _installers(ic.find_old_installs(system=machine))
    assert str(link) in record.shortcuts
    assert ic.remove_install(record, system=machine).ok

    assert not link.exists() and redirected.is_dir()


def test_start_menu_entries_of_another_spacr_are_left_alone(tmp_path):
    box = Box(tmp_path)
    root = windows_online(box, "1.5.0.1")
    programs = box.appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    shutil.rmtree(programs / "spaCR")
    other = _shortcut(box, programs / "spaCR" / "spaCR.lnk",
                      box.base / "D" / "spaCR" / "spacr.exe")
    not_a_folder = box.write(programs / OLD, "a file, not a Start Menu folder")
    machine = box.machine("windows")

    [record] = _installers(ic.find_old_installs(system=machine))
    assert record.root == str(root) and record.menu_entries == ()
    assert ic.remove_install(record, system=machine).ok

    assert other.exists() and not_a_folder.exists()


# ---------------------------------------------------------------------------
# macOS and Linux finder edges
# ---------------------------------------------------------------------------

def test_a_mac_app_with_a_damaged_info_plist_is_still_found_and_removed(tmp_path):
    box = Box(tmp_path)
    app = macos_disk_image(box, "1.5.0.5")
    (app / "Contents" / "Info.plist").write_bytes(b"not a property list")
    machine = box.machine("macos")

    [record] = _installers(ic.find_old_installs(system=machine))
    assert (record.root, record.version) == (str(app), None)
    assert ic.remove_install(record, system=machine).ok
    assert not app.exists()


def test_the_mac_support_folder_left_after_the_app_was_binned_is_removed(tmp_path):
    box = Box(tmp_path)
    support = macos_package(box, "1.5.0.5")
    shutil.rmtree(box.system("/Applications/spaCR.app"))
    machine = box.machine("macos")

    records = _installers(ic.find_old_installs(system=machine))
    [record] = [r for r in records if r.layout == "macos-app"]
    assert record.root == str(support)
    assert "its application bundle is already gone" in record.notes
    for other in records:
        assert ic.remove_install(other, system=machine).ok
    assert not support.exists()


def test_a_launcher_that_cannot_be_read_is_not_claimed(tmp_path, monkeypatch):
    box = Box(tmp_path)
    root = linux_online(box, "1.5.0.7")
    launcher = box.home / ".local" / "bin" / "spacr"

    def unreadable(file, *args, **kwargs):
        if str(file) == str(launcher):
            raise PermissionError(errno.EACCES, "Permission denied", str(file))
        return open(file, *args, **kwargs)

    monkeypatch.setattr(ic, "open", unreadable, raising=False)
    machine = box.machine("linux")

    [record] = _installers(ic.find_old_installs(system=machine))
    assert record.root == str(root) and record.launchers == ()
    assert ic.remove_install(record, system=machine).ok
    assert launcher.exists()


class _Dpkg:
    """A command runner answering dpkg questions from a table."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        return self.answers[tuple(argv)]


def test_the_debian_package_is_found_by_dpkg_and_removed_with_sudo(tmp_path):
    box = Box(tmp_path)
    query = ("dpkg-query", "-W", "-f=${Version}")
    dpkg = _Dpkg({query + ("python3-spacr",): (0, "1.5.0.1\n"),
                  query + ("spacr",): (1, "dpkg-query: no packages found matching spacr\n"),
                  ("sudo", "dpkg", "-r", "python3-spacr"): (0, "Removing python3-spacr\n")})
    machine = box.machine("linux", sudo=True)
    machine.packages = ic._SystemPackages(dpkg, which=lambda name: f"/usr/bin/{name}",
                                          geteuid=lambda: 1000)

    [record] = _installers(ic.find_old_installs(system=machine))
    assert (record.layout, record.version, record.registrations) == (
        "linux-deb", "1.5.0.1", ("deb:python3-spacr",))
    report = ic.remove_install(record, system=machine)

    assert report.ok and report.removed == ["deb:python3-spacr"]
    assert dpkg.calls[-1] == ["sudo", "dpkg", "-r", "python3-spacr"]


@pytest.mark.parametrize(("answer", "version"), [
    ((0, "1.5.0.1\n"), "1.5.0.1"),
    ((1, "dpkg-query: no packages found matching python3-spacr\n"), None),
    ((0, ""), None),
    ((0, None), None),
    ((0, "dpkg-query: warning: parsing file"), None),
])
def test_only_a_clean_dpkg_answer_is_a_version(answer, version):
    packages = ic._SystemPackages(lambda argv: answer, which=lambda name: "/usr/bin/x")
    assert packages.version("python3-spacr") == version


def test_without_dpkg_nothing_is_asked():
    packages = ic._SystemPackages(_refuse("dpkg"), which=lambda name: None)
    assert packages.version("python3-spacr") is None


@pytest.mark.parametrize(("euid", "sudo", "answer", "argv", "expected"), [
    (1000, False, None, None,
     (False, "needs administrator rights; run: sudo dpkg -r python3-spacr")),
    (1000, True, (0, ""), ["sudo", "dpkg", "-r", "python3-spacr"], (True, "")),
    (0, False, (0, ""), ["dpkg", "-r", "python3-spacr"], (True, "")),
    (0, False, (2, "Removing\ndpkg: error: package is in a very bad state\n"),
     ["dpkg", "-r", "python3-spacr"],
     (False, "dpkg: error: package is in a very bad state")),
    (0, False, (2, ""), ["dpkg", "-r", "python3-spacr"], (False, "exit code 2")),
], ids=["user-without-sudo", "user-with-sudo", "root", "dpkg-error", "silent-failure"])
def test_dpkg_removes_only_with_the_rights_to(euid, sudo, answer, argv, expected):
    calls = []
    packages = ic._SystemPackages(lambda command: calls.append(command) or answer,
                                  geteuid=lambda: euid)

    assert packages.remove("python3-spacr", sudo) == expected
    assert calls == ([argv] if argv else [])


def test_without_a_package_manager_the_debian_package_is_neither_found_nor_removed(
        tmp_path):
    box = Box(tmp_path)
    linux_deb(box, "1.5.0.1")
    records = ic.find_old_installs(system=box.machine("linux", sudo=True))
    machine = box.machine("linux", sudo=True)
    machine.packages = None
    installs = []

    assert ic.find_old_installs(system=machine) == []
    reports, result = ic.run_update_sequence(
        lambda: installs.append(1), records=records, system=machine)

    assert reports[0].failed == [("deb:python3-spacr", "no package manager to ask")]
    assert all(path.exists() for path in box.deb_files)
    assert result is None and installs == []


# ---------------------------------------------------------------------------
# Environments and versions
# ---------------------------------------------------------------------------

def test_environments_in_every_place_a_user_keeps_them_are_listed(tmp_path):
    box = Box(tmp_path)
    user_site = box.home / ".local" / "lib" / "python3.12"
    _dist(user_site / "site-packages", "1.5.0.3")
    venv = box.home / ".virtualenvs" / "imaging"
    box.write(venv / "bin" / "python3")
    _dist(venv / "lib" / "python3.11" / "site-packages", "1.4.0")
    conda = box.home / "tools" / "conda"
    env = conda / "envs" / "segmentation"
    (env / "conda-meta").mkdir(parents=True)
    box.write(env / "bin" / "python")
    _dist(env / "lib" / "python3.12" / "site-packages", "1.5.0.7")
    machine = box.machine("linux")
    machine.environ["CONDA_EXE"] = str(conda / "bin" / "conda")

    records = ic.find_old_installs(system=machine)

    assert {r.kind for r in records} == {"environment"}
    assert {(r.layout, r.root, r.version, r.python) for r in records} == {
        ("conda", str(env), "1.5.0.7", str(env / "bin" / "python")),
        ("venv", str(venv), "1.4.0", str(venv / "bin" / "python3")),
        ("user-site", str(user_site), "1.5.0.3", None)}


def test_a_ticked_user_site_is_uninstalled_only_with_a_python_found_on_path(tmp_path):
    box = Box(tmp_path)
    user_site = box.home / ".local" / "lib" / "python3.12"
    _dist(user_site / "site-packages", "1.5.0.3")
    machine = box.machine("linux", which={"python3.12": "/usr/bin/python3.12"}.get)

    [sandboxed] = ic.find_old_installs(system=machine)
    assert sandboxed.python is None, "a sandboxed computer never consults PATH"
    report = ic.remove_install(sandboxed, ticked=True, system=machine)
    assert report.failed == [(str(user_site), "no Python found in it; run pip uninstall spacr in that environment")]
    assert box.commands == []

    machine.real = True
    [record] = ic.find_old_installs(system=machine)
    assert record.python == "/usr/bin/python3.12"
    assert ic.remove_install(record, ticked=True, system=machine).ok
    assert box.commands == [["/usr/bin/python3.12", "-m", "pip", "uninstall", "-y",
                             "spacr"]]


def test_a_ticked_venv_made_without_pip_is_uninstalled_with_uv(tmp_path):
    box = Box(tmp_path)
    venv = box.home / ".venvs" / "analysis"
    box.write(venv / "bin" / "python")
    _dist(venv / "lib" / "python3.12" / "site-packages", "1.5.0.7")
    machine = box.machine("linux", which={"uv": "/opt/uv/bin/uv"}.get)
    machine.real = True

    [record] = ic.find_old_installs(system=machine)
    assert ic.remove_install(record, ticked=True, system=machine).ok

    assert box.commands == [["/opt/uv/bin/uv", "pip", "uninstall", "--python",
                             str(venv / "bin" / "python"), "spacr"]]


@pytest.mark.parametrize(("answer", "reason"), [
    ((1, "Found existing installation: spacr 1.5.0.7\n"
         "ERROR: Cannot uninstall spacr, RECORD file not found.\n"),
     "ERROR: Cannot uninstall spacr, RECORD file not found."),
    ((2, ""), "exit code 2"),
])
def test_a_failed_uninstall_from_your_environment_does_not_hold_the_install_back(
        tmp_path, answer, reason):
    box = Box(tmp_path)
    prefix = _conda_env(box, "analysis")
    machine = box.machine("linux")
    machine.runner = lambda argv: box.commands.append(list(argv)) or answer
    installs = []

    reports, result = ic.run_update_sequence(
        lambda: installs.append(1) or 0, ticked=[str(prefix)], system=machine)

    assert reports[0].failed == [(f"spacr in {prefix}", reason)]
    assert result == 0 and installs == [1]


@pytest.mark.parametrize(("direct_url", "kind"), [
    ({"url": "https://example.org/spacr-1.5.0.7-py3-none-any.whl",
      "archive_info": {}}, "environment"),
    ({"url": "", "dir_info": {"editable": True}}, "environment"),
    ({"url": "https://example.org/spacr", "dir_info": {"editable": True}}, "checkout"),
], ids=["wheel", "editable-without-a-source", "editable-elsewhere"])
def test_only_an_editable_install_is_a_checkout_and_a_checkout_is_never_touched(
        tmp_path, direct_url, kind):
    box = Box(tmp_path)
    prefix = _conda_env(box, "dev")
    info = prefix / "lib" / "python3.12" / "site-packages" / "spacr-1.5.0.7.dist-info"
    (info / "direct_url.json").write_text(json.dumps(direct_url), encoding="utf-8")
    machine = box.machine("linux")
    before = box.files()

    [record] = ic.find_old_installs(system=machine)
    reports, result = ic.run_update_sequence(
        lambda: "installed", records=[record], ticked=[record.root], system=machine)

    assert record.kind == kind
    assert record.root == (direct_url["url"] if kind == "checkout" else str(prefix))
    assert result == "installed" and box.files() == before
    if kind == "checkout":
        assert box.commands == [] and reports[0].skipped
    else:
        assert box.commands == [[str(prefix / "bin" / "python"), "-m", "pip",
                                 "uninstall", "-y", "spacr"]]


def test_a_version_comes_from_the_folder_name_when_the_metadata_gives_none(tmp_path):
    box = Box(tmp_path)
    root = linux_online(box, "1.5.0.7")
    site = root / "venv" / "lib" / "python3.12" / "site-packages"
    (site / "spacr-1.5.0.7.dist-info" / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: spacr\nVersion: \n\n", encoding="utf-8")
    box.write(site / "spacr_nightly-1.5.1.dev3.dist-info" / "METADATA",
              "Metadata-Version: 2.1\nName: spacr-nightly\n\nVersion: in the body\n")
    old = box.home / ".venvs" / "old"
    box.write(old / "bin" / "python")
    (old / "lib" / "python3.12" / "site-packages" / "spacr-1.4.2.dist-info").mkdir(
        parents=True)
    truncated = box.home / ".venvs" / "truncated"
    box.write(truncated / "bin" / "python")
    box.write(truncated / "lib" / "python3.12" / "site-packages" / "spacr-1.3.0.dist-info"
              / "METADATA", "Metadata-Version: 2.1\nName: spacr\n")

    records = ic.find_old_installs(system=box.machine("linux"))

    assert {r.root: r.version for r in records} == {
        str(root): "1.5.1.dev3", str(old): "1.4.2", str(truncated): "1.3.0"}


# ---------------------------------------------------------------------------
# The computer itself: defaults, commands, processes, downloads
# ---------------------------------------------------------------------------

def test_the_default_computer_is_this_one():
    machine = ic._Machine()

    assert machine.platform == ("windows" if os.name == "nt" else
                                "macos" if sys.platform == "darwin" else "linux")
    assert machine.fs_root == os.path.abspath("/")
    assert machine.running_prefix == sys.prefix
    assert machine.executable == sys.executable
    assert machine.which is shutil.which
    assert machine.path("/Applications") == "/Applications"
    assert machine.unmapped("/Applications/spaCR.app") == "/Applications/spaCR.app"


@pytest.mark.parametrize(("platform", "registry", "packages"), [
    ("windows", ic._WindowsRegistry, type(None)),
    ("linux", type(None), ic._SystemPackages),
    ("macos", type(None), type(None)),
])
def test_a_real_computer_gets_the_registry_or_package_manager_of_its_platform(
        platform, registry, packages):
    machine = ic._Machine(platform=platform)

    assert type(machine.registry) is registry
    assert type(machine.packages) is packages
    if packages is ic._SystemPackages:
        assert machine.packages._runner is machine.runner
        assert machine.packages._which is machine.which


def test_a_sandboxed_computer_gets_neither_and_keeps_its_own_spellings(tmp_path):
    machine = ic._Machine(platform="windows", environ={}, fs_root=str(tmp_path))

    assert not machine.real
    assert (machine.registry, machine.packages, machine.running_prefix) == (
        None, None, None)
    assert machine.unmapped(machine.path("/Applications/spaCR.app")) == (
        "/Applications/spaCR.app")
    outside = str(tmp_path.parent / "elsewhere")
    assert machine.unmapped(outside) == outside


def test_a_real_computer_with_no_home_variable_asks_the_operating_system(
        tmp_path, monkeypatch):
    monkeypatch.delenv("HOME")
    monkeypatch.delenv("USERPROFILE")
    monkeypatch.setattr(os.path, "expanduser",
                        lambda path: str(tmp_path) if path == "~" else path)

    assert ic._Machine(platform="linux").home == str(tmp_path)
    assert ic._Machine(platform="linux", environ={}, fs_root=str(tmp_path)).home == ""


def test_the_command_runner_returns_the_exit_code_and_everything_said():
    code, output = _REAL_RUN([
        sys.executable, "-c",
        "import sys; print('out'); sys.stdout.flush(); print('err', file=sys.stderr); "
        "sys.exit(3)"])
    assert (code, output) == (3, "out\nerr\n")


def test_a_command_that_is_not_there_is_exit_code_127(tmp_path):
    code, output = _REAL_RUN([str(tmp_path / "Uninstall.exe")])
    assert code == 127 and "could not run" in output


def test_a_command_that_never_finishes_is_given_up_on():
    code, output = _REAL_RUN([sys.executable, "-c", "import time; time.sleep(60)"],
                             timeout=0.5)
    assert code == 124 and "did not finish" in output


@pytest.mark.parametrize(("os_name", "detached"), [
    ("posix", {"start_new_session": True}),
    ("nt", {"creationflags": 0x8 | 0x200 | 0x8000000}),
])
def test_the_helper_is_started_detached_from_spacr(tmp_path, os_name, detached):
    started = []

    class Popen:
        def __init__(self, argv, **options):
            started.append((argv, options))
            self.pid = 4242

    pid = _REAL_SPAWN([str(tmp_path / "uv"), "run", 3], {"A": "1"}, str(tmp_path),
                      os_name=os_name, popen=Popen)

    assert pid == 4242
    assert started == [([str(tmp_path / "uv"), "run", "3"], {
        "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL, "close_fds": True, "cwd": str(tmp_path),
        "env": {"A": "1"}, **detached})]


@pytest.mark.skipif(os.name == "nt", reason="a POSIX session")
def test_a_detached_helper_really_runs_in_a_session_of_its_own(tmp_path):
    marker = tmp_path / "session"
    pid = _REAL_SPAWN([sys.executable, "-c",
                       f"import os, pathlib; pathlib.Path({str(marker)!r})"
                       f".write_text(str(os.getsid(0)))"], None, str(tmp_path))

    _pid, status = os.waitpid(pid, 0)

    assert os.waitstatus_to_exitcode(status) == 0
    assert int(marker.read_text()) == pid


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_the_helper_waits_until_spacr_has_exited():
    clock = _Clock()
    polls = []
    answers = iter([None, None, ProcessLookupError()])

    def kill(pid, signal):
        polls.append((pid, signal))
        answer = next(answers)
        if answer:
            raise answer

    assert ic._wait_for_exit("4242", timeout=10, os_name="posix", kill=kill,
                             sleep=clock.sleep, clock=clock)
    assert polls == [(4242, 0)] * 3 and clock.now == 101.0


def test_a_process_it_may_not_signal_is_waited_for_until_the_timeout():
    clock = _Clock()

    def kill(pid, signal):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    assert not ic._wait_for_exit(4242, timeout=2, os_name="posix", kill=kill,
                                 sleep=clock.sleep, clock=clock)
    assert clock.now == 102.0


def test_with_no_time_to_wait_nothing_is_asked():
    assert not ic._wait_for_exit(4242, timeout=0, os_name="posix",
                                 kill=_refuse("kill"), sleep=_refuse("sleep"),
                                 clock=lambda: 100.0)


@pytest.mark.skipif(os.name == "nt", reason="signal 0 is POSIX")
def test_a_process_that_already_exited_is_not_waited_for():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert ic._wait_for_exit(child.pid, timeout=5)


@pytest.mark.parametrize(("handle", "waited", "ended"), [
    (0, None, True), (77, 0, True), (77, 258, False)],
    ids=["already-gone", "exited", "still-running"])
def test_on_windows_the_helper_waits_on_the_process_handle(
        monkeypatch, handle, waited, ended):
    calls = []
    kernel32 = types.SimpleNamespace(
        OpenProcess=lambda access, inherit, pid: calls.append(
            ("open", access, inherit, pid)) or handle,
        WaitForSingleObject=lambda h, ms: calls.append(("wait", h, ms)) or waited,
        CloseHandle=lambda h: calls.append(("close", h)))
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(kernel32=kernel32),
                        raising=False)

    assert ic._wait_for_exit(4242, timeout=1.5, os_name="nt") is ended
    expected = [("open", 0x00100000, False, 4242)]
    if handle:
        expected += [("wait", 77, 1500), ("close", 77)]
    assert calls == expected


def test_the_helper_downloads_only_from_the_spacr_releases(tmp_path):
    target = tmp_path / "setup.run"
    for url in ("https://example.org/spaCR-9.9.9-Linux-x86_64-Online.run",
                ic._RELEASE_DOWNLOAD + ".example.org/v9.9.9/setup.run"):
        with pytest.raises(ValueError, match="refusing"):
            ic._download(url, str(target))
    assert not target.exists()


@pytest.mark.parametrize("body", [b"#!/bin/sh\necho installer\n", b""],
                         ids=["installer", "empty"])
def test_a_downloaded_installer_is_executable_and_an_empty_one_is_refused(
        tmp_path, monkeypatch, body):
    requests = []

    def urlopen(request, timeout):
        requests.append((request.full_url, request.get_header("User-agent"), timeout))
        return io.BytesIO(body)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    url = f"{ic._RELEASE_DOWNLOAD}/v9.9.9/spaCR-9.9.9-Linux-x86_64-Online.run"
    target = tmp_path / "setup.run"

    if body:
        ic._download(url, str(target))
        assert target.read_bytes() == body
        if os.name != "nt":
            assert stat.S_IMODE(target.stat().st_mode) == 0o755
    else:
        with pytest.raises(OSError, match="was empty"):
            ic._download(url, str(target))

    # A SECOND REQUEST SINCE 2026-09-20, and it is item 416's checksum.
    # `_download` now asks the release for the SHA256SUMS.txt beside the
    # asset and refuses a file that does not match, because the next thing
    # that happens to this file is chmod 755 and being run. This stand-in
    # answers every URL with the same body, so the sums file it sees names
    # no asset and no digest is enforced here -- which is the case
    # `test_a_sums_file_that_does_not_name_this_asset_is_not_a_failure`
    # covers on purpose in
    # tests/test_the_downloaded_installer_is_checked_before_it_is_run.py.
    # An empty download is refused before the sums are fetched at all, so
    # that arm still makes exactly one request.
    sums = f"{ic._RELEASE_DOWNLOAD}/v9.9.9/{ic._SUMS_NAME}"
    if body:
        assert requests == [(url, "spacr-updater", 120),
                            (sums, "spacr-updater", 60)]
    else:
        assert requests == [(url, "spacr-updater", 120)]


# ---------------------------------------------------------------------------
# The helper plan, beyond the cases above
# ---------------------------------------------------------------------------

def test_no_helper_starts_when_the_running_spacr_is_not_installer_made(tmp_path):
    box = Box(tmp_path / "box")
    prefix = _conda_env(box, "spacr")
    machine = box.machine("linux", running_prefix=str(prefix))
    records = ic.find_old_installs(system=machine)

    plan = ic.start_update_helper(records, "9.9.9", system=machine,
                                  spawn=_refuse("helper"))

    assert plan["command"] is None
    assert plan["error"] == "the running spaCR is not an installer-made copy"
    assert plan["pid"] == os.getpid()
    assert ic._inside(plan["workdir"], tempfile.gettempdir())
    written = json.loads((Path(plan["workdir"]) / "plan.json").read_text())
    assert written["error"] == plan["error"] and written["install"] is None


def test_a_mac_helper_opens_the_new_package_and_starts_the_app_again(tmp_path):
    box = Box(tmp_path / "box")
    macos_package(box, "1.5.0.5")
    runtime = box.home / "Library" / "Application Support" / "spaCR"
    machine = box.machine("macos", running_prefix=str(runtime / "venv"),
                          executable=str(runtime / "venv" / "bin" / "python"))
    records = ic.find_old_installs(system=machine)
    assert [r.running for r in _installers(records)] == [False, True]

    plan = ic.start_update_helper(records, "9.9.9", workdir=str(tmp_path / "h"),
                                  system=machine, spawn=lambda *args: None)

    assert plan["fetch"]["url"].endswith("/v9.9.9/spaCR-9.9.9-macOS-Universal-Online.pkg")
    assert plan["install"] == ["open", "-W", plan["fetch"]["path"]]
    assert plan["relaunch"] == ["open", "-a", "/Applications/spaCR.app"]
    assert plan["command"][0] == str(tmp_path / "h" / "uv")


@pytest.mark.parametrize("where", ["outside", "inside"])
def test_on_a_real_computer_the_helper_may_borrow_a_uv_from_path(tmp_path, where):
    box = Box(tmp_path / "box")
    root, machine, records = _running_desktop(box)
    (root / "bootstrap" / "uv").unlink()
    uv = box.write((tmp_path / "tools" / "uv") if where == "outside"
                   else (root / "python" / "uv"), "#!/bin/sh\n")
    machine.which = {"uv": str(uv)}.get
    machine.real = True

    plan = ic.start_update_helper(records, "9.9.9", workdir=str(tmp_path / "h"),
                                  system=machine, spawn=lambda *args: None)

    if where == "outside":
        assert plan["error"] is None
        assert plan["command"][0] == str(tmp_path / "h" / "uv")
        assert (tmp_path / "h" / "uv").read_text() == "#!/bin/sh\n"
    else:
        assert plan["command"] is None and "run the new installer" in plan["error"]


def _edit_plan(path, **changes):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data.update(changes)
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def test_the_helper_starts_nothing_when_the_installer_fails(tmp_path):
    box, root, machine, plan = _helper_plan(tmp_path)
    events = []

    code = ic._run_plan(plan, wait=lambda pid: True, fetch=lambda url, path: None,
                        run=lambda argv: 9,
                        spawn=lambda *args: events.append("relaunch"), system=machine)

    assert code == 9 and events == []
    log = (box.home / ".spacr" / "logs" / "update.log").read_text()
    assert "The installer failed with exit code 9." in log


def test_a_plan_without_a_download_relaunch_or_log_only_deletes_and_installs(
        tmp_path, capsys):
    box, root, machine, plan = _helper_plan(tmp_path)
    _edit_plan(plan, fetch=None, relaunch=None, log=None)
    events = []

    code = ic._run_plan(
        plan, wait=lambda pid: True, fetch=lambda *args: events.append("fetch"),
        run=lambda argv: events.append(("install", root.exists())) or (0, ""),
        spawn=lambda *args: events.append("relaunch"), system=machine)

    assert code == 0 and events == [("install", False)]
    assert "Installed." in capsys.readouterr().out
    assert not (box.home / ".spacr" / "logs" / "update.log").exists()


def test_an_update_log_that_cannot_be_written_does_not_stop_the_helper(
        tmp_path, capsys):
    box, root, machine, plan = _helper_plan(tmp_path)
    blocker = box.write(tmp_path / "a-file-not-a-folder")
    _edit_plan(plan, log=str(blocker / "update.log"))

    code = ic._run_plan(plan, wait=lambda pid: False, system=machine)

    assert code == 3 and "did not close" in capsys.readouterr().out


def test_a_plan_pointing_at_another_download_deletes_nothing(tmp_path):
    box, root, machine, plan = _helper_plan(tmp_path)
    _edit_plan(plan, fetch={"url": "https://example.org/setup.run",
                            "path": str(tmp_path / "setup.run")})
    events = []

    code = ic._run_plan(plan, wait=lambda pid: True,
                        run=lambda argv: events.append("install"),
                        remove=lambda *args, **kwargs: events.append("delete"),
                        system=machine)

    assert code == 4 and events == [] and root.exists()
    log = (box.home / ".spacr" / "logs" / "update.log").read_text()
    assert "refusing to download" in log


def test_the_helper_log_lists_your_environment_without_reporting_on_it(tmp_path):
    box = Box(tmp_path / "box")
    prefix = _conda_env(box, "analysis")
    root, machine, records = _running_desktop(box)
    ic.start_update_helper(records, "9.9.9", pid=1, workdir=str(tmp_path / "helper"),
                           system=machine, spawn=lambda *args: None)

    code = ic._run_plan(str(tmp_path / "helper" / "plan.json"),
                        wait=lambda pid: True, fetch=lambda url, path: None,
                        run=lambda argv: (0, ""), spawn=lambda *args: None,
                        system=machine)

    log = (box.home / ".spacr" / "logs" / "update.log").read_text()
    assert code == 0 and box.commands == []
    assert f"{prefix}  [your environment, left in place" in log
    assert "not ticked" not in log and f"removed: {root}" in log


def test_the_listing_marks_the_running_copy_and_its_notes():
    record = ic.InstallRecord(kind="installer", layout="linux-online",
                              platform="linux", root="/r", running=True,
                              notes=("its own uninstaller is missing",))
    odd = dataclasses.replace(record, kind="something-new", running=False,
                              version="2.0", notes=())

    assert ic._format_records([record, odd]) == [
        "  /r  [older copy, will be removed; linux-online, spaCR unknown version, "
        "running, its own uninstaller is missing]",
        "  /r  [something-new; linux-online, spaCR 2.0]"]


# ---------------------------------------------------------------------------
# The command line, in this process
# ---------------------------------------------------------------------------

def test_the_command_line_without_a_command_prints_its_help(capsys):
    assert ic._main([]) == 2
    assert "run-plan" in capsys.readouterr().out


def test_the_command_line_lists_what_it_finds(tmp_path, monkeypatch, capsys):
    box = Box(tmp_path)
    root = linux_online(box, "1.5.0.1")
    prefix = _conda_env(box, "analysis")
    monkeypatch.setenv("HOME", str(box.home))

    assert ic._main(["find", "--root", str(box.fs)]) == 0
    text = capsys.readouterr().out
    assert ic._main(["find", "--json", "--root", str(box.fs)]) == 0
    listed = json.loads(capsys.readouterr().out)

    assert text.startswith("Found 2 spaCR installation(s).")
    assert str(root) in text and str(prefix) in text
    assert [(r["kind"], r["root"]) for r in listed] == [
        ("installer", str(root)), ("environment", str(prefix))]


@pytest.mark.parametrize("denied", [False, True])
def test_the_command_line_exit_code_says_whether_every_old_copy_went(
        tmp_path, monkeypatch, capsys, denied):
    box = Box(tmp_path)
    root = linux_online(box, "1.5.0.7")
    prefix = _conda_env(box, "analysis")
    launcher = box.home / ".local" / "bin" / "spacr"
    monkeypatch.setenv("HOME", str(box.home))
    real_unlink = os.unlink

    def unlink(path, *args, **kwargs):
        if denied and str(path) == str(launcher):
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)

    code = ic._main(["remove", "--root", str(box.fs), "--keep",
                     str(root / "install.log")])

    out = capsys.readouterr().out
    assert code == (1 if denied else 0)
    assert f"left: {root / 'install.log'} (kept: part of the new installation)" in out
    assert (root / "install.log").exists() and not (root / "venv").exists()
    assert prefix.is_dir() and box.commands == []
    if denied:
        assert f"could not remove: {launcher} (in use or not permitted; close anything using it and try again)" in out
        assert "nothing new was installed" in out
    else:
        assert not launcher.exists()


@pytest.mark.parametrize("argv", [["find"], ["remove", "--sudo"]])
def test_the_command_line_looks_at_this_computer_but_not_as_the_running_spacr(
        monkeypatch, capsys, argv):
    seen = []
    monkeypatch.setattr(ic, "find_old_installs",
                        lambda system: seen.append(system) or [])

    assert ic._main(argv) == 0

    [machine] = seen
    assert machine.real and machine.running_prefix is None
    assert machine.sudo is ("--sudo" in argv)
    assert "0" in capsys.readouterr().out


def test_the_command_line_hands_a_plan_to_the_helper(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ic, "_run_plan", lambda path: seen.append(path) or 5)

    assert ic._main(["run-plan", str(tmp_path / "plan.json")]) == 5
    assert seen == [str(tmp_path / "plan.json")]


def test_the_module_runs_as_a_script(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["install_cleanup.py"])

    with pytest.raises(SystemExit) as exit_:
        runpy.run_path(ic.__file__, run_name="__main__")

    assert exit_.value.code == 2
    assert "run-plan" in capsys.readouterr().out
