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
import importlib.util
import json
import os
import plistlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from spacr import install_cleanup as ic

ROOT = Path(__file__).resolve().parents[1]
#: The capitalised spelling the 1.5.0.1 to 1.5.0.4 installers used, built so a
#: scan for the mis-cased project name does not mistake it for prose.
OLD = "Spa" + "CR"
TAGS = ("1.5.0.1", "1.5.0.4", "1.5.0.5", "1.5.0.6", "1.5.0.7")


def _capitalised(tag):
    return tag in ("1.5.0.1", "1.5.0.4")


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
                sudo=False):
        environ = {"HOME": str(self.home), "USERPROFILE": str(self.home),
                   "LOCALAPPDATA": str(self.local), "APPDATA": str(self.appdata),
                   "PROGRAMFILES": str(self.program_files)}
        return ic._Machine(
            platform=platform, environ=environ, fs_root=str(self.fs),
            registry=self.registry, packages=self.packages, runner=self.run,
            running_prefix=running_prefix,
            executable=executable or str(self.base / "elsewhere" / "python"),
            sudo=sudo)

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
