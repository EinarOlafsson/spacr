"""Item 703: purge and uninstall choose exactly what they may delete, and stop on change.

The purge lists only spaCR-named folders once, outermost first, and says what
it could not delete. The uninstall data scan honours a home-relative
``SPACR_HOME``, relocated caches and the running interpreter, and treats an
unreadable folder as something to keep. The confirmed scope is refused when
it is malformed or no longer matches the computer.

Everything lives below ``tmp_path``; the registry and dpkg are fakes.
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

from spacr import install_cleanup as ic
from tests.test_install_cleanup import Box, linux_online
from tests.test_install_cleanup import (
    _nothing_here_reaches_this_computer as _nothing_here_reaches_this_computer,
)


def _machine(box, platform="linux", running_prefix=None, **extra):
    environ = {"HOME": str(box.home), "USERPROFILE": str(box.home),
               "LOCALAPPDATA": str(box.local), "APPDATA": str(box.appdata),
               "PROGRAMFILES": str(box.program_files), **extra}
    return ic._Machine(platform=platform, environ=environ, fs_root=str(box.fs),
                       registry=box.registry, packages=box.packages, runner=box.run,
                       running_prefix=running_prefix,
                       executable=str(box.base / "elsewhere" / "python"))


def test_dpkg_without_its_query_tool_owns_nothing():
    calls = []
    packages = ic._SystemPackages(lambda argv: calls.append(argv) or (0, ""),
                                  which=lambda name: None)
    assert packages.owns("spacr", "/opt/spacr/bin/spacr") is False
    assert calls == []


def test_purge_targets_skip_foreign_names_and_folders_inside_a_chosen_one(tmp_path):
    box = Box(tmp_path)
    app_home = box.home / ".spacr"
    (app_home / "backends" / "spacr-envs").mkdir(parents=True)
    foreign = box.home / "research-data"
    foreign.mkdir()
    machine = _machine(box, SPACR_HOME=str(foreign),
                       SPACR_BACKENDS_DIR=str(app_home / "backends" / "spacr-envs"))
    targets = ic._purge_targets(machine)
    assert str(foreign) not in targets
    assert str(app_home) in targets
    assert not any(ic._inside(path, str(app_home)) and path != str(app_home) for path in targets)


class _StuckRegistry:
    """A settings key whose subkey cannot be deleted."""

    def __init__(self):
        self.deleted = []

    def values(self, key):
        return {"theme": "dark"}

    def subkeys(self, key):
        return ["locked"] if key == ic._SETTINGS_REGISTRY_KEY else []

    def delete_key(self, key):
        if key.endswith("locked"):
            raise OSError(13, "Access is denied")
        self.deleted.append(key)


def test_a_registry_key_that_cannot_be_deleted_is_reported_and_kept(tmp_path, capsys):
    box = Box(tmp_path)
    machine = _machine(box)
    machine.registry = _StuckRegistry()
    assert ic._purge_command(machine, yes=True) == 1
    out = capsys.readouterr().out
    assert f"registry: HKCU\\{ic._SETTINGS_REGISTRY_KEY}" in out
    assert f"could not remove: registry:{ic._SETTINGS_REGISTRY_KEY}\\locked" in out
    assert machine.registry.deleted == [ic._SETTINGS_REGISTRY_KEY]


def test_a_purge_with_nothing_to_delete_says_so(tmp_path, capsys):
    machine = _machine(Box(tmp_path))
    assert ic._purge_command(machine, yes=True) == 0
    assert capsys.readouterr().out.strip() == "No spaCR caches, backends or user data were found."


class _Terminal(io.StringIO):
    def isatty(self):
        return True


def test_the_purge_question_is_asked_only_on_a_terminal(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    assert ic._ask_on_terminal("Purge? ") is None
    monkeypatch.setattr(sys, "stdin", _Terminal())
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "purge")
    assert ic._ask_on_terminal("Purge? ") == "purge" and asked == ["Purge? "]

    def closed(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", closed)
    assert ic._ask_on_terminal("Purge? ") is None


def test_uninstall_data_follows_home_relative_and_relocated_locations(tmp_path):
    box = Box(tmp_path)
    app_home = box.home / "custom-spacr"
    (app_home / "logs").mkdir(parents=True)
    models = tmp_path / "disk" / "spacr-models"
    news = tmp_path / "disk" / "not-spacr-news"
    models.mkdir(parents=True)
    news.mkdir()
    (app_home / ic._CACHE_LOCATIONS_FILE).write_text(json.dumps(
        {"SPACR_MODEL_CACHE": str(models), "SPACR_NEWS_CACHE": str(news),
         "SPACR_BACKENDS_DIR": 7}), encoding="utf-8")
    machine = _machine(box, SPACR_HOME="~/custom-spacr")
    rows = {path: (category, owned) for category, path, owned in
            ic._uninstall_data_candidates(machine)}
    assert rows[str(app_home / "logs")] == ("logs", True)
    assert rows[str(models)] == ("models", True)
    assert str(news) not in rows


def test_uninstall_data_with_unreadable_cache_locations_keeps_only_the_defaults(tmp_path):
    box = Box(tmp_path)
    (box.home / ".spacr" / "cache").mkdir(parents=True)
    (box.home / ".spacr" / ic._CACHE_LOCATIONS_FILE).write_text("{not json", encoding="utf-8")
    rows = ic._uninstall_data_candidates(_machine(box))
    assert ("caches", str(box.home / ".spacr" / "cache"), True) in rows


def test_the_running_interpreter_and_an_unreadable_parent_are_never_owned(tmp_path, monkeypatch):
    box = Box(tmp_path)
    plugins = box.home / ".spacr" / "plugins"
    (plugins / "env").mkdir(parents=True)
    machine = _machine(box, running_prefix=str(plugins),
                       SPACR_PLUGIN_HOME=str(plugins / "env"))
    rows = {path: owned for _category, path, owned in ic._uninstall_data_candidates(machine)}
    assert rows[str(plugins)] is False

    machine = _machine(box, SPACR_PLUGIN_HOME=str(plugins / "env"))
    real_listdir = os.listdir

    def unreadable(path):
        if os.fspath(path) == str(plugins):
            raise PermissionError(13, "Permission denied", str(plugins))
        return real_listdir(path)

    monkeypatch.setattr(ic.os, "listdir", unreadable)
    rows = ic._uninstall_data_candidates(machine)
    assert ("other", str(plugins), False) in rows


@pytest.fixture
def installation(tmp_path):
    box = Box(tmp_path)
    root = linux_online(box, "1.5.0.7")
    return box, root, box.machine("linux", running_prefix=str(root / "venv"))


def test_a_preview_needs_an_operation_and_reports_unverifiable_copies(installation, monkeypatch):
    box, root, machine = installation
    with pytest.raises(ValueError, match="operation_id"):
        ic._uninstall_preview("  ", system=machine)
    real_state = ic._uninstall_path_state

    def unverifiable(path):
        if os.path.normpath(path) == str(root):
            raise PermissionError(13, "denied")
        return real_state(path)

    monkeypatch.setattr(ic, "_uninstall_path_state", unverifiable)
    preview = ic._uninstall_preview("op", system=machine)
    row = next(row for row in preview["installations"] if row["record"]["root"] == str(root))
    assert row["reason"] == "unverified" and not row["selectable"] and row["identity"] == []


def test_settings_in_the_registry_are_a_selectable_cleanup_row(installation):
    box, _root, machine = installation
    box.registry.set(ic._SETTINGS_REGISTRY_KEY, theme="dark")
    preview = ic._uninstall_preview("op", system=machine)
    row = next(row for row in preview["cleanup"] if row["path"].startswith("registry:"))
    assert row == {"category": "settings", "path": "registry:" + ic._SETTINGS_REGISTRY_KEY,
                   "identity": [], "selectable": True, "reason": ""}
    selection = ic._uninstall_selection(preview, categories=["settings"])
    records, cleanup = ic._validate_uninstall_selection(preview, selection, machine,
                                                        after_shutdown=True)
    result = ic._apply_uninstall_records(preview, selection, [], cleanup, machine)
    assert result["ok"]
    assert "registry:" + ic._SETTINGS_REGISTRY_KEY in result["cleanup"]["removed"]
    assert not box.registry.exists(ic._SETTINGS_REGISTRY_KEY)


@pytest.mark.parametrize("scope, categories, error", [
    ("everything", None, "scope"),
    ("current", ["caches", "caches"], "categories"),
    ("current", ["nonsense"], "categories"),
])
def test_a_malformed_scope_is_refused(installation, scope, categories, error):
    _box, _root, machine = installation
    preview = ic._uninstall_preview("op", system=machine)
    with pytest.raises(ValueError, match=error):
        ic._uninstall_selection(preview, scope=scope, categories=categories)


def test_a_scope_with_nothing_removable_is_refused(installation):
    _box, _root, machine = installation
    preview = ic._uninstall_preview("op", system=machine)
    for row in preview["installations"]:
        row["selectable"] = False
    with pytest.raises(ValueError, match="No removable spaCR installation"):
        ic._uninstall_selection(preview)


def test_a_selected_copy_that_disappeared_stops_the_uninstall(installation, tmp_path):
    box, root, machine = installation
    other = Box(tmp_path / "second")
    linux_online(other, "1.5.0.6")
    preview = ic._uninstall_preview("op", system=machine)
    selection = ic._uninstall_selection(preview, scope="current")
    stale = dict(preview, installations=[dict(preview["installations"][0], key="unselected",
                                              current=False)]
                 + [dict(row, key="gone" + row["key"][4:])
                    if row["key"] in selection["installations"]
                    else row for row in preview["installations"]])
    stale_selection = dict(selection, installations=[
        "gone" + key[4:] for key in selection["installations"]])
    with pytest.raises(ValueError, match="installation changed"):
        ic._validate_uninstall_selection(stale, stale_selection, machine, after_shutdown=True)


def test_cleanup_that_changed_after_review_is_not_deleted(installation):
    box, _root, machine = installation
    logs = box.home / ".spacr" / "logs"
    box.write(logs / "today.log")
    preview = ic._uninstall_preview("op", system=machine)
    selection = ic._uninstall_selection(preview, categories=["logs"])
    _records, cleanup = ic._validate_uninstall_selection(preview, selection, machine,
                                                         after_shutdown=True)
    assert [row["path"] for row in cleanup] == [str(logs)]
    os.rename(logs, logs.with_name("logs-old"))
    logs.mkdir()
    with pytest.raises(ValueError, match="cleanup paths changed"):
        ic._apply_uninstall_records(preview, selection, [], cleanup, machine)
    assert logs.with_name("logs-old").joinpath("today.log").exists()


@pytest.fixture
def uv_cache_installation(tmp_path):
    if sys.platform != "linux":
        pytest.skip("linux-online UV caches are verified only on Linux "
                    "(/proc, O_NOFOLLOW, geteuid, symlinks and FIFOs)")
    box = Box(tmp_path)
    root = linux_online(box, "1.5.1.5")
    box.write(root / "cache/CACHEDIR.TAG", "Signature: 8a477f597d28d172789f06886806bc55\n")
    box.write(root / "keep-user-file.txt", "user-owned")
    return box, root, _machine(box, running_prefix=str(root / "venv"))


def test_legacy_online_uv_cache_is_previewed_and_purgeable_without_registration(uv_cache_installation):
    box, root, machine = uv_cache_installation
    cache = str(root / "cache")
    assert ("caches", cache, True) in ic._uninstall_data_candidates(machine)
    preview = ic._uninstall_preview("cache-preview", system=machine)
    row = next(row for row in preview["cleanup"] if row["path"] == cache)
    assert row["selectable"] and row["category"] == "caches" and row["identity"]
    assert cache in ic._purge_targets(machine)
    assert str(root) not in ic._purge_targets(machine)
    assert not (box.home / ".spacr" / ic._CACHE_LOCATIONS_FILE).exists()


@pytest.mark.parametrize("selected", [False, True])
def test_confirmed_uninstall_respects_uv_cache_optout_and_unknown_user_files(uv_cache_installation, selected):
    box, root, machine = uv_cache_installation
    preview = ic._uninstall_preview("cache-choice", system=machine)
    choice = ic._uninstall_selection(preview, categories=["caches"] if selected else [])
    outcome = ic._apply_uninstall_selection(preview, choice, system=machine, after_shutdown=True)
    assert outcome["ok"]
    assert (root / "cache").exists() is not selected
    assert (root / "keep-user-file.txt").read_text() == "user-owned"
    assert not (root / "venv").exists()


def test_cli_purge_deletes_verified_uv_cache_but_preserves_unknown_data(uv_cache_installation):
    box, root, machine = uv_cache_installation
    assert ic._purge_command(machine, yes=True) == 0
    assert not (root / "cache").exists()
    assert (root / "venv").is_dir()
    assert (root / "keep-user-file.txt").read_text() == "user-owned"


def test_installer_cache_registration_merges_existing_locations_atomically(uv_cache_installation):
    box, root, machine = uv_cache_installation
    manifest = box.write(box.home / ".spacr" / ic._CACHE_LOCATIONS_FILE,
                         json.dumps({"SPACR_MODEL_CACHE": "/kept/spacr-models", "other": "kept"}))
    assert ic._register_installer_cache(str(root), system=machine) == str(root / "cache")
    assert json.loads(manifest.read_text()) == {
        "SPACR_MODEL_CACHE": "/kept/spacr-models", "other": "kept",
        "SPACR_INSTALLER_CACHE": str(root / "cache")}
    assert sorted(path.name for path in manifest.parent.iterdir()) == [ic._CACHE_LOCATIONS_FILE]


@pytest.mark.parametrize("problem", ["cache_link", "tag_link", "bootstrap_link", "missing_tag", "bad_tag", "bad_metadata", "editable", "malformed_url", "malformed_url_directory", "tag_fifo", "metadata_fifo", "url_fifo", "url_link", "running", "shared"])
def test_unverified_linked_or_shared_cache_is_never_authorized(uv_cache_installation, tmp_path, problem):
    box, root, machine = uv_cache_installation
    cache = root / "cache"
    outside = tmp_path / "shared-data"
    outside.mkdir()
    sentinel = box.write(outside / "keep.txt", "untouched")
    if problem == "cache_link":
        cache.rename(outside / "original-cache")
        cache.symlink_to(outside / "original-cache", target_is_directory=True)
    elif problem in {"tag_link", "bootstrap_link"}:
        path = cache / "CACHEDIR.TAG" if problem == "tag_link" else root / "bootstrap/uv"
        target = outside / path.name
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    elif problem == "missing_tag":
        (cache / "CACHEDIR.TAG").unlink()
    elif problem == "bad_tag":
        (cache / "CACHEDIR.TAG").write_text("not a UV cache")
    elif problem == "bad_metadata":
        metadata = next((root / "venv").rglob("METADATA"))
        metadata.write_text("Name: unrelated\nVersion: 1.5.1.5\n")
    elif problem == "editable":
        metadata = next((root / "venv").rglob("METADATA"))
        metadata.with_name("direct_url.json").write_text(json.dumps(
            {"dir_info": {"editable": True}, "url": "file:///not-an-installer"}))
    elif problem == "malformed_url_directory":
        metadata = next((root / "venv").rglob("METADATA"))
        metadata.with_name("direct_url.json").write_text('{"dir_info": []}')
    elif problem == "malformed_url":
        metadata = next((root / "venv").rglob("METADATA"))
        metadata.with_name("direct_url.json").write_text("[]")
    elif problem in {"url_fifo", "url_link"}:
        path = next((root / "venv").rglob("METADATA")).with_name("direct_url.json")
        if problem == "url_fifo":
            os.mkfifo(path)
        else:
            target = box.write(outside / "direct_url.json", "{}")
            path.symlink_to(target)
    elif problem in {"tag_fifo", "metadata_fifo"}:
        path = cache / "CACHEDIR.TAG" if problem == "tag_fifo" else next((root / "venv").rglob("METADATA"))
        path.unlink()
        os.mkfifo(path)
    elif problem == "running":
        machine.running_prefix = str(cache / "active-python")
    elif problem == "shared":
        machine.environ["HF_HOME"] = str(cache / "shared-models")
    assert str(cache) not in ic._verified_installer_caches(machine)
    assert str(cache) not in ic._purge_targets(machine)
    with pytest.raises(ValueError, match="unverified_installer_cache"):
        ic._register_installer_cache(str(root), system=machine)
    assert sentinel.read_text() == "untouched"


@pytest.mark.parametrize("content", ["{broken", "[]"])
def test_registration_refuses_malformed_registry_without_overwriting(uv_cache_installation, content):
    box, root, machine = uv_cache_installation
    manifest = box.write(box.home / ".spacr" / ic._CACHE_LOCATIONS_FILE, content)
    with pytest.raises(ValueError):
        ic._register_installer_cache(str(root), system=machine)
    assert manifest.read_text() == content


def test_registration_refuses_linked_registry_and_forged_entry_cannot_authorize(uv_cache_installation, tmp_path):
    box, root, machine = uv_cache_installation
    foreign = box.write(tmp_path / "shared/spacr-sensitive/keep.txt", "preserved")
    manifest = box.write(box.home / ".spacr" / ic._CACHE_LOCATIONS_FILE,
                         json.dumps({ic._INSTALLER_CACHE_KEY: str(foreign.parent)}))
    assert str(foreign.parent) not in ic._purge_targets(machine)
    assert all(path != str(foreign.parent) for _category, path, _owned in ic._uninstall_data_candidates(machine))
    original = manifest.read_bytes()
    moved = manifest.with_name("real-registry.json")
    manifest.rename(moved)
    manifest.symlink_to(moved)
    with pytest.raises(OSError):
        ic._register_installer_cache(str(root), system=machine)
    assert moved.read_bytes() == original and foreign.read_text() == "preserved"


def test_general_cli_remove_purge_captures_cache_before_runtime_metadata_disappears(uv_cache_installation, monkeypatch):
    box, root, machine = uv_cache_installation
    monkeypatch.setattr(ic, "_Machine", lambda **kwargs: machine)
    assert ic._main(["remove", "--purge", "--yes"]) == 0
    assert not (root / "venv").exists()
    assert not (root / "cache").exists()
    assert (root / "keep-user-file.txt").read_text() == "user-owned"


def test_cli_purge_refuses_replaced_installer_cache_after_confirmation(uv_cache_installation):
    box, root, machine = uv_cache_installation
    cache = root / "cache"
    captured = [(str(cache), ic._uninstall_path_state(str(cache)))]

    def replace_at_confirmation(prompt):
        cache.rename(root / "original-cache")
        cache.mkdir()
        box.write(cache / "keep.txt", "replacement")
        return "purge"

    assert ic._purge_command(machine, ask=replace_at_confirmation, installer_caches=captured) == 1
    assert (cache / "keep.txt").read_text() == "replacement"
    assert (root / "original-cache/CACHEDIR.TAG").exists()
    assert (root / "keep-user-file.txt").read_text() == "user-owned"
