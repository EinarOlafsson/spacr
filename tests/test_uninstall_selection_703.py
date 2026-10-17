"""Confirmed uninstall scopes remove only disposable spaCR software and selected data."""
from __future__ import annotations

import json

import pytest

from spacr import install_cleanup as ic
from tests.test_install_cleanup import (
    Box,
    _conda_env,
    linux_online,
    macos_package,
)
from tests.test_install_cleanup import (
    _nothing_here_reaches_this_computer as _nothing_here_reaches_this_computer,
)


@pytest.fixture
def installation(tmp_path):
    box = Box(tmp_path)
    root = linux_online(box, "1.5.0.7")
    return box, root, box.machine("linux", running_prefix=str(root / "venv"))


def _preview_choice(machine, *, categories=None, scope="current"):
    preview = ic._uninstall_preview("this-dialog-only", system=machine)
    return preview, ic._uninstall_selection(preview, scope=scope, categories=categories)


def test_current_and_all_are_distinct_scopes_and_checkouts_stay(installation):
    box, root, machine = installation
    environment = _conda_env(box, "shared-science")
    source = box.home / "projects" / "source"
    box.write(source / "analysis.py")
    _conda_env(box, "editable", editable_source=source)
    preview, current = _preview_choice(machine)
    all_copies = ic._uninstall_selection(preview, scope="all")
    rows = {row["key"]: row for row in preview["installations"]}
    assert [rows[key]["record"]["root"] for key in current["installations"]] == [str(root)]
    assert {rows[key]["record"]["root"] for key in all_copies["installations"]} == {str(root), str(environment)}
    assert all_copies["categories"] == list(ic._UNINSTALL_CATEGORIES)
    checkout = next(row for row in rows.values() if row["record"]["kind"] == "checkout")
    assert not checkout["selectable"] and checkout["reason"] == "checkout"


def test_current_macos_online_scope_includes_its_application_launcher(tmp_path):
    box = Box(tmp_path)
    support = macos_package(box, "1.5.0.7")
    runtime = box.home / "Library" / "Application Support" / "spaCR"
    machine = box.machine("macos", running_prefix=str(runtime / "venv"))
    preview, choice = _preview_choice(machine, categories=[])
    selected = [row for row in preview["installations"] if row["key"] in choice["installations"]]
    assert {row["record"]["root"] for row in selected} == {str(support), str(runtime)}
    assert any(str(box.system("/Applications/spaCR.app")) in row["paths"] for row in selected)


@pytest.mark.parametrize("selected", ic._UNINSTALL_CATEGORIES)
def test_each_category_can_be_deleted_while_all_others_are_kept(installation, selected):
    box, root, machine = installation
    names = {"caches": "news", "models": "models", "logs": "logs", "settings": "settings",
             "history": "runs", "temporary": "tmp", "downloads": "downloads", "other": "plugins"}
    paths = {category: box.write(box.home / ".spacr" / name / "owned-file")
             for category, name in names.items()}
    project = box.write(box.home / ".spacr" / "my-research" / "measurements.db")
    preview, choice = _preview_choice(machine, categories=[selected])
    outcome = ic._apply_uninstall_selection(preview, choice, system=machine, after_shutdown=True)
    assert outcome["ok"]
    assert not (root / "venv" / "bin" / "python").exists()
    assert not paths[selected].exists()
    assert all(path.exists() for category, path in paths.items() if category != selected)
    assert project.read_text() == "x"


def test_current_application_is_never_removed_before_shutdown(installation):
    box, root, machine = installation
    preview, choice = _preview_choice(machine)
    before = box.files()
    with pytest.raises(ValueError, match="must close"):
        ic._apply_uninstall_selection(preview, choice, system=machine)
    assert box.files() == before and box.commands == []


def test_cancel_or_empty_data_choice_preserves_every_data_category(installation):
    box, root, machine = installation
    model = box.write(box.home / ".spacr" / "models" / "weights")
    preview, choice = _preview_choice(machine, categories=[])
    before = box.files()
    assert box.files() == before
    outcome = ic._apply_uninstall_selection(preview, choice, system=machine, after_shutdown=True)
    assert outcome["ok"] and model.exists()


def test_all_removes_only_spacr_from_shared_environments(installation):
    box, root, machine = installation
    prefix = _conda_env(box, "shared-science")
    unrelated = box.write(prefix / "lib" / "python3.12" / "site-packages" / "numpy" / "keep.py")
    preview, choice = _preview_choice(machine, scope="all", categories=[])
    outcome = ic._apply_uninstall_selection(preview, choice, system=machine, after_shutdown=True)
    assert outcome["ok"] and prefix.is_dir() and unrelated.exists()
    assert box.commands == [[str(prefix / "bin" / "python"), "-m", "pip", "uninstall", "-y", "spacr"]]


def test_shared_external_cache_and_project_override_are_displayed_but_kept(installation):
    box, root, machine = installation
    project = box.home / "research"
    file = box.write(project / "patient-data.csv")
    machine.environ["SPACR_MODEL_CACHE"] = str(project)
    preview, choice = _preview_choice(machine)
    row = next(row for row in preview["cleanup"] if row["path"] == str(project))
    assert not row["selectable"] and row["reason"] == "unverified"
    assert ic._apply_uninstall_selection(preview, choice, system=machine, after_shutdown=True)["ok"]
    assert file.exists()


def test_portable_model_opt_out_survives_parent_cache_cleanup(installation):
    box, root, machine = installation
    cache = box.home / ".spacr" / "cache"
    model = box.write(cache / "huggingface" / "weights")
    disposable = box.write(cache / "preview" / "image.png")
    machine.environ["HF_HOME"] = str(cache / "huggingface")
    preview, choice = _preview_choice(machine, categories=["caches"])
    assert ic._apply_uninstall_selection(preview, choice, system=machine, after_shutdown=True)["ok"]
    assert model.exists() and not disposable.exists()


def test_opted_out_models_nested_inside_removed_installation_are_preserved(installation):
    box, root, machine = installation
    data = root / "spacr-data"
    model = box.write(data / "models" / "weights")
    cache = box.write(data / "news" / "releases.json")
    machine.environ["SPACR_HOME"] = str(data)
    preview, choice = _preview_choice(machine, categories=["caches"])
    assert ic._apply_uninstall_selection(preview, choice, system=machine, after_shutdown=True)["ok"]
    assert not (root / "venv" / "bin" / "python").exists()
    assert model.exists() and not cache.exists()


@pytest.mark.parametrize("variable", ["HF_HOME", "TORCH_HOME", "SPACR_MODEL_CACHE"])
def test_spacr_named_external_folder_is_not_ownership_evidence(installation, variable):
    box, root, machine = installation
    project = box.home / "spacr-my-project"
    data = box.write(project / "valuable-model")
    machine.environ[variable] = str(project)
    preview, choice = _preview_choice(machine)
    assert not next(row for row in preview["cleanup"] if row["path"] == str(project))["selectable"]
    assert ic._apply_uninstall_selection(preview, choice, system=machine, after_shutdown=True)["ok"]
    assert data.exists()


@pytest.mark.parametrize("ancestor", [False, True])
def test_linked_data_paths_are_never_followed(installation, ancestor):
    box, root, machine = installation
    outside = box.home / "project"
    secret = box.write(outside / "models" / "important")
    app_home = box.home / ".spacr"
    if ancestor:
        app_home.symlink_to(outside, target_is_directory=True)
    else:
        app_home.mkdir()
        (app_home / "models").symlink_to(outside / "models", target_is_directory=True)
    preview, choice = _preview_choice(machine)
    row = next(row for row in preview["cleanup"] if row["category"] == "models")
    assert not row["selectable"]
    assert ic._apply_uninstall_selection(preview, choice, system=machine, after_shutdown=True)["ok"]
    assert secret.exists()


def test_replaced_installation_refuses_before_any_removal(installation):
    box, root, machine = installation
    preview, choice = _preview_choice(machine)
    root.rename(root.with_name("retained-original"))
    linux_online(box, "1.5.0.7")
    before = box.files()
    with pytest.raises(ValueError, match="installation changed"):
        ic._apply_uninstall_selection(preview, choice, system=machine, after_shutdown=True)
    assert box.files() == before and box.commands == []


def test_changed_cleanup_root_refuses_before_application_removal(installation):
    box, root, machine = installation
    models = box.home / ".spacr" / "models"
    box.write(models / "old")
    preview, choice = _preview_choice(machine)
    models.rename(models.with_name("retained-models"))
    box.write(models / "new")
    before = box.files()
    with pytest.raises(ValueError, match="cleanup paths changed"):
        ic._apply_uninstall_selection(preview, choice, system=machine, after_shutdown=True)
    assert box.files() == before


def test_normal_atomic_settings_save_during_shutdown_keeps_confirmed_file_scope(installation):
    box, root, machine = installation
    settings = box.write(box.home / ".spacr" / "network.json", "old settings")
    preview, choice = _preview_choice(machine, categories=["settings"])
    replacement = box.write(settings.with_suffix(".new"), "settings saved while closing")
    replacement.replace(settings)
    assert ic._apply_uninstall_selection(preview, choice, system=machine, after_shutdown=True)["ok"]
    assert not settings.exists()


def test_whitespace_home_override_uses_the_normal_app_home(installation):
    box, root, machine = installation
    model = box.write(box.home / ".spacr" / "models" / "weights")
    machine.environ["SPACR_HOME"] = "  "
    preview, choice = _preview_choice(machine, categories=["models"])
    assert any(row["path"] == str(model.parent) for row in preview["cleanup"])
    assert ic._apply_uninstall_selection(preview, choice, system=machine, after_shutdown=True)["ok"]
    assert not model.exists()


def test_selection_cannot_add_an_undisplayed_installation(installation):
    box, root, machine = installation
    preview, choice = _preview_choice(machine)
    choice["installations"].append("a-different-root")
    with pytest.raises(ValueError, match="selection changed"):
        ic._apply_uninstall_selection(preview, choice, system=machine, after_shutdown=True)
    assert (root / "venv" / "bin" / "python").exists()


def test_failed_software_removal_keeps_selected_data(installation, monkeypatch):
    box, root, machine = installation
    model = box.write(box.home / ".spacr" / "models" / "weights")
    preview, choice = _preview_choice(machine)
    def denied(record, **kwargs):
        return ic.RemovalReport(record, failed=[(record.root, "access denied")])
    monkeypatch.setattr(ic, "remove_install", denied)
    result = ic._apply_uninstall_selection(preview, choice, system=machine, after_shutdown=True)
    assert not result["ok"] and result["cleanup"] is None and model.exists()


def test_user_site_uninstall_does_not_match_other_packages_starting_with_spacr(tmp_path):
    box = Box(tmp_path)
    root = box.home / ".local" / "lib" / "python3.12"
    for name in ("spacr", "spacr_helpers"):
        box.write(root / "site-packages" / f"{name}-1.5.0.7.dist-info" / "METADATA",
                  f"Name: {name}\nVersion: 1.5.0.7\n")
    python = str(box.write(box.base / "python"))
    record = ic.InstallRecord("environment", "user-site", "linux", str(root), python=python)
    report = ic.remove_install(record, ticked=True, system=box.machine("linux"))
    assert report.ok and box.commands == [[python, "-m", "pip", "uninstall", "-y", "spacr"]]
    assert (root / "site-packages" / "spacr_helpers-1.5.0.7.dist-info" / "METADATA").exists()


def _prepared_helper(installation, tmp_path):
    box, root, machine = installation
    box.write(box.home / ".spacr" / "logs" / "old.log")
    preview, choice = _preview_choice(machine)
    calls = []
    plan = ic._start_uninstall_helper(preview, choice, system=machine,
        workdir=str(tmp_path / "outside-helper"), pid=98765,
        spawn=lambda *arguments: calls.append(arguments))
    return box, root, machine, plan, calls


def test_preparation_does_not_remove_anything_and_uses_outside_copied_source(installation, tmp_path):
    box, root, machine, plan, calls = _prepared_helper(installation, tmp_path)
    assert (root / "venv" / "bin" / "python").exists()
    assert plan["command"][-2:] == ["run-uninstall-plan", str(tmp_path / "outside-helper" / "plan.json")]
    assert calls == [(plan["command"], None, plan["workdir"])]
    assert (tmp_path / "outside-helper" / "install_cleanup.py").is_file()
    assert not (tmp_path / "outside-helper" / "approved.json").exists()


def test_approved_helper_requires_verified_process_exit_and_records_result(installation, tmp_path):
    box, root, machine, plan, calls = _prepared_helper(installation, tmp_path)
    handshake = ic._FrozenUpdateHandshake(plan)
    handshake.ready()
    handshake.approve()
    observed = []
    result = ic._run_uninstall_plan(str(tmp_path / "outside-helper" / "plan.json"),
        system=machine, wait=lambda pid: observed.append(pid) or True)
    assert result == 0 and observed == [98765]
    assert not (root / "venv" / "bin" / "python").exists()
    assert not (box.home / ".spacr" / "logs").exists()
    assert json.loads((tmp_path / "outside-helper" / "result.json").read_text())["ok"]


def test_cancelled_helper_does_not_remove_application_or_data(installation, tmp_path):
    box, root, machine, plan, calls = _prepared_helper(installation, tmp_path)
    ic._FrozenUpdateHandshake(plan).cancel()
    with pytest.raises(RuntimeError, match="cancelled"):
        ic._run_uninstall_plan(str(tmp_path / "outside-helper" / "plan.json"),
                              system=machine, wait=lambda pid: True)
    assert (root / "venv" / "bin" / "python").exists()
    assert (box.home / ".spacr" / "logs" / "old.log").exists()
    assert box.commands == []


def test_approval_without_process_exit_cannot_remove_application(installation, tmp_path):
    box, root, machine, plan, calls = _prepared_helper(installation, tmp_path)
    handshake = ic._FrozenUpdateHandshake(plan)
    handshake.ready()
    handshake.approve()
    def still_running(pid):
        assert pid == 98765
        handshake.cancel()
        return False
    with pytest.raises(RuntimeError, match="cancelled"):
        ic._run_uninstall_plan(str(tmp_path / "outside-helper" / "plan.json"),
                              system=machine, wait=still_running)
    assert (root / "venv" / "bin" / "python").exists()
    assert (box.home / ".spacr" / "logs" / "old.log").exists()


def test_saved_plan_cannot_change_cleanup_after_approval(installation, tmp_path):
    box, root, machine, plan, calls = _prepared_helper(installation, tmp_path)
    path = tmp_path / "outside-helper" / "plan.json"
    changed = json.loads(path.read_text())
    changed["selection"]["categories"] = []
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="binding"):
        ic._run_uninstall_plan(str(path), system=machine, wait=lambda pid: True)
    assert (root / "venv" / "bin" / "python").exists()


@pytest.mark.parametrize("blocked", ["application", "data"])
def test_permission_refusal_happens_before_helper_spawn_or_application_exit(installation, tmp_path, monkeypatch, blocked):
    box, root, machine = installation
    data = box.home / ".spacr" / "models"
    box.write(data / "weights")
    preview, choice = _preview_choice(machine)
    blocked_parent = root.parent if blocked == "application" else data.parent
    real_access = ic.os.access
    monkeypatch.setattr(ic.os, "access", lambda path, mode: False if str(path) == str(blocked_parent) else real_access(path, mode))
    before = box.files()
    spawned = []
    with pytest.raises(PermissionError, match="spaCR will stay open"):
        ic._start_uninstall_helper(preview, choice, system=machine,
            workdir=str(tmp_path / "refused-helper"), spawn=lambda *args: spawned.append(args))
    assert not spawned and box.files() == before
    assert not (tmp_path / "refused-helper").exists()


def test_shared_environment_package_permissions_are_checked_before_closing(installation, tmp_path, monkeypatch):
    box, root, machine = installation
    environment = _conda_env(box, "shared-science")
    preview, choice = _preview_choice(machine, scope="all", categories=[])
    shared = next(row for row in preview["installations"] if row["record"]["root"] == str(environment))
    package = next(path for path in shared["paths"] if path.endswith(".dist-info"))
    parent = ic.os.path.dirname(package)
    access = ic.os.access
    monkeypatch.setattr(ic.os, "access", lambda path, mode: False if str(path) == parent else access(path, mode))
    before = box.files()
    with pytest.raises(PermissionError, match="spaCR will stay open"):
        ic._start_uninstall_helper(preview, choice, system=machine,
            workdir=str(tmp_path / "refused-helper"), spawn=lambda *args: pytest.fail("no helper may start"))
    assert box.files() == before and box.commands == []


def test_shared_environment_parent_permission_is_not_needed_to_remove_only_spacr(installation, tmp_path, monkeypatch):
    box, root, machine = installation
    environment = _conda_env(box, "shared-science")
    preview, choice = _preview_choice(machine, scope="all", categories=[])
    access = ic.os.access
    monkeypatch.setattr(ic.os, "access", lambda path, mode: False if str(path) == str(environment.parent) else access(path, mode))
    calls = []
    plan = ic._start_uninstall_helper(preview, choice, system=machine,
        workdir=str(tmp_path / "allowed-helper"), spawn=lambda *args: calls.append(args))
    assert len(calls) == 1 and plan["selection"]["scope"] == "all"
    assert environment.is_dir() and box.commands == []


@pytest.fixture
def native_removal(tmp_path):
    import ctypes
    import shutil
    import subprocess
    from pathlib import Path

    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler unavailable")
    source = Path(__file__).resolve().parents[1] / "packaging/online/macos_gui_process.c"
    library = tmp_path / "native-removal.so"
    subprocess.run([compiler, "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                    str(source), "-o", str(library)], check=True)
    native = ctypes.CDLL(str(library)).spacr_uninstall_tree
    native.argtypes = [ctypes.c_char_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_uint,
                       ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t, ctypes.c_int]
    native.restype = ctypes.c_int

    def remove(path, keep=(), apply=False, identity=None):
        import os

        details = path.lstat()
        identities = identity or (details.st_dev, details.st_ino)
        kept = (ctypes.c_char_p * len(keep))(*(os.fsencode(item) for item in keep))
        return native(os.fsencode(path), *identities, os.getuid(), kept, len(keep), apply)

    return remove


def test_native_removal_preserves_nested_opt_out_and_never_follows_links(tmp_path, native_removal):
    root = tmp_path / "Application.app"
    contents = root / "Contents"
    contents.mkdir(parents=True)
    root.chmod(0o755)
    contents.chmod(0o755)
    (contents / "application").write_text("software")
    models = contents / "spacr-data" / "models"
    models.mkdir(parents=True)
    models.parent.chmod(0o755)
    models.chmod(0o755)
    model = models / "weights"
    model.write_text("preserved")
    outside = tmp_path / "project"
    outside.mkdir()
    valuable = outside / "measurements.csv"
    valuable.write_text("preserved")
    link = contents / "linked-project"
    link.symlink_to(outside, target_is_directory=True)
    assert native_removal(root, [models]) == 0
    assert (contents / "application").exists() and link.is_symlink()
    assert native_removal(root, [models], apply=True) == 0
    assert model.read_text() == valuable.read_text() == "preserved"
    assert not link.exists() and not (contents / "application").exists()


def test_native_preflight_refuses_writable_subtree_before_any_removal(tmp_path, native_removal):
    import errno

    root = tmp_path / "Application.app"
    child = root / "Contents"
    child.mkdir(parents=True)
    root.chmod(0o755)
    software = child / "application"
    software.write_text("software")
    child.chmod(0o777)
    assert native_removal(root) == errno.EPERM
    assert software.exists()


def test_native_removal_refuses_replaced_root_and_keeps_similar_sibling(tmp_path, native_removal):
    import errno

    root = tmp_path / "Application.app"
    root.mkdir()
    root.chmod(0o755)
    identity = (root.stat().st_dev, root.stat().st_ino)
    root.rename(tmp_path / "previous")
    root.mkdir()
    root.chmod(0o755)
    payload = root / "application"
    payload.write_text("new")
    assert native_removal(root, apply=True, identity=identity) == errno.ESTALE
    assert payload.exists()
    assert native_removal(root, [tmp_path / "Application.app-other"], apply=True) == 0
    assert not root.exists()


@pytest.fixture
def protected_macos(tmp_path, monkeypatch):
    box = Box(tmp_path)
    support = macos_package(box, "1.5.0.7")
    app = box.system("/Applications/spaCR.app")
    helper = box.write(app / "Contents/MacOS/install-spacr-gui")
    box.write(app / "Contents/Resources/uninstall-capabilities.json", '{"native_uninstall_schema":1}\n')
    for folder in (app, helper.parent.parent, helper.parent, helper.parent.parent / "Resources"):
        folder.chmod(0o755)
    helper.chmod(0o755)
    (helper.parent.parent / "Resources/uninstall-capabilities.json").chmod(0o644)
    launcher = box.system("/usr/local/bin/spacr")
    launcher.unlink()
    launcher.symlink_to(app / "Contents/MacOS/spaCR")
    runtime = box.home / "Library/Application Support/spaCR"
    machine = box.machine("macos", running_prefix=str(runtime / "venv"))
    preview, choice = _preview_choice(machine, categories=[])
    actual_access = ic.os.access
    denied = {str(app.parent), str(support.parent), str(launcher.parent)}
    monkeypatch.setattr(ic.os, "access", lambda path, mode: str(path) not in denied and actual_access(path, mode))
    records, _ = ic._validate_uninstall_selection(preview, choice, machine, after_shutdown=True)
    return box, machine, preview, choice, records, helper


def test_native_macos_preparation_authorizes_only_the_fixed_installed_helper(protected_macos, tmp_path):
    box, machine, preview, choice, records, helper = protected_macos
    adapter = ic._macos_native_uninstall_adapter(preview, records, machine)
    assert adapter["helper"] == str(helper)
    assert {target["path"] for target in adapter["request"]["targets"]} == {
        str(box.system("/Applications/spaCR.app")), str(box.system("/Library/Application Support/spaCR"))}
    before = box.files()
    plan = ic._start_uninstall_helper(preview, choice, system=machine,
        workdir=str(tmp_path / "outside"), spawn=lambda *args: None)
    assert plan["native"] == adapter
    assert box.files() == before | {str(tmp_path / "outside/install_cleanup.py"), str(tmp_path / "outside/plan.json")}
    command = ic._macos_native_uninstall_command(adapter, str(tmp_path / "phase"), 12345)
    assert command[:2] == ["/usr/bin/osascript", "-e"]
    assert "with administrator privileges" in command[2] and "--spacr-uninstall-native" in command[2]
    assert "install_cleanup.py" not in command[2]


@pytest.mark.parametrize("invalid", ["old_binary", "writable_binary", "linked_binary"])
def test_native_macos_preparation_refuses_untrusted_or_older_installed_helper(protected_macos, invalid):
    box, machine, preview, choice, records, helper = protected_macos
    if invalid == "old_binary":
        (helper.parent.parent / "Resources/uninstall-capabilities.json").unlink()
    elif invalid == "writable_binary":
        helper.chmod(0o777)
    else:
        helper.unlink()
        helper.symlink_to(box.write(box.home / "arbitrary-code"))
    with pytest.raises((OSError, ValueError)):
        ic._macos_native_uninstall_adapter(preview, records, machine)


@pytest.mark.parametrize("cancelled", [False, True])
def test_native_authorization_precedes_readiness_and_deletion_waits_for_exit(installation, tmp_path, monkeypatch, cancelled):
    box, root, machine, plan, calls = _prepared_helper(installation, tmp_path)
    handshake = ic._FrozenUpdateHandshake(plan)
    events = []

    class Channel:
        _connection = object()
        endpoint = str(tmp_path / "private-descriptor")

        def __init__(self, workdir):
            assert workdir == plan["workdir"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def _receive(self, phases, **kwargs):
            phase = next(iter(phases))
            events.append(phase)
            if phase == "prepared":
                assert handshake.status() is None
            return {"phase": phase, "details": {}}

        def _send(self, phase):
            events.append(phase)
            if phase == "activate":
                assert events.index("exited") < events.index("activate")

    class Process:
        def __init__(self, *args, **kwargs):
            events.append("authorization")

        def poll(self):
            return None if "complete" not in events else 0

        def wait(self, timeout):
            return 0

        def terminate(self):
            events.append("terminated")

    def ready():
        events.append("ready")
        ic._FrozenUpdateHandshake.ready(handshake)
        if cancelled:
            handshake.cancel()
        else:
            handshake.approve()

    def exited(pid):
        events.append("exited")
        return True

    monkeypatch.setattr(ic, "_UpdatePhaseServer", Channel, raising=False)
    monkeypatch.setattr(ic.subprocess, "Popen", Process)
    monkeypatch.setattr(handshake, "ready", ready)
    monkeypatch.setattr(ic, "_native_uninstall_command", lambda *args: ["native-authorization"])
    if cancelled:
        with pytest.raises(RuntimeError, match="cancelled"):
            ic._run_native_uninstall(plan, {"adapter": "macos-online-native"}, handshake, wait=exited)
        assert "activate" not in events and events[-2:] == ["cancel", "terminated"]
    else:
        ic._run_native_uninstall(plan, {"adapter": "macos-online-native"}, handshake, wait=exited)
        assert events == ["authorization", "prepared", "ready", "exited", "activate", "complete"]
    assert (root / "venv/bin/python").exists()


@pytest.mark.parametrize("decision,exited", [("activate", True), ("cancel", True), ("activate", False)])
def test_native_debian_removes_only_spacr_after_authorization_and_actual_exit(monkeypatch, decision, exited):
    from contextlib import nullcontext

    events = []
    helper = "/opt/spacr/_internal/spacr-update-helper"
    desktop = "/opt/spacr/spacr"
    monkeypatch.setattr(ic, "_debian_authenticated_user", lambda uid: events.append(("authorized", uid)))
    monkeypatch.setattr(ic, "_desktop_process_identity", lambda pid: (1000, desktop), raising=False)
    monkeypatch.setattr(ic, "_debian_check_package_owner", lambda version: helper)
    monkeypatch.setattr(ic.sys, "frozen", True, raising=False)
    monkeypatch.setattr(ic.sys, "executable", helper)
    monkeypatch.setattr(ic, "_debian_frontend_lock", nullcontext)
    monkeypatch.setattr(ic, "_wait_for_exit", lambda pid, timeout: events.append(("exited", exited)) or exited)
    monkeypatch.setattr(ic, "_debian_root_runner", lambda command: (events.append(command) or 0, ""))

    class Channel:
        def __init__(self, endpoint, pid, executable):
            assert (endpoint, pid, executable) == ("private-descriptor", 12345, desktop)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def _send(self, phase):
            events.append(phase)

        def _receive(self, phases, **kwargs):
            assert phases == {"activate", "cancel"}
            return {"phase": decision}

    monkeypatch.setattr(ic, "_UpdatePhaseClient", Channel, raising=False)
    if decision == "activate" and not exited:
        with pytest.raises(RuntimeError, match="must close"):
            ic._uninstall_debian_root("private-descriptor", 12345, desktop, 1000, "1.5.0.7")
    else:
        assert ic._uninstall_debian_root("private-descriptor", 12345, desktop, 1000, "1.5.0.7") == (0 if decision == "activate" else 3)
    assert ["/usr/bin/dpkg", "--no-act", "--remove", "spacr"] in events
    removal = ["/usr/bin/dpkg", "--remove", "spacr"]
    assert (removal in events) == (decision == "activate" and exited)
    if removal in events:
        assert events.index("prepared") < events.index(("exited", True)) < events.index(removal) < events.index("complete")


def test_native_debian_refuses_different_desktop_owner_before_package_commands(monkeypatch):
    monkeypatch.setattr(ic, "_debian_authenticated_user", lambda uid: uid)
    monkeypatch.setattr(ic, "_desktop_process_identity", lambda pid: (2000, "/opt/spacr/spacr"), raising=False)
    monkeypatch.setattr(ic, "_debian_check_package_owner", lambda version: pytest.fail("package inspection preceded owner binding"))
    with pytest.raises(ValueError, match="could not be identified"):
        ic._uninstall_debian_root("private-descriptor", 12345, "/opt/spacr/spacr", 1000, "1.5.0.7")


def test_copied_uninstall_helper_binds_original_desktop_executable(installation, tmp_path, monkeypatch):
    box, root, machine, plan, calls = _prepared_helper(installation, tmp_path)
    assert plan["desktop_executable"] == machine.executable
    observed = []

    def described(**values):
        observed.append(values)
        return machine

    handshake = ic._FrozenUpdateHandshake(plan)
    handshake.ready()
    handshake.approve()
    monkeypatch.setattr(ic, "_Machine", described)
    assert ic._run_uninstall_plan(str(tmp_path / "outside-helper/plan.json"), wait=lambda pid: True) == 0
    assert observed == [{"running_prefix": machine.running_prefix, "executable": machine.executable}]


@pytest.mark.parametrize("invalid", ["linked_marker", "writable_marker", "writable_resources", "oversized_marker"])
def test_native_macos_requires_trusted_bounded_capability_marker(protected_macos, invalid):
    box, machine, preview, choice, records, helper = protected_macos
    marker = helper.parent.parent / "Resources/uninstall-capabilities.json"
    if invalid == "linked_marker":
        marker.unlink()
        marker.symlink_to(box.write(box.home / "marker", '{"native_uninstall_schema":1}'))
    elif invalid == "writable_marker":
        marker.chmod(0o666)
    elif invalid == "writable_resources":
        marker.parent.chmod(0o777)
    else:
        marker.write_text('{"native_uninstall_schema":1}' + " " * 128 + "unexpected")
    with pytest.raises((OSError, ValueError)):
        ic._macos_native_uninstall_adapter(preview, records, machine)


@pytest.fixture
def protected_windows(tmp_path, monkeypatch):
    import hashlib

    root = tmp_path / "Program Files" / "spaCR"
    menu = tmp_path / "Common Programs"
    menu.mkdir()
    files = {"spacr.exe": b"desktop", "_internal/spacr-update-helper.exe": b"helper",
             "_internal/spacr-1.5.0.7.dist-info/METADATA": b"Name: spacr\nVersion: 1.5.0.7\n",
             "_internal/spacr-uninstall-protocol.json": b'{"schema":1,"adapter":"windows-frozen-uninstall-v1"}'}
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = root / "spacr-payload.json"
    manifest.write_text(json.dumps({"schema": 1, "version": "1.5.0.7",
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}))
    (root / "Uninstall.exe").write_bytes(b"native uninstaller")
    (root / "spacr-uninstaller.json").write_text(json.dumps({"schema": 1,
        "adapter": "windows-frozen-uninstall-v1", "sha256": ic._macos_digest(str(root / "Uninstall.exe"))}))
    shortcut = menu / "spaCR.lnk"
    shortcut.write_text(str(root))
    project = root / "my-project" / "measurements.csv"
    project.parent.mkdir()
    project.write_text("unrelated")
    empty_kept = root / "_internal" / "empty-cache"
    empty_kept.mkdir()
    monkeypatch.setattr(ic, "_windows_known_folder", lambda identifier: str(menu if identifier == 0x17 else root.parent))
    monkeypatch.setattr(ic, "_windows_system_owned", lambda path: None)
    request = {"targets": [{"root": str(root), "identity": list(ic._identity(root)),
        "version": "1.5.0.7", "manifest": ic._macos_digest(str(manifest)),
        "keep": [str(empty_kept)], "shortcuts": [str(shortcut)]}]}
    return root, request, project, empty_kept, shortcut


@pytest.mark.parametrize("invalid", ["outside_root", "changed_manifest", "foreign_shortcut", "outside_keep"])
def test_windows_native_scope_refuses_changed_or_unowned_paths(protected_windows, tmp_path, invalid):
    root, request, project, kept, shortcut = protected_windows
    target = request["targets"][0]
    if invalid == "outside_root":
        target["root"] = str(root.parent / "other-application")
    elif invalid == "changed_manifest":
        (root / "spacr.exe").write_text("changed")
    elif invalid == "foreign_shortcut":
        shortcut.write_text("unrelated application")
    else:
        target["keep"] = [str(tmp_path / "unrelated")]
    with pytest.raises(ValueError):
        ic._windows_uninstall_targets(request)
    assert project.read_text() == "unrelated" and (root / "spacr.exe").exists()


@pytest.mark.parametrize("decision,exited", [("activate", True), ("cancel", True), ("activate", False)])
def test_windows_native_removal_keeps_projects_and_empty_optouts_until_actual_exit(
        protected_windows, tmp_path, monkeypatch, decision, exited):
    root, request, project, kept, shortcut = protected_windows
    staged = tmp_path / "native-private" / "spacr-update-helper.exe"
    staged.parent.mkdir()
    staged.write_bytes((root / "_internal/spacr-update-helper.exe").read_bytes())
    request_path = tmp_path / "private-request.json"
    request_path.write_text(json.dumps(request))
    desktop = str(root / "spacr.exe")
    events = []
    monkeypatch.setattr(ic, "_windows_process_elevated", lambda: True)
    monkeypatch.setattr(ic.sys, "frozen", True, raising=False)
    monkeypatch.setattr(ic.sys, "executable", str(staged))
    monkeypatch.setattr(ic, "_desktop_process_identity", lambda pid: ("desktop-sid", desktop), raising=False)
    monkeypatch.setattr(ic, "_windows_pip_private_access", lambda path, **kwargs: events.append(("request-owner", kwargs)))
    monkeypatch.setattr(ic, "_wait_for_exit", lambda pid, timeout: events.append("exited") or exited)

    class Channel:
        def __init__(self, endpoint, pid, executable):
            assert (endpoint, pid, executable) == ("private-descriptor", 12345, desktop)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def _send(self, phase):
            events.append(phase)
            if phase == "prepared":
                assert (root / "spacr.exe").exists() and shortcut.exists()

        def _receive(self, phases, **kwargs):
            assert phases == {"activate", "cancel"}
            return {"phase": decision}

    monkeypatch.setattr(ic, "_UpdatePhaseClient", Channel, raising=False)
    def call():
        return ic._uninstall_frozen_windows(str(request_path), str(root), "private-descriptor", 12345, desktop)
    if decision == "activate" and not exited:
        with pytest.raises(RuntimeError, match="must close"):
            call()
    else:
        assert call() == (0 if decision == "activate" else 3)
    removed = decision == "activate" and exited
    assert (root / "spacr.exe").exists() is not removed
    assert shortcut.exists() is not removed
    assert project.read_text() == "unrelated" and kept.is_dir() and staged.exists()
    assert events[0] == ("request-owner", {"owner_sid": "desktop-sid", "private_read": True})
    if removed:
        assert events.index("prepared") < events.index("exited") < events.index("complete")


@pytest.mark.parametrize("owner,foreign_write,accepted", [
    ("S-1-5-32-544", False, True), ("desktop-sid", False, False), ("S-1-5-18", True, False)])
def test_windows_native_helper_requires_system_ownership_and_no_foreign_write(tmp_path, monkeypatch, owner, foreign_write, accepted):
    helper = tmp_path / "helper.exe"
    helper.write_bytes(b"helper")
    evidence = {"owner": owner, "has_dacl": True, "rules": [
        {"type": 0, "rights": 0x1F01FF, "propagation": 0, "sid": "S-1-5-32-544"}]}
    if foreign_write:
        evidence["rules"].append({"type": 0, "rights": 2, "propagation": 0, "sid": "desktop-sid"})
    monkeypatch.setattr(ic, "_windows_pip_acl", lambda path: evidence)
    if accepted:
        ic._windows_system_owned(str(helper))
    else:
        with pytest.raises(PermissionError):
            ic._windows_system_owned(str(helper))


def test_frozen_macos_descriptor_removal_preserves_optouts_and_external_link_targets(tmp_path):
    root = tmp_path / "spaCR.app"
    executable = root / "Contents/MacOS/spacr"
    executable.parent.mkdir(parents=True)
    executable.write_text("desktop")
    kept = root / "Contents/Resources/models"
    kept.mkdir(parents=True)
    (kept / "weights").write_text("keep")
    empty = root / "Contents/Resources/logs"
    empty.mkdir()
    project = tmp_path / "unrelated-project"
    project.mkdir()
    (project / "analysis.csv").write_text("keep outside")
    (root / "Contents/Resources/linked").symlink_to(project, target_is_directory=True)
    for path in [root, *[path for path in root.rglob("*") if path.is_dir() and not path.is_symlink()]]:
        path.chmod(0o755)
    identity = ic._identity(str(root))
    ic._macos_uninstall_tree(str(root), identity, [str(kept), str(empty)], owner=ic.os.getuid())
    assert executable.exists()
    ic._macos_uninstall_tree(str(root), identity, [str(kept), str(empty)], apply=True, owner=ic.os.getuid())
    assert not executable.exists() and (kept / "weights").read_text() == "keep" and empty.is_dir()
    assert (project / "analysis.csv").read_text() == "keep outside"


@pytest.mark.parametrize("invalid", ["writable_subtree", "changed_root", "outside_keep"])
def test_frozen_macos_descriptor_preflight_refuses_before_mutation(tmp_path, invalid):
    root = tmp_path / "spaCR.app"
    child = root / "Contents"
    child.mkdir(parents=True)
    payload = child / "payload"
    payload.write_text("old application")
    root.chmod(0o755)
    child.chmod(0o755)
    identity = ic._identity(str(root))
    keep = []
    if invalid == "writable_subtree":
        child.chmod(0o777)
    elif invalid == "changed_root":
        identity = (identity[0], identity[1] + 1)
    else:
        keep = [str(tmp_path / "unrelated")]
    with pytest.raises((ValueError, PermissionError)):
        ic._macos_uninstall_tree(str(root), identity, keep, owner=ic.os.getuid())
    assert payload.read_text() == "old application"


@pytest.mark.parametrize("decision,exited", [("activate", True), ("cancel", True), ("activate", False)])
def test_frozen_macos_authorized_helper_requires_activation_and_actual_exit(monkeypatch, decision, exited):
    from types import SimpleNamespace

    stage = "/private/var/root/spacr-helper.ABC12345/helper"
    target = "/Applications/spaCR.app"
    desktop = "/opt/spacr-desktop/venv/bin/python"
    events = []
    actual_lstat = ic.os.lstat
    actual_listdir = ic.os.listdir
    monkeypatch.setattr(ic.os, "geteuid", lambda: 0)
    monkeypatch.setattr(ic.sys, "frozen", True, raising=False)
    monkeypatch.setattr(ic.sys, "executable", stage)
    monkeypatch.setattr(ic.os, "lstat", lambda path, **kwargs: SimpleNamespace(st_uid=0, st_mode=0o700)
        if str(path) in {stage, ic.os.path.dirname(stage)} else actual_lstat(path, **kwargs))
    monkeypatch.setattr(ic.os, "listdir", lambda path: ["Contents", "my-project"]
        if path == target else actual_listdir(path))
    monkeypatch.setattr(ic, "_desktop_process_identity", lambda pid: (1000, desktop), raising=False)
    monkeypatch.setattr(ic, "_macos_frozen_trusted_helper", lambda: target + "/Contents/Frameworks/spacr-update-helper", raising=False)
    monkeypatch.setattr(ic, "_macos_digest", lambda path: "a" * 64)
    monkeypatch.setattr(ic, "_macos_bundle_identity", lambda *args: events.append("identity"))
    monkeypatch.setattr(ic, "_wait_for_exit", lambda pid, timeout: events.append("exited") or exited)
    monkeypatch.setattr(ic, "_macos_uninstall_tree", lambda path, identity, keep, **kwargs:
        events.append(("remove" if kwargs.get("apply") else "preflight", path, identity, keep)))

    class Channel:
        def __init__(self, endpoint, pid, executable):
            assert (endpoint, pid, executable) == ("descriptor", 12345, desktop)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def _send(self, phase):
            events.append(phase)

        def _receive(self, phases, **kwargs):
            assert phases == {"activate", "cancel"}
            return {"phase": decision}

    monkeypatch.setattr(ic, "_UpdatePhaseClient", Channel, raising=False)
    if decision == "activate" and not exited:
        with pytest.raises(RuntimeError, match="must close"):
            ic._uninstall_macos_root("descriptor", 12345, desktop, "1.5.0.7", "3:9", "[]")
    else:
        assert ic._uninstall_macos_root("descriptor", 12345, desktop, "1.5.0.7", "3:9", "[]") == (0 if decision == "activate" else 3)
    removals = [event for event in events if isinstance(event, tuple) and event[0] == "remove"]
    assert bool(removals) == (decision == "activate" and exited)
    if removals:
        assert removals == [("remove", target, (3, 9), [target + "/my-project"])]
        assert events.index("prepared") < events.index("exited") < events.index(removals[0]) < events.index("complete")


@pytest.mark.parametrize("cancelled", [False, True])
def test_windows_uninstall_bootstrap_exit_is_not_authenticated_completion(installation, tmp_path, monkeypatch, cancelled):
    from contextlib import contextmanager
    from threading import Event

    box, root, machine, plan, calls = _prepared_helper(installation, tmp_path)
    handshake = ic._FrozenUpdateHandshake(plan)
    uninstaller = tmp_path / "Uninstall.exe"
    uninstaller.write_bytes(b"native fixture")
    completed_bootstrap = Event()
    events = []
    pinned = []
    adapter = {"uninstaller": str(uninstaller), "request": {"targets": []},
               "uninstaller_sha256": ic._macos_digest(str(uninstaller)),
               "desktop_executable": machine.executable}

    @contextmanager
    def pin(path, digest):
        assert path == str(uninstaller)
        assert digest == adapter["uninstaller_sha256"]
        pinned.append(True)
        try:
            yield
        finally:
            pinned.clear()

    def authorize(path, arguments, **kwargs):
        assert path == str(uninstaller) and kwargs == {"visible": False}
        assert "/SPACRUNINSTALL=" in arguments and "/SPACRREQUEST=" in arguments
        events.append("bootstrap-exit")
        completed_bootstrap.set()
        return 0, ""

    class Channel:
        _connection = object()
        endpoint = str(tmp_path / "private-descriptor")

        def __init__(self, workdir):
            assert workdir == plan["workdir"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def _receive(self, phases, **kwargs):
            assert completed_bootstrap.wait(timeout=2)
            assert kwargs["stopped"]() is False
            phase = next(iter(phases))
            if phase == "prepared":
                assert pinned and handshake.status() is None
            else:
                assert not pinned and events[-1] == "activate"
            events.append(phase)
            return {"phase": phase}

        def _send(self, phase):
            events.append(phase)

    def ready():
        events.append("ready")
        ic._FrozenUpdateHandshake.ready(handshake)
        handshake.cancel() if cancelled else handshake.approve()

    monkeypatch.setattr(ic, "_windows_pip_private_access", lambda *args, **kwargs: None)
    monkeypatch.setattr(ic, "_windows_frozen_installer_pin", pin)
    monkeypatch.setattr(ic, "_windows_authorized_process", authorize)
    monkeypatch.setattr(ic, "_UpdatePhaseServer", Channel)
    monkeypatch.setattr(handshake, "ready", ready)
    def wait(pid):
        events.append("exited")
        return True
    if cancelled:
        with pytest.raises(RuntimeError, match="cancelled"):
            ic._run_windows_native_uninstall(plan, adapter, handshake, wait=wait)
        assert "activate" not in events and events[-1] == "cancel"
    else:
        ic._run_windows_native_uninstall(plan, adapter, handshake, wait=wait)
        assert events == ["bootstrap-exit", "prepared", "ready", "exited", "activate", "complete"]


@pytest.mark.parametrize("invalid", ["missing_receipt", "old_uninstaller", "oversized_receipt"])
def test_windows_new_payload_marker_does_not_authorize_an_old_uninstaller(protected_windows, invalid):
    root, request, project, kept, shortcut = protected_windows
    receipt = root / "spacr-uninstaller.json"
    if invalid == "missing_receipt":
        receipt.unlink()
    elif invalid == "old_uninstaller":
        (root / "Uninstall.exe").write_bytes(b"legacy uninstaller without the protocol")
    else:
        receipt.write_text(receipt.read_text() + " " * 1024)
    with pytest.raises((OSError, ValueError)):
        ic._windows_uninstall_targets(request)
    assert (root / "_internal/spacr-uninstall-protocol.json").exists()
    assert project.read_text() == "unrelated" and (root / "spacr.exe").exists()


def test_frozen_macos_authorizes_nonwritable_bundle_even_with_writable_applications(tmp_path, monkeypatch):
    box = Box(tmp_path)
    app = box.system("/Applications/spaCR.app")
    helper = box.write(app / "Contents/Frameworks/spacr-update-helper")
    box.write(app / "Contents/MacOS/spacr")
    machine = box.machine("macos")
    machine.real = True
    actual_access = ic.os.access
    protected = {str(app), str(app / "Contents"), str(helper.parent)}
    monkeypatch.setattr(ic.os, "access", lambda path, mode: False if str(path) in protected else actual_access(path, mode))
    assert ic.os.access(app.parent, ic.os.W_OK | ic.os.X_OK)
    checked = []
    monkeypatch.setattr(ic, "_macos_frozen_trusted_helper", lambda: checked.append("trusted") or str(helper))
    monkeypatch.setattr(ic, "_macos_bundle_identity", lambda path, version: {"SPACRPackageVersion": version})
    record = ic.InstallRecord(kind="installer", layout="macos-app", platform="macos", root=str(app),
        version="1.5.0.7", registrations=(str(app),), needs_admin=True)
    adapter = ic._macos_frozen_uninstall_adapter({"cleanup": []}, [record], machine)
    assert adapter["adapter"] == "macos-frozen-native" and checked == ["trusted"]
    assert adapter["roots"] == [str(app)]
