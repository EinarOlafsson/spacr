"""Confirmed uninstall scopes remove only disposable spaCR software and selected data."""
from __future__ import annotations

import json

import pytest

from spacr import install_cleanup as ic
from tests.test_install_cleanup import (
    Box,
    _conda_env,
    _nothing_here_reaches_this_computer as _nothing_here_reaches_this_computer,
    linux_online,
    macos_package,
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
