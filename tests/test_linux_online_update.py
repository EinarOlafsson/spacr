"""Linux online updates use the private POSIX transaction and saved choices."""
from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from spacr import install_cleanup as ic
from tests.test_macos_online_update_n627 import (
    installation as mac_installation,
    desktop_boundary,
    no_cleanup,
    test_upgrade_waits_verifies_then_relaunches_once as shared_upgrade,
    test_cancelled_shutdown_never_upgrades_on_later_exit as shared_cancel,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Exercises the Linux POSIX handshake")


@pytest.fixture
def installation(mac_installation):
    record, _, kept = mac_installation
    record = replace(record, platform="linux", layout="linux-online")
    (Path(record.root) / "install-profile.json").write_text(json.dumps({
        "schema": 1, "requested_backend": "cpu", "active_backend": "cpu", "without_wgpu": False}), encoding="utf-8")
    machine = ic._Machine(platform="linux", environ={"HOME": str(Path(record.root).parents[2])},
                          fs_root=str(Path(record.root).parents[3]),
                          executable=record.python, running_prefix=str(Path(record.python).parents[1]))
    return record, machine, kept


def test_linux_plan_preserves_cpu_and_optional_dependency_choices(tmp_path, installation):
    record, machine, kept = installation
    profile = {"schema": 1, "requested_backend": "cpu", "active_backend": "cpu", "without_wgpu": True}
    (Path(record.root) / "install-profile.json").write_text(json.dumps(profile), encoding="utf-8")
    plan = ic.start_update_helper([record], "1.5.1.1", workdir=str(tmp_path / "helper"),
                                  system=machine, spawn=lambda *args: None)
    assert plan["error"] is None
    assert plan["adapter"] == "linux-online-uv-v1" and plan["fetch"] is None
    assert plan["command"][:2] == [record.python, "-I"]
    assert plan["install"][6:8] == ["--torch-backend", "cpu"]
    override = Path(plan["install"][plan["install"].index("--override") + 1])
    assert override.read_text(encoding="utf-8").strip() == 'wgpu; sys_platform == "never"'
    assert kept.exists()


def test_linux_stages_before_shutdown_then_relaunches(tmp_path, installation, monkeypatch):
    shared_upgrade(tmp_path, installation, monkeypatch)


@pytest.mark.parametrize("cancel_at", ["before_ready", "after_ready", "while_waiting"])
def test_linux_cancel_preserves_active_environment(tmp_path, installation, monkeypatch, cancel_at):
    shared_cancel(tmp_path, installation, monkeypatch, cancel_at)


@pytest.mark.parametrize("missing", ["bootstrap/uv", "venv/bin/python"])
def test_linux_missing_private_bootstrap_never_falls_back_to_removal(tmp_path, installation, missing):
    record, machine, kept = installation
    (Path(record.root) / missing).unlink()
    plan = ic.start_update_helper([record], "1.5.1.1", workdir=str(tmp_path / "helper"),
                                  system=machine, spawn=no_cleanup)
    assert plan["adapter"] == "linux-online-uv-v1"
    assert plan["command"] is None and "cannot stage a verified update" in plan["error"]
    assert kept.exists()


def test_linux_older_target_cannot_mutate_the_running_runtime(tmp_path, installation):
    record, machine, kept = installation
    plan = ic.start_update_helper([record], "1.0.0", workdir=str(tmp_path / "helper"),
                                  system=machine, spawn=lambda *args: None)
    assert ic._run_plan(str(Path(plan["workdir"]) / "plan.json"), system=machine,
                        run=no_cleanup, remove=no_cleanup, spawn=no_cleanup) == 6
    assert kept.exists()
    assert ic._FrozenUpdateHandshake(plan).status()["state"] == "error"


@pytest.mark.parametrize("profile", [None, "not json"])
def test_linux_unavailable_profile_refuses_instead_of_changing_cpu_backend(tmp_path, installation, profile):
    record, machine, kept = installation
    path = Path(record.root) / "install-profile.json"
    if profile is None:
        path.unlink()
    else:
        path.write_text(profile, encoding="utf-8")
    plan = ic.start_update_helper([record], "1.5.1.1", workdir=str(tmp_path / "helper"),
                                  system=machine, spawn=no_cleanup)
    assert plan["command"] is None and "cannot stage a verified update" in plan["error"]
    assert kept.exists()
