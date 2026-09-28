"""An update must retain the user's installed CPU/GPU wheel selection."""
import json

import pytest

from spacr import updater


@pytest.mark.parametrize("backend", ["cpu", "cu128", "auto", "cu126"])
def test_private_uv_update_uses_saved_choice_without_redetecting_hardware(
        tmp_path, monkeypatch, backend):
    """A CPU choice on an NVIDIA host stays CPU, even for a pinned update."""
    profile = tmp_path / "install-profile.json"
    profile.write_text(json.dumps({
        "schema": 1, "requested_backend": backend,
        "detected_accelerator": "nvidia", "active_backend": "cpu",
    }), encoding="utf-8")
    monkeypatch.setenv("SPACR_INSTALL_PROFILE", str(profile))
    monkeypatch.setattr(updater, "find_uv", lambda: "/private/bootstrap/uv")
    command = updater.upgrade_command(target_version="1.5.1.1")
    assert command[0] == "/private/bootstrap/uv"
    assert command[command.index("--torch-backend") + 1] == backend
    assert command[command.index("--python") + 1] == updater.sys.executable
    assert command[-1] == "spacr==1.5.1.1"


@pytest.mark.parametrize("uv", [None, "/ordinary/uv"])
def test_ordinary_environment_without_an_installer_profile_keeps_its_resolver(
        tmp_path, monkeypatch, uv):
    """Only an actual installer profile adds the uv-specific backend option."""
    monkeypatch.setenv("SPACR_INSTALL_PROFILE", str(tmp_path / "absent.json"))
    monkeypatch.setattr(updater, "find_uv", lambda: uv)
    command = updater.upgrade_command(pre_release=True)
    assert "--torch-backend" not in command
    assert "--pre" in command
    assert command[-1] == "spacr"
    assert command[0] == (uv or updater.sys.executable)


def test_invalid_profile_cannot_inject_extra_installer_arguments(tmp_path, monkeypatch):
    """The existing profile validator remains the boundary for resolver values."""
    profile = tmp_path / "install-profile.json"
    profile.write_text(json.dumps({
        "schema": 1, "requested_backend": "cpu --index-url https://invalid.example",
        "active_backend": "cpu",
    }), encoding="utf-8")
    monkeypatch.setenv("SPACR_INSTALL_PROFILE", str(profile))
    monkeypatch.setattr(updater, "find_uv", lambda: "/private/bootstrap/uv")
    command = updater.upgrade_command()
    assert "--torch-backend" not in command
    assert not any("invalid.example" in part for part in command)
