"""An accepted update must install and verify the offered version."""
import sys

import pytest

from spacr import updater


@pytest.mark.parametrize("uv", [None, "/private/runtime/uv"])
def test_offered_version_is_pinned_in_the_active_interpreter(monkeypatch, uv):
    monkeypatch.setattr(updater, "find_uv", lambda: uv)
    command = updater.upgrade_command(target_version="1.5.0.9")
    assert command[-1] == "spacr==1.5.0.9"
    assert sys.executable in command


@pytest.mark.parametrize("probe,success", [((0, "1.5.0.9\n"), True),
                                        ((0, "1.5.0.8\n"), False),
                                        ((1, "No package metadata was found"), False)])
def test_success_requires_a_fresh_verified_version(monkeypatch, probe, success):
    monkeypatch.setattr(updater, "editable_install_location", lambda: None)
    monkeypatch.setattr(updater, "find_uv", lambda: None)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return (0, "pip completed") if len(calls) == 1 else probe
    monkeypatch.setattr(updater, "run_install_command", run)
    result = updater.run_pip_upgrade(target_version="1.5.0.9")
    assert (result[0] == 0) == success
    assert calls[1][:3] == [sys.executable, "-I", "-c"]
    if not success:
        assert "did not verify" in result[1]


def test_source_checkout_cannot_report_a_requested_release_as_installed(monkeypatch):
    monkeypatch.setattr(updater, "editable_install_location", lambda: "/source/spacr")
    monkeypatch.setattr(updater, "run_install_command",
                        lambda *args, **kwargs: pytest.fail("source checkout must remain untouched"))
    code, explanation = updater.run_pip_upgrade(target_version="1.5.0.9")
    assert code != 0
    assert "git pull" in explanation


def test_relaunch_uses_this_interpreter_and_a_surviving_directory(monkeypatch, tmp_path):
    from spacr import install_cleanup
    monkeypatch.setenv("HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(install_cleanup, "_spawn_detached",
                        lambda command, **kwargs: calls.append((command, kwargs)) or 123)
    assert updater.launch_updated_app() == 123
    assert calls == [([sys.executable, "-m", "spacr.qt"], {"cwd": str(tmp_path)})]
