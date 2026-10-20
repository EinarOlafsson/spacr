"""Saved installer choices constrain both resolution and staged verification."""
import json
from pathlib import Path

import pytest

from spacr import install_cleanup as ic, install_profile
from tests.test_macos_online_update_n627 import installation, make_plan, approve_when_ready, _identity


def profile_for(record, **changes):
    profile = {"schema": 1, "requested_backend": "cpu", "active_backend": "cpu",
               "without_wgpu": True}
    profile.update(changes)
    path = Path(record.root) / "install-profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


@pytest.mark.parametrize("choice", [False, True, "unwritable", "unsupported"])
def test_online_update_binds_resolution_to_its_own_saved_profile(installation, tmp_path, monkeypatch, choice):
    record, _machine, _kept = installation
    profile_for(record, without_wgpu=choice is not False)
    other = tmp_path / "unrelated-profile.json"
    other.write_text(json.dumps({"schema": 1, "requested_backend": "cu128", "active_backend": "cuda"}), encoding="utf-8")
    monkeypatch.setenv("SPACR_INSTALL_PROFILE", str(other))
    written = []

    def override(profile, path):
        assert profile["without_wgpu"] is True
        written.append(path)
        return None if choice == "unwritable" else path

    monkeypatch.setattr(install_profile, "no_wgpu_override", override, raising=False)
    if choice == "unsupported":
        monkeypatch.delattr(install_profile, "no_wgpu_override")
    if choice in {"unwritable", "unsupported"}:
        plan = ic.start_update_helper([record], "1.5.1.1", workdir=str(tmp_path / "helper"),
            system=installation[1], spawn=lambda *_: pytest.fail("cannot start without the saved choice"))
        assert plan["command"] is None and "without wgpu" in plan["error"]
        return
    plan, _ = make_plan(tmp_path, installation)
    command = plan["install"]
    assert command[command.index("--torch-backend") + 1] == "cpu"
    assert ("--override" in command) is choice
    assert written == ([Path(plan["workdir"]) / "no-wgpu-override.txt"] if choice else [])
    if choice:
        assert command[command.index("--override") + 1] == str(written[0])
    assert command[-1] == "spacr"


@pytest.mark.parametrize("profile", ["broken json", '{"schema": 99}',
    '{"schema": 1, "requested_backend": "cpu", "active_backend": "cpu", "without_wgpu": "false"}'])
def test_unreadable_saved_choice_refuses_without_starting_update(installation, tmp_path, profile):
    record, machine, _ = installation
    (Path(record.root) / "install-profile.json").write_text(profile, encoding="utf-8")
    plan = ic.start_update_helper([record], "1.5.1.1", workdir=str(tmp_path / "helper"),
        system=machine, spawn=lambda *_: pytest.fail("invalid profile cannot start an update"))
    assert plan["command"] is None and "choices could not be read" in plan["error"]


def test_changing_saved_backend_during_staging_refuses_activation(installation, tmp_path, monkeypatch):
    record, machine, kept = installation
    path = profile_for(record, without_wgpu=False)
    before = kept.read_bytes()
    plan, plan_path = make_plan(tmp_path, installation)
    approve_when_ready(monkeypatch, plan)

    def wait(_pid):
        profile_for(record, without_wgpu=False, requested_backend="auto")
        return True

    result = ic._run_plan(str(plan_path), system=machine, wait=wait,
        run=lambda argv: (0, _identity("1.5.1.1")) if "-c" in argv else (0, "Updated"),
        spawn=lambda *_: pytest.fail("changed choice cannot activate"))
    assert result != 0
    assert "choices changed" in Path(plan["log"]).read_text(encoding="utf-8")
    assert kept.read_bytes() == before and path.exists()
    assert not list(Path(record.root).glob(".spacr-runtime-update-*"))


OMISSION = "The package `spacr` requires `wgpu>=0.25,<0.33; python_version >= '3.11'`, but it's not installed"
CHECK = "Checked 123 packages in 12ms\nFound 1 incompatibility\n" + OMISSION


@pytest.mark.parametrize("change", [None, "another-missing", "other-owner", "incompatible", "unknown",
    "duplicate", "wrong-exit", "pip-output", "blank", "different-name", "url"])
def test_only_one_explicit_missing_wgpu_diagnostic_is_eligible(change):
    output, code = CHECK, 1
    if change == "another-missing":
        output = CHECK.replace("1 incompatibility", "2 incompatibilities") + "\nThe package `spacr` requires `numpy`, but it's not installed"
    elif change == "other-owner":
        output = CHECK.replace("`spacr`", "`other-package`")
    elif change == "incompatible":
        output = CHECK.replace("it's not installed", "`0.1` is installed")
    elif change == "unknown":
        output += "\nUnknown metadata failure"
    elif change == "duplicate":
        output += "\n" + OMISSION
    elif change == "wrong-exit":
        code = 2
    elif change == "pip-output":
        output = "spacr 1.5.1.1 requires wgpu, which is not installed."
    elif change == "blank":
        output = ""
    elif change == "different-name":
        output = CHECK.replace("wgpu>=", "wgpu-utils>=")
    elif change == "url":
        output = CHECK.replace("wgpu>=0.25,<0.33; python_version >= '3.11'", "wgpu @ https://invalid.example/wgpu.whl")
    assert ic._only_saved_wgpu_omission(code, output) is (change is None)


@pytest.mark.parametrize("saved, extra", [(True, False), (False, False), (True, True)])
def test_dependency_exception_requires_saved_choice_and_preserves_other_failures(
        installation, tmp_path, monkeypatch, saved, extra):
    record, machine, kept = installation
    profile_for(record, without_wgpu=saved)
    monkeypatch.setattr(install_profile, "no_wgpu_override", lambda _profile, path: path, raising=False)
    plan, plan_path = make_plan(tmp_path, installation)
    offered = []

    def ready(handshake):
        offered.append(True)
        handshake.cancel()

    monkeypatch.setattr(ic._FrozenUpdateHandshake, "ready", ready)

    def run(argv):
        if argv[1:3] == ["pip", "check"]:
            assert (argv[-2:] == ["--color", "never"]) is saved
            output = CHECK
            if extra:
                output = output.replace("1 incompatibility", "2 incompatibilities") + "\nThe package `spacr` requires `numpy`, but it's not installed"
            return 1, output
        return (0, _identity("1.5.1.1")) if "-c" in argv else (0, "Updated")

    assert ic._run_plan(str(plan_path), system=machine, run=run,
        spawn=lambda *_: pytest.fail("the running desktop must be preserved")) != 0
    assert bool(offered) is (saved and not extra)
    assert kept.exists()
    assert not list(Path(record.root).glob(".spacr-runtime-update-*"))
