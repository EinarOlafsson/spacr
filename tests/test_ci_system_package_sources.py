"""CI installs Ubuntu libraries without querying unrelated hosted vendor feeds."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
UBUNTU_SOURCES = "/etc/apt/sources.list.d/ubuntu.sources"


def _system_steps():
    steps = []
    for name in ("tests.yml", "_pytest-suite.yml"):
        workflow = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
        for job, definition in workflow["jobs"].items():
            for step in definition.get("steps", []):
                if step.get("name", "").startswith("Install system libraries"):
                    steps.append((f"{name}:{job}", step["run"]))
    workflow = yaml.safe_load((ROOT / ".github/workflows/compat-matrix.yml").read_text())
    for step in workflow["jobs"]["python315-linux"]["steps"]:
        if step.get("name", "").startswith("Install system libraries"):
            steps.append(("compat-matrix.yml:python315-linux", step["run"]))
    return steps


STEPS = _system_steps()


def test_all_four_system_library_steps_use_existing_ubuntu_definitions():
    assert len([name for name, _source in STEPS
                if name != "compat-matrix.yml:python315-linux"]) == 4
    assert [name for name, _source in STEPS
            if name == "compat-matrix.yml:python315-linux"] == [
                "compat-matrix.yml:python315-linux"]
    for _name, source in STEPS:
        assert f"ubuntu_sources={UBUNTU_SOURCES}" in source
        assert "allow-unauthenticated" not in source
        assert "AllowInsecure" not in source and "trusted=yes" not in source
        assert "libegl1" in source and "ffmpeg" in source


@pytest.mark.skipif(sys.platform != "linux", reason="The Ubuntu CI APT shell runs only on Linux")
@pytest.mark.parametrize("step", STEPS, ids=lambda row: row[0])
@pytest.mark.parametrize("failure", ["", "update", "install", "missing-sources"])
def test_unrelated_feed_is_not_queried_but_ubuntu_and_install_failures_are_fatal(
        tmp_path, step, failure):
    source = tmp_path / "ubuntu.sources"
    if failure != "missing-sources":
        source.write_text("Types: deb\nURIs: mirror+file:/etc/apt/apt-mirrors.txt\n"
                          "Suites: noble noble-updates\nComponents: main universe\n"
                          "Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n")
    original = source.read_bytes() if source.exists() else None
    commands = tmp_path / "commands.jsonl"
    sudo = tmp_path / "sudo"
    sudo.write_text('#!/bin/sh\nexec "$@"\n')
    sudo.chmod(0o755)
    apt = tmp_path / "apt-get"
    apt.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, sys
arguments = sys.argv[1:]
options = {}
while arguments and arguments[0] == "-o":
    key, value = arguments[1].split("=", 1)
    options[key] = value
    arguments = arguments[2:]
with open(os.environ["COMMANDS"], "a") as stream:
    stream.write(json.dumps([options, arguments]) + "\\n")
if options.get("Dir::Etc::sourceparts") != "-":
    print("Microsoft vendor feed: 403 Forbidden")
    sys.exit(100)
assert pathlib.Path(options["Dir::Etc::sourcelist"]).is_file()
if os.environ["FAIL_STAGE"] == arguments[0]:
    print("Required Ubuntu operation failed")
    sys.exit(100)
''')
    apt.chmod(0o755)
    environment = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
                       COMMANDS=str(commands), FAIL_STAGE=failure)
    script = step[1].replace(UBUNTU_SOURCES, str(source))
    result = subprocess.run(["bash", "-e", "-c", script], env=environment,
                            capture_output=True, text=True, timeout=10)
    calls = [json.loads(line) for line in commands.read_text().splitlines()] if commands.exists() else []
    if failure == "missing-sources":
        assert result.returncode != 0 and calls == []
        return
    assert source.read_bytes() == original, "mirror and signature definitions were modified"
    assert [call[1][0] for call in calls] == (["update"] if failure == "update" else ["update", "install"])
    assert all(call[0] == {"Dir::Etc::sourcelist": str(source), "Dir::Etc::sourceparts": "-"} for call in calls)
    assert result.returncode == (100 if failure else 0), result.stdout + result.stderr


def test_native_apt_source_selection_excludes_vendor_uris_without_network(tmp_path):
    apt = shutil.which("apt-get")
    if apt is None:
        pytest.skip("native APT source parser is Linux-specific")
    parts = tmp_path / "sources.list.d"
    parts.mkdir()
    source = parts / "ubuntu.sources"
    source.write_text("Types: deb\nURIs: https://archive.ubuntu.com/ubuntu\n"
                      "Suites: noble\nComponents: main universe\n"
                      "Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n")
    (parts / "vendor.list").write_text("deb https://packages.microsoft.com/ubuntu/24.04/prod noble main\n")
    (tmp_path / "lists/partial").mkdir(parents=True)
    base = [apt, "-o", f"Dir::Etc={tmp_path}", "-o", f"Dir::Etc::sourcelist={source}",
            "-o", f"Dir::State::lists={tmp_path / 'lists'}", "-o", "Debug::NoLocking=1"]
    control = subprocess.run([*base, "--print-uris", "update"], capture_output=True, text=True, timeout=10)
    assert control.returncode == 0, control.stderr
    assert "packages.microsoft.com" in control.stdout
    selected = subprocess.run([*base, "-o", "Dir::Etc::sourceparts=-", "--print-uris", "update"],
                              capture_output=True, text=True, timeout=10)
    assert selected.returncode == 0, selected.stderr
    assert "archive.ubuntu.com" in selected.stdout
    assert "packages.microsoft.com" not in selected.stdout


def test_python315_installs_required_qt_libraries_before_native_controls():
    """Qt-first controls need real runtime libraries even with offscreen QPA."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/compat-matrix.yml").read_text())
    steps = workflow["jobs"]["python315-linux"]["steps"]
    installs = [(index, step) for index, step in enumerate(steps)
                if step.get("name") == "Install system libraries"]
    assert len(installs) == 1
    index, install = installs[0]
    assert not install.get("if") and not install.get("continue-on-error", False)
    native = next(index for index, step in enumerate(steps)
                  if step.get("name") == "Check installed native imports in both cold-start orders")
    assert index < native
    shared = dict(STEPS)["_pytest-suite.yml:pytest"]
    assert install["run"] == shared
