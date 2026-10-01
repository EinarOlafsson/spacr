"""Standalone updater preparation must not borrow the removed app's Python."""

import ast
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from spacr import install_cleanup as cleanup


def _frozen(tmp_path, monkeypatch, platform="linux", environ=None):
    """Create bundle-shaped bytes without launching a substitute executable."""
    installation = tmp_path / "installation"
    bundle = installation / "_internal"
    bundle.mkdir(parents=True)
    name = "spacr-update-helper" + (".exe" if platform == "windows" else "")
    (bundle / name).write_bytes(b"standalone helper fixture")
    executable = installation / "spacr"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    machine = SimpleNamespace(platform=platform, executable=str(executable),
                              environ=dict(environ or {}))
    record = cleanup.InstallRecord(kind="installer", layout="windows-offline",
                                   platform=platform, root=str(installation), running=True)
    return machine, record, bundle, name


@pytest.mark.parametrize("platform", ["linux", "windows", "macos"])
def test_copied_helper_survives_original_bundle_removal(tmp_path, monkeypatch, platform):
    """Copied bytes and extraction roots must survive deleting all old app files."""
    machine, record, bundle, name = _frozen(tmp_path, monkeypatch, platform)
    work = tmp_path / "update"
    plan = work / "plan.json"
    command, environment, error = cleanup._helper_command(
        [record], str(work), "nonexistent archived source.pyc", str(plan), machine)
    assert error is None
    assert command == [str(work / name), "run-plan", str(plan)]
    assert "-I" not in command
    assert not any("python" in argument.lower() for argument in command)
    shutil.rmtree(record.root)
    assert Path(command[0]).read_bytes() == b"standalone helper fixture"
    if os.name != "nt":
        assert stat.S_IMODE(Path(command[0]).stat().st_mode) == 0o700
    assert environment["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
    assert Path(environment["TMPDIR"]).is_dir()
    assert environment["TMP"] == environment["TEMP"] == environment["TMPDIR"]


def test_helper_environment_drops_parent_python_and_restores_loader_path(tmp_path, monkeypatch):
    """An independent extraction must not inherit the removed runtime's paths."""
    original = {"PYTHONPATH": "/old", "PYTHONHOME": "/old", "LD_LIBRARY_PATH": "/bundle",
                "LD_LIBRARY_PATH_ORIG": "/system", "LIBPATH": "/bundle", "PATH": "/usr/bin"}
    machine, record, _, _ = _frozen(tmp_path, monkeypatch, environ=original)
    work = tmp_path / "update"
    command, environment, error = cleanup._standalone_helper_command(
        [record], str(work), str(work / "plan.json"), machine)
    assert command and error is None
    assert environment["LD_LIBRARY_PATH"] == "/system"
    assert not {"PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH_ORIG", "LIBPATH"} & environment.keys()
    assert machine.environ == original


@pytest.mark.parametrize("escape", ["workdir", "symlink", "plan"])
def test_helper_refuses_paths_that_do_not_survive_removal(tmp_path, monkeypatch, escape):
    """Both symlinks and direct nested work folders must fail before copying."""
    machine, record, bundle, name = _frozen(tmp_path, monkeypatch)
    work = tmp_path / "update"
    plan = work / "plan.json"
    if escape == "workdir":
        work = Path(record.root) / "update"
        plan = work / "plan.json"
    elif escape == "symlink":
        (bundle / name).unlink()
        outside = tmp_path / "unrelated"
        outside.write_bytes(b"not bundled")
        (bundle / name).symlink_to(outside)
    else:
        plan = Path(record.root) / "plan.json"
    command, _, error = cleanup._standalone_helper_command([record], str(work), str(plan), machine)
    assert command is None and error
    assert not (work / name).exists()


def test_helper_refuses_mac_resources_workdir_even_without_discovery_record(tmp_path, monkeypatch):
    """The executable identifies its whole app, including sibling Resources."""
    app = tmp_path / "Custom.app"
    executable = app / "Contents/MacOS/spacr"
    bundle = app / "Contents/Frameworks"
    bundle.mkdir(parents=True)
    (bundle / "spacr-update-helper").write_bytes(b"bundled")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    machine = SimpleNamespace(platform="macos", executable=str(executable), environ={})
    work = app / "Contents/Resources/update"
    command, _, error = cleanup._standalone_helper_command([], str(work), str(work / "plan.json"), machine)
    assert command is None and "inside an installation" in error
    assert not work.exists()


def test_helper_canonicalizes_aliased_doomed_root(tmp_path, monkeypatch):
    """A symlinked installation record still owns its actual directory."""
    machine, record, bundle, name = _frozen(tmp_path, monkeypatch)
    alias = tmp_path / "alias"
    alias.symlink_to(record.root, target_is_directory=True)
    record = cleanup.replace(record, root=str(alias))
    machine.executable = str(tmp_path / "unrelated" / "separate-executable")
    work = bundle.parent / "update"
    command, _, error = cleanup._standalone_helper_command([record], str(work), str(work / "plan.json"), machine)
    assert command is None and "inside an installation" in error
    assert not (work / name).exists()


def test_helper_rejects_symlinked_plan_outside_private_folder(tmp_path, monkeypatch):
    """Lexically nested plan paths cannot redirect the surviving helper outside."""
    machine, record, _, _ = _frozen(tmp_path, monkeypatch)
    work = tmp_path / "update"
    work.mkdir(mode=0o700)
    plan = work / "plan.json"
    plan.symlink_to(tmp_path / "untrusted-plan.json")
    command, _, error = cleanup._standalone_helper_command([record], str(work), str(plan), machine)
    assert command is None and "plan must stay inside" in error


def test_helper_does_not_fold_case_on_a_case_sensitive_filesystem(tmp_path, monkeypatch):
    """Case-distinct sibling trees are outside the actual bundle on Linux."""
    machine, record, bundle, name = _frozen(tmp_path, monkeypatch)
    outside = bundle.parent / "_INTERNAL"
    if outside.exists():
        pytest.skip("the actual filesystem is case-insensitive")
    outside.mkdir()
    (outside / name).write_bytes(b"outside bundle")
    (bundle / name).unlink()
    (bundle / name).symlink_to(outside / name)
    work = tmp_path / "update"
    command, _, error = cleanup._standalone_helper_command([record], str(work), str(work / "plan.json"), machine)
    assert command is None and "outside its bundle" in error


def test_helper_returns_canonical_plan_path(tmp_path, monkeypatch):
    """A safe alias resolves once before it becomes detached process argv."""
    machine, record, _, _ = _frozen(tmp_path, monkeypatch)
    work = tmp_path / "update"
    work.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(work, target_is_directory=True)
    command, _, error = cleanup._standalone_helper_command([record], str(work), str(alias / "plan.json"), machine)
    assert error is None and command[-1] == str(work / "plan.json")


def test_helper_never_overwrites_existing_destination(tmp_path, monkeypatch):
    """A preexisting target or symlink is not trusted executable material."""
    machine, record, _, name = _frozen(tmp_path, monkeypatch)
    work = tmp_path / "update"
    work.mkdir(mode=0o700)
    target = work / name
    target.write_bytes(b"preserve")
    command, _, error = cleanup._standalone_helper_command([record], str(work), str(work / "plan.json"), machine)
    assert command is None and error
    assert target.read_bytes() == b"preserve"


@pytest.mark.skipif(os.name == "nt", reason="POSIX folder permission semantics")
def test_helper_refuses_a_shared_workdir(tmp_path, monkeypatch):
    """Other users must not be able to replace a prepared updater's files."""
    machine, record, _, name = _frozen(tmp_path, monkeypatch)
    work = tmp_path / "shared"
    work.mkdir()
    work.chmod(0o777)
    command, _, error = cleanup._standalone_helper_command([record], str(work), str(work / "plan.json"), machine)
    assert command is None and "private" in error
    assert not (work / name).exists()


@pytest.mark.parametrize("layout,platform", [("windows-offline", "windows"), ("linux-deb", "linux"), ("macos-app", "macos")])
def test_frozen_family_cannot_fall_back_to_online_replacement(tmp_path, layout, platform):
    """A usable helper must not activate the unrelated online installer recipe."""
    root = tmp_path / ("spacr.app" if platform == "macos" else "spacr")
    record = cleanup.InstallRecord(kind="installer", layout=layout, platform=platform,
                                   root=str(root), running=True)
    machine = cleanup._Machine(environ={}, fs_root=str(tmp_path))
    with pytest.raises(ValueError, match="frozen replacement adapter"):
        cleanup._reinstall_steps(record, "1.5.1.1", str(tmp_path / "update"), machine)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"records": [asdict(record)], "pid": 123,
                                "version": "1.5.1.1", "log": str(tmp_path / "update.log")}))

    def forbidden(*args, **kwargs):
        """No wait, fetch, removal, installation or relaunch is authorized."""
        pytest.fail("unsupported frozen plan continued")

    assert cleanup._run_plan(str(path), system=machine, wait=forbidden, fetch=forbidden,
                             remove=forbidden, run=forbidden, spawn=forbidden) == 2
    assert "nothing was changed" in (tmp_path / "update.log").read_text()


def test_bundle_builds_helper_from_existing_stdlib_entry_and_embeds_it():
    """The helper must be an independent EXE containing binaries and stdlib."""
    spec = Path(__file__).resolve().parents[1] / "packaging/spacr.spec"
    tree = ast.parse(spec.read_text())
    assignments = {node.targets[0].id: node.value for node in tree.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    analysis = assignments["helper_analysis"]
    assert analysis.func.id == "Analysis"
    assert "install_cleanup.py" in ast.unparse(analysis.args[0])
    arguments = {keyword.arg: keyword.value for keyword in analysis.keywords}
    assert ast.literal_eval(arguments["hiddenimports"]) == []
    assert {"spacr", "torch", "PySide6", "numpy"} <= set(ast.literal_eval(arguments["excludes"]))
    executable = assignments["helper_exe"]
    assert executable.func.id == "EXE"
    assert any(ast.unparse(argument) == "helper_analysis.binaries" for argument in executable.args)
    assert not ast.literal_eval(next(keyword.value for keyword in executable.keywords if keyword.arg == "exclude_binaries"))
    assert any(isinstance(node, ast.AugAssign) and ast.unparse(node.target) == "binaries"
               and "helper_exe.name" in ast.unparse(node.value) for node in tree.body)


@pytest.mark.parametrize("path", [
    "packaging/build_debian.sh", "packaging/build_macos.sh",
    "packaging/build_windows.ps1", ".github/workflows/frozen-installers.yml",
])
def test_every_helper_builder_requires_the_environment_reset_bootloader(path):
    """PyInstaller introduced the public independent-extraction flag in 6.10."""
    source = (Path(__file__).resolve().parents[1] / path).read_text()
    assert "pyinstaller>=6.10,<7" in source
    assert "pyinstaller>=6,<7" not in source


@pytest.mark.parametrize("job,step_name,source,copy,removal,launch", [
    ("smoke-frozen", "Install and launch the actual Windows artifact",
     "_internal/spacr-update-helper.exe", "Copy-Item -LiteralPath $HelperSource",
     "$Uninstall = Start-Process", "$Child = Start-Process $Helper"),
    ("smoke-frozen", "Install and launch the actual macOS artifact",
     "Contents/Frameworks/spacr-update-helper", 'cp -p "$helper_source" "$helper"',
     'rm -rf -- "$install_root/spaCR.app"', "run_helper help --help"),
    ("smoke-debian", "Install Debian artifact without source or pip supplementation",
     "/opt/spacr/_internal/spacr-update-helper", 'cp -p "$helper_source" "$helper"',
     "dpkg --remove spacr", "/usr/bin/timeout --kill-after=5s 60s"),
])
def test_native_workflow_requires_copied_helper_after_actual_removal(job, step_name, source, copy, removal, launch):
    """Native jobs must require real packaged bytes, isolated CLI and preserved data."""
    import yaml

    workflow = Path(__file__).resolve().parents[1] / ".github/workflows/frozen-installers.yml"
    steps = yaml.safe_load(workflow.read_text())["jobs"][job]["steps"]
    assert not any("actions/setup-python" in step.get("uses", "") for step in steps)
    step = next(item for item in steps if item.get("name") == step_name)
    assert not step.get("continue-on-error", False)
    script = step["run"]
    assert source in script
    assert script.index(copy) < script.index(removal) < script.index(launch)
    assert "helper-survival.json" in script and "helper-sha256.txt" in script
    assert "run-plan" in script and "--help" in script
    assert "not update execution" in script
    assert "PYINSTALLER_RESET_ENVIRONMENT" in script
    if job == "smoke-frozen" and "Windows" in step_name:
        assert "$env:PATH = ''" in script and "Remove-Item -LiteralPath ('Env:'" in script
        assert "WaitForExit(60000)" in script
        assert "Test-Path $Result.database" in script and "Test-Path $Unrelated" in script
        assert "Test-Path $HelperSource" in script
    else:
        assert "/usr/bin/env -i" in script and "PATH=" in script
        assert 'test -s "$database"' in script and 'test ! -e "$helper_source"' in script
        if "macOS" in step_name:
            assert script.index("native_menu.quit_observed") < script.index(removal)
            assert "/bin/sleep 60" in script


@pytest.mark.parametrize("version,accepted", [("6.9.0", False), ("6.10.0", True), ("6.22.3", True)])
def test_spec_rejects_old_bootloaders_even_when_dependency_install_is_skipped(version, accepted):
    """Preprovisioned builders must not silently omit the helper isolation mechanism."""
    from packaging.version import Version
    spec = Path(__file__).resolve().parents[1] / "packaging/spacr.spec"
    tree = ast.parse(spec.read_text())
    guard = next(node for node in tree.body if isinstance(node, ast.If)
                 and "PYINSTALLER_VERSION" in ast.unparse(node.test))
    code = compile(ast.Module(body=[guard], type_ignores=[]), str(spec), "exec")
    if accepted:
        exec(code, {"Version": Version, "PYINSTALLER_VERSION": version})
    else:
        with pytest.raises(RuntimeError, match="requires PyInstaller"):
            exec(code, {"Version": Version, "PYINSTALLER_VERSION": version})


def test_native_helper_shell_blocks_parse_without_running_installers():
    """Validate the actual shell programs without performing privileged actions."""
    import yaml

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required to parse the POSIX native smoke programs")
    workflow = Path(__file__).resolve().parents[1] / ".github/workflows/frozen-installers.yml"
    jobs = yaml.safe_load(workflow.read_text())["jobs"]
    checked = 0
    for name in ("smoke-frozen", "smoke-debian"):
        for step in jobs[name]["steps"]:
            if step.get("shell") != "bash" or "helper-survival.json" not in step.get("run", ""):
                continue
            result = subprocess.run([bash, "-n"], input=step["run"], text=True,
                                    capture_output=True, timeout=10)
            assert result.returncode == 0, (step["name"], result.stderr)
            checked += 1
    assert checked == 2


def test_actual_frozen_helper_runs_after_copy_without_source_or_python(tmp_path):
    """Optional native acceptance executes the produced helper, never a script substitute."""
    supplied = os.environ.get("SPACR_TEST_FROZEN_HELPER")
    if not supplied:
        pytest.skip("requires an actual native PyInstaller helper artifact")
    source = Path(supplied).resolve(strict=True)
    assert source.name in {"spacr-update-helper", "spacr-update-helper.exe"}
    magic = source.read_bytes()[:4]
    assert magic == b"\x7fELF" or magic[:2] == b"MZ" or magic in {
        b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",
    }, "the native acceptance input must be an executable binary, not a script"
    original = tmp_path / "disposable-original"
    original.mkdir()
    fixture = original / source.name
    shutil.copy2(source, fixture)
    target = tmp_path / source.name
    shutil.copy2(fixture, target)
    shutil.rmtree(original)
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LIBPATH"}}
    environment.update(PATH="", PYINSTALLER_RESET_ENVIRONMENT="1", TMPDIR=str(tmp_path), TMP=str(tmp_path), TEMP=str(tmp_path))
    completed = subprocess.run([str(target), "--help"], cwd=tmp_path, env=environment,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert "run-plan" in completed.stdout and "install_cleanup" in completed.stdout
    plan = tmp_path / "refused-plan.json"
    sentinel = tmp_path / "analysis.txt"
    sentinel.write_text("preserve unrelated data")
    plan.write_text(json.dumps({"frozen_application": True, "records": [], "pid": os.getpid(),
                                "version": "1.5.1.1", "log": str(tmp_path / "refusal.log")}))
    refused = subprocess.run([str(target), "run-plan", str(plan)], cwd=tmp_path, env=environment,
                             capture_output=True, text=True, timeout=60)
    assert refused.returncode == 2, refused.stdout + refused.stderr
    assert "nothing was changed" in (tmp_path / "refusal.log").read_text()
    assert sentinel.read_text() == "preserve unrelated data"
    assert source.is_file(), "the supplied native artifact must remain untouched"
