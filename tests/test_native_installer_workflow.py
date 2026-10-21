"""The legacy installer jobs must validate installed artifacts on fresh runners."""
from pathlib import Path
import hashlib
import importlib.util
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_native_smoke_jobs_cannot_borrow_the_build_checkout():
    yaml = pytest.importorskip('yaml')
    workflow = yaml.safe_load((ROOT / '.github/workflows/frozen-installers.yml').read_text(encoding="utf-8"))
    events = workflow.get('on', workflow.get(True))
    assert 'push' in events and events['push']['branches'] == ['nightly']
    assert 'workflow_dispatch' in events
    assert 'spacr/run_journal.py' in events['push']['paths']
    assert workflow['permissions'] == {'contents': 'read'}
    for key in ('smoke-frozen', 'smoke-debian'):
        steps = workflow['jobs'][key]['steps']
        actions = [row.get('uses', '') for row in steps]
        assert not any('checkout' in action or 'setup-python' in action for action in actions)
        assert any('download-artifact' in action for action in actions)
        commands = '\n'.join(row.get('run', '') for row in steps)
        assert 'pip install' not in commands
        assert 'empty-cwd' in commands
        assert 'smoke.json' in commands
    assert 'continue-on-error' not in workflow['jobs']['smoke-debian']
    assert workflow['jobs']['build-debian']['container'] == 'ubuntu:22.04'
    assert set(workflow['jobs']['smoke-debian']['strategy']['matrix']['container']) == {
        'ubuntu:22.04', 'debian:12'}


def test_windows_nsis_template_keeps_runtime_dollars_and_requires_actual_executable():
    source = (ROOT / 'packaging/build_windows.ps1').read_text(encoding="utf-8")
    template = source.split("$Template = @'\n", 1)[1].split("\n'@", 1)[0]
    assert 'InstallDir "$PROGRAMFILES64\\spaCR"' in template
    assert 'SetOutPath "$INSTDIR"' in template
    assert 'CreateShortcut "$SMPROGRAMS\\spaCR.lnk" "$INSTDIR\\spacr.exe"' in template
    assert '-RequireInstaller' in (ROOT / '.github/workflows/frozen-installers.yml').read_text(encoding="utf-8")
    assert 'if ($LASTEXITCODE -ne 0)' in source
    assert 'Join-Path $Source "spacr.exe"' in source
    assert 'RMDir /r "$INSTDIR"' not in template
    assert 'RMDir /r "$INSTDIR\\_internal"' not in template
    extract = template.index('SetOutPath "$PLUGINSDIR\\spaCR-payload"')
    copy = template.index('File /r "@SOURCE@\\*"')
    helper = template.index('CopyFiles /SILENT "$PLUGINSDIR\\spaCR-payload\\_internal\\spacr-update-helper.exe"')
    transaction = template.index('install-frozen-windows --source')
    result = template.index('Pop $0', transaction)
    shortcut = template.index('CreateShortcut "$SMPROGRAMS\\spaCR.lnk"')
    assert extract < copy < helper < transaction < result < shortcut
    assert 'IfErrors frozen_install_failed' in template
    assert 'PYINSTALLER_RESET_ENVIRONMENT' in template
    assert '--destination "$INSTDIR" --version "@VERSION@"' in template
    assert '${If} $0 != 0\n    Goto frozen_install_failed' in template
    assert 'frozen_install_failed:\n  SetErrorLevel 1\n  Abort' in template
    assert '@DELETE_FILES@' in template and '@DELETE_DIRECTORIES@' in template


def test_debian_xcb_shape_dependency_exists_on_builder_and_fresh_host():
    """Both clean targets failed to load Qt because libxcb-shape was absent."""
    source = (ROOT / 'packaging/build_debian.sh').read_text(encoding="utf-8")
    builder = re.search(r'needed=\((.*?)\)\nmissing=', source, re.S).group(1).split()
    depends = re.search(r'^Depends: (.+)$', source, re.M).group(1)
    runtime = {value.strip().split()[0] for value in depends.split(',')}
    assert 'libxcb-shape0' in builder
    assert 'libxcb-shape0' in runtime


def _release_helper():
    spec = importlib.util.spec_from_file_location("native_release_helper", ROOT / "packaging/release.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def _native_builds(tmp_path):
    source = tmp_path / "builds"
    version, commit = "1.5.1.5", "a" * 40
    names = {"windows": f"spaCR-{version}-setup.exe", "macos": f"spaCR-{version}.dmg",
             "debian": f"spacr_{version}_amd64.deb"}
    for platform, name in names.items():
        root = source / f"legacy-build-{platform}"
        (root / "acceptance").mkdir(parents=True)
        (root / "dist").mkdir()
        payload = (platform + " disposable test bytes").encode()
        (root / "dist" / name).write_bytes(payload)
        (root / "acceptance/source-commit.txt").write_text(commit + "\n", encoding="utf-8")
        (root / "acceptance/version.txt").write_text(version + "\n", encoding="utf-8")
        recorded = "dist/" + name if platform == "debian" else name
        (root / "acceptance/SHA256SUMS").write_text(hashlib.sha256(payload).hexdigest() + "  " + recorded + "\n", encoding="utf-8")
        (root / "acceptance/build.log").write_text("CI-only build output", encoding="utf-8")
    updater = f"spaCR-{version}-update.exe"
    root = source / "legacy-build-windows"
    (root / "dist" / updater).write_bytes(b"mandatory phase updater")
    with (root / "acceptance/SHA256SUMS").open("a") as stream:
        stream.write(hashlib.sha256(b"mandatory phase updater").hexdigest() + "  " + updater + "\n")
    return source, version, commit, names


def test_frozen_release_collection_copies_only_verified_updater_payloads(tmp_path):
    source, version, commit, names = _native_builds(tmp_path)
    windows = source / "legacy-build-windows/dist"
    (windows / f"spaCR-{version}-windows.zip").write_bytes(b"portable archive, not the installer")
    (windows / "spacr-update-helper.exe").write_bytes(b"standalone helper, not the installer")
    destination = tmp_path / "public"
    destination.mkdir()
    (destination / "online.run").write_bytes(b"existing online asset")
    result = subprocess.run([sys.executable, str(ROOT / "packaging/release.py"), "collect-frozen",
                             "--source", str(source), "--destination", str(destination),
                             "--version", version, "--source-commit", commit],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert {path.name for path in destination.iterdir()} == {*names.values(), f"spaCR-{version}-update.exe", "online.run"}
    for platform, name in names.items():
        assert (destination / name).read_bytes() == (source / f"legacy-build-{platform}/dist" / name).read_bytes()
    assert (destination / "online.run").read_bytes() == b"existing online asset"


@pytest.mark.parametrize("platform", ["windows", "macos", "debian"])
@pytest.mark.parametrize("fault", ["source", "version", "checksum", "missing", "extra", "duplicate-checksum"])
def test_frozen_release_collection_refuses_wrong_or_incomplete_builds_before_copy(tmp_path, platform, fault):
    source, version, commit, names = _native_builds(tmp_path)
    root = source / f"legacy-build-{platform}"
    payload = root / "dist" / names[platform]
    manifest = root / "acceptance/SHA256SUMS"
    if fault == "source":
        (root / "acceptance/source-commit.txt").write_text("b" * 40, encoding="utf-8")
    elif fault == "version":
        (root / "acceptance/version.txt").write_text("1.5.1.4", encoding="utf-8")
    elif fault == "checksum":
        payload.write_bytes(b"changed payload")
    elif fault == "missing":
        payload.unlink()
    elif fault == "extra":
        (root / "dist" / names[platform].replace(version, "1.5.1.4")).write_bytes(b"wrong version")
    else:
        manifest.write_text(manifest.read_text(encoding="utf-8") * 2, encoding="utf-8")
    destination = tmp_path / "public"
    with pytest.raises((ValueError, FileNotFoundError)):
        _release_helper().collect_frozen_installers(source, destination, version, commit)
    assert not destination.exists()


def test_frozen_release_collection_never_clobbers_existing_asset(tmp_path):
    source, version, commit, names = _native_builds(tmp_path)
    destination = tmp_path / "public"
    destination.mkdir()
    previous = destination / names["debian"]
    previous.write_bytes(b"previous published bytes")
    with pytest.raises(ValueError, match="already exists"):
        _release_helper().collect_frozen_installers(source, destination, version, commit)
    assert list(destination.iterdir()) == [previous]
    assert previous.read_bytes() == b"previous published bytes"


def test_release_waits_for_exact_source_native_acceptance_before_tagging_or_upload():
    yaml = pytest.importorskip("yaml")
    release = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    jobs = release["jobs"]
    native = jobs["frozen-installers"]
    assert native["uses"] == "./.github/workflows/frozen-installers.yml"
    assert native["needs"] == ["bump", "installers"]
    assert native["with"] == {"source_ref": "${{ needs.installers.outputs.release_commit }}",
                              "release_version": "${{ needs.bump.outputs.version }}"}
    publish = jobs["github-release"]
    assert "frozen-installers" in publish["needs"]
    assert "needs.frozen-installers.result == 'success'" in publish["if"]
    steps = publish["steps"]
    download = next(step for step in steps if step.get("with", {}).get("pattern") == "legacy-build-*")
    assert download["with"] == {"pattern": "legacy-build-*", "path": "frozen-assets"}
    verify = next(step for step in steps if step.get("name") == "Verify and select exact native release payloads")
    assert verify["env"] == {"RELEASE_SHA": "${{ needs.installers.outputs.release_commit }}",
                             "VERSION": "${{ needs.bump.outputs.version }}"}
    assert 'collect-frozen --source frozen-assets --destination release-assets' in verify["run"]
    assert '--source-commit "$RELEASE_SHA" --version "$VERSION"' in verify["run"]
    names = [step.get("name") for step in steps]
    assert names.index(verify["name"]) < names.index("Create checksums") < names.index("Tag exact release commit")
    assert names.index("Create checksums") < names.index("Publish release notes and downloadable installers")


def test_reusable_native_build_and_verifiers_use_selected_commit_not_caller_sha():
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load((ROOT / ".github/workflows/frozen-installers.yml").read_text(encoding="utf-8"))
    events = workflow.get("on", workflow.get(True))
    assert events["workflow_call"]["inputs"] == {
        "source_ref": {"required": True, "type": "string"},
        "release_version": {"required": True, "type": "string"}}
    assert workflow["env"]["SPACR_ACCEPTANCE_REF"] == "${{ inputs.source_ref || github.sha }}"
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["ref"] == "${{ inputs.source_ref || github.sha }}"
    for name in ("build-frozen", "build-debian"):
        commands = "\n".join(step.get("run", "") for step in workflow["jobs"][name]["steps"])
        assert 'test "$(git rev-parse HEAD)" = "$SPACR_ACCEPTANCE_REF"' in commands
        assert 'test "$version" = "$SPACR_RELEASE_VERSION"' in commands
        assert '"$version" > acceptance/version.txt' in commands
    source = (ROOT / ".github/workflows/frozen-installers.yml").read_text(encoding="utf-8")
    assert "$env:GITHUB_SHA" not in source and '"$GITHUB_SHA"' not in source
    assert "-SourceCommit $env:SPACR_ACCEPTANCE_REF" in source
    assert '-e ACCEPTANCE_COMMIT="$SPACR_ACCEPTANCE_REF"' in source
    assert '"$SPACR_ACCEPTANCE_REF" macos-hosted' in source


def test_windows_native_verifier_keeps_default_and_explicit_commit_refusal(tmp_path):
    source = (ROOT / "packaging/verify_claude_native_install.ps1").read_text(encoding="utf-8")
    assert "[string]$SourceCommit = $env:GITHUB_SHA" in source
    assert "$Proof.source_commit -ne $SourceCommit" in source
    powershell = shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is required to execute the native receipt guard")
    prefix = source.split("$Hint =", 1)[0]
    guard = tmp_path / "guard.ps1"
    guard.write_text(prefix, encoding="utf-8")
    receipt = tmp_path / "smoke.json"
    receipt.write_text('{"status":"passed","frozen":true,"qt_platform":"windows","source_commit":"selected"}', encoding="utf-8")
    common = [powershell, "-NoProfile", "-NonInteractive", "-File", str(guard),
              "-NativeReceipt", str(receipt), "-Shell", "cmd", "-Output", str(tmp_path / "out")]
    import os
    env = dict(os.environ, GITHUB_SHA="selected")
    default = subprocess.run(common, env=env, capture_output=True, text=True, timeout=60)
    assert default.returncode == 0, default.stderr
    explicit = subprocess.run(common + ["-SourceCommit", "selected"],
                              env=dict(env, GITHUB_SHA="caller"), capture_output=True, text=True, timeout=60)
    assert explicit.returncode == 0, explicit.stderr
    wrong = subprocess.run(common + ["-SourceCommit", "caller"], env=env,
                           capture_output=True, text=True, timeout=60)
    assert wrong.returncode != 0 and "another commit" in wrong.stderr


@pytest.mark.parametrize("fault", ["missing", "corrupt", "unchecksummed"])
def test_release_cannot_publish_without_verified_mandatory_phase_windows_asset(tmp_path, fault):
    source, version, commit, names = _native_builds(tmp_path)
    root = source / "legacy-build-windows"
    updater = root / "dist" / f"spaCR-{version}-update.exe"
    if fault == "missing":
        updater.unlink()
    elif fault == "corrupt":
        updater.write_bytes(b"unverified updater")
    else:
        checksum = root / "acceptance/SHA256SUMS"
        checksum.write_text("".join(line for line in checksum.read_text(encoding="utf-8").splitlines(keepends=True)
                                   if updater.name not in line), encoding="utf-8")
    destination = tmp_path / "public"
    with pytest.raises((ValueError, FileNotFoundError)):
        _release_helper().collect_frozen_installers(source, destination, version, commit)
    assert not destination.exists()


def test_windows_update_asset_requires_phase_flags_and_shares_the_built_payload():
    source = (ROOT / "packaging/build_windows.ps1").read_text(encoding="utf-8")
    assert source.count('@("-m", "PyInstaller",') == 1
    assert "!ifdef SPACR_UPDATE_ONLY" in source
    assert "/DSPACR_UPDATE_ONLY" in source
    assert 'spaCR-$Version-update.exe' in source
    assert '${GetOptions} $R0 "/SPACRUPDATE="' in source
    assert '${GetOptions} $R0 "/SPACRPID="' in source
    assert '--phase-channel "$R1" --desktop-pid "$R2"' in source
    workflow = (ROOT / ".github/workflows/frozen-installers.yml").read_text(encoding="utf-8")
    assert 'dist/*.exe' in workflow
    release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "collect-frozen" in release and "release-assets/*" in release
