"""Static and dry-run contracts for the small cross-platform installers."""

from __future__ import annotations

import re
import subprocess
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ONLINE = ROOT / "packaging" / "online"
UNIX = ONLINE / "install_spacr_unix.sh"
WINDOWS = ONLINE / "install_spacr_windows.ps1"
NSIS = ONLINE / "spacr_online_installer.nsi"
APP_ICON = ROOT / "spacr" / "resources" / "icons" / "app_icon.png"
WINDOWS_ICON = ROOT / "spacr" / "resources" / "icons" / "app_icon.ico"
WORKFLOW = ROOT / ".github" / "workflows" / "online-installers.yml"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def _release_module():
    spec = importlib.util.spec_from_file_location(
        "spacr_release_helper", ROOT / "packaging" / "release.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_online_installers_are_small_bootstraps_not_frozen_applications():
    files = list(ONLINE.iterdir())
    assert files
    assert sum(path.stat().st_size for path in files if path.is_file()) < 250_000
    combined = "\n".join(_text(path) for path in files if path.is_file())
    assert "PyInstaller" not in combined
    assert "conda install" not in combined
    assert "miniconda" not in combined.lower()


def test_bootstraps_pin_uv_and_private_python():
    for path in (UNIX, WINDOWS):
        source = _text(path)
        assert re.search(r"0\.11\.32", source)
        assert re.search(r"3\.12", source)
        assert "UV_PYTHON_INSTALL_DIR" in source
        assert "UV_CACHE_DIR" in source
        assert "spacr[" in source


def test_bootstraps_use_tls_and_portable_pytorch_by_default():
    unix = _text(UNIX)
    windows = _text(WINDOWS)
    assert "https://astral.sh/uv/" in unix
    assert "--proto '=https'" in unix
    assert "--tlsv1.2" in unix
    assert "Tls12" in windows
    assert 'TORCH_BACKEND="${SPACR_TORCH_BACKEND:-cpu}"' in unix
    assert '--torch-backend "$TORCH_BACKEND"' in unix
    assert '$TorchBackend = "cpu"' in windows
    assert "--torch-backend $TorchBackend" in windows
    assert 'DEFAULT_EXTRAS="qt"' in unix
    assert '$DefaultExtras = "qt"' in windows
    assert "qt,zernike,btrack,czi" not in unix
    assert "qt,zernike,btrack,czi" not in windows


def test_bootstraps_do_not_register_their_private_python_globally():
    unix = _text(UNIX)
    windows = _text(WINDOWS)
    assert 'python install "$PYTHON_VERSION" --managed-python --no-bin' in unix
    assert (
        "python install $PythonVersion --managed-python --no-bin --no-registry"
        in windows
    )


def test_bootstraps_guard_python_312_numba_and_llvmlite_resolution():
    """SHAP's unbounded requirements must not select 2021 source releases."""
    for path in (UNIX, WINDOWS):
        source = _text(path)
        assert "numba>=0.60,<1.0" in source
        assert "llvmlite>=0.43,<1.0" in source
        assert "ResolverGuards" in source or "RESOLVER_GUARDS" in source


def test_install_is_validated_before_the_previous_environment_is_replaced():
    for path in (UNIX, WINDOWS):
        source = _text(path)
        assert "--relocatable" in source
        check = source.index("pip check")
        import_check = source.index("import spacr, PySide6, torch")
        activate = source.index("venv-previous")
        assert "-I" in source[:import_check]
        assert check < activate
        assert import_check < activate


def test_windows_validation_passes_isolated_flag_as_an_argument():
    """PowerShell must not bind Python's ``-I`` to Invoke-Checked itself."""
    windows = _text(WINDOWS)
    assert "Invoke-Checked -Command $StagePython -Arguments @(" in windows
    assert '"-I"' in windows
    assert "Invoke-Checked $StagePython -I" not in windows


def test_installers_preserve_a_diagnostic_log():
    unix = _text(UNIX)
    windows = _text(WINDOWS)
    assert 'INSTALL_LOG="$INSTALL_ROOT/install.log"' in unix
    assert "tee -a" in unix
    assert '$LogPath = Join-Path $InstallRoot "install.log"' in windows
    assert "Start-Transcript" in windows


def test_linux_installer_creates_desktop_launcher_and_uninstaller():
    source = _text(UNIX)
    assert "[Desktop Entry]" in source
    assert "Terminal=false" in source
    assert "uninstall-spacr.sh" in source
    assert "apt-get" in source
    assert "dnf" in source
    assert "zypper" in source
    assert "pacman" in source
    assert "app_icon.png" in source


def test_application_icon_has_exact_background_and_transparent_corners():
    from PIL import Image

    with Image.open(APP_ICON).convert("RGBA") as icon:
        assert icon.size == (1024, 1024)
        assert icon.getpixel((0, 0))[3] == 0
        assert icon.getpixel((512, 32)) == (0, 55, 55, 255)
    with Image.open(WINDOWS_ICON) as icon:
        assert icon.format == "ICO"
        assert icon.size in {(16, 16), (256, 256)}


def test_windows_installer_is_per_user_and_registers_uninstall():
    bootstrap = _text(WINDOWS)
    nsis = _text(NSIS)
    assert "LOCALAPPDATA" in bootstrap
    assert "RequestExecutionLevel user" in nsis
    assert "CurrentVersion\\Uninstall\\spaCR" in nsis
    assert "CreateShortcut" in nsis
    assert "pythonw.exe" in nsis
    assert "Refusing unsafe install root" in bootstrap
    assert 'Section /o "NVIDIA GPU acceleration (large download)"' in nsis
    assert '-TorchBackend "$1"' in nsis
    assert "app_icon.ico" in nsis
    assert 'File /oname=spacr.ico "${SPACR_ICON}"' in nsis


def test_macos_builder_creates_application_and_pkg_with_uninstall_helper():
    source = _text(ONLINE / "build_macos_online.sh")
    assert "/Applications/SpaCR.app" in source
    assert "pkgbuild" in source
    assert "codesign" in source
    assert "iconutil -c icns" in source
    assert "CFBundleIconFile" in source
    assert "uninstall-spacr.sh" in source
    assert "PRODUCTSIGN_IDENTITY" in source


def test_unix_bootstrap_parses_and_dry_run_never_downloads(tmp_path):
    subprocess.run(["bash", "-n", str(UNIX)], check=True)
    result = subprocess.run(
        [
            "bash",
            str(UNIX),
            "--platform",
            "linux",
            "--dry-run",
            "--skip-system-deps",
            "--no-launch",
            "--install-root",
            str(tmp_path / "spacr"),
            "--package-spec",
            "spacr[qt]==9.9.9",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "spacr[qt]==9.9.9" in result.stdout
    assert "PyTorch backend: cpu" in result.stdout
    assert "DRY RUN" in result.stdout
    assert not (tmp_path / "spacr").exists()


def test_release_workflow_builds_all_platforms_with_node24_actions():
    workflow = _text(WORKFLOW)
    assert "workflow_call:" in workflow
    assert "\n  push:" not in workflow
    for job in ("linux:", "windows:", "macos:"):
        assert job in workflow
    assert "actions/checkout@v7" in workflow
    assert "actions/setup-python@v7" in workflow
    assert "actions/upload-artifact@v7" in workflow
    assert "actions/download-artifact@v8" in workflow
    assert "python packaging/release.py collect" in workflow
    assert "spacr/application" in workflow
    for platform in ("Linux", "Windows", "macOS"):
        assert f"Install and import the released {platform} application" in (
            workflow
        )
    assert workflow.count("timeout-minutes: 30") == 3
    assert workflow.count("assert torch.version.cuda is None") == 3
    assert workflow.count("install.log") >= 3
    assert "sudo installer -verboseR -pkg" in workflow
    assert "Start-Process" in workflow
    macos_job = workflow[workflow.index("  macos:"):workflow.index("  collect:")]
    assert 'find dist/online -name \'*macOS*Online.pkg\'' in macos_job
    assert "packaging/release.py version" not in macos_job


def test_one_click_release_orders_version_pypi_installers_and_github():
    workflow = _text(RELEASE_WORKFLOW)
    assert "workflow_dispatch:" in workflow
    assert "branches: [main]" in workflow
    assert '"setup.py"' in workflow
    assert "python packaging/release.py bump" in workflow
    assert "--allow-current" in workflow
    assert "github.actor != 'github-actions[bot]'" in workflow
    assert "VERSION remains $version" in workflow
    assert "release_required: ${{ steps.version.outputs.release_required }}" in workflow
    assert "if: needs.bump.outputs.release_required == 'true'" in workflow
    assert "needs.verify-pypi.result == 'success'" in workflow
    assert "python packaging/release.py version" in workflow
    assert "if: github.event_name == 'workflow_dispatch'" in workflow
    assert "uses: ./.github/workflows/online-installers.yml" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "environment:" in workflow and "name: pypi" in workflow
    assert "id-token: write" in workflow
    assert '"setuptools>=64"' in workflow
    assert "python -m pytest --noconftest" in workflow
    assert "gh release create" in workflow
    assert "gh release upload" in workflow
    assert "release-assets/*" in workflow
    assert "SHA256SUMS.txt" in workflow
    assert 'git config user.name "github-actions[bot]"' in workflow
    assert (
        'git config user.email '
        '"41898282+github-actions[bot]@users.noreply.github.com"'
        in workflow
    )
    assert "needs.bump.outputs.pypi_exists != 'true'" in workflow
    assert "verify-pypi:" in workflow
    assert "https://pypi.org/pypi/spacr/$VERSION/json" in workflow
    assert "needs: [bump, package, publish-pypi]" in workflow
    assert "needs: [bump, verify-pypi]" in workflow
    assert (
        "needs: [bump, package, publish-pypi, verify-pypi, installers]"
        in workflow
    )
    # The installer stage cannot start until PyPI has returned HTTP 200.
    assert workflow.index("  publish-pypi:") < workflow.index("  verify-pypi:")
    assert workflow.index("  verify-pypi:") < workflow.index("  installers:")


def test_release_helper_bumps_only_to_a_newer_valid_version(tmp_path):
    helper = _release_module()

    setup = tmp_path / "setup.py"
    setup.write_text('name = "spacr"\nVERSION = "1.2.3"\n', encoding="utf-8")
    assert helper.bump_version(setup, "1.2.4") == "1.2.4"
    assert helper.read_version(setup) == "1.2.4"
    assert setup.read_text(encoding="utf-8").count("VERSION") == 1

    with pytest.raises(ValueError, match="greater than"):
        helper.bump_version(setup, "1.2.4")
    assert helper.bump_version(
        setup, "1.2.4", allow_current=True) == "1.2.4"
    assert helper.read_version(setup) == "1.2.4"
    with pytest.raises(ValueError, match="valid Python package version"):
        helper.bump_version(setup, "not a version")


def test_release_helper_collects_current_installers_and_rewrites_links(tmp_path):
    helper = _release_module()

    version = "2.3.4"
    setup = tmp_path / "setup.py"
    setup.write_text(f'VERSION = "{version}"\n', encoding="utf-8")
    readme = tmp_path / "README.rst"
    readme.write_text(
        "Before\n\n.. spacr-installer-links-begin\nold\n"
        ".. spacr-installer-links-end\n\nAfter\n",
        encoding="utf-8",
    )
    source = tmp_path / "artifacts"
    source.mkdir()
    names = [
        f"SpaCR-{version}-Windows-Online-Setup.exe",
        f"SpaCR-{version}-macOS-Universal-Online.pkg",
        f"SpaCR-{version}-Linux-x86_64-Online.run",
    ]
    for index, name in enumerate(names):
        nested = source / f"job-{index}"
        nested.mkdir()
        (nested / name).write_bytes(f"installer-{index}".encode())
    destination = tmp_path / "application"
    destination.mkdir()
    old = destination / "SpaCR-1.0.0-Linux-x86_64-Online.run"
    old.write_bytes(b"old")

    copied = helper.collect_installers(
        source, destination, readme, setup, branch="nightly")

    assert {path.name for path in copied} == set(names)
    assert not old.exists()
    links = readme.read_text(encoding="utf-8")
    assert "old" not in links
    for name in names:
        assert name in links
        assert (
            f"https://github.com/EinarOlafsson/spacr/releases/download/"
            f"v{version}/{name}"
        ) in links
        assert (destination / name).is_file()
    assert "/raw/nightly/" not in links
    manifest = (destination / "README.rst").read_text(encoding="utf-8")
    assert f"Current version: ``{version}``" in manifest
    assert manifest.count("SHA-256") == 4  # heading + one line per installer


def test_builders_read_version_without_importing_setup_py():
    unix_builders = (
        ONLINE / "build_linux_online.sh",
        ONLINE / "build_macos_online.sh",
    )
    for path in unix_builders:
        source = _text(path)
        assert 'ast.parse' in source
        assert "setup.py --version" not in source
    windows = _text(ONLINE / "build_windows_online.ps1")
    assert "Select-String" in windows
    assert "setup.py --version" not in windows
