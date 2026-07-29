"""Static and dry-run contracts for the small cross-platform installers."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ONLINE = ROOT / "packaging" / "online"
UNIX = ONLINE / "install_spacr_unix.sh"
WINDOWS = ONLINE / "install_spacr_windows.ps1"
NSIS = ONLINE / "spacr_online_installer.nsi"
WORKFLOW = ROOT / ".github" / "workflows" / "online-installers.yml"


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


def test_bootstraps_use_tls_and_hardware_aware_pytorch():
    unix = _text(UNIX)
    windows = _text(WINDOWS)
    assert "https://astral.sh/uv/" in unix
    assert "--proto '=https'" in unix
    assert "--tlsv1.2" in unix
    assert "Tls12" in windows
    assert "--torch-backend auto" in unix
    assert "--torch-backend auto" in windows


def test_install_is_validated_before_the_previous_environment_is_replaced():
    for path in (UNIX, WINDOWS):
        source = _text(path)
        assert "--relocatable" in source
        check = source.index("pip check")
        import_check = source.index("import spacr, PySide6, torch")
        activate = source.index("venv-previous")
        assert check < activate
        assert import_check < activate


def test_linux_installer_creates_desktop_launcher_and_uninstaller():
    source = _text(UNIX)
    assert "[Desktop Entry]" in source
    assert "Terminal=false" in source
    assert "uninstall-spacr.sh" in source
    assert "apt-get" in source
    assert "dnf" in source
    assert "zypper" in source
    assert "pacman" in source


def test_windows_installer_is_per_user_and_registers_uninstall():
    bootstrap = _text(WINDOWS)
    nsis = _text(NSIS)
    assert "LOCALAPPDATA" in bootstrap
    assert "RequestExecutionLevel user" in nsis
    assert "CurrentVersion\\Uninstall\\spaCR" in nsis
    assert "CreateShortcut" in nsis
    assert "pythonw.exe" in nsis
    assert "Refusing unsafe install root" in bootstrap


def test_macos_builder_creates_application_and_pkg_with_uninstall_helper():
    source = _text(ONLINE / "build_macos_online.sh")
    assert "/Applications/SpaCR.app" in source
    assert "pkgbuild" in source
    assert "codesign" in source
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
    assert "DRY RUN" in result.stdout
    assert not (tmp_path / "spacr").exists()


def test_release_workflow_builds_all_platforms_with_node24_actions():
    workflow = _text(WORKFLOW)
    assert "branches: [spacr-codex, spacr-nightly]" in workflow
    for job in ("linux:", "windows:", "macos:"):
        assert job in workflow
    assert "actions/checkout@v6" in workflow
    assert "actions/setup-python@v6" in workflow
    assert "actions/upload-artifact@v6" in workflow
    assert "actions/download-artifact@v6" in workflow
    assert "https://pypi.org/pypi/spacr/$version/json" in workflow
    assert "gh release upload" in workflow


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
