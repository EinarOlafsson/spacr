"""Static and dry-run contracts for the small cross-platform installers."""

from __future__ import annotations

import os
import re
import subprocess
import importlib.util
import json
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


def _installer_i18n_module():
    spec = importlib.util.spec_from_file_location(
        "spacr_installer_i18n", ROOT / "packaging" / "i18n" / "render.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _standalone_unix_installer(tmp_path, version="9.9.9"):
    renderer = _installer_i18n_module()
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "installer_messages.sh").write_text(
        renderer.render_shell(renderer.catalogs()), encoding="utf-8"
    )
    renderer.OUTPUT_DIR = generated
    installer = tmp_path / "spaCR-Linux-x86_64-Online.run"
    renderer.embed_unix(UNIX, installer, version)
    return installer


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


def test_bootstraps_allow_an_explicit_ci_package_source():
    unix = _text(UNIX)
    windows = _text(WINDOWS)
    workflow = _text(WORKFLOW)

    assert 'PACKAGE_SPEC="${SPACR_PACKAGE_SPEC:-}"' in unix
    assert '$PackageSpec = $env:SPACR_PACKAGE_SPEC' in windows
    assert '--package-spec "$GITHUB_WORKSPACE[qt]"' in workflow
    assert '$env:SPACR_PACKAGE_SPEC = "$((Resolve-Path .).Path)[qt]"' in workflow
    assert 'SPACR_PACKAGE_SPEC="$GITHUB_WORKSPACE[qt]"' in workflow


def test_windows_generated_catalog_is_bom_encoded_before_use():
    builder = _text(ONLINE / "build_windows_online.ps1")
    workflow = _text(WORKFLOW)

    for source in (builder, workflow):
        assert "System.Text.UTF8Encoding($true)" in source
        assert "installer_messages.ps1" in source
    assert workflow.index("System.Text.UTF8Encoding($true)") < workflow.index(
        "Language.Parser]::ParseFile"
    )
    assert builder.index("System.Text.UTF8Encoding($true)") < builder.index(
        "& $MakeNsis"
    )


def test_bootstraps_use_tls_and_detect_acceleration_by_default():
    unix = _text(UNIX)
    windows = _text(WINDOWS)
    assert "https://astral.sh/uv/" in unix
    assert "--proto '=https'" in unix
    assert "--tlsv1.2" in unix
    assert "Tls12" in windows
    assert 'TORCH_BACKEND="${SPACR_TORCH_BACKEND:-}"' in unix
    assert "nvidia-smi -L" in unix
    assert 'DETECTED_ACCELERATOR="apple-silicon"' in unix
    assert '--torch-backend "$TORCH_BACKEND"' in unix
    assert 'Get-Command "nvidia-smi.exe"' in windows
    assert '$TorchBackend = "auto"' in windows
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


def test_unix_bootstrap_uses_last_intel_macos_llvmlite_wheel_line():
    source = _text(UNIX)
    assert '"$PLATFORM" == "macos"' in source
    assert '"$(uname -m)" == "x86_64"' in source
    assert "numba>=0.60,<0.63" in source
    assert "llvmlite>=0.43,<0.46" in source
    assert "numpy>=1.26,<2.0" in source
    assert "opencv-python-headless<4.12" in source


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


def test_bootstraps_persist_the_requested_and_actual_backend_for_doctor():
    unix = _text(UNIX)
    windows = _text(WINDOWS)
    for source in (unix, windows):
        assert "spacr.install_profile" in source
        assert "install-profile.json" in source
        assert "requested" in source
        assert "detected" in source
        assert "13x faster Cellpose segmentation" in source
        assert "20x faster ResNet classification" in source
    assert "torch.cuda.is_available()" in unix
    assert "torch.backends.mps.is_available()" in unix
    assert "torch.cuda.is_available()" in windows
    assert "MUI_DESCRIPTION_TEXT ${SecGpu}" in _text(NSIS)


def test_installer_consent_is_optional_off_by_default_and_persisted():
    unix = _text(UNIX)
    windows = _text(WINDOWS)
    nsis = _text(NSIS)
    for source in (unix, windows):
        assert "ShareDiagnostics" in source or "SHARE_DIAGNOSTICS=0" in source
        assert "ReportIssues" in source or "REPORT_ISSUES=0" in source
        assert "SignInNow" in source or "SIGN_IN_NOW=0" in source
        assert "spacr.install_profile" in source
    assert "Page custom ConsentPage ConsentPageLeave" in nsis
    assert nsis.count("off by default") >= 3
    assert "PUBLIC spaCR GitHub repository" in nsis
    assert "cannot be reliably unpublished" in nsis
    assert "-ConsentCollected $ConsentCollectedValue" in nsis


def test_unix_validation_exercises_torch_numpy_abi():
    source = _text(UNIX)
    assert "torch.from_numpy" in source
    assert "dtype=np.float32" in source


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
    assert 'Get-SpacrInstallerMessage "unsafe_root"' in bootstrap
    assert 'Section /o "$(SPACR_NSIS_GPU)"' in nsis
    assert '-TorchBackend "$1"' in nsis
    assert "nvidia-smi -L" in nsis
    assert "SectionSetFlags ${SecGpu} ${SF_SELECTED}" in nsis
    assert "app_icon.ico" in nsis
    assert 'File /oname=spacr.ico "${SPACR_ICON}"' in nsis


def test_installer_locales_cover_every_supported_ui_language():
    from spacr.qt.i18n import LANGUAGES

    renderer = _installer_i18n_module()
    values = renderer.catalogs()
    locale_dir = ROOT / "packaging" / "i18n"
    expected_languages = {language.code for language in LANGUAGES}
    expected_keys = set(json.loads(_text(locale_dir / "en.json")))
    assert expected_languages == {
        path.stem for path in locale_dir.glob("*.json")
    }
    for language in expected_languages:
        table = json.loads(_text(locale_dir / f"{language}.json"))
        assert set(table) == expected_keys
        assert all(str(value).strip() for value in table.values())
    shell_messages = renderer.render_shell(values)
    assert "spacr_install_language" in shell_messages
    assert "defaults read -g AppleLocale" in shell_messages
    assert "CurrentUICulture" in renderer.render_powershell(values)
    nsis_messages = renderer.render_nsis(values)
    for name in (
        "English", "Swedish", "German", "Spanish", "SimpChinese",
        "Portuguese", "Hindi", "Korean", "Icelandic", "French",
    ):
        assert f'MUI_LANGUAGE "{name}"' in nsis_messages


def test_installer_renderer_outputs_are_deterministic():
    renderer = _installer_i18n_module()
    values = renderer.catalogs()
    first = (
        renderer.render_shell(values),
        renderer.render_powershell(values),
        renderer.render_nsis(values),
    )
    second = (
        renderer.render_shell(values),
        renderer.render_powershell(values),
        renderer.render_nsis(values),
    )

    assert tuple(values) == renderer.LANGUAGES
    assert first == second


@pytest.mark.parametrize(
    ("delimiter", "quoted"),
    (
        ("'", "'left''right'"),
        ("\u2018", "'left‘‘right'"),
        ("\u2019", "'left’’right'"),
        ("\u201b", "'left‛‛right'"),
    ),
)
def test_powershell_quote_doubles_every_single_quote_delimiter(
    delimiter, quoted
):
    renderer = _installer_i18n_module()
    original = f"left{delimiter}right"

    assert renderer._ps_quote(original) == quoted
    assert quoted[1:-1].replace(delimiter * 2, delimiter) == original


def test_embedded_unix_catalog_preserves_markers_and_is_idempotent(
    tmp_path, monkeypatch
):
    renderer = _installer_i18n_module()
    source = tmp_path / "installer.sh"
    once = tmp_path / "installer-once.sh"
    twice = tmp_path / "installer-twice.sh"
    begin = "# @SPACR_INSTALLER_MESSAGES_BEGIN@"
    end = "# @SPACR_INSTALLER_MESSAGES_END@"
    source.write_text(
        f"#!/bin/sh\n{begin}\nsource generated/messages.sh\n{end}\n"
        'VERSION="@SPACR_VERSION@"\n',
        encoding="utf-8",
    )
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "installer_messages.sh").write_text(
        renderer.render_shell(renderer.catalogs()), encoding="utf-8"
    )
    monkeypatch.setattr(renderer, "OUTPUT_DIR", generated)

    renderer.embed_unix(source, once, "7.8.9")
    renderer.embed_unix(once, twice, "7.8.9")

    rendered = _text(once)
    assert rendered == _text(twice)
    assert rendered.count(begin) == rendered.count(end) == 1
    assert "source generated/messages.sh" not in rendered
    assert "spacr_install_language()" in rendered
    assert 'VERSION="7.8.9"' in rendered


@pytest.mark.parametrize(
    "language", ("en", "sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")
)
def test_unix_dry_run_detects_every_installer_language(
    language, tmp_path
):
    renderer = _installer_i18n_module()
    catalog = renderer.catalogs()[language]
    installer = _standalone_unix_installer(tmp_path)
    env = os.environ.copy()
    env.update(
        SPACR_INSTALL_LANGUAGE=language,
        LC_ALL="C",
        LANG="C",
    )
    result = subprocess.run(
        [
            "bash",
            str(installer),
            "--platform",
            "linux",
            "--dry-run",
            "--skip-system-deps",
            "--no-launch",
            "--install-root",
            str(tmp_path / language / "spacr"),
            "--package-spec",
            "spacr[qt]==9.9.9",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert catalog["installer_title"] in result.stdout
    assert catalog["application"] in result.stdout
    assert catalog["dry_download"].replace(
        "%s", "https://astral.sh/uv/0.11.32/install.sh"
    ) in result.stdout
    assert not (tmp_path / language).exists()


@pytest.mark.parametrize(
    "language", ("en", "sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")
)
def test_powershell_and_nsis_dry_run_language_paths_keep_exact_catalogs(
    language,
):
    renderer = _installer_i18n_module()
    values = renderer.catalogs()
    powershell = renderer.render_powershell(values)
    nsis_messages = renderer.render_nsis(values)
    nsis_installer = _text(NSIS)

    # PowerShell's dry-run bootstrap reads this exact per-language table.
    ps_start = powershell.index(f"  '{language}' = @{{")
    ps_end = powershell.index("  }", ps_start)
    ps_catalog = powershell[ps_start:ps_end]
    for key, value in values[language].items():
        expected = (
            f"    {renderer._ps_quote(key)} = "
            f"{renderer._ps_quote(renderer._ps_format(value))}"
        )
        assert expected in ps_catalog

    nsis_name = renderer.NSIS_LANGUAGE[language]
    assert f'!insertmacro MUI_LANGUAGE "{nsis_name}"' in nsis_messages
    for key in (
        "nsis_launch", "nsis_gpu", "nsis_application",
        "nsis_downloading", "nsis_failed", "nsis_uninstall",
    ):
        expected = (
            f"LangString SPACR_{key.upper()} ${{LANG_{nsis_name.upper()}}} "
            f'"{renderer._nsis_escape(values[language][key])}"'
        )
        assert expected in nsis_messages

    if language == "en":
        assert 'StrCpy $3 "en"' in nsis_installer
    else:
        selector = re.compile(
            rf"\$LANGUAGE == \$\{{LANG_{nsis_name.upper()}\}}.*?"
            rf'StrCpy \$3 "{re.escape(language)}"',
            re.DOTALL,
        )
        assert selector.search(nsis_installer)
    assert '-Language "$3"' in nsis_installer


def test_installer_catalogs_use_reviewed_software_and_screening_terms():
    locale_dir = ROOT / "packaging" / "i18n"
    expected = {
        "sv": ("program", "CRISPR-screeningar", "PyTorch-backend"),
        "de": ("Anwendung", "CRISPR-Screens", "PyTorch-Backend"),
        "es": ("aplicación", "cribados CRISPR", "backend de PyTorch"),
        "zh_CN": ("应用程序", "CRISPR 筛选", "PyTorch 后端"),
        "pt": ("aplicativo", "triagens CRISPR", "backend do PyTorch"),
        "hi": ("एप्लिकेशन", "CRISPR स्क्रीनिंग", "PyTorch बैकएंड"),
        "ko": ("애플리케이션", "CRISPR 스크리닝", "PyTorch 백엔드"),
        "is": ("forrit", "CRISPR-skimunum", "PyTorch-bakendi"),
        "fr": ("application", "criblages CRISPR", "backend PyTorch"),
    }
    for language, (application, screening, backend) in expected.items():
        table = json.loads(_text(locale_dir / f"{language}.json"))
        assert table["application"] == application
        assert screening in table["desktop_comment"]
        assert backend in table["pytorch_backend"]


def test_windows_selected_installer_language_reaches_bootstrap():
    nsis = _text(NSIS)
    powershell = _text(WINDOWS)
    for code in ("sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr"):
        assert f'StrCpy $3 "{code}"' in nsis
    assert '-Language "$3"' in nsis
    assert '[string]$Language = ""' in powershell
    assert '$env:SPACR_INSTALL_LANGUAGE = $Language' in powershell


def test_unix_installer_uses_external_localized_messages():
    source = _text(UNIX)
    assert "@SPACR_INSTALLER_MESSAGES_BEGIN@" in source
    assert 'source "$SPACR_INSTALLER_DIR/generated/installer_messages.sh"' in source
    assert "spacr_say installed" in source
    for english in (
        "spaCR installed successfully.",
        "Downloading the pinned uv bootstrap...",
        "Creating an isolated spaCR environment...",
    ):
        assert english not in source


def test_macos_builder_creates_application_and_pkg_with_uninstall_helper():
    source = _text(ONLINE / "build_macos_online.sh")
    assert "/Applications/spaCR.app" in source
    assert "pkgbuild" in source
    assert "codesign" in source
    assert "iconutil -c icns" in source
    assert "CFBundleIconFile" in source
    assert "uninstall-spacr.sh" in source
    assert "PRODUCTSIGN_IDENTITY" in source
    assert "install-for-user.sh" in source
    assert "osascript" in source
    assert '$HOME/Library/Application Support/spaCR' in source
    assert "--no-command-launcher" in source
    postinstall = source[source.index('cat > "$SCRIPTS/postinstall"'):]
    assert "install-online.sh" not in postinstall


def test_unix_bootstrap_parses_and_dry_run_never_downloads(tmp_path):
    installer = _standalone_unix_installer(tmp_path)
    subprocess.run(["bash", "-n", str(installer)], check=True)
    env = os.environ.copy()
    env["SPACR_TORCH_BACKEND"] = "cpu"
    result = subprocess.run(
        [
            "bash",
            str(installer),
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
        env=env,
    )
    assert "spacr[qt]==9.9.9" in result.stdout
    assert "PyTorch backend: cpu" in result.stdout
    assert "DRY RUN" in result.stdout
    assert not (tmp_path / "spacr").exists()


@pytest.mark.parametrize(
    ("platform_name", "machine", "nvidia_status", "expected"),
    (
        ("linux", "x86_64", 0, "auto"),
        ("linux", "x86_64", 1, "cpu"),
        ("macos", "arm64", 1, "auto"),
        ("macos", "x86_64", 1, "cpu"),
    ),
)
def test_unix_backend_default_follows_detected_hardware(
    tmp_path, platform_name, machine, nvidia_status, expected
):
    installer = _standalone_unix_installer(tmp_path)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    nvidia = fake_bin / "nvidia-smi"
    nvidia.write_text(
        f"#!/bin/sh\nexit {nvidia_status}\n", encoding="utf-8"
    )
    nvidia.chmod(0o755)
    uname = fake_bin / "uname"
    uname.write_text(f"#!/bin/sh\necho {machine}\n", encoding="utf-8")
    uname.chmod(0o755)
    env = os.environ.copy()
    env.pop("SPACR_TORCH_BACKEND", None)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    result = subprocess.run(
        [
            "bash", str(installer), "--platform", platform_name,
            "--dry-run", "--install-root", str(tmp_path / "spacr"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert f"PyTorch backend: {expected}" in result.stdout


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
    assert "--localized-readme-dir docs/i18n/readme" in workflow
    assert "README.{sv,de,es,zh_CN,pt,hi,ko,is,fr}.rst" in workflow
    assert "for language in en sv de es zh_CN pt hi ko is fr" in workflow
    assert (
        '$Languages = @("en", "sv", "de", "es", "zh_CN", "pt", '
        '"hi", "ko", "is", "fr")'
    ) in workflow
    assert "Language.Parser]::ParseFile" in workflow
    assert "generated\\installer_messages.ps1" in workflow
    assert workflow.index("python .\\packaging\\i18n\\render.py") < (
        workflow.index("Language.Parser]::ParseFile")
    )
    assert workflow.index("Language.Parser]::ParseFile") < workflow.index(
        "$Languages = @"
    )
    assert "PowerShell dry run failed for $Language" in workflow
    assert "spacr/application" in workflow
    for platform in ("Linux", "Windows", "macOS"):
        assert f"Install and import the checked-out {platform} application" in (
            workflow
        )
    assert workflow.count("timeout-minutes: 30") == 3
    assert workflow.count("assert torch.version.cuda is None") == 3
    assert workflow.count("smoke_installed.py") == 3
    assert workflow.count("install-profile.json") >= 3
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
    assert 'git config user.name "Einar Olafsson"' in workflow
    assert (
        'git config user.email '
        '"einar.olafsson@gmail.com"'
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
    old_version = "1.0.0"

    def write_readme(path, translated_label):
        links = [helper.README_BEGIN, ""]
        for label, suffix in helper.PLATFORMS:
            old_name = f"spaCR-{old_version}-{suffix}"
            old_url = (
                f"{helper.RELEASE_DOWNLOAD_ROOT}/v{old_version}/{old_name}"
            )
            links.append(
                f"* `{translated_label} {label}: spaCR {old_version} "
                f"localized-action <{old_url}>`_"
            )
        links.extend(["", helper.README_END])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "Before\n\n" + "\n".join(links) + "\n\nAfter\n",
            encoding="utf-8",
        )

    write_readme(readme, "English download")
    localized_dir = tmp_path / "docs" / "i18n" / "readme"
    for code in helper.LOCALIZED_README_CODES:
        write_readme(localized_dir / f"README.{code}.rst", f"translated-{code}")
    source = tmp_path / "artifacts"
    source.mkdir()
    names = [
        f"spaCR-{version}-Windows-Online-Setup.exe",
        f"spaCR-{version}-macOS-Universal-Online.pkg",
        f"spaCR-{version}-Linux-x86_64-Online.run",
    ]
    for index, name in enumerate(names):
        nested = source / f"job-{index}"
        nested.mkdir()
        (nested / name).write_bytes(f"installer-{index}".encode())
    destination = tmp_path / "application"
    destination.mkdir()
    old = destination / "spaCR-1.0.0-Linux-x86_64-Online.run"
    old.write_bytes(b"old")

    copied = helper.collect_installers(
        source, destination, readme, setup, branch="nightly")

    assert {path.name for path in copied} == set(names)
    assert not old.exists()
    all_readmes = [readme] + [
        localized_dir / f"README.{code}.rst"
        for code in helper.LOCALIZED_README_CODES
    ]
    assert len(all_readmes) == 10
    for index, current_readme in enumerate(all_readmes):
        links = current_readme.read_text(encoding="utf-8")
        translated_label = (
            "English download" if index == 0
            else f"translated-{helper.LOCALIZED_README_CODES[index - 1]}"
        )
        assert translated_label in links
        assert old_version not in links
        for name in names:
            assert name in links
            assert (
                f"https://github.com/EinarOlafsson/spacr/releases/download/"
                f"v{version}/{name}"
            ) in links
        assert "/raw/nightly/" not in links
    for name in names:
        assert (destination / name).is_file()
    manifest = (destination / "README.rst").read_text(encoding="utf-8")
    assert f"Current version: ``{version}``" in manifest
    assert manifest.count("SHA-256") == 4  # heading + one line per installer


def test_release_helper_refuses_an_incomplete_localized_readme_set(tmp_path):
    helper = _release_module()
    version = "2.3.4"
    setup = tmp_path / "setup.py"
    setup.write_text(f'VERSION = "{version}"\n', encoding="utf-8")
    readme = tmp_path / "README.rst"
    original = (
        "Before\n.. spacr-installer-links-begin\n"
        ".. spacr-installer-links-end\nAfter\n"
    )
    readme.write_text(original, encoding="utf-8")
    locale_dir = tmp_path / "docs" / "i18n" / "readme"
    locale_dir.mkdir(parents=True)
    for code in helper.LOCALIZED_README_CODES[:-1]:
        (locale_dir / f"README.{code}.rst").write_text(
            original, encoding="utf-8"
        )
    source = tmp_path / "artifacts"
    source.mkdir()
    for _label, suffix in helper.PLATFORMS:
        (source / f"spaCR-{version}-{suffix}").write_bytes(b"installer")
    destination = tmp_path / "application"

    with pytest.raises(FileNotFoundError, match="README.fr.rst"):
        helper.collect_installers(
            source, destination, readme, setup, branch="nightly"
        )

    assert _text(readme) == original
    assert not destination.exists()


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
