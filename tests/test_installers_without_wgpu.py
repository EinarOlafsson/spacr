"""The online installers can install spaCR without wgpu, and do on old glibc."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ONLINE = ROOT / "packaging" / "online"
I18N = ROOT / "packaging" / "i18n"
UNIX = ONLINE / "install_spacr_unix.sh"
WINDOWS = ONLINE / "install_spacr_windows.ps1"
NSIS = ONLINE / "spacr_online_installer.nsi"
MACOS = ONLINE / "macos_installer.m"
LANGUAGES = ("en", "sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")
NEW_KEYS = (
    "help_without_wgpu", "without_wgpu", "without_wgpu_description",
    "wgpu_plan", "wgpu_plan_install", "wgpu_plan_skip", "wgpu_glibc_too_old",
)
OVERRIDE = 'wgpu; sys_platform == "never"'


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _renderer():
    spec = importlib.util.spec_from_file_location(
        "spacr_installer_i18n_wgpu", I18N / "render.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def standalone(tmp_path):
    renderer = _renderer()
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "installer_messages.sh").write_text(
        renderer.render_shell(renderer.catalogs()), encoding="utf-8")
    renderer.OUTPUT_DIR = generated
    installer = tmp_path / "spaCR-Linux-x86_64-Online.run"
    renderer.embed_unix(UNIX, installer, "9.9.9")
    return installer


def _dry_run(installer, tmp_path, *extra, env_extra=None, platform="linux"):
    env = os.environ.copy()
    env["SPACR_INSTALL_LANGUAGE"] = "en"
    env["SPACR_TORCH_BACKEND"] = "cpu"
    env.pop("SPACR_WITHOUT_WGPU", None)
    env.pop("SPACR_GLIBC_VERSION", None)
    env.update(env_extra or {})
    result = subprocess.run(
        ["bash", str(installer), "--platform", platform, "--dry-run",
         "--skip-system-deps", "--no-launch",
         "--install-root", str(tmp_path / "spacr"), *extra],
        check=True, capture_output=True, text=True, env=env,
    )
    assert not (tmp_path / "spacr").exists()
    return result.stdout


def test_unix_installer_offers_without_wgpu_off_by_default():
    source = _text(UNIX)
    assert "--without-wgpu)" in source
    assert "help_without_wgpu" in source
    assert 'WITHOUT_WGPU="${SPACR_WITHOUT_WGPU:-0}"' in source
    assert "local wgpu_default=FALSE" in source
    assert '"$wgpu_default" without_wgpu "$(spacr_say without_wgpu)"' in source
    assert "child_options+=(--without-wgpu)" in source


def test_unix_installer_skips_wgpu_at_resolve_time_with_an_override():
    source = _text(UNIX)
    assert OVERRIDE in source
    assert 'override_args=(--override "$wgpu_overrides")' in source
    install = source[source.index('spacr_say downloading_dependencies'):
                     source.index('spacr_say validating_install')]
    assert install.count('${override_args[@]+"${override_args[@]}"}') == 2
    assert "pip uninstall" not in install


def test_unix_installer_checks_glibc_2_28_and_tolerates_unknown_libc():
    source = _text(UNIX)
    assert "getconf GNU_LIBC_VERSION" in source
    assert "ldd --version" in source
    assert "glibc_minor < 28" in source
    assert 'WGPU_SKIP_REASON="glibc"' in source
    assert "spacr_say wgpu_glibc_too_old" in source


def test_windows_installer_offers_without_wgpu_switch():
    source = _text(WINDOWS)
    assert "[switch]$WithoutWgpu" in source
    assert OVERRIDE.replace('"', '`"') in source
    assert '$OverrideArgs = @("--override", $WgpuOverrides)' in source
    assert source.count("@OverrideArgs") == 2


def test_nsis_without_wgpu_section_is_unticked_and_passes_the_switch():
    source = _text(NSIS)
    assert 'Section /o "$(SPACR_WITHOUT_WGPU)" SecWithoutWgpu' in source
    assert "SectionSetFlags ${SecWithoutWgpu} 0" in source
    assert 'StrCpy $5 " -WithoutWgpu"' in source
    assert "-ReportIssues $ConsentIssuesValue$5'" in source
    assert "$(SPACR_WITHOUT_WGPU_DESCRIPTION)" in source


def test_nsis_catalog_defines_the_without_wgpu_strings():
    renderer = _renderer()
    rendered = renderer.render_nsis(renderer.catalogs())
    for name in renderer.NSIS_LANGUAGE.values():
        for key in ("SPACR_WITHOUT_WGPU ", "SPACR_WITHOUT_WGPU_DESCRIPTION "):
            assert f"LangString {key}${{LANG_{name.upper()}}}" in rendered


def test_macos_window_has_an_unchecked_without_wgpu_choice():
    source = _text(MACOS)
    keys = re.search(r"NSArray \*keys = @\[(.*?)\];", source).group(1)
    flags = re.search(r"NSArray \*flags = @\[(.*?)\];", source).group(1)
    assert '@"without_wgpu"' in keys
    assert '@"--without-wgpu"' in flags
    assert keys.split(",").index(' @"without_wgpu"') == (
        flags.split(",").index(' @"--without-wgpu"') + 1)
    assert "index == 0 ? NSControlStateValueOn : NSControlStateValueOff" in source


def test_every_installer_locale_has_the_new_keys():
    english = json.loads(_text(I18N / "en.json"))
    for language in LANGUAGES:
        table = json.loads(_text(I18N / f"{language}.json"))
        for key in NEW_KEYS:
            assert table.get(key), (language, key)
            assert table[key].count("%s") == english[key].count("%s"), (language, key)
        if language != "en":
            assert table["without_wgpu_description"] != english["without_wgpu_description"]
    _renderer().catalogs()


def test_dry_run_installs_wgpu_by_default(standalone, tmp_path):
    out = _dry_run(standalone, tmp_path, env_extra={"SPACR_GLIBC_VERSION": "2.39"})
    assert "wgpu 3D rendering: install" in out
    assert "glibc" not in out


def test_dry_run_reports_without_wgpu(standalone, tmp_path):
    out = _dry_run(standalone, tmp_path, "--without-wgpu")
    assert "wgpu 3D rendering: skip (3D uses OpenGL, else the CPU)" in out


@pytest.mark.parametrize("version", ("2.17", "2.27", "1.9"))
def test_dry_run_skips_wgpu_on_old_glibc(standalone, tmp_path, version):
    out = _dry_run(standalone, tmp_path, env_extra={"SPACR_GLIBC_VERSION": version})
    assert f"wgpu needs glibc 2.28 or newer (this system has {version})" in out
    assert "wgpu 3D rendering: skip" in out


def test_old_glibc_is_ignored_on_macos(standalone, tmp_path):
    out = _dry_run(standalone, tmp_path, platform="macos",
                   env_extra={"SPACR_GLIBC_VERSION": "2.17"})
    assert "wgpu 3D rendering: install" in out


def _fake_tools(tmp_path, getconf, ldd):
    tools = tmp_path / "tools"
    tools.mkdir()
    for name, body in (("getconf", getconf), ("ldd", ldd)):
        if body is None:
            continue
        path = tools / name
        path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        path.chmod(0o755)
    return {"PATH": f"{tools}{os.pathsep}{os.environ['PATH']}"}


@pytest.mark.parametrize(
    ("getconf", "ldd", "skipped"),
    (
        ("echo glibc 2.17", None, True),
        ("echo glibc 2.28", None, False),
        ("exit 1", "echo 'ldd (GNU libc) 2.17'", True),
        ("exit 1", "echo 'ldd (Ubuntu GLIBC 2.39-0ubuntu8) 2.39'", False),
        ("exit 1", "echo 'musl libc (x86_64)' >&2; echo 'Version 1.2.4' >&2; exit 1", False),
    ),
)
def test_glibc_probe_reads_getconf_then_ldd(standalone, tmp_path, getconf, ldd, skipped):
    out = _dry_run(standalone, tmp_path, env_extra=_fake_tools(tmp_path, getconf, ldd))
    assert ("wgpu 3D rendering: skip" in out) is skipped
    assert ("wgpu needs glibc 2.28" in out) is skipped


def test_in_app_updates_keep_wgpu_out_of_an_install_that_left_it_out(tmp_path, monkeypatch):
    """An install made without wgpu stays without it after an in-app update."""
    from spacr import install_profile, updater

    monkeypatch.setenv("SPACR_INSTALL_PROFILE", str(tmp_path / "install-profile.json"))
    monkeypatch.setattr(updater, "find_uv", lambda: "uv")
    profile = {"requested_backend": "auto", "without_wgpu": True}
    monkeypatch.setattr(install_profile, "read_profile", lambda path=None: dict(profile))
    command = updater.upgrade_command()
    override = Path(command[command.index("--override") + 1])
    assert override == tmp_path / install_profile.NO_WGPU_OVERRIDE_NAME
    assert override.read_text(encoding="utf-8").strip() == 'wgpu; sys_platform == "never"'
    profile["without_wgpu"] = False
    assert "--override" not in updater.upgrade_command()
    parsed = install_profile.build_parser().parse_args(
        ["--path", "p", "--requested", "auto", "--detected", "none", "--without-wgpu", "1"])
    assert parsed.without_wgpu == "1"


@pytest.mark.parametrize("problem", ("missing_uv", "unwritable_override"))
def test_an_update_cannot_silently_drop_the_without_wgpu_choice(tmp_path, monkeypatch, problem):
    from spacr import install_profile, updater

    monkeypatch.setenv("SPACR_INSTALL_PROFILE", str(tmp_path / "install-profile.json"))
    monkeypatch.setattr(install_profile, "read_profile", lambda path=None: {
        "requested_backend": "cpu", "without_wgpu": True,
    })
    monkeypatch.setattr(updater, "find_uv", lambda: None if problem == "missing_uv" else "uv")
    if problem == "unwritable_override":
        (tmp_path / install_profile.NO_WGPU_OVERRIDE_NAME).mkdir()

    with pytest.raises(RuntimeError, match="without wgpu"):
        updater.upgrade_command(target_version="1.5.1.5")
