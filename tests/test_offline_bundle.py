"""Contracts of the offline (air-gapped) installer bundle.

The builder runs with network access and packs uv, CPython, every wheel,
Cellpose weights and Mask test data; the online installers install from it
with ``--offline-bundle`` and never open a socket. Nothing here downloads.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OFFLINE = ROOT / "packaging" / "offline"
ONLINE = ROOT / "packaging" / "online"
UNIX = ONLINE / "install_spacr_unix.sh"
WINDOWS = ONLINE / "install_spacr_windows.ps1"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load("spacr_offline_builder", OFFLINE / "build_offline_bundle.py")


@pytest.fixture(scope="module")
def mask_check():
    return _load("spacr_offline_mask_check", OFFLINE / "offline_mask_check.py")


def _unix_installer(tmp_path):
    renderer = _load("spacr_installer_i18n_offline", ROOT / "packaging" / "i18n" / "render.py")
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "installer_messages.sh").write_text(
        renderer.render_shell(renderer.catalogs()), encoding="utf-8")
    renderer.OUTPUT_DIR = generated
    installer = tmp_path / "install.sh"
    renderer.embed_unix(UNIX, installer, "9.9.9")
    return installer


def test_bundle_names_carry_platform_and_torch_line(builder):
    name = builder.release.offline_bundle_name("1.5.1.0", "linux-x86_64", "cu126")
    assert name == "spaCR-1.5.1.0-Linux-x86_64-Offline-cu126"
    with pytest.raises(ValueError):
        builder.release.offline_bundle_name("1.5.1.0", "linux-x86_64", "auto;rm")
    with pytest.raises(ValueError):
        builder.release.offline_bundle_name("1.5.1.0", "solaris", "cpu")
    assert set(builder.TARGETS) == set(builder.release.OFFLINE_PLATFORMS)


def test_builder_pins_the_same_uv_and_guards_as_the_online_installer(builder):
    assert builder.uv_version() == "0.11.32"
    assert builder.resolver_guards() == ["numba>=0.60,<1.0", "llvmlite>=0.43,<1.0"]


def test_builder_defaults_to_core_package_and_accepts_explicit_extras(builder):
    """An offline bundle uses core spaCR unless extras are requested."""
    parser = builder.build_parser()
    default = parser.parse_args([])
    assert default.extras == ""
    assert builder.package_requirement("9.9.9", default.extras) == "spacr==9.9.9"
    chosen = parser.parse_args(["--extras", "qt"])
    assert builder.package_requirement("9.9.9", chosen.extras) == "spacr[qt]==9.9.9"
    assert parser.parse_args(["--package-spec", "spacr[qt]==9.9.9"]).package_spec == (
        "spacr[qt]==9.9.9")


def test_python_download_matches_the_target_and_its_mirror_path(builder):
    entries = [
        {"implementation": "cpython", "variant": "freethreaded", "os": "linux",
         "arch": "x86_64", "libc": "gnu", "url": "https://x/a/free.tar.gz"},
        {"implementation": "cpython", "variant": "default", "os": "linux",
         "arch": "x86_64", "libc": "musl", "url": "https://x/a/musl.tar.gz"},
        {"implementation": "cpython", "variant": "default", "os": "linux",
         "arch": "x86_64", "libc": "gnu", "version": "3.12.13", "key": "k",
         "url": "https://h/releases/download/20260718/"
                "cpython-3.12.13%2B20260718-x86_64-unknown-linux-gnu-install_only.tar.gz"},
    ]
    entry = builder.choose_python_download(entries, builder.TARGETS["linux-x86_64"])
    assert entry["libc"] == "gnu" and entry["variant"] == "default"
    assert builder.mirror_relative_path(entry["url"]) == Path(
        "20260718/cpython-3.12.13+20260718-x86_64-unknown-linux-gnu-install_only.tar.gz")
    with pytest.raises(SystemExit):
        builder.choose_python_download(entries, builder.TARGETS["windows-x86_64"])


def test_lock_pins_skip_options_comments_and_markers(builder, tmp_path):
    lock = tmp_path / "requirements.txt"
    lock.write_text("--index-url https://pypi.org/simple\n# comment\n"
                    "numpy==2.1.0\ntorch==2.8.0+cpu ; sys_platform == 'linux'\n\n")
    assert builder.locked_pins(lock) == ["numpy==2.1.0", "torch==2.8.0+cpu"]
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / "numpy-2.1.0-cp312-cp312-manylinux_2_28_x86_64.whl").touch()
    assert builder._have_wheel("numpy==2.1.0", wheels)
    assert not builder._have_wheel("numpy==2.2.0", wheels)


def test_test_data_keeps_whole_fields(builder, tmp_path):
    names = [f"plate1_E01_T0001F00{f}L01A0{c}Z01C0{c}.tif"
             for f in (1, 2, 3) for c in (1, 2, 3)]
    chosen = builder.select_test_fields([tmp_path / n for n in names], 2)
    assert len(chosen) == 6
    assert {p.name[:20] for p in chosen} == {"plate1_E01_T0001F001", "plate1_E01_T0001F002"}
    assert len(builder.select_test_fields([tmp_path / n for n in names], 0)) == 9


def test_checksums_cover_every_file_but_themselves(builder, tmp_path):
    (tmp_path / "wheels").mkdir()
    (tmp_path / "wheels" / "a.whl").write_bytes(b"a")
    (tmp_path / "bundle.json").write_text("{}")
    sums = builder.write_checksums(tmp_path).read_text().splitlines()
    assert [line.split("  ")[1] for line in sums] == ["bundle.json", "wheels/a.whl"]
    result = subprocess.run(["sha256sum", "--check", "--quiet", "SHA256SUMS"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_mask_check_replaces_only_models_the_bundle_cannot_resolve(mask_check, tmp_path):
    settings = tmp_path / "gen_mask_settings.csv"
    settings.write_text("Key,Value\ncell_model_name,cpsam\n"
                        "pathogen_model_name,<models>/cpsam_v2_toxo_r2\nsrc,<src>\n")
    assert mask_check.model_overrides(settings, "cpsam") == ["pathogen_model_name=cpsam"]


def test_unix_offline_dry_run_uses_the_bundle_and_never_downloads(tmp_path):
    installer = _unix_installer(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bundle.json").write_text(json.dumps(
        {"torch_backend": "cu126", "package_spec": "spacr[qt]==9.9.9"}, indent=2))
    env = dict(os.environ, SPACR_TORCH_BACKEND="auto")
    result = subprocess.run(
        ["bash", str(installer), "--platform", "linux", "--dry-run", "--no-launch",
         "--offline-bundle", str(bundle), "--install-root", str(tmp_path / "spacr")],
        check=True, capture_output=True, text=True, env=env)
    assert f"Offline bundle: {bundle}" in result.stdout
    assert "PyTorch backend: cu126" in result.stdout
    assert "astral.sh" not in result.stdout
    assert not (tmp_path / "spacr").exists()


def test_unix_offline_refuses_a_folder_that_is_not_a_bundle(tmp_path):
    installer = _unix_installer(tmp_path)
    result = subprocess.run(
        ["bash", str(installer), "--platform", "linux", "--dry-run",
         "--offline-bundle", str(tmp_path), "--install-root", str(tmp_path / "spacr")],
        capture_output=True, text=True)
    assert result.returncode == 2
    assert "Not an offline bundle" in result.stderr


def test_offline_installs_verify_then_install_without_an_index():
    unix = UNIX.read_text(encoding="utf-8")
    windows = WINDOWS.read_text(encoding="utf-8")
    assert unix.index("sha256sum --check") < unix.index('cp "$OFFLINE_BUNDLE/uv/uv"')
    assert '--offline --no-index' in unix
    assert 'UV_PYTHON_INSTALL_MIRROR="file://$OFFLINE_BUNDLE/python"' in unix
    assert windows.index("Get-FileHash") < windows.index('Copy-Item (Join-Path $OfflineBundle "uv\\uv.exe")')
    assert "--offline --no-index" in windows
    assert "$env:UV_PYTHON_INSTALL_MIRROR" in windows
    for source in (unix, windows):
        assert "offline_mask_check.py" in source
        assert ".cellpose" in source
