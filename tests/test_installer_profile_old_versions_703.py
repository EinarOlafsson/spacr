"""Current installers write profiles without calling an older package's CLI."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ONLINE = ROOT / "packaging" / "online"
PROFILE = ROOT / "spacr" / "install_profile.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(os.name == "nt" or not shutil.which("bash"), reason="Unix installer invocation")
@pytest.mark.parametrize("embedded", [False, True], ids=["checkout", "standalone"])
@pytest.mark.parametrize("without_wgpu", ["0", "1"])
def test_installer_profile_uses_current_writer_with_old_installed_package(
    tmp_path, embedded, without_wgpu,
):
    renderer = _load("profile_compat_renderer", ROOT / "packaging" / "i18n" / "render.py")
    source = (ONLINE / "install_spacr_unix.sh").read_text(encoding="utf-8")
    if embedded:
        source = renderer._embed_profile_module(source)
        assert renderer._embed_profile_module(source) == source
    start = source.index("spacr_write_profile_module() {")
    end = source.index("# @SPACR_PROFILE_MODULE_END@", start)
    writer = tmp_path / "installer-owned-profile.py"
    subprocess.run(
        ["bash", "-c", source[start:end] + '\nspacr_write_profile_module "$1"',
         "profile-test", str(writer)],
        env={**os.environ, "SPACR_SOURCE_DIR": str(ONLINE)}, check=True,
    )
    assert writer.read_bytes() == PROFILE.read_bytes()

    stage = tmp_path / "stage"
    venv.EnvBuilder(with_pip=False).create(stage)
    python = stage / "bin" / "python"
    site = Path(subprocess.check_output(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True,
    ).strip())
    old_package = site / "spacr"
    old_package.mkdir()
    (old_package / "__init__.py").write_text("__version__ = '1.5.0.7'\n")
    (old_package / "install_profile.py").write_text(
        "import argparse\np = argparse.ArgumentParser()\n"
        "for name in ('path', 'requested', 'detected', 'consent-collected', 'share-diagnostics', 'report-issues'):\n"
        "    p.add_argument('--' + name)\n"
        "p.parse_args()\n",
    )
    rejected = subprocess.run(
        [str(python), "-I", "-m", "spacr.install_profile", "--without-wgpu", without_wgpu],
        capture_output=True, text=True,
    )
    assert rejected.returncode == 2
    assert "unrecognized arguments: --without-wgpu" in rejected.stderr
    (site / "torch.py").write_text(
        "from types import SimpleNamespace as N\n"
        "__version__ = 'staged-torch'\nversion = N(cuda='test-cuda')\n"
        "cuda = N(is_available=lambda: True)\n"
        "backends = N(mps=N(is_available=lambda: False))\n",
    )
    target = tmp_path / "profile.json"
    invocation = source[source.index('"$stage_python" -I "$profile_module"'):]
    invocation = invocation[:invocation.index("\n\n")]
    completed = subprocess.run(
        ["bash", "-c", invocation], check=True, capture_output=True, text=True,
        env={**os.environ, "stage_python": str(python), "profile_module": str(writer),
             "stage_profile": str(target), "TORCH_BACKEND": "cu128", "DETECTED_ACCELERATOR": "nvidia",
             "CONSENT_COLLECTED": "1", "SHARE_DIAGNOSTICS": "0", "REPORT_ISSUES": "1",
             "WITHOUT_WGPU": without_wgpu},
    )
    profile = json.loads(target.read_text())
    assert json.loads(completed.stdout) == profile
    assert profile["schema"] == 1
    assert profile["without_wgpu"] is (without_wgpu == "1")
    assert profile["consent"] == {"collected": True, "share_diagnostics": False, "report_issues": True}
    assert profile["requested_backend"] == "cu128"
    assert profile["detected_accelerator"] == "nvidia"
    assert profile["torch_version"] == "staged-torch"
    assert profile["torch_cuda_build"] == "test-cuda"
    assert profile["active_backend"] == "cuda"
    assert profile["cuda_available"] is True and profile["mps_available"] is False
    assert not list(tmp_path.glob(".profile.json.*.tmp"))


def test_windows_installer_and_distributables_carry_the_current_writer(tmp_path, monkeypatch):
    windows = (ONLINE / "install_spacr_windows.ps1").read_text()
    assert '$ProfileModule = Join-Path $PSScriptRoot "install_profile.py"' in windows
    assert '$ProfileModule = Join-Path $PSScriptRoot "..\\..\\spacr\\install_profile.py"' in windows
    invocation = re.search(
        r'Invoke-Checked -Command \$StagePython -Arguments @\(\s*"-I",\s*"-X",\s*"utf8",\s*\$ProfileModule,.*?\n    \)',
        windows, re.DOTALL,
    )
    assert invocation is not None
    assert '"--without-wgpu"' in invocation.group()
    assert '$StageProfile' in invocation.group()
    assert windows.index(invocation.group()) < windows.index('$OldVenv =')
    assert '"-m",\n        "spacr.install_profile"' not in windows
    nsis = (ONLINE / "spacr_online_installer.nsi").read_text()
    assert 'File "..\\..\\spacr\\install_profile.py"' in nsis
    assert 'Delete "$PLUGINSDIR\\spaCR-online-installer\\install_profile.py"' in nsis
    builder = _load("profile_compat_offline", ROOT / "packaging" / "offline" / "build_offline_bundle.py")
    renderer = _load("profile_compat_windows_messages", ROOT / "packaging" / "i18n" / "render.py")
    online = tmp_path / "source"
    generated = online / "generated"
    generated.mkdir(parents=True)
    for name in ("windows_crt.py", "windows_crt.json"):
        (online / name).write_bytes((ONLINE / name).read_bytes())
    (generated / "installer_messages.ps1").write_text(
        renderer.render_powershell(renderer.catalogs()), encoding="utf-8",
    )
    monkeypatch.setattr(builder, "ONLINE", online)
    monkeypatch.setattr(builder, "run", lambda *args, **kwargs: None)
    builder.render_installer(builder.TARGETS["windows-x86_64"], "1.5.1.5", tmp_path)
    assert (tmp_path / "install_profile.py").read_bytes() == PROFILE.read_bytes()
    assert (tmp_path / "install_cleanup.py").read_bytes() == (ROOT / "spacr" / "install_cleanup.py").read_bytes()
    for name in ("windows_crt.py", "windows_crt.json"):
        assert (tmp_path / name).read_bytes() == (ONLINE / name).read_bytes()
