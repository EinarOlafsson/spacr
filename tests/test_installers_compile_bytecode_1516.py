"""Fresh installs compile bytecode, so the first module open does not pay for it.

Measured on a Mac after a fresh install (1.5.1.5): the first Regression open took about 5 s, compiling the
modules it imports. Every installer package step now passes --compile-bytecode to uv; the macOS first-run
check runs the shell installer, so it is covered by the same step.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ONLINE = ROOT / "packaging" / "online"


def _uv_installs(text, start):
    """Each uv pip install command in ``text``, joined across line continuations."""
    joined = re.sub(r"\\\n\s*|`\n\s*", " ", text)
    return [line for line in joined.splitlines() if start in line and "pip install" in line]


def test_the_shell_installer_compiles_bytecode_in_every_install():
    commands = _uv_installs((ONLINE / "install_spacr_unix.sh").read_text(encoding="utf-8"), '"$UV_BIN" pip install')
    assert len(commands) == 2
    assert all("--compile-bytecode" in command for command in commands)


def test_the_windows_installer_compiles_bytecode_in_every_install():
    commands = _uv_installs((ONLINE / "install_spacr_windows.ps1").read_text(encoding="utf-8"),
                            "$UvExe pip install")
    assert len(commands) == 2
    assert all("--compile-bytecode" in command for command in commands)


def test_the_macos_first_run_check_installs_through_the_shell_installer():
    wrapper = (ONLINE / "install_for_macos_user.sh").read_text(encoding="utf-8")
    assert 'INSTALLER="$SUPPORT_DIR/install-online.sh"' in wrapper and '"$INSTALLER" --platform macos' in wrapper
    assert "pip install" not in wrapper
