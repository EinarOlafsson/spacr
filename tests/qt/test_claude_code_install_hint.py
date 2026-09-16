"""Claude Code installs with curl on every system, Windows included (item 414).

The maintainer, on a Windows machine without Node.js: "just allways use
curl". The Windows hint used to be ``npm install -g @anthropic-ai/claude-code``,
which fails with "'npm' is not recognized" wherever Node.js is absent.

The commands are Anthropic's own native installers, copied from
https://code.claude.com/docs/en/setup (retrieved 2026-09-15). The Windows one
is the page's "Windows CMD" command. In PowerShell it fails twice, because
``curl`` can be an alias for ``Invoke-WebRequest`` and Windows PowerShell 5.1
rejects ``&&``. So the hint names its shell (``cmd /c "..."``), which
makes the one copied line run whole whichever of the two shells it is pasted
into.

``install_hint`` is chosen from ``sys.platform`` while the class body runs, so
one machine only ever sees its own branch. Each platform is therefore read
from a private copy of the module executed with ``sys.platform`` swapped in.
That copy is never registered in ``sys.modules``, so the provider classes
every other test uses are left alone.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import spacr.qt.ai.providers as live_providers

PROVIDERS_PY = Path(live_providers.__file__)

DOCUMENTED_MACOS_LINUX = "curl -fsSL https://claude.ai/install.sh | bash"
DOCUMENTED_WINDOWS_CMD = (
    "curl -fsSL https://claude.ai/install.cmd -o install.cmd"
    " && install.cmd && del install.cmd"
)

PLATFORMS = ("win32", "linux", "darwin")


def _claude_install_hint_on(platform: str) -> str:
    """The Claude Code install hint providers.py produces on ``platform``."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "platform", platform)
        spec = importlib.util.spec_from_file_location(
            f"_claude_install_hint_probe_{platform}", PROVIDERS_PY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.ClaudeCliProvider.install_hint


def test_the_probe_reads_the_same_code_this_process_imported():
    assert "/spacr/qt/ai/providers.py" in PROVIDERS_PY.as_posix()
    assert (_claude_install_hint_on(sys.platform)
            == live_providers.ClaudeCliProvider.install_hint)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_no_claude_code_install_hint_starts_with_npm(platform):
    hint = _claude_install_hint_on(platform)
    assert not hint.lstrip().startswith("npm"), (platform, hint)
    assert "npm" not in hint.split(), (platform, hint)
    assert "curl" in hint, (platform, hint)


def test_the_windows_hint_names_its_shell_or_uses_curl_exe():
    hint = _claude_install_hint_on("win32")
    names_its_shell = hint.lower().startswith(("cmd /c ", "cmd.exe /c "))
    assert names_its_shell or "curl.exe" in hint, hint


def test_the_windows_hint_is_the_documented_cmd_installer_run_by_cmd():
    assert (_claude_install_hint_on("win32")
            == f'cmd /c "{DOCUMENTED_WINDOWS_CMD}"')


@pytest.mark.parametrize("platform", ("linux", "darwin"))
def test_macos_and_linux_hints_are_the_documented_installer(platform):
    assert _claude_install_hint_on(platform) == DOCUMENTED_MACOS_LINUX
