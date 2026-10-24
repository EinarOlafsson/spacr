"""The online installer runs on macOS's bash 3.2 when it is given no arguments.

Found 2026-10-24 on the first Apple Silicon test machine (M2). macOS ships
``/bin/bash`` 3.2, where ``set -u`` makes ``"${array[@]}"`` of an EMPTY array
an unbound-variable error (fixed only in bash 4.4). The installer saves its
arguments in ``SPACR_ORIGINAL_ARGS`` and re-executes itself as arm64 when it
finds itself translated by Rosetta -- so the plain, argument-free
``curl ... | bash`` from a Terminal under Rosetta, the case the re-exec exists
for, died there instead::

    install_spacr_unix.sh: line 204: SPACR_ORIGINAL_ARGS[@]: unbound variable

Linux CI never saw it: every Linux bash there is 4.4 or newer.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "packaging" / "online" / "install_spacr_unix.sh"
BASH = "/bin/bash" if Path("/bin/bash").exists() else shutil.which("bash")


def _reexec_line() -> str:
    lines = [line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
             if "exec /usr/bin/arch -arm64" in line]
    assert len(lines) == 1, lines
    return lines[0].strip()


@pytest.mark.posix_only
@pytest.mark.skipif(BASH is None, reason="needs bash")
@pytest.mark.parametrize("arguments", [(), ("--no-launch",), ("--install-root", "a dir")])
def test_the_arm64_reexec_passes_exactly_the_arguments_it_was_given(tmp_path, arguments):
    """The line itself, run by this system's bash under the script's own options."""
    line = _reexec_line().replace("exec /usr/bin/arch -arm64 /bin/bash",
                                  "printf '<%s>' ", 1)
    harness = tmp_path / "reexec.sh"
    harness.write_text("set -Eeuo pipefail\nSPACR_ORIGINAL_ARGS=(\"$@\")\n" + line + "\n",
                       encoding="utf-8")
    result = subprocess.run([BASH, str(harness), *arguments], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "".join(f"<{part}>" for part in (str(harness), *arguments))


def test_no_array_in_the_installer_is_expanded_unguarded_where_it_can_be_empty():
    """The arrays that can be empty: the arguments, the GUI child's, and
    ``elevate``, which is empty when the installer already runs as root."""
    source = SCRIPT.read_text(encoding="utf-8")
    for name in ("SPACR_ORIGINAL_ARGS", "child_args", "elevate", "build_deps"):
        assert f'"${{{name}[@]}}"' not in source.replace(
            f'${{{name}[@]+"${{{name}[@]}}"}}', ""), name


def _translated_bash_available() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    probe = subprocess.run(["/usr/bin/arch", "-x86_64", "/usr/bin/true"],
                           capture_output=True)
    return probe.returncode == 0


@pytest.mark.skipif(not _translated_bash_available(),
                    reason="needs an Apple Silicon Mac with Rosetta 2")
@pytest.mark.skipif(not (SCRIPT.parent / "generated" / "installer_messages.sh").exists(),
                    reason="render packaging/i18n/render.py first")
def test_a_translated_terminal_with_no_arguments_reaches_the_plan():
    """The real script, run as Intel code with no arguments, in dry-run mode."""
    result = subprocess.run(
        ["/usr/bin/arch", "-x86_64", "/bin/bash", str(SCRIPT)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120,
        env={**os.environ, "SPACR_INSTALL_DRY_RUN": "1"})
    assert "unbound variable" not in result.stderr, result.stderr
    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
