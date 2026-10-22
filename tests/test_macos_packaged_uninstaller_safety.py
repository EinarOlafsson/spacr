"""Execute the packaged shell body with only its fixed paths relocated to a sandbox."""
import os
from pathlib import Path
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.posix_only


@pytest.fixture
def installation(tmp_path):
    home = tmp_path / "home"
    app = tmp_path / "Applications/spaCR.app"
    support = tmp_path / "Library/Application Support/spaCR"
    launcher = tmp_path / "usr/local/bin/spacr"
    for directory in (app, support, launcher.parent):
        directory.mkdir(parents=True)
    (app / "sentinel").write_text("application")
    (support / "installer_messages.sh").write_text(
        "spacr_say() { printf '%s\\n' \"$*\"; }\n")
    source = (ROOT / "packaging/online/build_macos_online.sh").read_text()
    body = source.split('cat > "$SUPPORT/uninstall-spacr.sh" <<\'EOF\'\n', 1)[1].split("\nEOF\n", 1)[0]
    body = body.replace("$HOME/Library/Application Support/spaCR", "SPACR_TEST_USER_RUNTIME")
    for original, replacement in (("/Applications/spaCR.app", app),
                                  ("/Library/Application Support/spaCR", support),
                                  ("/usr/local/bin/spacr", launcher)):
        body = body.replace(original, str(replacement))
    body = body.replace("SPACR_TEST_USER_RUNTIME", "$HOME/Library/Application Support/spaCR")
    script = support / "uninstall-spacr.sh"
    script.write_text(body + "\n")
    python = home / "Library/Application Support/spaCR/venv/bin/python"
    python.parent.mkdir(parents=True)
    log = tmp_path / "python-arguments.txt"
    python.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > " + shlex.quote(str(log)) +
                      '\nexit "${PURGE_EXIT:-0}"\n')
    python.chmod(0o755)
    return home, app, support, launcher, script, log


def run_uninstaller(installation, *args, purge_exit=0):
    home, _app, _support, _launcher, script, _log = installation
    return subprocess.run(["sh", str(script), *args], text=True, capture_output=True,
                          env=dict(os.environ, HOME=str(home), PURGE_EXIT=str(purge_exit)), timeout=10)


@pytest.mark.parametrize("kind", ["owned", "regular", "unrelated", "different-app-command"])
def test_only_the_exact_owned_launcher_symlink_is_removed(installation, kind):
    _home, app, support, launcher, _script, _log = installation
    if kind == "regular":
        launcher.write_text("unrelated command")
    else:
        target = (app / "Contents/MacOS/spaCR" if kind == "owned" else
                  app / "Contents/MacOS/other" if kind == "different-app-command" else
                  launcher.parent / "another-program")
        launcher.symlink_to(target)
    result = run_uninstaller(installation)
    assert result.returncode == 0, result.stderr
    assert not app.exists() and not support.exists()
    assert os.path.lexists(launcher) is (kind != "owned")
    if kind == "regular":
        assert launcher.read_text() == "unrelated command"
    elif kind != "owned":
        assert launcher.readlink() == target


@pytest.mark.parametrize("purge", [False, True])
def test_dry_run_lists_paths_without_removing_software_or_data(installation, purge):
    _home, app, support, launcher, _script, log = installation
    launcher.symlink_to(app / "Contents/MacOS/spaCR")
    args = ["--purge", "--yes", "--dry-run"] if purge else ["--dry-run"]
    result = run_uninstaller(installation, *args)
    assert result.returncode == 0, result.stderr
    assert (app / "sentinel").read_text() == "application"
    assert support.is_dir() and launcher.is_symlink()
    for path in (app, support, launcher):
        assert str(path) in result.stdout
    if purge:
        assert log.read_text().splitlines() == ["-I", str(support / "verify_runtime.py"), "purge", "--dry-run", "--yes"]
    else:
        assert not log.exists()


def test_unknown_option_is_rejected_before_removal(installation):
    _home, app, support, _launcher, _script, log = installation
    result = run_uninstaller(installation, "--dr-run")
    assert result.returncode == 2 and "--dr-run" in result.stderr
    assert (app / "sentinel").read_text() == "application" and support.is_dir()
    assert not log.exists()


@pytest.mark.parametrize("code", [1, 3])
def test_failed_or_declined_purge_keeps_the_software(installation, code):
    _home, app, support, _launcher, _script, log = installation
    result = run_uninstaller(installation, "--purge", purge_exit=code)
    assert result.returncode == code
    assert (app / "sentinel").read_text() == "application" and support.is_dir()
    assert log.read_text().splitlines() == ["-I", str(support / "verify_runtime.py"), "purge"]


def test_purge_without_its_python_stops_before_software_removal(installation):
    home, app, support, _launcher, _script, _log = installation
    (home / "Library/Application Support/spaCR/venv/bin/python").unlink()
    result = run_uninstaller(installation, "--purge", "--yes")
    assert result.returncode == 1
    assert (app / "sentinel").read_text() == "application" and support.is_dir()


def test_successful_explicit_purge_precedes_software_removal(installation):
    _home, app, support, _launcher, _script, log = installation
    result = run_uninstaller(installation, "--yes", "--purge")
    assert result.returncode == 0, result.stderr
    assert log.read_text().splitlines() == ["-I", str(support / "verify_runtime.py"), "purge", "--yes"]
    assert not app.exists() and not support.exists()
