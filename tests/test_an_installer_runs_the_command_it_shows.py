"""The Install button runs the provider's own install hint, and says how it went.

Item 420, asked for on 2026-09-16: "instead of giving the user a curl link it
uses the curl link to downloade e.g. claude and then installs it", and the
same for the GitHub CLI. These tests are about the mechanism,
:mod:`spacr.qt.ai.cli_install`, which is Qt-free; the screen is tested in
``tests/qt/test_the_setup_screen_installs_the_provider_it_offers.py``.

NOTHING IS INSTALLED BY THIS FILE. Every test that is not explicitly about a
real process runs with ``cli_install._spawn`` replaced, and the autouse guard
below fails any test that reaches the real one by accident. The three
real-process tests run a shell script written into ``tmp_path``.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from spacr.qt.ai import cli_install
from spacr.qt.ai import providers
from spacr.qt.ai.providers import (INSTALL_METHODS, InstallMethod,
                                   UNVERIFIED_PLATFORMS, CommandLineTool,
                                   GitHubCli, github_cli, install_hint_for,
                                   install_methods_for, platform_family)


@pytest.fixture(autouse=True)
def _nothing_is_really_run(monkeypatch):
    """Fail loudly if a test would start a real installer or signal a group."""
    def refuse(*args, **kwargs):
        raise AssertionError(f"a real process was about to start: {args!r}")

    monkeypatch.setattr(cli_install, "_spawn", refuse)
    monkeypatch.setattr(cli_install, "_query", refuse)
    monkeypatch.setattr(cli_install, "_signal_group", refuse)


class FakeStdout:
    """An installer's output, handed out in the chunks a pipe would give."""

    def __init__(self, chunks, gate=None):
        self._chunks = list(chunks)
        self._gate = gate
        self.closed = False

    def read1(self, _size):
        if self._chunks:
            return self._chunks.pop(0)
        if self._gate is not None:
            self._gate.wait(10)
        return b""

    def close(self):
        self.closed = True


class FakeProcess:
    """A child process that prints ``chunks`` and exits with ``status``."""

    pid = 424242

    def __init__(self, chunks=(), status=0, gate=None, on_exit=None):
        self.stdout = FakeStdout(chunks, gate)
        self._status = status
        self._gate = gate
        self._on_exit = on_exit
        self.returncode = None
        self.signals = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self._gate is not None and not self._gate.is_set():
            if not self._gate.wait(timeout if timeout is not None else 10):
                raise subprocess.TimeoutExpired("fake", timeout)
        if self.returncode is None:
            self.returncode = self._status
            if self._on_exit is not None:
                self._on_exit()
        return self.returncode

    def terminate(self):
        self.signals.append("terminate")
        self._end(-15)

    def kill(self):
        self.signals.append("kill")
        self._end(-9)

    def _end(self, status):
        self._status = status
        if self._gate is not None:
            self._gate.set()


class Tool(CommandLineTool):
    """A tool with whatever install rows a test gives it."""

    def __init__(self, methods, cli="fakecli", label="Fake"):
        self.install_methods = tuple(methods)
        self.install_hint = "   # or ".join(m.shown for m in methods)
        self.cli_name = cli
        self.label = label


CURL = InstallMethod(("curl", "bash"),
                     "curl -fsSL https://example.invalid/install.sh | bash",
                     "shell")
NPM = InstallMethod(("npm",), "npm install -g fake-cli")
BREW = InstallMethod(("brew",), "brew install fake-cli")


def which_from(found):
    """A ``shutil.which`` that knows only the programs in ``found``."""
    def which(name, mode=os.F_OK | os.X_OK, path=None):
        if path is not None:
            return found.get((path, name))
        return found.get(name)
    return which


def spawner(process, seen):
    """A ``_spawn`` that records its call and returns ``process``."""
    def spawn(command, **kwargs):
        seen.append((command, kwargs))
        return process
    return spawn


def run_with(monkeypatch, process, method=CURL, found=None, **kwargs):
    """Run one install of a tool whose only row is ``method``."""
    if found is None:
        found = {"curl": "/usr/bin/curl", "bash": "/usr/bin/bash",
                 "npm": "/usr/bin/npm"}
    monkeypatch.setattr(cli_install.shutil, "which", which_from(found))
    monkeypatch.setattr(cli_install, "likely_folders", lambda platform=None: [])
    seen = []
    monkeypatch.setattr(cli_install, "_spawn", spawner(process, seen))
    tool = Tool([method])
    plan = cli_install.plan_install(tool, platform="linux")
    lines = []
    outcome = cli_install.run_install(plan, lines.append, **kwargs)
    return outcome, lines, seen, found


# ----------------------------------------------------------- one definition


@pytest.mark.parametrize("name", sorted(INSTALL_METHODS))
@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_the_hint_is_built_from_the_rows_that_run(name, platform):
    """The button and the hint cannot drift apart: one is made of the other."""
    rows = install_methods_for(name, platform)
    hint = install_hint_for(name, platform)
    assert rows, (name, platform)
    assert rows[0].shown in hint
    if platform == "win32":
        assert hint == rows[0].shown, "cmd has no # comment to hide the rest"
    else:
        assert hint == "   # or ".join(row.shown for row in rows)


@pytest.mark.parametrize("provider", providers.list_providers()
                         + [github_cli()])
def test_every_tool_on_this_system_carries_its_own_rows(provider):
    assert provider.install_methods == install_methods_for(provider.name,
                                                           sys.platform)
    assert provider.install_hint == install_hint_for(provider.name,
                                                     sys.platform)


def test_the_hints_that_existed_before_are_unchanged():
    assert install_hint_for("codex", "linux") == (
        "npm install -g @openai/codex   # or brew install codex")
    assert install_hint_for("gemini", "darwin") == (
        "npm install -g @google/gemini-cli   # or brew install gemini-cli")
    assert install_hint_for("claude", "linux") == (
        "curl -fsSL https://claude.ai/install.sh | bash")


def test_windows_and_macos_are_recorded_as_unverified():
    assert set(UNVERIFIED_PLATFORMS) == {"darwin", "win32"}


@pytest.mark.parametrize("value,family", [
    ("win32", "win32"), ("cygwin", "linux"), ("darwin", "darwin"),
    ("linux", "linux"), ("freebsd14", "linux")])
def test_the_platform_family(value, family):
    assert platform_family(value) == family


def test_an_unknown_tool_has_nothing_to_run():
    assert install_methods_for("nothing", "linux") == ()
    assert install_hint_for("nothing", "linux") == ""


def test_a_cmd_row_shows_the_shell_it_needs():
    row = install_methods_for("claude", "win32")[0]
    assert row.shown == f'cmd /c "{row.command}"'
    assert NPM.shown == NPM.command


# ------------------------------------------------------------------- the plan


def test_the_first_row_whose_programs_exist_is_chosen(monkeypatch):
    monkeypatch.setattr(cli_install.shutil, "which",
                        which_from({"brew": "/opt/homebrew/bin/brew"}))
    plan = cli_install.plan_install(Tool([NPM, BREW]))
    assert plan.method is BREW and plan.automatic
    assert plan.command == "brew install fake-cli"
    assert plan.platform == sys.platform


def test_with_no_row_that_can_run_the_plan_names_what_is_missing(monkeypatch):
    monkeypatch.setattr(cli_install.shutil, "which", which_from({}))
    tool = Tool([NPM, BREW, InstallMethod(("npm",), "npm i other")])
    plan = cli_install.plan_install(tool, platform="linux")
    assert not plan.automatic
    assert plan.needs == "npm (Node.js) or Homebrew"
    assert plan.command == tool.install_hint


def test_a_tool_without_rows_has_a_plan_without_a_method():
    plan = cli_install.plan_install(object(), platform="linux")
    assert plan.method is None and plan.command == "" and plan.needs == ""


def test_a_pipe_runs_under_pipefail_so_a_failed_download_fails(monkeypatch):
    """`curl ... | bash` exits 0 when curl fails and bash runs nothing."""
    monkeypatch.setattr(cli_install.shutil, "which",
                        which_from({"bash": "/usr/bin/bash"}))
    assert cli_install.launch_command(CURL, "linux") == [
        "/usr/bin/bash", "-o", "pipefail", "-c", CURL.command]


def test_a_program_row_is_split_and_looked_up(monkeypatch):
    monkeypatch.setattr(cli_install.shutil, "which",
                        which_from({"npm": "C:/node/npm.cmd"}))
    assert cli_install.launch_command(NPM, "win32") == [
        "C:/node/npm.cmd", "install", "-g", "fake-cli"]
    monkeypatch.setattr(cli_install.shutil, "which", which_from({}))
    assert cli_install.launch_command(NPM, "linux")[0] == "npm"
    assert cli_install.launch_command(CURL, "linux")[0] == "bash"


def test_the_conda_row_names_spacrs_own_environment(tmp_path, monkeypatch):
    """Review of 420, 2026-09-19: conda installed into whatever was active.

    With no ``--prefix`` conda picked the environment itself -- base, when
    spaCR was started without activating one -- and ``locate`` then could
    not find the ``gh`` it had just installed. The row now names spaCR's
    own environment on screen and adds ``gh`` without updating the rest.
    """
    env = tmp_path / "my envs" / "spacr"
    (env / "conda-meta").mkdir(parents=True)
    row = providers.gh_conda_row(str(env), "linux")
    assert row.needs == ("conda",)
    assert row.command == (f"conda install --yes --prefix '{env}' "
                           f"--freeze-installed gh --channel conda-forge")
    monkeypatch.setattr(cli_install.shutil, "which",
                        which_from({"conda": "/opt/conda/bin/conda"}))
    assert cli_install.launch_command(row, "linux") == [
        "/opt/conda/bin/conda", "install", "--yes", "--prefix", str(env),
        "--freeze-installed", "gh", "--channel", "conda-forge"]


def test_outside_a_conda_environment_the_conda_row_is_condas_own(tmp_path):
    row = providers.gh_conda_row(str(tmp_path), "linux")
    assert row.command == "conda install --yes gh --channel conda-forge"


def test_a_windows_environment_path_survives_the_split(monkeypatch):
    prefix = r"C:\Users\John Doe\miniconda3\envs\spacr"
    monkeypatch.setattr(providers.os.path, "isdir", lambda path: True)
    row = providers.gh_conda_row(prefix, "win32")
    assert f'--prefix "{prefix}" ' in row.command
    monkeypatch.setattr(cli_install.shutil, "which", which_from({}))
    assert cli_install.launch_command(row, "win32")[:5] == [
        "conda", "install", "--yes", "--prefix", prefix]
    plain = r"C:\envs\spacr"
    assert providers.quote_path(plain, "win32") == plain
    assert cli_install.split_command(
        f"conda install --prefix {plain} gh", "win32")[3] == plain


def test_this_system_installs_gh_into_spacrs_own_environment():
    rows = [row for row in github_cli().install_methods
            if row.needs == ("conda",)]
    assert rows == [providers.gh_conda_row(sys.prefix, sys.platform)]


def test_the_windows_row_runs_as_the_line_it_shows(monkeypatch):
    monkeypatch.setattr(cli_install.shutil, "which", which_from({}))
    row = install_methods_for("claude", "win32")[0]
    assert cli_install.launch_command(row, "win32") == row.shown
    assert cli_install.launch_command(row, "linux")[0] == "curl"


def test_the_installer_gets_a_group_of_its_own():
    assert cli_install._start_options("linux") == {"start_new_session": True}
    flags = cli_install._start_options("win32")["creationflags"]
    assert flags & 0x200 and flags & 0x8000000


# -------------------------------------------------------------------- the run


def test_a_successful_install_streams_its_output_and_finds_the_cli(
        monkeypatch):
    process = FakeProcess([b"\x1b[32mDownloading\x1b[0m 10%\r20%",
                           b"\r100%\nInstalled\n", b"\n  \n", b"last"])
    found = {"curl": "/usr/bin/curl", "bash": "/usr/bin/bash"}
    process._on_exit = lambda: found.update(fakecli="/home/u/.local/bin/x")
    outcome, lines, seen, _found = run_with(monkeypatch, process,
                                            found=found)
    assert lines == ["Downloading 10%", "20%", "100%", "Installed", "last"]
    assert outcome.ok and outcome.kind == cli_install.INSTALLED
    assert outcome.location == "/home/u/.local/bin/x"
    assert outcome.exit_status == 0
    command, kwargs = seen[0]
    assert command[-1] == CURL.command
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.STDOUT
    assert kwargs["start_new_session"] is True
    assert not Path(kwargs["cwd"]).exists(), "the scratch folder is removed"
    assert process.stdout.closed


def test_an_installer_that_exits_0_and_leaves_nothing_is_not_a_success(
        monkeypatch):
    outcome, *_ = run_with(monkeypatch, FakeProcess([b"done\n"]))
    assert outcome.kind == cli_install.NOT_FOUND and not outcome.ok


def test_an_install_can_run_with_nobody_reading_its_output(monkeypatch):
    monkeypatch.setattr(cli_install.shutil, "which",
                        which_from({"curl": "c", "bash": "b"}))
    monkeypatch.setattr(cli_install, "likely_folders",
                        lambda platform=None: [])
    monkeypatch.setattr(cli_install, "_spawn",
                        spawner(FakeProcess([b"quiet\n"], status=3), []))
    plan = cli_install.plan_install(Tool([CURL]), platform="linux")
    outcome = cli_install.run_install(plan)
    assert outcome.tail == "quiet" and outcome.kind == cli_install.FAILED


@pytest.mark.parametrize("output,status,method,kind", [
    (b"curl: (6) Could not resolve host: claude.ai\n", 6, CURL,
     cli_install.NO_NETWORK),
    (b"", 7, CURL, cli_install.NO_NETWORK),
    (b"", 22, CURL, cli_install.REFUSED),
    (b"curl: (22) The requested URL returned error: 403\n", 22, CURL,
     cli_install.REFUSED),
    (b"npm ERR! code EACCES\nnpm ERR! Error: EACCES: permission denied\n",
     243, NPM, cli_install.PERMISSION),
    (b"npm ERR! code ENOTFOUND\n", 1, NPM, cli_install.NO_NETWORK),
    (b"something else went wrong\n", 6, NPM, cli_install.FAILED),
    (b"", 3, CURL, cli_install.FAILED),
])
def test_each_failure_is_named(monkeypatch, output, status, method, kind):
    outcome, *_ = run_with(monkeypatch, FakeProcess([output], status),
                           method=method)
    assert outcome.kind == kind
    assert outcome.exit_status == status
    assert not outcome.ok


def test_a_failure_quotes_the_last_lines_only():
    lines = [f"line {i}" for i in range(10)]
    assert cli_install.quote_tail(lines) == "line 8 line 9"
    long = cli_install.quote_tail(["x" * 500])
    assert len(long) == cli_install.QUOTED_CHARS and long.endswith("…")


def test_a_plan_with_nothing_to_run_runs_nothing(monkeypatch):
    monkeypatch.setattr(cli_install.shutil, "which", which_from({}))
    plan = cli_install.plan_install(Tool([NPM]), platform="linux")
    assert cli_install.run_install(plan).kind == cli_install.NOT_AUTOMATIC


def test_an_installer_that_cannot_start_says_why(monkeypatch):
    def refuse(*_a, **_k):
        raise FileNotFoundError("no such file: bash")

    monkeypatch.setattr(cli_install.shutil, "which",
                        which_from({"curl": "c", "bash": "b"}))
    monkeypatch.setattr(cli_install, "_spawn", refuse)
    plan = cli_install.plan_install(Tool([CURL]), platform="linux")
    outcome = cli_install.run_install(plan)
    assert outcome.kind == cli_install.NOT_STARTED
    assert "bash" in outcome.tail


def test_cancel_stops_the_installer_and_says_so(monkeypatch):
    gate = threading.Event()
    process = FakeProcess([b"Downloading\n"], status=0, gate=gate)
    signalled = []

    def signal_group(pid, sig):
        signalled.append((pid, sig))
        process._end(-sig)

    monkeypatch.setattr(cli_install, "_signal_group", signal_group)
    monkeypatch.setattr(cli_install, "WATCH_INTERVAL_S", 0.01)
    stop = threading.Event()
    stop.set()
    outcome, lines, *_ = run_with(monkeypatch, process, stop=stop)
    assert outcome.kind == cli_install.CANCELLED
    assert outcome.exit_status is None
    assert signalled == [(process.pid, signal.SIGTERM)]
    assert lines == ["Downloading"]


def test_an_installer_that_never_finishes_is_stopped(monkeypatch):
    gate = threading.Event()
    process = FakeProcess([], gate=gate)
    monkeypatch.setattr(cli_install, "_signal_group",
                        lambda pid, sig: process._end(-sig))
    monkeypatch.setattr(cli_install, "WATCH_INTERVAL_S", 0.01)
    outcome, *_ = run_with(monkeypatch, process, timeout_s=0.05)
    assert outcome.kind == cli_install.TIMED_OUT


# --------------------------------------------------------------- stopping it


def test_an_exited_installer_still_has_its_group_stopped(monkeypatch):
    """A program the installer started may still hold its output open.

    Review of 420 (2026-09-19): stop_process returned early when the
    installer itself had exited, so Cancel and the time limit could not
    reach a child that was keeping the pipe open, and the install hung.
    """
    sent = []
    process = FakeProcess()
    process.returncode = 0
    monkeypatch.setattr(cli_install, "_signal_group",
                        lambda pid, sig: sent.append((pid, sig)))
    cli_install.stop_process(process, "linux")
    assert sent == [(process.pid, signal.SIGTERM)]
    assert process.signals == [], "the reaped installer is not signalled"


def test_an_exited_installer_with_no_group_left_is_left_alone(monkeypatch):
    def gone(_pid, _sig):
        raise ProcessLookupError("no such process group")

    process = FakeProcess()
    process.returncode = 0
    monkeypatch.setattr(cli_install, "_signal_group", gone)
    cli_install.stop_process(process, "linux")
    assert process.signals == []
    monkeypatch.setattr(cli_install, "_signal_group", None)
    cli_install.stop_process(process, "linux")
    assert process.signals == []


def test_an_exited_installer_on_windows_is_left_alone(monkeypatch):
    process = FakeProcess()
    process.returncode = 0
    cli_install.stop_process(process, "win32")
    assert process.signals == []


def test_what_is_left_of_a_group_is_killed(monkeypatch):
    sent = []
    process = FakeProcess()
    monkeypatch.setattr(cli_install, "_signal_group",
                        lambda pid, sig: sent.append(sig))
    cli_install._kill_group(process, "linux")
    cli_install._kill_group(process, "win32")
    assert sent == [signal.SIGKILL]

    def gone(_pid, _sig):
        raise ProcessLookupError("no such process group")

    monkeypatch.setattr(cli_install, "_signal_group", gone)
    cli_install._kill_group(process, "linux")
    assert process.signals == []


def test_cancel_kills_a_child_that_keeps_the_output_open(monkeypatch):
    """The installer has exited; something it started holds the pipe."""
    gate = threading.Event()
    process = FakeProcess([b"started\n"], status=0, gate=gate)
    process.returncode = 0
    sent = []

    def signal_group(pid, sig):
        sent.append(sig)
        if sig == signal.SIGKILL:
            gate.set()

    monkeypatch.setattr(cli_install, "_signal_group", signal_group)
    monkeypatch.setattr(cli_install, "WATCH_INTERVAL_S", 0.01)
    monkeypatch.setattr(cli_install, "STOP_GRACE_S", 0.05)
    stop = threading.Event()
    stop.set()
    outcome, lines, *_ = run_with(monkeypatch, process, stop=stop)
    assert sent == [signal.SIGTERM, signal.SIGKILL]
    assert outcome.kind == cli_install.CANCELLED
    assert lines == ["started"]


def test_an_install_that_finishes_as_cancel_is_pressed_is_installed(
        monkeypatch):
    """Cancel racing a normal exit: the install is done, so it says so."""
    gate = threading.Event()
    process = FakeProcess([b"done\n"], status=0, gate=gate)
    monkeypatch.setattr(cli_install, "_signal_group",
                        lambda pid, sig: gate.set())
    monkeypatch.setattr(cli_install, "WATCH_INTERVAL_S", 0.01)
    found = {"curl": "/usr/bin/curl", "bash": "/usr/bin/bash",
             "fakecli": "/usr/bin/fakecli"}
    stop = threading.Event()
    stop.set()
    outcome, *_ = run_with(monkeypatch, process, found=found, stop=stop)
    assert outcome.kind == cli_install.INSTALLED
    assert outcome.location == "/usr/bin/fakecli"


def test_a_group_that_ignores_sigterm_is_killed(monkeypatch):
    sent = []
    process = FakeProcess(gate=threading.Event())
    monkeypatch.setattr(cli_install, "_signal_group",
                        lambda pid, sig: sent.append(sig))
    monkeypatch.setattr(cli_install, "STOP_GRACE_S", 0.01)
    cli_install.stop_process(process, "linux")
    assert sent == [signal.SIGTERM, signal.SIGKILL]


def test_without_process_groups_the_process_itself_is_stopped(monkeypatch):
    process = FakeProcess(gate=threading.Event())
    monkeypatch.setattr(cli_install, "_signal_group", None)
    monkeypatch.setattr(cli_install, "STOP_GRACE_S", 0.01)
    cli_install.stop_process(process, "linux")
    assert process.signals == ["terminate"]
    stubborn = FakeProcess(gate=threading.Event())
    stubborn.terminate = lambda: stubborn.signals.append("terminate")
    cli_install.stop_process(stubborn, "linux")
    assert stubborn.signals == ["terminate", "kill"]


def test_windows_ends_the_whole_tree(monkeypatch):
    seen = []
    monkeypatch.setattr(cli_install, "_spawn",
                        lambda argv, **kw: seen.append(argv))
    process = FakeProcess(gate=threading.Event())
    cli_install.stop_process(process, "win32")
    assert seen == [["taskkill", "/T", "/F", "/PID", str(process.pid)]]

    def refuse(*_a, **_k):
        raise OSError("no taskkill")

    monkeypatch.setattr(cli_install, "_spawn", refuse)
    cli_install.stop_process(process, "win32")
    assert process.signals == ["kill"]


# -------------------------------------------------------------- finding it


def test_a_cli_on_path_is_found_where_it_is(monkeypatch):
    monkeypatch.setattr(cli_install.shutil, "which",
                        which_from({"fakecli": "/usr/bin/fakecli"}))
    assert cli_install.locate(Tool([])) == "/usr/bin/fakecli"
    assert cli_install.locate(object()) == ""


def test_a_cli_off_path_is_found_in_a_folder_installers_use(monkeypatch):
    monkeypatch.setattr(cli_install, "likely_folders",
                        lambda platform=None: ["/a", "/b"])
    monkeypatch.setattr(cli_install.shutil, "which",
                        which_from({("/b", "fakecli"): "/b/fakecli"}))
    assert cli_install.locate(Tool([])) == "/b/fakecli"
    monkeypatch.setattr(cli_install.shutil, "which", which_from({}))
    assert cli_install.locate(Tool([])) == ""


def test_the_folders_looked_in_on_linux(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CONDA_PREFIX", "/envs/spacr")
    monkeypatch.setattr(cli_install.shutil, "which", which_from({}))
    folders = cli_install.likely_folders("linux")
    assert folders[0] == str(tmp_path / ".local" / "bin")
    assert "/envs/spacr/bin" in folders
    assert "/opt/homebrew/bin" in folders
    monkeypatch.setenv("CONDA_PREFIX", sys.prefix)
    folders = cli_install.likely_folders("linux")
    assert folders.count(str(Path(sys.prefix) / "bin")) == 1


def test_the_folders_looked_in_on_windows(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(cli_install.shutil, "which",
                        which_from({"npm": "npm"}))
    monkeypatch.setattr(
        cli_install, "_query",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, "C:/npm\n", ""))
    folders = cli_install.likely_folders("win32")
    assert str(tmp_path / "local" / "Microsoft/WinGet/Links") in folders
    assert folders[-1] == "C:/npm"


@pytest.mark.parametrize("answer,expected", [
    (subprocess.CompletedProcess([], 0, "/usr/local\n", ""),
     str(Path("/usr/local") / "bin")),
    (subprocess.CompletedProcess([], 1, "/usr/local\n", ""), ""),
    (subprocess.CompletedProcess([], 0, "", ""), ""),
    (subprocess.CompletedProcess([], 0, None, ""), ""),
])
def test_npm_is_asked_where_its_programs_go(monkeypatch, answer, expected):
    monkeypatch.setattr(cli_install.shutil, "which",
                        which_from({"npm": "/usr/bin/npm"}))
    monkeypatch.setattr(cli_install, "_query", lambda *a, **k: answer)
    assert cli_install._npm_folder("linux") == expected


def test_an_npm_that_does_not_answer_adds_no_folder(monkeypatch):
    def hang(*_a, **_k):
        raise subprocess.TimeoutExpired("npm", 20)

    monkeypatch.setattr(cli_install.shutil, "which",
                        which_from({"npm": "/usr/bin/npm"}))
    monkeypatch.setattr(cli_install, "_query", hang)
    assert cli_install._npm_folder("linux") == ""
    monkeypatch.setattr(cli_install.shutil, "which", which_from({}))
    assert cli_install._npm_folder("linux") == ""


def test_the_folder_is_put_on_path_once(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", "/usr/bin")
    cli = tmp_path / "bin" / "claude"
    assert cli_install.put_on_path(str(cli)) is True
    assert os.environ["PATH"] == f"{tmp_path / 'bin'}{os.pathsep}/usr/bin"
    assert cli_install.put_on_path(str(cli)) is False
    assert cli_install.put_on_path("") is False
    monkeypatch.setenv("PATH", "")
    assert cli_install.put_on_path(str(cli)) is True
    assert os.environ["PATH"] == str(tmp_path / "bin")


# --------------------------------------------------------- signed in or not


def test_a_tool_is_asked_whether_it_is_signed_in(monkeypatch):
    seen = []
    monkeypatch.setattr(providers, "_run_quietly",
                        lambda argv, timeout: seen.append(argv) or 0)
    monkeypatch.setattr(providers.shutil, "which", lambda name: None)
    tool = github_cli()
    assert tool.check_signed_in() is True
    assert seen == [["gh", "auth", "token"]]
    monkeypatch.setattr(providers, "_run_quietly", lambda argv, timeout: 1)
    assert tool.check_signed_in() is False
    assert CommandLineTool().check_signed_in() is None


def test_a_status_command_that_cannot_run_means_signed_out(monkeypatch):
    def refuse(argv, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(providers, "_run_quietly", refuse)
    assert github_cli().check_signed_in() is False


def test_the_github_cli_has_the_providers_three_states(monkeypatch):
    monkeypatch.setattr(providers.shutil, "which", lambda name: None)
    assert not github_cli().is_installed()
    assert not github_cli().is_logged_in()
    monkeypatch.setattr(providers.shutil, "which", lambda name: "/bin/gh")
    monkeypatch.setattr(providers, "_run_quietly", lambda argv, timeout: 0)
    assert github_cli().is_logged_in()
    assert isinstance(github_cli(), GitHubCli)


def test_the_quiet_run_really_runs_and_discards_output():
    status = providers._run_quietly(
        [sys.executable, "-c", "print('secret'); raise SystemExit(3)"], 30)
    assert status == 3


def test_claude_says_it_is_signed_in_by_its_own_status_command():
    """Measured 2026-09-19: `claude auth status` exits 1 signed out, 0 in."""
    assert providers.get_provider("claude").status_command == (
        "claude", "auth", "status")


# ------------------------------------------------------ with a real process


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX script")
def test_a_real_installer_process_streams_and_is_found(monkeypatch,
                                                       tmp_path):
    """A shell script stands in for npm: real pipes, real exit, real PATH."""
    monkeypatch.setattr(cli_install, "_spawn", subprocess.Popen)
    monkeypatch.setattr(cli_install, "_query", subprocess.run)
    home = tmp_path / "home"
    target = home / ".local" / "bin"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_npm = bin_dir / "npm"
    fake_npm.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = prefix ]; then echo /nonexistent; exit 0; fi\n"
        "printf 'added 1 package\\r'\n"
        f"mkdir -p '{target}'\n"
        f"printf '#!/bin/sh\\nexit 0\\n' > '{target}/fakecli'\n"
        f"chmod +x '{target}/fakecli'\n"
        "echo done\n")
    fake_npm.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    plan = cli_install.plan_install(Tool([NPM]), platform=sys.platform)
    lines = []
    outcome = cli_install.run_install(plan, lines.append)
    assert lines == ["added 1 package", "done"]
    assert outcome.ok, outcome
    assert outcome.location == str(target / "fakecli")


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX groups")
def test_cancel_ends_every_process_the_installer_started(monkeypatch,
                                                         tmp_path):
    """`curl | bash` is two processes; Cancel must not leave one behind."""
    monkeypatch.setattr(cli_install, "_spawn", subprocess.Popen)
    monkeypatch.setattr(cli_install, "_signal_group", os.killpg)
    monkeypatch.setattr(cli_install, "likely_folders",
                        lambda platform=None: [])
    pid_file = tmp_path / "child.pid"
    row = InstallMethod(("bash",),
                        f"sleep 60 & echo $! > '{pid_file}'; echo started; "
                        "wait", "shell")
    plan = cli_install.plan_install(Tool([row]), platform=sys.platform)
    stop = threading.Event()
    lines = []

    def on_output(line):
        lines.append(line)
        stop.set()

    began = time.monotonic()
    outcome = cli_install.run_install(plan, on_output, stop)
    assert outcome.kind == cli_install.CANCELLED
    assert time.monotonic() - began < 10
    child = int(pid_file.read_text())
    for _ in range(50):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(child, signal.SIGKILL)
        pytest.fail("the installer's child outlived Cancel")


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX groups")
def test_cancel_ends_a_child_left_holding_the_output(monkeypatch, tmp_path):
    """The installer exits at once; the `sleep` it left keeps the pipe open.

    Before the review fix of 2026-09-19 this took the full sleep: Cancel
    found the installer gone and did nothing, and `read1` waited for the
    `sleep` to close its end of the pipe.
    """
    monkeypatch.setattr(cli_install, "_spawn", subprocess.Popen)
    monkeypatch.setattr(cli_install, "_signal_group", os.killpg)
    monkeypatch.setattr(cli_install, "likely_folders",
                        lambda platform=None: [])
    pid_file = tmp_path / "child.pid"
    row = InstallMethod(("bash",),
                        f"sleep 30 & echo $! > '{pid_file}'; echo started",
                        "shell")
    plan = cli_install.plan_install(Tool([row]), platform=sys.platform)
    stop = threading.Event()

    def on_output(_line):
        time.sleep(0.5)
        stop.set()

    began = time.monotonic()
    outcome = cli_install.run_install(plan, on_output, stop)
    assert time.monotonic() - began < 10
    assert outcome.kind == cli_install.CANCELLED
    child = int(pid_file.read_text())
    for _ in range(50):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(child, signal.SIGKILL)
        pytest.fail("the child holding the output outlived Cancel")
