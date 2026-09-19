"""Run a vendor command-line tool's own installer, and say how it went.

Qt-free. :mod:`spacr.qt.widgets.cli_setup_panel` calls :func:`run_install` on
a worker thread and shows what it reports; nothing here touches a widget.

THE COMMANDS ARE NOT HERE. Each tool's ``install_methods`` in
:mod:`spacr.qt.ai.providers` are the rows its ``install_hint`` is built from,
and :func:`plan_install` picks the first of those rows whose programs are on
``PATH``. The command the Install button runs and the command the screen
shows are therefore one definition.

What a run reports is a kind (:data:`INSTALLED`, :data:`NO_NETWORK`, ...)
rather than a sentence, so the screen can say something the user can act on,
in the user's language, for each way an install goes wrong.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Union

from .providers import InstallMethod, platform_family

#: The CLI is installed and was found afterwards.
INSTALLED = "installed"
#: The user pressed Cancel and the installer was stopped.
CANCELLED = "cancelled"
#: The installer ran past :data:`INSTALL_TIMEOUT_S` and was stopped.
TIMED_OUT = "timed out"
#: The download could not reach its server.
NO_NETWORK = "no network"
#: The server answered and refused the download.
REFUSED = "refused"
#: The installer was not allowed to write where it installs.
PERMISSION = "permission"
#: The installer exited 0 and the CLI still cannot be found.
NOT_FOUND = "not found"
#: The installer could not be started at all.
NOT_STARTED = "not started"
#: No row of the tool's install methods can run on this computer.
NOT_AUTOMATIC = "not automatic"
#: The installer failed for a reason none of the other kinds names.
FAILED = "failed"

#: Seconds an installer may run before it is stopped as hung.
INSTALL_TIMEOUT_S = 20 * 60

#: Seconds a stopped installer is given to exit before it is killed.
STOP_GRACE_S = 3

#: Seconds between two checks for Cancel or the time limit.
WATCH_INTERVAL_S = 0.2

#: How many of the installer's last output lines are kept for the verdict.
KEPT_LINES = 40

#: How many of those lines a failure message quotes.
QUOTED_LINES = 2

#: The longest quotation of an installer's output, in characters.
QUOTED_CHARS = 300

#: What a missing prerequisite is called when a message names it.
REQUIREMENT_NAMES = {
    "npm": "npm (Node.js)",
    "brew": "Homebrew",
    "conda": "conda",
    "winget": "winget",
    "curl": "curl",
    "bash": "bash",
}

#: Output that means the installer could not reach the network.
NETWORK_TEXT = (
    "could not resolve host", "enotfound", "eai_again", "etimedout",
    "econnrefused", "econnreset", "enetunreach", "network is unreachable",
    "temporary failure in name resolution", "name or service not known",
    "failed to connect", "connection timed out", "connection refused",
    "connection failed", "could not connect", "getaddrinfo",
)

#: Output that means a server answered and refused the download.
REFUSED_TEXT = (
    "the requested url returned error", "403 forbidden", "404 not found",
    "401 unauthorized", "e403", "e404", "http error 4", "http error 5",
)

#: Output that means the installer could not write where it installs.
PERMISSION_TEXT = (
    "eacces", "eperm", "permission denied", "access is denied",
    "operation not permitted", "not writable",
)

#: curl's exit statuses for a network that could not be reached.
CURL_NETWORK_EXITS = (5, 6, 7, 28, 35, 56)

#: curl's exit status for an HTTP error answer under ``--fail``.
CURL_REFUSED_EXIT = 22

_ESCAPES = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]"
                      r"|\x1b\][^\x07]*\x07"
                      r"|[\x00-\x08\x0b-\x1f\x7f]")
_LINE_BREAKS = re.compile(rb"[\r\n]")

_spawn = subprocess.Popen
_query = subprocess.run
_signal_group = getattr(os, "killpg", None)


@dataclass(frozen=True)
class InstallPlan:
    """What pressing Install would run for one tool on this computer.

    :ivar tool: the tool, a :class:`spacr.qt.ai.providers.CommandLineTool`.
    :ivar method: the row that would run, or ``None`` when none of the
        tool's rows can run here.
    :ivar platform: the ``sys.platform`` value the plan was made for.
    """

    tool: Any
    method: Optional[InstallMethod]
    platform: str

    @property
    def automatic(self) -> bool:
        """Whether a row was found that spaCR can run."""
        return self.method is not None

    @property
    def command(self) -> str:
        """The command to show: the row that would run, else the hint."""
        if self.method is not None:
            return self.method.shown
        return str(getattr(self.tool, "install_hint", "") or "")

    @property
    def needs(self) -> str:
        """The programs that would make an install possible, for a message.

        :returns: the first program each row needs, named for a reader and
            joined with " or ", such as ``"npm (Node.js) or Homebrew"``.
        """
        names: List[str] = []
        for method in getattr(self.tool, "install_methods", ()) or ():
            for need in method.needs[:1]:
                name = REQUIREMENT_NAMES.get(need, need)
                if name not in names:
                    names.append(name)
        return " or ".join(names)


@dataclass(frozen=True)
class InstallOutcome:
    """How one run of an installer ended.

    :ivar kind: one of the module's kinds, such as :data:`INSTALLED`.
    :ivar exit_status: the installer's exit status, or ``None`` when it did
        not run or was stopped before it exited on its own.
    :ivar tail: the installer's last output, quoted for a failure message.
    :ivar location: where the CLI was found afterwards, or ``""``.
    """

    kind: str
    exit_status: Optional[int] = None
    tail: str = ""
    location: str = ""

    @property
    def ok(self) -> bool:
        """Whether the tool is now installed and findable."""
        return self.kind == INSTALLED


def plan_install(tool: Any, platform: Optional[str] = None) -> InstallPlan:
    """Choose the row of ``tool``'s install methods that can run here.

    :param tool: a :class:`spacr.qt.ai.providers.CommandLineTool`, or any
        object with the same attributes; one with no ``install_methods``
        gets a plan with no method.
    :param platform: a ``sys.platform`` value; the running one by default.
    :returns: the plan, whose ``method`` is the first row whose ``needs``
        are all on ``PATH``, or ``None`` when there is no such row.
    """
    platform = platform or sys.platform
    for method in getattr(tool, "install_methods", ()) or ():
        if all(shutil.which(need) for need in method.needs):
            return InstallPlan(tool, method, platform)
    return InstallPlan(tool, None, platform)


def launch_command(method: InstallMethod,
                   platform: str) -> Union[List[str], str]:
    """The command line that starts ``method``'s installer.

    :param method: the row to run.
    :param platform: a ``sys.platform`` value.
    :returns: for ``"cmd"`` on Windows, the shown line itself as one string,
        which Windows hands to ``cmd`` unchanged; for ``"shell"``, ``bash -o
        pipefail -c`` and the command, so a failed download in ``curl ... |
        bash`` is a failed install rather than an empty script run
        successfully; otherwise the command split into a program, looked up
        on ``PATH``, and its arguments.
    """
    if method.runner == "cmd" and platform_family(platform) == "win32":
        return method.shown
    if method.runner == "shell":
        return [shutil.which("bash") or "bash", "-o", "pipefail", "-c",
                method.command]
    argv = shlex.split(method.command)
    argv[0] = shutil.which(argv[0]) or argv[0]
    return argv


def _start_options(platform: str) -> dict:
    """Keyword arguments that put the installer in a group of its own.

    :param platform: a ``sys.platform`` value.
    :returns: a new session on POSIX, so Cancel can stop the whole
        ``curl | bash`` pipeline and not only its first process; a new
        process group with no console window on Windows.
    """
    if platform_family(platform) == "win32":
        flags = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
                 | getattr(subprocess, "CREATE_NO_WINDOW", 0x8000000))
        return {"creationflags": flags}
    return {"start_new_session": True}


def clean_line(raw: bytes) -> str:
    """One line of installer output, decoded and without terminal codes.

    :param raw: the bytes between two line breaks.
    :returns: the text with colour codes and control characters removed and
        the ends stripped.
    """
    text = raw.decode("utf-8", "replace")
    return _ESCAPES.sub("", text).strip()


def stop_process(proc: Any, platform: str) -> None:
    """End an installer and everything it started.

    :param proc: the running installer.
    :param platform: a ``sys.platform`` value.
    """
    if proc.poll() is not None:
        return
    if platform_family(platform) == "win32":
        try:
            _spawn(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
        except OSError:
            proc.kill()
        return
    try:
        _signal_group(proc.pid, signal.SIGTERM)
    except (OSError, TypeError):
        proc.terminate()
    try:
        proc.wait(timeout=STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        try:
            _signal_group(proc.pid, signal.SIGKILL)
        except (OSError, TypeError):
            proc.kill()


def _watch(proc: Any, stop: Optional[threading.Event], deadline: float,
           finished: threading.Event, fired: List[str], platform: str) -> None:
    """Stop ``proc`` when Cancel is pressed or the time limit passes.

    Runs on a thread of its own, because the thread running the install is
    blocked reading the installer's output and cannot notice either.

    :param proc: the running installer.
    :param stop: set by Cancel, or ``None``.
    :param deadline: the ``time.monotonic()`` value past which it is hung.
    :param finished: set once the installer has exited, to end this watch.
    :param fired: receives :data:`CANCELLED` or :data:`TIMED_OUT` when this
        watch is what stopped the installer.
    :param platform: a ``sys.platform`` value.
    """
    while not finished.wait(WATCH_INTERVAL_S):
        if stop is not None and stop.is_set():
            fired.append(CANCELLED)
        elif time.monotonic() >= deadline:
            fired.append(TIMED_OUT)
        else:
            continue
        stop_process(proc, platform)
        return


def classify(exit_status: int, lines: Sequence[str],
             method: InstallMethod) -> str:
    """Name what went wrong with an installer that exited non-zero.

    :param exit_status: its exit status.
    :param lines: its last output lines.
    :param method: the row that ran, whose command says whether a curl exit
        status can be read as curl's.
    :returns: :data:`NO_NETWORK`, :data:`REFUSED`, :data:`PERMISSION` or
        :data:`FAILED`.
    """
    text = "\n".join(lines).lower()
    if any(marker in text for marker in NETWORK_TEXT):
        return NO_NETWORK
    if any(marker in text for marker in REFUSED_TEXT):
        return REFUSED
    if any(marker in text for marker in PERMISSION_TEXT):
        return PERMISSION
    if method.command.startswith("curl "):
        if exit_status in CURL_NETWORK_EXITS:
            return NO_NETWORK
        if exit_status == CURL_REFUSED_EXIT:
            return REFUSED
    return FAILED


def quote_tail(lines: Sequence[str]) -> str:
    """The installer's last lines, short enough for a message.

    :param lines: its output lines, oldest first.
    :returns: the last :data:`QUOTED_LINES` joined by a space and cut to
        :data:`QUOTED_CHARS`.
    """
    text = " ".join(list(lines)[-QUOTED_LINES:])
    if len(text) > QUOTED_CHARS:
        text = text[:QUOTED_CHARS - 1].rstrip() + "…"
    return text


def run_install(plan: InstallPlan,
                on_output: Optional[Callable[[str], None]] = None,
                stop: Optional[threading.Event] = None, *,
                timeout_s: float = INSTALL_TIMEOUT_S) -> InstallOutcome:
    """Run ``plan``'s installer to the end and say how it went.

    Blocks until the installer exits, so run it off the GUI thread. The
    installer gets no input -- a prompt reads end-of-file rather than
    waiting forever -- and runs in a scratch folder that is removed
    afterwards, because the Windows command downloads ``install.cmd`` into
    the folder it runs in.

    :param plan: from :func:`plan_install`.
    :param on_output: called with each non-blank line of output, on the
        calling thread, as it arrives.
    :param stop: set it to stop the installer.
    :param timeout_s: seconds after which the installer is stopped as hung.
    :returns: the outcome; :data:`INSTALLED` only when the CLI is found
        afterwards, with ``location`` saying where.
    """
    if plan.method is None:
        return InstallOutcome(NOT_AUTOMATIC)
    command = launch_command(plan.method, plan.platform)
    workdir = tempfile.mkdtemp(prefix="spacr-install-")
    kept: deque = deque(maxlen=KEPT_LINES)
    fired: List[str] = []
    try:
        try:
            proc = _spawn(command, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          cwd=workdir, env=os.environ.copy(),
                          **_start_options(plan.platform))
        except (OSError, ValueError) as exc:
            return InstallOutcome(NOT_STARTED, tail=str(exc))
        finished = threading.Event()
        watcher = threading.Thread(
            target=_watch, name="spacr-install-watch", daemon=True,
            args=(proc, stop, time.monotonic() + timeout_s, finished, fired,
                  plan.platform))
        watcher.start()
        pending = b""
        try:
            while True:
                chunk = proc.stdout.read1(4096)
                if not chunk:
                    break
                pending += chunk
                *complete, pending = _LINE_BREAKS.split(pending)
                for raw in complete:
                    _keep(raw, kept, on_output)
            _keep(pending, kept, on_output)
            exit_status = proc.wait()
        finally:
            finished.set()
            watcher.join(timeout=STOP_GRACE_S + 1)
            proc.stdout.close()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    tail = quote_tail(kept)
    if fired:
        return InstallOutcome(fired[0], None, tail)
    if exit_status != 0:
        return InstallOutcome(classify(exit_status, kept, plan.method),
                              exit_status, tail)
    location = locate(plan.tool, plan.platform)
    return InstallOutcome(INSTALLED if location else NOT_FOUND, 0, tail,
                          location)


def _keep(raw: bytes, kept: deque,
          on_output: Optional[Callable[[str], None]]) -> None:
    """Record one raw output line and pass it on, unless it is blank.

    :param raw: the bytes of the line.
    :param kept: the recent lines, which this appends to.
    :param on_output: the caller's line callback, or ``None``.
    """
    line = clean_line(raw)
    if not line:
        return
    kept.append(line)
    if on_output is not None:
        on_output(line)


def likely_folders(platform: Optional[str] = None) -> List[str]:
    """Folders installers put a CLI in that may not be on ``PATH``.

    A program started from a desktop icon often has a shorter ``PATH`` than a
    terminal, and Claude's own installer puts ``claude`` in
    ``~/.local/bin``, which is exactly the folder such a ``PATH`` leaves
    out.

    :param platform: a ``sys.platform`` value; the running one by default.
    :returns: candidate folders, most likely first, without duplicates.
    """
    platform = platform or sys.platform
    home = Path.home()
    prefixes = [Path(sys.prefix)]
    if os.environ.get("CONDA_PREFIX"):
        prefixes.append(Path(os.environ["CONDA_PREFIX"]))
    if platform_family(platform) == "win32":
        local = Path(os.environ.get("LOCALAPPDATA", home / "AppData/Local"))
        roaming = Path(os.environ.get("APPDATA", home / "AppData/Roaming"))
        programs = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        folders = [home / ".local" / "bin", local / "Microsoft/WinGet/Links",
                   programs / "GitHub CLI", roaming / "npm"]
        for prefix in prefixes:
            folders += [prefix / "Scripts", prefix / "Library" / "bin"]
    else:
        folders = [home / ".local" / "bin", home / ".claude" / "local",
                   Path("/opt/homebrew/bin"), Path("/usr/local/bin"),
                   Path("/home/linuxbrew/.linuxbrew/bin"),
                   home / ".linuxbrew" / "bin"]
        folders += [prefix / "bin" for prefix in prefixes]
    npm = _npm_folder(platform)
    if npm:
        folders.append(Path(npm))
    seen: List[str] = []
    for folder in folders:
        text = str(folder)
        if text not in seen:
            seen.append(text)
    return seen


def _npm_folder(platform: str) -> str:
    """Where ``npm install -g`` puts programs, or ``""``.

    :param platform: a ``sys.platform`` value.
    :returns: the global prefix's ``bin`` folder on POSIX and the prefix
        itself on Windows, as ``npm prefix -g`` reports it; ``""`` when
        there is no npm or it does not answer.
    """
    npm = shutil.which("npm")
    if not npm:
        return ""
    try:
        answer = _query([npm, "prefix", "-g"], stdin=subprocess.DEVNULL,
                        capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return ""
    prefix = (answer.stdout or "").strip()
    if answer.returncode != 0 or not prefix:
        return ""
    if platform_family(platform) == "win32":
        return prefix
    return str(Path(prefix) / "bin")


def locate(tool: Any, platform: Optional[str] = None) -> str:
    """Find ``tool``'s CLI on ``PATH`` or in a folder installers use.

    :param tool: a :class:`spacr.qt.ai.providers.CommandLineTool`.
    :param platform: a ``sys.platform`` value; the running one by default.
    :returns: the executable's path, or ``""`` when it is nowhere.
    """
    name = str(getattr(tool, "cli_name", "") or "")
    if not name:
        return ""
    found = shutil.which(name)
    if found:
        return found
    for folder in likely_folders(platform):
        found = shutil.which(name, path=folder)
        if found:
            return found
    return ""


def put_on_path(location: str) -> bool:
    """Make the folder holding ``location`` part of this process's ``PATH``.

    So a CLI installed into a folder the desktop's ``PATH`` leaves out works
    in spaCR at once, without a restart. Changes this process's environment,
    so call it on the GUI thread.

    :param location: the path of an executable, from :func:`locate`.
    :returns: ``True`` when ``PATH`` was changed.
    """
    if not location:
        return False
    folder = os.path.dirname(os.path.abspath(location))
    current = os.environ.get("PATH", "")
    parts = [os.path.normcase(os.path.abspath(p))
             for p in current.split(os.pathsep) if p]
    if os.path.normcase(folder) in parts:
        return False
    os.environ["PATH"] = folder + (os.pathsep + current if current else "")
    return True

