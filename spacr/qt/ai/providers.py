"""
AI chat providers through vendor command-line tools or spaCR's hosted relay. The vendor
providers invoke the user's installed CLI; authentication and account usage follow that
CLI's configuration.

* Anthropic Claude → the `claude` CLI ("Claude Code")
* OpenAI ChatGPT   → the `codex`  CLI
* Google Gemini    → the `gemini` CLI

A fourth, :class:`HostedProvider`, needs no CLI and no account: it asks spaCR's relay
for hosted advice. It is the default while network access is allowed unless a configured
vendor CLI is explicitly preferred (:func:`default_provider`). Configuration checks do
not guarantee that a request will succeed or that quota remains.

Each vendor-CLI provider::

    is_installed()   — is the CLI on PATH?
    is_logged_in()   — best-effort check; falls back to "assume yes if
                       installed" (the actual auth error surfaces on
                       the first stream chunk).
    stream_chat()    — spawn the CLI subprocess, yield stdout chunks.

For ordinary vendor-CLI requests, conversation history is flattened into one prompt with
role labels. An approved Claude session resume sends the newest message with the saved
session id instead. These vendor-CLI prompts have no size limit enforced by this module.

How each CLI is installed is data, not prose: :data:`INSTALL_METHODS` holds
one row per way of installing it on each operating system, and both the
``install_hint`` a screen shows and the command
:mod:`spacr.qt.ai.cli_install` runs are read from those rows. The GitHub CLI
is described the same way by :class:`GitHubCli`, which shares the
:class:`CommandLineTool` interface without being a chat provider.
"""
from __future__ import annotations

import sys as _sys

import json
import os
import shlex
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple


class InstallMethod(NamedTuple):
    """One way of installing a command-line tool on one operating system.

    :ivar needs: executables that must already be on ``PATH`` for this way to
        work, such as ``("npm",)``.
    :ivar command: the command as a user would type it.
    :ivar runner: how :mod:`spacr.qt.ai.cli_install` starts it: ``"exec"``
        runs the command's first word as a program with the rest as its
        arguments, ``"shell"`` hands the command to ``bash -o pipefail -c``
        (for a pipe such as ``curl ... | bash``), and ``"cmd"`` runs it as
        ``cmd /c "<command>"`` on Windows.
    """

    needs: Tuple[str, ...]
    command: str
    runner: str = "exec"

    @property
    def shown(self) -> str:
        """The command as it is shown on screen and copied.

        :returns: ``cmd /c "<command>"`` for the ``"cmd"`` runner, because
            that line runs whole in both Windows shells; otherwise the
            command itself.
        """
        if self.runner == "cmd":
            return f'cmd /c "{self.command}"'
        return self.command


_CLAUDE_POSIX = (
    InstallMethod(("curl", "bash"),
                  "curl -fsSL https://claude.ai/install.sh | bash", "shell"),
)
_CLAUDE_WINDOWS_PATH_SETUP = (
    "try{if(-not([IO.File]::Exists([IO.Path]::Combine("
    "[Environment]::GetFolderPath('UserProfile'),'.local\\bin\\claude.exe'))))"
    "{throw('Claude_native_executable_missing')}"
    "if(-not([Environment]::ExpandEnvironmentVariables([string]"
    "[Environment]::GetEnvironmentVariable('Path','User')).ToLowerInvariant()"
    ".Split(';').Contains([Environment]::GetFolderPath('UserProfile')"
    ".ToLowerInvariant()+'\\.local\\bin')))"
    "{[Environment]::SetEnvironmentVariable('Path',([string]"
    "[Environment]::GetEnvironmentVariable('Path','User'))+';'"
    "+[Environment]::GetFolderPath('UserProfile')+'\\.local\\bin','User')}}"
    "catch{[Console]::Error.WriteLine('Claude_PATH_registration_failed');exit(1)}"
)
_CLAUDE_WINDOWS = (
    InstallMethod(("curl",),
                  "curl -fsSL https://claude.ai/install.cmd -o install.cmd"
                  " && install.cmd && del install.cmd"
                  " && powershell.exe -NoProfile -NonInteractive -Command "
                  + _CLAUDE_WINDOWS_PATH_SETUP.replace("(", "^(").replace(")", "^)"), "cmd"),
)
_CODEX_NPM = InstallMethod(("npm",), "npm install -g @openai/codex")
_GEMINI_NPM = InstallMethod(("npm",), "npm install -g @google/gemini-cli")


def quote_path(path: str, platform: str) -> str:
    """A path as it is typed in a command on ``platform``.

    :param path: the path.
    :param platform: a ``sys.platform`` value.
    :returns: on Windows the path in double quotes when it holds a space,
        else unchanged; elsewhere quoted for a POSIX shell.
    """
    path = str(path)
    if str(platform).startswith("win"):
        return f'"{path}"' if any(ch.isspace() for ch in path) else path
    return shlex.quote(path)


def gh_conda_row(prefix: str, platform: str) -> InstallMethod:
    """The row that installs the GitHub CLI with conda.

    :param prefix: spaCR's own environment, ``sys.prefix``.
    :param platform: a ``sys.platform`` value, for quoting ``prefix``.
    :returns: when ``prefix`` is a conda environment, a command that names
        it with ``--prefix`` and adds ``gh`` with ``--freeze-installed``,
        so the command on screen says which environment changes and conda
        does not update the packages spaCR runs on; otherwise conda's plain
        command, which installs into whichever environment conda treats as
        active.
    """
    if os.path.isdir(os.path.join(str(prefix), "conda-meta")):
        return InstallMethod(
            ("conda",),
            f"conda install --yes --prefix {quote_path(prefix, platform)}"
            f" --freeze-installed gh --channel conda-forge")
    return InstallMethod(("conda",),
                         "conda install --yes gh --channel conda-forge")


_GH_CONDA = gh_conda_row(_sys.prefix, _sys.platform)
_GH_POSIX = (InstallMethod(("brew",), "brew install gh"), _GH_CONDA)
_CODEX_POSIX = (_CODEX_NPM, InstallMethod(("brew",), "brew install codex"))
_GEMINI_POSIX = (_GEMINI_NPM,
                 InstallMethod(("brew",), "brew install gemini-cli"))

#: Every way spaCR knows to install each tool, per operating system, in the
#: order they are tried: the first row whose ``needs`` are all on ``PATH`` is
#: the one that runs.
#:
#: Keyed by tool name, then by ``"linux"``, ``"darwin"`` or ``"win32"`` (see
#: :func:`platform_family`). Claude's rows are Anthropic's documented native
#: installers; the npm and Homebrew rows are the vendors' package
#: names; the GitHub CLI rows are from cli.github.com. The rows for
#: :data:`UNVERIFIED_PLATFORMS` have not been run on those systems.
INSTALL_METHODS: Dict[str, Dict[str, Tuple[InstallMethod, ...]]] = {
    "claude": {"linux": _CLAUDE_POSIX, "darwin": _CLAUDE_POSIX,
               "win32": _CLAUDE_WINDOWS},
    "codex": {"linux": _CODEX_POSIX, "darwin": _CODEX_POSIX,
              "win32": (_CODEX_NPM,)},
    "gemini": {"linux": _GEMINI_POSIX, "darwin": _GEMINI_POSIX,
               "win32": (_GEMINI_NPM,)},
    "gh": {"linux": _GH_POSIX, "darwin": _GH_POSIX,
           "win32": (InstallMethod(("winget",),
                                   "winget install --id GitHub.cli --exact"
                                   " --accept-source-agreements"
                                   " --accept-package-agreements"),
                     _GH_CONDA)},
}

#: Operating systems whose :data:`INSTALL_METHODS` rows were copied from the
#: vendors' documentation and have not been run on that system.
UNVERIFIED_PLATFORMS: Tuple[str, ...] = ("darwin", "win32")


def platform_family(platform: str) -> str:
    """Name the operating system :data:`INSTALL_METHODS` is keyed by.

    :param platform: a ``sys.platform`` value.
    :returns: ``"win32"`` for any Windows value, ``"darwin"`` for macOS, and
        ``"linux"`` for everything else.
    """
    platform = str(platform)
    if platform.startswith("win"):
        return "win32"
    if platform == "darwin":
        return "darwin"
    return "linux"


def install_methods_for(name: str, platform: str) -> Tuple[InstallMethod, ...]:
    """The ways of installing tool ``name`` on ``platform``, in order.

    :param name: a key of :data:`INSTALL_METHODS`, such as ``"claude"``.
    :param platform: a ``sys.platform`` value.
    :returns: the rows, or ``()`` for a tool spaCR cannot install.
    """
    return INSTALL_METHODS.get(str(name), {}).get(platform_family(platform),
                                                  ())


def install_hint_for(name: str, platform: str) -> str:
    """The one-line install hint for tool ``name`` on ``platform``.

    Built from the same rows the Install button runs, so the two cannot
    drift apart. On macOS and Linux the alternatives are joined by three
    spaces and then ``# or``, which a POSIX shell reads as a comment; on
    Windows only the first row is shown, because ``cmd`` has no such
    comment and would hand the words after ``#`` to the installer as
    arguments.

    :param name: a key of :data:`INSTALL_METHODS`.
    :param platform: a ``sys.platform`` value.
    :returns: the hint, or ``""`` when the tool has no rows.
    """
    methods = install_methods_for(name, platform)
    if platform_family(platform) == "win32":
        methods = methods[:1]
    return "   # or ".join(method.shown for method in methods)


def _run_quietly(argv: List[str], timeout: float) -> int:
    """Run ``argv`` with no input and every output discarded.

    :param argv: the command line.
    :param timeout: seconds to wait before giving up.
    :returns: the exit status.
    :raises OSError: when the program cannot be started.
    :raises subprocess.TimeoutExpired: when it runs past ``timeout``.
    """
    return subprocess.run(argv, stdin=subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL,
                          timeout=timeout).returncode


class CommandLineTool:
    """A vendor command-line tool spaCR can find, install and sign in to.

    The AI providers and the GitHub CLI share this, so "is it there?" and
    "is it signed in?" are asked one way for all four.

    :ivar name: short id, and the key of :data:`INSTALL_METHODS`.
    :ivar label: human-readable label shown in the UI.
    :ivar cli_name: executable expected on ``PATH``.
    :ivar install_methods: the ways of installing it on this system, tried
        in order.
    :ivar install_hint: the one-line hint built from ``install_methods``.
    :ivar login_command: the command a user runs to sign in.
    :ivar status_command: a command that exits 0 when the tool is signed in,
        or ``()`` when the tool has none.
    """

    name: str = ""
    label: str = ""
    cli_name: str = ""
    install_methods: Tuple[InstallMethod, ...] = ()
    install_hint: str = ""
    login_command: str = ""
    status_command: Tuple[str, ...] = ()

    #: Seconds a status command may take before it counts as signed out.
    STATUS_TIMEOUT_S = 20

    def is_installed(self) -> bool:
        """Return True when the tool's executable is on ``PATH``."""
        return shutil.which(self.cli_name) is not None

    def check_signed_in(self) -> Optional[bool]:
        """Ask the tool itself whether it is signed in.

        Runs ``status_command`` with its output discarded -- for the GitHub
        CLI that output is the token -- so call it off the GUI thread.

        :returns: ``True`` when the command exits 0, ``False`` when it exits
            otherwise, cannot be started or runs past
            :attr:`STATUS_TIMEOUT_S`, and ``None`` when the tool has no
            status command to ask.
        """
        if not self.status_command:
            return None
        argv = list(self.status_command)
        argv[0] = shutil.which(argv[0]) or argv[0]
        try:
            return _run_quietly(argv, self.STATUS_TIMEOUT_S) == 0
        except (OSError, subprocess.SubprocessError):
            return False


class GitHubCli(CommandLineTool):
    """The GitHub CLI, ``gh``: installed and signed in like a provider.

    spaCR reads a token from it (:mod:`spacr.qt.ai.github_auth`) to file an
    issue without a browser round-trip. Signing in is ``gh auth login``;
    ``gh auth token`` exits 0 exactly when there is a token to read.
    """

    name = "gh"
    label = "GitHub CLI"
    cli_name = "gh"
    install_methods = install_methods_for("gh", _sys.platform)
    install_hint = install_hint_for("gh", _sys.platform)
    login_command = "gh auth login"
    status_command = ("gh", "auth", "token")

    def is_logged_in(self) -> bool:
        """Return True when ``gh auth token`` has a token to give.

        Runs a process, so call it off the GUI thread.
        """
        return self.is_installed() and self.check_signed_in() is True


_GITHUB_CLI = GitHubCli()


def github_cli() -> GitHubCli:
    """The GitHub CLI, described the way the AI providers are."""
    return _GITHUB_CLI


class ChatProvider(CommandLineTool, ABC):
    """Common interface for vendor-CLI and hosted AI chat providers.

    Subclasses set the ``name``/``label``/``cli_name``/``install_hint``/
    ``login_command`` class attributes and implement :meth:`stream_chat`.

    :ivar name: short id ("claude" / "codex" / "gemini" / "spacr_hosted").
    :ivar label: human-readable label shown in the UI.
    :ivar cli_name: executable expected on ``PATH``.
    :ivar install_hint: shell one-liner suggested for installation, built
        from ``install_methods``.
    :ivar login_command: shell one-liner the user runs to authenticate.
    """

    def __init__(self):
        """Create the provider with no child process running.

        The running process is tracked so that cancelling a stream can actually
        terminate it -- otherwise iterating the child's stdout blocks
        indefinitely and the worker thread never exits.
        """
        self._current_proc: Optional[subprocess.Popen] = None
        self._approved_resume = ""
        self._approved_tools: List[str] = []
        self.last_permission_denials: List[Dict] = []
        self.last_session_id = ""

    def is_logged_in(self) -> bool:
        """Best-effort — override per provider if a cheap check exists.

        Default: assume yes when installed. The real auth error will
        surface as a normal subprocess failure on the first send."""
        return self.is_installed()

    def is_configured(self) -> bool:
        """Return True when the install and best-effort login checks pass. The default login
        check assumes an installed CLI is signed in."""
        return self.is_installed() and self.is_logged_in()

    def source_of_key(self) -> str:
        """Compatibility text for the old KeysDialog, reporting whether the CLI is installed and
        where it was found; it does not verify sign-in."""
        if not self.is_installed():
            return "CLI not installed"
        return f"CLI found at {shutil.which(self.cli_name)}"

    def cancel_stream(self) -> None:
        """Request termination of the running subprocess, if any, with bounded waits and a kill
        attempt after a timeout.

        Terminating the process is what unblocks a stream that is waiting
        on stdout; a Python flag is only checked between chunks, and the
        next chunk may never arrive."""
        proc = self._current_proc
        if proc is None:
            return
        _mark_stopped_by_spacr(proc)
        try:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        except Exception:
            pass

    @abstractmethod
    def stream_chat(self, messages: List[Dict], system: str = "",
                     model: Optional[str] = None) -> Iterator[str]:
        """Yield text chunks streaming from the CLI subprocess.

        :param messages: conversation history as ``{role, content}`` dicts,
            oldest first; each implementation formats it into the CLI prompt.
        """



_NOISE_LINE_PREFIXES = (
    "Permission deny rule",
    "Permission allow rule",
    "Permission ask rule",
)


#: How many of a failed CLI's last output lines :class:`ProviderFailed` quotes.
_FAILURE_TAIL_LINES = 3

#: The longest quotation of a failed CLI's output, in characters.
_FAILURE_TAIL_CHARS = 400


class ProviderFailed(RuntimeError):
    """A provider request failed. For a vendor CLI, a non-zero exit status means its output
    is an error message rather than an answer.

    The three vendor CLIs report a failure the way any command-line tool does:
    a line on stdout or stderr, then a non-zero exit. A signed-out ``claude``
    prints ``Not logged in · Please run /login`` and exits 1. In GitHub #117
    an expired one printed ``Failed to authenticate: OAuth session expired
    and could not be refreshed``. Streamed as if it were a reply, that line
    was shown as spaCR AI's answer to a crash, and it was filed into GitHub
    issues as "spaCR AI's analysis of this error".

    :param cli: the executable that failed, for the message.
    :param exit_status: its exit status.
    :param output_tail: the last lines it printed, already stripped.
    :param login_command: the command that signs in to this provider, or
        ``""`` when the caller did not say which provider this was.
    :ivar exit_status: the exit status, for a caller that needs the number.
    :ivar output_tail: the quoted output, for a caller that needs the text.
    """

    def __init__(self, cli: str, exit_status: int, output_tail: str,
                 login_command: str = ""):
        """Build the message a user reads after ``[AI error]``."""
        self.exit_status = exit_status
        self.output_tail = output_tail
        said = f": {output_tail}" if output_tail else " and printed nothing"
        message = f"{cli} stopped with exit status {exit_status}{said}"
        if login_command:
            message += (
                f". If it says you are signed out, sign in again by running "
                f"`{login_command}` in a terminal, then ask again")
        super().__init__(message)


def _failure_tail(lines: List[str]) -> str:
    """Join the last non-blank lines a CLI printed into one short quotation.

    :param lines: the lines, in the order they were printed.
    :returns: at most :data:`_FAILURE_TAIL_LINES` lines joined by a space and
        cut to :data:`_FAILURE_TAIL_CHARS`, or ``""`` when every line was blank.
    """
    kept = [line.strip() for line in lines if line.strip()]
    text = " ".join(kept[-_FAILURE_TAIL_LINES:])
    if len(text) > _FAILURE_TAIL_CHARS:
        text = text[:_FAILURE_TAIL_CHARS - 1].rstrip() + "…"
    return text


#: Every provider subprocess currently being read, newest last.
#:
#: A stream is read by a worker thread that BLOCKS on the child's stdout,
#: and the only reliable way to unblock it is to end the child --
#: :meth:`ChatProvider.cancel_stream` says so and is right. But that
#: method reaches one provider's own process, and the thing that goes
#: wrong is nobody holding the provider any more: the owner is gone, the
#: thread is still blocked on a read, and Qt aborts the process the
#: moment that thread's QThread wrapper is collected.
#:
#: So the live processes are also findable from here, without a provider
#: in hand. Entries are removed as each stream ends; a crash that skips
#: the removal leaves a dead Popen, which
#: :func:`terminate_all_streams` steps over.
_LIVE_STREAMS: List[subprocess.Popen] = []

#: Every wait in process cleanup is bounded. A provider CLI is external code;
#: shutdown must not hang forever because that code ignored a signal.
_PROCESS_EXIT_TIMEOUT = 1


def _process_has_exited(proc: subprocess.Popen) -> bool:
    """Whether ``proc`` is known to have exited; uncertainty means still live."""
    try:
        return proc.poll() is not None
    except Exception:                                      # noqa: BLE001
        return False


def _mark_stopped_by_spacr(proc: subprocess.Popen) -> None:
    """Record on ``proc`` that spaCR itself asked it to stop.

    A child ended by Cancel, by quitting, or by the reader's own cleanup exits
    with a status that is not zero -- a negative signal number on POSIX, and
    ``1`` on Windows, where ``terminate`` is ``TerminateProcess(handle, 1)``.
    That status is spaCR's doing, not the CLI reporting a failure, and
    :func:`_stream_process` must not quote it back as one.

    :param proc: the child about to be signalled.
    """
    try:
        proc._spacr_stopped_it = True
    except Exception:                                      # noqa: BLE001
        pass


def _was_stopped_by_spacr(proc: subprocess.Popen) -> bool:
    """Whether :func:`_mark_stopped_by_spacr` was called on ``proc``.

    :param proc: the child.
    :returns: ``True`` only for a child spaCR signalled.
    """
    return getattr(proc, "_spacr_stopped_it", False) is True


def _kill_and_reap(proc: subprocess.Popen) -> bool:
    """Kill ``proc``, then bounded-wait to reap it; report confirmed exit."""
    _mark_stopped_by_spacr(proc)
    try:
        proc.kill()
    except Exception:                                      # noqa: BLE001
        return _process_has_exited(proc)
    try:
        proc.wait(timeout=_PROCESS_EXIT_TIMEOUT)
        return True
    except Exception:                                      # noqa: BLE001
        return _process_has_exited(proc)


def _terminate_and_reap(
        proc: subprocess.Popen, *, known_running: bool = False,
) -> tuple[bool, bool]:
    """Request termination and return ``(requested, confirmed_exited)``.

    A failed signal is not evidence that the child died. Callers use the
    second result to keep an uncertain, possibly live child registered for a
    later retry instead of losing the only handle that can unblock its reader.
    """
    if not known_running and _process_has_exited(proc):
        return False, True
    _mark_stopped_by_spacr(proc)
    try:
        proc.terminate()
    except Exception:                                      # noqa: BLE001
        return False, _process_has_exited(proc)
    try:
        proc.wait(timeout=_PROCESS_EXIT_TIMEOUT)
        return True, True
    except Exception:                                      # noqa: BLE001
        return True, _kill_and_reap(proc)


def _discard_stream(proc: subprocess.Popen) -> None:
    """Remove a confirmed-finished stream, tolerating a cleanup race."""
    try:
        _LIVE_STREAMS.remove(proc)
    except ValueError:
        pass


def terminate_all_streams() -> int:
    """Request termination of active provider subprocesses and retain uncertain processes
    for a later retry.

    :returns: Number of subprocesses for which termination was requested.
    """
    ended = 0
    for proc in list(_LIVE_STREAMS):
        requested, finished = _terminate_and_reap(proc)
        if requested:
            ended += 1
        if finished:
            _discard_stream(proc)
    return ended


def _stream_process(argv: List[str], stdin_text: Optional[str] = None,
                     env_extra: Optional[Dict[str, str]] = None,
                     provider: Optional["ChatProvider"] = None,
                     ) -> Iterator[str]:
    """Spawn a subprocess and yield stdout as it arrives.

    Reads line-by-line so noise-filtering can drop specific warnings
    (e.g. Claude Code's per-file permission-rule reminders from the
    user's ~/.claude/settings.json). Merges stderr into stdout so
    real errors show up inline.

    When ``provider`` is supplied, its process reference is registered so
    :meth:`ChatProvider.cancel_stream` can terminate a blocked read. This also
    prevents the worker thread from outliving its Python owner during exit.

    A CLI that ends with a non-zero exit status failed, and what it printed
    was its error message. Every line has already been yielded by then, so
    the failure is raised after the last one, as :class:`ProviderFailed`. A
    child spaCR stopped itself -- Cancel, quitting, or this function's own
    cleanup after a child that would not exit -- is not a failure, whatever
    status it ends with.

    :param argv: the command line to run.
    :param stdin_text: text written to the child's stdin, or ``None`` for no
        stdin pipe.
    :param env_extra: variables layered over a copy of ``os.environ``.
    :param provider: the provider this stream belongs to, or ``None``.
    :returns: an iterator over the child's output lines, noise dropped.
    :raises RuntimeError: when ``argv[0]`` cannot be run at all.
    :raises ProviderFailed: when the child exits on its own with a non-zero
        status; the message quotes its last lines and, with ``provider``,
        names the command that signs in again.
    """
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE if stdin_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not run {argv[0]!r} — is the CLI installed and on PATH?"
        ) from e

    if provider is not None:
        provider._current_proc = proc
    _LIVE_STREAMS.append(proc)

    printed: List[str] = []
    exit_status = None
    try:
        if stdin_text is not None and proc.stdin is not None:
            try:
                proc.stdin.write(stdin_text)
                proc.stdin.close()
            except BrokenPipeError:
                pass
        assert proc.stdout is not None
        for line in proc.stdout:
            if any(line.startswith(prefix) for prefix in _NOISE_LINE_PREFIXES):
                continue
            if line.strip():
                printed.append(line)
                del printed[:-_FAILURE_TAIL_LINES]
            yield line
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        finished = False
        try:
            exit_status = proc.wait(timeout=_PROCESS_EXIT_TIMEOUT)
            finished = True
        except Exception:                                  # noqa: BLE001
            _requested, finished = _terminate_and_reap(
                proc, known_running=True)
        if provider is not None and finished:
            provider._current_proc = None
        if finished:
            _discard_stream(proc)

    if (isinstance(exit_status, int) and exit_status != 0
            and not _was_stopped_by_spacr(proc)):
        raise ProviderFailed(
            os.path.basename(str(argv[0])) if argv else "the AI CLI",
            exit_status, _failure_tail(printed),
            login_command=getattr(provider, "login_command", "") or "")


def _format_conversation(messages: List[Dict], system: str = "") -> str:
    """Flatten the {role, content} history into a single prompt.

    Used by CLIs whose non-interactive mode takes one prompt string
    per invocation. Prior turns get simple role prefixes so the model
    knows who said what.
    """
    parts: List[str] = []
    if system:
        parts.append(f"System:\n{system}\n")
    for m in messages[:-1]:
        role = m.get("role", "user")
        prefix = "User" if role == "user" else "Assistant"
        parts.append(f"{prefix}:\n{m.get('content','')}\n")
    if messages:
        last = messages[-1]
        role = last.get("role", "user")
        prefix = "User" if role == "user" else "Assistant"
        parts.append(f"{prefix}:\n{last.get('content','')}")
    return "\n".join(parts)




def _provider_arguments(provider):
    """Use snapshotted reporting arguments without creating headless Qt settings."""
    if hasattr(provider, "_reporting_args"):
        return list(provider._reporting_args)
    from . import settings
    return settings.provider_args(provider.name)


class ClaudeCliProvider(ChatProvider):
    """Anthropic Claude via the ``claude`` (Claude Code) CLI."""

    name = "claude"
    label = "Claude (via Claude Code)"
    cli_name = "claude"
    install_methods = install_methods_for("claude", _sys.platform)
    install_hint = install_hint_for("claude", _sys.platform)
    login_command = "claude auth login"
    status_command = ("claude", "auth", "status")

    def stream_chat(self, messages: List[Dict], system: str = "",
                     model: Optional[str] = None) -> Iterator[str]:
        """Stream a chat completion from the ``claude`` CLI.

        :param messages: conversation history as ``{role, content}`` dicts.
        :param system: optional system prompt appended via
            ``--append-system-prompt``.
        :param model: optional model override passed via ``--model``.
            When None the current response-speed setting supplies one.
        :returns: iterator yielding stdout text chunks.
        """
        resume, allowed = self._approved_resume, self._approved_tools
        self._approved_resume, self._approved_tools = "", []
        self.last_permission_denials = []
        self.last_session_id = ""
        if resume and messages:
            argv = ["claude", "-p", str(messages[-1].get("content", "")),
                    "--resume", resume]
            if allowed:
                argv += ["--allowedTools", *allowed]
        else:
            prompt = _format_conversation(messages, system=system)
            argv = ["claude", "-p", prompt]
        if system:
            argv += ["--append-system-prompt", system]
        if model:
            argv += ["--model", model]
        else:
            argv += _provider_arguments(self)
        argv += ["--output-format", "json"]
        yield from self._read_result(_stream_process(argv, provider=self))

    def _read_result(self, lines: Iterator[str]) -> Iterator[str]:
        """Yield the reply from ``claude --output-format json`` output.

        The CLI prints one JSON result object. Its ``result`` text is the
        reply; its ``permission_denials`` are the tool calls that needed the
        user's approval, which ``claude -p`` cannot ask for (GitHub #147).
        They are kept in :attr:`last_permission_denials` with the session in
        :attr:`last_session_id`, so the console can ask the user and resume
        the session with :meth:`allow_once`. Lines that are not a result
        object pass through unchanged.

        :param lines: the CLI's output lines.
        :returns: an iterator over the reply text.
        :raises ProviderFailed: when the CLI failed, quoting its result text
            rather than the raw JSON.
        """
        result: Dict = {}
        try:
            for line in lines:
                parsed = _result_object(line)
                if parsed is None:
                    yield line
                    continue
                result = parsed
                self.last_session_id = str(parsed.get("session_id") or "")
                self.last_permission_denials = [
                    d for d in parsed.get("permission_denials") or []
                    if isinstance(d, dict)]
                text = parsed.get("result")
                if isinstance(text, str) and text and not parsed.get("is_error"):
                    yield text if text.endswith("\n") else text + "\n"
        except ProviderFailed as failed:
            text = result.get("result") if result else ""
            if isinstance(text, str) and text.strip():
                raise ProviderFailed(
                    "claude", failed.exit_status, _failure_tail([text]),
                    login_command=self.login_command) from None
            raise

    def allow_once(self, denials: List[Dict], session_id: str) -> List[str]:
        """Let the next :meth:`stream_chat` resume ``session_id`` with these
        tool calls allowed, for that one reply only.

        :param denials: entries of :attr:`last_permission_denials` the user
            approved.
        :param session_id: the session they were denied in.
        :returns: the ``--allowedTools`` rules that will be passed.
        """
        rules = [rule for rule in (tool_rule(d) for d in denials) if rule]
        self._approved_resume = str(session_id or "")
        self._approved_tools = list(dict.fromkeys(rules))
        return list(self._approved_tools)



def _result_object(line: str) -> Optional[Dict]:
    """The ``claude --output-format json`` result object on ``line``, or None."""
    text = line.strip()
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    if isinstance(parsed, dict) and parsed.get("type") == "result":
        return parsed
    return None


def tool_rule(denial: Dict) -> str:
    """Build a ``--allowedTools`` rule from a denied tool call.

    A shell command is allowed as written, ``Bash(gh auth status)``; a web
    fetch for its host; any other tool by its name.

    :param denial: one ``permission_denials`` entry.
    :returns: the rule, or ``""`` when the entry names no tool.
    """
    name = str(denial.get("tool_name") or "").strip()
    given = denial.get("tool_input")
    given = given if isinstance(given, dict) else {}
    if not name:
        return ""
    if name == "Bash" and str(given.get("command") or "").strip():
        return f"Bash({str(given['command']).strip()})"
    if name == "WebFetch" and given.get("url"):
        from urllib.parse import urlparse

        host = urlparse(str(given["url"])).hostname
        return f"WebFetch(domain:{host})" if host else name
    return name


def describe_tool_call(denial: Dict) -> str:
    """One line naming what a denied tool call would have done.

    :param denial: one ``permission_denials`` entry.
    :returns: the command for a shell call, otherwise the tool's name and
        its path, address or pattern.
    """
    name = str(denial.get("tool_name") or "").strip() or "?"
    given = denial.get("tool_input")
    given = given if isinstance(given, dict) else {}
    if name == "Bash" and given.get("command"):
        return str(given["command"]).strip()
    for key in ("file_path", "path", "url", "pattern", "notebook_path"):
        if given.get(key):
            return f"{name}: {given[key]}"
    return name


class CodexCliProvider(ChatProvider):
    """OpenAI ChatGPT via the ``codex`` CLI."""

    name = "codex"
    label = "ChatGPT (via Codex CLI)"
    cli_name = "codex"
    install_methods = install_methods_for("codex", _sys.platform)
    install_hint = install_hint_for("codex", _sys.platform)
    login_command = "codex login"
    status_command = ("codex", "login", "status")

    def stream_chat(self, messages: List[Dict], system: str = "",
                     model: Optional[str] = None) -> Iterator[str]:
        """Stream a chat completion from the ``codex`` CLI.

        :param messages: conversation history as ``{role, content}`` dicts.
        :param system: optional system prompt folded into the prompt body.
        :param model: optional model override passed via ``--model``.
            When None the current response-speed setting supplies one.
        :returns: iterator yielding stdout text chunks.
        """
        prompt = _format_conversation(messages, system=system)
        argv = ["codex", "exec", prompt]
        if model:
            argv += ["--model", model]
        else:
            argv += _provider_arguments(self)
        yield from _stream_process(argv, provider=self)



class GeminiCliProvider(ChatProvider):
    """Google Gemini via the ``gemini`` CLI."""

    name = "gemini"
    label = "Gemini (via Gemini CLI)"
    cli_name = "gemini"
    install_methods = install_methods_for("gemini", _sys.platform)
    install_hint = install_hint_for("gemini", _sys.platform)
    login_command = "gemini"

    def stream_chat(self, messages: List[Dict], system: str = "",
                     model: Optional[str] = None) -> Iterator[str]:
        """Stream a chat completion from the ``gemini`` CLI.

        :param messages: conversation history as ``{role, content}`` dicts.
        :param system: optional system prompt folded into the prompt body.
        :param model: optional model override passed via ``-m``.
            When None the current response-speed setting supplies one.
        :returns: iterator yielding stdout text chunks.
        """
        prompt = _format_conversation(messages, system=system)
        argv = ["gemini", "-p", prompt]
        if model:
            argv += ["-m", model]
        else:
            args = _provider_arguments(self)
            if args and args[0] == "--model":
                argv += ["-m", args[1]]
            elif args:
                argv += args
        yield from _stream_process(argv, provider=self)



#: Short id of the hosted provider, as stored by the provider preference.
HOSTED_PROVIDER_NAME = "spacr_hosted"


class HostedProviderRefused(ProviderFailed):
    """The hosted request gave no advice because of a quota, network or other relay error.

    A :class:`ProviderFailed`, so the console writes the message as it
    stands after "[AI error]".

    :param message: the sentence the user reads.
    :param reason: ``"quota"``, ``"network"`` or ``"error"``, from
        :class:`spacr.crashreport.RelayRefused`, kept on the exception as
        ``reason``.
    """

    def __init__(self, message: str, reason: str = "error"):
        """Keep the reason; the message is the whole text shown."""
        RuntimeError.__init__(self, message)
        self.reason = reason
        self.exit_status = 429 if reason == "quota" else 1
        self.output_tail = ""


def _hosted_request(messages: List[Dict]) -> Tuple[Dict, str]:
    """Turn a conversation into what the hosted provider may send.

    The newest traceback anywhere in the conversation becomes the
    anonymised summary of :func:`spacr.crashreport.public_summary`. The
    newest message, with its console context, code blocks and traceback
    removed, is the question. Nothing else of the conversation is sent.

    :param messages: conversation history as ``{role, content}`` dicts.
    :returns: ``(summary, question)``.
    """
    import re

    from ... import crashreport

    traceback_text, app = "", ""
    for message in reversed(messages or []):
        content = str(message.get("content", "") or "")
        at = content.rfind("Traceback (most recent call last)")
        if at < 0:
            continue
        block = content[at:]
        for end in ("```", "</spacr_console_context>", "--- complete"):
            cut = block.find(end)
            if cut > 0:
                block = block[:cut]
        traceback_text = block
        found = re.search(r"^Active app:\s*(\S+)", content, re.MULTILINE)
        app = found.group(1) if found else ""
        break
    question = ""
    if messages:
        question = str(messages[-1].get("content", "") or "")
        question = re.sub(r"<spacr_console_context>.*?</spacr_console_context>",
                          " ", question, flags=re.DOTALL)
        question = re.sub(r"```.*?```", " ", question, flags=re.DOTALL)
        at = question.find("Traceback (most recent call last)")
        if at >= 0:
            question = question[:at]
        question = re.sub(r"(?m)^(?:Active app:.*|Traceback:)\s*$", " ",
                          question)
    summary = crashreport.public_summary(traceback_text=traceback_text,
                                         module=app)
    return summary, question.strip()


def question_cache_key(messages: List[Dict]) -> str:
    """The key a fresh question's answer is saved under on this computer.

    Only a conversation consisting of a single user message is eligible. A message
    containing ``Traceback (most recent call last)`` is excluded even when short. The
    original message, including console context, code and active-app metadata, must fit
    within :data:`spacr.crashreport.QUESTION_CACHE_CONTEXT_MAX` characters.
    :func:`spacr.crashreport.question_key` validates and hashes the cleaned question and
    active app; the key does not depend on the provider.

    :param messages: conversation history as ``{role, content}`` dicts.
    :returns: the key, or ``""`` when the answer must not be saved.
    """
    import re

    from ... import crashreport

    asked = [m for m in messages or [] if m.get("role", "user") == "user"]
    if len(asked) != 1 or len(messages or []) != 1:
        return ""
    content = str(asked[0].get("content", "") or "")
    if "Traceback (most recent call last)" in content:
        return ""
    found = re.search(r"^Active app:\s*(\S+)", content, re.MULTILINE)
    question = re.sub(r"<spacr_console_context>.*?</spacr_console_context>",
                      " ", content, flags=re.DOTALL)
    question = re.sub(r"```.*?```", " ", question, flags=re.DOTALL)
    question = re.sub(r"(?m)^Active app:.*$", " ", question).strip()
    context = len(content) - len(question)
    return crashreport.question_key(question, found.group(1) if found else "",
                                    context)


class HostedProvider(ChatProvider):
    """spaCR's hosted assistant: advice from spaCR's relay, no account needed.

    It sends only what :func:`_hosted_request` keeps, the anonymised error summary and
    the question, to the relay's ``/diagnose`` route through
    :func:`spacr.crashreport.diagnose_remotely`. The answer is generated advice and is
    labelled as such. It is the default while network access is allowed unless a
    configured vendor CLI is explicitly preferred (:func:`default_provider`). It is
    unavailable when the network is forbidden
    (:func:`spacr.crashreport.network_allowed`).
    """

    name = HOSTED_PROVIDER_NAME
    label = "spaCR AI (free)"
    cli_name = ""
    install_hint = ""
    login_command = ""

    #: The allowance the relay reported with the last answer, or ``None``.
    last_allowance: Optional[Dict] = None

    #: The model tier of the last answer: ``"best"`` or ``"light"``.
    last_tier: str = "best"

    def is_installed(self) -> bool:
        """Nothing to install: the hosted provider is always present."""
        return True

    def is_logged_in(self) -> bool:
        """Nothing to sign in to: the relay needs no account."""
        return True

    def is_configured(self) -> bool:
        """Whether the hosted provider may be used: the network is allowed."""
        from ... import crashreport

        return crashreport.network_allowed()

    def check_signed_in(self) -> Optional[bool]:
        """There is no sign-in to check.

        :returns: ``None``.
        """
        return None

    def source_of_key(self) -> str:
        """Say where answers come from, in place of a CLI's location."""
        from ... import crashreport

        return f"spaCR relay at {crashreport.relay_url()}"

    def stream_chat(self, messages: List[Dict], system: str = "",
                    model: Optional[str] = None) -> Iterator[str]:
        """Ask the relay and yield its advice, then a line saying what it is.

        :param messages: conversation history as ``{role, content}`` dicts;
            only the anonymised summary and the newest question leave this
            computer.
        :param system: ignored; the relay holds spaCR's own prompt.
        :param model: ignored; the relay chooses its free model.
        :returns: iterator yielding the answer.
        :raises HostedProviderRefused: when the request is refused because of a quota, a
            forbidden or unavailable network, or another relay error.
        """
        from ... import crashreport
        from ..i18n import tr

        summary, question = _hosted_request(messages)
        try:
            answer = crashreport.diagnose_remotely(summary, question)
        except crashreport.RelayRefused as exc:
            raise HostedProviderRefused(
                hosted_refusal_message(exc.reason, exc.scope, exc.resets_at),
                exc.reason) from exc
        self.last_allowance = answer.get("allowance")
        self.last_tier = answer.get("tier", "best")
        yield answer["advice"].rstrip() + "\n\n"
        if self.last_tier == "light":
            yield "[" + tr("lighter model") + "] "
        yield tr("Generated advice from spaCR hosted AI ({model}). It can be "
                 "wrong; check it before you act on it.",
                 model=answer.get("model") or tr("free model"))


def local_reset_time(resets_at=None) -> str:
    """The time the free allowances reset, in this computer's local time.

    :param resets_at: an aware datetime or an ISO-8601 string; defaults to
        the next 00:00 UTC.
    :returns: ``HH:MM``, followed by the weekday when it is not today.
    """
    from datetime import datetime

    from ... import crashreport

    moment = resets_at
    if isinstance(moment, str):
        moment = crashreport._parse_reset(moment)
    if not isinstance(moment, datetime):
        moment = crashreport.next_reset_utc()
    local = moment.astimezone()
    text = local.strftime("%H:%M")
    if local.date() != datetime.now().astimezone().date():
        text = f"{text} {local.strftime('%a')}"
    return text


def hosted_cli_hint() -> str:
    """The hint for connecting a CLI that uses the user's own provider account.

    :returns: the translated sentence.
    """
    from ..i18n import tr

    return tr("To use your own provider account, connect your own Claude, Codex or Gemini "
              "CLI under Preferences → AI → Providers.")


def hosted_allowance_text(allowance: Optional[Dict]) -> str:
    """One line saying how many free hosted-AI questions are left today.

    :param allowance: what :func:`spacr.crashreport.fetch_allowance`
        returns, or ``None`` when it is not known.
    :returns: the translated line, or ``""`` when nothing is known.
    """
    from ... import crashreport
    from ..i18n import tr

    if not allowance:
        return ""
    ai = allowance.get("ai") or {}
    best = int(ai.get("best_left", 0) or 0)
    light = int(ai.get("light_left", 0) or 0)
    when = local_reset_time(allowance.get("resets_at"))
    pool = allowance.get("pool", "best")
    if pool == "exhausted":
        return tr("spaCR's free AI capacity is used up for today · resets "
                  "at {time}", time=when)
    if best <= 0 and light <= 0:
        return tr("No free spaCR AI questions left today · resets at {time}",
                  time=when)
    if pool == "light":
        return tr("Free capacity is low, so everyone gets the lighter model "
                  "now · {light} questions left today · resets at {time}",
                  light=light, time=when)
    if best <= 0:
        return tr("Best-model questions used today · {light} of {total} "
                  "lighter-model questions left · resets at {time}",
                  light=light, total=crashreport.AI_LIGHT_PER_DAY, time=when)
    return tr("{best} of {total} best-model questions left today · then "
              "{light} on the lighter model · resets at {time}",
              best=best, total=crashreport.AI_BEST_PER_DAY, light=light,
              time=when)


def hosted_refusal_message(reason: str, scope: str = "",
                           resets_at=None) -> str:
    """The sentence shown when the hosted provider gives no advice.

    :param reason: ``"quota"``, ``"network"`` or anything else.
    :param scope: for a quota refusal, ``"install"`` (this install's daily
        questions are used), ``"global"`` (spaCR's shared free capacity is)
        or ``"ip"``.
    :param resets_at: when the limit resets; defaults to the next 00:00 UTC.
    :returns: the translated sentence; a quota refusal also carries
        :func:`hosted_cli_hint`.
    """
    from ..i18n import tr

    if reason == "quota":
        when = local_reset_time(resets_at)
        if scope == "global":
            text = tr("spaCR's free AI capacity is used up for today. Try "
                      "again after {time}.", time=when)
        else:
            text = tr("You've used today's free spaCR AI questions. They "
                      "reset at {time}.", time=when)
        return f"{text} {hosted_cli_hint()}"
    if reason == "network":
        return tr("spaCR hosted AI could not be reached. Check the network "
                  "connection and try again later.")
    return tr("spaCR hosted AI could not answer this time. Try again later.")


_PROVIDERS: List[ChatProvider] = [
    ClaudeCliProvider(),
    CodexCliProvider(),
    GeminiCliProvider(),
]

_HOSTED = HostedProvider()


def list_providers() -> List[ChatProvider]:
    """Return every vendor CLI provider, regardless of install state.

    The hosted provider is not one of them: it has nothing to install or
    sign in to. :func:`hosted_provider` returns it.
    """
    return list(_PROVIDERS)


def hosted_provider() -> HostedProvider:
    """Return spaCR's hosted provider."""
    return _HOSTED


def configured_providers() -> List[ChatProvider]:
    """Return providers that pass configuration checks, spaCR AI (free) first.

    The hosted provider leads whenever the network is allowed. A vendor CLI follows when
    its install and best-effort login checks pass; the current vendor classes treat an
    executable on ``PATH`` as configured. An explicit saved ``"none"`` choice
    returns an empty list without discovery. These checks do not confirm relay
    capacity or a working authenticated CLI session.
    """
    from ... import crashreport
    policy = crashreport._ERROR_POLICY_CACHE.get("affirmed_policy") or {}
    if policy.get("provider") == "none":
        return []
    ready = [_HOSTED] if _HOSTED.is_configured() else []
    return ready + [p for p in _PROVIDERS if p.is_configured()]


def default_provider(preferred: str = "") -> Optional[ChatProvider]:
    """Return the provider the AI switch uses.

    :param preferred: the provider Preferences names, or ``""``.
    :returns: the matching ``preferred`` provider from :func:`configured_providers`;
        otherwise the first provider in that list, or ``None`` if it is empty. The
        hosted provider leads while the network is allowed; otherwise vendor CLIs are
        considered in their registered order. An explicit ``"none"`` returns
        ``None`` without provider discovery. This is a configuration choice, not a
        guarantee of authentication, remaining quota or request success.
    """
    if preferred == "none":
        return None
    ready = configured_providers()
    for p in ready:
        if preferred and p.name == preferred:
            return p
    return ready[0] if ready else None


#: The choices Preferences → AI offers, in order: ``(name, label, what it
#: needs)``. Every one is listed whether or not it is set up.
PROVIDER_CHOICES: Tuple[Tuple[str, str, str], ...] = (
    (HOSTED_PROVIDER_NAME, "spaCR AI (free, daily allowance)",
     "Needs nothing: no account and no install. Answers come from spaCR's "
     "relay within a daily allowance."),
    ("claude", "Claude", "Uses your Claude Code login (the claude CLI)."),
    ("codex", "GPT (Codex)",
     "Uses your ChatGPT login through the Codex CLI."),
    ("gemini", "Gemini", "Uses your Google login through the Gemini CLI."),
    ("none", "None", ""),
)


def effective_provider_name(preferred: str = "") -> str:
    """Return the name of the provider Preferences shows as chosen.

    :param preferred: the stored preference; ``""`` (never chosen) means
        spaCR AI (free).
    :returns: a name from :data:`PROVIDER_CHOICES`.
    """
    names = {name for name, _label, _needs in PROVIDER_CHOICES}
    return preferred if preferred in names else HOSTED_PROVIDER_NAME


def apply_provider_choice(name: str) -> bool:
    """Save a provider choice and try to apply it to open consoles.

    The choice is saved even when its configuration checks fail. Open consoles are given
    the chosen provider when those checks pass, otherwise the fallback from
    :func:`default_provider`, which can be another vendor CLI or ``None`` when the
    network is forbidden. Choosing ``"none"`` unsets the provider without a
    fallback. Exceptions while updating consoles are ignored.

    :param name: a name from :data:`PROVIDER_CHOICES`.
    :returns: whether the chosen provider passed configuration checks, regardless of
        whether updating every console succeeded.
    """
    from ..preferences import set_preferred_provider

    name = effective_provider_name(name)
    set_preferred_provider(name)
    provider = get_provider(name)
    ready = bool(provider is not None and provider.is_configured())
    target = None if name == "none" else (provider if ready else default_provider())
    try:
        from PySide6.QtWidgets import QApplication

        from ..widgets.console_panel import ConsolePanel

        for widget in QApplication.allWidgets():
            if isinstance(widget, ConsolePanel):
                widget.set_ai_provider(target.name if target else None)
    except Exception:
        pass
    return ready


def write_report_summary(summary: Dict, provider: Optional[ChatProvider],
                         timeout: Optional[float] = None) -> str:
    """Have the user's own AI summarise an anonymised report, or return ``""``.

    The provider is given :func:`spacr.crashreport.ai_summary_prompt` as a single user
    message, and a worker thread collects its reply. If the worker is still running
    after ``timeout`` seconds, cancellation is requested and ``""`` is returned; a
    provider exception or an empty reply also gives ``""``. Blocks, so call it off the
    GUI thread.

    :param summary: a summary from :func:`spacr.crashreport.public_summary`.
    :param provider: a vendor CLI provider, or ``None``.
    :param timeout: seconds to wait; defaults to
        :data:`spacr.crashreport.AI_SUMMARY_TIMEOUT_S`.
    :returns: the summary text, scrubbed and cut to
        :data:`spacr.crashreport.MAX_AI_SUMMARY_CHARS`.
    """
    import threading

    from ... import crashreport

    if provider is None or provider.name == HOSTED_PROVIDER_NAME:
        return ""
    wait = crashreport.AI_SUMMARY_TIMEOUT_S if timeout is None else timeout
    prompt = crashreport.ai_summary_prompt(summary)
    chunks: List[str] = []
    failed: List[BaseException] = []

    def _ask() -> None:
        """Collect the provider's answer."""
        try:
            for chunk in provider.stream_chat(
                    [{"role": "user", "content": prompt}]):
                chunks.append(str(chunk))
        except Exception as exc:
            failed.append(exc)

    worker = threading.Thread(target=_ask, name="spacr-report-summary",
                              daemon=True)
    worker.start()
    worker.join(wait)
    if worker.is_alive():
        try:
            provider.cancel_stream()
        except Exception:
            pass
        return ""
    if failed:
        return ""
    return crashreport.scrub_public_lines(
        "".join(chunks).strip(), crashreport.MAX_AI_SUMMARY_CHARS, True)


def with_ai_summary(summary: Dict, provider: Optional[ChatProvider]) -> Dict:
    """Prepare a report to request or include an AI summary.

    For spaCR AI (free), set ``summarize: true`` so report submission can request a
    relay-written summary; this function does not contact the relay itself. A vendor CLI
    writes the summary here (:func:`write_report_summary`) and a non-empty result is
    included as ``ai_summary``. With no provider, or when the CLI gives no text, the
    copied report data is unchanged.

    :param summary: a summary from :func:`spacr.crashreport.public_summary`.
    :param provider: the active provider, or ``None``.
    :returns: a new summary dict.
    """
    out = dict(summary)
    if provider is None:
        return out
    if provider.name == HOSTED_PROVIDER_NAME:
        out["summarize"] = True
        return out
    text = write_report_summary(out, provider)
    if text:
        out["ai_summary"] = text
    return out


def explain_failure(summary: Dict, provider: Optional[ChatProvider],
                    timeout: Optional[float] = None) -> Dict:
    """Return a cached explanation or request one from the selected provider.

    After a local cache miss, spaCR AI (free) asks the relay's ``/diagnose`` with the
    anonymised summary and :data:`spacr.crashreport.EXPLAIN_QUESTION`; a vendor CLI
    answers :func:`spacr.crashreport.ai_summary_prompt` here
    (:func:`write_report_summary`). The returned text can be used for the console and
    the report's ``ai_summary``. After a local miss, this function attempts one provider
    request; a relay reply can itself be cached. Blocks, so call it off the GUI thread.

    :param summary: a summary from :func:`spacr.crashreport.public_summary`.
    :param provider: the active provider, or ``None``.
    :param timeout: seconds a vendor CLI may take; defaults to
        :data:`spacr.crashreport.AI_SUMMARY_TIMEOUT_S`.
    :returns: ``{"text": explanation or "", "tier": "best"|"light"|"", "refused": "" or
        "quota"|"network"|"error"|"none"}``. A local cache hit or a completed provider
        call also includes ``"cause"``. ``"saved"`` is ``"local"`` for a local hit,
        ``"relay"`` for reused non-empty relay advice, or ``""`` for a fresh hosted
        reply; other branches omit it.

    With no provider, return ``"refused": "none"`` without consulting the local cache.
    Otherwise a valid explanation from :func:`spacr.crashreport.saved_explanation` is
    reused before any request, for every provider. Cache entries can expire or be
    evicted, and failed saves are ignored, so the same error can require another
    request.
    """
    from ... import crashreport

    if provider is None:
        return {"text": "", "tier": "", "refused": "none"}
    signature = crashreport.explanation_signature(summary)
    saved = crashreport.saved_explanation(signature)
    if saved:
        return {"text": saved["text"], "tier": saved["tier"], "refused": "",
                "saved": "local", "cause": saved.get("cause", "")}
    if provider.name == HOSTED_PROVIDER_NAME:
        try:
            answer = crashreport.diagnose_remotely(
                summary, crashreport.EXPLAIN_QUESTION,
                timeout=crashreport.AI_SUMMARY_TIMEOUT_S)
        except crashreport.RelayRefused as exc:
            return {"text": "", "tier": "", "refused": exc.reason}
        except Exception:
            return {"text": "", "tier": "", "refused": "error"}
        allowance = answer.get("allowance")
        if allowance:
            provider.last_allowance = allowance
        text = crashreport.scrub_public_lines(
            answer.get("advice", ""), crashreport.MAX_AI_SUMMARY_CHARS, True)
        tier = answer.get("tier", "best")
        text, cause = crashreport.split_cause(text)
        cause = answer.get("cause", "") or cause
        crashreport.save_explanation(signature, text, tier,
                                     answer.get("model", ""), cause=cause)
        return {"text": text, "tier": tier,
                "refused": "" if text else "error",
                "saved": "relay" if answer.get("cached") and text else "",
                "cause": cause}
    text, cause = crashreport.split_cause(
        write_report_summary(summary, provider, timeout))
    crashreport.save_explanation(signature, text, cause=cause)
    return {"text": text, "tier": "", "refused": "" if text else "error",
            "cause": cause}


#: What each vendor CLI can say about its remaining usage, checked
#: 2026-10-14. None has a reliable, free, non-interactive, machine-readable
#: source: Claude Code shows plan limits only inside an interactive session
#: (``/usage``, and the JSON it hands a status-line command, which only a
#: running session invokes); ``claude -p --output-format json`` reports
#: the tokens and cost of the call just made, not what is left. The Codex
#: CLI shows limits in its interactive ``/status`` view only. The Gemini
#: CLI's ``/stats`` is interactive and covers the current session. So no
#: bar is drawn for them and no number is invented.
CLI_USAGE_TEXT: Dict[str, str] = {
    "claude": "Your own Claude account · limits set by Anthropic",
    "codex": "Your own ChatGPT account (Codex) · limits set by OpenAI",
    "gemini": "Your own Google account (Gemini) · limits set by Google",
}


def usage_bar(provider_name: Optional[str],
              allowance: Optional[Dict] = None) -> Tuple[Optional[int], str]:
    """What the usage bar under the AI chat box shows for a provider.

    For spaCR AI (free) the bar is the user's share of today's shared free
    capacity, ``neurons_left / share_neurons``, from the relay's dynamic
    allowance. For a vendor CLI there is no reliable usage source
    (:data:`CLI_USAGE_TEXT`), so the bar is hidden and only a line is shown.

    :param provider_name: the selected provider's name, or ``None``.
    :param allowance: what :func:`spacr.crashreport.fetch_allowance`
        returned, for spaCR AI.
    :returns: ``(percent or None to hide the bar, translated text)``.
    """
    from ..i18n import tr

    if provider_name in CLI_USAGE_TEXT:
        return None, tr(CLI_USAGE_TEXT[provider_name])
    if provider_name != HOSTED_PROVIDER_NAME:
        return None, ""
    if not allowance:
        return None, hosted_cli_hint()
    ai = allowance.get("ai") or {}
    share = ai.get("share_neurons")
    left = ai.get("neurons_left")
    if share is None or left is None:
        return None, hosted_allowance_text(allowance)
    share, left = int(share or 0), int(left or 0)
    if share <= 0 or left <= 0 or allowance.get("pool") == "exhausted":
        return 0, tr("Daily share used. Errors are still reported without an "
                     "AI summary.")
    percent = max(0, min(100, round(100 * left / share)))
    text = tr("{pct}% of your share left · {n} users sharing today · resets "
              "{time}", pct=percent, n=int(ai.get("active_users", 0) or 0),
              time=local_reset_time(allowance.get("resets_at")))
    if ai.get("tier_now") == "light":
        text += " · " + tr("lighter model")
    return percent, text


def hosted_daily_allowance_text() -> str:
    """The free allowance of spaCR AI, as Preferences states it.

    :returns: the translated sentence.
    """
    from ... import crashreport
    from ..i18n import tr

    return tr("Free every day: {best} questions on the best model, then "
              "{light} on a lighter model. Resets at {time}.",
              best=crashreport.AI_BEST_PER_DAY,
              light=crashreport.AI_LIGHT_PER_DAY,
              time=local_reset_time())


def get_provider(name: str) -> Optional[ChatProvider]:
    """Look up a provider by its short id.

    :param name: provider id (``"claude"``, ``"codex"``, ``"gemini"`` or
        :data:`HOSTED_PROVIDER_NAME`).
    :returns: the matching provider, or ``None`` if no such id.
    """
    for p in _PROVIDERS + [_HOSTED]:
        if p.name == name:
            return p
    return None
