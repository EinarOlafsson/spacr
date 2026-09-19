"""
Provider abstraction — one class per AI vendor. Each shells out to
the vendor's own coding-agent CLI so authentication piggy-backs on
the user's chat subscription (Claude.ai Pro, ChatGPT Plus/Pro/Team,
Google account) — no separate API billing.

* Anthropic Claude → the `claude` CLI ("Claude Code")
* OpenAI ChatGPT   → the `codex`  CLI
* Google Gemini    → the `gemini` CLI

Each provider::

    is_installed()   — is the CLI on PATH?
    is_logged_in()   — best-effort check; falls back to "assume yes if
                       installed" (the actual auth error surfaces on
                       the first stream chunk).
    stream_chat()    — spawn the CLI subprocess, yield stdout chunks.

Conversation context is carried by concatenating the full message
history into each prompt (simplest approach that works uniformly
across all three CLIs). For subscription users token count is not a
concern.

How each CLI is installed is data, not prose: :data:`INSTALL_METHODS` holds
one row per way of installing it on each operating system, and both the
``install_hint`` a screen shows and the command
:mod:`spacr.qt.ai.cli_install` runs are read from those rows. The GitHub CLI
is described the same way by :class:`GitHubCli`, which shares the
:class:`CommandLineTool` interface without being a chat provider.
"""
from __future__ import annotations

import sys as _sys

import os
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
_CLAUDE_WINDOWS = (
    InstallMethod(("curl",),
                  "curl -fsSL https://claude.ai/install.cmd -o install.cmd"
                  " && install.cmd && del install.cmd", "cmd"),
)
_CODEX_NPM = InstallMethod(("npm",), "npm install -g @openai/codex")
_GEMINI_NPM = InstallMethod(("npm",), "npm install -g @google/gemini-cli")
_GH_CONDA = InstallMethod(("conda",),
                          "conda install --yes gh --channel conda-forge")
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
#: installers (item 414); the npm and Homebrew rows are the vendors' package
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
    drift apart. On macOS and Linux the alternatives are joined with
    ``   # or ``, which a POSIX shell reads as a comment; on Windows only the
    first row is shown, because ``cmd`` has no such comment and would hand
    the words after ``#`` to the installer as arguments.

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
    """Abstract base for AI chat providers that shell out to a vendor CLI.

    Subclasses set the ``name``/``label``/``cli_name``/``install_hint``/
    ``login_command`` class attributes and implement :meth:`stream_chat`.

    :ivar name: short id ("claude" / "codex" / "gemini").
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

    def is_logged_in(self) -> bool:
        """Best-effort — override per provider if a cheap check exists.

        Default: assume yes when installed. The real auth error will
        surface as a normal subprocess failure on the first send."""
        return self.is_installed()

    def is_configured(self) -> bool:
        """Return True when the CLI is both installed and logged in."""
        return self.is_installed() and self.is_logged_in()

    def source_of_key(self) -> str:
        """Compat string for the old KeysDialog — now describes the
        CLI's install/login state."""
        if not self.is_installed():
            return "CLI not installed"
        return f"CLI found at {shutil.which(self.cli_name)}"

    def cancel_stream(self) -> None:
        """Kill the running subprocess (if any).

        This is the ONLY reliable way to unblock a stream that's stuck
        waiting on stdout — flipping a Python flag would only unblock
        between chunks, which may never come."""
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
        """Yield text chunks streaming from the CLI subprocess."""



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
    """A provider CLI exited with a non-zero status, so what it printed is an
    error message and not an answer.

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
    """Terminate active provider subprocesses and release their readers.

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
        from . import settings as ai_settings
        prompt = _format_conversation(messages, system=system)
        argv = ["claude", "-p", prompt]
        if system:
            argv += ["--append-system-prompt", system]
        if model:
            argv += ["--model", model]
        else:
            argv += ai_settings.provider_args(self.name)
        yield from _stream_process(argv, provider=self)



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
        from . import settings as ai_settings
        prompt = _format_conversation(messages, system=system)
        argv = ["codex", "exec", prompt]
        if model:
            argv += ["--model", model]
        else:
            argv += ai_settings.provider_args(self.name)
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
        from . import settings as ai_settings
        prompt = _format_conversation(messages, system=system)
        argv = ["gemini", "-p", prompt]
        if model:
            argv += ["-m", model]
        else:
            args = ai_settings.provider_args(self.name)
            if args and args[0] == "--model":
                argv += ["-m", args[1]]
            elif args:
                argv += args
        yield from _stream_process(argv, provider=self)



_PROVIDERS: List[ChatProvider] = [
    ClaudeCliProvider(),
    CodexCliProvider(),
    GeminiCliProvider(),
]


def list_providers() -> List[ChatProvider]:
    """Return every registered provider, regardless of install state."""
    return list(_PROVIDERS)


def configured_providers() -> List[ChatProvider]:
    """Return only providers whose CLI is installed and logged in."""
    return [p for p in _PROVIDERS if p.is_configured()]


def get_provider(name: str) -> Optional[ChatProvider]:
    """Look up a registered provider by its short id.

    :param name: provider id (``"claude"``, ``"codex"``, ``"gemini"``).
    :returns: the matching provider, or ``None`` if no such id.
    """
    for p in _PROVIDERS:
        if p.name == name:
            return p
    return None
