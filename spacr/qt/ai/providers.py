"""
Provider abstraction — one class per AI vendor. Each shells out to
the vendor's own coding-agent CLI so authentication piggy-backs on
the user's chat subscription (Claude.ai Pro, ChatGPT Plus/Pro/Team,
Google account) — no separate API billing.

* Anthropic Claude → the `claude` CLI ("Claude Code")
* OpenAI ChatGPT   → the `codex`  CLI
* Google Gemini    → the `gemini` CLI

Each provider:
    is_installed()   — is the CLI on PATH?
    is_logged_in()   — best-effort check; falls back to "assume yes if
                       installed" (the actual auth error surfaces on
                       the first stream chunk).
    stream_chat()    — spawn the CLI subprocess, yield stdout chunks.

Conversation context is carried by concatenating the full message
history into each prompt (simplest approach that works uniformly
across all three CLIs). For subscription users token count is not a
concern.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional


class ChatProvider(ABC):
    name: str = ""            # short id: "claude" / "codex" / "gemini"
    label: str = ""           # human-readable label
    cli_name: str = ""        # the executable on PATH
    install_hint: str = ""    # shell one-liner to install
    login_command: str = ""   # shell one-liner the user should run

    def __init__(self):
        # Tracks the currently-running child process so cancel_stream()
        # can actually terminate it — otherwise `for line in proc.stdout`
        # blocks indefinitely and the worker thread never exits.
        self._current_proc: Optional[subprocess.Popen] = None

    def is_installed(self) -> bool:
        return shutil.which(self.cli_name) is not None

    def is_logged_in(self) -> bool:
        """Best-effort — override per provider if a cheap check exists.

        Default: assume yes when installed. The real auth error will
        surface as a normal subprocess failure on the first send."""
        return self.is_installed()

    def is_configured(self) -> bool:
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


# ---------------------------------------------------------------------------
# Shared subprocess helper
# ---------------------------------------------------------------------------

# Noise the vendor CLIs emit that we drop before showing to the user.
# Match on line prefix (case-sensitive).
_NOISE_LINE_PREFIXES = (
    "Permission deny rule",
    "Permission allow rule",
    "Permission ask rule",
)


def _stream_process(argv: List[str], stdin_text: Optional[str] = None,
                     env_extra: Optional[Dict[str, str]] = None,
                     provider: Optional["ChatProvider"] = None,
                     ) -> Iterator[str]:
    """Spawn a subprocess and yield stdout as it arrives.

    Reads line-by-line so noise-filtering can drop specific warnings
    (e.g. Claude Code's per-file permission-rule reminders from the
    user's ~/.claude/settings.json). Merges stderr into stdout so
    real errors show up inline.

    If `provider` is passed we register the Popen on it so that
    provider.cancel_stream() can terminate the subprocess and unblock
    the caller's iteration. Without this, a stream that hangs on
    a `for line in proc.stdout` read can never be cancelled and the
    worker QThread will outlive its Python reference on quit — which
    is exactly the crash the user reported.
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
            bufsize=1,           # line-buffered
            env=env,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not run {argv[0]!r} — is the CLI installed and on PATH?"
        ) from e

    if provider is not None:
        provider._current_proc = proc

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
            yield line
    finally:
        # Always tear the child down cleanly — cancel_stream() may have
        # already terminated it; ok to call terminate again defensively.
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=1)
        except Exception:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except Exception:
                    proc.kill()
            except Exception:
                pass
        if provider is not None:
            provider._current_proc = None


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


# ---------------------------------------------------------------------------
# Anthropic Claude — via `claude` (Claude Code)
# ---------------------------------------------------------------------------

class ClaudeCliProvider(ChatProvider):
    name = "claude"
    label = "Claude (via Claude Code)"
    cli_name = "claude"
    install_hint = (
        "curl -fsSL https://claude.ai/install.sh | bash   # or "
        "npm install -g @anthropic-ai/claude-code"
    )
    login_command = "claude setup-token"

    def stream_chat(self, messages: List[Dict], system: str = "",
                     model: Optional[str] = None) -> Iterator[str]:
        prompt = _format_conversation(messages, system=system)
        argv = ["claude", "-p", prompt]
        if system:
            argv += ["--append-system-prompt", system]
        if model:
            argv += ["--model", model]
        yield from _stream_process(argv, provider=self)


# ---------------------------------------------------------------------------
# OpenAI ChatGPT — via `codex` (OpenAI Codex CLI)
# ---------------------------------------------------------------------------

class CodexCliProvider(ChatProvider):
    name = "codex"
    label = "ChatGPT (via Codex CLI)"
    cli_name = "codex"
    install_hint = (
        "npm install -g @openai/codex   # or brew install codex"
    )
    login_command = "codex login"

    def stream_chat(self, messages: List[Dict], system: str = "",
                     model: Optional[str] = None) -> Iterator[str]:
        prompt = _format_conversation(messages, system=system)
        argv = ["codex", "exec", prompt]
        if model:
            argv += ["--model", model]
        yield from _stream_process(argv, provider=self)


# ---------------------------------------------------------------------------
# Google Gemini — via `gemini` CLI
# ---------------------------------------------------------------------------

class GeminiCliProvider(ChatProvider):
    name = "gemini"
    label = "Gemini (via Gemini CLI)"
    cli_name = "gemini"
    install_hint = (
        "npm install -g @google/gemini-cli   # or brew install gemini-cli"
    )
    login_command = "gemini"

    def stream_chat(self, messages: List[Dict], system: str = "",
                     model: Optional[str] = None) -> Iterator[str]:
        prompt = _format_conversation(messages, system=system)
        argv = ["gemini", "-p", prompt]
        if model:
            argv += ["-m", model]
        yield from _stream_process(argv, provider=self)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PROVIDERS: List[ChatProvider] = [
    ClaudeCliProvider(),
    CodexCliProvider(),
    GeminiCliProvider(),
]


def list_providers() -> List[ChatProvider]:
    return list(_PROVIDERS)


def configured_providers() -> List[ChatProvider]:
    return [p for p in _PROVIDERS if p.is_configured()]


def get_provider(name: str) -> Optional[ChatProvider]:
    for p in _PROVIDERS:
        if p.name == name:
            return p
    return None
