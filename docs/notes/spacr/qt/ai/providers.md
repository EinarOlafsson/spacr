# Notes from `spacr/qt/ai/providers.py`

Prose lifted out of `spacr/qt/ai/providers.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ChatProvider](#chatprovider) (5 entries)
- [ChatProvider.__init__](#chatprovider__init__) (1 entry)
- [Module level](#module-level) (1 entry)
- [_stream_process](#_stream_process) (2 entries)
- [ClaudeCliProvider](#claudecliprovider) (1 entry)
- [GeminiCliProvider.stream_chat](#geminicliproviderstream_chat) (1 entry)
- [_stream_process and ProviderFailed, 2026-09-19](#_stream_process-and-providerfailed-2026-09-19) (1 entry)

## ChatProvider

### line 47, trailing

```python
name: str = ""
```

short id: "claude" / "codex" / "gemini"

### line 48, trailing  _(unsure)_

```python
label: str = ""
```

human-readable label

### line 49, trailing  _(unsure)_

```python
cli_name: str = ""
```

the executable on PATH

### line 50, trailing  _(unsure)_

```python
install_hint: str = ""
```

shell one-liner to install

### line 51, trailing  _(unsure)_

```python
login_command: str = ""
```

shell one-liner the user should run

## ChatProvider.__init__

### lines 54-56

```python
"""Create the provider with no child process running.
```

Tracks the currently-running child process so cancel_stream() can actually terminate it — otherwise `for line in proc.stdout` blocks indefinitely and the worker thread never exits.

## Module level

### lines 119-120  _(unsure)_

```python
_NOISE_LINE_PREFIXES = (
```

Noise the vendor CLIs emit that we drop before showing to the user. Match on line prefix (case-sensitive).

## _stream_process

### line 240, trailing  _(unsure)_

```python
bufsize=1,
```

line-buffered

### lines 265-266  _(unsure)_

```python
try:
```

Always tear the child down cleanly — cancel_stream() may have already terminated it; ok to call terminate again defensively.

## ClaudeCliProvider

### lines 316-326

```python
install_hint = (
```

ONE COMMAND, WHOLE, PER PLATFORM. It was a single line carrying both forms joined by "# or", which pastes correctly -- the shell comments the rest away -- and copies badly: the maintainer took the curl half without `| bash` on 2026-09-10, and curl then printed the installer to the terminal instead of running it. Nothing failed and nothing installed, which is the worst shape a copied command can have.

CURL ON EVERY SYSTEM, WINDOWS INCLUDED (item 414, 2026-09-15). Windows used to get `npm install -g @anthropic-ai/claude-code`, because install.sh refuses Windows outright ("Windows is not supported by this script"). That form fails with "'npm' is not recognized" on any machine without Node.js, which is what the maintainer met on Windows. Windows now gets Anthropic's own CMD installer from https://code.claude.com/docs/en/setup (retrieved 2026-09-15), `curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd`, run as `cmd /c "..."`. The wrapper is the point: pasted into PowerShell the bare command fails twice, because `curl` can be an alias for Invoke-WebRequest there and Windows PowerShell 5.1 rejects `&&`. Named as a `cmd /c` command, the one copied line runs whole from CMD or PowerShell. Windows 10 (1803+) and 11 ship curl.exe, and install.cmd ends with `exit /b`, so the trailing `del` still runs. macOS and Linux keep the documented `curl -fsSL https://claude.ai/install.sh | bash`. `tests/qt/test_claude_code_install_hint.py` pins both.

## GeminiCliProvider.stream_chat

### line 423  _(unsure)_

```python
args = ai_settings.provider_args(self.name)
```

SPEED_MAP uses --model; translate to -m for the gemini CLI

## _stream_process and ProviderFailed, 2026-09-19

```python
if (isinstance(exit_status, int) and exit_status != 0
```

A provider CLI that exits non-zero has failed, and the line it printed is its error message, not an answer. Before this, `_stream_process` never looked at the exit status, so the worker reported success with that line as the reply. GitHub #117 (jak18015, macOS, 1.5.0.8) is the case: `claude` printed `Failed to authenticate: OAuth session expired and could not be refreshed` and exited, and the console showed it as spaCR AI's answer, with no `[AI error]` and no hint to sign in again. The console kept it as the explanation of the crash, so `ai_explanation_of` handed it to the bug reporter, and issues #118 and #121 were filed with that line under "spaCR AI's analysis of this error".

Measured 2026-09-19 with Claude Code 2.1.274 in an empty HOME: `claude -p "say hi"` prints `Not logged in · Please run /login` and exits 1. The expired-session message in #117 needs an expired session, which could not be produced here, so its exit status is not measured. The check depends only on the status, never on the wording.

The lines are still streamed as they arrive, because a long answer has to show while it is written. The failure is raised after the last line, once the child has been reaped. The message quotes the last three non-blank lines, cut to 400 characters. When a provider is given, it also names that provider's `login_command`. The command is the one the Providers dialog already shows, so the two places agree.

A child that spaCR itself ended does not count as failing. Cancel, quitting through `terminate_all_streams`, and this function's own escalation all mark the Popen before they signal it. The status such a child exits with is spaCR's doing. On POSIX it is a negative signal number. On Windows it is 1, because `Popen.terminate` is `TerminateProcess(handle, 1)`, so it looks exactly like a CLI reporting a failure. The mark is the only way to tell the two apart.
