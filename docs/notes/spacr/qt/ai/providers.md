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

NOT JUST THE curl FORM, which is what was asked for: install.sh refuses Windows outright ("Windows is not supported by this script"), so the npm form is the only one that works there. Choosing by platform gives every reader exactly one command that is whole and correct for them.

## GeminiCliProvider.stream_chat

### line 423  _(unsure)_

```python
args = ai_settings.provider_args(self.name)
```

SPEED_MAP uses --model; translate to -m for the gemini CLI
