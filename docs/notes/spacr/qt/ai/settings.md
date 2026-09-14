# Notes from `spacr/qt/ai/settings.py`

Prose lifted out of `spacr/qt/ai/settings.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [get_auto_file_issues](#get_auto_file_issues) (1 entry)
- [get_route_errors_through_ai](#get_route_errors_through_ai) (1 entry)
- [get_console_aware](#get_console_aware) (1 entry)

## Module level

### lines 31-33

```python
"claude": {
```

Claude Code CLI supports --model to pick between Haiku (fast) / Sonnet (balanced) / Opus (deep). Newer builds also honour ``--reasoning-effort low|medium|high`` — safest is model.

### lines 39-41

```python
"codex": {
```

Codex CLI: model picks fast (o4-mini) / balanced (o1-preview) / deep (o1). The exact model IDs may drift; provider falls back to CLI default if the flag is unrecognised.

### lines 47-48  _(unsure)_

```python
"gemini": {
```

Gemini CLI: model picks flash (fast) / pro (balanced) / pro thinking (deep, via same model with --thinking flag).

## get_auto_file_issues

### lines 165-167

```python
def get_auto_file_issues() -> bool:
```

Auto-file GitHub issue on error (opt-in)

## get_route_errors_through_ai

### line 192, trailing  _(unsure)_

```python
raw = _settings().value(_KEY_ROUTE_ERRORS, True)
```

default ON

## get_console_aware

### line 210, trailing  _(unsure)_

```python
raw = _settings().value(_KEY_CONSOLE_AWARE, True)
```

default ON
