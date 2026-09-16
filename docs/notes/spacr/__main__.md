# Notes from `spacr/__main__.py`

Prose lifted out of `spacr/__main__.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## main

### lines 73-76

```python
if args.command in ("gui", "mask", "measure", "classify", "annotate",
```

EVERY WINDOW COMMAND OPENS THE Qt APPLICATION. The seven Tk screens these used to start are tabs in it, so a script that still says `python -m spacr mask` lands on the Mask tab rather than failing to import a module that no longer exists.

### lines 81-82

```python
key = _APP_KEYS.get(args.command)
```

`run` takes the argv the launcher would have had, and its first positional IS the screen to open on.

### lines 86-88

```python
parser.error(f"Unknown command: {args.command}")
```

`parser.error` is annotated NoReturn and raises SystemExit(2); a `return 2` after it is unreachable, and an unreachable line is a line no test can ever justify.
