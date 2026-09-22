# Notes from `spacr/qt/ai/worker.py`

Prose lifted out of `spacr/qt/ai/worker.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [StreamWorker.run](#streamworkerrun) (2 entries)
- [make_stream_thread](#make_stream_thread) (2 entries)
- [StreamWorker.run, 2026-09-19](#streamworkerrun-2026-09-19) (1 entry)

## StreamWorker.run

### lines 89-91

```python
tb = traceback.format_exc()
```

BaseException — even a KeyboardInterrupt during a blocking network call should let the UI recover instead of leaving _thread wedged forever.

### line 93  _(unsure)_

```python
try:
```

Print to real stderr so users can see it while we iterate.

## make_stream_thread

### line 150, trailing

```python
worker.setParent(None)
```

worker moves to thread, no parent

### lines 154-155

```python
thread.finished.connect(thread.deleteLater, Qt.QueuedConnection)
```

The QThread is GUI-affine, so its deferred delete is flushed by the GUI thread's own loop. That one is safe, and it is the only one.

## StreamWorker.run, 2026-09-19

```python
if self._cancelled:
```

A reader whose child was ended can raise before its first chunk, for example on a closed pipe. When the user pressed Cancel, `finished` says `Cancelled.` rather than quoting that exception as the provider's failure. `_stream_process` already declines to raise for a child spaCR stopped (see `providers.md`, 2026-09-19); this covers any other exception a provider raises on the way out.

### added 2026-09-19 (432)

```python
if not isinstance(e, ProviderFailed) or not detail:
```

THE CLASS NAME IS PREFIXED ONLY WHEN IT SAYS SOMETHING. What `finished` carries here is written to the console after "[AI error] ", and `ProviderFailed`'s whole message is composed to be read there -- which CLI stopped, with what status, what it printed, and the command that signs it back in. Prefixing the class turned that into "[AI error] ProviderFailed: claude stopped with exit status 1: Failed to authenticate ... run `claude auth login`", putting a Python class name in front of the one sentence the user can act on. It was the shape 432 shipped and the last nit its trailing note left.

Every other exception keeps its class, deliberately: a bare `[Errno 2] No such file or directory` does not say it is a `FileNotFoundError`, and for most exceptions the type is the only part that names the fault. `ProviderFailed` is the exception because its message already does. An empty message falls back to the prefixed form rather than emitting "[AI error] " over nothing.

Held by `test_a_failed_provider_finishes_not_ok` (the sentence, and no class name) and `test_any_other_failure_keeps_the_class_that_names_it`.
