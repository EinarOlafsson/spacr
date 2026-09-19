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
