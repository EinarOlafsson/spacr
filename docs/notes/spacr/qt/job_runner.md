# Notes from `spacr/qt/job_runner.py`

Prose lifted out of `spacr/qt/job_runner.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [JobRunner.__init__](#jobrunner__init__) (1 entry)
- [JobRunner.submit](#jobrunnersubmit) (4 entries)
- [JobRunner._relay](#jobrunner_relay) (1 entry)
- [JobRunner._on_settled](#jobrunner_on_settled) (1 entry)

## JobRunner.__init__

### lines 139-141

```python
_LIVE_RUNNERS.add(self)
```

REGISTERED THE MOMENT IT EXISTS, so a runner cannot be created and then missed by the quit-time drain. Weak, so this holds nothing alive that Qt would otherwise collect.

## JobRunner.submit

### lines 186-189

```python
thread, worker = make_thread(
```

journal=False: this is read-only UI housekeeping, not an analysis run. A reproducibility record for "the user opened a table" is noise, and `RunRegistry.cancel_all` treats journalled jobs as a reason to refuse to close the application.

### lines 194-196

```python
self._jobs[job_id] = (thread, worker)
```

Strong references. PySide6 does not keep the worker alive through the started->run connection alone, and a collected worker means the thread spins forever without ever calling run().

### lines 200-204

```python
worker.finished.connect(
```

A closure -- deliberately. `worker.finished` is emitted on the WORKER thread, and all this one does is call `_relay`, which re-emits a Signal; emitting is safe from any thread. The Signal's receiver (`_on_settled`) is a bound method of this GUI-thread object, so Qt queues the real work back onto the GUI thread.

### lines 207-208

```python
thread.finished.connect(self._retire_finished_jobs)
```

A BOUND METHOD -- and the contrast with the line above is the whole point. See the module docstring.

## JobRunner._relay

### lines 235-240

```python
self._pending.pop(job_id, None)
```

The runner's C++ half died with its parent before this worker finished.  No queued receiver remains to retire the pending result, but the Python closure still owns ``self`` and may clear its bookkeeping safely under the GIL.  Do not drop ``_jobs`` here: its strong references keep the QThread alive until the worker has actually stopped.

## JobRunner._on_settled

### lines 251-253

```python
if ok and on_done is not None and generation == self._generation:
```

Bookkeeping happens for every job; only *use* of the result is conditional. A cancelled load that skipped this would leave the runner permanently busy.
