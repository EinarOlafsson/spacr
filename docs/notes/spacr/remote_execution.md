# Notes from `spacr/remote_execution.py`

Prose lifted out of `spacr/remote_execution.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [RemoteJobManager.submit](#remotejobmanagersubmit) (1 entry)
- [RemoteJobManager.refresh](#remotejobmanagerrefresh) (1 entry)

## RemoteJobManager.submit

### lines 1413-1415

```python
job_dir = self.jobs.path.parent / "jobs" / job_id
```

Keep settings beside the selected JobStore.  Besides making custom installations coherent, this ensures a portable/test store never leaks files into the user's normal state directory.

## RemoteJobManager.refresh

### lines 1472-1474

```python
job.error = f"{type(exc).__name__}: {exc}"
```

A transient SSH/cloud outage must not turn a still-running remote job into a permanent failure.  Preserve its prior state and make the polling error visible.
