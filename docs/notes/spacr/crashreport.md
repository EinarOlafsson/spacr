# Notes from `spacr/crashreport.py`

Prose lifted out of `spacr/crashreport.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_collect](#_collect) (1 entry)
- [_tail](#_tail) (1 entry)
- [collect.environment](#collectenvironment) (1 entry)
- [write_crash_report](#write_crash_report) (1 entry)
- [Module level](#module-level) (1 entry)

## _collect

### lines 243-247

```python
if isinstance(exc, KeyboardInterrupt):
```

BaseException rather than Exception: a collector that trips a MemoryError or a RecursionError must still leave a report behind, and re-raising here would lose every section gathered before it. KeyboardInterrupt is re-raised, because a user pressing Ctrl-C while a report is being written means stop, not "record that".

## _tail

### lines 278-279

```python
handle.readline()
```

The seek lands mid-line; drop the partial first line rather than present half a message as a whole one.

## collect.environment

### lines 685-687

```python
redacted = _redacted_environment()
```

Inside the collector, not beside it. Everything in this function that runs outside _collect is a way for the whole bundle to be lost, and two of them were found here by the tests that say so.

## write_crash_report

### lines 741-742  _(unsure)_

```python
archive.writestr("summary.txt", report.summary())
```

summary first, manifest last: the two files a reader wants at the top and the bottom of the listing.

## Module level

### line 878, trailing  _(unsure)_

```python
if __name__ == "__main__":
```

the module entry point
