# Notes from `spacr/qt/logging_util.py`

Prose lifted out of `spacr/qt/logging_util.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [log_dir](#log_dir) (1 entry)
- [_RecordRelay](#_recordrelay) (1 entry)
- [QtLogHandler.__init__](#qtloghandler__init__) (1 entry)
- [QtLogHandler.emit](#qtloghandleremit) (1 entry)
- [setup_logging](#setup_logging) (2 entries)

## log_dir

### lines 35-38  _(unsure)_

```python
def log_dir() -> Path:
```

Path shims — kept for backwards compatibility with existing callers / tests that import log_dir/log_path from spacr.qt.logging_util.

## _RecordRelay

### lines 56-58  _(unsure)_

```python
class _RecordRelay(QObject):
```

Qt-side log handler — bridges Python logging → Qt signal

## QtLogHandler.__init__

### lines 97-98  _(unsure)_

```python
self.record_ready = self._record_relay.record_ready
```

Keep the existing ``handler.record_ready.connect(...)`` contract; only the QObject that owns the signal changes.

## QtLogHandler.emit

### line 134

```python
self.handleError(record)
```

Never let a logging failure crash the app

## setup_logging

### lines 167-169

```python
_package_setup_logging(level=level, log_file=log_path())
```

Package-scope file handler — installed once, shared by every spacr subsystem. Explicitly pass log_path() so tests that monkey-patch the Qt-side path are honoured.

### line 172  _(unsure)_

```python
qt_h = get_signal_handler()
```

Qt signal handler — only relevant when a QApplication exists.
