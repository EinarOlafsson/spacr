# Notes from `spacr/qt/__init__.py`

Prose lifted out of `spacr/qt/__init__.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_quiet_vispy_logging](#_quiet_vispy_logging) (1 entry)
- [_install_quiet_qt_logging.handler](#_install_quiet_qt_logginghandler) (2 entries)
- [_quiet_library_warnings](#_quiet_library_warnings) (1 entry)
- [_missing_qt_extra](#_missing_qt_extra) (2 entries)
- [_prefer_a_context_the_shaders_can_run_on](#_prefer_a_context_the_shaders_can_run_on) (3 entries)
- [run](#run) (3 entries)
- [Module level](#module-level) (9 entries)

## _quiet_vispy_logging

### line 121  _(unsure)_

```python
pass
```

vispy is optional; a machine without it has no backdrop to quiet.

## _install_quiet_qt_logging.handler

### lines 154-179

```python
if "cannot be started from another thread" in (message or "") or \
```

AND INTO THE LOG. This handler printed to stderr and nowhere else, so every Qt warning was visible to whoever was watching the terminal and invisible to everyone reading ~/.spacr/logs/spacr.log afterwards. That cost real time on 2026-08-19: "QBasicTimer::start: Timers cannot be started from another thread" arrives immediately before a crash on the maintainer's machine, and the log had ZERO occurrences of it so the one line that mattered could only be obtained by asking them to copy it out of a terminal that the crash had already closed.

A crash report is written from the log, not from a screen someone happened to be looking at. A THREAD-AFFINITY WARNING GETS A PYTHON STACK. `QBasicTimer::start` is called from Qt's own C++ internals, so the Python-level guard on QTimer.start never sees it -- but THIS handler runs in the emitting thread at the moment of the warning, so the stack here names the Python call that entered Qt.

Only for this family. A stack on every Qt warning would bury the one that matters, which is the mistake the guard's own test exists to prevent. "STOPPED" BELONGS HERE TOO, and its absence cost a day. The started/created pair was matched; `killTimer` and `~QObject` say "cannot be STOPPED from another thread" and fell through with no stack -- which is the pair that precedes the cyclic-collector crash spacr.qt.gc_policy documents, so the one crash that most needed a Python stack was the one family that never got one.

### line 206

```python
pass
```

Never let logging a warning become a second failure.

## _quiet_library_warnings

### lines 263-265

```python
if any(action == "ignore" and category is UserWarning
```

A filters entry is (action, message_re, category, module_re, lineno) with the two patterns compiled, so `.pattern` recovers what was asked for and the comparison is against the request rather than the object.

## _missing_qt_extra

### lines 328-329

```python
root = (getattr(exc, "name", None) or "").split(".", 1)[0]
```

ModuleNotFoundError sets `.name` to the module that could not be found; a failed `from PySide6.QtCore import ...` sets it to `PySide6.QtCore`.

### lines 333-334  _(unsure)_

```python
text = str(exc)
```

Import hooks and hand-raised ImportErrors may leave `.name` unset, so fall back to the message text before giving up on the friendly path.

## _prefer_a_context_the_shaders_can_run_on

### line 364, trailing

```python
return
```

the caller chose; do not overrule them

### line 367, trailing  _(unsure)_

```python
return
```

not Wayland; the context is already fine

### line 369, trailing  _(unsure)_

```python
return
```

no XWayland to ask for

## run

### lines 410-414

```python
from . import timing as _timing
```

FIRST IN THE PUBLIC ENTRY POINT.  ``app`` imports PySide, and the registration pass below may import modules that own live hooks.  A clock begun inside ``launch()`` misses both and cannot claim process-to-interactive timing.  The timing module itself is stdlib-only while disabled; begin() is a single environment-guarded return.

### lines 419-422

```python
_quiet_gtk_accessibility()
```

Before anything imports Qt, GTK or torch: the AT-SPI variable is only read while GTK loads, the Qt handler has to be in place before the first widget lays out text, and the warning filter has to be in place before the pipeline preloader reaches cellpose.

### lines 445-447

```python
return launch(argv)
```

Deliberately outside the `try`: an ImportError raised *during* a run — a screen lazily importing an optional reader, say — is a real failure and must not be reported as "Qt is not installed".

## Module level

### lines 476-478

```python
"spacr.qt.chaining",
```

Not an app of its own: it registers a screen FACTORY for every module that declares ports, so the generic AppScreen gains the auto-chaining / staleness / next-step strip without a line inside the shared screen.

### lines 480-484

```python
"spacr.qt.prerun",
```

Also not an app: it decorates the Measure screen with the segmentation verdict seg_qc already computed and the Mask screen with the diameter estimator, by wrapping whatever factory is registered for those two keys. Listed AFTER chaining so the normal launch order composes onto chaining's screen rather than the other way round; both orders work.

### lines 488-492

```python
"spacr.qt.screens.trellis",
```

Three Explore screens built on the Graph Builder's spec engine. Each owns a tested, idempotent register() that fans its name, intro, CLI note, api_module and nine translations out of one register_app call. All that was ever missing was the row that runs it: the agent that wrote them could not add it while this file was being edited.

### lines 496-499

```python
"spacr.qt.screens.outliers",
```

Flags an object — or a whole well, which is the more common failure — as extreme by a robust rule, and writes a COLUMN rather than dropping a row. Safe to have on by default for exactly that reason: nothing it decides is destructive until the user acts on it.

### lines 501-505

```python
"spacr.qt.screens.dose_response",
```

Four-parameter logistic with a confidence interval on EC50 -- and a refusal wherever an EC50 would be a guess: an incomplete curve reports a one-sided bound instead of a number, and non-monotone data is not fitted at all. Filed under Design, with Power: an EC50 is fitted to choose the concentration the next experiment will use.

### lines 508-510

```python
"spacr.qt.screens.control_chart",
```

A control's measured value plate by plate across a campaign, with limits estimated from a STATED baseline and applied forward, so a drift is visible before it has ruined the screen rather than after.

### lines 512-515

```python
"spacr.qt.screens.project_browser",
```

Every project on disk in one table -- stage, size, last run, what is stale -- built entirely on `spacr.projects`, which is built on ports, artifacts, data_manager and chaining. A project the registry has never seen is listed too; that is the case it exists for.

### lines 517-521

```python
"spacr.qt.resource_cleanup",
```

Not an app: it connects the pre-run cleanup to the run registry and performs whatever launch cleanup the chosen spaCR mode asks for. In Balanced — the default — both of those are a preference read and a return, so this row costs a user who never opens the Performance tab nothing at all.

### lines 523-527

```python
"spacr.qt.maturity",
```

Not an app either: it corrects the maturity label on the modules whose evidence no longer matches "alpha". Listed LAST, after every module that registers an app of its own, because it can only reassess apps that are in the registry by the time it runs — a module registered after it would keep whatever stage it declared.
