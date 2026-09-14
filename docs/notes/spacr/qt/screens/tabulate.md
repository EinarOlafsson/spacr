# Notes from `spacr/qt/screens/tabulate.py`

Prose lifted out of `spacr/qt/screens/tabulate.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [TabulateScreen.__init__](#tabulatescreen__init__) (4 entries)
- [TabulateScreen.load_path](#tabulatescreenload_path) (1 entry)
- [TabulateScreen.closeEvent](#tabulatescreencloseevent) (1 entry)
- [Module level](#module-level) (1 entry)

## TabulateScreen.__init__

### lines 111-113

```python
self._jobs = JobRunner(self, threaded=threaded, app_key="tabulate")
```

Every table read goes through here, so it never runs on the GUI thread and always shows up in the run registry (and so in the background-activity spinner).

### lines 165-172

```python
from ..preferences import scaled_px
```

SCALED, NOT A DEVICE-PIXEL CONSTANT. This cap exists to stop the settings column eating the figure beside it, and 320 px is the right answer at 100 %% -- and only there. The glyphs inside it double at 200 %% and the box did not, which is the same defect instruction 350 already fixed on UsageBar's fixed 48 px caption column. Measured on Control Charts: the column's own sizeHint wants 586 px at 100 %%, 707 at 125 %% and 1107 at 200 %%, against a cap that stayed 330 in all three.

### lines 189-190  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

### lines 193-195

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## TabulateScreen.load_path

### lines 263-266

```python
self._jobs.cancel()
```

A second load supersedes the first. Without this, switching table twice in quick succession delivers the frames in whatever order the reads happen to finish, and the picker ends up disagreeing with the pivot below it.

## TabulateScreen.closeEvent

### lines 361-364

```python
"""Stop background work and unlink before going away.
```

Abandon an in-flight read rather than let it outlive the screen: Qt aborts the process if a running QThread is destroyed, and a worker that delivers into a closed widget is a use-after-free.

## Module level

### lines 385-389

```python
_ROW = declared_app(APP_KEY)
```

The row this screen puts in the registry is declared in

`spacr.qt.app_catalog`, which is what lets the app be registered without importing this module -- the launch reads the table, not the screen. These read the same row back rather than restating it, so the name, the blurb and the nine translations have one spelling and no second copy to drift from.
