# Notes from `spacr/qt/screens/trellis.py`

Prose lifted out of `spacr/qt/screens/trellis.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [TrellisScreen.__init__](#trellisscreen__init__) (3 entries)
- [TrellisScreen.load_path](#trellisscreenload_path) (1 entry)
- [TrellisScreen.closeEvent](#trellisscreencloseevent) (1 entry)
- [Module level](#module-level) (1 entry)

## TrellisScreen.__init__

### lines 124-131

```python
from ..preferences import scaled_px
```

SCALED, NOT A DEVICE-PIXEL CONSTANT. This cap exists to stop the settings column eating the figure beside it, and 360 px is the right answer at 100 %% -- and only there. The glyphs inside it double at 200 %% and the box did not, which is the same defect instruction 350 already fixed on UsageBar's fixed 48 px caption column. Measured on Control Charts: the column's own sizeHint wants 586 px at 100 %%, 707 at 125 %% and 1107 at 200 %%, against a cap that stayed 330 in all three.

### lines 143-144  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

### lines 147-149

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## TrellisScreen.load_path

### lines 214-215  _(unsure)_

```python
self._jobs.cancel()
```

A second load supersedes the first, so switching table twice does not deliver the frames in whatever order the reads happen to finish.

## TrellisScreen.closeEvent

### lines 284-285

```python
"""Let the panel close first, so it can unlink its canvas.
```

Abandon an in-flight read rather than let it outlive the screen: Qt aborts the process if a running QThread is destroyed.

## Module level

### lines 300-304

```python
_ROW = declared_app(APP_KEY)
```

The row this screen puts in the registry is declared in

`spacr.qt.app_catalog`, which is what lets the app be registered without importing this module -- the launch reads the table, not the screen. These read the same row back rather than restating it, so the name, the blurb and the nine translations have one spelling and no second copy to drift from.
