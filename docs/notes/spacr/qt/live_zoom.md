# Notes from `spacr/qt/live_zoom.py`

Prose lifted out of `spacr/qt/live_zoom.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [_scaled_font](#_scaled_font) (1 entry)
- [LiveZoomFilter.__init__](#livezoomfilter__init__) (1 entry)
- [LiveZoomFilter.eventFilter](#livezoomfiltereventfilter) (2 entries)
- [LiveZoomFilter._wheeled](#livezoomfilter_wheeled) (1 entry)
- [LiveZoomFilter._begin](#livezoomfilter_begin) (1 entry)
- [LiveZoomFilter._apply](#livezoomfilter_apply) (1 entry)
- [LiveZoomFilter._announce](#livezoomfilter_announce) (1 entry)
- [LiveZoomFilter.settle](#livezoomfiltersettle) (3 entries)

## Module level

### lines 52-55

```python
from PySide6.QtWidgets import QApplication
```

LOCAL, NOT AT IMPORT TIME. `spacr.qt.i18n` pulls the catalogs in, and this module is on the startup path; the readout is the only thing here that needs a translated string, so the import lives in the one function that uses it rather than costing every launch.

## _scaled_font

### lines 142-143

```python
return None
```

A font with neither a pixel nor a point size is unresolved; leaving it alone lets it keep inheriting rather than pinning it at a guess.

## LiveZoomFilter.__init__

### lines 184-186

```python
self._settle_timer.timeout.connect(
```

NOT `self.settle` directly: the wheel going quiet ends the gesture but does NOT end the hold, and the two are different answers to "is Z still down?".

## LiveZoomFilter.eventFilter

### lines 204-207

```python
if self._held and self._is_the_key(event) \
```

X11 sends a release/press PAIR for every auto-repeat tick, so a filter that trusts KeyRelease disarms itself a few hundred milliseconds into the hold and the rest of the gesture scrolls the list instead.

### lines 215-216

```python
self.settle()
```

Alt-tabbing away while Z is down means the KeyRelease is delivered to another application and never arrives here.

## LiveZoomFilter._wheeled

### lines 275-279

```python
target = round(target, 4)
```

ROUNDED, because this number is read back with `int(x * 100)`. Four notches down from 1.0 in binary floating point is 0.7999999999999998, which the Preferences slider truncates to 79 % -- the gesture and the control disagreeing by a percent for no reason a user could ever discover.

## LiveZoomFilter._begin

### lines 305-310

```python
self._baseline = [(w, QFont(w.font()), w.testAttribute(Qt.WA_SetFont))
```

WA_SetFont says whether the font on the widget is the widget's own or one QSS resolved onto it -- Qt sets the attribute in `QWidget::setFont` and NOT in the style sheet's font pass. The settle needs the difference: putting a QSS-derived font back with `setFont` would pin it, and a pinned font outlives the sheet that was supposed to own it.

## LiveZoomFilter._apply

### lines 341-342  _(unsure)_

```python
continue
```

Deleted mid-gesture -- a dialog the user closed, a screen that rebuilt itself. Ordinary, not exceptional.

## LiveZoomFilter._announce

### lines 381-382

```python
return
```

The window went away mid-gesture, or carries no real status bar. Feedback is not worth an exception on the input path.

## LiveZoomFilter.settle

### lines 413-420

```python
for widget, base, own in self._baseline:
```

PUT EACH WIDGET BACK THE WAY IT WAS HELD, not merely back at the old size. A widget whose font came from the sheet has to end this with no font of its own again: `setFont` sets WA_SetFont, and the only rule that then still reaches it is one the sheet names explicitly -- so a widget outside the blanket QWidget rule would keep this gesture's size for the rest of the session. A widget that really did own its font (the console's monospace, the AI toggle's size) gets that font back verbatim.

### lines 433-444

```python
from .preferences import apply_preferences_to_app, set_font_scale
```

RE-STYLE EVEN WHEN THE SCALE CAME BACK TO WHERE IT STARTED, and this is a fix rather than tidiness. Clearing a QSS-dressed widget with `setFont(QFont())` above leaves it INHERITING rather than styled -- the sheet's font-size does not come back until something re-polishes it. Returning here because the number happened to be unchanged left every widget the gesture touched with no styled font at all: measured, Z + one notch up + one notch down release took the visible widgets from 13 px to unset, with nothing scheduled to repair them.

`set_font_scale` is still skipped in that case -- there is nothing to persist -- but the re-polish is not optional.

### lines 452-456

```python
if window is not None and _alive(window):
```

Icons, tile geometry and the dock are rebuilt from Python rather than from QSS; only the window knows how. Just the one window the wheel was over -- walking topLevelWidgets() reaches windows whose C++ side is already being torn down, and rebuilding one of those segfaults rather than raising (see `_refresh_owner_window`).
