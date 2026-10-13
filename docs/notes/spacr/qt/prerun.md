# Notes from `spacr/qt/prerun.py`

Prose lifted out of `spacr/qt/prerun.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_qss](#_qss) (2 entries)
- [Module level](#module-level) (1 entry)
- [_JobMixin._start_job](#_jobmixin_start_job) (1 entry)
- [SegQCBanner.__init__](#segqcbanner__init__) (2 entries)
- [SegQCBanner.refresh](#segqcbannerrefresh) (3 entries)
- [SegQCBanner._on_refreshed](#segqcbanner_on_refreshed) (2 entries)
- [SegQCBanner._draw](#segqcbanner_draw) (1 entry)
- [SegQCBanner._draw_findings](#segqcbanner_draw_findings) (1 entry)
- [SegQCBanner._on_copy_clicked](#segqcbanner_on_copy_clicked) (1 entry)
- [SegQCBanner._on_score_clicked](#segqcbanner_on_score_clicked) (3 entries)
- [SegQCBanner._on_scored](#segqcbanner_on_scored) (2 entries)
- [DiameterPanel.__init__](#diameterpanel__init__) (1 entry)
- [DiameterPanel._draw_rows](#diameterpanel_draw_rows) (1 entry)
- [DiameterPanel.apply](#diameterpanelapply) (1 entry)
- [install_qc_banner](#install_qc_banner) (1 entry)
- [install_diameter_panel](#install_diameter_panel) (1 entry)
- [register](#register) (1 entry)

## _qss

### lines 133-152

```python
def _qss(palette: Dict[str, Any], opacity: Any) -> str:
```

Styling

The block is registered at IMPORT time, at the bottom of this section, and `spacr.qt.prerun` is listed in `theme.WIDGET_QSS_MODULES`. Both halves are required and neither is optional -- see INVARIANTS 1.

It used to be registered only from `register()`, which runs after app.py has imported. The application stylesheet is built and applied before that, so `QFrame#MeasureQCBanner` was not in the sheet when the sheet was made: the panel fell through to the blanket `QWidget { background-color: bg }` and `bg` is #000000 on the dark theme. The verdict text sat on a solid black slab while every container around it was translucent -- which is exactly the symptom INVARIANTS 1 describes, arrived at by a different route.

Measured on a fresh interpreter: 'MeasureQCBanner' in theme.stylesheet() was False at launch and True only after register_self_registering_modules().

### lines 164-185

```python
surface = palette["surface_alt"]
```

Straight off the palette, which is what a REGISTERED block is handed: `register_widget_qss` documents that `surface`, `surface_alt` and `surface_hi` arrive already rendered through the user's page opacity, so this is the value the built-in rules interpolate and the panel matches the app by construction.

It used to call `pane_surface("surface_alt", palette.get("theme"), opacity)`. Two things were wrong with that and neither was visible while this block was missing from the sheet:

the palette carries no "theme" key, so that argument was always None, and `opacity` is None for a registered block -- so pane_surface fell through to reading the LIVE preference. A stylesheet that reads live preferences is the thing `test_the_sheet_does_not_read_the_live_page_opacity` exists to forbid; it emitted rgba() on the opaque themes, which have no scrim, so the panel carried a translucency the theme never authorised (`test_opaque_themes_still_emit_plain_hex`).

Both tests were green only because the block was not reaching the sheet they inspect.

## Module level

### lines 236-237

```python
LOG.exception("could not register the pre-run stylesheet at import")
```

INVARIANTS 10: a stylesheet that cannot be registered costs this panel its background, not the Measure module its run.

## _JobMixin._start_job

### lines 447-449

```python
thread, worker = make_thread(fn, box, app_key=app_key,
```

journal=False: this is read-only UI housekeeping, not an analysis run, and a reproducibility manifest per button press would bury the runs that are.

## SegQCBanner.__init__

### lines 579-588

```python
self._findings_box = _transparent(QWidget(self))
```

Scaffolding, so it must paint nothing (INVARIANTS 3). A plain QWidget used as a layout container inherits the blanket `QWidget { background-color: bg }` rule, and `bg` is the WINDOW colour -- #000000 on the dark theme. The findings text sat on a solid black rectangle inside a panel that was otherwise a translucent surface, which is exactly what it looked like: a black box behind the text.

The panel's own background already follows the page opacity

(`pane_surface` in `_qss`); this is what lets it show through.

### lines 628-637

```python
self.hide()
```

HIDDEN UNTIL THERE IS SOMETHING TO SAY. `install_qc_banner` now SCHEDULES the first read instead of doing it inline -- that is the freeze fix -- so for the 450 ms of the debounce the banner would otherwise sit in the layout visible and empty: a title, two buttons and no verdict, on every Measure screen build, appearing and vanishing. Measured against HEAD: with no src, HEAD had it hidden from the start and the working tree showed it for 1.2 s.

Every path that has something to draw calls `show()` itself, so this only removes the flash.

## SegQCBanner.refresh

### lines 752-754

```python
box: Dict[str, Any] = {
```

Both cache fields are read HERE and compared on the worker. The job body may not read the banner's state: by the time it runs, the GUI thread may have changed it.

### lines 770-777

```python
self._refresh_again = True
```

COALESCED -- neither dropped nor queued. Twenty keystrokes are twenty requests for one answer: a job each would ask the same question twenty times, and dropping them loses the last one, which is the only one that matters. `_pending_work` runs exactly one catch-up when the slot frees. This is `ChainingBar._refresh`'s `_resolve_again`, and it replaces a re-armed debounce that re-asked every 450 ms for as long as a sleeping mount took to answer.

### lines 782-783

```python
if not self._start_job(self._refresh_job, box, self._on_refreshed,
```

user_visible=False: nobody asked for this, and a job that claims a run banner would put "measure - running" on Home for a CSV read.

## SegQCBanner._on_refreshed

### lines 815-824

```python
return
```

THE ANSWER TO A QUESTION NOBODY IS ASKING. The src field has moved on since this read was issued, so painting it would put the previous source's verdict on screen under the current source's name -- and after a CLEARED field would un-hide a banner `refresh` had just hidden. The catch-up below asks what is actually outstanding.

`_job_settled` is not generation-guarded and cannot be: it is shared with the diameter panel. The guard belongs here, where what was asked is known.

### lines 836-837  _(unsure)_

```python
digest = self._digest
```

The cards on disk are the ones already parsed. This is the whole point of the fingerprint: ten visits, one parse.

## SegQCBanner._draw

### lines 889-890

```python
self._close_field_browser()
```

A refreshed scorecard can point at a different set of fields.  Do not leave an already-open browser navigating the previous digest.

## SegQCBanner._draw_findings

### lines 962-965

```python
block = _transparent(QWidget(self._findings_box))
```

One per finding, and each is its own anonymous QWidget, so each needs tagging: making only the parent transparent left a black rectangle behind every finding's text -- which is what the first attempt at this fixed and what the user still saw.

## SegQCBanner._on_copy_clicked

### line 1109

```python
def _on_copy_clicked(self) -> None:
```

the one expensive path, and only on request

## SegQCBanner._on_score_clicked

### lines 1146-1147

```python
self._score_again = False
```

Includes a queued click whose source has since been cleared:

put the button back rather than leave it disabled forever.

### line 1153, trailing  _(unsure)_

```python
return
```

a scoring pass is already running

### lines 1156-1158

```python
self._title.setText("Segmentation QC — scoring the masks…")
```

The caption the pass itself uses. Scoring is what happens next and it needs no further input, so a second wording for the same state would only be a second thing to read.

## SegQCBanner._on_scored

### line 1198, trailing  _(unsure)_

```python
self._cache_key = None
```

the cards on disk have just changed

### lines 1203-1205

```python
self._pending_work()
```

A read asked for while this pass held the slot. It is worth running even now: `src` may have changed under the scoring pass, and this is what puts the current source back on screen.

## DiameterPanel.__init__

### line 1285  _(unsure)_

```python
self._rows_box = _transparent(QWidget(self))
```

Same as the QC panel's findings box: scaffolding paints nothing.

## DiameterPanel._draw_rows

### lines 1441-1443

```python
line.addWidget(_label(
```

The evidence, not just the number. A proposal whose object count is 3 is a different claim from one whose count is 800, and the user cannot tell them apart from the value alone.

## DiameterPanel.apply

### lines 1483-1488

```python
value = int(round(float(est.diameter)))
```

An int, because `spacr.settings.expected_types` declares these keys int and `collect()` hands a float straight back as the *string* "24.0" — which then reaches check_settings as a string. Sub-pixel precision is meaningless here anyway: the value's only effect is a 30/diameter rescale. The panel still SHOWS the measured value to a decimal, so nothing about the measurement is hidden.

## install_qc_banner

### lines 1580-1583

```python
banner.schedule_refresh()
```

SCHEDULED, NOT CALLED. This runs inside `MainWindow._build_screen`, which does not yield -- so anything done here is done before the screen can be painted, and until 2026-09-04 that included statting the user's src folder. See `SegQCBanner.refresh`.

## install_diameter_panel

### line 1605  _(unsure)_

```python
return None
```

A screen with no diameter to set has no use for an estimate.

## register

### lines 1703-1706

```python
from .theme import register_widget_qss
```

Already registered at import (see the Styling section). Repeated here because `teardown()` unregisters it, so a register/teardown/ register cycle -- which the tests do -- has to put it back. `replace=True` makes the ordinary case a no-op.
