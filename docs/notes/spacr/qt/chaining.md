# Notes from `spacr/qt/chaining.py`

Prose lifted out of `spacr/qt/chaining.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ChainingBar.__init__](#chainingbar__init__) (1 entry)
- [ChainingBar.__init__._root_answered](#chainingbar__init___root_answered) (1 entry)
- [ChainingBar.__init__._let_go](#chainingbar__init___let_go) (2 entries)
- [ChainingBar._capture_edits](#chainingbar_capture_edits) (1 entry)
- [ChainingBar.search_roots](#chainingbarsearch_roots) (1 entry)
- [ChainingBar._refresh](#chainingbar_refresh) (2 entries)
- [ChainingBar._paint](#chainingbar_paint) (2 entries)
- [ChainingBar.steps](#chainingbarsteps) (1 entry)
- [install_chaining](#install_chaining) (1 entry)
- [Module level](#module-level) (1 entry)
- [register](#register) (1 entry)

## ChainingBar.__init__

### lines 253-255

```python
from . import path_probe as _probe
```

A root skipped because the probe had not answered yet comes back when it does -- otherwise `search_roots` would drop it for the life of the screen and the strip would silently stop chaining.

## ChainingBar.__init__._root_answered

### line 272, trailing  _(unsure)_

```python
pass
```

the strip has gone; the signal outlived it

## ChainingBar.__init__._let_go

### lines 276-280

```python
def _let_go(*_args) -> None:
```

AND DISCONNECTED WHEN THE STRIP GOES. `probes` is process-wide and outlives any one screen, so a connection left behind is a signal delivered to a Python wrapper whose C++ half has been deleted which raises out of whatever happened to emit it. `destroyed` fires while the wrapper is still usable, which is the moment to let go.

### line 293, trailing  _(unsure)_

```python
pass
```

already disconnected, or the source is gone

## ChainingBar._capture_edits

### lines 448-452

```python
if key in self._seen:
```

Empty is only a *clearing* if the field held something first. On the first refresh of a fresh screen it just means nobody has typed anything yet, and unpinning there would throw away the path the user chose in a previous session — the exact promise this store exists to keep.

## ChainingBar.search_roots

### lines 638-643

```python
if path_probe.isdir(candidate):
```

BELT AND BRACES, on top of running this off the GUI thread. `isdir` answers False for a root it has not probed yet, which is the pessimistic direction and the right one here: skipping a root costs one refresh, and the probe signal below brings it back the moment the answer lands. Stating it costs however long a sleeping mount takes.

## ChainingBar._refresh

### lines 688-689  _(unsure)_

```python
self._resolve_again = bool(self._resolve_again) or finished
```

One question, asked once. `finished` is sticky so a completed run's next-step offer is not lost to a coalesced refresh.

### lines 728-729  _(unsure)_

```python
self._resolving = False
```

The runner refused -- shutting down, or already busy. The strip simply does not update, which is what `refresh` promises.

## ChainingBar._paint

### lines 736-740

```python
widgets = self._widgets()
```

Everything the resolution decided — a restored pin as much as a chained default — goes into the field, but only where the field is empty. That single rule is what makes a pin survive a restart (the widget starts on its placeholder and the pin fills it) while never overwriting anything the user can see.

### lines 761-764

```python
self.setVisible(any(not w.isHidden() for w in (
```

``isHidden`` and not ``isVisible``: a widget whose window has not been shown yet is not *visible*, so asking that question during the screen's construction would answer "nothing to say" every time and latch the strip hidden for the life of the screen.

## ChainingBar.steps

### line 867  _(unsure)_

```python
@property
```

introspection, for tests and for the next module's Continue

## install_chaining

### lines 946-957

```python
try:
```

SWEEP THE STRIP'S OWN CONTAINERS. The screen was themed when it was built, and this arrives afterwards -- so the page-surface sweep that ran then never saw the rows inside it. An anonymous QWidget holding a layout inherits the blanket `QWidget { background-color: bg }` rule and paints the WINDOW colour, which is not a surface and which no opacity setting can reach. That is the black box the user reported behind the pinned-input row and its "Use it" button, directly above Run.

The bar itself is a QFrame and is deliberately NOT swept: it is a component that paints on purpose. Only the scaffolding inside it is tagged, which is the same rule the screen sweep uses.

## Module level

### lines 1024-1029

```python
"timelapse": "mask",
```

Timelapse is the mask pipeline with tracking on, and the GUI folded it into Mask Generation as a settings category with a switch. The port graph still declares it because `spacr-run timelapse` still runs it, so a chain whose next step is timelapse must offer the screen that now carries it -- otherwise "what comes next" simply stops mentioning a step that is perfectly runnable.

## register

### lines 1115-1117

```python
continue
```

Somebody else owns this screen — a plugin, or a module that ships its own. Theirs wins; a strip is not worth overriding a whole screen for.
