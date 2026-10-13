# Notes from `spacr/qt/screens/project_browser.py`

Prose lifted out of `spacr/qt/screens/project_browser.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ProjectBrowserScreen.__init__](#projectbrowserscreen__init__) (3 entries)
- [ProjectBrowserScreen.add_root](#projectbrowserscreenadd_root) (2 entries)
- [ProjectBrowserScreen.choose_root](#projectbrowserscreenchoose_root) (1 entry)
- [ProjectBrowserScreen._start_directory](#projectbrowserscreen_start_directory) (1 entry)
- [ProjectBrowserScreen.rescan](#projectbrowserscreenrescan) (1 entry)
- [ProjectBrowserScreen._fill_table](#projectbrowserscreen_fill_table) (2 entries)
- [ProjectBrowserScreen.closeEvent](#projectbrowserscreencloseevent) (1 entry)
- [make_project_browser_screen](#make_project_browser_screen) (1 entry)
- [Module level](#module-level) (1 entry)

## ProjectBrowserScreen.__init__

### lines 206-208

```python
mark_surface(self._root_list, self._table, self._detail)
```

All three regions of this screen ARE the page: there is no card and no tab pane behind any of them, so without this the sweep leaves them showing the backdrop straight through the text.

### lines 217-218  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

### lines 221-223

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## ProjectBrowserScreen.add_root

### lines 238-244

```python
path_probe.isdir(path)
```

Queue a background check on it, and throw the answer away: what is wanted is the CACHE ENTRY, so that `_start_directory` can offer this folder to the chooser on the next click -- it refuses a folder `path_probe` has never been asked about. `_start_directory` does ask for itself as well, but only once the user has clicked, which is one click too late to answer that click. Asking here is what makes a root added this session usable by the chooser at all.

### lines 251-252

```python
LOG.debug("could not record the recent folder", exc_info=True)
```

Remembering the folder is a convenience; failing to remember it must not cost the scan the user asked for.

## ProjectBrowserScreen.choose_root

### lines 270-282

```python
path = os.path.abspath(os.path.expanduser(str(path)))
```

The dialog only ever returns a folder it has just listed, so this is a fact already in hand rather than an excuse for another stat -- which is exactly what `path_probe.prime` is for. Primed under the SAME spelling `add_root` and `push_recent_source` store, because the cache is keyed on the string: priming `/data/plate/` would leave the `/data/plate` every other screen asks about still unknown, and the prime would buy nothing.

`prime` records the "does it exist?" answer only; the "is it a directory?" answer that :meth:`_start_directory` reads has no priming entry point, and `add_root` queues a real probe for it one line down. That probe is cheap by construction -- the dialog has just listed the folder, so the kernel answers from cache.

## ProjectBrowserScreen._start_directory

### lines 328-330

```python
if path_probe.isdir(root):
```

Asking is also how an unprobed root gets probed: `isdir` queues the check it cannot answer, so a "no" here is what makes the NEXT click a "yes". That is the whole recovery this gate needs.

## ProjectBrowserScreen.rescan

### lines 361-362  _(unsure)_

```python
self._jobs.cancel()
```

A second scan supersedes the first, so clicking Scan twice does not deliver two tables in whatever order the walks happen to finish.

## ProjectBrowserScreen._fill_table

### lines 424-426

```python
self._table.setSortingEnabled(False)
```

Sorting is switched off while rows are inserted: a sorted table re-orders on every setItem, and the row index the next call writes into is then not the row it just wrote.

### lines 443-445

```python
item.setData(Qt.UserRole, summary.root)
```

The root travels with the row so a re-sorted table still selects the project the user clicked rather than the one that happens to be at that index now.

## ProjectBrowserScreen.closeEvent

### lines 533-535

```python
"""Stop background work and unlink before going away.
```

Abandon in-flight work rather than let it outlive the screen: Qt aborts the process if a running QThread is destroyed, and a worker delivering into a closed widget is a use-after-free.

## make_project_browser_screen

### lines 606-619

```python
roots = tuple(dict.fromkeys(
```

This factory runs on the GUI thread -- `MainWindow._build_screen` calls it inline, because Qt forbids building widgets anywhere else and every path here is one the user typed at some other screen. `os.path.isdir` on the maintainer's remembered `/nas_mnt` root had not returned after TWENTY SECONDS on 2026-09-04, so this filter froze the whole window for as long as the automount slept, and opening the browser was reported as a crash with no traceback. `exists(..., want_dir=True)` answers from the probe cache instead, optimistically: `isdir()` would default to False and open the browser empty on the first run of every session, which is the one thing this seeding exists to prevent. Optimism costs nothing here because the seeded roots go straight to `rescan`, whose walk runs on a JobRunner worker -- a root that turns out to be gone is discovered off the GUI thread and simply lists no projects.

## Module level

### lines 628-632

```python
_ROW = declared_app(APP_KEY)
```

The row this screen puts in the registry is declared in

`spacr.qt.app_catalog`, which is what lets the app be registered without importing this module -- the launch reads the table, not the screen. These read the same row back rather than restating it, so the name, the blurb and the nine translations have one spelling and no second copy to drift from.
