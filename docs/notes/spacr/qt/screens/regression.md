# Notes from `spacr/qt/screens/regression.py`

Prose lifted out of `spacr/qt/screens/regression.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [DiagnosticsOpener._folder](#diagnosticsopener_folder) (1 entry)
- [DiagnosticsOpener.verdict](#diagnosticsopenerverdict) (1 entry)
- [DiagnosticsOpener.open](#diagnosticsopeneropen) (1 entry)
- [install_extras](#install_extras) (1 entry)
- [install_folds](#install_folds) (3 entries)

## Module level

### lines 97-104

```python
HITS_TAB_TITLE = "Hits"
```

NONE OF THE THREE HAS A REGISTRY ROW. What each said as a TILE -- the name, the sentence and the maturity colour a button has to go on carrying now that the registry answers "stable" and a title-cased key for all three lives in `spacr.qt.screens.map_barcodes.FOLD_FALLBACK`, because `map_barcodes.fold_description` is what `restate_fold_button` reads, and that is the only table it looks in; `fold_strip.folded_fallback` reaches the same entries by walking the hosts. A second copy stood here, beside these three keys, and nothing consulted it.

### lines 485-489

```python
"investigate_hit": _build_investigate_hit,
```

BOTH STILL HOLD REGISTRY ROWS, unlike the three above. They keep their name, sentence and maturity colour from the registry, so neither needs a `FOLD_FALLBACK` entry -- and both stay reachable from the command palette, which is the route that must cover every module whether or not it has a tile.

### lines 500-506

```python
DIAGNOSTICS_KEY: (
```

SHADOWED SINCE f37f7d553, and kept in step rather than deleted. `map_barcodes.FOLD_FALLBACK` gained a `regression_diagnostics` row that `fold_description` reaches first, so this one no longer decides anything -- but two tables answering the same question must not DISAGREE, and this said "beta" where the registry row and the shared record both say "alpha". A reader comparing them would not know which was current.

## DiagnosticsOpener._folder

### lines 548-551

```python
newest, newest_at = None, -1.0
```

THE NEWEST RUN, not the first found. A project accumulates results folders and the one the user just produced is the one they mean; offering an older one silently would show panels for a fit they are not looking at.

## DiagnosticsOpener.verdict

### lines 595-596

```python
LOG.debug("could not read the diagnostics summary", exc_info=True)
```

A summary that cannot be read is not a verdict. The button still opens the folder, which is where the panels are.

## DiagnosticsOpener.open

### lines 614-618

```python
try:
```

THE REASON, NOT AN EMPTY FOLDER. `ml` writes this file when the backend cannot produce residuals -- RRA ranks rather than fits, so "residual" has no meaning for it. A reader who opens a folder with fewer panels than they expected cannot tell that from a failure.

## install_extras

### lines 693-696

```python
from .hit_list import connect_investigation
```

"Investigate selected…" used to be connected by the registry factory, which ran only while the hit list was still a tile. It is a tab now, so the connection is made where the tab is built -- otherwise the button emits the selected result into nothing.

## install_folds

### lines 757-758  _(unsure)_

```python
install_extras(screen)
```

The panel is prepared BEFORE the buttons exist, so no button can be pressed into a panel that has not got its tab yet.

### lines 765-773

```python
if hasattr(opener, "verdict") and hasattr(button, "set_verdict"):
```

THE BADGE, item 3 of instruction 322. Only the diagnostics opener has a verdict to show; the rest are modules, and a module has no verdict on a run.

Read once, here, rather than on every repaint: it is a file on disk, and a run finishing is what changes it. Failing to read it leaves the button unbadged, which is the same thing a run that has not happened yet shows -- and is right, because in both cases there is no verdict to report.

### line 785

```python
screen._fold_openers = openers
```

The openers outlive this call only because the screen holds them.
