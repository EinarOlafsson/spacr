# Notes from `spacr/qt/command_palette.py`

Prose lifted out of `spacr/qt/command_palette.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [CommandPalette.__init__](#commandpalette__init__) (1 entry)
- [CommandPalette._collect_commands](#commandpalette_collect_commands) (6 entries)
- [CommandPalette._render](#commandpalette_render) (1 entry)
- [CommandPalette.keyPressEvent](#commandpalettekeypressevent) (1 entry)

## CommandPalette.__init__

### line 75  _(unsure)_

```python
from .preferences import scaled_px
```

Frameless-ish look — big centred dialog on top of the app.

## CommandPalette._collect_commands

### lines 128-130

```python
apps = []
```

`apps` is empty on this path, so the Apps loop below never runs and `app_stage` is never reached — only `app_is_visible` is, from the recent-runs loop, so only it needs a stand-in.

### lines 139-143

```python
localized_name = tr(name)
```

The badge is the app's category — the same single grouping

Home and the sidebar use, now that maturity is a colour rather than a second set of sections. How finished an app is is a KEYWORD instead: "alpha" is a useful thing to be able to type, and a useless thing to sort a list by.

### line 174  _(unsure)_

```python
self._commands.append(Command(
```

Providers dialog

### line 183  _(unsure)_

```python
self._commands.append(Command(
```

Cheat sheet

### lines 211-213

```python
self._collect_settings_commands()
```

Settings of the module on screen. Without these the palette answered "which app?" and nothing else, while the thing a user is most often hunting for is one setting among a hundred and ninety.

### lines 216-224

```python
try:
```

Menu bar actions.

Menus are reached through `bar.findChildren(QMenu)`, not by walking `menuBar().actions()` and calling `QAction.menu()`: on PySide6 6.11 the QMenu wrapper the latter returns is only valid while the QAction wrapper it came off is alive, so it went stale as soon as this loop moved on — and every menu command in the palette raised "Internal C++ object already deleted" when triggered. `findChildren` hands back children the bar owns in C++.

## CommandPalette._render

### line 347  _(unsure)_

```python
for i in range(self._list.count()):
```

Skip the first section header when auto-selecting

## CommandPalette.keyPressEvent

### line 393  _(unsure)_

```python
"""Move through the results, or run the highlighted command.
```

Arrow keys → move selection; everything else falls through
