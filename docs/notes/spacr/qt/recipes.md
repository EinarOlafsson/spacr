# Notes from `spacr/qt/recipes.py`

Prose lifted out of `spacr/qt/recipes.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [compatibility_note](#compatibility_note) (1 entry)
- [RecipeDialog.__init__](#recipedialog__init__) (1 entry)
- [install](#install) (1 entry)
- [install_help_action](#install_help_action) (1 entry)
- [install_window_hooks](#install_window_hooks) (2 entries)

## Module level

### lines 107-110

```python
try:
```

AT IMPORT TIME, so the failure is not a missing background it is the module not importing, which takes down whatever imports it. Driven in tests/qt/test_a_theme_that_refuses_does_not_stop_an_import.py.

## compatibility_note

### lines 328-331

```python
known = (set(getattr(model, "_widgets", {}) or {})
```

Conditional settings stay supported while their rows are absent from this particular form shape. ``collect()`` carries them in ``_defaults``, so judging only the currently rendered widgets calls a fresh recipe stale and asks the user to confirm every ordinary apply.

## RecipeDialog.__init__

### lines 439-453

```python
row = FlowLayout(spacing=6)
```

A FLOW, NOT A BOX, AND THE REASON IS A MEASURED CLIP. Five buttons in a `QHBoxLayout` want more width than this dialog's 520 px floor in German -- "Aktuelle Einstellungen speichern…" asks for 440 px and was given 399 -- and Qt's answer to a box it cannot satisfy is to shrink every child BELOW its hint rather than to wrap. The caption goes, silently.

`app_screen._WrappingButtonStrip` records the same defect on the module action row, with its own numbers: a single box made that row's minimum 1,092 px in German against 908 in English. The remedy there and here is that buttons wrap instead of squeezing.

WHAT IS LOST IS THE STRETCH between {Save, Import} and {Share, Delete, Apply}, which a flow layout has no notion of. That grouping was a nicety; a caption nobody can read is not.

## install

### lines 686-690

```python
caption = "Recipes"
```

The button joins the strip after the window has already run its one language pass over the screen, so it translates its own caption. The English stays on the widget as the source `retranslate_widget_tree` reads, so a later language change still has something to translate rather than a translation.

## install_help_action

### lines 800-803

```python
from .menus import set_menu_role
```

"Settings recipes…" contains the word "settings", which is enough for Qt to claim it as the macOS Preferences item and move it out of Help into the application menu -- where it opened instead of Preferences. NoRole is what stops that. See spacr.qt.menus.

## install_window_hooks

### lines 869-871

```python
stack.currentChanged.connect(watcher.on_current_changed)
```

Queued after the search strip's own watcher, which is connected first in `shortcuts._install_window_hooks` — Qt delivers to slots in connection order, so the strip exists by the time this runs.

### lines 877-879

```python
QTimer.singleShot(0, watcher.install_current)
```

Scheduled after the search strip's own deferred install, because Qt runs zero-timers in the order they were started and `shortcuts` installs the strip first.
