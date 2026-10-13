# Notes from `spacr/qt/walkthrough.py`

Prose lifted out of `spacr/qt/walkthrough.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_WalkthroughHandler.on_current_changed](#_walkthroughhandleron_current_changed) (1 entry)
- [install_help_menu](#install_help_menu) (2 entries)

## _WalkthroughHandler.on_current_changed

### lines 478-480

```python
if getattr(screen, "_settings_model", None) is None:
```

Only for modules that render the shared settings form; a bespoke screen has no groups to describe and the derived steps would be three sentences of nothing.

## install_help_menu

### lines 541-545

```python
action.setProperty("moduleAppKey", key)
```

Carry the module identity so the retranslation pass rebuilds this status tip through the reviewed per-module summaries instead of translating the sentence word by word. Two thirds of the module descriptions have no catalog row of their own, so the word-level fallback leaves them wholly English; the summaries cover them all.

### lines 571-576

```python
from .menus import pin_menu_roles
```

Every action here, and the submenu's own action, gets an explicit macOS role. None of these texts happens to contain "settings" or "options" today -- but leaving the role unset means Qt decides from the text, so RENAMING a walkthrough could move it into the application menu on macOS, and nobody makes that connection while renaming a menu item. See spacr.qt.menus.
