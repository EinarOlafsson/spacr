# Notes from `spacr/qt/shortcuts.py`

Prose lifted out of `spacr/qt/shortcuts.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (5 entries)
- [discover](#discover) (1 entry)
- [install](#install) (4 entries)
- [_install_window_hooks](#_install_window_hooks) (2 entries)
- [_bind](#_bind) (2 entries)
- [ShortcutOverlay.__init__](#shortcutoverlay__init__) (6 entries)
- [show_cheat_sheet](#show_cheat_sheet) (1 entry)

## Module level

### lines 82-83  _(unsure)_

```python
ShortcutSpec("Ctrl+End",     "Jump to the newest console line",
```

Bound at window scope so it is available whenever a module console exists, and listed under the interface area it controls.

### lines 94-105

```python
ShortcutSpec("Z + scroll",   "Resize the interface text",
```

HOLD Z AND SCROLL, from 378. It is on the map for the reason 378's own rejection note gives: "a gesture nobody can guess belongs on that map". It was left off at the time only because every spec's label, category and scope are rows in the compact caption ratchet, and the catalogs were being rebuilt by other work that week. They are stable now, so the condition that deferred it is met.

NOT A KEY SEQUENCE, which is why the first column reads as prose rather than as an accelerator. Qt binds a shortcut to a key press; this is a modifier held while the wheel turns, caught in an event filter, and there is no QKeySequence that can express it. The cheat sheet is a MAP of what the hands can do, not a list of what QShortcut owns.

### lines 108-109

```python
ShortcutSpec("F11",          "Full screen",            "Actions"),
```

BOUND ON WINDOW ACTIONS: the window carries these, so they belongs on the map, and `install()` is not the one that creates it.

### lines 179-186

```python
"Ctrl+H", "Ctrl+P",
```

Ctrl+H and Ctrl+P joined on 2026-09-08. The spaCR menu builds Home and Preferences as QActions carrying these sequences so the menu can print the accelerator beside the item -- which BINDS them -- and `install` bound a QShortcut for each on the same window as well. Qt answers a key with two holders by firing neither and logging "QAction::event: Ambiguous shortcut overload", so both keys were dead. They stay in SHORTCUTS because the cheat sheet must still teach them; they leave `installed()` because the action owns them.

### lines 929-932

```python
try:
```

AT IMPORT TIME, so the failure is not a missing background it is the module not importing, which takes down whatever imports it. Driven in tests/qt/test_a_theme_that_refuses_does_not_stop_an_import.py.

## discover

### lines 254-256

```python
known = {native(spec.keys) for spec in mapped()}
```

Include declared per-screen bindings as well as window-wide ones. A screen that already exists under the window must not make its declared shortcut appear a second time as a dynamically discovered key.

## install

### lines 288-290

```python
_bind(window, "Ctrl+K", lambda: _open_palette(window))
```

Ctrl+H and Ctrl+P are NOT bound here: the spaCR menu's Home and Preferences actions already carry them, and a second holder makes the key ambiguous. See BOUND_ELSEWHERE.

### lines 293-299

```python
end = _bind(window, "Ctrl+End", lambda: _jump_to_the_newest_line(window))
```

THE WINDOW OWNS Ctrl+End, and the consoles stand down. Binding it here as well as on every console panel would make it AMBIGUOUS, which in Qt means NEITHER fires -- measured: with both live, `activated` stays silent on both and `activatedAmbiguously` goes to one of them in turn. `_hand_ctrl_end_to_the_window` disables the panels' own copies, so exactly one binding is live and the key works from anywhere in the window rather than only while a console happens to exist.

### lines 301-309

```python
if end is not None:
```

If a console the sweep never reached is on screen, the press arrives ambiguously instead of cleanly. Answering it anyway means the jump still happens, and the handler stands that console down on its way past, so the next press is clean.

`_bind` returns None when a menu QAction already holds the key, which no menu currently does for Ctrl+End -- but the guard is here rather than in a comment, because the failure it would cause is this line raising during window construction.

### line 318  _(unsure)_

```python
for i in range(1, 10):
```

Ctrl+1 .. Ctrl+9 → nth app in the sidebar

## _install_window_hooks

### lines 361-366

```python
try:
```

THE FOLD STRIPS. A folded module is reached from its host's masthead, and the generic settings screens the hosts are built from know nothing about who folded into them -- the strip is hung on each of them from outside, as the stack reaches it. Without this call no host's strip ever reaches a running window, which for Mask Generation's tracking switch means the module folded into it has no way in at all.

### lines 373-377

```python
try:
```

LAST. Everything above may have added menu-bar actions, and an action with no explicit macOS menu role is one Qt assigns from its TEXT which is how "Settings recipes…" became the Preferences item of the macOS application menu. Re-sweeping here means a module added later is covered without its author knowing this problem exists.

## _bind

### lines 407-425

```python
for action in window.findChildren(QAction):
```

A QAction HOLDS A SHORTCUT TOO, and the loop above cannot see one. The menu bar builds "Home" and "Preferences..." as QActions carrying Ctrl+H and Ctrl+P so the menu can print the accelerator beside the item; binding a QShortcut for the same sequence here gave each key a second holder, and Qt answers a key with two holders by firing NEITHER -- it logs "QAction::event: Ambiguous shortcut overload" and the key does nothing. Reported 2026-09-08 for both keys, which is exactly the pair the menu also declares.

The action wins: it is the one the user can see, and a menu item printing an accelerator that does not work is worse than no accelerator. Nothing is returned because there is no QShortcut to hand back -- callers that connect `ambiguousActivation` are guarding against this very case and have nothing left to guard. `findChildren`, not `window.actions()`: a menu item is a child of its QMenu, so the window's own action list does not contain it. The menu bar is built before `install` runs, which is what makes this reachable BOUND_ELSEWHERE is the declarative half that does not depend on that order.

### lines 430-435

```python
sc.setContext(Qt.WindowShortcut)
```

One spaCR window owns one set of bindings.  ApplicationShortcut makes every still-live window's copy eligible, including a window waiting on deferred deletion after a rebuild/test teardown.  Qt then calls the key ambiguous and fires neither copy.  WindowShortcut still reaches every child control in the active window, which is the promised scope, while another open spaCR window keeps its own independent bindings.

## ShortcutOverlay.__init__

### lines 650-654

```python
self._scroll = QScrollArea(self._card)
```

The complete map normally fits as one centred card. A short window or larger UI font must not clip its last shortcuts, so only the inside becomes scrollable when the natural card is taller/wider than the overlay. The surrounding card and its visual treatment do not change.

### lines 675-679

```python
for spec in mapped() + discover(self.parent()):
```

THE DECLARED TABLE PLUS WHATEVER IS LIVE (197 A). A per-screen binding is declared, because it does not exist until that screen is built and the map has to describe it anyway; anything else the window happens to carry is discovered, so a shortcut added at runtime appears without a list being edited.

### lines 683-687

```python
room = max(int(self.width() * 0.9), 640)
```

THE CARD HAS TO FIT THE WINDOW. One column-pair per category made the card 1,640 px wide against a 1,280 px overlay the moment the map grew from 17 rows to 33 -- a map that runs off the screen is the same fault as a map that leaves keys out. Categories are laid out in as many pairs as fit and then wrapped.

### lines 689-692

```python
per_pair = 420
```

A pair contains the key plus a possibly scoped description. About 420 px per pair keeps two pairs inside a 1280 px window even with the longest scope text; narrower estimates let the size hint grow beyond the overlay on hosted Open Sans rasterizers.

### lines 707-708

```python
keys = QLabel(native(spec.keys), self._card_content)
```

PRINTED IN THE PLATFORM'S OWN SPELLING. `Ctrl` is the

Command symbol on macOS and Qt already knows.

### lines 713-715

```python
said = tr(spec.label)
```

AND WHERE IT WORKS, when that is not everywhere. A key that works on one screen and is listed without saying so sends a user to press it somewhere it does nothing.

## show_cheat_sheet

### line 858  _(unsure)_

```python
by_cat: dict[str, list[ShortcutSpec]] = {}
```

Group by category
