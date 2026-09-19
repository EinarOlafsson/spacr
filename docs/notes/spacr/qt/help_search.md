# Notes from `spacr/qt/help_search.py`

Written by hand on 2026-09-19 for instruction 422. The module carries no
comments, so its reasons are in its docstrings; this file holds what is about
the feature rather than about a function.

## Where "beside the Help dropdown" actually is

Help is the last menu on the bar, and what follows it is the menu bar's
top-right corner widget — `WindowChrome`, which already holds the minimise,
full-screen and close marks, because this window is frameless and has no
title bar of its own. The field is inserted at index 0 of that row, so it sits
immediately after the Help menu and in front of the window marks. There is no
second place it could go that is still "beside Help": the left corner is
occupied by the spaCR menu, and a toolbar would be a new strip.

`install()` falls back to creating its own corner widget if there is no
`WindowChrome` — a window built without the chrome (a test, a future frameless
variant) still gets the field.

## Why Ctrl+Shift+H and not Ctrl+F

`Ctrl+F` is already "find a setting on THIS module" and is bound to the
per-module search strip. A key that means one thing on a settings screen and
another on Home is a key nobody trusts. `Ctrl+Shift+F` is the background
full-screen toggle and `Ctrl+K` is the command palette, so `Ctrl+Shift+H` —
free, and next to the menu it opens — is what was left.

The command palette (`Ctrl+K`) and this field are deliberately different
things. The palette lists ~30 ACTIONS: apps, recent runs, menu entries. This
searches 11,000 NOUNS: settings, preferences, API symbols. Merging them would
have meant either drowning the palette's actions or capping this at a size
that defeats the point.

## Reveal, not filter

The obvious implementation of "open the module with only the matching category
expanded" is to type the setting's key into the per-module search strip: it
already collapses everything with no match and expands what it keeps. It was
rejected. A filtered panel hides every other setting, so a user who arrived
from a search and then wanted to look at the rows around the one they searched
for has to first work out what happened to the form — and the strip's box
holding a query they did not type is the least discoverable state in the
module.

`SettingsSearchBar.reveal(key)` instead clears the filter, collapses every
section except the one holding the row, scrolls to it and outlines it for four
seconds. Nothing is rebuilt and no value is read or written, which is the same
property the filter has and for the same reason: the row was already on the
form.

Measured on Mask: 22 sections, 1 open, 21 collapsed, and 190 settings still on
the form.

**And the disclosure level is raised only when it has to be.** Switching to
All settings unconditionally works and also rewrites the module's remembered
Essentials/All choice every time anyone looks something up — a setting the
user chose, changed as a side effect. So the filter is cleared first and the
level is raised only if the row is still off the form afterwards, which is
exactly the case where Essentials is what was hiding it.

## The mark is static on purpose

Anything that moves owes an answer to the Animation preferences and to the
reduced-motion equivalents. A border that appears and then goes away owes
neither and is not lost on a reader who has turned motion off. The previous
stylesheet is restored rather than cleared, so a field that carried one of its
own still carries it afterwards.

## No dead links, and how that is kept without blocking

The instruction is explicit that a dead link is worse than a result that says
the page is not available, and the reason is in the instruction too: a search
result is believed. So an API result never opens a browser at a page that has
not answered.

`_DocsReach` has THREE states. `"unknown"` is the honest answer before anybody
has asked, and it is treated as "do not claim the link works": the offline
view opens, with its **Open the web page** button disabled, and the button
becomes live if and when the probe says the site is there. A two-state version
has to guess, and guessing "reachable" is exactly how a search result becomes
a 404.

The probe is a plain daemon thread — the same shape `spacr/qt/path_probe.py`
uses — and it is started only when an API result is actually opened. spaCR
does not reach the network because somebody typed in a search box.

## The offline docstring is read, not imported

`local_docstring` parses the module's source with `ast` and walks to the
symbol. Importing `spacr.core` to read one docstring would drag the
scientific stack into the process on the offline path, which is the path most
likely to be on a machine that is short of everything.
`test_the_local_docstring_is_read_without_importing_the_module` runs a
subprocess and asserts `spacr.core` is not in `sys.modules` afterwards.

## Settings survive a result being opened, and always did

The maintainer asked for this to be verified and made true if it was not. It
was already true: `MainWindow._on_nav_selected` keeps every screen it has
built in `self._screens` and switches the stack to it, so a half-filled form
is still half-filled when the user comes back. The only path that replaces a
screen is `_rebuild_for_scale`, which fires when the interface font scale has
changed since the screen was built — a different event from navigation. The
test now pins it, so a future rebuild-on-navigate cannot take it away quietly.
