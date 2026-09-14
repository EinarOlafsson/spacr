# Notes from `spacr/qt/theme.py`

Prose lifted out of `spacr/qt/theme.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (87 entries)
- [rim_colour](#rim_colour) (1 entry)
- [_solve_scrims](#_solve_scrims) (1 entry)
- [pane_alpha](#pane_alpha) (1 entry)
- [panel_alpha](#panel_alpha) (1 entry)
- [field_chrome](#field_chrome) (1 entry)
- [_splash_roles](#_splash_roles) (2 entries)
- [_scrim_bounds](#_scrim_bounds) (1 entry)
- [_solve_scrims_over_drift](#_solve_scrims_over_drift) (1 entry)
- [_apply_dressing](#_apply_dressing) (1 entry)
- [palette_for](#palette_for) (1 entry)
- [page_colour](#page_colour) (1 entry)
- [menu_bar_background](#menu_bar_background) (1 entry)
- [_bare_image_rules](#_bare_image_rules) (1 entry)
- [font_px](#font_px) (1 entry)
- [clear_container_surfaces](#clear_container_surfaces) (2 entries)
- [_window_block](#_window_block) (3 entries)
- [_SheetsEveryWindowThatAppears.eventFilter](#_sheetseverywindowthatappearseventfilter) (1 entry)
- [_sheet_one_window](#_sheet_one_window) (4 entries)
- [set_a_sheeted_widgets_own_rule](#set_a_sheeted_widgets_own_rule) (1 entry)
- [_forget_window_stylesheets](#_forget_window_stylesheets) (1 entry)
- [_the_windows_own_stylesheet](#_the_windows_own_stylesheet) (2 entries)
- [apply_stylesheet_per_window](#apply_stylesheet_per_window) (2 entries)
- [clear_widget_qss_overlays](#clear_widget_qss_overlays) (1 entry)
- [stylesheet](#stylesheet) (13 entries)
- [close_mark_side](#close_mark_side) (1 entry)
- [apply_close_mark](#apply_close_mark) (1 entry)
- [size_close_mark](#size_close_mark) (1 entry)
- [_CloseMarkResizer.eventFilter](#_closemarkresizereventfilter) (1 entry)
- [_CloseMarkWatcher._sweep](#_closemarkwatcher_sweep) (1 entry)
- [mark_tab_bar](#mark_tab_bar) (1 entry)

## Module level

### lines 45-52

```python
DARK_PALETTE = {
```

Dark palette — kept aligned with the Tk gui_elements apply_theme dict so switching between the two GUIs feels visually consistent.

Named DARK_PALETTE, not PALETTE. The old name read like "the palette" and was imported as one by two dozen widgets, which is how the light theme ended up drawing dark panels. See the module docstring and :func:`active_palette`.

### line 54

```python
"bg":          "#000000",   # main window background
```

Surfaces — pure black bg, subtle depth via layered near-blacks

### line 55, trailing  _(unsure)_

```python
"bg":          "#000000",
```

main window background

### line 56, trailing  _(unsure)_

```python
"page":        "#23252a",
```

the page BEHIND the panels — see `page`

### line 57, trailing  _(unsure)_

```python
"surface":     "#0d0e10",
```

sidebar / panels — barely lifted from bg

### line 58, trailing  _(unsure)_

```python
"surface_alt": "#161719",
```

cards / grouped sections — one more step

### line 59, trailing  _(unsure)_

```python
"surface_hi":  "#1f2124",
```

hovered surfaces

### line 60, trailing  _(unsure)_

```python
"border":      "#2a2d33",
```

visible dividers

### line 61, trailing  _(unsure)_

```python
"border_soft": "#1c1e22",
```

hairline card borders

### line 64, trailing  _(unsure)_

```python
"fg_muted":    "#a1a6ad",
```

secondary text (was #9ba0a6)

### line 65, trailing  _(unsure)_

```python
"fg_dim":      "#6b6f76",
```

disabled / hints

### line 67, trailing  _(unsure)_

```python
"accent":      "#4A9EFF",
```

primary interactive

### line 68, trailing  _(unsure)_

```python
"accent_hi":   "#66B2FF",
```

hover

### line 69, trailing  _(unsure)_

```python
"accent_lo":   "#2F80D9",
```

pressed

### line 70, trailing  _(unsure)_

```python
"accent_soft": "#1e3550",
```

accent-tinted surface (chips, highlights)

### line 71  _(unsure)_

```python
"success":     "#3fb950",
```

Status

### lines 73-75

```python
"chip_class":  "#1fb6ad",   # teal  -- the class name
```

The class/value bubbles of the Classify selector. Semantic roles, not literals at the call site, so they follow the theme -- and named for what they MEAN rather than for the colour, so a retheme can move them.

### line 76, trailing  _(unsure)_

```python
"chip_class":  "#1fb6ad",
```

teal  -- the class name

### line 77, trailing  _(unsure)_

```python
"chip_value":  "#3fb950",
```

green -- its value

### lines 84-100

```python
LIGHT_PALETTE = {
```

Light palette — mirrors the dark set. Used when the user picks the light theme, or when the OS colour scheme is light and theme=system. Kept close in structure to DARK_PALETTE so a caller can swap between them by key without touching consumers.

Every value below was picked (or corrected) against :data:`CONTRAST_RULES`. The originals failed AA in several places — `accent` at 4.10:1 on `surface_hi`, `accent_hi` at 2.81:1 on `accent_soft`, `warning` at 3.82:1, `fg_dim` at 2.54:1 — because in a light theme "hover" has to go *darker*, not brighter, and the first cut of this palette mirrored the dark one literally.

Every key here also exists in :data:`DARK_PALETTE` — see

``test_theme.py`` — so a caller can swap palettes without touching consumers. That is what makes :func:`active_palette` a drop-in for the old module-level import.

### line 103, trailing  _(unsure)_

```python
"page":        "#e1e4e6",
```

see `page` — a light theme goes the other way

### line 113, trailing  _(unsure)_

```python
"accent_hi":   "#0851a3",
```

hover = darker in a light theme

### lines 117-119

```python
"chip_class":  "#0a6f6a",   # teal  -- the class name
```

The class/value bubbles of the Classify selector. Semantic roles, not literals at the call site, so they follow the theme -- and named for what they MEAN rather than for the colour, so a retheme can move them.

### line 120, trailing  _(unsure)_

```python
"chip_class":  "#0a6f6a",
```

teal  -- the class name

### line 121, trailing  _(unsure)_

```python
"chip_value":  "#0f7030",
```

green -- its value

### lines 128-135

```python
SPACE_PALETTE = {
```

Space palette — a dark theme layered over a generated deep-space image.

The surface colours here are *scrim* colours: they are painted at the alpha in :data:`SCRIM_ALPHA` over whatever the background image happens to be. They are therefore chosen so that even composited over a pure white star they stay dark enough for `fg`, `fg_muted` and `accent` to clear AA — see :func:`effective_surface`.

### line 137, trailing  _(unsure)_

```python
"bg":          "#04060d",
```

fallback flat sky when no image is cached

### line 138, trailing  _(unsure)_

```python
"page":        "#232b3d",
```

see `page`

### line 147, trailing  _(unsure)_

```python
"accent":      "#6cb6ff",
```

brighter than the dark theme's accent:

### line 148, trailing  _(unsure)_

```python
"accent_hi":   "#9bcdff",
```

it has to clear AA on a translucent

### line 149, trailing

```python
"accent_lo":   "#3d8ddb",
```

scrim, not on solid #161719

### lines 152-154

```python
"chip_class":  "#4fd6ce",   # teal  -- the class name
```

The class/value bubbles of the Classify selector. Semantic roles, not literals at the call site, so they follow the theme -- and named for what they MEAN rather than for the colour, so a retheme can move them.

### line 155, trailing  _(unsure)_

```python
"chip_class":  "#4fd6ce",
```

teal  -- the class name

### line 156, trailing  _(unsure)_

```python
"chip_value":  "#5fd97a",
```

green -- its value

### lines 163-178

```python
CELL_PALETTE = {
```

Cell palette — a dark theme layered over the user's own micrographs.

Same construction as Space (scrim colours, judged composited over a white worst case), re-hued to the imagery: the microtubule network is cyan and the filopodia are cyan-to-green, so a blue accent would fight the picture and a warm one would clash with it.

`accent`, `fg_muted` and `error` are *brighter* than their Space counterparts on purpose. They are the roles that get painted straight onto the wallpaper, so their luminance is what :func:`max_background_luma` turns into the exposure the imagery is allowed — a dim accent here would mean a needlessly murky background everywhere. As set, the limit is 0.109 rather than Space's 0.059, which is the difference between the microtubule frame reading as a photograph and reading as a stain.

### line 180, trailing  _(unsure)_

```python
"bg":          "#02080b",
```

fallback flat field when no image is cached

### line 181, trailing  _(unsure)_

```python
"page":        "#24363f",
```

see `page`

### lines 195-197

```python
"chip_class":  "#5fe0d8",   # teal  -- the class name
```

The class/value bubbles of the Classify selector. Semantic roles, not literals at the call site, so they follow the theme -- and named for what they MEAN rather than for the colour, so a retheme can move them.

### line 198, trailing  _(unsure)_

```python
"chip_class":  "#5fe0d8",
```

teal  -- the class name

### line 199, trailing  _(unsure)_

```python
"chip_value":  "#6fe39a",
```

green -- its value

### lines 206-214

```python
GLASS_PALETTE = {
```

Glass palette — neutral translucent material over a built-in light field.

Qt stylesheets do not expose the compositor's native macOS/iOS backdrop blur or Liquid Glass lensing. The cross-platform approximation is a layered material: a bounded neutral light field behind low-alpha charcoal surfaces, brighter top-edge highlights, soft lower shading, generous concentric corners, and selective tint only for actions. This avoids the old result, which was simply opaque navy when the global Page opacity was 100%.

### lines 217-220

```python
"page":        "#0b0d11",
```

The one palette whose `bg` was already a page: its surfaces are charcoal, not near-black, so `bg` clears them by 12.0 L* / 1.301:1 without moving. Stated rather than defaulted so `page` is never "the key dark forgot" — see `page`.

### lines 230-231

```python
"accent":      "#8cc8ff",
```

Tint is deliberately reserved for actions and state. The material itself stays neutral, matching Apple's guidance not to tint everything.

### lines 237-239

```python
"chip_class":  "#6fe0d8",   # teal  -- the class name
```

The class/value bubbles of the Classify selector. Semantic roles, not literals at the call site, so they follow the theme -- and named for what they MEAN rather than for the colour, so a retheme can move them.

### line 240, trailing  _(unsure)_

```python
"chip_class":  "#6fe0d8",
```

teal  -- the class name

### line 241, trailing  _(unsure)_

```python
"chip_value":  "#78dfa3",
```

green -- its value

### lines 259-262

```python
"space": SPACE_PALETTE,
```

Kept, though "space" is no longer offered: `palette_for` is called with whatever is persisted, and a settings file written by an older spaCR can still say "space". Resolving it beats raising; it simply cannot be chosen any more, and `THEMES` no longer lists it so nothing iterates it.

### lines 334-345

```python
STAGE_HOVER = {
```

Maturity — how finished a module is, drawn as a colour

The app→stage table lives in :data:`spacr.qt.app.APP_STAGE`; this is only what each stage LOOKS like. Both the tile hover and the legend under the Home aside read these, so a stage cannot be one colour in the legend and another on the tile.

The three hues were chosen by the user. They are deliberately not palette accents: the point is that "this module is alpha" is not a theme decision, and the same green means the same thing on Space as on Light.

### line 348, trailing  _(unsure)_

```python
"stable": "#3B82F6",
```

blue

### line 349, trailing  _(unsure)_

```python
"beta":   "#FF00FF",
```

magenta

### line 350, trailing  _(unsure)_

```python
"alpha":  "#00CEC8",
```

green-cyan

### lines 376-393

```python
TILE_W = 172
```

The module tile's geometry — here, because the QSS needs it too

These would live happily in :class:`spacr.qt.widgets.home.HomePage` were it not for `min-height`, and `min-height` is not optional.

The app stylesheet carries `QPushButton { min-height: 22px }`, and QStyleSheetStyle turns that into a real `setMinimumHeight(22)` on every button when it polishes it. `qSmartMinSize` lets an EXPLICIT minimum override the widget's own `minimumSizeHint()`, so a tile that answers "124 px" through its hints is still, as far as the layout is concerned, collapsible to 22 — and on a page that does not fit, it collapses. The symptom is not a clipped tile: it is the name label drawn on top of the icon, with no warning and no scrollbar.

The fix is for the QSS to state the floor itself, which means the QSS has to know the number. So the number lives here, the widget reads it from here, and there is one of it.

### lines 400-416

```python
WORST_CASE_UNDER = "#ffffff"
```

Scrims — per-theme opacity of each surface role

Only the image themes are translucent. Everything else resolves to 1.0 and the QSS emits plain hex, byte-identical to what it emitted before scrims existed.

The alphas are **solved, not chosen** — see :func:`solve_scrim_alpha`. The first cut of them was a hand-picked 0.86/0.88/0.90/0.93, which passes every contrast rule and hides the wallpaper: a 0.90 scrim transmits a 1.10:1 range of the picture, i.e. a ghost. Users reported the image themes as "not implemented — I can't see the cells", which was a fair reading of a 10 % image.

`elevated` (menus, tooltips, combo popups) is deliberately opaque even there: those are separate top-level windows, and a translucent popup without a compositor shows the desktop, not the app.

### lines 457-459

```python
GLASS_BACKDROP_UNDER = "#454950"
```

Brightest stop in Glass's built-in `_window_block` light field. Unlike Space, Glass cannot accept an arbitrary photograph, so this is a hard rendering contract rather than a hopeful estimate.

### lines 711-730

```python
PANE_OPACITY_MIN = 0.0
```

The page pane — the one surface whose opacity the USER sets

Everything above solves an alpha. This does not: the rounded box behind Home's tiles is the one surface the user asked to be able to see through, and the whole point of a preference is that the answer is theirs.

What the solver still owns is the FLOOR. `legible_scrim_floor` is the thinnest this pane can be painted and still have the tile names on it clear WCAG AA over the worst pixel that theme's wallpaper can present, so the preference is clamped UP to it and text can never be dragged into illegibility.

The solver's other bound, `present_scrim_ceiling`, is deliberately NOT applied. That one exists to guarantee the wallpaper is still visible through a panel — and a user who drags this slider to 100 % is explicitly asking for the wallpaper to be hidden behind the app they are trying to use. A floor protects them from a mistake; a ceiling would only overrule a choice.

### lines 892-922

```python
FIELD_FADE_EXPONENT = 3.0
```

The field fade — the one surface the page-opacity preference does NOT own

Everything above answers "how solid is this panel?" with a single number. An input field answers it with a ramp instead, and it is deliberately *exempt* from :func:`panel_alpha`:

"the fields should not be subject to the occupacy setting. the fields could gradually become fully transparent (not the text in the field but the container) with the transparency growing faster towards the right. outlines should also be subject to the same effect."

So a field is painted fully opaque at its left edge — whatever the page slider says — and dissolves to nothing at its right edge. Two properties fall out of that and both matter:

The page-opacity slider cannot make a field harder to read. Its left edge, where the value starts, is always a solid surface. The ramp has to be *convex* in transparency, not linear. A linear ramp is already 50 % gone at the midpoint, which is where the text still is.

The curve is therefore a cubic ease-in on TRANSPARENCY:

transparency(t) = t ** FIELD_FADE_EXPONENT alpha(t)        = 1 - t ** FIELD_FADE_EXPONENT

with ``t`` the fraction of the way across the field, 0 at the left edge and 1 at the right. At the midpoint the container is still 87.5 % opaque; it is 58 % at three-quarters, 27 % at nine-tenths, and gone at the edge. That is "faster towards the right" spelled as a number: d(transparency)/dt is 0 at the left edge and 3 at the right.

### lines 1092-1140

```python
SPACEOUT_HUES: Dict[str, float] = {
```

spaceout — the same application, wearing something else

`spaceout` is a console entry point beside `spacr` and `spacr-qt` (:mod:`spacr.qt.spaceout`). It starts the same application — the same screens, the same modules, the same settings — and changes only the dressing: the palette goes rainbow, and the ambient backdrop draws moving fractals instead of drifting blobs.

THE CHOICE IS MADE ONCE, AT LAUNCH, AND IS STORED NOWHERE. It lives here as process state because this module is the funnel every colour in the application already passes through: `palette_for` is what `stylesheet`, `apply_qpalette`, `active_palette`, `page_colour`, every contrast check and every widget that paints its own pixels end up calling, so nothing else has to learn that the mode exists. Handing it to `spacr.qt.preferences` would make it survive a restart and leak the dressing into an ordinary `spacr` start, which is the one thing the request rules out — so it is written nowhere, no Preferences control offers it, and the entry point is the only way in.

THE THEME CONTRACT DOES NOT CHANGE. `resolve_effective_theme` still answers one of `THEMES`, `THEMES` is still those four, and the light/dark handling every screen reads goes on working. spaceout re-hues whichever theme was resolved; it does not become a fifth one, and a light start stays light.

READABILITY IS THE CONSTRUCTION HERE, not a table of colours somebody eyeballed afterwards. Every role moves in HUE ONLY and keeps its own WCAG relative luminance, and three checks fall out of that identity:

`contrast_ratio` is a function of relative luminance and of nothing else, so every rule in `CONTRAST_RULES` measures what it measured on the theme being re-hued; `lightness` (CIE L*) is a function of relative luminance too, so `page_separation_report` — can you see the panel — is preserved with it; `max_background_luma` is a minimum over luminances, so the exposure the imagery is solved down to does not move either.

The identity is exact in the reals and within 8-bit rounding on screen; `_hue_shift` picks the closest representable colour on the hue line rather than the first one that fits, which keeps the drift to a few thousandths of a luminance level.

The one place it does not carry on its own is a TRANSLUCENT surface. An image theme composites its panels over the wallpaper channel by channel in sRGB, and two colours of equal luminance but different hue do not composite to equal luminance. That is why `SCRIM_ALPHA` is re-solved when the mode is enabled — `_solve_scrims` was written as a solve rather than a table for exactly this case, and it says so.

### line 1160  _(unsure)_

```python
"bg":          285.0,
```

Surfaces, sweeping violet -> magenta -> blue -> cyan -> green.

### lines 1168-1171

```python
"fg":          210.0,
```

Text. `fg` is the extreme of its theme's range — white on the dark themes, near-black on the light one — and a hue cannot move a colour that is already at the top or the bottom of the luminance scale, so this entry mostly documents where the light theme's ink goes.

### line 1175  _(unsure)_

```python
"accent":      190.0,
```

Interactive.

### line 1181  _(unsure)_

```python
"success":     130.0,
```

Status — see the note above.

### lines 1187-1189

```python
"button_accent":     305.0,
```

The theme-invariant button roles. They are re-hued by the same table for every theme, so they stay invariant across themes inside the dressing exactly as they are outside it.

### lines 1371-1406

```python
SPACEOUT_DRIFT_TURN = 540.0
```

The drift — the spectrum turns, and the readability turns with it

The dressing above is a *table*: every role lands on one hue and stays there. This turns the whole table, slowly and without a loop a watcher can learn, so the application is not one rainbow but a rainbow that moves.

WHY THIS COSTS NOTHING IN READABILITY, and why that is a property of the construction rather than a claim: `_hue_shift` moves a role in hue and leaves its WCAG relative luminance where it was, and `contrast_ratio` is a function of relative luminance and of NOTHING ELSE. Adding a constant to every hue therefore leaves every ratio in `CONTRAST_RULES` exactly where it was — at every point on the drift, not only at the one it started from.

TWO THINGS DO NOT CARRY ON THEIR OWN, and both are solved rather than hoped for:

a TRANSLUCENT surface. An image theme composites its panels over the wallpaper channel by channel in sRGB, and two colours of equal luminance and different hue do not composite to equal luminance. The alphas were already re-solved when the dressing went on; they are now solved for the WORST POINT ON THE DRIFT instead of for one palette. Measured: keeping the alphas solved at the starting hue, the wallpaper stops reading through the panels at 74 of the 360 one-degree offsets, down to 1.37:1 against a 1.50:1 rule. the INK, which is the other half of the request — see :data:`SPACEOUT_INK_ROLES`.

THE CLOCK IS DRIVEN, NOT READ. `advance_spaceout_drift` is called by the things that are already painting frames — the ambient backdrop's tick and the setup card's — rather than by `palette_for` reading a wall clock. Two reasons, and the second is the load-bearing one: a backdrop the user turned off should not be quietly replaced by a palette animating instead, and `palette_for` is on the path of every stylesheet build and every widget that paints, so a wall clock inside it would make the palette a different value on two calls in the same frame.

### lines 2097-2117

```python
_FROZEN_DARK = MappingProxyType(DARK_PALETTE)
```

The name that was a trap

`PALETTE` is served by module ``__getattr__`` rather than bound as a global, for three reasons:

1. It cannot be assigned to. `theme.PALETTE = ...` and

`PALETTE["fg"] = ...` both fail now, so nobody can "fix" a theme by mutating it and have the change silently apply to every module that already imported it — or, worse, not apply, because half of them copied the value into an f-string at import time. 2. `grep -n 'PALETTE ='` over this file no longer finds a definition for it. The dark dict is spelled `DARK_PALETTE` at its definition, where a reader decides whether it is the right thing to import. 3. Importing it warns, naming the exact failure, so the remaining call sites are discoverable with `-W error::DeprecationWarning` instead of by eye.

It is kept (rather than deleted) only because the migration is not finished: two dozen screens and widgets still import it. Each one is a light-theme rendering bug until it moves to `active_palette()`.

### lines 2318-2319  _(unsure)_

```python
("bg", "accent", 4.5),
```

`bg` is the ink on filled accent/danger surfaces: the selected menu row, a pressed button, DangerButton on hover.

### lines 2375-2388

```python
PAGE_PANEL_ROLES: Tuple[str, ...] = ("surface", "surface_alt")
```

Does the page separate from the panels?

:data:`CONTRAST_RULES` asks "can you read the text on it". This asks the other question, the one nobody was asking when the dark page was ``#000000``: can you see the *panel*. They are different measurements, and the second one has no WCAG tier to borrow, because WCAG has nothing to say about two surfaces neither of which is text.

So it is measured twice. The ratio is kept because it is the number the rest of this module speaks in; CIE L* is the one that decides, because down at the black end the ratio stops discriminating — every pair of near-blacks is "about 1.1:1" — while L* stays linear in perceived lightness all the way to zero.

### lines 2548-2566

```python
CONSTANT_ROLES = {
```

Theme-invariant colour roles

The user should be able to recognise interactive controls by colour regardless of theme. If the AI/Live toggle went accent-blue in dark and a different accent-blue in light, that recognition breaks. These keys resolve to the SAME value in both DARK_PALETTE and LIGHT_PALETTE so button / toggle styling can rely on them.

`button_accent`     — primary button + toggle "on" colour

`button_accent_hi`  — hover `button_accent_lo`  — pressed `button_accent_ink` — text drawn ON those fills Chosen to read well on both surface_alt colours (near-black + near-white).

The ink is near-black, not white. White on #4A9EFF measures 2.75:1 — well under AA — so the "Run" button had unreadable small text in every theme until this was measured. Near-black on the same fill is 6.96:1, and still 4.75:1 on the darker pressed shade.

### lines 2572-2576

```python
}
```

NOTE: the loading screen's background and ink are NOT here. They follow the theme's own `bg` and `fg` -- see `palette_for`. Putting them here as one fixed colour is what instruction 78 undid: a splash one shade off the window behind it makes the handover flash, and the point of the loading screen is that the transition is invisible.

### lines 2580-2586

```python
_SOLVED_SCRIMS[False] = _solve_scrims()
```

Every ingredient the scrim solver needs — the palettes (including these constant roles, which `palette_for` folds in), the contrast rules, the colour maths and the exposure ceiling — exists by this point, so the alphas can be solved. Done at import so that `scrim_alpha` stays a dict lookup on the hot path (the QSS asks for it once per role per theme change) and so a palette edit that makes a theme unsolvable fails loudly here rather than three screens later.

### lines 2591-2593  _(unsure)_

```python
SPACING = {
```

Spacing / radius scale — 4/8-based, matches Tk gui_elements.

### line 2611, trailing  _(unsure)_

```python
"xs":      11,
```

inline metadata, table cell suffixes

### line 2612, trailing  _(unsure)_

```python
"small":   12,
```

captions, muted secondary text, form hints

### line 2613, trailing  _(unsure)_

```python
"body":    13,
```

default body text

### line 2614, trailing  _(unsure)_

```python
"label":   13,
```

form field labels

### line 2615, trailing  _(unsure)_

```python
"header":  15,
```

card / section titles

### line 2616, trailing  _(unsure)_

```python
"subtitle":17,
```

dialog headings, secondary display

### line 2617, trailing  _(unsure)_

```python
"title":   22,
```

screen-level headings

### line 2618, trailing  _(unsure)_

```python
"display": 30,
```

startup screen brand title

### line 2619, trailing  _(unsure)_

```python
"hero":    42,
```

empty-state hero numerals

### lines 3220-3244

```python
_WIDGET_QSS: Dict[str, object] = {}
```

The widget QSS seam — how a new widget styles itself without editing this

`stylesheet()` is one 1100-line f-string, and every widget added to the app used to need a block inside it. That makes this file the bottleneck for anyone building a widget, and it makes two people building two widgets a merge conflict in a literal nobody can review line by line.

A widget registers its own block instead, at import time, from its own module:

from spacr.qt.theme import register_widget_qss, pane_surface

def _gate_editor_qss(palette, opacity):

return f''' QFrame#GateEditor {{ background: {pane_surface("surface_alt", palette["theme"], opacity)}; border: 1px solid {palette["border_soft"]}; }}'''

register_widget_qss("GateEditor", _gate_editor_qss)

Registered blocks are appended LAST, after the glass layer, so a widget can override a general rule that would otherwise win on specificity — and so the built-in look is unchanged while nothing is registered.

### lines 3276-3277  _(unsure)_

```python
"spacr.qt.screens.classify",
```

The lightweight Classify footer registers the theme-native box without importing FlowView itself.  Its renderer remains lazy until expansion.

### lines 3283-3286

```python
"spacr.qt.screens.gate_editor",
```

The Gate Editor's Filter/Search tab strip. It tried to register its own block from the screen's __init__, against a `theme.register_qss` that has never existed, so the strip has been falling through to the blanket `QWidget { background-color: bg }` since it was written.

### lines 3290-3293

```python
"spacr.qt.screens.map_barcodes",
```

The fold page strip every module screen grows when a fold is opened. A page can be opened long after launch, so its import-time registration is applied to its host screen then; exhaustive sheets include it up front too.

### lines 3303-3307

```python
"spacr.qt.prerun",
```

Not a screen: the Measure segmentation-QC banner and the Mask diameter panel. Its block used to be registered only from `prerun.register()`, which runs after the stylesheet has been built and applied, so the banner fell through to the blanket QWidget rule and the verdict text sat on a solid black slab.

### lines 3311-3316

```python
"spacr.qt.widgets.gate_editor",
```

Registers at import, through `ensure_field_fade_qss()` at module scope, but nothing imported it while the sheet was being built -- so the block was absent at launch. Found by the strengthened registrar check, not by a bug report. The gate list had no QSS block at all, so its rows fell back to Qt's default black text on the theme's surface -- unreadable on dark.

### lines 3436-3440

```python
_LOCAL_WIDGET_QSS_START = "/* --- local registered widget QSS: start --- */"
```

A late block lives on the root of the screen that needed it.  The outer markers make that suffix replaceable as more modules are imported and removable before the next whole-application preference rebuild.  The attribute holds the exact suffix, including its leading newline, so a screen's own stylesheet is restored byte-for-byte rather than reparsed.

### lines 5593-5602

```python
CLOSE_MARK = "✕"
```

Shared marks — one glyph per gesture, drawn the same way everywhere

A close mark restyled at each site drifts at the next one. The montage's well tabs drew their own `×` at the tab bar's font, the gate-editor chip drew one in the chip's ink, the folded pages got whatever small pixmap the platform style felt like, and the value chips coloured theirs `fg_muted`. There is ONE mark now: the glyph, its size, its hit target and its two colours live here, and every site asks for it instead of describing it again.

## rim_colour

### lines 276-283

```python
def rim_colour(theme: str = "dark") -> str:
```

Two colours that are a *function of the theme*, not entries in it

Both used to be inlined at the call site, which is the exact mistake `PALETTE` exists as a warning about: a hex typed into a widget is a hex that stays dark on the light theme. Neither is a palette role because neither is a colour you would ever choose independently — each is derived from one that is already there.

## _solve_scrims

### lines 696-697  _(unsure)_

```python
solved["elevated"] = 1.00
```

Popups are separate top-level windows. Translucency there shows the desktop, not the wallpaper.

## pane_alpha

### lines 765-768

```python
wanted *= scrim_alpha("glass", "surface")
```

Glass has material translucency of its own. Page opacity controls how strongly that material is present; 100% means the designed glass, not an opaque navy panel. This keeps the preference and the theme as two genuinely separate features.

## panel_alpha

### lines 885-887

```python
wanted *= scrim_alpha("glass", role)
```

Relative material strength: even the 100% setting retains the role's designed translucency. Other themes keep literal 0..100% surface opacity, so this does not change their established control.

## field_chrome

### line 985

```python
"radius": 10.0 if glass else float(RADIUS["sm"]),
```

Glass rounds its inputs to 10px; everything else uses RADIUS.sm.

## _splash_roles

### lines 1018-1023

```python
dim = splash_dim_alpha(ink, bg)
```

The dim weight is SOLVED, not fixed, for the same reason the scrims are. A fixed alpha does not mean a fixed contrast: dark ink fading toward a light surface loses contrast faster than white ink fading toward black gains it, so the 110 that read at 3.04:1 on the dark theme read at 2.31:1 on the light one -- under the floor, on the screen that is up while the user has nothing else to look at.

### lines 1025-1029

```python
return {
```

COMPOSITED TO OPAQUE HEX, not stored as `rgba(...)`. Every palette value is #rrggbb -- `test_palette_values_are_hex` is the contract and these three are only ever painted ON the splash background, so flattening them against it is exact rather than an approximation. The widget then draws opaque colours and does no alpha maths at all.

## _scrim_bounds

### lines 1772-1781

```python
base = _channels(palette[colour_role or role])
```

`colour_role or role`, matching the two published solvers exactly which this docstring insists on and, until 310 A7, did not do. Both `picture_contrast` and `present_scrim_ceiling` take `colour_role` as Optional and fall back to `role`; here `role` was accepted and never read, so a caller passing the documented `colour_role=None` got `KeyError: None` out of the drift solver at import rather than that role's own bounds. Not reachable today -- every SCRIM_ROLES value is a non-None string -- but the dead parameter also hid the divergence from anyone comparing the three implementations, which is the comparison the docstring asks them to make.

## _solve_scrims_over_drift

### lines 1851-1852  _(unsure)_

```python
solved["elevated"] = 1.00
```

Popups are separate top-level windows. Translucency there shows the desktop, not the wallpaper.

## _apply_dressing

### lines 1903-1904  _(unsure)_

```python
damping = _solve_page_damping()
```

The damping first: it moves the panel colours, and the scrims are solved from those.

## palette_for

### lines 1967-1969

```python
drift = (spaceout_drift_step() if _SOLVE_DRIFT is None
```

Same keys, same surface luminances, different hues — see the spaceout block above. Applied before the splash roles so those are derived from the colours the window will actually be painted with.

## page_colour

### lines 2003-2066

```python
def page_colour(theme: str = "dark") -> str:
```

`page` — the colour of the surface the panels float ON

`bg` is the WINDOW colour. It is `QPalette.Window`, it is the ink on a filled accent button (`QPalette.HighlightedText`), it is the blanket `QWidget { background-color: bg }` in `_window_block`, and in the dark theme it is literally `#000000`. Thirty-six uses, most of which are not "the page".

For a long time it was the page anyway, by omission. A module screen clears its layout containers (`clear_container_surfaces`) so the backdrop shows between the settings cards — and when the ambient animation is switched off there IS no backdrop, so what showed through was the blanket window fill. Pure black. Measured on a real AppScreen with `ambient_enabled=False`: 40 of 74 samples down the settings column were exactly (0,0,0), and (0,0,0) was the single most common colour on the whole screen (17,621 samples, against 11,420 for `surface`). Users reported it three times as "a black box behind the settings categories"; two fixes swept more containers transparent, which is the right sweep and made the hole bigger, because the thing behind the containers had no colour of its own.

So the page gets one. `page` is a separate role from `bg` precisely so that giving the page a colour cannot change selected-text rendering, `QPalette.Window`, or the ink on a pressed button.

The values are solved, not chosen. A page has to satisfy, at once:

Separation from the resting panels.** `surface` and `surface_alt` are what a settings category is painted with, and they have to read as panels sitting on something. Judged in CIE L*, because near black the WCAG ratio saturates and stops discriminating — on the dark theme `bg`-to-`surface` is 1.087:1 whatever you do, and `#000000` against `#0d0e10` is 3.95 L* on a scale where the palette's own deliberate step is 3.75. The bar is one full palette step (>=3.5 L*) and >=1.15:1 from each. See `page_separation_report`. Survival at 60 % page opacity**, where a panel is composited 0.6 * panel + 0.4 * page and the separation shrinks by ~40 %. The bar there is >=2.0 L* and >=1.08:1. Text still lands on it.** Every rule in :data:`CONTRAST_RULES` is enforced against `page` as a surface, so `fg`/`fg_muted`/`accent` clear 4.5:1 and `fg_dim` and the status hues clear 3.0:1 — hint text and disabled controls do sit straight on the page.

On the dark theme those three close to a band two values wide. `surface_alt` separation pushes from below and `fg_dim` at 3.0:1 caps from above; `#23252a` sits in it at 1.170:1 / 6.96 L* from `surface_alt` and 1.259:1 / 10.72 L* from `surface`, with `fg_dim` at 3.04:1. The light theme is the same solve mirrored — the page goes *down* so white cards lift off it — and closes just as tightly, between `surface_alt` from below and `accent` at 4.5:1 from above.

The consequence is that on the dark theme the page is now *lighter* than the cards, which is not the usual dark-UI layering and is deliberate: it is what the app already looks like with the animation on. The ambient backdrop is brighter than `#0d0e10` nearly everywhere (2 of 74 samples pure black with it enabled, against 40 with it off), so dark cards floating on a lighter field is the established look and `page` is the still frame of it, not a new idea.

`surface_hi` is deliberately NOT in the solve. It is the *hover* colour: converging toward the page as a card lifts is what hover is for, and requiring a step from it as well pushes the page past `fg_dim`'s AA ceiling with no band left at all.

## menu_bar_background

### lines 2229-2232

```python
try:
```

The theme ON SCREEN, resolved the way `active_palette` does the chrome is restyled on a theme change like everything else, so reading a constant here would leave the corner painting the dark bar's colour under the light one.

## _bare_image_rules

### lines 2472-2484

```python
def _bare_image_rules() -> Tuple[Tuple[str, float], ...]:
```

Contrast against a *background image*

:func:`contrast_report` resolves the `bg` role to the palette's flat fallback colour, which is right for the opaque themes and for the gradient an image theme falls back to — but it is not what an image theme actually shows. There, `bg` is the photograph, and text painted with no surface under it (a hero subtitle, a tile caption, a ghost button) lands on whatever pixels happen to be there.

So the rules that name `bg` are the ones a wallpaper has to satisfy, and they are read straight out of CONTRAST_RULES rather than restated — a role added there is automatically enforced against the imagery.

## font_px

### lines 2649-2651

```python
try:
```

Lazy: `preferences` imports this module, so a module-level import would be circular. Degrade to 1.0 rather than raising — an unscaled label is cosmetic, an exception in a paint is not.

## clear_container_surfaces

### lines 2829-2831

```python
continue
```

The screen says this view IS the page. Tagging it — or its viewport, which is the half that actually paints — would put the backdrop straight underneath the text.

### lines 2841-2843

```python
if type(widget) is QWidget and not widget.objectName():
```

`type(widget) is QWidget` on purpose, not isinstance: a subclass is a component someone wrote and may well paint deliberately. Only the bare scaffolding qualifies.

## _window_block

### lines 3069-3072

```python
sky = f'background-color: {P["bg"]};'
```

THE THEME'S OWN GROUND, NOT A PICTURE. An opaque theme running an animated backdrop needs `QWidget` transparent so the animation is not covered, and the window itself opaque so a desktop without a compositor does not show through. It has no master to paint.

### lines 3080-3082

```python
sky = (
```

A neutral off-axis light field gives translucent surfaces something to optically respond to. It is intentionally not blue: colour belongs to content and selected actions, not to every piece of chrome.

### lines 3091-3095

```python
sky = ("background-color: qlineargradient(\n"
```

No cached image (first run mid-generation, unwritable home, a source build with the masters stripped, tests): a gradient in the theme's own hues. Dimmer than the real thing but every scrim, border and text colour still lands correctly, so the theme degrades to "plain dark" rather than to broken.

## _SheetsEveryWindowThatAppears.eventFilter

### lines 3566-3569

```python
_sheet_one_window(watched)
```

A NOMINATED ROOT COMING BACK ON SCREEN. It was left out of the last change on purpose because nobody could see it; this is the moment that stops being true, and it is before the first paint.

## _sheet_one_window

### lines 3636-3640

```python
if (window.property(_WINDOW_SHEET_SERIAL) == serial
```

SEEN IT AND STILL WEARING IT, which are two questions. The second is what catches a widget whose sheet was replaced by somebody else since -- and re-sheeting is then correct rather than wasteful, because whoever replaced it did so believing the application carried the theme.

### lines 3646-3649

```python
own = _the_windows_own_stylesheet(window)
```

THE WINDOW'S OWN RULES SURVIVE, AND GO LAST so they still win. Remembered the first time, because by the second pass the sheet on the widget is ours and reading it back would fold the global rules into "its own" for ever.

### lines 3651-3654

```python
text = preserve_widget_qss_overlay(window, sheet + own)
```

PRESERVE A SCREEN-LOCAL SUFFIX. `preserve_widget_qss_overlay` owns the other end of this: a root that has been given its own late block keeps it appended, or setting the window sheet would strand that screen on the previous preference values.

### lines 3658-3661

```python
window.setProperty(_WINDOW_SHEET_BASE_LEN, len(sheet))
```

WHERE OUR PART ENDS, remembered so the next pass can tell an append from a replacement. The app attribute is no use for this: by the time the next pass reads it, it already holds the NEW sheet while the widget still wears the OLD one.

## set_a_sheeted_widgets_own_rule

### lines 3694-3696

```python
widget.setProperty(_WINDOW_SHEET_SERIAL, None)
```

Force the re-sheet: the serial says we have already dressed this widget for the current sheet, and what changed is the half that is appended after it.

## _forget_window_stylesheets

### lines 3737-3763

```python
try:
```

NOT JUST THE TOP-LEVEL WINDOWS, AND THAT IS THE WHOLE CHANGE. Since `MainWindow` nominates roots, the sheet lives on widgets that are NOT top level -- the menu bar, the status bar, the dock, the visible page. Walking only `topLevelWidgets()` left them wearing it. Measured: of 28 widgets carrying the sheet, this removed 22 and SIX kept it QMenuBar, QStatusBar, HomePage and a layout container, every one of them `isWindow() == False`.

That is the cross-test contamination this function exists to stop, reopened by the change that moved where the sheet lives. The stamp is what identifies our own work, so the stamp is what is searched for.

`allWidgets()` RATHER THAN A WALK, and the difference from the segfault this codebase already paid for is WHEN. Calling it from `pytest_sessionfinish` crashed the interpreter -- Qt was already tearing objects down and reading a half-destroyed one is not an exception, it is a fault. This runs BETWEEN tests with a live application, which is the state the call is defined for. The walk it replaces missed a `QMenu` that is a window and is a child of nothing reachable from a top-level widget: 27 of 28 is not isolation. THE APPLICATION SHEET COMES DOWN FIRST, and the order is the fix rather than an ordering preference. Clearing a widget's stylesheet is itself a `setStyleSheet`, which triggers a `Polish` -- and the event filter, seeing a sheet still installed on the application, RE-SHEETS the widget we have just cleared. Traced: one QMenu came out of the loop wearing 72,618 characters again with its stamp gone, so it could not even be found on a second pass.

## _the_windows_own_stylesheet

### lines 3833-3834

```python
current = current[:-len(suffix)]
```

The late screen block belongs to `preserve_widget_qss_overlay`, which re-appends it; keeping it here too would double it.

### lines 3836-3845

```python
base_len = window.property(_WINDOW_SHEET_BASE_LEN)
```

AN APPEND IS NOT A REPLACEMENT, and telling them apart is the whole of this. `ensure_widget_qss_applied` appends a late block, which changes the digest exactly as somebody else's `setStyleSheet` would and what is left after stripping the recorded suffix is OUR OWN GLOBAL SHEET. Adopting that as the widget's "own" rules re-appends it under the next global sheet, so the text grows by a whole copy of itself at every theme change. Measured before this check: 1.01x, 2.01x, 3.01x, 4.01x, 5.00x over five themes -- 359 KB where 72 KB was right. That is the accumulation the original captured-once comment existed to prevent, reintroduced by making this re-read.

## apply_stylesheet_per_window

### lines 3887-3898

```python
if app.styleSheet():
```

TAKE THE APPLICATION SHEET DOWN WHEN THE WINDOWS TAKE OVER. An application sheet left standing keeps applying to every widget including the ones inside a window, where an OLD one then shows through as a colour from the wrong theme. Measured: `HomePage inlines #000000 (dark bg)` under the light, cell and glass themes, in a run of 1,045 tests and not one of them alone, because what was stale was left by whichever earlier test had set a sheet globally.

Guarded, because clearing it is itself a full repolish. In production it is empty after the first call, so this is paid once at startup with one small window on screen -- not on the preference save this whole function exists to make cheap.

### lines 3907-3908  _(unsure)_

```python
app.removeEventFilter(_WINDOW_SHEET_FILTER)
```

Re-installing is how the filter survives an application being torn down and rebuilt in one process, which the test suite does constantly.

## clear_widget_qss_overlays

### line 3945  _(unsure)_

```python
pass
```

The C++ widget was deleted while Qt was draining its queue.

## stylesheet

### lines 4175-4177

```python
if load_widget_registrars:
```

Exhaustive callers get every known block in one static sheet. The live application opts out; MainWindow installs blocks registered by a lazy module on that module's screen root before it is shown.

### lines 4183-4186

```python
P = _widget_qss_palette(theme, font_scale, surface_opacity)
```

Surface roles are re-rendered through the theme's scrim alpha. For dark and light every alpha is 1.0 and this is a no-op that emits the same hex it always did; for Space each one becomes an ``rgba()`` so the background image reads through the panel.

### line 4188

```python
ELEVATED = css_color(
```

Opaque variants for the places translucency would be wrong.

### lines 4210-4217

```python
TILE_BG = css_color(base["surface"],
```

Tiles take page opacity on every theme. Over an image they always did; on the flat themes they were `transparent`, which looked identical to the window and meant the tile itself could not be dialled — the user asked for the tiles AND the boxes they sit in to follow the setting. `transparent` is still the right answer at 100%, because a fully opaque tile over an identical window colour is what the flat themes looked like before, so the alpha is applied to the surface colour rather than replacing it with a hard block.

### lines 4220-4227

```python
TROUGH = "transparent" if over_image else page_colour(theme)
```

Scrollbar troughs paint over the window; over a photograph they must not be an opaque black block. Group-box titles are transparent below.

`page`, not `bg`, on the flat themes. A trough runs the full height of the settings column and it took the WINDOW colour, so on the dark theme it was a black stripe down the page — the last 168 pure-black samples in that column after the page itself had a colour, and the only ones left. It is a groove IN the page, so the page is what it should be.

### lines 4229-4232

```python
CONSOLE_BG = (P["surface_alt"] if over_image else css_color(
```

The console honours page opacity on EVERY theme, not just the image ones. `#0a0b0d` was a hard-coded near-black, so on dark and light the console stayed a solid slab no matter where the slider was — one of the containers the preference visibly failed to reach.

### lines 4235-4251

```python
DOCK_BG = (dock_colour(theme) if over_image else css_color(
```

The dock takes page opacity on the FLAT themes and stays opaque over an image. Two user instructions meet here and both are kept:

16j — "the dock to the left should never have a transparent background, either dark gray or white". A navigation column is chrome: it is what you look at when you have lost your place. It used to paint `surface`, which the image themes re-render through `scrim_alpha`, so on Space the app list was a ghost with a galaxy behind every row.

Later — page opacity should reach "the dock" as well.

On dark and light there is no wallpaper to show through, only the ambient animation, so thinning the dock does exactly what was asked and nothing #16j was protecting against. Over Space or Cell the picture is behind it and the old complaint applies verbatim — and the legibility floor does NOT save it there (Cell floors at 0.047), so the split is explicit rather than left to the solver.

### lines 4260-4263

```python
RIM = rim_colour(theme)
```

The theme ink used by outlined horizontal tiles, and the three maturity hues a module tile switches to on hover. Resting AppTiles are rimless; the hover fill is the stage colour at a low alpha so the tile lights UP rather than being replaced by a block of magenta.

### lines 4276-4282

```python
FOLD_STAGE_RULES = "\n".join(
```

THE FOLDED MODULES LIGHT UP LIKE THE TILES THEY REPLACED.

A module folded into a host screen is a button on that host's masthead rather than a tile on Home, and the maturity it promises did not change when it moved. Same hue, same two states, read from the same STAGE_HOVER table -- so signing a module off recolours its button and its tile together, and neither can drift from the other.

### lines 4303-4305

```python
TILE_MIN_H = max(1, int(round(TILE_H * font_scale)))
```

Scaled with the font, like every other Python-set size: a 150 % font makes the name taller, and a tile that did not grow with it would clip the thing it exists to show.

### lines 4308-4309  _(unsure)_

```python
F = {k: max(6, int(round(v * font_scale)))
```

Scaled font sizes so the "Font scale" preference actually resizes the whole app, not just the base body text.

### lines 4316-4321

```python
WIDGET_QSS = registered_widget_qss(P, surface_opacity)
```

Contributed blocks, rendered against the same surfaces the rules below use. `theme` and `font_scale` ride along because a widget that wants `pane_surface`/`pane_alpha` needs the theme name, and this callback is the only place it would otherwise have to guess it (guessing means reading the preference again, which is wrong while a stylesheet is being generated for a theme that is not yet live).

### lines 4323-4326

```python
CLOSE_MARK_RULES = close_mark_rules(theme, F["body"])
```

LAST, after the contributed blocks. The close mark is the one glyph the whole application shares, so a widget block that grows a rule for its own X loses the tie instead of quietly winning it. See `close_mark_rules`.

## close_mark_side

### lines 5706-5708

```python
ink = metrics.tightBoundingRect(CLOSE_MARK)
```

The glyph's own box, not the font's line box. A line box carries ascent, descent and leading for text that is not there, and sizing the square from it made a mark half again as tall as the tab holding it.

## apply_close_mark

### lines 5727-5729

```python
repolish(button)
```

Polished BEFORE it is measured: the style resolves the sheet's font-size onto the widget, so both the glyph below and the control's own size hint describe what will actually be painted.

## size_close_mark

### lines 5754-5760

```python
button.setFixedSize(width, height)
```

Unconditional, and deliberately so. This once read

`if button.size() != (width, height):`, which compares a QSize against a tuple and is therefore true for every size there is -- so the box has always been re-fixed on every FontChange and StyleChange the style delivered. Making that guard real would change WHEN setFixedSize re-applies its minimum and maximum, so the skip is left out rather than introduced here.

## _CloseMarkResizer.eventFilter

### line 5793  _(unsure)_

```python
pass
```

The button went away under the event. Nothing to size.

## _CloseMarkWatcher._sweep

### lines 5844-5845  _(unsure)_

```python
pass
```

The bar went away between the event and this turn of the loop. Nothing left to mark.

## mark_tab_bar

### lines 5890-5892

```python
if hidden:
```

AFTER, not before. `setTabButton` shows whatever it is given, so a mark hidden on the way in comes back visible and puts an X on the page that must not close.
