# Notes from `spacr/figures/style.py`

Prose lifted out of `spacr/figures/style.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Palette](#palette) (7 entries)
- [Module level](#module-level) (7 entries)
- [user_overrides](#user_overrides) (1 entry)
- [rc](#rc) (6 entries)
- [theme_target](#theme_target) (1 entry)
- [panel_letter](#panel_letter) (1 entry)

## Palette

### line 34, trailing  _(unsure)_

```python
GREY = "#B4B4B4"
```

default data, non-significant, comparisons

### line 35, trailing  _(unsure)_

```python
GREY_DARK = "#7F7F7F"
```

secondary series, mean bars

### line 36, trailing  _(unsure)_

```python
BLUE = "#2E77BC"
```

the primary highlight / the gene of interest

### line 37, trailing  _(unsure)_

```python
BLUE_LIGHT = "#7FB3E0"
```

a second strain

### line 38, trailing  _(unsure)_

```python
GREEN = "#2E7D4F"
```

wild type, upregulated

### line 39, trailing  _(unsure)_

```python
RUST = "#C4441C"
```

downregulated, the other highlight

### line 40, trailing  _(unsure)_

```python
CORAL = "#E8A88C"
```

density and histogram fills

## Module level

### line 54, trailing  _(unsure)_

```python
"data": Palette.GREY,
```

every guide that is not the point

### line 55, trailing  _(unsure)_

```python
"up": Palette.GREEN,
```

positive effect, called

### line 56, trailing  _(unsure)_

```python
"down": Palette.RUST,
```

negative effect, called

### line 57, trailing  _(unsure)_

```python
"highlight": Palette.BLUE,
```

the selected gene

### line 60, trailing  _(unsure)_

```python
"fill": Palette.CORAL,
```

histogram and density fills

### line 61, trailing

```python
"reference": Palette.GREY_DARK,
```

thresholds, limits, 1:1 lines

### line 68, trailing  _(unsure)_

```python
"label": 7.0,
```

the 1.0x reference

## user_overrides

### lines 221-223

```python
if not general and not per_graph:
```

Both stores are EMPTY until the user changes something -- they hold the deltas, not the defaults, on purpose -- so this is the common case and it costs nothing.

## rc

### lines 264-266

```python
picked_ink = ink or chosen_ink()
```

Read each control ONCE. `picked_*` is None while that half follows the theme, and it is also the flag that decides whether the choice outranks the house-style panel's own colours further down.

### lines 279-285

```python
"font.sans-serif": [_FIGURE_FAMILY, "Helvetica", "Arial",
```

OPEN SANS SHIPS WITH spaCR, so it is always there to resolve. Naming "Helvetica" first meant a Linux machine without it fell silently back to DejaVu Sans, and figures came out in a different face from the interface around them -- and in a different face on each contributor's machine. `use_open_sans_for_figures` registers the bundled files with the font manager, which is what makes the name resolve at all; the rest of the list stays as a fallback.

### lines 296-297  _(unsure)_

```python
"axes.grid": False,
```

NO GRIDLINES. EVER. The published figures have none, and a grid is the fastest way to make a panel look like a spreadsheet.

### lines 301-304

```python
"xtick.color": line_colour, "ytick.color": line_colour,
```

THE MARK IS A LINE, THE LABEL IS TEXT. `xtick.color` is the little dash beside the axis; `xtick.labelcolor` is the number printed next to it. Matplotlib's default for the second is "inherit", so both are named here or the two can never be told apart.

### lines 325-326  _(unsure)_

```python
params.update(user_overrides(kind))
```

LAST, so the user wins. Everything above is the published look; this is the handful of settings they went into Preferences and changed.

### lines 328-333

```python
if picked_ink:
```

LATER STILL, and only for a colour the user actually named. The two figure-colour controls are the dedicated ones for these roles, so they outrank the graph-style panel's general `foreground` — which resolves to `xtick.color` and would otherwise repaint the tick marks the line control was just told to own. Nothing is written here while both halves follow the theme, so an untouched store keeps the house style exactly.

## theme_target

### lines 385-386  _(unsure)_

```python
return "print" if text in ("white", "#ffffff", "#fff") else "screen"
```

A white or very light ground means the figure is destined for paper, whatever the GUI theme is doing.

## panel_letter

### lines 390-392  _(unsure)_

```python
def panel_letter(ax, letter: str, dx: float = -0.16, dy: float = 1.06) -> None:
```

The small vocabulary every panel shares
