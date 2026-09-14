# Notes from `spacr/figure_style.py`

Prose lifted out of `spacr/figure_style.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (11 entries)
- [rc_params](#rc_params) (6 entries)
- [apply](#apply) (1 entry)
- [to_rgb](#to_rgb) (2 entries)
- [export_colour](#export_colour) (4 entries)
- [figure_save_mode](#figure_save_mode) (1 entry)
- [theme_ink](#theme_ink) (1 entry)
- [saved_figure_appearance](#saved_figure_appearance) (1 entry)

## Module level

### lines 15-18

```python
"font_family": "Open Sans",
```

The face spaCR ships, registered with the font manager by

`spacr.figure_font` so the name resolves on a machine that never had Open Sans installed. DejaVu Sans was matplotlib's fallback, not a choice.

### lines 24-25

```python
"palette": "colorblind",
```

Colour-blind-safe and print-safe. A screen's categories are nominal, so a sequential map would imply an order that is not there.

### lines 27-36

```python
"background": "none",
```

TRANSPARENT, NOT WHITE. The store keeps deltas from this table, so choosing the default in the panel stores nothing and the figure renders on whatever is behind it -- which meant the declared default and the observed behaviour disagreed, and picking '#FFFFFF' deliberately was the one thing the panel could not express.

'none' is also the right answer for the job: a figure going into a paper or a slide takes the ground it is placed on, and a white rectangle behind it is visible on every dark background it lands on. A user who wants white can still say so, and now it stores.

### line 43, trailing  _(unsure)_

```python
"spines": "left_bottom",
```

all | left_bottom | none

### lines 51-63

```python
"chrome_colour": "",
```

D: THE FURNITURE IS ONE INK (instruction 200)

ONE CONTROL FOR THE LINES, not one per line. The axis spines, the periphery box and the grid are the same thing -- they are the FRAME and making the user set each is the "user chooses each line" complaint. `chrome_colour` is what a user reaches for; `grid_colour` and the rest stay as the per-element override underneath it, and `chrome_of` below resolves the two.

EMPTY MEANS "FOLLOW THE INK", not black. `resolve_ink(theme_target())` is what a figure's text already does, for the reason 178 measured eleven times, and a frame pinned to a literal while the text follows the theme is a figure that looks wrong in one of the two themes.

### lines 100-102

```python
"legend": False,
```

27 LOPIT compartments is a legend taller than the plot, and it costs 40 ms of every redraw. Colour identifies them; the legend only names them.

### lines 110-111  _(unsure)_

```python
"aspect": "equal",
```

A plate is 24x16 wells. Forcing it square stops the wells being square, which is the whole point of looking at one.

### lines 144-145

```python
"error_bars": "sem",
```

The set this may take is STYLE_CHOICES["error_bars"], not the comment that used to be on this line.

### lines 167-174

```python
"mark_colouring": ("group", "uniform", "random"),
```

C: THE MARKS (instruction 200)

BY GROUP is the default and the house rule is why: everything is grey except what the sentence is about. UNIFORM is that rule taken all the way. RANDOM was asked for by name and is for telling points apart by eye -- a random colour per point carries no information, so it belongs in a working view rather than in a figure that makes a claim, and its tooltip says so.

### lines 178-183

```python
"page_shape": ("square", "portrait", "landscape", "wide", "custom"),
```

E: THE SHAPE OF THE PAGE

A NAMED RATIO rather than two boxes of inches. The inches stay for a user who wants a journal's exact column width; the ratio is what somebody choosing how a figure LOOKS is actually choosing, and it keeps the two axes consistent when the size changes.

### lines 472-491

```python
SAVE_MODES = ("print", "screen", "transparent")
```

A SAVED FIGURE IS FOR PAPER, NOT FOR THE SCREEN.

Instruction 150, reported 2026-08-18: "when a graph is saved and the user is on a dark theme white elements are changed to black for saving (text lines, etc)". On a dark theme `spacr.qt.preferences.get_figure_colors()` hands both renderers a WHITE foreground, so the axes, ticks, labels, title and legend are white -- and nothing anywhere inverted them at export time. A PNG saved with a transparent ground even looks right in a dark file manager and disappears when it is pasted into a manuscript, which means the user finds out at the point of writing the paper.

THE DECISION LIVES HERE AND THE APPLICATION DOES NOT. This half is matplotlib-free and Qt-free, like the rest of the module, so the pyqtgraph exporter (`FastPlot._paint_scene`, instruction 150 C) can import `saved_figure_appearance` and get the same answer as `spacr.plot.print_ready` without either of them owning the rule. Two renderers deciding separately what "print" means is the same defect as two engines deciding which statistical test applies.

## rc_params

### lines 313-315

```python
from .figure_font import use_open_sans_for_figures
```

NAMING THE FAMILY IS NOT ENOUGH. A family matplotlib cannot resolve is a silent fallback to DejaVu Sans, not an error, so the bundled files have to be in the font manager before the name is used. Idempotent.

### lines 351-355

```python
spine_ink = chrome_of(style, "spine")
```

THE FRAME IS ONE INK. The spines, the tick marks and the grid are the same furniture, so `chrome_colour` colours all three at once and a per-element value overrides it where the user set one. An empty `chrome_colour` leaves each element exactly where it was, so a style that never mentions the frame renders as before.

### lines 363-365

```python
frame_ink = str(style.get("chrome_colour", "") or "").strip()
```

The grid carries its own colour by default, and a default is not an override: the one control wins over it, and only a grid colour the user actually chose wins back.

### lines 371-375

```python
marker = str(style.get("marker_style", "") or "").strip()
```

THE MARK. `marker_style` is the shape drawn at each point of a line or a series; scatter marks pass their own and are unaffected. Emitted only when it is not the default shape, because these params are pushed into the global rcParams: naming the default would put a marker on every line ever drawn, which is a decision no user made.

### lines 380-383

```python
shape = str(style.get("page_shape", "") or "").strip()
```

THE SHAPE OF THE PAGE. One number in, two out: naming the ratio keeps the two axes consistent when the size changes. `custom` has no ratio it means the caller's own inches -- so it emits no size at all, and neither does the default shape, for the reason above.

### lines 387-392

```python
colours = palette_colours(style.get("palette"))
```

THE PALETTE IS AN rcParam TOO, and it has to be one here rather than only inside `apply`. `spacr.figures.style.rc` -- the only supported way a figure gets this style -- builds its overrides by DIFFING two `rc_params` dicts, so a setting this function does not emit cannot reach a drawn figure at all: a user who picked a palette in Preferences got the Matplotlib default cycle back.

## apply

### line 433, trailing

```python
except Exception:
```

never fail a run over styling

## to_rgb

### line 552, trailing  _(unsure)_

```python
return None
```

fully transparent is not a colour

### line 564, trailing  _(unsure)_

```python
try:
```

the colour spellings only matplotlib reads

## export_colour

### lines 718-721

```python
return None
```

THE DATA NEVER MOVES. A white data point turned black is, on a volcano, the colour of "not a hit" -- section A exists to prevent exactly that, and it is the one line of this function that must never grow a special case.

### lines 725-727

```python
luminance = relative_luminance(current)
```

Only a DARK ground is repainted. A deliberately tinted light background is somebody's choice, and 'transparent' has no ground to argue about -- the writer owns that.

### lines 734-735

```python
replacement = look.grid if kind == "grid" else look.ink
```

A grid repainted in the ink is a cage over the data, so an illegible grid becomes the faint print grey instead.

### lines 737-742

```python
if to_rgb(replacement) == to_rgb(current):
```

AND NOTHING IS "REPAINTED" IN THE COLOUR IT ALREADY IS. The light-mode grid default IS `PRINT_GRID`, and #DDDDDD on white is 1.27 contrast deliberately faint, correctly below the chrome floor, and already the colour it would be changed to. Saying None here is what makes "a light-mode save changes nothing at all" true of the artists as well as of the pixels, and it leaves the caller nothing to restore.

## figure_save_mode

### line 830, trailing  _(unsure)_

```python
try:
```

the GUI's own answer, when there is one

## theme_ink

### lines 871-873

```python
return (PRINT_INK, PRINT_GRID) if theme == "light" else (DARK_INK, DARK_GRID)
```

`resolve_effective_theme` says so itself: compare against "light" and treat everything else as dark, because Space and Cell are dark themes and "system" has already been resolved by the time it answers.

## saved_figure_appearance

### lines 898-913

```python
ink, grid = theme_ink()
```

THE INK FOLLOWS THE THEME HERE, and only here.

Asked for twice: "the background of the figures should be transparent and the lines should be white on a dark theme and black on a light one", and then reported as a fault -- "a lot of the text in the figures is black on the dark theme and the axes as well".

This mode used to keep the PRINT ink on the ground that it removes, on the argument that dark ink on a transparent ground is still unreadable on a dark slide. That argument is right about `print` and wrong about this: transparent MEANS the ground is whatever the figure is pasted onto, and the only thing that knows what that is, is the user -- who says so by the theme they are working in.

`print` is unchanged and is still the default, so a figure going into a manuscript is untouched by this.
