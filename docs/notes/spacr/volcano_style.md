# Notes from `spacr/volcano_style.py`

Prose lifted out of `spacr/volcano_style.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [VolcanoStyle](#volcanostyle) (2 entries)
- [VolcanoStyle.to_dict](#volcanostyleto_dict) (1 entry)
- [_resolve_effect_threshold](#_resolve_effect_threshold) (4 entries)
- [render_volcano](#render_volcano) (2 entries)
- [_colour_values](#_colour_values) (1 entry)
- [_draw_by_localization](#_draw_by_localization) (1 entry)
- [_draw_points](#_draw_points) (1 entry)
- [_scatter_by_shape](#_scatter_by_shape) (1 entry)
- [_annotate](#_annotate) (1 entry)
- [_finish_axes](#_finish_axes) (4 entries)
- [validate_style](#validate_style) (2 entries)

## Module level

### lines 21-24

```python
from .style_base import SCALES as _SCALES
```

THE SHARED VOCABULARY (108 point 1). `FigureStyle` holds every field a reader would recognise on any figure -- axes, type, grid, legend, page so a house style saved on a volcano can be applied to another figure that shares those names, and "font size" is one setting in spaCR.

### lines 32-34

```python
from .figures.style import figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

## VolcanoStyle

### lines 121-125

```python
x_label: str = "Standardized marginal effect"
```

axes

`y_label`, `title`, the two scales, the two limits and the two inverts are INHERITED. Only the default LABEL is restated: a base class cannot know what this figure's x axis is, and the volcano's has been "Standardized marginal effect" since it was written.

### lines 239-241

```python
annotations: dict = field(default_factory=dict)
```

text: the SIZES are inherited (font_family, font_size, title_font_size, label_font_size, tick_font_size, font_weight). What is here is the labelling, which is a volcano's own question.

## VolcanoStyle.to_dict

### lines 247-249

```python
def to_dict(self) -> dict:
```

frame: INHERITED in full (figure_width, figure_height, dpi, grid, grid_axis, grid_color, grid_width, hide_top_right_spines, legend, legend_location, background_color, transparent).

## _resolve_effect_threshold

### lines 330-334

```python
median = float(np.median(controls))
```

MAD rather than std: one control that went wrong should not widen the null it is supposed to define. 1.4826 makes it a consistent estimator of sigma under normality, the same scaling the 'mad' branch uses, so the two are directly comparable and the difference between them is exactly "did the hits inflate it".

### lines 338-339

```python
raise ValueError(
```

Controls all identical -- degenerate, and a zero cut would mark every guide significant. Say so instead.

### line 354

```python
return mad * 1.4826 * multiplier
```

1.4826 makes the MAD a consistent estimator of sigma under normality.

### line 357

```python
quantile = min(max(multiplier, 0.5), 0.999999)
```

The multiplier is the quantile itself here, e.g. 0.99.

## render_volcano

### lines 425-428

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 463-468

```python
from .plot import save_figure
```

108 point 6, through the one writer -- and the style still wins where it has an opinion. THE EXTENSION IS THE CALLER'S: this is the headless renderer and its `save_path` is a filename someone chose, so `fmt` is taken from it rather than from the preference, which would rename the file under them. What it gains is the DPI rule, the TrueType embedding, and the repaint for paper.

## _colour_values

### line 486  _(unsure)_

```python
if numeric.notna().mean() > 0.9:
```

A column that is mostly unparseable is a category, whatever its dtype.

## _draw_by_localization

### lines 563-564  _(unsure)_

```python
wanted = list(dict.fromkeys(str(name) for name in style.localizations))
```

`dict.fromkeys` keeps the offered order and drops a repeat, so the same combination is drawn the same however it was ticked.

## _draw_points

### lines 598-599  _(unsure)_

```python
return None
```

No mappable: a compartment is a category, and a colour bar over categories is a scale that reads as continuous when it is not.

## _scatter_by_shape

### line 654  _(unsure)_

```python
same_source = style.shape_by == style.color_by
```

When colour and shape encode the SAME column, "GRA · GRA" is noise.

## _annotate

### lines 722-724

```python
edges = []
```

A break has no data coordinate to draw on. Put the anchor on its nearest visible edge and move the text towards the panel's interior, or clipping erases the label while retaining its Text.

## _finish_axes

### line 835  _(unsure)_

```python
panels[0].spines["bottom"].set_visible(False)
```

The break marks. Hide the shared edge, then draw the diagonal ticks.

### lines 839-841

```python
break_ink = str(style.axis_color or "").strip() or "#404040"
```

THE BREAK MARKS ARE AXIS FURNITURE, so they take the axis ink rather than a grey of their own: a black-axes volcano with two grey ticks at the break reads as a rendering fault.

### lines 871-872  _(unsure)_

```python
_paint_ink(figure, panels, style, ground)
```

LAST, so it reaches the legend and the colour bar the lines above have only just created.

### lines 874-875

```python
if not style.split_axis and not (mappable is not None and style.show_colorbar):
```

tight_layout cannot lay out a broken axis or a figure-level colorbar and warns instead of doing nothing, so it is only run when it applies.

## validate_style

### lines 960-964

```python
try:
```

The control rule is the one that can fail on DATA rather than on a typo -- too few controls, or controls with no spread -- so it is asked rather than guessed, and the resolver is the thing that knows. Attributed to `threshold_method`, because that is the control the reader chose and can take back.

### lines 975-976

```python
pass
```

A fault that is not about this setting -- a missing x column, say -- is already reported against the setting it belongs to.
