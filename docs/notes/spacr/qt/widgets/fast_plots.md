# Notes from `spacr/qt/widgets/fast_plots.py`

Prose lifted out of `spacr/qt/widgets/fast_plots.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (5 entries)
- [_Absorbs.__iter__](#_absorbs__iter__) (1 entry)
- [_Absorbs.__bool__](#_absorbs__bool__) (1 entry)
- [style_field_kind](#style_field_kind) (3 entries)
- [style_field_label](#style_field_label) (1 entry)
- [add_style_file_entries._load](#add_style_file_entries_load) (1 entry)
- [add_style_file_entries._make_default](#add_style_file_entries_make_default) (1 entry)
- [add_style_file_entries._clear_default](#add_style_file_entries_clear_default) (1 entry)
- [add_style_file_entries](#add_style_file_entries) (2 entries)
- [apply_default_style](#apply_default_style) (1 entry)
- [_add_style_entry](#_add_style_entry) (3 entries)
- [_ask_style_value](#_ask_style_value) (2 entries)
- [_figure_colors](#_figure_colors) (1 entry)
- [_violin_profile](#_violin_profile) (2 entries)
- [FastPlot.__init__](#fastplot__init__) (13 entries)
- [FastPlot._reset_scene](#fastplot_reset_scene) (1 entry)
- [FastPlot.add_smoother](#fastplotadd_smoother) (1 entry)
- [FastPlot._refresh_level_control](#fastplot_refresh_level_control) (1 entry)
- [FastPlot.shape_columns](#fastplotshape_columns) (1 entry)
- [FastPlot._install_axis_hooks](#fastplot_install_axis_hooks) (2 entries)
- [FastPlot._describe_item](#fastplot_describe_item) (7 entries)
- [FastPlot._register_drawn](#fastplot_register_drawn) (1 entry)
- [FastPlot.log_reason](#fastplotlog_reason) (3 entries)
- [FastPlot._install_split_ticks](#fastplot_install_split_ticks) (1 entry)
- [FastPlot._place](#fastplot_place) (1 entry)
- [FastPlot._apply_log](#fastplot_apply_log) (1 entry)
- [FastPlot._axis_label_text](#fastplot_axis_label_text) (1 entry)
- [FastPlot._relabel_axes](#fastplot_relabel_axes) (2 entries)
- [FastPlot._reapply_pinned](#fastplot_reapply_pinned) (1 entry)
- [FastPlot.auto_range_axes](#fastplotauto_range_axes) (1 entry)
- [FastPlot.apply_text_style](#fastplotapply_text_style) (5 entries)
- [FastPlot.axis_items](#fastplotaxis_items) (1 entry)
- [FastPlot.set_line_style](#fastplotset_line_style) (5 entries)
- [FastPlot.colour_by_column](#fastplotcolour_by_column) (1 entry)
- [FastPlot._gated](#fastplot_gated) (1 entry)
- [FastPlot.build_style_menu](#fastplotbuild_style_menu) (22 entries)
- [FastPlot._categorical_opacity](#fastplot_categorical_opacity) (1 entry)
- [FastPlot._export_pdf](#fastplot_export_pdf) (1 entry)
- [FastPlot._ask_point_colour](#fastplot_ask_point_colour) (1 entry)
- [FastPlot._ask_axis_limits](#fastplot_ask_axis_limits) (1 entry)
- [FastPlot._toggle_legend](#fastplot_toggle_legend) (1 entry)
- [FastPlot.add_scatter](#fastplotadd_scatter) (8 entries)
- [FastPlot.set_keys](#fastplotset_keys) (1 entry)
- [FastPlot.highlight_key](#fastplothighlight_key) (2 entries)
- [FastPlot._draw_marker](#fastplot_draw_marker) (1 entry)
- [FastPlot.highlight_keys](#fastplothighlight_keys) (2 entries)
- [FastPlot._install_rubber_band.drag](#fastplot_install_rubber_banddrag) (1 entry)
- [FastPlot._clear_extra_highlights](#fastplot_clear_extra_highlights) (1 entry)
- [FastPlot.add_ranked_bars](#fastplotadd_ranked_bars) (1 entry)
- [FastPlot._beeswarm_offsets](#fastplot_beeswarm_offsets) (1 entry)
- [FastPlot.add_radar](#fastplotadd_radar) (3 entries)
- [FastPlot.add_line](#fastplotadd_line) (2 entries)
- [FastPlot.add_group_mark](#fastplotadd_group_mark) (6 entries)
- [FastPlot._on_points_clicked](#fastplot_on_points_clicked) (1 entry)
- [FastPlot._describe](#fastplot_describe) (1 entry)
- [FastPlot.export](#fastplotexport) (3 entries)
- [FastPlot._write_export](#fastplot_write_export) (2 entries)
- [FastPlot._dressed_for_the_file](#fastplot_dressed_for_the_file) (3 entries)
- [FastPlot.styled_snapshot](#fastplotstyled_snapshot) (1 entry)
- [FastPlot.export_bundle](#fastplotexport_bundle) (1 entry)
- [FastPlot._offer_graph_kinds](#fastplot_offer_graph_kinds) (3 entries)
- [FastPlot._shape_the_image](#fastplot_shape_the_image) (1 entry)
- [FastPlot.snapshot](#fastplotsnapshot) (1 entry)
- [FastPlot._render_snapshot](#fastplot_render_snapshot) (4 entries)
- [FastPlot.restyle](#fastplotrestyle) (3 entries)
- [VolcanoPlot.__init__](#volcanoplot__init__) (1 entry)
- [VolcanoPlot.set_p_axis](#volcanoplotset_p_axis) (1 entry)
- [VolcanoPlot._q_strength](#volcanoplot_q_strength) (2 entries)
- [VolcanoPlot._q_ramp](#volcanoplot_q_ramp) (2 entries)
- [VolcanoPlot._q_opacity](#volcanoplot_q_opacity) (1 entry)
- [VolcanoPlot.set_correction](#volcanoplotset_correction) (1 entry)
- [VolcanoPlot.set_results](#volcanoplotset_results) (25 entries)
- [VolcanoPlot._add_significance_lines](#volcanoplot_add_significance_lines) (4 entries)
- [VolcanoPlot._build_caption](#volcanoplot_build_caption) (2 entries)
- [VolcanoPlot._offer_p_axes](#volcanoplot_offer_p_axes) (1 entry)
- [VolcanoPlot._offer_encodings](#volcanoplot_offer_encodings) (1 entry)
- [EffectRankPlot.__init__](#effectrankplot__init__) (1 entry)
- [EffectRankPlot.set_results](#effectrankplotset_results) (6 entries)
- [EffectRankPlot._label_series](#effectrankplot_label_series) (1 entry)
- [BinnedPlot.__init__](#binnedplot__init__) (1 entry)
- [BinnedPlot._fill_bins](#binnedplot_fill_bins) (1 entry)
- [BinnedPlot._on_scene_clicked](#binnedplot_on_scene_clicked) (2 entries)
- [BinnedPlot.highlight_bin](#binnedplothighlight_bin) (2 entries)
- [PValueHistogram.set_p_values](#pvaluehistogramset_p_values) (1 entry)
- [QQPlot.set_p_values](#qqplotset_p_values) (3 entries)
- [ResidualPlot.set_residuals](#residualplotset_residuals) (2 entries)
- [InfluencePlot.set_influence](#influenceplotset_influence) (2 entries)
- [GroupedPlot.set_mark](#groupedplotset_mark) (1 entry)
- [ControlSeparation.set_groups](#controlseparationset_groups) (5 entries)
- [GuideAgreementPlot.set_support](#guideagreementplotset_support) (5 entries)
- [ResultsTable.__init__](#resultstable__init__) (3 entries)
- [ResultsTable.set_frame](#resultstableset_frame) (3 entries)
- [ResultsTable._apply_filter](#resultstable_apply_filter) (2 entries)
- [ResultsTable.select_keys](#resultstableselect_keys) (1 entry)

## Module level

### line 29, trailing  _(unsure)_

```python
try:
```

exercised by the import guard test

### lines 90-95

```python
pg = _Absorbs()
```

THE MODULE TOO, not only the widget. Thirty call sites in this file go through `pg.` -- mkBrush, mkPen, ScatterPlotItem, InfiniteLine -- and `pg = None` turns every one into an AttributeError the moment a table arrives. The panel would then BUILD and die on its first redraw, which is a worse failure than the original: the app looks fine until the user loads data.

### lines 286-297

```python
_Drawn = namedtuple("_Drawn", "item x y blocks kind counts")
```

One positionable thing on a plot, with the DATA coordinates it was handed. Private on purpose: it is bookkeeping for the log transform, not a contract anyone outside this module holds.

x, y     float arrays -- or None where the item does not move on that axis, as a horizontal reference line does not move in x. blocks   {axis: why this item can NEVER be logged on that axis} kind     "points", "bar", or "line" -- so a refusal can name what is in the way rather than quoting a count of anonymous "values". counts   {axis: (at or below zero, finite in total, the lowest of them)}, measured once at registration rather than re-measured every time a menu opens.

### lines 379-386

```python
_STYLE_GROUPS = (
```

Which group of the right-click menu a style field belongs on, as ``(group, name fragments)`` tried in order; anything unmatched is Appearance. A plain module constant rather than a documented one, because it is a layout table for this module's menu and not a contract anyone outside it holds.

ORDER IS THE ORDER OF USE, the same as `build_style_menu`: what changes what the figure CLAIMS, then its axes, then how it looks, then how big it is.

### lines 401-403

```python
STYLE_FIELD_KINDS = ("flag", "colour", "choice", "multi", "number", "pair",
```

What a style field can be edited WITH. A field whose kind is "unsupported" still gets an entry, greyed and saying so: a setting silently absent from the menu is one the user is told exists and cannot find.

## _Absorbs.__iter__

### lines 62-65

```python
def __iter__(self):
```

Chains have to survive whole, not one link at a time. The subclasses run `self.plot.scene().sigMouseClicked.connect(...)` in their own init__ -- three links -- so returning a plain [] from the first call only moves the AttributeError one step along.

## _Absorbs.__bool__

### lines 75-76

```python
"""False, so ``if self._highlight:`` reads as "nothing is drawn".
```

`if self._highlight:` must read as "nothing is drawn", which is true, rather than as a live artist to remove.

## style_field_kind

### lines 430-433

```python
if (isinstance(value, (tuple, list))
```

A CLOSED SET HELD IN A CONTAINER IS TICKED, NOT PICKED. "Dense granules and rhoptries 1" is one question rather than two, so a field whose value is a tuple of members of the offered list gets a submenu where any number of them can be on at once.

### lines 452-454

```python
return "unsupported"
```

A NESTED SHAPE HAS NO DIALOG. Offering the split axis's pair of pairs as one pair of numbers would write a value the renderer cannot read -- worse than saying it is not editable here.

### line 464  _(unsure)_

```python
if str(name).endswith(("_lim", "_lims")):
```

Nothing declared and nothing held: the name is the only clue left.

## style_field_label

### lines 519-521

```python
chosen = [str(item) for item in (value or ())]
```

WHAT IS TICKED, spelled out. A multi-select whose label said only its name would make "is anything selected?" a question the reader has to open the submenu to answer.

## add_style_file_entries._load

### lines 770-773

```python
_say(f"Could not load that style: {exc}")
```

NAMED, not swallowed. The two ways this fails -- a file that is not a style and a style of another kind -- are both things the user can act on, and a menu entry that silently does nothing is the one failure this module keeps being written to avoid.

## add_style_file_entries._make_default

### line 786, trailing  _(unsure)_

```python
except Exception:
```

no settings store

## add_style_file_entries._clear_default

### line 798, trailing  _(unsure)_

```python
except Exception:
```

no settings store

## add_style_file_entries

### line 822, trailing  _(unsure)_

```python
except Exception:
```

no settings store

### lines 825-827

```python
action.setEnabled(has_default)
```

Greyed when there is nothing to clear (106), and it doubles as the readout for "is a house style in force here?", which is the question a user arrives with when a figure does not look like the package's.

## apply_default_style

### line 849, trailing  _(unsure)_

```python
except Exception:
```

no settings store

## _add_style_entry

### lines 865-867

```python
named = dict((labels or {}).get(name) or {})
```

WHAT A VALUE IS CALLED, where that is not what it is. A marker stored as "o" and a line style stored as "--" are what a saved style file says and are not what a reader picks off a menu.

### lines 897-898

```python
entry = submenu.addAction(_shown(option))
```

`None` is a real option -- it is how a colour-by column is taken back off -- and "None" is not what that reads as on a menu.

### line 912  _(unsure)_

```python
action = submenu.menuAction()
```

The GROUP's action is what a reader meets, so it carries the name.

## _ask_style_value

### lines 987-991

```python
colour = pick_colour(parent, value or "#000000", pretty)
```

A SEVENTH CALL SITE, which is the reason `pick_colour` exists. Instruction 151 counted six unguarded `QColorDialog.getColor` calls in the tree; this one -- the figure style's own colour fields -- was not among them, so the count was low and the flag would have been forgotten here even after the six were fixed.

### lines 1015-1017

```python
_apply_style(style, name, (first, second), on_change)
```

CANCELLING THE SECOND ABANDONS THE FIRST, the same rule the axis-limit dialog follows: a half-set pair is a range nobody chose.

## _figure_colors

### line 1093, trailing  _(unsure)_

```python
except Exception:
```

no settings store available

## _violin_profile

### lines 1176-1185

```python
peak = float(counts.max())
```

NO `peak <= 0` GUARD. The histogram's range IS the data's own min and max, and the guard above has already required both to be finite with high > low -- so every value of v lies inside the range and at least one bin is populated. `counts.max()` cannot be zero.

NaN cannot sneak past either: `np.min` PROPAGATES NaN rather than ignoring it, so an array holding one fails the finite check above rather than arriving here with an empty histogram. Checked against 20,000 random finite arrays spanning twelve orders of magnitude every one had a populated bin.

### lines 1188-1189

```python
centres = np.concatenate([[low], centres, [high]])
```

Pinned shut at both ends, so the outline closes on the data's range instead of stopping mid-air at the first and last bin's width.

## FastPlot.__init__

### lines 1234-1236

```python
self._init_axis_state()
```

BEFORE THE BRANCH, so both constructors have it: the hooks below fire on the very first setLabel, and the pyqtgraph-absent path still has to answer log_axes() without raising.

### lines 1249-1257

```python
self._background, self._foreground = _figure_colors()
```

BACKGROUND None IS TRANSPARENT, WHICH WAS ALREADY RIGHT. The ink was not: `foreground="k"` hardcoded BLACK axes, ticks and labels, so on a dark theme the plot drew black-on-transparent over a dark surface and the axes were invisible. The matplotlib path has resolved this correctly for a while via preferences.get_figure_colors(), which returns TRANSPARENT_FIGURE_BG plus theme-correct ink and honours an explicit colour the user has chosen; pyqtgraph simply never asked it. Same source for both renderers, so a theme switch cannot move one and not the other.

### lines 1265-1267

```python
self._header = QHBoxLayout()
```

ABOVE THE PLOT, BESIDE THE TITLE. Empty it costs no height, and it stays empty on every plot whose host offers no levels -- which is every plot but the volcano.

### lines 1274-1276

```python
self._install_axis_hooks()
```

EVERY ITEM AND EVERY LABEL, CAUGHT ON THE WAY IN -- see :meth:`_install_axis_hooks`. It goes in before the first label is set, because the label is one of the two things it catches.

### lines 1278-1281

```python
self._install_rubber_band()
```

Modified left-drag selects a region (instruction 206). In here rather than in a subclass because the funnel every plot emits through is on this class, and a band that only some plots had would be a gesture the user cannot rely on.

### lines 1287-1291

```python
self.plot.setBackground(None)
```

A transparent pyqtgraph background is not enough on its own: the QWidget it lives in still paints the theme's `bg` under the blanket QWidget rule, so the plot sits on an opaque slab regardless. The theme's own helper is what every other transparent surface here uses -- see the hyperparam screen, which does exactly this.

### line 1297, trailing  _(unsure)_

```python
except Exception:
```

theme absent

### lines 1301-1306

```python
controls = QHBoxLayout()
```

THE STRIP CARRIES WHAT IS PRESSED, NOT WHAT IS SET. Log x, log y and grid were checkboxes here and are entries on the right-click menu now (instruction 148 C): they are set once and then read off the axis, so a permanent row of them under every plot spent screen on three states nobody looks at twice. The legend stays, because it is the one a reader flicks on and off while looking at the figure.

### lines 1318-1320

```python
reset.clicked.connect(self.auto_range_axes)
```

`auto_range_axes`, NOT `plot.autoRange`: the bare call freezes the axes on today's points, so the NEXT redraw opens inside this run's window. See that method.

### lines 1323-1332

```python
save = QPushButton("Save figure…")
```

NO EXPORT BUTTON (187 D). Reported 2026-08-20: "the export button dosnt cause any errors but the exported figure is broken, with massive text and so on ... actually remove the export button, save styled is enough."

TWO DOORS AND ONE OF THEM WROTE BLIND. `export` writes with no preview and no styling pass, so a page sized in millimetres got text scaled for the screen -- which is the massive text. Save styled shows what it is about to write, which is the difference, and it is the door that stays.

### lines 1358-1360

```python
self._shape_legend = None
```

The other two layered channels' keys, set by `set_results` and read by the legend. Declared here so a plot that has never drawn still answers what its legend would say: nothing.

### lines 1365-1369

```python
self._keys: Sequence[str] = ()
```

THE KEY JOIN. Row-to-point highlighting is joined on the identifier the row carries, never on a position -- a table sorted by effect and a scatter drawn in input order are the same points in two orders, and joining them by index lights up the WRONG guide silently, in exactly the direction nobody questions, because something lit up.

### line 1431  _(unsure)_

```python
self.plot.setContextMenuPolicy(Qt.CustomContextMenu)
```

Right-click to restyle, the same gesture the matplotlib figures use.

## FastPlot._reset_scene

### lines 1512-1515

```python
self._drawn = []
```

The log transform's bookkeeping goes with the artists it described. THE SCALE ITSELF STAYS: a user who asked for a log axis has not unasked for it by filtering the table, and every item the redraw puts back is transformed as it arrives.

## FastPlot.add_smoother

### lines 1684-1685

```python
return str(refusal)
```

A refusal is the answer, not a failure: a Gaussian process asked for more rows than it can take says the number.

## FastPlot._refresh_level_control

### lines 1780-1782

```python
blocked = self._level_box.blockSignals(True)
```

`activated` fires on a USER's choice only, so refilling cannot re-enter the callback -- blocked anyway, because a future currentIndexChanged here would, and silently.

## FastPlot.shape_columns

### line 1916, trailing  _(unsure)_

```python
except Exception:
```

an unhashable cell

## FastPlot._install_axis_hooks

### lines 2061-2064

```python
try:
```

A MOUSE DRAG FORGETS A TYPED LIMIT. Panning or zooming by hand is a user saying "not that window any more"; remembering the old one and snapping back to it the next time the scale changes would make the plot argue with the person driving it.

### line 2067, trailing  _(unsure)_

```python
except Exception:
```

no such signal

## FastPlot._describe_item

### line 2078  _(unsure)_

```python
def _describe_item(self, item):
```

what is drawn, and what it says about being logged

### lines 2094-2100

```python
if angle == 90.0:
```

THE ANGLE IS ASKED FIRST, AND THAT ORDER IS THE POINT. `InfiniteLine.value()` answers with a scalar only for the two orthogonal angles; an oblique line answers with the whole ``[x, y]`` position, and `float()` of a list is a TypeError so reading the value before the branch made adding a diagonal line to any plot in this module raise out of `addItem` instead of reaching the branch written to ignore it.

### line 2105, trailing  _(unsure)_

```python
else:
```

an oblique line moves on neither axis

### line 2112, trailing  _(unsure)_

```python
except Exception:
```

an odd bar spec

### lines 2113-2114

```python
blocks = {"x": "one of the bars cannot be re-measured",
```

A BAR WE CANNOT READ IS A BAR WE CANNOT MOVE, and a scale that leaves one item behind is the bug this file is fixing.

### lines 2124-2127

```python
blocks["y"] = ("the bars are measured from zero, which has "
```

THE BASELINE IS THE POINT. A histogram's bars are measured from zero, and zero has no logarithm; saying that is a better answer than "50 of 100 values are at or below zero", which is true and tells the reader nothing they can act on.

### line 2136, trailing  _(unsure)_

```python
except Exception:
```

an empty curve

## FastPlot._register_drawn

### lines 2166-2169

```python
refused = [axis for axis in ("x", "y")
```

A REDRAW CAN MAKE A LIVE LOG SCALE IMPOSSIBLE. The level filter admits a coefficient of zero, a compartment's p-values reach 1 and log10 of those is not a number, so the points would silently leave the plot. The scale comes off instead, and says so.

## FastPlot.log_reason

### line 2200, trailing  _(unsure)_

```python
except Exception:
```

absent axis

### lines 2203-2205

```python
return (f"log {axis}: this axis names its groups rather than "
```

THE AXIS IS A LIST OF GROUPS. A control panel's x and an effect ranking's y carry names at hand-placed positions; a logarithm of a position that stands for "nc" is not a quantity.

### lines 2220-2222

```python
return (f"log {axis}: {point_bad:,} of {point_total:,} points "
```

THE SENTENCE INSTRUCTION 148 ASKED FOR, to the comma. A count and a total, because "some points cannot be logged" leaves the reader unable to judge whether the scale is worth having.

## FastPlot._install_split_ticks

### line 2455, trailing  _(unsure)_

```python
except Exception:
```

absent axis

## FastPlot._place

### lines 2542-2545

```python
if xs is not None:
```

IN PLACE, not through setData. `ScatterPlotItem.setData` clears the item and re-adds the points, which drops the per-point row index every click on this plot is resolved through -- so the dots would move and stop identifying anything.

## FastPlot._apply_log

### lines 2568-2569  _(unsure)_

```python
continue
```

Removed since it was registered -- the previous selection ring, a bar that was outlined and then was not.

## FastPlot._axis_label_text

### lines 2590-2594

```python
notes = []
```

SAY IT WHERE THE AXIS IS. A tick on a menu nobody has open is not notice; "-log10(p)" was already this idea, and a logged axis owes the reader the same sentence. A SPLIT axis owes it more: its ruler is piecewise linear, and a reader measuring a distance on it without being told is measuring the wrong thing.

## FastPlot._relabel_axes

### lines 2616-2618

```python
continue
```

AN UNLABELLED AXIS STAYS UNLABELLED. `setLabel` calls

`showLabel()`, so writing the empty string onto the control panel's deliberately bare x-axis grows a blank strip there.

### lines 2622-2623  _(unsure)_

```python
self.apply_text_style()
```

`setLabel` takes the style with the text, so relabelling drops a font the user chose off the menu unless it is put back.

## FastPlot._reapply_pinned

### lines 2746-2748

```python
self._pinned[axis] = None
```

The scale just changed under a limit that cannot survive it. Forgetting it is the honest answer -- the axis goes back to its data rather than to a bound nobody typed.

## FastPlot.auto_range_axes

### lines 2765-2775

```python
box.enableAutoRange(x=True, y=True)
```

ENABLED LAST, AND THAT ORDER IS THE WHOLE FIX. pyqtgraph's

`autoRange()` ends in `setRange(..., disableAutoRange=True)`, so calling it turns auto-ranging OFF -- and this method, whose entire job is to give the axes back to the data, was leaving them frozen on whatever happened to be drawn at the moment it ran.

The panel calls this between runs, BEFORE drawing the new table. Measured: run A spanning +-13 followed by run B spanning +-0.6 opened run B inside run A's window, twenty times too wide, and "Reset view" -- which autoranges the points that are now there put it right. That is exactly what was reported.

## FastPlot.apply_text_style

### line 2845, trailing  _(unsure)_

```python
except Exception:
```

absent axis

### lines 2854-2858

```python
if axis.labelText:
```

AN AXIS WITH NO LABEL IS LEFT ALONE. `setLabel` calls

`showLabel()`, so restyling the empty string would make the control panel -- whose x-axis is deliberately unlabelled, because its ticks already name the groups -- grow a blank strip under it the first time anyone changed the font.

### lines 2871-2878

```python
for item in self.line_items():
```

A THRESHOLD LINE'S CAPTION IS TEXT, NOT PART OF THE LINE. "p=0.05" and "FDR 5%" used to be recoloured by `set_line_style`, on the reasoning that a red word beside a green line looks wrong. The maintainer's decision (instruction 152 B) is the other way and it is the one that can be stated in a sentence: "a font color that controls the color of all font in the graph". A caption that followed the line would make "all font" untrue, and would be the one string on the figure the font control could not reach.

### line 2885, trailing  _(unsure)_

```python
except Exception:
```

not a labelled line

### line 2895, trailing  _(unsure)_

```python
except Exception:
```

an odd legend item

## FastPlot.axis_items

### line 2938, trailing  _(unsure)_

```python
except Exception:
```

absent axis

## FastPlot.set_line_style

### lines 2983-2986

```python
return 0
```

NOTHING WAS DRAWN, so there is nothing to restyle. Answering zero is the honest reply; reaching into a plot that was never built is the half-built-widget trap `_build_without_pyqtgraph` exists to close.

### lines 2995-2998

```python
if not hasattr(item, "_spacr_base_colour"):
```

THE COLOUR IT WAS DRAWN WITH, REMEMBERED ONCE. Without it

"Follow the theme" has nothing to go back to and would have to invent a colour, which is the same class of mistake as persisting a resolved default.

### lines 3014-3018

```python
try:
```

THE TICK MARKS, SEPARATELY. `setPen` paints the spine; pyqtgraph draws the little dashes with `tickPen` and falls back to the spine's pen only while none is set -- so an axis that has ever been given one keeps drawing its ticks in the old ink unless this line is here.

### line 3021, trailing  _(unsure)_

```python
except Exception:
```

older pyqtgraph

### lines 3023-3024

```python
return touched
```

THE CAPTIONS ARE NOT TOUCHED HERE. "p=0.05" is text and follows the font control -- see :meth:`apply_text_style`.

## FastPlot.colour_by_column

### lines 3062-3066

```python
if colormap not in COLORMAPS:
```

ASKED BEFORE pyqtgraph IS. `pg.colormap.get` resolves a name by opening a file in its own package directory, so an unknown one raises FileNotFoundError naming a path inside site-packages -- a traceback about the library's install layout, in answer to a user picking a colour scale. Measured on 'jet'.

## FastPlot._gated

### lines 3402-3406

```python
return menu.addAction(label)
```

A CHECKABLE ENTRY WIRES ITSELF. `addAction(text, callable)` connects `triggered`, whose bool is dropped for a slot that does not ask for one -- so a checkable entry connected that way reports the state it had BEFORE the press, i.e. never turns anything on. Those connect `toggled` themselves.

## FastPlot.build_style_menu

### lines 3478-3481

```python
menu.setToolTipsVisible(True)
```

A DISABLED ENTRY'S REASON HAS TO BE READABLE. Qt hides action tooltips unless a menu asks for them, so without this the greyed entries would be exactly the "present but inert" control that instruction 106 forbids. Each group repeats it -- see `_group`.

### lines 3484-3493

```python
menu.addAction("Save figure…", lambda: self.save_styled())
```

ONE DOOR (187 D). There were two -- "Export…", which wrote the plot as it looked, and "Save styled", which opened the preview where ink, background, grid and text size are chosen FOR THE FILE. Export wrote with no preview and no styling pass, which is where the reported "massive text and a tiny misaligned graph" came from: the page is sized in millimetres and nothing scaled the text to it.

A second door that produces a worse file is not a shortcut, so the remaining one shows what it will write. `export` itself stays as the API both paths and every test use.

### lines 3495-3504

```python
self._offer_graph_kinds(menu)
```

AND THE WHOLE THING (instruction 223). Beside "Save figure…" rather than replacing it: a user who wants a png for Slack should not have to take a folder of five files, and one who wants the figure checkable six months from now should not have to assemble it by hand. Two doors here is not the 187 D duplication -- they produce DIFFERENT things, and each says which. AND CHANGE WHAT KIND OF GRAPH IT IS. On the base class, so every plot that holds a spec gets it and no plot that does not shows an empty submenu. `_offer_graph_kinds` returns without adding anything when there is nothing to offer.

### lines 3519-3520

```python
self._checkable(self._group(menu, "Show"), self._levels)
```

WHICH ROWS, not how they look. First, because a filtered plot that looks like a restyled one is read as the whole screen.

### lines 3524-3526

```python
self._checkable(self._group(menu, "p-value"), self._p_values)
```

THE Y-AXIS ITSELF. Above the effect-size cut because it changes what the axis MEANS, while the cut changes where a line is drawn on it.

### lines 3530-3533

```python
group = self._group(menu, "Correction")
```

WHICH MULTIPLE-TESTING CORRECTION IS DRAWN. Beside the p-value axis because the two are one question -- what the height and the colour MEAN -- and above the effect-size cut for the same reason the axis is.

### lines 3537-3542

```python
group.addAction("Write this correction as a table…",
```

NOT LEFT AMBIGUOUS. A plot recorrected to something other than the run's is showing a different analysis from the results.csv beside it; the status line says so, and this is the other half of the answer -- the numbers on screen, written out, so the table and the figure can be made to agree rather than merely be known to differ.

### lines 3547-3550

```python
self._checkable(self._group(menu, "Show the FDR as"),
```

WHICH CHANNEL CARRIES THE FDR. Under the correction rather than beside "Colour by", because it changes what the picture MEANS and not how it looks -- and because two of its entries take the colour channel away from the colouring below.

### lines 3556-3558

```python
cut = self._group(menu, "Effect-size cut")
```

It changes which points count as hits, so it belongs neither with the restyling below nor with the re-fit at the end -- it re-reads a fit that has already happened, like the baseline.

### lines 3568-3570

```python
self._checkable(self._group(menu, "Measured from"),
```

WHAT THE EFFECTS ARE MEASURED FROM. It moves the points and does NOT change the fit: it changes where zero is drawn on a fit that has already happened.

### lines 3575-3576  _(unsure)_

```python
self._checkable(
```

NOT under a heading that could read as a choice of fit. These are drawn on top of one; the heading says which it is.

### lines 3586-3589

```python
self._checkable(self._group(menu, "Draw as"), self._marks)
```

Every option is offered -- including the ones that mislead for the data on screen, because a menu that hides them cannot explain why -- and the plot says so in its status line once the choice is made.

### lines 3597-3599

```python
self._checkable(self._group(colour, "Colour by localisation"),
```

ITS OWN LIST, because this is the one that can be long -- and it holds only what this screen actually has, so a choice that would colour nothing is not offered at all.

### lines 3610-3614

```python
axes.addAction("Lock axis scales (1 y unit = n x units)…",
```

NAMED FOR WHAT IT DOES. It was "Aspect ratio", which everybody read as "make the figure square" -- and it is a statement about the DATA: one y unit drawn as n x units, which is what a Q-Q's 45-degree diagonal needs and is nothing to do with the page. The page is "Shape", under Appearance.

### lines 3617-3620

```python
axes.addAction("Split the y axis…", self._ask_y_split)
```

THE SPLIT, ASKED FOR BY NAME. Under Axes because it is a statement about the RULER: it takes an empty stretch out so the rest of the screen gets its height back. It does NOT make a stepped adjusted P continuous, and the status line it writes says so.

### lines 3627-3629

```python
reason = self.log_reason(axis)
```

CHECKABLE, AND GATED. The tick is the state, and an axis that cannot be logged says why in the entry itself rather than sitting there live and doing nothing.

### lines 3651-3652  _(unsure)_

```python
entry.setData(name)
```

The stored name travels with the entry, so a caller reading the menu back does not have to un-translate the label.

### lines 3655-3658

```python
look.addAction("Font colour…", self._ask_font_colour)
```

EXACTLY TWO COLOUR CONTROLS, split by what a mark IS rather than by which part of the code draws it (instruction 152 B). Font colour is every piece of text, tick LABELS included; Line colour is every line, the axis spines and tick MARKS included.

### lines 3676-3679

```python
size = self._group(menu, "Size")
```

NAMED SEPARATELY BECAUSE THEY ARE DIFFERENT QUANTITIES. "Dimensions" as one entry is the misleading version: on the live plot it is the widget's size, on a saved figure it is the page, and a user who sets one and inspects the other finds nothing changed.

### lines 3687-3689

```python
style, on_change, choices = self._style
```

THE FIGURE'S OWN SETTINGS, under one heading and below the plot's, because they belong to whoever supplied them and a reader has to be able to tell the two apart.

### lines 3693-3696

```python
group.addSeparator()
```

SAVABLE, not only editable -- the half the maintainer restated on 2026-08-16 ("each figure should be editable and savable"). A restyle the user cannot keep is a restyle they redo every time they need the picture.

### lines 3702-3703  _(unsure)_

```python
menu.addSection("Re-runs the analysis")
```

A SECTION, not another line in the list. Everything above restyles; below here the numbers change.

## FastPlot._categorical_opacity

### lines 3857-3859

```python
out.extend(base[len(out):])
```

A frame shorter than the drawn points leaves the rest untouched rather than dropping them: a missing level is not a reason for a point to vanish.

## FastPlot._export_pdf

### lines 4000-4020

```python
writer.setResolution(cls._pdf_resolution(source.width(), width_mm))
```

THE DEVICE SCALE MUST MATCH THE SCENE, and 600 was the bug.

Reported 2026-08-20: "the png saving seems to work but the pdf still has ginormous text and on a tiny missaligned graph." Reproduced by rasterising the PDF beside the PNG: the tick labels came out several times the height of the plot and the axes sat off the page.

WHY IT ONLY HITS THE PDF. `scene.render` maps the scene rect onto the page geometrically -- but pyqtgraph draws its tick labels with `ItemIgnoresTransformations`, which is what keeps them upright and legible while a user zooms. Those items render at the DEVICE's own scale, untouched by the mapping. At 600 dpi a 180 mm page is ~4250 device units wide while the scene is ~900, so everything geometric was scaled 4.7x and the text was not -- it stayed device-sized and so came out 4.7x too big relative to everything around it. The PNG path never had this because ImageExporter renders at the item's own pixel size, 1:1.

Resolution chosen so one scene unit is one device unit. A PDF is vector at any resolution -- this sets the coordinate scale, not the fidelity -- so the text stays text and the lines stay lines.

## FastPlot._ask_point_colour

### lines 4137-4138

```python
brush = pg.mkBrush(colour)
```

One brush for everything: this is the deliberate override of a category colouring, and it is also the fastest path there is.

## FastPlot._ask_axis_limits

### lines 4218-4221

```python
units = {axis: (" in data units, not log10" if self._log[axis] else "")
```

THE DIALOG SAYS WHICH UNITS IT WANTS. They are always the data's own -- a logged axis is drawn in log10 and typed in the quantity and a prompt that does not say so is a number the user has to guess the meaning of on the one axis where the two differ.

## FastPlot._toggle_legend

### line 4438, trailing  _(unsure)_

```python
except Exception:
```

already detached

## FastPlot.add_scatter

### lines 4549-4550  _(unsure)_

```python
drawn = np.nonzero(keep)[0]
```

Positions in the arrays as handed in: what indexes `colours` and `brush_list`, which are drawn up alongside x and y.

### lines 4552-4554

```python
original = drawn if rows is None else np.asarray(rows)[drawn]
```

Indices into the ORIGINAL frame, so a click still identifies the right row after unplottable points have been dropped -- and, when `rows` says the arrays were reordered, after that too.

### lines 4556-4559

```python
rows_drawn = original.tolist()
```

ONE numpy->Python conversion, not a loop of them. `.tolist()` is a C-level bulk convert; `[int(row) for row in original]` is 1,215 interpreter round trips and measurably re-slowed the volcano the first time this was written that way.

### line 4564  _(unsure)_

```python
brushes = [brush_list[i] for i in drawn]
```

Already one reusable brush per point; nothing to build.

### lines 4567-4577

```python
colours = list(colours)
```

ONE BRUSH PER DISTINCT COLOUR, REUSED -- not one per point.

pg.mkBrush() per point builds 1,215 QBrush objects and defeats pyqtgraph's fast path completely. Measured on the real volcano:

a brush constructed per point      39.5 ms 27 brushes, indexed per point       3.5 ms a single brush for everything       1.6 ms

The colours themselves were never the problem; allocating them was. This is the whole of the lag on the last graph.

### lines 4589-4590

```python
sizes = size
```

`data` must go in with the points: calling setData afterwards ADDS points rather than annotating the ones already there.

### lines 4596-4600

```python
symbols = [symbol_list[int(i)] for i in drawn]
```

SUBSET BY `drawn`, exactly as the sizes and the brushes are. A per-point array indexed in frame order against points that have been filtered is the same misalignment `rows` exists to prevent, and it would put the wrong shape on the right dot silently, because something is drawn.

### lines 4611-4612  _(unsure)_

```python
self._row_xy.update(zip(rows_drawn,
```

Where each row ended up, so a selection arriving later can be drawn without re-deriving the transform that put it there.

## FastPlot.set_keys

### lines 4636-4640

```python
self._keys = [None if key is None or key != key else str(key)
```

A MISSING KEY IS None, NOT THE STRING "nan". A frame column carries its blanks as float NaN, and str() turns every one of them into the same four characters -- which would make one bogus identifier that several unrelated rows answer to, i.e. exactly the collision this method's duplicate rule exists to prevent.

## FastPlot.highlight_key

### lines 4677-4678  _(unsure)_

```python
self._selected_keys = [] if key is None else [key]
```

Single-select REPLACES. Leaving the multi list alone here is how the count on screen and the list the consumers read drift apart.

### line 4684, trailing  _(unsure)_

```python
except Exception:
```

already gone

## FastPlot._draw_marker

### lines 4705-4706

```python
self._highlight = pg.ScatterPlotItem(
```

An open ring, not a filled dot: filling it would hide the point it is meant to identify, including its category colour.

## FastPlot.highlight_keys

### line 4745, trailing  _(unsure)_

```python
except Exception:
```

already gone

### lines 4753-4754

```python
for key in wanted[:-1]:
```

The LAST picked one keeps `_highlight`, because that is the one a single-select consumer means by "the selection".

## FastPlot._install_rubber_band.drag

### lines 4854-4855  _(unsure)_

```python
box.updateScaleBox(event.buttonDownPos(), event.pos())
```

The band is pyqtgraph's own scale box, so it looks exactly like the rectangle zoom the user already knows.

## FastPlot._clear_extra_highlights

### line 4879, trailing  _(unsure)_

```python
except Exception:
```

already gone

## FastPlot.add_ranked_bars

### lines 4943-4947

```python
rows = np.arange(len(heights), dtype=float)
```

LARGEST AT THE TOP, so `rank` counts DOWN the screen. pyqtgraph's y grows upward, so the first bar takes the highest number and the axis is inverted rather than the data being reversed -- the same picture, but the values stay in the order the caller can read off the frame they handed over.

## FastPlot._beeswarm_offsets

### line 5170  _(unsure)_

```python
step = ((rank + 1) // 2) * (1 if rank % 2 else -1)
```

Centred: rank 0 in the middle, then alternating out.

## FastPlot.add_radar

### lines 5265-5267

```python
angles = np.array([math.pi / 2.0 - 2.0 * math.pi * i / len(names)
```

CLOCKWISE FROM THE TOP, which is how every radar chart anybody has seen is laid out; counter-clockwise from the right is the maths convention and reads as a different chart.

### lines 5294-5296

```python
area = pg.PlotDataItem(x=closed_x, y=closed_y,
```

THE FILL IS ITS OWN ITEM. `fillLevel` fills to a horizontal line, which on a closed polygon shades a half-moon rather than the inside -- so the interior is a second, brushed curve.

### lines 5310-5312

```python
for side in ("left", "bottom"):
```

A RADAR HAS NO AXES: the numbers live on the rings, and a pair of cartesian scales beside a polar chart is two coordinate systems on one picture.

## FastPlot.add_line

### lines 5338-5341

```python
colour = self._line_colour
```

A LINE ADDED AFTER THE CONTROL WAS USED STILL OBEYS IT. A redraw puts new threshold lines on the plot, and without this they would arrive in the default red beside the ones the user recoloured.

### line 5344

```python
ink = self._font_colour or self._foreground
```

THE CAPTION FOLLOWS THE FONT, NOT THE LINE (instruction 152 B).

## FastPlot.add_group_mark

### lines 5415-5418

```python
level = float(np.median(v) if centre == "median" else np.mean(v))
```

THE SUMMARY LINE IS THE POINT OF "points". Bare points with no summary answer nothing; the rule this menu follows is "individual points WITH a mean line", and the line is the half that carries the comparison between the groups.

### lines 5425-5433

```python
self.add_group_mark(position, values,
```

THE SUMMARY FIRST, THE POINTS ON TOP. Drawn in that order so the observations are not hidden behind the shape that summarises them -- which is the whole reason a composite is a different request from either half.

AND THE POINTS KEEP THEIR ROWS, so a composite stays CLICKABLE where the bare box and bar do not. That is what makes this the honest default: the reader sees the distribution and can still name any observation in it.

### lines 5444-5447

```python
level = float(np.median(v) if centre == "median" else np.mean(v))
```

ONE POINT PER GROUP, and the JOINING is the caller's: this method draws one group at a time and cannot see its neighbours. The marker is what a line chart is made of, and `GroupedPlot` connects them once every group has been drawn.

### lines 5468-5475

```python
from ...figures.spread import SPREAD_NONE, spread_of
```

THE SPREAD, ON THE BAR, AND THE USER SAYS WHICH ONE. A bar already hides every observation; one with no interval at all hides that there was any spread to hide, which is the version of this chart that gets published and then argued about.

THROUGH `spacr.figures.spread`, which is the one vocabulary a second definition of SEM here would let two screens draw whiskers sqrt(n) apart and label them identically.

### lines 5503-5506

```python
beyond = (v > top) | (v < bottom)
```

OUTLIERS STAY POINTS, and stay clickable. They are the rows a reader of a box plot actually wants to name, and they are individual observations, so the rule above lets them keep their rows.

### lines 5517-5519

```python
return self.add_group_mark(position, values, "points",
```

Every value identical: a density has no width and the outline would be a vertical line pretending to be a shape. Fall back to the honest mark rather than drawing that.

## FastPlot._on_points_clicked

### lines 5563-5567

```python
if self._adding_to_selection():
```

THE PLATFORM GESTURE, NOT A BESPOKE MODE (instruction 206). Ctrl or Shift adds and removes; a plain click replaces. Read from the application rather than the event because pyqtgraph's click carries a scene event whose modifiers are not always populated on every platform.

## FastPlot._describe

### lines 5622-5626

```python
key = self.key_for_row(index)
```

THE IDENTIFIER IS ALREADY THE ANSWER. A diagnostic plot holds no frame -- it is handed an array of p-values -- so without this a click on the Q-Q reported an empty status line while quietly selecting the right row somewhere else. The key IS the guide's name; saying it costs one lookup and is what the user clicked for.

## FastPlot.export

### lines 5650-5661

```python
if isinstance(path, bool):
```

`clicked` AND `triggered` BOTH CARRY A BOOL, and this method takes an optional first argument, so Qt hands the checked state straight into `path`. `False is None` is False, the dialog never opened, and `False` travelled all the way to QImage.save -- which is where the user saw it:

TypeError: 'QImage.save' called with wrong argument types:

QImage.save(bool)

The connections below now pass no argument, but this stays: `export` is public, and the next person to wire a button to it should not have to know that Qt's signal has an argument it does not want.

### lines 5676-5685

```python
restore = self._wear_the_print_look(item)
```

INSTRUCTION 150. Around the WHOLE export and not around

`_paint_scene`, which was the first attempt and reached only two of the three formats: PDF and SVG paint the scene themselves, and PNG goes through pyqtgraph's ImageExporter, which does not. A rule that covers two formats out of three is worse than none, because the one it misses is the default.

AND NOT AROUND `snapshot()`, deliberately. That render is the tile in the gallery, which is the SCREEN version -- 139 C: the tile and the file differ on purpose and the difference is the point.

### lines 5688-5690

```python
with self._held_at_the_page_shape():
```

THE SCENE TAKES THE SHAPE FIRST. `export_size` above has already put the shape on the PAGE; without this the page and the scene disagree and Qt letterboxes the difference.

## FastPlot._write_export

### lines 5706-5712

```python
try:
```

THE PAGE THE SAVE ASKS FOR (150 B). Transparent was the old answer and it is only right for the `transparent` mode: dark ink on no background is still unreadable on a dark slide, and a figure going into a manuscript is going onto white. `print` the default -- writes an explicit light page, `screen` keeps what is on screen, and `transparent` keeps the old behaviour for anyone compositing onto their own colour.

### line 5715, trailing  _(unsure)_

```python
except (KeyError, TypeError):
```

older pyqtgraph

## FastPlot._dressed_for_the_file

### lines 5773-5777

```python
before_ground = getattr(self, "_chosen_ground", "")
```

THE PAGE, not just the scene. Restyling the scene colours what is DRAWN; the raster exporter fills the page behind it separately, and it reads `_export_ground`. Recording the choice here is what lets that method prefer it over the global look. Empty means the dialog said "transparent", which is not a colour and must not become one.

### lines 5781-5784

```python
self._canvas_shape = str(canvas_shape)
```

THE STATE, NOT `set_canvas_shape`. That one re-lays the widget out on screen, and this styling is for the file: the scene is given the proportion by `_held_at_the_page_shape` around the render itself.

### lines 5826-5829

```python
self._font_size = before_font
```

BACK TO WHAT IT WAS, INCLUDING None. `set_font_size` takes an int, so "no size of its own" -- the default state of every plot nobody has resized -- can only be restored by putting the attribute back and re-applying.

## FastPlot.styled_snapshot

### lines 5867-5870

```python
return self.snapshot(width, ground=self._export_ground())
```

THE PAGE THE FILE WOULD GET. `_export_ground` is what the raster exporter fills behind the scene when the file is written, so a preview that left it out showed a transparent page for a file that will not have one.

## FastPlot.export_bundle

### line 5897, trailing  _(unsure)_

```python
if isinstance(folder, bool):
```

a signal's checked state

## FastPlot._offer_graph_kinds

### lines 5950-5952

```python
action = show_as.addAction(str(GRAPH_NAMES.get(kind, kind)))
```

THE NAME ON THE ENTRY, THE DESCRIPTION IN THE TOOLTIP. A menu reading "one value per group" instead of "Bar" cannot be scanned, which is what a menu is for.

### lines 5958-5960

```python
action.setEnabled(False)
```

GREYED WITH THE REASON, never absent: a list that silently shortens leaves the user wondering whether they misremembered (instruction 106).

### lines 5966-5973

```python
if current:
```

AND A WAY TO MAKE IT THE STARTING POINT. Asked for 2026-08-28: the right-click menu was right, but it only ever changed the graph in front of you -- the next one of the same shape was drawn the old way again. This is the same choice, remembered.

HERE, where the user is already choosing a graph type, rather than only in Preferences: the moment somebody decides they prefer a violin is the moment they are looking at one.

## FastPlot._shape_the_image

### line 6115, trailing  _(unsure)_

```python
except Exception:
```

a different exporter API

## FastPlot.snapshot

### lines 6160-6161

```python
return None
```

A picture is never worth taking the screen down for. The caller pins nothing, which is the same thing that happens before a run.

## FastPlot._render_snapshot

### lines 6177-6181

```python
try:
```

THE SHAPE, IN THE PIXELS TOO. The scene is already held at the proportion, but the exporter's linked height is recomputed from the source rect it saw when the width was written -- so a preview could come back one pixel out of square, which is the difference between "the shape worked" and "nearly".

### line 6186, trailing  _(unsure)_

```python
except Exception:
```

other exporter

### lines 6189-6197

```python
exporter.parameters()["background"] = (
```

TRANSPARENT BY DEFAULT, like the tile behind it. The exporter otherwise uses pyqtgraph's configured background, and a tile painted onto an opaque slab is the "the graphs still have a black background" report all over again.

A CALLER MAY ASK FOR THE PAGE, and the save dialog does:

its preview is meant to be the file, and a file written onto white while its preview showed transparent is a preview of something else.

### line 6200, trailing  _(unsure)_

```python
except (KeyError, TypeError):
```

old pyqtgraph

## FastPlot.restyle

### line 6226, trailing  _(unsure)_

```python
except Exception:
```

absent axis

### lines 6233-6236

```python
if self._line_colour is not None:
```

A THEME SWITCH MUST NOT UNDO A CHOICE THE USER MADE. The loop above has just painted the theme's ink over every axis; if the user set a font or a line colour off the menu, that is what they asked this plot to look like and it goes back on top.

### lines 6241-6244

```python
self.apply_text_style()
```

The captions are text and were just repainted with the axes' ink by nothing at all -- they are LabelItems, which the loop above does not reach. apply_text_style is what carries the theme to them, so it runs on a plain theme switch too.

## VolcanoPlot.__init__

### lines 6342-6347

```python
self.plot.getAxis("bottom").enableAutoSIPrefix(False)
```

NO SI PREFIX ON AN EFFECT SIZE. pyqtgraph factors a common power of ten out of the tick labels and states it once in the axis title, so a partial correlation running -0.06 to 0.50 was drawn as an axis reading -100 to 400 titled "coefficient (x0.001)". That is correct and unreadable: the number a reader wants to quote is 0.50, and they should not have to multiply it back themselves.

## VolcanoPlot.set_p_axis

### lines 6418-6427

```python
if kind == "adjusted" and self.correction() in ("none", "", None):
```

THE RUN-TIME HALF OF THE GREYING RULE. The menu entry for the adjusted axis is DISABLED when no correction is in force -- because the adjusted p then IS the raw p, and the axis is the one above with a label that reads "adjusted p (None (raw P values))". But the menu is not the only way in: the host drives this axis directly and redraws on every level, baseline and compartment change, so a user who chose the adjusted axis on a CORRECTED run and then moved to an uncorrected one got exactly the label this instruction removed. Same shape as `ml._require_backend`: a rule that lives only in the widget that greys it is a rule with one entry point unguarded.

## VolcanoPlot._q_strength

### lines 6502-6503  _(unsure)_

```python
strength[usable] = 0.5
```

Every test on the plot holds the same q. Half way up, so the ramp is one colour and is honest about being one colour.

### lines 6507-6508  _(unsure)_

```python
strength[np.isfinite(q) & (q <= 0)] = 1.0
```

A q of exactly zero underflowed; it is the strongest evidence there is, not a missing value.

## VolcanoPlot._q_ramp

### lines 6548-6549  _(unsure)_

```python
for fraction in np.linspace(1.0, 0.0, self.Q_LEGEND_STOPS):
```

The stops are named by the q they stand for, strongest first, so the key reads in the direction a volcano is read.

### line 6553  _(unsure)_

```python
logged = -np.log10(np.clip(finite, 1e-300, 1.0))
```

The q at this stop, inverted back out of the log scale.

## VolcanoPlot._q_opacity

### line 6589, trailing  _(unsure)_

```python
except Exception:
```

an odd brush

## VolcanoPlot.set_correction

### lines 6624-6628

```python
self._p_axis = "raw"
```

Choosing "no correction" while the height IS the adjusted p would leave the axis labelled "-log10(adjusted p, none)" over numbers that are the raw p. The menu greys the adjusted entry for the same reason; this is the one path that can reach the state anyway, so it is unwound rather than left.

## VolcanoPlot.set_results

### lines 6732-6735

```python
self._results_call = (frame, dict(
```

THE CALL IS REMEMBERED, NOT THE PICTURE. Switching the axis or the correction redraws from the table the host handed over, so neither costs a round trip through the host and neither can drift from what is on screen.

### lines 6752-6753

```python
try:
```

CANONICAL, so "bh" in an older table and "fdr_bh" in a new one are the same run method and the menu ticks the same entry for both.

### lines 6758-6767

```python
raw_column = _first_column(frame, ("p_value", "p", "pvalue"))
```

A HOST STILL DRIVING THE AXIS THROUGH `p_column` IS HONOURED UNTIL THE USER SAYS OTHERWISE. `RegressionResultsPanel` switches raw/adjusted by handing over a different column, and reading that as "the host asked for the adjusted axis" keeps its control working while this plot owns the choice.

A CHOICE MADE ON THE PLOT WINS FROM THEN ON, and that is not a nicety: the host redraws on every level change, baseline change and compartment change, so without this any one of them would silently put the axis back and the user would watch their choice undo itself.

### lines 6776-6789

```python
untested = 0
```

NUISANCE TERMS ARE NOT HYPOTHESES, AND THEY OWN THE AXIS.

The intercept and the plate row/column effects are covariates: they are fitted so the guide effects come out clean, not so anyone can ask whether they differ from zero. spacr.ml already draws that line it leaves them out of the multiple-testing family, which is why they leave a fit with q_value = NaN -- and plotting them draws a different experiment from the one the q-values describe.

It is not a rounding error. On plate1_dv the intercept sits at -log10(p) = 45.5 against 12.5 for the strongest real hit and 2.3 at the 99th percentile, so ONE untestable row makes the y-axis 3.6x taller than the data and flattens the whole screen into the bottom of it. A fit carrying row and column terms has ~25 of them.

### lines 6810-6817

```python
method = self.correction()
```

the correction

RECOMPUTED, AND WITHIN THE RIGHT FAMILY. The correction applies within a LEVEL -- a run at level='both' fits twice and each fit is its own family (instruction 128 R) -- so pooling whatever happens to be on screen would change n and with it every q value, quietly and in the direction that makes the run look weaker than it is. `hits.family_labels` is the single statement of that split.

### lines 6830-6834

```python
self._lfdr_values = None
```

THE LOCAL FDR IS NOT COMPUTED UNLESS IT IS WANTED. Measured on the real screen's 1,215 coefficients: the mixture fit is 25 ms of a 40 ms redraw -- more than drawing the plot -- and the default axis is the raw P, which does not use it. It is computed when the axis asks for it and when a click asks for it, once, and cached.

### lines 6842-6854

```python
critical = (float(np.max(raw[mask][fam_called]))
```

THE CRITICAL RAW P, WHICH IS EXACT AND IS NOT ALPHA.

Every correction here is monotone in the raw P within a family, so the set it calls is a lower set: there is a rank k with every p <= p_(k) called and everything above it not. For BH that is the textbook identity q_(i) <= alpha iff p_(i) <= alpha*i/n at the largest such i. One horizontal line at -log10(p_(k)) therefore divides this plot EXACTLY as the FDR does, on a continuous axis, with no steps anywhere.

Drawing it at -log10(alpha) instead is the mistake this replaces: that is the UNCORRECTED threshold and it calls far too much of the screen.

### lines 6860-6865

```python
finite_raw = raw[np.isfinite(raw)]
```

IS THE RAW P ITSELF QUANTISED? The permutation path is discrete TWICE OVER and that is why it looks worst: a permutation p is (1 + #{null <= observed}) / (n + 1), so 1,000 permutations admit only 1,001 possible values and many guides already share one before BH ever runs. Saying so is the difference between a user raising `guide_permutations` and a user concluding the plot is broken.

### lines 6876-6879

```python
agrees = None
```

DOES THE PLOT AGREE WITH THE TABLE BESIDE IT? results.csv carries the run's own q values; recomputing the run's own method on the run's own family has to reproduce them, and if it does not, the user is looking at two analyses and is entitled to know.

### lines 6902-6903

```python
smallest = np.nanmin(values[values > 0]) if np.any(values > 0) \
```

A p of exactly zero is a real result underflowing, not a mistake; clamping keeps it on the plot instead of sending it to infinity.

### lines 6913-6917

```python
ramp_on = self._q_colour == "ramp" and bool(np.isfinite(q).any())
```

F.5 -- THE RAMP TAKES THE COLOUR CHANNEL, and what it takes it from is named. A dot cannot be coloured for its condition and for its q at once, so the colouring that would otherwise be in force is not drawn and the caption says which one it was. Silently dropping it is how a reader ends up reading a q ramp as a condition.

### lines 6929-6937

```python
from ...localisation import of as compartment_of
```

EVERY COMPARTMENT AT ONCE, asked for on 2026-08-20: "Colour by lets me color by a single location, all should be an option."

ONE AT A TIME WAS A DECISION, NOT AN OVERSIGHT -- "everything is grey except what the sentence is about", and the 27-colour legend measured 40 ms of a 49 ms redraw. So this is offered BESIDE that, never instead of it, and it is built on the CATEGORICAL path below rather than a Python loop over brushes, which is where that 40 ms actually went.

### lines 6945-6947

```python
from ...localisation import mask as compartment_mask
```

ONE COMPARTMENT AGAINST GREY. Two brushes and a two-entry legend: the 27-colour version is what the house style forbids and, measured, its legend cost 40 ms of a 49 ms redraw.

### lines 6958-6960

```python
import pandas as _pd
```

Categorical codes are computed in C; the alternative is a Python loop over 1,215 pandas values plus a QColor.rgba() per point, which cost 45 ms of the 48 ms this used to take.

### lines 6966-6976

```python
elif called.any() or np.isfinite(q).any():
```

THE COUNT BESIDE EACH LABEL. Asked for 2026-08-17: "beside the label on the graph should be the count of each label".

It is not decoration on a screen. `nc` and `pc` are three and twenty-four points among twelve hundred, and a legend that names them without saying so invites reading a two-point cluster as a group -- the same reason the compartment legend and the gene/guide menu already carry theirs.

Counted with np.bincount over the CODES, not by grouping the frame -- see `_categorical_brushes`.

### lines 6978-6982

```python
here, elsewhere = pg.mkBrush(HIGHLIGHT), pg.mkBrush(MUTED)
```

THE FIELD'S OWN VOLCANO: continuous height, binary colour, and the colour carrying the claim. Only when nothing else has claimed the colour channel -- a dot cannot be coloured for its condition and for its q at once, and the legend says which is in force.

### lines 6989-6996

```python
self._frame = frame
```

NO PER-POINT WORK BEFORE DRAWING.

This used to build a label string for all 1,215 rows up front, three `frame[col].iloc[i]` lookups each. Pandas scalar indexing in a Python loop is ~3,600 lookups to draw a scatter plot, and it cost more than the drawing did. The frame is kept instead and a label is formatted for the ONE point that gets clicked -- which is the only one anybody ever reads.

### lines 7003-7006

```python
key = key_column or ("feature" if "feature" in frame.columns
```

`feature` is the design-matrix term name and is one-to-one with the row -- checked on the real screen: 1,213 rows, 1,213 distinct. `gene` and `grna` are NOT keys, because a gene has several guides and several rows, so joining on either highlights an arbitrary one.

### lines 7011-7015

```python
size_list = None
```

F.6 -- THE ENCODING THAT COMPOSES. Size and opacity are channels the colour is not using, so either can be on at the same time as any of the colourings above: a volcano coloured by condition with its marks sized by q says both things at once, which is exactly what the ramp cannot do.

### lines 7021-7023

```python
symbol_list = None
```

THE SECOND AND THIRD CHANNELS. Only when the FIRST is in force: a shape that means one thing beside a colour that means the q value is two claims on one dot with nothing saying which.

### lines 7035-7037

```python
self._opacity_legend = (opacity_column,
```

THE COLUMN AND ITS LEVELS, because a fade nobody can name is not an encoding. The legend draws them at the alphas the points got, from the same ramp.

### lines 7053-7065

```python
self._legend_colours = legend
```

THE LEGEND IS OPT-IN, AND IT IS THE REASON WHY.

Twenty-seven entries cost 40 ms of a 49 ms redraw -- each one builds a ScatterPlotItem and a LabelItem. It is the identical cost that made matplotlib's version 63 ms, so bringing it across unchanged would have carried the lag over to the new library and wasted the switch.

scatter alone, 1,215 points        3.4 ms the same plus a 27-entry legend   43.7 ms

So the plot draws without one and offers a checkbox. Colour still identifies the compartments; the legend only names them, and naming them is worth 40 ms when asked for and not before.

### lines 7067-7069

```python
entries = len(self._legend_entries())
```

THE OTHER TWO CHANNELS COUNT TOO. A volcano whose shape carries a column and whose legend can only be switched on when a COLOUR column is chosen is a picture with an unreachable key.

### lines 7079-7081

```python
if self._selected_key is not None:
```

A SELECTION SURVIVES A REDRAW. plot.clear() took the marker with it, so it goes back on -- otherwise changing the colouring, or any other setting, silently deselects whatever the user was looking at.

### lines 7093-7094

```python
note += (f" {untested} nuisance "
```

Reported, not silently removed: the difference between a filter and a lie.

## VolcanoPlot._add_significance_lines

### lines 7105-7106  _(unsure)_

```python
def _add_significance_lines(self, level: str) -> int:
```

the line and the sentence

### line 7117  _(unsure)_

```python
name = "q" if self._p_axis == "adjusted" else "lfdr"
```

The corrected axis is the one place alpha IS the threshold.

### lines 7127-7133

```python
distinct = sorted(set(criticals.values()))
```

DEDUPED ON THE EXACT VALUE, not on a rounded one. `round(p, 15)` was here and it sent every P value below 1e-15 to zero -- so a screen whose critical P is 1.7e-20 drew its line at -log10(0), and pyqtgraph died on the infinity ("cannot convert float NaN to integer") rather than on anything a reader could act on. Rounding to DECIMAL places is never right for a P value; the two families' criticals are computed from the same array and compare exactly.

### lines 7142-7145

```python
self.add_line(y=-np.log10(max(value, self._y_floor)), label=text)
```

THE LINE SITS ON THE SAME FLOOR THE POINTS DO. A P value that underflowed to zero is a real result, and the scatter already clamps it rather than sending it to infinity; a threshold line that did not use the same floor would leave the plot.

## VolcanoPlot._build_caption

### lines 7173-7176

```python
if self._q_colour == "ramp" and displaced:
```

ONE SENTENCE PER FIGURE saying which encoding has which channel. A ramp that silently replaced a condition colouring, or a mark sized by q with nothing saying so, is a figure that shows one thing and is read as another.

### lines 7198-7206

```python
parts.append(
```

WHY THE LINE ALWAYS TOUCHES A POINT, said once, because it reads as a coincidence and is not one. The threshold is `max(p over the called tests)` -- an OBSERVED value, not a formula evaluated in the abstract -- so it is necessarily some test's own p, and the line therefore lands exactly on the last test it called. A reader who notices two genes sitting on the line is seeing the borderline, and asked whether those are in or out; `p<=` in the label answers it and this says which points those are.

## VolcanoPlot._offer_p_axes

### lines 7250-7263

```python
uncorrected = ("" if str(method) != "none" else
```

NO CORRECTION MEANS THERE IS NO ADJUSTED AXIS. With

`multiple_testing_method='none'` the q value written for every row EQUALS its raw p, so this entry offered a second copy of the axis above it under a name -- "adjusted p — None (raw P values) (stepped)" -- that promises a number the run never computed. A user who picks it sees an identical plot and concludes the correction made no difference, which is the reading this module exists to prevent.

GREYED, NOT REMOVED (INVARIANTS 6), and it says why: the answer to "where is the adjusted axis" is "this run applied no correction", which a missing entry does not give. The condition is the correction IN FORCE and not the run's, because a plot recorrected from the menu has real q values whatever the run did.

## VolcanoPlot._offer_encodings

### lines 7311-7313

```python
flat = ("every q on this plot is the same number, so a ramp "
```

The staircase at its limit: one q for the whole screen. A ramp over it is one colour, and offering it live would be the present-but-inert control instruction 106 forbids.

## EffectRankPlot.__init__

### lines 7434-7436

```python
self.plot.getViewBox().invertY(True)
```

RANK 1 AT THE TOP. A ranked list is read downwards, and pyqtgraph's y-axis grows upwards, so without this the strongest effect sits at the bottom of the panel and the reader starts at the weakest.

## EffectRankPlot.set_results

### lines 7511-7514

```python
frame = frame.reset_index(drop=True)
```

POSITIONAL FROM HERE ON. Every row index this method hands to `add_scatter`, and every index `_detail` is later asked about, is a position in THIS frame; a caller's filtered frame arrives with holes in its index and `.iloc` would then disagree with `.loc`.

### lines 7541-7543

```python
order = np.argsort(-np.abs(effects), kind="stable")
```

numpy puts NaN last under an ascending sort whatever its sign, so a coefficient that did not converge ranks below every one that did rather than at the top of the list.

### lines 7549-7551

```python
code = np.where(called, np.where(x > 0, 1, 2), 0)
```

0 grey, 1 up, 2 down -- as a code array rather than a list of colours, so the intervals can be grouped by ink in three passes instead of one PlotCurveItem per coefficient.

### lines 7555-7558

```python
usable = np.isfinite(x) & np.isfinite(widths) & (widths > 0)
```

THE INTERVALS FIRST, so the dots sit on top of them. One curve per ink with `connect="pairs"` -- 1,213 disconnected segments in three items rather than 1,213 items, which is the difference between a plot that opens and one that hangs.

### lines 7575-7579

```python
names = self._label_series(frame, label_column)
```

THE NAMES ARE Y-TICKS HERE AND ANNOTATIONS IN THE SAVED PANEL, and the difference is deliberate. A tick label is drawn outside the axes, so on a sheet a long gene id reaches into the cell to its left which is why the static panel puts them inside. A tab has no neighbouring cell, and a tick is the axis a reader can then zoom.

### lines 7590-7593

```python
self.set_status(self._sentence(plotted, len(order), shown, error,
```

COUNTED OVER WHAT IS ON THE PICTURE. `called` is computed over every row, and a coefficient that did not come out can still carry a q-value -- so counting one would put a number in the status line that no reader can reach by counting the coloured dots.

## EffectRankPlot._label_series

### line 7616, trailing  _(unsure)_

```python
except Exception:
```

figures unavailable

## BinnedPlot.__init__

### lines 7717-7720

```python
self.plot.scene().sigMouseClicked.connect(self._on_scene_clicked)
```

A BAR IS NOT A POINT, so there is no sigClicked to connect to. The scene reports where the user pressed and the bin is worked out from the x coordinate, which is also the only definition of "which bar" that stays right when the axis is zoomed.

## BinnedPlot._fill_bins

### line 7763

```python
self._row_bin = np.full(len(held_all), -1, dtype="int64")
```

Row -> its bar, as a dense array rather than a dict of thousands.

## BinnedPlot._on_scene_clicked

### line 7840, trailing  _(unsure)_

```python
except Exception:
```

no viewbox to map into

### lines 7842-7844

```python
index = self.bin_at(self._to_data(point.x(), "x"))
```

THE BAR IS FOUND IN DATA UNITS. `mapSceneToView` answers in DRAWN units, which are log10 of the data while the x axis is logged, and a bin looked up with those lands in the wrong bar or in none.

## BinnedPlot.highlight_bin

### line 7858, trailing  _(unsure)_

```python
except Exception:
```

already gone

### lines 7860-7863

```python
self._highlight = pg.BarGraphItem(
```

An OUTLINE, not a refill: the same reason the scatter marker is an open ring. A solid bar in the highlight colour would hide how tall it is against its neighbours, which is the only thing the panel is for.

## PValueHistogram.set_p_values

### lines 7926-7930

```python
held = self._fill_bins(values, bins, span=(0.0, 1.0))
```

PINNED TO [0, 1], because that is what a p-value's axis MEANS. A histogram of p over the observed range would put its left edge at the smallest p in the screen, and the spike at zero -- the whole signal this panel exists to show -- would then be the first bar of every screen, calibrated or not.

## QQPlot.set_p_values

### line 8091  _(unsure)_

```python
rows = np.nonzero(~np.isnan(p) & (p > 0))[0]
```

Frame rows, kept alongside their p-values through the sort.

### line 8104

```python
chi = np.median(observed) / np.median(expected) if np.median(expected) else float("nan")
```

Genomic inflation: the ratio at the median. 1.0 is calibrated.

### lines 8112-8114

```python
if self._selected_key is not None:
```

A SELECTION SURVIVES A REDRAW, here for the same reason it does on the volcano: the user picked a guide, and reloading or recolouring is not them un-picking it.

## ResidualPlot.set_residuals

### line 8174

```python
slope, intercept = np.polyfit(f[good], r[good], 1)
```

A crude trend line: if this is not flat, the mean is wrong.

### lines 8182-8183

```python
curve = self.add_smoother(f[good], r[good],
```

THE STRAIGHT LINE CAN BE FLAT WHILE THE RESIDUALS BEND, which is the case a slope cannot report and a smoother can.

## InfluencePlot.set_influence

### lines 8314-8316

```python
cut = 4.0 / n
```

4/n, the conventional screening rule and the one the saved report draws. The stricter D > 1 almost never fires on a few hundred wells, which makes it a rule that separates nothing.

### line 8330  _(unsure)_

```python
self.add_line(x=2.0 * int(n_params) / n, colour="#DD8452",
```

2p/n: the standard "this row has an unusual design" rule.

## GroupedPlot.set_mark

### lines 8413-8414  _(unsure)_

```python
self._offer_marks()
```

The menu is rebuilt from scratch on every right-click, so the tick only moves if the stored list moves with it.

## ControlSeparation.set_groups

### lines 8541-8542  _(unsure)_

```python
flat_keys: list = []
```

One flat row space over every group, so a key means the same thing whichever group it came from.

### lines 8554-8555

```python
if len(given) != len(v):
```

A short or long key list is a caller bug that would silently shift every row after it; pad rather than mis-join.

### lines 8561-8564

```python
self._spans.append((base[name], base[name] + len(v), name))
```

WHICH GROUP A ROW IS IN, AS A SPAN RATHER THAN AS 1,186 DICT

ENTRIES. Only the clicked point is ever asked, and the module's whole performance argument is that nothing is computed per point before a click. Three tuples answer it in a scan of three.

### lines 8577-8582

```python
self.add_group_mark(position, v[finite], self._mark, size=7,
```

THE MEDIAN IS THE SENTENCE OF THIS PANEL -- whether the classes separate is read off those lines -- so `MARK_CENTRE` keeps the summary line on the median whatever mark the user picks, and the line is drawn in the plot's own ink. It was hardcoded BLACK, which on every dark spaCR theme but one is a line nobody can see, and it is the one mark here that must be visible.

### lines 8597-8602

```python
axis.setTicks([[(i, f"{name}\n(n={size})")
```

THE COUNT BESIDE THE LABEL, not only in the note below the plot. Asked for 2026-08-17. "pc" and "nc" are three and twenty-four points, and a label that does not say so lets a three-point group be read as a group -- which is the same reason the mark advice exists. Taken from the SAME `sizes` the note and the advice use, so the axis cannot disagree with the sentence under it.

## GuideAgreementPlot.set_support

### lines 8738-8741

```python
self._frame = frame
```

THE TABLE THE RESTYLE MENU READS. It is the RESET frame, not the argument: the row indices carried into every scatter below are positions in this one, so a column mapped onto a colour or a shape has to be indexed the same way or it would shade the wrong genes.

### lines 8754-8760

```python
rows = np.arange(len(frame))
```

JITTERED, for the same reason the static panel is: guides per gene is a small integer and agreement is a handful of fractions, so several hundred genes stack into a dozen dots and the panel looks like it holds no data. Seeded, so the picture is the same every time, and recorded per row -- the ring a selection draws reads its coordinates back out of `_row_xy`, so it lands on the dot the user actually sees rather than on the un-jittered lattice point.

### lines 8763-8770

```python
spread = 0.22 if self._mark == "jitter" else 0.0
```

THE HOUSE-RULE COLOURING SURVIVES ONLY WHILE THE MARKS ARE

POINTS. Grey for a gene its own guides corroborate, colour for one that rests on a single guide, which is the sentence this panel exists to make -- and a sentence about individual genes that a box plot cannot carry, because a box holds both kinds at once. So the point marks keep this path and the summarising marks take the grouped one below, rather than one path drawing a compromise neither picture wanted.

### lines 8784-8786

```python
x = counts
```

ONE MARK PER GUIDE COUNT. The x-axis is already a small integer, so the groups are the counts themselves and no tick remapping is needed -- "3" on the axis still means three guides.

### lines 8809-8821

```python
beyond = 0
```

ONE GENE MUST NOT OWN THE AXIS. The library gives a gene two to four guides; the non-targeting control block parses as a single "gene" carrying all 24 of them, and on autorange that ONE point stretches the x-axis six times wider than the data and squashes all 388 real genes into the left fifth of the panel. It is the identical failure the intercept caused on the volcano, measured the same way.

The volcano's answer was to DROP the offender, because a nuisance term is not a hypothesis. This one is different: an over-represented gene is still a gene, and dropping it would lose a real point. So it is drawn and merely left outside the OPENING view -- "Reset view" reaches it, and the status says it is out there rather than letting it disappear silently.

## ResultsTable.__init__

### lines 8924-8925  _(unsure)_

```python
install_sorting(self.table)
```

The application's one sorting contract: descending on the first click, ascending on the second, the frame's own order on the third.

### lines 8928-8930

```python
self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
```

EXTENDED, NOT SINGLE (instruction 206). A band over the volcano selects several guides, and a table that can only hold one of them would show a different guide from the plot that fed it.

### lines 8943-8953

```python
self._significance: Optional[str] = None
```

EVERY PIECE OF STATE _apply_filter READS IS BORN HERE.

`_significance` was created only in set_frame, and the filter controls are connected in this constructor -- so any path that touched a control before the first frame arrived crashed the application on startup with AttributeError. configure() is one such path: it can uncheck "significant only", which emits toggled.

A widget must be fully usable the moment it exists. Half-built state that only becomes valid after some other method has been called is how a constructor turns into a trap.

## ResultsTable.set_frame

### lines 8966-8967

```python
self._key_restriction = None
```

A new table is a new experiment: a set of keys chosen off the last one names nothing here, and leaving it on would hide every row.

### lines 8979-8980  _(unsure)_

```python
self.table.setSortingEnabled(False)
```

Sorting must be off while filling: with it on, Qt re-sorts after every insert and the rows end up interleaved.

### line 8989  _(unsure)_

```python
item.setData(Qt.UserRole, row)
```

The frame row, so a click still maps home after sorting.

## ResultsTable._apply_filter

### lines 9034-9036

```python
significance = self._significance if self._frame is not None else None
```

The significance cut needs the frame to find its column in. Without one there is nothing to cut on, and asking for the column would be the same crash one line further down.

### lines 9067-9068

```python
note += (f" — narrowed to {len(self._key_restriction)} chosen on "
```

Said out loud, because a table that has silently narrowed itself is indistinguishable from a table that has lost its rows.

## ResultsTable.select_keys

### lines 9180-9182

```python
self.table.setRowHidden(row, False)
```

A hidden row is unhidden to select it, for the same reason

`select_key` does: silently dropping the point the user just dragged over reads as a broken gesture.
