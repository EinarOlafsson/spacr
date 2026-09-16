# Notes from `spacr/qt/widgets/figure_settings.py`

Prose lifted out of `spacr/qt/widgets/figure_settings.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_as_hex](#_as_hex) (2 entries)
- [_colour_button._choose](#_colour_button_choose) (1 entry)
- [FigureSettingsDialog](#figuresettingsdialog) (1 entry)
- [FigureSettingsDialog.__init__](#figuresettingsdialog__init__) (8 entries)
- [FigureSettingsDialog._build_umap_tab](#figuresettingsdialog_build_umap_tab) (1 entry)
- [FigureSettingsDialog.reject](#figuresettingsdialogreject) (4 entries)
- [FigureSettingsDialog._add_series_rules.apply_palette](#figuresettingsdialog_add_series_rulesapply_palette) (2 entries)
- [FigureSettingsDialog._redraw_now](#figuresettingsdialog_redraw_now) (2 entries)
- [FigureSettingsDialog._statistics_tab](#figuresettingsdialog_statistics_tab) (1 entry)
- [FigureSettingsDialog._figure_tab](#figuresettingsdialog_figure_tab) (4 entries)
- [FigureSettingsDialog._figure_tab.set_all_text](#figuresettingsdialog_figure_tabset_all_text) (1 entry)
- [FigureSettingsDialog._figure_tab.set_line_ink](#figuresettingsdialog_figure_tabset_line_ink) (1 entry)
- [FigureSettingsDialog._axes_tab](#figuresettingsdialog_axes_tab) (9 entries)
- [FigureSettingsDialog._axes_tab.apply_limits](#figuresettingsdialog_axes_tabapply_limits) (1 entry)
- [FigureSettingsDialog._axes_tab.apply_grid](#figuresettingsdialog_axes_tabapply_grid) (1 entry)
- [FigureSettingsDialog._axes_tab.apply_legend](#figuresettingsdialog_axes_tabapply_legend) (1 entry)
- [FigureSettingsDialog._axes_tab.set_colour](#figuresettingsdialog_axes_tabset_colour) (1 entry)
- [apply_line_colour](#apply_line_colour) (3 entries)
- [apply_font_colour](#apply_font_colour) (1 entry)
- [figure_follows_the_theme](#figure_follows_the_theme) (1 entry)
- [graph_style_as_dict](#graph_style_as_dict) (1 entry)
- [_pairs_from_axes](#_pairs_from_axes) (3 entries)
- [derive_replot_recipe](#derive_replot_recipe) (2 entries)
- [_which_types_fit](#_which_types_fit) (1 entry)
- [_add_group_colours._recolour](#_add_group_colours_recolour) (1 entry)
- [_add_group_colours](#_add_group_colours) (1 entry)
- [save_figure_bundle](#save_figure_bundle) (1 entry)
- [save_figure_bundle._render](#save_figure_bundle_render) (1 entry)
- [build_figure_context_menu](#build_figure_context_menu) (10 entries)
- [save_figure_as](#save_figure_as) (5 entries)
- [export_sidecars](#export_sidecars) (1 entry)
- [Module level](#module-level) (2 entries)
- [FigureStylePreferences](#figurestylepreferences) (1 entry)
- [FigureStylePreferences.__init__](#figurestylepreferences__init__) (2 entries)
- [FigureStylePreferences._save_to_file](#figurestylepreferences_save_to_file) (1 entry)
- [FigureStylePreferences._control](#figurestylepreferences_control) (1 entry)
- [FigureStylePreferences._control._set_colour](#figurestylepreferences_control_set_colour) (1 entry)

## _as_hex

### line 74  _(unsure)_

```python
if isinstance(value, np.ndarray):
```

A collection stores one row per element; they share a colour here.

### line 81, trailing  _(unsure)_

```python
except Exception:
```

genuinely unreadable colour

## _colour_button._choose

### lines 100-101

```python
"""Ask for a colour and keep it if the dialog returned one."""
```

Qt's own dialog, never the platform one -- see :mod:`spacr.qt.widgets.colour_picker`.

## FigureSettingsDialog

### lines 135-140

```python
class FigureSettingsDialog(QDialog):
```

THE ARGUMENTS ARE DOCUMENTED ON `__init__`, ONCE. They were listed here too, as a NumPy ``Parameters`` section, and AutoAPI runs with ``class_content='both'``: the class docstring and ``__init__``'s are concatenated before Napoleon sees them, the section became a field list, and ``__init__``'s opening prose then ended it mid-way -- "Field list ends without a blank line", which `sphinx-build -W` makes fatal.

## FigureSettingsDialog.__init__

### lines 183-188

```python
self._snapshot = None
```

A SNAPSHOT TO GO BACK TO. The dialog this replaced restored the figure on Cancel, and said why: "live apply with no way out is a trap: the user drags a spin box to see what it does and there is no longer an 'as it was'". This dialog changes far more than that one did, so the trap is correspondingly worse. The copy is the same one the preview renderer takes, ~14 ms, and buys a working Cancel.

### line 193, trailing  _(unsure)_

```python
except Exception:
```

artists that will not pickle

### lines 195-198

```python
try:
```

The per-figure text size is an ATTRIBUTE, and `reject` restores the figure by copying axes out of the snapshot rather than by swapping the object -- so the attribute would survive a Cancel that undid everything it applies to. Kept here and put back explicitly.

### line 202, trailing  _(unsure)_

```python
except Exception:
```

figure_queue unavailable

### lines 205-207

```python
self._rendering = False
```

Coalesce redraws. Every control calls _changed(); this restarts a single-shot timer, so a burst of twenty value changes costs one render instead of twenty.

### lines 221-223

```python
if getattr(figure, "_spacr_groups", None):
```

STATISTICS, only for a figure that actually compares groups. A tab offering a t-test on a Q-Q plot would be an invitation to report a number that means nothing -- see `_statistics_tab`.

### lines 231-234

```python
self._umap_settings = None
```

The Image UMAP half (instruction 75): every UMAP setting, live against this figure. Only for a figure carrying the embedding it was drawn from -- without it "live" would mean re-running the reduction and every point would move.

### lines 241-244

```python
self._block_wheel_on_inputs()
```

Scrolling the panel must scroll it, not edit whatever is under the pointer. Qt gives spin boxes and combos the wheel by default, so a scroll gesture over this dialog changed a dozen settings and triggered a render for each -- which is what made it unusable.

## FigureSettingsDialog._build_umap_tab

### line 271, trailing  _(unsure)_

```python
except Exception:
```

UMAP support absent

## FigureSettingsDialog.reject

### lines 327-329

```python
self._figure.clear()
```

Copy the restored state back INTO the figure the queue holds, rather than swapping the object -- everything else refers to the original by identity.

### lines 332-337

```python
axis.remove()
```

DETACHED FIRST. matplotlib refuses to put one artist in two figures, and a restored axes still belongs to the figure the snapshot was unpickled into -- so re-homing it without the detach raised on the first axes and Cancel left a CLEARED figure behind, with neither the size nor the ground it opened with.

### line 344, trailing  _(unsure)_

```python
except Exception:
```

restore is best-effort

### line 350, trailing  _(unsure)_

```python
except Exception:
```

figure_queue unavailable

## FigureSettingsDialog._add_series_rules.apply_palette

### lines 392-394

```python
colour = (colormap(index % colormap.N) if colormap.N <= 32
```

A qualitative map is indexed by position; a continuous one is sampled across its range. Using the wrong one gives every series nearly the same colour.

### line 399, trailing  _(unsure)_

```python
except Exception:
```

artist without colour

## FigureSettingsDialog._redraw_now

### lines 538-549

```python
if self._rendering:
```

RENDERS MUST NOT STACK.

A preview blocks the GUI thread for ~150 ms. Qt keeps delivering events during that render -- spin-box auto-repeat, wheel, the timer itself -- and without this guard each one lands another render behind the current one. The queue grows faster than it drains and the window stops responding: the hang.

Instead a request that arrives mid-render only sets a flag, and one final redraw runs afterwards. Interaction stays smooth because the thread is always free between renders, and the picture still ends up matching the controls.

### line 558  _(unsure)_

```python
self._on_change()
```

A caller that does not know about preview rendering.

## FigureSettingsDialog._statistics_tab

### line 614, trailing  _(unsure)_

```python
except Exception:
```

module absent

## FigureSettingsDialog._figure_tab

### lines 722-742

```python
all_text = QSpinBox()
```

One control that reaches EVERY text object at once, because "make the fonts bigger" is a single intention.

GitHub issue #108 (2026-08-17): "Font size is by default to large to be visible. Adjusting font size ... from 10 to 2, does not reduce the font size, in fact increases it, and when returning ... the font size has been returned to 10."

All three symptoms were this control, and the cause is what it did NOT reach. Measured on a volcano-shaped figure: 23 text objects, 20 reached, and the three it missed were

('EAF1', 22.0)      an ax.texts annotation -- a GENE LABEL, and the LARGEST text on the figure ('a run', 12.0)     the figure suptitle ('condition', 10.0) the legend's title

So shrinking "all text" shrank everything EXCEPT the biggest thing on the plot, which then dominated it -- and reads exactly as "the font got bigger". The volcano annotates its hits by name, so this is the common case, not a corner one.

### lines 744-746

```python
all_text.setRange(2, 96)
```

Down to 2, because 2 is what the reporter typed. A 2pt font is unreadable and that is their business; a control that silently clamps is one that lies about what it did.

### line 811, trailing  _(unsure)_

```python
except Exception:
```

odd colour spec

### line 817, trailing  _(unsure)_

```python
except Exception:
```

odd colour spec

## FigureSettingsDialog._figure_tab.set_all_text

### lines 754-768

```python
set_figure_text_size_override(figure, size)
```

AND REMEMBER IT ON THE FIGURE. Setting the sizes alone was not enough and that is issue #108's third symptom: the next full render calls `render_figure_to_png`, which re-applies the GLOBAL text-size preference to every text object, so the user's choice survived only until the dialog closed -- and reopening the dialog, which reads the size off the figure, showed the preference again. The override is per FIGURE and is not written to the preference, for the same reason the colour buttons on this tab are not: this dialog restyles the figure in front of the user, and the setting for every figure is Preferences.

Connected AFTER `setValue` above, so this runs only when a user moves the control -- seeding never writes back. That is the rule at the head of the figure colour section in `spacr/qt/preferences.py`: NEVER PERSIST A RESOLVED DEFAULT.

## FigureSettingsDialog._figure_tab.set_line_ink

### lines 779-794

```python
def set_line_ink(colour):
```

AND THE COLOUR OF ALL OF IT -- IN TWO CONTROLS, NOT ONE.

There was a size control here and no colour control at all, so the background could be changed and the writing on top of it could not, which on a dark background is a figure with invisible axes and no way to fix it. The first version of the fix was ONE "All text colour" that also drove the spines and the tick marks.

The maintainer's decision (instruction 152 B) splits it by what a mark IS rather than by which code draws it: "line color which should change the color of all lines including axis lines and ticks, and then a font color that controls the color of all font in the graph". So a user can now say "dark axes, coloured labels" or the other way round, and the first report -- "doesnt look like there is an option to change the axis color" -- has an answer that is not "change your text as well".

## FigureSettingsDialog._axes_tab

### line 854

```python
for label, getter, setter in (
```

Scales -- the data-bound controls a saved page could never offer.

### lines 868-870

```python
for label, getter, setter in (
```

Limits. Four boxes and an autoscale switch, because "zoom the volcano to the part with the hits in it" is the single most common thing anyone wants from a plot and there was no way to ask for it.

### lines 883-884  _(unsure)_

```python
box.setRange(-1e12, 1e12)
```

Room to move well outside the data, and enough precision for a log axis where the interesting range can be tiny.

### line 889, trailing  _(unsure)_

```python
box.setKeyboardTracking(False)
```

not one redraw per keystroke

### line 970  _(unsure)_

```python
spine_width = QDoubleSpinBox()
```

Spines and ticks

### lines 1008-1010

```python
handles, _labels = axis.get_legend_handles_labels()
```

Legend -- only offered when there is one, or something to make one from. A legend row on a figure with no labelled series is a control that does nothing.

### lines 1062-1069

```python
if len(series) > self.SERIES_DETAIL_LIMIT:
```

MANY SERIES GET A RULE, NOT A CONTROL EACH.

A volcano scatters once per compartment, so an axis can hold 27 collections. One block each is 135 controls and reads as styling individual data points, which is not a thing anyone wants to do to a screen. Past the threshold the dialog offers what actually governs the appearance: a palette applied across the series, and one set of size/opacity controls that reach all of them.

### line 1074  _(unsure)_

```python
for label, artist in series:
```

Few enough to be worth naming individually.

### lines 1100-1102

```python
try:
```

A collection returns an ARRAY of widths, one per element, not a scalar. float() on it happens to work today and is deprecated; take the first explicitly.

## FigureSettingsDialog._axes_tab.apply_limits

### line 901, trailing  _(unsure)_

```python
return
```

a zero-width axis throws; wait for the other box

## FigureSettingsDialog._axes_tab.apply_grid

### lines 941-945

```python
"""Show or hide the grid, passing line properties ONLY when enabling.
```

Line properties are passed ONLY when enabling. matplotlib warns "First parameter to grid() is false, but line properties are supplied" and then turns the grid ON regardless -- so the unconditional version made the checkbox unable to switch the grid off, which is the opposite of what it says.

## FigureSettingsDialog._axes_tab.apply_legend

### lines 1034-1037

```python
handles, _labels = axis.get_legend_handles_labels()
```

Rebuilding needs labelled artists. Calling legend() without them warns "No artists with labels found to put in legend" and returns nothing, losing the legend the figure already had -- so an existing legend is restyled in place instead.

## FigureSettingsDialog._axes_tab.set_colour

### line 1087, trailing  _(unsure)_

```python
except Exception:
```

artist without colour

## apply_line_colour

### line 1210, trailing  _(unsure)_

```python
artist.set_edgecolor(colour)
```

a spine

### line 1214, trailing  _(unsure)_

```python
except Exception:
```

odd spec

### lines 1217-1221

```python
try:
```

THE TICK MARKS, SEPARATELY, and `color=` only. `colors=` would set the LABEL as well, which is the conflation the two controls exist to undo -- and it is done through `tick_params` rather than over the current ticks because matplotlib rebuilds them on every draw, so a colour set on the objects is lost at the next autoscale.

## apply_font_colour

### lines 1253-1254

```python
try:
```

The labels are regenerated on every draw, so the colour has to be set on the TICK rather than only on today's label objects.

## figure_follows_the_theme

### line 1276, trailing  _(unsure)_

```python
except Exception:
```

no store

## graph_style_as_dict

### line 1316, trailing  _(unsure)_

```python
except Exception:
```

no store

## _pairs_from_axes

### lines 1547-1548  _(unsure)_

```python
try:
```

BARS. Height is the value and the bar's centre picks the tick label. Patches that span the whole axes are backgrounds, not data.

### line 1567

```python
try:
```

SCATTER AND STRIP.

### lines 1579-1580

```python
try:
```

LINES AND MARKERS. A line with no marker and two points is usually a reference line rather than data, and is left out.

## derive_replot_recipe

### lines 1618-1619

```python
return None
```

ONE AXES ONLY. A grid of panels redrawn as a single violin would throw away every panel but one, silently.

### lines 1625-1634

```python
return {
```

NO `nunique() < 1` GUARD. The group column comes only from

`_named`, which returns either a tick label or `f"{float(x):g}"` always a str, never a missing value, and "nan" for a NaN x rather than NaN itself. With at least two rows guaranteed above, `nunique()` is 1 or more by construction.

It could only be reached by making `_pairs_from_axes` return actual NaN group names, which is not a figure. Instruction 310 A15 counted it, and a reader maintaining this was being told nameless groups are a case that occurs.

## _which_types_fit

### lines 1718-1722

```python
alias = {"bar_jitter": ("jitter_bar", "jitter_box")}
```

The two vocabularies differ: `graph_types` says `bar_jitter` where the drawer says `jitter_bar`, and it has no `jitter_box`. Map the ones that mean the same thing rather than renaming either -- one is the analysis vocabulary and the other the drawer's, and each is right in its own module.

## _add_group_colours._recolour

### lines 1768-1771

```python
recipe["colors"] = current
```

STORED ON THE RECIPE AND REDRAWN, not painted onto the artists. Setting an artist's colour lasts until the next redraw and then silently reverts -- which is what "changing the colors changes nothing" looks like from the other side.

## _add_group_colours

### lines 1783-1785

```python
note = colours.addAction(
```

NAMED, NOT SILENTLY DROPPED. A menu that shows the first twenty-four of ninety groups and says nothing looks like a menu that has them all.

## save_figure_bundle

### lines 1839-1841

```python
groups = {str(key): part[value].dropna().to_numpy()
```

`observed=True`: a categorical grouping column would otherwise yield a group per unused CATEGORY as well, and an empty group is a comparison arm with no observations in it.

## save_figure_bundle._render

### lines 1848-1850

```python
"""Render one file of the bundle through the SHARED export path.
```

A bundle deliberately contains both formats, but each rendering still uses the shared export path so print colours, embedded fonts, and raster DPI match every other figure the user keeps.

## build_figure_context_menu

### lines 1909-1914

```python
owner = parent if parent is not None else menu
```

AN OWNER FOR THE ACTIONS, WHICH IS NOT ALWAYS `parent`. `QMenu.addAction` does not adopt an action built here, so a QAction whose only reference is a local name and whose parent is `None` is collected the moment this function returns -- and the menu comes back holding Save, the two submenus and nothing else. `add_graph_style_file_entries` already falls back this way for the same reason.

### lines 1924-1927

```python
recipe = getattr(figure, "_spacr_replot", None)
```

SHOW THE SAME DATA ANOTHER WAY (178 A). Offered only where the figure carries its own recipe -- `create_grouped_plot` attaches one -- because a menu entry that cannot redraw the figure it is on is worse than an absent one. Every other figure in spaCR simply does not get the group.

### lines 1930-1934

```python
derived = derive_replot_recipe(figure)
```

NO RECIPE, SO READ ONE BACK OFF THE AXES. Only

`create_grouped_plot` attaches `_spacr_replot`, which left the menu on a handful of figures and absent from every other plot in the software. Derived recipes are marked so a redraw does not claim to be the original data.

### lines 1944-1951

```python
show_as = QMenu(tr("Graph type"), menu)
```

Give Python and C++ an explicit ownership chain. ``addMenu(str)`` can leave the Python wrapper as the submenu's only live owner, so a caller retrieving it through the parent action gets an already deleted QMenu. This is the same lifetime rule used by Appearance and Axis scale below. NAMED "Graph type", which is what it was asked for by: "an option when i right click on a graph, called graph type that would allow the user to switch between graph types".

### lines 1955-1957

```python
fits, why_not = _which_types_fit(recipe)
```

ONLY THE TYPES THAT FIT THE DATA (instruction 200 A), and the rest greyed with the reason rather than absent -- a list that silently shortens leaves the user wondering whether they misremembered.

### lines 1973-1977

```python
_add_group_colours(menu, figure, recipe, on_change, parent)
```

THE GROUPS, NOT THE ELEMENTS (reported 2026-08-21: "i also want to modify thing on the group level not individual points and barts"). The Appearance menu below colours the FURNITURE -- spines, ticks, text -- which is why changing a colour there appeared to do nothing to the bars: it was never about them.

### lines 1980-1983

```python
_add_bundle_save(menu, figure, parent)
```

AND THE WHOLE THING, on every figure that has its data (instruction 223). This was on the pyqtgraph plots only, which is not where these graphs are drawn -- so a feature that existed was unreachable from where the user was looking.

### line 2040, trailing  _(unsure)_

```python
scales = QMenu(tr("Axis scale"), menu)
```

see "Appearance" below for why

### lines 2053-2062

```python
appearance = QMenu(tr("Appearance"), menu)
```

THE TWO COLOUR CONTROLS, ON THE RIGHT-CLICK ITSELF (instruction 152 B). They are two clicks away behind "Figure settings…", and the report that opened 152 was a user who could not find an axis colour at all -- a control nobody can find is a control that does not exist. BUILT WITH AN EXPLICIT PARENT, not `menu.addMenu("Appearance")`. `addMenu(str)` hands back a QMenu that PySide does not keep alive: the Python wrapper is the only owner, and the moment it goes out of scope the C++ object is deleted under the still-visible parent action. Driving the entry then raises "Internal C++ object (QMenu) already deleted", which is what a user would see as a submenu that opens empty.

### lines 2111-2114

```python
styled = QAction(tr("Save figure with a preview…"), owner)
```

STYLE IT FOR THE FILE FIRST (178 C.2). "the user should be able to change all of theis for the saved graph, get a preview then save." Beside the direct save rather than replacing it: writing what is on screen is one click and remains one click.

## save_figure_as

### lines 2242-2255

```python
extension = os.path.splitext(path)[1].lower().lstrip(".")
```

THROUGH `spacr.plot.save_figure`, WHICH IS THE POINT OF INSTRUCTION 108 POINT 6. This function used to write the file itself, with the SCREEN's background and no print rule, which made it one of the twenty-three `savefig` calls that bypass the one place a figure the user keeps gets written -- and on a dark theme it produced exactly instruction 150's report: white text on a transparent ground, invisible the moment it is pasted into a manuscript. `save_figure` applies the DPI preference and `print_ready`'s ink rule, and a light-mode save is unchanged by it.

THE USER'S OWN EXTENSION WINS over the format preference, and that is the one thing this cannot delegate: `save_figure` corrects the extension to the chosen FORMAT, so a user who typed `figure.pdf` while the preference says PNG would get a PNG. The extension is passed as `fmt` when it is one `save_figure` knows.

### line 2259, trailing  _(unsure)_

```python
except Exception:
```

Qt-only build

### lines 2272-2275

```python
try:
```

SVG and EPS are NOT among `FIGURE_FORMATS`, so they cannot go through `save_figure` without having their extension rewritten under them and they are offered in the dialog because they are what a journal asks for. They still get the print rule, which is the half that matters.

### line 2281, trailing  _(unsure)_

```python
except Exception:
```

no settings store

### lines 2291-2292  _(unsure)_

```python
vector = extension in ("pdf", "svg", "eps")
```

Vector formats have no meaningful DPI, and passing one makes matplotlib rasterise text in some backends.

## export_sidecars

### lines 2352-2354

```python
labels = list(usable)
```

EVERY PAIR, corrected across them. Six pairwise tests at 0.05 is a 26% chance of one false positive and the individual p-values give no hint of it.

## Module level

### lines 2391-2417

```python
_FALLBACK_CHOICES = {
```

INSTRUCTION 118 -- FIGURE PREFERENCES: GENERAL, AND PER GRAPH TYPE

"in the general app preferences in the figure tab theere should be general graph settings and specialized settings for al the possible different sets of graphs"

The MODEL for this already existed: `spacr.figure_style` holds GENERAL_DEFAULTS, GRAPH_DEFAULTS and `resolve`, and `spacr.figures.style` lays a user's deltas over the publication house style. What did not exist was any way to SET them -- the Figures tab held format, DPI, cache size and the dynamic switch, and nothing at all about how a plot looks.

BUILT FROM `figure_style`'S OWN TABLES, not from a hand-written list. That is the same decision `add_style_entries` made for instruction 108 and for the same reason: a style gains a key, the panel gains a control, and the two cannot fall out of step. It also means this file never has to know what a volcano is.

THE STORE HOLDS DELTAS, NOT THE RESOLVED STYLE, and that contract is older than this panel -- `get_figure_style` returns {} on a fresh install and `figures.style.user_overrides` returns only the keys the user MOVED, so a user who has never opened Preferences gets the published house style exactly. Writing the whole resolved style here would replace the house style for everybody, which is the same class of mistake as instruction 152 A's persisted resolution.

### lines 2419-2421

```python
_FALLBACK_CHOICES = {
```

Fallback choices keep the preferences panel usable if ``figure_style`` cannot be imported. Normal operation reads the canonical choices from that module through ``style_choices_for``.

## FigureStylePreferences

### lines 2520-2525

```python
class FigureStylePreferences(QWidget):
```

THE ARGUMENTS ARE DOCUMENTED ON `__init__`, ONCE. They were listed here too, as a NumPy ``Parameters`` section, and AutoAPI runs with ``class_content='both'``: the class docstring and ``__init__``'s are concatenated before Napoleon sees them, the section became a field list, and ``__init__``'s opening prose then ended it mid-way -- "Field list ends without a blank line", which `sphinx-build -W` makes fatal.

## FigureStylePreferences.__init__

### lines 2609-2612

```python
file_row = QHBoxLayout()
```

SAVE / LOAD, instruction 108 point 5, on the panel that owns these settings. The same file the figure's right-click menu reads and writes -- one format, two ways in, and no third place a graph style can live.

### lines 2629-2636

```python
from ..screens.settings_model import retarget_field_tooltips
```

HOVER HELP BELONGS ON THE SETTING'S NAME, NOT ON THE CONTROL

(instruction 113, restated across every module 2026-08-19: "the tooltip should only be visable when hovering the mouse over the setting name text, and not when hovering over the field, checkbox, or whatever the setting controlls"). One post-pass rather than a convention every hand-built row has to remember -- which is what `tests/test_tooltips_are_on_the_setting_not_the_field.py` exists to catch, and did catch this screen.

## FigureStylePreferences._save_to_file

### line 2640  _(unsure)_

```python
def _save_to_file(self) -> None:
```

a house style as a file

## FigureStylePreferences._control

### lines 2727-2730

```python
combo.addItem(f"{value} (not offered)", value)
```

A stored value the package no longer offers. Kept and shown rather than snapped to the first entry, because silently changing a user's setting while showing them a settings dialog is the worst place to do it.

## FigureStylePreferences._control._set_colour

### lines 2749-2753

```python
b.setText(str(v))
```

THE SWATCH TOO. `_colour_button` paints itself from its own state, so writing the holder alone would leave the button showing the old colour -- a reset the user can see did not happen. Rebuilt in place rather than reaching into the button's private state.
