# Notes from `spacr/qt/preferences.py`

Prose lifted out of `spacr/qt/preferences.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (17 entries)
- [_mandelbrot_defaults](#_mandelbrot_defaults) (2 entries)
- [_LazyDefaults._load](#_lazydefaults_load) (1 entry)
- [speed_group_values](#speed_group_values) (1 entry)
- [scale_group_values](#scale_group_values) (1 entry)
- [explain_a_fractal_number](#explain_a_fractal_number) (1 entry)
- [set_figure_style_default](#set_figure_style_default) (1 entry)
- [auto_figure_colors](#auto_figure_colors) (1 entry)
- [_migrate_frozen_figure_colors](#_migrate_frozen_figure_colors) (2 entries)
- [_unfreeze_figure_colors_that_fight_the_theme](#_unfreeze_figure_colors_that_fight_the_theme) (4 entries)
- [get_figure_colors](#get_figure_colors) (1 entry)
- [set_figure_colors_auto](#set_figure_colors_auto) (1 entry)
- [get_figure_text_size](#get_figure_text_size) (1 entry)
- [theme_choices](#theme_choices) (1 entry)
- [get_theme_choice](#get_theme_choice) (1 entry)
- [set_theme_choice](#set_theme_choice) (1 entry)
- [theme_background_path](#theme_background_path) (1 entry)
- [resolve_effective_theme](#resolve_effective_theme) (3 entries)
- [get_ambient_theme](#get_ambient_theme) (2 entries)
- [set_ambient_theme](#set_ambient_theme) (1 entry)
- [ambient_default_palette](#ambient_default_palette) (1 entry)
- [_ambient_ranges](#_ambient_ranges) (1 entry)
- [_migrate_ambient_motion](#_migrate_ambient_motion) (1 entry)
- [_ambient_multiplier](#_ambient_multiplier) (1 entry)
- [apply_ambient_preferences](#apply_ambient_preferences) (6 entries)
- [get_fractal_settings._number](#get_fractal_settings_number) (1 entry)
- [set_fractal_settings](#set_fractal_settings) (4 entries)
- [laptop_mode_note](#laptop_mode_note) (1 entry)
- [get_performance_level](#get_performance_level) (1 entry)
- [set_performance_level](#set_performance_level) (2 entries)
- [set_spacr_mode](#set_spacr_mode) (1 entry)
- [get_spinner_delay](#get_spinner_delay) (1 entry)
- [set_field_fade_enabled](#set_field_fade_enabled) (1 entry)
- [color_blind_categorical_palette](#color_blind_categorical_palette) (2 entries)
- [apply_preferences_to_app](#apply_preferences_to_app) (17 entries)
- [_disk_report_runner](#_disk_report_runner) (1 entry)
- [_disk_button](#_disk_button) (1 entry)
- [_still_asking](#_still_asking) (2 entries)
- [_start_disk_report](#_start_disk_report) (2 entries)
- [_start_disk_report.done](#_start_disk_reportdone) (2 entries)
- [explain_every_row](#explain_every_row) (4 entries)
- [PreferencesDialog.__new__](#preferencesdialog__new__) (1 entry)
- [PreferencesDialog._build_the_dialog](#preferencesdialog_build_the_dialog) (54 entries)
- [PreferencesDialog._build_the_dialog._reload_ambient_palettes](#preferencesdialog_build_the_dialog_reload_ambient_palettes) (2 entries)
- [PreferencesDialog._build_the_dialog._percent_row._update](#preferencesdialog_build_the_dialog_percent_row_update) (1 entry)
- [PreferencesDialog._build_the_dialog._percent_row](#preferencesdialog_build_the_dialog_percent_row) (1 entry)
- [PreferencesDialog._build_the_dialog._sync_mode_note](#preferencesdialog_build_the_dialog_sync_mode_note) (2 entries)
- [PreferencesDialog._build_the_dialog._whole](#preferencesdialog_build_the_dialog_whole) (1 entry)
- [PreferencesDialog._build_the_dialog._quit_spacr](#preferencesdialog_build_the_dialog_quit_spacr) (1 entry)
- [PreferencesDialog._build_the_dialog._resource_button](#preferencesdialog_build_the_dialog_resource_button) (2 entries)
- [PreferencesDialog._build_the_dialog._workspace_copying](#preferencesdialog_build_the_dialog_workspace_copying) (1 entry)
- [PreferencesDialog._build_the_dialog._reset_to_defaults](#preferencesdialog_build_the_dialog_reset_to_defaults) (1 entry)
- [PreferencesDialog._build_the_dialog._save](#preferencesdialog_build_the_dialog_save) (14 entries)
- [_everything_explains_itself_in_the_strip](#_everything_explains_itself_in_the_strip) (2 entries)

## Module level

### lines 163-165

```python
_KEY_LOG_FILE_LEVELS = "prefs/log_file_levels"
```

Stored as level names ("INFO,WARNING,ERROR") rather than numbers: QSettings round-trips strings predictably across platforms, and a settings file a human might open says what it means.

### line 567  _(unsure)_

```python
_KEY_FIG_FORMAT = "prefs/figure_format"
```

Figure rendering

### lines 578-584

```python
_KEY_FIG_LIVE_CACHE = "prefs/figure_live_cache"
```

How many of the most recent figures keep their LIVE matplotlib Figure, and what happens to the ones past that.

A live Figure is what makes a figure restylable: it still has a legend to toggle, an axis to set log, series to recolour. A pixmap has none of those — it is a picture of a figure. Keeping every Figure forever is not an option either, since each holds its own data arrays.

### lines 717-719  _(unsure)_

```python
_KEY_FIG_STYLE = "figures/style_general"
```

Figures — display format (png / pdf) + png resolution

### lines 1159-1182

```python
_KEY_FIG_BG = "prefs/figure_bg"
```

Figure colours. Stored as TOKENS, never as answers: either an explicit colour the user picked, or "auto" (the default), which is resolved against the live theme on every read.

NEVER PERSIST A RESOLVED DEFAULT.

This is the rule the whole section exists to enforce, and it is written here because this is where the next person will be standing when they are about to break it. Writing back what "auto" happened to resolve to turns a preference that TRACKS into a preference that FREEZES, and the damage outlives the session that caused it: a store holding "#ffffff" cannot be told apart from a user who chose white, so nothing downstream can ever undo it. That is not hypothetical -- `_FigureSettingsDialog` seeded itself from `get_figure_colors()` (resolved) and wrote the same pair back on OK, so opening the dialog once on a dark theme and pressing OK without touching anything froze every future figure white, including on a light theme. See `_migrate_frozen_figure_colors` for the clean-up that costs.

The same reasoning is already recorded one screen up for `get_figure_style`, which stores {} rather than today's defaults for exactly this reason.

So: anything that can WRITE this preference back seeds itself from `get_figure_color_tokens()`, shows `auto_figure_colors()` as a labelled PREVIEW, and passes "auto" to `set_figure_colors` unless the user picked.

### lines 2215-2217  _(unsure)_

```python
SPACR_MODES = ("extra_performance", "performance", "balanced")
```

spaCR mode — how hard the app tries to stay out of the machine's way

### lines 2372-2375

```python
"balanced": {
```

A LEVEL SETS THE GROUP'S CONTROL, not its members. Scale derives render scale and supersampling, so a preset that also named those fought the derivation and whichever was written last won -- which is how choosing High stopped giving High's supersampling.

### lines 2389-2391

```python
"scale": 2.5,
```

2.5 is where supersampling reaches three samples a side, which is nine a pixel. Past three the difference is smaller than the screen can show and the cost is the square.

### lines 3535-3537  _(unsure)_

```python
VALID_DOCK_MODES = ("locked", "hidden")
```

The left dock — revealed, pinned, or gone

### lines 3592-3594  _(unsure)_

```python
DEFAULT_PANE_OPACITY_PCT = 60
```

Page opacity — how solid shared page and module surfaces are

### lines 3646-3648

```python
DEFAULT_HASH_INPUTS = True
```

The field fade — the exception to the setting above

### lines 3981-3983  _(unsure)_

```python
DEFAULT_DB_BROWSER_EDITABLE = False
```

Database Browser — editing is opt-in

### lines 4037-4039  _(unsure)_

```python
DEFAULT_SHOW_ALPHA = True
```

Module visibility — whether unfinished modules and settings are shown

### line 4600  _(unsure)_

```python
"Error bars": "Statistic represented by error bars: standard deviation, standard error or confide...
```

figures: statistics drawn on the plot

### lines 4620-4625

```python
"Lock axis scales": "Whether one y unit is drawn the same length as one x "
```

KEYED BY THE LABEL THE ROW SHOWS, which for the `aspect` setting comes from `style_setting_label`. The setting itself offers 'equal' or 'auto', so it is the axis-scale lock -- one y unit drawn the same length as one x unit -- and not the shape of the figure. The shape is 'Page shape' below, and the explanation says so rather than describing shapes this control cannot take.

### lines 7103-7105  _(unsure)_

```python
_KEY_SAVE_WORKSPACE = "runs/save_workspace"
```

Workspace content retained with a saved run

### lines 7174-7180

```python
_KEY_RIM_LENGTH = "rim/length_px"
```

The travelling rim

The accent that runs round a settings card follows the pointer, and how it does that is a matter of taste rather than of correctness -- so it is three settings rather than three constants.

## _mandelbrot_defaults

### lines 243-247

```python
return dict(DEFAULTS)
```

EXACTLY WHAT WAS GIVEN. An earlier version overrode the cost numbers with the `balanced` preset to keep the first impression light; the maintainer then handed over the command line they actually run, supersampling 2 and render scale 1.0 among it, and a later instruction wins over an earlier inference.

### lines 250-258

```python
return {"supersampling": 2, "seconds_per_decade": 24.0,
```

A LITERAL COPY, because this branch exists for the build where the renderer cannot be imported at all -- so it cannot read the numbers it is mirroring. It had drifted from them twice: `supersampling` was still 1 and `max_depth` still 34.0 from the version that deliberately shipped a lighter preset, against the renderer's 2 and 21.0. A headless install therefore opened on a different pattern from the one that is documented. `test_the_fractal_defaults_are_the_published_ones_or_the_written_ fallback` now compares every shared key, so a third drift fails.

## _LazyDefaults._load

### lines 292-293

```python
self._loaded = True
```

Set FIRST: `_mandelbrot_defaults` cannot recurse into this, but a failure part-way through must not leave it retrying on every read.

## speed_group_values

### lines 432-433  _(unsure)_

```python
values["seconds_per_decade"] = round(
```

A DURATION, so it goes the other way: twice the speed is half the time a decade takes.

## scale_group_values

### line 448

```python
values["supersampling"] = 1 if amount < 1.5 else (2 if amount < 2.5
```

Whole samples a side: 1 below 1.5, then 2, then 3 at 2.5 and above.

## explain_a_fractal_number

### line 470, trailing  _(unsure)_

```python
if number != number:
```

NaN

## set_figure_style_default

### lines 962-964

```python
settings.setValue(_KEY_FIG_STYLE_DEFAULTS, json.dumps(stored))
```

JSON rather than a nested QVariant map: QSettings' INI writer flattens a dict of dicts into keys containing the field names, and a style field called `x_label` would then be indistinguishable from a group.

## auto_figure_colors

### lines 1247-1248

```python
dark = resolve_effective_theme() != "light"
```

Light is the only light theme; Space is a dark one, so a `== "dark"` test here would have handed it white figures.

## _migrate_frozen_figure_colors

### lines 1282-1286

```python
for key, frozen, label in (
```

The line key is examined on the same pass rather than behind a scale bump, and that is safe rather than lucky: it did not exist before this migration shipped, so a store already marked cannot hold a frozen one. A key that CAN predate the marker needs the bump; this one cannot.

### lines 1301-1312

```python
LOG.warning(
```

WARNING AND NOT INFO, AND THE LEVEL IS THE MESSAGE'S JOB. This line exists so that changing a stored preference underneath the user is not silent -- the docstring above says "with a line in the console". It was not reaching one: `spacr.qt` is pinned by the app's level policy, so an INFO logged from `spacr.qt.preferences` is dropped at the source before any handler sees it. Measured after pressing Save in Preferences: `spacr.qt` at level 30, `spacr.qt.preferences` at NOTSET, so the effective level for this module is WARNING.

It is warning-grade on its own merits too: something the user did not ask for happened to their settings.

## _unfreeze_figure_colors_that_fight_the_theme

### line 1349, trailing  _(unsure)_

```python
return
```

chosen through the current dialog/API

### line 1353, trailing  _(unsure)_

```python
return
```

nothing frozen to hand back

### line 1355, trailing

```python
return
```

a colour "auto" never produced

### line 1360, trailing  _(unsure)_

```python
return
```

frozen, but at today's answer anyway

## get_figure_colors

### lines 1430-1431  _(unsure)_

```python
if figure_color_is_auto(bg):
```

An EXPLICIT colour the user has chosen is still honoured; only the "auto" halves are substituted.

## set_figure_colors_auto

### lines 1465-1466

```python
set_figure_line_colour(AUTO_FIGURE_COLOR)
```

All THREE, because "follow the theme" that left one of them frozen would be the trap this function exists to be the way out of.

## get_figure_text_size

### line 1475, trailing  _(unsure)_

```python
return 0
```

0 = leave matplotlib's own sizes alone

## theme_choices

### lines 1518-1520

```python
choices.extend(
```

Space is gone; its variants are no longer offered. The wallpapers are named for what they show rather than prefixed with a theme the user never has to think about.

## get_theme_choice

### lines 1531-1537

```python
if theme == "cell":
```

NO `space` BRANCH. "space" is not in VALID_THEMES -- `set_theme` refuses it and `theme_choices` offers no `space:` token -- so a branch for it could not be reached by any route through this module, and coverage counted three items nothing could execute. The Space ARTWORK still exists and `spaceout` still draws it; what is gone is the theme by that name, which is why the variant accessors below stay.

_Corrected 2026-09-19:_ the variant accessors did not stay. They were retired on 2026-09-09 (61c896555, instruction 364), because `spaceout`'s "space" pattern is `widgets/fractal_space.py` and reads neither key. What was left behind was the two stored values: the key names were kept "so a stored value can still be recognised and cleared", and nothing cleared them. `_forget_the_space_theme_keys`, called from `get_theme`, now removes `prefs/space_variant` and `prefs/space_seed` from a store the first time the theme is read. It writes only when a key was there and skips safe mode, which reads nothing it was given.

## set_theme_choice

### lines 1549-1550  _(unsure)_

```python
if choice.startswith("cell:"):
```

Likewise no `space:` prefix: the validity check above rejects any token `theme_choices` does not offer, and it offers none.

## theme_background_path

### lines 1617-1619

```python
if theme == "cell":
```

NO `space` BRANCH. It was unreachable -- see the retirement note above `space_variants`' former home -- and a branch nothing can enter is a branch that will be read as live by the next person to touch this.

## resolve_effective_theme

### line 1636  _(unsure)_

```python
try:
```

system — poll Qt's palette hint

### lines 1642-1644

```python
bg = app.palette().color(QPalette.ColorRole.Window)
```

THE ENUM, NOT THE INSTANCE. `palette.Window` was removed in

PySide6 6.x, and the bare except below would swallow the AttributeError and hand every desktop the dark theme.

### line 1646

```python
lum = (0.299 * bg.red() + 0.587 * bg.green()
```

crude luminance test — < 128 → dark scheme

## get_ambient_theme

### lines 1800-1802

```python
from .widgets.ambient import (
```

Aliased on import: this module's own DEFAULT_THEME is the *app* theme (dark), and shadowing it here would be a trap for the next reader.

### lines 1807-1810

```python
return raw if raw in AMBIENT_THEMES else DEFAULT_AMBIENT_THEME
```

"none" lands on the default here rather than propagating: this getter's whole contract is that its answer can be painted, and the callers that must not paint at all are gated on `get_ambient_enabled()`, which is already False in that case.

## set_ambient_theme

### line 1831  _(unsure)_

```python
stored = str(settings.value(_KEY_AMBIENT_PALETTE, ""))
```

Repair the companion key against the theme we just stored.

## ambient_default_palette

### lines 1850-1853

```python
return DEFAULT_PALETTE
```

"Never raises for an unknown theme" was a claim this function did not keep: the real `palettes_for` raises ValueError on a name it has no engine for, which "none" now is, and the exception escaped a Qt slot in the middle of refilling the palette picker.

## _ambient_ranges

### lines 1893-1899

```python
def _ambient_ranges():
```

blur, speed and size

Three multipliers on whatever the chosen animation already does, rather than absolute pixels or seconds. 1.0 means "as shipped" in every theme, so a user who never opens these controls sees the animation that was designed; the ranges and the clamping live in the widget module next to the engines that honour them, because a number this module accepted and the engine then rejected would be a preference that silently does nothing.

## _migrate_ambient_motion

### line 1956, trailing  _(unsure)_

```python
if old == old and old > 0:
```

not NaN

## _ambient_multiplier

### line 1988, trailing  _(unsure)_

```python
if value != value:
```

NaN — a hand-edited INI can hold one

## apply_ambient_preferences

### line 2161  _(unsure)_

```python
theme = get_ambient_theme() if enabled else None
```

Only read — and only repaint into — what is going to be shown.

### line 2175  _(unsure)_

```python
widget.set_animating(False)
```

Stop first, then hide: no last frame on the way out.

### lines 2180-2190

```python
widget.set_animating(True)
```

Unconditionally True, NOT `widget.isVisible()`. A module screen the user is not currently looking at has an invisible backdrop, so the isVisible() form latched `_animating = False` onto every background tab the moment Preferences was saved — and because `showEvent` honours that flag (which is what makes a pause a pause), those screens never animated again for the rest of the session. Un-pausing an off-screen widget costs nothing: `AmbientWidget._should_run` already refuses to tick while hidden, and the widget's own showEvent starts it when the tab comes back — which is what this function's docstring already says is supposed to happen.

### lines 2192-2196

```python
try:
```

Everything cosmetic goes *after* the run state, and in its own guard. The bug described above cost a session's worth of animation because one step of this loop threw and skipped the rest; a backdrop that comes back in last week's palette is a smaller failure than a backdrop that never moves again.

### lines 2200-2201  _(unsure)_

```python
widget.set_blur(blur)
```

After the theme: a theme change rebuilds the engine, and all of these ride on it.

### lines 2016-2018

```python
if (sys.modules.get(f"{__package__}.widgets.ambient") is None
```

A WIDGET OF A CLASS WHOSE MODULE WAS NEVER IMPORTED CANNOT EXIST, so when the backdrop is off and `spacr.qt.widgets.ambient` is not in `sys.modules` there is nothing here to hide, and importing the module to find that out was the cost. It was the last thing that imported the backdrop module into `safespacr`, which never builds one (296; measured 2026-09-15: `launch` -> `apply_preferences_to_app` -> here). Asked in this order, `get_ambient_enabled` answers from `SPACR_NO_BACKDROP` before it reads the stored animation, which would import the module itself. When the backdrop is on, or a widget may exist, the walk below runs exactly as before.

## get_fractal_settings._number

### line 2457, trailing  _(unsure)_

```python
if value != value:
```

NaN

## set_fractal_settings

### lines 2575-2578

```python
"speed": (_KEY_FRACTAL_SPEED, (0.0, None)),
```

ASKED FOR 2026-08-28: capped at 1000000, not 8. The speed is a multiplier on the flight's own clock, so there is no physical ceiling to respect -- past a few hundred the picture becomes a blur, and someone who wants that has asked for it.

### lines 2633-2636

```python
derived = (speed_group_values(value) if name == "speed"
```

ONE CONTROL, ITS WHOLE GROUP. Six fields answered "how fast" and three answered "how finely"; set by hand they could contradict each other, and changing the one a user thinks of as Speed left the other five where they were.

### lines 2647-2650

```python
from .widgets.fractal_mandelbrot import steering_from_one_number
```

ONE CONTROL, THREE NUMBERS, derived together so they cannot contradict each other. Set by hand a short interval and a long duration make the camera re-target before it has finished moving, which is the jerkiness reported on 2026-08-28.

### lines 2662-2666

```python
low, high = bounds
```

THE FIELD'S NUMBER IS KEPT. Only a value that cannot work at all is moved, and `explain_a_fractal_number` is what tells the user about it before they get here -- a store that silently reduced 8000 to 8 would make the field a control that lies about what it did.

## laptop_mode_note

### lines 2806-2809

```python
_on, why = wanted({**measure(), "override": None})
```

Asked with the override cleared, because what the note has to report is what the MEASUREMENT says -- an environment variable set for one launch would otherwise be read back as the machine's own answer.

## get_performance_level

### lines 2914-2916

```python
try:
```

WRITTEN BEFORE THE OLD KEYS ARE TRUSTED AGAIN, so a crash between the two cannot lose the answer: the worst case is a migration that runs twice and reaches the same level.

## set_performance_level

### lines 2935-2936  _(unsure)_

```python
posture = spacr_mode_for_level(level)
```

THROUGH `set_spacr_mode`, so the visual stashing that entering and leaving Extra Performance does still happens. It writes both keys.

### lines 2939-2940  _(unsure)_

```python
settings = _settings()
```

`set_spacr_mode` wrote the posture as the level; correct it to the level the caller actually asked for, which is the finer value.

## set_spacr_mode

### lines 3034-3038

```python
settings.setValue(_KEY_PERFORMANCE_LEVEL, mode)
```

AND THE LEVEL, because that is the value everything reads now. Each of the three modes is also a level, so setting one is an unambiguous statement about the other -- and leaving them to disagree is exactly the two-answers-to-one-question defect 286 removed. Written directly rather than through `set_performance_level`, which calls back here.

## get_spinner_delay

### line 3196, trailing  _(unsure)_

```python
if value != value:
```

NaN

## set_field_fade_enabled

### lines 3713-3714

```python
pass
```

Headless, or PySide6 not importable: the cache cannot be stale if it was never built.

## color_blind_categorical_palette

### line 3781  _(unsure)_

```python
return ["#4A9EFF", "#3fb950", "#f0883e", "#a78bfa",
```

Default spaCR categorical palette — matches theme accents

### line 3784

```python
return ["#0072B2", "#E69F00", "#009E73", "#F0E442",
```

Okabe-Ito — see https://jfly.uni-koeln.de/color/

## apply_preferences_to_app

### lines 4092-4094  _(unsure)_

```python
def apply_preferences_to_app(app=None) -> None:
```

Wire prefs into the running QApplication

### lines 4124-4127

```python
try:
```

Instruction 180. Pushed on every preferences save and not only at startup: the run journal reads a module-level default it can see without Qt, and a user who changed the setting mid-session would otherwise not see it take effect until the next launch.

### lines 4137-4143

```python
background = theme_background_path(theme)
```

Only the image themes want a picture, and only they pay for producing one. Everything here degrades to None on any failure, and the stylesheet renders a gradient in that case.

This is also the ONLY call site that can decode a master. It runs at startup and on a preferences save — never from a resize, never from a paint. See :func:`spacr.qt.imagery.decode_count`.

### lines 4146-4150

```python
try:
```

Fields before the stylesheet, not after: importing the module is what registers its QSS block, and dropping the cached preference is what lets that block agree with the painter about whether the effect is on. Do it the other way round and the first save after a toggle emits the previous state's stylesheet.

### lines 4159-4164

```python
style_signature = (
```

Setting one application stylesheet asks Qt to unpolish and repolish EVERY live widget. Preferences used to do that even when the user only changed a logging, cache or export option, and a mature session can own thousands of controls. The complete visual input is small and stable; remember it and pay the global rebuild only when one of those inputs or the set of late widget-QSS registrars -- actually changed.

### lines 4176-4186

```python
or window_stylesheet(app)
```

The live sheet is public state.  Tests, embedding hosts and theme integrations may replace it without going through this function, so the signature is only valid while the exact sheet it describes is still installed.  Checking the text is cheap beside a global Qt repolish and also catches a non-empty foreign sheet.

READ FROM THE WINDOWS, NOT FROM THE APPLICATION. The sheet goes on every top-level window now rather than on the QApplication, so `app.styleSheet()` is empty here and comparing it would report a change on every single save -- which is exactly the rebuild this guard exists to skip.

### lines 4189-4194

```python
or bool(app.styleSheet())
```

AND A NON-EMPTY APPLICATION SHEET IS FOREIGN BY CONSTRUCTION. `apply_stylesheet_per_window` clears it every time, so anything there was put there by somebody else -- a test, an embedding host, a theme integration -- and it applies to every widget including the ones inside our windows. That is the case this guard was written for and the window comparison above cannot see it.

### lines 4198-4200

```python
set_widget_qss_context(app, theme, scale, pane_opacity)
```

Record the exact inputs any screen-local late block must share with the application sheet. The local copies are absorbed into the complete global rebuild below once that sheet is composed.

### lines 4206-4208

```python
clear_widget_qss_overlays(app)
```

A local sheet outranks the application sheet. Remove old-theme copies only after the replacement exists, then install the complete sheet that now contains every block registered so far.

### lines 4210-4218

```python
apply_stylesheet_per_window(app, sheet)
```

PER WINDOW, NOT ON THE APPLICATION, and instruction 380 has the measurement: `QApplication.setStyleSheet` repolishes every widget the process owns, and a session that has opened four modules owns 9,045 of which 6,111 are on screens nobody can see. Measured on this box, offscreen: 7,500 ms that way against 1,900 ms this way, for the same picture. A window born after the change is covered by the filter `apply_stylesheet_per_window` installs, which is the property `tests/qt/test_a_dialog_never_opens_in_the_previous_theme` was written to hold before this line could be changed.

### lines 4223-4225

```python
try:
```

A field whose QSS did not change still has to redraw when the paint hook is turned off. Field fade is part of the signature, so this is needed on visual changes and never on an unrelated save.

### lines 4231-4232  _(unsure)_

```python
from .button_roles import install_button_roles
```

Run/Propagate and Stop/Close-style buttons are tagged centrally, including QDialogButtonBox buttons created after startup.

### lines 4236-4238

```python
apply_ambient_preferences(app)
```

The animated background follows the same rule as the theme: it is re-applied here, so toggling it (or switching palette) lands on the screens that are already open instead of at the next launch.

### lines 4241-4244

```python
try:
```

The console and AI chat paint their own entries with an explicit point size, so the stylesheet's font scale never reaches them. Push Zoom into every open one here, or changing it would only affect consoles opened afterwards.

### lines 4253-4257

```python
try:
```

Apply the verbose-logger preference too — cheap to re-apply, and this is the one place that runs on every prefs save. Also attaches the rotating file handler if it isn't already, so every spaCR launch drops a trail into ~/.spacr/logs/ regardless of whether the user turned verbose logging on.

### lines 4264-4266

```python
from ..logging_util import apply_level_policy
```

AFTER apply_verbose_logging, which still sets a blanket threshold on the attached loggers. The per-level switches are the finer statement and have to be the one that lands last.

### lines 4270-4271

```python
pass
```

Logger module is optional at import time — never let its absence prevent the app from theming itself.

## _disk_report_runner

### lines 4368-4379

```python
try:
```

Remade rather than reused when an application has appeared since: a runner that decided to be inline while there was no event loop would otherwise keep blocking its caller for the rest of the process.

RETIRED, NEVER DROPPED. `cancel` abandons the results but cannot interrupt a stat already in the kernel, and the old runner's `_jobs` is the only strong reference to that QThread. Collecting it here would destroy a running QThread and take the process with it, so the retired runner is kept for the life of the module. There is at most one per transition, and transitions happen when an application appears or goes.

## _disk_button

### line 4399  _(unsure)_

```python
return None
```

Not a widget, or its C++ half has already gone.

## _still_asking

### line 4416  _(unsure)_

```python
return False
```

The C++ half went with the dialog.

### line 4419  _(unsure)_

```python
return True
```

Not a widget. Nothing to close, so nothing to drop.

## _start_disk_report

### lines 4467-4470

```python
button.setToolTip(tr("Reading the disk…"))
```

Instruction 106: disabled and SAYING WHY, never inert. The button was unpressable while the report ran before this change too — the application was frozen — so keeping it unpressable while the worker reads is the same affordance, minus the freeze.

### lines 4517-4518

```python
restore()
```

Nothing was started, or `done` itself raised. A button left disabled would be the one failure the user cannot recover from.

## _start_disk_report.done

### lines 4501-4503

```python
LOG.warning("the disk could not be read", exc_info=report)
```

Same visible outcome as before this moved to a worker, where the exception left the clicked slot and no box was shown but the button is usable again and the reason is in the log.

### lines 4512-4513  _(unsure)_

```python
LOG.debug("the disk report outlived its dialog", exc_info=True)
```

The dialog was closed while the disk was being read. There is nothing left to show the report to.

## explain_every_row

### lines 4712-4722

```python
if not _widget_is_alive(label) or not _widget_is_alive(field):
```

ALIVE ON THE C++ SIDE, checked before anything is read through it. `isinstance` does NOT establish that: a Python wrapper keeps its type after Qt has deleted the object it wraps, so the check below passes and `label.text()` then reads freed memory. That is not an exception, it is a segfault, and this walk runs while a dialog is being rebuilt -- exactly when a row's widgets are being replaced underneath it.

The same shape took the whole process in

`settings_model._sibling_label_for` (52f3642b6). Found here by sweeping for it rather than by meeting it.

### lines 4729-4737

```python
is_action = isinstance(field, (QPushButton, QToolButton))
```

A BUTTON IS NOT A SETTING, and its tooltip is not a row's explanation -- it says what pressing it DOES, which is often the confirmation the press will ask for. Moving that onto the label and clearing the button leaves the user hovering the thing they are about to press and being told nothing.

So a button row gets its label explained and KEEPS its own tooltip. Every other kind of field hands its explanation over, which is the rule: on the setting's text, never on its field.

### lines 4740-4741

```python
tip = (field.toolTip() or "").strip()
```

WHATEVER THE ROW ALREADY SAID, moved rather than duplicated: a tooltip on both reads as two answers.

### lines 4747-4758

```python
explain_through_the_bar(field)
```

A BUTTON SAYS WHAT PRESSING IT DOES, AND IT SAYS IT AT

THE FOOT OF THE WINDOW. A tooltip appears over the button where the pointer already is, and where the user is about to click -- so the sentence covers the thing it describes. The bar is out of the way, holds a long sentence without hiding anything, and does not flicker as the pointer crosses a row of buttons. This is what the module tiles on Home already do.

With no bar in the window the tooltip STAYS: a control that explains itself nowhere is worse than one that explains itself awkwardly.

## PreferencesDialog.__new__

### lines 4534-4542

```python
shadowed = _settings
```

IN SAFE MODE THE DIALOG SHOWS WHAT IS STORED, and only while it is being built (296). Safe mode STARTS on defaults so that a broken value cannot stop the window appearing, but this dialog is where a value is repaired. Built on defaults it showed every control at its default, and Save writes every control, so repairing one value reset every other preference it owns -- measured 2026-09-15, a stored font scale of 1.25 came back as 1. It also made the broken value look fixed already: the Animation dropdown showed Blobs over a stored Cells, and choosing Blobs, the one on screen, would have changed nothing.

So the store the getters read is pointed at the real one for the build, the same move `_reset_to_defaults` makes towards an empty one, and put back afterwards whatever happens. Untouched controls then write back exactly what is stored. The build is synchronous, so nothing else in the process reads a preference while it runs; handlers that run later, and `apply_preferences_to_app` after Save, still read defaults, so the running safe session does not take on the stored values. Showing a value is not what kills a start: every getter returns rather than raises on garbage (115 keys x 5 kinds, measured), and the dialog builds no backdrop.

## PreferencesDialog._build_the_dialog

### lines 4816-4819

```python
from .dialogs import detach_from_window_manager
```

Detached from the parent for the window manager's purposes, so the user can put it where they like. It is still parented, still modal and still exec()s: only the window TYPE changes. See spacr.qt.dialogs for what a WM does with an attached modal dialog.

### lines 4826-4833

```python
tabs = QTabWidget()
```

One scrollable column had grown to thirty controls, which is a column nobody reads to the bottom of: Module visibility and the figure format sat below five animation sliders, and the only way to find out whether a setting existed was to scroll past everything else. The tabs are by WHAT A SETTING IS ABOUT rather than by how often it is touched — a reader looking for "how much of my machine does this use" has one place to go, and so does a reader looking for "why is the text so small".

### lines 4859-4861

```python
form = _page("General", "PreferencesTabGeneral")
```

General first because Language is in it, and a reader who cannot read the interface has to be able to find that one without understanding any of the others.

### lines 4863-4889

```python
appearance = _page("Appearance", "PreferencesTabAppearance")
```

THREE TABS WHERE APPEARANCE USED TO BE ONE, split on 2026-09-03: "in preferences the appearence tab has to much information in it. please divide the settings in it among the new tabs Appearence, Theme, Annimation."

It had grown to seventeen rows -- which is the same complaint that produced the tabs in the first place, one level down. The split is by WHAT A SETTING IS ABOUT, on the same principle as the note above:

Appearance  chrome and layout: where a tooltip goes, how a setting is laid out, which weight the interface font is drawn at. Nothing here has a colour or moves. Theme       the theme itself, and how its surfaces render: the page's opacity, the field fade, and the six rim controls. The rim is the accent light around a card, so it belongs with the palette that colours it rather than with the ambient animations. Animation   anything that moves on its own: the ambient theme and its palette, the setting animations, and the backdrop behind a settings popup.

The Theme PICKER moved here out of General with them. A tab called Theme that does not contain the theme is the kind of thing a reader looks for twice; Language stays in General, because somebody who cannot read the interface has to find that one without understanding any other tab's name.

### lines 4899-4909

```python
log_level_toggles = {}
```

Two independent switches per level rather than one severity threshold. A threshold cannot express "record DEBUG but not INFO", which is the shape of most triage: one chatty subsystem is wanted and the routine progress chatter is not.

The file column gates the console column. Both are fed by the same records, so a line shown in the console but absent from the log file would be a line the user can see and then cannot produce when asked for the log -- the console switch is disabled whenever its file switch is off, and unticking a file switch takes its console switch with it.

### line 4960  _(unsure)_

```python
language_combo = QComboBox()
```

Language is first so it remains discoverable even on a small screen.

### lines 4987-4991

```python
from .widgets.ambient import palette_label, palettes_for, theme_label
```

Animated background — the drifting shapes behind every module page. The first entry is None, and it is an animation choice rather than a separate switch: a user who finds the motion distracting reads one row, not a checkbox and a dropdown that can disagree with each other. Applied on Save, without a restart.

### lines 4997-5000

```python
from .widgets.ambient import AMBIENT_THEMES
```

A build (or a test double) whose ambient module predates the

None entry still gets a working dialog with its six animations, rather than a Preferences window that will not open because of a decorative setting.

### lines 5006-5008

```python
from .widgets.ambient import animation_note
```

Purely descriptive, and resolved through the same import as everything else above so the dialog cannot end up reading two different ambient modules in one function.

### lines 5073-5075

```python
ambient_dir_combo = QComboBox()
```

Which way the starfield goes. Only meaningful for that one animation, so it is shown only when that animation is chosen rather than sitting greyed out under five others.

### lines 5107-5114

```python
(blur_lo, blur_hi) = _ambient_ranges()[0][0]
```

The shape-of-the-motion controls, beside the animation they shape. Each is a percentage of what the chosen animation already does, so 100 % is the designed look in every theme and every one of them starts there — except blur, whose designed value is 0 %, because the animation ships unsoftened and the softening is what this one adds. Percentages rather than pixels or seconds because "40 px" means nothing to a starfield and "6 seconds" means nothing to a blob.

### lines 5234-5237

```python
tooltips_box_check = Toggle(tr("Tooltips box"))
```

THE TWO TOOLTIP SURFACES, instruction 371. Two switches rather than one three-way control, because the request is explicit that both on, both off, and either alone are all legal -- and "both off" is a choice a user is allowed to make, not a state to be prevented.

### lines 5300-5302

```python
spinner_slider = QSlider(Qt.Horizontal)
```

How long work has to run before the busy indicator appears. Seconds, not a percentage: this one is a real duration and the reader is entitled to see it as one.

### line 5356  _(unsure)_

```python
dock_combo = QComboBox()
```

The left dock — a permanent column, or gone.

### line 5376  _(unsure)_

```python
opacity_slider = QSlider(Qt.Horizontal)
```

Page opacity — shared by Home and every module surface.

### lines 5378-5381

```python
opacity_slider.setObjectName("PaneOpacity")
```

Named because (0, 100) stopped identifying it: the spinner-delay slider is (0, SPINNER_DELAY_MAX * 10) = (0, 100) too, and anything picking this control out of findChildren by its range silently got that one instead.

### lines 5427-5428

```python
field_fade_check = Toggle(tr("Fade fields towards the right"))
```

The one surface Page opacity does not reach, and why it sits directly under the slider: this is the exception to the row above.

### lines 5441-5445

```python
rim_length_slider = QSlider(Qt.Horizontal)
```

The travelling rim

THREE SETTINGS, NOT THREE CONSTANTS. How long the light is, how hard it chases and whether it sits centred on the pointer are matters of taste, and taste is what a preference is for.

### lines 5473-5474

```python
rim_lag_slider.setRange(int(RIM_LAG_RANGE[0] * 100),
```

Stored as a fraction; shown as a percentage, because a slider from 0.02 to 1.0 is a slider with no readable numbers on it.

### lines 5528-5529

```python
rim_period_slider.setRange(int(RIM_PERIOD_RANGE[0] * 10),
```

Stored in seconds, shown in tenths, because a slider from 0.4 to 12.0 has no readable integer positions.

### line 5567  _(unsure)_

```python
cb_combo = QComboBox()
```

Colour-blind mode

### lines 5582-5585

```python
verbose_check = Toggle(tr("Enable verbose logging"))
```

Verbose logging — one toggle, wired at Save time. When on, spaCR's loggers go to DEBUG and `cellpose` to INFO, and the log files keep DEBUG. torch, PIL and matplotlib are left alone. The console still shows only the levels switched on for it on the Logging tab. Aimed at bug reports. Corrected 2026-09-19: this note used to say every record echoes into the ConsolePanel and that torch, PIL and matplotlib are dialled up, and the code does neither.

THE DEBUG FILE SWITCH FOLLOWS THIS TOGGLE (2026-09-19, item 294). While verbose is ticked, the Logging tab's DEBUG "Log file" switch is held on and disabled, because verbose writes DEBUG whatever it says. Save stores the user's own DEBUG choice, remembered from before verbose held it, and never the held value. Unticking verbose hands the switch back with that choice.

Before this, the tab was built from `get_log_file_levels()`, which adds DEBUG while verbose is on, and Save wrote every switch back through `set_log_levels`. With verbose on by default, that meant:

    fresh store, open Preferences, untick verbose, Save
    stored log_file_levels   "DEBUG,INFO,WARNING,ERROR,CRITICAL"
    get_verbose_logging()    False
    spacr, spacr.io          still DEBUG

So the switch could not be turned off from its own default, and every Save made with verbose on wrote DEBUG into the user's stored levels, where it could no longer be told apart from a DEBUG the user chose. `get_log_file_levels`' docstring had promised both that DEBUG is not stored and that it "goes away again when they turn verbose off".

`set_log_levels` clamps the console against the levels the files actually keep, verbose's DEBUG included, and applies those to the live handlers. Without that, a console DEBUG switch ticked while verbose is on would be dropped at Save, because the stored file levels no longer carry DEBUG.

### lines 5637-5640

```python
db_edit_check = Toggle(tr("Allow editing in the Database Browser"))
```

Database Browser — off by default. The browser opens measurements.db with mode=ro; this is the only switch that lets it open a read-write connection at all, and even then the user has to arm edit mode per session and confirm it.

### lines 5652-5661

```python
ai_provider_combo = QComboBox()
```

THE PROVIDER PICKER LIVES HERE NOW, not on the actions row of every module. It was a chevron beside the AI switch on each screen, which put a preference -- "which assistant do I use" -- in the place where per-run choices are made, and repeated it on every module. It has one answer for the whole application, which is what a preference is.

`get_preferred_provider` already existed and was READ BY NOTHING: the slot was here the whole time and the chevron wrote to the console instead, so a provider chosen on one screen was forgotten by the next.

### lines 5712-5714

```python
alpha_check = Toggle(tr("Show Alpha modules and settings"))
```

Module visibility. Both are opt-out: existing users and fresh installs continue to see every feature until they choose a quieter, stable-only interface.

### lines 5736-5740

```python
figure_save_mode_combo = QComboBox()
```

The appearance shared by every saved-figure renderer. This differs from Figure format below: format governs the app's Figures panel; save mode governs the page and figure-element colours used when a figure is written. `spacr.figure_style.figure_save_mode` is the one production resolver and gives the environment override precedence.

### lines 5761-5769

```python
fig_format_combo = QComboBox()
```

Figures — display format (png = lighter / faster, pdf = vector editable via the figure-settings button) and the PNG resolution.

Both tooltips say plainly what these two settings reach, because their labels invite a bigger reading than the truth. They govern the figures the app renders into the Figures panel; the figures a pipeline saves into its own results directory are written by ``savefig`` calls inside spacr.plot / spacr.submodules / spacr.ml, which choose their own format and DPI and never read preferences.

### lines 5786-5791

```python
from ..graph_types import (DATA_SHAPES, GRAPH_NAMES, DEFAULTS,
```

WHICH GRAPH IS DRAWN FIRST. Asked for 2026-08-28. One row per data SHAPE and not a single control, because "Bar" is not an answer for two continuous axes -- a bar needs groups to summarise and there are none -- so one setting would be ignored by most graphs and look broken. `graph_types.WHY_NOT` says why for each pair, and only the types that FIT a shape are offered here.

### lines 5836-5839

```python
live_cache_spin = QSpinBox()
```

How many figures stay EDITABLE, and what happens to the rest. The panel keeps a live matplotlib Figure for the most recent N: those can be restyled, because they still have a legend to toggle and axes to rescale. Older ones keep only their rendered page.

### lines 5852-5855

```python
montage_columns_spin = QSpinBox()
```

CELLS PER ROW IN A MONTAGE, decided rather than measured. The well tab used to divide the panel width by a fixed cell size, so the montage changed shape whenever the window did and two wells looked at side by side were not laid out the same way.

### lines 5880-5892

```python
from .widgets.figure_settings import FigureStylePreferences
```

HOW THE GRAPHS LOOK (instruction 118)

Everything above this line is about the FILE: its format, its resolution, how many stay editable. Nothing above it is about how a plot LOOKS -- no font, no palette, no marker size, no grid default, and nothing specific to any one kind of graph -- which is what "the graphs look pretty ugly" means in practice: every plot inherited matplotlib's defaults.

The panel builds itself from `spacr.figure_style`'s own tables, so a style key added there gains a control here without this file being touched. It stores DELTAS only; see its docstring for why that is not an optimisation.

### lines 5904-5914

```python
mode_combo = QComboBox()
```

Performance

The mode, then the four things the two performance modes press on your behalf. They are in the same tab deliberately: a mode that says "cleanup runs at launch" should be read next to the buttons that say exactly what a cleanup is, or "cleanup" is a word the user has to take on trust. ONE SELECTOR, FIVE LEVELS, ordered by how much of the machine spaCR keeps for itself. Laptop used to be a second control that quietly overrode this one, so a user could choose a posture on one row and have another row undo it -- two answers to one question. It is the most constrained end of the same scale.

### lines 5954-5963

```python
from .memory_budget import (DEFAULT_CACHE_CEILING_MB,
```

THE MEMORY BUDGET, under the level it takes its suggestions from. Asked for 2026-08-27; the headroom floor comes FIRST because the other two say what may be kept and this says when keeping it stops being acceptable.

NOTHING HERE CLAIMS TO UNLOAD A LIBRARY. Measured: importing torch costs 477 MB and deleting every torch entry from sys.modules returns none of it, because CPython has never supported unloading a C extension. What can be returned is caches, weights and GPU allocations, so that is what these are named for.

### lines 6028-6030

```python
font_weight = QComboBox()
```

NO SEPARATE LAPTOP CONTROL. It is the first entry of the selector above, which is the whole point of 286: one value, not two that can disagree.

### lines 6039-6041

```python
if spaceout_enabled():
```

THE SPACEOUT FRACTAL, and ONLY under spaceout. An ordinary launch builds none of these rows, so the hidden mode stays hidden: a settings page advertising it would be the giveaway.

### lines 6054-6055

```python
fractal.addRow(tr("Pattern"), fractal_pattern)
```

FIRST in the tab: it decides which fractal the rows below it are describing, and the two have different costs.

### lines 6178-6181

```python
fractal_speed_min = None
```

SLOWEST AND FASTEST ARE GONE FROM THE PANEL. Variable speed breathes around the Speed above it; two more fields to say how far is three controls answering one question, and the three could be set to contradict each other.

### lines 6185-6188

```python
fractal_ss = _whole(
```

SUPERSAMPLING, called out as "a super important setting". It is the one that decides whether the picture is smooth or aliased, and it costs its own square: 2 is four samples a pixel, 3 is nine.

### lines 6199-6221

```python
fractal_path = QComboBox()
```

THE MANDELBROT RENDERER'S OWN SETTINGS. They mean nothing for the other three patterns and are shown regardless, because a row that appears and disappears as the pattern changes is the form rearranging itself under the reader -- and these are all numbers a user may want to set BEFORE choosing the pattern. SUB-CATEGORIES, asked for 2026-08-28. Twenty-one fields in one column is a wall; three headings say which question each group answers, and a reader looking for the zoom does not have to read the steering to find it. ONE CONTROL PER QUESTION, and the same questions whichever pattern is chosen. Asked for 2026-08-28: "there need to be fewer options for speed so one option for speed that is user facing, one option for steering and so on... mak the settings easy to navigate aross themes."

Speed is already one field above. Steering is one here, and it DERIVES the three numbers it stands for -- set by hand they contradict each other, and a short interval with a long duration is the jerkiness that was reported. THE PATH, and Steering only means anything when it is guided. Fixed is the default because the search moves the camera and that is what shook -- smoothing the motion does not remove the fact that it is being moved.

### lines 6226-6229

```python
("tour", "Tour the interesting places")):
```

THE TWENTY REGIONS, REACHABLE (327). `RegionTour` and `fractal_regions` were built and tested with no caller and no door; this is the door.

### lines 6263-6267

```python
fractal_depth = _tenths(
```

DEPTH, asked for by name on 2026-08-28: "i want controll over the decades". It was a setting all along and was taken off the panel in the cut-down, which is the same mistake as hiding the numbers behind Advanced -- a control somebody asks for and cannot find is not a control.

### lines 6300-6311

```python
fractal_mandel = {}
```

THE DETAIL IS NOT ON THE PANEL. Asked for 2026-08-28:

"there need to be fewer options... one option for speed that is user facing, one option for steering and so on."

The twelve numbers behind Speed, Steering and Quality are still settings -- the renderer reads them and a settings file can carry them -- but they are DERIVED from the controls above, and offering both is what let a hand-set combination contradict itself: strength 0 with an interval of 0.01 and a duration of 0.1 moved the camera "every second in a random direction". A panel that offers a number and a control that overwrite each other is not a choice, it is a trap.

### lines 6314-6315

```python
fractal_pointer = Toggle(tr("Mouse gravity"))
```

MOUSE GRAVITY. Asked for 2026-08-28, with a size and a strength, applying to every pattern and both backends.

### lines 6328-6347

```python
fractal_pointer_size = _tenths(
```

SIZE IS OFFERED, STRENGTH IS NOT, asked for 2026-09-08: "id like a setting that controlls the size of the gravity ball".

The note this replaces said "ONE MOUSE CONTROL, not three. Size and strength answer the same question -- how much does it pull -- and offering both let a size of 3 fight a strength of 0." The second half of that is still true and is why STRENGTH stays off the panel: two controls over one feeling is what produced the setting that cancelled itself. The first half was too strong. Size is not strength. It is the REACH -- how far from the cursor the pattern feels the pull at all -- and it is the one a user can see themselves changing.

1.0 is the widget's short edge, which is why the range runs to 3.0 rather than to some rounder number: past about three short edges the whole backdrop is inside the ball and there is nothing left for it to reach toward. `_tenths` is a QDoubleSpinBox holding the value itself -- the name is the helper's, not a unit. Scale, Speed and Dream above are saved straight from `.value()` and so is this.

### lines 6363-6365

```python
fractal_speed_period = None
```

HOW OFTEN IT BREATHES IS NOT A QUESTION ANYBODY ASKED. Variable speed uses one sensible period; a field for it was a third control on the same question as Speed.

### lines 6368-6371

```python
_sync_mode_note()
```

NOTHING TO GREY ANY MORE. The three bounds this used to enable and disable are not on the panel: variable speed breathes around Speed by a fixed proportion, so there is one control and nothing that can disagree with it.

### lines 6443-6447

```python
hash_check = Toggle(tr("Hash inputs for the run manifest"))
```

Hashing is a COST setting, which is why it lives here beside the four that free resources rather than under Appearance. It is off by default: hashing every file under every path-valued setting is proportional to the data, not the run, and on a plate of raw images it is minutes of reading before the first mask is made.

### lines 6460-6463

```python
workspace_combo = QComboBox()
```

Instruction 180. On this tab and beside the hashing switch because it is the same kind of decision -- how much of the machine a saved run is allowed to use -- and the two are read together: a user who wants a run they can hand to somebody else wants both.

### lines 6518-6526

```python
quit_button = QPushButton(tr("Quit spaCR…"))
```

Quitting belongs on this tab and not with Save/Cancel: it is the last of the "this machine is not behaving" tools, next to the four that free what a wedged run is holding. It is the one to reach for when freeing memory was not enough.

Deliberately NOT wired to `dlg.accept()` first. A user reaching for this has a window that will not close; making them save preferences on the way out would be one more thing between them and leaving.

### lines 6540-6545

```python
buttons = QDialogButtonBox(
```

NO STANDING SENTENCES UNDER THE TABS. Two of them sat here on every visit -- what applies instantly on Save, and when colour-blind mode reaches a figure -- and a paragraph that is always true of the whole dialog is not read after the first time. Whatever a particular control does belongs to that control, and the hint bar below says it on hover.

### lines 6557-6560

```python
reset_button = buttons.addButton(
```

`ResetRole` is what puts it on the LEFT, away from Save and

Cancel: every Qt style groups the destructive-ish button apart from the two that close the dialog, which is what stops it being clicked by muscle memory aimed at Cancel.

### lines 6832-6836

```python
from .widgets.hint_bar import HintBar
```

A LINE AT THE FOOT, THE WAY THE HOME SCREEN DOES IT. Added before the rows are explained, because `explain_every_row` hands an action button's sentence to whatever bar its window has -- so the bar must exist by then or the button keeps a tooltip nobody asked for.

### lines 6839-6843

```python
layout = dlg.layout()
```

ABOVE THE BUTTONS, not under them. Asked for 2026-08-28. Appending put the explanation below Defaults/Close/Open, which reads as a footnote to the buttons rather than as the answer to the control the pointer is on -- and puts it furthest from the tabs it describes.

### lines 6850-6852

```python
explain_every_row(dlg)
```

EVERY ROW EXPLAINED, ON ITS LABEL. Done here, over the finished dialog, so a row added anywhere above is covered without the author having to remember the rule.

## PreferencesDialog._build_the_dialog._reload_ambient_palettes

### lines 5038-5041

```python
valid = () if theme_key == NO_ANIMATION else palettes_for(theme_key)
```

None has no palette to offer and no engine to ask, so the list is emptied rather than filled with the last theme's colours — an enabled-looking picker for a backdrop that is not being drawn is a control that lies.

### lines 5056-5058

```python
ambient_theme_combo.setToolTip(
```

Say what the selected animation actually looks like — the names alone ("Ripples", "Starfield") do not tell a user what they are about to put behind their work.

## PreferencesDialog._build_the_dialog._percent_row._update

### lines 5136-5137

```python
"""Show the percentage, saying when it is the designed value.
```

Say when it is the designed value, because "100%" alone does not tell a reader that it is the one to come back to.

## PreferencesDialog._build_the_dialog._percent_row

### lines 5151-5156

```python
(animation if target is None else target).addRow(
```

THE ANIMATION TAB BY DEFAULT. It was `appearance` until the split on 2026-09-03, and these five are what shape the ambient animation chosen two rows above them -- detail, blur, speed, size and density. Leaving them behind would have put five rows named "Animation ..." on a tab that no longer has the animation on it, which is the confusion the split was for.

## PreferencesDialog._build_the_dialog._sync_mode_note

### lines 5937-5940

```python
said = PERFORMANCE_NOTES.get(key) or mode_note(
```

EACH LEVEL SAYS WHICH HARDWARE IT IS FOR. A selector whose entries are five adjectives asks the user to guess; the note states the memory profile, what is retained and what that costs.

### lines 5947-5948

```python
text = f"{text}\n\n⚠ {tr(warning)}"
```

Warn on SELECTION, not on Save: a warning that arrives after the dialog has closed is a report, not a choice.

## PreferencesDialog._build_the_dialog._whole

### lines 6148-6153

```python
"""A whole-number field, equally uncapped."""
```

NO `suffix` PARAMETER. It had one, defaulting to "", and the single caller below never passed it -- so the `setSuffix` it guarded could not run, and coverage counted two items nothing could reach. A parameter with no caller is not extensibility, it is a branch that cannot be tested and a reader wondering what uses it.

## PreferencesDialog._build_the_dialog._quit_spacr

### lines 6413-6415

```python
watcher.start()
```

Parented to the window, not to the dialog: the dialog is about to close and a timer that dies with it would ask nothing.

## PreferencesDialog._build_the_dialog._resource_button

### lines 6422-6425

```python
def _resource_button(action, label_text, row_label):
```

The four buttons. Each one is confirmed by a dialog that NAMES what will happen — "are you sure?" is not something a user can consent to — and each reports what was actually freed, measured before and after, including when that is nothing.

### lines 6434-6437

```python
button.setToolTip(resource_cleanup.summary_text(action))
```

THE SHORT FORM ON HOVER. The confirmation still shows the full bulleted promise when the button is pressed; a hint bar that grew to eight lines made the dialog jump as the pointer moved between two buttons.

## PreferencesDialog._build_the_dialog._workspace_copying

### line 6505

```python
workspace_limit.setToolTip(workspace_limit.toolTip() if copying else tr(
```

Instruction 106: disabled and SAYING WHY, never inert.

## PreferencesDialog._build_the_dialog._reset_to_defaults

### lines 6615-6617

```python
style_panel.reset()
```

Told directly rather than re-read: this panel holds its controls, not its store, so the throwaway-settings trick every getter above uses does not reach it.

## PreferencesDialog._build_the_dialog._save

### lines 6651-6652

```python
"""Write every preference this dialog owns, rim first.
```

The rim first: every open card rereads these, and doing it before the theme work means one repaint rather than two.

### lines 6667-6670

```python
set_ambient_animation(ambient_theme_combo.currentData())
```

One write for the whole Animation row: it stores the choice, repairs a palette the new animation cannot draw, and turns the backdrop off for None (which is what makes "no timer" true — every install site reads `get_ambient_enabled` first).

### lines 6674-6678

```python
set_ambient_palette(palette_choice)
```

An animation that offers no palette leaves the combo empty. The theme write above already stored a usable value, so there is nothing to save here — and a decorative background must never be the reason the whole Preferences dialog refuses to close.

### lines 6695-6699

```python
_tell_the_screens_the_object_grid_changed()
```

AND TELL THE SCREENS THAT ARE ALREADY OPEN. The switch used to be read only while a settings panel was being built, so it did nothing at all until the module was closed and reopened -- in both directions, which is what makes a switch look broken rather than slow.

### lines 6706-6709

```python
set_workspace_copy_limit_mb(workspace_limit.value())
```

The limit FIRST: `set_save_workspace` pushes both down to spacr.workspace together, so setting the mode against a stale limit would leave the journal copying to the old ceiling until something else happened to push again.

### lines 6716-6718

```python
set_log_levels(
```

set_log_levels re-clamps rather than trusting the dialog: the console switch is disabled when its file switch is off, but a disabled QCheckBox still reports whatever it was last set to.

### lines 6731-6734

```python
set_default_graph_type(shape, combo.currentData() or "")
```

Empty data is the "Recommended" row, which CLEARS the saved choice rather than storing today's table: a stored copy of a default is a preference that has stopped tracking the package.

### lines 6743-6747

```python
if spaceout_enabled():
```

LAST of the writes, and deliberately: entering Extra

Performance overrides five of the settings written above with their minimums, and leaving it puts back what it stashed. Do it earlier and the dialog's own values would land on top, which would mean the mode silently did not take effect.

### lines 6757-6758

```python
speed_min=fractal_speed.value() * 0.55,
```

DERIVED FROM SPEED, not set beside it: variable speed breathes by a fixed proportion either way.

### lines 6762-6767

```python
pointer_size=(fractal_pointer_size.value()
```

THE REACH IS THE USER'S NOW, so it is read from the field rather than pinned at 1.0. Strength is still derived: off is a strength of zero, which stops the pull whatever reach is stored, and there is no control for it because two knobs over one feeling is what let a size of 3 fight a strength of 0.

### lines 6780-6782

```python
complaints = [
```

A NUMBER THAT CANNOT BE USED IS SAID SO, in words, rather than silently reduced. The fields take anything; this is where the tool answers.

### lines 6793-6797

```python
try:
```

ALWAYS BACK TO THE SURFACE. A dive that resumed where it was would apply the new numbers thirty decades down, where a changed starting scale or iteration count has nothing recognisable to act on -- so the change would look as though it had done nothing.

### lines 6802-6806

```python
apply_saved_controls()
```

THE NEW NUMBERS REACH THE RUNNING BACKDROP. It keeps the controls it was built with, so writing them to the store and restarting the dive left the old speed in place -- which is why changing Speed appeared to do nothing at all.

### lines 6819-6821

```python
set_performance_level(mode_combo.currentData())
```

ONE VALUE. `set_performance_level` mirrors the level into the three-mode posture the cleanup code speaks in, so there is nothing else to write and nothing that can disagree.

## _everything_explains_itself_in_the_strip

### lines 6888-6889

```python
if bar.explain(widget):
```

An empty `text` takes the widget's own tooltip and clears it, so the sentence MOVES rather than being said in two places.

### lines 6893-6894

```python
continue
```

Help that will not move is a blemish, never a reason for

Preferences not to open.

## get_issue_prompt_mode

### changed 2026-09-19, after review (whose 'ask' is it?)

```python
_KEY_ISSUE_PROMPT_CHOSEN = "ai/issue_prompt_chosen"
```

"'always' is the default for anyone who has not chosen" was true only of a profile with nothing stored, and almost no installed profile is in that state. `SetupSlides.accept()` AND `reject()` both call `setup_screen.apply(self.answers())`, and `issue_prompt` has been one of those answers since 6c57da8d6 (2026-08-21, shipped in 1.5.0.5 through 1.5.0.8). So every user who so much as opened first-run setup -- including one who dismissed it at the first slide -- has `'ask'` written into their profile by the default of the day, and reading that back as "an explicit earlier choice" left the maintainer's decision reaching new profiles only. The reporter of issue #117 was on 1.5.0.8 and would still have filed nothing after upgrading.

So `set_issue_prompt_mode` now writes a marker beside the value, and a stored `'ask'` WITHOUT that marker reads as the current default. `'never'` and `'always'` are returned as they stand whether marked or not: the superseded default was `'ask'`, so neither of those was ever written on a user's behalf. Every writer -- the setup slides, the Preferences dialog, the AI Console and the installer's consent page -- goes through the setter, so from here on an 'ask' in the store is an answer somebody gave.

Nothing is filed on the strength of this alone. The terms go from 4.1 to 4.2 in the same change, so the profile is asked again, `AppScreen._the_terms_allow_automatic_filing` files nothing until 4.2 is accepted, and the slide that carries the setting is on the page the user accepts them from -- showing 'always', with the switch to change it, before any run can fail.

## 2026-09-19 — `sound/music_file` (item 427, part B)

A WAV of the user's own for the music bed to play instead of the
synthesized one, and, because there is only ever one thing playing, the
thing the Resonance backdrop is driven by too. Empty by default.

`get_sound_music_file` deliberately does NOT check that the file is there.
It is read on the GUI thread on every settings read, and a `stat` on a
network home directory is exactly the stall `spacr/qt/path_probe.py`
exists for. `spacr.qt.sound` looks for the file on its audio thread, and a
chosen file that has gone falls through to spaCR's own bed rather than to
silence.
