# Notes from `spacr/qt/widgets/home.py`

Prose lifted out of `spacr/qt/widgets/home.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_fmt_elapsed](#_fmt_elapsed) (1 entry)
- [AppTile.__init__](#apptile__init__) (3 entries)
- [Panel.__init__](#panel__init__) (4 entries)
- [RunningBanner.__init__](#runningbanner__init__) (2 entries)
- [RunningBanner._on_quit](#runningbanner_on_quit) (1 entry)
- [RunningBanner._terminate](#runningbanner_terminate) (2 entries)
- [RunningBanner.refresh](#runningbannerrefresh) (2 entries)
- [QueuedPanel.refresh](#queuedpanelrefresh) (1 entry)
- [RecentRunsPanel._registry](#recentrunspanel_registry) (1 entry)
- [RecentRunsPanel.read](#recentrunspanelread) (2 entries)
- [SystemPanel.__init__](#systempanel__init__) (1 entry)
- [_nvml](#_nvml) (1 entry)
- [TotalsPanel._totals_since](#totalspanel_totals_since) (1 entry)
- [StageLegend.__init__](#stagelegend__init__) (1 entry)
- [StageLegend._legend_row](#stagelegend_legend_row) (1 entry)
- [NewsPanel.__init__](#newspanel__init__) (3 entries)
- [NewsPanel._release_block](#newspanel_release_block) (2 entries)
- [NewsPanel.render_body](#newspanelrender_body) (1 entry)
- [HomePage.__init__](#homepage__init__) (6 entries)
- [HomePage.page_fill](#homepagepage_fill) (1 entry)
- [HomePage._install_ambient](#homepage_install_ambient) (3 entries)
- [HomePage._discard_ambient](#homepage_discard_ambient) (1 entry)
- [HomePage._clear_page_surfaces](#homepage_clear_page_surfaces) (6 entries)
- [HomePage._build_hero](#homepage_build_hero) (9 entries)
- [HomePage._build_tabs](#homepage_build_tabs) (3 entries)
- [HomePage._build_home_tab](#homepage_build_home_tab) (2 entries)
- [HomePage._build_category_tab](#homepage_build_category_tab) (3 entries)
- [HomePage._make_tile](#homepage_make_tile) (1 entry)
- [HomePage._scrolled](#homepage_scrolled) (1 entry)
- [HomePage._fill_grid](#homepage_fill_grid) (2 entries)
- [HomePage._columns_for](#homepage_columns_for) (1 entry)
- [HomePage._build_aside](#homepage_build_aside) (2 entries)
- [HomePage._version](#homepage_version) (1 entry)
- [HomePage._on_runs_changed](#homepage_on_runs_changed) (1 entry)
- [HomePage.set_reserved_content](#homepageset_reserved_content) (1 entry)
- [HomePage.eventFilter](#homepageeventfilter) (2 entries)

## _fmt_elapsed

### lines 154-162

```python
def _fmt_elapsed(seconds: float) -> str:
```

`elide_to_lines` used to live here. It shortened a tile's one-line description until it wrapped into at most three lines, because a word-wrapped QLabel in a fixed-height box does not elide — it just stops painting. #16j removed the descriptions from the tiles, so there is no fixed-height wrapped label left on this page and nothing to shorten. The identical helper in `spacr/resources/home/versions/_generators/parts.py` is still live: that renders the archived home-screen candidates, several of which do carry a blurb.

## AppTile.__init__

### lines 254-256

```python
from ..theme import STAGE_LABEL
```

The stage goes in the accessible description as a WORD, not only as a colour: a legend keyed on hue is no legend at all to a screen reader, and colour alone fails WCAG 1.4.1.

### lines 262-272

```python
col = QVBoxLayout(self)
```

NO TOOLTIP on the tile. The description already appears in the hint bar at the bottom of the Home screen, updated by HomePage's eventFilter on the same hover -- so a popup was a second copy of the same sentence, drawn ON TOP of the grid the user is reading to choose between. These blurbs run to several hundred characters, which is fine in a fixed line the eye can skip and wrong in a box covering the tiles.

The accessible name and description above are set independently and are what a screen reader reads, so removing the tooltip costs no assistive text.

### lines 289-304

```python
name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
```

`Ignored` horizontally, and added with NO alignment flag, so the layout hands it the tile's whole content width. Both halves matter:

``setFixedWidth`` — the obvious way to write this, and what this line used to be — leaves the label at that width with ``WA_Resized`` still FALSE (``setMaximumSize`` restores the flag after the resize it does internally). ``ElidingLabel`` deliberately does not elide before its first real layout pass, so a fixed-width label never elides at all: a long name is painted and cut off, which is the exact bug this widget family exists to prevent. an alignment flag would make QGridLayout/QBoxLayout give the item only its ``sizeHint`` — which for an ElidingLabel is the width of the FULL text — and the tile would be dragged wider by its longest name.

## Panel.__init__

### lines 407-411

```python
head = QWidget()
```

THE CAPTION IS A ROW, not a label, because the panels carry an action word on the right of their heading. Kept out of the box below so the word sits on the page beside the caption rather than inside the panel's own frame, which is where a reader looks for a control that acts on the whole panel.

### lines 435-440

```python
from ..theme import pane_surface
```

The rounded box KEEPS its dark-grey fill — at the page opacity. `active_palette()` returns raw hex, which is why the preference never reached these panels; `pane_surface` reads it. Making the box itself transparent (tried, reverted) left nothing but a floating outline: the fill is what makes it read as a panel. What has to go is the CONTAINER behind it, which is handled in `_clear_page_surfaces`.

### lines 453-457

```python
make_transparent(self)
```

The Panel wrapper positions a header and the box; it paints nothing. Untagged it takes the blanket `QWidget { background-color: bg }` rule, and six of these stacked down the aside read as one large black column behind every panel — which is exactly what they looked like.

### lines 459-462

```python
make_transparent(self._head)
```

The caption ROW too. Untagged it takes the blanket

`QWidget { background-color: bg }` rule and draws a black strip above every panel, which is the mistake the note above records for the wrapper.

## RunningBanner.__init__

### lines 659-663

```python
self._btn_pause.setStyleSheet(
```

The app stylesheet's `QPushButton:disabled` rule loses to its own `QPushButton#GhostButton` rule (an ID selector outranks a pseudo-state), so a disabled ghost button renders identically to a live one. This scoped rule puts the difference back — a control that cannot be used has to *look* like it.

### lines 672-675

```python
self._btn_quit = QPushButton("Quit")
```

Quit, in red, because a run that will not stop is the state this banner is most often being read in. `Pause` asks the gate and `Open` navigates; this is the only control here that can end a job whose worker has stopped checking whether it should.

## RunningBanner._on_quit

### lines 715-717

```python
self._quit_watcher = GracefulQuitWatcher(
```

Keep the watcher on the banner rather than on the handle: the handle is retired the moment the job stops, and a timer parented to a dead object is a crash rather than a missed prompt.

## RunningBanner._terminate

### lines 747-752

```python
try:
```

PARK a thread that will not stop; never terminate it. terminate() is pthread_cancel and every thread here runs Python -- cancelled holding the GIL it freezes the whole process, which is the opposite of what a force-quit button is for. drain_thread keeps a reference to a stubborn thread so nothing drops a running QThread, and returns immediately either way.

### lines 758-759  _(unsure)_

```python
pass
```

Already gone: the job finished between the prompt and here, which is the good outcome and not an error.

## RunningBanner.refresh

### line 804, trailing  _(unsure)_

```python
self._bar.setRange(0, 0)
```

indeterminate

### lines 813-815

```python
tail = handle.last_line
```

The last line is the *other* thing the job said. When it is the progress line the count above already came from, repeating it is noise ("41 of 96 · Progress: 41/96, operation_type: …").

## QueuedPanel.refresh

### lines 947-950

```python
from ..i18n import tr
```

THE COUNT IS A VALUE, NOT PART OF THE KEY. Composing "+3 more" first asks the catalog for a phrase that changes with the queue depth, so no row could ever match it; tr() formats the count into the translation instead.

## RecentRunsPanel._registry

### lines 1013-1014  _(unsure)_

```python
return None
```

Home is mid-construction, or the registry is gone. Filtering nothing is the safe answer: it is what the panel did before.

## RecentRunsPanel.read

### lines 1070-1073

```python
entries = recent_runs(limit=max(self._limit * 8, 32))
```

A MARGIN, because the filter below can drop most of what comes back. Bounded rather than unlimited: the whole point of `limit` in `recent_runs` is that the cost must not grow with the size of the journal.

### lines 1085-1089

```python
continue
```

ISO-8601 UTC strings compare lexicographically in time order, which is the whole reason the watermark is stored as one. An entry with NO start time is kept: it is a real run whose manifest is incomplete, and hiding it would be guessing that it is old.

## SystemPanel.__init__

### lines 1175-1176

```python
self._refresh = self.add_action(
```

BLUE, not red: it takes nothing away. Asked for on 2026-09-03 "should have a refresh button (like clear button but blue)".

## _nvml

### line 1272  _(unsure)_

```python
_NVML = None
```

No NVIDIA driver, no package, or a driver too old to init.

## TotalsPanel._totals_since

### lines 1353-1355

```python
for entry in recent_runs(limit=None):
```

`limit=None`, NOT a negative number. `recent_runs` truncates with `all_entries[:limit]`, so -1 quietly drops the OLDEST entry measured: 11,027 runs come back for None and 11,026 for -1.

## StageLegend.__init__

### lines 1413-1414  _(unsure)_

```python
for stage in ("alpha", "beta", "stable"):
```

Least finished first: the row a user needs to have read before they trust a number is the one they should meet first.

## StageLegend._legend_row

### lines 1439-1445

```python
chip.setStyleSheet(
```

Filled with the hue and rimmed like the tiles, so the swatch is a small picture of the thing it stands for.

The rule is scoped to the chip's own object name on purpose:

``Panel.add`` sets an unscoped ``background: transparent`` on the row this chip lives in, and an unscoped rule on a parent is exactly what strips the fill off its children.

## NewsPanel.__init__

### lines 1531-1533

```python
"""Build the panel and its caption.
```

THE HEADING AND THE VERSION ARE SEPARATE. The catalog is keyed on "News", so composing the release into the caption first leaves the only aside panel that names a build in English.

### lines 1541-1545

```python
super().__init__(f"{heading} \u00b7 spaCR {version}" if version
```

NOT `beta=True` ANY MORE. The mark meant "this panel is a slot with nothing in it yet", which was true for as long as the body said "Reserved for featured content". It now lists every release with its notes and its links, so the mark would be labelling a finished panel as unfinished.

### lines 1572-1575

```python
f"color: {P['fg_muted']}; font-size: {font_px(11)}px;"
```

`fg_muted`, for the same reason as the date stamps below: this is the panel telling the reader there is nothing to show, not a disabled control. `fg_dim` reads 3.66:1 dark and 3.11:1 light against a 4.5:1 floor.

## NewsPanel._release_block

### lines 1631-1633

```python
title = QLabel(
```

THE TITLE IS THE LINK to the release page, so the whole record is one click from the panel even when its body carries no URL of its own -- which is true of four of the seven releases bundled today.

### lines 1648-1661

```python
f"color: {P['fg_muted']}; font-size: {font_px(10)}px;"
```

`fg_muted`, NOT `fg_dim`. A release-note date is real content that happens to be secondary -- it is not disabled and it is not a hint, which is what `fg_dim` is for. Chosen as the wrong ROLE rather than the wrong colour, and it showed up as a contrast failure: `fg_dim` reads 3.66:1 on the dark card and 3.11:1 on the light one, against the 4.5:1 floor `test_theme_blind_widgets` holds. `fg_muted` is the secondary-text role and clears it in both.

Fixing the role rather than lightening `fg_dim` keeps disabled text receding everywhere else -- that colour has 73 call sites, and making it louder to satisfy one panel would change screens nobody complained about.

## NewsPanel.render_body

### lines 1697-1698

```python
text = re.sub(
```

Bare URLs become anchors. Applied to the ESCAPED text, so the pattern cannot match inside a tag -- there are none yet.

## HomePage.__init__

### lines 1852-1854

```python
from ..job_runner import JobRunner
```

The run-journal walk behind Recent runs and Totals goes through here, so returning to Home never blocks on it. journal=False: reading the journal is not itself a run.

### lines 1856-1867

```python
self._journal_jobs = JobRunner(self, app_key="home journal",
```

`user_visible=False`: NOTHING THE USER STARTED. This walk is housekeeping -- it reads the run journal to fill Recent runs and Totals on the way back to Home -- and a visible job claims a run BANNER, which is the blue "home journal - running" box reported over the Home screen on 2026-09-03. It still turns the activity spinner, because something is genuinely running.

The same mistake the usage poller made, and `JobRunner`'s own docstring records that one: "without this Home flashes '<module> usage - running' on and off for as long as a module screen is open." A journal of 11,000 runs takes long enough for this one to sit there rather than flash.

### lines 1895-1897

```python
self._running_host = QWidget()
```

One row per active run, oldest first. Keep ``_banner`` as the first row for compatibility with integrations that predate concurrent module runs.

### lines 1915-1921

```python
from .module_hint_bar import ModuleHintBar
```

A `ModuleHintBar`, not a plain QLabel, since 2026-09-03: the strip carries an API link and a Tutorial link now and holds the last module hovered for thirty seconds, because a link that vanishes when the pointer moves toward it cannot be clicked. It sizes itself from the font for the reason the plain bar did -- a hard 32 px is a promise about text metrics that breaks the moment the font scale or the theme's font stack changes, and the hint needed 35 px.

### lines 1926-1929

```python
from .. import bridge
```

Live job state. The registry is process-wide and outlives this page (Home is rebuilt on every theme change), so the connection is made with a bound method and dropped in closeEvent — a lambda would keep a destroyed page alive as a receiver.

### lines 1946-1956

```python
self._clear_page_surfaces()
```

And unconditionally, whatever happened above — the same correction the module screens already carry. This used to run only inside the successful-install arm, on the reasoning that a page with nothing behind it should stay opaque; that reasoning is what left Home a solid `bg` slab for anyone with the ambient preference off or the Animation preference set to `none`. There is never nothing behind it: `paintEvent` paints the page, which is what these containers are supposed to be showing. `clear_container_surfaces` is idempotent, so the call inside `_install_ambient` stays where it is for its own ordering reasons and this one costs a second pass over the tree.

## HomePage.page_fill

### lines 1972-1976

```python
if (self._ambient is not None
```

A backdrop the WINDOW owns counts as "a backdrop is installed". Without this the guard that declines to build a second one would leave `_ambient` None, `page_fill` would return the flat page colour, and the screen would paint that colour straight over the window's animation -- the black slab, reported three times.

## HomePage._install_ambient

### lines 2032-2036

```python
return
```

A RETRY THAT IS NO LONGER NEEDED. The handler below comes back on a timer while the backdrop is waiting for a heavy import, and Home is also rebuilt on a theme change -- so two installs can be in flight, and the second would leave the first parented, ticking and invisible behind it.

### lines 2047-2053

```python
if not _the_heavy_import_lock_is_free():
```

NOT WHILE A HEAVY IMPORT IS RUNNING. Home is built at startup, which is exactly when the pipeline preloader is importing torch under the lock the spaceout backdrop's GL context needs. Trying anyway costs the bounded wait on the GUI thread and then fails; asking first costs a non-blocking acquire. The screen is what matters and the backdrop is decoration, so the decoration is what waits.

### lines 2069-2077

```python
try:
```

The peek above is a check, not a reservation: the preloader re-takes the lock between two imports, so a refusal can still arrive here. It means "not yet", and Home has no second chance of its own -- it is built once and rebuilt only on a theme change -- so without this a spaceout launch would lose Home's backdrop for the session. DEFENSIVELY: a missing ambient module is one of the things this handler exists to absorb, so it cannot be the thing asked to classify the failure without a guard of its own.

## HomePage._discard_ambient

### line 2112

```python
return
```

If the import is what failed, nothing was constructed.

## HomePage._clear_page_surfaces

### lines 2153-2158

```python
clear_container_surfaces(self)
```

The generic sweep FIRST. Home used to hand-list five widgets it guessed were responsible, which is why measuring found three that were not on it: the hero's own QLabels, Qt's internal `qt_tabwidget_tabbar`, and the anonymous row hosts the tiles sit in. Naming widgets one at a time cannot keep up with a layout; sweeping by rule can.

### lines 2161-2162

```python
for bar in self.findChildren(QTabBar):
```

Qt builds the tab bar itself, so it is neither anonymous nor ours to name at construction — it has to be reached through the tab widget.

### lines 2166-2169

```python
hero = self.findChild(QWidget, "Hero")
```

The hero's labels: the mark, the wordmark and the subtitle. They are type on the page, and a QLabel with no rule of its own takes the blanket window fill — which is what left a black band across the masthead after the Hero FRAME was already transparent.

### lines 2180-2182

```python
tabs = getattr(self, "_tabs", None)
```

Every scroll area, its viewport, and the stacked pages the tabs keep their tab bodies in. These are the containers behind the tile rows and their headings.

### lines 2188-2189  _(unsure)_

```python
for i in range(tabs.count()):
```

The direct page widgets of each tab: one per category, each the host for that category's rows and headings.

### lines 2194-2195  _(unsure)_

```python
layout = self.layout()
```

The body wrapper is the first child of `outer` and has no object name of its own, so it is reached through the layout.

## HomePage._build_hero

### lines 2256-2258

```python
hero.setObjectName("Hero")
```

Named so `_clear_page_surfaces` can find it and clear the fill off the labels inside it; without the name that sweep reaches nothing and the masthead's type keeps the blanket window background.

### lines 2260-2266

```python
try:
```

THE MASTHEAD IS A PANEL, NOT A BLACK BAND. The mark and the wordmark sat on the blanket window fill, which over a light theme or an animated backdrop reads as a black box drawn behind the logo. It wears the surface every other panel in the application wears -- the translucent one, at the user's own pane opacity, with the same corner radius -- so it sits ON the backdrop rather than punching a hole in it.

### lines 2271-2276

```python
hero.setStyleSheet(
```

THE SAME CALL EVERY OTHER PANEL MAKES, told which theme it is in. Left to resolve its own, it answered a different question from "what is this page painted in" and put a near-white panel behind the wordmark on a dark page. Told, it matches the rest of the application by construction: the same surface, the same scrim, and the user's own pane-opacity preference.

### lines 2284-2287

```python
import logging
```

This module has no module-level logger, and the one other place that logs imports it where it is used. Same here: a masthead without its surface is a smaller masthead, and the reason belongs in the log rather than on the screen.

### lines 2293-2294

```python
row.setContentsMargins(SPACING["md"], SPACING["sm"],
```

Padding, because the panel now has an edge: type flush against a rounded corner reads as clipped.

### lines 2299-2302

```python
logo_px = font_px(HERO_LOGO_PX)
```

The mark and the wordmark move together, so both take the Zoom multiplier: scaling only the text leaves a 52 px logo beside 78 px lettering, which is the one thing the two constants exist to prevent.

### lines 2321-2323

```python
follow_device_ratio(label, _draw_mark)
```

The masthead is the one picture in the application that is up for the whole session and never rebuilt, so it is the one that would stay soft after a move onto a denser display.

### lines 2335-2341

```python
from .loading_screen import strap_line
```

THE SAME SENTENCE THE LOADING SCREEN SHOWS, from the same place. This used to be a literal here that had drifted from it -- "single- cell measurements" against "single-cell image analysis", and "genotype-phenotype" against "genotype-to-phenotype" -- so a user saw two slightly different claims about the same product within five seconds of launching it. One string, one definition, one set of translation rows.

### lines 2348-2350

```python
return hero
```

No "All apps" button: the first tab IS all apps. The edge drawer still exists for the screens that are not Home, and is reachable there from the spaCR menu or Ctrl+Shift+A.

## HomePage._build_tabs

### lines 2371-2373

```python
self._tabs.setDocumentMode(False)
```

Keep ordinary tab geometry; the QSS deliberately suppresses the pane's decorative frame, while the selected tab keeps its own meaningful indicator.

### lines 2377-2381

```python
from PySide6.QtWidgets import QStackedWidget
```

The QStackedWidget QTabWidget keeps its pages in is a plain

QWidget, and the blanket `QWidget { background-color: bg }` rule makes it paint the window colour over the ::pane it sits on. Every other layer between the pane and the tiles is tagged in `_scrolled`; this is the one that is not ours to construct.

### lines 2388-2391

```python
from ..i18n import tr
```

THE NAME AND THE COUNT ARE TRANSLATED SEPARATELY. The catalog is keyed on "Core", not on "Core  (6)", so composing first and translating after finds nothing and leaves the one part of the Home screen that is a proper noun in English.

## HomePage._build_home_tab

### lines 2402-2410

```python
def _build_home_tab(self) -> QWidget:
```

tab 1: everything

One band per category, in the order the app registry hands them over, each app in exactly one. There is deliberately no membership table here: the page this replaced had Prepare/Run/Review bands with a section→band map and a three-app override list *in this file*, so "which group is Plate Queue in" had two answers depending on which tab you were looking at, and a renamed section silently dropped its apps into a fallback band.

### lines 2427-2429

```python
grid.setHorizontalSpacing(SPACING["xs"])
```

The gap is the quiet separation between rimless resting tiles. Keep this vertical rhythm even though there is no decorative edge to reinforce it.

## HomePage._build_category_tab

### lines 2486-2488

```python
col.setSpacing(SPACING["xs"])
```

`xs`, not `sm`: the heading block, the rule and the grid are one unit, and every gap above the grid is a gap that decides whether the biggest tab scrolls.

### lines 2491-2496

```python
head = QWidget()
```

Heading + note in one block on 2 px, so adding the note costs a line rather than a line plus a layout gap.

The heading is redundant with the tab label for a sighted user, but it is what a screen reader lands on inside the page and what the category-coverage test reads.

### lines 2525-2526  _(unsure)_

```python
grid.setHorizontalSpacing(SPACING["xs"])
```

Match Home's quiet separation: resting tiles have no decorative rim, and this small gap alone keeps adjacent launchers distinct.

## HomePage._make_tile

### lines 2546-2548

```python
tile.setMaximumWidth(scaled_px(self.TILE_MAX_W))
```

Preferred/Fixed + a maximum: the tile widens to reach the edge of its column (see ``_fill_grid``) but stops at TILE_MAX_W, and never changes height.

## HomePage._scrolled

### lines 2581-2592

```python
from ..theme import make_transparent
```

No stylesheet on the scroll area: an unscoped

`background: transparent` on a QScrollArea cascades to every descendant and strips the fill off the tiles inside it. `make_transparent` tags each widget with a property instead, and the QSS rule matches only widgets carrying it.

This is what lets the pane's colour reach the eye. In the two opaque themes the blanket `QWidget { background-color: bg }` rule made this page — and its scroll viewport — paint the window colour straight over the rounded box behind them, so the box was a border around nothing and the opacity preference could not have shown a difference at any setting.

## HomePage._fill_grid

### lines 2617-2622

```python
rows = 0
```

No alignment flags at all: QGridLayout gives an *aligned* item exactly its sizeHint and positions it in the cell, so even Qt.AlignTop alone leaves a 172 px tile sitting in a 205 px column with a gap after it. Unaligned, the item is handed the whole cell; the tile's Fixed vertical policy keeps the height, and its maximumWidth caps how far it stretches.

### lines 2630-2636

```python
span = max(grid.columnCount(), columns)
```

Stretch is set over `columns` columns even when fewer are occupied. ``grid.columnCount()`` counts columns that HAVE an item, so a band with a single app got one column, that column took the whole width, and the unaligned tile floated to the middle of the page under a left-aligned heading. Naming the empty columns puts the tile back at the left edge where the heading is.

## HomePage._columns_for

### lines 2651-2653

```python
return max(1, available // (tile_w + SPACING["xs"]))
```

No cap at six any more: the tiles are 172 px, and capping the count is how a wide window ends up with a row that stops two-thirds of the way across and a page that reads as empty.

## HomePage._build_aside

### lines 2677-2680

```python
make_transparent(aside)
```

The column itself paints nothing. Untagged it runs the full height of the window as one black slab behind every panel in it — the "one large black box spanning all right side elements and going down to the bottom".

### lines 2697-2710

```python
for panel in (self._queued, self._recent, self._news,
```

THE ORDER IS THE ANSWER TO A QUESTION EACH PANEL ANSWERS, and it was rearranged on 2026-09-03 at the maintainer's request: "system is fine but should be at the bottom". What is happening now (Queued), what just happened (Recent runs), what is new in the build (News) and how much has been done (Totals) are all about the work; the machine's GPU and disk are about the machine, and belong after them.

`Module state` -- the colour-to-maturity legend -- is NOT in this column any more: "you can remove modual state". The class stays and `HomePage.legend` still answers, because the tiles' hover colours are still drawn from `StageLegend.swatch_colour` and the tests that keep the swatch and the tile from drifting apart go through it. It is simply not built into the page.

## HomePage._version

### lines 2745-2746

```python
return "" if version.lower() in ("", "dev", "unknown") else version
```

"dev" / "" are not a release, and heading a panel "spaCR dev" says less than heading it "News".

## HomePage._on_runs_changed

### lines 2752-2755

```python
active = [h for h in self._registry.active()
```

`user_visible` is False for housekeeping the user did not start the two-second usage poll above all. Without this filter Home flashed a blue "<module> usage - running" banner on and off continuously while any module screen was open.

## HomePage.set_reserved_content

### line 2802  _(unsure)_

```python
def set_reserved_content(self, widget: QWidget) -> None:
```

API kept from the page this replaces

## HomePage.eventFilter

### lines 2867-2872

```python
mark = STAGE_LABEL.get(
```

The stage goes in the hint bar as a WORD. It used to ride on the tile's tooltip, which is gone -- and the tile's hover HUE cannot be the only carrier, because colour alone fails WCAG 1.4.1. The accessible description covers screen readers; this covers a sighted colour-blind user, who reads neither the hue nor the accessibility tree.

### lines 2877-2885

```python
if not self._hint_bar.is_holding():
```

THE STRIP IS NOT CLEARED ON LEAVE, which is the whole point of the thirty-second hold: the API and Tutorial words appeared only while the pointer was on the tile, so moving toward them removed them and neither could ever be pressed. The hold in `ModuleHintBar` puts the prompt back instead, thirty seconds later or as soon as another module is hovered.

A tile with nothing registered still clears, because it never wrote anything to reach.
