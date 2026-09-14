# Notes from `spacr/qt/screens/map_barcodes.py`

Prose lifted out of `spacr/qt/screens/map_barcodes.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (10 entries)
- [fold_description](#fold_description) (2 entries)
- [restate_fold_button](#restate_fold_button) (1 entry)
- [show_as_window](#show_as_window) (1 entry)
- [host_pages](#host_pages) (3 entries)
- [hide_as_page](#hide_as_page) (1 entry)
- [show_as_page](#show_as_page) (3 entries)
- [FoldOpener.open](#foldopeneropen) (1 entry)
- [install_fold_strip](#install_fold_strip) (1 entry)
- [install_window_hooks](#install_window_hooks) (1 entry)
- [_widget_keys](#_widget_keys) (1 entry)
- [CategoryFold.mount](#categoryfoldmount) (6 entries)
- [CategoryFold._build_section](#categoryfold_build_section) (1 entry)
- [CategoryFoldSet._set_button](#categoryfoldset_set_button) (1 entry)
- [_planned_reference_tables](#_planned_reference_tables) (1 entry)
- [_prepare_barcode_search](#_prepare_barcode_search) (1 entry)
- [_same_setting_value](#_same_setting_value) (1 entry)
- [BarcodeSearchPanel.__init__](#barcodesearchpanel__init__) (1 entry)
- [BarcodeSearchPanel._build_ui](#barcodesearchpanel_build_ui) (5 entries)
- [BarcodeSearchPanel._on_prepared](#barcodesearchpanel_on_prepared) (1 entry)
- [BarcodeSearchPanel._finish](#barcodesearchpanel_finish) (1 entry)
- [BarcodeSearchPanel._stop_with](#barcodesearchpanel_stop_with) (1 entry)
- [BarcodeSearchPanel._absorb](#barcodesearchpanel_absorb) (1 entry)
- [BarcodeSearchPanel._fill_findings](#barcodesearchpanel_fill_findings) (3 entries)
- [BarcodeSearchPanel._render_proposal](#barcodesearchpanel_render_proposal) (1 entry)
- [BarcodeSearchPanel.changeEvent](#barcodesearchpanelchangeevent) (1 entry)
- [install_barcode_search](#install_barcode_search) (1 entry)

## Module level

### lines 77-96

```python
"investigate_hit": (
```

THE THREE REGRESSION FOLDS THAT HAD NO FALLBACK, added 2026-09-08.

`regression.FOLDED_APPS` named six keys and this table answered for three of them. The other three still have standalone registry rows, so `restate_fold_button` finds their name and description there and the buttons read correctly TODAY -- which is exactly why the gap was invisible. The day those rows are dropped, as the fold intends, three buttons on Regression go mute.

`test_the_fold_fallback_is_in_the_table_that_is_actually_read` is the assertion that caught it, and its own docstring says why this table and no other: `install_folds` restates through `map_barcodes.restate_fold_button`, which looks here and nowhere else. THE TEXT IS THE REGISTRY'S, CHARACTER FOR CHARACTER, and the first version of these three was not: a trailing full stop and a `beta` where the row says `alpha` were both caught by `test_the_fold_fallback_agrees_with_whatever_still_knows`. A fallback that paraphrases is a second source of truth, and it drifts silently the moment the row it stands in for is edited -- which is precisely the day this table starts being read.

### line 154  _(unsure)_

```python
"image_scatter": (
```

Image UMAP's two other projections of the same measurement table.

### line 164

```python
"volcano_explorer": (
```

Regression's three: the figure, the list and the write-up.

### lines 398-417

```python
PAGES_NAME = "FoldPages"
```

A fold that is a page on its host, rather than a window over it

"some new module could take space above the console or become a tab. anything to integrate the new module naturally ... if you cannot find any other way, then do your new window idea."

A folded module that has a screen of its own -- a bundle browser, a SHAP panel, a kappa table, a settings form and its Run button -- is a VIEW ON THE HOST'S DATA rather than a set of settings the host already has. So it becomes a page beside the host's own: the module itself, whole, but inside the window the user is already in rather than floating over it.

NOTHING IS REIMPLEMENTED AND NOTHING IS LOST. It is the same widget the window held, with the same signals wired to the same host; only where it is mounted changes. Closing its tab keeps the built screen, so the state it had -- a loaded bundle, a typed path, a finished run -- is still there when the button is pressed again, which is more than the window managed.

### lines 459-463

```python
try:
```

Registered at import as well, so a session that builds its stylesheet before any fold page exists already carries the rule. The import-time registration is what ``theme.WIDGET_QSS_MODULES`` loads; the call above is what covers a page made after the sheet was composed. Both, because either alone leaves one of the two orders unstyled.

### lines 776-784

```python
FOLD_HOST_MODULES: Dict[str, str] = {
```

Reaching the screens the window builds

A host screen is the generic ``AppScreen``, which knows nothing about who folded into it and should not have to. The strips are hung on it from outside, as each screen reaches the window's stack -- the route :mod:`spacr.qt.preview_registry` and :mod:`spacr.qt.recipes` take to put their own controls on a screen they do not own.

### lines 795-798

```python
"mask": "mask",
```

Mask Generation's two folds are settings categories rather than windows, but they reach their host by the same walk: the screen is the generic `AppScreen`, built by the window, and the strip is hung on it from outside.

### lines 802-805

```python
"graph_builder": "graph_builder",
```

Instruction 318's folds. Each of these hosts gained two buttons for modules that used to hold a Home tile of their own -- a tile says "start here", and none of the six is a job anyone sets out to do: they are second views of something the host is already showing.

### lines 809-810  _(unsure)_

```python
"foreign": "foreign",
```

Import: Format Converter and External Masks, folded onto the screen that was Import Project.

### lines 1281-1337

```python
SEARCH_CARD_NAME = "BarcodeSearchCard"
```

The live barcode search: settings read off the reads instead of guessed

"implementing a search function that searches for barcodes and sets settings automatically, and the ability to find more than 3, i.e. an arbitrary number of barcodes ... The automatic settings mode should be like live mode ... the user should also see the matching barcodes in a text window with 1 read per row and the different barcodes matches visualized by coloring different barcodes different colors."

THE RUN THIS EXISTS TO PREVENT. The Map Barcodes tutorial could not be written because a real paired run produced 8,611 consensus rows and zero mapped counts, and finished normally while doing it. The reads of a pair are read from opposite ends of one fragment, so a barcode that is plain in one mate is reverse complemented in the other, and spaCR ships its reference tables in one orientation only. Measured on that run, per barcode table and per mate:

barcode type   R1 plain   R1 RC    R2 plain   R2 RC    chance gRNA              0.1%    79.7%      79.0%     0.2%     ~0% column            0.9%    27.9%       2.7%     1.0%     5.2% row               6.8%     6.9%       6.8%     1.3%     6.9%

Reading column barcodes off R1 against the shipped table therefore matched 0.9% of reads against a 5.2% coincidence rate -- below noise -- and the run mapped nothing without ever saying so.

WHY EVERY RATE ON THIS PANEL IS PRINTED BESIDE ITS CHANCE RATE. Look at the row barcodes in that table. They match 6.8% of reads and they are not there at all: thirty-two barcodes of eight bases scanned across a hundred and fifty base read match somewhere by pure coincidence in 32 * (150 - 8 + 1) / 4**8 of reads, which is 6.98%. A panel that showed "row barcodes found, 6.8%" would send someone hunting for a bug that does not exist, or would auto-configure a mapping that produces garbage. So the chance rate has a column of its own directly beside the observed one, the enrichment between them has a third, and no verdict says a table is present unless the engine's own thresholds clear coincidence by a wide margin. That pairing is the feature. Dropping the chance column to save width would turn this panel back into the thing it was built to replace.

HOW IT RUNS WITHOUT FREEZING ANYTHING. The engine hands back a complete report after every chunk of reads, so this submits one chunk at a time through `JobRunner` and submits the next when the previous one lands on the GUI thread -- the same route the live preview takes, which is also what puts the work in the process-wide run registry that turns the activity spinner. Nothing here reads a file on the GUI thread: the folder scan, the reference tables, the reads and the annotation all happen inside a submitted job and come back as plain data. Measured against the run above, a chunk of two thousand reads from each of two mates against four reference tables takes 0.25 s, so a twenty thousand read sample settles in about two and a half seconds and refines visibly while it does.

AND APPLYING IS A SEPARATE PRESS. The search proposes; the user disposes. Silently rewriting settings somebody typed is not acceptable even when the rewrite is right, so the proposal is rendered as old value beside new value and nothing reaches the form until the Apply button is pressed.

## fold_description

### lines 201-211

```python
try:
```

THE DECLARED CATALOGUE, before any hand-written table. Several folded modules never had a registry row at all -- they are declared in `app_catalog` and built from it -- and that declaration already carries the name, the sentence and the maturity this button needs. Copying those three strings into a per-host `FOLD_FALLBACK` is the same knowledge written twice, and the copy is the one that goes stale.

Only consulted when the registry had nothing: a module that is BOTH registered and declared must present as the registry says, because that is what its tile and its menu entry say.

### lines 224-228

```python
try:
```

NOT EVERY FOLD LANDS HERE. This table holds what the modules folded into THIS screen said; a module folded into Measure or Classify keeps its record on that host instead. The shared resolver walks them all, so a button asks one question rather than each host having to know about every other host's folds.

## restate_fold_button

### lines 258-261

```python
set_stage = getattr(button, "set_stage", None)
```

Asked of the button rather than done here: a switch also carries a widget-local ":checked" fill computed from the stage it was built with, and setting the property alone left it lighting stable-blue when it was on while hovering in its own colour.

## show_as_window

### lines 387-389

```python
from ..theme import ensure_widget_qss_applied
```

A fallback window bypasses MainWindow._theme_screen. Its builder may have imported another late registrar, so give the window its own scope before the first show just as the ordinary screen-host path does.

## host_pages

### lines 507-509

```python
_ensure_pages_qss(screen)
```

Building the next folded module may have registered more QSS since this strip was created. Refresh the host's one owned suffix before the new child is mounted; otherwise only the first fold is styled.

### lines 529-536

```python
bar = pages.tabBar()
```

The host's own page has no close button: there is nothing behind it.

HIDDEN, NOT CLEARED. `QTabBar.setTabButton(index, side, None)` destroys the button that was there, and the tab bar goes on holding a pointer to it -- which lands as a segmentation fault in whatever the process happens to be doing when that memory is next touched, three tests away from the line that caused it. Hiding it leaves ownership where Qt put it.

### lines 542-545

```python
install_close_marks(pages, tooltip=tr("Close"))
```

THE APPLICATION'S CLOSE MARK, NOT THIS STRIP'S. The host page's button stays hidden -- `install_close_marks` carries that across so folding still costs the host nothing. See `theme.install_close_marks`.

## hide_as_page

### lines 587-588

```python
if index <= 0:
```

Never index 0: that is the host's own body, which has no close mark for the same reason -- there is nothing behind it.

## show_as_page

### lines 610-612

```python
install_close_marks(pages, tooltip=tr("Close"))
```

Qt builds its own small close button for a new tab. Ask for the application's mark here rather than waiting for the strip's watcher, so the page never appears carrying the wrong one.

### lines 614-618

```python
key = str(getattr(screen, "app_key", "") or "")
```

THE MODULE'S OWN MARK ON ITS TAB. A folded module gave up its tile, and the icon is the thing a user already associates with it -- so a page carrying only a title asks them to re-learn a name for something they could recognise at a glance. The key is taken from the screen itself, so a page opened by any host is marked the same.

### lines 624-626

```python
from ..app import _icon_for_app
```

See the note in `widgets/fold_strip.py`: resolving by filename alone ignores `_ICON_OVERRIDES` and hands a borrowing module the wrong picture.

## FoldOpener.open

### lines 675-676

```python
built = self.window = None
```

Qt deleted the C++ side under us. Build a fresh one rather than try to resurrect a dangling wrapper.

## install_fold_strip

### line 742

```python
screen._fold_openers = openers
```

The openers outlive this call only because the screen holds them.

## install_window_hooks

### lines 885-886  _(unsure)_

```python
QTimer.singleShot(0, watcher.install_current)
```

The first screen is already current when this runs, and no currentChanged is coming for it.

## _widget_keys

### lines 891-904

```python
def _widget_keys(model) -> Dict[int, str]:
```

A fold that is not a window: the module as settings categories on its host

A WINDOW IS THE LAST RESORT. Some folded modules are not a second screen at all -- they are the host's own pipeline with a gate turned on and a few extra knobs. Timelapse and Motility on Mask Generation are the case the maintainer named: "these buttons just need to toggle the visability of their settings categories as they share the rest with [the host]".

So the button reveals the module's own settings CATEGORIES on the host's form and turns the pipeline flag they belong to on. Nothing opens, nothing is replaced, and the settings the two modules share are edited once, in the place the user is already looking.

## CategoryFold.mount

### lines 978-982

```python
already = set(getattr(host_model, "_widgets", {}))
```

ONLY WHAT THIS FOLD ADDS. The loop below keeps a row exactly when the host does not already hold its key, so building the rest was 96% waste: the timelapse fold on the mask screen built 364 settings to keep 14, at 1,148 ms on every module open. The host's own keys are skipped up front instead.

### lines 987-998

```python
try:
```

AND NOT WHAT THIS RUN HAS NO OBJECT FOR. The host builds every object's rows and hides the ones whose channel is unset, so a fold that mounted them would put the host's own hidden category on the form a second time -- the timelapse fold mounted a second PATHOGEN SEGMENTATION card on mask for the one pathogen setting mask's registry spells differently. Judged against the HOST's channels, because it is the host's run these rows would join. THE HOST'S KEYS GO IN WITH THE FOLD'S. The rule gates a role only when that role's switch is on the same panel, so that it never hides a row whose switch lives on a screen the user cannot reach. Here the switch IS reachable -- it is on the host, one card up and the panel these rows would join is the union of the two.

### lines 1020-1021

```python
layout.insertWidget(max(0, layout.count() - 1), section)
```

Before the trailing stretch the panel ends with, or the categories would be pushed off the bottom of the column.

### lines 1030-1032

```python
host_model._widgets.update(
```

THE HOST NOW COLLECTS THEM. `collect()` walks `_widgets`, so a control that is on the host's form and not in this map is a control the run never sees.

### lines 1035-1039

```python
for name, value in getattr(model, "_defaults", {}).items():
```

And the module's settings that have no control -- the ones its own screen does not render either -- ride along as defaults, so the pipeline is handed the same dict its own module would have handed it. The gates are excluded: they are this fold's switch, and their value is decided by the button rather than inherited.

### lines 1044-1047

```python
if run_hides:
```

AND SO DO THE ROWS THE RUN HAS NO OBJECT FOR. They get no control the host already shows that object's category -- but the module's pipeline still reads them by name, so their value rides along exactly as a setting with no control does.

## CategoryFold._build_section

### lines 1089-1091

```python
widget.setToolTip("")
```

The help lives on the label, as it does on every other row:

a tooltip on the field itself pops while the user is typing into it.

## CategoryFoldSet._set_button

### lines 1219-1221

```python
button.setChecked(bool(on))
```

The toggle comes back through `set_active`, so the fold and everything it implies are switched by the same path a user pressing the button takes.

## _planned_reference_tables

### lines 1466-1468

```python
LOG.debug("could not read the barcode set from the settings",
```

A set that cannot be read is a settings mistake the run itself reports in full. The search still has the three shipped references to work with, and saying so twice helps nobody.

## _prepare_barcode_search

### lines 1559-1567

```python
def _prepare_barcode_search(settings, max_reads, chunk_reads):
```

The three pieces of work that happen off the GUI thread

Each of these takes data and returns data. None of them touches a widget, which is what lets `JobRunner` hand them to a worker thread, and each returns its failure in the result rather than raising, because an exception on a worker thread has nobody to catch it and a panel that goes quiet is worse than one that says what went wrong.

## _same_setting_value

### lines 1666-1668  _(unsure)_

```python
def _same_setting_value(left, right):
```

Rendering measurements as something a person can act on

## BarcodeSearchPanel.__init__

### lines 1834-1837

```python
self._jobs = JobRunner(self, threaded=threaded,
```

Every file read goes through here rather than through a thread this file owns, for the reason the live preview gives: `JobRunner` submits through `bridge.make_thread`, and that is what puts the work in the run registry the activity spinner watches.

## BarcodeSearchPanel._build_ui

### lines 1912-1913  _(unsure)_

```python
install_sorting(self.findings)
```

Every view in the package asks for this, and this one did not: the columns hold read counts and chance rates, which are numbers.

### lines 1920-1924

```python
self.proposal_label = QLabel(self)
```

THE TWO HALVES SHARE ONE HEIGHT, and which of them deserves it depends on what the reader is doing. Checking a verdict wants rows of reads; comparing tables wants rows of measurements. A splitter lets that be answered by the person looking rather than by a number chosen here, and neither half can be collapsed to nothing.

### lines 1931-1934

```python
self.proposal_label.setProperty("i18nSkipText", True)
```

WHAT A MEASUREMENT SAID IS NOT CHROME. The sentences here are the search engine's own account of what it found, assembled per run and naming files, rates and settings keys, so the translator leaves them alone rather than looking each assembled paragraph up as a caption.

### lines 1936-1939

```python
notes = QScrollArea(self)
```

SCROLLED RATHER THAN GROWN. The notes are one sentence per barcode plus one per decision, so a run that decodes six barcodes writes a paragraph. Left to size itself, the label took that height out of the reads below it and pushed them off the card entirely.

### lines 1951-1956

```python
split = QSplitter(Qt.Vertical, self)
```

THE THREE PANES SHARE ONE HEIGHT, and which of them deserves it depends on what the reader is doing. Checking a verdict wants rows of reads; comparing tables wants rows of measurements; deciding whether to apply wants the notes. A splitter lets that be answered by the person looking rather than by a number chosen here, and no pane can be collapsed to nothing.

## BarcodeSearchPanel._on_prepared

### line 2152

```python
def _on_prepared(self, result) -> None:
```

the steps, as their results arrive on the GUI thread

## BarcodeSearchPanel._finish

### lines 2251-2253

```python
settings = self.current_settings()
```

Read once. Collecting the form walks every widget on it, and a proposal compared against a second reading would be a proposal compared against a different dictionary.

## BarcodeSearchPanel._stop_with

### lines 2277-2279

```python
self._set_status("{problem}", problem=str(problem))
```

Through the same helper as every other line, so that a language changed afterwards re-renders this one rather than restoring whatever sentence the status carried before the search failed.

## BarcodeSearchPanel._absorb

### line 2283

```python
def _absorb(self, report) -> None:
```

showing what was measured

## BarcodeSearchPanel._fill_findings

### lines 2336-2340

```python
item = self.findings.item(row, column)
```

REUSED WHERE THERE IS ONE, because this is redrawn after every chunk of reads. Replacing the items would drop the row the reader had selected and the place they had scrolled to, three times a second, which is a display nobody can read while it is working.

### lines 2343-2346

```python
item = table_item()
```

THE SHARED ITEM, not `QTableWidgetItem`. A bare Qt cell sorts its text as words, so the read counts in this table would order 10 before 9 the moment a reader clicked the header.

### lines 2350-2353

```python
item.setToolTip(str(finding.reason))
```

The sentence the engine wrote about this finding says which check it passed or failed, which is the one thing a reader who disagrees with a verdict needs and the one thing no column is wide enough to hold.

## BarcodeSearchPanel._render_proposal

### lines 2465-2470

```python
lines.append(tr(
```

THE WINDOW IS DERIVED FROM WHAT WAS ESTABLISHED, and nothing else. A run whose column and row references never cleared coincidence gets a window around the guide alone, which is the honest answer to what was measured and is still not a window that will decode the reads the regex describes. Saying so above the numbers costs one line and saves a run that maps nothing.

## BarcodeSearchPanel.changeEvent

### line 2532  _(unsure)_

```python
def changeEvent(self, event) -> None:                    # noqa: N802
```

living in a themed, closable window

## install_barcode_search

### lines 2676-2677

```python
toggle.setParent(screen)
```

No strip on this screen, so the toggle goes above the card rather than nowhere, or the search would be installed and unreachable.
