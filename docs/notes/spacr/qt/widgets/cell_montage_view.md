# Notes from `spacr/qt/widgets/cell_montage_view.py`

Prose lifted out of `spacr/qt/widgets/cell_montage_view.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [experiment_root](#experiment_root) (1 entry)
- [parse_channels](#parse_channels) (2 entries)
- [_thumb_px_of](#_thumb_px_of) (1 entry)
- [fits_on_a_page](#fits_on_a_page) (1 entry)
- [_crop_settings](#_crop_settings) (2 entries)
- [load](#load) (12 entries)
- [montage_figure](#montage_figure) (1 entry)
- [_Thumb.__init__](#_thumb__init__) (2 entries)
- [_Thumb.paintEvent](#_thumbpaintevent) (1 entry)
- [_WellTab.__init__](#_welltab__init__) (3 entries)
- [_WellTab.resizeEvent](#_welltabresizeevent) (1 entry)
- [_WellTab.fill](#_welltabfill) (2 entries)
- [_WellTab._show_cell_detail](#_welltab_show_cell_detail) (1 entry)
- [_WellTab.clear](#_welltabclear) (1 entry)
- [_tooltip](#_tooltip) (1 entry)
- [_pixmap](#_pixmap) (1 entry)
- [CellMontageView.__init__](#cellmontageview__init__) (20 entries)
- [CellMontageView.set_coefficient](#cellmontageviewset_coefficient) (3 entries)
- [CellMontageView.reason](#cellmontageviewreason) (2 entries)
- [CellMontageView._multivariate_is_ready](#cellmontageview_multivariate_is_ready) (1 entry)
- [CellMontageView._on_loaded](#cellmontageview_on_loaded) (5 entries)
- [CellMontageView.picture_settings](#cellmontageviewpicture_settings) (2 entries)
- [CellMontageView.edit_picture_settings](#cellmontageviewedit_picture_settings) (4 entries)
- [CellMontageView.workspace_state](#cellmontageviewworkspace_state) (1 entry)
- [CellMontageView.apply_workspace_state](#cellmontageviewapply_workspace_state) (1 entry)
- [CellMontageView.compare_a_measurement](#cellmontageviewcompare_a_measurement) (1 entry)
- [CellMontageView.save](#cellmontageviewsave) (1 entry)
- [CellMontageView._apply_shape_availability](#cellmontageview_apply_shape_availability) (2 entries)
- [CellMontageView._on_settings_changed](#cellmontageview_on_settings_changed) (4 entries)
- [CellMontageView._ensure_graph_tab](#cellmontageview_ensure_graph_tab) (1 entry)
- [CellMontageView.annotate_the_cells](#cellmontageviewannotate_the_cells) (1 entry)
- [CellMontageView._ensure_annotation_panel](#cellmontageview_ensure_annotation_panel) (2 entries)
- [CellMontageView._refresh_controls](#cellmontageview_refresh_controls) (1 entry)
- [CellMontageView._announce](#cellmontageview_announce) (2 entries)
- [CellMontageView._open_well_tab](#cellmontageview_open_well_tab) (1 entry)
- [CellMontageView._fill](#cellmontageview_fill) (2 entries)

## Module level

### lines 39-42

```python
from ...crops import (LOAD_IMAGES, LOAD_IMAGES_LABEL, STREAM_IMAGES,
```

The headless half's own vocabulary, so the controls speak it rather than re-declaring their defaults. `spacr.cell_montage` costs numpy and pandas and nothing else -- crops and io are lazy inside it -- so this is not the torch import the module docstring is careful about.

## experiment_root

### lines 158-160

```python
def experiment_root(db_path: str) -> str:
```

The job -- everything below runs on the WORKER thread and touches no widget

## parse_channels

### lines 208-220

```python
letter = part.strip().lower()
```

THE ANNOTATION APP'S SPELLING IS ACCEPTED HERE. It asks which COLOUR PLANES to show and writes them 'r,g,b' -- `_csv_to_list` keeps whatever strings it is given and `filter_channels_pil` reads the letters directly. This box asks a different question (which SOURCE channels to cut) and answers it in indices, so typing the annotator's answer here raised ValueError and the tab refused to draw anything at all: "'rgb' is not a list of channel indeces ... and this blocks the user from being able to spawn any images".

THE LETTERS ARE TRANSLATED THROUGH THE MAPPING, never by position. spaCR's default is {r: 2, g: 1, b: 0}, so 'r' is source channel TWO; reading it as 0 would cut the planes in reverse and produce a crop that looks entirely plausible and is wrong.

### lines 225-229

```python
if letter and all(c in _COLOUR_TO_SOURCE for c in letter):
```

'rgb' AND 'rg' TOO, not only 'r,g,b'. "the user should be able to type in r,g,b or any combination" -- and a token made only of colour letters has exactly one reading, so refusing it is pedantry with a blocked montage on the other end. This is the form the report actually used.

## _thumb_px_of

### lines 502-503  _(unsure)_

```python
return max(24, min(value, 512)) if value else 0
```

Bounded: a thumbnail larger than the panel is a scroll bar, and one smaller than a few pixels is not a picture.

## fits_on_a_page

### lines 528-532

```python
thumb = max(1, int(thumb_px))
```

THE LAST ROW NEEDS NO TRAILING SPACING, and charging it one loses a whole row whenever the fit is close -- reported as "never more that two rows ... where there could be 3 almost 4". n items span n*thumb + (n-1)*spacing, so the capacity is (available + spacing) divided by the step, not available // step.

## _crop_settings

### lines 560-565

```python
if request.picture:
```

WHAT THE USER ASKED FOR IN THE SETTINGS WINDOW (instruction 170 B), translated into the crop layer's own vocabulary by `picture_settings.to_crop_settings` -- which carries ONLY the settings that change how a crop is CUT, and only the ones this mode uses. A settings window whose values never reached the picture would be worse than no settings window.

### lines 571-573

```python
settings["png_dims"] = list(request.channels)
```

``png_dims`` is the PNG path's own name for "which intensity planes become the picture", so the user's choice is expressed in the vocabulary the crop spec already speaks instead of a new one.

## load

### lines 643-647

```python
folder = request.results_path
```

THE FOLDER, NEVER THE COEFFICIENT TABLE ITSELF. `results_path` is what the panel loaded -- `results.csv`, the coefficients -- and `read_well_guide_fractions` takes a CSV at its word. Handed the coefficient table it reports "names no well", which is true and is the wrong file being read rather than a missing column.

### lines 651-660

```python
counts = None
```

THE RUN FOLDER IS A CONVENIENCE, NOT A REQUIREMENT. It holds

`regression_data.csv`, which is the score/count join PERSISTED -- and the per-well guide fractions in it are `count / well total`, computable from the count CSVs the input table already names. Requiring the folder is what produced "the loaded coefficient table was not read from a run folder" for a user who had loaded their scores and counts and was looking at the coefficients those files produced.

So: read the folder when there is one, and BUILD from the counts when there is not. Same arithmetic as `ml.process_reads`, same number.

### lines 698-701

```python
objects = objects.copy()
```

WHICH FOLDER THIS ROW'S PIXELS LIVE IN, carried on the row itself. Two plates are two experiment folders and can have two different answers -- one with its crops exported and one with only merged/ so "which source" is a property of the row, not of the montage.

### lines 704-712

```python
from ...crops import reanchor_frame
```

THE PATHS MUST SURVIVE THE FOLDER MOVING MACHINE (instruction 155 F). The database records absolute paths as they were on the machine that wrote it, and this is the one place that knows both the recorded path and the root it is under NOW. Every path-bearing column is re-anchored -- png_path for the exported crops AND path_name / merged_path for the arrays they are cut from -- and whatever carries no recognisable anchor is COUNTED and one is NAMED, because a silent pass-through is how a re-anchor that had already lost the file name stayed invisible.

### lines 729-731

```python
sources: Dict[str, Any] = {}
```

THE CROP SOURCE, ONE PER PLATE, RESOLVED BEFORE ANYTHING IS SELECTED. A montage nobody can draw is worth refusing before the selection runs, and the plan's caption has to carry which source drew it.

### lines 739-742

```python
choice = resolve_montage_crop_source(
```

THE ROUTE'S OWN REQUIREMENTS, CHECKED UP FRONT (instruction 155 E). The two routes to pixels need different things, and a user missing a channel list has to be told THAT rather than told there is no source -- which is what a check made at cutting time reports.

### lines 757-759

```python
if requirements is not None and requirements.shapes:
```

THE SHAPE IS APPLIED HERE OR IT IS SAID NOT TO BE. An object-shaped crop this route cannot cut must never quietly become a bounding box.

### lines 799-801

```python
selection["show_all"] = _show_all_of(request.picture)
```

THE WHOLE WELL, OR ONLY THE CHOSEN CELLS (instruction 172). Not the default: the two answer different questions, and a reader who cannot see the well cannot judge how many of it the fraction claims.

### lines 803-806

```python
picking = str((request.picture or {}).get("cell_picking") or "rank")
```

WHICH CELLS BELONG TO THE COEFFICIENT (172, 173). 'attributed' and 'assigned' both need every guide's fitted effect in the well, because a posterior is a comparison -- so they read them out of the run the montage is already showing.

### lines 815-825

```python
from ...control_names import common_prefix
```

The results name guides as the DESIGN did (`225160_1`) while the counts name them as the library does (`TGGT1_225160_1`), so the two are matched on the design spelling.

THE PREFIX IS MEASURED, NOT HARD-CODED. This read

`str(g).split("TGGT1_")[-1]`, which is one organism's name written into the matching rule: every guide of a Plasmodium or a human library kept its prefix, matched nothing in `raw`, and the whole `effects` map came back empty -- at which point the attribution silently has no competition to compare against. See instruction 184, which is this same assumption in the control fields.

### lines 841-846

```python
from ...cell_montage import effects_grid_from_results
```

THE GRID NOTHING EVER SET (186 A). `select_montage` has taken an `effects_grid` since option C shipped and no caller supplied one, so multivariate could never run: it found None every time, fell back to the single-score attribution, and said so in the caption. The fallback worked exactly as designed and hid the fact that what it fell back FROM was unreachable.

### lines 880-881

```python
offered: Optional[set] = None
```

WHAT THE ROUTES CAN ACTUALLY CUT, so the shape control can disable what they cannot rather than accepting the click and doing something else.

## montage_figure

### lines 973-975

```python
caption_lines = sum(len(c.splitlines()) for c in captions) + 2 * len(plans)
```

Height budget: one inch per row of crops, plus a line per caption line. The caption is long by design and clipping it would be the same failure as omitting it.

## _Thumb.__init__

### lines 1047-1048  _(unsure)_

```python
self.setAttribute(Qt.WA_TranslucentBackground, True)
```

Transparent, so the rounded tile sits on the grid without a grey square peeking out at the corners.

### lines 1051-1055

```python
self.highlight = str(highlight or "")
```

THE ANNOTATION APP'S OWN BORDER, asked for by appearance:

"highlight the cells most likely to be whatever gene is picked ... as if they were annotated in the annotations app". `label_to_hex` is where that colour is decided, and it is theme-aware because contrast is -- so this borrows it rather than picking a blue.

## _Thumb.paintEvent

### lines 1093-1097

```python
paint_tile(painter, float(self.width()), float(self.height()),
```

`current=False` and the colour swapped instead: one ring, which hover RECOLOURS rather than surrounds. A picked cell therefore keeps its blue everywhere the cursor is not -- which is the whole of show-all, where the point is to compare the cells that carry the inference against the ones that do not.

## _WellTab.__init__

### lines 1171-1174

```python
self._note = QLabel()
```

WHY THIS TAB IS EMPTY, above the grid rather than inside it. In the grid it would be indistinguishable from a thumbnail to anything counting them -- including the 'no crop' placeholders, which ARE one per object and must stay countable.

### lines 1190-1194

```python
self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
```

NO SCROLLBAR (instruction 211). The visible area IS the page, and a page that scrolls is not a page -- it is a grid with a smaller window over it, which is what this replaces. If anything is below the fold the page size is wrong, and hiding the bar makes that visible instead of navigable.

### lines 1199-1201

```python
self._thumb_px = THUMBNAIL_PX
```

HOW BIG THE CELLS ARE DRAWN AND HOW MANY FIT ON A PAGE. Both are the user's, set from the picture settings; the defaults are what this tab has always used, so a panel nobody touches is unchanged.

## _WellTab.resizeEvent

### line 1351

```python
self.show_crop(anchor)
```

THE FIRST IMAGE ON THE PAGE STAYS PUT, not the page number.

## _WellTab.fill

### lines 1411-1416

```python
shown = self._page_slice()
```

ONE PAGE AT A TIME when a well has more cells than a page holds. A well capped at 300 objects drawn as one grid is a scroll nobody reads to the end of, and the alternative that was NOT taken is silently truncating it -- a montage that shows some of a well and says it showed the well is the failure this whole panel avoids everywhere else.

### lines 1423-1426

```python
if hasattr(thumb, "clicked"):
```

CLICK FOR THE PROVENANCE. The tile already carries it on its tooltip -- which well, which object, which route cut it -- and a tooltip is unreadable the moment you want to compare two of them or copy a number out.

## _WellTab._show_cell_detail

### lines 1444-1445

```python
layout.addWidget(view)
```

SELECTABLE, because the reason to open this is usually to copy a path or an object id out of it.

## _WellTab.clear

### lines 1473-1476

```python
widget.setParent(None)
```

setParent(None) as well as deleteLater: a widget removed from the layout is still a visible child of the body at its old geometry until the deferred delete runs, which is how a rebuilt grid paints the previous montage under the new one.

## _tooltip

### lines 1496-1498

```python
parts.append("consistent with the effect — membership is inferred")
```

NOT "carries this guide". The pooled design cannot say that of any single object, and a tooltip is exactly where that claim would get made by accident.

## _pixmap

### lines 1574-1576

```python
px = int(size or THUMBNAIL_PX)
```

`.copy()` is not optional: QImage does not own the numpy buffer, and without it the pixmap points at freed memory the moment the array goes out of scope.

## CellMontageView.__init__

### lines 1685-1688

```python
self._key: str = ""
```

EVERY PIECE OF STATE A CONTROL READS IS BORN HERE, before a single signal is connected. A widget whose controls are live before its state exists is the `_significance` crash that took this application down at launch, and the rule earned its own test file.

### lines 1724-1730

```python
self._controls_row = FlowHost(self)
```

THIS ROW WRAPS BEFORE IT CLIPS. Its four buttons alone measure 553--569 px across the two shipped Open Sans weights/rasterizers; the guide selector used to be `Ignored`, so a 560 px panel met its nominal minimum only by squeezing that required control down to a few pixels. A flow keeps every control at its full size and spends height, rather than forcing the regression splitter wider, whenever the panel or the user's Zoom leaves too little room for one line.

### lines 1747-1758

```python
self._channels = QLineEdit(self)
```

CHANNELS LIVES IN THE SETTINGS WINDOW, not on the toolbar. Asked 2026-08-19: "in the cell tab there is no need to have object channels outisde of the settings pannel, moove it to the settings pannel with the other settings". It was already offered there under the annotator's own name, so the toolbar copy was a second control for one setting -- and two controls for one setting is two places for it to be wrong.

The widget stays as a hidden CHILD because `channels()` and the run's saved state both read it and a removal would have been a rename disguised as a deletion. Parenting it keeps the source of truth owned by this tab when the screen closes.

### line 1790

```python
self._picture_settings: dict = {}
```

THE ANNOTATOR'S CONTROL OVER THE PICTURE (instruction 170 B).

### lines 1808-1811

```python
self._compare_button = QPushButton("Compare a measurement…")
```

COMPARE THE CELLS THIS TAB PICKED against the rest (177 F). It sits beside the picture settings because it asks the same question of the same selection: these cells, versus the ones the picker did not choose.

### lines 1823-1828

```python
self._annotation_panel = None
```

HOW THE CELLS GET ANNOTATED is a tab rather than another persistent button in this strip. A button used to take the one-line row from 553 px to 713 px; the strip now wraps, but the tab remains the right home for forty controls that are deliberately built only when they are asked for. The same width failure is why the stringency row is not duplicated here either.

### lines 1846-1847

```python
self._save.clicked.connect(lambda: self.save())
```

No argument: `clicked` carries a bool that `save` would read as its path. The guard in `save` covers it too; this says the intent.

### lines 1852-1859

```python
stringency = QHBoxLayout()
```

THE STRINGENCY ROW. Every control here changes WHICH CELLS a reader is looking at, so every one is written into the caption and a non-default value says so ON the montage (:meth:`spacr.cell_montage.MontagePlan.settings_line`). They are PER SCREEN, NEVER PER GENE: one width for every coefficient, because a width chosen per gene is a width that can be tuned until the pictures look right and nothing in the output would show that it had been.

### lines 1899-1908

```python
self._cap.setRange(1, 1_000_000)
```

AS HIGH AS THE SETTINGS WINDOW CAN GO. This box is the value

`request()` reads, and the window writes its choice back into it through `setValue`, which clamps silently -- so a lower ceiling here would quietly turn a cap the user chose into a different one with nothing on screen saying so.

The floor stays at 1 rather than following the window down to 0: a cap of 0 reaches `select_montage` as "no cap given" and draws the default 2,000 under a caption that says 0, so it is the one value in the window's range this control must not pass on.

### lines 1919-1922

```python
stringency.addStretch(1)
```

THE ROW IS GONE FROM THE TOOLBAR. Its four controls live in the settings window now; the widgets stay because `request()` and the saved state both read them, and the window writes back to them so the two cannot drift.

### lines 1925-1930

```python
for box in (self._object, self._shape, self._source, self._per_guide,
```

THE TAB LIVES IN THE LEFT HALF OF THE FIGURES SPLITTER, which starts at 780 px. Two rows of controls with their prose spelled out in the widgets pushed this widget's minimum width to 1346 px, which is not a cosmetic problem: a minimum wider than the splitter forces the whole screen wider. The words live in the tooltips and the combos elide.

### lines 1936-1938

```python
horizontal = (QSizePolicy.Minimum
```

Hidden settings do not participate in the layout. The visible guide selector does: making it `Ignored` gave FlowLayout a zero-width item and clipped the selected mode completely.

### lines 1950-1953

```python
self._tabs = QTabWidget()
```

ONE TAB PER WELL, CLOSED ONLY BY ITS OWN X. The first tab is the summary and is not closable: it is where the arithmetic, the settings and every refusal are read, and it has to be somewhere even when there is no montage at all.

### lines 1964-1970

```python
self._caption = QPlainTextEdit()
```

THE CAPTION IS PART OF THE OUTPUT, NOT A TOOLTIP. It states the wells, the score window, EVERY SETTING that decided which cells these are, the whole arithmetic a reader can check the sum from, and -- last, so it is the sentence a reader leaves with -- that guide membership is INFERRED from a well-level fraction rather than observed. Selectable, because the reason to want it is usually to paste it into a methods section.

### lines 1979-1984

```python
self._annotation_page = QWidget()
```

THE ANNOTATE TAB IS ALWAYS THERE AND ITS CONTENT IS NOT. The tab is named from the start, so the strategies are a place a user can find rather than a button that has to be discovered; the panel inside it -- forty controls and a fitting runner -- is built the first time somebody opens it. A montage that is never annotated therefore costs a label and a layout.

### lines 1999-2003

```python
self._graph_tab = None
```

THE GRAPH TAB (179 A), beside Summary and empty until a montage has been generated: before that there are no groups to graph, and a tab offering to would be a control that cannot work. It is created here so it keeps its place in the tab ORDER -- added later it would arrive after whichever well tabs a run opened.

### lines 2006-2008

```python
for side in (QTabBar.LeftSide, QTabBar.RightSide):
```

The summary tab has no x. Qt puts the close button on whichever side the style names; removing BOTH is the only way that does not depend on the style.

### lines 2012-2014

```python
install_close_marks(self._tabs)
```

THE APPLICATION'S CLOSE MARK, NOT THIS WIDGET'S. Asked for once; the strip keeps it as tabs are opened and closed. See `theme.install_close_marks`.

### line 2025

```python
self._reflow.timeout.connect(self._relayout)
```

A bound method of this GUI-thread object, per job_runner's rules.

### lines 2034-2039

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember -- the same call class_editor, pca_view and pivot_builder each end their __init with, and its absence here is why this was the last screen still putting help on an editable field.

## CellMontageView.set_coefficient

### lines 2067-2069

```python
self._drop_montage()
```

THE GRID CANNOT SHOW ONE GENE WHILE THE VOLCANO RINGS ANOTHER. The caption names the gene it was built for, so leaving it up under a moved selection is a picture that reads as the new one.

### lines 2071-2076

```python
self._jobs.cancel()
```

AND THE LOAD IN FLIGHT IS FOR THE OLD GENE. Clicking through points faster than a merged-source load returns is the ordinary case, not an edge one -- at the measured 13.36 ms/crop a montage can be seconds behind the click. `cancel` retires the job (the bookkeeping must still run) and drops its result by generation, so the answer for a point the user has left never lands.

### lines 2079-2081

```python
self._unavailable = ""
```

A NEW COEFFICIENT IS A NEW QUESTION, so a remembered "this run has no crop source" is re-earned rather than assumed -- the databases may have been attached in between.

## CellMontageView.reason

### lines 2206-2211

```python
frame = self._frame()
```

A MISSING RUN FOLDER IS NO LONGER FATAL. The guide fractions are `count / well total`, which the input table's COUNT CSVs carry outright, so the folder is one of two sources rather than a requirement. This branch is now reached only when BOTH are absent -- which is the one case where there is genuinely nothing to compute a fraction from.

### lines 2230-2233

```python
return ("This coefficient table names no Intercept term, so "
```

SAID, NEVER FALLEN BACK FROM. Quietly using the median while the user has asked for the intercept moves the target and therefore which cells are shown, and the caption would say 'the screen median' under a setting reading 'fitted intercept'.

## CellMontageView._multivariate_is_ready

### lines 2413-2414

```python
self._force_picking("rank")
```

Their choice, recorded where the caption will read it, so the montage says "rank" because it IS rank.

## CellMontageView._on_loaded

### lines 2447-2454

```python
self._pending = expected
```

A SECOND BELT, AND `expected is None` IS THE IMPORTANT CASE. `cancel` drops what a superseded click started, but a result can still arrive after this widget has stopped waiting for one the click moved on, or a second load is in flight because the settings changed without the coefficient doing so. An answer nobody is waiting for is by definition an answer to a question the user has left, and painting it is the montage-under-the- wrong-name failure this whole tab is careful about.

### lines 2471-2474

```python
self._loaded_signature = self._load_signature()
```

WHAT THESE CROPS ANSWER. A later settings change compares against it and redraws in place when only the DISPLAY settings moved "if they have been loaded it should take a verry shourt amoutn of time to change nd reapply the settings".

### lines 2479-2484

```python
self.remember_inventory(objects=result.objects)
```

THE INVENTORY, WHICH NOTHING WAS EVER FILLING. `remember_inventory` has existed since the picture-settings window learned to offer this screen's own mask planes and object columns, and it had no caller: `_last_objects` was None every time, so those choosers silently fell back to their generic lists. It is called here because this is the one place that HAS the answer.

### lines 2492-2494

```python
if self._annotation_panel is not None:
```

THE WELLS MOVED WITH THE COEFFICIENT. A strategy panel still showing the previous gene's wells would take its positives from wells the montage on screen has nothing to do with.

### lines 2498-2501

```python
self._build_the_next_queued()
```

AND THE NEXT ONE, if `build_every_selected` queued any. Chained here rather than started together, so the tabs arrive in the order the user picked the guides rather than in the order the disk happened to answer.

## CellMontageView.picture_settings

### lines 2541-2543

```python
out.update({k: v for k, v in self._read_widgets().items()
```

The widgets are the source of truth for the ones that used to be on the toolbar, so a value set before this window ever opened is shown in it rather than replaced by a default.

### lines 2548-2553

```python
override = getattr(self, "_picking_override", "")
```

THE PICKER THE USER AGREED TO, when they were told multivariate could not run and chose rank rather than Cancel. Applied here and not by editing their settings, because it is a decision about THIS montage: their saved choice of multivariate is still what they want once the sweep exists, and silently rewriting it would take that away for every future run too.

## CellMontageView.edit_picture_settings

### lines 2563-2566

```python
source = getattr(self, "_last_source", None)
```

BUILT FROM THIS SCREEN. The mask planes come from whatever crop source the montage resolved, and the coordinate columns from the object table it actually loaded -- so the choosers list what is there rather than asking the user to remember it.

### lines 2574-2576

```python
self._picture_settings = dialog.values()
```

EVERY key, not only the ones this mode uses: a user who set a streaming setting, switched to load images and switched back must find it where they left it.

### lines 2578-2579

```python
self.clear_picking_override()
```

A NEW CHOICE OF PICKER DESERVES A NEW QUESTION. "rank, just this once" must not quietly become permanent.

### lines 2581-2583

```python
self._write_back(self._picture_settings)
```

ONE SETTING, ONE VALUE. `channels` is read from the hidden field by `channels()` and from the picture settings by the renderer, so the window writes it back rather than letting the two drift.

## CellMontageView.workspace_state

### line 2702

```python
def workspace_state(self) -> dict:
```

instruction 180: what this panel contributes to a saved run

## CellMontageView.apply_workspace_state

### lines 2761-2763

```python
self.set_coefficient(str(key))
```

LAST, and through the setter. It re-reads the frame and rebuilds the level and the effect from it, so a coefficient applied before the widgets would be described by the old settings.

## CellMontageView.compare_a_measurement

### lines 2785-2787

```python
return None
```

The tab could not be built, so there is nothing to switch to. Returning None is the behaviour; the alternative is an AttributeError out of a button press.

## CellMontageView.save

### lines 2921-2925

```python
if isinstance(path, bool):
```

`clicked` CARRIES A BOOL and this takes an optional first argument, so Qt hands the checked state into `path`. `False is None` is False, so the dialog never opened and `False` went on to the writer -- the same fault that reached the user out of FastPlotWidget.export as "QImage.save(bool)". See that method.

## CellMontageView._apply_shape_availability

### lines 3027-3030

```python
self._shape.setEnabled(not answered or bool(offered))
```

NO SHAPE CHOICE AT ALL is a real answer and not "everything is available": the exported PNGs were cut when the run wrote them, so neither entry does anything. The whole control greys out with that sentence rather than offering two options that both do nothing.

### lines 3050-3052

```python
for index in range(self._shape.count()):
```

The choice on screen is not the one that was cut. Move it, and say so -- a control reading 'object-shaped' over bounding-box crops is the silent substitution this is here to prevent.

## CellMontageView._on_settings_changed

### line 3068  _(unsure)_

```python
showing = bool(self._plans)
```

The load in flight is answering the previous settings.

### lines 3072-3074

```python
self._apply_shape_availability(MontageLoad())
```

And a shape greyed out by the LAST route is re-armed: forcing 'merged' after a run whose PNGs are gone is exactly the case where a remembered refusal would keep a real choice unavailable.

### lines 3076-3081

```python
if showing:
```

A SETTING CHANGED WHILE CELLS WERE ON SCREEN MEANS REDRAW THEM. Reported 2026-08-19: "after the cells are loaded it looks like i cannot reapply the settings". This cancelled the load in flight and stopped, so a montage already drawn kept the OLD settings and nothing said why -- indistinguishable from a control that does nothing.

### lines 3083-3093

```python
if self._can_redraw_without_loading():
```

REDRAW FROM THE CROPS ALREADY IN HAND WHERE THAT IS ENOUGH. Reported 2026-08-19: "if they have been loaded it should take a verry shourt amoutn of time to change nd reapply the settings ... i think the current behaviour is that they are reloaded every time something changes" -- which it was.

`picture_settings` already separates the settings that decide what is CUT from disk (channels, size, crop shape, object type, source) from the ones that only decide how an obtained crop is DRAWN (normalise, outline, edge, percentiles). Only the first kind can need new pixels.

## CellMontageView._ensure_graph_tab

### lines 3150-3153

```python
self._graph_tab = self._tabs.insertTab(1, self._graph_panel,
```

AFTER Summary, which is index 0, and before any well tab. NAMED FOR THE BUTTON THAT OPENS IT. It was "Graph", and the control the user presses says "Compare a measurement" -- one thing under two names reads as two things.

## CellMontageView.annotate_the_cells

### lines 3166-3167

```python
panel = self._ensure_annotation_panel()
```

BUILT BEFORE THE TAB IS RAISED, so the hundred widgets go into a page Qt is not in the middle of showing.

## CellMontageView._ensure_annotation_panel

### lines 3222-3224

```python
self._annotation_placeholder.setVisible(False)
```

HIDDEN, NOT DELETED. Destroying a widget out of the layout of the page on screen is a teardown worth not asking Qt for, and a hidden label costs one widget.

### lines 3228-3230

```python
panel.finished.connect(self._on_annotation_finished)
```

THE OUTCOME REACHES THE STATUS LINE, so a user who ran a strategy and went back to the pictures is told it landed rather than having to go and look.

## CellMontageView._refresh_controls

### lines 3292-3302

```python
if self._annotation_panel is not None:
```

AND THE COMPARE BUTTON, which had neither (reported 2026-08-21: "now i press compare a measurement and nothing hapens").

It was ENABLED with nothing to compare, and its refusal went to the status line -- which is exactly a button that appears to do nothing. Show and Save have greyed with a reason since they were written; this one was simply missed, and the same rule applies: a control that cannot act says so before it is pressed rather than after. THE ANNOTATE TAB GREYS ITS OWN RUN BUTTON, with the reason, so it follows the montage the same way the Compare tab does.

## CellMontageView._announce

### lines 3334-3338

```python
if not self._name:
```

WHICH RUN THIS IS ABOUT. Instruction 154 G: the choice of loaded run has to be visible from the views that depend on it, not only from the tab that sets it -- otherwise a montage built from the wrong run looks exactly like one built from the right one.

### lines 3343-3346

```python
selection = self.selected_coefficients()
```

THE COUNT, WHENEVER THERE IS MORE THAN ONE. A selection you cannot count is one you cannot trust (instruction 206), and a grid showing one guide out of four with nothing saying so reads as a grid of the whole selection.

## CellMontageView._open_well_tab

### lines 3406-3408

```python
taken = {t.label for t in self.well_tabs()}
```

A LABEL NO OTHER OPEN TAB ALREADY HAS. The rule above makes a collision unlikely rather than impossible, and two identical tabs is precisely the failure this label exists to prevent.

## CellMontageView._fill

### lines 3465-3467

```python
picture = self.picture_settings()
```

THE VIEW'S OWN SETTINGS. `_fill` runs from `self._plans` and has no request in scope -- reaching for one raised NameError on the real screen the moment a montage was drawn, after the run had succeeded.

### lines 3511-3516

```python
for key, tab in self._well_tabs.items():
```

A TAB THIS RUN NO LONGER FILLS MUST NOT KEEP SHOWING THE OLD ONE. Driving the real widget found it: narrow the cap and re-run, and the wells that dropped out kept their previous thumbnails under a summary describing the new settings. The tab is NOT closed -- only its x does that -- it is emptied and says why, which is the same rule the empty montage follows.
