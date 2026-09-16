# Notes from `spacr/qt/widgets/figure_queue.py`

Prose lifted out of `spacr/qt/widgets/figure_queue.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [_style_figure_colors](#_style_figure_colors) (1 entry)
- [_export_vector_pdf](#_export_vector_pdf) (3 entries)
- [render_figure_to_png](#render_figure_to_png) (5 entries)
- [render_pdf_to_image](#render_pdf_to_image) (2 entries)
- [render_pdf_to_image._page_rendered](#render_pdf_to_image_page_rendered) (1 entry)
- [_ClearFiguresLabel.__init__](#_clearfigureslabel__init__) (1 entry)
- [_ClearFiguresLabel._restyle](#_clearfigureslabel_restyle) (2 entries)
- [_ClearFiguresLabel.flash](#_clearfigureslabelflash) (1 entry)
- [_ClearFiguresLabel.mouseReleaseEvent](#_clearfigureslabelmousereleaseevent) (1 entry)
- [FigureQueue.__init__](#figurequeue__init__) (7 entries)
- [FigureQueue._build_ui](#figurequeue_build_ui) (12 entries)
- [FigureQueue._rerender_for_size](#figurequeue_rerender_for_size) (1 entry)
- [FigureQueue._on_resize_rendered](#figurequeue_on_resize_rendered) (1 entry)
- [FigureQueue.refresh_current_figure](#figurequeuerefresh_current_figure) (5 entries)
- [FigureQueue._open_figure_settings](#figurequeue_open_figure_settings) (1 entry)
- [FigureQueue.add_figure](#figurequeueadd_figure) (5 entries)
- [FigureQueue._refresh_live_figure](#figurequeue_refresh_live_figure) (1 entry)
- [FigureQueue.show_index](#figurequeueshow_index) (2 entries)
- [FigureQueue.mark_run](#figurequeuemark_run) (1 entry)
- [FigureQueue.forget_run](#figurequeueforget_run) (3 entries)
- [FigureQueue.clear](#figurequeueclear) (2 entries)
- [FigureQueue._request_pdf_refinement](#figurequeue_request_pdf_refinement) (4 entries)
- [FigureQueue._on_view_zoomed](#figurequeue_on_view_zoomed) (2 entries)
- [FigureQueue._on_pdf_rendered](#figurequeue_on_pdf_rendered) (1 entry)
- [FigureQueue.show_live_canvas](#figurequeueshow_live_canvas) (6 entries)
- [FigureQueue._on_canvas_scroll](#figurequeue_on_canvas_scroll) (2 entries)
- [FigureQueue._teardown_canvas](#figurequeue_teardown_canvas) (5 entries)
- [FigureQueue._preview_target_px](#figurequeue_preview_target_px) (1 entry)
- [FigureQueue._render_preview_async](#figurequeue_render_preview_async) (2 entries)
- [FigureQueue._render_preview_async.work](#figurequeue_render_preview_asyncwork) (2 entries)
- [FigureQueue._on_preview_rendered](#figurequeue_on_preview_rendered) (2 entries)
- [FigureQueue._paint_preview](#figurequeue_paint_preview) (1 entry)
- [FigureQueue._render_preview](#figurequeue_render_preview) (1 entry)
- [FigureQueue._trim_live_figures](#figurequeue_trim_live_figures) (1 entry)
- [FigureQueue._evict_live_figure](#figurequeue_evict_live_figure) (1 entry)
- [FigureQueue.replace_figure](#figurequeuereplace_figure) (1 entry)
- [FigureQueue.figure_for](#figurequeuefigure_for) (1 entry)
- [FigureQueue._pixmap_for](#figurequeue_pixmap_for) (3 entries)
- [FigureQueue._refresh_nav](#figurequeue_refresh_nav) (1 entry)
- [FigureQueue._shutdown_jobs](#figurequeue_shutdown_jobs) (1 entry)
- [FigureQueue.closeEvent](#figurequeuecloseevent) (1 entry)
- [FigureQueue.__del__](#figurequeue__del__) (2 entries)
- [_FigureSettingsDialog](#_figuresettingsdialog) (1 entry)
- [_FigureSettingsDialog.__init__](#_figuresettingsdialog__init__) (8 entries)
- [_FigureSettingsDialog._paint_colour_buttons](#_figuresettingsdialog_paint_colour_buttons) (3 entries)
- [_FigureSettingsDialog._apply_and_accept](#_figuresettingsdialog_apply_and_accept) (3 entries)

## Module level

### lines 154-157

```python
_LIVE_QUEUES: "weakref.WeakSet[FigureQueue]" = weakref.WeakSet()
```

The budget service holds no FigureQueue strongly.  Closing a screen remains sufficient to retire its queue; while it is live, the service can account for the two genuinely reclaimable layers it owns (editable Figures and decoded full-resolution pixmaps).

### lines 479-480  _(unsure)_

```python
RAM_CAP = 100
```

Number of full-resolution pixmaps kept in RAM. Older figures live only as PNGs on disk until viewed.

## _style_figure_colors

### lines 185-186  _(unsure)_

```python
ax.tick_params(color=ink, labelcolor=fg, which="both")
```

`colors=` sets the mark AND the label together, which is exactly the conflation the two controls exist to undo.

## _export_vector_pdf

### lines 266-271

```python
fig.savefig(str(pdf_path), dpi=dpi, bbox_inches="tight",
```

BYPASSES `plot.save_figure` DELIBERATELY (instruction 108 point 6): this page is not a file the user keeps, it is the vector source the queue rasterises at 2200 px to put ON SCREEN, so the print rule would make every figure flash to a light page a moment after it appeared. The format and DPI preferences are read above, which is the part `save_figure` exists for.

### lines 276-279

```python
LOG.warning("vector PDF export failed for %s: %s", pdf_path, exc)
```

This used to be a bare ``except Exception: pass``, which made a failed export indistinguishable from a successful one: the PNG appeared, the caller returned True, and the only symptom a user could ever see was that the figure never sharpened.

### lines 281-284

```python
try:
```

A half-written page is worse than no page — the queue would rasterise it and show a torn figure — and a *stale* one left over from an earlier render of this slot would show the wrong figure entirely. Either way the right state is "absent".

## render_figure_to_png

### lines 348-352

```python
text_size = figure_text_size_override(fig) or text_size
```

THE FIGURE'S OWN CHOICE BEATS THE GLOBAL DEFAULT, and this line is the whole of issue #108's third symptom. Without it every full render put the preference back over the size the user had just set in "Figure settings…", so the control appeared to do nothing and reopening the dialog showed the old number again.

### lines 355-357

```python
try:
```

Cap the DISPLAY raster so a big multi-panel figure at a high DPI can't balloon into a slow-to-decode PNG. Screen never needs > ~4000 px on the long side; the vector .pdf keeps full quality for export.

### lines 365-367

```python
from ..preferences import figure_bg_is_transparent
```

`transparent=True` when the background is "none": savefig otherwise falls back to the rcParam and writes an opaque page, so setting the facecolor alone is not enough.

### lines 369-373

```python
fig.savefig(png_path, dpi=display_dpi, bbox_inches="tight",
```

BYPASSES `plot.save_figure` DELIBERATELY (108 point 6): this is the screen raster, into a temp directory, at a capped display DPI. It is what the user is LOOKING at, so it follows the theme rather than the page. The file they keep is written by `figure_settings.save_figure_as`, which does go through `save_figure`.

### lines 378-390

```python
if not _retry_on_a_fresh_canvas(fig, png_path, display_dpi, bg):
```

A DEAD Qt CANVAS IS NOT A DEAD FIGURE. `savefig` renders through whatever canvas the figure currently holds, and a figure that was ever shown in a Qt widget holds a FigureCanvasQTAgg -- which Qt destroys with the widget, leaving `Internal C++ object (FigureCanvasQTAgg) already deleted`. Seventy of those are in the maintainer's log, and every one is a tile that silently did not render.

The figure itself is intact; only its painter is gone. Attaching a fresh Agg canvas gives it one that has no Qt object to lose, which is also the right canvas for a WORKER-THREAD render Agg touches no Qt at all, which is why the render was put on a worker in the first place.

## render_pdf_to_image

### line 432, trailing  _(unsure)_

```python
doc = QPdfDocument()
```

NO parent — see the docstring.

### lines 461-463

```python
guard = QTimer()
```

An explicit timer rather than QTimer.singleShot: this one is a local and dies with the frame, so nothing is left armed against a QEventLoop that has already been collected.

## render_pdf_to_image._page_rendered

### lines 450-451

```python
"""Take the rendered page. Queued back onto THIS thread.
```

Runs on THIS thread (queued from Qt's render thread), so the only Python that ever holds the GIL is this handful of lines.

## _ClearFiguresLabel.__init__

### lines 515-516

```python
self.setFocusPolicy(Qt.StrongFocus)
```

Focusable and Enter/Space-activatable: it is a control, and a control reachable only by mouse is one some users cannot reach.

## _ClearFiguresLabel._restyle

### lines 530-533

```python
colour = (palette["accent"] if self._flash.active
```

Flash stays the ACCENT: it is the app-wide "your click landed" mark, shared with the console's copy glyph, and a control that invented its own flash colour would be inconsistent in the other direction. Resting is `error`.

### line 537  _(unsure)_

```python
colour = "#4A9EFF" if self._flash.active else "#f85149"
```

A palette that will not load is not a reason to draw nothing.

## _ClearFiguresLabel.flash

### lines 547-549

```python
QTimer.singleShot(FLASH_MS + 10, self._restyle)
```

Flash.trigger repaints via update(), which a stylesheet colour does not follow, so the restyle is scheduled explicitly just after the shared duration.

## _ClearFiguresLabel.mouseReleaseEvent

### lines 553-554

```python
"""Clear the figures on a click inside the label.
```

Release rather than press, so dragging off the label cancels, which is what every other clickable in the app does.

## FigureQueue.__init__

### line 603  _(unsure)_

```python
self._fig_index: Dict[int, int] = {}
```

id(fig) -> index, for dedup of repeated emits of the same fig.

### line 605  _(unsure)_

```python
self._png_paths: Dict[int, str] = {}
```

index -> temp PNG path (every figure has one).

### lines 607-610

```python
self._figures: "OrderedDict[int, object]" = OrderedDict()
```

index -> matplotlib Figure, kept so a figure can be restyled and re-rendered rather than only looked at. An LRU: capped by the "Editable figures kept" preference, ordered by USE so restoring an old figure does not immediately evict it again.

### lines 614-626

```python
self._titles: Dict[int, str] = {}
```

index -> the figure's own name, taken ONCE when it arrives.

A NAME IS NOT A CACHE. `figure_titles` used to read the label off the live Figure every time it was asked, so a figure's caption survived exactly as long as its Figure did: the moment `_trim_live_figures` spilled it past the live cap the grid fell back to the temp file's stem. Measured with the default cap of 20 three runs of the house-style panels put 21 figures in, and the first three tiles were captioned `fig_00000`, `fig_00001`, `fig_00002`. On a screen that has done twelve runs that is almost every caption on the grid. A name costs a short string and is what the figure itself said it was called; nothing about running low on RAM makes that untrue.

### lines 628-633

```python
self._runs: list = []
```

WHERE EACH RUN'S FIGURES START. The queue accumulates across runs that is the point of it, an earlier run stays reachable -- but the grid letters its cells, and panel letters belong to a FIGURE. A second run continuing at L rather than restarting at A is what the maintainer reported: "the figure letters just keep climing i want each run seperated into sections".

### line 635  _(unsure)_

```python
self._ram: "OrderedDict[int, QPixmap]" = OrderedDict()
```

LRU cache of index -> full-res QPixmap (capped at ram_cap).

### lines 641-645

```python
self._jobs = JobRunner(self, app_key="figures")
```

Crisp vector-page renders run off the GUI thread. JobRunner is the one approved way to do that (see spacr.qt.job_runner) — in particular it is what wires ``thread.finished`` to a bound method rather than a closure, which is the bug this widget would otherwise have re-derived.

## FigureQueue._build_ui

### lines 694-695  _(unsure)_

```python
self._list.setContextMenuPolicy(Qt.CustomContextMenu)
```

Thumbnails are right-clickable too: the figure a user wants to restyle is often not the one currently shown.

### lines 703-704

```python
self._view.zoom_changed.connect(self._on_view_zoomed)
```

ZOOMING PAST THE RASTER RE-RENDERS THE VECTOR PAGE, rather than magnifying pixels of it. See `_on_view_zoomed`.

### lines 706-716

```python
self._view.setFrameShape(QFrame.NoFrame)
```

THE CONTAINER DOES NOT PAINT A BACKGROUND. Instruction 118 asks for figures with "not black not white just transparent", and a transparent PNG dropped into a container that paints its own base is a transparent figure on an opaque slab -- reported as "the figure container has a black background".

A QGraphicsView is three surfaces, not one: the widget, its viewport, and the SCENE's own background brush, which is what a QGraphicsView actually paints behind its items. Clearing the first two and leaving the third is the way to get this half-right and see no change at all.

### lines 723-726

```python
self._view.viewport().setAutoFillBackground(False)
```

A QGraphicsView's viewport sets autoFillBackground on ITSELF, so the theme helper's property is not enough here -- it has to be turned off explicitly or the viewport keeps painting the palette's Base, which is white.

### lines 734-736

```python
self._view.setContextMenuPolicy(Qt.CustomContextMenu)
```

Right-click anywhere on the figure, or on a thumbnail, to restyle it. Without this the panel had one button and three controls, and clicking a figure did nothing at all.

### lines 739-740  _(unsure)_

```python
self._view.clicked.connect(self.figure_clicked)
```

Re-emitted so a caller can react to "the user clicked the figure" without reaching into a private view.

### lines 742-751

```python
self._resize_timer = QTimer(self)
```

Re-render the figure when the container changes size, rather than scaling the raster. A UMAP draws its thumbnails with `OffsetImage(zoom=...)`, which is in DISPLAY pixels -- so a figure re-rendered at a larger size spreads the points out and leaves every thumbnail the same size on screen, which is what makes a crowded embedding readable. Scaling the PNG magnifies the thumbnails with everything else, which is the opposite.

Debounced: a drag emits a resize per frame and re-rendering a figure with a few thousand thumbnails is not a per-frame cost.

### lines 759-776

```python
self._stack = QStackedWidget(self)
```

THE LIVE CANVAS. A figure that still has its matplotlib Figure is shown by matplotlib itself rather than as a picture of itself.

Everything above this line is a raster pipeline: draw the figure, encode it, hand Qt a pixmap, and scale that pixmap into the view. It is blurry whenever the view is not exactly the size the raster was drawn at -- which is most of the time, and always when zoomed and it pays a full render just to LOOK at a figure.

A FigureCanvasQTAgg redraws from the figure at the widget's own device resolution, so it is crisp at any size and at any zoom, and showing a figure costs nothing at all: no render, no encode, no copy. A restyle is one draw_idle, which matplotlib coalesces to one draw per event-loop turn.

The raster view stays for figures whose Figure is gone (spilled past the live window, or loaded from a PDF), which genuinely are only a picture.

### line 778, trailing  _(unsure)_

```python
self._stack.addWidget(self._view)
```

index 0: raster

### line 783, trailing  _(unsure)_

```python
self._stack.addWidget(self._canvas_host)
```

index 1: live canvas

### lines 784-787

```python
try:
```

ONE OPAQUE CONTAINER IS ENOUGH TO BURY THE BACKDROP, and there are four between the figure and the theme's wallpaper: this widget, the stack, the canvas host and the thumbnail strip. Tagging three of them and missing one looks exactly like tagging none.

### lines 804-806

```python
nav = QHBoxLayout()
```

Navigation is via the thumbnail strip (click a thumbnail) — no separate Prev/Next buttons. A "Figure settings…" button (shown only when figures are rendered as PDF/vector) restyles the current figure.

## FigureQueue._rerender_for_size

### lines 852-860

```python
idx = self._current
```

OFF the GUI thread. `render_figure_to_png` is pure matplotlib and documents itself as safe to call from a worker, and re-rendering a real figure is not cheap: doing it inline stalled the GUI thread for 1321 ms against a 250 ms budget and `test_adding_a_pdf_figure_does_not_freeze_the_gui_thread` caught it. The same `_jobs.submit` seam the PDF refinement uses.

The callable touches no widget and returns a path, not a QPixmap: QPixmap is GUI-thread-only.

## FigureQueue._on_resize_rendered

### line 888  _(unsure)_

```python
self._pdf_state.pop(idx, None)
```

Any crisp render cached for this slot is of the old size.

## FigureQueue.refresh_current_figure

### lines 919-920  _(unsure)_

```python
fig = self.figure_for(self._current)
```

figure_for, not a dict lookup: an evicted figure is restored from its spill, so restyling an old figure redraws it too.

### lines 925-928

```python
if self.show_live_canvas(fig):
```

A live figure is drawn by matplotlib, not rasterised and copied. One draw_idle, coalesced by matplotlib to one draw per event-loop turn, and the result is crisp because it is drawn at the widget's own resolution rather than stretched from a raster.

### line 931  _(unsure)_

```python
self._render_figure(fig, Path(png))
```

The files on disk still have to match what is on screen.

### lines 936-937

```python
return True
```

The worker will deliver it; the picture on screen stays put for ~100 ms rather than the window freezing for that long.

### lines 944-945  _(unsure)_

```python
self._pdf_state.pop(self._current, None)
```

The sibling .pdf was rewritten too, so any crisp render already cached for this slot is of the OLD styling.

## FigureQueue._open_figure_settings

### lines 995-999

```python
FigureSettingsDialog(
```

NO REFRESH AFTER exec(). The dialog's closeEvent already lands a full-quality redraw before it returns, so a second call here rendered the same figure twice on every close -- measured at ~263 ms each on a 823-point volcano, i.e. half a second of dead GUI for one of them to overwrite the other with an identical picture.

## FigureQueue.add_figure

### lines 1103-1105

```python
name = self._figure_name(fig)
```

BEFORE the trim, not after: this figure is the newest, but a

`live_figure_cap` of 1 evicts it on the very next arrival and a name read later would already be gone.

### line 1114  _(unsure)_

```python
try:
```

Adopt the worker-rendered PNG (and its sibling .pdf, if any).

### line 1126  _(unsure)_

```python
pixmap = self._render_figure(fig, png_path)
```

No usable prerender — fall back to rendering here.

### line 1132  _(unsure)_

```python
item = QListWidgetItem(f"#{idx + 1}")
```

Thumbnail (small icon) — always kept.

### lines 1139-1151

```python
if self._following_the_tail(idx):
```

FOLLOW THE TAIL, BUT ONLY WHILE THE USER IS AT IT.

This used to jump to the newest figure unconditionally, so a user reading figure 3 of a Cellpose run -- which produces figures steadily, for minutes -- was thrown off it every few seconds and could not read any of them. It is also most of the queue's cost while a run streams: every arrival tore down the live canvas and built another, whether or not anyone was looking at the result.

"At the tail" means the view is on what WAS the newest figure, which is where it sits when nobody has navigated. The moment the user clicks a tile, the view stops following and new figures arrive quietly in the list -- which is where they can then click them.

## FigureQueue._refresh_live_figure

### lines 1212-1215

```python
self._pdf_state.pop(idx, None)
```

Both the .png and the .pdf under this slot just changed, so a crisp render already cached (or still in flight) for it is of the previous frame. Dropping the state supersedes the in-flight one and lets the new page be rendered.

## FigureQueue.show_index

### lines 1239-1248

```python
live = self._figures.get(idx) if self.has_live_figure(idx) else None
```

Prefer the live canvas: matplotlib draws the figure at the widget's own resolution, so it is crisp at any size and any zoom, and showing it costs no render at all. Only a figure that is genuinely just a picture -- spilled past the live window, or loaded from a PDF falls back to the raster view. has_live_figure, NOT figure_for: figure_for restores a spilled figure from disk, so using it here would un-spill a figure merely because the user navigated past it -- which is exactly what the live-figure cap exists to prevent. Viewing an old figure keeps showing the picture; restyling it is what earns a restore.

### lines 1263-1273

```python
pixmap = self._explanation_pixmap(self._why_not_shown(idx))
```

SAY WHY, RATHER THAN OPEN BLANK (199 B). Reported as "i can not see #2 when i click on it this run": the tile HAS a picture on it -- `set_pinned` refuses a null pixmap -- so the raster existed and the full-size view showed nothing.

Leaving the view untouched is worse than empty: it keeps the

PREVIOUS figure, so clicking figure 2 shows figure 1 and the user reads it as figure 2.

"figure 2 could not be restored from its spill" is an answer. An empty panel is the fault this is against.

## FigureQueue.mark_run

### lines 1323-1325

```python
self._runs[-1]["label"] = label or self._runs[-1]["label"]
```

Two starts with nothing between them: the earlier run produced no figures at all. Keep the later label rather than an empty section for each.

## FigureQueue.forget_run

### lines 1351-1352  _(unsure)_

```python
self._runs = [r for r in self._runs if r.get("label") != wanted]
```

A section that drew nothing: forget the MARK so the label stops appearing, but there is nothing to renumber.

### lines 1375-1377

```python
self._fig_index = {
```

`_fig_index` is the one keyed the OTHER way round -- id(fig) to index -- so it is rebuilt rather than shifted, and the ids of the forgotten figures go with it.

### lines 1394-1397

```python
if self._current >= end:
```

WHERE THE VIEW IS LOOKING. A current index inside the forgotten span names a figure that no longer exists; one above it has moved. Left alone, the queue shows the wrong figure or none, which is the same class of bug as a mark pointing at a run that is not on screen.

## FigureQueue.clear

### line 1499  _(unsure)_

```python
self._shutdown_jobs()
```

Before the temp dir goes: a worker is reading its PDF out of it.

### lines 1501-1503

```python
self._show_raster()
```

The live canvas owns callbacks into the Figure objects cleared below. Tear it down first so neither those callbacks nor a queued matplotlib idle draw can outlive the objects they refer to.

## FigureQueue._request_pdf_refinement

### lines 1595-1598

```python
if not self._figure_format_is_pdf():
```

Normally gated on the PDF preference. But a page written earlier, under a preference since switched to PNG, is still a vector page on disk -- and "load the PDF if it exists" is exactly what the dynamic figures option promises for a figure whose live Figure is gone.

### lines 1606-1612

```python
LOG.warning(
```

PDF mode is on and the vector page is not there: the export failed (:func:`_export_vector_pdf` logged why) or this slot was filled from a prerender that never carried one. Silently returning made that indistinguishable from a page still on its way, which is how a broken PDF export stayed invisible. Record it as failed instead — one log line, and no re-stat of the same missing file on every subsequent navigation to this figure.

### lines 1621-1622

```python
self._pdf_render_px[idx] = PDF_DISPLAY_MAX_PX
```

The baseline a later zoom compares against: without it the first wheel notch cannot tell whether a finer render would show anything.

### lines 1625-1627

```python
self._jobs.submit(
```

The submitted callable runs on a worker thread: it touches no widget, no member of this object, and builds a QImage rather than a QPixmap. Everything it needs is bound as a default argument.

## FigureQueue._on_view_zoomed

### line 1659  _(unsure)_

```python
wanted = int(width * float(scale))
```

What the view is actually asking the page for, in pixels.

### line 1666, trailing  _(unsure)_

```python
return
```

already as fine as it will ever get

## FigureQueue._on_pdf_rendered

### lines 1704-1705

```python
pixmap = QPixmap.fromImage(image)
```

QPixmap is GUI-thread-only, which is why the worker returned a QImage and the conversion happens here.

## FigureQueue.show_live_canvas

### lines 1755-1759

```python
canvas.setStyleSheet("background: transparent;")
```

THE CANVAS DOES NOT PAINT A BACKGROUND EITHER. It is the surface actually showing the figure most of the time, so a transparent raster view with an opaque canvas in front of it is no better than before. Qt's widget base and matplotlib's own figure patch are two different opaque layers and both have to go.

### lines 1763-1767

```python
transparent = canvas.palette()
```

AND ITS PALETTE. matplotlib's paintEvent erases the rect before blitting the Agg buffer, and eraseRect fills with the widget's palette brush -- which is Base, i.e. white, whatever the stylesheet says. That erase is why a figure whose patch is already 'none' still sits on a white rectangle.

### lines 1772-1777

```python
canvas.setAttribute(Qt.WA_OpaquePaintEvent, False)
```

matplotlib's Qt backend sets WA_OpaquePaintEvent, which tells Qt nothing is behind this widget and it need not clear -- so the canvas must paint every pixel itself, and it paints the ones the Agg buffer left transparent as WHITE. That single attribute is why a figure with facecolor 'none' still shows a white plot rectangle with a transparent margin around it.

### line 1780  _(unsure)_

```python
canvas.setContextMenuPolicy(Qt.CustomContextMenu)
```

Right-click must still restyle, exactly as on the raster view.

### lines 1784-1785

```python
self._canvas_layout.addWidget(toolbar)
```

Pan and zoom re-render from the figure, so zooming in gives more detail rather than bigger pixels.

### lines 1788-1799

```python
canvas.mpl_connect("scroll_event", self._on_canvas_scroll)
```

WHEEL ZOOM, because the raster view has it and this one did not.

The two views are meant to be interchangeable -- the user is not told which one they are looking at, and should not have to care. But `_ZoomView` zooms on a plain wheel turn while a matplotlib canvas ignores the wheel entirely and offers zoom only through the toolbar's magnifier. So whether scrolling zoomed depended on whether the figure was still live, which is invisible: a recent Cellpose figure is live and refused to zoom, and the same figure zoomed fine an hour later once it had spilled to a raster. Reported as "i cant zoom into the figures generated from cellpose".

## FigureQueue._on_canvas_scroll

### lines 1844-1845  _(unsure)_

```python
factor = (1.0 / step) if getattr(event, "button", "") == "up" else step
```

'up' is towards the screen, which everywhere else in this application means closer -- so it narrows the limits.

### lines 1849-1851

```python
axes.set_xlim(x - (x - left) * factor, x + (right - x) * factor)
```

Anchored on the pointer: each edge keeps its DISTANCE RATIO to the cursor, so the data under it does not move. Written to survive inverted axes, which every `imshow` panel has.

## FigureQueue._teardown_canvas

### lines 1879-1885

```python
try:
```

``FigureCanvasQT.draw_idle`` posts ``_draw_idle`` through a zero-delay QTimer and exposes no cancellation handle. Merely deleting the widget therefore leaves that bound callback in the event queue; when it runs, matplotlib asks ``height()`` of a canvas whose C++ object is already gone. Clearing the pending flag is matplotlib's own no-op path through that callback and must happen before ``deleteLater`` below.

### lines 1890-1891

```python
for attribute in ("_id_press", "_id_release", "_id_drag",
```

The toolbar's own connection ids, then anything else left on the registry: a stale callback of any kind is a crash here.

### lines 1900-1904

```python
try:
```

Anything else still bound to the two dying widgets. Disconnected through mpl_disconnect rather than by clearing the registry: matplotlib keeps its OWN entries in there (the pylab figure manager's _cidgcf among them) and emptying the dict behind its back makes its later disconnect raise KeyError instead.

### lines 1909-1915

```python
if isinstance(proxy, weakref.ReferenceType):
```

CallbackRegistry stores bound methods as

``weakref.WeakMethod`` and other callbacks in Matplotlib's ``_StrongRef`` wrapper. Neither has the ``.func`` attribute older versions exposed. Do not call an unknown proxy: a future registry may store the callback itself, and teardown must never execute user code.

### lines 1926-1937

```python
if widget is None or not isinstance(widget, QWidget):
```

ASKED BEFORE HANDED OVER. `removeWidget`, `setParent` and

`deleteLater` are C++ calls, and passing something that is not a QWidget fails inside PySide6's binding layer rather than in Python. The `except` below looked sufficient and is not: under `coverage run` the tracing perturbs that failure into a SEGMENTATION FAULT, reproducible every time and never once without coverage.

It reads as a flaky coverage tool and is not: it is a real object handed to a real C++ API that cannot take it. A canvas that has lost its C++ half still passes this check, which is why the try/except stays.

## FigureQueue._preview_target_px

### lines 1977-1978

```python
return float(min(max(longest, 600.0), 2400.0))
```

A sane floor for a view that has not been laid out yet, and a ceiling so a maximised 4K window does not ask for a 6000 px draw.

## FigureQueue._render_preview_async

### lines 2003-2007

```python
if self._preview_busy:
```

One draw in flight at a time. The copy is cheap but not free, and a worker per control change would spend the interaction copying figures whose renders are stale before they land. A change arriving mid-draw is remembered and drawn once, from the figure as it stands when the worker frees up.

### lines 2012-2018

```python
try:
```

UNDER `FIGURE_LOCK` (instruction 166). The copy is taken on the GUI thread, and the figure it walks may be one a WORKER is rendering at that moment: `bridge._capture_show` re-renders any figure marked `_spacr_live_update` -- the training monitor -- for as long as the fit runs, and `_rerender_for_size` hands the current figure to a worker on every resize. Pickling a Figure mid-draw is the same C-layer race as restyling one, and matplotlib is not thread-safe.

## FigureQueue._render_preview_async.work

### lines 2032-2033

```python
def work(_blob=blob, _target=target, _token=token,
```

Runs on a worker thread: it touches no widget and no member of this object, and returns a QImage because QPixmap is GUI-thread-only.

### line 2052  _(unsure)_

```python
image = QImage(canvas.buffer_rgba(), width, height,
```

.copy() detaches from the canvas buffer, which is freed with it.

## FigureQueue._on_preview_rendered

### lines 2068-2071

```python
self._paint_preview(payload)
```

PAINT BEFORE STARTING THE NEXT ONE. Starting it first bumps the sequence, and this payload -- freshly drawn, perfectly good -- would then be discarded as stale by its own successor, so a continuous drag would show nothing at all until the user stopped moving.

### lines 2074-2075  _(unsure)_

```python
if self._preview_pending:
```

Whatever changed while this was drawing still has to reach the picture, and now there is a free worker to draw it.

## FigureQueue._paint_preview

### lines 2095-2096

```python
self._pdf_state.pop(idx, None)
```

A refinement started before this restyle would repaint the OLD picture over the new one when it lands.

## FigureQueue._render_preview

### lines 2118-2129

```python
with FIGURE_LOCK:
```

NO bbox_inches='tight' HERE. It measures the tight box by doing a complete extra draw, which on the volcano is a flat ~125 ms on top of the ~150 ms render -- the single largest cost in the live path, and it buys only trimmed whitespace nobody is looking at mid-drag. The full render on dialog close still trims.

BYPASSES `plot.save_figure` DELIBERATELY (108 point 6): this writes to a BytesIO for a drag preview and never to a file. UNDER `FIGURE_LOCK` (instruction 166), for the same reason the restyle is: this is a GUI-thread draw of a figure a worker may be rendering. The lock is re-entrant and this holds it only for the draw.

## FigureQueue._trim_live_figures

### lines 2308-2313

```python
for old in list(self._figures)[:len(self._figures) - cap]:
```

LEAST RECENTLY USED, not lowest-numbered. Trimming by index looks equivalent while figures only ever arrive in order -- but the moment an old figure is restored so the user can restyle it, index order says it is the oldest and evicts the very figure just asked for. `_figures` is insertion-ordered and every access moves its key to the end, so the front of it is genuinely the coldest.

## FigureQueue._evict_live_figure

### lines 2325-2326  _(unsure)_

```python
try:
```

Close it, or matplotlib's own registry keeps it alive and the cache policy has removed a lookup without releasing the object.

## FigureQueue.replace_figure

### lines 2401-2402

```python
try:
```

The spill holds a pickle of the OLD figure; leaving it would let a later eviction restore the picture this call replaced.

## FigureQueue.figure_for

### line 2425, trailing  _(unsure)_

```python
self._figures.move_to_end(idx)
```

asked for = most recently used

## FigureQueue._pixmap_for

### line 2508, trailing  _(unsure)_

```python
self._ram.move_to_end(idx)
```

mark as recently used

### lines 2515-2517

```python
self._pdf_state.pop(idx, None)
```

Reaching here means the RAM copy was evicted, and with it any crisp render this slot had. ``"done"`` no longer holds, so clear it and let the refinement run again.

### lines 2521-2523

```python
if (not self.has_live_figure(idx)
```

The figure itself is gone, so nothing will re-render it from source. Its vector page is the only remaining way to show it sharply, and this is the moment the user asked for it.

## FigureQueue._refresh_nav

### lines 2549-2551

```python
self._fig_settings_btn.setVisible(self._count > 0)
```

Figure settings (background/text colour + size) restyle the figure and re-render, so they apply in both PNG and PDF mode — show whenever there's a figure to tweak.

## FigureQueue._shutdown_jobs

### line 2575  _(unsure)_

```python
LOG.debug("figure render shutdown failed", exc_info=True)
```

Reachable from __del__, where the C++ half may already be gone.

## FigureQueue.closeEvent

### lines 2594-2596

```python
"""Stop background rendering before going away.
```

Cancel matplotlib's queued idle draw before Qt destroys the canvas. A pending QTimer otherwise calls into the deleted C++ widget during the next event-loop drain.

## FigureQueue.__del__

### lines 2607-2611

```python
"""Best-effort cleanup if the widget is collected without being closed.
```

Best-effort temp cleanup if the widget is GC'd without close. The canvas goes first so its queued idle draw cannot run after Python has released the owning widget. The workers follow because they read out of the directory about to be removed, and a live QThread must not be left holding a runner whose last reference is being dropped.

### line 2624  _(unsure)_

```python
pass
```

The C++ half may already be gone when Qt initiated destruction.

## _FigureSettingsDialog

### lines 2636-2638  _(unsure)_

```python
class _FigureSettingsDialog(QDialog):
```

Figure settings dialog — restyle a matplotlib figure (PDF/vector mode)

## _FigureSettingsDialog.__init__

### lines 2679-2687

```python
self._bg, self._fg = get_figure_color_tokens()
```

THE STORED TOKENS, not the resolved pair. `get_figure_colors()` answers "what colour is the text right now", which on a dark theme is "#ffffff" whether the user chose it or not -- and this dialog WRITES ITS SEED BACK on OK, so seeding from the answer turned "follow the theme" into a hard white for every future figure the first time anybody pressed OK. That is instruction 152 section A, and the rule it cost is stated at the head of the figure colour section in `spacr/qt/preferences.py`: NEVER PERSIST A RESOLVED DEFAULT.

### lines 2690-2696

```python
self._stored_size = get_figure_text_size()
```

THE SAME RULE, IN THE SIZE KEY. 0 means "leave every figure the sizes it was drawn with", and 10 is only what the box SHOWS while that is true. `_apply_and_accept` used to write the shown number back, so pressing OK once -- to change a colour, or by accident -- froze 10 into every figure this user would ever draw, which is issue #108's "the font size is by default too large". `_size_touched` is what tells the two apart.

### lines 2700-2701

```python
self._bg, self._fg, _init_size = "auto", "auto", 10
```

The literal rather than AUTO_FIGURE_COLOR: this branch exists for the case where importing preferences failed.

### lines 2709-2714

```python
self._line_btn = _QPB("Line colour…")
```

THE SECOND OF THE TWO CONTROLS (instruction 152 B). "Text colour" was the only ink this dialog offered and it drove the spines and the tick marks as well, so there was no way to say "dark axes, coloured labels" or the other way round -- and the first report ("doesnt look like there is an option to change the axis color") was about exactly the half that had no control.

### lines 2722-2724

```python
self._auto_btn = _QPB("Follow the theme")
```

The explicit route back. A user frozen by the old dialog -- or by their own click -- otherwise has no way to un-set a colour, and a preference that can only ever be set is a trap.

### lines 2770-2771

```python
self._auto_btn.setToolTip(
```

Set after the sweep above, which owns the other three tooltips. This one is not a documented setting — it is the way out of one.

### lines 2776-2779

```python
self._line_btn.setToolTip(
```

Not routed through `install_api_tooltips`: that maps a widget onto a DOCUMENTED setting key, and this control is one half of a split that the settings documentation still spells as one ("figure_text_color"). Mapping it onto that key would put the wrong sentence on it.

### lines 2785-2788

```python
install_api_tooltips(self._umap_settings, "umap")
```

Scoped to the section rather than to the dialog: a second sweep over the whole dialog would re-decorate the three figure controls under the "umap" app key, and their documentation lives under "figure".

## _FigureSettingsDialog._paint_colour_buttons

### lines 2926-2929

```python
auto_line = auto_fg if self._is_auto(self._fg) else self._fg
```

The line button's automatic answer is the FONT's, not the theme's directly: "auto" on the line half means "follow the text", so a user who has chosen a green font sees "Automatic (#00ff00)" and is told what the axes are actually about to be drawn in.

### lines 2944-2945

```python
btn.setStyleSheet("")
```

A transparent background has no swatch to show; painting one would be a lie about what the figure will look like.

### lines 2947-2949

```python
self._auto_btn.setEnabled(
```

Greyed when it would do nothing, per instruction 106 — and it is also the readout for "am I frozen?", which is the question a user bitten by this arrives with.

## _FigureSettingsDialog._apply_and_accept

### lines 2987-2991

```python
size = int(self._size.value()) if self._size_touched \
```

0 = automatic, and it stays 0 unless the user moved the box. The colours are seeded from their TOKENS for this reason and the size is seeded from a RESOLVED 10, which is why it needed the flag rather than a comparison: a store of 0 and a box showing 10 are not the same state, and only one of them may be written.

### lines 3006-3008

```python
set_figure_text_size_override(self._fig, 0)
```

This box is the size for EVERY figure, so a per-figure override left on the figure in front of the user would make it the one figure that ignored what they just asked for.

### lines 3011-3012

```python
self._umap_settings.flush()
```

A value still sitting on the debounce timer is a value the user typed and would otherwise lose by pressing OK promptly.
