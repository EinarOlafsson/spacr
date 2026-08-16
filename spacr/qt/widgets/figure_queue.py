"""
Figure queue — the panel that collects matplotlib figures a pipeline
emits, shows them above the console, and lets the user scrub through
them without blowing up RAM.

Behaviour (per the spec):

* **Thumbnail strip** on the left — one small icon per figure, all
  kept in memory (icons are tiny).
* **Zoomable enlarged view** on the right — a QGraphicsView showing the
  current figure's full-resolution render. Wheel = zoom, fit-on-load,
  scales with the container (reuses the live-preview _ZoomView).
* **Forward / back navigation** — ◀ / ▶ buttons plus the thumbnail
  list, with an "N / total" position label.
* **RAM cap + temp spill** — the 100 most-recent figures keep their
  full-resolution QPixmap in RAM. When figure #101 arrives, figure #1's
  pixmap is dropped from RAM (its PNG stays on disk in a temp dir);
  #102 evicts #2, and so on — a 100-wide sliding window. Navigating
  back to an evicted figure reloads it from its temp PNG on demand.
* **Cleanup** — the temp directory is deleted when the queue is
  cleared or the owning screen is destroyed.
* **Progressive refinement in PDF mode** — the PNG-derived pixmap is shown
  immediately and the true vector page is rasterised at 2200 px on a
  worker thread, then swapped in. Doing that render inline used to freeze
  the GUI thread for the better part of a second per figure (measured:
  815 ms for a nine-panel 16x12" figure), on every arriving figure *and*
  on every navigation click that reloaded a spilled one.

Every figure is rendered to a temp PNG as soon as it arrives, so the
spill copy always exists and the RAM pixmap is just a cache.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PySide6.QtCore import Signal, QEvent, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QImage, QPalette, QPixmap
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from ..i18n import tr
from ..job_runner import JobRunner
from ..theme import active_palette
from .flash import FLASH_MS, Flash
from .live_preview import _ZoomView

LOG = logging.getLogger("spacr.qt.figure_queue")

#: Longest edge, in pixels, of the crisp vector-page render swapped in behind
#: the PNG. Enough for a figure filling a 4K panel; beyond it the extra pixels
#: cost render time nobody can see.
PDF_DISPLAY_MAX_PX = 2200


def _style_figure_colors(fig, bg: str, fg: str, text_size: int = 0) -> None:
    """Recolour a figure's background + all text to (bg, fg)."""
    try:
        fig.patch.set_facecolor(bg)
        for ax in fig.get_axes():
            ax.set_facecolor(bg)
            for sp in ax.spines.values():
                sp.set_color(fg)
            ax.tick_params(colors=fg)
            texts = ([ax.title, ax.xaxis.label, ax.yaxis.label]
                     + ax.get_xticklabels() + ax.get_yticklabels())
            leg = ax.get_legend()
            if leg is not None:
                texts += list(leg.get_texts())
            for t in texts:
                t.set_color(fg)
                if text_size:
                    t.set_fontsize(text_size)
    except Exception:
        pass


def _sibling_pdf(png_path) -> Path:
    """The ``.pdf`` that belongs beside ``png_path``.

    Every place in this module that reaches for a figure's vector page comes
    through here, because the *only* thing that pairs the two files is that
    the writer and the readers derive the same name — nothing records it.
    ``Path.with_suffix`` on its own does not guarantee that: it replaces
    whatever follows the last dot, so a caller handing over a path with a
    dotted name and no extension (``run_2.5``) gets ``run_2.pdf`` — and so
    does ``run_2.6``, which is two figures rasterising one page. Only a
    trailing ``.png`` is treated as an extension here; anything else keeps its
    whole name and gains a suffix, which cannot collide.

    Both real callers (:mod:`spacr.qt.bridge` and :class:`FigureQueue`) pass
    ``*.png``, so this changes nothing for them. It is here so that the one
    invariant the vector page depends on is stated once instead of being
    re-derived, identically, at every place that writes, moves, deletes or
    rasterises a page.
    """
    path = Path(png_path)
    if path.suffix.lower() == ".png":
        return path.with_suffix(".pdf")
    return path.with_name(path.name + ".pdf")


def _export_vector_pdf(fig, pdf_path: Path, dpi: int, bg: str) -> bool:
    """Write ``fig`` to ``pdf_path`` as an *editable* vector page.

    Three things are decided here rather than left to matplotlib's defaults.

    **Fonts are embedded as TrueType** (``pdf.fonttype = 42``) for the length
    of the save. matplotlib's default is Type 3, which draws every glyph as
    its own little content stream: the file is still vector, but Illustrator
    and Inkscape open the text as unselectable outlines. The preference that
    turns this whole path on is labelled "PDF (vector, editable)", and Type 3
    delivers only the first half of that. The setting is scoped with
    ``rc_context`` so a pipeline that has deliberately chosen fonttype 3 for
    its own ``savefig`` calls is not changed underneath it.

    **The requested DPI is passed in**, even though a PDF page is
    resolution-independent. Vector art ignores it; an ``imshow`` panel does
    not — and spaCR figures are full of them (cell montages, mask overlays,
    plate heatmaps). Without it those panels are embedded at the figure's own
    100 DPI while the user has asked for 300, so the "vector" export is a
    vector frame around a blurry bitmap. Note this is the *uncapped*
    preference value: the display cap in :func:`render_figure_to_png` exists
    to keep a screen raster quick to decode, and this file is not a screen
    raster. The cost is real and worth stating — a full-page montage at the
    1200 DPI the preference offers is a big file — but it is the number the
    user chose, and silently substituting a smaller one is the bug this
    function is fixing.

    **The background stays the app theme's**, i.e. black under a dark theme.
    That looks wrong for an export and is nevertheless right here. The PDF is
    not only the export: :meth:`FigureQueue._request_pdf_refinement`
    rasterises this same file at 2200 px and swaps it in as the on-screen
    pixmap, so a white page would make every figure flash from dark to light a
    moment after it appeared. And a white page would not fix anything on its
    own — :func:`_style_figure_colors` has already painted the axes black and
    the labels white *on the Figure object*, so ``facecolor="white"`` alone
    yields black panels and white-on-white text, which is worse than a
    consistent dark page. Restyling every artist for print is what the
    "Figure settings…" dialog does, and it re-renders both files.

    Returns True if the page was written.
    """
    pdf_path = Path(pdf_path)
    try:
        from matplotlib import rc_context
        with rc_context({"pdf.fonttype": 42}):
            fig.savefig(str(pdf_path), dpi=dpi, bbox_inches="tight",
                        facecolor=bg)
        return True
    except Exception as exc:
        # This used to be a bare ``except Exception: pass``, which made a
        # failed export indistinguishable from a successful one: the PNG
        # appeared, the caller returned True, and the only symptom a user
        # could ever see was that the figure never sharpened.
        LOG.warning("vector PDF export failed for %s: %s", pdf_path, exc)
        # A half-written page is worse than no page — the queue would
        # rasterise it and show a torn figure — and a *stale* one left over
        # from an earlier render of this slot would show the wrong figure
        # entirely. Either way the right state is "absent".
        try:
            pdf_path.unlink()
        except OSError:
            pass
        return False


def render_figure_to_png(fig, png_path: str) -> bool:
    """Style ``fig`` per the app theme and save it as a display-capped PNG —
    plus, in PDF mode, a genuinely vector ``.pdf`` beside it
    (:func:`_export_vector_pdf`). Pure matplotlib — no Qt — so it is
    SAFE TO CALL FROM A WORKER THREAD, which is how the pipeline bridge keeps
    the GUI responsive while lots of figures are produced.

    Returns True once the PNG — the raster the GUI actually displays — is on
    disk. A failed *sibling PDF* does not turn that into False, and the
    asymmetry is deliberate rather than sloppy: the callers turn False into
    "no pixmap, no thumbnail", so reporting a missing export that way would
    delete the figure from the gallery over a file nothing has asked for yet.
    It is logged at WARNING instead, and
    :meth:`FigureQueue._request_pdf_refinement` notices the absent page,
    says so, and stops waiting for a render that will never arrive.

    Note what the DPI preference does and does not reach. It sets the PNG's
    resolution (subject to the display cap below) and the resolution of any
    raster *inside* the PDF. It does not reach the figures a pipeline saves to
    its own results directory: those go through ``savefig`` calls in
    :mod:`spacr.plot`, :mod:`spacr.submodules` and friends, which hard-code
    their own format and DPI and never consult preferences at all.
    """
    try:
        from ..preferences import (get_figure_png_dpi, get_figure_format,
                                   get_figure_colors, get_figure_text_size)
        dpi = get_figure_png_dpi()
        bg, fg = get_figure_colors()
        text_size = get_figure_text_size()
        fmt = get_figure_format()
    except Exception:
        dpi, fmt = 200, "png"
        bg, fg, text_size = "#ffffff", "#000000", 0
    _style_figure_colors(fig, bg, fg, text_size)
    # Cap the DISPLAY raster so a big multi-panel figure at a high DPI can't
    # balloon into a slow-to-decode PNG. Screen never needs > ~4000 px on the
    # long side; the vector .pdf keeps full quality for export.
    try:
        w_in, h_in = fig.get_size_inches()
        longest_in = max(float(w_in), float(h_in)) or 1.0
        display_dpi = min(dpi, max(72, int(4000 / longest_in)))
    except Exception:
        display_dpi = min(dpi, 200)
    try:
        # `transparent=True` when the background is "none": savefig
        # otherwise falls back to the rcParam and writes an opaque page,
        # so setting the facecolor alone is not enough.
        from ..preferences import figure_bg_is_transparent
        fig.savefig(png_path, dpi=display_dpi, bbox_inches="tight",
                    facecolor=bg,
                    transparent=figure_bg_is_transparent(bg))
    except Exception as e:
        LOG.info("figure render failed: %s", e)
        return False
    if fmt == "pdf":
        _export_vector_pdf(fig, _sibling_pdf(png_path), dpi, bg)
    return True

def render_pdf_to_image(pdf_path: str, max_px: int = PDF_DISPLAY_MAX_PX,
                        timeout_ms: int = 30000):
    """Rasterise page 0 of ``pdf_path`` and return it as a ``QImage``.

    Touches no widget and builds no ``QPixmap`` — both are GUI-thread-only —
    so this is SAFE TO CALL FROM A WORKER THREAD, which is the entire point.
    The caller turns the returned QImage into a QPixmap on the GUI thread.

    **Why this does not simply call** ``QPdfDocument.render()``. That call
    does not release the GIL: PySide6 6.11 marks it blocking, so the Python
    interpreter is held for its entire duration. Moving it to a worker thread
    therefore moves the freeze without removing it — measured, on a nine-panel
    16x12" figure whose page takes 640 ms to rasterise: with the render on a
    QThread, a 1 ms QTimer on the GUI thread fired **twice** in 640 ms.
    Rendering the page in horizontal strips does not help either (pdfium
    re-walks the whole page per strip: 5 strips cost 2.1 s and the worst
    single strip was still 643 ms).

    :class:`QPdfPageRenderer` in ``MultiThreaded`` mode does the rasterising
    inside Qt's own C++ thread, which no Python call is standing in — so the
    GIL is free throughout. This function starts one, then waits on a
    **nested QEventLoop** on the calling worker thread; ``QEventLoop.exec``
    does release the GIL. Same 640 ms render, same pixels (verified equal to
    the one-shot render), worst GUI-thread gap 1.7 ms.

    The wait is bounded twice over: by ``timeout_ms``, and by
    ``QThread.quit()`` — which exits *nested* event loops too, so
    :meth:`FigureQueue._shutdown_jobs` can still stop a render in flight.

    Every QObject here is constructed with **no parent**. One parented to the
    widget would be created with the worker thread's affinity but owned by a
    GUI-thread object, and would then be destroyed from whichever thread got
    there first.

    Call this from a worker thread only. On the GUI thread the nested loop
    would re-enter the application's own event loop and deliver user input in
    the middle of a render — reentrancy, not a freeze, but no better.
    :meth:`FigureQueue._request_pdf_refinement` always submits it to a
    threaded :class:`~spacr.qt.job_runner.JobRunner`.

    Returns ``None`` on any failure. A *missing* file is the most likely one
    and is not an error: :class:`FigureQueue` deletes its temp directory when
    it closes, and a render already in flight is expected to survive that
    rather than raise on the worker thread.
    """
    try:
        from PySide6.QtCore import QEventLoop, QSize as _QSize, QTimer
        from PySide6.QtPdf import QPdfDocument, QPdfPageRenderer

        if not Path(pdf_path).is_file():
            return None
        doc = QPdfDocument()            # NO parent — see the docstring.
        if doc.load(str(pdf_path)) != QPdfDocument.Error.None_:
            return None
        if doc.pageCount() < 1:
            return None
        sz = doc.pagePointSize(0)
        longest = max(sz.width(), sz.height()) or 1.0
        scale = max_px / longest
        target = _QSize(max(1, int(sz.width() * scale)),
                        max(1, int(sz.height() * scale)))

        renderer = QPdfPageRenderer()
        renderer.setRenderMode(QPdfPageRenderer.RenderMode.MultiThreaded)
        renderer.setDocument(doc)
        loop = QEventLoop()
        box = {}

        def _page_rendered(_page, _size, image, _options, _request_id):
            # Runs on THIS thread (queued from Qt's render thread), so the
            # only Python that ever holds the GIL is this handful of lines.
            box["image"] = image
            loop.quit()

        renderer.pageRendered.connect(_page_rendered)
        # An explicit timer rather than QTimer.singleShot: this one is a local
        # and dies with the frame, so nothing is left armed against a
        # QEventLoop that has already been collected.
        guard = QTimer()
        guard.setSingleShot(True)
        guard.timeout.connect(loop.quit)
        guard.start(max(1, int(timeout_ms)))
        renderer.requestPage(0, target)
        loop.exec()
        guard.stop()

        img = box.get("image")
        return img if img is not None and not img.isNull() else None
    except Exception:
        LOG.debug("pdf page render failed for %s", pdf_path, exc_info=True)
        return None


# Number of full-resolution pixmaps kept in RAM. Older figures live
# only as PNGs on disk until viewed.
RAM_CAP = 100


#: How long a resize has to settle before the figure is redrawn. A drag
#: emits a resize per frame, and re-rendering a figure carrying a few
#: thousand thumbnails is not a per-frame cost -- so the raster is scaled
#: during the drag and the true render lands when the user lets go.
FIGURE_RESIZE_DEBOUNCE_MS = 220


class _ClearFiguresLabel(QLabel):
    """"Clear figures" as plain text, flashing the accent when clicked.

    NOT a QPushButton, deliberately. Clearing is destructive and rare; button
    chrome would give it the same visual weight as the controls beside it that
    people use constantly, and it should sit quieter than those. The
    pointing-hand cursor is what makes it discoverable as clickable without a
    border having to say so -- the same trick the console's copy glyph uses.

    The flash is the whole feedback: clearing an already-empty queue looks
    identical to a click that never landed, so the mark is what says the click
    was received.
    """

    #: Emitted on a completed click or keyboard activation.
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(tr("Clear figures"), parent)
        self.setObjectName("FigureQueueClear")
        self.setCursor(Qt.PointingHandCursor)
        # Focusable and Enter/Space-activatable: it is a control, and a
        # control reachable only by mouse is one some users cannot reach.
        self.setFocusPolicy(Qt.StrongFocus)
        self._flash = Flash(self)
        self._restyle()

    def _restyle(self) -> None:
        """Paint at the resting or the flashing colour, from the palette.

        Both colours are palette roles rather than literals, so the control
        follows a theme switch -- a hex typed in here is a hex that stays dark
        on the light theme.
        """
        try:
            palette = active_palette()
            colour = (palette["accent"] if self._flash.active
                      else palette["fg_dim"])
        except Exception:
            # A palette that will not load is not a reason to draw nothing.
            colour = "#4A9EFF" if self._flash.active else "#888888"
        self.setStyleSheet(
            f"QLabel#FigureQueueClear {{ color: {colour}; "
            "background: transparent; }")

    def flash(self) -> None:
        """Light the text briefly, then return it to its resting colour."""
        self._flash.trigger()
        self._restyle()
        # Flash.trigger repaints via update(), which a stylesheet colour does
        # not follow, so the restyle is scheduled explicitly just after the
        # shared duration.
        QTimer.singleShot(FLASH_MS + 10, self._restyle)

    def mouseReleaseEvent(self, event):        # noqa: N802 (Qt naming)
        # Release rather than press, so dragging off the label cancels, which
        # is what every other clickable in the app does.
        if (event.button() == Qt.LeftButton
                and self.rect().contains(event.pos())):
            self.flash()
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):            # noqa: N802 (Qt naming)
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.flash()
            self.clicked.emit()
            return
        super().keyPressEvent(event)


class FigureQueue(QWidget):
    """Scrollable, RAM-bounded gallery of pipeline figures."""

    #: The displayed figure was clicked (not dragged).
    figure_clicked = Signal()

    def __init__(self, ram_cap: int = RAM_CAP, parent=None):
        super().__init__(parent)
        self._ram_cap = int(ram_cap)
        self._count = 0
        # id(fig) -> index, for dedup of repeated emits of the same fig.
        self._fig_index: Dict[int, int] = {}
        # index -> temp PNG path (every figure has one).
        self._png_paths: Dict[int, str] = {}
        # index -> matplotlib Figure, kept so a figure can be restyled and
        # re-rendered rather than only looked at. An LRU: capped by the
        # "Editable figures kept" preference, ordered by USE so restoring an
        # old figure does not immediately evict it again.
        self._figures: "OrderedDict[int, object]" = OrderedDict()
        # LRU cache of index -> full-res QPixmap (capped at ram_cap).
        self._ram: "OrderedDict[int, QPixmap]" = OrderedDict()
        self._tempdir: Optional[Path] = None
        self._current = -1
        # Crisp vector-page renders run off the GUI thread. JobRunner is the
        # one approved way to do that (see spacr.qt.job_runner) — in
        # particular it is what wires ``thread.finished`` to a bound method
        # rather than a closure, which is the bug this widget would otherwise
        # have re-derived.
        self._jobs = JobRunner(self, app_key="figures")
        #: index -> the in-flight render's token (an int), or ``"done"`` /
        #: ``"failed"`` once settled. Absent means "may be refined". See
        #: :meth:`_request_pdf_refinement`.
        self._pdf_state: Dict[int, Any] = {}
        self._pdf_seq = 0
        #: Newest live-preview render; older results are dropped on arrival.
        self._preview_seq = 0
        #: A preview draw is on a worker right now.
        self._preview_busy = False
        #: A change landed mid-draw and still needs to reach the picture.
        self._preview_pending = False
        #: Set by the owning screen; see :meth:`set_propagate_callback`.
        self._propagate_cb = None
        self._build_ui()

    # -- construction ------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)

        self._list = QListWidget()
        self._list.setObjectName("FiguresList")
        self._list.setFixedWidth(160)
        self._list.setIconSize(QSize(140, 90))
        # Thumbnails are right-clickable too: the figure a user wants to
        # restyle is often not the one currently shown.
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._list_context_menu)
        self._list.setSpacing(4)
        self._list.currentRowChanged.connect(self._on_row_changed)
        body.addWidget(self._list)

        self._view = _ZoomView(self)
        # THE CONTAINER DOES NOT PAINT A BACKGROUND. Instruction 118 asks for
        # figures with "not black not white just transparent", and a
        # transparent PNG dropped into a container that paints its own base
        # is a transparent figure on an opaque slab -- reported as "the figure
        # container has a black background".
        #
        # A QGraphicsView is three surfaces, not one: the widget, its
        # viewport, and the SCENE's own background brush, which is what a
        # QGraphicsView actually paints behind its items. Clearing the first
        # two and leaving the third is the way to get this half-right and see
        # no change at all.
        self._view.setFrameShape(QFrame.NoFrame)
        self._view.setBackgroundBrush(Qt.NoBrush)
        try:
            self._view.scene().setBackgroundBrush(Qt.NoBrush)
        except Exception:               # pragma: no cover - no scene yet
            pass
        # A QGraphicsView's viewport sets autoFillBackground on ITSELF, so
        # the theme helper's property is not enough here -- it has to be
        # turned off explicitly or the viewport keeps painting the palette's
        # Base, which is white.
        self._view.viewport().setAutoFillBackground(False)
        try:
            from ..theme import make_transparent

            make_transparent(self._view, self._view.viewport())
        except Exception:               # pragma: no cover - theme absent
            pass
        # Right-click anywhere on the figure, or on a thumbnail, to restyle
        # it. Without this the panel had one button and three controls, and
        # clicking a figure did nothing at all.
        self._view.setContextMenuPolicy(Qt.CustomContextMenu)
        self._view.customContextMenuRequested.connect(self._view_context_menu)
        # Re-emitted so a caller can react to "the user clicked the
        # figure" without reaching into a private view.
        self._view.clicked.connect(self.figure_clicked)
        # Re-render the figure when the container changes size, rather than
        # scaling the raster. A UMAP draws its thumbnails with
        # `OffsetImage(zoom=...)`, which is in DISPLAY pixels -- so a figure
        # re-rendered at a larger size spreads the points out and leaves
        # every thumbnail the same size on screen, which is what makes a
        # crowded embedding readable. Scaling the PNG magnifies the
        # thumbnails with everything else, which is the opposite.
        #
        # Debounced: a drag emits a resize per frame and re-rendering a
        # figure with a few thousand thumbnails is not a per-frame cost.
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(FIGURE_RESIZE_DEBOUNCE_MS)
        self._resize_timer.timeout.connect(self._rerender_for_size)
        self._view.installEventFilter(self)
        self._view.setMinimumHeight(280)

        # THE LIVE CANVAS. A figure that still has its matplotlib Figure is
        # shown by matplotlib itself rather than as a picture of itself.
        #
        # Everything above this line is a raster pipeline: draw the figure,
        # encode it, hand Qt a pixmap, and scale that pixmap into the view.
        # It is blurry whenever the view is not exactly the size the raster
        # was drawn at -- which is most of the time, and always when zoomed --
        # and it pays a full render just to LOOK at a figure.
        #
        # A FigureCanvasQTAgg redraws from the figure at the widget's own
        # device resolution, so it is crisp at any size and at any zoom, and
        # showing a figure costs nothing at all: no render, no encode, no
        # copy. A restyle is one draw_idle, which matplotlib coalesces to one
        # draw per event-loop turn.
        #
        # The raster view stays for figures whose Figure is gone (spilled past
        # the live window, or loaded from a PDF), which genuinely are only a
        # picture.
        self._stack = QStackedWidget(self)
        self._stack.addWidget(self._view)          # index 0: raster
        self._canvas_host = QWidget(self)
        self._canvas_layout = QVBoxLayout(self._canvas_host)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas_layout.setSpacing(0)
        self._stack.addWidget(self._canvas_host)   # index 1: live canvas
        # ONE OPAQUE CONTAINER IS ENOUGH TO BURY THE BACKDROP, and there are
        # four between the figure and the theme's wallpaper: this widget, the
        # stack, the canvas host and the thumbnail strip. Tagging three of
        # them and missing one looks exactly like tagging none.
        try:
            from ..theme import make_transparent

            make_transparent(self, self._stack, self._canvas_host, self._list,
                             self._list.viewport())
        except Exception:               # pragma: no cover - theme absent
            pass
        self._canvas = None
        self._canvas_toolbar = None
        #: Live figures go to the canvas. Turned off to exercise the raster
        #: pipeline, which is still what a spilled or PDF-only figure uses.
        self._live_canvas_enabled = True
        self._stack.setMinimumHeight(280)
        body.addWidget(self._stack, 1)
        root.addLayout(body, 1)

        # Navigation is via the thumbnail strip (click a thumbnail) — no
        # separate Prev/Next buttons. A "Figure settings…" button (shown only
        # when figures are rendered as PDF/vector) restyles the current figure.
        nav = QHBoxLayout()
        self._pos_label = QLabel("0 / 0", self)
        self._pos_label.setAlignment(Qt.AlignCenter)
        self._fig_settings_btn = QPushButton("Figure settings…", self)
        self._fig_settings_btn.clicked.connect(self._open_figure_settings)
        self._clear_label = _ClearFiguresLabel(self)
        self._clear_label.clicked.connect(self.clear)
        nav.addWidget(self._pos_label, 1)
        nav.addWidget(self._clear_label)
        nav.addWidget(self._fig_settings_btn)
        root.addLayout(nav)
        self._refresh_nav()

    def eventFilter(self, obj, event):
        """Debounce the view's resizes into one re-render."""
        if obj is self._view and event.type() == QEvent.Resize:
            self._resize_timer.start()
        return super().eventFilter(obj, event)

    def _rerender_for_size(self) -> None:
        """Redraw the current figure at the view's current size.

        The EMBEDDING is untouched -- this only changes the canvas the same
        points are drawn on. A resize that re-embeds is a resize that loses
        the user's place, and on a UMAP that means every neighbour
        relationship the user was reading moves.
        """
        fig = self._figures.get(self._current)
        png = self._png_paths.get(self._current)
        if fig is None or not png:
            return
        size = self._view.size()
        if size.width() < 80 or size.height() < 80:
            return
        try:
            dpi = float(fig.get_dpi()) or 100.0
            want = (max(2.0, size.width() / dpi), max(2.0, size.height() / dpi))
            if (abs(fig.get_size_inches()[0] - want[0]) < 0.05
                    and abs(fig.get_size_inches()[1] - want[1]) < 0.05):
                return
            fig.set_size_inches(*want)
        except Exception:
            LOG.debug("could not resize the figure", exc_info=True)
            return

        # OFF the GUI thread. `render_figure_to_png` is pure matplotlib and
        # documents itself as safe to call from a worker, and re-rendering a
        # real figure is not cheap: doing it inline stalled the GUI thread
        # for 1321 ms against a 250 ms budget and
        # `test_adding_a_pdf_figure_does_not_freeze_the_gui_thread` caught
        # it. The same `_jobs.submit` seam the PDF refinement uses.
        #
        # The callable touches no widget and returns a path, not a QPixmap:
        # QPixmap is GUI-thread-only.
        idx = self._current
        self._resize_seq = getattr(self, "_resize_seq", 0) + 1
        token = self._resize_seq
        self._jobs.submit(
            lambda _i=idx, _t=token, _f=fig, _p=str(png): (
                _i, _t, render_figure_to_png(_f, _p), _p),
            self._on_resize_rendered)

    def _on_resize_rendered(self, payload) -> None:
        """Show a finished resize render. Always on the GUI thread.

        Discarded when a later resize has already been dispatched, or when
        the user has navigated to another figure since -- both are ordinary
        during a drag, and showing a stale render is worse than showing the
        scaled raster for another moment.
        """
        if not payload:
            return
        idx, token, ok, png = payload
        if not ok or idx != self._current:
            return
        if token != getattr(self, "_resize_seq", 0):
            return
        pixmap = QPixmap(png)
        if pixmap.isNull():
            return
        self._cache_pixmap(idx, pixmap)
        # Any crisp render cached for this slot is of the old size.
        self._pdf_state.pop(idx, None)
        self._view.set_pixmap(self._display_pixmap(idx, pixmap))

    def set_propagate_callback(self, callback) -> None:
        """Register ``callback(dict)`` for the settings window's Propagate.

        The same seam the Mask live preview and the UMAP explorer use
        (``SettingsWidgets.set_value_for_key`` behind an owner method), so a
        value tuned against a finished figure lands in the settings panel and
        is saved with the run instead of living in a dialog that is about to
        close. Optional: a queue built in a test has none, and the button
        says so rather than doing nothing.
        """
        self._propagate_cb = callback

    def refresh_current_figure(self, preview: bool = False) -> bool:
        """Re-rasterise the figure on screen after something restyled it.

        Writes through :meth:`_render_figure`, so the PNG (at the preference
        DPI) and its sibling vector page are both rewritten — the format the
        user asked for is the format the view and the export agree on.

        :param preview: render the raster only, skipping the vector page. A
            full render writes the PNG *and* exports a PDF at the preference
            DPI, which is the better part of a second on a large figure. That
            is right once; it is ruinous while a control is moving, and doing
            it per change is what made the settings dialog hang. The vector
            page is rewritten when the dialog closes, so nothing stays stale.
        :returns: True when the view was updated.
        """
        # figure_for, not a dict lookup: an evicted figure is restored from
        # its spill, so restyling an old figure redraws it too.
        fig = self.figure_for(self._current)
        png = self._png_paths.get(self._current)
        if fig is None or not png:
            return False
        # A live figure is drawn by matplotlib, not rasterised and copied.
        # One draw_idle, coalesced by matplotlib to one draw per event-loop
        # turn, and the result is crisp because it is drawn at the widget's
        # own resolution rather than stretched from a raster.
        if self.show_live_canvas(fig):
            if not preview:
                # The files on disk still have to match what is on screen.
                self._render_figure(fig, Path(png))
                self._pdf_state.pop(self._current, None)
            return True
        if preview and self._render_preview_async(fig):
            # The worker will deliver it; the picture on screen stays put for
            # ~100 ms rather than the window freezing for that long.
            return True
        pixmap = (self._render_preview(fig, Path(png)) if preview
                  else self._render_figure(fig, Path(png)))
        if pixmap is None:
            return False
        self._cache_pixmap(self._current, pixmap)
        # The sibling .pdf was rewritten too, so any crisp render already
        # cached for this slot is of the OLD styling.
        self._pdf_state.pop(self._current, None)
        shown = self._display_pixmap(self._current, pixmap)
        self._view.set_pixmap(shown)
        item = self._list.item(self._current)
        if item is not None and not shown.isNull():
            item.setIcon(QIcon(shown.scaled(
                140, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        return True

    def refresh_figure(self, index: int, preview: bool = False) -> bool:
        """Re-rasterise a figure that is NOT the one on screen.

        Restyling from a grid tile has to redraw that tile. Without this the
        edit lands on the matplotlib object, the picture the grid is built
        from stays as it was, and the user sees a menu that appears to do
        nothing -- so they do it again, and again.

        The view is deliberately not touched: the whole point of editing from
        the grid is that the grid stays put.
        """
        index = int(index)
        if index == self._current:
            return self.refresh_current_figure(preview)
        fig = self.figure_for(index)
        png = self._png_paths.get(index)
        if fig is None or not png:
            return False
        pixmap = (self._render_preview(fig, Path(png)) if preview
                  else self._render_figure(fig, Path(png)))
        if pixmap is None:
            return False
        self._cache_pixmap(index, pixmap)
        self._pdf_state.pop(index, None)
        item = self._list.item(index)
        if item is not None and not pixmap.isNull():
            item.setIcon(QIcon(pixmap.scaled(
                140, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        return True

    def _open_figure_settings(self) -> None:
        """Open the full settings dialog for the current figure.

        ``figure_for`` rather than a dict lookup, so a figure past the live
        window is restored from its spill and is editable too -- which is the
        whole reason it is spilled as a Figure rather than only as a picture.
        """
        from .figure_settings import FigureSettingsDialog

        figure = self.figure_for(self._current)
        if figure is None:
            return
        # NO REFRESH AFTER exec(). The dialog's closeEvent already lands a
        # full-quality redraw before it returns, so a second call here
        # rendered the same figure twice on every close -- measured at ~263 ms
        # each on a 823-point volcano, i.e. half a second of dead GUI for one
        # of them to overwrite the other with an identical picture.
        FigureSettingsDialog(
            figure, self, on_change=self.refresh_current_figure,
            propagate_callback=self._propagate_cb).exec()

    def show_figure_menu(self, position, idx: Optional[int] = None,
                         navigate: bool = True) -> None:
        """Right-click menu for a figure, from the view, a thumbnail or a tile.

        The panel had one button offering three controls -- background, text
        colour, text size -- and no context menu at all, so a figure could not
        be restyled by clicking on it.

        :param navigate: whether ``idx`` becomes the current figure first. The
            thumbnail strip wants that; the figure grid does not, because a
            grid is for comparing figures and jumping to one loses the
            comparison the user was making.
        """
        from .figure_settings import build_figure_context_menu

        index = self._current if idx is None else int(idx)
        if navigate and index != self._current and 0 <= index < self._count:
            self.show_index(index)
        figure = self.figure_for(index)

        def _redraw(preview: bool = False, _i=index) -> bool:
            return self.refresh_figure(_i, preview)

        menu = build_figure_context_menu(
            self, figure, on_change=_redraw,
            open_settings=self._open_figure_settings)
        menu.exec(position)
        return menu

    def _view_context_menu(self, point) -> None:
        self.show_figure_menu(self._view.mapToGlobal(point))

    def _list_context_menu(self, point) -> None:
        item = self._list.itemAt(point)
        row = self._list.row(item) if item is not None else self._current
        self.show_figure_menu(self._list.mapToGlobal(point), row)

    # -- temp dir ----------------------------------------------------------

    def _ensure_tempdir(self) -> Path:
        if self._tempdir is None:
            self._tempdir = Path(tempfile.mkdtemp(prefix="spacr_figq_"))
        return self._tempdir

    # -- public API --------------------------------------------------------

    def add_figure(self, fig, prerendered_png: Optional[str] = None) -> int:
        """Render + append ``fig`` (a matplotlib Figure). Returns its
        index. Re-emitting the same figure object re-selects it instead
        of duplicating.

        ``prerendered_png`` is a PNG the pipeline bridge already rendered in a
        WORKER thread — when supplied we just adopt it (a fast file move + a
        cheap QPixmap load) instead of doing the expensive savefig on the GUI
        thread, so the UI stays responsive while many figures stream in."""
        if id(fig) in self._fig_index:
            idx = self._fig_index[id(fig)]
            if prerendered_png and Path(prerendered_png).is_file():
                self._refresh_live_figure(idx, prerendered_png)
            self.show_index(idx)
            return idx

        idx = self._count
        self._count += 1
        self._fig_index[id(fig)] = idx
        self._figures[idx] = fig
        self._trim_live_figures()

        png_path = self._ensure_tempdir() / f"fig_{idx:05d}.png"
        pixmap = None
        if prerendered_png and Path(prerendered_png).is_file():
            # Adopt the worker-rendered PNG (and its sibling .pdf, if any).
            try:
                shutil.move(prerendered_png, str(png_path))
                src_pdf = _sibling_pdf(prerendered_png)
                if src_pdf.is_file():
                    shutil.move(str(src_pdf), str(_sibling_pdf(png_path)))
                pixmap = QPixmap(str(png_path))
                if pixmap.isNull():
                    pixmap = None
            except Exception:
                pixmap = None
        if pixmap is None:
            # No usable prerender — fall back to rendering here.
            pixmap = self._render_figure(fig, png_path)
        self._png_paths[idx] = str(png_path)
        if pixmap is not None:
            self._cache_pixmap(idx, pixmap)

        # Thumbnail (small icon) — always kept.
        item = QListWidgetItem(f"#{idx + 1}")
        item.setTextAlignment(Qt.AlignCenter)
        if pixmap is not None and not pixmap.isNull():
            item.setIcon(QIcon(pixmap.scaled(
                140, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        self._list.addItem(item)

        self._list.setCurrentRow(idx)   # jump to the newest
        self.show_index(idx)
        return idx

    def _refresh_live_figure(self, idx: int, prerendered_png: str) -> None:
        """Replace one live figure's raster while preserving its gallery slot.

        The vector page has to move with the raster, *including when there
        isn't one*. A replacement that arrives without a sibling ``.pdf``
        leaves the slot's previous page in place, and the refinement dispatched
        a few lines below would then rasterise it and paint the figure this
        call just superseded over the new one. That is not a corner case: the
        training monitor re-emits a ``_spacr_live_update`` figure every epoch,
        so the user would watch each new plot appear and then revert to the
        first one. Deleting the orphan is what keeps the pairing honest.
        """
        target = Path(self._png_paths[idx])
        try:
            shutil.move(prerendered_png, str(target))
            src_pdf = _sibling_pdf(prerendered_png)
            dst_pdf = _sibling_pdf(target)
            if src_pdf.is_file():
                shutil.move(str(src_pdf), str(dst_pdf))
            elif dst_pdf.is_file():
                try:
                    dst_pdf.unlink()
                except OSError:
                    LOG.debug("could not drop stale vector page %s", dst_pdf)
            pixmap = QPixmap(str(target))
            if pixmap.isNull():
                return
            # Both the .png and the .pdf under this slot just changed, so a
            # crisp render already cached (or still in flight) for it is of
            # the previous frame. Dropping the state supersedes the in-flight
            # one and lets the new page be rendered.
            self._pdf_state.pop(idx, None)
            self._cache_pixmap(idx, pixmap)
            pixmap = self._display_pixmap(idx, pixmap)
            item = self._list.item(idx)
            if item is not None:
                item.setIcon(QIcon(pixmap.scaled(
                    140, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            if self._current == idx:
                self._view.set_pixmap(pixmap)
        except Exception as exc:
            LOG.info("live figure refresh failed: %s", exc)

    def show_index(self, idx: int) -> None:
        if not (0 <= idx < self._count):
            return
        self._current = idx
        # Prefer the live canvas: matplotlib draws the figure at the widget's
        # own resolution, so it is crisp at any size and any zoom, and showing
        # it costs no render at all. Only a figure that is genuinely just a
        # picture -- spilled past the live window, or loaded from a PDF --
        # falls back to the raster view.
        # has_live_figure, NOT figure_for: figure_for restores a spilled
        # figure from disk, so using it here would un-spill a figure merely
        # because the user navigated past it -- which is exactly what the
        # live-figure cap exists to prevent. Viewing an old figure keeps
        # showing the picture; restyling it is what earns a restore.
        live = self._figures.get(idx) if self.has_live_figure(idx) else None
        if live is not None and self.show_live_canvas(live):
            if self._list.currentRow() != idx:
                self._list.blockSignals(True)
                self._list.setCurrentRow(idx)
                self._list.blockSignals(False)
            self._refresh_nav()
            return
        self._show_raster()
        pixmap = self._pixmap_for(idx)
        if pixmap is not None:
            self._view.set_pixmap(pixmap)
        if self._list.currentRow() != idx:
            self._list.blockSignals(True)
            self._list.setCurrentRow(idx)
            self._list.blockSignals(False)
        self._refresh_nav()

    def show_prev(self) -> None:
        if self._current > 0:
            self.show_index(self._current - 1)

    def show_next(self) -> None:
        if self._current < self._count - 1:
            self.show_index(self._current + 1)

    def all_pixmaps(self):
        """Every figure the queue holds, in order, for the grid view.

        Reads the PNG from disk for a figure whose pixmap has been evicted
        rather than promoting it in the RAM cache: building a grid is a bulk
        read of everything, and letting it reorder the cache would evict
        exactly the figures the user is currently looking at.
        """
        from PySide6.QtGui import QPixmap

        out = []
        for index in range(self._count):
            pixmap = self._ram.get(index)
            if pixmap is None:
                path = self._png_paths.get(index)
                pixmap = QPixmap(path) if path else None
            out.append(pixmap if pixmap is not None and not pixmap.isNull()
                       else None)
        return out

    def figure_titles(self):
        """A short name per figure, from its file, for the grid captions."""
        import os as _os

        titles = []
        for index in range(self._count):
            path = self._png_paths.get(index)
            titles.append(_os.path.splitext(_os.path.basename(path))[0]
                          if path else f"figure {index + 1}")
        return titles

    def count(self) -> int:
        return self._count

    def ram_resident(self) -> int:
        """How many full-res pixmaps are currently held in RAM."""
        return len(self._ram)

    def spilled_count(self) -> int:
        """How many figures have been evicted from RAM to disk-only."""
        return max(0, self._count - len(self._ram))

    def active_jobs(self) -> int:
        """How many crisp-render worker threads are still winding down."""
        return self._jobs.active_jobs()

    def is_busy(self) -> bool:
        """True while a crisp vector-page render has not been delivered yet."""
        return self._jobs.is_busy()

    def clear(self) -> None:
        """Drop everything and delete the temp dir."""
        # Before the temp dir goes: a worker is reading its PDF out of it.
        self._shutdown_jobs()
        self._list.clear()
        self._ram.clear()
        self._png_paths.clear()
        self._fig_index.clear()
        self._figures.clear()
        self._pdf_state.clear()
        self._count = 0
        self._current = -1
        self._view.set_pixmap(QPixmap())
        self._delete_tempdir()
        self._refresh_nav()

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _style_figure(fig, bg: str, fg: str, text_size: int = 0) -> None:
        """Recolour a figure's background + all text to (bg, fg), so plots
        follow the app theme (dark → black bg + white text)."""
        try:
            fig.patch.set_facecolor(bg)
            for ax in fig.get_axes():
                ax.set_facecolor(bg)
                for sp in ax.spines.values():
                    sp.set_color(fg)
                ax.tick_params(colors=fg)
                texts = ([ax.title, ax.xaxis.label, ax.yaxis.label]
                         + ax.get_xticklabels() + ax.get_yticklabels())
                leg = ax.get_legend()
                if leg is not None:
                    texts += list(leg.get_texts())
                for t in texts:
                    t.set_color(fg)
                    if text_size:
                        t.set_fontsize(text_size)
        except Exception:
            pass

    @staticmethod
    def _figure_format_is_pdf() -> bool:
        try:
            from ..preferences import get_figure_format
            return get_figure_format() == "pdf"
        except Exception:
            return False

    def _display_pixmap(self, idx: int,
                        fallback: Optional[QPixmap]) -> Optional[QPixmap]:
        """What to show for figure ``idx`` **right now**, plus a refinement.

        Returns the cheap PNG-derived pixmap immediately and, in PDF mode,
        dispatches the 2200 px vector-page render to a worker thread;
        :meth:`_on_pdf_rendered` swaps the crisper result in when it lands.
        The figure therefore appears at once and then sharpens, instead of the
        window freezing until the crisp render is ready.
        """
        self._request_pdf_refinement(idx)
        return fallback

    def _request_pdf_refinement(self, idx: int) -> None:
        """Start the crisp vector-page render for ``idx`` off the GUI thread.

        Only for the figure actually on screen, and that is a decision rather
        than a shortcut: a pipeline streams figures through
        :meth:`add_figure` one after another, and refining every one at
        2200 px would start a worker thread per figure to produce pixmaps
        nobody ever looks at — a worse problem than the freeze this replaces.
        Every other figure is refined the moment it is navigated to, which is
        the first moment the result could be seen.

        At most one render is in flight per slot: ``_pdf_state`` holds the
        token while it runs and ``"done"`` / ``"failed"`` afterwards, so this
        is a no-op on repeat visits.
        """
        if idx != self._current or idx in self._pdf_state:
            return
        png = self._png_paths.get(idx)
        if not png:
            return
        # Normally gated on the PDF preference. But a page written earlier,
        # under a preference since switched to PNG, is still a vector page on
        # disk -- and "load the PDF if it exists" is exactly what the dynamic
        # figures option promises for a figure whose live Figure is gone.
        if not self._figure_format_is_pdf():
            if not (self.dynamic_figures_enabled()
                    and not self.has_live_figure(idx)
                    and _sibling_pdf(png).is_file()):
                return
        pdf = _sibling_pdf(png)
        if not pdf.is_file():
            # PDF mode is on and the vector page is not there: the export
            # failed (:func:`_export_vector_pdf` logged why) or this slot was
            # filled from a prerender that never carried one. Silently
            # returning made that indistinguishable from a page still on its
            # way, which is how a broken PDF export stayed invisible. Record
            # it as failed instead — one log line, and no re-stat of the same
            # missing file on every subsequent navigation to this figure.
            LOG.warning(
                "figure #%d has no vector page at %s — showing the raster; "
                "the PDF export did not happen", idx + 1, pdf)
            self._pdf_state[idx] = "failed"
            return
        self._pdf_seq += 1
        token = self._pdf_seq
        self._pdf_state[idx] = token
        path = str(pdf)
        # The submitted callable runs on a worker thread: it touches no
        # widget, no member of this object, and builds a QImage rather than a
        # QPixmap. Everything it needs is bound as a default argument.
        self._jobs.submit(
            lambda _i=idx, _t=token, _p=path: (_i, _t, render_pdf_to_image(_p)),
            self._on_pdf_rendered)

    def _on_pdf_rendered(self, payload: Optional[Tuple]) -> None:
        """Swap in a finished crisp render. Always on the GUI thread.

        Three separate things can have happened while the worker rendered, and
        each is checked here rather than assumed away:

        * the slot was re-pointed at a different figure by
          :meth:`_refresh_live_figure`, or the queue was cleared — caught by
          the token, which no longer matches;
        * the user navigated away — caught by the index check. The result is
          dropped rather than cached, so the RAM window stays exactly the
          sliding window of what has been *viewed*, and the next visit renders
          it again;
        * the PDF would not render at all — remembered as ``"failed"`` so a
          broken page is not retried on every single navigation.
        """
        if not isinstance(payload, tuple) or len(payload) != 3:
            return
        idx, token, image = payload
        if self._pdf_state.get(idx) != token:
            return
        del self._pdf_state[idx]
        if image is None or image.isNull():
            self._pdf_state[idx] = "failed"
            return
        if idx != self._current or not (0 <= idx < self._count):
            return
        # QPixmap is GUI-thread-only, which is why the worker returned a
        # QImage and the conversion happens here.
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            self._pdf_state[idx] = "failed"
            return
        self._pdf_state[idx] = "done"
        self._cache_pixmap(idx, pixmap)
        item = self._list.item(idx)
        if item is not None:
            item.setIcon(QIcon(pixmap.scaled(
                140, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        self._view.set_pixmap(pixmap)

    def _render_figure(self, fig, png_path: Path) -> Optional[QPixmap]:
        """Save ``fig`` to ``png_path`` (raster, for display) and return a
        QPixmap of it. The figure background + text follow the app theme
        (dark → black bg + white text) unless overridden in figure settings."""
        if not render_figure_to_png(fig, str(png_path)):
            return None
        pm = QPixmap(str(png_path))
        return pm if not pm.isNull() else None

    #: Longest edge of the throwaway raster drawn while a control is moving.
    #: Small enough to be instant, large enough to judge a legend or a colour
    #: by. The real render lands the moment the dialog closes.
    PREVIEW_MAX_PX = 1100

    def show_live_canvas(self, fig) -> bool:
        """Show ``fig`` through matplotlib itself. True if the canvas is up.

        This is what makes a figure crisp: the canvas redraws from the Figure
        at the widget's device resolution every time it changes size or zoom,
        so there is never a raster being stretched to fit. It is also what
        makes it fast -- looking at a figure costs no render at all.
        """
        if not self._live_canvas_enabled:
            return False
        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg, NavigationToolbar2QT)
        except Exception as error:  # pragma: no cover - no Qt backend
            LOG.debug("no Qt matplotlib backend, staying on the raster: %s",
                      error)
            return False
        try:
            if self._canvas is not None and self._canvas.figure is fig:
                self._canvas.draw_idle()
                self._stack.setCurrentIndex(1)
                return True
            self._teardown_canvas()
            canvas = FigureCanvasQTAgg(fig)
            # THE CANVAS DOES NOT PAINT A BACKGROUND EITHER. It is the surface
            # actually showing the figure most of the time, so a transparent
            # raster view with an opaque canvas in front of it is no better
            # than before. Qt's widget base and matplotlib's own figure patch
            # are two different opaque layers and both have to go.
            canvas.setStyleSheet("background: transparent;")
            canvas.setAttribute(Qt.WA_TranslucentBackground, True)
            canvas.setAutoFillBackground(False)
            # AND ITS PALETTE. matplotlib's paintEvent erases the rect before
            # blitting the Agg buffer, and eraseRect fills with the widget's
            # palette brush -- which is Base, i.e. white, whatever the
            # stylesheet says. That erase is why a figure whose patch is
            # already 'none' still sits on a white rectangle.
            transparent = canvas.palette()
            for role in (QPalette.Window, QPalette.Base):
                transparent.setColor(role, QColor(0, 0, 0, 0))
            canvas.setPalette(transparent)
            # matplotlib's Qt backend sets WA_OpaquePaintEvent, which tells Qt
            # nothing is behind this widget and it need not clear -- so the
            # canvas must paint every pixel itself, and it paints the ones the
            # Agg buffer left transparent as WHITE. That single attribute is
            # why a figure with facecolor 'none' still shows a white plot
            # rectangle with a transparent margin around it.
            canvas.setAttribute(Qt.WA_OpaquePaintEvent, False)
            canvas.setAttribute(Qt.WA_NoSystemBackground, True)
            # Right-click must still restyle, exactly as on the raster view.
            canvas.setContextMenuPolicy(Qt.CustomContextMenu)
            canvas.customContextMenuRequested.connect(self._view_context_menu)
            toolbar = NavigationToolbar2QT(canvas, self._canvas_host)
            # Pan and zoom re-render from the figure, so zooming in gives more
            # detail rather than bigger pixels.
            self._canvas_layout.addWidget(toolbar)
            self._canvas_layout.addWidget(canvas, 1)
            self._canvas = canvas
            self._canvas_toolbar = toolbar
            self._stack.setCurrentIndex(1)
            canvas.draw_idle()
            return True
        except Exception as error:  # noqa: BLE001 - fall back to the raster
            LOG.debug("live canvas failed, staying on the raster: %s", error)
            self._teardown_canvas()
            return False

    def set_live_canvas_enabled(self, enabled: bool) -> None:
        """Turn the live canvas off to force the raster pipeline.

        The raster path is not legacy -- a figure spilled past the live window
        or loaded from a PDF has no Figure to draw and can only be a picture.
        This makes that path reachable on demand, so the machinery that keeps
        it off the GUI thread stays under test.
        """
        self._live_canvas_enabled = bool(enabled)
        if not enabled:
            self._show_raster()

    def _teardown_canvas(self) -> None:
        """Drop the current canvas. A Figure may only live on one canvas.

        MATPLOTLIB'S EVENT WIRING MUST GO FIRST.

        NavigationToolbar2QT connects BOUND METHODS of itself to the canvas --
        mouse_move, the zoom/pan handlers -- and those live in the figure's
        callback registry, which the Figure owns and which outlives both
        widgets. deleteLater() destroys the C++ side while Python still holds
        the wrapper, so the next mouse move over the panel calls
        toolbar.mouse_move -> locLabel.setText on a dead QLabel and raises

            RuntimeError: libshiboken: Internal C++ object
            (PySide6.QtWidgets.QLabel) already deleted.

        once per mouse event -- thousands of tracebacks, and the same again
        for the canvas via set_cursor. Disconnecting before deleting leaves
        nothing holding a pointer into freed memory.
        """
        canvas, toolbar = self._canvas, self._canvas_toolbar
        if canvas is not None:
            # The toolbar's own connection ids, then anything else left on
            # the registry: a stale callback of any kind is a crash here.
            for attribute in ("_id_press", "_id_release", "_id_drag",
                              "_id_zoom", "_id_pan"):
                cid = getattr(toolbar, attribute, None)
                if cid is not None:
                    try:
                        canvas.mpl_disconnect(cid)
                    except Exception:  # pragma: no cover - already gone
                        pass
            # Anything else still bound to the two dying widgets. Disconnected
            # through mpl_disconnect rather than by clearing the registry:
            # matplotlib keeps its OWN entries in there (the pylab figure
            # manager's _cidgcf among them) and emptying the dict behind its
            # back makes its later disconnect raise KeyError instead.
            try:
                doomed = {id(canvas), id(toolbar)}
                for signal, entries in list(canvas.callbacks.callbacks.items()):
                    for cid, proxy in list(entries.items()):
                        owner = getattr(getattr(proxy, "func", None),
                                        "__self__", None)
                        if owner is not None and id(owner) in doomed:
                            canvas.mpl_disconnect(cid)
            except Exception:  # pragma: no cover - registry shape changed
                pass
        for widget in (toolbar, canvas):
            if widget is not None:
                try:
                    self._canvas_layout.removeWidget(widget)
                    widget.setParent(None)
                    widget.deleteLater()
                except Exception:  # pragma: no cover - already destroyed
                    pass
        self._canvas = None
        self._canvas_toolbar = None

    def _show_raster(self) -> None:
        """Fall back to the pixmap view, for a figure that is only a picture."""
        self._teardown_canvas()
        self._stack.setCurrentIndex(0)

    def _preview_target_px(self) -> float:
        """Longest edge, in real device pixels, of the area showing the figure.

        Rendering to a fixed cap and letting Qt scale the result up to the
        view is what made a restyled figure look "super pixelated": the
        preview was 1100 px, the view is larger than that on a normal screen,
        and the difference was made up by interpolation. Rendering at the size
        it will actually be displayed costs no more and is sharp.
        """
        try:
            size = self._view.size()
            ratio = float(self._view.devicePixelRatioF() or 1.0)
            longest = max(size.width(), size.height()) * ratio
            # A sane floor for a view that has not been laid out yet, and a
            # ceiling so a maximised 4K window does not ask for a 6000 px draw.
            return float(min(max(longest, 600.0), 2400.0))
        except Exception:  # pragma: no cover - headless
            return float(self.PREVIEW_MAX_PX)

    def _render_preview_async(self, fig) -> bool:
        """Draw the live preview on a worker thread. True if it was started.

        An Agg draw of the volcano is ~110 ms, of which the 27-entry legend
        alone is ~63 ms, and none of that gets cheaper by lowering the
        resolution -- the cost is text layout and marker-path geometry, not
        pixels. Run on the GUI thread it is felt as lag on every single
        control change no matter how it is debounced.

        Agg releases the GIL while it draws, so the same work on a worker
        thread stalls the GUI by ~1 ms over idle. The figure is copied first
        because the user goes on moving controls while the worker draws, and
        mutating a figure mid-draw is a crash rather than a glitch; the copy
        costs ~14 ms, which is inside a frame.

        :returns: False if the figure could not be copied, in which case the
            caller should fall back to rendering synchronously.
        """
        import pickle

        # One draw in flight at a time. The copy is cheap but not free, and a
        # worker per control change would spend the interaction copying
        # figures whose renders are stale before they land. A change arriving
        # mid-draw is remembered and drawn once, from the figure as it stands
        # when the worker frees up.
        if self._preview_busy:
            self._preview_pending = True
            return True

        try:
            blob = pickle.dumps(fig)
        except Exception as error:  # noqa: BLE001 - artists may not pickle
            LOG.debug("figure will not copy, rendering inline: %s", error)
            return False

        self._preview_busy = True
        self._preview_seq += 1
        token = self._preview_seq
        target = self._preview_target_px()
        facecolor = fig.get_facecolor()

        # Runs on a worker thread: it touches no widget and no member of this
        # object, and returns a QImage because QPixmap is GUI-thread-only.
        def work(_blob=blob, _target=target, _token=token,
                 _idx=self._current, _face=facecolor):
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            copy = pickle.loads(_blob)
            longest = max(copy.get_size_inches()) or 1.0
            copy.set_dpi(max(min(_target / longest, 300.0), 30.0))
            copy.patch.set_facecolor(_face)
            canvas = FigureCanvasAgg(copy)
            canvas.draw()
            width, height = canvas.get_width_height()
            # .copy() detaches from the canvas buffer, which is freed with it.
            image = QImage(canvas.buffer_rgba(), width, height,
                           QImage.Format_RGBA8888).copy()
            return (_idx, _token, image)

        self._jobs.submit(work, self._on_preview_rendered)
        return True

    def _on_preview_rendered(self, payload) -> None:
        """Show a finished preview. Always on the GUI thread.

        Only the newest render is shown: the user keeps changing controls
        while a draw is in flight, so earlier results are stale by the time
        they land and painting them would make the figure flicker backwards.
        """
        self._preview_busy = False
        # PAINT BEFORE STARTING THE NEXT ONE. Starting it first bumps the
        # sequence, and this payload -- freshly drawn, perfectly good -- would
        # then be discarded as stale by its own successor, so a continuous
        # drag would show nothing at all until the user stopped moving.
        self._paint_preview(payload)

        # Whatever changed while this was drawing still has to reach the
        # picture, and now there is a free worker to draw it.
        if self._preview_pending:
            self._preview_pending = False
            pending = self.figure_for(self._current)
            if pending is not None:
                self._render_preview_async(pending)

    def _paint_preview(self, payload) -> None:
        """Put a finished preview on screen, if it is still the current one."""
        if not isinstance(payload, tuple) or len(payload) != 3:
            return
        idx, token, image = payload
        if token != self._preview_seq or idx != self._current:
            return
        if image is None or image.isNull():
            return
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            return
        self._cache_pixmap(idx, pixmap)
        # A refinement started before this restyle would repaint the OLD
        # picture over the new one when it lands.
        self._pdf_state.pop(idx, None)
        self._view.set_pixmap(pixmap)
        item = self._list.item(idx)
        if item is not None:
            item.setIcon(QIcon(pixmap.scaled(
                140, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)))

    def _render_preview(self, fig, png_path: Path) -> Optional[QPixmap]:
        """A fast raster for live restyling: no vector page, capped size.

        :func:`render_figure_to_png` also exports the sibling PDF at the
        preference DPI, which is most of the cost of a render and pointless
        twenty times a second while a slider moves. This draws straight to a
        buffer at a modest size and does not touch the disk at all, so the
        saved files keep the last FULL render until the dialog closes.
        """
        try:
            from io import BytesIO

            longest = max(fig.get_size_inches()) or 1.0
            dpi = max(min(self.PREVIEW_MAX_PX / longest, 160.0), 40.0)
            buffer = BytesIO()
            # NO bbox_inches='tight' HERE. It measures the tight box by doing a
            # complete extra draw, which on the volcano is a flat ~125 ms on
            # top of the ~150 ms render -- the single largest cost in the live
            # path, and it buys only trimmed whitespace nobody is looking at
            # mid-drag. The full render on dialog close still trims.
            fig.savefig(buffer, format="png", dpi=dpi,
                        facecolor=fig.get_facecolor())
            pixmap = QPixmap()
            if pixmap.loadFromData(buffer.getvalue(), "PNG"):
                return pixmap
        except Exception as error:  # noqa: BLE001 - a preview may always fail
            LOG.debug("preview render failed: %s", error)
        return None

    def _cache_pixmap(self, idx: int, pixmap: QPixmap) -> None:
        """Insert into the LRU RAM cache, evicting the oldest beyond the
        cap. The PNG on disk is untouched, so an evicted figure can be
        reloaded on demand."""
        self._ram[idx] = pixmap
        self._ram.move_to_end(idx)
        while len(self._ram) > self._ram_cap:
            old_idx, _ = self._ram.popitem(last=False)
            LOG.debug("spilled figure #%d from RAM (PNG kept)", old_idx)

    def live_figure_cap(self) -> int:
        """How many recent figures keep their live matplotlib Figure."""
        try:
            from ..preferences import get_figure_live_cache
            return int(get_figure_live_cache())
        except Exception:  # pragma: no cover - headless / no QSettings
            return 20

    def dynamic_figures_enabled(self) -> bool:
        """Whether an evicted figure reloads from its vector page on demand."""
        try:
            from ..preferences import get_figure_dynamic
            return bool(get_figure_dynamic())
        except Exception:  # pragma: no cover
            return True

    def live_figure_count(self) -> int:
        """How many live Figures are currently retained."""
        return len(self._figures)

    def _trim_live_figures(self) -> None:
        """Keep only the most recent N live Figures, SPILLING the rest.

        A live Figure is what makes a figure restylable -- it still has a
        legend to toggle and axes to rescale. A pixmap is a picture of one.
        But each Figure holds its own data arrays, and a screen emits dozens,
        so every one retained forever is a leak in all but name.

        An evicted Figure is therefore pickled to the temp directory before it
        is closed. That is the whole point: a pickled Figure restores as a
        REAL Figure, with its artists, its data and its scales, so an old
        figure is fully editable again rather than only recolourable.

        The alternative considered was editing the saved vector page. A PDF
        does allow a stroke to be recoloured, a width changed, a font resized
        or grid paths deleted -- but not anything data-bound, because a log
        axis has to recompute every position. Pickling costs disk instead of
        RAM, which is exactly the trade the cap exists to make, and gives back
        everything rather than a subset. Measured on a scatter-plus-imshow
        figure: 1.55 MB, 4 ms to write, 3 ms to restore.

        A figure that cannot be pickled -- a custom artist, a live callback --
        is closed anyway and falls back to its rendered page. Failing to spill
        must never cost the cap.
        """
        cap = max(int(self.live_figure_cap()), 1)
        if len(self._figures) <= cap:
            return
        import matplotlib.pyplot as plt

        # LEAST RECENTLY USED, not lowest-numbered. Trimming by index looks
        # equivalent while figures only ever arrive in order -- but the moment
        # an old figure is restored so the user can restyle it, index order
        # says it is the oldest and evicts the very figure just asked for.
        # `_figures` is insertion-ordered and every access moves its key to
        # the end, so the front of it is genuinely the coldest.
        for old in list(self._figures)[:len(self._figures) - cap]:
            figure = self._figures.pop(old, None)
            self._spill_figure(old, figure)
            # Close it, or matplotlib's own registry keeps it alive and the
            # cap frees nothing.
            try:
                plt.close(figure)
            except Exception:  # pragma: no cover - defensive
                pass

    def _spill_path(self, idx: int) -> Optional[Path]:
        """Where ``idx``'s pickled Figure lives, if the temp dir exists."""
        if self._tempdir is None:
            return None
        return self._tempdir / f"fig_{idx:05d}.pkl"

    def _spill_figure(self, idx: int, figure) -> bool:
        """Pickle ``figure`` beside its rendered page. True when written."""
        if figure is None:
            return False
        path = self._spill_path(idx)
        if path is None:
            return False
        import pickle

        try:
            with open(path, "wb") as handle:
                pickle.dump(figure, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as error:  # noqa: BLE001 - spilling is best effort
            LOG.debug("figure %d could not be spilled: %s", idx, error)
            try:
                path.unlink()
            except OSError:
                pass
            return False
        return True

    def has_live_figure(self, idx: int) -> bool:
        """Whether ``idx``'s Figure is in memory right now.

        A query, not a use: it does not promote the entry, so asking whether
        something is live cannot change what gets evicted next.
        """
        return idx in self._figures

    def is_restorable(self, idx: int) -> bool:
        """Whether ``idx`` can be made editable again from its spill."""
        if idx in self._figures:
            return True
        path = self._spill_path(idx)
        return bool(path and path.is_file())

    def figure_for(self, idx: int):
        """The live Figure for ``idx``, restoring it from spill if needed.

        This is what a restyling menu asks for. A figure inside the live
        window is returned directly; one past it is unpickled, put back into
        the live set (so repeated edits do not re-read the disk) and the cap
        re-applied. Returns ``None`` only when the figure was never spillable.
        """
        if idx in self._figures:
            self._figures.move_to_end(idx)     # asked for = most recently used
            return self._figures[idx]
        if not self.dynamic_figures_enabled():
            return None
        path = self._spill_path(idx)
        if not (path and path.is_file()):
            return None
        import pickle

        try:
            with open(path, "rb") as handle:
                figure = pickle.load(handle)
        except Exception as error:  # noqa: BLE001
            LOG.debug("figure %d could not be restored: %s", idx, error)
            return None
        self._figures[idx] = figure
        self._trim_live_figures()
        return self._figures.get(idx, figure)

    def _pixmap_for(self, idx: int) -> Optional[QPixmap]:
        """Return the full-res pixmap for ``idx`` — from RAM if resident,
        otherwise reloaded from the temp PNG (and re-cached).

        When the live Figure for ``idx`` has been released and *dynamic
        figures* is on, the vector page is preferred over the display-capped
        raster: navigating back to an old figure then shows the PDF rather
        than a soft enlargement of a thumbnail-grade image.
        """
        if idx in self._ram:
            self._ram.move_to_end(idx)   # mark as recently used
            return self._display_pixmap(idx, self._ram[idx])
        path = self._png_paths.get(idx)
        if path and Path(path).is_file():
            pm = QPixmap(path)
            if not pm.isNull():
                # Reaching here means the RAM copy was evicted, and with it
                # any crisp render this slot had. ``"done"`` no longer holds,
                # so clear it and let the refinement run again.
                self._pdf_state.pop(idx, None)
                self._cache_pixmap(idx, pm)
                pixmap = self._display_pixmap(idx, pm)
                # The figure itself is gone, so nothing will re-render it from
                # source. Its vector page is the only remaining way to show it
                # sharply, and this is the moment the user asked for it.
                if (not self.has_live_figure(idx)
                        and self.dynamic_figures_enabled()
                        and _sibling_pdf(path).is_file()):
                    self._request_pdf_refinement(idx)
                return pixmap
        return None

    def _on_row_changed(self, row: int) -> None:
        if 0 <= row < self._count and row != self._current:
            self.show_index(row)

    def _refresh_nav(self) -> None:
        self._pos_label.setText(
            f"{self._current + 1} / {self._count}" if self._count
            else "0 / 0")
        # Figure settings (background/text colour + size) restyle the figure
        # and re-render, so they apply in both PNG and PDF mode — show whenever
        # there's a figure to tweak.
        self._fig_settings_btn.setVisible(self._count > 0)

    def _shutdown_jobs(self, timeout_ms: int = 2000) -> None:
        """Stop every in-flight crisp render and wait briefly for its thread.

        Must run **before** :meth:`_delete_tempdir`, and the ordering is not
        cosmetic. A worker is reading its PDF out of that directory:
        :func:`render_pdf_to_image` tolerates the file vanishing, but Qt
        aborts the process if a running QThread is destroyed, and the runner
        (and its threads) go with this widget. ``JobRunner.shutdown`` also
        bumps the generation, so a result that arrives anyway is dropped
        instead of being handed to a widget on its way out.

        Bounded, never unbounded: a render that outlasts the budget is parked
        by :func:`spacr.qt.bridge.drain_thread` rather than terminated, so
        closing cannot hang.
        """
        jobs = getattr(self, "_jobs", None)
        if jobs is None:
            return
        try:
            jobs.shutdown(timeout_ms=timeout_ms)
        except Exception:
            # Reachable from __del__, where the C++ half may already be gone.
            LOG.debug("figure render shutdown failed", exc_info=True)

    def _delete_tempdir(self) -> None:
        if self._tempdir is not None:
            try:
                shutil.rmtree(self._tempdir, ignore_errors=True)
            except Exception:
                pass
            self._tempdir = None

    # -- lifecycle ---------------------------------------------------------

    def closeEvent(self, event):
        self._shutdown_jobs()
        self._delete_tempdir()
        super().closeEvent(event)

    def __del__(self):
        # Best-effort temp cleanup if the widget is GC'd without close. The
        # workers go first — they read out of the directory about to be
        # removed, and a live QThread must not be left holding a runner whose
        # last reference is being dropped right now.
        try:
            self._shutdown_jobs()
        except Exception:
            pass
        try:
            self._delete_tempdir()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Figure settings dialog — restyle a matplotlib figure (PDF/vector mode)
# ---------------------------------------------------------------------------

class _FigureSettingsDialog(QDialog):
    """Adjust a figure's background colour, text colour and text size, then
    re-render. Only offered for vector (PDF) figures.

    An Image UMAP figure gets a second half: every Image UMAP setting, live
    against the figure on screen (instruction 75). The section only appears
    for a figure that carries ``_spacr_umap_payload`` — the embedding it was
    drawn from — because without the embedding "live" would mean re-running
    the reduction, and every point would move.

    NO API DOTS. The three teal dots this dialog used to draw are gone; the
    same help, with the same ``href``, is on the labels' hover tooltips, and
    a form whose every row carries a dot reads as a column of dots rather
    than a column of settings. Same change, same reason, as the Mask live
    preview, the Annotate settings and the UMAP search dialog before it.
    """

    def __init__(self, fig, parent=None, propagate_callback=None,
                 render_callback=None):
        super().__init__(parent)
        self._fig = fig
        self._propagate_cb = propagate_callback
        self._render_cb = render_callback
        self.setWindowTitle("Figure settings")
        from PySide6.QtWidgets import (
            QFormLayout, QDialogButtonBox, QSpinBox, QPushButton as _QPB,
            QVBoxLayout as _QVBox, QWidget as _QWidget)
        outer = _QVBox(self)
        holder = _QWidget(self)
        form = QFormLayout(holder)
        form.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(holder)

        try:
            from ..preferences import get_figure_colors, get_figure_text_size
            self._bg, self._fg = get_figure_colors()
            _init_size = get_figure_text_size() or 10
        except Exception:
            self._bg, self._fg, _init_size = "#ffffff", "#000000", 10
        self._bg_btn = _QPB("Background…")
        self._bg_btn.clicked.connect(lambda: self._pick("_bg", self._bg_btn))
        self._fg_btn = _QPB("Text colour…")
        self._fg_btn.clicked.connect(lambda: self._pick("_fg", self._fg_btn))
        form.addRow("Background", self._bg_btn)
        form.addRow("Text colour", self._fg_btn)

        self._bg_btn.setStyleSheet(f"background-color: {self._bg};")
        self._fg_btn.setStyleSheet(f"background-color: {self._fg};")

        self._size = QSpinBox()
        self._size.setRange(4, 48)
        self._size.setValue(int(_init_size))
        form.addRow("Text size", self._size)

        self._umap_settings = None
        self._umap_payload = getattr(fig, "_spacr_umap_payload", None)
        if isinstance(self._umap_payload, dict):
            self._build_umap_section(outer)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._propagate_btn = _QPB("Propagate settings")
        self._propagate_btn.setToolTip(
            "Write these values into the module's settings panel, so the "
            "next run starts from them and they are saved with it.")
        self._propagate_btn.clicked.connect(self._propagate)
        self._propagate_btn.setEnabled(callable(propagate_callback))
        if not callable(propagate_callback):
            self._propagate_btn.setToolTip(
                "Available on a module screen, which is what owns the "
                "settings panel these values would be written into.")
        bb.addButton(self._propagate_btn, QDialogButtonBox.ActionRole)
        bb.accepted.connect(self._apply_and_accept)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)
        from ..screens.settings_model import install_api_tooltips
        install_api_tooltips(self, "figure", {
            self._bg_btn: "figure_background",
            self._fg_btn: "figure_text_color",
            self._size: "figure_text_size",
        }, api_dots=False)
        if self._umap_settings is not None:
            # Scoped to the section rather than to the dialog: a second sweep
            # over the whole dialog would re-decorate the three figure
            # controls under the "umap" app key, and their documentation
            # lives under "figure".
            install_api_tooltips(self._umap_settings, "umap", api_dots=False)

    # -- the Image UMAP half -----------------------------------------------

    def _build_umap_section(self, outer) -> None:
        """Add every Image UMAP setting, live against this figure."""
        from PySide6.QtWidgets import QScrollArea
        from .umap_figure_settings import UmapFigureSettings

        values = dict(self._umap_payload.get("settings") or {})
        self._umap_settings = UmapFigureSettings(values, self)
        self._umap_settings.settings_changed.connect(self._on_umap_changed)
        area = QScrollArea(self)
        area.setWidgetResizable(True)
        area.setWidget(self._umap_settings)
        area.setMinimumHeight(320)
        outer.addWidget(area, 1)
        self._umap_applied = dict(self._umap_settings.values())

    def _on_umap_changed(self, values: dict) -> None:
        """Push a changed Image UMAP setting at the figure, now.

        The embedding is read, never recomputed — see
        :func:`spacr.qt.widgets.umap_figure_settings.redraw_umap_figure`.
        """
        from .umap_figure_settings import apply_to_figure

        mode = apply_to_figure(self._fig, self._umap_payload, values,
                               getattr(self, "_umap_applied", {}))
        self._umap_applied = dict(values)
        if mode and callable(self._render_cb):
            try:
                self._render_cb()
            except Exception:
                LOG.debug("could not re-render the figure", exc_info=True)

    def umap_values(self) -> dict:
        """Every Image UMAP setting the window holds, or ``{}``."""
        if self._umap_settings is None:
            return {}
        return self._umap_settings.values()

    def _propagate(self) -> None:
        """Send the current values into the module's settings panel."""
        if not callable(self._propagate_cb):
            return
        values = dict(self.umap_values())
        values.update({
            "figure_background": self._bg,
            "figure_text_color": self._fg,
            "figure_text_size": int(self._size.value()),
        })
        try:
            self._propagate_cb(values)
        except Exception:
            LOG.debug("could not propagate the figure settings", exc_info=True)

    def reject(self):
        """Put the figure back the way the window found it, then close.

        Live apply with no way out is a trap: the user drags a spin box to
        see what it does and there is no longer an "as it was". Cancel is
        that way out, and it costs one redraw.
        """
        settings = self._umap_settings
        if settings is not None:
            initial = settings.initial_values()
            if initial != getattr(self, "_umap_applied", initial):
                self._on_umap_changed(initial)
        super().reject()

    def _pick(self, attr, btn):
        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtGui import QColor
        c = QColorDialog.getColor(QColor(getattr(self, attr)), self)
        if c.isValid():
            setattr(self, attr, c.name())
            btn.setStyleSheet(f"background-color: {c.name()};")

    def _apply_and_accept(self):
        """Persist the chosen colours/size (so every figure follows them) and
        apply them to this figure, then accept — the caller re-renders."""
        size = int(self._size.value())
        try:
            from ..preferences import set_figure_colors, set_figure_text_size
            set_figure_colors(self._bg, self._fg)
            set_figure_text_size(size)
        except Exception:
            pass
        if self._umap_settings is not None:
            # A value still sitting on the debounce timer is a value the user
            # typed and would otherwise lose by pressing OK promptly.
            self._umap_settings.flush()
        FigureQueue._style_figure(self._fig, self._bg, self._fg, size)
        self.accept()
