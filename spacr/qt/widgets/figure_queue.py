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

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
    QVBoxLayout, QWidget,
)

from ..job_runner import JobRunner
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
        fig.savefig(png_path, dpi=display_dpi, bbox_inches="tight",
                    facecolor=bg)
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


class FigureQueue(QWidget):
    """Scrollable, RAM-bounded gallery of pipeline figures."""

    def __init__(self, ram_cap: int = RAM_CAP, parent=None):
        super().__init__(parent)
        self._ram_cap = int(ram_cap)
        self._count = 0
        # id(fig) -> index, for dedup of repeated emits of the same fig.
        self._fig_index: Dict[int, int] = {}
        # index -> temp PNG path (every figure has one).
        self._png_paths: Dict[int, str] = {}
        # index -> matplotlib Figure (kept so the figure-settings dialog can
        # restyle + re-render it).
        self._figures: Dict[int, object] = {}
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
        self._list.setSpacing(4)
        self._list.currentRowChanged.connect(self._on_row_changed)
        body.addWidget(self._list)

        self._view = _ZoomView(self)
        self._view.setMinimumHeight(280)
        body.addWidget(self._view, 1)
        root.addLayout(body, 1)

        # Navigation is via the thumbnail strip (click a thumbnail) — no
        # separate Prev/Next buttons. A "Figure settings…" button (shown only
        # when figures are rendered as PDF/vector) restyles the current figure.
        nav = QHBoxLayout()
        self._pos_label = QLabel("0 / 0", self)
        self._pos_label.setAlignment(Qt.AlignCenter)
        self._fig_settings_btn = QPushButton("Figure settings…", self)
        self._fig_settings_btn.clicked.connect(self._open_figure_settings)
        nav.addWidget(self._pos_label, 1)
        nav.addWidget(self._fig_settings_btn)
        root.addLayout(nav)
        self._refresh_nav()

    def _open_figure_settings(self) -> None:
        """Open the figure-settings dialog for the current figure."""
        fig = self._figures.get(self._current)
        if fig is None:
            return
        dlg = _FigureSettingsDialog(fig, self)
        if dlg.exec():
            # Re-render the restyled figure in place.
            png = self._png_paths.get(self._current)
            if png:
                pm = self._render_figure(fig, Path(png))
                if pm is not None:
                    self._cache_pixmap(self._current, pm)
                    # The sibling .pdf was rewritten too, so any crisp render
                    # already cached for this slot is of the OLD styling.
                    self._pdf_state.pop(self._current, None)
                    self._view.set_pixmap(
                        self._display_pixmap(self._current, pm))

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
        if not self._figure_format_is_pdf():
            return
        png = self._png_paths.get(idx)
        if not png:
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

    def _cache_pixmap(self, idx: int, pixmap: QPixmap) -> None:
        """Insert into the LRU RAM cache, evicting the oldest beyond the
        cap. The PNG on disk is untouched, so an evicted figure can be
        reloaded on demand."""
        self._ram[idx] = pixmap
        self._ram.move_to_end(idx)
        while len(self._ram) > self._ram_cap:
            old_idx, _ = self._ram.popitem(last=False)
            LOG.debug("spilled figure #%d from RAM (PNG kept)", old_idx)

    def _pixmap_for(self, idx: int) -> Optional[QPixmap]:
        """Return the full-res pixmap for ``idx`` — from RAM if resident,
        otherwise reloaded from the temp PNG (and re-cached)."""
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
                return self._display_pixmap(idx, pm)
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
    re-render. Only offered for vector (PDF) figures."""

    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self._fig = fig
        self.setWindowTitle("Figure settings")
        from PySide6.QtWidgets import (
            QFormLayout, QDialogButtonBox, QSpinBox, QPushButton as _QPB)
        form = QFormLayout(self)

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

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._apply_and_accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        from ..screens.settings_model import install_api_tooltips
        install_api_tooltips(self, "figure", {
            self._bg_btn: "figure_background",
            self._fg_btn: "figure_text_color",
            self._size: "figure_text_size",
        })

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
        FigureQueue._style_figure(self._fig, self._bg, self._fg, size)
        self.accept()
