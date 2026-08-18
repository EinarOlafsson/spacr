"""Click a gene, see everything spaCR knows about it -- in one panel.

Instruction 121. Two modules already hold the two halves of the answer and
neither of them is a widget, which is deliberate: the parts that can be WRONG
are testable without a window.

    :mod:`spacr.gene_tile`    WHICH gene this dot is -- guide to gene, the
                              ambiguous protospacers, and THIS SCREEN's own
                              effect / p / q / guides, read out of the results
                              frame that is already on screen.
    :mod:`spacr.gene_facts`   WHAT that gene is -- product, topology with the
                              DeepTMHMM coordinates, hyperLOPIT compartment,
                              the published CRISPR fitness screens and the
                              stage expression, all of it out of
                              :mod:`spacr.annotation`.

This module puts them one above the other and does nothing else with the
numbers. THE PANEL IS NOT A SECOND SOURCE OF TRUTH: the coefficient, the
p-value, the q-value and the guide agreement are whatever the table on screen
says they are, because two places computing one number is how they start
disagreeing.

THE GUI THREAD DOES NOT READ FILES
----------------------------------

Cold, the first click costs 360 ms of CSV reading -- five bundled annotation
tables, DeepTMHMM's 8,140 rows, and the gRNA reference and metadata indices
:mod:`spacr.gene_tile` keeps -- inside a mouse press. A plot that freezes for
a third of a second when clicked reads as broken.

So :func:`warm_annotation` does all of it on a worker thread, through
:class:`spacr.qt.job_runner.JobRunner`, which is the module that already
gets the threading rules right: the worker's ``finished`` is relayed through
a Signal whose receiver is a BOUND METHOD of a GUI-thread object, never a
closure, so the handler runs on the GUI thread and the QThread is retired.
Given the screen's own terms it warms every gene in one join -- 400 genes
cost the same 21 ms as one -- after which a click is a dictionary lookup.

UNTIL IT IS WARM, THE PANEL SAYS SO, and the one control it has is greyed out
with the reason on it (instruction 106). A gene with no annotation likewise
says "no row in the bundled annotation" rather than showing a form of empty
fields, which reads as "measured, found nothing".
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Sequence, Tuple

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout,
                               QLabel, QPushButton, QSizePolicy, QSplitter,
                               QTextBrowser, QVBoxLayout, QWidget)

from ..theme import SPACING
from .gene_tile import TILE_WIDTH, GeneTilePanel

LOG = logging.getLogger("spacr.qt.gene_panel")

__all__ = ["GenePanel", "warm_annotation"]

#: Shown in the lower half before the annotation has finished loading. Not a
#: blank: a blank pane beside a plot reads as broken rather than as busy.
LOADING_TEXT = "Loading the bundled Toxoplasma annotation…"

#: Shown in the lower half before anything has been clicked.
IDLE_TEXT = ("What spaCR knows about the gene appears here — product,"
             " topology, compartment, fitness screens and stage expression.")

#: The term used to build :mod:`spacr.gene_tile`'s three indices during the
#: warm-up when the caller passed no terms of its own. Any term does; this one
#: is a control guide, so it resolves without depending on a gene being in the
#: bundled reference.
_WARMING_TERM = "fraction:grna[000000_1]"


def warm_annotation(features: Sequence[Any] = ()) -> Tuple[str, ...]:
    """Read every table a gene tile needs. RUNS ON A WORKER THREAD.

    :param features: the terms the user might click -- pass the results
        table's whole ``feature`` column. Every gene among them is joined in
        one pass, which costs the same as joining one.
    :returns: the annotation columns that came out available; empty when the
        bundled tables are not installed, which is a state the panel shows
        rather than hides.

    Touches no widget and returns only data -- that is the contract for
    anything handed to :meth:`spacr.qt.job_runner.JobRunner.submit`.

    It warms :mod:`spacr.gene_tile` too, by resolving one term. That module
    keeps its own indices over the gRNA reference and the curated metadata,
    and they are just as cold on the first click as the annotation tables
    are; warming one and not the other would move the freeze rather than
    remove it.
    """
    from ... import gene_facts, gene_tile

    columns = gene_facts.warm(features)
    try:
        gene_tile.gene_tile(features[0] if len(features) else _WARMING_TERM)
    except Exception:                                          # noqa: BLE001
        # The warm-up is an optimisation. A reference file this install does
        # not have must not stop the panel from opening -- the click path
        # says what it could not resolve, which is the answer either way.
        LOG.debug("gene panel: could not warm the gene_tile indices",
                  exc_info=True)
    return columns


class GenePanel(QWidget):
    """The gene tile: which gene this is, and everything known about it.

    :param frame_provider: called with no arguments for the current results
        frame. A callable rather than a stored frame so a newly loaded
        regression is never answered out of the previous one.
    :param threaded: ``False`` warms inline instead of on a worker thread, so
        a test can drive the panel synchronously without the behaviour
        diverging -- :class:`~spacr.qt.job_runner.JobRunner` emits the same
        signals in the same order either way.
    :param parent: the usual.
    """

    #: Emitted with the feature string whenever a tile is built for it.
    tile_shown = Signal(str)

    #: Emitted with the available column count once the annotation is loaded;
    #: ``0`` means the bundled tables are not installed.
    annotation_ready = Signal(int)

    def __init__(self, frame_provider: Optional[Callable[[], Any]] = None,
                 *, threaded: bool = True, parent=None):
        super().__init__(parent)
        from ..job_runner import JobRunner

        self._columns: Tuple[str, ...] = ()
        self._warm = False
        self._facts: Tuple[Any, ...] = ()
        #: The terms the last warm-up covered. A panel is re-set_frame'd with
        #: the SAME table whenever the gene/guide filter moves, and a QThread
        #: per redraw is a thread for no new genes at all.
        self._warmed: Tuple[Any, ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.get("xs", 4))

        split = QSplitter(Qt.Vertical, self)
        #: The record half: identity, this screen's numbers, the guides.
        self.summary = GeneTilePanel(frame_provider=frame_provider)
        split.addWidget(self.summary)

        self._known = QTextBrowser()
        self._known.setOpenLinks(False)
        self._known.setOpenExternalLinks(False)
        # A gene id must survive translation intact: TGGT1_239740 is not a
        # phrase, and a catalog that "translated" it would be renaming a gene.
        self._known.setProperty("i18nSkipText", True)
        self._known.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        split.addWidget(self._known)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 4)
        self.split = split
        layout.addWidget(split, 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: palette(mid); font-size: 10px;")
        footer.addWidget(self._status, 1)
        #: Writes this gene's full DeepTMHMM record -- every segment's
        #: coordinates -- as its own CSV. It is the one thing on this panel
        #: that leaves a file behind, so it is the one thing that has to say
        #: why it cannot: see :meth:`topology_reason`.
        self.topology_button = QPushButton("Save topology CSV…")
        self.topology_button.clicked.connect(self._ask_to_save_topology)
        footer.addWidget(self.topology_button)
        layout.addLayout(footer)

        self.clear()

        self._runner = JobRunner(self, threaded=bool(threaded),
                                 app_key="gene annotation",
                                 user_visible=False)
        self._runner.submit(warm_annotation, self._annotation_loaded)
        # BELT AND BRACES ON THE THREAD'S LIFETIME. Qt aborts the process if
        # a running QThread is destroyed, and a panel can be dropped without
        # ever being closed -- a tab rebuilt, a screen replaced, an
        # interpreter shutting down. `closeEvent` covers the ordinary path;
        # this covers the one where nobody closed anything.
        application = QApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self._shut_down_warming)

    # ------------------------------------------------------------------ state

    @property
    def tile(self):
        """The :class:`spacr.gene_tile.GeneTile` on screen, or ``None``."""
        return self.summary.tile

    @property
    def feature(self) -> str:
        """The feature string the current tile was built from."""
        return self.summary.feature

    @property
    def facts(self) -> Tuple[Any, ...]:
        """One :class:`spacr.gene_facts.GeneFacts` per candidate gene."""
        return self._facts

    def is_warm(self) -> bool:
        """Has the annotation finished loading off the GUI thread?"""
        return self._warm

    def annotation_columns(self) -> Tuple[str, ...]:
        """The annotation columns this install can show. Empty until warm."""
        return self._columns

    def set_frame_provider(self, provider: Optional[Callable[[], Any]]) -> None:
        """Point the panel at where the current results frame lives."""
        self.summary.set_frame_provider(provider)

    def clear(self) -> None:
        """Back to the waiting state, which says what it is waiting for."""
        self.summary.clear()
        self._facts = ()
        self._say(LOADING_TEXT if not self._warm else IDLE_TEXT)
        self._update_topology_button()

    # ------------------------------------------------------------------ warming

    def warm_for(self, frame) -> bool:
        """Warm the annotation for every gene in ``frame``. Call on load.

        :param frame: the results table that was just loaded.
        :returns: whether a warm-up was started.

        The terms are read off the frame HERE, on the GUI thread, because
        that is a list comprehension over a column that is already in memory.
        The join they feed is what goes to the worker.
        """
        terms = tuple(_terms_of(frame))
        if not terms or terms == self._warmed:
            return False
        self._warmed = terms
        return bool(self._runner.submit(
            lambda features=list(terms): warm_annotation(features),
            self._annotation_loaded))

    def _annotation_loaded(self, columns) -> None:
        """The warm-up landed. GUI THREAD ONLY -- JobRunner guarantees it."""
        self._columns = tuple(columns or ())
        self._warm = True
        if self.summary.feature:
            # A click that beat the warm-up is re-answered rather than left
            # showing "loading" over a tile that is already on screen.
            self.show_feature(self.summary.feature)
        else:
            self._say(IDLE_TEXT)
        self._update_topology_button()
        self.annotation_ready.emit(len(self._columns))

    # ------------------------------------------------------------------ slots

    def show_feature(self, key) -> None:
        """Build and show the tile for one clicked feature.

        THE SLOT ``key_selected`` CONNECTS TO. It takes the feature string and
        nothing else, so a volcano click and a results-row click reach it
        identically -- and it is connected once, on the table, because that is
        the funnel both directions already pass through.

        Driven straight rather than off ``summary.tile_shown``: that signal
        is not emitted when the resolver RAISES, and the one case where the
        lower half must not be left showing the previous gene is exactly the
        one where the upper half failed.
        """
        self.summary.show_feature(key)
        self._render_known(self.summary.tile)
        self._update_topology_button()
        self.tile_shown.emit(self.summary.feature or str(key))

    def _render_known(self, tile) -> None:
        import html as _html

        from ... import gene_facts

        self._facts = ()
        if tile is None:
            self._say("The gene could not be resolved, so there is nothing "
                      "to look up. The plot is unaffected.")
            return
        if not self._warm:
            self._say(LOADING_TEXT)
            return

        reason = _nothing_to_look_up(tile)
        if reason:
            self._say(reason)
            return

        unavailable = gene_facts.unavailable_reason()
        if unavailable:
            self._say(unavailable)
            return

        found = gene_facts.facts_for(c.gene for c in tile.candidates)
        self._facts = tuple(found[c.gene] for c in tile.candidates
                            if c.gene in found)
        if not self._facts:
            self._say("No gene number could be parsed out of this term, so "
                      "there is no annotation to look up.")
            return

        parts: List[str] = []
        for candidate, known in zip(tile.candidates, self._facts):
            if len(self._facts) > 1:
                # Named per gene, because the whole point of the ambiguous
                # case is that these blocks are alternatives and not one
                # record: three products under one heading would read as one
                # protein with three names.
                parts.append("<h3 style='margin-bottom:0'>what spaCR knows "
                             f"about {_html.escape(candidate.name)}</h3>")
            parts.append(known.to_html())
        self._known.setHtml("".join(parts))
        self._status.setText("")

    def _say(self, text: str) -> None:
        """Put a sentence in the lower half instead of a table of facts."""
        import html as _html

        self._known.setHtml(
            f"<p style='color:#888'>{_html.escape(text)}</p>")
        self._status.setText("")

    # ------------------------------------------------------------------ topology

    def topology_reason(self) -> str:
        """Why "Save topology CSV" cannot run, or ``""`` when it can.

        Instruction 106: a control that cannot do anything is greyed out AND
        says why. This is the sentence, and it is the button's tooltip.
        """
        if not self._warm:
            return ("The DeepTMHMM table is still loading. The button turns "
                    "on when it is ready.")
        if not self._columns:
            return ("The bundled Toxoplasma annotation is not installed with "
                    "this copy of spaCR, so there is no topology to save.")
        if not self._facts:
            return "Click a gene first — there is no gene to save topology for."
        genes = [known for known in self._facts if known.segments]
        if not genes:
            named = ", ".join(known.gene for known in self._facts)
            return (f"DeepTMHMM found no signal peptide and no transmembrane "
                    f"segment in {named}, so its topology table would be "
                    f"empty.")
        return ""

    def _update_topology_button(self) -> None:
        reason = self.topology_reason()
        self.topology_button.setEnabled(not reason)
        self.topology_button.setToolTip(
            reason or "Write this gene's full DeepTMHMM record — every "
                      "segment's coordinates — as a CSV.")

    def save_topology(self, path) -> bool:
        """Write the clicked gene's full DeepTMHMM record to ``path``.

        :returns: whether a file was written.

        Straight through :func:`spacr.annotation.supplementary`, which is the
        function that defines what that table is. Rewriting the columns here
        would be a second definition of the supplementary file, differing
        from the one an export writes in ways nobody would notice until a
        reviewer compared them.
        """
        from ... import annotation

        genes = [known.gene for known in self._facts if known.gene]
        if not genes:
            return False
        return annotation.supplementary(genes, path) is not None

    def _ask_to_save_topology(self) -> None:
        genes = "_".join(known.gene for known in self._facts if known.gene)
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save DeepTMHMM topology",
            f"deeptmhmm_{genes or 'gene'}.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            self.save_topology(path)
        except Exception as error:                             # noqa: BLE001
            LOG.exception("gene panel: could not write the topology table")
            self._status.setText(f"Could not write {path}: {error}")
        else:
            self._status.setText(f"Topology written to {path}")

    # ------------------------------------------------------------------ grid

    def to_pixmap(self, width: int = TILE_WIDTH) -> QPixmap:
        """The whole tile -- both halves -- as one ``QPixmap``.

        The figure grid's cells take a pixmap and size themselves from its
        aspect ratio, so rendering to one lets the gene tile be a tile in
        that grid without ``_FigureCell`` learning about text.
        """
        top = self.summary.to_pixmap(width)
        bottom = _document_pixmap(self._known.toHtml(), width)
        height = max(top.height() + bottom.height(), 1)
        out = QPixmap(QSize(max(int(width), 1), height))
        out.fill(Qt.transparent)
        painter = QPainter(out)
        try:
            painter.drawPixmap(0, 0, top)
            painter.drawPixmap(0, top.height(), bottom)
        finally:
            painter.end()
        return out

    # ------------------------------------------------------------------ close

    def _shut_down_warming(self) -> None:
        """Stop the warm-up. A BOUND METHOD, and that is the point.

        It is connected to ``QApplication.aboutToQuit``; a closure there
        would make the application object the receiver and the call would be
        dropped when this panel is destroyed first, leaving the very thread
        it was meant to stop still running.
        """
        try:
            self._runner.shutdown()
        except RuntimeError:
            # The C++ half can already be gone when a whole window closes at
            # once; there is nothing left to shut down and nothing to report.
            pass

    def closeEvent(self, event):                                # noqa: N802
        """Stop the warm-up before the widget goes.

        Qt aborts the process if a running QThread is destroyed, and a
        warm-up outliving its panel is exactly that.
        """
        self._shut_down_warming()
        super().closeEvent(event)


def _terms_of(frame) -> List[Any]:
    """The clickable terms of a results frame, or ``[]``.

    ``feature`` is the key every plot and the table join on -- see
    :class:`spacr.qt.widgets.regression_results.RegressionResultsPanel`. A
    frame without it is not a coefficient table and there is nothing here to
    warm.
    """
    columns = getattr(frame, "columns", None)
    if columns is None or "feature" not in columns:
        return []
    try:
        return list(frame["feature"])
    except Exception:                                          # noqa: BLE001
        return []


def _nothing_to_look_up(tile) -> str:
    """Why this tile has no gene to annotate, or ``""``.

    A control guide, a model covariate and an unrecognised string are three
    DIFFERENT answers and each of them is an answer. The one thing none of
    them may produce is a panel of empty fields, which reads as "measured,
    found nothing".
    """
    if tile.kind == "control":
        return ("This is a non-targeting control guide. There is no gene "
                "behind it, so there is nothing to annotate — its effect is "
                "the assay's own baseline.")
    if tile.kind == "nuisance":
        return (f"{tile.feature} is a model covariate, not a gene. It is "
                "fitted so the real effects are estimated cleanly and is not "
                "itself a hypothesis.")
    if not tile.candidates:
        return (tile.unresolved[0] if tile.unresolved else
                f"{tile.feature} does not name a gene spaCR recognises.")
    return ""


def _document_pixmap(document_html: str, width: int) -> QPixmap:
    """One HTML document rendered to a pixmap of the given width."""
    from PySide6.QtGui import QTextDocument

    document = QTextDocument()
    document.setHtml(document_html)
    document.setTextWidth(max(int(width), 1))
    size = document.size().toSize()
    pixmap = QPixmap(QSize(max(int(width), 1), max(size.height(), 1)))
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    try:
        document.drawContents(painter)
    finally:
        painter.end()
    return pixmap
