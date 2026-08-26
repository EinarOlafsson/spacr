"""Regression, and the three modules that are the rest of the visit.

A regression run is not finished when the coefficients appear. Three
screens carry on from there, and none of them duplicates anything the
results panel does:

* **Volcano Explorer** is the only publication-figure path in spaCR. Its
  volcano is a matplotlib render behind a 56-field style that can be
  saved and reloaded as JSON, every called point labelled, colour or
  shape driven by an annotation file merged on an inferred key, arbitrary
  x/y columns, and a vector re-render at journal size. The panel's own
  volcano is pyqtgraph and can honour none of it, so the explorer is
  offered from that plot's own menu as "Publication figure…", seeded with
  the frame already on screen.
* **Hit List** displays one ranked row per gene for the loaded regression
  run. Backends that report p-values receive Benjamini–Hochberg q-values
  computed across the genes tested; penalised backends instead rank by
  bootstrap selection frequency and do not report q-values. Each row includes
  the effect estimate, a 95% interval when a standard error is available, and
  gRNA sign agreement when guide-level coefficients are available. Annotation
  CSVs are collapsed to one row per gene before validated many-to-one joins.
  The complete :class:`~spacr.qt.screens.hit_list.HitListScreen` is installed
  as the **Hits** tab after Guide support, follows the run loaded by the
  results panel, and exports the displayed filtered list as CSV, Markdown, or
  self-contained HTML.
* **Methods & Results** builds the run digest -- package versions,
  timings, seed and error policy, per-module parameters parsed out of the
  emitted macro, the segmentation verdict, artifact counts, held-out
  metrics -- drafts the two sections from it and then mechanically checks
  every number in the draft back against the digest. It opens seeded with
  the project the regression screen is already pointed at, so the path it
  otherwise asks the user to type is filled in.

Each of the three is also a button on the Regression masthead: the
module's own icon with no text, its one-line description as the tooltip,
lit on hover in the maturity colour its tile used -- see
:class:`spacr.qt.widgets.fold_strip.FoldStrip`. The button and the place
the capability lives are the same door: the Hit List button raises the
Hits tab rather than opening a second hit list, and the Volcano Explorer
button opens exactly what "Publication figure…" opens.

ONE CORRECTION FAMILY ON ONE VOLCANO. A guide permutation writes every
minimum-support family stacked into one long table, and fits one family
per response when several outcomes were fitted. Each of those is its own
Benjamini-Hochberg family, so drawing the table unfiltered puts a guide on
the plot two to four times at different heights and pools two corrections
into one picture. :func:`single_correction_family` is the cut, and
:func:`install_correction_families` puts it in front of the panel's own
``set_frame`` so every route into the panel -- a finished run, a folder
opened by hand, a dropped bundle -- draws one family.

The shared half of a fold -- opening a module in a window, wiring the host
signals a sidebar row used to wire, and hanging the strip off the
masthead -- lives in :mod:`spacr.qt.screens.map_barcodes` and is imported
rather than repeated.
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import Callable, Dict, Optional, Tuple

from PySide6.QtWidgets import QWidget

from ..widgets.fold_strip import FoldStrip
from .map_barcodes import FoldOpener, restate_fold_button

LOG = logging.getLogger(__name__)

#: Registry key of the screen this module hangs its strip on.
HOST_KEY = "regression"

#: Registry keys of the modules folded into it, in the order the strip
#: draws them: the figure, then the list, then the write-up -- which is
#: the order the three are wanted in after a run finishes.
FOLDED_APPS: Tuple[str, ...] = ("volcano_explorer", "hit_list",
                                "methods_export")

# What each of those three said as a TILE -- the name, the sentence and the
# maturity colour a button has to go on carrying once the row is dropped --
# lives in `spacr.qt.screens.map_barcodes.FOLD_FALLBACK`, because
# `map_barcodes.fold_description` is what `restate_fold_button` reads, and
# that is the only table it looks in. A second copy stood here, beside these
# three keys, and nothing consulted it.

#: What the "Hits" tab is called, and the tab it is inserted after.
HITS_TAB_TITLE = "Hits"
HITS_TAB_AFTER = "Guide support"

#: The directory a regression writes its runs into, inside the project
#: -- ``<project>/results/<score>/<kind>``. Named so
#: :func:`project_path` can walk back up to the project the digest
#: wants without a second spelling of the layout.
RESULTS_DIRNAME = "results"

#: The entry the regression volcano's own right-click menu grows.
PUBLICATION_FIGURE_LABEL = "Publication figure…"

#: The heading that entry sits under. Its own section, like the re-fit
#: entry: everything above restyles the plot on screen, and this hands the
#: same rows to a different renderer.
PUBLICATION_FIGURE_SECTION = "Publication figure"

# ---------------------------------------------------------------------------
# One correction family per volcano
# ---------------------------------------------------------------------------

def single_correction_family(frame):
    """``frame`` reduced to ONE multiple-testing family, ready to plot.

    A guide permutation fits the same guides once per minimum-support
    threshold and once per response, stacks the lot into one long table,
    and corrects each stack separately. Two things follow, and both are
    wrong on a plot: the same guide is drawn two to four times at
    different heights, and two Benjamini-Hochberg corrections share one
    axis -- so a q-value read off the picture belongs to whichever family
    the point came from, which nothing on the picture says.

    The primary family is the smallest ``minimum_wells_threshold``, which
    is what ``perform_regression`` writes ``results.csv`` from when
    ``guide_primary_min_wells`` is left blank; where several responses
    were fitted the first is kept, because each response is its own
    correction family and the explorer's column controls can switch.

    A frame with neither column is returned unchanged, which is every
    parametric run -- so this costs an ordinary table nothing.

    :param frame: a coefficient table, or None.
    :returns: the same object when there was nothing to cut, otherwise a
        new frame holding one family.
    """
    if frame is None:
        return frame
    columns = list(getattr(frame, "columns", ()))
    if not columns or not len(frame):
        return frame
    cut = frame
    if "minimum_wells_threshold" in columns:
        thresholds = cut["minimum_wells_threshold"].dropna().unique()
        if len(thresholds) > 1:
            cut = cut.loc[cut["minimum_wells_threshold"] == min(thresholds)]
    if "outcome" in columns and cut["outcome"].nunique() > 1:
        cut = cut.loc[cut["outcome"] == cut["outcome"].iloc[0]]
    if cut is frame:
        return frame
    return cut.reset_index(drop=True)


def install_correction_families(panel) -> bool:
    """Put :func:`single_correction_family` in front of ``panel.set_frame``.

    Every route into the results panel ends in ``set_frame`` -- a finished
    run, a folder opened through "Load results…", a dropped bundle, a
    frame handed straight in -- so one wrapper there is what makes the
    volcano show one correction family whichever way the table arrived.

    Idempotent, so installing twice does not stack two cuts.

    :param panel: the :class:`RegressionResultsPanel`, or None.
    :returns: True when this call installed the cut.
    """
    if panel is None or getattr(panel, "_one_correction_family", False):
        return False
    original = getattr(panel, "set_frame", None)
    if not callable(original):
        return False

    def set_frame(frame, source: str = "") -> bool:
        return original(single_correction_family(frame), source=source)

    panel.set_frame = set_frame
    panel._one_correction_family = True
    return True


# ---------------------------------------------------------------------------
# The publication figure
# ---------------------------------------------------------------------------

def _run_folder(panel) -> str:
    """The run folder behind ``panel``, or "" when it came from nowhere."""
    reader = getattr(panel, "run_folder", None)
    if not callable(reader):
        return ""
    try:
        return str(reader() or "")
    except Exception:
        LOG.debug("Could not read the panel's run folder", exc_info=True)
        return ""


def build_publication_figure(panel, host_window=None) -> QWidget:
    """Volcano Explorer's own screen, seeded with what is on screen.

    The frame is preferred over the folder: the panel may be showing a
    table nobody can find again -- a live run, a bare CSV, a frame handed
    in -- and re-reading the folder would draw a different table from the
    one the user asked to publish. The folder is the fallback, and the
    explorer's own "Open results…" is there when there is neither.

    Every capability the explorer has arrives with it, because what is
    built is the module itself: the 56-field style with its JSON save and
    load, the annotation merge, arbitrary x/y columns and the vector
    re-render at page size.

    :param panel: the results panel to seed from, or None.
    :param host_window: the main window, for navigation. Unused by the
        explorer today; taken so the builder matches every other fold's.
    :returns: the Volcano Explorer screen.
    """
    from . import volcano as volcano_module

    screen = volcano_module._make_screen(host=host_window)
    frame = None
    reader = getattr(panel, "results_frame", None)
    if callable(reader):
        try:
            frame = reader()
        except Exception:
            LOG.debug("Could not read the panel's frame", exc_info=True)
    if frame is not None and len(frame):
        screen.explorer.set_results(single_correction_family(frame))
        folder = _run_folder(panel)
        screen._path_label.setText(folder or "The run on screen")
    else:
        folder = _run_folder(panel)
        if folder:
            screen.load(folder)
    return screen


def install_publication_figure(panel, opener: Callable[[], object]) -> bool:
    """Offer ``opener`` as "Publication figure…" on the panel's volcano.

    The entry goes on the pyqtgraph volcano's own right-click menu, under
    a section of its own, because that is where a user who wants the
    figure is already looking -- and it is deliberately NOT ``offer_style``
    on that plot, which would hang 56 style fields off a renderer that can
    honour none of them.

    The menu is built fresh on every right-click, so the entry is added by
    wrapping the builder rather than by holding a menu: a plot that
    rebuilt its menu would otherwise drop the entry the first time the
    user changed anything.

    Idempotent. Returns False when there is no volcano to offer it on.
    """
    plot = getattr(panel, "volcano", None) if panel is not None else None
    if plot is None or getattr(plot, "_publication_figure", False):
        return False
    build = getattr(plot, "build_style_menu", None)
    if not callable(build):
        return False

    def build_style_menu(*args, **kwargs):
        menu = build(*args, **kwargs)
        menu.addSection(PUBLICATION_FIGURE_SECTION)
        entry = menu.addAction(PUBLICATION_FIGURE_LABEL,
                               lambda: opener())
        entry.setToolTip(
            "The same rows through the publication renderer: a saved style, "
            "labelled points, an annotation file joined on, and a vector "
            "re-render at journal size. This plot is drawn for speed and "
            "cannot do any of it.")
        return menu

    plot.build_style_menu = build_style_menu
    plot._publication_figure = True
    return True


# ---------------------------------------------------------------------------
# The Hits tab
# ---------------------------------------------------------------------------

def install_hits_tab(panel):
    """Add the Hit List to ``panel`` as a tab, beside Guide support.

    The whole screen goes in, not a copy of its table: the filter bar, the
    metadata picker and the three export buttons ARE the capability the
    panel has none of, and a tab that reimplemented the list would keep
    whichever parts the person doing the folding thought of.

    The tab follows the panel: whenever a run is loaded the hit list is
    pointed at the same folder, so it is never showing one run's hits
    beside another run's coefficients.

    Idempotent.

    :param panel: the results panel.
    :returns: the :class:`~spacr.qt.screens.hit_list.HitListScreen`, or
        None when the tab could not be built.
    """
    if panel is None:
        return None
    existing = getattr(panel, "hits", None)
    if existing is not None:
        return existing
    tabs = getattr(panel, "tabs", None)
    if tabs is None:
        return None
    from .hit_list import HitListScreen

    try:
        hits = HitListScreen(parent=panel, folder=_run_folder(panel))
    except Exception:
        LOG.exception("Could not build the Hits tab")
        return None
    index = tabs.count()
    for position in range(tabs.count()):
        if tabs.tabText(position) == HITS_TAB_AFTER:
            index = position + 1
            break
    tabs.insertTab(index, hits, HITS_TAB_TITLE)
    tabs.setTabToolTip(
        tabs.indexOf(hits),
        "One row per gene rather than per coefficient: the effect with its "
        "95% interval, a q-value recomputed over the genes actually tested, "
        "how many of the gene's own guides agree in sign, and any annotation "
        "CSV joined on. Export the exact list on screen as CSV, Markdown or "
        "a self-contained HTML page.")
    panel.hits = hits
    loaded = getattr(panel, "loaded", None)
    if loaded is not None:
        try:
            loaded.connect(lambda _path, p=panel: _follow_the_run(p))
        except Exception:
            LOG.debug("The Hits tab could not follow the panel", exc_info=True)
    return hits


def _follow_the_run(panel) -> None:
    """Point the Hits tab at whatever run the panel just loaded."""
    hits = getattr(panel, "hits", None)
    folder = _run_folder(panel)
    if hits is None or not folder:
        return
    try:
        hits.load_folder(folder)
    except Exception:
        LOG.debug("The Hits tab could not follow the run", exc_info=True)


def raise_hits_tab(panel) -> bool:
    """Bring the Hits tab to the front. False when there is none."""
    hits = getattr(panel, "hits", None) if panel is not None else None
    tabs = getattr(panel, "tabs", None) if panel is not None else None
    if hits is None or tabs is None:
        return False
    tabs.setCurrentWidget(hits)
    return True


# ---------------------------------------------------------------------------
# Methods & Results
# ---------------------------------------------------------------------------

def project_path(screen) -> str:
    """Resolve the project directory associated with a regression screen.

    The function first examines the run loaded in the results panel and
    returns the parent of its nearest ``results`` directory. If no project can
    be resolved from that run, it reads ``src`` from the screen's settings
    model. A sequence-valued ``src`` contributes its first entry.

    :param screen: Regression screen or compatible host.
    :returns: Project path, or ``""`` when it cannot be determined.
    """
    if screen is None:
        return ""
    folder = _run_folder(results_panel(screen))
    while folder and folder != os.path.dirname(folder):
        parent = os.path.dirname(folder)
        if os.path.basename(folder) == RESULTS_DIRNAME:
            return parent
        folder = parent
    model = getattr(screen, "_settings_model", None)
    collect = getattr(model, "collect", None) if model is not None else None
    if not callable(collect):
        return ""
    try:
        source = (collect() or {}).get("src")
    except Exception:
        LOG.debug("Could not read the regression source", exc_info=True)
        return ""
    if isinstance(source, (list, tuple)):
        source = source[0] if source else ""
    source = str(source or "").strip()
    if not source:
        return ""
    return os.path.abspath(os.path.expanduser(source))


def build_methods_export(host_window: Optional[QWidget] = None,
                         screen: Optional[QWidget] = None) -> QWidget:
    """Methods & Results' own screen, seeded with the run it will describe.

    Both sources the regression screen knows are filled in: the project,
    which supplies the provenance summary and the segmentation verdict,
    and the results folder, which supplies the hit statistics. The other
    two -- the run journal and the classifier checkpoint -- are the user's
    to name, and every one of them is optional on that screen.
    """
    from .methods_export import MethodsExportScreen

    panel = results_panel(screen)
    return MethodsExportScreen(project=project_path(screen),
                               results_folder=_run_folder(panel))


# ---------------------------------------------------------------------------
# The strip
# ---------------------------------------------------------------------------

def results_panel(screen):
    """``screen``'s results panel, or None on a screen that has none."""
    return getattr(screen, "_results_panel", None) if screen is not None \
        else None


def open_publication_figure(host_window: Optional[QWidget] = None,
                            screen: Optional[QWidget] = None
                            ) -> Optional[QWidget]:
    """Build the Volcano Explorer seeded from ``screen``'s results panel."""
    return build_publication_figure(results_panel(screen), host_window)


#: One builder per folded module that opens a WINDOW. Each takes the main
#: window and the host screen; :func:`install_folds` binds the screen, so a
#: builder still has the one-argument shape
#: :class:`spacr.qt.screens.map_barcodes.FoldOpener` calls it with.
#:
#: ``hit_list`` is not here: its home is a tab on the results panel, so its
#: button raises that tab rather than opening anything -- see
#: :class:`HitsOpener`, which falls back to a window on a screen that has no
#: panel to hold a tab.
BUILDERS: Dict[str, Callable[..., Optional[QWidget]]] = {
    "volcano_explorer": open_publication_figure,
    "methods_export": build_methods_export,
}


class HitsOpener:
    """The Hits button: raise the tab, or open the module when there is none.

    ONE HIT LIST, not two. Where the results panel exists the list is
    already a tab on it, loaded with the run on screen, so the button goes
    there -- and it brings the Results page forward first, because a button
    that silently changed a tab behind the settings form would read as a
    button that does nothing.

    A bare settings screen has no panel and so no tab; then the module
    opens in a window of its own, like every other fold, with the run
    folder seeded.

    :param screen: the host screen the button sits on.
    """

    key = "hit_list"

    def __init__(self, screen: QWidget) -> None:
        self.screen = screen
        self._window = FoldOpener(screen, self.key, self._build_window)

    def _build_window(self, _host_window) -> QWidget:
        """The Hit List module itself, seeded with the run on screen."""
        from .hit_list import HitListScreen

        return HitListScreen(folder=_run_folder(results_panel(self.screen)))

    def open(self, _checked: bool = False) -> Optional[QWidget]:
        """Show the hit list, wherever this screen keeps it."""
        panel = results_panel(self.screen)
        raise_results = getattr(self.screen, "_raise_the_results_tab", None)
        if panel is not None and callable(raise_results):
            try:
                raise_results()
            except Exception:
                LOG.debug("Could not raise the results page", exc_info=True)
        if raise_hits_tab(panel):
            return panel.hits
        return self._window.open()


def install_extras(screen: QWidget) -> bool:
    """Give ``screen``'s results panel the three folded capabilities.

    Separate from the strip because the strip is only the way IN: the Hits
    tab, the publication-figure entry and the one-family cut are on the
    panel whether or not a masthead could be found to hang buttons on.

    :returns: True when a panel was found and prepared.
    """
    if getattr(screen, "app_key", None) != HOST_KEY:
        return False
    panel = results_panel(screen)
    if panel is None:
        return False
    install_correction_families(panel)
    install_hits_tab(panel)
    install_publication_figure(panel, publication_opener(screen).open)
    return True


def publication_opener(screen) -> FoldOpener:
    """Return the shared Volcano Explorer opener for ``screen``.

    The opener is created on first use and retained on the screen. Both the
    **Publication figure…** command and the masthead action use this instance,
    so reopening the explorer raises the existing window rather than creating
    a duplicate with independent state.

    :param screen: Regression screen that owns the publication workflow.
    :returns: Persistent :class:`~spacr.qt.screens.map_barcodes.FoldOpener`.
    """
    opener = getattr(screen, "_publication_opener", None)
    if not isinstance(opener, FoldOpener):
        opener = FoldOpener(screen, "volcano_explorer",
                            partial(open_publication_figure, screen=screen))
        screen._publication_opener = opener
    return opener


def install_folds(screen: QWidget) -> Optional[FoldStrip]:
    """Put Regression's fold strip on ``screen``'s masthead.

    Built here rather than through
    :func:`spacr.qt.screens.map_barcodes.install_fold_strip` because one of
    the three buttons does not open a window: the Hits button raises a tab
    on the screen the user is already looking at.

    Idempotent, and defensive by design: a screen that opens without its
    fold buttons is a smaller screen, while an exception raised here would
    be no regression screen at all.

    :returns: the strip, or None when this screen cannot carry one -- it is
        not the host, it has no masthead, or one is already installed.
    """
    if getattr(screen, "app_key", None) != HOST_KEY:
        return None
    existing = getattr(screen, "_fold_strip", None)
    if isinstance(existing, FoldStrip):
        return existing
    header = getattr(screen, "_header", None)
    if header is None or not hasattr(header, "add_trailing"):
        return None
    openers = []
    for key in FOLDED_APPS:
        if key == HitsOpener.key:
            openers.append(HitsOpener(screen))
        elif key == "volcano_explorer":
            openers.append(publication_opener(screen))
        else:
            openers.append(FoldOpener(screen, key,
                                      partial(BUILDERS[key], screen=screen)))
    # The panel is prepared BEFORE the buttons exist, so no button can be
    # pressed into a panel that has not got its tab yet.
    install_extras(screen)
    try:
        strip = FoldStrip([(o.key, o.open) for o in openers], header)
        for opener in openers:
            restate_fold_button(strip.button_for(opener.key), opener.key)
        header.add_trailing(strip)
    except Exception:
        LOG.debug("Could not build the regression fold strip", exc_info=True)
        return None
    # The openers outlive this call only because the screen holds them.
    screen._fold_openers = openers
    screen._fold_strip = strip
    return strip
