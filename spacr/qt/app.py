"""
QApplication bootstrap + MainWindow.

`launch(argv)` is the public entry point called by `spacr-qt` and
`python -m spacr.qt`.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import traceback
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QStackedWidget,
    QStatusBar,
    QStyle,
    QVBoxLayout,
    QWidget,
)


from . import iconset
# Nothing in this module uses a colour, a spacing or the palette API any
# more — the Home page, the sidebar QSS and `apply_preferences_to_app`
# each own their own. The import stayed behind after they moved out, and
# because it named `PALETTE` it fired the deprecation warning on every
# `import spacr.qt.app` for a value nobody read.
from .widgets.eliding import ElidingPushButton

LOG = logging.getLogger(__name__)


class _UpdateWorker(QThread):
    """Run one updater operation without blocking the GUI event loop.

    :param operation: stable operation name used in error messages.
    :param fn: zero-argument callable whose return value is emitted.
    :param parent: Qt owner; normally the :class:`MainWindow`.
    """

    succeeded = Signal(object)
    failed = Signal(str, str)

    def __init__(self, operation, fn, parent=None):
        super().__init__(parent)
        self.operation = str(operation)
        self._fn = fn

    def run(self) -> None:
        """Execute the operation and surface every exception."""
        try:
            result = self._fn()
        except Exception:
            details = traceback.format_exc()
            LOG.exception("Updater %s failed", self.operation)
            self.failed.emit(self.operation, details)
            return
        self.succeeded.emit(result)


class _PipelinePreloader:
    """Warms up the heavy pipeline modules so the first click on
    Mask/Measure/Classify/etc. doesn't stall the UI while
    torch/cellpose/pandas/etc. resolve.

    IMPORTANT: preloading runs on the MAIN (GUI) thread, one module
    per event-loop tick, NOT on a background daemon thread. Importing
    C-extension modules that initialise CUDA/GL (torch, cellpose) from
    a non-main thread concurrent with Qt's own GPU init is a classic
    cause of intermittent "Segmentation fault (core dumped)" at
    startup. Each import here blocks the event loop only briefly, and
    a QTimer tick between imports lets Qt process repaints/clicks so
    the UI stays responsive without the off-thread race.
    """

    _MODULES = (
        "spacr.core",
        "spacr.measure",
        "spacr.deep_spacr",
        "spacr.ml",
        "spacr.sequencing",
        "spacr.submodules",
        "spacr.spacr_cellpose",
    )

    def __init__(self):
        self._i = 0
        self._started = False

    def start(self) -> None:
        """Begin the main-thread import chain (no-op if already begun)."""
        if self._started:
            return
        self._started = True
        self._i = 0
        self._step()

    def _step(self) -> None:
        """Import the next module, then schedule the following one on the
        next event-loop tick."""
        from PySide6.QtCore import QTimer
        if self._i >= len(self._MODULES):
            return
        mod = self._MODULES[self._i]
        self._i += 1
        try:
            import importlib
            importlib.import_module(mod)
        except Exception:
            # Preloading is optional, but an import failure still belongs in
            # the diagnostic log so a later first-use failure has context.
            LOG.debug("Could not preload %s", mod, exc_info=True)
        # 50 ms between imports so Qt drains its event queue (repaints,
        # input) before the next potentially-blocking import.
        QTimer.singleShot(50, self._step)


# ---------------------------------------------------------------------------
# The app registry
# ---------------------------------------------------------------------------
#
# Two axes, and only ONE of them is a place.
#
# An app is filed under WHAT IT DOES — Core, Data, Segmentation models,
# Results & QC, Toxoplasma. That is stable: Format Converter is a Data
# app whether or not anyone has finished it, and it is the axis the
# sidebar, the command palette and the Home tabs all group by.
#
# An app is *also* staged by HOW FINISHED IT IS — alpha, beta or stable
# — and 22 of the 30 are not yet signed off. #16i made that second axis
# two extra CATEGORIES, which meant three of the five subject tabs
# drained to nothing and "where is the format converter" acquired two
# answers. #16j undoes exactly that: maturity is now drawn as the
# tile's HOVER COLOUR, with a legend beside the tiles, so it is visible
# on the same tile that is in the right category. One table
# (:data:`APP_STAGE`), one place per app, no second grouping.
#
#   Core            the end-to-end pipeline: images in, single-object
#                   measurements out, hits called.
#   Data            get images and tables in, run many plates, get the
#                   numbers back out.
#   Segmentation    build, train, pick and check the Cellpose models
#     models        the Mask step runs.
#   Results & QC    read what came out and decide whether to believe it.
#   Toxoplasma      parasite-specific readouts.
#
# A section holds AT MOST `MAX_APPS_PER_SECTION` apps. Past that nobody
# reads the row — which is exactly how "Tools" grew to sixteen entries and
# became unusable. If a section is full, the honest fix is a new section
# with a name that means something, not a longer row.
#
# The names are as short as they can be and still mean something: they
# are TAB LABELS on Home, where long names would not fit on one line, and
# a tab that has to elide is a tab nobody can read.
SECTION_CORE = "Core"
SECTION_DATA = "Data"
SECTION_MODELS = "Segmentation models"
SECTION_RESULTS = "Results & QC"
SECTION_TOXO = "Toxoplasma"

#: Tab, sidebar and heading order, in workflow order: the end-to-end
#: pipeline first, then getting data in and running it at scale, then
#: the segmentation models that pipeline depends on, then reading the
#: results, then the Toxoplasma-specific assays.
SECTIONS = (SECTION_CORE, SECTION_DATA, SECTION_MODELS, SECTION_RESULTS,
            SECTION_TOXO)

#: Hard cap on apps per section. Enforced by tests, not at runtime — a
#: violation is a design mistake to fix in this table, not something to
#: discover at startup.
#:
#: Raised from 9 to 13 by #16i and kept there. Nine was the width of the
#: Core pipeline and nothing more; a cap that is exactly the size of the
#: biggest section is a cap that fires on the next app added rather than
#: when a section stops being readable.
MAX_APPS_PER_SECTION = 13

APPS = [
    # (key, human name, description, section)
    #
    # `section` is what the app IS ABOUT. How finished it is lives in
    # APP_STAGE below and is drawn as a colour, not as a place.
    #
    # NOTE: keys are load-bearing. bridge.resolve_pipeline_entry,
    # cli.INTERACTIVE_ONLY, validate.APP_FUNCTIONS, dnd_handlers,
    # settings_model.resolve_default_settings and saved user state all
    # key off them. Renaming a key silently breaks those; renaming the
    # display name or moving an app between sections is free.
    #
    # -- Core pipeline: images in, single-object measurements out, hits
    #    called. Ctrl+1..9 map to these nine, in this order.
    ("mask",           "Mask",           "Generate cellpose masks for cells, nuclei and pathogens",   SECTION_CORE),
    ("timelapse",      "Timelapse",      "Segment and track objects across the frames of a time series", SECTION_CORE),
    ("motility",       "Motility Assay", "Automated motility assay: track velocity + infection QC",     SECTION_CORE),
    ("measure",        "Measure",        "Measure single-object intensity + morphology features",       SECTION_CORE),
    ("annotate",       "Annotate",       "Annotate single-object images on a grid; save to database",  SECTION_CORE),
    ("classify",       "Classify (CV)",  "Train Torch CNNs/Transformers to classify single objects",   SECTION_CORE),
    ("ml_analyze",     "Classify (ML)",  "Classical ML (XGBoost / random forest / …) on screen features", SECTION_CORE),
    ("map_barcodes",   "Map Barcodes",   "Map sequencing barcodes to screen data",                      SECTION_CORE),
    ("regression",     "Regression",     "Regression analysis of screen scores",                        SECTION_CORE),
    # -- Data & batch runs: get images and tables into a spaCR project,
    #    run many plates unattended, get the numbers back out.
    ("align",          "Align & Stitch", "Register tiles into one stitched canvas, written incrementally so a 20000x20000 mosaic never has to fit in RAM", SECTION_DATA),
    ("convert",        "Format Converter", "ND2/CZI/LIF/OME-TIFF into Yokogawa TIFFs: preview the mapping, then a map file back to the originals", SECTION_DATA),
    ("foreign",        "Import Project", "Someone else's images, masks and measurement table into a spaCR project, with their columns mapped onto spaCR's", SECTION_DATA),
    ("external_masks", "External Masks", "Turn images and externally generated label masks into a measured spaCR project ready for annotation", SECTION_DATA),
    ("queue",          "Plate Queue",    "Chain multiple plates through the same pipeline",             SECTION_DATA),
    ("batch",          "Batch Runner",   "Queue any modules, plates and settings and run them overnight", SECTION_DATA),
    ("db_browser",     "Database Browser", "Browse and export measurements.db without the sqlite3 CLI", SECTION_DATA),
    # -- Segmentation models: build, train, pick and check the Cellpose
    #    models the Mask step runs.
    ("make_masks",     "Make Masks",     "Fine-tune Cellpose models for your dataset",                  SECTION_MODELS),
    ("train_cellpose", "Train Cellpose", "Train custom Cellpose models",                                SECTION_MODELS),
    ("cellpose_masks", "Cellpose Masks", "Cellpose mask generation",                                    SECTION_MODELS),
    ("model_compare",  "Model Compare",  "Two Cellpose models on the same fields: masks side by side, object-count and ARI deltas", SECTION_MODELS),
    ("model_zoo",      "Model Zoo",      "Browse, verify, download and bench Cellpose + classifier models on three of your fields", SECTION_MODELS),
    # -- Results & QC: look at what came out, decide whether to believe it,
    #    and hand it to someone else.
    ("plate_view",     "Plate Viewer",   "Any measurement as a plate heatmap + edge-effect detection",  SECTION_RESULTS),
    ("agreement",      "Annotator Agreement", "Cohen's/Fleiss' κ between annotation columns + a disagreement review", SECTION_RESULTS),
    ("umap",           "Image UMAP",     "Generate UMAP embeddings with image glyphs",                  SECTION_RESULTS),
    ("activation",     "Activation",     "Generate activation maps",                                    SECTION_RESULTS),
    ("train_compare",  "Training Runs",  "Overlay several training runs' curves with their settings diffed side by side", SECTION_RESULTS),
    ("report",         "Report",         "One-click shareable HTML/PDF: QC verdict, figures, stats, settings, versions", SECTION_RESULTS),
    # -- Toxoplasma assays: parasite-specific readouts.
    ("analyze_plaques", "Plaque Assay",  "Analyze plaque assay data",                                   SECTION_TOXO),
    ("recruitment",    "Recruitment",    "Analyze recruitment data",                                    SECTION_TOXO),
    ("invasion",       "Invasion Assay", "Two-colour outside/inside stain: attached vs invaded parasites, invasion efficiency per well", SECTION_TOXO),
    ("replication",    "Replication Assay", "Endodyogeny: parasites per vacuole, scored into replication rate per condition", SECTION_TOXO),
]


# ---------------------------------------------------------------------------
# Maturity — the second axis, drawn as colour rather than as a place
# ---------------------------------------------------------------------------

STAGE_STABLE = "stable"
STAGE_BETA = "beta"
STAGE_ALPHA = "alpha"

#: Every stage, least finished first — the order the legend lists them.
STAGES = (STAGE_ALPHA, STAGE_BETA, STAGE_STABLE)

#: **The** classification. app key → :data:`STAGE_ALPHA` or
#: :data:`STAGE_BETA`; anything absent is :data:`STAGE_STABLE`.
#:
#: This is the single table the hover colour and the legend both read
#: (through :func:`app_stage` and :func:`home_stages`), and it is the
#: only place maturity is written down — #16i's mistake was that
#: maturity was *also* a section, so an app could be alpha in one table
#: and Data in another and the two could disagree.
#:
#: Signing an app off is deleting its line here. Nothing else moves: the
#: app is already filed under what it does.
APP_STAGE = {
    # -- alpha: built and reachable, not yet trusted end to end (14)
    "align":           STAGE_ALPHA,
    "model_zoo":       STAGE_ALPHA,
    "convert":         STAGE_ALPHA,
    "foreign":         STAGE_ALPHA,
    "external_masks":  STAGE_ALPHA,
    "model_compare":   STAGE_ALPHA,
    "queue":           STAGE_ALPHA,
    "batch":           STAGE_ALPHA,
    "invasion":        STAGE_ALPHA,
    "db_browser":      STAGE_ALPHA,
    "plate_view":      STAGE_ALPHA,
    "agreement":       STAGE_ALPHA,
    "train_compare":   STAGE_ALPHA,
    "report":          STAGE_ALPHA,
    # -- beta: further along, in regular use, still not signed off (9)
    "make_masks":      STAGE_BETA,
    "train_cellpose":  STAGE_BETA,
    "cellpose_masks":  STAGE_BETA,
    "timelapse":       STAGE_BETA,
    "motility":        STAGE_BETA,
    "analyze_plaques": STAGE_BETA,
    "replication":     STAGE_BETA,
    "umap":            STAGE_BETA,
    "activation":      STAGE_BETA,
}


def app_stage(key: str) -> str:
    """How finished ``key`` is — one of :data:`STAGES`.

    Unknown keys read as stable rather than raising: a stage is an
    annotation on an app, and an app with no annotation is one nobody
    has flagged.
    """
    return APP_STAGE.get(key, STAGE_STABLE)


def app_is_visible(key: str) -> bool:
    """Whether ``key`` should appear in module navigation.

    Preferences are imported lazily so this registry remains safe for
    packaging and headless callers. If preferences cannot be read, preserve
    the historical all-modules-visible behaviour.
    """
    try:
        from .preferences import maturity_is_visible
        return maturity_is_visible(app_stage(key))
    except Exception:
        return True


def visible_apps() -> List[Tuple[str, str, str, str]]:
    """Registry rows allowed by the Alpha/Beta visibility preferences."""
    return [row for row in APPS if app_is_visible(row[0])]


def home_stages() -> dict:
    """app key → stage, for every app in :data:`APPS`.

    What :func:`make_home_page` hands the Home page, so the tiles and
    the legend colour from the same table this module owns and the
    widget still knows nothing about what a stage means.
    """
    return {row[0]: app_stage(row[0]) for row in APPS}


def section_members(
    section: str,
    apps: Optional[List[Tuple[str, str, str, str]]] = None,
) -> List[Tuple[str, str, str, str]]:
    """The ``APPS`` rows a category's tab shows, in registry order."""
    source = APPS if apps is None else apps
    return [row for row in source if row[3] == section]


def home_categories(
    apps: Optional[List[Tuple[str, str, str, str]]] = None,
) -> List[Tuple[str, List[str]]]:
    """``(section, [app key])`` for every tab after Home, in tab order.

    Computed rather than written down, so a section cannot acquire a tab
    it has no apps for or lose one it does.
    """
    result = []
    for section in SECTIONS:
        rows = section_members(section, apps)
        if rows:
            result.append((section, [row[0] for row in rows]))
    return result


def home_bands(
    apps: Optional[List[Tuple[str, str, str, str]]] = None,
) -> List[Tuple[str, List[Tuple[str, str, str, str]]]]:
    """``(band, rows)`` for the Home tab, in :data:`SECTIONS` order.

    The same grouping the tabs use — every app once, under what it is
    about. Home is the "all of it" view and each later tab is a filter
    of it, which is only true while both read this one table.

    Bands with no apps are dropped rather than drawn empty. None are,
    today; the guard is what stops a heading appearing over nothing the
    day a section's last app is retired.
    """
    grouped = [(s, section_members(s, apps)) for s in SECTIONS]
    return [(s, rows) for s, rows in grouped if rows]


#: One line per section, drawn under its heading on that category's tab.
#: A category with two apps in it looks broken until it says why.
SECTION_NOTES = {
    SECTION_CORE: "Images in, single-object measurements out, hits called.",
    SECTION_DATA: ("Images and tables into a spaCR project, many plates run "
                   "unattended, the numbers back out."),
    SECTION_MODELS: ("Build, train, pick and check the Cellpose models the "
                     "Mask step runs."),
    SECTION_RESULTS: ("Read what came out, decide whether to believe it, "
                      "hand it to someone else."),
    SECTION_TOXO: "Parasite-specific readouts.",
}


def make_home_page(parent=None):
    """Build the Home page exactly as the running app builds it.

    The grouping, the stages, the notes and the icon provider are four
    arguments that have to agree, and a test that assembles its own
    HomePage is testing a page that does not ship — which is the exact
    class of bug #16i was: Home grouped by one table and the tabs by
    another. One constructor call, used by :class:`MainWindow` and by
    the suite.
    """
    from .widgets.home import HomePage
    apps = visible_apps()
    return HomePage(
        apps, _icon_for_app, parent,
        section_notes=SECTION_NOTES,
        categories=home_categories(apps),
        bands=[(s, [r[0] for r in rows]) for s, rows in home_bands(apps)],
        stages=home_stages())


# Explicit key -> icon-filename overrides for cases where the app_key
# doesn't match any resource filename. Add entries here rather than
# renaming resource files.
_ICON_OVERRIDES = {
    "analyze_plaques": "plaque.png",
    "train_cellpose":  "cellpose_masks.png",  # share the Cellpose Masks icon
    "agreement":       "annotate.png",     # shares the Annotate glyph: it
                                           # scores annotation columns
    "plate_view":      "map_barcodes.png", # ruled bars read as a well grid
    "model_compare":   "mask.png",         # mask.png is one field split down
                                           # the middle -- raw objects on one
                                           # side, contours on the other. That
                                           # IS Model Compare: the same field,
                                           # segmented two ways, side by side.
    "model_zoo":       "download.png",     # the zoo is where models come from
    #
    # FOUR entries were REMOVED here — `timelapse`→run.png,
    # `motility`→recruitment.png, `db_browser`→map_barcodes.png and
    # `train_compare`→classify.png — because the user chose artwork for
    # each and it is now installed as `<key>.png`, which `app_icon`
    # finds without being told. An override is for an app that BORROWS
    # another app's picture; it is not the place to record "this app has
    # an icon". (`align` and `foreign` gained artwork in the same round;
    # neither was ever in this table — `align` was in _FORCE_GLYPH below
    # and `foreign` had nothing at all.)
    #
    # `motility` is the one worth remembering. It borrowed
    # recruitment.png, so re-skinning Recruitment silently re-skinned
    # Motility Assay as well — the old recruitment drawing had to be
    # kept and installed as motility.png to stop that. A borrowed icon
    # is a coupling between two apps that nothing declares.
    #
    # Of the six left, FOUR are genuine sharing and are a debt:
    # `train_cellpose` shows Cellpose Masks' picture, `agreement` shows
    # Annotate's, `plate_view` shows Map Barcodes', `model_compare`
    # shows Mask's. The other two are only renames — no app is keyed
    # `plaque` or `download`, so `analyze_plaques` and `model_zoo` are
    # the sole users of that artwork and share it with nobody.
    #
    # queue.png / batch.png / invasion.png / replication.png are drawn for
    # these apps and named after them, so they need no override. They used
    # to: `queue` and `batch` BOTH aliased sequencing.png (a DNA helix,
    # "the closest visual match for now"), which made two different apps
    # render identically as a picture of neither. The new pair carries the
    # distinction that matters -- queue is the same settings over many
    # plates, batch is arbitrary module+plate combinations in sequence.
}

# Keys that render their qtawesome glyph instead of a bundled PNG.
#
# EMPTY, deliberately, and kept rather than deleted: it is the documented
# fallback for an app whose meaning no bundled artwork carries, and
# emptying the set is not the same as removing the escape hatch.
#
# ``align`` was the last entry. No bundled PNG read as "tiles registered
# into ONE canvas", so it drew ``fa5s.border-all`` — a square divided
# into four by its own seams. The user has since chosen
# ``cellpose_all_01`` for it, which is that judgement overruled by the
# person whose app it is; the PNG is installed as ``align.png`` and the
# glyph is out of the way.
#
# WORTH KNOWING: ``cellpose_all_01.png`` was already installed as
# ``cellpose_all.png``, so ``align.png`` is now byte-identical to it.
# No two Qt tiles collide — ``cellpose_all`` is a Tk-only module and is
# not in :data:`APPS` — but the Tk GUI's "Cellpose All" and Qt's
# Align & Stitch draw the same picture, and re-inking one re-inks both.
# The user chose it explicitly; this is the note, not an objection.
#
# ``invasion`` left for the same reason earlier ("no bundled PNG reads
# as inside vs outside") once it had artwork that did.
_FORCE_GLYPH: set = set()


def _icon_for_app(key: str) -> Optional[QIcon]:
    """Return a QIcon for an app key.

    The bundled PNG is re-inked for the active theme by
    :func:`spacr.qt.iconset.app_icon`. Loading it raw is what left
    Format Converter — a solid-black PNG — with no visible icon at all
    on the black home page, and every white icon invisible on the light
    theme. Falls back to a themed qtawesome glyph when there is no PNG.
    """
    # Keys that should use their themed qtawesome glyph rather than a bundled
    # PNG (e.g. train_cellpose got a fresh 'brain' glyph).
    if key in _FORCE_GLYPH:
        return iconset.icon(key)
    return iconset.app_icon(key, override=_ICON_OVERRIDES.get(key))


class Sidebar(QWidget):
    """Left navigation column. Emits `nav_selected(str key)` when a tile
    is clicked. `Home` reverts to the startup page."""

    from PySide6.QtCore import Signal
    nav_selected = Signal(str)

    #: Width bounds in px at 100 % font scale. The column starts at
    #: ``WIDTH_MIN`` and widens — up to ``WIDTH_MAX`` — if the longest app
    #: name needs it, so a new long name can't quietly get cut in half.
    WIDTH_MIN = 220
    WIDTH_MAX = 320

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        title = QLabel("spaCR")
        title.setObjectName("SidebarTitle")
        outer.addWidget(title)

        # The nav rows scroll. Measured at 1440x900 -- the realistic laptop
        # size -- this column stacks a row per app, a heading per section
        # and the title in a plain QVBoxLayout and asks for ~1300 px against
        # 850 available, so the last few apps were simply UNREACHABLE. The
        # title stays pinned; only the rows move.
        #
        # No stylesheet is set on the scroll area on purpose: an unscoped
        # `background: transparent` on a QScrollArea cascades to every
        # descendant and strips the fill off the buttons inside it.
        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("SidebarScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.viewport().setAutoFillBackground(False)

        inner = QWidget()
        inner.setObjectName("SidebarInner")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._scroll.setWidget(inner)
        outer.addWidget(self._scroll, 1)

        self._items: list[ElidingPushButton] = []
        self._section_headers: dict[str, QLabel] = {}

        home = self._make_item("Home", "Back to the start page", "__home__")
        home.setIcon(iconset.icon("home"))
        home.clicked.connect(lambda: self.nav_selected.emit("__home__"))
        layout.addWidget(home)

        # Group apps by section, in APPS order
        current_section = None
        for key, name, desc, section in APPS:
            if section != current_section:
                header = QLabel(section)
                header.setObjectName("SidebarSection")
                layout.addWidget(header)
                self._section_headers[section] = header
                current_section = section
            btn = self._make_item(name, desc, key)
            icon = _icon_for_app(key)
            if icon is not None:
                btn.setIcon(icon)
                btn.setIconSize(QSize(16, 16))
            btn.clicked.connect(lambda checked=False, k=key: self.nav_selected.emit(k))
            layout.addWidget(btn)

        layout.addStretch(1)
        self.refresh_visibility()

    def refresh_visibility(self) -> None:
        """Apply Alpha/Beta filters without rebuilding or reparenting the dock."""
        section_by_key = {key: section for key, _name, _desc, section in APPS}
        visible_sections = set()
        for btn in self._items:
            key = str(btn.property("navKey") or "")
            if key == "__home__":
                btn.setVisible(True)
                continue
            visible = app_is_visible(key)
            btn.setVisible(visible)
            if visible and key in section_by_key:
                visible_sections.add(section_by_key[key])
        for section, header in self._section_headers.items():
            header.setVisible(section in visible_sections)
        self.setFixedWidth(self.fitting_width())

    def refresh_icons(self) -> None:
        """Re-ink every nav icon for the current theme.

        A QIcon bakes its pixmap when it is built, so re-applying the
        stylesheet does not recolour icons that already exist: switch to
        the light theme with a sidebar on screen and every white glyph
        stays white, on white. Each row carries its app key in the
        ``navKey`` property, so rebuilding them is just a re-lookup.
        """
        for btn in self._items:
            key = btn.property("navKey")
            if not key:
                continue
            icon = (iconset.icon("home") if key == "__home__"
                    else _icon_for_app(key))
            if icon is not None:
                btn.setIcon(icon)
                btn.setIconSize(QSize(16, 16))

    def _make_item(self, name: str, desc: str,
                   nav_key: str) -> "ElidingPushButton":
        """Build one navigation row.

        The label elides instead of clipping, and the tooltip leads with
        the app NAME (not just its description) so a shortened label is
        still identifiable on hover.

        :param nav_key: the app key this row navigates to, exposed as the
            ``navKey`` Qt property so callers (and tests) can tell app
            rows apart from the Home row without parsing labels.
        """
        # "&&" so Qt draws the ampersand instead of eating it as a
        # mnemonic: "Align & Stitch" was rendering as "Align _Stitch" in
        # the nav column (QPushButton reads a lone & as an accelerator;
        # the Home tiles are unaffected because they draw their name in
        # a QLabel, which does not).
        btn = ElidingPushButton(f"  {name.replace('&', '&&')}")
        btn.setObjectName("SidebarItem")
        btn.setProperty("navKey", nav_key)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(f"{name} — {desc}" if desc else name)
        # Accessibility — screen readers announce the app name +
        # its one-line description as the button's role.
        btn.setAccessibleName(name)
        btn.setAccessibleDescription(desc)
        self._items.append(btn)
        return btn

    def fitting_width(self) -> int:
        """Width that shows the longest nav label in full, within bounds.

        Font scale moves both bounds (a 150 % font needs a 150 % column),
        and the widest button's own size hint moves the result inside
        them. Names longer than ``WIDTH_MAX`` elide with a tooltip rather
        than pushing the column across the window.

        Public because the locked dock has to re-apply it: the sidebar is
        re-parented out of the drawer, which had resized it to the
        drawer's width.
        """
        from .preferences import scaled_px
        widest = max((b.sizeHint().width() for b in self._items
                      if not b.isHidden()), default=0)
        return max(scaled_px(self.WIDTH_MIN),
                   min(widest + scaled_px(12), scaled_px(self.WIDTH_MAX)))

    def clipped_items(self) -> list:
        """Nav buttons whose label had to be shortened to fit.

        Empty in a healthy layout; a test asserts that, and anything in
        here still carries its full name in its tooltip.
        """
        return [b for b in self._items if b.is_elided()]


class MainWindow(QMainWindow):
    """Top-level window: sidebar + stacked screens + status bar.

    :param initial_app: optional app key to navigate to on show; when
        omitted the window opens on the Home startup page.
    """

    def __init__(self, initial_app: Optional[str] = None):
        super().__init__()
        self._closing = False
        self.setWindowTitle("spaCR")
        self.setMinimumSize(1200, 720)

        self._build_menu_bar()

        # Central layout: a row that holds an (initially empty) dock slot
        # and the screen stack. By default the app list is a REVEAL over
        # the stack's left edge rather than anything in that slot.
        #
        # The column was 220-320 px of the 1440 a laptop has, on every
        # screen, holding a list most sessions never touch — and it is
        # the reason Home could not fit its five categories plus a state
        # column without scrolling. As a drawer it costs 6 px of trigger
        # strip and is one hover, one click, or Ctrl+B away.
        #
        # The slot is what "lock the dock" fills: see
        # :meth:`apply_dock_mode`. It exists whatever the mode, because a
        # QMainWindow's central widget cannot be swapped without
        # re-parenting the stack, and re-parenting a stack that already
        # holds live screens is how a locked dock would cost you the
        # screen you were looking at.
        #
        # `self._sidebar` is the SAME `Sidebar` object it always was, only
        # reparented: the tutorial highlights it, the command palette and
        # the tests all reach it by that name.
        self._stack = QStackedWidget()
        central = QWidget()
        central.setObjectName("CentralRow")
        row = QHBoxLayout(central)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self._dock_slot = QWidget()
        self._dock_slot.setObjectName("DockSlot")
        slot_col = QVBoxLayout(self._dock_slot)
        slot_col.setContentsMargins(0, 0, 0, 0)
        slot_col.setSpacing(0)
        self._dock_slot.hide()
        row.addWidget(self._dock_slot)
        row.addWidget(self._stack, 1)
        self.setCentralWidget(central)

        self._sidebar = Sidebar()
        self._sidebar.nav_selected.connect(self._on_nav_selected)
        self._sidebar.nav_selected.connect(self._on_drawer_navigated)

        from .widgets.drawer import EdgeDrawer
        self._app_drawer = EdgeDrawer(self._stack, self._sidebar,
                                      width=self._sidebar.width())
        self.apply_dock_mode()

        # Register screens lazily — created on first navigation.
        self._screens: dict[str, QWidget] = {}
        #: App keys in the order the user last LOOKED at them, oldest
        #: first. Not the same thing as ``_screens`` order, which is
        #: creation order and never changes when an app is revisited —
        #: see :meth:`_snapshot_current_screen_settings`.
        self._visit_order: list[str] = []
        self._install_startup_page()

        # Rich status bar: transient message (left) + active app + version
        status = QStatusBar()
        self._status_app_label = QLabel("Home")
        self._status_app_label.setObjectName("Muted")
        self._status_version_label = QLabel(f"spaCR {self._resolve_version()}")
        self._status_version_label.setObjectName("Caption")
        status.addPermanentWidget(self._status_app_label)
        status.addPermanentWidget(self._status_version_label)
        status.showMessage("Ready")
        self.setStatusBar(status)

        # The AI Console now lives inside each pipeline app's Console
        # panel (see spacr.qt.widgets.console_panel). No side-dock.

        # Preload heavy pipeline imports in a background thread AFTER
        # the first screen has been built. Kicking it off pre-nav caused
        # a real circular-import race in spacr.core/IPython on some
        # systems ("partially initialized module 'IPython'"), so we wait
        # a moment before starting.
        from PySide6.QtCore import QTimer
        self._preloader = _PipelinePreloader()
        QTimer.singleShot(1500, self._preloader.start)

        # Keyboard shortcuts — Ctrl+H, Ctrl+1..9, Ctrl+K, F1/?, etc.
        try:
            from . import shortcuts
            shortcuts.install(self)
        except Exception:
            pass

        if initial_app:
            self._on_nav_selected(initial_app)

        # Apply the persisted language after every startup widget exists.
        # New lazy screens are translated separately when first constructed.
        self.refresh_language()

        # First-launch tour — coach-marks over the home layout the
        # first time this user boots spacr. State stored in QSettings,
        # so subsequent launches are silent. Delayed a beat so the
        # window has time to render before the overlay attaches.
        try:
            from PySide6.QtCore import QTimer
            from .first_run import maybe_show_tour
            QTimer.singleShot(800, lambda: maybe_show_tour(self))
        except Exception:
            pass

    def _resolve_version(self) -> str:
        """Return the installed spacr version string, or ``"dev"`` on failure."""
        try:
            import spacr
            return getattr(spacr, "__version__", "") or "dev"
        except Exception:
            return "dev"

    # -- menu -------------------------------------------------------------
    def _build_menu_bar(self):
        mb = self.menuBar()

        app_menu = mb.addMenu("&spaCR")
        self._app_actions: dict[str, QAction] = {}
        for key, name, desc, section in APPS:
            act = QAction(name, self)
            act.setStatusTip(desc)
            act.triggered.connect(lambda checked=False, k=key: self._on_nav_selected(k))
            app_menu.addAction(act)
            self._app_actions[key] = act
        self._refresh_app_action_visibility()
        app_menu.addSeparator()
        act_home = QAction("Home", self)
        act_home.setShortcut(QKeySequence("Ctrl+H"))
        act_home.triggered.connect(lambda: self._on_nav_selected("__home__"))
        app_menu.addAction(act_home)
        # The keyboard + menu route into the edge reveal. A panel you can
        # only summon by hovering a 6 px strip is a panel a keyboard user
        # does not have, so this is not optional decoration.
        act_all = QAction("All apps", self)
        act_all.setShortcut(QKeySequence("Ctrl+B"))
        act_all.setStatusTip(
            "Show the full app list. Also revealed by moving the pointer "
            "to the left edge of the window.")
        act_all.triggered.connect(self.toggle_app_drawer)
        app_menu.addAction(act_all)
        self.addAction(act_all)
        #: Kept so :meth:`apply_dock_mode` can grey it out — a Ctrl+B that
        #: silently does nothing because the dock is hidden is worse than
        #: a menu entry that says so.
        self._act_all_apps = act_all
        app_menu.addSeparator()
        act_prefs = QAction("Preferences…", self)
        act_prefs.setShortcut(QKeySequence("Ctrl+,"))
        act_prefs.triggered.connect(self._open_preferences)
        app_menu.addAction(act_prefs)
        app_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        app_menu.addAction(act_quit)

        demo_menu = mb.addMenu("&Demos")
        self._demo_actions: dict[str, QAction] = {}
        for app_key, label in (
            ("mask",      "Mask demo…"),
            ("measure",   "Measure demo…"),
            ("crop",      "Crop demo…"),
            ("classify",  "Classify demo…"),
            ("timelapse", "Timelapse demo…"),
            ("map_barcodes", "Sequencing demo…"),
        ):
            act = QAction(label, self)
            act.setStatusTip(
                f"Generate a synthetic {app_key} dataset and open it "
                "in the matching app.")
            act.triggered.connect(
                lambda checked=False, k=app_key: self._on_load_demo(k))
            demo_menu.addAction(act)
            target = self.DEMO_TARGETS.get(app_key, (app_key, ""))[0]
            self._demo_actions[app_key] = act
            act.setVisible(app_is_visible(target))
        demo_menu.addSeparator()
        act_e2e = QAction("End-to-end (Mask → Measure → Annotate) real dataset…", self)
        act_e2e.setStatusTip(
            "Download the toxo_mito HF demo dataset + settings pack, "
            "then chain Mask → Measure → Annotate on it.")
        act_e2e.triggered.connect(self._on_e2e_demo)
        demo_menu.addAction(act_e2e)

        help_menu = mb.addMenu("&Help")
        act_tutorial = QAction("Tutorial (web)", self)
        act_tutorial.setIcon(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_MessageBoxInformation
            )
        )
        act_tutorial.triggered.connect(
            lambda: self._open_url("https://einarolafsson.github.io/spacr/tutorial/"))
        help_menu.addAction(act_tutorial)
        act_docs = QAction("Documentation (web)", self)
        act_docs.setIcon(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_MessageBoxInformation
            )
        )
        act_docs.triggered.connect(
            lambda: self._open_url("https://einarolafsson.github.io/spacr/index.html"))
        help_menu.addAction(act_docs)
        act_about = QAction("About spaCR", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)
        help_menu.addSeparator()
        act_update = QAction("Check for updates…", self)
        act_update.triggered.connect(self._check_for_updates)
        help_menu.addAction(act_update)
        act_log = QAction("Open log folder…", self)
        act_log.setStatusTip(
            "Open the ~/.spacr/logs folder. Attach the newest log file "
            "to any bug report — it contains a full trace of every "
            "spaCR run including function entries, button presses, "
            "and pipeline output.")
        act_log.triggered.connect(self._open_log_folder)
        help_menu.addAction(act_log)

    def _open_url(self, url: str):
        """Open ``url`` in the system browser; surface failures in the status bar."""
        import webbrowser
        try:
            webbrowser.open(url)
        except Exception as e:
            self.statusBar().showMessage(f"Failed to open {url}: {e}", 5000)

    # -- demos -----------------------------------------------------------
    # Map each demo key to (target-app key, generator function name).
    # Kept as a class constant so tests can introspect it without launching
    # the file dialog.
    DEMO_TARGETS = {
        "mask":      ("mask",       "generate_mask_demo"),
        "measure":   ("measure",    "generate_measure_demo"),
        "crop":      ("measure",    "generate_crop_demo"),
        "classify":  ("annotate",   "generate_classify_demo"),
        # The timelapse demo writes a settings CSV with timelapse=True; the
        # Mask module no longer has a widget for that key, so it has to land
        # in the Timelapse module or the flag would be silently dropped.
        "timelapse": ("timelapse",  "generate_timelapse_demo"),
        "map_barcodes": ("map_barcodes", "generate_map_barcodes_demo"),
    }

    def _on_load_demo(self, demo_key: str) -> None:
        """Generate a synthetic demo dataset, save its settings, then
        navigate to the matching app and pre-populate it."""
        from PySide6.QtWidgets import QFileDialog
        from pathlib import Path

        target_app, gen_name = self.DEMO_TARGETS[demo_key]

        default = str(Path.home() / "spacr-demos" / demo_key)
        dst = QFileDialog.getExistingDirectory(
            self, f"Choose destination for {demo_key} demo",
            default,
            QFileDialog.ShowDirsOnly | QFileDialog.DontConfirmOverwrite,
        )
        if not dst:
            return
        try:
            layout = self._run_demo_generator(demo_key, dst)
        except Exception as e:
            QMessageBox.warning(self, "Demo generation failed", str(e))
            return

        self._on_nav_selected(target_app)
        widget = self._screens.get(target_app)
        if widget is None:
            return
        try:
            self._apply_demo_to_screen(widget, layout)
            self.statusBar().showMessage(
                f"Loaded {demo_key} demo from {layout.src}", 5000)
        except Exception as e:
            QMessageBox.warning(self, "Demo load failed", str(e))

    def _on_e2e_demo(self) -> None:
        """Confirm, prompt for a folder, download the HF demo dataset,
        then chain Mask -> Measure -> Annotate on it.

        Flow (matches the spec agreed with the user):
          1. Yes/No modal: "do you want to test mask -> Measure ->
             Annotate on a real dataset?"
          2. Folder picker for the local download destination.
          3. QProgressDialog while the toxo_mito + spacr_settings repos
             download in a background thread.
          4. On success, kick off Mask -> Measure -> Annotate. Users
             see the run inside each app's normal console.
        """
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path

        answer = QMessageBox.question(
            self, "End-to-end demo",
            "Do you want to test Mask → Measure → Annotate on a real "
            "dataset?\n\n"
            "This will download the toxo_mito demo dataset "
            "(~a few hundred MB) plus the matching settings pack from "
            "Hugging Face, then run the pipeline chain against it.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        default = str(Path.home() / "spacr-demos" / "toxo_mito_e2e")
        dst = QFileDialog.getExistingDirectory(
            self, "Choose folder for the demo dataset",
            default,
            QFileDialog.ShowDirsOnly | QFileDialog.DontConfirmOverwrite,
        )
        if not dst:
            return

        from .hf_download import download_toxo_mito_demo

        def _on_download_done(result, error):
            if result is None:
                QMessageBox.warning(self, "Download",
                    f"The download did not complete:\n{error or 'unknown error'}")
                return
            self.statusBar().showMessage(
                f"Downloaded demo dataset to {result.dataset_path}", 6000)
            self._run_e2e_chain(result.dataset_path,
                                    result.settings_path)

        download_toxo_mito_demo(self, dst, _on_download_done)

    def _run_e2e_chain(self, dataset_path, settings_path) -> None:
        """Run Mask → Measure → Annotate against the freshly-downloaded
        dataset, prompting before each stage.

        The user gets a Continue/Stop popup before each stage kicks off
        so they can inspect the previous run's output before letting
        the next one loose. Non-interactive stages (mask, measure) run
        their pipeline immediately after Continue; the annotate stage
        just opens the annotation UI at the dataset root so the user
        can start labelling.
        """
        from PySide6.QtWidgets import QMessageBox
        from pathlib import Path

        dataset_path  = Path(dataset_path)
        settings_path = Path(settings_path)

        # Helper — load the app's default settings, then override with
        # whatever CSV pack we downloaded for that app.
        def _settings_for(app_key: str) -> dict:
            from .screens.settings_model import resolve_default_settings
            settings = dict(resolve_default_settings(app_key))
            csv = settings_path / f"{app_key}_settings.csv"
            if csv.is_file():
                import csv as _csv
                with csv.open() as fh:
                    for row in _csv.reader(fh):
                        if not row or row[0].startswith("#") or len(row) < 2:
                            continue
                        k, v = row[0].strip(), row[1]
                        if v.lower() in ("true", "false"):
                            v = v.lower() == "true"
                        else:
                            try:
                                v = int(v)
                            except ValueError:
                                try:
                                    v = float(v)
                                except ValueError:
                                    pass
                        settings[k] = v
            settings["src"] = str(dataset_path)
            return settings

        stages = (
            ("mask",     "Mask generation",
             "Ready to start mask generation with the downloaded settings?"),
            ("measure",  "Measurement",
             "Mask stage finished. Ready to start measurement?"),
            ("annotate", "Annotation",
             "Measurement stage finished. Ready to open the annotation UI?"),
        )
        for stage, title, prompt in stages:
            answer = QMessageBox.question(
                self, title, prompt,
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                self.statusBar().showMessage(
                    f"E2E chain stopped at '{stage}' stage.", 6000)
                return
            settings = _settings_for(stage)
            self._on_nav_selected(stage)
            widget = self._screens.get(stage)
            if widget is None:
                QMessageBox.warning(self, "E2E",
                    f"Couldn't open the '{stage}' screen.")
                return
            try:
                if hasattr(widget, "apply_settings_dict"):
                    widget.apply_settings_dict(settings)
                # Kick off the pipeline automatically for mask + measure;
                # annotate is interactive and opens directly at the
                # loaded dataset so the user can start labelling.
                if stage != "annotate" and hasattr(widget, "_on_run"):
                    widget._on_run()
            except Exception as e:
                QMessageBox.warning(self, f"E2E: {stage} failed", str(e))
                return
        self.statusBar().showMessage(
            "E2E chain launched — check each app's console for progress.",
            8000)

    def _run_demo_generator(self, demo_key: str, dst: str):
        """Isolated for tests — invoke the named generator function
        with `dst` and return whatever it returned."""
        from spacr.qt import synthetic as syn
        _, gen_name = self.DEMO_TARGETS[demo_key]
        gen = getattr(syn, gen_name)
        return gen(dst)

    def _apply_demo_to_screen(self, widget, layout) -> None:
        """Push the demo layout into a target screen, in whatever way
        that screen supports (settings CSV, source folder, or DB path)."""
        from spacr.utils import load_settings

        # AppScreen: load the CSV into its settings model
        if hasattr(widget, "apply_settings_dict") and layout.settings_csv:
            loaded = load_settings(
                str(layout.settings_csv),
                setting_key="Key", setting_value="Value",
            )
            if isinstance(loaded, dict):
                widget.apply_settings_dict(loaded)
                return
        # AnnotateScreen: takes a src folder directly
        if hasattr(widget, "_open_source"):
            widget._open_source(str(layout.src))
            return
        # MakeMasksScreen: opens a folder directly
        if hasattr(widget, "_open_folder"):
            widget._open_folder(str(layout.src))
            return

    def _show_about(self):
        """Show the About dialog with the installed spacr version."""
        try:
            import spacr
            version = spacr.__version__
        except Exception:
            version = "unknown"
        QMessageBox.about(self, "About spaCR",
                          f"<h3>spaCR</h3>"
                          f"<p>Spatial single-cell analysis for microscopy data.</p>"
                          f"<p><b>Version:</b> {version}</p>"
                          f"<p>© Olafsson Lab</p>")

    def _open_log_folder(self):
        """Open the ~/.spacr/logs folder in the OS file browser."""
        from .verbose_logger import log_dir
        import webbrowser
        try:
            webbrowser.open(f"file://{log_dir()}")
        except Exception as e:
            self.statusBar().showMessage(
                f"Failed to open log folder: {e}", 5000)

    def _open_preferences(self):
        """Open the Preferences dialog (theme, font size, colour-blind)."""
        try:
            from .preferences import PreferencesDialog
        except Exception as e:
            self.statusBar().showMessage(
                f"Preferences unavailable: {e}", 5000)
            return
        PreferencesDialog(self).exec()
        self.refresh_theme()

    def refresh_theme(self) -> None:
        """Rebuild everything preferences cannot update through QSS alone.

        Three things do not follow a ``setStyleSheet`` call: the Home
        tiles set sizes/margins from the font scale in Python, every
        QIcon baked its pixmap at the theme in force when it was built,
        and the dock's mode and the page's opacity are layout decisions
        rather than colours. Called after the Preferences dialog closes,
        and found by duck-typing from
        :func:`spacr.qt.preferences.apply_preferences_to_app` so a
        preference change from anywhere reaches the widgets.
        """
        try:
            self._sidebar.refresh_icons()
            self._sidebar.refresh_visibility()
        except Exception:
            pass
        try:
            self._refresh_app_action_visibility()
        except Exception:
            pass
        for screen in getattr(self, "_screens", {}).values():
            refresh = getattr(screen, "refresh_maturity_visibility", None)
            if callable(refresh):
                try:
                    refresh()
                except Exception:
                    pass
        try:
            self.apply_dock_mode()
        except Exception:
            pass
        try:
            self._rebuild_startup_page()
        except Exception:
            pass
        self.refresh_language()
        try:
            self._sidebar.refresh_visibility()
        except Exception:
            pass

    def refresh_language(self) -> None:
        """Apply the persisted language to existing static UI text."""
        try:
            from .i18n import retranslate_widget_tree
            retranslate_widget_tree(self)
        except Exception:
            LOG.exception("Could not apply the selected UI language")

    def _refresh_app_action_visibility(self) -> None:
        """Keep the spaCR menu in sync with module maturity preferences."""
        for key, action in getattr(self, "_app_actions", {}).items():
            action.setVisible(app_is_visible(key))
        for demo_key, action in getattr(self, "_demo_actions", {}).items():
            target = self.DEMO_TARGETS.get(demo_key, (demo_key, ""))[0]
            action.setVisible(app_is_visible(target))

    def _rebuild_startup_page(self):
        """Recreate the Home page (e.g. after a font-scale change)."""
        old = getattr(self, "_startup", None)
        was_current = (old is not None
                       and self._stack.currentWidget() is old)
        self._install_startup_page()
        if old is not None:
            self._stack.removeWidget(old)
            # close() before deleteLater() so the outgoing page drops its
            # subscription to the run registry now, rather than staying a
            # live receiver until the deferred delete is flushed.
            try:
                old.close()
            except Exception:
                pass
            old.deleteLater()
        if was_current:
            self._stack.setCurrentWidget(self._startup)

    def _check_for_updates(self):
        """Query PyPI/GitHub in a background thread, prompt to upgrade.

        Both the network call and an accepted ``pip`` upgrade run on
        :class:`_UpdateWorker`; only dialogs and status updates run here.
        """
        try:
            from spacr.updater import check_for_updates
        except Exception as e:
            LOG.exception("Could not import the spaCR updater")
            QMessageBox.warning(self, "Updates",
                                f"Update check unavailable: {e}")
            return

        self.statusBar().showMessage("Checking for updates…", 4000)
        self._start_update_worker(
            "check", check_for_updates, self._on_update_check_done)

    def _start_update_worker(self, operation, fn, on_done) -> None:
        """Start one updater callable and retain it until shutdown."""
        worker = _UpdateWorker(operation, fn, self)
        worker.succeeded.connect(on_done)
        worker.failed.connect(self._on_update_worker_failed)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        self._update_worker = worker

    def _on_update_check_done(self, info) -> None:
        """Handle an :class:`spacr.updater.UpdateInfo` on the GUI thread."""
        if self._closing:
            LOG.debug("Discarding an update result during shutdown")
            return
        if info.error and not info.latest_release:
            QMessageBox.warning(
                self, "Updates",
                f"Couldn't reach update server:\n{info.error}")
            return
        if not info.upgrade_available:
            QMessageBox.information(
                self, "Updates",
                f"You're on {info.installed_version}. No updates.")
            return

        msg = (f"A new version is available.\n\n"
               f"Installed: {info.installed_version}\n"
               f"Latest:    {info.latest_release}\n\n"
               f"Run pip install --upgrade spacr now?")
        if QMessageBox.question(
                self, "Update available", msg) != QMessageBox.Yes:
            return
        try:
            from spacr.updater import run_pip_upgrade
        except Exception as exc:
            LOG.exception("Could not import the spaCR upgrade helper")
            QMessageBox.warning(
                self, "Updates", f"Upgrade unavailable: {exc}")
            return
        self.statusBar().showMessage("Upgrading spaCR…", 4000)
        self._start_update_worker(
            "upgrade", run_pip_upgrade, self._on_upgrade_done)

    def _on_upgrade_done(self, return_code) -> None:
        """Report a completed package upgrade on the GUI thread."""
        if self._closing:
            LOG.debug("Discarding an upgrade result during shutdown")
            return
        if return_code == 0:
            QMessageBox.information(
                self, "Updates",
                "Upgrade finished. Restart spaCR to use it.")
        else:
            QMessageBox.warning(
                self, "Updates",
                f"pip returned exit code {return_code}. "
                "Check the terminal for details.")

    def _on_update_worker_failed(self, operation: str, details: str) -> None:
        """Report an updater exception instead of losing it in a QThread."""
        if self._closing:
            LOG.debug("Updater %s failed during shutdown:\n%s",
                      operation, details)
            return
        last = next(
            (line for line in reversed(details.splitlines()) if line.strip()),
            "unknown error")
        label = "Update check" if operation == "check" else "Upgrade"
        QMessageBox.warning(self, "Updates", f"{label} failed:\n{last}")

    # -- shutdown ----------------------------------------------------------
    def closeEvent(self, event):
        """Cancel every active AI stream + wait for its QThread to exit
        BEFORE Qt starts destroying widgets. Prevents the
        'QThread: Destroyed while thread is still running / Aborted'
        crash on quit."""
        self._closing = True
        from .widgets.console_panel import ConsolePanel
        for panel in self.findChildren(ConsolePanel):
            try:
                panel.shutdown()
            except Exception:
                pass
        # Help → "Check for updates…" runs its network call on a QThread
        # parented to this window. Quitting while it's in flight destroys
        # a live QThread, which is the same abort the console drain above
        # exists to prevent. The updater's own socket timeouts are a few
        # seconds, so the wait is bounded twice over.
        worker = getattr(self, "_update_worker", None)
        if worker is not None:
            try:
                worker.wait(5000)
            except RuntimeError:
                pass          # already deleted — nothing left to wait for
        super().closeEvent(event)

    # -- the app-list drawer ----------------------------------------------
    def dock_mode(self) -> str:
        """The user's dock preference — ``auto`` / ``locked`` / ``hidden``.

        Read through here rather than inlined so a headless build, or one
        with an unwritable settings file, still gets the default rather
        than an exception during ``__init__``.
        """
        try:
            from .preferences import get_dock_mode
            return get_dock_mode()
        except Exception:
            return "locked"

    def apply_dock_mode(self, mode: Optional[str] = None) -> None:
        """Put the app list where the preference says it goes.

        Three modes, one ``Sidebar`` object:

        ``auto``    the sidebar lives inside the :class:`EdgeDrawer` and
                    reveals on dwell against the left edge.
        ``locked``  the sidebar is re-parented into the window's dock
                    slot, where it is an ordinary column: it never
                    animates, never overlays the page, and the hot strip
                    is switched off so it cannot also slide in on top of
                    itself.
        ``hidden``  the drawer is closed, its hot strip hidden, and the
                    slot stays empty. The "All apps" action is disabled
                    with a tooltip that says where to turn it back on —
                    a control that silently does nothing is worse than
                    one that is greyed out.

        Idempotent, and safe to call before the menu exists.
        """
        mode = mode or self.dock_mode()
        drawer = getattr(self, "_app_drawer", None)
        sidebar = getattr(self, "_sidebar", None)
        slot = getattr(self, "_dock_slot", None)
        if drawer is None or sidebar is None or slot is None:
            return
        self._dock_mode = mode

        if mode == "locked":
            drawer.close()
            drawer.set_enabled(False)
            if sidebar.parent() is not slot:
                slot.layout().addWidget(sidebar)
            sidebar.setFixedWidth(sidebar.fitting_width())
            sidebar.show()
            slot.show()
        else:
            if sidebar.parent() is not drawer:
                drawer.adopt(sidebar)
            slot.hide()
            drawer.set_enabled(mode == "auto")
            if mode == "hidden":
                drawer.close()

        action = getattr(self, "_act_all_apps", None)
        if action is not None:
            action.setEnabled(mode != "hidden")
            action.setToolTip(
                "The app dock is hidden. Turn it back on in Preferences → "
                "App dock." if mode == "hidden" else
                "Show the full app list.")

    def toggle_app_drawer(self) -> None:
        """Open (and focus) or close the slide-in app list.

        The keyboard and menu path into the reveal, so the panel is not
        hover-only. Note what the drawer is *for* now that Home's first
        tab lists every app: it is not Home's app list, it is the app
        list on **every other screen** — the replacement for the
        permanent 220 px column. From inside Mask, this is the only
        pointer-driven way to reach Measure without going Home first.

        A no-op when the dock is locked (it is already on screen and not
        going anywhere) or hidden (the user asked for it not to be
        there; a shortcut that overrules a preference is a bug).
        """
        if getattr(self, "_dock_mode", "auto") != "auto":
            return
        drawer = getattr(self, "_app_drawer", None)
        if drawer is not None:
            drawer.toggle()

    def _on_drawer_navigated(self, _key: str) -> None:
        """A row in the drawer was clicked — it has done its job, close it.

        Nothing to close when the dock is locked: it is a column, and a
        column that vanished every time you used it would be worse than
        the reveal it replaced.
        """
        if getattr(self, "_dock_mode", "auto") != "auto":
            return
        drawer = getattr(self, "_app_drawer", None)
        if drawer is not None:
            drawer.close()

    # -- navigation -------------------------------------------------------
    def _install_startup_page(self):
        """Instantiate the Home page and add it to the stack."""
        self._startup = make_home_page()
        self._startup.tile_clicked.connect(self._on_nav_selected)
        self._startup.update_check_requested.connect(self._check_for_updates)
        # The hero's "All apps" button is the labelled twin of the edge
        # reveal — a discoverable way in for anyone who never finds the
        # hot strip, and the thing a screenshot can point at.
        try:
            self._startup._btn_all_apps.clicked.connect(self.toggle_app_drawer)
        except Exception:
            pass
        self._stack.addWidget(self._startup)

    def _on_nav_selected(self, key: str):
        """Navigate to app ``key``, lazily instantiating its screen on first use."""
        if key == "__home__":
            # Re-read the things that go stale while Home is off screen:
            # the plate queue, the run journal and the disk/GPU figures.
            # Cheap (a JSON read and three stat calls) and only on a
            # deliberate return to Home, not on a timer.
            try:
                self._startup.refresh()
            except Exception:
                pass
            self._stack.setCurrentWidget(self._startup)
            self._status_app_label.setText("Home")
            self.statusBar().showMessage("Home", 2000)
            return
        if key not in self._screens:
            self._screens[key] = self._build_screen(key)
            self._stack.addWidget(self._screens[key])
            try:
                from .i18n import retranslate_widget_tree
                retranslate_widget_tree(self._screens[key])
            except Exception:
                LOG.exception("Could not translate the %s screen", key)
        self._stack.setCurrentWidget(self._screens[key])
        # Move this app to the end of the visit list. Revisiting an app
        # has to count as the most recent visit — otherwise "Add current
        # plate" on the Queue screen picks up whichever app was OPENED
        # last rather than the one that was on screen a moment ago.
        if key in self._visit_order:
            self._visit_order.remove(key)
        self._visit_order.append(key)
        # Find nice display name
        from .i18n import tr
        name = tr(next((n for k, n, _d, _s in APPS if k == key), key))
        self._status_app_label.setText(name)
        self.statusBar().showMessage(tr("Opened {name}", name=name), 2000)

    def _on_zoo_compare_requested(self, request: dict) -> None:
        """Open Model Compare preloaded with the two models the zoo selected.

        Goes through ``ModelCompareScreen.configure`` rather than reaching into
        the panels, so restructuring either panel cannot silently break the
        hand-off.

        :param request: ``{'model_a', 'model_b', 'folder', 'n_fields'}``.
        """
        self._on_nav_selected("model_compare")
        screen = self._screens.get("model_compare")
        if screen is None or not hasattr(screen, "configure"):
            return
        screen.configure(
            model_a=request.get("model_a", ""),
            model_b=request.get("model_b", ""),
            folder=request.get("folder", ""),
            n_fields=int(request.get("n_fields", 0) or 0),
        )

    def _build_screen(self, key: str) -> QWidget:
        """Return a freshly-built screen widget for the given app ``key``."""
        if key == "annotate":
            from .screens.annotate import AnnotateScreen
            screen = AnnotateScreen()
            screen.train_requested.connect(self._on_train_requested)
            return screen
        if key == "make_masks":
            from .screens.make_masks import MakeMasksScreen
            return MakeMasksScreen()
        if key == "queue":
            from .screens.queue import QueueScreen
            screen = QueueScreen()
            screen.wire_add_current(self._snapshot_current_screen_settings)
            return screen
        if key == "db_browser":
            from .screens.db_browser import DbBrowserScreen
            return DbBrowserScreen()
        if key == "agreement":
            from .screens.agreement import AgreementScreen
            return AgreementScreen()
        if key == "plate_view":
            from .screens.plate_view import PlateViewScreen
            return PlateViewScreen()
        if key == "model_compare":
            from .screens.model_compare import ModelCompareScreen
            return ModelCompareScreen()
        if key == "align":
            from .screens.align import AlignScreen
            return AlignScreen()
        if key == "convert":
            from .screens.convert import ConvertScreen
            return ConvertScreen()
        if key == "foreign":
            from .screens.foreign import ForeignScreen
            return ForeignScreen()
        if key == "batch":
            from .screens.batch import BatchScreen
            return BatchScreen()
        if key == "model_zoo":
            from .screens.model_zoo import ModelZooScreen
            screen = ModelZooScreen()
            screen.compare_requested.connect(self._on_zoo_compare_requested)
            return screen
        if key == "report":
            from .screens.report import ReportScreen
            return ReportScreen()
        if key == "train_compare":
            from .screens.train_compare import TrainCompareScreen
            return TrainCompareScreen()
        from .screens.app_screen import AppScreen
        screen = AppScreen(app_key=key)
        screen.error_explain_requested.connect(self._on_explain_error)
        return screen

    def _snapshot_current_screen_settings(self):
        """Return ``(app_key, settings_dict)`` for the AppScreen the user
        was looking at when they hit "Add current plate" on the Queue
        screen. Raises when the active screen isn't a normal app."""
        widget = self._stack.currentWidget()
        # Prefer the most-recently-viewed AppScreen — the Queue screen
        # itself isn't one.
        from .screens.app_screen import AppScreen
        if isinstance(widget, AppScreen):
            return widget.app_key, dict(widget._settings_model.collect())
        # Fall back to the last non-queue AppScreen the user visited.
        # Walk the VISIT order, not `_screens` (creation) order: a user
        # who opens Mask, then Measure, then goes back to Mask and hits
        # "Add current plate" means Mask — creation order would hand
        # them Measure's settings under Mask's nose.
        for key in reversed(self._visit_order):
            scr = self._screens.get(key)
            if isinstance(scr, AppScreen):
                return scr.app_key, dict(scr._settings_model.collect())
        raise RuntimeError(
            "No active plate settings — open Mask/Measure/Classify first.")

    def _on_explain_error(self, traceback_text: str, active_app: str) -> None:
        """Legacy hook — the AI now lives inside each AppScreen's
        Console panel, which handles Explain-error directly. This
        method is kept only for backward-compat with subclasses."""
        pass

    def _on_train_requested(self, target_key: str, seed: dict) -> None:
        """Navigate to `target_key` (creating the screen if needed) and
        push `seed` values into its settings model. Called by the
        annotate screen's Train CV / Train XG buttons."""
        self._on_nav_selected(target_key)
        widget = self._screens.get(target_key)
        if widget is None:
            return
        model = getattr(widget, "_settings_model", None)
        if model is None:
            return
        widgets = getattr(model, "_widgets", {})
        for key, value in seed.items():
            w = widgets.get(key)
            if w is None:
                continue
            try:
                self._apply_seed_value(w, value)
            except Exception:
                LOG.warning(
                    "Could not seed %s.%s with %r",
                    target_key, key, value, exc_info=True)

    @staticmethod
    def _apply_seed_value(w: QWidget, value) -> None:
        from PySide6.QtWidgets import (
            QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox,
        )
        setter = getattr(w, "set_value", None)
        if callable(setter):
            setter(value)
        elif isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        elif isinstance(w, QSpinBox):
            w.setValue(int(float(value)))
        elif isinstance(w, QDoubleSpinBox):
            w.setValue(float(value))
        elif isinstance(w, QComboBox):
            for i in range(w.count()):
                if w.itemData(i) == value or w.itemText(i) == str(value):
                    w.setCurrentIndex(i)
                    break
        elif isinstance(w, QLineEdit):
            w.setText("" if value is None else str(value))


def _load_bundled_fonts() -> None:
    """Register the bundled Open Sans TTFs with :class:`QFontDatabase`.

    Idempotent — the fonts are only loaded once even if called
    multiple times (Qt tracks the file path).
    """
    from PySide6.QtGui import QFontDatabase
    here = os.path.dirname(os.path.abspath(__file__))
    fonts_dir = os.path.join(here, "resources", "fonts")
    if not os.path.isdir(fonts_dir):
        return
    for name in os.listdir(fonts_dir):
        if name.lower().endswith((".ttf", ".otf")):
            QFontDatabase.addApplicationFont(os.path.join(fonts_dir, name))


def launch(argv: Optional[list[str]] = None) -> int:
    """Bootstrap QApplication and show the main window."""
    if argv is None:
        argv = sys.argv[1:]

    # Support `spacr-qt <app>` to open directly into an app.
    initial_app = argv[0] if argv else None

    # Enable high-DPI early.
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QApplication(sys.argv[:1])
    app.setApplicationName("spaCR")
    app.setOrganizationName("Olafsson Lab")
    app.setWindowIcon(QIcon(os.path.join(
        iconset.RESOURCE_DIR, "app_icon.png")))

    # Lift Qt's default 256 MB QImageReader allocation limit. Large multi-panel
    # figures rendered at high DPI decode to well over 256 MB, and hitting the
    # limit makes QPixmap loads fail (blank figures) and the UI hang. 0 = no
    # limit; the figure queue still caps display resolution for sanity.
    try:
        from PySide6.QtGui import QImageReader
        QImageReader.setAllocationLimit(0)
    except Exception:
        LOG.debug("This Qt build does not expose QImageReader allocation "
                  "limits", exc_info=True)

    # Bundle Open Sans (Regular + Light + SemiBold) so the app renders
    # the same on every OS regardless of what fonts the user has
    # installed. Registered before applying the stylesheet so any
    # `font-family: "Open Sans"` rule resolves.
    _load_bundled_fonts()

    # Apply user preferences (theme + font scale) — falls back to the
    # dark defaults on the first launch when nothing is stored yet.
    from .preferences import apply_preferences_to_app
    apply_preferences_to_app(app)

    # Real Python logging → rotating file + Qt signal so ConsolePanel
    # can render records inline. Set it up before the launch breadcrumb and
    # MainWindow construction so neither is lost.
    from .logging_util import setup_logging
    setup_logging()

    # Every launch drops a timeline marker into the diagnostic log.
    import logging as _lg
    import sys as _sys
    from .verbose_logger import current_log_file
    _lg.getLogger("spacr").info(
        "spaCR launched (python=%s.%s.%s, log=%s)",
        _sys.version_info.major, _sys.version_info.minor,
        _sys.version_info.micro, current_log_file())

    win = MainWindow(initial_app=initial_app)
    win.show()

    # Pre-warm the heavy imports that a module screen needs (spacr.gui_utils
    # pulls torch + cv2 ≈ 3-4 s; spacr.settings ≈ 1 s) in a BACKGROUND thread
    # while the user looks at the home screen. By the time they open a module
    # these are cached, so the module snaps open instead of freezing on the
    # first import. Importing modules (no Qt objects) off-thread is safe.
    def _prewarm():
        try:
            import importlib
            for mod in ("spacr.settings", "spacr.gui_utils"):
                importlib.import_module(mod)
        except Exception:
            LOG.debug("Could not prewarm GUI settings imports", exc_info=True)
    threading.Thread(target=_prewarm, name="spacr-prewarm",
                     daemon=True).start()

    # aboutToQuit fires no matter how the app exits (window closed,
    # Ctrl+C, SIGTERM, …). Belt-and-suspenders with MainWindow's
    # closeEvent: ensure every ConsolePanel drains its AI thread
    # before Qt starts destroying widgets.
    def _drain_ai():
        from .widgets.console_panel import ConsolePanel
        for panel in win.findChildren(ConsolePanel):
            try:
                panel.shutdown()
            except Exception:
                LOG.debug("Could not shut down a console panel",
                          exc_info=True)
        # Also kill any subprocess still tracked by a provider
        try:
            from . import ai as _ai
            for p in _ai.list_providers():
                p.cancel_stream()
        except Exception:
            LOG.debug("Could not cancel every AI provider", exc_info=True)
    app.aboutToQuit.connect(_drain_ai)

    return app.exec()
