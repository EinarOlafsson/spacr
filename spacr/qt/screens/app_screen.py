"""
AppScreen — the reusable layout every non-interactive spacr app uses.

Structure (horizontal splitter):
    ┌───────────────────────┬─────────────────────────────┐
    │ Settings (scrollable) │ Console (top)               │
    │                       │ Usage bars   |  Run/Stop... │
    │  QGroupBox sections   │ Progress bar                │
    │  QFormLayout inside   │                             │
    └───────────────────────┴─────────────────────────────┘
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from functools import partial
from html import escape
from typing import Optional

from PySide6.QtCore import QSize, QEvent, Qt, QTimer, QThread, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..bridge import make_thread, resolve_pipeline_entry
from ..i18n import tr
from ..job_runner import JobRunner
from ..theme import (SPACING, ensure_widget_qss_applied, register_widget_qss)
from ..widgets import Card, Divider, InfoLink, Section, UsageBar
from .settings_model import (
    CATEGORY_TOOLTIPS,
    SettingsWidgets,
    category_tooltip,
)

LOG = logging.getLogger(__name__)


#: Object name the settings column carries, and what the block below keys
#: off. It is the column itself, and — see `_settings_panel_qss` — the
#: rule that names it is a rule saying *paint nothing*.
SETTINGS_PANEL_NAME = "SettingsBox"

#: The "Point <module> at some data" banner at the top of that column.
EMPTY_STATE_NAME = "EmptyStateBanner"


def _settings_panel_qss(palette: dict, opacity=None) -> str:
    """The settings column paints nothing. The categories are the panels.

    Three positions were tried on this column, and the third is the one
    that was asked for:

    1. **An opaque slab.** A ``QScrollArea``'s viewport auto-fills with the
       palette's **Window** brush — not a surface — so no page-opacity
       setting could reach it and the column sat as a black rectangle over
       the animated backdrop.
    2. **A panel of its own.** Turning the auto-fill off left the column
       with nothing, so it was given the console box's treatment: a
       dark-grey rounded surface at the page opacity. That put a box round
       a column of boxes, and every category then composited two
       translucent greys — 0.51 at a requested 30 %, a shade no position of
       the slider can produce.
    3. **Nothing at all**, which is this. The categories float directly on
       the theme as separate rounded panels with the backdrop visible in
       the gaps between them. Most module screens are a list of categories
       and a list needs no box round it; a container would be a box around
       a box.

    So the rules here are all subtractive, and the surface the user sees is
    ``QFrame#SectionCard`` in the shared stylesheet, which already goes
    through the page-opacity roles. Removing the column's fill is what lets
    the slider reach the categories: they were never broken, they were
    composited onto things that were.

    An ID selector on the column anyway, rather than leaving it unstyled —
    an unstyled ``QScrollArea`` inherits the blanket
    ``QWidget {{ background-color: bg }}``, the WINDOW colour, which is
    exactly position 1. Saying it explicitly also keeps the decision
    findable, and the viewport keeps the container sweep's transparent tag.

    The banner is the same story one layer in. "Point <module> at some
    data" is an :class:`~spacr.qt.widgets.EmptyState` — a ``QWidget``
    subclass, so the sweep's ``type(w) is QWidget`` test skips it — renamed
    to ``EmptyStateBanner``, which also takes it out of the ``QFrame#Card``
    rule it would otherwise have matched. Between the two it had no rule at
    all and painted the window colour: a black box at the top of the
    column. It is a line of type on the page, so it paints nothing.
    """
    return f"""
QScrollArea#{SETTINGS_PANEL_NAME} {{
    background: transparent;
    border: none;
}}
QWidget#{EMPTY_STATE_NAME} {{
    background: transparent;
    border: none;
}}
"""


# ``replace=True``: this module owns the name, and a reimport must
# re-register rather than raise and leave every module screen unstyled.
register_widget_qss(SETTINGS_PANEL_NAME, _settings_panel_qss, replace=True)


#: What a module screen tells the user to do when it has nothing more
#: specific to say. Every ``AppScreen`` is the same gesture — fill the form
#: in, press Run — so the instruction is the same sentence.
DEFAULT_INSTRUCTION = "Configure settings, then press Run."


class ModuleHeader(QWidget):
    """The masthead every module page wears: name, description, instruction.

    Three pieces of text in a fixed relationship, and the relationship is
    the point:

    * the **module name**, large — ``DisplayHeading``, which is 30 px
      against a 13 px body;
    * the **description** beside it, one muted line saying what the module
      is for, with the API documentation link after it;
    * the **instruction** under the name, one muted line saying what to do
      on this page.

    Trailing controls — a source label, a table picker, a Load button —
    go on the same row through :meth:`add_trailing`, right-aligned past
    the stretch, so a screen that had its own header row keeps it.

    This was written inline inside :class:`AppScreen` and stayed there,
    which is the whole of the defect it now fixes. Roughly twenty-five
    screens arrived in two days, none of them an ``AppScreen``, and each
    rolled its own header: a ``QLabel`` tagged ``ScreenTitle`` — an object
    name with **no rule anywhere in the stylesheet**, so it rendered at
    body size — or, on four of them, a bare paragraph with no title at
    all. Copying the styling into each of them would have set the same
    trap for the twenty-sixth; a shared component cannot drift.

    Transparent by construction. A header is a page region, not a card,
    and an untagged ``QWidget`` inherits the blanket
    ``QWidget {{ background-color: bg }}`` — the WINDOW colour, which no
    page-opacity setting can reach — so it would sit as an opaque band
    across the backdrop. ``AppScreen`` used to tag its header by hand in
    ``_clear_page_surfaces``; every screen gets it here instead.

    :param title: the module name, shown large.
    :param description: one line to the right of the name. Never wrapped —
      it may shrink below its ideal width rather than force the window
      wider — and repeated as a tooltip so a truncated one is readable.
    :param instruction: one line under the name. Omitted if empty.
    :param app_key: registry key. Given one, the description gets the
      module's API documentation link beside it.
    """

    def __init__(self, title: str, description: str = "",
                 instruction: str = "", *, app_key: Optional[str] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ModuleHeader")
        from ..theme import make_transparent
        make_transparent(self)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["lg"])

        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(2)
        self.title_label = QLabel(str(title))
        self.title_label.setObjectName("DisplayHeading")
        title_col.addWidget(self.title_label)
        self.instruction_label = QLabel(str(instruction or ""))
        self.instruction_label.setObjectName("Muted")
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setVisible(bool(instruction))
        title_col.addWidget(self.instruction_label)
        row.addLayout(title_col)

        self.description_label: Optional[QLabel] = None
        self.info_link = None
        if description:
            intro_row = QHBoxLayout()
            intro_row.setContentsMargins(0, 0, 0, 0)
            intro_row.setSpacing(SPACING["sm"])
            blurb = QLabel(str(description))
            blurb.setObjectName("Muted")
            # One line, flush left. The label may shrink below its ideal
            # width so a long blurb never forces the window wider.
            blurb.setWordWrap(False)
            blurb.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            blurb.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
            blurb.setMinimumWidth(0)
            blurb.setToolTip(str(description))
            intro_row.addWidget(blurb)
            self.description_label = blurb
            if app_key:
                from .settings_model import api_docs_url
                info = InfoLink(api_docs_url(app_key),
                                tooltip=str(description), parent=self)
                info.setObjectName("ModuleInfoLink")
                info.setProperty("moduleApiAppKey", app_key)
                intro_row.addWidget(info)
                self.info_link = info
            row.addLayout(intro_row)

        row.addStretch(1)
        self._row = row

    def add_trailing(self, widget: QWidget, stretch: int = 0) -> QWidget:
        """Put ``widget`` on the header row, right of the stretch.

        For the screens whose header row also carries controls — Control
        Chart's table picker and Load button, Graph Builder's source
        label. They keep their row; they stop having to build the title
        part of it themselves.
        """
        self._row.addWidget(widget, stretch)
        return widget


# The hover description must never reflow the runtime controls. Four lines
# are enough to scan the curated setting descriptions while the full rich
# tooltip remains available beside the field.
HINT_STRIP_LINES = 4

# The category strip sits above it and holds a shorter blurb, so three lines
# is enough. Fixed, for the same reason: the runtime controls above must not
# jump when the pointer crosses a category header.
CATEGORY_STRIP_LINES = 3


# One blurb per settings CATEGORY, keyed by the uppercased category title.
# The table itself lives beside the category map in `settings_model`, because
# that is what decides which categories exist; this module only renders them.
# Re-exported under the historical name so integrations and tests that read
# `app_screen.SECTION_HINTS` keep working.
#
# The blurbs are shown in the strip UNDER the Run / Stop actions row (see
# `_build_runtime_panel` and `_wire_category_hints`), not as a popup over the
# form: a category description is three lines long and a floating tooltip
# covers the very settings it is describing.
SECTION_HINTS = CATEGORY_TOOLTIPS


# Settings whose VALUE is the name of a database column. Each gets a "SQL"
# button that opens the run's measurements.db read-only and shows what is
# actually in it, so a typo cannot silently create a second near-identical
# column. The value is the table to preselect; None lets the user choose.
#
# dependent_variable is deliberately absent: it names a column of the score
# CSV, not of measurements.db, and pointing the picker at the wrong file
# would be worse than having no picker at all.
COLUMN_TABLES = {
    "annotation_column":  "png_list",
    "annotation_columns": "png_list",
    # custom_measurement is gone: it was collected and never read, so a SQL
    # column picker for it offered to fill in a control that did nothing.
    "measurement":        None,
    "exclude":            None,
    "heatmap_feature":    None,
    "location_column":    None,
    "filter_column":      None,
    "col_to_compare":     None,
    "color_by":           None,
    "metadata_type_by":   None,
    "infection_xgb_proba_column": None,
}


def settings_section_maturity(app_key: str, title: str) -> str:
    """Return the least-mature stage applying to one settings section.

    An alpha or beta module colours every one of its settings. A stable
    module can still contain an explicitly experimental ``(Beta)``/``(Alpha)``
    category, in which case that section receives the more cautious stage.
    """
    from ..app import app_stage

    module_stage = app_stage(app_key)
    normalized = str(title or "").strip().lower()
    section_stage = "stable"
    if normalized == "alpha" or "(alpha)" in normalized:
        section_stage = "alpha"
    elif normalized == "beta" or "(beta)" in normalized:
        section_stage = "beta"
    risk = {"alpha": 0, "beta": 1, "stable": 2}
    return min((module_stage, section_stage), key=risk.__getitem__)


APP_TITLES = {
    "mask":            "Mask Generation",
    "timelapse":       "Timelapse",
    "motility":        "Motility Assay",
    "measure":         "Measure",
    "classify_merged": "Classify",
    "classify":        "Classify (CV)",
    "umap":            "Image UMAP",
    "train_cellpose":  "Train Cellpose",
    "cellpose_masks":  "Cellpose Masks",
    "cellpose_all":    "Cellpose (All)",
    "map_barcodes":    "Map Barcodes",
    "queue":           "Plate Queue",
    "ml_analyze":      "Classify (ML)",
    "regression":      "Regression",
    "recruitment":     "Recruitment",
    "activation":      "Activation Maps",
    "analyze_plaques": "Plaque Analysis",
    "invasion":        "Invasion Assay",
    "replication":     "Replication Assay",
    "annotate":        "Annotate",
    "make_masks":      "Make Masks",
    "db_browser":      "Database Browser",
    "agreement":       "Annotator Agreement",
    "plate_view":      "Plate Viewer",
    "model_compare":   "Model Compare",
    "align":           "Align & Stitch",
    "convert":         "Format Converter",
    "foreign":         "Import Project",
    "external_masks":  "External Masks",
    "batch":           "Batch Runner",
    "model_zoo":       "Model Zoo",
    "report":          "Report",
    "train_compare":   "Training Runs",
    "classifier_evaluation": "Classifier Evaluation",
    "run_history":     "Run History",
    "distributed_jobs": "Distributed Jobs",
}


# Short "what this module does" blurbs shown to the right of the header.
APP_INTROS = {
    "mask":            "Segment cells, nuclei, pathogens and organelles with Cellpose and build the merged image+mask arrays.",
    "timelapse":       "Segment each frame of a time series and link objects across frames into tracks, then export per-channel movies.",
    "motility":        "Rebuild per-frame tracks, score velocity and straightness per object, and split them by infection state.",
    "measure":         "Extract per-object intensity + morphology features from masks and write them to the measurements database.",
    "annotate":        "Review single-object image crops on a grid and label them; annotations save back to the database.",
    "classify_merged": "Train a classifier on single objects. Pick the training basis - metadata, annotation or measurement - and the classifier family: Torch on object crops, or gradient boosting on measured features. Classify (CV) and Classify (ML) remain available separately.",
    "classify":        "Train and test Torch computer-vision models (CNNs / transformers) to classify single-object images.",
    "ml_analyze":      "Train a classical ML classifier (XGBoost, random forest, …) on per-object features and score every well.",
    "map_barcodes":    "Map sequencing reads to gRNA barcodes and link them to screen wells.",
    "regression":      "Regress screen phenotypes against guide/gene effects across plates.",
    "make_masks":      "Fine-tune a Cellpose model interactively on your own images.",
    "train_cellpose":  "Train a custom Cellpose model from a labelled dataset.",
    "cellpose_masks":  "Run a Cellpose model over images to generate masks.",
    "activation":      "Generate activation / attention maps to see what a trained model focuses on.",
    "umap":            "Embed single-object images into a UMAP and render the map with image glyphs.",
    "queue":           "Chain several plates through the same pipeline configuration.",
    "db_browser":      "Preview any table in a measurements database, search its columns, and export filtered rows to CSV — read-only.",
    "agreement":       "Score how well two or more annotation passes agree with Cohen's or Fleiss' κ, then review every crop they labelled differently.",
    "plate_view":      "Draw any per-well measurement as a plate heatmap and test whether the outer ring reads differently from the interior — the edge artefact that turns a screen hit into a failed follow-up.",
    "model_compare":   "Segment the same three fields with two Cellpose models and see what changed: masks side by side, object counts, the background-excluded ARI, and whether the extra objects are new cells or fragments of old ones.",
    "report":          "Turn a finished run folder into one self-contained HTML or PDF file — the QC verdict, the key figures, the statistics, the exact settings and the package versions — that a collaborator can open without spaCR.",
    "foreign":         "Take a third party's images, label masks and measurement table and produce a working spaCR project: Yokogawa-named TIFFs, merged arrays, and a measurements.db carrying their columns next to spaCR's plateID/rowID/columnID/fieldID/object_label — with the column mapping shown, editable and unit-checked before anything is written.",
    "external_masks":  "Drop intensity images and label masks made outside spaCR, review the detected cell, nucleus, pathogen and organelle assignments, then build merged arrays, measurements and single-object crops ready for Annotate.",
    "align":           "Stitch an arbitrary number of tiles into one canvas. Offsets are solved globally rather than accumulated, so error does not walk down a row; tiles that failed to register are shown and recorded rather than quietly placed by stage position; and the canvas is written band by band, so its size is never a memory limit.",
    "convert":         "Turn ND2, CZI, LIF or OME-TIFF acquisitions into Yokogawa-named TIFFs spaCR can read, after showing you exactly which source file becomes which target — and write a map file so the original metadata can be joined back onto the measurements afterwards.",
    "batch":           "Stack any combination of modules, plates and settings into a queue and run it unattended — each job is validated when you add it, runs in its own process, and reports what failed, what was skipped because an upstream job failed, and what finished only partly.",
    "model_zoo":       "Every Cellpose and classifier checkpoint this machine can reach, with what it was trained on, whether its bytes check out against a published checksum, and what it does to three of your fields.",
    "train_compare":   "Overlay the loss and accuracy curves of several training runs on one axis and see, beside them, exactly which settings differed — with environment drift bucketed away from the knobs you actually turned.",
    "classifier_evaluation": "Inspect held-out predictions from grouped or nested cross-validation, calibration, confusion matrices, per-plate performance and explicit train/test leakage checks.",
    "run_history":     "Search every recorded job and inspect its exact settings, hashed inputs and outputs, warnings, failure traceback, software versions, seeds and performance.",
    "distributed_jobs": "Submit resolved spaCR settings to SSH workstations, Slurm clusters or cloud/HPC command profiles and monitor them locally.",
    "analyze_plaques": "Detect and quantify plaques in plaque-assay images.",
    "recruitment":     "Quantify recruitment of a marker to a compartment across conditions.",
    "invasion":        "Score every parasite attached or invaded from a two-colour outside/inside stain, with the threshold derived per field and flagged when the two populations it assumes are not actually there.",
    "replication":     "Count the parasites in every vacuole and turn that into a replication rate: endodyogeny doubles a vacuole 1 -> 2 -> 4 -> 8, so the distribution of counts per vacuole is the readout, not the mean.",
}

try:
    from spacr.plugins import plugin_apps as _plugin_apps
    for _plugin_app in _plugin_apps():
        APP_TITLES.setdefault(_plugin_app.key, _plugin_app.name)
        APP_INTROS.setdefault(_plugin_app.key, _plugin_app.description)
except Exception:
    # Discovery records individual failures. Metadata lookup must not prevent
    # the built-in AppScreen class from importing.
    pass


def _absorb_registered_app_metadata() -> None:
    """Take the header and blurb of every registered app into the tables above.

    The PULL half of the app-registration seam.
    :func:`spacr.qt.app.register_app` PUSHES a new app's title and intro
    into these two dicts when this module is already imported; this picks
    up the apps that registered before it was. Between them, which module
    is imported first stops mattering — and it used to matter a great
    deal: a screen that registers itself could not be given a header
    without a hand-edit in this file, so four finished features shipped
    unreachable.

    Read out of :data:`sys.modules` rather than imported, because
    ``spacr.qt.app`` builds this screen and importing it from here would
    be a cycle. ``setdefault``, so the hand-written entries above — where
    the header deliberately differs from the sidebar name ("Mask
    Generation" over the "Mask" tile) — stay the more specific answer.
    """
    app = sys.modules.get("spacr.qt.app")
    # `getattr(..., None)`: `spacr.qt.app` may be half-built when this
    # runs (it imports the widget package before `register_app` exists),
    # in which case there is nothing to pull and the push half of the
    # seam delivers every row later.
    pull = getattr(app, "registered_metadata", None) if app else None
    if pull is None:
        return
    for key, title in pull("title").items():
        APP_TITLES.setdefault(key, title)
    for key, intro in pull("intro").items():
        APP_INTROS.setdefault(key, intro)


_absorb_registered_app_metadata()


#: Apps that get a live DNA-rain backdrop. Sequencing only — every
#: other app key misses this set and the hook below does nothing, so no
#: other screen changes in any way.
DNA_RAIN_APPS = frozenset({"map_barcodes"})


def uses_ambient_background(app_key: str) -> bool:
    """Whether ``app_key``'s screen gets the generic ambient backdrop.

    Every module screen **except** the ones that already animate
    something of their own — which today is exactly
    :data:`DNA_RAIN_APPS`. Sequencing's rain is *about* sequencing:
    bases falling behind the screen that maps reads to barcodes. Putting
    a second, unrelated animation behind it would fight it — two
    independent motions competing for the same pixels, neither readable,
    and two animation timers running on the one screen that already had
    one. So a screen gets one animated background or none, never both.

    Written as a rule with a name rather than as an ``else`` on the
    rain's ``if``: the two backdrops are chosen by *one* decision, and
    the day a second module earns a themed animation of its own, adding
    its key to :data:`DNA_RAIN_APPS`-style membership is all that is
    needed for the ambient one to step aside.

    :param app_key: id of the app (see ``APPS`` in ``spacr.qt.app``).
    :returns: True when the ambient backdrop belongs on that screen.
    """
    return app_key not in DNA_RAIN_APPS


def _discard_widget(widget) -> None:
    """Unparent and delete ``widget``, tolerating any state it is in.

    Used on the failure paths below. A half-installed backdrop that is
    still a child of the screen would keep painting and keep its timer;
    dropping the Python reference alone does not remove it, because Qt
    owns it through its parent.
    """
    if widget is None:
        return
    try:
        widget.setParent(None)
        widget.deleteLater()
    except Exception:
        pass


def _theme_wallpaper():
    """Path of the wallpaper the current theme is painting, or ``None``.

    ``None`` for dark and light, which have no picture — and the rain
    then keeps its cheap opaque-strip fast path. For Space and Cell this
    is the *same* cached file the stylesheet points at: the theme has
    already been applied by the time any screen is built, so the cache
    is warm and this resolves without decoding a master. Handing it to
    the rain is what stops the rain painting flat black over the very
    image the theme exists to show.

    Never raises: no wallpaper is a cosmetic miss, not a broken screen.
    """
    try:
        from ..preferences import (resolve_effective_theme,
                                   theme_background_path)
        return theme_background_path(resolve_effective_theme())
    except Exception:
        return None


#: Settings that were renamed, mapped old -> new. A dict handed to a screen
#: comes from a CSV, a demo pack or another screen, and any of those may have
#: been written before the rename -- so the translation belongs here, at the
#: point a dict meets the widgets, rather than in every producer.
_RENAMED_SETTING_KEYS = {"png_dims": "png_channel_mapping"}


def _translate_legacy_setting_keys(settings: dict) -> dict:
    """Rename retired setting keys so their values still reach a widget.

    Without this a settings CSV holding `png_dims` loads into a screen that
    renders `png_channel_mapping`, finds no widget for it, and drops the
    value on the floor -- the run then uses the module default and the user
    is never told. Caught by
    `test_demo_settings_survive_the_widget_round_trip`.

    The new key wins when both are present: someone who has said outright
    which channel is red must not have it overridden by a stale list.
    `ChannelMappingWidget.set_value` accepts the list form directly, so no
    value conversion is needed here -- only the name.

    :param settings: a settings dict, not modified.
    :returns: a new dict with retired keys renamed.
    """
    out = dict(settings)
    for old, new in _RENAMED_SETTING_KEYS.items():
        if old in out:
            value = out.pop(old)
            out.setdefault(new, value)
    return out


class AppScreen(QWidget):
    """Generic settings + runtime screen used by every non-interactive app.

    Composes the settings model on the left with the console, usage bars,
    figures card, and actions row on the right.

    :param app_key: id of the app (see ``APPS`` in ``spacr.qt.app``).
    :ivar error_explain_requested: emitted with ``(traceback, app_key)``
        when the user clicks "Explain error"; MainWindow routes it to
        the AI Console for backward compatibility.
    """

    # Emitted when the user clicks "Explain error" with the last
    # captured traceback + the app key so MainWindow can route to the
    # AI Console.
    error_explain_requested = Signal(str, str)
    # Hand an immutable settings snapshot to the Distributed Jobs screen.
    # MainWindow owns navigation, so the reusable screen does not reach into
    # the application stack itself.
    remote_submit_requested = Signal(str, dict)

    # Backdrop state, declared on the class so a Qt event that arrives
    # mid-construction (showEvent is delivered from inside a nested
    # layout activation on some styles) finds an answer rather than an
    # AttributeError.
    _ambient = None
    _ambient_applied = None
    _backdrop_applied = None
    _dna_rain = None

    def __init__(self, app_key: str, parent=None):
        super().__init__(parent)
        self.app_key = app_key
        self._last_error_text: str = ""
        self._hint_map: dict = {}       # widget → plain-text hint
        self._html_tip_map: dict = {}   # widget → HTML tooltip (sticky popup)

        # This module is imported lazily by `app.py`, long after the launch
        # stylesheet was generated, so the block registered above is not in
        # it. Without this the settings column opens unpanelled — see
        # `ensure_widget_qss_applied`.
        ensure_widget_qss_applied(SETTINGS_PANEL_NAME)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                  SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        # ─── Header ───────────────────────────────────────────────────
        # The shared masthead — see `ModuleHeader`. This screen was where it
        # was written and for a long time where it stayed, which is how
        # twenty-odd screens ended up with a title at body size.
        header = ModuleHeader(
            APP_TITLES.get(app_key, app_key.title()),
            description=APP_INTROS.get(app_key) or "",
            instruction=DEFAULT_INSTRUCTION,
            app_key=app_key,
        )
        self._header = header
        outer.addWidget(header)

        outer.addWidget(Divider())

        # ─── Body splitter ────────────────────────────────────────────
        body = QSplitter(Qt.Horizontal)
        self._body_splitter = body
        body.setChildrenCollapsible(False)

        # Settings panel (left)
        body.addWidget(self._build_settings_panel())
        # Runtime panel (right)
        body.addWidget(self._build_runtime_panel())

        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 2)
        body.setSizes([400, 800])
        outer.addWidget(body, 1)

        # The live-preview autoload watches ``src``, and can only be wired
        # once BOTH panels exist: the settings panel owns the src field and
        # the runtime panel owns the preview. It used to be wired from
        # _build_empty_state_banner (inside the settings panel), where
        # ``self._live_preview`` does not exist yet — so it never fired.
        self._wire_live_preview_autoload()

        # Same ordering constraint: the sections are built by the settings
        # panel, the strip they describe themselves into belongs to the
        # runtime panel, so the two can only be connected once both exist.
        self._wire_category_hints()

        # Two runners, not one, and the split is deliberate. Both of these are
        # background work — the usage poll shells out to nvidia-smi, filing an
        # issue shells out to `gh` and then talks to api.github.com — but they
        # run on wildly different clocks. `_refresh_usage` skips a tick while
        # its own sample is still out, so that a machine slow enough to still
        # be inside nvidia-smi 2 s later does not accumulate a backlog. Share
        # a runner with the issue report and that guard also swallows every
        # poll for the up-to-28 s an issue report can take, freezing the usage
        # bars for the whole of it.
        self._usage_jobs = JobRunner(self, app_key=f"{self.app_key} usage",
                                    user_visible=False)
        self._jobs = JobRunner(self, app_key=f"{self.app_key} background")

        # Timer to poll RAM/GPU/CPU periodically
        self._usage_timer = QTimer(self)
        self._usage_timer.setInterval(2000)
        self._usage_timer.timeout.connect(self._refresh_usage)
        # A stacked module page may be constructed hours before the user
        # opens it.  Polling every hidden page wastes a thread and a
        # ``nvidia-smi`` subprocess every two seconds; in a long-lived Qt
        # process those orphan polls also made GPUtil's subprocess boundary
        # eventually segfault.  showEvent starts the one page the user can
        # actually see, and hideEvent stops it again.

        # Threading state
        self._thread: Optional[QThread] = None

        # Drag & drop — install a dropzone with this app's per-module
        # handler. Universally accepts settings CSVs; folder policy
        # is app-specific (see spacr.qt.dnd_handlers).
        try:
            from ..dnd import install_dropzone
            from ..dnd_handlers import get_handler
            install_dropzone(self, get_handler(self.app_key), self)
        except Exception:
            pass

        # DNA rain backdrop (sequencing only). Sits behind every other
        # child, takes no focus and no mouse events, and stops its timer
        # whenever this screen is not visible, so it costs nothing while
        # the pipeline runs on another tab. Its colour / speed /
        # visibility / font controls live in a popover behind a DNA
        # button beside the AI toggle — they used to be a permanent bar
        # across the bottom of the page, which is more chrome than a
        # backdrop is worth.
        self._dna_rain = None
        if self.app_key in DNA_RAIN_APPS:
            try:
                from ..widgets.dna_rain import install_dna_rain
                # The rain is lowered behind its siblings, so it is only
                # ever as visible as those siblings are transparent.
                # Under dark and light every container is an opaque `bg`
                # and it was buried completely: the animation ran, cost
                # its frames, and reached the eye only through the few
                # pixels of layout spacing between widgets.
                self._clear_page_surfaces()
                # The page colour follows whether a backdrop got installed, so
                # it is resolved wherever that is decided -- and it has to reach
                # QPalette.Window, not just paintEvent, or Qt's pre-paint erase
                # still uses `bg` and flashes black. See _sync_page_palette.
                self._sync_page_palette()
                self._dna_rain = install_dna_rain(
                    self, outer, backdrop=_theme_wallpaper())
            except Exception:
                self._dna_rain = None

        # Ambient backdrop — the drifting blobs (or whichever theme the
        # user picked) behind every screen that does NOT already animate
        # something of its own. See `uses_ambient_background` for why
        # that is one rule and not an `else` on the branch above; the
        # two are mutually exclusive by construction, so no screen ever
        # carries both.
        #
        # Same hard-won contract as the rain: lowered behind every
        # sibling, no focus, no mouse events, and its timer stops
        # whenever this screen is not visible — these screens stay open
        # while the pipeline runs on another tab, so an animation that
        # kept ticking off-screen would cost a core for nobody.
        self._ambient = None
        #: (theme, palette) last pushed at — or attempted on — the
        #: widget, so a tab switch that changed nothing neither restarts
        #: the animation nor retries an install that already failed.
        self._ambient_applied = None
        if uses_ambient_background(self.app_key):
            self._install_ambient()

        # And unconditionally, whatever happened above. This used to run ONLY
        # as a side effect of installing an animation — the DNA rain calls it
        # before, `_install_ambient` after — on the reasoning that a screen
        # with nothing behind it should be left opaque rather than transparent
        # over emptiness.
        #
        # That reasoning was wrong, and it is what made the settings half of
        # every module screen a solid black rectangle for anybody who had
        # turned the ambient backdrop off in Preferences: `_install_ambient`
        # returns early when the preference is off, so the sweep never ran, so
        # every layout container on the page kept the blanket
        # `QWidget { background-color: bg }` — the WINDOW colour, which no
        # page-opacity setting can reach. Measured over a probe backdrop with
        # the preference off, the settings column, the categories, the gaps
        # between them and the console box all read 0.000: the whole page was
        # one opaque slab and only the cards on top of it looked deliberate.
        #
        # There is never "nothing behind it". With no animation the thing
        # behind is the window's own `bg`, which is the theme — exactly what
        # the page is supposed to show between the floating category panels.
        # `clear_container_surfaces` is idempotent, so the calls inside the
        # two install paths stay where they are for their own ordering
        # reasons and this one costs a second pass over the tree.
        self._clear_page_surfaces()
        # The page colour follows whether a backdrop got installed, so
        # it is resolved wherever that is decided -- and it has to reach
        # QPalette.Window, not just paintEvent, or Qt's pre-paint erase
        # still uses `bg` and flashes black. See _sync_page_palette.
        self._sync_page_palette()

    # ------------------------------------------------------------------
    # Ambient backdrop
    # ------------------------------------------------------------------
    def _install_ambient(self) -> None:
        """Build the ambient backdrop for this screen, if it is wanted.

        Never raises. A decorative background must never be able to stop
        a module screen from opening: a missing widget module, a bad
        persisted theme name, a driver that cannot make the pixmap — any
        of those leaves ``self._ambient`` at ``None`` and the screen
        exactly as it would have been without the feature.

        Two deliberate orderings:

        * the preference is read **before** anything is constructed. Off
          means *not built*, not built-and-hidden — the construction is
          itself the cost the toggle exists to avoid on a machine that
          is running Cellpose on the GPU and a 40-plate pipeline.
        * :meth:`_clear_page_surfaces` runs **after** a successful
          install, where the DNA rain runs it before. It is needed for
          the same reason (under every theme the containers are an
          opaque ``bg``, and one of them is enough to bury the animation
          completely — it would run, cost its frames and reach the eye
          through a few pixels of layout spacing), but doing it second
          means a screen whose install failed is left opaque and normal
          rather than transparent with nothing behind it.

        A failure is remembered, not retried. ``_ambient_applied`` holds
        the (theme, palette) pair that was last *attempted*, and the
        same pair is never attempted twice — otherwise a machine with no
        working ambient module would re-import it and re-fail on every
        palette event, which a stylesheet re-apply raises. A preference
        change moves the pair and the attempt happens again.
        """
        if self._ambient is not None:
            return
        widget = None
        try:
            from ..preferences import (get_ambient_enabled,
                                       get_ambient_palette,
                                       get_ambient_theme)
            if not get_ambient_enabled():
                return
            wanted = (get_ambient_theme(), get_ambient_palette())
            if wanted == self._ambient_applied:
                return
            self._ambient_applied = wanted
            from ..widgets.ambient import install_ambient
            widget = install_ambient(self, None, theme=wanted[0],
                                     palette=wanted[1],
                                     backdrop=_theme_wallpaper())
            self._clear_page_surfaces()
            # The page colour follows whether a backdrop got installed, so
            # it is resolved wherever that is decided -- and it has to reach
            # QPalette.Window, not just paintEvent, or Qt's pre-paint erase
            # still uses `bg` and flashes black. See _sync_page_palette.
            self._sync_page_palette()
            self._ambient = widget
        except Exception:
            self._ambient = None
            _discard_widget(widget)
            self._discard_orphan_ambient()

    def _discard_orphan_ambient(self) -> None:
        """Remove an ambient widget an aborted install left parented here.

        ``install_ambient`` makes the widget a child of this screen
        before it finishes wiring it up, so an installer that raises
        half way through does not hand anything back to unparent — and
        an invisible leftover would still be a child with a timer. The
        screen owns its children, so it is the one that can find it.
        """
        try:
            from ..widgets.ambient import AmbientWidget
        except Exception:
            # No class, no way to recognise one — and if the import is
            # what failed, nothing was constructed to leave behind.
            return
        for child in list(self.children()):
            if isinstance(child, AmbientWidget):
                try:
                    child.set_animating(False)
                except Exception:
                    pass
                _discard_widget(child)

    def _remove_ambient(self) -> None:
        """Tear the ambient backdrop down. Safe when there is none."""
        widget, self._ambient = self._ambient, None
        self._ambient_applied = None
        if widget is None:
            return
        try:
            widget.set_animating(False)
        except Exception:
            pass
        _discard_widget(widget)
        # `page_fill` returns a colour only while there is no backdrop, so
        # taking the animation away is exactly the moment this screen
        # becomes responsible for its own page. Without the repaint the
        # Preferences toggle leaves the hole it used to leave for good.
        self._sync_page_palette()
        self.update()

    def refresh_ambient_background(self) -> None:
        """Re-read the ambient preferences and apply them to this screen.

        The restart-free path for the Preferences toggle: turning it off
        deletes the widget outright rather than hiding it, turning it on
        builds one on a screen that has been open all along, and a new
        theme/palette is pushed at the existing one without rebuilding
        it. Idempotent, and cheap enough to call on every show.

        Never raises, for the same reason the install does not.
        """
        if not uses_ambient_background(self.app_key):
            # Belt and braces: sequencing must not acquire one through
            # this path either.
            self._remove_ambient()
            return
        try:
            from ..preferences import (get_ambient_enabled,
                                       get_ambient_palette,
                                       get_ambient_theme)
            enabled = bool(get_ambient_enabled())
        except Exception:
            return
        if not enabled:
            self._remove_ambient()
            return
        if self._ambient is None:
            self._install_ambient()
            return
        try:
            wanted = (get_ambient_theme(), get_ambient_palette())
        except Exception:
            return
        if wanted == self._ambient_applied:
            # Nothing changed. Re-applying would restart the animation
            # every time the user switches back to this tab.
            return
        try:
            self._ambient.set_theme(wanted[0])
            self._ambient.set_palette(wanted[1])
            self._ambient_applied = wanted
        except Exception:
            pass

    def changeEvent(self, event) -> None:
        """Follow a live theme switch.

        Only Home is rebuilt when the theme changes; every other screen
        is re-styled in place by re-applying the QSS. That is enough for
        anything whose colours come from the stylesheet, and not enough
        for a backdrop that paints itself — the DNA rain and the ambient
        background both capture their flat fill colour and their
        wallpaper at construction. Switching from dark to light left a
        black rain rectangle on a white page, and switching into Cell
        left it painting flat black over the micrograph the theme had
        just loaded. The ambient backdrop has exactly the same two
        captured values and therefore exactly the same two bugs.

        Both palette events count, and that is the whole reason this
        works. ``QApplication.setPalette`` — which is what
        :func:`spacr.qt.theme.apply_qpalette` ends in — delivers
        ``ApplicationPaletteChange`` **only to top-level widgets** (Qt
        6.11, verified); every child, including every AppScreen inside
        MainWindow's stack, gets ``PaletteChange`` instead. Listening
        for the application event alone meant this handler fired in the
        tests that synthesised it and never once in the running app.

        Saving Preferences goes through the same call, so this is also
        where an ambient *preference* change lands on a screen that is
        already open — including the toggle switching back **on**, which
        has to build a widget that does not exist yet and so cannot be
        done by anything walking the live widget tree.

        What is deliberately *not* re-applied is anything the user
        picked: the rain's trail colour (it has a swatch in its settings
        bar) and the ambient theme + palette (they are Preferences
        entries). Silently resetting a choice the user made is worse
        than a slightly off-theme one.
        """
        super().changeEvent(event)
        if event.type() not in (QEvent.ApplicationPaletteChange,
                                QEvent.PaletteChange):
            return
        self.refresh_ambient_background()
        self._retheme_backdrops()
        # The page colour is resolved at paint time from the live theme,
        # so a theme switch has to ask for a repaint — nothing else on
        # this screen invalidates it. The palette moves with it, or Qt
        # keeps erasing to the OLD theme's page between the two.
        self._sync_page_palette()
        self.update()

    def _retheme_backdrops(self) -> None:
        """Re-apply the current theme's fill + wallpaper to both backdrops.

        Resolved once and compared against what was last pushed, because
        ``PaletteChange`` is a far chattier event than the application
        one: re-applying a stylesheet raises it too, and every
        ``set_background_color`` costs the rain its whole pre-rendered
        strip cache and a full repaint. Nothing changed means nothing is
        touched.

        ``set_backdrop`` is optional. The DNA rain has one; the ambient
        widget's published API is ``set_background_color`` /
        ``set_theme`` / ``set_palette`` / ``set_animating``. A backdrop
        without the method keeps whatever wallpaper it was built with,
        which is a cosmetic miss on the image themes only — not a reason
        to skip the flat fill, which is what fixes the black-rectangle
        case on dark -> light.

        The fill is ``page``, not ``bg``. It used to be ``bg``, which
        meant that on the dark theme every palette event — and re-applying
        the stylesheet raises one — pushed ``#000000`` back into a
        backdrop that had been built with the page colour. A backdrop
        that is correct only until the next theme refresh is not correct.
        """
        backdrops = [w for w in (getattr(self, "_dna_rain", None),
                                 getattr(self, "_ambient", None))
                     if w is not None]
        if not backdrops:
            return
        try:
            from ..theme import page_colour
            from ..preferences import resolve_effective_theme
            theme = resolve_effective_theme()
            fill = page_colour(theme)
            wallpaper = _theme_wallpaper()
        except Exception:
            return
        if (fill, wallpaper) == self._backdrop_applied:
            return
        self._backdrop_applied = (fill, wallpaper)
        for widget in backdrops:
            try:
                widget.set_background_color(fill)
            except Exception:
                pass
            try:
                set_backdrop = getattr(widget, "set_backdrop", None)
                if callable(set_backdrop):
                    set_backdrop(wallpaper)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # The page itself
    # ------------------------------------------------------------------
    def page_fill(self):
        """The flat colour this screen paints itself, or ``None``.

        ``_clear_page_surfaces`` makes every layout container transparent
        so that whatever is behind them shows through. That is right, and
        it is only half a page: something still has to *be* behind them.
        With an animation installed that something is the animation. With
        the ambient preference off, or the Animation preference set to
        ``none``, nothing was — so the containers showed the blanket
        ``QWidget {{ background-color: bg }}``, which on the dark theme is
        ``#000000``. That is the black box behind the settings categories,
        reported three times: not a container the sweep missed, a page
        with no colour of its own.

        ``None`` — meaning "let the stylesheet paint what it always did" —
        in exactly two cases:

        * a backdrop is installed. It covers the screen and paints its own
          fill, so a second full-rect fill under it is wasted work.
        * an image theme. There the window paints the wallpaper (or, with
          no cached image, a gradient in the theme's own hues) and
          ``QWidget`` is transparent precisely so it shows through; a flat
          fill here would paint over the picture the theme exists for.

        Never raises: a page that cannot resolve its colour falls back to
        the rendering it had before this existed.
        """
        if self._ambient is not None or self._dna_rain is not None:
            return None
        try:
            from ..preferences import resolve_effective_theme
            from ..theme import IMAGE_THEMES, page_colour
            theme = resolve_effective_theme()
            if theme in IMAGE_THEMES:
                return None
            return QColor(page_colour(theme))
        except Exception:
            return None

    def _sync_page_palette(self) -> None:
        """Put the page colour in ``QPalette.Window``, not only in the paint.

        ``paintEvent`` alone is not enough, and the difference is visible.
        Qt erases a damaged region to the widget's background *before*
        calling ``paintEvent``, and this widget's background role is
        ``bg`` — ``#000000`` on the dark theme. In a settled frame the fill
        lands on top and nothing shows. Between the erase and the paint —
        during the repaint storms that come with a resize, an expose, a
        theme switch, or the Preferences ambient toggle — the erase is what
        is on screen: a black box that appears and disappears on its own.

        That is also why rendering the screen offscreen could not
        reproduce it. ``QWidget.render`` forces one full synchronous paint,
        so the erase never gets a chance to be seen, and the measurement
        came back clean for a screen the user was watching flash.

        Setting the role makes Qt's own erase the page colour, so there is
        no ordering in which black can appear. The ``paintEvent`` fill
        stays: it is what covers the stylesheet's ``bg`` slab, which the
        palette does not reach.
        """
        # Re-entrancy guard, and it is not theoretical: `setPalette` posts a
        # `PaletteChange`, `changeEvent` handles `PaletteChange` by calling
        # this method, and the second call sets the palette again. That
        # recursed until the stack ran out -- a core dump on startup, not a
        # flicker. The flag makes the nested call a no-op; the outer one
        # finishes the work.
        if getattr(self, "_syncing_page", False):
            return
        colour = self.page_fill()
        # Idempotent, so the re-polish a stylesheet change triggers cannot
        # turn into a repaint loop of its own on a screen that is already
        # showing the right colour.
        applied = getattr(self, "_page_applied", "unset")
        wanted = None if colour is None else colour.name()
        if applied == wanted:
            return

        self._syncing_page = True
        try:
            if colour is None:
                # Back to whatever the stylesheet and the app palette say.
                self.setAutoFillBackground(False)
                self.setPalette(QPalette())
                self.setStyleSheet("")
            else:
                palette = QPalette(self.palette())
                palette.setColor(QPalette.Window, colour)
                self.setPalette(palette)
                self.setAutoFillBackground(True)
                # And in the screen's OWN stylesheet, which is what makes
                # this stick. `autoFillBackground` is not ours to hold: the
                # surface sweep and the theme passes both walk this tree
                # setting it, screens are built and re-themed in an order
                # that is not fixed, and more than one AppScreen is alive
                # during startup. Whoever runs last wins, and when the loser
                # was this method the erase went back to `bg` -- black at
                # launch, cured by any Preferences change that re-ran the
                # sync, black again on the next launch. Exactly the report.
                #
                # A type selector, so it applies to this screen and not to
                # the children it would otherwise cascade to: the cards and
                # panels carry their own surface colour at the page opacity,
                # and painting the page colour onto them would flatten the
                # layering the scheme is built on.
                self.setStyleSheet(
                    f"AppScreen {{ background-color: {colour.name()}; }}")
            self._page_applied = wanted
        finally:
            self._syncing_page = False

    def paintEvent(self, event) -> None:
        """Paint the page under everything this screen lays out.

        Deliberately does **not** chain to ``super()`` when it fills. The
        base implementation is what draws the stylesheet background, and
        the stylesheet background is the ``bg`` slab being replaced —
        calling it afterwards would paint black straight back over this.
        """
        colour = self.page_fill()
        if colour is None:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), colour)
        finally:
            painter.end()

    def _clear_page_surfaces(self) -> None:
        """Stop this screen's layout containers painting over the backdrop.

        The header (which carries the screen title), the body splitter,
        the settings scroll area with its viewport and content widget,
        and the two runtime wrappers are *pages*: they position things
        and should show whatever is behind them. The cards inside them —
        ``Section``, ``Card``, the console — are not tagged and stay the
        readable surface the settings form sits on, at the page opacity,
        which is exactly the "grey categories over the animated
        background" layering this screen is supposed to have.

        Every plain ``QWidget`` used as a container has to be listed. An
        untagged one inherits the blanket ``QWidget {{ background-color: bg }}``
        rule and paints the WINDOW colour — not a surface — so no opacity
        setting can reach it. That is what left a black slab spanning the
        console and the chat box, and black boxes behind the live-view images,
        after the boxes on top of them were thinned.
        """
        from ..theme import clear_container_surfaces, make_transparent

        # The same generic sweep every other screen uses. This used to be a
        # hand-written list plus scroll areas and splitters, which is the
        # shape that kept missing things on Home too: an ANONYMOUS QWidget
        # used as a container has no QSS rule of its own, so it paints the
        # window colour and no opacity setting can reach it. That is what left
        # a black box under the AI chat box and a dead black rectangle between
        # the chat and the System panel.
        clear_container_surfaces(self)

        make_transparent(
            getattr(self, "_header", None),
            getattr(self, "_body_splitter", None),
            getattr(self, "_settings_scroll", None),
            getattr(self, "_settings_content", None),
            getattr(self, "_runtime_wrap", None),
            getattr(self, "_console_wrap", None),
            # The Run / Stop / Import / Clear strip. It has no object name of
            # its own, so without this it takes the blanket window fill and
            # sits as an opaque band across the backdrop.
            getattr(self, "_actions_row", None),
            # The category blurb under it. Named (so a stylesheet can reach
            # it), which is exactly why the generic anonymous-container sweep
            # above leaves it alone.
            getattr(self, "_category_hint", None),
        )

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------
    def _build_settings_panel(self) -> QWidget:
        scroll = QScrollArea()
        self._settings_scroll = scroll
        # The column paints nothing — see `_settings_panel_qss`. The name is
        # what that block keys off; without it the scroll area falls through
        # to the blanket window fill, which is where this started.
        scroll.setObjectName(SETTINGS_PANEL_NAME)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        if self.app_key == "umap":
            # The action strip now carries GPU, search, interactive and AI
            # toggles. Never satisfy their combined width by crushing the
            # reducer settings into an unreadable sliver; the top-level window
            # may grow, while the splitter remains user-resizable.
            scroll.setMinimumWidth(280)
        # A QScrollArea's viewport auto-fills by default, and what it fills
        # with is the WINDOW colour -- not a surface -- so no opacity setting
        # can reach it and the settings column reads as an opaque slab over
        # the animated backdrop. The sidebar (app.py) and Home
        # (widgets/home.py) already say this for their own scroll areas; this
        # one was the odd one out. The column still scrolls; it just does not
        # paint.
        scroll.viewport().setAutoFillBackground(False)

        content = QWidget()
        self._settings_content = content
        layout = QVBoxLayout(content)
        # No box round the categories, so the only inset is the gutter that
        # keeps them clear of the scrollbar. The spacing below is what makes
        # them read as separate floating panels: it is where the theme shows
        # between one category and the next.
        layout.setContentsMargins(0, 0, SPACING["sm"], 0)
        layout.setSpacing(SPACING["sm"])

        self._settings_model = SettingsWidgets(self.app_key, parent=content)
        try:
            sections = self._settings_model.build_sections()
        except Exception as e:
            err = QLabel(f"Failed to build settings for '{self.app_key}': {e}")
            err.setWordWrap(True)
            layout.addWidget(err)
            scroll.setWidget(content)
            return scroll

        # Empty-state banner — shown ONLY when the src widget is
        # empty. It's a compact "Drop a plate folder here or pick a
        # Demo dataset" card that sits above the settings form; it
        # auto-hides as soon as the user sets src (drag/drop or
        # typing). Users who load settings via Import don't see it
        # a second time.
        self._empty_state_card = self._build_empty_state_banner()
        if self._empty_state_card is not None:
            layout.addWidget(self._empty_state_card)

        self._settings_sections = []
        self._maturity_notice = QLabel()
        self._maturity_notice.setObjectName("MaturityVisibilityNotice")
        self._maturity_notice.setWordWrap(True)
        self._maturity_notice.hide()
        layout.addWidget(self._maturity_notice)

        if not sections:
            layout.addWidget(QLabel("No settings defined for this app."))
        # Map widget → plain-text hint so the bottom hint strip AND our
        # sticky HoverTooltip can look up the description for the object
        # under the cursor. Initialized in __init__.
        for title, rows in sections:
            section = Section(title)
            section.set_maturity(
                settings_section_maturity(self.app_key, title)
            )
            self._settings_sections.append(section)
            # The category blurb. Its primary home is the strip under the
            # actions row (see `_wire_category_hints`); `set_hint` keeps the
            # same text on the header for screen readers and for the
            # beta/alpha caution note it appends. `category_tooltip` resolves
            # the module's own override first, then the shared table, then a
            # generic sentence, so a section is never left without text.
            section.set_hint(category_tooltip(self.app_key, title))
            for label, widget in rows:
                lbl_widget = QLabel(label)
                # Give the label a subtle affordance so users know
                # it's the hover target for tooltips (fields can be
                # focused / clicked — tooltips on labels are calmer).
                lbl_widget.setCursor(Qt.WhatsThisCursor)
                field_key = None
                for key, w in getattr(self._settings_model, "_widgets", {}).items():
                    if w is widget:
                        field_key = key
                        html = widget.toolTip()
                        hint = self._settings_model.plain_tooltip_for(key)
                        body_source = widget.property(
                            "apiTooltipDescriptionSource") or ""
                        lbl_widget.setProperty("settingsAppKey", self.app_key)
                        lbl_widget.setProperty("settingKey", key)
                        lbl_widget.setProperty(
                            "apiTooltipDescriptionSource", body_source)
                        lbl_widget.setProperty(
                            "apiTooltipDescription", body_source)
                        lbl_widget.setProperty("apiTooltipHtml", html)
                        lbl_widget.setProperty(
                            "apiTooltipDisplayRole", "tooltip")
                        # Tooltips live on the LABEL only — hovering
                        # the input field itself is left alone so
                        # focus / edit interactions aren't disturbed.
                        widget.setToolTip("")
                        # SettingsWidgets may already have disabled an
                        # algorithm-specific field before this visual label
                        # exists. Bind them now and mirror the state; later
                        # reducer switches update both through the same link.
                        widget._spacr_setting_label = lbl_widget
                        lbl_widget.setEnabled(widget.isEnabled())
                        self._hint_map[lbl_widget] = hint
                        self._html_tip_map[lbl_widget] = html
                        lbl_widget.installEventFilter(self)
                        break
                # No API link dot on the settings form. It sat between the
                # label and the field and carried a tooltip of its own, so
                # the help popped when the pointer was over the row's
                # right-hand side -- which reads as "the field has a
                # tooltip", because from the user's side of the screen that
                # is exactly what it looks like. 191 of them on the Mask
                # form alone.
                #
                # Nothing is lost but the mark: the API link is still in the
                # label's tooltip HTML (the `href=` several tests assert on),
                # so the reference is one hover and one click away, and the
                # help itself is unchanged and still on the label.
                #
                # The host stays, though, and is built here rather than by
                # `Section.add_row` (which only makes one when there is an
                # info widget to put in it). It is what right-aligns the
                # label against the field: dropping it left the label
                # left-aligned and half the row's width was suddenly the
                # page showing through rather than the category surface.
                section.add_row(lbl_widget, widget, info_widget=None,
                                wrap_label=True)
                self._attach_column_picker(field_key, widget)
            layout.addWidget(section)

        self.refresh_maturity_visibility()
        layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def refresh_maturity_visibility(self) -> None:
        """Show/hide Alpha and Beta settings without discarding typed values."""
        from ..preferences import maturity_is_visible

        hidden_stages = set()
        for section in getattr(self, "_settings_sections", []):
            visible = maturity_is_visible(section.maturity())
            section.setVisible(visible)
            if not visible:
                hidden_stages.add(section.maturity())

        notice = getattr(self, "_maturity_notice", None)
        if notice is None:
            return
        if hidden_stages:
            labels = " and ".join(stage.title()
                                  for stage in ("alpha", "beta")
                                  if stage in hidden_stages)
            notice.setText(
                f"{labels} settings are hidden by Preferences. "
                "Enable them in Preferences → Feature maturity."
            )
            notice.show()
        else:
            notice.hide()

    def _attach_column_picker(self, key, widget) -> None:
        """Give a column-name field its "SQL" button; a no-op for anything else.

        :param key: the settings key this widget collects, or None.
        :param widget: the input widget already installed in its Section.
        """
        if key not in COLUMN_TABLES:
            return
        from ..widgets.column_picker import attach_column_picker
        attach_column_picker(widget, self._settings_src_path,
                             COLUMN_TABLES[key])

    def _settings_src_path(self) -> str:
        """Return the run folder the src field currently names.

        Read on demand rather than captured, so the picker follows the source
        folder the user has typed rather than whatever it was at build time.
        Classify's source is list-valued; its SQL picker uses the first plate,
        which is the same database the single-source picker historically
        opened.
        """
        from PySide6.QtWidgets import QLineEdit
        src = getattr(self._settings_model, "_widgets", {}).get("src")
        if isinstance(src, QLineEdit):
            return src.text().strip()
        getter = getattr(src, "get_value", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:
                return ""
            if isinstance(value, (list, tuple)):
                return next(
                    (str(item).strip() for item in value
                     if str(item).strip()),
                    "",
                )
            return "" if value is None else str(value).strip()
        return ""

    def _build_empty_state_banner(self):
        """Return a compact "Drop or pick a demo" card, or None.

        The card is inserted at the top of the settings scroll. It
        hides once the ``src`` widget contains anything so users
        who've already pointed the app at data see the normal form.
        """
        from PySide6.QtWidgets import QLineEdit
        from ..widgets import EmptyState

        src_widget = None
        try:
            src_widget = self._settings_model._widgets.get("src")
        except Exception:
            pass
        if src_widget is None:
            return None

        # If src already points at a real path, don't show the banner.
        # `path`, `""` and None are all placeholders the settings dicts
        # use as "no src set yet".
        existing = ""
        if isinstance(src_widget, QLineEdit):
            existing = (src_widget.text() or "").strip()
        placeholders = {"", "path", "/path/to/src", "/path"}
        if existing and existing not in placeholders:
            return None

        # Human-friendly title varies per app; the body is the same.
        title = f"Point {APP_TITLES.get(self.app_key, self.app_key).lower()} at some data"
        # ...and so does the demo. This named "Demos → Mask demo…" on
        # every screen, so Measure, Timelapse, Classify and Sequencing all
        # offered a dataset that opens a DIFFERENT module: following the
        # hint on the Measure screen generates raw images, navigates away
        # to Mask, and leaves the empty screen the user was trying to fill
        # exactly as empty. Ask which demo lands HERE, and say nothing
        # specific when none does.
        try:
            from ..app import demo_label_for_app
            demo = demo_label_for_app(self.app_key)
        except Exception:
            demo = None
        offer = (f"use Demos → {demo} for a synthetic dataset"
                 if demo else "pick a dataset from the Demos menu")
        subtitle = (
            f"Drop a folder of images anywhere on this window, or {offer}. "
            "You can also type a path into the src field below."
        )
        card = EmptyState(
            title=title, subtitle=subtitle,
            cta_label="Open Demos menu",
            on_action=lambda: self._open_demos_menu(),
        )
        # Auto-hide once the user sets src
        if isinstance(src_widget, QLineEdit):
            src_widget.textChanged.connect(self._maybe_hide_empty_state)
        card.setObjectName(EMPTY_STATE_NAME)
        return card

    def _wire_live_preview_autoload(self) -> None:
        """Feed the first tile under ``src`` into the live-preview panel.

        Called from ``__init__`` once both panels exist. Deferred through a
        single-shot timer so a user typing a path doesn't trigger a directory
        walk per keystroke.

        Wiring this from ``_build_empty_state_banner`` (as it used to be) was
        a no-op twice over: that runs while the SETTINGS panel is being built,
        before ``_build_runtime_panel`` has created ``_live_preview``, and it
        only ran at all when the banner was shown — i.e. never for a screen
        whose ``src`` was already set.
        """
        from PySide6.QtWidgets import QLineEdit
        if getattr(self, "_live_preview", None) is None:
            return
        src_widget = getattr(self._settings_model, "_widgets", {}).get("src")
        if not isinstance(src_widget, QLineEdit):
            return
        self._live_src_timer = QTimer(self)
        self._live_src_timer.setSingleShot(True)
        self._live_src_timer.setInterval(400)
        self._live_src_timer.timeout.connect(
            lambda w=src_widget: self._autoload_live_preview(w.text()))
        src_widget.textChanged.connect(lambda _t: self._live_src_timer.start())

    def _maybe_hide_empty_state(self, text: str) -> None:
        card = getattr(self, "_empty_state_card", None)
        if card is None:
            return
        t = (text or "").strip()
        placeholders = {"", "path", "/path/to/src", "/path"}
        if t and t not in placeholders:
            card.hide()

    def _autoload_live_preview(self, src: str) -> None:
        """Ask the preview panel to discover/decode ``src`` asynchronously.

        Silent if ``src`` is empty or a placeholder. Directory traversal and
        image decoding both happen in the panel's worker so a large plate or
        slow NAS mount cannot freeze Qt.
        """
        panel = getattr(self, "_live_preview", None)
        if panel is None:
            return
        s = (src or "").strip()
        if not s or s in {"path", "/path/to/src", "/path"}:
            return
        panel.load_source_async(s)

    def _open_demos_menu(self) -> None:
        try:
            mw = self.window()
            if mw is None:
                return
            for act in mw.menuBar().actions():
                if act.text().replace("&", "") == "Demos":
                    m = act.menu()
                    if m is not None:
                        # Show the menu at the top-left of the window
                        m.exec(mw.mapToGlobal(mw.rect().topLeft()))
                    break
        except Exception:
            pass

    def eventFilter(self, obj, event):
        """Show/hide the hover tooltip and update the hint strip on Enter/Leave."""
        from PySide6.QtCore import QEvent
        from ..widgets.hover_tooltip import HoverTooltip
        # A settings CATEGORY header writes its own strip and nothing else:
        # it has no setting key, so falling through would blank the
        # per-setting strip every time the pointer crossed a header.
        category = obj.property("settingsCategory")
        if category:
            if event.type() == QEvent.Enter:
                self.show_category_hint(str(category))
            elif event.type() == QEvent.Leave:
                self.clear_category_hint()
            return super().eventFilter(obj, event)
        if event.type() == QEvent.Enter:
            key = obj.property("settingKey")
            if key:
                from .settings_model import refresh_api_tooltips
                refresh_api_tooltips(obj)
                hint = self._settings_model.plain_tooltip_for(str(key))
                html = obj.property("apiTooltipHtml")
                self._hint_map[obj] = hint
                self._html_tip_map[obj] = html
            else:
                hint = self._hint_map.get(obj)
                html = self._html_tip_map.get(obj)
            if hint and hasattr(self, "_hint_strip"):
                self._hint_strip.setText(hint)
            if html:
                HoverTooltip.instance().show_for(obj, html)
        elif event.type() == QEvent.Leave:
            if hasattr(self, "_hint_strip"):
                self._hint_strip.setText(self._default_hint())
            HoverTooltip.instance().start_hide()
        return super().eventFilter(obj, event)

    def _default_hint(self) -> str:
        return "Hover any setting for details, or select ⓘ for documentation."

    def _build_runtime_panel(self) -> QWidget:
        wrap = QWidget()
        self._runtime_wrap = wrap
        layout = QVBoxLayout(wrap)
        # Small left inset so the console, chat and button row sit slightly
        # away from the container's left edge (aligned with each other).
        layout.setContentsMargins(SPACING["sm"], 0, 0, 0)
        layout.setSpacing(SPACING["md"])

        # Figures card — hidden until the pipeline pushes a figure via
        # PipelineWorker.figure_ready. Sits ABOVE the console (like the
        # live-preview view). The FigureQueue widget owns the thumbnail
        # strip + zoomable enlarged view + forward/back nav + the
        # 100-in-RAM / temp-spill memory management. See
        # spacr.qt.widgets.figure_queue.
        from ..widgets.figure_queue import FigureQueue
        self._figures_card = Card(title="Figures")
        self._figure_queue = FigureQueue(parent=self._figures_card)
        self._figures_card.body_layout.addWidget(self._figure_queue, 1)
        # "Figure settings…" on the NON-LIVE figure holds every Image UMAP
        # setting, live against the figure on screen (instruction 75), and a
        # Propagate button. Propagate means the same thing here as everywhere
        # else in the app: write the values into THIS module's settings
        # panel, which is what the next Run reads and what is saved with the
        # run. Wired for every module, not just UMAP -- the figure colours
        # and text size propagate the same way.
        self._figure_queue.set_propagate_callback(
            self._propagate_live_settings)
        self._umap_explorer = None
        self._umap_payload_ready = False
        if self.app_key == "umap":
            from ..widgets import ImageUmapExplorer
            self._umap_explorer = ImageUmapExplorer(
                parent=self._figures_card)
            self._umap_explorer.hide()
            # The same propagate seam the Mask live preview uses, so a value
            # tuned in the explorer's display window lands in the settings
            # panel and is saved with the run rather than living only in the
            # widget. The getter lets that window open showing the CURRENT
            # run settings for the half it does not itself hold -- without
            # it, figure size and image count open as zeros.
            self._umap_explorer.set_propagate_callback(
                self._propagate_live_settings)
            self._umap_explorer._settings_getter = self._umap_display_defaults
            self._figures_card.body_layout.addWidget(
                self._umap_explorer, 1)
        self._figures_card.setMinimumHeight(360)
        self._figures_card.hide()
        layout.addWidget(self._figures_card, 1)

        # Live-preview segmentation — Mask app only. The card + the
        # console below live in a vertical QSplitter so the user can
        # drag the divider up (bigger console) or down (bigger preview)
        # depending on whether they're tuning parameters or watching
        # a run. Non-Mask apps get the console alone.
        from ..widgets import ConsolePanel
        app_title = APP_TITLES.get(self.app_key, self.app_key.title())
        console_wrap = QWidget()
        self._console_wrap = console_wrap
        console_col = QVBoxLayout(console_wrap)
        console_col.setContentsMargins(0, 0, 0, 0)
        console_col.setSpacing(4)
        console_header = QLabel("Console")
        console_header.setObjectName("CardTitle")
        console_col.addWidget(console_header)
        # `persist_key` is what lets the console remember where the user put
        # the divider between its output box and the AI chat box, per screen:
        # a tall chat box on Mask does not force one on Sequencing.
        self._console = ConsolePanel(active_app_label=app_title,
                                     persist_key=self.app_key)
        self._console.setMinimumHeight(180)
        console_col.addWidget(self._console, 1)

        # Exactly one of these cards occupies the slot above the console.
        # Nulled here rather than in every branch: the chain has grown to six
        # arms and a branch that forgets one leaves a stale attribute from a
        # previous screen.
        self._live_preview = self._live_preview_card = None
        self._measure_preview = self._measure_preview_card = None
        self._hyperparam = self._hyperparam_card = None
        self._timelapse_preview = self._timelapse_preview_card = None
        self._motility_preview = self._motility_preview_card = None
        self._runtime_splitter = None

        if self.app_key == "mask":
            splitter = QSplitter(Qt.Vertical)
            splitter.setChildrenCollapsible(False)
            self._live_preview, self._live_preview_card = (
                _build_live_preview_card(self))
            # Let the live preview push tuned settings into the main panel
            # when its "Propagate settings" toggle is on.
            self._live_preview.set_propagate_callback(
                self._propagate_live_settings)
            splitter.addWidget(self._live_preview_card)
            splitter.addWidget(console_wrap)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([420, 320])
            layout.addWidget(splitter, 1)
            self._runtime_splitter = splitter
        elif self.app_key == "timelapse":
            # Timelapse takes the same slot Mask and Measure use for Live
            # Preview. Segmenting the sequence is the expensive half and is
            # cached on a signature that deliberately excludes the tracking
            # settings, so re-linking while tuning them costs nothing.
            from ..widgets.timelapse_preview import build_timelapse_preview_card
            splitter = QSplitter(Qt.Vertical)
            splitter.setChildrenCollapsible(False)
            self._timelapse_preview, self._timelapse_preview_card = (
                build_timelapse_preview_card(self))
            self._timelapse_preview.set_propagate_callback(
                self._propagate_live_settings)
            splitter.addWidget(self._timelapse_preview_card)
            splitter.addWidget(console_wrap)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([420, 320])
            layout.addWidget(splitter, 1)
            self._runtime_splitter = splitter
        elif self.app_key == "motility":
            from ..widgets.motility_preview import build_motility_preview_card
            splitter = QSplitter(Qt.Vertical)
            splitter.setChildrenCollapsible(False)
            self._motility_preview, self._motility_preview_card = (
                build_motility_preview_card(self))
            self._motility_preview.set_propagate_callback(
                self._propagate_live_settings)
            splitter.addWidget(self._motility_preview_card)
            splitter.addWidget(console_wrap)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([420, 320])
            layout.addWidget(splitter, 1)
            self._runtime_splitter = splitter
        elif self.app_key == "measure":
            splitter = QSplitter(Qt.Vertical)
            splitter.setChildrenCollapsible(False)
            self._measure_preview, self._measure_preview_card = (
                _build_measure_preview_card(self))
            self._measure_preview.set_propagate_callback(
                self._propagate_live_settings)
            splitter.addWidget(self._measure_preview_card)
            splitter.addWidget(console_wrap)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([420, 320])
            layout.addWidget(splitter, 1)
            self._runtime_splitter = splitter
        elif _hyperparam_searchable(self.app_key):
            # umap / classify / ml_analyze get a Hyperparameter search card in
            # the slot Mask and Measure use for Live Preview: same shape, same
            # threading contract, and its Apply reuses the same route back into
            # the settings panel.
            from .hyperparam import build_hyperparam_card
            splitter = QSplitter(Qt.Vertical)
            splitter.setChildrenCollapsible(False)
            self._hyperparam, self._hyperparam_card = build_hyperparam_card(self)
            self._hyperparam.set_apply_callback(self._propagate_live_settings)
            self._hyperparam.set_settings_provider(
                lambda model=self._settings_model: model.collect())
            splitter.addWidget(self._hyperparam_card)
            splitter.addWidget(console_wrap)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([420, 320])
            layout.addWidget(splitter, 1)
            self._runtime_splitter = splitter
        else:
            layout.addWidget(console_wrap, 1)

        # Route the verbose logger (if the user turned it on in
        # Preferences) at THIS screen's console. Only the last-focused
        # screen receives the log stream — that's fine, users hit the
        # console they're looking at.
        try:
            from ..verbose_logger import register_console_target
            register_console_target(self._console)
        except Exception:
            pass

        # Usage card
        usage_card = Card(title="System")
        self._usage_ram = UsageBar("RAM")
        self._usage_gpu = UsageBar("GPU")
        self._usage_vram = UsageBar("VRAM")
        for w in (self._usage_ram, self._usage_gpu, self._usage_vram):
            usage_card.body_layout.addWidget(w)

        # CPU row: single "CPU" bar + a toggle chevron button.
        cpu_row = QHBoxLayout()
        cpu_row.setContentsMargins(0, 0, 0, 0)
        cpu_row.setSpacing(SPACING["sm"])
        self._usage_cpu = UsageBar("CPU")
        cpu_row.addWidget(self._usage_cpu, 1)
        self._btn_cpu_toggle = QPushButton("Per-core")
        self._btn_cpu_toggle.setCheckable(True)
        self._btn_cpu_toggle.setCursor(Qt.PointingHandCursor)
        self._btn_cpu_toggle.setToolTip("Toggle per-core CPU utilisation bars.")
        self._btn_cpu_toggle.toggled.connect(self._on_toggle_per_core)
        cpu_row.addWidget(self._btn_cpu_toggle)
        cpu_wrap = QWidget()
        # Transparent so the System card surface (not the global black QWidget
        # bg) shows behind the CPU bar + Per-core button.
        cpu_wrap.setStyleSheet("background: transparent;")
        cpu_wrap.setLayout(cpu_row)
        usage_card.body_layout.addWidget(cpu_wrap)

        # Per-core panel — hidden by default; one UsageBar per logical core.
        self._per_core_wrap = QWidget()
        self._per_core_wrap.setStyleSheet("background: transparent;")
        self._per_core_layout = QVBoxLayout(self._per_core_wrap)
        self._per_core_layout.setContentsMargins(0, 0, 0, 0)
        self._per_core_layout.setSpacing(2)
        self._per_core_bars: list[UsageBar] = []
        self._per_core_wrap.hide()
        usage_card.body_layout.addWidget(self._per_core_wrap)

        layout.addWidget(usage_card)

        # Actions row. Flush-left (no extra inset) so Run / Stop / Import /
        # Clear / Explain line up with the console, chat and System panel,
        # which all share the runtime panel's small left inset.
        actions = QWidget()
        # Kept so `_clear_page_surfaces` can tag it. Untagged it inherits the
        # blanket `QWidget { background-color: bg }` rule and paints an opaque
        # strip behind Run / Stop — a black box no opacity setting could reach,
        # because it is the window colour rather than a surface.
        self._actions_row = actions
        row = QHBoxLayout(actions)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["sm"])
        self._btn_run = QPushButton("Run")
        self._btn_run.setObjectName("PrimaryButton")
        self._btn_run.setCursor(Qt.PointingHandCursor)
        self._btn_run.clicked.connect(self._on_run)
        row.addWidget(self._btn_run)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setObjectName("DangerButton")
        self._btn_stop.setCursor(Qt.PointingHandCursor)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        row.addWidget(self._btn_stop)

        self._btn_import = QPushButton("Import settings…")
        self._btn_import.setObjectName("GhostButton")
        self._btn_import.setCursor(Qt.PointingHandCursor)
        self._btn_import.clicked.connect(self._on_import_settings)
        row.addWidget(self._btn_import)

        self._btn_remote = QPushButton("Submit remote…")
        self._btn_remote.setObjectName("PrimaryButton")
        self._btn_remote.setCursor(Qt.PointingHandCursor)
        self._btn_remote.setToolTip(
            "Send the current resolved settings to the Distributed Jobs "
            "screen for an SSH workstation, Slurm cluster, or configured "
            "cloud/HPC command."
        )
        self._btn_remote.clicked.connect(self._on_remote_submit)
        row.addWidget(self._btn_remote)

        self._btn_clear = QPushButton("Clear console")
        self._btn_clear.setObjectName("GhostButton")
        self._btn_clear.setCursor(Qt.PointingHandCursor)
        self._btn_clear.clicked.connect(lambda: self._console.clear())
        row.addWidget(self._btn_clear)

        # Beside Clear, because the two are the same kind of act on the same
        # thing — and because the console is what a bug report is made of.
        self._btn_copy_console = QPushButton("Copy console")
        self._btn_copy_console.setObjectName("GhostButton")
        self._btn_copy_console.setCursor(Qt.PointingHandCursor)
        self._btn_copy_console.setToolTip(
            "Copy everything in the console, section headers included.")
        self._btn_copy_console.clicked.connect(self._on_copy_console)
        row.addWidget(self._btn_copy_console)

        # Preferences, to the right of Copy console. Every module screen
        # gets it because every module screen is somewhere a user notices
        # the font is too small or the backdrop is costing frames -- and
        # the alternative is the menu bar, which is a trip out of the work.
        # Icon-only: the row is already three words wide and a gear is the
        # one glyph nobody has to be taught.
        from .. import iconset as _iconset_prefs

        self._btn_preferences = QPushButton()
        self._btn_preferences.setObjectName("GhostButton")
        self._btn_preferences.setIcon(_iconset_prefs.icon("settings"))
        # SIZE THE GEAR, or it is Qt's 16px default forever. This project
        # ships a 1.5 default font scale, so every label beside it renders
        # half again as large while the icon stays 16px in a 44px button --
        # which is why it was reported as "I cannot see the gear". The icon
        # was never missing; it was rendering at a third of the button.
        #
        # Scaled with the font rather than fixed, so it keeps its
        # proportion at every zoom level.
        from ..preferences import scaled_px
        self._btn_preferences.setIconSize(
            QSize(scaled_px(18), scaled_px(18)))
        self._btn_preferences.setCursor(Qt.PointingHandCursor)
        self._btn_preferences.setToolTip("Open Preferences (Ctrl+,).")
        self._btn_preferences.setAccessibleName("Preferences")
        self._btn_preferences.clicked.connect(self._open_preferences_dialog)
        row.addWidget(self._btn_preferences)

        # (The manual "Explain error" button was removed — errors now route to
        # the AI automatically when AI is enabled; see _on_pipeline_error.)
        from .. import iconset as _iconset

        # File as GitHub issue — same enable gate as Explain, plus the
        # user's opt-in in AI Settings. Opens a pre-filled issue URL
        # in the default browser; the user reviews and hits Submit.
        self._btn_file_issue = QPushButton("File as issue")
        self._btn_file_issue.setObjectName("GhostButton")
        self._btn_file_issue.setIcon(_iconset.icon("info"))
        self._btn_file_issue.setCursor(Qt.PointingHandCursor)
        self._btn_file_issue.setToolTip(
            "Open a pre-filled GitHub issue with the last traceback + "
            "environment. You review before submitting. Toggle on/off "
            "in AI Settings → Report errors as GitHub issues."
        )
        self._btn_file_issue.setEnabled(False)
        self._btn_file_issue.setVisible(False)
        self._btn_file_issue.clicked.connect(self._on_file_issue)
        row.addWidget(self._btn_file_issue)

        row.addStretch(1)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate until we know
        self._progress.setVisible(False)
        self._progress.setFixedWidth(240)
        row.addWidget(self._progress)

        # Runtime-preview toggle — every app with a preview gets the same
        # bottom-right control Mask established for Live Preview. Keeping this
        # in the shared actions row prevents Timelapse, Motility and Measure
        # from permanently taking half the console merely because their
        # preview card exists.
        from PySide6.QtWidgets import QMenu, QToolButton
        from ..widgets import AiToggleLabel
        preview_controls = {
            "mask": (
                "_live_preview_card",
                "Click to toggle Live Preview. When ON (blue), the "
                "interactive Cellpose preview appears above the console."),
            "timelapse": (
                "_timelapse_preview_card",
                "Click to toggle Track Preview for the timelapse."),
            "motility": (
                "_motility_preview_card",
                "Click to toggle Track Preview for the motility analysis."),
            "measure": (
                "_measure_preview_card",
                "Click to toggle Measurement Preview."),
        }
        preview_control = preview_controls.get(self.app_key)
        if preview_control is not None:
            card_attr, tooltip = preview_control
            if getattr(self, card_attr, None) is not None:
                self._preview_card_attr = card_attr
                self._preview_switch = AiToggleLabel(
                    text="Live", tooltip=tooltip)
                self._preview_switch.toggled.connect(
                    self._on_preview_switch)
                row.addWidget(self._preview_switch)
                # Preserve the public name used by existing Mask integrations.
                if self.app_key == "mask":
                    self._lp_switch = self._preview_switch
                self._on_preview_switch(False)

        # Image UMAP has one GPU switch for both its main run and its search.
        # It deliberately lives in the action strip instead of being repeated
        # in the settings form, and precedes Hyperparameter search as requested.
        self._gpu_switch = None
        if self.app_key == "umap" and getattr(
                self, "_hyperparam", None) is not None:
            self._gpu_switch = AiToggleLabel(
                text="GPU",
                tooltip=(
                    "Use the GPU for both the main dimensionality-reduction "
                    "run and Hyperparameter search. When ON (blue), spaCR "
                    "requires a working RAPIDS cuML backend. CPU and GPU "
                    "reducers can produce a DIFFERENT MAP, so compare rows "
                    "only within one backend."),
            )
            self._gpu_switch.toggled.connect(self._on_umap_gpu_switch)
            row.addWidget(self._gpu_switch)

        # Same slot, same behaviour, for the apps that have a hyperparameter
        # search instead of a live preview.
        if getattr(self, "_hyperparam", None) is not None:
            from .hyperparam import TOGGLE_TEXT, TOGGLE_TOOLTIP
            self._hp_switch = AiToggleLabel(text=TOGGLE_TEXT,
                                            tooltip=TOGGLE_TOOLTIP)
            self._hp_switch.toggled.connect(self._on_hyperparam_switch)
            row.addWidget(self._hp_switch)
            self._on_hyperparam_switch(False)   # start collapsed, like Live

        # Interactive image-UMAP explorer — UMAP only, immediately beside AI.
        # It starts off so ordinary runs retain the familiar static figure.
        # Turning it on before or after a run switches the same payload to the
        # click / image-preview / lasso / database-annotation interface.
        #
        # It says "Interactive", not "Live". A LIVE view re-renders a module's
        # own output from the current settings before a run — Mask, Timelapse,
        # Measure and Motility, all four of which now share one contract
        # (spacr.qt.widgets.preview_contract). This explorer is not one of
        # those: it makes an already-computed embedding clickable, and no
        # setting changes what it draws. One word for one thing.
        self._interactive_switch = None
        if self.app_key == "umap" and self._umap_explorer is not None:
            self._interactive_switch = AiToggleLabel(
                text="Interactive",
                tooltip=(
                    "Toggle the interactive image UMAP. When ON (blue), "
                    "click a point to preview its image, draw around a "
                    "cluster, and write manual or automatic labels to the "
                    "database."
                ),
            )
            self._interactive_switch.toggled.connect(
                self._on_interactive_switch)
            # Clicking the STATIC figure turns Live on. The request was "i
            # should be able to press every point" -- pressing a point on a
            # rendered PNG means hit-testing pixels back to the embedding,
            # a second and fragile implementation of what the explorer
            # already does properly. So the click takes you to the view
            # where pressing points works, instead of building that twice.
            queue = getattr(self, "_figure_queue", None)
            if queue is not None and hasattr(queue, "figure_clicked"):
                queue.figure_clicked.connect(self._on_static_figure_clicked)
            row.addWidget(self._interactive_switch)

        # AI toggle + provider dropdown, bottom-right of the actions row.
        # AI switch is a plain clickable text label — white when off,
        # accent blue when on. Chevron next to it exposes the provider
        # picker + install/login dialog.
        self._ai_switch = AiToggleLabel()
        self._ai_switch.toggled.connect(self._on_ai_switch)
        row.addWidget(self._ai_switch)

        self._ai_menu_btn = QToolButton()
        self._ai_menu_btn.setPopupMode(QToolButton.InstantPopup)
        self._ai_menu_btn.setCursor(Qt.PointingHandCursor)
        self._ai_menu_btn.setToolTip("Pick provider · Providers…")
        self._ai_menu_btn.setText("▾")
        self._ai_menu = QMenu(self._ai_menu_btn)
        self._ai_menu_btn.setMenu(self._ai_menu)
        row.addWidget(self._ai_menu_btn)
        self._refresh_ai_menu()

        layout.addWidget(actions)

        # Category strip — the settings CATEGORY blurb, immediately under the
        # Run / Stop row. A category groups tens of settings (Organelle
        # Segmentation groups fifty-three), so its description is a paragraph,
        # and a paragraph-sized popup hovering over the settings panel covers
        # the very controls it is describing. It gets a fixed region here
        # instead: hovering a category header fills it, expanding one pins it,
        # and it holds the pinned category while the pointer wanders back into
        # the form. The per-setting strip below shows the setting under the
        # cursor, so the two read as "where you are" then "what this does".
        self._category_hint_pinned = ""
        self._category_hint = QLabel(self._default_category_hint())
        self._category_hint.setObjectName("CategoryHintStrip")
        # Named widgets keep their fill under the blanket
        # `QWidget { background-color: bg }` rule, and this one is a caption
        # over the backdrop, not a surface — the same reason `cpu_wrap` above
        # carries the declaration.
        self._category_hint.setStyleSheet("background: transparent;")
        self._category_hint.setWordWrap(True)
        self._category_hint.setTextFormat(Qt.RichText)
        self._category_hint.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._sync_category_hint_height()
        layout.addWidget(self._category_hint)

        # Hint strip — hover-follows caption that shows the current
        # settings tooltip regardless of Qt HTML-tooltip rendering.
        self._hint_strip = QLabel(self._default_hint())
        self._hint_strip.setObjectName("SubtitleSmall")
        self._hint_strip.setWordWrap(True)
        self._sync_hint_strip_height()
        self._hint_strip.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._hint_strip.setOpenExternalLinks(True)
        layout.addWidget(self._hint_strip)

        return wrap

    def _sync_hint_strip_height(self) -> None:
        """Reserve four lines using the font Qt is actually painting."""
        hint = getattr(self, "_hint_strip", None)
        if hint is None:
            return
        hint.ensurePolished()
        hint.setFixedHeight(
            hint.fontMetrics().lineSpacing() * HINT_STRIP_LINES)

    # ------------------------------------------------------------------
    # Category help — the strip under the actions row
    # ------------------------------------------------------------------
    def _sync_category_hint_height(self) -> None:
        """Reserve three lines for the category strip, in the painted font."""
        strip = getattr(self, "_category_hint", None)
        if strip is None:
            return
        strip.ensurePolished()
        strip.setFixedHeight(
            strip.fontMetrics().lineSpacing() * CATEGORY_STRIP_LINES)

    def _default_category_hint(self) -> str:
        return ("Hover a settings category for what the group decides, "
                "or open one to keep it here.")

    def _wire_category_hints(self) -> None:
        """Route every category header at the strip under the actions row.

        Called once both panels exist — the settings panel builds the
        sections, the runtime panel owns the strip they write into.

        Idempotent on purpose, and by the same two mechanisms the per-setting
        decoration learned the hard way: a marker property so a second pass
        does not connect ``toggled`` twice, and ``removeEventFilter`` before
        ``installEventFilter``, because Qt keeps a LIST of filters and calls
        each installation separately — two installs on one header means one
        hover writing the strip twice.
        """
        for section in getattr(self, "_settings_sections", []):
            header = section.header()
            if header is None or header.property("categoryHintWired"):
                continue
            title = section.title()
            header.setProperty("settingsCategory", title)
            header.setProperty("categoryHintWired", True)
            header.removeEventFilter(self)
            header.installEventFilter(self)
            section.toggled.connect(
                partial(self._on_category_toggled, title))

    def _on_category_toggled(self, title: str, expanded: bool) -> None:
        """Pin an expanded category's blurb; unpin it when it collapses."""
        if expanded:
            self._category_hint_pinned = str(title)
            self.show_category_hint(title)
        elif self._category_hint_pinned == str(title):
            self._category_hint_pinned = ""
            self.clear_category_hint()

    def show_category_hint(self, title: str) -> None:
        """Show one category's blurb in the strip under the actions row."""
        strip = getattr(self, "_category_hint", None)
        if strip is None:
            return
        text = category_tooltip(self.app_key, title)
        heading = str(title or "").upper().strip()
        strip.setText(
            f"<b>{escape(heading)}</b> — {escape(text)}"
            if heading else escape(text)
        )
        strip.setAccessibleDescription(f"{heading}. {text}".strip())

    def clear_category_hint(self) -> None:
        """Fall back to the pinned (expanded) category, or to the prompt."""
        strip = getattr(self, "_category_hint", None)
        if strip is None:
            return
        pinned = getattr(self, "_category_hint_pinned", "")
        if pinned:
            self.show_category_hint(pinned)
            return
        strip.setText(self._default_category_hint())
        strip.setAccessibleDescription(self._default_category_hint())

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Re-measure the hover-help strip after stylesheet/font polishing.

        Also a second, independent chance to pick up an ambient-
        background preference that changed while this screen sat in the
        background. The first is :meth:`changeEvent`, which fires on
        every Preferences save; this one covers a preference written
        without ``apply_preferences_to_app`` behind it. Module screens
        are built once and kept, so without either of them a toggle
        would need a restart. It costs a settings read and returns
        without touching anything when nothing changed — see
        :meth:`refresh_ambient_background`.
        """
        super().showEvent(event)
        usage_timer = getattr(self, "_usage_timer", None)
        if usage_timer is not None and not usage_timer.isActive():
            usage_timer.start()
            self._refresh_usage()
        self._sync_hint_strip_height()
        self._sync_category_hint_height()
        self.refresh_ambient_background()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt override
        # Keep this Qt lifecycle hook out of the documented spaCR API: it is
        # only the inverse of the showEvent timer activation above.
        usage_timer = getattr(self, "_usage_timer", None)
        if usage_timer is not None:
            usage_timer.stop()
        super().hideEvent(event)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_run(self):
        from ..verbose_logger import log_button_press
        entry = resolve_pipeline_entry(self.app_key)
        if entry is None:
            log_button_press(
                f"{self.app_key}.Run",
                {"result": "not_runnable"})
            QMessageBox.information(
                self, "Not runnable",
                f"The '{self.app_key}' app is interactive-only in this Qt build. "
                f"Use the classic Tk GUI (`spacr`) for now.",
            )
            return
        try:
            settings = self._settings_model.collect()
        except Exception as e:
            log_button_press(
                f"{self.app_key}.Run",
                {"result": "bad_settings", "error": str(e)})
            QMessageBox.warning(self, "Bad settings", str(e))
            return
        if self.app_key == "umap":
            # Resolve GUI colours on the GUI thread and pass plain strings to
            # the worker. The UMAP canvas sits inside a Card, whose material is
            # ``surface_alt`` in every theme; matching that color avoids a
            # black/white rectangle inside dark, light, image, and glass
            # themes. Avoid reading QApplication/QSettings from the worker.
            from ..theme import active_palette
            palette = active_palette()
            settings["_plot_theme"] = {
                "background": palette["surface_alt"],
                "foreground": palette["fg"],
                "border": palette["fg"],
            }

        # Diagnostic breadcrumb — visible when the user has verbose
        # logging on. Shows exactly which app + entry-point ran and
        # (truncated) which settings were passed. Helps triage
        # "Starting mask… (hangs)" reports.
        log_button_press(
            f"{self.app_key}.Run",
            {
                "entry":    getattr(entry, "__qualname__", repr(entry)),
                "src":      settings.get("src"),
                "n_keys":   len(settings),
            },
        )
        # Also always print a compact one-liner into the Console so
        # non-verbose users see the entry point name — this is what
        # they were missing when the console just said "Starting mask…"
        # and nothing else.
        entry_name = getattr(entry, "__qualname__", repr(entry))
        # Tell the console which module/function this output is from so its
        # "spaCR output — <module> — <function>" banner is accurate.
        try:
            self._console.set_run_context(self.app_key, entry_name)
        except Exception:
            pass
        self._console.append_notice(
            "→ Starting {module} ({function}) with src={src} + "
            "{count} settings…\n",
            module=self.app_key,
            function=entry_name,
            src=repr(settings.get("src")),
            count=len(settings),
        )
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress.setVisible(True)

        # Remember start time so _on_finished can report elapsed to
        # the run journal + the OS notification.
        import time as _time
        self._run_started_at = _time.time()

        # The one preference the PIPELINE needs to know about, passed as an
        # ordinary setting. The pipeline must never read QSettings -- a
        # `from PySide6 import` in a pipeline module makes the package
        # unimportable on a cluster -- so the GUI reads it here and a
        # headless caller sets the same key itself.
        try:
            from ..preferences import get_hash_inputs
            settings.setdefault("hash_inputs", get_hash_inputs())
        except Exception:
            LOG.debug("could not read the hashing preference", exc_info=True)

        self._thread, worker = make_thread(entry, settings)
        # Keep a strong reference to the worker on ``self``. PySide6
        # does NOT keep a QObject alive through a bound-method signal
        # connection (thread.started → worker.run), so a local-only
        # ``worker`` can be garbage-collected before run() fires — the
        # thread then spins its event loop forever and the pipeline
        # never starts. Storing it here fixes an intermittent
        # "pressed Run, nothing happens" hang.
        self._worker = worker
        worker.line_ready.connect(self._console.append_stdout)
        worker.error.connect(self._on_pipeline_error)
        worker.figure_ready.connect(self._on_figure_ready)
        worker.finished.connect(self._on_finished)
        # Clear our Python references only once the QThread has genuinely
        # stopped (its event loop exited). Dropping them from _on_finished —
        # which runs on worker.finished, before thread.quit() has taken
        # effect — could destroy the QThread while it is still "running"
        # ("QThread: Destroyed while thread is still running" → abort).
        self._thread.finished.connect(self._clear_thread_refs)
        self._thread.start()

    def _open_preferences_dialog(self) -> None:
        """Open Preferences from this screen.

        Routed through the MainWindow when there is one, so the dialog is
        the same object the menu opens and a preference changed here reaches
        the same live-apply path. Falls back to constructing one directly so
        a screen built on its own -- which is how every test builds one --
        still works rather than raising.
        """
        window = self.window()
        opener = getattr(window, "_open_preferences", None)
        if callable(opener):
            opener()
            return
        try:
            from ..preferences import PreferencesDialog

            PreferencesDialog(self).exec()
        except Exception:
            LOG.exception("could not open Preferences from %s", self.app_key)

    def _on_copy_console(self) -> None:
        """Copy the whole console, and say how much went to the clipboard.

        A clipboard write is silent, so a button that appears to do nothing
        is indistinguishable from one that failed. The status line says what
        happened.
        """
        try:
            text = self._console.copy_all()
        except Exception as exc:
            self._console.append_error(f"Could not copy the console: {exc}\n")
            return
        lines = text.count("\n")
        self._btn_copy_console.setText("Copied")
        QTimer.singleShot(
            1200, lambda: self._btn_copy_console.setText("Copy console"))
        try:
            self.statusBar().showMessage(f"Copied {lines} lines", 3000)
        except Exception:
            pass

    def _on_remote_submit(self) -> None:
        """Validate current settings and hand a snapshot to MainWindow."""
        try:
            settings = dict(self._settings_model.collect())
        except Exception as exc:
            QMessageBox.warning(self, "Bad settings", str(exc))
            return
        self.remote_submit_requested.emit(self.app_key, settings)

    def _on_pipeline_error(self, tb: str):
        """Capture the traceback and either show it raw or route it through AI."""
        self._last_error_text = tb

        # Route through AI when AI is enabled with a provider AND the
        # route-errors-through-AI preference is on (the default). The user then
        # sees the AI's explanation + instructions; the raw traceback stays
        # hidden (the AI still has it, so the user can ask it to show the error).
        routed = False
        try:
            from ..ai import settings as _ai_settings
            if (self._console._ai_active
                    and self._console._current_provider() is not None
                    and _ai_settings.get_route_errors_through_ai()):
                self._console.open_error_flow(
                    tb, active_app=self.app_key, show_raw=False)
                routed = True
        except Exception:
            routed = False
        if not routed:
            self._console.append_error(tb)
        # File-as-issue button becomes visible only when the user has
        # opted in via AI Settings — otherwise it stays hidden so the
        # actions row doesn't grow noise for people who don't use it.
        try:
            from ..ai import settings as _ai_settings
            enabled = _ai_settings.get_auto_file_issues()
        except Exception:
            enabled = False
        self._btn_file_issue.setVisible(enabled)
        self._btn_file_issue.setEnabled(enabled)
        # Opting in reveals the action; it never submits in response to the
        # crash itself. Every report stops at an editable public-payload
        # preview and needs a report-specific Send click.

    # ------------------------------------------------------------------
    # AI toggle + provider menu — sits in the actions row (bottom right)
    # ------------------------------------------------------------------
    def _on_lp_switch(self, on: bool) -> None:
        """Compatibility route for callers that still name Mask's LP switch."""
        card = getattr(self, "_live_preview_card", None)
        if card is None:
            return
        card.setVisible(on)

    def _on_preview_switch(self, on: bool) -> None:
        """Show or hide this module's runtime preview card.

        Opening it also seeds the panel from the form, once. Before that,
        this screen wired only the push direction — ``set_propagate_callback``
        — so all four previews ran at their own hardcoded defaults. A user
        set ``cell_FT``, opened Live preview to check the segmentation, and
        the preview segmented at 0.4 regardless: the preview is consulted to
        make a decision, which is the worst place for it to disagree with
        the run.

        On FIRST show rather than at construction, mirroring what
        ``_PreviewHost.prime`` documents for the previews attached through
        :mod:`spacr.qt.preview_registry` — ``collect()`` is a pass over every
        widget on the screen, and a preview nobody opens should cost nothing.
        Once, not on every open, or re-opening the card would silently
        discard whatever the user had just tuned inside it.
        """
        attr = getattr(self, "_preview_card_attr", "")
        card = getattr(self, attr, None) if attr else None
        if card is None:
            return
        if on and not getattr(self, "_preview_primed", False):
            self._preview_primed = True
            self._prime_preview()
        card.setVisible(on)

    def _prime_preview(self) -> None:
        """Push the current settings into this screen's preview panel.

        Never raises: a preview that cannot be seeded is still worth showing,
        and the alternative is a module whose Live switch takes the window
        down.
        """
        attr = getattr(self, "_preview_card_attr", "")
        panel = getattr(self, attr[:-len("_card")], None) if attr else None
        model = getattr(self, "_settings_model", None)
        apply_settings = getattr(panel, "apply_settings", None)
        if model is None or not callable(apply_settings):
            return
        try:
            apply_settings(model.collect())
        except Exception:
            LOG.debug("could not seed the %s preview", self.app_key,
                      exc_info=True)

    def _on_hyperparam_switch(self, on: bool) -> None:
        """Show/hide the Hyperparameter search card when its toggle flips."""
        card = getattr(self, "_hyperparam_card", None)
        if card is None:
            return
        card.setVisible(on)
        if on:
            # Seed the search space from whatever is currently in the panel, so
            # the sweep starts from the user's settings rather than defaults.
            model = getattr(self, "_settings_model", None)
            if model is not None:
                self._hyperparam.apply_settings(model.collect())

    def _on_umap_gpu_switch(self, on: bool) -> None:
        """Keep one truthful GPU state across the main and search pipelines."""
        panel = getattr(self, "_hyperparam", None)
        model = getattr(self, "_settings_model", None)
        if panel is None or self.app_key != "umap":
            return
        enabled = bool(panel.request_gpu_enabled(bool(on)))
        if model is not None:
            model.set_hidden_value("gpu", enabled)
        switch = getattr(self, "_gpu_switch", None)
        if switch is not None and switch.isChecked() != enabled:
            switch.blockSignals(True)
            switch.setChecked(enabled)
            switch.blockSignals(False)


    def _on_static_figure_clicked(self) -> None:
        """A click on the static UMAP opens the interactive explorer.

        Only when there is a payload to explore -- clicking an ordinary
        figure, or one from a run that carried no embedding, does nothing
        rather than flipping a switch that then shows an empty panel.

        Says so in the console, because a view that changes under you with
        no explanation is worse than one that does not change.
        """
        if not getattr(self, "_umap_payload_ready", False):
            return
        switch = getattr(self, "_interactive_switch", None)
        if switch is None or switch.isChecked():
            return
        switch.setChecked(True)
        try:
            self._console.append_notice(
                "\nInteractive mode is on — click any point to preview its "
                "image. Turn it off with the Interactive toggle.\n")
        except Exception:
            LOG.debug("could not announce interactive mode", exc_info=True)

    def _on_interactive_switch(self, on: bool) -> None:
        """Switch UMAP results between the static figure and explorer.

        The toggle can be enabled before a run.  In that case the current
        console/figure layout stays put until a UMAP payload arrives, then
        :meth:`_on_figure_ready` opens the explorer automatically.
        """
        explorer = getattr(self, "_umap_explorer", None)
        queue = getattr(self, "_figure_queue", None)
        if explorer is None or queue is None:
            return
        if on and getattr(self, "_umap_payload_ready", False):
            queue.hide()
            explorer.show()
            self._figures_card.show()
            return
        explorer.hide()
        if queue.count():
            queue.show()

    def _on_ai_switch(self, on: bool) -> None:
        self._console.set_ai_active(on)
        if on:
            # Auto-pick first available provider if none selected yet.
            from .. import ai as ai_module
            if not self._console._current_provider_name:
                configured = ai_module.configured_providers()
                if configured:
                    self._console.set_ai_provider(configured[0].name)
                    self._refresh_ai_menu()
                else:
                    self._console.append_notice(
                        "[AI] No vendor CLI installed. Click ▾ next "
                        "to the AI switch → Providers…\n"
                    )
                    self._ai_switch.setChecked(False)

    def _refresh_ai_menu(self) -> None:
        """Rebuild the provider dropdown next to the AI switch."""
        from .. import ai as ai_module
        self._ai_menu.clear()
        configured = ai_module.configured_providers()
        current = self._console._current_provider_name
        if configured:
            for p in configured:
                act = self._ai_menu.addAction(p.label)
                act.setCheckable(True)
                act.setChecked(p.name == current)
                act.triggered.connect(
                    lambda _c=False, name=p.name: self._on_pick_provider(name)
                )
            self._ai_menu.addSeparator()
        else:
            source = "(no vendor CLI installed)"
            unavailable = self._ai_menu.addAction(tr(source))
            unavailable.setProperty("_spacr_i18n_text", source)
            unavailable.setEnabled(False)
            self._ai_menu.addSeparator()
        source = "Providers…"
        act_providers = self._ai_menu.addAction(tr(source))
        act_providers.setProperty("_spacr_i18n_text", source)
        act_providers.triggered.connect(self._on_open_providers_dialog)

    def _on_pick_provider(self, name: str) -> None:
        self._console.set_ai_provider(name)
        self._refresh_ai_menu()

    def _on_open_providers_dialog(self) -> None:
        from ..widgets.ai_chat_panel import _ProvidersDialog
        from PySide6.QtWidgets import QDialog
        dlg = _ProvidersDialog(self)
        if dlg.exec() == QDialog.Accepted:
            self._refresh_ai_menu()

    def _on_explain_error(self):
        if not self._last_error_text:
            return
        # Route the traceback into our own merged console — no more
        # side-panel navigation. Keep the legacy signal too, for
        # MainWindow's old dock path.
        self._console.open_error_flow(self._last_error_text, self.app_key)
        self.error_explain_requested.emit(self._last_error_text, self.app_key)

    def _on_file_issue(self) -> None:
        """Open a pre-filled GitHub issue for the last captured traceback.

        The reporting itself runs on a worker thread, and the reason is a
        number: :func:`spacr.qt.ai.issue_report.file_issue` resolves a GitHub
        token -- which falls through to ``subprocess.run(["gh", "auth",
        "token"], timeout=8)`` -- and then POSTs to ``api.github.com`` with
        ``urlopen(timeout=20)``. Run inline, as this was, the worst case is
        **28 seconds of a frozen window** with no cursor, no repaint and no
        way to cancel, in response to a single click. Measured with the
        event-loop watchdog at 2420 ms against 1.2 s stand-ins for both
        halves; the timeouts above are what it becomes on a bad network.

        Only the settings snapshot stays here, because reading a widget's
        value is the one part that *must* happen on the GUI thread. The
        console line is written when the worker returns.
        """
        if not self._last_error_text:
            return

        # The preview itself is the prompt and the consent boundary. The
        # legacy mode remains respected so Preferences can revoke reporting.
        from ..preferences import ISSUE_PROMPT_NEVER, get_issue_prompt_mode
        mode = get_issue_prompt_mode()
        if mode == ISSUE_PROMPT_NEVER:
            self._console.append_notice(
                "\nNot filing a report: issue reporting is set to 'never' in "
                "Preferences.\n")
            return
        # Best-effort settings snapshot from the current settings model
        # so the issue includes what the user was trying to run.
        settings_snapshot: dict = {}
        try:
            model = getattr(self, "_settings_model", None)
            if model is not None:
                for k, w in getattr(model, "_widgets", {}).items():
                    from PySide6.QtWidgets import (
                        QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit,
                        QSpinBox,
                    )
                    if isinstance(w, QCheckBox):
                        settings_snapshot[k] = w.isChecked()
                    elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                        settings_snapshot[k] = w.value()
                    elif isinstance(w, QComboBox):
                        settings_snapshot[k] = w.currentText()
                    elif hasattr(w, "get_value"):
                        # The chip editor is a QWidget, not a QLineEdit; a
                        # bug report that omitted every list setting was how
                        # the class_metadata crash arrived without its own
                        # value attached.
                        settings_snapshot[k] = w.get_value()
                    elif isinstance(w, QLineEdit):
                        settings_snapshot[k] = w.text()
        except Exception:
            settings_snapshot = {}
        from PySide6.QtWidgets import QDialog
        from ..ai.issue_preview import IssuePreviewDialog
        from ..ai.issue_report import build_report, submit_report
        from ..preferences import get_share_diagnostic_logs

        report = build_report(
            self._last_error_text,
            active_app=self.app_key,
            settings=settings_snapshot,
            include_log_tail=get_share_diagnostic_logs(),
        )
        preview = IssuePreviewDialog(report, self)
        if preview.exec() != QDialog.Accepted:
            self._console.append_notice(
                "[issue] cancelled — nothing was sent.\n")
            return
        approved_report = preview.approved_report()

        def _file():
            # The failure is carried back as data rather than raised. The
            # auto-file path used to wrap this call in `try/except` to print
            # "[issue] auto-file failed"; once the call is asynchronous that
            # `except` can no longer see it, and a report that silently fails
            # to send is worse than one that fails loudly.
            try:
                return {"url": submit_report(approved_report)}
            except Exception as exc:      # noqa: BLE001 - reported, not hidden
                return {"error": exc}

        self._console.append_notice(
            "[issue] sending the approved report to GitHub…\n")
        self._jobs.submit(_file, self._on_issue_filed)

    def _on_issue_filed(self, outcome: dict) -> None:
        """Say where the report went, or why it did not. GUI thread only."""
        error = (outcome or {}).get("error")
        if error is not None:
            self._console.append_notice(
                "[issue] auto-file failed: {detail}\n", detail=error)
            return
        self._console.append_notice(
            "[issue] report handoff completed.\n{url}...\n",
            url=str((outcome or {}).get("url") or "")[:100],
        )

    def _umap_display_defaults(self) -> dict:
        """The display settings the run is currently configured with.

        Read from the settings model rather than from the explorer: these
        are the ones the explorer cannot apply live, so it does not hold
        them, and a dialog that opens showing 0 for a setting the user set
        to 20 is worse than one that does not offer it at all.
        """
        model = getattr(self, "_settings_model", None)
        if model is None:
            return {}
        try:
            current = model.collect() or {}
        except Exception:
            LOG.debug("could not read the settings for the UMAP display "
                      "window", exc_info=True)
            return {}
        return {key: current[key]
                for key in ("figuresize", "image_nr", "img_zoom")
                if key in current and current[key] is not None}

    def _propagate_live_settings(self, settings: dict) -> None:
        """Write live-preview-tuned values into the main settings panel."""
        model = getattr(self, "_settings_model", None)
        if model is None:
            return
        for key, value in settings.items():
            model.set_value_for_key(key, value)

    def _on_figure_ready(self, fig, png_path: str = "") -> None:
        """Hand a matplotlib figure to the FigureQueue. ``png_path`` is a PNG
        the pipeline bridge already rendered in its worker thread, so the queue
        can adopt it (cheap) instead of re-rendering on the GUI thread — that's
        what keeps the UI responsive while many figures stream in."""
        payload = getattr(fig, "_spacr_umap_payload", None)
        explorer = getattr(self, "_umap_explorer", None)
        if payload is not None and explorer is not None:
            explorer.set_payload(payload)
            self._umap_payload_ready = True
            # Keep the ordinary figure too: switching Interactive off should
            # restore it immediately rather than requiring another UMAP run.
            self._figure_queue.add_figure(
                fig, prerendered_png=png_path or None)
            switch = getattr(self, "_interactive_switch", None)
            if switch is not None and switch.isChecked():
                self._figure_queue.hide()
                explorer.show()
            else:
                explorer.hide()
                self._figure_queue.show()
            self._figures_card.show()
            return
        self._figure_queue.add_figure(fig, prerendered_png=png_path or None)
        switch = getattr(self, "_interactive_switch", None)
        interactive_open = (
            explorer is not None
            and getattr(self, "_umap_payload_ready", False)
            and switch is not None
            and switch.isChecked()
        )
        if not interactive_open:
            self._figure_queue.show()
        self._figures_card.show()

    def closeEvent(self, event):
        """Cancel and join this screen's worker before destroying widgets.

        A worker that has not reached a safe boundary keeps the screen alive;
        dropping its references or force-terminating it could corrupt an
        output and triggers Qt's fatal "QThread destroyed while running".
        """
        th = getattr(self, "_thread", None)
        if th is not None:
            try:
                worker = getattr(self, "_worker", None)
                if worker is not None:
                    worker.request_cancel("screen closed")
                th.requestInterruption()
                th.wait(3000)
            except Exception:
                pass
            try:
                still_running = bool(th.isRunning())
            except (AttributeError, RuntimeError):
                still_running = False
            if still_running:
                self._console.append_notice(
                    "\nClose deferred: the current field is still finishing. "
                    "The window will remain open so its worker is not "
                    "destroyed mid-write; close it again after Stop completes.\n"
                )
                event.ignore()
                return
            self._thread = None
            self._worker = None
        # Stop polling before shutting the runner down, or the 2 s timer can
        # start one more job while `shutdown` is draining the last.
        try:
            self._usage_timer.stop()
        except (AttributeError, RuntimeError):
            pass
        # The usage poll and the issue report are abandoned rather than waited
        # for: neither writes anything a half-finished copy of would damage,
        # and `shutdown` parks any that outlast its budget instead of
        # terminating them mid-call.
        for name in ("_usage_jobs", "_jobs"):
            jobs = getattr(self, name, None)
            if jobs is not None:
                try:
                    jobs.shutdown()
                except RuntimeError:
                    pass
        # The settings panel's own background work goes with the screen. The
        # exclusion editor reads distinct values off a worker, and it is a
        # child widget, so navigation destroying the panel never gives it a
        # close event of its own to shut that down from.
        self._shutdown_settings_widgets()
        # Clean up the figure queue's temp dir if present.
        fq = getattr(self, "_figure_queue", None)
        if fq is not None:
            try:
                fq.clear()
            except Exception:
                pass
        explorer = getattr(self, "_umap_explorer", None)
        if explorer is not None:
            try:
                explorer.close()
            except Exception:
                pass
        super().closeEvent(event)

    def _shutdown_settings_widgets(self) -> None:
        """Stop any background work a settings widget owns.

        Only the exclusion editor has any today -- it reads a column's
        distinct values off a worker thread -- but the rule is stated by
        capability rather than by class name, so a settings widget that
        acquires a worker later is covered without this having to be
        remembered.
        """
        model = getattr(self, "_settings_model", None)
        widgets = getattr(model, "_widgets", None) if model is not None else None
        try:
            values = list(widgets.values()) if widgets else []
        except Exception:
            return
        for widget in values:
            shutdown = getattr(widget, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except (RuntimeError, TypeError):
                    pass

    def _on_finished(self, ok: bool):
        from ..button_roles import set_button_busy
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        set_button_busy(self._btn_run, False)
        set_button_busy(self._btn_stop, False)
        self._progress.setVisible(False)
        cancelled = bool(
            getattr(getattr(self, "_worker", None), "was_cancelled", False))
        if cancelled:
            self._console.append_notice(
                "■ Stopped safely at a field, trial, or job boundary\n")
        else:
            self._console.append_notice(
                "✓ Finished\n" if ok else
                "✗ Failed — see traceback above\n")
        # NOTE: do NOT drop self._thread / self._worker here. This slot runs
        # on worker.finished, i.e. before thread.quit() has actually stopped
        # the QThread's event loop; releasing the last reference now can
        # destroy the still-running QThread and abort the process. The
        # references are cleared from _clear_thread_refs, wired to the
        # QThread's own finished signal.
        # OS-level notification (libnotify / osascript / win10toast) so
        # users don't have to sit and watch. Always safe — the notify
        # module fails silently on any error.
        try:
            import time as _time
            elapsed = _time.time() - getattr(self, "_run_started_at",
                                                _time.time())
            from ..notify import announce_pipeline_finished
            announce_pipeline_finished(
                self.app_key,
                "cancelled" if cancelled else ("success" if ok else "failed"),
                elapsed,
            )
        except Exception:
            pass

    def _clear_thread_refs(self):
        """Release worker/thread references once the QThread has stopped.

        Wired to QThread.finished (not worker.finished), so by the time this
        runs the thread's event loop has exited and dropping the last Python
        reference cannot abort the process.
        """
        self._thread = None
        self._worker = None

    def _on_stop(self):
        """Stop the run, asking first whether to wait or to kill.

        It used to request a cooperative cancel, disable itself, and hope.
        That is why Stop "didn't seem to do much": cooperative cancellation
        cannot stop a worker wedged in a C extension (INVARIANTS 11) --
        cellpose, torch and cv2 calls never check the flag -- so the run kept
        going, and the button had disabled itself so there was no way to ask
        again. The user waited, believing the run was ending, while it was
        not.

        Now it offers the same choice the Home banner's quit button offers,
        through the same `ask_how_to_quit`, and if the cooperative attempt
        does not land a `GracefulQuitWatcher` comes back and asks again.

        Cooperative stays the DEFAULT and force is never what a stray Return
        does: a pipeline killed mid-write leaves a half-written .npy, and
        silent corruption found later is worse than waiting.
        """
        if self._thread is None:
            return
        from ..button_roles import set_button_busy
        from ..shutdown import (CANCEL, FORCE, GracefulQuitWatcher,
                                ask_how_to_quit)

        name = APP_TITLES.get(self.app_key, self.app_key)
        choice = ask_how_to_quit(
            self, what=name, verb="Stop",
            detail="This run is still working. Stopping cooperatively lets "
                   "it finish the field, trial or job it is on and stop at "
                   "the next point it can do so safely.")
        if choice == CANCEL:
            return

        if choice == FORCE:
            self._force_stop()
            return

        self._console.append_notice(
            "\nRequesting stop. The current field/trial/job will finish, then "
            "the resumable run will stop at its next safe boundary.\n")
        set_button_busy(self._btn_stop, True)
        # NOT disabled. A cooperative stop that never lands used to leave the
        # user with no way to escalate; the button stays live so pressing it
        # again reaches the same prompt, and the watcher asks unprompted.
        self._request_cooperative_stop()

        self._stop_watcher = GracefulQuitWatcher(
            self,
            lambda: bool(self._thread is not None
                         and self._thread.isRunning()),
            what=name,
            describe=lambda: "The run has not reached a safe stopping point "
                             "yet. It may be inside a step that cannot be "
                             "interrupted.",
            on_force=self._force_stop,
        )
        self._stop_watcher.start()

    def _request_cooperative_stop(self) -> None:
        """Ask the worker and its thread to retire, without waiting."""
        worker = getattr(self, "_worker", None)
        if worker is not None:
            try:
                worker.request_cancel("stopped by the user")
            except Exception:
                pass
        try:
            if self._thread is not None:
                self._thread.requestInterruption()
        except Exception:
            pass

    def _force_stop(self) -> None:
        """Give the window back, whether or not the worker cooperates.

        Never reached without the user having been shown what it costs. The
        cooperative request goes first, so a worker that IS still checking
        stops on its own terms.

        A worker that is NOT checking -- one inside a long C call in torch or
        cellpose, which is the case this button exists for -- is PARKED by
        :func:`spacr.qt.bridge.drain_thread` rather than terminated. Parking
        keeps a reference so nothing drops a running QThread, lets the call
        finish in the background, and returns the window immediately.

        THIS USED TO CALL ``thread.terminate()``, and that was worse than the
        problem it solved. ``terminate()`` is ``pthread_cancel``, and every
        thread here runs Python: cancelled while holding the GIL, the whole
        process stops making progress with every thread still alive -- so
        "kill" produced a permanently frozen application rather than a
        returned one -- and cancelled inside a Qt or PySide internal it
        corrupts the heap and the process dies later somewhere unrelated.
        Both were live symptoms in this project, which is why
        ``tests/qt/test_qt_worker_teardown.py`` refuses the call outright.
        """
        import logging

        logging.getLogger(__name__).warning(
            "Force-stopping %s at the user's request", self.app_key)
        self._request_cooperative_stop()
        thread = self._thread
        if thread is None:
            return
        from ..bridge import drain_thread
        stopped = drain_thread(thread, getattr(self, "_worker", None),
                               timeout_ms=2000)
        if stopped:
            self._console.append_notice(
                "\nStopped. Anything being written at that moment is left "
                "half-written.\n")
        else:
            # Parked: the window is usable now, and the run is still out
            # there. Say so -- a user who is told "stopped" and then sees the
            # file grow has been lied to.
            self._console.append_notice(
                "\nStopped waiting. The step would not interrupt -- it is "
                "still finishing in the background and may keep writing for "
                "a while. The window is yours again.\n")

    def _on_import_settings(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Import settings CSV",
            filter="Settings (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            loaded = self._load_settings_csv(path)
            applied = self.apply_settings_dict(loaded)
            self._console.append_notice(
                "Loaded {count} settings from {path}\n",
                count=applied, path=path,
            )
            self._warn_about_moved_settings(loaded)
        except Exception as e:
            QMessageBox.warning(self, "Import failed", str(e))

    #: Key/value column-name pairs a spaCR settings CSV can use, in the order
    #: they are tried. Mirrors ``spacr.cli._CSV_COLUMNS``: ``Key,Value`` is
    #: what :func:`spacr.utils.save_settings` writes next to every run, while
    #: ``setting_key,setting_value`` is the documented default of
    #: :func:`spacr.utils.load_settings` and what ``spacr.io`` /
    #: ``spacr.object`` / ``spacr.spacr_cellpose`` write.
    _CSV_COLUMNS = (
        ("Key", "Value"),
        ("setting_key", "setting_value"),
        ("key", "value"),
        ("Setting", "Value"),
        ("name", "value"),
    )

    @classmethod
    def _load_settings_csv(cls, path: str) -> dict:
        """Parse a two-column settings CSV, whichever header spelling it uses.

        ``load_settings`` raises when the column names it was told to expect
        are absent, so trying only ``Key``/``Value`` made every CSV written by
        ``spacr.io.save_settings_to_db`` — the ``setting_key``/
        ``setting_value`` spelling — fail to import with "Import failed".

        :param path: path to the CSV.
        :returns: the parsed settings dict.
        :raises ValueError: when no recognised column pair is present.
        """
        from spacr.utils import load_settings
        first_error = None
        for key_col, value_col in cls._CSV_COLUMNS:
            try:
                return load_settings(path, setting_key=key_col,
                                     setting_value=value_col)
            except ValueError as e:
                # Wrong header spelling (or an unparseable file) — remember
                # the first complaint and try the next spelling.
                if first_error is None:
                    first_error = e
        raise first_error

    @staticmethod
    def _truthy(val) -> bool:
        """Interpret a CSV-loaded value as a boolean (values arrive as strings)."""
        if isinstance(val, str):
            return val.strip().lower() in ("true", "1", "yes")
        return bool(val)

    def _warn_about_moved_settings(self, loaded: dict) -> None:
        """Tell the user, in the console, when an imported CSV enables a
        setting this module no longer owns.

        Timelapse and the automated motility assay left the Mask module for
        modules of their own; an old mask CSV with ``timelapse=True`` would
        otherwise be applied minus that flag with nothing to show for it.
        Console text, never a modal — a QMessageBox here would hang headless
        runs.
        """
        if self.app_key != "mask":
            return
        notes = []
        if self._truthy(loaded.get("timelapse", False)):
            notes.append(
                "timelapse=True was ignored — tracking now lives in the "
                "Timelapse module (sidebar > Core > Timelapse).")
        if self._truthy(loaded.get("motility_analysis", False)):
            notes.append(
                "motility_analysis=True was ignored — the assay now lives in "
                "the Motility Assay module (sidebar > Core > Motility Assay).")
        for note in notes:
            self._console.append_notice(
                "[settings] {note}\n", note=note)

    def apply_settings_dict(self, settings: dict) -> int:
        """Push key/value pairs from `settings` into whichever settings
        widgets this app exposes. Silently skips keys the current app
        does not have — the same dict can safely be applied across
        several apps. Returns the count of keys actually applied."""
        settings = _translate_legacy_setting_keys(settings)
        applied = 0
        for key, val in settings.items():
            w = self._settings_model._widgets.get(key)
            if w is None:
                continue
            try:
                self._apply_value(w, val)
                applied += 1
            except Exception:
                pass
        self._settings_model._refresh_contextual_widgets()
        return applied

    def _apply_value(self, widget, val):
        from PySide6.QtWidgets import QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit
        if isinstance(widget, QCheckBox):
            widget.setChecked(str(val).lower() in ("true", "1", "yes"))
        elif isinstance(widget, QSpinBox):
            try:
                widget.setValue(int(float(val)))
            except (ValueError, TypeError):
                pass
        elif isinstance(widget, QDoubleSpinBox):
            try:
                widget.setValue(float(val))
            except (ValueError, TypeError):
                pass
        elif isinstance(widget, QComboBox):
            for i in range(widget.count()):
                if widget.itemText(i) == str(val):
                    widget.setCurrentIndex(i)
                    break
        elif hasattr(widget, "set_value"):
            # _ListEditor / _ListEdit / _ScalarEdit all round-trip their own
            # value. Importing a settings CSV used to go through the plain
            # QLineEdit branch below, which str()'d a list back into the box;
            # the chip editor is not a QLineEdit at all, so it would have been
            # skipped entirely.
            widget.set_value(val)
        elif isinstance(widget, QLineEdit):
            widget.setText("" if val is None else str(val))

    # ------------------------------------------------------------------
    # Usage
    # ------------------------------------------------------------------
    def _on_toggle_per_core(self, checked: bool):
        """Show/hide the per-core CPU panel. Creates one UsageBar per
        logical core the first time it's opened."""
        if checked and not self._per_core_bars:
            try:
                import psutil
                n = int(psutil.cpu_count(logical=True) or 0)
            except Exception:
                n = 0
            for i in range(n):
                bar = UsageBar(f"C{i:02d}")
                self._per_core_bars.append(bar)
                self._per_core_layout.addWidget(bar)
        self._per_core_wrap.setVisible(checked)

    def _refresh_usage(self):
        """Sample RAM/CPU/GPU on a worker; paint the bars when it returns.

        ``GPUtil.getGPUs()`` spawns ``nvidia-smi`` and waits for it: **25 ms**,
        measured, every single call. This runs on a 2 s timer and once more
        during every screen build, so inline it was a guaranteed 25 ms hitch
        twice a minute per open module and a 25 ms tax on opening one. psutil
        is 0.13 ms and could have stayed, but sampling everything in one place
        means one job rather than a split rule about which half is cheap.

        Nothing here touches a widget except the two ``set_value`` calls in
        :meth:`_apply_usage`, which run on the GUI thread. Overlapping polls
        are skipped rather than queued -- a machine slow enough to still be
        inside nvidia-smi 2 s later must not accumulate a backlog of them.
        """
        if self._usage_jobs.is_busy():
            return
        # Read the toggle here: it is a widget, and the worker may not look
        # at one.
        per_core = bool(self._btn_cpu_toggle.isChecked()
                        and self._per_core_bars)
        self._usage_jobs.submit(lambda: _sample_usage(per_core),
                                self._apply_usage)

    def _apply_usage(self, sample: dict) -> None:
        """Paint one worker-taken usage sample. GUI thread only."""
        if not sample:
            return
        ram = sample.get("ram")
        if ram is not None:
            self._usage_ram.set_value(ram)
        cpu = sample.get("cpu")
        if cpu is not None:
            self._usage_cpu.set_value(cpu)
        for bar, pct in zip(self._per_core_bars, sample.get("per_core") or ()):
            bar.set_value(pct)
        gpu = sample.get("gpu")
        if gpu is not None:
            self._usage_gpu.set_value(gpu)
        vram = sample.get("vram")
        if vram is not None:
            self._usage_vram.set_value(vram)

    def active_jobs(self) -> int:
        """How many of this screen's background jobs are still winding down.

        The pipeline run is deliberately not counted: it has its own Stop
        button, its own console and its own refusal-to-close in
        :meth:`closeEvent`. This is the housekeeping work -- the usage poll
        and the issue report -- that a test drives to quiescence.
        """
        return self._jobs.active_jobs() + self._usage_jobs.active_jobs()

    def is_busy(self) -> bool:
        """True while a background job has not yet delivered its result."""
        return self._jobs.is_busy() or self._usage_jobs.is_busy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_usage(per_core: bool) -> dict:
    """Read RAM/CPU/GPU utilisation. Runs on a worker thread.

    Module-level and widget-free on purpose: this is the whole of what
    :meth:`AppScreen._refresh_usage` sends off the GUI thread, so it must be
    impossible for it to reach a widget. A missing psutil leaves those keys
    out rather than failing the sample. A CPU-only machine reports zero GPU
    and VRAM use without entering GPUtil's subprocess boundary.

    :param per_core: whether the per-core panel is open and wants its own
        reading. Decided by the caller, on the GUI thread, from the toggle.
    :returns: a plain dict; every key optional.
    """
    sample: dict = {}
    try:
        import psutil
        sample["ram"] = psutil.virtual_memory().percent
        sample["cpu"] = psutil.cpu_percent(interval=None)
        if per_core:
            sample["per_core"] = psutil.cpu_percent(interval=None, percpu=True)
    except Exception:
        pass
    # GPUtil shells out to nvidia-smi.  Calling it when the executable does
    # not exist is both pointless and, after hundreds of short-lived worker
    # threads in a Qt process, has crashed in CPython's subprocess boundary
    # (CI run 31869225004).  The cheap executable check keeps CPU-only hosts
    # entirely outside that native boundary.  A real NVIDIA host still uses
    # GPUtil's established parsing and reports the same values as before.
    if _nvidia_smi_available():
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                sample["gpu"] = gpus[0].load * 100
                sample["vram"] = gpus[0].memoryUtil * 100
            else:
                sample["gpu"] = 0
                sample["vram"] = 0
        except Exception:
            pass
    else:
        sample["gpu"] = 0
        sample["vram"] = 0
    return sample


def _nvidia_smi_available() -> bool:
    """Return whether GPU telemetry can invoke an actual ``nvidia-smi``."""
    if shutil.which("nvidia-smi"):
        return True
    if sys.platform == "win32":
        drive = os.environ.get("SystemDrive", "C:")
        candidate = os.path.join(
            drive, "Program Files", "NVIDIA Corporation", "NVSMI",
            "nvidia-smi.exe",
        )
        return os.path.isfile(candidate)
    return False


def QtGui_QListWidgetItem_helper(fig, idx: int):
    """Build a :class:`QListWidgetItem` with a low-DPI thumbnail render
    of ``fig`` — used in the figures panel's history strip."""
    from io import BytesIO
    from PySide6.QtWidgets import QListWidgetItem
    item = QListWidgetItem()
    item.setText(f"#{idx + 1}")
    item.setTextAlignment(Qt.AlignCenter)
    try:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=32, bbox_inches="tight",
                     facecolor=fig.get_facecolor())
        pix = QPixmap()
        pix.loadFromData(buf.getvalue(), "PNG")
        if not pix.isNull():
            item.setIcon(QIcon(pix.scaled(
                140, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
    except Exception:
        pass
    return item


def _hyperparam_searchable(app_key: str) -> bool:
    """Whether ``app_key`` has a hyperparameter search, without paying for it.

    Imported lazily: spacr.qt.screens.hyperparam pulls spacr.hyperparam, and
    every screen construction would otherwise carry that cost whether or not
    the app can be searched.
    """
    try:
        from .hyperparam import searchable
    except Exception:
        return False
    return searchable(app_key)


def _build_live_preview_card(host):
    """Build the ``Live preview`` card + panel pair without adding it
    to any layout.

    The Mask app screen embeds this into a QSplitter alongside the
    console so the two panels can be resized against each other. The panel
    starts hidden and is shown when the user clicks the Live toggle.
    """
    from ..widgets.live_preview import LivePreviewPanel
    card = Card(title="Live preview")
    panel = LivePreviewPanel(card)
    card.body_layout.addWidget(panel)
    card.setMinimumHeight(300)
    return panel, card


def _build_measure_preview_card(host):
    """Build the Measure ``Crop preview`` card + panel pair (not added to a
    layout). Mirrors the Mask live preview but shows object crops from a merged
    array, tuned with the crop settings the Measure run will use."""
    from ..widgets.measure_preview import MeasurePreviewPanel
    card = Card(title="Crop preview")
    panel = MeasurePreviewPanel(card)
    card.body_layout.addWidget(panel)
    card.setMinimumHeight(300)
    return panel, card
