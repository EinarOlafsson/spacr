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
import re
import os
import shutil
import sys
import time
from functools import partial
from html import escape
from typing import Callable, Optional
from weakref import WeakMethod

from PySide6.QtCore import (
    QEvent, QObject, QRect, QSize, Qt, QThread, QTimer, Signal,
)
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..bridge import make_thread, resolve_pipeline_entry
from ..hidpi import device_ratio, scaled_for
from ..i18n import tr
from ..job_runner import JobRunner
from ..theme import (SPACING, ensure_widget_qss_applied,
                     register_widget_qss,
                     set_a_sheeted_widgets_own_rule)
from ..widgets import ApiHelpLabel, Card, Divider, Section, UsageBar
from ..widgets.flow import FlowLayout
from .settings_model import (
    CATEGORY_TOOLTIPS,
    SettingsWidgets,
    category_tooltip,
    object_of_setting,
    object_switch_keys,
    section_tooltip,
)

LOG = logging.getLogger(__name__)


def _example_pack_console(screen, owner):
    """Return the registered form's console after a possible screen rebuild."""
    current = getattr(owner, "_screens", {}).get(screen.app_key)
    return getattr(current, "_console", screen._console)


def _append_example_pack_report(console, report, applied: int) -> None:
    """Localize migration details while distinguishing reader and form counts."""
    console.append_notice(
        "[example] {applied} settings applied to the form from {name}; "
        "{accepted} CSV keys accepted.\n",
        applied=applied, name=report.source,
        accepted=len(report.applied) + len(report.renamed))
    if report.renamed:
        console.append_notice(
            "[example] Renamed {count} settings: {renames}\n",
            count=len(report.renamed),
            renames=", ".join(f"{old} → {new}" for old, new in report.renamed))
    elsewhere = set(report.elsewhere)
    dropped = sorted(key for key in report.dropped if key not in elsewhere)
    if dropped:
        console.append_notice(
            "[example] Dropped {count} unknown or retired settings: {keys}\n",
            count=len(dropped), keys=", ".join(dropped))
    if elsewhere:
        console.append_notice(
            "[example] Ignored {count} settings available in this build "
            "but not on this form: {keys}\n",
            count=len(elsewhere), keys=", ".join(sorted(elsewhere)))
    if report.malformed:
        console.append_notice(
            "[example] Skipped {count} unreadable CSV row(s).\n",
            count=report.malformed)


#: `organelleb_model_name`, `organellec_model_name`, ... -- the
#: per-organelle model fields generated when a run has more than one.
_ORGANELLE_MODEL_KEY = re.compile(r"^organelle[a-z]?_model_name$")

#: Fingerprints whose automatic report is being filed right now, each with
#: the monotonic clock reading it started at.
#:
#: Shared by every screen, because two screens can fail on the same crash
#: before the first report has come back from GitHub.
#:
#: TIMED, BECAUSE A DROPPED RESULT NEVER CLEARS ITS ENTRY. The runner hands
#: nothing to `on_done` when `cancel()` has bumped the generation or the
#: worker reports not-ok, which is what a closed screen does to a report in
#: flight. A bare set kept that fingerprint for the rest of the process, and
#: every later failure with the same traceback -- in any screen -- said "This
#: error is being reported already" and filed nothing.
_REPORTS_BEING_FILED: dict = {}

#: How long a report is allowed to be in flight before another failure may
#: file it again. `gh auth token` is capped at 8 s and each API call at 20 s,
#: so a report that has not come back inside two minutes is not coming back.
REPORT_IN_FLIGHT_SECONDS = 120.0

#: Edge of the gear beside "Copy console", in logical pixels before the
#: interface scale. Named rather than written twice, because the size is
#: now set once at construction and re-derived from this number whenever
#: the scale moves -- the two have to be the same number or the gear does
#: not come back to where it started.
GEAR_ICON_PX = 18


def _a_report_is_in_flight(fingerprint: str) -> bool:
    """Whether this fingerprint is being filed right now, dropping stale ones.

    :param fingerprint: the traceback fingerprint to look for.
    :returns: ``True`` only while a report started less than
        :data:`REPORT_IN_FLIGHT_SECONDS` ago is outstanding.
    """
    now = time.monotonic()
    for key, started in list(_REPORTS_BEING_FILED.items()):
        if now - started >= REPORT_IN_FLIGHT_SECONDS:
            _REPORTS_BEING_FILED.pop(key, None)
    return fingerprint in _REPORTS_BEING_FILED


#: Object name the settings column carries, and what the block below keys
#: off. It is the column itself, and — see `_settings_panel_qss` — the
#: rule that names it is a rule saying *paint nothing*.
SETTINGS_PANEL_NAME = "SettingsBox"
#: The scroll area inside ONE category tab, when the settings column is
#: drawn as tabs rather than as a single scrolling list. It needs its own
#: name because the column's name is looked up with `findChild` to mean
#: the column specifically, and two widgets answering to that would make
#: the lookup return whichever Qt reached first.
SETTINGS_TAB_PAGE_NAME = "SettingsTabPage"
#: The tab strip itself. Named since it was written; registered for QSS
#: only now, which is why its pane took the blanket window fill.
SETTINGS_TABS_NAME = "SettingsCategoryTabs"

#: The "Point <module> at some data" banner at the top of that column.
EMPTY_STATE_NAME = "EmptyStateBanner"


class _ExplainerBrowser(QTextBrowser):
    """Text browser that asks its screen to rerender after a locale change.

    :param refresh: the screen's rerender, held as a WEAK method. The
        browser is a child of the screen, so a strong reference would be a
        cycle through a QObject; a dead one simply means there is nothing
        left to rerender.
    :param parent: parent widget; ownership only.
    """

    def __init__(
        self,
        refresh: Callable[[Optional[str]], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        """Hold the screen's rerender weakly and take it as parent."""
        super().__init__(parent)
        self._refresh_explainers = WeakMethod(refresh)

    def retranslate_dynamic_content(
        self,
        language: Optional[str] = None,
    ) -> None:
        """Invoke the owning screen's exact-template renderer."""
        refresh = self._refresh_explainers()
        if refresh is not None:
            refresh(language)


def _settings_panel_qss(palette: dict, opacity=None) -> str:
    """Return transparent styling for the settings column and empty state.

    Category cards provide the visible surfaces and apply the configured
    page opacity. Explicit rules keep the surrounding scroll area, viewport,
    and empty-state banner from inheriting the opaque window background.
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
/* THE TABBED LAYOUT NEEDS THE SAME THING SAID AGAIN, for different
   widgets. When the categories are drawn as tabs, each tab holds its own
   scroll area and the strip holds a pane, and none of them is the column
   above -- so the rule naming the column reaches none of them and they
   fall through to the blanket `QWidget {{ background-color: bg }}`, which
   is the WINDOW colour and is `#000000` on the dark theme. That is the
   black slab behind the settings, and it is the same defect the column
   itself had: the cards inside carry the visible surfaces, so everything
   behind them paints nothing. */
QScrollArea#{SETTINGS_TAB_PAGE_NAME} {{
    background: transparent;
    border: none;
}}
QTabWidget#{SETTINGS_TABS_NAME}::pane {{
    background: transparent;
    border: none;
}}
/* Qt builds `qt_tabwidget_tabbar` itself, and with no rule of its own it
   takes the blanket fill like anything else. The TABS keep the shipped
   look -- they are controls and belong on a surface; it is only the strip
   behind them that must not paint. */
QTabWidget#{SETTINGS_TABS_NAME} > QTabBar {{
    background: transparent;
}}
"""


register_widget_qss(SETTINGS_PANEL_NAME, _settings_panel_qss, replace=True)


#: What a module screen tells the user to do when it has nothing more
#: specific to say. Every ``AppScreen`` is the same gesture — fill the form
#: in, press Run — so the instruction is the same sentence.
DEFAULT_INSTRUCTION = "Configure settings, then press Run."



#: ``(dimension, label, tooltip)``, in the order the switches are drawn.
#:
#: The labels are the words the request used, and they are drawn as
#: :class:`spacr.qt.widgets.AiToggleLabel` -- the same lit-while-on toggle
#: Live and AI in this row already are, and the only state toggle in the
#: project that can carry a WORD. A fold button is the shape being matched,
#: but it is an icon and nothing else: it derives its picture, its name and
#: its maturity colour from a registry key, and "3D" is not a module, so it
#: has neither an icon nor a row to read. Reusing it would have put two
#: buttons labelled "3" and "T" on the masthead.
DIMENSION_TOGGLES = (
    ("z", "3D",
     "Click to show the volumetric settings — the ones that mean nothing "
     "until a field has a z axis: which axis it is, how the stack is "
     "segmented or projected, and the voxel geometry that says how far "
     "apart two planes really are. Hiding them keeps their values: they "
     "still reach the run and the settings file."),
    ("t", "Time",
     "Click to show the timelapse settings — the ones that mean nothing "
     "until a field has a time axis: the time axis itself, the frame "
     "interval, and how objects are linked from one frame to the next. "
     "Hiding them keeps their values: they still reach the run and the "
     "settings file."),
)

#: Screens that carry the switches. Mask Generation and Measure are the two
#: modules that read a plate off disk and decide what its axes are.
DIMENSION_TOGGLE_APPS = frozenset({"mask", "measure"})

#: The widest these two may force the action row -- and so the window -- to be.
#:
#: A QHBoxLayout cannot go below the sum of its children's minimum widths, and
#: the action row's minimum is the module screen's minimum, which the body
#: splitter then has to satisfy out of whatever the window is. So every pixel
#: a toggle demands here is a pixel taken off the settings column at any
#: window narrower than about 1500. Measured on Mask Generation at 1200x720:
#: 3D asks for 69 px and Time for 92 px unaided, which took the row from 899
#: to 1076 and the settings column from 260 px to 83 px -- a 624 px settings
#: card in an 83 px viewport, with the labels hanging out of the right of it.
#: That is the failure :data:`spacr.qt.widgets.ai_toggle_label
#: .ELIDE_ABOVE_PX` exists for, arriving from two short labels rather than
#: one long one, and a secondary control must not be able to starve the
#: primary.
#:
#: 24 px is a click target, not a word: the label still ASKS for its full
#: width through ``sizeHint``, so a row with room to spare draws "3D" and
#: "Time" in full and only a squeezed one clips them. The row comes out at
#: 963 and the settings column at 196 px, which is where the setting labels
#: fit inside their card again.
DIMENSION_TOGGLE_MIN_PX = 24

#: ``dimension -> the settings that ANNOUNCE it in a loaded settings file``.
#:
#: These are the keys whose whole job is to say the data has the dimension:
#: ``z_stack`` refuses to run unless the array has a real z axis, ``t_stack``
#: unless each field is a (T, Z, Y, X) volume, and ``timelapse`` is what
#: makes a mask run group its plate into time stacks. A CSV that sets one of
#: them is a CSV about that dimension, so loading it lights the switch --
#: without which importing a volumetric settings file would hide the very
#: settings it had just filled in.
DIMENSION_GATE_KEYS = {
    "z": ("z_stack",),
    "t": ("t_stack", "timelapse"),
}

#: Cache for :func:`dimension_settings`, filled on first use.
_DIMENSION_SETTINGS: Optional[dict] = None


def dimension_settings() -> dict:
    """``dimension -> the setting keys that only mean something in it``.

    READ OFF THE SETTINGS, NOT LISTED HERE. :data:`spacr.settings.categories`
    already carries the answer, because the split was made deliberately when
    the experimental axes were added:

    * ``"3D Settings (Beta)"`` is every key that presumes a z axis --
      ``z_stack``, the segmentation mode, the projection, the two voxel
      sizes and the ``anisotropy`` they derive (all three of which answer
      "how far apart are two planes"), and ``stitch_threshold``, which links
      labels BETWEEN planes.
    * ``"4D Settings (Beta)"`` is the time half of the same split, and the
      source says so where the split is written: ``z_axis`` is filed under
      3D "because 4D builds on the same z plan", which leaves the 4D list
      holding "only time-axis and inter-frame tracking controls".
    * the Timelapse and Motility modules' own key lists join the time set,
      because every key in them is a per-frame or between-frame quantity --
      the frame rate, which tracker links the frames, how far an object may
      move between two of them, how long it may vanish for.

    A key in neither set is dimension-free and stays visible whatever the
    switches say.

    :returns: a fresh dict of ``str -> frozenset``; callers may keep it.
    """
    global _DIMENSION_SETTINGS
    if _DIMENSION_SETTINGS is None:
        from ...settings import (categories, motility_advanced_settings,
                                 motility_settings, timelapse_settings)
        _DIMENSION_SETTINGS = {
            "z": frozenset(categories.get("3D Settings (Beta)", ())),
            "t": (frozenset(categories.get("4D Settings (Beta)", ()))
                  | frozenset(timelapse_settings) | {"timelapse"}
                  | frozenset(motility_settings)
                  | frozenset(motility_advanced_settings)),
        }
    return dict(_DIMENSION_SETTINGS)


def setting_dimension(key: str) -> str:
    """The dimension ``key`` depends on -- ``"z"``, ``"t"`` or ``""``.

    Blank for the great majority of settings, which mean the same thing
    whatever axes the plate has.

    :param key: settings key, matched exactly against the z (3-D) and t
        (time-lapse and motility) setting sets of :func:`dimension_settings`.
    """
    for dimension, keys in dimension_settings().items():
        if key in keys:
            return dimension
    return ""


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

    The shared component keeps headings and actions consistent across screens.
    It is transparent because a header is a page region rather than a card;
    this allows the configured page backdrop to remain visible.

    :param title: the module name, shown large.
    :param description: one line to the right of the name. Never wrapped —
      it may shrink below its ideal width rather than force the window
      wider — and repeated in its own hover help so a truncated one is
      readable.
    :param instruction: one line under the name. Omitted if empty.
    :param app_key: registry key. Given one, the module's API documentation
      link is the last line of the description's hover help. NOTHING IS
      DRAWN BESIDE THE DESCRIPTION: a dot used to be, and a masthead reads
      as one sentence rather than a sentence and a mark. The link is not
      lost with it -- see :class:`spacr.qt.widgets.ApiHelpLabel`, which
      carries the same help every setting's label carries.
    :param parent: parent widget; ownership only.
    """

    def __init__(self, title: str, description: str = "",
                 instruction: str = "", *, app_key: Optional[str] = None,
                 parent: Optional[QWidget] = None):
        """Build the shared module masthead.

        :param title: the module's name, set at display size.
        :param description: one-line blurb; with an ``app_key`` it becomes the
            link into that module's API help.
        :param instruction: what to do first, shown under the title.
        :param app_key: the module the description links to; without it the
            blurb is plain text.
        :param parent: parent widget, or ``None``.
        """
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
        self.instruction_label = QLabel(str(instruction or ""), self)
        self.instruction_label.setObjectName("Muted")
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setVisible(bool(instruction))
        title_col.addWidget(self.instruction_label)
        row.addLayout(title_col)

        self.description_label: Optional[ApiHelpLabel] = None
        #: The description, when it carries an API link; ``None`` otherwise.
        #: A screen that serves two modules from one masthead repoints it --
        #: see :meth:`ApiHelpLabel.set_api_app_key`.
        self.api_help: Optional[ApiHelpLabel] = None
        if description:
            blurb = ApiHelpLabel(str(description), str(app_key or ""),
                                 parent=self)
            blurb.setObjectName("Muted")
            blurb.setWordWrap(False)
            blurb.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            blurb.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
            blurb.setMinimumWidth(0)
            row.addWidget(blurb)
            self.description_label = blurb
            if app_key:
                self.api_help = blurb

        row.addStretch(1)
        self._row = row

    def add_trailing(self, widget: QWidget, stretch: int = 0) -> QWidget:
        """Put ``widget`` on the header row, right of the stretch.

        For the screens whose header row also carries controls — Control
        Chart's table picker and Load button, Graph Builder's source
        label. They keep their row; they stop having to build the title
        part of it themselves.

        :param widget: the control to append to the header row; it is also
            returned, so a caller can build and keep it in one expression.
        """
        self._row.addWidget(widget, stretch)
        return widget


HINT_STRIP_LINES = 4


def _fit_to_lines(text: str, label, lines: int) -> str:
    """``text`` shortened until it wraps into at most ``lines`` of ``label``.

    Qt elides a single line for you and does nothing for a wrapped one, so a
    four-line strip silently clipped anything longer. This walks back to a word
    boundary and ends with an ellipsis, so the cut reads as deliberate.

    Falls back to the original text when the label has no usable width yet --
    during construction it has none, and a guess there would trim against a
    width that is about to change.
    """
    from PySide6.QtGui import QFontMetrics

    text = " ".join(str(text).split())
    width = max(0, label.width() - 8)
    if width <= 0 or not text:
        return text
    metrics = QFontMetrics(label.font())

    def wrapped_lines(candidate: str) -> int:
        """How many lines ``candidate`` wraps to at the measured width.

        Measured against the font Qt is actually painting, so it stays right at
        any font scale rather than at the one this was written on.
        """
        rect = metrics.boundingRect(
            0, 0, width, 0, Qt.TextWordWrap, candidate)
        return max(1, round(rect.height() / max(1, metrics.lineSpacing())))

    if wrapped_lines(text) <= lines:
        return text
    words = text.split(" ")
    low, high = 0, len(words)
    while low < high:
        middle = (low + high + 1) // 2
        if wrapped_lines(" ".join(words[:middle]) + "…") <= lines:
            low = middle
        else:
            high = middle - 1
    return (" ".join(words[:low]) + "…") if low else text[:1] + "…"

CATEGORY_STRIP_LINES = 3


def _height_of_lines(metrics, lines: int) -> int:
    """The height ``lines`` wrapped lines occupy in the font ``metrics`` reads.

    ASK QT RATHER THAN REBUILD ITS ARITHMETIC. Every closed form tried here
    has been wrong on some font, because the height of a wrapped paragraph
    is a layout question and only the layout engine knows the answer.

    The two that were tried, and how they failed:

    ``lineSpacing() * lines`` is short wherever a font's OS/2 table asks for
    a NEGATIVE leading -- several URW and Bitstream faces do, by one to three
    pixels at interface sizes. Qt does not overlap glyphs to honour it, so
    the product reserved less than the text needed and the last line came out
    clipped.

    ``max(lineSpacing(), height()) * lines`` was the repair for that, and it
    is still wrong: measured against ``QLabel.heightForWidth`` over 40
    families x 28 pixel sizes at three lines, it UNDER-reserves 450 of 1,120
    combinations and over-reserves another 202. It fixed the negative-leading
    case and missed everything else.

    ``boundingRect`` with the same flags Qt lays the label out with is exact
    on all 1,120 -- never short, never over. It subsumes the negative-leading
    case rather than special-casing it, so the reasoning above is history
    rather than a rule to maintain.

    WHICH FONT IS PAINTING IS NOT SOMETHING THE PACKAGE DECIDES, which is why
    this matters at all. The stylesheet asks for Open Sans and the package
    ships it, but a machine that has not registered those files -- a test
    runner, or anything reading the interface through fontconfig's
    substitution -- paints with whatever it has. Open Sans has a leading of
    zero, so on a developer's machine the broken expressions agreed to the
    pixel and nothing looked wrong.

    :param metrics: the ``QFontMetrics`` of the label that will paint them.
    :param lines: how many lines to reserve; anything below one reserves one.
    :returns: the height in pixels.
    """
    count = max(1, int(lines))
    return metrics.boundingRect(
        QRect(0, 0, 1 << 20, 0),
        int(Qt.TextWordWrap | Qt.AlignTop | Qt.AlignLeft),
        "\n".join(["Xg"] * count)).height()


SECTION_HINTS = CATEGORY_TOOLTIPS


COLUMN_TABLES = {
    "annotation_column":  "png_list",
    "annotation_columns": "png_list",
    "classes":            "png_list",
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


def module_maturity(app_key: str) -> str:
    """The stage ``app_key`` is drawn in, registry row or not.

    ``app_stage`` answers ``stable`` for a key it has never heard of, which
    is the right answer for a typo and the wrong one for a module that was
    folded into a host and had its row deleted: its settings would go from
    beta-tinted to unmarked on the day the row went, claiming a maturity
    nobody signed off. So a key with no row is asked of the fold tables,
    which record what its tile said.

    :param app_key: the module's registry key.
    :returns: ``"alpha"``, ``"beta"`` or ``"stable"``.
    """
    from ..app import APPS, app_stage

    stage = app_stage(app_key)
    if any(row and row[0] == app_key for row in APPS):
        return stage
    try:
        from ..widgets.fold_strip import folded_fallback
        folded = folded_fallback(app_key)[2]
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not read the fold tables for %r", app_key,
                  exc_info=True)
        folded = ""
    return folded or stage


def settings_section_maturity(app_key: str, title: str) -> str:
    """Return the least-mature stage applying to one settings section.

    An alpha or beta module colours every one of its settings. A stable
    module can still contain an explicitly experimental ``(Beta)``/``(Alpha)``
    category, in which case that section receives the more cautious stage.

    :param app_key: the module's registry key, whose stage comes from
        :func:`module_maturity`.
    :param title: the section heading; compared case-insensitively, it is
        alpha when it is ``"alpha"`` or contains ``"(alpha)"``, beta
        likewise, and stable otherwise.
    :returns: ``"alpha"``, ``"beta"`` or ``"stable"``.
    """
    module_stage = module_maturity(app_key)
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
    "barcode_qc":      "Barcode QC",
    "anndata_export":  "AnnData Export",
    "illumination":    "Illumination Correction",
}


APP_INTROS = {
    "mask":            "Segment cells, nuclei, pathogens and organelles with Cellpose and build the merged image+mask arrays.",
    "timelapse":       "Segment each frame of a time series and link objects across frames into tracks, then export per-channel movies.",
    "motility":        "Rebuild per-frame tracks, score velocity and straightness per object, and split them by infection state.",
    "measure":         "Extract per-object intensity + morphology features from masks and write them to the measurements database.",
    "annotate":        "Review single-object image crops on a grid and label them; annotations save back to the database.",
    "classify_merged": "Train classifiers from image crops or measured features in one interface. Define classes from plate metadata or annotations, then choose a PyTorch image model or a gradient-boosted feature model.",
    "classify":        "Train and test Torch computer-vision models (CNNs / transformers) to classify single-object images.",
    "ml_analyze":      "Train a classical ML classifier (XGBoost, random forest, …) on per-object features and score every well.",
    "map_barcodes":    "Map sequencing reads to gRNA barcodes and link them to screen wells.",
    "regression":      "Regress screen phenotypes against guide/gene effects across plates.",
    "make_masks":      "Inspect and edit segmentation masks with brush, flood-fill, relabel, fill and object-removal tools; train, apply, compare and select segmentation models from the integrated workbench.",
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
    "train_compare":   "Compare loss and accuracy curves from multiple training runs on shared axes. The adjacent settings comparison separates model and training changes from differences in the software environment.",
    "classifier_evaluation": "Inspect held-out predictions from grouped or nested cross-validation, calibration, confusion matrices, per-plate performance and explicit train/test leakage checks.",
    "run_history":     "Search every recorded job and inspect its exact settings, hashed inputs and outputs, warnings, failure traceback, software versions, seeds and performance.",
    "distributed_jobs": "Submit resolved spaCR settings to SSH workstations, Slurm clusters or cloud/HPC command profiles and monitor them locally.",
    "analyze_plaques": "Detect and quantify plaques in plaque-assay images.",
    "recruitment":     "Quantify recruitment of a marker to a compartment across conditions.",
    "invasion":        "Classify attached and invaded parasites from two-colour differential staining. Estimate the intensity threshold per field and flag fields that lack two distinguishable populations.",
    "replication":     "Count the parasites in every vacuole and turn that into a replication rate: endodyogeny doubles a vacuole 1 -> 2 -> 4 -> 8, so the distribution of counts per vacuole is the readout, not the mean.",
    "barcode_qc":      "Evaluate a completed barcode-mapping run using read depth per well, low-depth wells, unmapped reads, barcode collisions, positional effects and library coverage. Given the intended number of gRNAs per well, estimate and report the abundance threshold and its sensitivity.",
    "anndata_export":  "Export the measurement tables as AnnData (.h5ad) - N objects x M features with per-object metadata, feature definitions, embeddings and provenance - so scanpy, scvi-tools and squidpy can read a spaCR run directly.",
    "illumination":    "No microscope lights a field evenly, so the same cell measures brighter at the centre than at a corner \u2014 routinely 10\u201340% on a widefield screen, and it does not average out of a per-well aggregate. This estimates the illumination field from the plate's own merged fields (a per-pixel median across fields, then a smooth low-order surface), QCs it, and installs it as a preprocessing hook that every measure worker applies before a single feature is computed.",
}

try:
    from spacr.plugins import plugin_apps as _plugin_apps
    for _plugin_app in _plugin_apps():
        APP_TITLES.setdefault(_plugin_app.key, _plugin_app.name)
        APP_INTROS.setdefault(_plugin_app.key, _plugin_app.description)
except Exception:
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
from ..settings_pack import _FORM_RENAMES as _RENAMED_SETTING_KEYS

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

    ONE SEMANTIC FOLD RUNS HERE TOO: the retired `Toxoplasma` / `toxo`
    switch, through `spacr.settings._fold_toxoplasma`. The form has no
    widget for the switch, so without it a file saying `Toxoplasma=False`
    would load with the annotation field still on its default and the run
    would annotate what the file had turned off.

    :param settings: a settings dict, not modified.
    :returns: a new dict with retired keys renamed.
    """
    out = dict(settings)
    for old, new in _RENAMED_SETTING_KEYS.items():
        if old in out:
            value = out.pop(old)
            out.setdefault(new, value)
    for key in list(out):
        replacement = _surviving_name_of(key)
        if not replacement or replacement == key:
            continue
        value = out.pop(key)
        for name in ((replacement,) if isinstance(replacement, str)
                     else tuple(replacement)):
            out.setdefault(name, value)
    from spacr.settings import _fold_toxoplasma

    return _fold_toxoplasma(out)


def _surviving_name_of(key: str):
    """What a retired setting is called now, or ``None`` if it is current.

    ONE RESOLVER, SHARED. This used to be the only place in spaCR that knew
    the organelle suffix rule, and it had its own regex to do it -- so the Qt
    panel migrated `organelleq_min_size` while the run ignored it and the
    doctor said nothing about it. The rule now lives in
    `spacr.settings.surviving_setting_name`, which all three consult, and the
    regex is gone rather than generalised.

    ALSO WHY THIS NO LONGER READS `RETIRED_SETTINGS` DIRECTLY: that table
    contains withdrawals and one SEMANTIC migration as well as renames.
    Reading it here folded `gradient_accumulation: False` straight onto
    `gradient_accumulation_steps`, where `int()` makes it ZERO -- and a step
    count of zero is not a state the code has. `steps = 1` is the off state,
    which is what `settings._fold_gradient_accumulation` exists to produce.

    :param key: the key a settings file carries.
    :returns: the name read today, a tuple for a split, or ``None``.
    """
    from spacr.settings import surviving_setting_name

    survivors = surviving_setting_name(key)
    if not survivors:
        return None
    return survivors[0] if len(survivors) == 1 else tuple(survivors)


def _elapsed_words(seconds: float) -> str:
    """``473.0`` -> ``"7 min 53 s"``. What the heartbeat says out loud.

    WHOLE UNITS ONLY, and never a bare float. "473.0 s" is a number a reader
    has to divide before it means anything, and the question the heartbeat
    answers -- has this been going long enough to worry -- is asked in
    minutes and hours.
    """
    total = int(max(0.0, float(seconds)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


class _LateCaptionTranslator(QObject):
    """Runs the language pass over a subtree that arrives after the screen.

    A screen is translated ONCE, by ``MainWindow`` when it builds it. What
    is parented into it afterwards has never met the pass and never will,
    so on a non-English screen it sits in English beside translated text.
    The declared preview :mod:`spacr.qt.preview_registry` installs, and the
    settings strip that :mod:`spacr.qt.settings_search` inserts, are both
    that: built the first time a module is opened, which is after the pass.

    A FOLDED MODULE IS THE THIRD, and the largest: its page is a whole
    module screen built by a fold button rather than by the window, so
    nothing routes it through the one call that runs the pass. Every
    caption on it -- masthead, instruction and each of its several hundred
    settings rows -- and the tab it arrived on opened in English inside a
    translated window.

    Installed as an event filter on the hosts those land in.
    ``QEvent.ChildAdded`` is the moment of parenting, and the pass is
    deferred one turn of the event loop from there: a widget can be
    parented before its own children exist, and
    :func:`spacr.qt.i18n.retranslate_widget_tree` caches whatever caption a
    widget holds the first time it sees one as that widget's English
    source, so a pass that ran too early would record an empty caption and
    opt the control out for good.
    """

    def eventFilter(self, watched, event):  # noqa: N802 - Qt override
        """Schedule a pass over ``watched``, WITHOUT touching the new child.

        THE ARRIVING CHILD MUST NOT BE ASKED FOR HERE -- no `child` call on
        this event -- and the reason is not
        style. `ChildAdded` is delivered synchronously from inside the
        child's C++ constructor, before Shiboken has registered the wrapper
        for the Python class being built. Asking for the child at that
        instant mints a BARE QWidget wrapper and registers THAT under the
        child's pointer; Shiboken never overwrites an existing key, so the
        real wrapper is never registered, and when the bare one is released
        the pointer is left mapped to nothing for the rest of the process.

        The child then loses its whole dynamic metaobject: its own signals
        vanish from it, so emitting one is a silent no-op, and
        `findChildren()` by type cannot see it. That is what produced
        "libpyside: addMetaMethod: ... No Wrapper found." once per screen
        built -- see Part 5 of `tools/diagnose_pyside_slot_warning.py`,
        which reproduces it and this fix in isolation.

        THE KEEPING IS WHAT DOES THE DAMAGE, not the call: a bare wrapper
        that is dropped immediately is collected again before the real
        registration and costs nothing, which is why deferring with
        `partial(..., child)` -- holding it for exactly one turn -- was the
        shape that broke it. So `watched` is deferred instead. It is
        already fully wrapped, and a turn later every child of it is too.
        """
        if event.type() != QEvent.ChildAdded:
            return False
        QTimer.singleShot(0, partial(self._on_arrivals_in, watched))
        return False

    #: Marks a child this filter has already run its pass over. A Qt
    #: dynamic property rather than a Python attribute or an id() set,
    #: because it has to survive the Python wrapper being collected and
    #: recreated, and id()s of dead widgets get reused.
    _HANDLED = "_spacr_late_caption_seen"

    def _on_arrivals_in(self, host) -> None:
        """Run the pass over children of ``host`` that have not had one.

        Scans rather than being handed the child, because the arrival is
        the one moment the child cannot safely be touched -- see
        :meth:`eventFilter`. A turn later they are ordinary widgets of
        their real classes.
        """
        try:
            children = [child for child in host.children()
                        if isinstance(child, QWidget)]
        except RuntimeError:
            return
        for widget in children:
            try:
                if widget.property(self._HANDLED):
                    continue
                widget.setProperty(self._HANDLED, True)
            except RuntimeError:
                continue
            self._on_arrival(widget)

    def _on_arrival(self, widget) -> None:
        """Follow and translate a subtree, a turn after it was parented.

        The widget arrives here as its REAL class, because nothing wrapped
        it while it was still being constructed. That is why the strip
        below is recognised with `isinstance` rather than by asking Qt with
        `inherits`: the bare-QWidget wrapper that made `isinstance`
        useless here no longer happens.
        """
        try:
            if isinstance(widget, QTabWidget):
                self._watch_pages_of(widget)
                self._translate(widget.parent() or widget)
                return
            self._translate(self._pass_root(widget))
        except RuntimeError:
            pass

    def _watch_pages_of(self, strip) -> None:
        """Watch the widget ``strip`` parents its pages into.

        A page is parented into the ``QStackedWidget`` INSIDE a tab widget
        rather than into the tab widget itself, so a filter that saw the
        strip arrive would never see a page land on it. The strip arrives
        empty and every module that becomes a page on it comes later, so
        without this only the first fold a user opens is translated.
        """
        for stack in strip.findChildren(QStackedWidget,
                                        options=Qt.FindDirectChildrenOnly):
            stack.removeEventFilter(self)
            stack.installEventFilter(self)

    @staticmethod
    def _pass_root(widget):
        """The subtree a pass for ``widget`` has to cover.

        A page carries a caption that is not its own: the tab it arrived on
        belongs to the strip above it, and translating the page alone
        leaves that tab in English over a translated page. So the strip is
        the root whenever the new widget is a page on one -- one walk that
        reaches the page and the tabs together.
        """
        parent = widget.parent()
        strip = parent.parent() if parent is not None else None
        if isinstance(strip, QTabWidget) and strip.indexOf(widget) >= 0:
            return strip
        return widget

    @staticmethod
    def _translate(widget) -> None:
        """Translate ``widget`` and its descendants, if it is still alive."""
        try:
            from ..i18n import retranslate_widget_tree

            retranslate_widget_tree(widget, only_new=True)
            from .settings_model import retarget_field_tooltips

            retarget_field_tooltips(widget)
        except RuntimeError:
            pass
        except Exception:
            LOG.exception("could not translate a late settings panel")


class _CaptionsBuiltWhenTheyAreAskedFor(dict):
    """``AppScreen._hint_map``, with the captions still waiting still to come.

    A caption is what carries a setting's key, its help and its hover
    behaviour, and this is the panel's index of them. A row the object rule
    has already hidden does not have one yet, so WALKING this index has to
    hand over the rest first -- the check that every setting's caption names
    its setting reads exactly this, and an index holding a fraction of the
    panel would pass it while checking almost nothing.

    LOOKING ONE UP DOES NOT BUILD ANYTHING, which is the whole reason the two
    are separated: a lookup is what a pointer moving across the panel does,
    several times a second, and the answer for a widget that is not a caption
    is "no hint" whatever else exists.
    """

    def __init__(self, caption_the_rest):
        """:param caption_the_rest: called once, to caption the waiting rows."""
        super().__init__()
        self._caption_the_rest = caption_the_rest

    def _complete(self) -> None:
        """Caption the rows that are still waiting. Idempotent."""
        build, self._caption_the_rest = self._caption_the_rest, None
        if build is not None:
            build()

    def __iter__(self):
        """Build the deferred rows first, then answer as a normal mapping.

        EVERY access point has to force the build, not just the obvious one. A
        reader that took ``len`` for a cheap check and ``in`` for a cheap test
        would see an empty mapping and conclude there is nothing here, which is
        the failure mode a lazy container has and a list does not.
        """
        self._complete()
        return super().__iter__()

    def __len__(self) -> int:
        """Build the deferred rows first, then answer as a normal mapping.

        EVERY access point has to force the build, not just the obvious one. A
        reader that took ``len`` for a cheap check and ``in`` for a cheap test
        would see an empty mapping and conclude there is nothing here, which is
        the failure mode a lazy container has and a list does not.
        """
        self._complete()
        return super().__len__()

    def keys(self):
        """The widgets that have a caption, building any still waiting."""
        self._complete()
        return super().keys()

    def items(self):
        """Every widget and its caption, building any still waiting."""
        self._complete()
        return super().items()

    def values(self):
        """Every caption, building any still waiting."""
        self._complete()
        return super().values()


class _RowsBuiltWhenTheyAreAskedFor(list):
    """A heading's rows, with the ones the run has no object for still to come.

    `AppScreen._build_settings_section` leaves a row's caption unbuilt when
    the object rule has already decided the row must not be on screen, which
    on Mask is most of the panel. The rows that ARE captioned go in here as
    usual, so the list is a real list holding real rows; what it adds is that
    ASKING for its contents builds the rest first.

    That is the whole reason it exists. Several checks find a settings row by
    walking ``Section._row_widgets`` -- the module smoke test reads every entry
    as a labelled setting row and asserts what it carries -- and a list that
    quietly held a twentieth of the panel's rows would let each of them pass
    while checking almost nothing. Deferring the work is only worth doing if
    nothing can tell the difference except by being faster.

    ``append`` is deliberately NOT one of the methods that builds: it is what
    ``Section.add_row`` calls while the rest of the rows are being built, and
    the callback is dropped before that starts, so the pass cannot re-enter.
    """

    def __init__(self, build_the_rest):
        """:param build_the_rest: called once, to lay out the waiting rows."""
        super().__init__()
        self._build_the_rest = build_the_rest

    def _complete(self) -> None:
        """Build the rows that have not been laid out yet. Idempotent."""
        build, self._build_the_rest = self._build_the_rest, None
        if build is not None:
            build()

    def __len__(self) -> int:
        """Build the deferred rows first, then answer as a normal list.

        EVERY access point has to force the build, not just the obvious one. A
        reader that took ``len`` for a cheap check and ``in`` for a cheap test
        would see an empty list and conclude there is nothing here, which is
        the failure mode a lazy container has and a list does not.
        """
        self._complete()
        return super().__len__()

    def __iter__(self):
        """Build the deferred rows first, then answer as a normal list.

        EVERY access point has to force the build, not just the obvious one. A
        reader that took ``len`` for a cheap check and ``in`` for a cheap test
        would see an empty list and conclude there is nothing here, which is
        the failure mode a lazy container has and a list does not.
        """
        self._complete()
        return super().__iter__()

    def __getitem__(self, index):
        """Build the deferred rows first, then answer as a normal list.

        EVERY access point has to force the build, not just the obvious one. A
        reader that took ``len`` for a cheap check and ``in`` for a cheap test
        would see an empty list and conclude there is nothing here, which is
        the failure mode a lazy container has and a list does not.
        """
        self._complete()
        return super().__getitem__(index)

    def __bool__(self) -> bool:
        """Build the deferred rows first, then answer as a normal list.

        EVERY access point has to force the build, not just the obvious one. A
        reader that took ``len`` for a cheap check and ``in`` for a cheap test
        would see an empty list and conclude there is nothing here, which is
        the failure mode a lazy container has and a list does not.
        """
        self._complete()
        return super().__len__() > 0

    def __contains__(self, item) -> bool:
        """Build the deferred rows first, then answer as a normal list.

        EVERY access point has to force the build, not just the obvious one. A
        reader that took ``len`` for a cheap check and ``in`` for a cheap test
        would see an empty list and conclude there is nothing here, which is
        the failure mode a lazy container has and a list does not.
        """
        self._complete()
        return super().__contains__(item)

    def __repr__(self) -> str:
        """Build the deferred rows, then show them.

        Forces the build like every other access: a repr that showed an empty
        list in a debugger would be a lie about a container that has contents.
        """
        self._complete()
        return super().__repr__()

#: Top-level section containing each module's example-data control. Modules
#: omitted from this mapping do not expose an example-data action.
EXAMPLE_DATA_SECTIONS = {
    "regression": "Input Tables",
    "mask": "Input & Metadata",
    "measure": "Input & Experiment",
    "classify": "Plate Sources & Workflow",
    "classify_merged": "Plate Sources & Workflow",
    "map_barcodes": "Sequencing Input",
    "analyze_plaques": "Input & Channels",
    "replication": "Assay Inputs",
    "recruitment": "Data source",
    "umap": "Input Data",
    "invasion": "Assay Inputs",
    'host_pathogen': 'Assay Inputs',
    "ops": "OPS input",
}


def _window_of(screen):
    """The window ``screen`` sits in, read before a load that may retire it.

    A screen a rebuild replaced has no parent left, so its window can only be
    found while it is still mounted.
    """
    window = getattr(screen, "window", None)
    try:
        return window() if callable(window) else None
    except RuntimeError:
        return None


def _screen_after_the_load(screen, window):
    """The screen holding ``screen``'s form once example settings are in.

    ITEM 514. Applying a shipped settings pack that changes the form's shape
    has the window build a new screen and destroy this one
    (:meth:`AppScreen.apply_settings_dict`). Every Load test data route then
    wrote ``src`` into the field it had looked up BEFORE the load, which
    belonged to the retired screen, so the screen on view kept the pack's
    ``<src>`` and its Live preview was never told. A second press rebuilt
    nothing, because the form already had the pack's shape, which is why it
    worked.

    :param screen: the screen the button was pressed on.
    :param window: its window, from :func:`_window_of` before the load.
    :returns: the screen that replaced it, or ``screen`` itself.
    """
    screens = getattr(window, "_screens", None) if window is not None else None
    key = getattr(screen, "app_key", None)
    fresh = screens.get(key) if isinstance(screens, dict) else None
    return fresh if fresh is not None else screen


def _live_is_on(screen) -> bool:
    """Whether ``screen``'s Live switch is on, without building anything."""
    switch = getattr(screen, "__dict__", {}).get("_preview_switch")
    try:
        return bool(switch is not None and switch.isChecked())
    except RuntimeError:
        return False


def _show_the_src_live(screen, *, live_was_on: bool = False) -> None:
    """Have ``screen``'s Live preview show what ``src`` holds, now.

    Called once ``src`` holds the test data. A preview that is on loads it at
    once rather than after the typing debounce; one that was on before a
    rebuild is switched on again on the new screen; one that is off keeps the
    source for when it is opened.

    :param screen: the screen whose ``src`` was just written.
    :param live_was_on: whether Live was on when the button was pressed.
    """
    state = getattr(screen, "__dict__", {})
    timer = state.get("_live_src_timer")
    if timer is not None:
        try:
            timer.stop()
        except RuntimeError:
            pass
    switch = state.get("_preview_switch")
    if switch is None:
        return
    source = screen._settings_src_path() or ""
    live_card = getattr(screen, "_preview_card_attr", "") == "_live_preview_card"
    was_primed = bool(getattr(screen, "_preview_primed", False))
    if live_was_on and not switch.isChecked():
        switch.setChecked(True)
    if not switch.isChecked():
        if live_card:
            screen._autoload_live_preview(source)
        else:
            screen._preview_primed = False
        return
    if not was_primed and getattr(screen, "_preview_primed", False):
        return
    if live_card:
        screen._autoload_live_preview(source)
    else:
        screen._prime_preview()



def _each_fractal_backdrop(screen):
    """Yield every fractal backdrop reachable from ``screen``.

    Walks the window rather than holding a reference: the backdrop belongs
    to whichever screen is showing, and a reference captured once would go
    stale the first time the user changed module.
    """
    window = screen.window() if hasattr(screen, "window") else None
    if window is None:
        return
    for child in window.findChildren(QWidget):
        if hasattr(child, "pause") and hasattr(child, "backend_name"):
            yield child


def _pause_the_fractal(screen) -> int:
    """Stop the spaceout fractal for the duration of a run.

    :returns: how many backdrops were paused, so a test can assert on a
        number rather than on a screenshot.

    Never raises: a run must not fail because a decoration would not stop.
    """
    paused = 0
    try:
        for backdrop in _each_fractal_backdrop(screen):
            try:
                if backdrop.pause():
                    paused += 1
            except Exception:                                # noqa: BLE001
                LOG.debug("could not pause the fractal", exc_info=True)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not reach the fractal", exc_info=True)
    return paused


def _resume_the_fractal(screen) -> int:
    """Start it again once the run is over. Returns how many resumed."""
    resumed = 0
    try:
        for backdrop in _each_fractal_backdrop(screen):
            try:
                if backdrop.resume():
                    resumed += 1
            except Exception:                                # noqa: BLE001
                LOG.debug("could not resume the fractal", exc_info=True)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not reach the fractal", exc_info=True)
    return resumed


class _WrappingButtonStrip(FlowLayout):
    """The action row's buttons, laid out so they wrap rather than squeeze.

    WHY THIS EXISTS, WITH THE NUMBERS. The action buttons want 651 px of
    caption in English, 836 in German and 741 in Icelandic,
    and with the progress bar and the switches beside them a single
    ``QHBoxLayout`` made the whole row's MINIMUM width 908 / 1092 / 1068 px
    in those three locales. A minimum that large does damage two ways, and
    both were measured on Measure with the layout settled:

    * AT 1200x850 IT TOOK THE WIDTH FROM ITS NEIGHBOUR. The screen asks its
      body splitter for ``setSizes([400, 800])``; what it got instead was a
      settings column of 251 px in English, 91 in Icelandic and 67 in
      German, because the runtime side could not be narrower than the row.
      67 px of settings column is not a settings column.
    * AND IN A WINDOW THAT CANNOT GROW, THE CAPTIONS WENT. A page in a
      ``QStackedWidget`` gets whatever the window has, its own minimum
      included, and Qt's answer to a box it cannot satisfy is to shrink
      every child below its hint. At 1000x850 that cut four captions in
      German ("Einstellungen importieren…" 106 px against a 192 hint), four
      in Icelandic, and one in ENGLISH. Asked to be 1000 px wide as a
      top-level window the screen simply refused, and came up 1193 px in
      German -- which is the same defect wearing the other face.

    English very nearly fits, which is the only reason this shipped: the
    developers' locale is the one where the row is 908 and not 1092.

    :class:`~spacr.qt.widgets.flow.FlowLayout` already wraps, and it is used
    here AS A SUB-LAYOUT of the row rather than inside a ``FlowHost`` widget
    of its own. Both of those choices were forced, and both are worth reading
    twice before anyone "simplifies" this:

    * THE ROW IS NOT A ROW OF BUTTONS. It is eight buttons and an activity
      spinner, then ``addStretch(1)``, then the progress bar and the 3D /
      Time / Live / GPU / sweep / hyperparameter / interactive / AI
      switches, and the stretch is load-bearing -- it is what holds the switches against the
      right edge while the buttons stay left. ``FlowLayout`` has no
      ``addStretch``, so the row itself cannot become one; an earlier attempt
      swapped the layout wholesale and died immediately on ``AttributeError:
      'FlowLayout' object has no attribute 'addStretch'``.
    * A WIDGET ADDED TO A SUB-LAYOUT IS PARENTED TO THE WIDGET THAT OWNS THE
      TOP-LEVEL ONE, so every button is still a Qt child of ``_actions_row``
      and nothing that reaches into this row had to move.
      ``tests/qt/test_chaining_gui.py`` asserts exactly that
      (``screen._btn_run.parent() is screen._actions_row``), six test files
      reach into the row by attribute, ``findChildren`` from the screen still
      finds every button, and ``_clear_page_surfaces`` still has ONE widget
      to tag -- a nested ``FlowHost`` would have broken the first and the
      third, and added a second anonymous container to keep transparent.

    THE ONE THING ``FlowLayout`` DOES NOT DO for this use is ask for a whole
    line, and that is what this subclass adds. Its ``sizeHint`` is its
    ``minimumSize`` -- the widest single chip -- which is right for a chip
    strip in a column that hands it all the width there is, and wrong inside
    a ``QHBoxLayout``, where a 192 px hint would let the stretch swallow
    everything else and wrap eight buttons onto five lines in a row with room
    for one. ``sizeHint`` below asks for the single line instead, so the box
    layout keeps the strip on one line while it can and wraps it only when it
    cannot -- which is the behaviour the row already had in English and now
    has in every locale.
    """

    def __init__(self, spacing: int) -> None:
        """Build the strip and remember its gap.

        :param spacing: the gap between buttons in px. KEPT SEPARATELY as
            well as passed down, because ``FlowLayout`` holds its gap
            privately and never calls ``setSpacing`` -- so ``spacing()``
            would answer the style's default, and :meth:`sizeHint` needs
            the real number.
        """
        super().__init__(None, spacing=spacing)
        self._gap = int(spacing)

    def sizeHint(self) -> QSize:                # noqa: N802 (Qt override)
        """Every button on ONE line: the shape the row prefers when it fits.

        Deliberately counts the gap after every item except the first even
        when that item is empty -- a hidden ``_btn_file_issue`` reports a
        zero-width hint through ``QWidgetItem`` while
        ``FlowLayout._do_layout`` still advances by the spacing. Matching
        that arithmetic exactly is what keeps "the width this hint asks for"
        and "the width one line actually needs" the same number, to the
        pixel; a hint one gap short would wrap the last button for nothing.
        """
        width = 0
        height = 0
        for index in range(self.count()):
            hint = self.itemAt(index).sizeHint()
            width += hint.width() + (self._gap if index else 0)
            height = max(height, hint.height())
        margins = self.contentsMargins()
        return QSize(width + margins.left() + margins.right(),
                     height + margins.top() + margins.bottom())


#: The strip's Animation word, as a link target.
#:
#: A PRIVATE SCHEME, not a URL: the strip has `setOpenExternalLinks(True)`
#: so its API link works, and anything that looks like a real address would
#: be handed to a browser.
_HINT_ANIMATION_HREF = "spacr:animation"


def _setting_has_an_animation(key: str) -> bool:
    """Whether this setting has an animation to offer. Never raises.

    A missing or unreadable catalogue means "no animation", not a broken
    screen: the strip is help, and help that can take the form down is
    worse than help that is absent.
    """
    if not key:
        return False
    try:
        from ..setting_animations import animation_for_setting

        return animation_for_setting(str(key)) is not None
    except Exception:                                        # noqa: BLE001
        return False


#: The regression results panel and everything built with it: the Runs,
#: Results, Measurements and Cells tabs, the figure grid and its pages.
_REGRESSION_RESULTS = "regression results"
#: The hyperparameter search panel inside its card.
_HYPERPARAM_PANEL = "hyperparameter panel"
#: Mask's Cellpose live preview panel inside its card.
_LIVE_PREVIEW = "live preview"
#: Measure's crop preview panel inside its card.
_MEASURE_PREVIEW = "measure preview"
_UMAP_EXPLORER = "UMAP explorer"


def _run_to_the_end(steps):
    """Run a generator of build steps to the end and return what it returns.

    The settings form's builders are generators so that the idle prebuild
    (:class:`_IdlePrebuild`) can run them a step at a time; everything else
    runs them through here, in one go, which is the same code doing the same
    work in the same order.
    """
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


class _IdlePrebuild(QObject):
    """Builds a screen's waiting settings categories while nobody is using it.

    A category closed at open is built when it is first opened
    (:meth:`AppScreen._build_a_waiting_heading`), and that first open costs
    its build on the click. This builds them beforehand, in the gaps: once
    the screen has been on show with no input for :attr:`IDLE_MS`, it runs
    the same steps a click runs (:meth:`AppScreen._run_a_step_of`), for about
    :attr:`SLICE_MS` at a time, then gives the event loop back.

    INPUT STOPS IT. While slices are running it watches the application's
    events, and any mouse, key, wheel or touch event pushes the next slice
    to :attr:`IDLE_MS` after that input, so a user never waits on more than
    the one slice already running. It stops when the screen is hidden and
    starts again when it is shown.

    THE WATCH IS OFF WHILE IT WAITS FOR POINTER INPUT TO STOP. An
    application-wide event filter is a Python call per event, and opening a
    module delivers tens of thousands of them: installed for the whole wait,
    it cost every module open about 10 ms in the benchmark. So after pointer
    input, and at the start, the wait ends by comparing where the pointer
    is now with where it was; after a key the watch stays on, so typing
    keeps the build waiting key after key. A key typed with the pointer
    still after a pointer wait is caught one slice late, by the watch that
    slice installs.

    :param screen: the :class:`AppScreen` whose categories it builds.
    """

    #: No input for this long, in ms, before a slice runs.
    IDLE_MS = 400

    #: How long, in ms, a slice keeps taking steps. A step is one control,
    #: one row, or one pass, so a slice ends within one step of this; a
    #: step that took half of it ends the slice on its own, so a costly step
    #: is not run on top of several cheap ones.
    SLICE_MS = 4.0

    _POINTER = frozenset({
        QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease,
        QEvent.Type.MouseButtonDblClick, QEvent.Type.MouseMove,
        QEvent.Type.Wheel, QEvent.Type.TouchBegin, QEvent.Type.TouchUpdate,
        QEvent.Type.TabletPress, QEvent.Type.TabletMove,
    })
    _KEYS = frozenset({QEvent.Type.KeyPress, QEvent.Type.KeyRelease,
                       QEvent.Type.ShortcutOverride})

    def __init__(self, screen) -> None:
        """Prepare, without starting; see :meth:`resume`."""
        super().__init__(screen)
        self._screen = screen
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._slice)
        self._watching = False
        self._pointer = None
        #: Wall time, in ms, of each slice run so far; read by the tests
        #: and the measurements, never by the application.
        self.slices_ms: list = []

    def _work_left(self) -> list:
        """The headings still to build, in the order the form shows them."""
        screen = self._screen
        try:
            return [section for section
                    in screen.rendered_settings_sections()
                    if screen._heading_is_waiting(section)]
        except RuntimeError:
            return []

    @staticmethod
    def _pointer_now():
        """Where the pointer is, and which buttons are down."""
        from PySide6.QtGui import QCursor

        return (QCursor.pos(), int(QApplication.mouseButtons().value))

    def _wait(self, ms: int, *, watch: bool = False) -> None:
        """Note the pointer and try again in ``ms``.

        :param watch: keep watching events while waiting. After a key, so
            that typing keeps the build waiting key after key; after
            pointer input the watch comes off, because a click is how a user
            opens the next module and the next module's build is where an
            event filter costs the most.
        """
        if not watch:
            self._unwatch()
        self._pointer = self._pointer_now()
        self._timer.start(ms)

    def _unwatch(self) -> None:
        """Remove the application event filter once when activity observation ends."""
        if self._watching:
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
            self._watching = False

    def resume(self) -> None:
        """Start, or restart after a hide, if there is anything to build."""
        if not self._work_left():
            self.stop()
            return
        self._wait(self.IDLE_MS)

    def stop(self) -> None:
        """Stop building and stop watching input."""
        self._timer.stop()
        self._unwatch()

    def eventFilter(self, watched, event):                   # noqa: N802
        """Input: stop, and wait :attr:`IDLE_MS` after it."""
        kind = event.type()
        if kind in self._POINTER:
            self._wait(self.IDLE_MS)
        elif kind in self._KEYS:
            self._wait(self.IDLE_MS, watch=True)
        return False

    def _slice(self) -> None:
        """Take steps for about :attr:`SLICE_MS`, then give the loop back."""
        import time

        screen = self._screen
        try:
            if not screen.isVisible():
                self.stop()
                return
        except RuntimeError:
            self.stop()
            return
        if not self._watching:
            if self._pointer_now() != self._pointer:
                self._wait(self.IDLE_MS)
                return
            app = QApplication.instance()
            if app is None:
                return
            app.installEventFilter(self)
            self._watching = True
        work = self._work_left()
        if not work:
            self.stop()
            return
        section = work[0]
        started = time.perf_counter()
        more = True
        while more:
            before = time.perf_counter()
            more = screen._run_a_step_of(section)
            after = time.perf_counter()
            if ((after - started) * 1000.0 >= self.SLICE_MS
                    or (after - before) * 1000.0 >= self.SLICE_MS / 2):
                break
        self.slices_ms.append((time.perf_counter() - started) * 1000.0)
        if not more and not self._work_left():
            self.stop()
            return
        if self._watching:
            self._timer.start(0)


class _BuiltOnFirstUse:
    """A screen attribute whose widgets are built the first time it is used.

    Opening a module polishes every widget on its screen
    against the whole stylesheet, so a panel nobody can see yet still costs
    its share of the stall. The largest such panels -- Regression's results
    tabs, the hyperparameter search -- sit in a card that starts hidden,
    and their attributes are read from dozens of places, most of them
    long after the screen opened.

    This descriptor keeps every one of those readers working unchanged.
    While the screen still owes the named PART, reading or assigning any of
    its attributes builds the part first, so the caller sees exactly what an
    eagerly-built screen would have shown it: the widget, ``None`` where the
    eager build fell back to ``None``, or ``AttributeError`` where the eager
    build never assigned the name.

    THE VALUE IS NOT KEPT UNDER THE ATTRIBUTE'S OWN NAME. Shiboken's
    attribute lookup reads a wrapper's instance dictionary BEFORE the
    class's data descriptors -- the reverse of plain Python -- so a value
    stored under ``_results_panel`` would answer every later read directly
    and this descriptor would never be asked again. It is kept under
    :attr:`slot` instead.

    :param part: which deferred part assigns this attribute.
    """

    def __init__(self, part: str) -> None:
        """Remember which part assigns the attribute this descriptor names."""
        self.part = part
        self.name = ""
        self.slot = ""

    def __set_name__(self, owner, name: str) -> None:
        """Learn the attribute name from the class body."""
        self.name = name
        self.slot = f"_built_on_first_use{name}"

    def __get__(self, screen, owner=None):
        """Build the owed part, then answer as a plain attribute would."""
        if screen is None:
            return self
        values = screen.__dict__
        owed = values.get("_parts_owed")
        if owed and self.part in owed:
            screen._build_owed_part(self.part)
        try:
            return values[self.slot]
        except KeyError:
            raise AttributeError(self.name) from None

    def __set__(self, screen, value) -> None:
        """Build the owed part first, so an assignment lands after it."""
        owed = screen.__dict__.get("_parts_owed")
        if owed and self.part in owed:
            screen._build_owed_part(self.part)
        screen.__dict__[self.slot] = value

    def __delete__(self, screen) -> None:
        """Forget the attribute, as ``del`` on a plain one would."""
        try:
            del screen.__dict__[self.slot]
        except KeyError:
            raise AttributeError(self.name) from None

    def peek(self, screen):
        """The value held now, WITHOUT building an owed part; else ``None``."""
        return screen.__dict__.get(self.slot)


class AppScreen(QWidget):
    """Generic settings + runtime screen used by every non-interactive app.

    Composes the settings model on the left with the console, usage bars,
    figures card, and actions row on the right.

    :param app_key: id of the app (see ``APPS`` in ``spacr.qt.app``).
    :param parent: parent widget; ownership only.
    :ivar error_explain_requested: emitted with ``(traceback, app_key)``
        when the user clicks "Explain error"; MainWindow routes it to
        the AI Console for backward compatibility.
    """

    error_explain_requested = Signal(str, str)
    remote_submit_requested = Signal(str, dict)

    _ambient = None
    _ambient_applied = None
    _backdrop_applied = None
    _backdrops_ready = False
    _dna_rain = None

    _results_panel = _BuiltOnFirstUse(_REGRESSION_RESULTS)
    _figure_grid = _BuiltOnFirstUse(_REGRESSION_RESULTS)
    _figure_detail = _BuiltOnFirstUse(_REGRESSION_RESULTS)
    _volcano_page = _BuiltOnFirstUse(_REGRESSION_RESULTS)
    _gene_split = _BuiltOnFirstUse(_REGRESSION_RESULTS)
    _figure_size = _BuiltOnFirstUse(_REGRESSION_RESULTS)
    _figures_stack = _BuiltOnFirstUse(_REGRESSION_RESULTS)
    _sweep_runs = _BuiltOnFirstUse(_REGRESSION_RESULTS)
    _results_split = _BuiltOnFirstUse(_REGRESSION_RESULTS)
    _results_page = _BuiltOnFirstUse(_REGRESSION_RESULTS)
    _scan_panel = _BuiltOnFirstUse(_REGRESSION_RESULTS)
    _column_run_handles = _BuiltOnFirstUse(_REGRESSION_RESULTS)
    _sweep_panel = _BuiltOnFirstUse(_REGRESSION_RESULTS)
    _cell_montage = _BuiltOnFirstUse(_REGRESSION_RESULTS)
    _results_tabs = _BuiltOnFirstUse(_REGRESSION_RESULTS)
    _figures_split = _BuiltOnFirstUse(_REGRESSION_RESULTS)
    _grid_refresh = _BuiltOnFirstUse(_REGRESSION_RESULTS)
    _hyperparam = _BuiltOnFirstUse(_HYPERPARAM_PANEL)
    _live_preview = _BuiltOnFirstUse(_LIVE_PREVIEW)
    _measure_preview = _BuiltOnFirstUse(_MEASURE_PREVIEW)
    _umap_explorer = _BuiltOnFirstUse(_UMAP_EXPLORER)

    def __init__(self, app_key: str, parent=None):
        """Build one module page: the settings column beside the runtime panel.

        Ordering matters throughout and is the reason for the length. The live
        preview watches ``src`` and can only be wired once both panels exist,
        because the settings panel owns the field and the runtime panel owns the
        preview. The category hints have the same constraint in reverse. The
        page-surface sweep runs on every route, not only where a backdrop was
        installed: with the ambient preference off, skipping it left every
        layout container carrying the blanket window fill, which is what made
        the settings half a solid slab.

        :param app_key: which module this page is; it selects the title, the
            blurb, the settings schema, the drop handler and the backdrop.
        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        self.app_key = app_key
        self._ambient = None
        self._ambient_applied = None
        self._backdrop_applied = None
        self._backdrops_ready = False
        self._dna_rain = None
        self._last_error_text: str = ""
        self._hint_map: dict = _CaptionsBuiltWhenTheyAreAskedFor(
            self._caption_every_waiting_row)
        self._html_tip_map: dict = {}
        self._model_explainer = None
        self._heartbeat = None
        self._heartbeat_said = 0.0
        self._slow_fit = False
        #: {section title: its prose box}. Born here for the same reason
        #: everything else in this block is: a screen that has not built its
        #: settings pane yet still answers "which sections carry a box".
        self._section_explainers = {}
        #: ``dimension -> is it on``. Born BEFORE the settings panel, which
        #: is what reads it: the panel's first `refresh_maturity_visibility`
        #: runs while the action row that carries the switches does not
        #: exist yet, so the state has to live on the screen rather than in
        #: the widgets. Off to begin with, which is the point -- a plate is
        #: flat and single-shot until its user says otherwise.
        self._dimension_on = {name: False for name, _label, _tip
                              in DIMENSION_TOGGLES}
        #: ``dimension -> the toggle in the action row``, once there is one.
        self._dimension_switches = {}

        ensure_widget_qss_applied(SETTINGS_PANEL_NAME)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                  SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        header = ModuleHeader(
            APP_TITLES.get(app_key, app_key.title()),
            description=APP_INTROS.get(app_key) or "",
            instruction=DEFAULT_INSTRUCTION,
            app_key=app_key,
        )
        self._header = header
        outer.addWidget(header)

        outer.addWidget(Divider())

        from ..widgets.collapsible_splitter import EDGE, CollapsibleSplitter

        body = CollapsibleSplitter(Qt.Horizontal,
                                   persist_key=f"{app_key}::body")
        self._body_splitter = body
        body.setChildrenCollapsible(False)

        self._settings_body = body
        from .. import screens as _screens_package

        self._categories_wait = bool(getattr(
            _screens_package, "_categories_wait_to_be_opened", False))
        self._settings_panel = self._build_settings_panel()
        body.add_pane(self._settings_panel, "Settings", mode=EDGE,
                      fold_key=f"{app_key}/Settings", stretch=1, extent=400)
        self.the_name_carries_the_help()
        self._form_shape_on_screen = self._form_shape()
        self._watch_the_settings_that_decide_the_form()
        body.add_pane(self._build_runtime_panel(), "Runtime", stretch=2,
                      extent=800)

        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 2)
        body.setSizes([400, 800])
        outer.addWidget(body, 1)
        self._shell_focus.target(body, "Settings")

        self._wire_live_preview_autoload()
        if self.app_key == "analyze_plaques":
            self._install_plaque_mode()

        self._wire_category_hints()

        self._watch_for_late_captions()

        self._usage_jobs = JobRunner(self, app_key=f"{self.app_key} usage",
                                    user_visible=False)
        self._jobs = JobRunner(self, app_key=f"{self.app_key} background")

        self._usage_generation = 0
        self._usage_timer = QTimer(self)
        self._usage_timer.setInterval(2000)
        self._usage_timer.timeout.connect(self._refresh_usage)

        self._thread: Optional[QThread] = None

        try:
            from ..dnd import install_dropzone
            from ..dnd_handlers import get_handler
            install_dropzone(self, get_handler(self.app_key), self)
        except Exception:
            pass

        if self.app_key in DNA_RAIN_APPS:
            try:
                from ..widgets.dna_rain import install_dna_rain
                self._clear_page_surfaces()
                self._sync_page_palette()
                self._dna_rain = install_dna_rain(
                    self, outer, backdrop=_theme_wallpaper())
            except Exception:
                self._dna_rain = None

        #: (theme, palette) last pushed at — or attempted on — the
        #: widget, so a tab switch that changed nothing neither restarts
        #: the animation nor retries an install that already failed.
        self._backdrops_ready = True
        if uses_ambient_background(self.app_key):
            self._install_ambient()

        # THE SWEEP IS UNCONDITIONAL (item 381). With no backdrop behind the
        # containers, `page_fill` gives the page its own colour, so a
        # transparent container shows the page and never the window's `bg`.
        if self._ambient is None:
            self._clear_page_surfaces()
        self._sync_page_palette()
        try:
            self.register_workspace()
        except Exception:                                       # noqa: BLE001
            LOG.debug("could not enrol the workspace sections", exc_info=True)
        try:
            from ..theme import take_the_scroll_arrows_off
            take_the_scroll_arrows_off(self)
        except Exception:                                       # noqa: BLE001
            LOG.debug("could not take the tab scroll arrows off",
                      exc_info=True)

    def _heavy_lock_is_free(self) -> bool:
        """Whether the heavy-import lock could be taken right now.

        Asked without blocking and released immediately: this is a peek,
        not a reservation. The backdrop's own constructor still takes the
        lock properly, so the guarantee that a GL context is never built
        while the preloader is bringing CUDA up is unchanged -- all this
        decides is whether to try at all on this event-loop turn.

        A tree with no lock to ask (the widget module is absent, or its
        import failed) answers yes, so a machine without the backdrop
        behaves exactly as it did before.
        """
        try:
            from ..widgets.ambient import _the_heavy_import_lock_is_free
        except Exception:                                    # noqa: BLE001
            return True
        return _the_heavy_import_lock_is_free()

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
        if not self._backdrops_ready or self._ambient is not None:
            return
        if not self._heavy_lock_is_free():
            from PySide6.QtCore import QTimer

            QTimer.singleShot(120, self._install_ambient)
            return
        widget = None
        try:
            from ..preferences import (get_ambient_enabled,
                                       get_ambient_palette,
                                       get_ambient_theme)
            if not get_ambient_enabled():
                return
            window = self.window()
            shared = getattr(window, "window_backdrop", None)
            if callable(shared) and shared() is not None:
                self._clear_page_surfaces()
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
            self._ambient = widget
            self._sync_page_palette()
        except Exception as error:
            self._ambient = None
            _discard_widget(widget)
            self._discard_orphan_ambient()
            try:
                from ..widgets.ambient import _the_backdrop_wants_a_retry
            except Exception:                                # noqa: BLE001
                return
            if _the_backdrop_wants_a_retry(error):
                self._ambient_applied = None
                from PySide6.QtCore import QTimer

                QTimer.singleShot(120, self._install_ambient)

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

        :param event: the state-change event; it is passed to the base
            class, and only its type is read -- ``ApplicationPaletteChange``
            and ``PaletteChange`` re-theme the backdrops.
        """
        super().changeEvent(event)
        if event.type() not in (QEvent.ApplicationPaletteChange,
                                QEvent.PaletteChange):
            return
        if not self._backdrops_ready:
            return
        self.refresh_ambient_background()
        self._retheme_backdrops()
        self._retheme_section_explainers()
        self._sync_page_palette()
        self.update()

    def _retheme_section_explainers(
        self,
        language: Optional[str] = None,
    ) -> None:
        """Re-render every prose box in the live theme and language."""
        from ..theme import active_palette
        from .settings_model import section_explainer_html

        boxes = getattr(self, "_section_explainers", {}) or {}
        if not boxes:
            return
        try:
            palette = active_palette()
        except Exception:                                        # noqa: BLE001
            return
        for title, box in boxes.items():
            if box is None:
                continue
            try:
                if box is getattr(self, "_model_explainer", None):
                    self._refresh_model_explainer(language)
                else:
                    box.setHtml(section_explainer_html(
                        self.app_key, title, palette=palette,
                        language=language))
            except (RuntimeError, AttributeError):
                LOG.debug("could not re-theme the %s box", title,
                          exc_info=True)

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
        if (self._ambient is not None or self._dna_rain is not None
                or getattr(self, "_uses_window_backdrop", False)):
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
        if getattr(self, "_syncing_page", False):
            return
        colour = self.page_fill()
        applied = getattr(self, "_page_applied", "unset")
        wanted = None if colour is None else colour.name()
        if applied == wanted:
            return

        self._syncing_page = True
        try:
            if colour is None:
                self.setAutoFillBackground(False)
                self.setPalette(QPalette())
                set_a_sheeted_widgets_own_rule(self, "")
            else:
                palette = QPalette(self.palette())
                palette.setColor(QPalette.Window, colour)
                self.setPalette(palette)
                self.setAutoFillBackground(True)
                set_a_sheeted_widgets_own_rule(
                    self, f"AppScreen {{ background-color: {colour.name()}; }}")
            self._page_applied = wanted
        finally:
            self._syncing_page = False

    def paintEvent(self, event) -> None:
        """Paint the page under everything this screen lays out.

        Deliberately does **not** chain to ``super()`` when it fills. The
        base implementation is what draws the stylesheet background, and
        the stylesheet background is the ``bg`` slab being replaced —
        calling it afterwards would paint black straight back over this.

        :param event: the paint event; handed to the base class only when
            there is no page fill. The fill always covers the whole screen,
            not just the event's region.
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

        clear_container_surfaces(self)

        make_transparent(
            getattr(self, "_header", None),
            getattr(self, "_body_splitter", None),
            getattr(self, "_settings_scroll", None),
            getattr(self, "_settings_content", None),
            getattr(self, "_runtime_wrap", None),
            getattr(self, "_console_wrap", None),
            getattr(self, "_actions_row", None),
            getattr(self, "_actions_section", None),
            getattr(self, "_actions_body", None),
            getattr(self, "_category_hint", None),
        )

    def _build_settings_panel(self) -> QWidget:
        """Build the settings column, with the UI language resolved once.

        Everything the layout below asks for is rendered in the UI
        language -- every tooltip body, type hint, label and
        documentation URL, several times per row -- and each of those
        asks reads the preference store. `language_resolved_once` makes
        the panel ask once instead. It is re-entrant, so `build_sections`
        opening its own scope inside this one shares this cache rather
        than starting a second.

        :returns: the scroll area holding the settings form.
        """
        from .settings_model import language_resolved_once

        with language_resolved_once():
            return self._lay_out_the_settings_panel()

    def _lay_out_the_settings_panel(self) -> QWidget:
        """Build the settings column itself. See `_build_settings_panel`."""
        scroll = QScrollArea()
        self._settings_scroll = scroll
        scroll.setObjectName(SETTINGS_PANEL_NAME)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        if self.app_key == "umap":
            scroll.setMinimumWidth(280)
        scroll.viewport().setAutoFillBackground(False)

        content = QWidget()
        self._settings_content = content
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, SPACING["sm"], 0)
        layout.setSpacing(SPACING["sm"])

        self._settings_model = SettingsWidgets(
            self.app_key, parent=content,
            current=AppScreen.values_the_next_screen_is_built_for)
        self._rows_awaiting_layout = {}
        self._run_has_no_object_for = None
        #: Headings that gained a caption since the last pass, so the language
        #: pass reaches them and nothing else.
        self._captioned_late = set()
        self._settings_model.rows_are_laid_out_by = \
            self._lay_out_the_rows_that_are_back
        self._settings_model.rows_are_filtered_by = \
            self._refilter_the_settings_search
        self._settings_model.rows_the_screen_hides = \
            self._rows_the_filters_hide
        #: ``key -> heading`` for each setting whose category waits to be
        #: opened; see :meth:`_build_a_waiting_heading`.
        self._waiting_heading_of = {}
        if (getattr(self, "_categories_wait", False)
                and str(self.app_key) not in self.SETTINGS_AS_TABS):
            self._settings_model.categories_may_wait = \
                self._a_category_may_wait
        try:
            sections = self._settings_model.build_sections()
        except Exception as e:
            err = QLabel(f"Failed to build settings for '{self.app_key}': {e}")
            err.setWordWrap(True)
            layout.addWidget(err)
            scroll.setWidget(content)
            return scroll

        self._empty_state_card = self._build_empty_state_banner()
        if self._empty_state_card is not None:
            layout.addWidget(self._empty_state_card)

        self._settings_sections = []
        self._discarded_settings_host = QWidget(content)
        self._discarded_settings_host.setObjectName(
            "DiscardedSettingsSections")
        self._discarded_settings_host.hide()
        self._discarded_settings_sections = []
        self._category_blurbs = {}
        self._settings_tabs = None
        if str(self.app_key) in self.SETTINGS_AS_TABS and len(sections) > 2:
            self._settings_tabs = QTabWidget()
            self._settings_tabs.setObjectName("SettingsCategoryTabs")
            self._settings_tabs.setDocumentMode(True)
            layout.addWidget(self._settings_tabs)
        self._maturity_notice = QLabel()
        self._maturity_notice.setObjectName("MaturityVisibilityNotice")
        self._maturity_notice.setWordWrap(True)
        self._maturity_notice.hide()
        layout.addWidget(self._maturity_notice)

        if not sections:
            layout.addWidget(QLabel("No settings defined for this app."))
        for section_order, spec in enumerate(sections):
            if self._spec_waits(spec) and self._spec_holds_anything(spec):
                section = self._build_a_waiting_heading(spec)
                section._settings_top_level_order = section_order
                layout.addWidget(section)
                continue
            if self._spec_waits(spec):
                spec = self._with_the_controls(spec)
            section = self._build_settings_section(spec)
            section._settings_top_level_order = section_order
            if not self._section_holds_anything(section):
                self._discard_settings_section(section)
                continue
            if self._settings_tabs is not None:
                page = QWidget()
                page_layout = QVBoxLayout(page)
                page_layout.setContentsMargins(0, 0, 0, 0)
                page_layout.addWidget(section)
                page_layout.addStretch(1)
                holder = QScrollArea()
                holder.setObjectName(SETTINGS_TAB_PAGE_NAME)
                holder.setWidgetResizable(True)
                holder.setFrameShape(QScrollArea.NoFrame)
                holder.viewport().setAutoFillBackground(False)
                holder.setWidget(page)
                self._settings_tabs.addTab(
                    holder, str(section.property("settingsCategorySource")))
                section.set_expanded(True)
            else:
                layout.addWidget(section)

        self._settings_layout = layout
        self._mount_the_object_grid(layout)
        self.refresh_maturity_visibility()

        model = getattr(self, "_settings_model", None)
        if model is not None:
            try:
                model.refresh_object_visibility()
            except Exception:                                # noqa: BLE001
                LOG.debug("could not decide the object rows at build",
                          exc_info=True)

        layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

    #: How many shared questions a module needs before the per-object table
    #: is offered at all. Below this the flat form is the smaller control;
    #: see :meth:`_mount_the_object_grid`.
    MIN_GRID_QUESTIONS = 3

    def _mount_the_object_grid(self, layout) -> None:
        """Show the per-object settings as one table, if preferences ask.

        78 of Mask's 201 settings are the same twenty-odd questions asked once
        per object type. This puts them in a grid -- one row per question, one
        column per object -- and hides the flat rows they came from.

        OFF UNLESS CHOSEN. This is the most-used screen in the application, so
        the grid arrives as an offer rather than as a change to what everyone
        already knows: `get_object_grid_enabled` is False by default and this
        method returns before touching anything.

        NOTHING DOWNSTREAM LEARNS THE GRID EXISTS. It writes through to the
        same widgets the flat rows do, so `collect()` is unchanged and a
        settings file written with this on is the same file written with it
        off. The flat rows are HIDDEN, not dropped, so the settings search
        still indexes them and every check that walks the form still finds
        them holding their values.

        A PROSE ROW, not a setting row: `Section.add_row` records the pair in
        `_row_widgets`, and the module smoke test takes every entry there to
        BE a labelled setting with its own `settingKey` and API help. A grid
        is neither, and `add_prose_row` is the seam that exists for exactly
        that distinction.
        """
        try:
            from ..preferences import get_object_grid_enabled

            if not get_object_grid_enabled():
                return
            model = getattr(self, "_settings_model", None)
            if model is None or not getattr(model, "_widgets", None):
                return

            from ..widgets.object_grid_binding import ObjectGridBinding
            from ..widgets.object_settings_grid import ObjectSettingsGrid
            from ..widgets.section import Section

            section = Section("Per-object settings", self)
            grid = ObjectSettingsGrid(section)
            grid.set_app_key(self.app_key)
            binding = ObjectGridBinding(grid, model, self)
            binding.seed()
            owned = binding.owned_keys()
            if len(grid.questions()) < self.MIN_GRID_QUESTIONS or not owned:
                section.deleteLater()
                return
            section.add_prose_row("", grid)
            self._object_grid = grid
            self._object_grid_binding = binding
            model.hide_the_rows_the_grid_speaks_for(owned)
            layout.insertWidget(self._index_before_the_stretch(layout),
                                section)
            self._settings_sections.append(section)
            section.set_expanded(True)
            # TAG WHAT WAS JUST MOUNTED (item 408). While the panel is being
            # built the screen's own sweep comes later and covers this, but a
            # Preferences save mounts it on a screen already on show, after
            # every sweep: the table's viewport then painted `QPalette.Base`,
            # opaque black, over the backdrop until the next show. Only this
            # subtree -- the whole-screen sweep re-polishes 201 settings.
            from ..theme import clear_container_surfaces

            clear_container_surfaces(section)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not mount the per-object grid", exc_info=True)

    def apply_object_grid_preference(self) -> bool:
        """Mount or unmount the per-object table to match preferences.

        WHAT THIS FIXES: the switch in Preferences was read once, while the
        settings panel was being built, so turning it on did nothing to a
        module already open and turning it off left the table on screen with
        the rows it speaks for still hidden. The preference is a view of the
        same settings either way -- nothing downstream knows the grid exists
        -- so there is no reason it should need the module reopened.

        IDEMPOTENT, and safe on a screen whose panel was never built: both
        directions check what is actually mounted rather than trusting a
        flag, so a repeated call is a no-op and the two states cannot drift
        apart.

        :returns: True if the screen changed, False if it was already right.
        """
        try:
            from ..preferences import get_object_grid_enabled

            wanted = get_object_grid_enabled()
        except Exception:                                    # noqa: BLE001
            return False
        grid = getattr(self, "_object_grid", None)
        try:
            mounted = grid is not None and grid.parent() is not None
        except RuntimeError:
            mounted = False
        if wanted == mounted:
            return False
        if wanted:
            layout = getattr(self, "_settings_layout", None)
            if layout is None:
                return False
            self._mount_the_object_grid(layout)
            return getattr(self, "_object_grid", None) is not grid
        self._unmount_the_object_grid()
        return True

    @staticmethod
    def _index_before_the_stretch(layout) -> int:
        """Where a widget goes to stay above ``layout``'s trailing spring.

        :param layout: the settings column's layout.
        :returns: the index of the trailing stretch, or the end of the
            layout when it has not been given one yet.
        """
        for index in range(layout.count() - 1, -1, -1):
            item = layout.itemAt(index)
            if item is not None and item.spacerItem() is not None:
                return index
        return layout.count()

    def _unmount_the_object_grid(self) -> None:
        """Take the table off the screen and give the flat rows back.

        THE ROWS COME BACK FIRST. They were hidden, not dropped, so all it
        takes is telling the model the grid speaks for nothing -- but if the
        section were deleted first and that call then raised, the settings it
        holds would be on no screen at all: not in a table, and not in a
        form. Neither the values nor `collect()` are touched either way.
        """
        model = getattr(self, "_settings_model", None)
        unhide = getattr(model, "hide_the_rows_the_grid_speaks_for", None)
        if callable(unhide):
            try:
                unhide(())
            except Exception:                                # noqa: BLE001
                LOG.debug("could not give the flat rows back", exc_info=True)
        grid = getattr(self, "_object_grid", None)
        section = None
        try:
            section = grid.parent() if grid is not None else None
            while section is not None and not hasattr(
                    section, "add_prose_row"):
                section = section.parent()
        except RuntimeError:
            section = None
        self._object_grid = None
        self._object_grid_binding = None
        if section is None:
            return
        try:
            self._settings_sections.remove(section)
        except (AttributeError, ValueError):
            pass
        section.setParent(None)
        section.deleteLater()

    def _widget_key_index(self) -> dict:
        """``id(widget) -> setting key`` for this panel's settings model.

        ``SettingsWidgets._widgets`` is keyed the other way round, and the
        panel needs the reverse: given the field it is about to lay out,
        which setting is it? Answering that by scanning is quadratic in the
        number of settings, and Mask has 1,538 of them.

        Rebuilt whenever the model is replaced or has gained settings.
        ``setdefault`` keeps the first key for a widget registered under two
        names, which is the answer the scan this replaced gave.

        The index is keyed by ``id`` and is therefore only ever as good as
        its stamp; :meth:`_key_of_field` is what consumers should call,
        because it checks the answer against the model before returning it.

        :returns: the index, empty when there is no model yet.
        """
        model = getattr(self, "_settings_model", None)
        widgets = getattr(model, "_widgets", None) or {}
        built = getattr(widgets, "built_items", None)
        pairs = built() if callable(built) else list(widgets.items())
        stamp = (id(model), len(pairs))
        if getattr(self, "_widget_key_stamp", None) != stamp:
            index: dict = {}
            for key, widget in pairs:
                index.setdefault(id(widget), key)
            self._widget_key_cache = index
            self._widget_key_stamp = stamp
        return self._widget_key_cache

    def _key_of_field(self, field) -> Optional[str]:
        """The setting ``field`` is, or ``None`` if the model does not own it.

        What the panel used to answer by scanning every entry of
        ``_widgets``. The index does it in one lookup, and the answer is
        CHECKED against the model before it is returned: an ``id`` is only
        unique among live objects, so an index built against a different
        ``_widgets`` -- or one holding a widget that has since been freed --
        could otherwise name the wrong setting. A disagreement rebuilds the
        index once and asks again, which is the same answer the scan gave
        and still not a scan.

        :param field: the widget laid out in a settings row.
        :returns: its setting key, or ``None``.
        """
        widgets = getattr(
            getattr(self, "_settings_model", None), "_widgets", None) or {}
        key = self._widget_key_index().get(id(field))
        if key is not None and widgets.get(key) is field:
            return key
        self._widget_key_stamp = None
        key = self._widget_key_index().get(id(field))
        return key if key is not None and widgets.get(key) is field else None

    @staticmethod
    def _section_holds_anything(section) -> bool:
        """Whether ``section`` has a row, a child heading, or prose.

        :param section: a built :class:`~spacr.qt.widgets.section.Section`.
        :returns: ``False`` when it would render as a heading over nothing.

        A NESTED HEADING COUNTS ONLY IF IT HOLDS SOMETHING ITSELF, or an
        umbrella over three empty sub-headings would survive as four empty
        headings instead of none.
        """
        from ..widgets.section import Section, _sections_below

        def already_built_rows(owner):
            """Iterate registered rows without forcing deferred captions."""
            rows = getattr(owner, "_row_widgets", None)
            if rows is None:
                return iter(())
            if isinstance(rows, _RowsBuiltWhenTheyAreAskedFor):
                return list.__iter__(rows)
            return iter(rows)

        def holds_an_active_slot(owner) -> bool:
            """Whether deferred rows belong to a count-requested organelle."""
            from ...organelle_types import organelle_role_of

            return any(
                key is not None and organelle_role_of(key) is not None
                for key, _label, _widget in
                (getattr(owner, "_spacr_declared_rows", None) or ())
            )

        if any(widget is not None for _label, widget
               in already_built_rows(section)):
            return True
        if holds_an_active_slot(section):
            return True
        for child in _sections_below(section):
            if not isinstance(child, Section):
                continue
            if any(widget is not None for _label, widget
                   in already_built_rows(child)):
                return True
            if holds_an_active_slot(child):
                return True
        return False

    def _detach_what_the_form_hides(self) -> int:
        """Take every category body nobody can see out of the page.

        Called by the window just before the page is first shown, which is
        when the stylesheet lands on it and every widget under the page is
        styled; see
        :meth:`spacr.qt.widgets.section.Section._detach_body_while_hidden`.
        By then the maturity preference, the dimension switches and the
        settings search's Essentials view have all decided which categories
        are on the form, so what leaves the page is exactly what the user
        first sees collapsed or hidden. Only top-level categories are
        detached; a sub-heading travels with its category.

        :returns: how many widgets left the page.
        """
        moved = 0
        for section in self.rendered_settings_sections():
            if getattr(section, "_settings_top_level_order", None) is None:
                continue
            detach = getattr(section, "_detach_body_while_hidden", None)
            if not callable(detach) or self._heading_is_waiting(section):
                continue
            try:
                section._body_came_back = self._a_category_body_came_back
                moved += int(detach())
            except RuntimeError:
                continue
        return moved

    def _a_category_body_came_back(self, section) -> None:
        """Give a body that was detached what the page had while it was away.

        The language pass and the move of field help onto captions run over
        the page, and a detached body is not on it; the language may have
        changed, or a caption may have been built, while it waited. Both are
        idempotent, so a body that missed nothing is left as it was.
        """
        try:
            from ..i18n import retranslate_widget_tree
            from .settings_model import retarget_field_tooltips

            retranslate_widget_tree(section._body)
            retarget_field_tooltips(self)
        except RuntimeError:
            pass
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not dress a category that came back",
                      exc_info=True)

    def rendered_settings_sections(self) -> tuple:
        """The section widgets actually mounted in the settings panel.

        ``_settings_sections`` deliberately also contains dormant forms for
        object-gated settings.  Visual consumers -- maturity and category
        hints in particular -- must use this subset instead.
        """
        return tuple(
            section for section in getattr(self, "_settings_sections", ())
            if not section.property("settingsSectionDiscarded")
        )

    def _discard_settings_section(self, section, restore_parent=None) -> None:
        """Keep a pruned section's form indexed without making its header live.

        Settings controls are also the model's value store, so deleting an
        empty section would leave ``SettingsWidgets._widgets`` pointing at
        deleted Qt objects.  Search and delayed captions likewise need its
        form and model row registration.  Park the whole tree under a hidden
        owned widget instead; visual consumers explicitly ignore it.
        """
        from ..widgets.section import Section, _sections_below

        section._settings_restore_parent = restore_parent
        for member in (section, *(child for child in _sections_below(section)
                                  if isinstance(child, Section))):
            member.setProperty("settingsSectionDiscarded", True)
            member.hide()
        if restore_parent is None:
            section.setParent(self._discarded_settings_host)
        section.hide()
        self._discarded_settings_sections.append(section)

    def _restore_settings_section(self, section) -> bool:
        """Mount a dormant heading whose object-gated rows now apply."""
        if not section.property("settingsSectionDiscarded"):
            return False

        parent = getattr(section, "_settings_restore_parent", None)
        if parent is not None:
            self._restore_settings_section(parent)
        else:
            layout = self._settings_content.layout()
            section.setParent(self._settings_content)
            order = getattr(section, "_settings_top_level_order", -1)
            before = [
                layout.indexOf(other)
                for other in self.rendered_settings_sections()
                if getattr(other, "_settings_restore_parent", None) is None
                and getattr(other, "_settings_top_level_order", -1) > order
                and layout.indexOf(other) >= 0
            ]
            if before:
                at = min(before)
            else:
                at = layout.count()
                if at and layout.itemAt(at - 1).spacerItem() is not None:
                    at -= 1
            layout.insertWidget(at, section)

        section.setProperty("settingsSectionDiscarded", False)
        self._discarded_settings_sections[:] = [
            dormant for dormant in self._discarded_settings_sections
            if dormant is not section
        ]
        return True

    #: What the screen ABOUT TO BE BUILT should be shaped for.
    #:
    #: A class attribute rather than something read off the window, because
    #: a screen is constructed BEFORE it is parented: inside
    #: `_build_settings_panel`, `self.window()` answers with the screen
    #: itself, so anything left on the MainWindow is invisible there. That
    #: is why the first attempt rebuilt the form to exactly the shape it
    #: already had.
    #:
    #: Set by `MainWindow.rebuild_app_screen` and cleared by it, so an
    #: ordinary module open -- which is every other caller -- builds from
    #: the module's own defaults as it always did.
    values_the_next_screen_is_built_for = None

    #: Settings whose value decides which OTHER settings exist on every
    #: applicable panel. Object switches are added from the widgets the panel
    #: actually owns by :meth:`_form_shaping_keys`: Mask uses ``*_channel``,
    #: Measure uses ``*_mask_dim``, and each active organelle slot has its own
    #: switch. Cell is deliberately excluded because its family is never
    #: gated.
    FORM_SHAPING_KEYS = ("number_of_organelles",)

    def _form_shaping_keys(self) -> tuple[str, ...]:
        """Return every committed value that shapes this panel's form.

        The list must follow the live widget inventory rather than a fixed
        trio of Mask keys. Measure owns mask-plane switches instead of image
        channels, and organelle switches appear only after the count creates
        their slots. A missing switch cannot shape this panel; the cell switch
        never shapes one because cell settings always remain available.
        """
        model = getattr(self, "_settings_model", None)
        widgets = getattr(model, "_widgets", {}) or {}
        keys = [key for key in self.FORM_SHAPING_KEYS if key in widgets]
        for key in widgets:
            role = object_of_setting(key)
            if role is None or role == "cell":
                continue
            if key in object_switch_keys(role):
                keys.append(key)
        return tuple(dict.fromkeys(keys))

    def _object_switches_on_this_form(self) -> tuple[str, ...]:
        """The committed values that only decide which objects are SHOWN.

        THE OTHER HALF OF :meth:`_form_shaping_keys`, and the distinction
        that matters: a channel number toggles the visibility of the
        categories it belongs to WITHOUT reloading the module. It does not
        add or remove a single row -- the panel already built
        a control for every object it can name -- so the rule that decides
        which of them are on screen is the whole of the work.

        `number_of_organelles` is deliberately NOT here. Its rows do not
        exist until it is raised, so they have to be spawned, which is the
        one case a rebuild is accepted for.
        """
        model = getattr(self, "_settings_model", None)
        widgets = getattr(model, "_widgets", {}) or {}
        keys = []
        for key in self._form_shaping_keys():
            if key in self.FORM_SHAPING_KEYS:
                continue
            if key in widgets:
                keys.append(key)
        return tuple(keys)

    def _show_the_objects_the_run_has(self, *_args) -> None:
        """Reveal or hide one object's settings, in place.

        WHAT THIS REPLACED, measured on Mask before it existed: committing a
        value into `nucleus_channel` called `rebuild_app_screen`, which took
        455 ms and put a DIFFERENT SCREEN OBJECT in the window's stack -- to
        change which rows are visible. Everything the user had not committed,
        every scroll position and every expanded fold went with it, and it is
        the same startup cost the screen works to avoid, being paid on a
        keystroke.

        `refresh_object_visibility` is what the rebuild was reaching for the
        long way round. It decides EVERY gated row and hides the headings of
        the slots the run lacks, so calling it is the whole of the change --
        and it lays out any row that was left unbuilt while it was hidden
        before it shows it.

        The remembered shape is refreshed too, so a later `number_of_
        organelles` change still compares like with like and rebuilds.
        """
        self._run_has_no_object_for = None
        model = getattr(self, "_settings_model", None)
        if model is None:
            return
        try:
            model.refresh_object_visibility()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not re-decide which objects the run has",
                      exc_info=True)
        try:
            self._form_shape_on_screen = self._form_shape()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not remember the form shape", exc_info=True)

    def _watch_the_settings_that_decide_the_form(self) -> None:
        """Rebuild the form when a value that shapes it is COMMITTED.

        NOT ON EVERY KEYSTROKE. That is what made the Mask module hang: the
        old rule ran a full pass over 1,551 widgets per character typed. A
        commit is one event per value the user actually settled on -- Enter,
        or leaving the field -- so typing "1" costs one rebuild rather than
        one per digit.

        The pair is deliberate. `editingFinished` is the commit for a field
        somebody types into; `valueChanged` is the commit for a spin box or
        a combo, where every change is already a decision. A field that has
        both fires once, because `_rebuild_the_form` compares the shape it
        would build against the one on screen and returns when they agree.

        The cell's channel is followed too, although it shapes nothing: cell
        rows are never hidden, but a cell channel brings Cell Segmentation
        into the Essentials view, so its commit runs the same in-place pass
        as the other object channels.
        """
        model = getattr(self, "_settings_model", None)
        if model is None:
            return
        switches = set(self._object_switches_on_this_form())
        watched = dict.fromkeys(self._form_shaping_keys())
        for key in object_switch_keys("cell"):
            if key in (getattr(model, "_widgets", {}) or {}):
                watched[key] = None
                switches.add(key)
        for key in watched:
            widget = getattr(model, "_widgets", {}).get(key)
            if widget is None:
                continue
            slot = (self._show_the_objects_the_run_has if key in switches
                    else self._rebuild_the_form)
            done = getattr(widget, "editingFinished", None)
            if done is not None:
                try:
                    done.connect(slot)
                    continue
                except Exception:                            # noqa: BLE001
                    pass
            for name in ("valueChanged", "currentIndexChanged"):
                signal = getattr(widget, name, None)
                if signal is None:
                    continue
                try:
                    signal.connect(slot)
                    break
                except Exception:                            # noqa: BLE001
                    continue
        self._watch_the_cellpose3_choosers(model)

    def _watch_the_cellpose3_choosers(self, model) -> None:
        """Show the Cellpose 3 rows as soon as a Cellpose 3 model is chosen.

        Item 503. The model zoo writes ``cellpose3:<model>`` into a model
        field with ``setText``, which is not a commit -- no
        ``editingFinished`` follows -- so the text itself is followed, and
        the visibility pass waits for the text to settle for a moment rather
        than running on every keystroke typed into the field.

        :param model: the screen's settings model.
        """
        from .settings_model import _cellpose3_choosers

        widgets = getattr(model, "_widgets", {}) or {}
        keys = _cellpose3_choosers(widgets)
        if not keys:
            return
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(250)
        timer.timeout.connect(self._show_the_objects_the_run_has)
        self._cellpose3_rows_timer = timer
        for key in keys:
            widget = widgets.get(key)
            for name in ("textChanged", "currentTextChanged", "valueChanged",
                         "currentIndexChanged"):
                signal = getattr(widget, name, None)
                if signal is None:
                    continue
                try:
                    signal.connect(lambda *_: timer.start())
                    break
                except Exception:                            # noqa: BLE001
                    continue

    def _form_shape(self) -> tuple:
        """What the form's shape currently depends on.

        :returns: the committed form-shaping key/value pairs.

        Compared before rebuilding so a signal that did not actually change
        the shape -- a second commit of the same value, or the two signals a
        spin box emits -- costs nothing.
        """
        model = getattr(self, "_settings_model", None)
        values = dict((model.collect() if model is not None else {}) or {})
        return tuple((key, str(values.get(key, "")).strip())
                     for key in self._form_shaping_keys())

    def _worker_thread_is_running(self) -> bool:
        """Whether this screen still owns a live pipeline QThread."""
        thread = getattr(self, "_thread", None)
        if thread is None:
            return False
        try:
            return bool(thread.isRunning())
        except (AttributeError, RuntimeError):
            return False

    def _rebuild_the_form(self, *_args) -> None:
        """Rebuild this screen for the shape the values now describe.

        THE WHOLE SCREEN, not the settings panel in place. Swapping the
        panel inside the splitter is the cheaper move and it is the one that
        did not work: the widget mounted there is not the scroll area the
        panel builder records, so the lookup answered -1 and the rebuild
        returned having done nothing at all. Asking the window to build the
        screen again is the path that already runs on every module open, and
        Mask opens in 1.45 s -- which is a fine price for a deliberate
        change to one value, and no price at all for the typing that used to
        trigger the old rule.

        WHAT THE USER HAS TYPED SURVIVES. Every current value is collected
        first and applied to the new screen, so a form that gains twenty
        nucleus settings does not lose the twelve already filled in.

        A RUN OWNS THIS SCREEN UNTIL ITS QTHREAD HAS STOPPED. A shaping edit
        made during a run is remembered and rebuilt afterwards; closing the
        screen here would request cancellation before ``closeEvent`` could
        decide to refuse the replacement.
        """
        if getattr(self, "_rebuilding_the_form", False):
            return
        model = getattr(self, "_settings_model", None)
        if getattr(model, "_applying_settings", False):
            return
        if self._worker_thread_is_running():
            deferred = dict(getattr(self, "_deferred_form_values", None) or {})
            try:
                deferred.update(dict((model.collect() if model else {}) or {}))
            except Exception:                               # noqa: BLE001
                pass
            self._deferred_form_values = deferred
            self._form_rebuild_deferred = True
            return
        deferred_values = getattr(self, "_deferred_form_values", None)
        if deferred_values is not None:
            merged = dict(deferred_values)
            try:
                merged.update(dict((model.collect() if model else {}) or {}))
            except Exception:                               # noqa: BLE001
                pass
            deferred_values = self._deferred_form_values = merged
        shape = self._form_shape()
        if (deferred_values is None
                and shape == getattr(self, "_form_shape_on_screen", None)):
            self._form_rebuild_deferred = False
            return
        window = self.window()
        rebuild = getattr(window, "rebuild_app_screen", None)
        if not callable(rebuild):
            self._form_rebuild_deferred = False
            return
        self._rebuilding_the_form = True
        try:
            keep = (dict(deferred_values) if deferred_values is not None
                    else dict((self._settings_model.collect() or {})))
            deferred_bulk = getattr(self, "_deferred_bulk_settings", None)
            rebuild(self.app_key, keep)
            fresh = getattr(window, "_screens", {}).get(self.app_key)
            if (deferred_bulk is not None and fresh is not None
                    and fresh is not self):
                fresh._refresh_after_bulk_apply(deferred_bulk)
            self._deferred_form_values = None
            self._deferred_bulk_settings = None
            self._form_rebuild_deferred = False
        except Exception:                                    # noqa: BLE001
            LOG.exception("could not rebuild the settings form")
        finally:
            self._rebuilding_the_form = False

    def _build_settings_section(self, spec, depth: int = 0, into=None):
        """Build one heading of the settings TREE, and everything under it.

        :meth:`_settings_section_steps` run to the end; see it for the
        parameters.

        :returns: the built :class:`Section` widget.
        """
        return _run_to_the_end(self._settings_section_steps(spec, depth, into))

    def _settings_section_steps(self, spec, depth: int = 0, into=None):
        """Build one heading of the settings TREE, and everything under it.

        ``SettingsWidgets.build_sections`` returns a
        :class:`~spacr.qt.screens.settings_model.SettingsSection`: still the
        ``(title, rows)`` pair it always was, with ``own_rows`` for the rows
        this heading owns itself and ``children`` for the headings nested
        below it. Reading only the pair draws every control exactly once but
        FLAT -- the umbrella renders as a single heading holding every
        advanced row, and the sub-headings that say which object a row
        belongs to are nowhere. This walks the tree instead: ``own_rows``
        here, a nested :class:`Section` per child, added with ``add_prose``
        because a heading is not a labelled setting row and
        ``tests/qt/test_all_module_smoke.py`` reads every ``_row_widgets``
        entry as one.

        Sections are recorded in ``_settings_sections`` DEEPEST FIRST, so
        everything that searches that list for the section holding a widget
        -- the command palette, the search strip -- finds the innermost
        heading rather than the umbrella two levels above it.

        :param spec: a ``SettingsSection`` or a plain ``(title, rows)`` pair.
        :param depth: 0 for a top-level category; deeper for a sub-heading.
        :param into: a heading built earlier by
            :meth:`_build_a_waiting_heading`, to lay the rows out in instead
            of a new one. It is already titled and recorded, so only its body
            is built.
        :returns: the built :class:`Section` widget.
        """
        title = getattr(spec, "title", None)
        if title is None:
            title = spec[0]
        title = str(title)
        own_rows = getattr(spec, "own_rows", None)
        rows = spec[1] if own_rows is None else own_rows
        children = tuple(getattr(spec, "children", ()) or ())
        if into is None:
            section = self._titled_heading(spec, title)
        else:
            section = into
        declared = tuple((self._key_of_field(widget), label, widget)
                         for label, widget in rows)
        no_object = self._keys_the_run_has_no_object_for()
        waiting = {key for key, _label, _widget in declared
                   if key is not None and key in no_object}
        section._spacr_declared_rows = declared
        try:
            self._settings_model.remember_section_rows(
                section,
                [key for key, _label, _widget in declared if key],
                bool(children))
        except AttributeError:
            pass
        if waiting:
            section._row_widgets = _RowsBuiltWhenTheyAreAskedFor(
                partial(self._lay_out_every_waiting_row, section))
        for key, label, widget in declared:
            if key in waiting:
                self._rows_awaiting_layout[key] = section
                section.add_prose(widget)
                yield
                continue
            self._lay_out_setting_row(section, label, widget)
            yield
        for child in children:
            nested = yield from self._settings_section_steps(child, depth + 1)
            if not self._section_holds_anything(nested):
                section.add_prose(nested)
                self._discard_settings_section(nested, section)
                continue
            section.add_prose(nested)
            nested.toggled.connect(
                partial(self._open_the_headings_above, section))
            yield
        yield
        from .settings_model import has_section_explainer

        if depth == 0 and has_section_explainer(self.app_key, title):
            self._install_section_explainer(section, title)
        if depth == 0 and title == EXAMPLE_DATA_SECTIONS.get(self.app_key):
            section.set_expanded(True)
            if self.app_key == "regression":
                self._install_example_data_button(section)
            elif self.app_key == "measure":
                self._install_measure_example_button(section)
            elif self.app_key in ("classify", "classify_merged"):
                self._install_annotate_example_button(section)
            elif self.app_key == "map_barcodes":
                self._install_sequencing_example_button(section)
            elif self.app_key == "analyze_plaques":
                self._install_plaque_example_button(section)
            elif self.app_key in ("replication", "recruitment", "invasion", 'host_pathogen'):
                from ..assay_examples import install_assay_example_button

                install_assay_example_button(self, section)
            elif self.app_key == "umap":
                self._install_measurements_example_button(section)
            elif self.app_key == "ops":
                self._install_ops_example_button(section)
            else:
                self._install_example_images_button(section)
        if into is None:
            self._settings_sections.append(section)
        return section

    def _titled_heading(self, spec, title: str):
        """A new, empty heading for ``spec``: title, maturity and blurb."""
        section = Section(title)
        section.setProperty("settingsCategorySource", title)
        section.set_maturity(
            settings_section_maturity(self.app_key, title)
        )
        blurb = section_tooltip(self.app_key, spec)
        section.set_hint(blurb)
        self._category_blurbs.setdefault(title, blurb)
        return section

    def _a_category_may_wait(self, title: str, keys=()) -> bool:
        """Whether top-level category ``title`` may wait to be opened.

        Not the one holding the module's example-data control, which
        :meth:`_build_settings_section` opens. Not one the settings search
        will open either: under the Essentials view -- where every module
        starts -- the strip opens each category holding an essential setting
        as it is installed, and a category built by being opened pays the
        passes an opening runs on top of what building it with the panel
        costs. The essentials are over-counted on purpose (every object's
        segmentation settings, whether or not its channel is set): a
        category wrongly kept is built as it always was, and one wrongly
        left waiting is built a moment later, so either mistake is safe.

        :param title: the category.
        :param keys: the settings laid out under it.
        """
        if str(title) == EXAMPLE_DATA_SECTIONS.get(self.app_key):
            return False
        return not (set(keys) & self._settings_the_essentials_view_opens())

    def _settings_the_essentials_view_opens(self) -> set:
        """The settings the Essentials view shows, if it is the view used."""
        found = getattr(self, "_essentials_it_opens", None)
        if found is not None:
            return found
        found = set()
        try:
            from ..settings_search import ESSENTIALS, disclosure_for
            from .settings_model import (
                _APP_ESSENTIALS_THAT_FOLLOW_THEIR_OBJECT,
                _expand_layout_tokens, categories_for_app, essential_keys,
                get_categories)

            if disclosure_for(self.app_key) == ESSENTIALS:
                cats = categories_for_app(self.app_key, get_categories())
                found.update(essential_keys(self.app_key, cats))
                found.update(_expand_layout_tokens(
                    cats, _APP_ESSENTIALS_THAT_FOLLOW_THEIR_OBJECT.get(
                        str(self.app_key), ())))
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not tell which settings Essentials shows",
                      exc_info=True)
        self._essentials_it_opens = found
        return found

    @staticmethod
    def _spec_waits(spec) -> bool:
        """Whether any row under ``spec`` is a control still to come."""
        from .settings_model import _ControlToCome

        return any(isinstance(widget, _ControlToCome)
                   for _label, widget in (spec[1] or ()))

    def _key_of_row(self, widget) -> Optional[str]:
        """The setting a spec row is for, whether or not it is built."""
        from .settings_model import _ControlToCome

        if isinstance(widget, _ControlToCome):
            return widget.key
        return self._key_of_field(widget)

    def _spec_holds_anything(self, spec) -> bool:
        """:meth:`_section_holds_anything`, answered from the keys alone.

        A heading holds something when a row under it is laid out -- every
        row whose object the run has -- or belongs to an organelle slot the
        count asked for, which is the same test the built heading is given.
        """
        from ...organelle_types import organelle_role_of

        lacking = self._keys_the_run_has_no_object_for()
        for _label, widget in spec[1] or ():
            key = self._key_of_row(widget)
            if key is None:
                continue
            if key not in lacking or organelle_role_of(key) is not None:
                return True
        return False

    def _with_the_controls(self, spec):
        """``spec`` with every control still to come built and in its row."""
        from .settings_model import SettingsSection, _ControlToCome

        model = self._settings_model
        widgets = model._widgets
        widgets.build([widget.key for _label, widget in spec[1] or ()
                       if isinstance(widget, _ControlToCome)])

        def swap(node):
            """``node`` rebuilt with its stand-ins replaced by controls."""
            own = getattr(node, "own_rows", None)
            own = node[1] if own is None else own
            rows = []
            for label, widget in own:
                if isinstance(widget, _ControlToCome):
                    widget = widgets.built(widget.key)
                    if widget is None:
                        continue
                rows.append((label, widget))
            children = [swap(child)
                        for child in getattr(node, "children", ()) or ()]
            title = getattr(node, "title", None)
            return SettingsSection(node[0] if title is None else title,
                                   rows, children)

        return swap(spec)

    def _build_a_waiting_heading(self, spec):
        """A category's heading, with its rows left until it is opened.

        WHAT WAITS. Every row of the category -- caption, field, help, and
        the sub-headings below it -- and every control in it that nothing
        has read. Measured on a real window, those were most of what a
        module built at open and nobody could see: 91 of Classify's 101
        settings, 101 of Mask's 132. The heading itself is built, titled,
        rated for maturity and recorded, so the form looks exactly as it
        does with the rows in place and closed.

        WHAT STILL ANSWERS FOR IT. The model knows every key, so ``in``,
        the search strip and the palette find its settings; a read of any of
        its controls builds that control (see
        :class:`~spacr.qt.screens.settings_model._ControlsBuiltWhenAskedFor`)
        so ``collect()``, recipes, imports and drops read and write real
        values. The object rule knows its keys through
        ``remember_section_rows``, the dimension switches through
        :meth:`_dimension_hidden_sections`.

        A control that was built anyway -- one nothing can read unbuilt, or
        one something read while the panel was being built -- is taken off
        the page until its row is laid out: it was made with the form as its
        parent, and a child of a shown widget that belongs to no row is drawn
        where it stands, at the form's top-left corner.

        WHAT OPENS IT: expanding it, revealing one of its settings, asking
        for its rows, or anything that needs a row on the form. See
        :meth:`_open_a_waiting_heading`.
        """
        title = str(getattr(spec, "title", None) or spec[0])
        children = tuple(getattr(spec, "children", ()) or ())
        section = self._titled_heading(spec, title)
        own = getattr(spec, "own_rows", None)
        own = spec[1] if own is None else own
        own_keys = [key for key in (self._key_of_row(widget)
                                    for _label, widget in own) if key]
        try:
            self._settings_model.remember_section_rows(
                section, own_keys, bool(children))
        except AttributeError:
            pass
        for _label, widget in spec[1] or ():
            if isinstance(widget, QWidget) and widget.parentWidget() is not None:
                widget.setParent(None)
        section._spacr_waiting_spec = spec
        section._spacr_declared_rows = ()
        opener = partial(self._open_a_waiting_heading, section)
        section._spacr_build_body = opener
        section._row_widgets = _RowsBuiltWhenTheyAreAskedFor(opener)
        for _label, widget in spec[1] or ():
            key = self._key_of_row(widget)
            if key:
                self._waiting_heading_of[key] = section
        self._settings_sections.append(section)
        return section

    def _heading_is_waiting(self, section) -> bool:
        """Whether ``section`` is a heading whose build is not finished.

        True from :meth:`_build_a_waiting_heading` until the last step of
        :meth:`_waiting_heading_steps`, including while an idle prebuild is
        part-way through it.
        """
        try:
            return (getattr(section, "_spacr_waiting_spec", None) is not None
                    or "_spacr_opening" in section.__dict__)
        except RuntimeError:
            return False

    def _open_the_heading_of(self, key: str) -> bool:
        """Build the waiting category ``key`` belongs to, if it waits.

        :returns: ``True`` when a category was built by this call.
        """
        section = (getattr(self, "_waiting_heading_of", None) or {}).get(
            str(key))
        if section is None:
            return False
        return self._open_a_waiting_heading(section)

    def _open_every_waiting_heading(self) -> int:
        """Build every category still waiting.

        :returns: how many were built.
        """
        opened = 0
        for section in list(getattr(self, "_settings_sections", ()) or ()):
            if self._heading_is_waiting(section):
                opened += int(self._open_a_waiting_heading(section))
        return opened

    def _open_a_waiting_heading(self, section) -> bool:
        """Build a waiting category's rows, as opening the screen would have.

        The rows are laid out by :meth:`_build_settings_section`, the path
        every category takes, into the heading already on the form. Then the
        category is given what the page gave every other one while it was
        opening, in the same order: its sub-headings are recorded ahead of
        it (deepest first, as the search strip and the palette expect), the
        category hints reach them, the surface sweep runs, the maturity and
        dimension switches and the object rule decide the new rows, the
        search strip indexes them, and the language pass and the move of
        help onto captions run last, after the rows are polished -- the
        order :meth:`_translate_a_late_part` explains.

        :returns: ``True`` when this call built the rows; ``False`` when the
            heading was built already.
        """
        steps = section.__dict__.get("_spacr_opening")
        if steps is None:
            if getattr(section, "_spacr_waiting_spec", None) is None:
                return False
            steps = section._spacr_opening = self._waiting_heading_steps(
                section)
        if section.__dict__.get("_spacr_opening_now"):
            return False
        from .. import timing
        from .settings_model import language_resolved_once

        title = str(section.property("settingsCategorySource") or "")
        section._spacr_opening_now = True
        try:
            with timing.span("build waiting category", title), \
                    language_resolved_once():
                _run_to_the_end(steps)
        finally:
            section._spacr_opening_now = False
        return True

    def _run_a_step_of(self, section) -> bool:
        """Run one step of building a waiting category.

        What :class:`_IdlePrebuild` calls. The steps are the ones
        :meth:`_open_a_waiting_heading` runs, from the same generator, so a
        category half built in idle time is finished by a click exactly as
        it would have been built by one.

        :returns: ``True`` while there is more to do; ``False`` once the
            category is built (or cannot be).
        """
        steps = section.__dict__.get("_spacr_opening")
        if steps is None:
            if getattr(section, "_spacr_waiting_spec", None) is None:
                return False
            steps = section._spacr_opening = self._waiting_heading_steps(
                section)
        if section.__dict__.get("_spacr_opening_now"):
            return True
        from .settings_model import language_resolved_once

        section._spacr_opening_now = True
        try:
            with language_resolved_once():
                next(steps)
        except StopIteration:
            return False
        except RuntimeError:
            LOG.debug("a waiting category went away mid-build", exc_info=True)
            section.__dict__.pop("_spacr_opening", None)
            return False
        finally:
            section._spacr_opening_now = False
        return True

    def _waiting_heading_steps(self, section):
        """The steps of building a waiting category, in the order they run.

        One control per step; the passes that grey controls from others once
        for the batch; one row, or one sub-heading, per step; then the
        passes the category needs once its rows exist, each its own step.
        Until the last step the heading counts as waiting
        (:meth:`_heading_is_waiting`) and opening it finishes the build first.
        """
        spec = section._spacr_waiting_spec
        model = self._settings_model
        widgets = model._widgets
        pending = [widget.key for _label, widget in spec[1] or ()
                   if self._is_control_to_come(widget)]
        arrived = []
        for key in pending:
            if key in widgets and not widgets.is_built(key):
                widgets.build((key,), decide=False)
                arrived.append(key)
                yield
        if arrived and model._decided_by_a_pass().intersection(arrived):
            yield from model._state_pass_steps()
        real = self._with_the_controls(spec)
        before = {id(other) for other in self._settings_sections}
        self._run_has_no_object_for = None
        try:
            yield from self._settings_section_steps(real, 0, into=section)
        finally:
            self._run_has_no_object_for = None
        section._spacr_waiting_spec = None
        rows = section.__dict__.get("_row_widgets")
        if isinstance(rows, _RowsBuiltWhenTheyAreAskedFor):
            rows._build_the_rest = None
        self._put_new_headings_ahead_of(section, before)
        self._wire_category_hints()
        yield
        self._clear_a_late_parts_surfaces(section._body)
        yield
        self.refresh_maturity_visibility()
        yield
        yield from self._rows_moved_steps(judge_them=True)
        yield from self._translate_a_late_part_steps(section._body)
        for _label, widget in spec[1] or ():
            key = self._key_of_row(widget)
            if key and self._waiting_heading_of.get(key) is section:
                del self._waiting_heading_of[key]
        section._spacr_build_body = None
        section.__dict__.pop("_spacr_opening", None)

    @staticmethod
    def _is_control_to_come(widget) -> bool:
        """Whether a spec row's widget is a stand-in for a waiting control."""
        from .settings_model import _ControlToCome

        return isinstance(widget, _ControlToCome)

    def _put_new_headings_ahead_of(self, section, before) -> None:
        """Record the sub-headings a category just built ahead of it.

        :param before: ``id()`` of every heading recorded until now.
        """
        fresh = [other for other in self._settings_sections
                 if id(other) not in before]
        if not fresh:
            return
        kept = [other for other in self._settings_sections
                if id(other) in before]
        at = next((i for i, other in enumerate(kept) if other is section),
                  len(kept))
        self._settings_sections[:] = kept[:at] + fresh + kept[at:]
        bar = getattr(self, "_settings_search", None)
        known = getattr(bar, "_sections", None)
        if isinstance(known, list):
            where = next((i for i, other in enumerate(known)
                          if other is section), len(known))
            known[where:where] = [other for other in fresh
                                  if not any(other is k for k in known)]

    def _keys_the_run_has_no_object_for(self) -> set:
        """The settings whose object this run does not have, right now.

        Asked once per panel build and remembered for it, because every
        heading asks the same question and the answer cannot change while
        the panel is being laid out.

        :returns: setting keys, empty when the model cannot answer.
        """
        answer = getattr(self, "_run_has_no_object_for", None)
        if answer is not None:
            return answer
        model = getattr(self, "_settings_model", None)
        answer = set()
        if model is not None:
            try:
                answer = set(model.keys_whose_object_the_run_lacks())
            except Exception:                                # noqa: BLE001
                LOG.debug("could not decide which objects the run has",
                          exc_info=True)
        self._run_has_no_object_for = answer
        return answer

    def _refilter_the_settings_search(self) -> None:
        """Apply the settings search again after the object rule has run.

        The object rule shows every row its objects allow, and the search
        strip's Essentials level, query and Modified filter then narrow that.
        Run in the other order, a channel committed under Essentials put
        rows the level excludes back on the form and left the new object's
        segmentation heading off it. Re-entry is refused: applying the
        filter can lay out a waiting row, and laying one out runs the object
        rule again. The filter is applied without reopening sections, so a
        section the user shut stays shut when an unrelated setting such as
        ``metadata_type`` runs the object rule.
        """
        bar = getattr(self, "_settings_search", None)
        if bar is None or getattr(self, "_refiltering_settings", False):
            return
        self._refiltering_settings = True
        try:
            bar.apply(reopen=False)
        except RuntimeError:
            LOG.debug("the settings search is gone", exc_info=True)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not re-apply the settings search", exc_info=True)
        finally:
            self._refiltering_settings = False

    def _rows_the_filters_hide(self) -> set:
        """The settings the search strip and the dimension switches hide.

        Asked by the object rule before it sets rows, so that a row one of
        these hides anyway is left hidden rather than shown and hidden again
        (see ``SettingsWidgets.rows_the_screen_hides``). Only asked while
        the strip is not already re-filtering: the pass it runs from inside
        a re-filter is answered by that re-filter.

        :returns: setting keys.
        """
        hides = set()
        bar = getattr(self, "_settings_search", None)
        if bar is not None and not getattr(self, "_refiltering_settings",
                                           False):
            try:
                hides.update(bar.keys_it_hides())
            except Exception:                                # noqa: BLE001
                LOG.debug("could not ask the search strip", exc_info=True)
        try:
            for _section, key, _field in self._dimension_rows():
                if self._dimension_is_gated(setting_dimension(key)):
                    hides.add(key)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not ask the dimension switches", exc_info=True)
        return hides

    def _headings_the_run_lacks(self) -> set:
        """``id()`` of each heading the object rule is holding off the form.

        :returns: the headings whose every row belongs to an object or an
            organelle slot the run does not have, as the model last decided.
        """
        model = getattr(self, "_settings_model", None)
        return set(getattr(model, "_headings_of_absent_slots", None) or ())

    def _lay_out_the_rows_that_are_back(self, hidden) -> None:
        """Caption every waiting row the object rule no longer hides.

        THE MODEL CALLS THIS BEFORE IT DECIDES ROW VISIBILITY, which is what
        makes the deferral invisible: a setting the rule is about to show is
        given its caption in the same pass, so the row the rule shows is a
        finished one rather than a field with its name missing.

        :param hidden: the keys the rule has just decided must stay off the
            form.
        """
        waiting = getattr(self, "_rows_awaiting_layout", None)
        if not waiting:
            return
        hidden = set(hidden or ())
        back = [key for key in waiting if key not in hidden]
        if not back:
            return
        owners = {waiting[key] for key in back}
        for key in back:
            self._lay_out_one_waiting_row(key)
        restored = False
        for section in owners:
            restored = self._restore_settings_section(section) or restored
        if restored:
            self.refresh_maturity_visibility()
            if getattr(self, "_category_hint", None) is not None:
                self._wire_category_hints()
        self._the_rows_moved(judge_them=False)

    def _lay_out_every_waiting_row(self, section) -> None:
        """Caption every one of ``section``'s rows, run or no run.

        What :class:`_RowsBuiltWhenTheyAreAskedFor` calls when something reads
        the heading's rows back, so that everything walking that list is
        handed the whole heading. The object rule is re-run afterwards and
        hides again everything it hid before, so ASKING what a heading holds
        cannot put a setting for an absent object on screen.
        """
        waiting = getattr(self, "_rows_awaiting_layout", None)
        if not waiting:
            return
        mine = [key for key, owner in waiting.items() if owner is section]
        if not mine:
            return
        for key in mine:
            self._lay_out_one_waiting_row(key)
        self._the_rows_moved(judge_them=True)

    def _caption_every_waiting_row(self) -> None:
        """Give every row still waiting its caption, on every heading.

        What the checks that walk the panel's captions ask for, and the
        coarsest of the three ways a waiting row is built. The object rule is
        re-run afterwards, so this cannot put a setting for an absent object
        on screen -- it only makes sure there is nothing left to find.
        """
        self._open_every_waiting_heading()
        waiting = getattr(self, "_rows_awaiting_layout", None)
        if not waiting:
            return
        for key in list(waiting):
            self._lay_out_one_waiting_row(key)
        self._the_rows_moved(judge_them=True)

    def _lay_out_one_waiting_row(self, key: str) -> None:
        """Give ``key``'s row its caption, in the form row it already holds.

        The field is on the form already, spanning a row of its own. Taking
        that row out, laying the pair out properly and moving the result back
        to the same index is what keeps a revealed setting where the module
        wrote it -- ``Section.add_row`` appends, and a row that appeared under
        the sub-headings instead of among its own would be a worse answer than
        the caption it was waiting for.
        """
        from PySide6.QtWidgets import QFormLayout

        waiting = getattr(self, "_rows_awaiting_layout", None)
        section = (waiting or {}).pop(key, None)
        if section is None:
            return
        row = next(((k, label, widget)
                    for k, label, widget in
                    getattr(section, "_spacr_declared_rows", ()) or ()
                    if k == key), None)
        if row is None:
            return
        form = getattr(section, "_form", None)
        if not isinstance(form, QFormLayout):
            return
        late = getattr(self, "_captioned_late", None)
        if late is None:
            late = self._captioned_late = set()
        late.add(section)
        try:
            at, _role = form.getWidgetPosition(row[2])
            if at >= 0:
                form.takeRow(at)
            self._lay_out_setting_row(section, row[1], row[2])
            last = form.rowCount() - 1
            if 0 <= at < last:
                taken = form.takeRow(last)
                label_item = getattr(taken, "labelItem", None)
                field_item = getattr(taken, "fieldItem", None)
                if field_item is None:
                    return
                if label_item is None:
                    form.insertRow(at, field_item.widget())
                else:
                    form.insertRow(at, label_item.widget(),
                                   field_item.widget())
        except RuntimeError:
            LOG.debug("no heading left to caption %s on", key, exc_info=True)

    def _the_rows_moved(self, judge_them: bool = True) -> None:
        """Put the panel's row-shaped answers back in step after a build.

        :meth:`_rows_moved_steps` run to the end; see it.

        :param judge_them: run the object rule over the new rows.
        """
        _run_to_the_end(self._rows_moved_steps(judge_them))

    def _rows_moved_steps(self, judge_them: bool = True):
        """Put the panel's row-shaped answers back in step after a build.

        A row that arrives after the panel was laid out has to be judged by
        everything that judges a row -- the object rule and the dimension
        switches decide whether it is on screen, the settings search has to be
        able to find it, and the section's own list of rows goes back into the
        order it was declared in rather than the order the rule got round to.

        :param judge_them: run the object rule over the new rows. False when
            the rule is what asked for them, because it decides every gated
            row itself the moment this returns -- and running it from inside
            itself would be a second pass saying the same thing.
        """
        for section in getattr(self, "_settings_sections", []) or []:
            if getattr(section, "_spacr_waiting_spec", None) is not None:
                continue
            rows = section.__dict__.get("_row_widgets")
            declared = getattr(section, "_spacr_declared_rows", None)
            if not isinstance(rows, list) or not declared:
                continue
            order = {id(widget): index
                     for index, (_k, _l, widget) in enumerate(declared)}
            if all(id(pair[1]) in order for pair in list.__iter__(rows)):
                rows.sort(key=lambda pair: order[id(pair[1])])
        yield
        if judge_them:
            model = getattr(self, "_settings_model", None)
            if model is not None:
                try:
                    model.refresh_object_visibility()
                except Exception:                            # noqa: BLE001
                    LOG.debug("could not re-decide the object rows",
                              exc_info=True)
            yield
        try:
            self._apply_dimension_visibility()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not re-apply the dimension switches",
                      exc_info=True)
        yield
        late = getattr(self, "_captioned_late", None) or set()
        self._captioned_late = set()
        if late:
            try:
                from ..i18n import retranslate_widget_tree

                for section in late:
                    retranslate_widget_tree(section)
            except RuntimeError:
                pass
            except Exception:                                # noqa: BLE001
                LOG.debug("could not translate a caption that arrived late",
                          exc_info=True)
            yield
        bar = getattr(self, "_settings_search", None)
        if bar is None:
            return
        try:
            bar._build_index()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not re-index the settings search", exc_info=True)
            return
        yield
        try:
            bar.apply(reopen=False)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not re-index the settings search", exc_info=True)

    def the_name_carries_the_help(self) -> int:
        """Move every settings tooltip off its field and onto its NAME.

        :returns: how many were moved, so a test can assert a number.

        THE HOVER TARGET IS THE SETTING'S NAME. Hovering the box you type
        in pops the help over the value you are reading or editing, and a
        field can be focused and clicked, so the popup fights the
        interaction. The name is inert, which makes it the calm target.

        `retarget_field_tooltips` is how the rest of the tool does this --
        every dialog and side panel calls it at the end of its `__init__`
        -- and the main settings form was the one place that never did.
        Measured on Mask: 1,538 fields carried their own tooltip and 13
        labels had one. `_lay_out_setting_row` moves the help for a row it
        lays out itself, which is 77 of that screen's 1,657; the rest
        arrive from the lazy row builder, the deferred "rows that are
        back" pass, and a fold mounting another module's categories.

        Run again after rows appear later, since it only ever moves a
        tooltip that is still on a field: it is idempotent by
        construction, and a second pass over rows already moved finds
        nothing to do.
        """
        from .settings_model import retarget_field_tooltips

        try:
            return int(retarget_field_tooltips(self))
        except Exception:
            return 0

    def _lay_out_setting_row(self, section, label, widget) -> None:
        """Put one setting on ``section``'s form: its label, then its field.

        Split out of :meth:`_build_settings_section` because a row is not
        always laid out while the panel is being built: a setting whose
        object the run does not have waits here until the rule says the
        run has it. See :meth:`_lay_out_the_rows_that_are_back`.

        :param section: the heading the row belongs to.
        :param label: the caption as the module wrote it. The language pass
            translates it afterwards, here as everywhere else.
        :param widget: the field ``SettingsWidgets`` built for the key.
        """
        setting_key = self._key_of(widget)
        field = widget
        widget = self._with_a_plate_map(widget, setting_key)
        widget = self._with_a_settings_advisor(widget, setting_key)
        widget = self._with_a_model_zoo_button(widget, setting_key)
        if widget is not field:
            widget.setProperty("settingKey", setting_key)
            widget.setProperty("settingsAppKey", self.app_key)
        lbl_widget = QLabel(label)
        lbl_widget.setCursor(Qt.WhatsThisCursor)
        field_key = None
        key = self._key_of_field(field)
        if key is not None:
            field_key = key
            html = field.toolTip()
            hint = self._settings_model.plain_tooltip_for(key)
            body_source = field.property(
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
            field.setProperty("apiTooltipDisplayRole", "metadata")
            field.setToolTip("")

            field._spacr_setting_label = lbl_widget
            lbl_widget.setEnabled(field.isEnabled())
            self._hint_map[lbl_widget] = hint
            self._html_tip_map[lbl_widget] = html
            lbl_widget.installEventFilter(self)
            self._put_the_greyed_reason_on(field, lbl_widget)
        section.add_row(lbl_widget, widget, info_widget=None,
                        wrap_label=True)
        self._attach_column_picker(field_key, field)

    @staticmethod
    def _put_the_greyed_reason_on(field, label) -> None:
        """Give a new caption the reason its field is greyed, if it is.

        A rule that greys a field before its row is laid out keeps the
        reason on the field (``settings_model._PENDING_NOTE_PROPERTY``, "so
        it can be put on a label that does not exist yet"), and nothing put
        it there: a greyed row's name, which is where the help lives, said
        nothing about why. The reason reached the name only if some later
        pass happened to grey the field again -- so whether it did depended
        on the order categories were built in.

        :param field: the setting's control.
        :param label: the caption just made for it.
        """
        from .settings_model import _PENDING_NOTE_PROPERTY, _note_on_label

        note = str(field.property(_PENDING_NOTE_PROPERTY) or "")
        if note and not field.isEnabled():
            _note_on_label(label, note)

    @staticmethod
    def _open_the_headings_above(parent, expanded: bool) -> None:
        """Open ``parent`` when a heading inside it is opened.

        Collapsing a sub-heading deliberately leaves the umbrella open: the
        user still has the rest of the group in front of them.
        """
        if not expanded:
            return
        try:
            if not parent.is_expanded():
                parent.set_expanded(True)
        except (AttributeError, RuntimeError):
            pass

    def _install_section_explainer(self, section, title) -> None:
        """Add a section's read-only explanatory panel above its controls.

        Sections listed in ``settings_model.SECTION_EXPLAINERS`` receive a
        selectable rich-text panel. Prose wraps normally, formulas remain
        unwrapped in ``<pre>`` blocks, and the object name supplies the
        monospace styling through QSS.
        """
        from PySide6.QtGui import QFontDatabase, QFontMetrics
        from PySide6.QtWidgets import QTextEdit

        from .settings_model import explainer_width

        box = _ExplainerBrowser(self._retheme_section_explainers)
        box.setObjectName("ModelExplainer")
        box.setReadOnly(True)
        box.setOpenExternalLinks(True)
        box.setLineWrapMode(QTextEdit.WidgetWidth)

        fixed = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        box.setFont(fixed)
        box.setStyleSheet(
            "QTextBrowser#ModelExplainer {"
            f' font-family: "{fixed.family()}", "DejaVu Sans Mono", "Menlo",'
            " \"Consolas\", monospace; }")
        box.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        advance = QFontMetrics(box.font()).horizontalAdvance("M") or 8
        box.setMinimumWidth(advance * (explainer_width() + 3))
        box.setMinimumHeight(220)
        section.add_prose(box, at_top=True)
        self._section_explainers[title] = box
        if title == "Model & Inference":
            self._model_explainer = box
        else:
            from ..theme import active_palette
            from .settings_model import section_explainer_html

            box.setHtml(section_explainer_html(self.app_key, title,
                                               palette=active_palette()))
            return

        widgets = getattr(self._settings_model, "_widgets", {})
        for key in ("regression_type", "level", "model_plate_position",
                    "random_row_column_effects", "inference",
                    "analysis_mode"):
            widget = widgets.get(key)
            if widget is None:
                continue
            for signal_name in ("currentTextChanged", "currentIndexChanged",
                                "toggled", "stateChanged",
                                "textChanged", "value_changed"):
                signal = getattr(widget, signal_name, None)
                if signal is not None:
                    signal.connect(self._on_model_selection_changed)
                    break
        self._refresh_model_explainer()

    def _on_model_selection_changed(self, *_args) -> None:
        """Re-render the explainer when the model or the level moves."""
        self._refresh_model_explainer()

    def _refresh_model_explainer(
        self,
        language: Optional[str] = None,
    ) -> None:
        """Render the selected model in the current palette and language."""
        box = self._model_explainer
        if box is None:
            return
        from ..theme import active_palette
        from .settings_model import regression_model_explainer_html

        widgets = getattr(self._settings_model, "_widgets", {})
        read = self._settings_model._read_widget

        def value(key, fallback):
            """One widget's value, or the fallback when there is no such widget."""
            widget = widgets.get(key)
            if widget is None:
                return fallback
            try:
                return read(widget)
            except Exception:
                return fallback

        scroll = box.verticalScrollBar().value()
        box.setHtml(regression_model_explainer_html(
            value("regression_type", "auto"), value("level", "both"),
            plate_position=value("model_plate_position", False),
            random_row_column=value("random_row_column_effects", False),
            palette=active_palette(),
            language=language,
            inference=value("inference", "auto"),
            analysis_mode=value("analysis_mode", "")))
        box.verticalScrollBar().setValue(scroll)

    #: The grid key the gene tile answers to. Not a figure index -- see below.
    GENE_TILE_KEY = "gene"

    def _gene_tile_entry(self, panel) -> list:
        """Return the selected gene panel as a figure-grid tile.

        The tile uses a stable live key instead of a figure index, so adding it
        does not shift saved-figure positions. A pixmap snapshot keeps grid
        resizing responsive; activating the tile opens the live gene panel.
        An empty list is returned when no gene is selected or rendering fails.
        """
        gene = getattr(panel, "gene", None)
        if gene is None or not hasattr(gene, "to_pixmap"):
            return []
        showing = getattr(gene, "feature", None)
        if not callable(showing) or not str(showing() or "").strip():
            return []
        try:
            pixmap = gene.to_pixmap()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not photograph the gene tile", exc_info=True)
            return []
        if pixmap is None or pixmap.isNull():
            return []
        return [(self.GENE_TILE_KEY, pixmap, "Gene")]

    def _with_a_plate_map(self, widget, key):
        """``widget``, with a plate-map button beside it when it takes wells.

        The button goes on `well_spec.WELL_ONLY_SETTINGS` -- the settings
        whose COMPLETE value is a well specification -- because pressing Done
        replaces the field. Every other setting gets the original widget
        untouched, including the mixed-vocabulary ones that may hold a well
        or a gene or a treatment name: replacing one of those would discard
        whatever else the user had typed.
        """
        from ...well_spec import WELL_ONLY_SETTINGS

        if str(key) not in WELL_ONLY_SETTINGS:
            return widget
        from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(widget, 1)
        button = QPushButton("Plate…", holder)
        button.setToolTip(
            "Pick the wells off a plate map. Rows (r1), columns (c1) and "
            "wells (A01) all read, and what it writes reads back.")
        button.clicked.connect(
            lambda *_, w=widget, k=str(key): self.pick_wells_for(w, k))
        row.addWidget(button)
        holder._spacr_field = widget
        return holder

    def _key_of(self, widget) -> str:
        """The setting name a widget holds, or ``''``.

        The panel is built from `(label, widget)` pairs and every rule that
        acts on a particular SETTING needs the key. Read off `_widgets`,
        which is the one place that maps the two -- through
        :meth:`_key_of_field`, so this is a lookup rather than the second
        full scan of that table per row.
        """
        key = self._key_of_field(widget)
        return "" if key is None else str(key)

    #: Settings whose value is a path to a Cellpose checkpoint, and which
    #: therefore get the model-zoo button. Named explicitly rather than
    #: matched on a suffix: `custom_model_path` in Classify holds a torch
    #: classifier, not a Cellpose model, and offering cpsam checkpoints there
    #: would offer something that screen cannot load.
    #: Settings whose value is a Cellpose checkpoint, named explicitly.
    #:
    #: Not matched on a suffix alone: `custom_model_path` in Classify holds a
    #: torch classifier, and offering cpsam checkpoints there would offer
    #: something that screen cannot load.
    _MODEL_ZOO_KEYS = (
        "pathogen_model", "pathogen_model_name",
        "cell_model_name", "nucleus_model_name", "organelle_model_name",
        "custom_model", "plaque_model",
    )

    @classmethod
    def _takes_a_cellpose_checkpoint(cls, key: str) -> bool:
        """Whether ``key`` names a Cellpose checkpoint field.

        THE SECOND ORGANELLE ONWARDS IS NOT IN THE LIST ABOVE, and cannot be:
        with more than one organelle the settings are generated per organelle
        as `organelleb_model_name`, `organellec_model_name` and so on
        (:data:`spacr.settings.DYNAMIC_ORGANELLE_SETTINGS`). A fixed tuple gave
        the button to organelle 1 and silently withheld it from every organelle
        after it, which is the shape of bug a list of literal names always
        eventually has when the names are generated.
        """
        name = str(key)
        if name in cls._MODEL_ZOO_KEYS:
            return True
        return bool(_ORGANELLE_MODEL_KEY.match(name))

    def _with_a_model_zoo_button(self, widget, key):
        """Add the model-zoo button beside a Cellpose-checkpoint field.

        Other settings are returned unchanged.

        A model setting takes a filesystem path, which is exact and unhelpful:
        the user has to already know a model exists, find where it lives, and
        type it. The button is the other way in -- browse what spaCR knows
        about, download one, and have its path written into this field.

        THE FIELD IS STILL WHAT THE PANEL COLLECTS FROM, the same rule the
        plate map and the advisor follow: a wrapper that made the value
        unreadable would be worse than no button. Typing a path by hand keeps
        working exactly as before; this only adds a second way in.
        """
        if not self._takes_a_cellpose_checkpoint(key):
            return widget
        from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(widget, 1)
        button = QPushButton("Model zoo…", holder)
        button.setToolTip(
            "Browse the models spaCR knows about, see what each was trained "
            "on, download one and fill in this field. You can still type a "
            "path yourself. "
            "API: spacr.qt.widgets.model_zoo_picker.choose_model.")
        button.clicked.connect(
            lambda *_, f=widget, k=key: self._choose_a_model_for(f, k))
        row.addWidget(button)
        holder._spacr_field = widget
        return holder

    @staticmethod
    def _model_kinds_for(key: str) -> tuple:
        """Which zoo kinds this model field can actually load.

        ``kinds`` is a rule rather than a parameter: the zoo also carries the
        YOLO well detector, and offering that in a Cellpose field would offer
        something no segmenter can load -- a choice that fails at segmentation
        time, long after the click that caused it.

        A ``*_model_name`` field gets ``cellpose3`` as well. Those are the
        settings :func:`spacr.settings._get_object_settings` reads BY NAME
        when ``segmentation_backend`` is ``'cellpose3'``, so cyto3, cyto2,
        cyto, nuclei and any bioimage.io Cellpose 3 checkpoint are real
        choices there. ``plaque_model`` and ``custom_model`` are not: those
        screens load the checkpoint with spaCR's own Cellpose 4, in spaCR's
        own process, and a Cellpose 3 name would fail there.
        """
        return (("cellpose", "cellpose3") if str(key).endswith("_model_name")
                else ("cellpose",))

    def _choose_a_model_for(self, field, key: str = "") -> None:
        """Open the picker and write the chosen path into ``field``.

        :param field: the widget the chosen value is written into.
        :param key: the setting the field stands for, which decides what the
            picker offers -- see :meth:`_model_kinds_for`.
        """
        from ..widgets.model_zoo_picker import choose_model

        path = choose_model(self, kinds=self._model_kinds_for(key))
        if not path:
            return
        if hasattr(field, "set_value"):
            field.set_value(path)
        elif hasattr(field, "setText"):
            field.setText(path)
        else:
            LOG.warning("no way to write a model path into %s",
                        type(field).__name__)

    def _with_a_settings_advisor(self, widget, key):
        """Add the regression settings-advisor button beside ``inference``.

        Other settings and application screens are returned unchanged.
        """
        if str(key) != "inference" or self.app_key != "regression":
            return widget
        from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        button = QPushButton("Settings for my data…", holder)
        button.setToolTip(
            "Read the attached tables, ask the few things they cannot "
            "answer, and propose settings with the reason beside each "
            "one and the current value beside the new one. Nothing is "
            "written until you accept it. "
            "API: spacr.settings_advisor.advise_the_screen.")
        button.clicked.connect(lambda *_: self.settings_for_my_data())
        row.addWidget(button)
        row.addWidget(widget, 1)
        holder._spacr_field = widget
        self._advisor_button = button
        return holder

    def the_advisor_can_run(self) -> str:
        """Return an empty string when the advisor can run, otherwise why not."""
        values = {}
        try:
            values = dict(self._settings_model.collect() or {})
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not collect the settings", exc_info=True)
        counts, scores = self._tables_for_the_advisor(values)
        if not counts and not scores:
            return ("No count or score table is attached yet. Fill in Input "
                    "Tables — or press 'Load test data…' there — "
                    "and this can read them.")
        return ""

    @staticmethod
    def _tables_for_the_advisor(values: dict):
        """The count and score paths, out of either input spelling.

        `paired_data` is the current shape -- one row per plate, naming both
        files -- and `count_data`/`score_data` are the legacy lists. Both are
        read, because a panel filled from an old settings CSV has the second.
        """
        counts, scores = [], []
        for row in values.get("paired_data") or ():
            if not isinstance(row, dict):
                continue
            if row.get("count"):
                counts.append(str(row["count"]))
            if row.get("score"):
                scores.append(str(row["score"]))
        for key, into in (("count_data", counts), ("score_data", scores)):
            got = values.get(key)
            if isinstance(got, str):
                got = [got]
            for path in got or ():
                if path and str(path) not in into:
                    into.append(str(path))
        return counts, scores

    def settings_for_my_data(self, *, answers: Optional[dict] = None) -> dict:
        """Propose settings from the attached tables and apply accepted values.

        Parameters
        ----------
        answers : dict, optional
            Answers to questions the data cannot resolve. Supplying this
            mapping skips the interactive question page.

        Returns
        -------
        dict
            Settings written to the panel, or an empty mapping when the
            advisor cannot run or the proposal is declined.
        """
        from PySide6.QtWidgets import QDialog

        from ...settings_advisor import advise_that_runs, read_the_screen
        from ..widgets.settings_advisor_dialog import SettingsAdvisorDialog

        why_not = self.the_advisor_can_run()
        if why_not:
            self._console.append_stdout(why_not + "\n")
            return {}
        values = dict(self._settings_model.collect() or {})
        counts, scores = self._tables_for_the_advisor(values)
        self._console.append_stdout(
            f"Reading {len(counts)} count table(s) and {len(scores)} score "
            f"table(s) to work out the settings…\n")
        reading = read_the_screen(
            counts, scores, str(values.get("dependent_variable") or ""))
        reading = self._reading_with_the_last_run(reading, values)
        for trouble in reading.trouble:
            self._console.append_stdout(f"  {trouble}\n")
        if reading.run_note:
            self._console.append_stdout(f"  {reading.run_note}\n")
        elif reading.run_folder:
            self._console.append_stdout(
                f"  Also reading the diagnostics of {reading.run_folder}\n")

        if answers is not None:
            chosen = advise_that_runs(reading, answers).as_settings()
        else:
            dialog = SettingsAdvisorDialog(reading, values, parent=self)
            if dialog.exec() != QDialog.Accepted:
                self._console.append_stdout(
                    "Nothing was changed.\n")
                return {}
            chosen = dialog.accepted_settings()
        if not chosen:
            self._console.append_stdout(
                "The proposal was never shown, so nothing was written.\n")
            return {}
        applied = self.apply_settings_dict(chosen)
        self._console.append_stdout(
            f"{applied} setting(s) written from the data.\n")
        return chosen

    def _console_folded(self, shut: bool) -> None:
        """Give the released height to whatever is above the console.

        MEASURED, NOT EYEBALLED. The console lives in a vertical splitter, and
        a splitter keeps its own sizes -- hiding the widget alone leaves the
        handle where it was and the space simply unused, which is the failure
        this instruction is actually about.
        """
        wrap = getattr(self, "_console_wrap", None)
        if wrap is None:
            return
        from ..widgets.collapsible_splitter import splitter_of

        if splitter_of(wrap) is not None:
            return
        wrap.setMinimumHeight(0 if shut else 180)
        splitter = getattr(self, "_console_splitter", None)
        if splitter is None:
            for parent in (wrap.parentWidget(),):
                if isinstance(parent, QSplitter):
                    splitter = parent
                    break
        if splitter is None:
            return
        index = splitter.indexOf(wrap)
        if index < 0:
            return
        sizes = list(splitter.sizes())
        if shut:
            self._console_height = sizes[index]
            freed = sizes[index] - wrap.sizeHint().height()
            sizes[index] -= freed
            above = index - 1 if index > 0 else (1 if len(sizes) > 1 else -1)
            if above >= 0:
                sizes[above] += freed
        else:
            back = getattr(self, "_console_height", 0)
            if back:
                freed = back - sizes[index]
                sizes[index] = back
                above = index - 1 if index > 0 else (1 if len(sizes) > 1
                                                     else -1)
                if above >= 0:
                    sizes[above] = max(0, sizes[above] - freed)
        splitter.setSizes(sizes)

    def _reading_with_the_last_run(self, reading, values):
        """Fold a finished run's diagnostics into ``reading``.

        SILENT WHEN THERE IS NO RUN, which is the ordinary case and the whole
        point of the button -- it answers before anything has been fitted.
        """
        from dataclasses import replace

        from ...settings_advisor import Reading, read_the_last_run

        folder = ""
        for source in (getattr(self, "_last_run_folder", ""),
                       self._results_panel.run_folder()
                       if getattr(self, "_results_panel", None) else ""):
            if source:
                folder = str(source)
                break
        if not folder:
            return reading
        try:
            extra = read_the_last_run(folder, values)
        except Exception:                                       # noqa: BLE001
            return reading
        known = set(Reading.__dataclass_fields__)
        extra = {k: v for k, v in extra.items() if k in known}
        return replace(reading, **extra) if extra else reading

    def pick_wells_for(self, field, key: str = "") -> str:
        """Open the plate map on ``field``'s value. Returns what was written.

        :param field: the settings field holding a well specification; its
            ``text()`` seeds the picker and ``setText()`` receives the choice
            when the field has them.
        :returns: the new specification, or ``""`` when the user closed
            without choosing -- in which case the field is untouched.
        """
        from ..widgets.plate_map_picker import PlateMapPicker

        reader = getattr(field, "text", None)
        before = str(reader() if callable(reader) else "")
        picker = PlateMapPicker(before, parent=self)
        if not picker.exec():
            return ""
        chosen = picker.value()
        setter = getattr(field, "setText", None)
        if callable(setter):
            setter(chosen)
        self._console.append_stdout(
            f"{key or 'wells'}: {chosen or 'nothing'} chosen from the "
            f"{picker._layout_size}-well map.\n")
        return chosen

    def _install_example_data_button(self, section) -> None:
        """A button that fetches the example screen and fills the two slots.

        `add_prose`, not `add_widget`: `_row_widgets` is taken to hold
        LABELLED SETTING ROWS by the module smoke test, and a button is
        neither a setting nor labelled -- the same reason the explainer boxes
        go in that way.
        """
        from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for label, kind, tip in (
                ("Score", "scores",
                 "Fetch the four score tables (about 19 MB) into the scores "
                 "slot."),
                ("Count", "counts",
                 "Fetch the four count tables (about 16 MB) into the counts "
                 "slot."),
        ):
            one = QPushButton(label)
            one.setToolTip(tip + " Cached afterwards, so pressing it again is "
                                 "instant.")
            one.clicked.connect(
                lambda _checked=False, _k=kind: self.load_the_example_screen(
                    kind=_k))
            layout.addWidget(one)
            setattr(self, f"_example_{kind}_button", one)

        feature = QPushButton("Measurements (.db)")
        feature.setToolTip(
            "Choose which of the screen's measurement databases to download. "
            "These are what the measurement and cell functions read. About "
            "0.5 GB per plate.")
        feature.clicked.connect(
            lambda: self.load_the_screen_data(kind="measurements"))
        self._screen_feature_button = feature
        layout.addWidget(feature)

        crops = QPushButton("Image crops")
        crops.setToolTip(
            "Choose which of the screen's crop folders to download. Only "
            "needed to display images — the measurement functions do not "
            "read them. About 8 GB per plate.")
        crops.clicked.connect(
            lambda: self.load_the_screen_data(kind="crops"))
        self._screen_crops_button = crops
        layout.addWidget(crops)

        layout.addStretch(1)

        section.add_prose_row("Download", row, at_top=True)

        button = QPushButton(tr("Load test data…"))
        button.setToolTip(
            "Fetch the four-plate example screen and put its count tables "
            "and score tables into the two slots below. About 35 MB the "
            "first time; cached afterwards, so pressing it again is instant.")
        button.clicked.connect(lambda: self.load_the_example_screen())
        self._example_data_button = button
        section.add_prose(button, at_top=True)

    #: Settings files an example dataset may ship, per module, best first.
    #:
    #: SEVERAL NAMES PER MODULE because the published sets were written by
    #: different runs at different times: a mask run saves
    #: `gen_mask_settings.csv`, the older pack shipped `gen_masks_settings.csv`,
    #: and the measure settings have been called both `measure_crop_settings`
    #: and `crop_measure_settings`. Trying each in turn is what makes an older
    #: archive still fill the form in, rather than silently filling in nothing.
    _EXAMPLE_SETTINGS_FILES = {
        "mask": ("gen_mask_settings.csv", "gen_masks_settings.csv"),
        "measure": ("measure_crop_settings.csv", "crop_measure_settings.csv"),
        "classify": ("classify_settings.csv",),
        "regression": ("regression_settings.csv",),
        "umap": ("umap_settings.csv",),
    }

    @staticmethod
    def reanchor_example_paths(loaded, destination) -> dict:
        """Re-root every absolute path in ``loaded`` onto ``destination``.

        A shipped example settings file records the paths of the machine
        that GENERATED it, which on any other machine names a user that does
        not exist::

            gen_masks_settings.csv    src,/home/carruthers/datasets/plate1
            crop_measure_settings.csv src,/home/carruthers/datasets/plate1/merged

        Both lines matter. The first is why the path is wrong; the second is
        why substituting the destination outright is not the fix -- the Measure
        set points at a SUBFOLDER, and collapsing it to the plate root would
        quietly measure the wrong directory rather than fail.

        LISTS AND TUPLES ARE NOT A CORNER CASE. Classify's ``src`` is
        list-valued and regression's ``count_data``, ``score_data`` and
        ``paired_data`` are too, and ``utils.load_settings`` turns any CSV cell
        starting with ``[``, ``(`` or ``{`` into a real Python container. An
        earlier version of this method tested ``isinstance(value, str)`` and
        skipped everything else, which skipped exactly the two modules whose
        loaders never write a local path as a fallback -- so for those the
        publisher's path was the panel's only source of truth. Containers are
        walked.

        Most of the per-path work is not new. :func:`spacr.portable_paths.reroot_crop_path`
        already picks the deepest recorded suffix that EXISTS below the current
        root, so a rewrite is only made when the reconstructed path is really
        there; it is tried first and its answer preferred. What it cannot do is
        the root itself -- its resolution needs a suffix to match, and ``src``
        pointing at the plate folder has none, which is the reported case. That
        one gap is filled by matching the destination's own folder name.

        A path that already resolves on this machine is left ALONE: a user who
        imported an example, edited ``src`` to their own data and saved would
        otherwise have that edit undone by the next example load.

        :param loaded: the settings as read from the shipped file.
        :param destination: the local folder the example actually unpacked to.
        :returns: a new mapping; the input is not modified.
        """
        from pathlib import Path, PurePosixPath, PureWindowsPath

        from ...portable_paths import reroot_crop_path

        destination = Path(destination)
        anchor = destination.name

        def rehome_text(text: str):
            """Repoint one path string at the folder the example now lives in."""
            stripped = text.strip()
            if not stripped:
                return text
            pure = (PureWindowsPath(stripped) if "\\" in stripped
                    else PurePosixPath(stripped))
            if not pure.is_absolute():
                return text
            if Path(stripped).exists():
                return text

            try:
                rerooted = reroot_crop_path(stripped, str(destination))
            except Exception:                                # noqa: BLE001
                rerooted = stripped
            if rerooted and rerooted != stripped:
                return rerooted

            parts = list(pure.parts)
            if anchor in parts:
                tail = parts[len(parts) - 1 - parts[::-1].index(anchor) + 1:]
                candidate = destination.joinpath(*tail) if tail else destination
                return str(candidate)
            return text

        def rehome(value):
            """Repoint every path in a value, whatever shape it is.

            Recurses through lists and tuples: a settings value may be one path or a
            list of them, and rehoming only the string case leaves the list pointing
            at a folder that is not there.
            """
            if isinstance(value, str):
                return rehome_text(value)
            if isinstance(value, (list, tuple)):
                return type(value)(rehome(item) for item in value)
            return value

        return {key: rehome(value) for key, value in loaded.items()}

    def apply_settings_that_came_with(self, folder, *,
                                      pack_folder=None) -> int:
        """Load the settings a downloaded example shipped, for THIS module.

        The point of shipping settings beside data: a user who has to work out
        which column holds the labels, what the mask dimensions are and which
        channels were measured has done most of the work the example was meant
        to save. With them applied, Run is the next action.

        Use the shared settings-pack reader to migrate old names and report
        renamed, dropped, and unreadable rows in the console. Apply only values
        supplied by the pack, leaving other form values alone. Re-anchor the
        publisher's paths onto the local dataset while preserving subfolders.

        :param folder: the unpacked dataset folder.
        :param pack_folder: optional folder of shipped settings CSVs, preferred
            over the dataset's own ``settings`` subfolder.
        :returns: how many settings were applied; 0 when no file was found,
            no supplied settings apply to this form, or applying them failed.
        """
        from pathlib import Path

        from ..settings_pack import settings_from_pack

        folder = Path(folder)
        # A form rebuild detaches this widget before bulk application returns.
        # Keep its owner so the report can reach the replacement's console.
        owner = self.window() if hasattr(self, "window") else None
        # THE SHIPPED PACK FIRST, THEN THE PLATE'S OWN FOLDER.
        #
        # A completed run writes `<src>/settings/<name>.csv` --
        # `utils.save_settings`, with name='gen_mask_settings' for Mask -- and
        # that is the same folder and the same filename this search looks in.
        # `_EXAMPLE_SETTINGS_FILES` even lists the run's spelling FIRST, which
        # its own comment says out loud: "a mask run saves
        # `gen_mask_settings.csv`, the older pack shipped
        # `gen_masks_settings.csv`".
        #
        # So on a cached example, once the user has run the module once, their
        # own output sits under the preferred name and wins forever, because a
        # cached example is never re-fetched. It cannot happen until you have
        # used the thing once, which is why it only bites returning users.
        #
        # The download already separates them -- the plate unpacks to
        # `<dest>/plate1` and the pack to `<dest>/settings`, a SIBLING that no
        # run writes into -- so the fix is to look there first rather than to
        # guess between two files with the same name.
        roots = []
        if pack_folder is not None:
            roots.append(Path(pack_folder))
        roots.append(folder / "settings")
        for root in roots:
            report = None
            path = root
            try:
                # Do not override src here: Measure's pack points at /merged,
                # which reanchor_example_paths preserves below the local plate.
                loaded, report = settings_from_pack(self.app_key, root)
                if not report.source:
                    continue
                # The reader returns defaults too; an example import must not
                # reset values the pack never supplied. A found pack remains
                # authoritative even when all its keys were dropped.
                supplied = set(report.applied)
                supplied.update(new for _old, new in report.renamed)
                loaded = {key: value for key, value in loaded.items()
                          if key in supplied}
                path = root / report.source
                try:
                    loaded = self.reanchor_example_paths(loaded, folder)
                except Exception:                            # noqa: BLE001
                    LOG.debug("could not re-home %s", path, exc_info=True)
                applied = self.apply_settings_dict(loaded) if loaded else 0
            except Exception as exc:                         # noqa: BLE001
                name = report.source if report and report.source else str(root)
                LOG.debug("could not apply %s", path, exc_info=True)
                _example_pack_console(self, owner).append_notice(
                    "[example] {name} could not be applied: {detail}\n",
                    name=name, detail=exc)
                return 0
            _append_example_pack_report(
                _example_pack_console(self, owner), report, applied)
            return applied
        return 0

    def screen_data_destination(self):
        """The shared example plate folder the screen pieces unpack into."""
        from ..hf_download import example_plate_folder

        return example_plate_folder()

    def load_the_screen_data(self, *, kind=None, ask=None,
                             choose=None) -> dict:
        """Ask which pieces of the screen to fetch, then fetch them.

        :param kind: ``"measurements"`` or ``"crops"``, so the Feature and
            Image-crops buttons each show only their own. Filtered rather than
            greyed: a Feature download listing eight rows and refusing four of
            them would be four chances to start a 30 GB transfer by mistake.
        :param choose: replaces the picker, for tests.
        :param ask: replaces the downloader, for tests.
        :returns: what was set on the panel, or an empty mapping.
        """
        from ...screen_data import SCREEN_REPO

        destination = self.screen_data_destination()
        destination.mkdir(parents=True, exist_ok=True)

        picker = choose
        if picker is None:
            from ..widgets.screen_data_picker import choose_screen_data as picker
        chosen = picker(self, destination, kind) or []
        if not chosen:
            return {}

        was = {"measurements": ("_screen_feature_button", "Feature"),
               "crops": ("_screen_crops_button", "Image crops")}.get(
                   kind, ("_screen_data_button", "Screen measurements\u2026"))
        button = getattr(self, was[0], None)
        if button is not None:
            button.setEnabled(False)
            button.setText(tr("Fetching\u2026"))

        placed: dict = {}

        def _done(result, error):
            """Restore the button whether the load worked or failed."""
            if button is not None:
                button.setEnabled(True)
                button.setText(tr(was[1]))
            if result is None:
                self._console.append_notice(
                    "[screen] not downloaded: {detail}\n",
                    detail=error or "cancelled")
                return
            self._console.append_stdout(
                tr("Screen data ready: {path}", path=str(destination)) + "\n")
            self.apply_settings_that_came_with(destination)
            placed["src"] = str(destination)

        download = ask
        if download is None:
            from ..hf_download import download_chosen_screen_data as download
        download(self, destination, [a.archive for a in chosen],
                 SCREEN_REPO, _done)
        return placed

    def _install_example_images_button(self, section) -> None:
        """Add the example-image control that populates ``src``."""
        from PySide6.QtWidgets import QPushButton

        button = QPushButton(tr("Load test data…"))
        button.setToolTip(tr(
            "Download the approximately 400 MB toxo_mito example plate and "
            "set its image directory as the source (src). Later requests "
            "reuse the cached files. The download includes compatible "
            "example settings."))
        button.clicked.connect(lambda: self.load_the_example_images())
        self._example_images_button = button
        section.add_prose(button, at_top=True)

    def _install_plaque_example_button(self, section) -> None:
        """Add Plaque Analysis's test-data control.

        The module is not in EXAMPLE_DATA_SECTIONS, so the dispatch above never
        reaches it and this builds its button instead. It offers TWO sets, because the module has two halves and they take
        different input: ten segmented plaque FIELDS, which is what the cpsam_plaque
        model was trained on, and ten whole plate FIGURES, which is what the pipeline
        actually consumes before it has found a well.

        The sample machinery is the shared example-dataset one, unchanged. What differs is what happens
        afterwards: Make Masks opens the folder in the editor, and this points ``src``
        at it.
        """
        from PySide6.QtWidgets import QPushButton

        from ..make_masks_datasets import install_dataset_button

        button = install_dataset_button(self, app_key="analyze_plaques",
                                        use=self.point_src_at)
        button.setText(tr("Load test data…"))
        button.setToolTip(tr(
            "Download ten example fields for Plaque Analysis and point src at them. "
            "Two sets to choose from: segmented plaque fields, which is what the "
            "plaque model was trained on, or whole plate figures, which is what the "
            "pipeline takes. Cached after the first download."))
        self._plaque_example_button = button
        section.add_prose(button, at_top=True)

    def _install_ops_example_button(self, section) -> None:
        """Add OPS's test-data control: two fields of a published screen.

        The sample is a well's last two sequencing fields with all
        eleven cycles and the guide library, so stitch, objects and decode all
        have something real to do. See :mod:`spacr.qt.ops_stitch_demo`.
        """
        from PySide6.QtWidgets import QPushButton

        button = QPushButton(tr("Load test data…"))
        button.setToolTip(tr(
            "Download about 390 MB from a published optical pooled screen: "
            "two sequencing fields of one well, all eleven cycles, and the "
            "guide library. The source, output folder, library and plate are "
            "filled in, so Run is the next action. There are no phenotype "
            "images, so the phenotype step is skipped. Cached afterwards."))
        button.clicked.connect(lambda: self.load_the_ops_example())
        self._ops_example_button = button
        section.add_prose(button, at_top=True)

    def load_the_ops_example(self, *, ask=None, folder=None) -> dict:
        """Fill the OPS settings with the test data, fetching it when needed.

        :param ask: replaces the downloader, for tests.
        :param folder: replaces the cache folder, for tests.
        :returns: the settings that were applied; empty while a download is
            still running or after a failure.
        """
        from ..ops_stitch_demo import load_the_test_data, ops_settings_for

        applied: dict = {}

        def use(where) -> None:
            """Apply the sample's settings and say so in the console."""
            values = ops_settings_for(where)
            self.apply_settings_dict(values)
            applied.update(values)
            self._console.append_stdout(
                tr("OPS test data ready: {path}", path=str(where)) + "\n")

        def report(text: str, _error: bool) -> None:
            """Put progress and failures in the console."""
            self._console.append_stdout(text + "\n")

        load_the_test_data(
            "ops", use=use, report=report,
            button=getattr(self, "_ops_example_button", None), parent=self,
            ask=ask, folder=folder)
        return applied

    def point_src_at(self, folder) -> bool:
        """Put ``folder`` in this module's ``src`` field. Returns whether it took.

        The counterpart of Make Masks' ``_open_folder`` for a module screen: a module
        does not open a folder, it runs on one. Reaches the widget the same way
        :meth:`_put_the_measure_example_in_place` does, so the two routes cannot drift
        on where ``src`` lives.

        :param folder: the folder to run on, as a path or a string.
        :returns: True when the screen has a ``src`` field and it took the value.
        """
        source = str(folder)
        model = getattr(self, "_settings_model", None)
        control = (model._widgets.get("src")
                   if model is not None and hasattr(model, "_widgets")
                   else None)
        if control is None or not hasattr(control, "setText"):
            return False
        control.setText(source)
        console = getattr(self, "_console", None)
        if console is not None:
            console.append_stdout(
                tr("Source directory (src): {path}", path=source) + "\n")
        _show_the_src_live(self)
        return True

    def _install_measure_example_button(self, section) -> None:
        """Add the example-data control that populates Measure's ``src``."""
        from PySide6.QtWidgets import QPushButton

        button = QPushButton(tr("Load test data…"))
        button.setToolTip(tr(
            "Download about 370 MB of example data for Measure: sixteen "
            "fields across four wells, already segmented, so Measure can be "
            "run without generating masks first. Cached afterwards, so "
            "pressing it again is instant."))
        button.clicked.connect(lambda: self.load_the_measure_example())
        self._measure_example_button = button
        section.add_prose(button, at_top=True)

    def measure_example_destination(self):
        """The shared example plate folder.

        The same one Mask and Annotate use: `merged/` and
        `measurements/measurements.db` are two halves of one plate, and
        downloading them into separate trees meant they could not be opened
        together.
        """
        from ..hf_download import example_plate_folder

        return example_plate_folder()

    def load_the_measure_example(self, *, ask=None) -> dict:
        """Download Measure's example plate and point ``src`` at it.

        :param ask: optional download function used in place of
            :func:`spacr.qt.hf_download.download_measure_example`.
        :returns: a mapping with the source directory, or empty when the
            download failed or was cancelled.
        """
        destination = self.measure_example_destination()
        destination.mkdir(parents=True, exist_ok=True)

        merged = destination / "merged"
        if merged.is_dir() and any(merged.glob("*.npy")):
            return self._put_the_measure_example_in_place(destination)

        button = getattr(self, "_measure_example_button", None)
        if button is not None:
            button.setEnabled(False)
            button.setText(tr("Fetching test data…"))

        placed: dict = {}

        def _done(result, error):
            """Restore the button whether the load worked or failed."""
            if button is not None:
                button.setEnabled(True)
                button.setText(tr("Load test data…"))
            if result is None:
                self._console.append_notice(
                    "[example] the measure example data was not "
                    "downloaded: {detail}\n", detail=error or "cancelled")
                return
            placed.update(self._put_the_measure_example_in_place(destination))

        download = ask
        if download is None:
            from ..hf_download import download_measure_example as download
        download(self, destination, _done)
        return placed

    def keep_the_src_openable(self, destination) -> str:
        """Take back a shipped ``src`` the panel cannot open. Return the value.

        THE RULE, once, for every route that applies example settings.

        A shipped settings file records the machine that GENERATED it, and
        `reanchor_example_paths` re-homes what it can. What it cannot resolve
        it deliberately leaves alone -- a template token, or a path whose
        folder name matches nothing here -- and that value then reaches the
        field verbatim -- which is how Mask Generation came to show the
        literal token ``<src>`` after loading the example images.

        A MORE SPECIFIC SHIPPED VALUE IS KEPT. Measure's example points `src`
        at the plate's `merged/` subfolder, and
        :meth:`reanchor_example_paths` records that collapsing that to the
        plate root "would quietly measure the wrong directory rather than
        fail". So this only ever replaces a value that is not a directory --
        never one that merely differs from ``destination``.

        `Path("")` IS THE WORKING DIRECTORY and its ``is_dir()`` is True, so
        an empty cell has to be rejected before the filesystem is asked or the
        run reads the cwd.

        :param destination: the folder the example was unpacked into.
        :returns: the value the field ends up holding.
        """
        from pathlib import Path

        fallback = str(Path(destination))
        model = getattr(self, "_settings_model", None)
        control = (model._widgets.get("src")
                   if model is not None and hasattr(model, "_widgets")
                   else None)
        if control is None or not hasattr(control, "setText"):
            return fallback
        applied = str(control.text() if hasattr(control, "text") else "")
        if applied.strip() and Path(applied).is_dir():
            return applied
        control.setText(fallback)
        self._console.append_stdout(
            tr("The example's recorded source does not exist here; "
               "using {path}", path=fallback) + "\n")
        return fallback

    def _put_the_measure_example_in_place(self, destination) -> dict:
        """Point ``src`` at the downloaded example and say so."""
        from pathlib import Path

        source = str(Path(destination))
        window = _window_of(self)
        live_was_on = _live_is_on(self)
        model = getattr(self, "_settings_model", None)
        control = (model._widgets.get("src")
                   if model is not None and hasattr(model, "_widgets")
                   else None)
        if control is not None and hasattr(control, "setText"):
            control.setText(source)
        self.apply_settings_that_came_with(destination)
        screen = _screen_after_the_load(self, window)
        screen._console.append_stdout(
            tr("Source directory (src): {path}", path=source) + "\n")
        kept = screen.keep_the_src_openable(destination)
        _show_the_src_live(screen, live_was_on=live_was_on)
        return {"src": kept}

    def _install_sequencing_example_button(self, section) -> None:
        """Add Map Barcodes' control for the published reads."""
        from PySide6.QtWidgets import QPushButton

        button = QPushButton(tr("Load test data…"))
        button.setToolTip(tr(
            "Download raw reads from the published screen (NCBI BioProject "
            "PRJNA1261935, four sequenced plates). You choose which runs and "
            "how many reads from each — the full set is about 20 GB, while a "
            "hundred thousand reads from each file is about 30 MB."))
        button.clicked.connect(lambda: self.load_the_sequencing_example())
        self._sequencing_example_button = button
        section.add_prose(button, at_top=True)

    def sequencing_example_destination(self):
        """Where the FASTQ goes: a reads folder beside the other examples."""
        from ..hf_download import example_plate_folder

        return example_plate_folder().parent / "sequencing"

    def load_the_sequencing_example(self, *, picker=None) -> dict:
        """Fetch published reads and point ``src`` at the folder they land in.

        :param picker: replaces the dialog, for tests.
        :returns: ``{"src": folder}`` when something was downloaded, else {}.
        """
        destination = self.sequencing_example_destination()
        destination.mkdir(parents=True, exist_ok=True)

        if picker is None:
            from ..widgets.sra_picker import SraPicker
            picker = SraPicker(destination, self)
        if hasattr(picker, "exec"):
            picker.exec()
        written = list(getattr(picker, "written", ()) or ())
        if not written:
            return {}

        source = str(destination)
        model = getattr(self, "_settings_model", None)
        control = (model._widgets.get("src")
                   if model is not None and hasattr(model, "_widgets")
                   else None)
        if control is not None and hasattr(control, "setText"):
            control.setText(source)
        self._console.append_stdout(
            tr("{count} read files ready: {path}",
               count=len(written), path=source) + "\n")
        return {"src": source, "files": written}

    def _install_annotate_example_button(self, section) -> None:
        """Add the example-data control for the Annotate/Classify set."""
        from PySide6.QtWidgets import QPushButton

        button = QPushButton(tr("Load test data…"))
        button.setToolTip(tr(
            "Download about 280 MB of example data: 2,341 single-cell crops "
            "with a measurements database, of which 88 are already labelled. "
            "Settings are filled in with it, so the module can be run "
            "straight away. Cached afterwards."))
        button.clicked.connect(lambda: self.choose_the_test_data())
        self._annotate_example_button = button
        section.add_prose(button, at_top=True)

    def _install_measurements_example_button(self, section) -> None:
        """Add the shared measurements-database example control.

        For a module that reads a finished plate -- its measurements database
        and the crops it indexes -- rather than one that makes either. The
        Annotate example is such a plate, so this reuses its download through
        :mod:`spacr.qt.widgets.measurements_example` and points ``src`` at
        the plate folder, without the crops-or-arrays chooser Classify asks
        through: a finished plate is the only half this module can read.
        """
        from ..widgets.measurements_example import install_test_data_button

        button = install_test_data_button(
            self, None, self._point_src_at_the_example,
            say=lambda message: self._console.append_stdout(message + "\n"))
        section.add_prose(button, at_top=True)

    def _point_src_at_the_example(self, folder, _database) -> dict:
        """Set ``src`` to the example plate folder and say so.

        :param folder: the example plate folder.
        :param _database: its measurements database, found from ``src``.
        :returns: ``{"src": folder}``.
        """
        self.apply_settings_dict({"src": str(folder)})
        self._console.append_stdout(
            tr("Example data ready: {path}", path=str(folder)) + "\n")
        return {"src": str(folder)}

    def choose_the_test_data(self, *, chooser=None, ask=None) -> dict:
        """Ask which half of the example plate to fetch, then fetch it.

        The same two routes Annotate offers, through the same dialog: the
        crops that are already cut, or the merged arrays they were cut from.
        Classify can train from either, and they differ by 110 MB, so the
        choice is worth describing before it is made rather than after.

        :param chooser: replaces the dialog, for tests.
        :param ask: replaces the downloader, for tests.
        :returns: the settings that were applied, or an empty mapping.
        """
        from ..widgets.test_data_chooser import TestDataChooser

        dialog = chooser if chooser is not None else TestDataChooser(self)
        if hasattr(dialog, "exec"):
            dialog.exec()
        route = str(getattr(dialog, "chosen", "") or "")
        if not route:
            return {}
        self._test_data_route = route
        if route == "stream":
            return self.load_the_measure_example(ask=ask)
        return self.load_the_annotate_example(ask=ask)

    def annotate_example_destination(self):
        """The shared example plate folder, which is where `data/` belongs."""
        from ..hf_download import example_plate_folder

        return example_plate_folder()

    def load_the_annotate_example(self, *, ask=None) -> dict:
        """Download the annotation example and fill this module's settings in.

        :param ask: optional download function used in place of
            :func:`spacr.qt.hf_download.download_annotate_example`.
        :returns: the settings that were applied, or an empty mapping.
        """
        destination = self.annotate_example_destination()
        destination.mkdir(parents=True, exist_ok=True)

        if (destination / "measurements" / "measurements.db").is_file():
            return self._apply_the_example_settings(destination)

        button = getattr(self, "_annotate_example_button", None)
        if button is not None:
            button.setEnabled(False)
            button.setText(tr("Fetching test data…"))

        applied: dict = {}

        def _done(result, error):
            """Restore the button whether the load worked or failed."""
            if button is not None:
                button.setEnabled(True)
                button.setText(tr("Load test data…"))
            if result is None:
                self._console.append_notice(
                    "[example] the annotation example was not downloaded: "
                    "{detail}\n", detail=error or "cancelled")
                return
            applied.update(self._apply_the_example_settings(destination))

        download = ask
        if download is None:
            from ..hf_download import download_annotate_example as download
        download(self, destination, _done)
        return applied

    def _apply_the_example_settings(self, destination) -> dict:
        """Load this module's example settings into the panel.

        THE POINT OF SHIPPING SETTINGS WITH THE DATA. The dataset is useless
        without knowing which column holds the labels, what size the crops are
        and which classes exist; making the user find that out first is most of
        the work the example was meant to save. The published file carries the
        answers, and this puts them in the panel so Run is the next action.
        """
        from pathlib import Path

        destination = Path(destination)
        window = _window_of(self)
        live_was_on = _live_is_on(self)
        applied = self.apply_settings_that_came_with(destination)
        screen = _screen_after_the_load(self, window)
        if not applied:
            screen._console.append_notice(
                "[example] no settings file for {app} in the example data\n",
                app=screen.app_key)
        screen._console.append_stdout(
            tr("Example data ready: {path}", path=str(destination)) + "\n")
        source = screen.keep_the_src_openable(destination)
        _show_the_src_live(screen, live_was_on=live_was_on)
        return {"src": source}

    def example_images_destination(self):
        """The shared example plate folder. See `hf_download.example_plate_folder`."""
        from ..hf_download import example_plate_folder

        return example_plate_folder()

    def load_the_example_images(self, *, ask=None) -> dict:
        """Download the example plate and populate the ``src`` setting.

        :param ask: Optional download function used in place of
            :func:`spacr.qt.hf_download.download_toxo_mito_demo`.
        :returns: A mapping containing the selected source directory and the
            downloaded settings path. Returns an empty mapping if the
            download fails or has not completed.
        """
        destination = self.example_images_destination()
        destination.mkdir(parents=True, exist_ok=True)

        plate = destination
        if plate.is_dir() and any(plate.glob("*.tif")):
            return self._put_the_example_images_in_place(plate, None)

        button = getattr(self, "_example_images_button", None)
        if button is not None:
            button.setEnabled(False)
            button.setText(tr("Fetching test data…"))

        placed = {}

        def done(result, error):
            """Re-enable the button whether the download worked or failed."""
            if button is not None:
                button.setEnabled(True)
                button.setText(tr("Load test data…"))
            if result is None:
                self._console.append_stdout(
                    tr(
                        "The example images could not be downloaded: "
                        "{error}",
                        error=error or tr("unknown error"),
                    ) + "\n"
                )
                return
            placed.update(self._put_the_example_images_in_place(
                result.dataset_path, result.settings_path))

        if ask is None:
            from ..hf_download import _MaskTarWorker, download_toxo_mito_demo

            def ask(parent, dest, on_done):
                """Start the demo download with the tar-aware worker."""
                download_toxo_mito_demo(parent, dest, on_done,
                                        worker_factory=_MaskTarWorker)
        ask(self, str(destination), done)
        return placed

    def _put_the_example_images_in_place(self, images, settings) -> dict:
        """Apply the shipped settings, then write the fetched folder into `src`.

        In that order, and on the screen that holds the form AFTERWARDS: a
        pack that reshapes the form replaces this screen, so the field and
        console are looked up once the settings are in (item 514).
        """
        window = _window_of(self)
        live_was_on = _live_is_on(self)
        self.apply_settings_that_came_with(images, pack_folder=settings)
        screen = _screen_after_the_load(self, window)

        screen._console.append_stdout(
            tr("Source directory (src): {path}", path=str(images)) + "\n"
        )
        if settings is not None:
            screen._console.append_stdout(
                tr(
                    "Compatible example settings: {path}",
                    path=str(settings),
                ) + "\n"
            )
        model = getattr(screen, "_settings_model", None)
        control = (model._widgets.get("src")
                   if model is not None and hasattr(model, "_widgets")
                   else None)
        if control is not None and hasattr(control, "setText"):
            control.setText(str(images))
        _show_the_src_live(screen, live_was_on=live_was_on)
        return {"src": str(images),
                "settings": str(settings) if settings else ""}

    def load_the_example_screen(self, *, download: bool = True,
                               kind=None) -> dict:
        """Fetch the example screen and fill `count_data` and `score_data`.

        :param download: If ``True``, download files that are not already in
            the local cache.
        :returns: A mapping of populated setting names to their file paths.
        """
        from ...example_data import ExampleDataError, fetch, missing

        button = getattr(self, f"_example_{kind}_button", None) if kind else None
        if button is None:
            button = getattr(self, "_example_data_button", None)
        absent = missing(kind=kind)
        if absent and button is not None:
            button.setEnabled(False)
            button.setText(tr("Fetching {count} file(s)\u2026",
                              count=len(absent)))
        try:
            got = fetch(download=download, kind=kind,
                        progress=self._say_the_download_is_moving)
        except ExampleDataError as error:
            self._console.append_stdout(f"{error}\n")
            return {}
        finally:
            if button is not None:
                button.setEnabled(True)
                button.setText(tr("Load test data…"))

        table = self._settings_model._widgets.get("paired_data")
        added = 0
        if table is not None and hasattr(table, "add_paths_for_side"):
            added += int(table.add_paths_for_side(list(got.scores), "score"))
            added += int(table.add_paths_for_side(list(got.counts), "count"))
        else:                                                # pragma: no cover
            added = self.apply_settings_dict(
                {"count_data": list(got.counts),
                 "score_data": list(got.scores)})
        self._console.append_stdout(
            f"{got.note()} Paired {len(got.scores)} score table(s) with "
            f"{len(got.counts)} count table(s) in Input Tables.\n")
        return {"counts": got.counts, "scores": got.scores,
                "applied": added, "folder": got.folder}

    def _say_the_download_is_moving(self, name, seen, total) -> None:
        """Progress on the button itself, which is where the user is looking."""
        button = getattr(self, "_example_data_button", None)
        if button is None or not total:
            return
        button.setText(f"{name} — {100 * seen // max(total, 1)}%")
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()

    def refresh_maturity_visibility(self) -> None:
        """Show/hide Alpha and Beta settings without discarding typed values.

        ALSO THE ONE PLACE A SECTION'S VISIBILITY IS DECIDED. Maturity is not
        the only reason a category is not on the form -- a category every one
        of whose settings needs a z axis is not on a flat plate's form
        either -- and two functions each calling ``setVisible`` on the same
        card is how a card comes back the next time Preferences is saved.
        So the dimension switches are answered here as well, and the settings
        search hands visibility back to this method for the same reason. A
        heading the object rule holds back, because the run has none of its
        objects, stays hidden here too.

        The notice below still speaks only for maturity: a category the 3D
        switch is holding back is not "hidden by Preferences", and saying so
        would send the user to a dialog that cannot bring it back.
        """
        from ..preferences import maturity_is_visible

        hidden_stages = set()
        gated = (self._dimension_hidden_sections()
                 | self._headings_the_run_lacks())
        for section in self.rendered_settings_sections():
            visible = maturity_is_visible(section.maturity())
            section.setVisible(visible and id(section) not in gated)
            if not visible:
                hidden_stages.add(section.maturity())
        self._apply_dimension_visibility()

        notice = getattr(self, "_maturity_notice", None)
        if notice is None:
            return
        if hidden_stages:
            stages = [stage for stage in ("alpha", "beta")
                      if stage in hidden_stages]
            labels = (tr("Alpha and Beta") if len(stages) > 1
                      else tr(stages[0].title()))
            notice.setText(tr(
                "{stages} settings are hidden by Preferences. Enable them "
                "in Preferences \u2192 Feature maturity.", stages=labels))
            notice.show()
        else:
            notice.hide()

    def _install_dimension_switches(self, row, toggle_cls) -> dict:
        """Put the 3D and Time switches in the action row, left of Live.

        A switch is built only where it has something to reveal: the screen
        is one of :data:`DIMENSION_TOGGLE_APPS` and its form actually renders
        at least one setting of that dimension. A toggle whose press changes
        nothing on screen is worse than no toggle, and the check is asked of
        the built form rather than of a list, so the day Measure stops
        offering voxel geometry its 3D switch stops being drawn.

        :param row: the action row's layout.
        :param toggle_cls: :class:`spacr.qt.widgets.AiToggleLabel`, passed in
            because the caller has already imported it.
        :returns: ``dimension -> toggle`` for the switches that were built.
        """
        self._dimension_switches = {}
        if str(self.app_key) not in DIMENSION_TOGGLE_APPS:
            return self._dimension_switches
        offered = {setting_dimension(key)
                   for _section, key, _field in self._dimension_rows()}
        offered.update(
            setting_dimension(key)
            for key in (getattr(self, "_waiting_heading_of", None) or {})
            if setting_dimension(key))
        for dimension, label, tooltip in DIMENSION_TOGGLES:
            if dimension not in offered:
                continue
            switch = toggle_cls(text=label, tooltip=tooltip)
            switch.setMinimumWidth(DIMENSION_TOGGLE_MIN_PX)
            switch.setChecked(bool(self._dimension_on.get(dimension)))
            switch.toggled.connect(partial(self._on_dimension_switch,
                                           dimension))
            row.addWidget(switch)
            self._dimension_switches[dimension] = switch
        if self._dimension_switches:
            self.refresh_maturity_visibility()
        return self._dimension_switches

    def _dimension_is_gated(self, dimension: str) -> bool:
        """Whether ``dimension``'s settings are being held back right now.

        A dimension is held back only by a switch that EXISTS and is off.
        Nothing is ever hidden on a screen with no switch to show it again:
        the Timelapse module's own screen renders the same z and t settings
        Mask does, and without this its acquisition categories would go
        with nothing on the page able to bring them back.
        """
        return bool(self.dimension_switch(dimension) is not None
                    and not self.dimension_is_on(dimension))

    def dimension_switch(self, dimension: str):
        """The 3D or Time toggle, or None on a screen that carries neither.

        :param dimension: ``"z"`` for the 3D toggle or ``"t"`` for the Time
            toggle.
        """
        return (getattr(self, "_dimension_switches", None) or {}).get(
            str(dimension))

    def dimension_is_on(self, dimension: str) -> bool:
        """Whether this screen is showing ``dimension``'s settings now.

        :param dimension: ``"z"`` (3D) or ``"t"`` (time); a dimension this
            screen does not track reads as off.
        """
        return bool((getattr(self, "_dimension_on", None) or {}).get(
            str(dimension)))

    def set_dimension(self, dimension: str, on: bool) -> None:
        """Switch a dimension's settings on or off.

        Driven through the toggle where there is one, so the switch never
        shows a state the form does not have; a screen built without the
        action row still moves, which is what lets the settings panel be
        gated before the row that gates it exists.

        :param dimension: ``"z"`` (3D) or ``"t"`` (time); a dimension this
            screen does not track is ignored.
        :param on: ``True`` to show the dimension's settings, ``False`` to
            hide them.
        """
        dimension = str(dimension)
        if dimension not in (getattr(self, "_dimension_on", None) or {}):
            return
        switch = self.dimension_switch(dimension)
        if switch is not None:
            switch.setChecked(bool(on))
            return
        self._on_dimension_switch(dimension, on)

    def _on_dimension_switch(self, dimension: str, on: bool) -> None:
        """Record the new state and re-decide what the form shows."""
        self._dimension_on[str(dimension)] = bool(on)
        self.refresh_maturity_visibility()

    def _dimension_rows(self) -> list:
        """``[(section, key, field), …]`` for each dimensional form row.

        Built from the model's ``key -> widget`` map and the sections the
        screen kept, the way the settings search builds its own index: the
        screen has already decided which key went where, and a second
        opinion here would be a second thing to keep in sync.

        Only the rows that HAVE a dimension are returned -- the answer is a
        few dozen rows out of several hundred, and every caller wants that
        subset.
        """
        from PySide6.QtWidgets import QFormLayout

        model = getattr(self, "_settings_model", None)
        widgets = getattr(model, "_widgets", None) or {}
        if not widgets or str(self.app_key) not in DIMENSION_TOGGLE_APPS:
            return []
        built = getattr(widgets, "built_items", None)
        pairs = built() if callable(built) else widgets.items()
        by_widget = {id(widget): key for key, widget in pairs}
        found = []
        for section in getattr(self, "_settings_sections", []) or []:
            form = getattr(section, "_form", None)
            if not isinstance(form, QFormLayout):
                continue
            for index in range(form.rowCount()):
                item = form.itemAt(index, QFormLayout.FieldRole)
                field = item.widget() if item is not None else None
                key = by_widget.get(id(field)) if field is not None else None
                if key and setting_dimension(key):
                    found.append((section, key, field))
        return found

    def _dimension_hidden_sections(self) -> set:
        """``id()`` of every category the switches are holding back.

        A category is held back only when EVERY row it has is dimensional
        and every one of those dimensions is off -- the Volumetric
        Processing card, which is nothing but z settings. A category that
        merely contains a few of them, as Measure's Mask & Channel Mapping
        contains ``timelapse``, keeps its card and loses those rows; hiding
        the whole card would take the channel mapping with it.
        """
        from PySide6.QtWidgets import QFormLayout

        by_section: dict = {}
        for section, key, _field in self._dimension_rows():
            by_section.setdefault(id(section), [section, []])[1].append(key)
        hidden = set()
        for marker, (section, keys) in by_section.items():
            form = getattr(section, "_form", None)
            if not isinstance(form, QFormLayout):
                continue
            if form.rowCount() != len(keys):
                continue
            if all(self._dimension_is_gated(setting_dimension(key))
                   for key in keys):
                hidden.add(marker)
        if str(self.app_key) not in DIMENSION_TOGGLE_APPS:
            return hidden
        for section in getattr(self, "_settings_sections", []) or []:
            spec = getattr(section, "_spacr_waiting_spec", None)
            if spec is None or getattr(spec, "children", ()):
                continue
            keys = [self._key_of_row(widget) for _label, widget in spec[1]]
            if keys and all(
                    key and setting_dimension(key)
                    and self._dimension_is_gated(setting_dimension(key))
                    for key in keys):
                hidden.add(id(section))
        return hidden

    def _apply_dimension_visibility(self) -> None:
        """Show or hide each dimensional ROW to match the switches.

        Rows rather than whole cards, because the dimensional settings are
        not all in dimensional cards: Measure keeps ``timelapse`` and
        ``timelapse_objects`` beside the channel mapping, where they belong
        for every other purpose.

        Hidden through ``QFormLayout.setRowVisible``, which is what reaches
        the label as well as the field -- a section builds the label side
        into a wrapper it does not hand back, so hiding the field alone
        leaves the caption stranded on an empty row. The two helpers come
        from the settings search, which hides rows for its own reason: one
        implementation, so a row hidden by a filter and a row hidden by a
        dimension are hidden the same way.

        Never touches a row that has no dimension, so anything else that
        decides a row's visibility keeps its own answers.
        """
        try:
            from ..settings_search import _set_row_visible
        except Exception:                                    # noqa: BLE001
            LOG.debug("no row-visibility helper; leaving the form alone",
                      exc_info=True)
            return
        for section, key, field in self._dimension_rows():
            _set_row_visible(
                section, field,
                not self._dimension_is_gated(setting_dimension(key)))

    def setting_row_is_visible(self, key: str) -> bool:
        """Whether ``key``'s row is currently on the form.

        The read-back the switches are checked against: "visible" for a
        setting is the state of its ROW, not of its widget, because the
        widget of a hidden row is still there holding the value it had.
        False for a key this screen does not render at all.

        ASKED OF THE FORM, NOT OF THE SCREEN. ``isVisible`` is false for
        everything on a page that has not been shown yet, which would make
        this answer "hidden" for the entire settings panel of a module the
        user has not opened. ``isHidden`` and ``QFormLayout.isRowVisible``
        answer what was hidden ON PURPOSE, which is the question.

        :param key: settings key of the row; a category still waiting to be
            built is built first so its row can be read.
        """
        from PySide6.QtWidgets import QFormLayout

        self._open_the_heading_of(str(key))
        model = getattr(self, "_settings_model", None)
        field = (getattr(model, "_widgets", None) or {}).get(str(key))
        if field is None:
            return False
        for section in getattr(self, "_settings_sections", []) or []:
            form = getattr(section, "_form", None)
            if not isinstance(form, QFormLayout):
                continue
            for index in range(form.rowCount()):
                item = form.itemAt(index, QFormLayout.FieldRole)
                if item is not None and item.widget() is field:
                    from ..settings_search import _row_is_visible
                    return bool(not section.isHidden()
                                and _row_is_visible(section, field))
        return False

    def _sync_dimension_switches(self, settings: dict) -> tuple:
        """Move the 3D and Time switches to match settings just applied.

        A settings file that turns ``z_stack`` on is a file about a
        volumetric run, and applying it with the 3D switch off would fill in
        every voxel size and then hide all of them -- the import would read
        as having done nothing. See :data:`DIMENSION_GATE_KEYS` for why
        these particular keys are the announcement.

        A file that says nothing about a dimension leaves that switch where
        the user put it: absence is not a request to hide anything.

        :param settings: the dict that was just applied.
        :returns: the dimensions the settings switched on.
        """
        turned_on = []
        for dimension, gate_keys in DIMENSION_GATE_KEYS.items():
            if dimension not in (getattr(self, "_dimension_on", None) or {}):
                continue
            named = [key for key in gate_keys if key in settings]
            if not named:
                continue
            wanted = any(self._truthy(settings.get(key)) for key in named)
            self.set_dimension(dimension, wanted)
            if wanted:
                turned_on.append(dimension)
        return tuple(turned_on)

    def _attach_column_picker(self, key, widget) -> None:
        """Give a column-name field its "SQL" button; a no-op for anything else.

        :param key: the settings key this widget collects, or None.
        :param widget: the input widget already installed in its Section.
        """
        if key not in COLUMN_TABLES:
            return
        from .settings_model import _CsvColumnField

        if isinstance(widget, _CsvColumnField):
            return
        from ..widgets.class_editor import ClassEditorWidget

        if isinstance(widget, ClassEditorWidget):
            widget.attach_sql_picker(self._settings_src_path,
                                     COLUMN_TABLES[key] or "png_list")
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
        """Return a compact source selection and test-data guidance card, or None.

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

        existing = self._settings_src_path()
        placeholders = {"", "path", "/path/to/src", "/path"}
        if existing and existing not in placeholders:
            return None

        title = tr(
            "Point {module} at some data",
            module=tr(APP_TITLES.get(self.app_key, self.app_key)).lower(),
        )
        subtitle = tr(
            "Drop a folder of images anywhere on this window or type a path "
            "into the Source field below. Use Load test data when available, "
            "or open Pipeline overviews on Home to choose a walkthrough."
        )
        card = EmptyState(
            title=title, subtitle=subtitle,
            cta_label="Choose source data",
            on_action=lambda: self.choose_source_folder(),
        )
        if isinstance(src_widget, QLineEdit):
            src_widget.textChanged.connect(self._maybe_hide_empty_state)
        else:
            changed = getattr(src_widget, "value_changed", None)
            if changed is not None and hasattr(changed, "connect"):
                changed.connect(self._refresh_empty_state)
        card.setObjectName(EMPTY_STATE_NAME)
        return card

    def _refresh_empty_state(self) -> None:
        """Show or hide the card from what ``src`` holds right now.

        Both directions, unlike :meth:`_maybe_hide_empty_state`, because a
        set can be emptied: removing the last database returns the screen to
        exactly the state the card describes, and leaving it hidden would
        leave a user with no data and nothing telling them how to get some.
        """
        card = getattr(self, "_empty_state_card", None)
        if card is None:
            return
        existing = self._settings_src_path()
        placeholders = {"", "path", "/path/to/src", "/path"}
        card.setVisible(not existing or existing in placeholders)

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
        if getattr(self, "_live_preview_card", None) is None:
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
        self._wire_live_preview_naming()

    def _install_plaque_mode(self) -> None:
        """Put Plaque Assay's Plaque | Figure switch on the settings column.

        Never raises: a screen without its switch still runs in the mode its
        form says.
        """
        try:
            from ..widgets.plaque_preview import install_plaque_mode

            install_plaque_mode(self)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not install the plaque mode switch",
                      exc_info=True)

    def _wire_live_preview_naming(self) -> None:
        """Regroup the preview's table when the file naming changes.

        ``metadata_type`` and ``custom_regex`` decide how the preview groups
        a folder's files into fields and channels. Changing either after
        loading left the table grouped the old way, which for a folder the
        old naming cannot read is every file under a single column. The same
        400 ms wait as the ``src`` field, so a pattern typed a character at a
        time is read once.
        """
        widgets = getattr(self._settings_model, "_widgets", {}) or {}
        if self._part_is_owed(_LIVE_PREVIEW):
            regroup = self._regroup_the_live_preview
        else:
            panel = getattr(self, "_live_preview", None)
            regroup = getattr(panel, "regroup_the_folder", None)
            if panel is None or not callable(regroup):
                return
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(400)
        timer.timeout.connect(regroup)
        self._live_naming_timer = timer
        for key in ("metadata_type", "custom_regex"):
            widget = widgets.get(key)
            if widget is None:
                continue
            for name in ("currentIndexChanged", "textChanged",
                         "value_changed"):
                signal = getattr(widget, name, None)
                if signal is None:
                    continue
                try:
                    signal.connect(lambda *_args: timer.start())
                except Exception:                            # noqa: BLE001
                    continue
                break

    def _maybe_hide_empty_state(self, text: str) -> None:
        """Hide the empty-state card once the source field names something real.

        :param text: the field's current text. The placeholders it ships with
            do not count as a source -- hiding the card for ``/path/to/src``
            would take the instruction away while nothing had been chosen.
        """
        card = getattr(self, "_empty_state_card", None)
        if card is None:
            return
        t = (text or "").strip()
        placeholders = {"", "path", "/path/to/src", "/path"}
        if t and t not in placeholders:
            card.hide()

    def _regroup_the_live_preview(self) -> None:
        """Regroup a live preview that exists; nothing to do for one that does not.

        A panel still waiting to be built has loaded no folder, so it has
        nothing to regroup, and the folder it loads when it is built is read
        with the naming the form holds then -- the same grouping a
        regrouping now would have produced.
        """
        if self._part_is_owed(_LIVE_PREVIEW):
            return
        panel = getattr(self, "_live_preview", None)
        regroup = getattr(panel, "regroup_the_folder", None)
        if callable(regroup):
            regroup()

    def _autoload_live_preview(self, src: str) -> None:
        """Ask the preview panel to discover/decode ``src`` asynchronously.

        Silent if ``src`` is empty or a placeholder. Directory traversal and
        image decoding both happen in the panel's worker so a large plate or
        slow NAS mount cannot freeze Qt.

        A panel that is not built yet is not built for this: the source is
        kept and loaded when the panel is, which is the first time the card
        is shown -- typing a path must not cost the preview's construction.
        """
        if self._part_is_owed(_LIVE_PREVIEW):
            self.__dict__["_live_src_waiting"] = src
            return
        panel = getattr(self, "_live_preview", None)
        if panel is None:
            return
        s = (src or "").strip()
        if not s or s in {"path", "/path/to/src", "/path"}:
            return
        panel.load_source_async(s)

    def choose_source_folder(self) -> str:
        """Ask for the run's source folder and put it in the src field.

        :returns: the folder chosen, or ``""`` if the dialog was cancelled
            or the screen has no src field to write to.
        """
        from PySide6.QtWidgets import QFileDialog, QLineEdit

        widgets = getattr(self._settings_model, "_widgets", None) or {}
        if "src" not in widgets:
            return ""
        chosen = QFileDialog.getExistingDirectory(
            self, tr("Choose source data"), self._settings_src_path() or "")
        if not chosen:
            return ""
        setter = getattr(self._settings_model, "set_value_for_key", None)
        if callable(setter):
            setter("src", chosen)
        else:                                                # pragma: no cover
            widget = widgets.get("src")
            if isinstance(widget, QLineEdit):
                widget.setText(chosen)
        self._refresh_empty_state()
        return chosen


    def eventFilter(self, obj, event):
        """Show/hide the hover tooltip and update the hint strip on Enter/Leave.

        :param obj: the watched widget; its ``settingsCategory`` or
            ``settingKey`` property decides whether a category blurb or a
            setting's help is shown.
        :param event: the filtered event; a ``ToolTip`` on a widget with
            hover help is swallowed, ``Enter`` shows the help and ``Leave``
            hides it, and every event is otherwise passed to the base class.
        """
        event_type = event.type()
        if event_type == QEvent.ToolTip:
            if hasattr(self, "_hint_strip") and (
                    obj in self._hint_map or obj.property("settingKey")):
                return True
        if event_type not in (QEvent.Enter, QEvent.Leave):
            return super().eventFilter(obj, event)
        from ..widgets.hover_tooltip import HoverTooltip
        category = obj.property("settingsCategory")
        if category:
            if event_type == QEvent.Enter:
                self.show_category_hint(str(category))
            else:
                self.clear_category_hint()
            return super().eventFilter(obj, event)
        if event_type == QEvent.Enter:
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
            from ..preferences import (get_tooltips_bottom_enabled,
                                       get_tooltips_box_enabled)
            want_bottom = get_tooltips_bottom_enabled()
            want_box = get_tooltips_box_enabled()
            shown_at_the_bottom = False
            if hint and want_bottom and hasattr(self, "_hint_strip"):
                link = ""
                if key:
                    try:
                        from .settings_model import api_docs_url
                        link = api_docs_url(self.app_key, str(key))
                    except Exception:                        # noqa: BLE001
                        link = ""
                self._hinted_widget = obj
                self._hinted_html = html
                self._write_hint(hint, link, hold=True,
                                 animated=_setting_has_an_animation(key))
                shown_at_the_bottom = True
            if html and (want_box or not shown_at_the_bottom):
                HoverTooltip.instance().show_for(obj, html)
        else:
            HoverTooltip.instance().start_hide()
        return super().eventFilter(obj, event)

    def show_module_hint(self, key: str, summary: str = "") -> bool:
        """Explain a MODULE in this screen's strip, for a dock hover.

        The window routes a dock hover to whichever page is in front (see
        `MainWindow._show_module_hint`), so on a module screen it arrives
        here. The strip is the same one the per-setting help writes to --
        deliberately, because it is the bottom of the window either way and a
        second strip stacked under it would be two places to look.

        THE HOLD IS THE MODULE ONE, not the setting one: thirty seconds
        rather than ten, because these links leave the application. A hovered
        SETTING overwrites this the moment the pointer reaches the form, and
        that is the right precedence -- the reader has moved on.

        :param key: the module to explain.
        :param summary: the sentence, already resolved and translated by
            `MainWindow._show_module_hint`. It has to arrive from there:
            `module_summary` falls back to the registry's English
            description, and a screen has no registry to look one up in.
        :returns: whether anything was written.
        """
        key = str(key or "")
        strip = getattr(self, "_hint_strip", None)
        summary = str(summary or "").strip()
        if not key or strip is None or not summary:
            return False
        try:
            from ..tutorials import tutorial_url
            from .settings_model import api_docs_url
        except Exception:                                        # noqa: BLE001
            return False
        from html import escape as _escape

        from ..i18n import tr as _tr
        words = []
        api = api_docs_url(key)
        if api:
            words.append(f'<a href="{_escape(api, quote=True)}">'
                         f'{_escape(_tr("API"))}</a>')
        lesson = tutorial_url(key)
        if lesson:
            words.append(f'<a href="{_escape(lesson, quote=True)}">'
                         f'{_escape(_tr("Tutorial"))}</a>')
        lines = HINT_STRIP_LINES - (1 if words else 0)
        fitted = _fit_to_lines(summary, strip, max(1, lines))
        strip.setText(
            f"{_escape(fitted)}<br>{'&nbsp;&nbsp;'.join(words)}" if words
            else _escape(fitted))
        strip.setToolTip(summary)
        self._hold_the_hint(True, self.MODULE_HINT_HOLD_MS)
        return True

    #: How long a MODULE stays in the strip, in milliseconds. Thirty seconds -- three times the per-setting hold
    #: because the API and Tutorial words open a browser, which is a larger
    #: decision than reaching for an animation.
    MODULE_HINT_HOLD_MS = 30_000

    def _default_hint(self) -> str:
        """The prompt the per-setting strip falls back to, in the UI language.

        Translated HERE rather than left to the language pass, because a
        pointer leaving a setting writes this back over whatever the pass
        rendered: the strip was Swedish until the first hover and English
        from then on.
        """
        return tr("Hover any setting for details and a link to its "
                  "documentation.")

    def _owe_part(self, part: str,
                  build: Callable[[], Optional[QWidget]]) -> None:
        """Record that ``part`` is built by ``build`` on its first use.

        See :class:`_BuiltOnFirstUse`, which does the building when one of
        the part's attributes is read or assigned.
        """
        owed = self.__dict__.get("_parts_owed")
        if owed is None:
            owed = self.__dict__["_parts_owed"] = {}
        owed[part] = build

    def _if_built(self, name: str):
        """The attribute ``name`` if it exists yet, never building a part.

        For teardown: a panel that was never built has nothing to shut
        down, and building 740 widgets to close them again is the worst
        moment to do it.
        """
        for klass in type(self).__mro__:
            slot = klass.__dict__.get(name)
            if isinstance(slot, _BuiltOnFirstUse):
                return slot.peek(self)
        return getattr(self, name, None)

    def _part_is_owed(self, part: str) -> bool:
        """Whether ``part`` has been deferred and not built yet."""
        return part in (self.__dict__.get("_parts_owed") or {})

    def _build_owed_part(self, part: str) -> bool:
        """Build ``part`` now if it is still owed.

        The part is struck off BEFORE it is built, so the builder assigns
        its attributes as plain ones instead of asking for itself again.
        The builder returns the root of what it built, or ``None`` when it
        fell back to building nothing. What follows keeps the ORDER an
        eager screen had: the surface sweep its construction ran, then
        whatever was queued with :meth:`_after_part_is_built` (Regression's
        fold strip adding the Hits tab, which came after that sweep), then
        the language pass and the polish, which reached the Hits tab too.

        :returns: ``True`` when this call built it.
        """
        owed = self.__dict__.get("_parts_owed") or {}
        build = owed.pop(part, None)
        if build is None:
            return False
        from .. import timing

        with timing.span("build deferred part", part):
            root = build()
            if root is not None:
                self._clear_a_late_parts_surfaces(root)
            waiting = (self.__dict__.get("_after_parts") or {}).pop(part, [])
            for callback in waiting:
                try:
                    callback()
                except Exception:                            # noqa: BLE001
                    LOG.exception("could not finish a deferred %s", part)
            if root is not None:
                self._translate_a_late_part(root)
        return True

    def _after_part_is_built(self, part: str, callback) -> bool:
        """Run ``callback`` once ``part`` is built, if it is still owed.

        For code that decorates a deferred part from outside the screen --
        Regression's fold strip adds its Hits tab to the results panel --
        and would otherwise build the part just to decorate it.

        :returns: ``True`` when the callback was queued; ``False`` when the
            part is not owed, in which case the caller does its work now.
        """
        if not self._part_is_owed(part):
            return False
        waiting = self.__dict__.get("_after_parts")
        if waiting is None:
            waiting = self.__dict__["_after_parts"] = {}
        waiting.setdefault(part, []).append(callback)
        return True

    def _results_panel_if_built(self):
        """Regression's results panel if it exists yet, WITHOUT building it.

        For a question an unbuilt panel answers the same way a fresh one
        would -- which run is loaded, when none can be -- so asking it does
        not cost the ~740 widgets the deferral saved.
        """
        return self._if_built("_results_panel")

    def _when_results_are_built(self, callback) -> bool:
        """Run ``callback`` once Regression's results panel exists.

        :returns: ``True`` when it was queued behind the deferred build;
            ``False`` when the panel is built already or never will be, and
            the caller should go ahead now.
        """
        return self._after_part_is_built(_REGRESSION_RESULTS, callback)

    def _clear_a_late_parts_surfaces(self, root: QWidget) -> None:
        """Make a late part's layout containers transparent, as opening did.

        The screen's construction swept its whole tree once
        (``_clear_page_surfaces``) and took tab scroll arrows off; a part
        built later missed both. The sweep is run from the part's PARENT
        because it tags the descendants of what it is given, and the
        part's own root is one of the containers it tags.

        Never raises: a part that keeps its fill still works.
        """
        try:
            from ..theme import (clear_container_surfaces,
                                 take_the_scroll_arrows_off)

            clear_container_surfaces(root.parentWidget() or root)
            take_the_scroll_arrows_off(root)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not clear a late part's surfaces",
                      exc_info=True)

    def _translate_a_late_part(self, root: QWidget) -> None:
        """Polish and translate a late part, as opening the screen did.

        THE POLISH IS THE SHEET, AND IT COMES FIRST. The page already
        carries the stylesheet, so nothing has to be applied, but an eager
        screen had its widgets polished before the language pass reached
        them -- laying the page out at construction asks every size hint,
        and a size hint polishes -- and hidden tab pages polished at the
        page's first show. ``ensurePolished`` walks the part the same way.
        The ORDER matters beyond the look: :mod:`spacr.qt.button_roles`
        classifies a button by its visible text when it is polished, and a
        Swedish "Kör förhandsgranskning" is not an English "Run", so a
        button translated first loses its Run colour.

        Then the language pass and the move of field help onto captions,
        which ``MainWindow`` ran once over the screen when it built it --
        the help move over the whole screen, because it keeps its event
        filter on the root it is handed and the screen's is the one every
        other caption uses.

        Never raises: a part in the wrong language still works.
        """
        _run_to_the_end(self._translate_a_late_part_steps(root))

    def _translate_a_late_part_steps(self, root):
        """:meth:`_translate_a_late_part`, a step at a time.

        The polish is taken one direct child of ``root`` at a time, then
        ``root`` itself, which polishes exactly what polishing ``root`` alone
        would (each child's whole subtree, then what is left), in the same
        order relative to the language pass. The language pass is taken the
        same way: each child's subtree in full, then ``root`` with
        ``only_new``, which visits what the children's passes did not -- the
        widgets they stamped are the ones it skips.
        """
        try:
            children = [child for child in root.children()
                        if isinstance(child, QWidget)]
        except RuntimeError:
            return
        for child in children:
            try:
                child.ensurePolished()
            except RuntimeError:
                pass
            yield
        try:
            root.ensurePolished()
        except RuntimeError:
            return
        yield
        try:
            from ..i18n import retranslate_widget_tree
            from .settings_model import retarget_field_tooltips

            for child in children:
                try:
                    retranslate_widget_tree(child)
                except RuntimeError:
                    pass
                yield
            retranslate_widget_tree(root, only_new=True)
            yield
            retarget_field_tooltips(self)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not translate a late part", exc_info=True)

    def _build_regression_results(self) -> Optional[QWidget]:
        """Build Regression's results tabs and figure pages into their card.

        Deferred from the screen's open to the first time the Figures card
        is shown or any of these attributes is used -- about 740 widgets
        that nobody can see until a run has results. See
        :class:`_BuiltOnFirstUse`. A failure falls back to the figure
        queue, as the eager build did. Dependencies are imported together
        before constructing the panels, so opening the screen does not
        load their data libraries and a missing dependency leaves no
        partially constructed results widgets.
        """
        try:
            from ..widgets.regression_results import RegressionResultsPanel
            from ..preferences import get_figure_grid_size
            from ..widgets.figure_grid_view import (
                MAX_CELL_PX, MIN_CELL_PX, FigureGridView)
            from ..widgets.sweep_runs import SweepRunsPanel
            from ..widgets.measurement_scan_panel import (
                MeasurementScanPanel)
            from ..widgets.sweep_panel import SweepPanel
            from ..widgets.cell_montage_view import CellMontageView

            self._results_panel = RegressionResultsPanel(
                self._figures_card, external_volcano=True)
            self._results_panel.refit_requested.connect(self._on_refit)

            self._figure_grid = FigureGridView(self._figures_card)
            self._figure_grid.figure_activated.connect(
                self._open_figure_from_grid)
            self._figure_grid.figure_menu_requested.connect(
                self._figure_grid_menu)

            detail = QWidget(self._figures_card)
            detail_layout = QVBoxLayout(detail)
            detail_layout.setContentsMargins(0, 0, 0, 0)
            detail_layout.setSpacing(4)
            back = QPushButton("← All figures")
            back.setFlat(True)
            back.setToolTip("Back to the grid of every figure this run "
                            "produced.")
            back.clicked.connect(self._show_figure_grid)
            row = QHBoxLayout()
            row.addWidget(back)
            row.addStretch(1)
            detail_layout.addLayout(row)
            detail_layout.addWidget(self._queue_the_results_hold, 1)
            self._figure_detail = detail

            volcano_page = QWidget(self._figures_card)
            volcano_layout = QVBoxLayout(volcano_page)
            volcano_layout.setContentsMargins(0, 0, 0, 0)
            volcano_layout.setSpacing(4)
            back_to_grid = QPushButton("← All figures")
            back_to_grid.setFlat(True)
            back_to_grid.clicked.connect(self._show_figure_grid)
            volcano_row = QHBoxLayout()
            volcano_row.addWidget(back_to_grid)
            volcano_row.addStretch(1)
            volcano_layout.addLayout(volcano_row)
            from ..widgets.collapsible_splitter import (
                EDGE, CollapsibleSplitter)
            gene_split = CollapsibleSplitter(Qt.Vertical, volcano_page)
            gene_split.add_pane(self._results_panel.volcano, "Volcano",
                                stretch=3)
            gene_split.add_pane(self._results_panel.gene, "Gene", mode=EDGE,
                                stretch=1)
            gene_split.set_collapsed("Gene", True, by_user=False)
            gene_split.setSizes([1000, 0])
            self._gene_split = gene_split
            volcano_layout.addWidget(gene_split, 1)
            self._volcano_page = volcano_page

            grid_page = QWidget(self._figures_card)
            grid_layout = QVBoxLayout(grid_page)
            grid_layout.setContentsMargins(0, 0, 0, 0)
            grid_layout.setSpacing(4)
            size_row = QHBoxLayout()
            size_row.addWidget(QLabel("Figure size"))
            self._figure_size = QSlider(Qt.Horizontal, grid_page)
            self._figure_size.setRange(MIN_CELL_PX, MAX_CELL_PX)
            self._figure_size.setValue(get_figure_grid_size())
            self._figure_size.setMaximumWidth(220)
            self._figure_size.setToolTip(
                "How wide each figure is drawn, which is also how tall: "
                "the tiles keep each figure's own aspect ratio. Fewer, "
                "bigger figures per row to the right.")
            self._figure_size.valueChanged.connect(self._on_figure_size)
            size_row.addWidget(self._figure_size)
            size_row.addStretch(1)
            grid_layout.addLayout(size_row)
            grid_layout.addWidget(self._figure_grid, 1)
            self._figure_grid.set_target_cell_width(
                self._figure_size.value())

            self._figures_stack = QStackedWidget(self._figures_card)
            self._figures_stack.addWidget(grid_page)
            self._figures_stack.addWidget(detail)
            self._figures_stack.addWidget(volcano_page)
            self._figure_grid.pinned_activated.connect(
                self._show_regression_graph)
            self._figure_grid.pinned_menu_requested.connect(
                self._pinned_menu)
            self._figure_grid.live_tile_activated.connect(
                self._open_live_tile)
            self._figure_grid.live_tile_menu_requested.connect(
                self._live_tile_menu)
            self._results_panel.table.key_selected.connect(
                self._on_guide_selected)

            self._sweep_runs = SweepRunsPanel(self._figures_card)
            self._sweep_runs.trial_activated.connect(self._show_trial)
            self._sweep_runs.loaded_run_changed.connect(self._show_trial)
            self._sweep_runs.loaded_run_changed.connect(
                self._on_loaded_run_changed_refresh_tabs)
            self._sweep_runs.runs_removed.connect(self._on_runs_removed)
            self._sweep_runs.compare_requested.connect(
                self.open_run_beside)
            self._sweep_runs.workspace_restore_requested.connect(
                self.restore_run_workspace)
            self._sweep_runs.set_photo_provider(self.run_photograph)
            left = QTabWidget(self._figures_card)
            left.addTab(self._sweep_runs, "Runs")
            self._results_split = QSplitter(Qt.Horizontal)
            self._results_split.setChildrenCollapsible(False)
            self._results_split.addWidget(self._results_panel)
            self._results_page = self._results_split
            left.addTab(self._results_split, "Results")
            left.setTabToolTip(0, "Every run: this session's own, its "
                                  "re-fits, and every trial the parameter "
                                  "sweep ran. Pick one to see its results "
                                  "and its figures.")
            left.setTabToolTip(1, "The selected run's coefficient table, "
                                  "its volcano and its diagnostics. "
                                  "Picking a row in Runs re-points this "
                                  "at that run.")

            self._scan_panel = MeasurementScanPanel(
                frame_provider=self._scan_source_frame,
                database_provider=self._attached_database_rows,
                destination_provider=self._measurements_destination,
                settings_provider=self._column_fit_settings,
                parent=left)
            self._column_run_handles = {}
            self._scan_panel.regression.fit_started.connect(
                self._on_column_fit_started)
            self._scan_panel.regression.fit_finished.connect(
                self._on_column_fit_finished)
            self._sweep_panel = SweepPanel(
                cells_provider=self._scan_panel.databases_frame,
                counts_provider=self._sweep_counts,
                scores_provider=self._sweep_scores,
                parent=left)
            self._sweep_panel.finished.connect(self._keep_the_effects_grid)
            self._scan_panel.add_section(self._sweep_panel,
                                         "Gene × measurement sweep")
            try:
                self._scan_panel.restore_section_layout()
            except Exception:                                # noqa: BLE001
                LOG.debug("could not restore the measurements layout",
                          exc_info=True)

            left.addTab(self._scan_panel, "Measurements")
            left.setTabToolTip(
                2, "Hold the model fixed and sweep the dependent "
                   "variable. Corrected ACROSS the scan, not only within "
                   "each measurement -- a measurement that passes alone "
                   "and fails across the scan is the one worth knowing "
                   "about.")
            self._cell_montage = CellMontageView(
                frame_provider=self._results_panel.results_frame,
                results_provider=self._results_source_path,
                database_provider=self._attached_database_rows,
                parent=left)
            cells_tab = left.addTab(self._cell_montage, "Cells")
            left.setTabToolTip(
                cells_tab,
                "The cells most consistent with the selected "
                "coefficient. This screen is POOLED: the sequencing says "
                "what fraction of a well carried a guide, never which "
                "cells did, so these are candidates consistent with the "
                "effect and the caption says so.")
            self._results_panel.table.key_selected.connect(
                self._cell_montage.set_coefficient)
            self._results_panel.table.keys_selected.connect(
                self._cell_montage.set_coefficients)

            left.currentChanged.connect(self._on_results_tab_changed)
            self._results_tabs = left
            left.setCurrentWidget(self._results_page)

            split = CollapsibleSplitter(Qt.Horizontal, self._figures_card)
            split.add_pane(left, "Results", mode=EDGE, stretch=1,
                           extent=780, fold_key="regression/Results")
            split.add_pane(self._figures_stack, "Figure pages", stretch=1)
            left.setMinimumWidth(520)
            self._figures_stack.setMinimumWidth(360)
            split.setSizes([780, 620])
            self._figures_split = split
            self._figures_card.body_layout.addWidget(split, 1)

            self._grid_refresh = QTimer(self)
            self._grid_refresh.setSingleShot(True)
            self._grid_refresh.setInterval(250)
            self._grid_refresh.timeout.connect(self._refresh_figure_grid)
        except Exception:
            LOG.debug("no fast results panel", exc_info=True)
            self._results_panel = None
            self._figures_stack = None
            self._cell_montage = None
            self._figures_card.body_layout.addWidget(
                self._queue_the_results_hold, 1)
            self._figures_card.setMinimumHeight(0)
            return None
        return self._figures_split

    def _build_hyperparam_panel(self) -> QWidget:
        """Build the hyperparameter search panel into its card.

        Deferred from the screen's open to the first time the card is
        shown or ``_hyperparam`` is used; see :class:`_BuiltOnFirstUse`.
        """
        from .hyperparam import _fill_hyperparam_card

        panel = _fill_hyperparam_card(self, self._hyperparam_card)
        self._hyperparam = panel
        panel.set_apply_callback(self._propagate_live_settings)
        panel.set_settings_provider(
            lambda model=self._settings_model: model.collect())
        return panel

    def _build_live_preview_panel(self) -> QWidget:
        """Build Mask's live preview panel into its card.

        Deferred from the screen's open to the first time the card is shown
        or ``_live_preview`` is used; see :class:`_BuiltOnFirstUse`. A
        source typed while the panel did not exist is loaded now, which is
        where the eager screen's hidden panel had already loaded it.
        """
        panel = _fill_live_preview_card(self, self._live_preview_card)
        self._live_preview = panel
        panel.set_propagate_callback(self._propagate_live_settings)
        waiting = self.__dict__.pop("_live_src_waiting", None)
        if waiting is not None:
            self._autoload_live_preview(waiting)
        return panel

    def _build_measure_preview_panel(self) -> QWidget:
        """Build Measure's crop preview panel into its card.

        Deferred from the screen's open to the first time the card is shown
        or ``_measure_preview`` is used; see :class:`_BuiltOnFirstUse`.
        """
        panel = _fill_measure_preview_card(self._measure_preview_card)
        self._measure_preview = panel
        panel.set_propagate_callback(self._propagate_live_settings)
        return panel

    def _build_runtime_panel(self) -> QWidget:
        """Build the right-hand column: figures, live preview, console and actions.

        :returns: the panel widget, ready to go into the body splitter.
        """
        wrap = QWidget()
        self._runtime_wrap = wrap
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(SPACING["sm"], 0, 0, 0)
        layout.setSpacing(SPACING["md"])

        from ..widgets.figure_queue import FigureQueue
        from ..widgets.card import _CardBuiltWhenShown
        self._figures_card = (
            _CardBuiltWhenShown if self.app_key == "regression" else Card)(
                title="Figures")
        self._figure_queue = FigureQueue(parent=self._figures_card)
        #: The queue the regression results page holds, kept apart from
        #: ``_figure_queue`` because that name can be rebound before the
        #: deferred results are built, and the page an eagerly-built screen
        #: laid out held the queue made HERE.
        self._queue_the_results_hold = self._figure_queue

        self._results_panel = None
        self._results_page = None
        self._results_split = None
        self._compare_panel = None
        self._run_photographs = {}
        self._figures_stack = None
        #: Cell-montage tab, when this screen supports regression results.
        #: Initialised before tab-change handlers can read it.
        self._cell_montage = None
        if self.app_key == "regression":
            self._owe_part(_REGRESSION_RESULTS,
                           self._build_regression_results)
            self._figures_card.build_body_when_first_shown(
                partial(self._build_owed_part, _REGRESSION_RESULTS))
        results_expected = (
            self._part_is_owed(_REGRESSION_RESULTS)
            or self._if_built("_results_panel") is not None)
        if results_expected:
            content = self._figures_card.body
            scroll = QScrollArea(self._figures_card)
            scroll.setFrameShape(QScrollArea.NoFrame)
            scroll.setWidgetResizable(True)
            self._figures_card._outer.replaceWidget(content, scroll)
            scroll.setWidget(content)
            self._figures_card.body = scroll
        if not results_expected:
            self._figures_card.body_layout.addWidget(self._figure_queue, 1)
        self._figure_queue.set_propagate_callback(
            self._propagate_live_settings)
        self._umap_explorer = None
        self._umap_payload_ready = False
        if self.app_key == "umap":
            self._owe_part(_UMAP_EXPLORER, self._build_umap_explorer)
        self._figures_card.setMinimumHeight(
            0 if results_expected else 360)
        self._figures_card.hide()

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
        self._console_header = console_header
        self._console = ConsolePanel(active_app_label=app_title,
                                     persist_key=self.app_key)
        self._console.setMinimumHeight(180)
        console_col.addWidget(self._console, 1)
        from ..widgets.foldable import make_foldable

        self._console_folder = make_foldable(
            console_header, self._console, name="Console",
            on_change=self._console_folded,
            persist_key=f"{self.app_key}/Console")

        self._live_preview = self._live_preview_card = None
        self._measure_preview = self._measure_preview_card = None
        self._hyperparam = self._hyperparam_card = None
        self._sweep = self._sweep_card = None
        self._timelapse_preview = self._timelapse_preview_card = None
        self._motility_preview = self._motility_preview_card = None
        self._runtime_splitter = None

        if self.app_key in ("mask", "analyze_plaques"):
            splitter = self._new_runtime_splitter()
            splitter.setChildrenCollapsible(False)
            if self.app_key == "analyze_plaques":
                from ..widgets.plaque_preview import build_plaque_preview_card

                self._live_preview, self._live_preview_card = (
                    build_plaque_preview_card(self))
                self._live_preview.set_propagate_callback(
                    self._propagate_live_settings)
            else:
                _, self._live_preview_card = _build_live_preview_card(
                    self, panel_later=True)
                self._owe_part(_LIVE_PREVIEW, self._build_live_preview_panel)
                self._live_preview_card.build_body_when_first_shown(
                    partial(self._build_owed_part, _LIVE_PREVIEW))
            splitter.addWidget(self._live_preview_card)
            splitter.insertWidget(0, self._figures_card)
            splitter.addWidget(console_wrap)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 3)
            splitter.setStretchFactor(2, 2)
            splitter.setSizes([420, 360, 300])
            layout.addWidget(splitter, 1)
            self._remember_runtime_splitter(splitter)
        elif self.app_key == "timelapse":
            from ..widgets.timelapse_preview import build_timelapse_preview_card
            splitter = self._new_runtime_splitter()
            splitter.setChildrenCollapsible(False)
            self._timelapse_preview, self._timelapse_preview_card = (
                build_timelapse_preview_card(self))
            self._timelapse_preview.set_propagate_callback(
                self._propagate_live_settings)
            splitter.addWidget(self._timelapse_preview_card)
            splitter.insertWidget(0, self._figures_card)
            splitter.addWidget(console_wrap)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 3)
            splitter.setStretchFactor(2, 2)
            splitter.setSizes([420, 360, 300])
            layout.addWidget(splitter, 1)
            self._remember_runtime_splitter(splitter)
        elif self.app_key == "motility":
            from ..widgets.motility_preview import build_motility_preview_card
            splitter = self._new_runtime_splitter()
            splitter.setChildrenCollapsible(False)
            self._motility_preview, self._motility_preview_card = (
                build_motility_preview_card(self))
            self._motility_preview.set_propagate_callback(
                self._propagate_live_settings)
            splitter.addWidget(self._motility_preview_card)
            splitter.insertWidget(0, self._figures_card)
            splitter.addWidget(console_wrap)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 3)
            splitter.setStretchFactor(2, 2)
            splitter.setSizes([420, 360, 300])
            layout.addWidget(splitter, 1)
            self._remember_runtime_splitter(splitter)
        elif self.app_key == "measure":
            splitter = self._new_runtime_splitter()
            splitter.setChildrenCollapsible(False)
            _, self._measure_preview_card = _build_measure_preview_card(
                self, panel_later=True)
            self._owe_part(_MEASURE_PREVIEW, self._build_measure_preview_panel)
            self._measure_preview_card.build_body_when_first_shown(
                partial(self._build_owed_part, _MEASURE_PREVIEW))
            splitter.addWidget(self._measure_preview_card)
            splitter.insertWidget(0, self._figures_card)
            splitter.addWidget(console_wrap)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 3)
            splitter.setStretchFactor(2, 2)
            splitter.setSizes([420, 360, 300])
            layout.addWidget(splitter, 1)
            self._remember_runtime_splitter(splitter)
        elif _sweepable(self.app_key):
            splitter = self._new_runtime_splitter()
            splitter.setChildrenCollapsible(False)
            from .parameter_sweep import build_parameter_sweep_card
            self._sweep, self._sweep_card = build_parameter_sweep_card(self)
            splitter.insertWidget(0, self._figures_card)
            splitter.addWidget(self._sweep_card)
            splitter.addWidget(console_wrap)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 2)
            splitter.setStretchFactor(2, 1)
            splitter.setSizes([720, 300, 220] if results_expected
                              else [480, 360, 240])
            layout.addWidget(splitter, 1)
            self._remember_runtime_splitter(splitter)
        elif _hyperparam_searchable(self.app_key):
            from .hyperparam import build_hyperparam_card
            splitter = self._new_runtime_splitter()
            splitter.setChildrenCollapsible(False)
            _, self._hyperparam_card = build_hyperparam_card(
                self, panel_later=True)
            self._owe_part(_HYPERPARAM_PANEL, self._build_hyperparam_panel)
            self._hyperparam_card.build_body_when_first_shown(
                partial(self._build_owed_part, _HYPERPARAM_PANEL))
            splitter.addWidget(self._hyperparam_card)
            splitter.insertWidget(0, self._figures_card)
            splitter.addWidget(console_wrap)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 3)
            splitter.setStretchFactor(2, 2)
            splitter.setSizes([420, 360, 300])
            layout.addWidget(splitter, 1)
            self._remember_runtime_splitter(splitter)
        else:
            splitter = self._new_runtime_splitter()
            splitter.setChildrenCollapsible(False)
            splitter.addWidget(self._figures_card)
            splitter.addWidget(console_wrap)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([560, 240])
            layout.addWidget(splitter, 1)
            self._remember_runtime_splitter(splitter)

        try:
            from ..verbose_logger import register_console_target
            register_console_target(self._console)
        except Exception:
            pass

        usage_card = Card(title="System", foldable=True,
                          fold_key=f"{self.app_key}/System")
        self._usage_card = usage_card
        self._usage_ram = UsageBar("RAM")
        self._usage_gpu = UsageBar("GPU")
        self._usage_vram = UsageBar("VRAM")
        for w in (self._usage_ram, self._usage_gpu, self._usage_vram):
            usage_card.body_layout.addWidget(w)

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
        cpu_wrap.setStyleSheet("background: transparent;")
        cpu_wrap.setLayout(cpu_row)
        usage_card.body_layout.addWidget(cpu_wrap)

        self._per_core_wrap = QWidget()
        self._per_core_wrap.setStyleSheet("background: transparent;")
        self._per_core_layout = QVBoxLayout(self._per_core_wrap)
        self._per_core_layout.setContentsMargins(0, 0, 0, 0)
        self._per_core_layout.setSpacing(2)
        self._per_core_bars: list[UsageBar] = []
        self._per_core_wrap.hide()
        usage_card.body_layout.addWidget(self._per_core_wrap)

        section = QWidget()
        self._actions_section = section
        section_col = QVBoxLayout(section)
        section_col.setContentsMargins(0, 0, 0, 0)
        section_col.setSpacing(4)
        actions_heading = QLabel("Actions")
        actions_heading.setObjectName("CardTitle")
        self._actions_heading = actions_heading
        self._actions_heading_row = QHBoxLayout()
        self._actions_heading_row.addWidget(actions_heading)
        self._actions_heading_row.addStretch(1)
        section_col.addLayout(self._actions_heading_row)
        actions_body = QWidget(section)
        self._actions_body = actions_body
        body_col = QVBoxLayout(actions_body)
        body_col.setContentsMargins(0, 0, 0, 0)
        body_col.setSpacing(SPACING["md"])
        section_col.addWidget(actions_body, 1)

        actions = QWidget()
        self._actions_row = actions
        row = QHBoxLayout(actions)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["sm"])

        buttons = _WrappingButtonStrip(SPACING["sm"])
        row.addLayout(buttons)

        self._btn_run = QPushButton("Run")
        self._btn_run.setObjectName("PrimaryButton")
        self._btn_run.setCursor(Qt.PointingHandCursor)
        self._btn_run.clicked.connect(self._on_run)
        buttons.addWidget(self._btn_run)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setObjectName("DangerButton")
        self._btn_stop.setCursor(Qt.PointingHandCursor)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        buttons.addWidget(self._btn_stop)

        self._btn_import = QPushButton("Import settings…")
        self._btn_import.setObjectName("GhostButton")
        self._btn_import.setCursor(Qt.PointingHandCursor)
        self._btn_import.clicked.connect(self._on_import_settings)
        buttons.addWidget(self._btn_import)

        self._btn_remote = QPushButton("Submit remote…")
        self._btn_remote.setObjectName("PrimaryButton")
        self._btn_remote.setCursor(Qt.PointingHandCursor)
        self._btn_remote.setToolTip(
            "Send the current resolved settings to the Distributed Jobs "
            "screen for an SSH workstation, Slurm cluster, or configured "
            "cloud/HPC command."
        )
        self._btn_remote.clicked.connect(self._on_remote_submit)
        buttons.addWidget(self._btn_remote)

        self._btn_clear = QPushButton("Clear console")
        self._btn_clear.setObjectName("GhostButton")
        self._btn_clear.setCursor(Qt.PointingHandCursor)
        self._btn_clear.clicked.connect(lambda: self._console.clear())
        buttons.addWidget(self._btn_clear)

        from ..widgets.activity_spinner import ActivitySpinner
        self._activity_spinner = ActivitySpinner(actions)
        buttons.addWidget(self._activity_spinner)

        self._btn_copy_console = QPushButton("Copy console")
        self._btn_copy_console.setObjectName("GhostButton")
        self._btn_copy_console.setCursor(Qt.PointingHandCursor)
        self._btn_copy_console.setToolTip(
            "Copy everything in the console, section headers included.")
        self._btn_copy_console.clicked.connect(self._on_copy_console)

        from .. import iconset as _iconset_prefs

        self._btn_preferences = QPushButton()
        self._btn_preferences.setObjectName("GhostButton")
        self._btn_preferences.setIcon(_iconset_prefs.icon("settings"))
        from ..preferences import _set_scaled_icon_size
        _set_scaled_icon_size(self._btn_preferences, GEAR_ICON_PX)
        self._btn_preferences.setCursor(Qt.PointingHandCursor)
        self._btn_preferences.setToolTip("Open Preferences (Ctrl+P).")
        self._btn_preferences.setAccessibleName("Preferences")
        self._btn_preferences.clicked.connect(self._open_preferences_dialog)

        copy_and_gear = QWidget()
        pair = QHBoxLayout(copy_and_gear)
        pair.setContentsMargins(0, 0, 0, 0)
        pair.setSpacing(SPACING["sm"])
        pair.addWidget(self._btn_copy_console)
        pair.addWidget(self._btn_preferences)
        buttons.addWidget(copy_and_gear)

        from .. import iconset as _iconset

        self._btn_file_issue = QPushButton("File as issue")
        self._btn_file_issue.setObjectName("GhostButton")
        self._btn_file_issue.setIcon(_iconset.icon("info"))
        from ..preferences import _SMALL_ICON_PX, _set_scaled_icon_size
        _set_scaled_icon_size(self._btn_file_issue, _SMALL_ICON_PX)
        self._btn_file_issue.setCursor(Qt.PointingHandCursor)
        self._btn_file_issue.setToolTip(
            "Open a pre-filled GitHub issue with the last traceback + "
            "environment. You review before submitting. Toggle on/off "
            "in AI Settings → Report errors as GitHub issues."
        )
        self._btn_file_issue.setEnabled(False)
        self._btn_file_issue.setVisible(False)
        self._btn_file_issue.clicked.connect(self._on_file_issue)
        buttons.addWidget(self._btn_file_issue)

        row.addStretch(1)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        self._progress.setFixedWidth(240)
        row.addWidget(self._progress)

        from ..widgets import AiToggleLabel

        self._install_dimension_switches(row, AiToggleLabel)

        preview_controls = {
            "mask": (
                "_live_preview_card",
                "Show or hide the interactive Cellpose segmentation "
                "preview above the console."),
            "timelapse": (
                "_timelapse_preview_card",
                "Show or hide a preview of linked object tracks across "
                "the selected time-series frames."),
            "motility": (
                "_motility_preview_card",
                "Show or hide the track preview used to inspect velocity, "
                "straightness and infection-state assignments."),
            "measure": (
                "_measure_preview_card",
                "Show or hide a preview of selected measurement overlays "
                "before processing the full dataset."),
            "analyze_plaques": (
                "_live_preview_card",
                "Show or hide the plaque preview above the console. In "
                "Plaque mode it segments one image; in Figure mode it finds "
                "the plaque images in one figure, reads their labels and "
                "segments them."),
        }
        from ..widgets.preview_refresh import install_refresh_button

        for panel_attr, part in (("_live_preview", _LIVE_PREVIEW),
                                 ("_measure_preview", _MEASURE_PREVIEW),
                                 ("_timelapse_preview", None),
                                 ("_motility_preview", None)):
            card = getattr(self, f"{panel_attr}_card", None)
            if card is None:
                continue
            if part is not None and self._part_is_owed(part):
                install_refresh_button(
                    self, card, None,
                    panel_getter=partial(getattr, self, panel_attr))
                continue
            panel = getattr(self, panel_attr, None)
            if panel is not None:
                install_refresh_button(self, card, panel)

        preview_control = preview_controls.get(self.app_key)
        if preview_control is not None:
            card_attr, tooltip = preview_control
            if getattr(self, card_attr, None) is not None:
                self._preview_card_attr = card_attr
                self._preview_switch = AiToggleLabel(
                    text="Live", tooltip=tooltip)
                self._preview_switch.toggled.connect(
                    self._on_preview_switch)
                card = getattr(self, card_attr, None)
                placed = False
                if hasattr(card, "add_title_action"):
                    try:
                        card.add_title_action(self._preview_switch)
                        placed = True
                    except Exception:                    # noqa: BLE001
                        LOG.debug("the preview card took no title action",
                                  exc_info=True)
                if not placed:
                    row.addWidget(self._preview_switch)
                if self.app_key == "mask":
                    self._lp_switch = self._preview_switch
                self._on_preview_switch(False)

        self._ops_switch = None
        if self.app_key == "mask":
            try:
                from .mask import (OPS_TOGGLE_TEXT, OPS_TOGGLE_TOOLTIP,
                                   install_ops_switch)

                self._ops_switch = AiToggleLabel(
                    text=OPS_TOGGLE_TEXT, tooltip=OPS_TOGGLE_TOOLTIP)
                row.addWidget(self._ops_switch)
                install_ops_switch(self, self._ops_switch)
            except Exception:                            # noqa: BLE001
                LOG.debug("Could not install the OPS switch", exc_info=True)

        self._gpu_switch = None
        if self.app_key == "umap" and getattr(
                self, "_hyperparam_card", None) is not None:
            self._gpu_switch = AiToggleLabel(
                text="GPU",
                tooltip=(
                    "Use the RAPIDS cuML backend for both the main "
                    "dimensionality-reduction run and hyperparameter search. "
                    "CPU and GPU reducers may produce different embeddings, "
                    "so compare search results only within the same backend."),
            )
            self._gpu_switch.toggled.connect(self._on_umap_gpu_switch)
            row.addWidget(self._gpu_switch)

        if getattr(self, "_sweep", None) is not None:
            from .parameter_sweep import (SWEEP_TOGGLE_TEXT,
                                          SWEEP_TOGGLE_TOOLTIP)
            self._sweep_switch = AiToggleLabel(text=SWEEP_TOGGLE_TEXT,
                                               tooltip=SWEEP_TOGGLE_TOOLTIP)
            self._sweep_switch.toggled.connect(self._on_sweep_switch)
            row.addWidget(self._sweep_switch)
            self._on_sweep_switch(False)

        if getattr(self, "_hyperparam_card", None) is not None:
            from .hyperparam import TOGGLE_TEXT, TOGGLE_TOOLTIP
            self._hp_switch = AiToggleLabel(text=TOGGLE_TEXT,
                                            tooltip=TOGGLE_TOOLTIP)
            self._hp_switch.toggled.connect(self._on_hyperparam_switch)
            row.addWidget(self._hp_switch)
            self._on_hyperparam_switch(False)

        self._interactive_switch = None
        if self.app_key == "umap":
            self._interactive_switch = AiToggleLabel(
                text="Interactive",
                tooltip=(
                    "Enable the interactive Image UMAP view. Select a point "
                    "to preview its image, draw a region around a cluster, "
                    "and write manual or model-assisted labels to the database."
                ),
            )
            self._interactive_switch.toggled.connect(
                self._on_interactive_switch)
            queue = getattr(self, "_figure_queue", None)
            if queue is not None and hasattr(queue, "figure_clicked"):
                queue.figure_clicked.connect(self._on_static_figure_clicked)
            row.addWidget(self._interactive_switch)

        self._ai_switch = AiToggleLabel()
        self._ai_switch.toggled.connect(self._on_ai_switch)
        row.addWidget(self._ai_switch)

        self._apply_ai_default()

        body_col.addWidget(actions)

        self._category_hint_pinned = ""
        self._category_hint = QLabel(self._default_category_hint())
        self._category_hint.setObjectName("CategoryHintStrip")
        self._category_hint.setStyleSheet("background: transparent;")
        self._category_hint.setWordWrap(True)
        self._category_hint.setTextFormat(Qt.RichText)
        self._category_hint.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._sync_category_hint_height()
        body_col.addWidget(self._category_hint)

        self._hint_strip = QLabel(self._default_hint())
        self._hint_strip.setObjectName("SubtitleSmall")
        self._hint_strip.setWordWrap(True)
        self._sync_hint_strip_height()
        self._hint_strip.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._hint_strip.setOpenExternalLinks(True)
        self._hint_strip.linkActivated.connect(self._on_hint_link)
        body_col.addWidget(self._hint_strip)

        self._actions_folder = make_foldable(
            actions_heading, actions_body, name="Actions",
            persist_key=f"{self.app_key}/Actions")
        self._install_the_shell_panes(layout, usage_card, section)
        return wrap

    #: Where a runtime splitter's state is stored. Distinct from the console
    #: splitter's key, which `console_panel` already persists under the bare
    #: app key -- two splitters on one screen would otherwise restore each
    #: other's blob, and `restoreState` on a mismatched one silently does
    #: nothing, which is a layout that ignores the user with no message.
    RUNTIME_SPLIT_SUFFIX = "::runtime"

    def _new_runtime_splitter(self):
        """The vertical splitter the runtime column's panes live in.

        A :class:`~spacr.qt.widgets.collapsible_splitter.CollapsibleSplitter`
        (item 471): every pane in it resizes by its edge and collapses at the
        limit, and it remembers the sizes the user dragged, per pane name,
        under this module's key.
        """
        from ..widgets.collapsible_splitter import CollapsibleSplitter

        return CollapsibleSplitter(
            Qt.Vertical,
            persist_key=f"{self.app_key}{self.RUNTIME_SPLIT_SUFFIX}")

    @staticmethod
    def _fold_a_card(card, name: str):
        """The Folder that collapses ``card`` by its title, made if need be.

        No persist key: a panel that opens because the user switched it on
        (a live preview) or because a run produced something (figures) must
        open, not come back as the strip it was left as last session.

        :returns: the Folder, or None for a card with no title to click.
        """
        if card is None:
            return None
        folder = getattr(card, "folder", None)
        if folder is not None:
            return folder
        title = getattr(card, "title_label", None)
        body = getattr(card, "body", None)
        if title is None or body is None:
            return None
        from ..widgets.foldable import make_foldable

        card.folder = make_foldable(title, body, name=name)
        return card.folder

    #: The runtime cards that are FOCUS panes: shown, they take the height
    #: (item 471). ``(attribute, name, height to open at)``; the name is the
    #: fallback for a card with no title, whose own title is used otherwise.
    _FOCUS_CARDS = (
        ("_figures_card", "Figures", 420),
        ("_live_preview_card", "Live preview", 420),
        ("_measure_preview_card", "Crop preview", 420),
        ("_timelapse_preview_card", "Track preview", 420),
        ("_motility_preview_card", "Motility preview", 420),
    )

    #: The runtime cards that collapse and resize but take nothing over.
    _TOOL_CARDS = (
        ("_sweep_card", "Parameter sweep", 300),
        ("_hyperparam_card", "Hyperparameter search", 300),
    )

    #: The panes a focus pane collapses, and the order they stack in.
    SHELL_TARGETS = ("Console", "System", "Actions")

    def _install_the_shell_panes(self, layout, usage_card, section) -> None:
        """Put System and the buttons in the splitter and name every pane.

        Item 471, slice B. The console, System and the buttons section each
        collapse and each resize by their edge; a collapsed one is its
        heading, at the bottom of the column. The figures panel and the
        previews are FOCUS panes: while one is shown the three below it and
        the settings column collapse (:class:`FocusCollapse`), and the user
        can open any of them again, which pins it open for the visit.

        Nothing here shows, measures or builds a card: a lazily-built panel
        is registered while hidden and stays unbuilt until its switch shows
        it (items 284/380).

        :param layout: the runtime column's layout, for the fallback.
        :param usage_card: the System card.
        :param section: the buttons section.
        """
        from ..widgets.collapsible_splitter import (CollapsibleSplitter,
                                                    FocusCollapse)

        focus = FocusCollapse(self)
        self._shell_focus = focus
        split = self._runtime_splitter
        if not isinstance(split, CollapsibleSplitter):
            layout.addWidget(usage_card)
            layout.addWidget(section)
            return
        tall = (self._part_is_owed(_REGRESSION_RESULTS)
                or self._if_built("_results_panel") is not None)
        focus_attrs = {attr for attr, _name, _extent in self._FOCUS_CARDS}
        for attr, name, extent in self._FOCUS_CARDS + self._TOOL_CARDS:
            card = getattr(self, attr, None)
            if card is None or split.indexOf(card) < 0:
                continue
            if attr == "_figures_card" and tall:
                extent = 720
            title = getattr(card, "title_label", None)
            name = (title.text().strip() if title is not None else "") or name
            is_focus = attr in focus_attrs
            split.add_pane(card, name, folder=self._fold_a_card(card, name),
                           focus=is_focus, extent=extent,
                           minimum=card.minimumHeight())
            if is_focus:
                focus.watch(card)
        split.add_pane(self._console_wrap, "Console",
                       folder=self._console_folder, extent=300, minimum=0)
        split.add_pane(usage_card, "System", folder=usage_card.folder,
                       stretch=0)
        split.add_pane(section, "Actions", folder=self._actions_folder,
                       stretch=0)
        for name in self.SHELL_TARGETS:
            focus.target(split, name)

    def adopt_runtime_pane(self, card, *, focus: bool = True):
        """Name a card someone else put in the runtime splitter.

        :mod:`spacr.qt.preview_registry` inserts a declared preview above the
        console; adopting it makes it collapse by its title, resize by its
        edge, and -- as a preview -- take the height when it is shown.

        :param card: the card, already in the runtime splitter.
        :param focus: whether showing it collapses the console, System, the
            buttons and the settings column.
        :returns: the pane, or None when this screen has no such splitter.
        """
        from ..widgets.collapsible_splitter import CollapsibleSplitter

        split = getattr(self, "_runtime_splitter", None)
        if not isinstance(split, CollapsibleSplitter) or card is None:
            return None
        if split.indexOf(card) < 0:
            return None
        title = getattr(card, "title_label", None)
        name = (title.text().strip() if title is not None else "") or \
            card.objectName() or "Preview"
        pane = split.add_pane(card, name, folder=self._fold_a_card(card, name),
                              focus=focus, extent=420,
                              minimum=card.minimumHeight())
        shell = getattr(self, "_shell_focus", None)
        if focus and shell is not None:
            shell.watch(card)
        return pane

    def reveal_settings(self) -> bool:
        """Open the settings column if it is collapsed; the user asked.

        Ctrl+F puts the caret in the settings search, which cannot take it
        from a column folded away to the left.

        :returns: whether the column is open now.
        """
        from ..widgets.collapsible_splitter import CollapsibleSplitter

        body = getattr(self, "_body_splitter", None)
        if not isinstance(body, CollapsibleSplitter):
            return True
        if body.is_collapsed("Settings"):
            body.set_collapsed("Settings", False, by_user=True)
        return not body.is_collapsed("Settings")

    def _remember_runtime_splitter(self, splitter) -> None:
        """Restore this screen's pane heights and persist each resize.

        Saving on every splitter move preserves the layout even when the
        application does not reach its normal shutdown path. A
        :class:`~spacr.qt.widgets.collapsible_splitter.CollapsibleSplitter`
        remembers its own sizes, per pane name, so it is only recorded here.
        """
        self._runtime_splitter = splitter
        if splitter is None:
            return
        from ..widgets.collapsible_splitter import CollapsibleSplitter

        if isinstance(splitter, CollapsibleSplitter):
            return
        key = f"{self.app_key}{self.RUNTIME_SPLIT_SUFFIX}"
        try:
            from ..widgets.console_panel import get_split_state, set_split_state
        except Exception:                                        # noqa: BLE001
            return
        try:
            state = get_split_state(key)
            if state is not None:
                splitter.restoreState(state)
        except Exception:                                        # noqa: BLE001
            pass

        def _save(*_args):
            """Store this splitter's layout under its key."""
            try:
                set_split_state(key, splitter.saveState())
            except Exception:                                    # noqa: BLE001
                pass

        splitter.splitterMoved.connect(_save)

    def _write_hint(self, text: str, url: str = "",
                    hold: bool = False, animated: bool = False) -> None:
        """Put ``text`` in the strip, trimmed to the lines the strip has.

        ``url`` adds the documentation link on its own line. Suppressing the
        popup would otherwise have taken the link with it -- the strip's own
        prompt promises "details AND a link to its documentation", and the
        link only ever lived in the popup. It is the per-setting target, so
        it points at the function that READS the setting rather than at the
        module page.

        The strip is a FIXED four lines so the panel below it does not move
        every time the pointer crosses a setting. A description longer than
        that used to be clipped mid-word by the layout, which reads as a
        rendering fault rather than as "there is more". Text always fits its
        container.

        Measured against the font Qt is actually painting and the width the
        strip actually has, so it stays correct at any font scale rather than
        at the one this was written on.
        """
        strip = getattr(self, "_hint_strip", None)
        if strip is None:
            return
        lines = HINT_STRIP_LINES - (1 if url else 0)
        fitted = _fit_to_lines(str(text), strip, max(1, lines))
        if url:
            from html import escape as _escape

            from ..i18n import tr as _tr
            animation_link = (
                f"&nbsp;&nbsp;<a href=\"{_HINT_ANIMATION_HREF}\">"
                f"{_escape(_tr('Animation'))}</a>" if animated else "")
            strip.setText(
                f"{_escape(fitted)}<br>"
                f"<a href=\"{_escape(str(url), quote=True)}\">"
                f"{_escape(_tr('API'))}</a>{animation_link}")
        else:
            strip.setText(fitted)
        strip.setToolTip(str(text))
        self._hold_the_hint(hold)

    #: How long the strip keeps the LAST hovered setting, in milliseconds.
    #:
    #: Ten seconds, and it is not a round number to
    #: tune down because it feels long while reading code. It is the budget
    #: for noticing the strip, crossing the window and pressing the link --
    #: the reach the hold exists to make possible. If a measurement ever says
    #: the reach takes longer, raise it and say so.
    HINT_HOLD_MS = 10_000

    def _hold_the_hint(self, holding: bool,
                       duration_ms: Optional[int] = None) -> None:
        """Start, restart or stop the strip's hold.

        RESTARTED ON EACH NEW SETTING, so reading down a form is not a race
        against a clock started by the first row. Stopped outright when the
        strip is being put back to its default, or the timer would blank a
        strip that is already blank and fight whatever wrote it next.

        :param duration_ms: how long to hold. ``None`` is the per-setting
            :data:`HINT_HOLD_MS`; a dock hover passes the longer module hold
            instead, because its links leave the application.
        """
        timer = getattr(self, "_hint_hold_timer", None)
        if timer is None:
            from PySide6.QtCore import QTimer

            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._release_the_hint)
            self._hint_hold_timer = timer
        timer.stop()
        if holding:
            timer.start(self.HINT_HOLD_MS if duration_ms is None
                        else int(duration_ms))

    def _on_hint_link(self, href: str) -> None:
        """Open the box for the setting the strip is showing, animation out.

        THE STRIP CANNOT PLAY THE ANIMATION ITSELF and should not try: it is
        four lines at the bottom of the window, and the popup already has a
        column sized for the square and the code to decode it once. So the
        word hands off rather than duplicating, and it does so for the
        setting the STRIP names -- which, ten seconds after the hover, is no
        longer the widget under the pointer.

        Deliberately ignores the `Tooltips box` preference. That switch says
        whether a box appears UNASKED; this is a press.
        """
        if href != _HINT_ANIMATION_HREF:
            return
        widget = getattr(self, "_hinted_widget", None)
        html = getattr(self, "_hinted_html", "")
        if widget is None or not html:
            return
        from ..widgets.hover_tooltip import HoverTooltip

        popup = HoverTooltip.instance()
        popup.show_for(widget, html)
        popup.toggle_animation()

    def _release_the_hint(self) -> None:
        """Put the strip back to its prompt once the hold has run out."""
        if getattr(self, "_hint_strip", None) is None:
            return
        self._write_hint(self._default_hint())

    def _sync_hint_strip_height(self) -> None:
        """Reserve four lines using the font Qt is actually painting."""
        hint = getattr(self, "_hint_strip", None)
        if hint is None:
            return
        hint.ensurePolished()
        hint.setFixedHeight(
            _height_of_lines(hint.fontMetrics(), HINT_STRIP_LINES))

    def _sync_category_hint_height(self) -> None:
        """Reserve three lines for the category strip, in the painted font."""
        strip = getattr(self, "_category_hint", None)
        if strip is None:
            return
        strip.ensurePolished()
        strip.setFixedHeight(
            _height_of_lines(strip.fontMetrics(), CATEGORY_STRIP_LINES))

    def _watch_for_late_captions(self) -> None:
        """Translate any subtree parented into this screen after it was built.

        `MainWindow._on_nav_selected` runs one language pass over a screen,
        once, when the screen is first constructed. Two things arrive after
        that and never see a pass of their own: the preview
        :mod:`spacr.qt.preview_registry` declares for a module -- built and
        inserted into the runtime panel the first time the module is opened
        -- and the toggle it puts on the settings strip. Measured on a
        Swedish cold start, that left the whole Plaque analysis preview and
        its toggle in English: 112 captions, on one screen.

        `ChildAdded` is the moment a widget is parented, and by then the
        widget it carries is fully built, so translating that subtree alone
        costs a walk of the new panel rather than of the screen. The pass is
        deferred by one turn of the event loop: a widget can be parented
        before its own children exist, and a pass that ran now would cache
        an empty caption as that child's English source.
        """
        hosts = (getattr(self, "_runtime_wrap", None),
                 getattr(self, "_body_splitter", None),
                 self)
        watcher = _LateCaptionTranslator(self)
        self._late_caption_watcher = watcher
        for host in hosts:
            if host is None:
                continue
            host.removeEventFilter(watcher)
            host.installEventFilter(watcher)

    def _default_category_hint(self) -> str:
        """The prompt the category strip falls back to, in the UI language.

        Same reason as :meth:`_default_hint`: leaving a header writes this
        back, so an untranslated literal here turns a translated strip
        English on the first hover and leaves it that way.
        """
        return tr("Hover a settings category for what the group decides, "
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
        for section in self.rendered_settings_sections():
            header = section.header()
            if header is None or header.property("categoryHintWired"):
                continue
            title = (section.property("settingsCategorySource")
                     or section.title())
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
        """Show one category's blurb in the strip under the actions row.

        The strip is fed by TITLE, because a hovered header carries its own
        name and nothing else. A heading nested under another resolves its
        help by PATH -- "Cell" under "Object filtration" is not the "Cell"
        segmentation category -- so what that resolved to when the section
        was built is read back here rather than looked up again by the bare
        word, which would hand a filtration sub-heading the blurb about
        Cellpose models.

        :param title: the category heading as built; its blurb is read from
            the section's recorded blurbs, falling back to
            :func:`category_tooltip`, and the translated heading is shown in
            capitals before it.
        """
        strip = getattr(self, "_category_hint", None)
        if strip is None:
            return
        text = (getattr(self, "_category_blurbs", None) or {}).get(
            str(title)) or category_tooltip(self.app_key, title)
        heading = tr(str(title or "")).upper().strip()
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

        :param event: the show event; passed to the base class, and only a
            non-spontaneous (application-initiated) show begins a new view
            for the focus-collapse rule.
        """
        super().showEvent(event)
        focus = getattr(self, "_shell_focus", None)
        if focus is not None and not event.spontaneous():
            focus.begin_view()
        usage_timer = getattr(self, "_usage_timer", None)
        if usage_timer is not None and not usage_timer.isActive():
            usage_timer.start()
            self._refresh_usage()
        self._sync_hint_strip_height()
        self._sync_category_hint_height()
        # ONCE, ON THE FIRST SHOW, AND THIS IS THE BLACK BOX.
        # `_clear_page_surfaces` runs during construction, and it tags what
        # exists THEN. Anything a screen builds afterwards -- a section that
        # mounts on demand, a grid the preferences turn on -- is never
        # tagged, inherits the blanket ``QWidget { background-color: bg }``
        # rule, and paints the window colour as a solid rectangle over the
        # backdrop.
        #
        # It looked intermittent because the repair was accidental:
        # `refresh_ambient_background` re-tags, but only when the ambient
        # preference actually CHANGED, and its docstring says so. Leaving
        # the screen and coming back happened to take that path, so the box
        # appeared on first open and was gone on the second -- which reads
        # like a paint race and is not one.
        #
        # Guarded by a flag rather than run on every show: tagging walks
        # every child and re-polishes it, and Mask carries 201 settings.
        if not getattr(self, "_surfaces_cleared_on_show", False):
            self._surfaces_cleared_on_show = True
            try:
                self._clear_page_surfaces()
            except Exception:                                # noqa: BLE001
                LOG.debug("could not clear the page surfaces on first show",
                          exc_info=True)
        self.refresh_ambient_background()
        self._prebuild_when_idle()

    def _prebuild_when_idle(self) -> None:
        """Build the waiting categories in idle time; see :class:`_IdlePrebuild`."""
        if not getattr(self, "_waiting_heading_of", None):
            return
        builder = self.__dict__.get("_idle_prebuild")
        if builder is None:
            builder = self._idle_prebuild = _IdlePrebuild(self)
        builder.resume()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Let the screen stop paying for things nobody can see.

        :param event: the Qt hide event.
        """
        builder = self.__dict__.get("_idle_prebuild")
        if builder is not None:
            builder.stop()
        self._usage_generation += 1
        usage_timer = getattr(self, "_usage_timer", None)
        if usage_timer is not None:
            usage_timer.stop()
        focus = getattr(self, "_shell_focus", None)
        if focus is not None and not event.spontaneous():
            focus.end_view()
        super().hideEvent(event)

    def _on_run(self, _checked=False, *, override=None):
        """Start the pipeline.

        :param _checked: swallows the bool a ``clicked`` signal passes.
        :param override: run THESE settings instead of what the panel holds.
            Keyword-only, so the clicked bool can never land in it.

        The override is what the re-fit uses. It builds its settings from the
        run already on screen rather than from the widgets, because the
        widgets may have been edited since -- and a re-fit of "what I am
        looking at" that quietly picked those up would compare two runs
        differing in more ways than the one the user chose.
        """
        from ..verbose_logger import log_button_press
        entry = resolve_pipeline_entry(self.app_key)
        if entry is None:
            log_button_press(
                f"{self.app_key}.Run",
                {"result": "not_runnable"})
            QMessageBox.information(
                self, tr("Not runnable"),
                tr("The '{app}' app is interactive-only in this Qt build. "
                   "Use the classic Tk GUI (`spacr`) for now.",
                   app=self.app_key),
            )
            return
        try:
            settings = (dict(override) if override is not None
                        else self._settings_model.collect())
        except Exception as e:
            log_button_press(
                f"{self.app_key}.Run",
                {"result": "bad_settings", "error": str(e)})
            QMessageBox.warning(self, tr("Bad settings"), str(e))
            return

        if self.app_key == "measure" and not self._confirm_crop_choices(
                settings):
            log_button_press(f"{self.app_key}.Run",
                             {"result": "cancelled_at_crop_warning"})
            return

        if self.app_key == "umap":
            from ..theme import active_palette
            palette = active_palette()
            settings["_plot_theme"] = {
                "background": palette["surface_alt"],
                "foreground": palette["fg"],
                "border": palette["fg"],
            }

        log_button_press(
            f"{self.app_key}.Run",
            {
                "entry":    getattr(entry, "__qualname__", repr(entry)),
                "src":      settings.get("src"),
                "n_keys":   len(settings),
            },
        )
        entry_name = getattr(entry, "__qualname__", repr(entry))
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

        import time as _time
        self._run_started_at = _time.time()
        _pause_the_fractal(self)
        import datetime as _dt
        from ..widgets.sweep_runs import SOURCE_REFIT, SOURCE_RUN
        source = SOURCE_REFIT if override is not None else SOURCE_RUN
        label = _dt.datetime.now().strftime(f"{source}  %H:%M:%S")
        try:
            self._figure_queue.mark_run(label)
        except Exception:
            LOG.debug("could not mark the run on the figure grid",
                      exc_info=True)
        self._run_handle = self._record_run_in_runs_tab(label, source, settings)

        try:
            from ..preferences import get_hash_inputs
            settings.setdefault("hash_inputs", get_hash_inputs())
        except Exception:
            LOG.debug("could not read the hashing preference", exc_info=True)

        self._announce_the_fit(settings)

        self._thread, worker = make_thread(entry, settings)
        self._worker = worker
        worker.line_ready.connect(self._console.append_stdout)
        worker.error.connect(self._on_pipeline_error)
        worker.figure_ready.connect(self._on_figure_ready)
        worker.result_ready.connect(self._on_pipeline_result)
        self._results_loaded_in_memory = False
        worker.finished.connect(self._on_finished)
        self._thread.finished.connect(self._clear_thread_refs)
        self._thread.start()


    #: When the heartbeat speaks, in seconds since the run started.
    #:
    #: SPARSE AND WIDENING, not a fixed tick. A line every thirty seconds is
    #: 120 lines in the hour this exists for, and a console holding 120
    #: identical lines is the same silence with more scrolling. The early
    #: entries are close together because that is when a user is deciding
    #: whether anything is happening at all.
    HEARTBEAT_SCHEDULE = (30, 60, 120, 300, 600, 1200, 1800, 2700, 3600)

    #: And every this many seconds after the last scheduled one.
    HEARTBEAT_INTERVAL = 1800

    #: How often the timer wakes to check the schedule. Cheap, and finer than
    #: the schedule so no entry is skipped by a slow event loop.
    HEARTBEAT_TICK_MS = 5000

    @staticmethod
    def _it_will_permute(settings) -> bool:
        """Return whether the settings explicitly select permutation inference.

        ``inference='auto'`` is intentionally excluded because its resolution
        depends on guide and well counts that are not available until the
        design scan completes.
        """
        settings = settings or {}
        inference = str(settings.get("inference") or "").strip().lower()
        mode = str(settings.get("analysis_mode") or "").strip().lower()
        return inference == "nonparametric" or mode == "guide_permutation"

    def _say_what_the_permutation_will_do(self, settings) -> str:
        """Display and return the configuration of a guide permutation run.

        The banner reports the statistic, blocking variable, output level,
        permutation count, and minimum attainable empirical P value. Invalid
        counts are reported without attempting to calculate a P-value floor;
        pre-flight validation then prevents the run from starting.

        :param settings: Regression settings associated with the pending run.
        :returns: The message appended to the run console.
        """
        from ..i18n import tr
        from .settings_model import normalise_regression_level

        settings = settings or {}
        raw_permutations = settings.get("guide_permutations", 200000)
        try:
            permutations = int(raw_permutations)
        except (TypeError, ValueError):
            permutations = 0
        if permutations < 1:
            note = tr(
                "→ Guide permutation test configuration is invalid: "
                "guide_permutations={value}. Enter an integer of at least 1 "
                "before running the analysis.",
                value=repr(raw_permutations),
            )
            self._console.append_notice("{note}\n", note=note)
            return note

        floor = 1.0 / (permutations + 1)
        level = normalise_regression_level(settings.get("level"))
        statistic = str(settings.get("grna_statistic") or "pearson")
        block = str(settings.get("guide_permutation_block") or "plateID")
        note = tr(
            "→ Guide permutation test: {permutations} within-{block} "
            "permutations per guide using the {statistic} statistic; "
            "reported level: {level}. No regression model is fitted, so "
            "regression_type is not used. Minimum attainable empirical P "
            "value: {floor} (1/(permutations+1)).",
            permutations=f"{permutations:,}",
            block=block,
            statistic=statistic,
            level=level,
            floor=f"{floor:.3g}",
        )
        self._console.append_notice("{note}\n", note=note)
        return note

    def _announce_the_fit(self, settings) -> None:
        """Say what is about to run, what it costs, and how big it is.

        THREE SENTENCES, IN THIS ORDER, and each is a different question:

        * WHAT IS RUNNING -- the model, its level and its backend, read off
          the settings the run was actually started with rather than off the
          widgets, so a re-fit describes the re-fit.
        * WHAT IT COSTS -- the measurement from
          :func:`~spacr.qt.screens.settings_model.mixed_cost_note`, which the
          model box states too. One source, so the two cannot drift.
        * HOW BIG THE DESIGN IS -- and that one needs the count files read,
          so it goes through the screen's `JobRunner` and arrives when it
          arrives. Reading a 500,000-row count CSV on the GUI thread to warn
          somebody about a slow fit would be its own freeze.

        A no-op outside the regression screens: no other module has a
        measured cost to quote, and "still running" over a segmentation is
        what the progress bar is already for.
        """
        if self.app_key not in ("regression", "sweep"):
            return
        from .settings_model import (SLOW_MODELS, mixed_cost_note,
                                     normalise_regression_level,
                                     regression_design_scan)

        model = str((settings or {}).get("regression_type") or "auto").lower()
        if self._it_will_permute(settings):
            self._say_what_the_permutation_will_do(settings)
            self._slow_fit = False
            self._start_the_heartbeat()
            return
        self._console.append_notice(
            "→ Model: {model}, level {level}, backend {backend}.\n",
            model=model,
            level=normalise_regression_level((settings or {}).get("level")),
            backend=str((settings or {}).get("regression_backend")
                        or "statsmodels (CPU)"),
        )
        if model == "mixed":
            self._console.append_notice("■ {note}\n", note=mixed_cost_note())
        self._slow_fit = model in SLOW_MODELS
        self._start_the_heartbeat()

        scan = dict(settings or {})
        try:
            self._jobs.submit(lambda: regression_design_scan(scan),
                              self._on_design_scanned)
        except Exception:                                        # noqa: BLE001
            LOG.debug("could not scan the design", exc_info=True)

    def _on_design_scanned(self, design) -> None:
        """Print the unfiltered count-file design size to the console.

        These counts precede score merging, read-fraction filtering, and well
        filtering, so they remain distinguishable from the run's cleaned
        design counts.
        """
        if not isinstance(design, dict):
            return
        genes, guides = design.get("genes"), design.get("guides")
        wells, note = design.get("wells"), str(design.get("note") or "")
        if genes is not None and guides is not None and wells is not None:
            self._console.append_notice(
                "■ The design in the count files: {genes} genes and {guides} "
                "guides over {wells} wells, from {rows} rows in {files} "
                "file(s). That is BEFORE the merge with the scores and "
                "before the filters — the run's own counts follow.\n",
                genes=genes, guides=guides, wells=wells,
                rows=design.get("rows", 0), files=design.get("files", 0))
        elif guides is not None:
            self._console.append_notice(
                "■ The count files hold {guides} distinct gRNAs in {rows} "
                "rows. {why}\n",
                guides=guides, rows=design.get("rows", 0),
                why=note or "The rest of the design could not be read.")
        else:
            self._console.append_notice(
                "■ Could not size the design from the count files: {why}\n",
                why=note or "nothing readable in them")

    def _start_the_heartbeat(self) -> None:
        """Begin reporting where a long run has got to."""
        self._heartbeat_said = 0.0
        timer = getattr(self, "_heartbeat", None)
        if timer is None:
            timer = QTimer(self)
            timer.setInterval(self.HEARTBEAT_TICK_MS)
            timer.timeout.connect(self._on_heartbeat)
            self._heartbeat = timer
        timer.start()

    def _stop_the_heartbeat(self) -> None:
        """The run is over: stop talking about it."""
        timer = getattr(self, "_heartbeat", None)
        if timer is not None:
            timer.stop()

    def _due_heartbeat(self, elapsed: float) -> bool:
        """Whether the schedule has passed a mark this run has not spoken at.

        Kept apart from the timer so the schedule can be driven directly: a
        test that had to wait 3,600 real seconds to check the last entry
        would not be written, and the entry nobody checks is the one that is
        wrong.
        """
        said = float(getattr(self, "_heartbeat_said", 0.0))
        due = [mark for mark in self.HEARTBEAT_SCHEDULE
               if said < mark <= elapsed]
        if due:
            self._heartbeat_said = float(due[-1])
            return True
        last = self.HEARTBEAT_SCHEDULE[-1]
        if elapsed < last + self.HEARTBEAT_INTERVAL:
            return False
        steps = int((elapsed - last) // self.HEARTBEAT_INTERVAL)
        mark = last + steps * self.HEARTBEAT_INTERVAL
        if said >= mark:
            return False
        self._heartbeat_said = float(mark)
        return True

    def _on_heartbeat(self) -> None:
        """One scheduled line saying the run is alive and where it has got to.

        NOT A SPINNER WITH NO CONTENT. The line names the elapsed time and,
        for a fit that holds one core by design, says that holding one core
        is what healthy looks like -- because "cpu at 100 percent" was the
        observation the user could not interpret.
        """
        import time as _time

        thread = getattr(self, "_thread", None)
        if thread is None or not thread.isRunning():
            self._stop_the_heartbeat()
            return
        started = getattr(self, "_run_started_at", None)
        if not started:
            return
        elapsed = _time.time() - float(started)
        if not self._due_heartbeat(elapsed):
            return
        if getattr(self, "_slow_fit", False):
            self._console.append_notice(
                "■ Still fitting — {elapsed} so far. This model is a "
                "single-threaded optimisation, so one core at 100% and the "
                "others idle is what a healthy fit looks like. Stop is "
                "live.\n", elapsed=_elapsed_words(elapsed))
        else:
            self._console.append_notice(
                "■ Still running — {elapsed} so far.\n",
                elapsed=_elapsed_words(elapsed))

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
        self._btn_copy_console.setText(tr("Copied"))
        QTimer.singleShot(
            1200,
            lambda: self._btn_copy_console.setText(tr("Copy console")))
        try:
            self.statusBar().showMessage(
                tr("Copied {count} lines", count=lines), 3000)
        except Exception:
            pass

    def _on_remote_submit(self) -> None:
        """Validate current settings and hand a snapshot to MainWindow."""
        try:
            settings = dict(self._settings_model.collect())
        except Exception as exc:
            QMessageBox.warning(self, tr("Bad settings"), str(exc))
            return
        self.remote_submit_requested.emit(self.app_key, settings)

    def _on_pipeline_error(self, tb: str):
        """Capture the traceback and either show it raw or route it through AI.

        With the report action switched on, "File as issue" is revealed and
        what happens to the report is decided here and carried out by
        :meth:`_on_finished`, under the failure line. With issue reporting
        set to 'always' the report is filed automatically; with 'ask' the
        console says nothing was sent and how to send it.
        """
        self._last_error_text = tb

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
        try:
            from ..ai import settings as _ai_settings
            enabled = _ai_settings.get_auto_file_issues()
        except Exception:
            enabled = False
        self._btn_file_issue.setVisible(enabled)
        self._btn_file_issue.setEnabled(enabled)
        always = bool(enabled) and self._reporting_is_set_to_always()
        agreed = always and self._the_terms_allow_automatic_filing()
        self._report_files_itself = agreed
        self._report_awaits_the_terms = always and not agreed
        self._report_waits_for_a_click = (
            bool(enabled) and not always
            and self._reporting_is_not_set_to_never())

    @staticmethod
    def _the_terms_allow_automatic_filing() -> bool:
        """Whether this profile has accepted the terms that allow it.

        Automatic filing rests on Section 5.6 of the terms of use. A profile
        can reach a failed run without accepting that version: a launch with
        ``--no-setup`` or ``SPACR_NO_SETUP``, an offscreen server, or a user
        who closed the terms slide. Until it accepts, nothing is filed
        automatically.

        :returns: ``False`` when the current terms have not been accepted,
            or when that cannot be read.
        """
        try:
            from ..terms import needs_agreement
            return not needs_agreement()
        except Exception:                                    # noqa: BLE001
            return False

    @staticmethod
    def _reporting_is_set_to_always() -> bool:
        """Whether a failed run files its own report, without a preview.

        :returns: ``True`` when issue reporting is 'always', which is the
            default for a profile that never chose. ``False`` when the
            preference cannot be read: an unreadable choice must not publish.
        """
        try:
            from ..preferences import (ISSUE_PROMPT_ALWAYS,
                                       get_issue_prompt_mode)
            return get_issue_prompt_mode() == ISSUE_PROMPT_ALWAYS
        except Exception:                                    # noqa: BLE001
            return False

    @staticmethod
    def _reporting_is_not_set_to_never() -> bool:
        """Whether "File as issue" would open a report if it were pressed.

        :returns: ``False`` when issue reporting is set to 'never' in
            Preferences, since the button then refuses to file, and ``False``
            when the preference cannot be read.
        """
        try:
            from ..preferences import (ISSUE_PROMPT_NEVER,
                                       get_issue_prompt_mode)
            return get_issue_prompt_mode() != ISSUE_PROMPT_NEVER
        except Exception:                                    # noqa: BLE001
            return False

    def _say_the_report_was_not_sent(self, failed: bool = True) -> None:
        """Say under a failed run that no report went to GitHub, and how to send one.

        Issue 117: a user with "Report errors as GitHub issues" on watched a
        run fail and expected an issue to have been filed. None was, by
        design: in 'ask' mode a report goes to the PUBLIC tracker only after
        a click on that specific report. The console never
        said so, and the button that files it had appeared in the row under
        the console with nothing pointing to it. This line goes where the user
        looks when a run fails, directly under "✗ Failed".

        :param failed: whether the run that just ended failed. The pending
            line is dropped either way, so a stopped run does not carry it
            over to the next one.
        """
        waiting = getattr(self, "_report_waits_for_a_click", False)
        self._report_waits_for_a_click = False
        if not (failed and waiting):
            return
        self._console.append_notice(
            "[issue] Nothing was sent to GitHub. Reports are public, so spaCR "
            "files one only when you press File as issue and then Send "
            "report.\n")

    def _settle_the_report(self, failed: bool = True) -> None:
        """Do what issue reporting is set to do, now that the run has ended.

        With 'always' the report is filed automatically
        (:meth:`_file_the_report_automatically`), once the profile has
        accepted the terms that allow it. Before that, the console says
        nothing was sent and why. With 'ask' the console says that nothing
        was sent (:meth:`_say_the_report_was_not_sent`). Every pending
        decision is dropped either way, so a stopped run does not carry one
        over to the next failure.

        :param failed: whether the run that just ended failed.
        """
        files_itself = getattr(self, "_report_files_itself", False)
        awaits_the_terms = getattr(self, "_report_awaits_the_terms", False)
        self._report_files_itself = False
        self._report_awaits_the_terms = False
        if failed and files_itself:
            self._report_waits_for_a_click = False
            self._file_the_report_automatically()
            return
        if failed and awaits_the_terms:
            self._report_waits_for_a_click = False
            self._console.append_notice(
                "[issue] Nothing was sent to GitHub. Automatic filing starts "
                "once you accept the terms of use (Help → Set spaCR up "
                "again…). To send this report now, press File as issue.\n")
            return
        self._say_the_report_was_not_sent(failed=failed)

    def _settings_snapshot(self) -> dict:
        """The settings form's current values, for a report.

        Read from the widgets, so it runs on the GUI thread.

        :returns: ``{key: value}``, or ``{}`` when the form cannot be read.
        """
        snapshot: dict = {}
        try:
            model = getattr(self, "_settings_model", None)
            if model is not None:
                for k, w in getattr(model, "_widgets", {}).items():
                    from PySide6.QtWidgets import (
                        QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit,
                        QSpinBox,
                    )
                    if isinstance(w, QCheckBox):
                        snapshot[k] = w.isChecked()
                    elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                        snapshot[k] = w.value()
                    elif isinstance(w, QComboBox):
                        snapshot[k] = w.currentText()
                    elif hasattr(w, "get_value"):
                        snapshot[k] = w.get_value()
                    elif isinstance(w, QLineEdit):
                        snapshot[k] = w.text()
        except Exception:                                    # noqa: BLE001
            snapshot = {}
        return snapshot

    def _file_the_report_automatically(self) -> None:
        """File the failed run's report without a preview ('always').

        Automatic filing is the default mode. Consent is Section 5.6 of the terms of use,
        and :meth:`_settle_the_report` calls this only for a profile that has
        accepted them (:meth:`_the_terms_allow_automatic_filing`).

        The report is the one "File as issue" would open, built by the same
        :func:`~spacr.qt.ai.issue_report.build_report`, and it is sent with
        the redaction the preview applies by default
        (:func:`~spacr.qt.ai.issue_report.public_report`). spaCR AI's
        analysis is attached only when the AI has already answered this
        error. A provider that failed leaves no analysis
        (``ai_explanation_of``).

        One crash is filed once. A fingerprint this profile has filed before
        is not filed again, nor is one still being filed. A fingerprint that
        already has an open issue gets a comment on it instead of a new
        issue (:func:`~spacr.qt.ai.issue_report.file_without_review`).

        Building and posting run on the background runner. Resolving the
        sign-in can run ``gh auth token``, and the log copy is file I/O.
        """
        tb = getattr(self, "_last_error_text", "") or ""
        if not tb:
            return
        from ..ai import issue_report
        from ..ai import settings as _ai_settings

        fingerprint = issue_report.fingerprint_of(tb)
        known = _ai_settings.auto_filed_url(fingerprint)
        if known:
            self._console.append_notice(
                "[issue] This error was reported from this computer before, "
                "so it was not filed again: {url}\n", url=known)
            return
        if _a_report_is_in_flight(fingerprint):
            self._console.append_notice(
                "[issue] This error is being reported already.\n")
            return
        try:
            analysis = self._console.ai_explanation_of(tb)
        except Exception:                                    # noqa: BLE001
            analysis = ""
        try:
            from ..preferences import get_share_diagnostic_logs
            keep_log = bool(get_share_diagnostic_logs())
        except Exception:                                    # noqa: BLE001
            keep_log = False
        settings_snapshot = self._settings_snapshot()
        app_key = self.app_key
        _REPORTS_BEING_FILED[fingerprint] = time.monotonic()

        def _file():
            """Build, redact and post the report. Off the GUI thread."""
            try:
                report = issue_report.public_report(issue_report.build_report(
                    tb, active_app=app_key, settings=settings_snapshot,
                    include_log_tail=keep_log, ai_response=analysis))
                outcome = issue_report.file_without_review(report)
            except Exception as exc:      # noqa: BLE001 - reported, not hidden
                outcome = {"status": issue_report.FAILED,
                           "detail": f"{type(exc).__name__}: {exc}"}
            outcome["fingerprint"] = fingerprint
            return outcome

        self._console.append_notice(
            "[issue] Filing a redacted report of this error on the public "
            "spaCR GitHub repository, because issue reporting is set to "
            "'always'…\n")
        if not self._jobs.submit(_file, self._on_report_filed_automatically):
            _REPORTS_BEING_FILED.pop(fingerprint, None)

    def _on_report_filed_automatically(self, outcome: dict) -> None:
        """Say where the automatic report went, or why it did not. GUI thread.

        A report that was filed, or added to an open issue, is remembered by
        its fingerprint, so the same crash is not filed again from here. One
        that was not sent is not remembered, so the next failure tries again.

        :param outcome: what
            :func:`~spacr.qt.ai.issue_report.file_without_review` returned,
            with the ``fingerprint`` added.
        """
        from ..ai import issue_report
        from ..ai import settings as _ai_settings

        outcome = dict(outcome or {})
        fingerprint = str(outcome.get("fingerprint", "") or "")
        _REPORTS_BEING_FILED.pop(fingerprint, None)
        status = outcome.get("status")
        url = str(outcome.get("url", "") or "")
        if status in (issue_report.FILED, issue_report.SEEN_AGAIN):
            try:
                _ai_settings.remember_auto_filed(fingerprint, url)
            except Exception:                                # noqa: BLE001
                LOG.debug("could not remember the filed report",
                          exc_info=True)
        if status == issue_report.FILED:
            self._console.append_notice(
                "[issue] Filed on GitHub: {url}\n", url=url)
        elif status == issue_report.SEEN_AGAIN:
            self._console.append_notice(
                "[issue] This error already has an open issue on GitHub. "
                "This occurrence was added to it: {url}\n", url=url)
        elif status == issue_report.SIGNED_OUT:
            self._console.append_notice(
                "[issue] Not filed: spaCR is not signed in to GitHub. Run "
                "`gh auth login` in a terminal once and later errors are "
                "filed automatically. To send this one, press File as "
                "issue.\n")
        else:
            self._console.append_notice(
                "[issue] Not filed: {detail}. Press File as issue to try "
                "again.\n",
                detail=str(outcome.get("detail", "") or "unknown error"))

    def _on_lp_switch(self, on: bool) -> None:
        """Compatibility route for callers that still name Mask's LP switch."""
        card = getattr(self, "_live_preview_card", None)
        if card is None:
            return
        card.setVisible(on)

    def _on_preview_switch(self, on: bool) -> None:
        """Show or hide the preview while keeping its Live switch reachable.

        When closed, the switch sits beside the Actions heading, outside its
        folding body. When open, it rides on the preview card. Hidden lazy
        previews therefore remain unbuilt until the user opens them.

        Opening it also seeds the panel from the form, once. Before that,
        this screen wired only the push direction — ``set_propagate_callback``
        — so all four previews ran at their own hardcoded defaults. A user
        set ``cell_flow_threshold``, opened Live preview to check the segmentation, and
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
        switch = getattr(self, "_preview_switch", None)
        heading_row = getattr(self, "_actions_heading_row", None)
        if switch is not None and heading_row is not None:
            if on and hasattr(card, "add_title_action"):
                heading_row.removeWidget(switch)
                card.add_title_action(switch)
            else:
                title_row = getattr(card, "_title_row", None)
                if title_row is not None:
                    title_row.removeWidget(switch)
                heading_row.addWidget(switch)
            blocked = switch.blockSignals(True)
            switch.setChecked(on)
            switch.blockSignals(blocked)
            switch.show()
        if on and not getattr(self, "_preview_primed", False):
            self._preview_primed = True
            self._prime_preview()
            if attr == "_live_preview_card":
                self._autoload_live_preview(self._settings_src_path() or "")
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
            model = getattr(self, "_settings_model", None)
            if model is not None:
                self._hyperparam.apply_settings(model.collect())

    def _on_sweep_switch(self, on: bool) -> None:
        """Show/hide the Parameter sweep card when its toggle flips."""
        card = getattr(self, "_sweep_card", None)
        if card is None:
            return
        card.setVisible(on)
        if on:
            model = getattr(self, "_settings_model", None)
            panel = getattr(self, "_sweep", None)
            if model is not None and panel is not None:
                try:
                    panel.apply_settings(model.collect())
                except Exception:
                    LOG.debug("could not seed the parameter sweep",
                              exc_info=True)

    def _on_umap_gpu_switch(self, on: bool) -> None:
        """Keep one truthful GPU state across the main and search pipelines."""
        panel = getattr(self, "_hyperparam", None)
        model = getattr(self, "_settings_model", None)
        if panel is None or self.app_key != "umap":
            return
        enabled = bool(panel.request_gpu_enabled(
            bool(on), anchor=getattr(self, "_gpu_switch", None)))
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

    def _build_umap_explorer(self) -> QWidget:
        """Build the explorer on its first payload or direct access.

        The common deferred-part lifecycle applies the current style and
        language. Switching between result modes reuses this panel and its
        payload rather than constructing another explorer.
        """
        from ..widgets import ImageUmapExplorer

        explorer = ImageUmapExplorer(parent=self._figures_card)
        explorer.hide()
        explorer.set_propagate_callback(self._propagate_live_settings)
        explorer._settings_getter = self._umap_display_defaults
        self._figures_card.body_layout.addWidget(explorer, 1)
        self._umap_explorer = explorer
        return explorer

    def _on_interactive_switch(self, on: bool) -> None:
        """Switch UMAP results between the static figure and explorer.

        The toggle can be enabled before a run.  In that case the current
        console/figure layout stays put until a UMAP payload arrives, then
        :meth:`_on_figure_ready` opens the explorer automatically.
        """
        explorer = self._if_built("_umap_explorer")
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

    def _apply_ai_default(self) -> None:
        """Turn the AI switch on at launch when the preference says to.

        Only ON is applied. The preference is "start with it on", not "keep
        it on": a user who turns the switch off mid-session means it, and
        re-asserting the default on the next screen would fight them.

        Never raises: a preference that cannot be read is not a reason for a
        module screen to fail to build.
        """
        try:
            from ..preferences import get_ai_on_by_default

            if not bool(get_ai_on_by_default()):
                return
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not read the AI default", exc_info=True)
            return
        try:
            self._ai_switch.setChecked(True)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not turn the AI switch on", exc_info=True)

    def _on_ai_switch(self, on: bool) -> None:
        """Turn the console's AI on or off, picking a provider if none is set.

        With no vendor CLI installed the switch turns itself back off and says
        where to configure one, rather than leaving an armed toggle that cannot
        answer.

        :param on: the switch's new state.
        """
        self._console.set_ai_active(on)
        if on:
            from .. import ai as ai_module
            if not self._console._current_provider_name:
                configured = ai_module.configured_providers()
                if configured:
                    self._console.set_ai_provider(
                        self._wanted_provider() or configured[0].name)
                else:
                    self._console.append_notice(
                        "[AI] No vendor CLI installed. "
                        "Preferences → AI → Providers…\n"
                    )
                    self._ai_switch.setChecked(False)

    def _wanted_provider(self):
        """The provider Preferences asks for, if it is actually installed.

        :returns: the chosen provider's name, or ``""`` to let the console
            take the first available one.
        """
        from .. import ai as ai_module
        from ..preferences import get_preferred_provider

        wanted = get_preferred_provider()
        if not wanted:
            return ""
        try:
            names = {p.name for p in ai_module.configured_providers()}
        except Exception:                                    # noqa: BLE001
            return ""
        return wanted if wanted in names else ""

    def _on_explain_error(self):
        """Send the last traceback to the console's error flow.

        The legacy signal is emitted as well, for ``MainWindow``'s older dock
        path. Nothing happens when no error has been seen.
        """
        if not self._last_error_text:
            return
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

        from ..preferences import ISSUE_PROMPT_NEVER, get_issue_prompt_mode
        mode = get_issue_prompt_mode()
        if mode == ISSUE_PROMPT_NEVER:
            self._console.append_notice(
                "\nNot filing a report: issue reporting is set to 'never' in "
                "Preferences.\n")
            return
        settings_snapshot = self._settings_snapshot()
        from PySide6.QtWidgets import QDialog
        from ..ai.issue_preview import IssuePreviewDialog
        from ..ai.issue_report import build_report, submit_report
        from ..preferences import get_share_diagnostic_logs

        try:
            ai_analysis = self._console.ai_explanation_of(self._last_error_text)
        except Exception:                                    # noqa: BLE001
            ai_analysis = ""
        report = build_report(
            self._last_error_text,
            active_app=self.app_key,
            settings=settings_snapshot,
            include_log_tail=get_share_diagnostic_logs(),
            ai_response=ai_analysis,
        )
        preview = IssuePreviewDialog(
            report, self, console=self._console,
            traceback_text=self._last_error_text)
        if preview.exec() != QDialog.Accepted:
            self._console.append_notice(
                "[issue] cancelled — nothing was sent.\n")
            return
        approved_report = preview.approved_report()

        def _file():
            """Submit the report, returning the failure AS DATA rather than raising.

            The call is asynchronous, so an ``except`` around the caller can no
            longer see it -- and a report that silently fails to send is worse than
            one that fails loudly.
            """
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

    def _crop_choice_warnings(self, settings) -> list:
        """What is worth saying about this run's crop settings, in order.

        Returned as a list rather than shown from here, so WHAT is warned
        about is testable without a dialog in the way.
        """
        notes = []
        if settings.get("normalize"):
            notes.append(
                "NORMALISING THE CROPS LOSES INFORMATION THAT CANNOT BE "
                "RECOVERED.\n\n"
                "The annotation viewer and the training pipeline each apply "
                "their own normalisation, at the point they display or learn "
                "from an image. Normalising here as well rescales the raw "
                "pixels before either of them sees them, so intensity "
                "differences between cells -- often the phenotype itself -- "
                "are flattened into one displayed range, and the crops on "
                "disk cannot be un-normalised afterwards.")
        if settings.get("use_bounding_box"):
            notes.append(
                "A BOUNDING BOX INCLUDES WHAT IS AROUND THE OBJECT, NOT ONLY "
                "THE OBJECT.\n\n"
                "The crop becomes the rectangle enclosing the mask, so "
                "neighbouring cells, debris and background inside that "
                "rectangle are kept. A classifier trained on those crops can "
                "learn the neighbourhood rather than the cell.")
        if notes:
            notes.append(
                "YOU DO NOT HAVE TO DECIDE THIS NOW.\n\n"
                "Once Measure has finished you can build a training or "
                "annotation set by streaming crops straight from the merged "
                "arrays, or from the coordinate columns in measurements.db, "
                "without measuring again. Streaming from the database gives "
                "BOUNDING-BOX crops only, because coordinates are all it "
                "stores; streaming from the arrays uses the object masks, so "
                "it can cut to the object itself.")
        return notes

    def _confirm_crop_choices(self, settings) -> bool:
        """Ask before a crop setting that changes every downstream image.

        :returns: whether the run should go ahead. ``True`` when there is
            nothing to warn about, so an ordinary run is never interrupted.
        """
        notes = self._crop_choice_warnings(settings)
        if not notes:
            return True
        answer = QMessageBox.question(
            self, tr("Check the crop settings"),
            "\n\n".join(notes) + "\n\n" + tr("Run with these settings?"),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

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
        """Write live-preview-tuned values into the main settings panel.

        The panel speaks Mask's setting names. A module that declares a
        translation in :data:`spacr.qt.preview_registry.PREVIEWS` -- Plaque
        Assay segments with ``diameter`` and ``plaque_model``, not
        ``cell_diameter`` and ``model_name`` -- gets its own names, and a
        name it has no use for is dropped rather than written to nothing.
        """
        model = getattr(self, "_settings_model", None)
        if model is None:
            return
        from ..preview_registry import PREVIEWS

        spec = PREVIEWS.get(str(getattr(self, "app_key", "")))
        rename = spec.propagation if spec is not None and self.app_key != "mask" else None
        for key, value in settings.items():
            target = rename.get(key) if rename else key
            if target is None:
                continue
            model.set_value_for_key(target, value)

    def _on_figure_ready(self, fig, png_path: str = "") -> None:
        """Hand a matplotlib figure to the FigureQueue. ``png_path`` is a PNG
        the pipeline bridge already rendered in its worker thread, so the queue
        can adopt it (cheap) instead of re-rendering on the GUI thread — that's
        what keeps the UI responsive while many figures stream in."""
        payload = getattr(fig, "_spacr_umap_payload", None)
        explorer = (self._umap_explorer
                    if payload is not None and self.app_key == "umap"
                    else self._if_built("_umap_explorer"))
        if payload is not None and explorer is not None:
            explorer.set_payload(payload)
            self._umap_payload_ready = True
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
        self._queue_figure_grid_refresh()

    def closeEvent(self, event):
        """Cancel and join this screen's worker before destroying widgets.

        A worker that has not reached a safe boundary keeps the screen alive;
        dropping its references or force-terminating it could corrupt an
        output and triggers Qt's fatal "QThread destroyed while running".

        :param event: the close event; ignored (so the screen stays open)
            when the worker is still running three seconds after the cancel
            request.
        """
        builder = self.__dict__.get("_idle_prebuild")
        if builder is not None:
            builder.stop()
        self._stop_the_heartbeat()
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
        try:
            self._usage_generation += 1
            self._usage_timer.stop()
        except (AttributeError, RuntimeError):
            pass
        for name in ("_usage_jobs", "_jobs"):
            jobs = getattr(self, name, None)
            if jobs is not None:
                try:
                    jobs.shutdown()
                except RuntimeError:
                    pass
        montage = self._if_built("_cell_montage")
        if montage is not None:
            try:
                montage.shutdown()
            except RuntimeError:
                pass
        flowview = getattr(self, "_flowview_section", None)
        if flowview is not None:
            try:
                flowview.shutdown()
            except RuntimeError:
                pass
        self._shutdown_settings_widgets()
        try:
            self.unregister_workspace()
        except Exception:                                       # noqa: BLE001
            LOG.debug("could not withdraw the workspace sections", exc_info=True)
        fq = getattr(self, "_figure_queue", None)
        if fq is not None:
            try:
                fq.clear()
            except Exception:
                pass
        explorer = self._if_built("_umap_explorer")
        if explorer is not None:
            try:
                explorer.close()
            except Exception:
                pass
        try:
            from ..widget_cleanup import retire_pyqtgraph_menus

            retire_pyqtgraph_menus(self)
        except (ImportError, RuntimeError):
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
            built = getattr(widgets, "built_items", None)
            if callable(built):
                values = [widget for _key, widget in built()]
            else:
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
        """Return the page to its resting state when a run ends.

        The heartbeat is stopped before anything else: one firing after the run
        would print "still fitting" under "Finished", and the last console
        line is the one a user reads. The backdrop gets its cores back here too,
        since this is the single door both a finished and a failed run take.

        :param ok: whether the run succeeded.
        """
        from ..button_roles import set_button_busy
        self._stop_the_heartbeat()
        _resume_the_fractal(self)
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        set_button_busy(self._btn_run, False)
        set_button_busy(self._btn_stop, False)
        self._progress.setVisible(False)
        cancelled = bool(
            getattr(getattr(self, "_worker", None), "was_cancelled", False))
        import time as _elapsed_time
        self._update_run_in_runs_tab(
            status=("stopped" if cancelled else ("ok" if ok else "failed")),
            seconds=round(_elapsed_time.time() - getattr(
                self, "_run_started_at", _elapsed_time.time()), 1))
        if cancelled:
            self._console.append_notice(
                "■ Stopped safely at a field, trial, or job boundary\n")
        else:
            self._console.append_notice(
                "✓ Finished\n" if ok else
                "✗ Failed — see traceback above\n")
        self._settle_the_report(failed=not ok and not cancelled)
        if (ok and not cancelled and getattr(self, "_results_panel", None)
                and not getattr(self, "_results_loaded_in_memory", False)):
            try:
                self._load_regression_results()
            except Exception:
                LOG.debug("could not open the regression results",
                          exc_info=True)

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

    def _record_run_in_runs_tab(self, label, source, settings):
        """Record a starting regression run on the Runs tab.

        Return the run handle, or ``None`` when this screen has no Runs tab.
        """
        runs = getattr(self, "_sweep_runs", None)
        if runs is None:
            return None
        try:
            return runs.record_run(label, source, settings)
        except Exception:
            LOG.debug("could not record the run in the Runs tab",
                      exc_info=True)
            return None

    def _update_run_in_runs_tab(self, **fields) -> bool:
        """Update the row for the run this screen last started."""
        runs = getattr(self, "_sweep_runs", None)
        handle = getattr(self, "_run_handle", None)
        if runs is None or handle is None:
            return False
        try:
            return bool(runs.update_run(handle, **fields))
        except Exception:
            LOG.debug("could not update the run in the Runs tab",
                      exc_info=True)
            return False

    def _refresh_figure_grid(self) -> None:
        """Rebuild the grid from whatever the queue is holding.

        Called on the debounce timer rather than per figure: a run streams
        seventeen in and the grid is on screen the whole time, so one rebuild
        per arrival is seventeen full relayouts of the same view.
        """
        grid = getattr(self, "_figure_grid", None)
        if grid is None:
            return
        try:
            grid.set_figures(self._figure_queue.all_pixmaps(),
                             self._figure_queue.figure_titles(),
                             sections=self._figure_queue.run_sections())
        except Exception:
            LOG.debug("could not build the figure grid", exc_info=True)
        self._pin_regression_graph()

    #: What the live tile is captioned. It has to distinguish itself from the
    #: run's own `volcano` panel, which is a cell of the same grid: they are
    #: the same numbers drawn twice on purpose -- one is the publication panel
    #: and one is the tool -- and a reader who cannot tell which is which has
    #: two identical-looking tiles and no reason to press either.
    LIVE_TILE_TITLE = "regression — interactive"

    def _pin_regression_graph(self) -> None:
        """Put the LIVE regression graph on the all-figures grid.

        "the regression plot isnt shown in all figures (i want it also shown
        there)". It was not, and there was no route to it from the grid at
        all: every tile opens a saved picture, and the only other way back to
        the interactive volcano was to select a row in the coefficient table.
        A view you can leave and not return to is the trap the "← All figures"
        button exists to avoid, pointing the other way.

        THE TILE WAS TRIED ONCE AND REMOVED, and the reason it was removed was
        not the tile. It grabbed a widget sitting on a stacked page nobody had
        opened, which on the real screen is 100x9 pixels inside a collapsed
        splitter, so `grab()` returned a one-colour rectangle -- the "blank
        box with a caption under it". :meth:`FastPlot.snapshot` sizes the
        widget and lays it out before grabbing, and the same widget yields a
        readable picture. Measured: 1 distinct colour before, 260 after.
        """
        grid = getattr(self, "_figure_grid", None)
        panel = getattr(self, "_results_panel", None)
        if grid is None or panel is None:
            return
        try:
            frame = panel.results_frame()
            if frame is None or not len(frame):
                grid.set_pinned(None, "")
                return

            from ..widgets.figure_grid_view import live_tiles_from_panels

            grid.set_live_tiles(live_tiles_from_panels([
                ("regression", self.LIVE_TILE_TITLE, panel.volcano),
                ("effect_rank", "Effect ranking", panel.effect_rank),
                ("p_values", "p-values", panel.p_values),
                ("qq", "Q-Q", panel.qq),
                ("controls", "Controls", panel.controls),
                ("agreement", "Guide support", panel.agreement),
                ("residuals", "Residuals", panel.residuals),
                ("scale_location", "Scale-location", panel.scale_location),
                ("influence", "Influence", panel.influence),
            ]) + self._gene_tile_entry(panel))
        except Exception:
            LOG.debug("could not pin the live regression graph", exc_info=True)

    #: Live-tile key -> the panel attribute holding the live widget, for
    #: the right-click menu. Only panels that HAVE a style menu appear:
    #: absent here means the tile opens on a left click and has no menu,
    #: which is the honest state, not a bug.
    _LIVE_TILE_WIDGETS = {
        "regression": "volcano",
        "effect_rank": "effect_rank",
        "p_values": "p_values",
        "qq": "qq",
        "controls": "controls",
        "agreement": "agreement",
        "residuals": "residuals",
        "scale_location": "scale_location",
        "influence": "influence",
    }

    def _open_live_tile(self, key: str) -> None:
        """A live tile was pressed: raise the panel it photographs.

        THE MISSING DOOR (199). "i am able to click on the colcano plot to
        open it but not the other pyqtgraphs" -- because the grid emits
        `live_tile_activated` for all nine live tiles and this screen was
        connected only to `pinned_activated`, which the grid emits for the
        volcano alone. Eight tiles were drawn, took a click, and reached a
        signal with no receiver.

        A TILE THAT DOES NOT OPEN IS WORSE THAN NO TILE: the user clicks
        twice and concludes the application is broken rather than that this
        particular picture has no door.
        """
        from ..widgets.figure_grid_view import PINNED_KEY

        if str(key) == PINNED_KEY:
            return
        panel = getattr(self, "_results_panel", None)
        if panel is None:
            return
        if not panel.show_panel(str(key)):
            self._console.append_notice(
                "■ That panel has no tab in this run's results.\n")
            return
        self._raise_the_results_tab()

    def _live_tile_menu(self, key: str, position) -> None:
        """Right-click on any live tile: that graph's OWN menu, in place.

        The general form of :meth:`_pinned_menu`, and missing for the same
        reason the open route was: `live_tile_menu_requested` had no
        receiver, so the gesture worked on the volcano and nowhere else.
        """
        from ..widgets.figure_grid_view import PINNED_KEY

        if str(key) == PINNED_KEY:
            return
        panel = getattr(self, "_results_panel", None)
        attribute = self._LIVE_TILE_WIDGETS.get(str(key))
        if panel is None or attribute is None:
            return
        widget = getattr(panel, attribute, None)
        builder = getattr(widget, "build_style_menu", None)
        if builder is None:
            return
        try:
            builder().exec(position)
        except Exception:
            LOG.debug("could not open the live tile's menu", exc_info=True)
        self._pin_regression_graph()

    def _pinned_menu(self, position) -> None:
        """Right-click on the live tile: the graph's OWN menu, in place.

        Not the figure queue's menu. The queue builds one from a matplotlib
        figure at an index, and this tile is neither -- it is a photograph of
        a live widget that has its own right-click menu with its own restyle,
        baselines, colour-by and re-fit on it. Showing that one means the
        gesture does the same thing on the tile as on the graph itself.
        """
        panel = getattr(self, "_results_panel", None)
        if panel is None:
            return
        try:
            panel.volcano.build_style_menu().exec(position)
        except Exception:
            LOG.debug("could not open the live tile's menu", exc_info=True)
        self._pin_regression_graph()

    def _show_publication_sheet(self) -> None:
        """Draw every panel as ONE publication-ready figure, and open it.

        "the all figures section should look like a publication ready
        figure". The tile grid answers "what did this run draw"; the sheet
        answers "what did this run FIND", which is a different question and
        the one a reader has.

        Built on demand rather than on every refresh: it is seven panels of
        matplotlib and the grid is rebuilt on a debounce while a run streams.
        """
        panel = getattr(self, "_results_panel", None)
        frame = panel.results_frame() if panel is not None else None
        if frame is None or not len(frame):
            self._console.append_stdout(
                "No coefficient table loaded, so there is nothing to draw a "
                "figure from. Open a finished run first.\n")
            return
        try:
            from ...figures import build_sheet
        except Exception:
            from spacr.figures import build_sheet
        try:
            sheet = build_sheet(frame, width="double")
        except Exception as error:      # noqa: BLE001 - never take the GUI down
            LOG.debug("could not build the sheet", exc_info=True)
            self._console.append_stdout(
                f"Could not draw the publication figure: {error}\n")
            return
        self._figure_queue.add_figure(sheet.figure)
        self._figure_queue.show_index(self._figure_queue.count() - 1)
        stack = getattr(self, "_figures_stack", None)
        if stack is not None and getattr(self, "_figure_detail", None):
            stack.setCurrentWidget(self._figure_detail)
        if sheet.skipped:
            self._console.append_stdout(
                "Publication figure: "
                + "; ".join(f"{p.title} not shown ({p.reason})"
                            for p in sheet.skipped) + "\n")
        self._console.append_stdout(sheet.legend() + "\n")

    def _queue_figure_grid_refresh(self) -> None:
        """Ask for a rebuild soon. Coalesces a burst into one."""
        timer = getattr(self, "_grid_refresh", None)
        if timer is not None:
            timer.start()

    def _attached_database_rows(self):
        """The regression input table's rows, for the Measurements tab.

        A zero-argument callable rather than a snapshot, because the user
        drops databases AFTER the panel is built and a list captured at
        construction would never grow.

        Returns [] rather than raising when this screen has no paired-data
        widget -- the Measurements tab is built on the regression screen and
        the provider must not assume it.
        """
        model = getattr(self, "_settings_model", None)
        widget = getattr(model, "_widgets", {}).get("paired_data") \
            if model is not None else None
        if widget is None:
            return []
        try:
            return list(widget.get_value() or [])
        except Exception:                                        # noqa: BLE001
            return []

    def _measurements_destination(self) -> str:
        """Where the Measurements tab writes its merged frame.

        THE SAME ROOT THE RUNS WRITE UNDER, in a folder of its own, so the
        artefact sits beside the results it produced rather than in a
        temporary directory the user cannot find.

        THE RULE IS `spacr.refit.destination`'s RULE, in the same order, so
        the merged frame and the runs fitted from it cannot end up in
        different projects: ``src`` when the module has one -- the regression
        screen does NOT, checked, which is why the count file is the live
        branch here -- then the first count file's folder, then the plate
        folder of the first attached database for a project whose counts are
        not named yet.
        """
        import os as _os

        model = getattr(self, "_settings_model", None)
        settings = {}
        if model is not None:
            try:
                settings = model.collect() or {}
            except Exception:                                    # noqa: BLE001
                settings = {}
        root = str(settings.get("src") or "").strip()
        if root in ("path", "/path", "/path/to/src"):
            root = ""
        if not root:
            root = _os.path.dirname(_first_count_file(settings))
        if not root:
            for row in self._attached_database_rows():
                database = (row.get("database") or row.get("db") or "") \
                    if isinstance(row, dict) else ""
                if database:
                    folder = _os.path.dirname(str(database))
                    root = (_os.path.dirname(folder)
                            if _os.path.basename(folder) == "measurements"
                            else folder)
                    break
        return _os.path.join(root, "measurements") if root else ""

    def _column_fit_settings(self) -> dict:
        """This screen's settings, for one fit of the column queue.

        Read LIVE and copied by the queue at the moment Run is pressed --
        never per fit, which would let a user editing the panel mid-queue fit
        twelve different models and compare them as though only the response
        had changed.
        """
        model = getattr(self, "_settings_model", None)
        if model is None:
            return {}
        try:
            return dict(model.collect() or {})
        except Exception:                                        # noqa: BLE001
            LOG.debug("could not read the settings for a column fit",
                      exc_info=True)
            return {}

    def _on_column_fit_started(self, column: str, settings: dict) -> None:
        """A column's fit began: give it a row in the Runs tab."""
        import datetime as _dt

        from ..widgets.sweep_runs import SOURCE_MEASUREMENT

        label = _dt.datetime.now().strftime(f"{column}  %H:%M:%S")
        handle = self._record_run_in_runs_tab(label, SOURCE_MEASUREMENT,
                                              settings)
        if handle is not None:
            self._column_run_handles[str(column)] = handle

    def _on_column_fit_finished(self, column: str, outcome: dict) -> None:
        """A column's fit is decided: say so on ITS row.

        Its OWN handle, not `_run_handle`. A queue puts several rows up at
        once and `_update_run_in_runs_tab` only knows about the last run this
        screen started -- using it here would move every outcome onto one row
        and the other eleven would say "running" for ever.
        """
        runs = getattr(self, "_sweep_runs", None)
        handle = self._column_run_handles.pop(str(column), None)
        if runs is None or handle is None:
            return
        ok = bool((outcome or {}).get("ok"))
        try:
            runs.update_run(
                handle,
                status="ok" if ok else "failed",
                folder=str((outcome or {}).get("folder") or "") or None,
                n_results=(outcome or {}).get("n_results") if ok else None,
                error_type=None if ok else
                str((outcome or {}).get("error") or "did not fit"))
        except Exception:                                        # noqa: BLE001
            LOG.debug("could not update the column fit's row", exc_info=True)


    #: How many runs may be LIVE at once. Two: a comparison needs two, and
    #: every one after that is bought at 74.99 ms a frame.
    MAX_LIVE_RUNS = 2

    #: Said out loud when the bound refuses. A bound discovered through a
    #: refusal with no reason is indistinguishable from a broken button.
    LIVE_RUNS_NOTE = (
        "Two runs can be live at once. A live plot costs about 75 ms a frame "
        "against 5 ms for a still, on a 17 ms budget, so a third would be "
        "paid for on every drag of the window. Close the run beside this one "
        "to open another.")

    def live_run_count(self) -> int:
        """How many runs have a live, interactive plot on screen right now."""
        count = 1 if getattr(self, "_results_panel", None) is not None else 0
        if getattr(self, "_compare_panel", None) is not None:
            count += 1
        return count

    def open_run_beside(self, record) -> bool:
        """Open a second run's results beside the loaded one.

        DELIBERATE, NOT THE DEFAULT. Reached from the Runs
        tab's context menu; nothing opens a second run on its own.

        :param record: a Runs-tab row, or a run folder.
        :returns: whether a second run is now live.
        """
        folder = (str(record.get("folder") or "") if isinstance(record, dict)
                  else str(record or ""))
        if self._results_panel is None or self._results_split is None:
            return False
        if not folder:
            self._console.append_notice(
                "■ That run has no folder on disk, so there is nothing "
                "to open beside this one.\n")
            return False
        if self._same_run_folder(self._results_panel.run_folder(), folder):
            self._console.append_notice(
                "■ That run is the one already on screen.\n")
            return False
        if self.live_run_count() >= self.MAX_LIVE_RUNS:
            self._console.append_notice("■ {note}\n",
                                        note=self.LIVE_RUNS_NOTE)
            return False

        from ..widgets.regression_results import RegressionResultsPanel

        panel = RegressionResultsPanel(self._results_split)
        if not panel.load(folder):
            panel.setParent(None)
            panel.deleteLater()
            self._console.append_notice(
                "■ No results in {folder} to open beside this run.\n",
                folder=folder)
            return False
        self._compare_panel = panel
        self._results_split.addWidget(panel)
        self._results_split.setSizes([1, 1])
        self._raise_the_results_tab()
        self._console.append_notice(
            "■ {name} is open beside the loaded run. {note}\n",
            name=panel.run_name(), note=self.LIVE_RUNS_NOTE)
        return True

    def close_run_beside(self) -> bool:
        """Take the second run's live plot away, keeping its PHOTOGRAPH.

        A still stands in for a run that is not live. The picture stays at
        about 5 ms a frame instead of 75, and the
        run's plot STATE was never in the widget -- so making it live again
        is cheap, and that is what the bound buys.
        """
        panel = getattr(self, "_compare_panel", None)
        if panel is None:
            return False
        folder = panel.run_folder()
        photo = None
        try:
            photo = panel.volcano.grab()
        except Exception:                                        # noqa: BLE001
            LOG.debug("could not photograph the run beside", exc_info=True)
        if folder and photo is not None and not photo.isNull():
            self._run_photographs[os.path.abspath(folder)] = photo
        self._compare_panel = None
        panel.setParent(None)
        panel.deleteLater()
        return True

    def run_photograph(self, folder):
        """The still kept for a run that is no longer live, or ``None``.

        :param folder: the run's output folder; looked up by absolute path,
            and an empty value gives ``None``.
        """
        if not folder:
            return None
        return self._run_photographs.get(os.path.abspath(str(folder)))

    def _raise_the_results_tab(self) -> None:
        """Bring the Results page forward, whatever is sharing it.

        The page is a SPLITTER rather than the panel, since a second run can
        be opened beside the first, and `setCurrentWidget` only accepts a
        widget the tab bar itself owns -- so a call naming the panel becomes
        a silent no-op the moment the panel stops being the page.
        """
        tabs = getattr(self, "_results_tabs", None)
        page = (getattr(self, "_results_page", None)
                or getattr(self, "_results_panel", None))
        if tabs is None or page is None:
            return
        self._figures_card.show()
        folder = getattr(self._figures_card, "folder", None)
        if folder is not None:
            folder.set_shut(False, by_user=False)
        try:
            tabs.setCurrentWidget(page)
        except (RuntimeError, TypeError):
            LOG.debug("could not raise the results tab", exc_info=True)

    def _on_runs_removed(self, records) -> None:
        """Discard retained views for runs removed from the Runs tab.

        This prevents a later run at the same path from inheriting the removed
        run's level, colors, limits, thresholds, selection, or figure tiles.
        """
        panel = getattr(self, "_results_panel", None)
        if panel is None or not records:
            return
        beside = getattr(self, "_compare_panel", None)
        for record in records:
            folder = str((record or {}).get("folder") or "")
            if not folder:
                continue
            self._run_photographs.pop(os.path.abspath(folder), None)
            if beside is not None and self._same_run_folder(
                    beside.run_folder(), folder):
                self.close_run_beside()
                self._run_photographs.pop(os.path.abspath(folder), None)
                beside = None
            try:
                panel.forget_run(folder)
            except Exception:                                    # noqa: BLE001
                LOG.debug("could not forget the run in %s", folder,
                          exc_info=True)

            queue = getattr(self, "_figure_queue", None)
            label = str((record or {}).get("run") or "")
            if queue is not None and label:
                try:
                    if queue.forget_run(label):
                        self._queue_figure_grid_refresh()
                except Exception:                                # noqa: BLE001
                    LOG.debug("could not forget %s's figures", label,
                              exc_info=True)

    def _on_loaded_run_changed_refresh_tabs(self, _record=None) -> None:
        """Re-read the Cells and Measurements tabs when the run changes.

        Opening a tab already re-reads it. That is not enough: the tab a user
        is LOOKING AT when they load another run is never opened again, so it
        kept the previous run's content while every other view moved. Showing
        one run's cells under another run's name is the plausible-and-wrong
        output this screen is most careful about.

        The montage's grid is emptied rather than rebuilt. Its contents
        answer a coefficient selected from the previous run's table, which
        means nothing for the new one; the selection that arrives with the
        new table fills it again. Never raises -- a tab that cannot refresh
        must not take the run change down with it.
        """
        montage = getattr(self, "_cell_montage", None)
        if montage is not None:
            for step, call in (("empty", getattr(montage, "clear", None)),
                               ("re-read", getattr(montage, "refresh", None))):
                try:
                    if call is not None:
                        call()
                except Exception:                                # noqa: BLE001
                    LOG.debug("could not %s the cells tab", step,
                              exc_info=True)
        scan = getattr(self, "_scan_panel", None)
        if scan is not None:
            try:
                scan.refresh_databases()
            except Exception:                                    # noqa: BLE001
                LOG.debug("could not refresh the measurements tab",
                          exc_info=True)

    def _on_results_tab_changed(self, index: int) -> None:
        """Opening a tab re-reads what it shows.

        The Runs tab re-reads the sweep's table; the Measurements tab
        re-reads the databases attached to the input table, which is the
        whole reason it needs an on-open refresh -- they are dropped while it
        is not the visible tab.

        It used to return early unless the widget was the Runs tab, so the
        Measurements tab never refreshed at all.
        """
        tabs = getattr(self, "_results_tabs", None)
        if tabs is None:
            return
        current = tabs.widget(index)

        montage = getattr(self, "_cell_montage", None)
        if montage is not None and current is montage:
            try:
                montage.refresh()
            except Exception:
                LOG.debug("could not refresh the cells tab", exc_info=True)
            return

        scan = getattr(self, "_scan_panel", None)
        if scan is not None and current is scan:
            try:
                scan.refresh_databases()
            except AttributeError:
                pass
            except Exception:
                LOG.debug("could not refresh the measurements tab",
                          exc_info=True)
            return

        runs = getattr(self, "_sweep_runs", None)
        if runs is None or current is not runs:
            return
        folder = self._sweep_destination()
        if folder:
            runs.load(folder)

    def _results_source_path(self) -> str:
        """Return the folder for the regression run shown on this screen.

        Prefer the selected Runs-tab entry so every view follows the active
        run. Fall back to the results panel for tables loaded directly from
        disk without a Runs-tab entry.
        """
        runs = getattr(self, "_sweep_runs", None)
        if runs is not None:
            try:
                folder = runs.loaded_run_folder()
            except Exception:                                    # noqa: BLE001
                folder = ""
            if folder:
                return str(folder)
        panel = getattr(self, "_results_panel", None)
        if panel is None:
            return ""
        try:
            return str(panel.run_folder() or "")
        except AttributeError:
            return str(getattr(panel, "_path", "") or "")

    def _keep_the_effects_grid(self, result) -> None:
        """Write a finished sweep's effects grid beside the run.

        Failure is logged and never raised: a sweep that produced its answer
        has not failed because the montage could not be told about it.
        """
        import os

        effects = getattr(result, "effects", None)
        if effects is None or not len(effects):
            return
        try:
            from ...cell_montage import write_effects_grid

            source = str(self._results_source_path() or "")
            folder = source if os.path.isdir(source) else os.path.dirname(source)
            if folder:
                write_effects_grid(effects, folder)
        except Exception:                                        # noqa: BLE001
            LOG.debug("could not keep the sweep's effects grid", exc_info=True)

    def _sweep_counts(self):
        """The per-well guide fractions the sweep needs.

        The SAME reader the montage uses, from the SAME input table -- the
        count CSVs the run was fitted on. A second way of turning counts into
        fractions would be a second answer to what a fraction is.
        """
        from ...cell_montage import fractions_from_counts

        try:
            paths = [row.get("count", "") for row in
                     (self._attached_database_rows() or [])
                     if isinstance(row, dict)]
            paths = [p for p in paths if p]
            return fractions_from_counts(paths) if paths else None
        except Exception:                                # noqa: BLE001
            LOG.debug("could not build the sweep's counts", exc_info=True)
            return None

    def _sweep_scores(self):
        """The per-object classification scores, for the circularity column.

        The merged measurements frame has NO score column -- it is the
        measurement tables -- so without these the column is NaN and the sweep
        says so rather than reporting zeros.
        """
        import pandas as pd

        try:
            frames = []
            for row in (self._attached_database_rows() or []):
                path = row.get("score", "") if isinstance(row, dict) else ""
                if path and os.path.isfile(str(path)):
                    frames.append(pd.read_csv(str(path)))
            if not frames:
                return None
            joined = pd.concat(frames, ignore_index=True, sort=False)
            from ...utils import correct_metadata_column_names

            return correct_metadata_column_names(joined)
        except Exception:                                # noqa: BLE001
            LOG.debug("could not read the sweep's scores", exc_info=True)
            return None

    def _scan_source_frame(self):
        """The well-level frame the measurement scan should run on.

        Taken from the regression's own merged data when a run has written
        one, because that frame already carries the gene assignment beside
        the measurements -- which is exactly what the scan needs and what
        nothing else in the app assembles.
        """
        import os as _os

        folder = self._results_source_path()
        if not folder:
            return None
        folder = _os.path.abspath(folder)
        for name in ("regression_data.csv", "merged_data.csv"):
            candidate = _os.path.join(folder, name)
            if _os.path.isfile(candidate):
                import pandas as _pd
                try:
                    return _pd.read_csv(candidate)
                except Exception:
                    LOG.debug("could not read %s", candidate, exc_info=True)
                    return None
        return None

    def _sweep_destination(self) -> str:
        """Where the sweep card was told to write, if it exists.

        Asked of the card rather than stored, so the two cannot disagree
        about which folder the sweep is filling.
        """
        sweep = getattr(self, "_sweep", None)
        field = getattr(sweep, "destination", None)
        try:
            return field.text().strip()
        except Exception:
            return ""

    def _show_trial(self, record: dict) -> None:
        """Show a selected run's saved results, figures, and retained view.

        Saved output is loaded without refitting. The Results tab is raised
        only after a table loads successfully; failed or incomplete trials
        remain visible in the Runs tab with their diagnostic message.
        """
        from ..widgets.sweep_runs import STATUS_RUNNING

        if not isinstance(record, dict):
            return
        named = record.get("run")
        trial = (named.strip() if isinstance(named, str) and named.strip()
                 else f"Trial {record.get('trial_id', '?')}")
        status = str(record.get("status", "ok"))
        if status == STATUS_RUNNING:
            self._console.append_stdout(
                f"{trial} is still going. Its results appear here when it "
                "finishes.\n")
            self._the_run_did_not_open(f"{trial} is still going.")
            return
        if status != "ok":
            self._console.append_stdout(
                f"{trial} did not produce a regression: "
                f"{record.get('error_type', '')} "
                f"{record.get('error', 'no reason recorded')}\n")
            self._the_run_did_not_open(
                f"{trial} did not produce a regression.")
            return
        folder = record.get("folder")
        panel = getattr(self, "_results_panel", None)
        if panel is None:
            return
        if folder and self._same_run_folder(panel.run_folder(), folder):
            self._figures_card.show()
            self._console.append_stdout(
                f"{trial} is already loaded — open the Results tab to "
                "see it.\n")
            return
        if folder and not os.path.isdir(str(folder)):
            folder = ""
        if not folder:
            self._console.append_stdout(
                f"{trial} has no saved results on disk. Re-run it from "
                "the sweep panel to draw them.\n")
            self._the_run_did_not_open(
                f"{trial} has no saved results on disk.")
            return
        self._pending_trial = (trial, str(folder))
        if not getattr(self, "_trial_load_wired", False):
            panel.load_finished.connect(self._on_trial_loaded)
            self._trial_load_wired = True
        if not panel.start_load(folder):
            return
        self._figures_card.show()

    def _on_trial_loaded(self, ok: bool) -> None:
        """The asynchronous half of :meth:`_show_trial`.

        Split off because the read is on a worker now: success is not known
        when `_show_trial` returns, so the figures and the rollback both have
        to wait for the answer rather than assuming it.
        """
        pending = getattr(self, "_pending_trial", None)
        if not pending:
            return
        trial, folder = pending
        self._pending_trial = None
        if not ok:
            self._console.append_stdout(
                f"{trial} has no saved results on disk. Re-run it from "
                "the sweep panel to draw them.\n")
            self._the_run_did_not_open(
                f"{trial} has no saved results on disk.")
            return
        runs = getattr(self, "_sweep_runs", None)
        if runs is not None and hasattr(runs, "the_load_succeeded"):
            runs.the_load_succeeded()
        if not self._load_trial_figures(str(folder)):
            self._console.append_stdout(
                f"{trial} saved a results table but no figures, so the grid "
                "is empty rather than showing the last run's.\n")
        self._figures_card.show()
        self._console.append_stdout(
            f"{trial} is loaded — open the Results tab to see it.\n")

    @staticmethod
    def _same_run_folder(one, other) -> bool:
        """Whether two paths name the same run folder.

        Both ends are noisy: the panel holds a CSV for a run opened off disk
        and a directory for one handed straight over by `perform_regression`
        (`run_folder` reconciles that), and a row's folder comes off a
        concatenated frame, so it can be NaN rather than a string.
        """
        try:
            if not one or not other:
                return False
            return (os.path.abspath(os.path.expanduser(str(one)))
                    == os.path.abspath(os.path.expanduser(str(other))))
        except (TypeError, ValueError):
            return False

    def _the_run_did_not_open(self, why: str) -> None:
        """Tell the Runs tab its newly-marked run could not be shown.

        THE MARK IS A CONSEQUENCE OF THE RUN BEING SHOWN (157). The tab moves
        it and asks; this is the other half of that conversation, and without
        it a failed load leaves a mark pointing at nothing on screen.
        """
        runs = getattr(self, "_sweep_runs", None)
        if runs is None:
            return
        try:
            runs.the_load_failed(why)
        except Exception:                                        # noqa: BLE001
            LOG.debug("could not hand the mark back to the Runs tab",
                      exc_info=True)

    @staticmethod
    def _figure_names_under(folder: str, recursive: bool = True):
        """Relative names of every picture in ``folder``, sorted.

        :param recursive: walk subfolders. True for a RUN folder, because most
            of a run's figures are in one (~19 QC panels, the permutation
            plots, a summary per measurement). False for the SCREEN folder,
            because its subfolders are the other runs -- recursing it would
            pull every sibling run's figures into the grid under the heading
            of this one, which is worse than missing them.
        """
        names = []
        try:
            if recursive:
                for root, _dirs, files in os.walk(folder):
                    for name in sorted(files):
                        names.append(os.path.relpath(
                            os.path.join(root, name), folder))
            else:
                for name in sorted(os.listdir(folder)):
                    if os.path.isfile(os.path.join(folder, name)):
                        names.append(name)
        except OSError:
            return []
        return sorted(name for name in names
                      if name.lower().endswith((".png", ".pdf")))

    #: The folder a run's own results directory sits inside.
    #: `perform_regression` builds `<src>/results/<kind>_<n>`, so a run folder
    #: whose parent is named this has the SCREEN folder one level further up.
    SCREEN_RESULTS_DIRNAME = "results"

    @classmethod
    def _screen_folders_above(cls, folder: str):
        """The per-screen folders a run's shared figures were written into.

        The same two :func:`spacr.ml._screen_figure_folders` names, reached
        from the other end. A sequencing helper writes
        ``<src>/results/fraction_threshold.pdf`` and
        ``<src>/results/cell_min_threshold.pdf``, but
        ``plot_plates(dst=<src>)`` writes
        ``<src>/plate_heatmap_unique_counts.pdf`` -- the screen ROOT, one
        level further up again. Two of the three are in the same place and
        the third is not, which is why this returns a list rather than a
        parent.

        THE CLIMB STOPS AT A FOLDER NAMED ``results``, and that is what keeps
        it from wandering. Reaching a second level up is only meaningful for
        the layout `_next_results_folder` actually builds; handed some other
        folder -- a bare run directory a user pointed at -- this returns its
        parent alone rather than guessing that its grandparent is a screen.
        """
        folder = os.path.normpath(folder)
        parent = os.path.dirname(folder)
        if not parent or parent == folder:
            return []
        folders = [parent]
        if os.path.basename(parent) == cls.SCREEN_RESULTS_DIRNAME:
            screen = os.path.dirname(parent)
            if screen and screen != parent:
                folders.append(screen)
        return folders

    @staticmethod
    def _pictures_from(folder: str, names):
        """``(pixmaps, titles)`` for ``names`` under ``folder``.

        A name that will not decode is skipped rather than left as a null
        pixmap: the grid drops nulls anyway, and a null in the list shifts
        every index after it away from the caption beside it.
        """
        from PySide6.QtGui import QPixmap

        pixmaps, titles = [], []
        for name in names:
            path = os.path.join(folder, name)
            pixmap = QPixmap()
            if name.lower().endswith(".pdf"):
                try:
                    from ..widgets.figure_queue import render_pdf_to_image
                    image = render_pdf_to_image(path)
                    if image is not None:
                        pixmap = QPixmap.fromImage(image)
                except Exception:
                    continue
            elif not pixmap.load(path):
                continue
            if pixmap.isNull():
                continue
            pixmaps.append(pixmap)
            titles.append(os.path.splitext(name)[0].replace(os.sep, " / "))
        return pixmaps, titles

    def _load_trial_figures(self, folder: str) -> int:
        """Load a trial's saved figures into the grid and return their count.

        Run-specific figures are loaded recursively from the run folder.
        Shared preprocessing figures are loaded non-recursively from the
        screen-level folders and placed in a separate section. Basenames
        already found in the run are skipped to prevent duplicates, and the
        grid is cleared when the selected trial has no figures.
        """
        grid = getattr(self, "_figure_grid", None)
        if grid is None:
            return 0

        run_names = self._figure_names_under(folder, recursive=True)
        pixmaps, titles = self._pictures_from(folder, run_names)
        sections = []
        if pixmaps:
            sections.append((os.path.basename(os.path.normpath(folder))
                             or str(folder), 0, len(pixmaps)))

        already = {os.path.basename(name) for name in run_names}
        extra, extra_titles = [], []
        for screen in self._screen_folders_above(folder):
            names = [name for name
                     in self._figure_names_under(screen, recursive=False)
                     if os.path.basename(name) not in already]
            found, found_titles = self._pictures_from(screen, names)
            already.update(os.path.basename(name) for name in names)
            extra.extend(found)
            extra_titles.extend(found_titles)
        if extra:
            sections.append(("the screen's own figures — not this run's",
                             len(pixmaps), len(extra)))
            pixmaps.extend(extra)
            titles.extend(extra_titles)

        grid.set_figures(pixmaps, titles, sections=sections)
        stack = getattr(self, "_figures_stack", None)
        if stack is not None:
            stack.setCurrentIndex(0)
        return len(pixmaps)

    def _figure_grid_menu(self, index: int, position) -> None:
        """Right-click on a tile: restyle or save THAT figure, in place.

        Deliberately without switching to the detail view first. The point of
        a grid is comparing figures, and navigating away to change one of them
        loses the comparison you were making.
        """
        try:
            self._figure_queue.show_figure_menu(position, int(index),
                                                navigate=False)
        except Exception:
            LOG.debug("could not open the tile menu", exc_info=True)
            return
        self._refresh_figure_grid()

    def _on_figure_size(self, pixels: int) -> None:
        """Redraw the grid at this tile width, and remember it.

        Remembered because it is a reading preference, not a property of the
        run: a user who wants big figures wants them on the next run too.
        """
        from ..preferences import set_figure_grid_size

        grid = getattr(self, "_figure_grid", None)
        if grid is not None:
            grid.set_target_cell_width(int(pixels))
        try:
            set_figure_grid_size(int(pixels))
        except Exception:                                        # noqa: BLE001
            LOG.debug("could not store the figure grid size", exc_info=True)

    #: Screens whose settings categories are TABS rather than a stack. Named
    #: rather than inferred from the count: a screen with many categories may
    #: still read better as a column, and this is a judgement about the
    #: screen and not about arithmetic.
    SETTINGS_AS_TABS = frozenset()

    @staticmethod
    def _is_the_page(container, widget) -> bool:
        """Return whether ``widget`` is visible on the container's page.

        The target may be the page itself or a descendant of a wrapper page,
        such as a grid inside a scroll area or results inside a splitter.
        """
        if container is None or widget is None:
            return False
        page = container.currentWidget()
        if page is None:
            return False
        return page is widget or page.isAncestorOf(widget)

    def showing_the_figure_grid(self) -> bool:
        """Whether the grid of every figure is the page on screen."""
        return self._is_the_page(getattr(self, "_figures_stack", None),
                                 getattr(self, "_figure_grid", None))

    def showing_the_live_graph(self) -> bool:
        """Whether the interactive volcano is the page on screen."""
        return self._is_the_page(getattr(self, "_figures_stack", None),
                                 getattr(self, "_volcano_page", None))

    def showing_the_results(self) -> bool:
        """Whether the coefficient results are the tab on screen."""
        return self._is_the_page(getattr(self, "_results_tabs", None),
                                 getattr(self, "_results_panel", None))

    def _show_figure_grid(self) -> None:
        """Back to every figure at once."""
        stack = getattr(self, "_figures_stack", None)
        if stack is not None:
            self._refresh_figure_grid()
            stack.setCurrentIndex(0)

    def _show_regression_graph(self) -> None:
        """Fill the results container with the interactive volcano plot."""
        stack = getattr(self, "_figures_stack", None)
        page = getattr(self, "_volcano_page", None)
        if stack is not None and page is not None:
            stack.setCurrentWidget(page)

    def _on_guide_selected(self, _key: str) -> None:
        """A guide was picked: raise the graph, and open the gene tile.

        Drawing a ring on a view nobody is looking at is the same as not
        drawing one, and a tile that stays collapsed is the same as no tile.
        """
        self._show_regression_graph()
        split = getattr(self, "_gene_split", None)
        if split is None:
            return
        if not getattr(self, "_gene_opened", False) and split.sizes()[1] == 0:
            self._gene_opened = True
            if getattr(split, "is_collapsed", None) and \
                    split.is_collapsed("Gene"):
                split.set_collapsed("Gene", False, by_user=False)
            total = sum(split.sizes()) or split.height() or 600
            split.setSizes([int(total * 0.6), int(total * 0.4)])

    def _open_figure_from_grid(self, index: int) -> None:
        """A pressed tile fills the container with that figure."""
        try:
            self._figure_queue.show_index(int(index))
        except Exception:
            return
        stack = getattr(self, "_figures_stack", None)
        if stack is not None:
            stack.setCurrentWidget(self._figure_detail)

    def _on_refit(self, settings) -> bool:
        """Run the same screen again through the model the user just picked.

        :returns: True if a run was started.

        REFUSED WHILE A RUN IS GOING. Two regressions writing at once is not
        a comparison, it is two runs competing for the same folder counter --
        `_next_results_folder` claims a name by looking, so both would look
        before either wrote and both would claim it.
        """
        if settings is None:
            return False
        if getattr(self, "_thread", None) is not None and self._thread.isRunning():
            self._console.append_notice(
                "■ A run is already going. Wait for it to finish, or stop it, "
                "before re-fitting.\n")
            return False
        self._console.append_notice(
            "→ Re-fitting: {model}, {correction}. This is a NEW run — the "
            "results you are looking at are not touched.\n",
            model=repr(settings.get("regression_type")),
            correction=repr(settings.get("multiple_testing_method")),
        )
        self._on_run(override=settings)
        return True

    def _on_pipeline_result(self, payload) -> None:
        """Take the coefficient table straight from the run that made it.

        `perform_regression` returns {'results': coef_df, 'res_folder': ...}.
        Using it directly is both faster and CORRECT: there is no path to
        guess, no newest-run heuristic, and no chance of reading a different
        run's table than the one that just finished.

        The on-disk search stays as the fallback for opening an old run.
        """
        if not isinstance(payload, dict):
            return
        self._update_run_in_runs_tab(
            folder=str(payload.get("res_folder") or "") or None,
            n_results=(len(payload["results"])
                       if payload.get("results") is not None else None))
        self._say_the_qc_verdict(payload)
        panel = getattr(self, "_results_panel", None)
        if panel is None:
            return
        frame = payload.get("results")
        if frame is None or not len(frame):
            return
        folder = payload.get("res_folder") or ""
        if folder:
            self._last_run_folder = str(folder)
        try:
            if panel.set_frame(frame, source=str(folder)):
                try:
                    panel.set_run_settings(payload.get("settings"))
                except Exception:
                    LOG.debug("could not hand the run's settings to the "
                              "results panel", exc_info=True)
                try:
                    panel.set_diagnostics(
                        payload.get("model"),
                        regression_type=payload.get("regression_type"))
                    panel.set_summary(
                        payload.get("model"),
                        regression_type=payload.get("regression_type"))
                except Exception:
                    LOG.debug("could not hand the run's model to the results "
                              "panel", exc_info=True)
                self._results_loaded_in_memory = True
                self._show_figure_grid()
                self._figures_card.show()
                self._console.append_stdout(
                    f"{len(frame)} coefficients loaded from the run itself"
                    + (f"; its files are in {folder}" if folder else "")
                    + ".\n")
        except Exception:
            LOG.debug("could not show the returned results", exc_info=True)

    def _load_regression_results(self) -> bool:
        """Point the Results tab at what the run just wrote.

        The settings' ``src`` is where the run was told to write, so it is
        searched first; the folder holding the count data is the fallback,
        because that is where output landed before the caller's choice was
        honoured and older runs are still there.
        """
        panel = getattr(self, "_results_panel", None)
        if panel is None:
            return False
        model = getattr(self, "_settings_model", None)
        settings = model.collect() if model is not None else {}

        candidates = []
        source = settings.get("src")
        if isinstance(source, str) and source.strip():
            candidates.append(source.strip())
        for key in ("count_data", "score_data"):
            value = settings.get(key)
            if isinstance(value, (list, tuple)) and value:
                candidates.append(os.path.dirname(str(value[0])))
            elif isinstance(value, str) and value.strip():
                candidates.append(os.path.dirname(value.strip()))

        from ..widgets.regression_results import find_results_tables

        ranked = []
        for candidate in candidates:
            tables = find_results_tables(candidate)
            if tables:
                try:
                    ranked.append((os.path.getmtime(tables[0]), candidate))
                except OSError:
                    continue
        ranked.sort(reverse=True)

        for _stamp, candidate in ranked:
            if panel.load(candidate):
                self._show_figure_grid()
                self._figures_card.show()
                return True

        if ranked:
            pass
        elif candidates:
            panel.load(candidates[0])
        else:
            panel.say(
                "The run finished, but its settings name no output folder -- "
                "src, count_data and score_data are all empty -- so there is "
                "nowhere to look for a results table. Use \u201cLoad "
                "results\u2026\u201d to point at one.")
        self._figures_card.show()
        return False

    def _clear_thread_refs(self):
        """Release worker/thread references once the QThread has stopped.

        Wired to QThread.finished (not worker.finished), so by the time this
        runs the thread's event loop has exited and dropping the last Python
        reference cannot abort the process.
        """
        self._thread = None
        self._worker = None
        if getattr(self, "_form_rebuild_deferred", False):
            QTimer.singleShot(0, self._rebuild_the_form)

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
        from ..shutdown import (CANCEL, FORCE, RESTART, GracefulQuitWatcher,
                                ask_how_to_quit)

        name = APP_TITLES.get(self.app_key, self.app_key)
        choice = ask_how_to_quit(
            self, what=name, verb="Stop",
            detail="This run is still working. Stopping cooperatively lets "
                   "it finish the field, trial or job it is on and stop at "
                   "the next point it can do so safely.",
            offer_restart=True,
            restart_detail=self._restart_warning())
        if choice == CANCEL:
            return

        if choice == RESTART:
            self.force_restart()
            return

        if choice == FORCE:
            self._force_stop()
            return

        self._console.append_notice(
            "\nRequesting stop. The current field/trial/job will finish, then "
            "the resumable run will stop at its next safe boundary.\n")
        set_button_busy(self._btn_stop, True)
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
            self._console.append_notice(
                "\nStopped waiting. The step would not interrupt -- it is "
                "still finishing in the background and may keep writing for "
                "a while. The window is yours again.\n")

    def _on_import_settings(self):
        """Load a settings CSV into the form and report what was applied.

        The dialog's caption and filter are translated here rather than by the
        application-wide dialog pass: it is built and executed in one
        expression, so that pass never sees it before it is on screen.
        """
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Import settings CSV"),
            filter=f"{tr('Settings')} (*.csv);;{tr('All files')} (*)",
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
            QMessageBox.warning(self, tr("Import failed"), str(e))

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
        """Say, in the console, what an imported CSV did to the fold switches.

        Tracking is a set of settings CATEGORIES on this form rather than a
        module of its own, and its pipeline gate has no control for a bulk
        apply to land in — the switch on the masthead is the control.
        :meth:`apply_settings_dict` moves it, and this reports what moved,
        so a user who loaded a Timelapse settings file can see that tracking
        became part of the run rather than being left to wonder.

        A gate the file asks for that no switch answered is reported as
        ignored, because that is what it is: nothing on this screen collects
        it and it will not reach the run. That covers the automated motility
        assay, which reads finished masks and writes a measurements table and
        so is a module Measure opens, not a category here.

        Console text, never a modal — a QMessageBox here would hang headless
        runs.
        """
        if self.app_key != "mask":
            return
        switched = set(getattr(self, "_folds_last_switched_on", ()))
        notes = []
        if self._truthy(loaded.get("timelapse", False)):
            if "timelapse" in switched:
                notes.append(
                    "timelapse=True switched Timelapse on — its tracking "
                    "categories are on the form and the run links objects "
                    "across frames.")
            else:
                notes.append(
                    "timelapse=True was ignored — this screen is carrying no "
                    "Timelapse switch, so the tracking categories are not on "
                    "the form and the flag will not reach the run.")
        if self._truthy(loaded.get("motility_analysis", False)):
            notes.append(
                "motility_analysis=True was ignored — the assay runs on masks "
                "that already exist, so it is a module of its own that Measure "
                "opens (Measure masthead > Motility Assay).")
        for note in notes:
            self._console.append_notice(
                "[settings] {note}\n", note=note)

    def running_modules(self) -> list:
        """Return active modules across the application.

        Each result contains the module label, its application key, and the
        elapsed run time in seconds when that information is available. The
        list is used to identify work that a forced restart will interrupt.
        """
        import time

        out = []
        for screen in self._sibling_screens():
            thread = getattr(screen, "_thread", None)
            if thread is None or not thread.isRunning():
                continue
            started = getattr(screen, "_run_started_at", None)
            out.append({
                "module": APP_TITLES.get(screen.app_key, screen.app_key),
                "app_key": screen.app_key,
                "seconds": (None if started is None
                            else max(0.0, time.time() - float(started))),
            })
        return out

    def _sibling_screens(self) -> list:
        """Every AppScreen in this application, including this one."""
        from PySide6.QtWidgets import QApplication

        screens = []
        for widget in QApplication.allWidgets():
            if isinstance(widget, AppScreen):
                screens.append(widget)
        return screens or [self]

    def _restart_warning(self) -> str:
        """What Force restart will cost, in the dialog's own words."""
        from ...restart_state import warning_text

        folders = []
        for screen in self._sibling_screens():
            folder = getattr(screen, "_last_run_folder", "") or ""
            if folder:
                folders.append(str(folder))
        try:
            return warning_text(
                [entry for entry in self.running_modules()
                 if entry.get("app_key") != self.app_key],
                folders)
        except Exception:                                       # noqa: BLE001
            LOG.debug("could not describe the restart", exc_info=True)
            return ""

    def force_restart(self, *, launcher=None, exiter=None) -> bool:
        """Save this module and its settings, then restart spaCR.

        Parameters
        ----------
        launcher, exiter
            Optional process hooks forwarded to
            :func:`spacr.qt.shutdown.restart_spacr`.

        Returns
        -------
        bool
            ``True`` when a replacement process was started. If saving fails,
            returns ``False`` without stopping the current process.
        """
        from ..shutdown import restart_spacr

        try:
            settings = dict(self._settings_model.collect() or {})
        except Exception:                                       # noqa: BLE001
            LOG.debug("could not collect the settings to restart with",
                      exc_info=True)
            settings = {}
        folders = [str(getattr(screen, "_last_run_folder", "") or "")
                   for screen in self._sibling_screens()]
        started = restart_spacr(
            self.app_key, settings,
            running=self.running_modules(),
            run_folders=[f for f in folders if f],
            launcher=launcher, exiter=exiter)
        if not started:
            self._say(
                "spaCR did NOT restart: the module and its settings could not "
                "be saved, and restarting without them would lose more than "
                "the stuck run. Nothing was stopped. The log says why.")
        return started

    def _say_the_qc_verdict(self, payload) -> str:
        """Put the run's worst QC verdict on the console. Returns what it said.

        THE WORST, not a summary and not a count. Nineteen panels passing and
        one saying the design is rank deficient is a run whose coefficients
        are one of infinitely many solutions, and "95% passed" is a sentence
        that hides exactly the panel the suite was run for --
        `regression_qc.worst_verdict` says so where it is computed and this
        is the other end of that.

        SILENT WHEN THERE IS NOTHING TO SAY. A run with QC turned off, or one
        whose suite could not build, has no verdict -- and printing "unknown"
        after every such run would train a reader to skip the line that
        matters.
        """
        if not isinstance(payload, dict):
            return ""
        verdict = payload.get("qc_verdict")
        level = str(payload.get("qc_verdict_level") or "").lower()
        if verdict is None:
            return ""
        detail = str(getattr(verdict, "detail", "") or verdict)
        name = str(getattr(verdict, "name", "") or "")
        where = f" ({name})" if name else ""
        if level in ("fail", "failed", "bad"):
            said = (f"\nREGRESSION QC: {level.upper()}{where} — {detail}\n"
                    f"The full report is in the run folder as "
                    f"regression_qc_report.txt.\n")
        elif level in ("warn", "warning", "caution"):
            said = (f"\nRegression QC: {level}{where} — {detail}\n")
        else:
            said = f"\nRegression QC: {level or 'ok'}{where} — {detail}\n"
        self._say(said)
        return said

    def restore_run_workspace(self, record) -> dict:
        """Restore the workspace recorded for a saved run.

        The console reports sections that could not be restored and files
        that have moved or changed.

        Parameters
        ----------
        record
            Run record or run-folder path accepted by
            :func:`spacr.workspace.load`.

        Returns
        -------
        dict
            Restore report with ``restored``, ``skipped``, and ``files``
            entries.
        """
        from ...workspace import load, providers, report_text, restore

        folder = str((record or {}).get("folder") or "") if isinstance(
            record, dict) else str(record or "")
        empty = {"restored": [], "skipped": [], "files": []}
        if not folder:
            return empty
        document = load(folder)
        if document is None:
            self._say(f"{folder} carries no saved workspace.")
            return empty
        report = restore(providers(), document, run_dir=folder)
        self._say(f"Restored the workspace of {os.path.basename(folder)}:\n"
                  f"{report_text(report)}")
        return report

    def _say(self, message: str) -> None:
        """Put a sentence on the console, or in the log if there is none."""
        console = getattr(self, "_console", None)
        writer = getattr(console, "append_stdout", None)
        if callable(writer):
            try:
                writer(str(message))
                return
            except Exception:                               # noqa: BLE001
                LOG.debug("could not write to the console", exc_info=True)
        LOG.info("%s", message)


    #: The workspace sections this screen owns, ``{name: attribute}``. Named
    #: rather than discovered so a saved run's section names are stable
    #: across releases -- a document written by one version is read back by
    #: another, and a section keyed by a widget's class name would go missing
    #: the first time a class was renamed.
    _WORKSPACE_PANELS = {
        "regression": "_results_panel",
        "montage": "_cell_montage",
        "figures": "_figure_grid",
    }

    def register_workspace(self) -> None:
        """Enrol this screen and its panels with the workspace registry.

        Registered as CALLABLES returning the attribute, never as the widget:
        `_results_panel` is rebuilt when the module changes, and a captured
        reference would hand a run journal a deleted C++ peer to ask.
        """
        from ...workspace import register

        key = str(self.app_key)
        register(f"{key}:settings", lambda: self)
        for name, attribute in self._WORKSPACE_PANELS.items():
            register(f"{key}:{name}",
                     lambda attribute=attribute: getattr(self, attribute, None))

    def unregister_workspace(self) -> None:
        """Withdraw this screen's sections. Called when it is torn down."""
        from ...workspace import unregister

        key = str(self.app_key)
        unregister(f"{key}:settings")
        for name in self._WORKSPACE_PANELS:
            unregister(f"{key}:{name}")

    def workspace_state(self) -> dict:
        """The settings AS EDITED, which is a superset of the run's dict.

        `open_run` writes the settings the PIPELINE was given. Keys the run
        did not consume -- a path typed into a module the user then switched
        away from, a threshold set for the next run -- are still the user's
        work and are not in that file. This is the screen's whole state.
        """
        try:
            values = dict(self._settings_model.collect() or {})
        except Exception:                                       # noqa: BLE001
            LOG.debug("could not collect the settings state", exc_info=True)
            return {"app_key": str(self.app_key), "settings": {}}
        return {"app_key": str(self.app_key), "settings": values}

    def apply_workspace_state(self, state) -> bool:
        """Put a screen's settings back. Returns whether any key applied.

        Only into the module it came from. Settings keys are shared across
        modules by name and mean different things -- `level` is the
        regression's fit level and the proportion plots' unit -- so replaying
        a measure screen's state into a regression screen would set keys that
        happen to collide and leave the rest.

        :param state: mapping with ``app_key`` and ``settings`` keys, as
            saved with the workspace; anything that is not a dict, names
            another module, or has no settings applies nothing.
        """
        if not isinstance(state, dict):
            return False
        if str(state.get("app_key") or self.app_key) != str(self.app_key):
            return False
        settings = state.get("settings")
        if not isinstance(settings, dict) or not settings:
            return False
        return bool(self.apply_settings_dict(settings))

    def apply_settings_dict(self, settings: dict) -> int:
        """Push key/value pairs from `settings` into whichever settings
        widgets this app exposes.

        A bulk load is one transaction. If it changes the form's shape, the
        complete merged mapping builds one replacement screen before values
        are applied; no signal may replace the screen halfway through the
        loop. Silently skips keys the current app does not have — the same
        dict can safely be applied across several apps. Returns the count of
        keys actually applied.

        :param settings: setting key to value mapping; legacy key names are
            translated first, and it is merged over the current values.
        """
        settings = _translate_legacy_setting_keys(settings)
        settings = self._migrate_control_wells(settings)
        model = getattr(self, "_settings_model", None)
        settings = self._infer_legacy_organelle_count(settings, model)
        try:
            current = dict((model.collect() if model is not None else {}) or {})
        except Exception:                                   # noqa: BLE001
            current = {}
        target = dict(current)
        target.update(settings)

        if (self._bulk_apply_changes_form_shape(settings, current)
                and not getattr(self, "_built_for_this_bulk_apply", False)):
            window = self.window()
            rebuild = getattr(window, "rebuild_app_screen", None)
            if callable(rebuild):
                if self._worker_thread_is_running():
                    self._deferred_form_values = target
                    deferred_bulk = dict(getattr(
                        self, "_deferred_bulk_settings", None) or {})
                    deferred_bulk.update(settings)
                    self._deferred_bulk_settings = deferred_bulk
                    self._form_rebuild_deferred = True
                else:
                    rebuild(self.app_key, target)
                    fresh = getattr(window, "_screens", {}).get(self.app_key)
                    if fresh is not None and fresh is not self:
                        fresh._built_for_this_bulk_apply = True
                        try:
                            return fresh.apply_settings_dict(settings)
                        finally:
                            fresh._built_for_this_bulk_apply = False

        applied = 0
        if model is not None:
            model._applying_settings = True
        try:
            applied = self._apply_each_setting(settings)
        finally:
            if model is not None:
                model._applying_settings = False
        self._refresh_after_bulk_apply(settings)
        return applied

    def _refresh_after_bulk_apply(self, settings: dict) -> None:
        """Apply cross-setting rules once, after a complete bulk mapping."""
        model = getattr(self, "_settings_model", None)
        if model is not None:
            try:
                model.apply_organelle_presets_from_mapping(settings)
                model._refresh_setting_dependencies()
            except Exception:                               # noqa: BLE001
                LOG.debug("could not refresh dependencies after a bulk "
                          "apply", exc_info=True)
        self._sync_folded_switches(settings)
        try:
            self._sync_dimension_switches(settings)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not sync the dimension switches after a bulk "
                      "apply", exc_info=True)

    @staticmethod
    def _infer_legacy_organelle_count(settings: dict, model) -> dict:
        """Add the count implied by slots in a pre-count settings file."""
        from ...organelle_types import (NUMBER_OF_ORGANELLES, organelle_count,
                                        organelle_role_of)

        defaults = getattr(model, "_defaults", {}) if model is not None else {}
        if (NUMBER_OF_ORGANELLES not in defaults
                or NUMBER_OF_ORGANELLES in settings
                or not any(organelle_role_of(key) is not None
                           for key in settings)):
            return settings
        migrated = dict(settings)
        migrated[NUMBER_OF_ORGANELLES] = organelle_count(settings)
        return migrated

    def _bulk_apply_changes_form_shape(
            self, settings: dict, current: dict) -> bool:
        """Whether supplied values require a differently shaped form.

        Only a switch this form carries can shape it, which is the rule
        :meth:`_form_shaping_keys` already follows for a committed edit. A
        switch the form neither shows nor holds is never in
        ``model.collect()``, so it could never compare equal after a
        rebuild either: an older recruitment file that still names
        ``nucleus_mask_dim``, a Mask file imported on Measure, or a Measure
        file imported on Mask would each ask for a rebuild on every screen
        the rebuild produced.

        :param settings: the mapping about to be applied.
        :param current: what the form collects now.
        :returns: True when the form must be rebuilt before the values go in.
        """
        from ...organelle_types import (NUMBER_OF_ORGANELLES, organelle_count,
                                        organelle_number)
        from ..settings_diff import _values_equal

        model = getattr(self, "_settings_model", None)
        carried = set(current) | set(getattr(model, "_widgets", {}) or {})
        supports_organelles = NUMBER_OF_ORGANELLES in current
        target = dict(current)
        target.update(settings)
        target_count = organelle_count(target) if supports_organelles else 0
        for key, value in settings.items():
            key = str(key)
            if key == NUMBER_OF_ORGANELLES and supports_organelles:
                if not _values_equal(current.get(key), value):
                    return True
                continue
            role = object_of_setting(key)
            if role is None or role == "cell":
                continue
            if key not in object_switch_keys(role):
                continue
            if key not in carried:
                continue
            if role not in ("nucleus", "pathogen"):
                if not supports_organelles:
                    continue
                try:
                    if organelle_number(role) > target_count:
                        continue
                except ValueError:
                    continue
            if not _values_equal(current.get(key), value):
                return True
        return False

    def _sync_folded_switches(self, settings: dict) -> tuple:
        """Move this screen's fold switches to match settings just applied.

        A module folded in as settings CATEGORIES contributes its pipeline
        GATE — ``timelapse`` on Mask Generation — and a gate has no widget
        on the form, because the masthead switch is its one control. The
        bulk apply above therefore fills in every tracking knob a Timelapse
        settings file names and leaves tracking switched off, which is a run
        the file did not ask for.

        Safe on every screen: one with no category folds returns an empty
        tuple, and a failure here costs the switch position rather than the
        import.

        :param settings: the dict that was just applied.
        :returns: the folded keys the settings switched on.
        """
        try:
            from .mask import sync_folds
            switched = tuple(sync_folds(self, settings))
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not sync the fold switches after a bulk apply",
                      exc_info=True)
            switched = ()
        self._folds_last_switched_on = switched
        return switched

    def _migrate_control_wells(self, settings: dict) -> dict:
        """Turn a retired control-well trio into the Classes rows it means.

        ``location_column``, ``positive_control`` and ``negative_control``
        were how Classify (ML) said "these two wells are my classes". The
        merged Classify module says it once, in the Classes editor, so it
        renders no widget for any of the three -- and a settings file written
        by the old module therefore lost all three on load, leaving the
        Classes editor empty and the run using the module's OWN default wells
        instead of the ones the file named. Silently, and reporting success.

        The run path already performs exactly this translation
        (:func:`spacr.classify_classes.normalize_settings`); doing it here is
        what lets the user SEE the wells the file asked for before starting.

        :param settings: settings about to be applied, not modified.
        :returns: the same dict, or a copy carrying an equivalent ``classes``.
        """
        trio = ("location_column", "positive_control_id", "negative_control_id")
        if not any(key in settings for key in trio):
            return settings
        model = getattr(self, "_settings_model", None)
        widgets = getattr(model, "_widgets", {}) if model is not None else {}
        if any(key in widgets for key in trio) or "classes" not in widgets:
            return settings
        if settings.get("classes"):
            return settings
        try:
            from ...classify_classes import normalize_settings
            classes = normalize_settings(dict(settings)).get("classes")
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not migrate the control wells", exc_info=True)
            return settings
        if not classes:
            return settings
        settings = dict(settings)
        settings["classes"] = classes
        return settings

    def _apply_each_setting(self, settings: dict) -> int:
        """Push each key into its widget. Returns how many landed."""
        applied = 0
        for key, val in settings.items():
            w = self._settings_model._widgets.get(key)
            if w is None:
                if self._settings_model.set_hidden_value(key, val):
                    from .settings_model import _APP_HIDDEN_KEYS
                    if key in _APP_HIDDEN_KEYS.get(self.app_key, set()):
                        applied += 1
                continue
            try:
                self._apply_value(w, val)
                applied += 1
            except Exception:
                pass
        self._settings_model._refresh_contextual_widgets()
        return applied

    def _apply_value(self, widget, val):
        """Write one loaded value into whichever kind of control holds it.

        A double spin box that also offers ``auto`` is set through the shared
        helper: ``float("auto")`` raises, and a swallowed failure would leave
        the control showing 1 -- the one value that cannot mean an unpenalised
        model.

        :param widget: the control to write into.
        :param val: the value as read from the settings file; coerced to the
            control's own type, and left alone when it cannot be.
        """
        from PySide6.QtWidgets import QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit
        if isinstance(widget, QCheckBox):
            widget.setChecked(str(val).lower() in ("true", "1", "yes"))
        elif isinstance(widget, QSpinBox):
            try:
                widget.setValue(int(float(val)))
            except (ValueError, TypeError):
                pass
        elif isinstance(widget, QDoubleSpinBox):
            from .settings_model import AUTO_TEXT, _set_auto_or_number

            if str(widget.specialValueText() or "") == AUTO_TEXT:
                _set_auto_or_number(widget, val)
            else:
                try:
                    widget.setValue(float(val))
                except (ValueError, TypeError):
                    pass
        elif isinstance(widget, QComboBox):
            index = widget.findData(val)
            if index < 0 and val is not None:
                index = widget.findData(str(val))
            if index < 0:
                index = widget.findText(str(val))
            if index >= 0:
                widget.setCurrentIndex(index)
        elif hasattr(widget, "set_value"):
            widget.set_value(val)
        elif isinstance(widget, QLineEdit):
            widget.setText("" if val is None else str(val))

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
        per_core = bool(self._btn_cpu_toggle.isChecked()
                        and self._per_core_bars)
        generation = self._usage_generation
        self._usage_jobs.submit(
            lambda: _sample_usage(per_core),
            lambda sample, generation=generation: self._apply_usage(
                sample, generation),
        )

    def _apply_usage(self, sample: dict,
                     request_generation: Optional[int] = None) -> None:
        """Paint one worker-taken usage sample. GUI thread only."""
        if (request_generation is not None
                and request_generation != self._usage_generation):
            return
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


#: The history strip's thumbnail, in LOGICAL pixels.
STRIP_THUMB_PX = (140, 90)

#: Matplotlib dots per inch for a strip thumbnail on an ordinary display.
#: Multiplied by the device pixel ratio, because a render is only as sharp
#: as the raster it came from -- scaling a 32-dpi picture up to a dense
#: panel enlarges the blur rather than removing it.
STRIP_THUMB_DPI = 32


def QtGui_QListWidgetItem_helper(fig, idx: int, target=None):
    """Build a :class:`QListWidgetItem` with a thumbnail render of ``fig``.

    Used in the figures panel's history strip. ``target`` is the widget the
    strip is on; the render is sized for that screen's pixel density, and
    falls back to the primary screen when no widget is given.

    :param fig: the Matplotlib figure to render as a PNG thumbnail; a render
        failure leaves the item without an icon.
    :param idx: zero-based position in the history; the item's text is
        ``#<idx + 1>``.
    """
    from io import BytesIO
    from PySide6.QtWidgets import QListWidgetItem
    item = QListWidgetItem()
    item.setText(f"#{idx + 1}")
    item.setTextAlignment(Qt.AlignCenter)
    try:
        ratio = device_ratio(target)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=STRIP_THUMB_DPI * ratio,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        pix = QPixmap()
        pix.loadFromData(buf.getvalue(), "PNG")
        if not pix.isNull():
            item.setIcon(QIcon(scaled_for(pix, target, STRIP_THUMB_PX)))
    except Exception:
        pass
    return item


def _first_count_file(settings: dict) -> str:
    """Return the first sgRNA-count CSV selected by ``settings``.

    Inspect paired score/count rows before the legacy flat ``count_data``
    list. Return an empty string when neither source contains a count file.
    """
    for row in (settings.get("paired_data") or []):
        count = (row.get("count") if isinstance(row, dict)
                 else (list(row)[1] if len(list(row)) > 1 else ""))
        if str(count or "").strip():
            return str(count).strip()
    counts = settings.get("count_data")
    if isinstance(counts, (list, tuple)):
        counts = counts[0] if counts else ""
    return str(counts or "").strip()


def _sweepable(app_key: str) -> bool:
    """Whether ``app_key`` has a parameter sweep. Import-guarded like its twin."""
    try:
        from .parameter_sweep import sweepable
        return sweepable(app_key)
    except Exception:  # pragma: no cover - sweep module unavailable
        return False


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


def _build_live_preview_card(host, *, panel_later: bool = False):
    """Build the ``Live preview`` card + panel pair without adding it
    to any layout.

    The Mask app screen embeds this into a QSplitter alongside the
    console so the two panels can be resized against each other. The panel
    starts hidden and is shown when the user clicks the Live toggle.

    The panel is told WHOSE run it previews -- the host's ``app_key`` --
    because Mask, Cellpose Masks and Plaque Assay each name the model with a
    different setting, and :mod:`spacr.qt.preview_registry` mounts this same
    builder for all three. Without it the plaque preview seeded
    ``model_name`` while the plaque run segments with ``plaque_model``.
    """
    from ..widgets.live_preview import LivePreviewPanel
    if panel_later:
        from ..widgets.card import _CardBuiltWhenShown
        card = _CardBuiltWhenShown(title="Live preview")
        card.setMinimumHeight(300)
        return None, card
    card = Card(title="Live preview")
    card.setMinimumHeight(300)
    panel = LivePreviewPanel(
        card, module=str(getattr(host, "app_key", "") or ""))
    card.body_layout.addWidget(panel)
    return panel, card


def _fill_live_preview_card(host, card):
    """Build the live preview panel into a card from
    :func:`_build_live_preview_card`, and return the panel.

    Separate so a screen can build the card at open and the panel -- about
    280 widgets -- the first time the card is shown. The
    card builder still imports the panel's module, so the widget blocks it
    registers reach the page's sheet at open as they always did.
    """
    from ..widgets.live_preview import LivePreviewPanel
    panel = LivePreviewPanel(
        card, module=str(getattr(host, "app_key", "") or ""))
    card.body_layout.addWidget(panel)
    return panel


def _build_measure_preview_card(host, *, panel_later: bool = False):
    """Build the Measure ``Crop preview`` card + panel pair (not added to a
    layout). Mirrors the Mask live preview but shows object crops from a merged
    array, tuned with the crop settings the Measure run will use."""
    from ..widgets.measure_preview import MeasurePreviewPanel
    if panel_later:
        from ..widgets.card import _CardBuiltWhenShown
        card = _CardBuiltWhenShown(title="Crop preview")
        card.setMinimumHeight(300)
        return None, card
    card = Card(title="Crop preview")
    card.setMinimumHeight(300)
    panel = MeasurePreviewPanel(card)
    card.body_layout.addWidget(panel)
    return panel, card


def _fill_measure_preview_card(card):
    """Build the crop preview panel into a card from
    :func:`_build_measure_preview_card`, and return the panel."""
    from ..widgets.measure_preview import MeasurePreviewPanel
    panel = MeasurePreviewPanel(card)
    card.body_layout.addWidget(panel)
    return panel
