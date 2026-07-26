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

from typing import Optional

from PySide6.QtCore import QSize, Qt, QTimer, QThread, Signal
from PySide6.QtGui import QFontDatabase, QIcon, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..bridge import make_thread, resolve_pipeline_entry
from ..theme import PALETTE, SPACING
from ..widgets import Card, Divider, Section, UsageBar
from .settings_model import SettingsWidgets


# Hover-tooltip text for each settings section. Keys match the
# uppercased section title (e.g. "PATHS", "CELL"). Sections that
# don't have an entry fall back to a generic "Settings that
# control <title>."
SECTION_HINTS = {
    "PATHS":            "Source folder + destination folder + which "
                        "sub-folders spaCR should read images from.",
    "GENERAL":          "High-level knobs: metadata source (Yokogawa "
                        "vs Cellvoyager vs custom regex), channel "
                        "layout, magnification, plotting toggles.",
    "CELL":             "Cellpose settings for the *cell* mask: "
                        "channel, model, diameter, cellprob threshold, "
                        "background floor.",
    "NUCLEUS":          "Cellpose settings for the *nucleus* mask: "
                        "channel, model, diameter, cellprob threshold, "
                        "background floor.",
    "PATHOGEN":         "Cellpose settings for the *pathogen* mask: "
                        "channel, model, diameter, cellprob threshold, "
                        "background floor.",
    "ORGANELLE":        "Everything for the organelle mask, in the order you "
                        "set it up: shape family and detection method, the "
                        "background/contrast correction applied first, the "
                        "knobs belonging to the method you chose (adaptive, "
                        "spot, ridge/hysteresis, ring, irregular, Cellpose or "
                        "U-Net), the size/intensity/border filters applied to "
                        "the objects found, and which parent compartments the "
                        "organelles are summarised into.",
    "CELLPOSE":         "Shared Cellpose knobs for the training and "
                        "mask-finetune tools: model, diameter, cellprob and "
                        "flow thresholds, resize and normalisation.",
    "SEGMENTATION QC":  "Automatic pass/fail checks on the finished masks — "
                        "object counts, size and split ratios, border and "
                        "foreground fractions, per-plate failure tolerance.",
    "MEASUREMENTS":     "Which per-object features are computed — intensity, "
                        "morphology, texture, radial distribution, "
                        "colocalisation — and which of them survive into the "
                        "analysis table.",
    "OBJECT CROPS":     "Per-object crop dimensions, which mask each crop is "
                        "centred on, and which channels get baked into each "
                        "saved PNG or array.",
    "PLATE LAYOUT & CONTROLS":
                        "The plate map: which wells hold which cell line, "
                        "pathogen strain and treatment, which wells or gRNAs "
                        "are the positive and negative controls, and the "
                        "labels they are given.",
    "TRAINING DATASET": "How the labelled training set is assembled from the "
                        "database — annotation column vs well metadata, which "
                        "crop type, and how many objects to sample.",
    "MODEL TRAINING":   "The image classifier: backbone, classes, input "
                        "channels and size, epochs, optimizer, learning-rate "
                        "schedule, loss, augmentation, and the "
                        "train/validation/test split.",
    "ML CLASSIFIER":    "The classical (non-image) screen classifier fitted "
                        "on measured features — algorithm, tree count, "
                        "regularisation, feature pruning, and permutation "
                        "importance.",
    "EMBEDDING & CLUSTERING":
                        "UMAP/t-SNE reduction of the feature table and the "
                        "clustering run on top of it — neighbourhood size, "
                        "metric, DBSCAN/KMeans parameters, noise handling.",
    "ACTIVATION MAPS":  "Grad-CAM / saliency settings — attribution method, "
                        "which layer to hook, overlay rendering, and the "
                        "input normalisation used at inference.",
    "PLOT":             "What spaCR plots inline during a run — channel "
                        "arrays, mask overlays, per-object diagnostic "
                        "figures — plus the styling of the embedding "
                        "scatter and the plate heatmaps.",
    "TIMELAPSE":        "Enable + tune temporal linking of masks "
                        "across frames when your data has a T axis.",
    "ADVANCED":         "Rarely-touched knobs — batch sizes, worker "
                        "counts, memory tuning, experimental options.",
    "BETA":             "Experimental features that may change or be "
                        "removed. Use with caution.",
    "MOTILITY (BETA)":  "Beta motility-assay analysis toggle + "
                        "per-object tracking parameters.",
    "MOTILITY ADVANCED (BETA)":
                        "Fine-grained control over the beta motility "
                        "pipeline — feature selection, filter windows.",
    "REGRESSION":       "Regression model + covariates for mapping "
                        "screen scores to gRNA effect sizes, plus the "
                        "control-based threshold used to call hits.",
    "INVASION ASSAY":   "The two-colour invasion assay: which channels hold the "
                        "outside and total stains, how the outside signal is "
                        "measured, how its threshold is chosen and checked, and "
                        "which objects count as parasites at all.",
    "SEQUENCING":       "FASTQ inputs, barcode reference, mapping "
                        "chunk size, and QC thresholds.",
}


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
    "custom_measurement": None,
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


APP_TITLES = {
    "mask":            "Mask Generation",
    "timelapse":       "Timelapse",
    "motility":        "Motility Assay",
    "measure":         "Measure",
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
    "annotate":        "Annotate",
    "make_masks":      "Make Masks",
    "db_browser":      "Database Browser",
    "agreement":       "Annotator Agreement",
    "plate_view":      "Plate Viewer",
    "model_compare":   "Model Compare",
    "align":           "Align & Stitch",
    "convert":         "Format Converter",
    "foreign":         "Import Project",
    "batch":           "Batch Runner",
    "model_zoo":       "Model Zoo",
    "report":          "Report",
    "train_compare":   "Training Runs",
}


# Short "what this module does" blurbs shown to the right of the header.
APP_INTROS = {
    "mask":            "Segment cells, nuclei, pathogens and organelles with Cellpose and build the merged image+mask arrays.",
    "timelapse":       "Segment each frame of a time series and link objects across frames into tracks, then export per-channel movies.",
    "motility":        "Rebuild per-frame tracks, score velocity and straightness per object, and split them by infection state.",
    "measure":         "Extract per-object intensity + morphology features from masks and write them to the measurements database.",
    "annotate":        "Review single-object image crops on a grid and label them; annotations save back to the database.",
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
    "align":           "Stitch an arbitrary number of tiles into one canvas. Offsets are solved globally rather than accumulated, so error does not walk down a row; tiles that failed to register are shown and recorded rather than quietly placed by stage position; and the canvas is written band by band, so its size is never a memory limit.",
    "convert":         "Turn ND2, CZI, LIF or OME-TIFF acquisitions into Yokogawa-named TIFFs spaCR can read, after showing you exactly which source file becomes which target — and write a map file so the original metadata can be joined back onto the measurements afterwards.",
    "batch":           "Stack any combination of modules, plates and settings into a queue and run it unattended — each job is validated when you add it, runs in its own process, and reports what failed, what was skipped because an upstream job failed, and what finished only partly.",
    "model_zoo":       "Every Cellpose and classifier checkpoint this machine can reach, with what it was trained on, whether its bytes check out against a published checksum, and what it does to three of your fields.",
    "train_compare":   "Overlay the loss and accuracy curves of several training runs on one axis and see, beside them, exactly which settings differed — with environment drift bucketed away from the knobs you actually turned.",
    "analyze_plaques": "Detect and quantify plaques in plaque-assay images.",
    "recruitment":     "Quantify recruitment of a marker to a compartment across conditions.",
    "invasion":        "Score every parasite attached or invaded from a two-colour outside/inside stain, with the threshold derived per field and flagged when the two populations it assumes are not actually there.",
}


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

    def __init__(self, app_key: str, parent=None):
        super().__init__(parent)
        self.app_key = app_key
        self._last_error_text: str = ""
        self._hint_map: dict = {}       # widget → plain-text hint
        self._html_tip_map: dict = {}   # widget → HTML tooltip (sticky popup)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                  SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        # ─── Header ───────────────────────────────────────────────────
        # Title + subtitle on the left, followed on the same row by a short
        # single-line "what this does" blurb and a docs link. Everything is
        # left-aligned; the trailing stretch takes up the slack.
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(SPACING["lg"])

        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(2)
        title = QLabel(APP_TITLES.get(app_key, app_key.title()))
        title.setObjectName("DisplayHeading")
        title_col.addWidget(title)
        subtitle = QLabel("Configure settings, then press Run.")
        subtitle.setObjectName("Muted")
        title_col.addWidget(subtitle)
        header_layout.addLayout(title_col)

        intro_text = APP_INTROS.get(app_key)
        if intro_text:
            from ..screens.settings_model import api_docs_url
            intro_row = QHBoxLayout()
            intro_row.setContentsMargins(0, 0, 0, 0)
            intro_row.setSpacing(SPACING["sm"])
            blurb = QLabel(intro_text)
            blurb.setObjectName("Muted")
            # One line, flush left. The label may shrink below its ideal
            # width so a long blurb never forces the window wider.
            blurb.setWordWrap(False)
            blurb.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            blurb.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
            blurb.setMinimumWidth(0)
            blurb.setToolTip(intro_text)
            intro_row.addWidget(blurb)
            docs = QLabel(
                f'<a href="{api_docs_url(app_key)}" '
                f'style="color:{PALETTE["accent"]};">Docs&nbsp;→</a>')
            docs.setOpenExternalLinks(True)
            docs.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            intro_row.addWidget(docs)
            header_layout.addLayout(intro_row)
        header_layout.addStretch(1)
        outer.addWidget(header)

        outer.addWidget(Divider())

        # ─── Body splitter ────────────────────────────────────────────
        body = QSplitter(Qt.Horizontal)
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

        # Timer to poll RAM/GPU/CPU periodically
        self._usage_timer = QTimer(self)
        self._usage_timer.setInterval(2000)
        self._usage_timer.timeout.connect(self._refresh_usage)
        self._usage_timer.start()
        self._refresh_usage()

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

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------
    def _build_settings_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
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

        if not sections:
            layout.addWidget(QLabel("No settings defined for this app."))
        # Map widget → plain-text hint so the bottom hint strip AND our
        # sticky HoverTooltip can look up the description for the object
        # under the cursor. Initialized in __init__.
        for title, rows in sections:
            section = Section(title)
            # Attach a per-section tooltip so hovering the header tells
            # users what the settings inside actually control. Falls
            # back to a generic "settings for <TITLE>" if the section
            # is one we don't have a curated blurb for.
            section.set_hint(SECTION_HINTS.get(
                title.upper().strip(),
                f"Settings that control {title.lower().strip()}.",
            ))
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
                        # Tooltips live on the LABEL only — hovering
                        # the input field itself is left alone so
                        # focus / edit interactions aren't disturbed.
                        widget.setToolTip("")
                        self._hint_map[lbl_widget] = hint
                        self._html_tip_map[lbl_widget] = html
                        lbl_widget.installEventFilter(self)
                        break
                section.add_row(lbl_widget, widget)
                self._attach_column_picker(field_key, widget)
            layout.addWidget(section)

        layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

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
        """
        from PySide6.QtWidgets import QLineEdit
        src = getattr(self._settings_model, "_widgets", {}).get("src")
        return src.text().strip() if isinstance(src, QLineEdit) else ""

    def _build_empty_state_banner(self):
        """Return a compact "Drop or pick a demo" card, or None.

        The card is inserted at the top of the settings scroll. It
        hides once the ``src`` widget contains anything so users
        who've already pointed the app at data see the normal form.
        """
        from PySide6.QtWidgets import QFrame, QLineEdit, QPushButton
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
        subtitle = (
            "Drop a folder of images anywhere on this window, or use "
            "Demos → Mask demo… for a synthetic dataset. You can also "
            "type a path into the src field below."
        )
        card = EmptyState(
            title=title, subtitle=subtitle,
            cta_label="Open Demos menu",
            on_action=lambda: self._open_demos_menu(),
        )
        # Auto-hide once the user sets src
        if isinstance(src_widget, QLineEdit):
            src_widget.textChanged.connect(self._maybe_hide_empty_state)
        card.setObjectName("EmptyStateBanner")
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
        """Load the first supported image found under ``src`` into the
        live-preview panel. Silent if ``src`` is empty, a placeholder,
        or contains no images — the panel already handles that.
        """
        panel = getattr(self, "_live_preview", None)
        if panel is None:
            return
        s = (src or "").strip()
        if not s or s in {"path", "/path/to/src", "/path"}:
            return
        from pathlib import Path
        root = Path(s)
        if not root.is_dir():
            if root.is_file() and root.suffix.lower() in {".tif", ".tiff",
                                                            ".png", ".jpg",
                                                            ".jpeg"}:
                panel.load_image(root)
            return
        # Pick the first image at any depth (breadth-limited)
        for pattern in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
            hits = sorted(root.rglob(pattern))
            if hits:
                panel.load_image(hits[0])
                return

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
        if event.type() == QEvent.Enter:
            hint = self._hint_map.get(obj)
            if hint and hasattr(self, "_hint_strip"):
                self._hint_strip.setText(hint)
            html = self._html_tip_map.get(obj)
            if html:
                HoverTooltip.instance().show_for(obj, html)
        elif event.type() == QEvent.Leave:
            if hasattr(self, "_hint_strip"):
                self._hint_strip.setText(self._default_hint())
            HoverTooltip.instance().start_hide()
        return super().eventFilter(obj, event)

    def _default_hint(self) -> str:
        return "Hover any setting to see its description and docs link."

    def _build_runtime_panel(self) -> QWidget:
        wrap = QWidget()
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
        console_col = QVBoxLayout(console_wrap)
        console_col.setContentsMargins(0, 0, 0, 0)
        console_col.setSpacing(4)
        console_header = QLabel("Console")
        console_header.setObjectName("CardTitle")
        console_col.addWidget(console_header)
        self._console = ConsolePanel(active_app_label=app_title)
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
        self._per_core_bars: List[UsageBar] = []
        self._per_core_wrap.hide()
        usage_card.body_layout.addWidget(self._per_core_wrap)

        layout.addWidget(usage_card)

        # Actions row. Flush-left (no extra inset) so Run / Stop / Import /
        # Clear / Explain line up with the console, chat and System panel,
        # which all share the runtime panel's small left inset.
        actions = QWidget()
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

        self._btn_clear = QPushButton("Clear console")
        self._btn_clear.setObjectName("GhostButton")
        self._btn_clear.setCursor(Qt.PointingHandCursor)
        self._btn_clear.clicked.connect(lambda: self._console.clear())
        row.addWidget(self._btn_clear)

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

        # LP (Live Preview) toggle — Mask app only. Same styling as the
        # AI switch: white when off, accent blue when on. Toggling
        # hides / shows the Live Preview card.
        from PySide6.QtWidgets import QMenu, QToolButton
        from ..widgets import AiToggleLabel
        if getattr(self, "_live_preview", None) is not None:
            self._lp_switch = AiToggleLabel(
                text="LP",
                tooltip=("Click to toggle Live Preview. When ON (blue), "
                          "the interactive Cellpose preview appears above "
                          "the console for tuning a sample tile."),
            )
            # Default LP OFF so the panel starts collapsed; user opts in.
            self._lp_switch.toggled.connect(self._on_lp_switch)
            row.addWidget(self._lp_switch)
            self._on_lp_switch(False)   # hide the LP card initially

        # Same slot, same behaviour, for the apps that have a hyperparameter
        # search instead of a live preview.
        if getattr(self, "_hyperparam", None) is not None:
            from .hyperparam import TOGGLE_TEXT, TOGGLE_TOOLTIP
            self._hp_switch = AiToggleLabel(text=TOGGLE_TEXT,
                                            tooltip=TOGGLE_TOOLTIP)
            self._hp_switch.toggled.connect(self._on_hyperparam_switch)
            row.addWidget(self._hp_switch)
            self._on_hyperparam_switch(False)   # start collapsed, like LP

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

        # Hint strip — hover-follows caption that shows the current
        # settings tooltip regardless of Qt HTML-tooltip rendering.
        self._hint_strip = QLabel(self._default_hint())
        self._hint_strip.setObjectName("SubtitleSmall")
        self._hint_strip.setWordWrap(True)
        self._hint_strip.setMinimumHeight(24)
        self._hint_strip.setOpenExternalLinks(True)
        layout.addWidget(self._hint_strip)

        return wrap

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
        self._console.append_stdout(
            f"→ Starting {self.app_key} ({entry_name}) with "
            f"src={settings.get('src')!r} + {len(settings)} settings…\n")
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress.setVisible(True)

        # Remember start time so _on_finished can report elapsed to
        # the run journal + the OS notification.
        import time as _time
        self._run_started_at = _time.time()

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
        # When the user has opted into automatic issue filing, actually file
        # it — previously this only revealed the button, so nothing was ever
        # sent unless the user also clicked. Open the pre-filled report now.
        if enabled:
            try:
                self._on_file_issue()
            except Exception as e:
                self._console.append_stdout(
                    f"[issue] auto-file failed: {e}\n")

    # ------------------------------------------------------------------
    # AI toggle + provider menu — sits in the actions row (bottom right)
    # ------------------------------------------------------------------
    def _on_lp_switch(self, on: bool) -> None:
        """Show/hide the Live Preview card when the LP toggle flips."""
        card = getattr(self, "_live_preview_card", None)
        if card is None:
            return
        card.setVisible(on)

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
                    self._console.append_stdout(
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
            self._ai_menu.addAction(
                "(no vendor CLI installed)"
            ).setEnabled(False)
            self._ai_menu.addSeparator()
        act_providers = self._ai_menu.addAction("Providers…")
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
        """Open a pre-filled GitHub issue for the last captured traceback."""
        if not self._last_error_text:
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
                    elif isinstance(w, QLineEdit):
                        settings_snapshot[k] = w.text()
        except Exception:
            settings_snapshot = {}
        from ..ai.issue_report import file_issue
        url = file_issue(self._last_error_text,
                          active_app=self.app_key,
                          settings=settings_snapshot)
        self._console.append_stdout(
            f"[issue] opened pre-filled report in your browser — "
            f"review + submit to complete filing.\n{url[:100]}...\n"
        )

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
        self._figure_queue.add_figure(fig, prerendered_png=png_path or None)
        self._figures_card.show()

    def closeEvent(self, event):
        """Stop any running pipeline thread before the widget is torn
        down. Destroying a QWidget while a child QThread is still
        running aborts the process (this also protects the test suite,
        where screens are created + destroyed rapidly)."""
        th = getattr(self, "_thread", None)
        if th is not None:
            try:
                th.requestInterruption()
                th.quit()
                # Bounded wait so we don't destroy the widget mid-run,
                # but only from closeEvent (main thread, not triggered
                # by the thread's own finished signal — safe here).
                th.wait(3000)
            except Exception:
                pass
            self._thread = None
            self._worker = None
        # Clean up the figure queue's temp dir if present.
        fq = getattr(self, "_figure_queue", None)
        if fq is not None:
            try:
                fq.clear()
            except Exception:
                pass
        super().closeEvent(event)

    def _on_finished(self, ok: bool):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.setVisible(False)
        self._console.append_stdout(
            "✓ Finished\n" if ok else "✗ Failed — see traceback above\n")
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
                self.app_key, "success" if ok else "failed", elapsed
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
        if self._thread is None:
            return
        # QThread.terminate is unsafe but the pipelines have no cooperative
        # cancellation; document the caveat in the console.
        self._console.append_stdout(
            "\nRequesting stop (worker cancellation isn't cooperative — "
            "the current task may finish before it exits).\n")
        try:
            self._thread.requestInterruption()
        except Exception:
            pass

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
            self._console.append_stdout(
                f"Loaded {applied} settings from {path}\n"
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
            self._console.append_stdout(f"[settings] {note}\n")

    def apply_settings_dict(self, settings: dict) -> int:
        """Push key/value pairs from `settings` into whichever settings
        widgets this app exposes. Silently skips keys the current app
        does not have — the same dict can safely be applied across
        several apps. Returns the count of keys actually applied."""
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
        # RAM
        try:
            import psutil
            self._usage_ram.set_value(psutil.virtual_memory().percent)
            self._usage_cpu.set_value(psutil.cpu_percent(interval=None))
            if self._btn_cpu_toggle.isChecked() and self._per_core_bars:
                per_core = psutil.cpu_percent(interval=None, percpu=True)
                for bar, pct in zip(self._per_core_bars, per_core):
                    bar.set_value(pct)
        except Exception:
            pass
        # GPU / VRAM
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                self._usage_gpu.set_value(gpu.load * 100)
                self._usage_vram.set_value(gpu.memoryUtil * 100)
            else:
                self._usage_gpu.set_value(0)
                self._usage_vram.set_value(0)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    console so the two panels can be resized against each other. LP
    starts hidden and is shown when the user clicks the LP toggle.
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
