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
    "ORGANELLE":        "Settings for a fourth-channel organelle mask "
                        "when your dataset includes one.",
    "ORGANELLE PREPROCESSING":
                        "Denoising / background correction applied to "
                        "the organelle channel before segmentation.",
    "ORGANELLE SPOT DETECTION":
                        "Blob-detector settings for organelle spots "
                        "(radius range, threshold, exclusion border).",
    "ORGANELLE NETWORK DETECTION":
                        "Ridge / tubular filter settings for network-"
                        "shaped organelles (e.g. mitochondria).",
    "ORGANELLE RING DETECTION":
                        "Ring / annulus detector for hollow organelle "
                        "structures.",
    "ORGANELLE IRREGULAR DETECTION":
                        "Watershed / adaptive-threshold detection for "
                        "irregular organelle shapes.",
    "ORGANELLE CELLPOSE":
                        "Cellpose model + parameters for organelle "
                        "segmentation when neither spots nor networks "
                        "capture the structure.",
    "ORGANELLE UNET":   "U-Net segmentation for organelles when "
                        "Cellpose is not accurate enough.",
    "ORGANELLE ADAPTIVE THRESHOLD":
                        "Local-threshold parameters used as a fallback "
                        "when trained models aren't available.",
    "PLOT":             "What spaCR plots inline during a run — "
                        "channel arrays, mask overlays, per-object "
                        "diagnostic figures.",
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
    "CROP":             "Per-object crop dimensions + which channels "
                        "get baked into each saved PNG.",
    "MEASURE":          "Which per-object measurements to compute — "
                        "intensity percentiles, morphology, colocalisation.",
    "CLASSIFY":         "Model type, training epochs, class balance, "
                        "augmentation, and evaluation split.",
    "REGRESSION":       "Regression model + covariates for mapping "
                        "screen scores to gRNA effect sizes.",
    "SEQUENCING":       "FASTQ inputs, barcode reference, mapping "
                        "chunk size, and QC thresholds.",
}


APP_TITLES = {
    "mask":            "Mask Generation",
    "measure":         "Measure",
    "classify":        "Classify",
    "umap":            "UMAP Embedding",
    "train_cellpose":  "Train Cellpose",
    "cellpose_masks":  "Cellpose Masks",
    "cellpose_all":    "Cellpose (All)",
    "map_barcodes":    "Map Barcodes",
    "ml_analyze":      "ML Analyze",
    "regression":      "Regression",
    "recruitment":     "Recruitment",
    "activation":      "Activation Maps",
    "analyze_plaques": "Plaque Analysis",
    "annotate":        "Annotate",
    "make_masks":      "Make Masks",
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
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(2)
        title = QLabel(APP_TITLES.get(app_key, app_key.title()))
        title.setObjectName("DisplayHeading")
        header_layout.addWidget(title)
        subtitle = QLabel("Configure settings, then press Run.")
        subtitle.setObjectName("Muted")
        header_layout.addWidget(subtitle)
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
                for key, w in getattr(self._settings_model, "_widgets", {}).items():
                    if w is widget:
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
            layout.addWidget(section)

        layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

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
            # Feed the first tile in ``src`` into the live-preview panel
            # so a Mask-app user sees something to segment as soon as
            # they pick a folder. Deferred to a timer to debounce rapid
            # typing.
            if getattr(self, "_live_preview", None) is not None:
                self._live_src_timer = QTimer(self)
                self._live_src_timer.setSingleShot(True)
                self._live_src_timer.setInterval(400)
                self._live_src_timer.timeout.connect(
                    lambda w=src_widget: self._autoload_live_preview(w.text()))
                src_widget.textChanged.connect(
                    lambda _t: self._live_src_timer.start())
        card.setObjectName("EmptyStateBanner")
        return card

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
        else:
            self._live_preview = None
            self._live_preview_card = None
            self._runtime_splitter = None
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

        # Explain error — enabled once a pipeline error is captured.
        from .. import iconset as _iconset
        self._btn_explain = QPushButton("Explain error")
        self._btn_explain.setObjectName("GhostButton")
        self._btn_explain.setIcon(_iconset.icon("info"))
        self._btn_explain.setCursor(Qt.PointingHandCursor)
        self._btn_explain.setToolTip(
            "Send the last traceback to the AI Console for a "
            "step-by-step fix."
        )
        self._btn_explain.setEnabled(False)
        self._btn_explain.clicked.connect(self._on_explain_error)
        row.addWidget(self._btn_explain)

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
        self._btn_explain.setEnabled(True)

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

    def _on_figure_ready(self, fig) -> None:
        """Hand a matplotlib figure to the FigureQueue, which renders it,
        thumbnails it, and manages the RAM/temp-spill window. The queue
        auto-selects the newest figure so the user sees each fresh result
        as it arrives."""
        self._figure_queue.add_figure(fig)
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
            from spacr.utils import load_settings
            loaded = load_settings(path, setting_key="Key", setting_value="Value")
            if not isinstance(loaded, dict):
                loaded = load_settings(path)
            if isinstance(loaded, dict):
                applied = self.apply_settings_dict(loaded)
                self._console.append_stdout(
                    f"Loaded {applied} settings from {path}\n"
                )
        except Exception as e:
            QMessageBox.warning(self, "Import failed", str(e))

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
