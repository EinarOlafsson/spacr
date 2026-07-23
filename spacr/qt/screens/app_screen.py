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

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QFontDatabase
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

        if not sections:
            layout.addWidget(QLabel("No settings defined for this app."))
        # Map widget → plain-text hint so the bottom hint strip AND our
        # sticky HoverTooltip can look up the description for the object
        # under the cursor. Initialized in __init__.
        for title, rows in sections:
            section = Section(title)
            for label, widget in rows:
                lbl_widget = QLabel(label)
                for key, w in getattr(self._settings_model, "_widgets", {}).items():
                    if w is widget:
                        html = widget.toolTip()
                        hint = self._settings_model.plain_tooltip_for(key)
                        # Clear Qt's native tooltip — the sticky popup
                        # takes its place so users can move into it and
                        # click the API docs link.
                        widget.setToolTip("")
                        self._hint_map[widget] = hint
                        self._hint_map[lbl_widget] = hint
                        self._html_tip_map[widget] = html
                        self._html_tip_map[lbl_widget] = html
                        widget.installEventFilter(self)
                        lbl_widget.installEventFilter(self)
                        break
                section.add_row(lbl_widget, widget)
            layout.addWidget(section)

        layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["md"])

        # Figures card — hidden until the pipeline pushes a figure via
        # PipelineWorker.figure_ready.
        self._figures_card = Card(title="Figures")
        from PySide6.QtWidgets import QScrollArea as _Scroll
        self._figures_scroll = _Scroll()
        self._figures_scroll.setWidgetResizable(True)
        self._figures_scroll.setFrameShape(_Scroll.NoFrame)
        self._figures_scroll.setMinimumHeight(240)
        self._figures_holder = QWidget()
        self._figures_layout = QVBoxLayout(self._figures_holder)
        self._figures_layout.setContentsMargins(0, 0, 0, 0)
        self._figures_layout.setSpacing(SPACING["sm"])
        self._figures_layout.addStretch(1)
        self._figures_scroll.setWidget(self._figures_holder)
        self._figures_card.body_layout.addWidget(self._figures_scroll, 1)
        self._figures_card.hide()
        layout.addWidget(self._figures_card, 1)

        # Merged Console (pipeline stdout + spaCR AI chat, share the
        # same scroll surface separated by topic bars).
        from ..widgets import ConsolePanel
        app_title = APP_TITLES.get(self.app_key, self.app_key.title())
        # Header label above the console so it still reads "Console"
        console_header = QLabel("Console")
        console_header.setObjectName("CardTitle")
        layout.addWidget(console_header)
        self._console = ConsolePanel(active_app_label=app_title)
        self._console.setMinimumHeight(320)
        layout.addWidget(self._console, 1)

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
        cpu_wrap.setLayout(cpu_row)
        usage_card.body_layout.addWidget(cpu_wrap)

        # Per-core panel — hidden by default; one UsageBar per logical core.
        self._per_core_wrap = QWidget()
        self._per_core_layout = QVBoxLayout(self._per_core_wrap)
        self._per_core_layout.setContentsMargins(0, 0, 0, 0)
        self._per_core_layout.setSpacing(2)
        self._per_core_bars: List[UsageBar] = []
        self._per_core_wrap.hide()
        usage_card.body_layout.addWidget(self._per_core_wrap)

        layout.addWidget(usage_card)

        # Actions row
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

        # AI toggle + provider dropdown, bottom-right of the actions row.
        # AI switch is a plain clickable text label — white when off,
        # accent blue when on. Chevron next to it exposes the provider
        # picker + install/login dialog.
        from PySide6.QtWidgets import QMenu, QToolButton
        from ..widgets import AiToggleLabel
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
        entry = resolve_pipeline_entry(self.app_key)
        if entry is None:
            QMessageBox.information(
                self, "Not runnable",
                f"The '{self.app_key}' app is interactive-only in this Qt build. "
                f"Use the classic Tk GUI (`spacr`) for now.",
            )
            return
        try:
            settings = self._settings_model.collect()
        except Exception as e:
            QMessageBox.warning(self, "Bad settings", str(e))
            return

        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress.setVisible(True)
        self._console.append_stdout(f"→ Starting {self.app_key}…\n")

        self._thread, worker = make_thread(entry, settings)
        worker.line_ready.connect(self._console.append_stdout)
        worker.error.connect(self._on_pipeline_error)
        worker.figure_ready.connect(self._on_figure_ready)
        worker.finished.connect(self._on_finished)
        self._thread.start()

    def _on_pipeline_error(self, tb: str):
        """Capture the traceback so the user can hit "Explain error"."""
        self._last_error_text = tb
        self._console.append_error(tb)
        self._btn_explain.setEnabled(True)
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

    # ------------------------------------------------------------------
    # AI toggle + provider menu — sits in the actions row (bottom right)
    # ------------------------------------------------------------------
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

    def _on_figure_ready(self, fig) -> None:
        """Embed a matplotlib figure emitted by the worker as a new
        FigureCanvas above the console. Duplicates are skipped."""
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        except Exception:
            try:
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            except Exception:
                return
        # Skip if this exact figure is already embedded
        for i in range(self._figures_layout.count()):
            item = self._figures_layout.itemAt(i)
            w = item.widget() if item is not None else None
            if isinstance(w, FigureCanvasQTAgg) and w.figure is fig:
                w.draw_idle()
                return
        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumHeight(320)
        canvas.draw_idle()
        # Insert BEFORE the stretch item at the end
        self._figures_layout.insertWidget(self._figures_layout.count() - 1, canvas)
        self._figures_card.show()

    def _on_finished(self, ok: bool):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.setVisible(False)
        self._console.append_stdout(
            "✓ Finished\n" if ok else "✗ Failed — see traceback above\n")
        self._thread = None

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
