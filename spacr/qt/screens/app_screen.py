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

from PySide6.QtCore import Qt, QTimer, QThread
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
    def __init__(self, app_key: str, parent=None):
        super().__init__(parent)
        self.app_key = app_key

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
        for title, rows in sections:
            section = Section(title)
            for label, widget in rows:
                # Build a QLabel so we can copy the tooltip from widget →
                # label (hover either shows the same tip)
                lbl_widget = QLabel(label)
                lbl_widget.setToolTip(widget.toolTip())
                lbl_widget.setOpenExternalLinks(True)
                section.add_row(lbl_widget, widget)
            layout.addWidget(section)

        layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

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

        # Console card
        console_card = Card(title="Console")
        self._console = QPlainTextEdit()
        self._console.setReadOnly(True)
        self._console.setObjectName("Console")
        # Use monospaced font for log readability
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self._console.setFont(mono)
        self._console.setPlaceholderText("Pipeline output will appear here…")
        console_card.body_layout.addWidget(self._console, 1)
        layout.addWidget(console_card, 1)

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
        self._btn_clear.clicked.connect(lambda: self._console.setPlainText(""))
        row.addWidget(self._btn_clear)

        row.addStretch(1)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate until we know
        self._progress.setVisible(False)
        self._progress.setFixedWidth(240)
        row.addWidget(self._progress)

        layout.addWidget(actions)

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
        self._console.appendPlainText(f"→ Starting {self.app_key}…\n")

        self._thread, worker = make_thread(entry, settings)
        worker.line_ready.connect(self._console.insertPlainText)
        worker.error.connect(lambda tb: self._console.appendPlainText(f"[error]\n{tb}"))
        worker.figure_ready.connect(self._on_figure_ready)
        worker.finished.connect(self._on_finished)
        self._thread.start()

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
        self._console.appendPlainText(
            "✓ Finished\n" if ok else "✗ Failed — see traceback above\n")
        self._thread = None

    def _on_stop(self):
        if self._thread is None:
            return
        # QThread.terminate is unsafe but the pipelines have no cooperative
        # cancellation; document the caveat in the console.
        self._console.appendPlainText(
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
                # Try the default column names
                loaded = load_settings(path)
            if isinstance(loaded, dict):
                self._console.appendPlainText(f"Loaded {len(loaded)} settings from {path}\n")
                # Push into widgets where keys match.
                # (Silent skip for keys the current app doesn't expose.)
                for key, val in loaded.items():
                    w = self._settings_model._widgets.get(key)
                    if w is None:
                        continue
                    try:
                        self._apply_value(w, val)
                    except Exception:
                        pass
        except Exception as e:
            QMessageBox.warning(self, "Import failed", str(e))

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
