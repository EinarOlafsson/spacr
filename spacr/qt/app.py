"""
QApplication bootstrap + MainWindow.

`launch(argv)` is the public entry point called by `spacr-qt` and
`python -m spacr.qt`.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


from . import iconset
from .theme import PALETTE, SPACING, apply_qpalette, stylesheet


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
            # Never fail loud — this is a background optimisation, and a
            # spacr module that can't import today isn't a bug we should
            # turn into a crash.
            pass
        # 50 ms between imports so Qt drains its event queue (repaints,
        # input) before the next potentially-blocking import.
        QTimer.singleShot(50, self._step)


APPS = [
    # (key, human name, description, section)
    ("mask",           "Mask",           "Generate cellpose masks for cells, nuclei and pathogens",   "Core"),
    ("measure",        "Measure",        "Measure single-object intensity + morphology features",       "Core"),
    ("annotate",       "Annotate",       "Annotate single-object images on a grid; save to database",  "Core"),
    ("make_masks",     "Make Masks",     "Fine-tune Cellpose models for your dataset",                  "Core"),
    ("classify",       "Classify",       "Train Torch CNNs/Transformers to classify single objects",   "Core"),
    ("umap",           "UMAP",           "Generate UMAP embeddings with image glyphs",                 "Analysis"),
    ("ml_analyze",     "ML Analyze",     "ML analysis of screen features",                             "Analysis"),
    ("regression",     "Regression",     "Regression analysis of screen scores",                       "Analysis"),
    ("recruitment",    "Recruitment",    "Analyze recruitment data",                                    "Analysis"),
    ("activation",     "Activation",     "Generate activation maps",                                    "Analysis"),
    ("analyze_plaques", "Plaque",        "Analyze plaque data",                                         "Analysis"),
    ("train_cellpose", "Train Cellpose", "Train custom Cellpose models",                                "Cellpose"),
    ("cellpose_masks", "Cellpose Masks", "Cellpose mask generation",                                    "Cellpose"),
    ("cellpose_all",   "Cellpose All",   "Run cellpose on all images",                                  "Cellpose"),
    ("map_barcodes",   "Map Barcodes",   "Map barcodes to data",                                        "Sequencing"),
    ("queue",          "Plate Queue",    "Chain multiple plates through the same pipeline",             "Batch"),
]


# Explicit key -> icon-filename overrides for cases where the app_key
# doesn't match any resource filename. Add entries here rather than
# renaming resource files.
_ICON_OVERRIDES = {
    "analyze_plaques": "plaque.png",
    "queue":           "sequencing.png",   # closest visual match for now
}


def _icon_for_app(key: str) -> Optional[QIcon]:
    """Return a QIcon for an app key. Uses the bundled spacr PNG icon if
    present; falls back to a themed qtawesome glyph via iconset."""
    here = os.path.dirname(os.path.abspath(__file__))
    resources_dir = os.path.normpath(os.path.join(here, "..", "resources", "icons"))
    candidates = []
    if key in _ICON_OVERRIDES:
        candidates.append(_ICON_OVERRIDES[key])
    candidates += [f"{key}.png", f"{key.replace('_', ' ')}.png"]
    for candidate in candidates:
        p = os.path.join(resources_dir, candidate)
        if os.path.exists(p):
            return QIcon(p)
    return iconset.icon(key)


class Sidebar(QWidget):
    """Left navigation column. Emits `nav_selected(str key)` when a tile
    is clicked. `Home` reverts to the startup page."""

    from PySide6.QtCore import Signal
    nav_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        # Sidebar width tracks the user's font scale so at 150 % the
        # labels still fit without being clipped mid-word. See
        # preferences.scaled_px for the helper.
        from .preferences import scaled_px
        self.setFixedWidth(scaled_px(220))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title = QLabel("spaCR")
        title.setObjectName("SidebarTitle")
        layout.addWidget(title)

        home = QPushButton("  Home")
        home.setObjectName("SidebarItem")
        home.setCursor(Qt.PointingHandCursor)
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
                current_section = section
            btn = QPushButton(f"  {name}")
            btn.setObjectName("SidebarItem")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(desc)
            # Accessibility — screen readers announce the app name +
            # its one-line description as the button's role.
            btn.setAccessibleName(name)
            btn.setAccessibleDescription(desc)
            icon = _icon_for_app(key)
            if icon is not None:
                btn.setIcon(icon)
                btn.setIconSize(QSize(16, 16))
            btn.clicked.connect(lambda checked=False, k=key: self.nav_selected.emit(k))
            layout.addWidget(btn)

        layout.addStretch(1)


class MainWindow(QMainWindow):
    """Top-level window: sidebar + stacked screens + status bar.

    :param initial_app: optional app key to navigate to on show; when
        omitted the window opens on the Home startup page.
    """

    def __init__(self, initial_app: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("spaCR")
        self.setMinimumSize(1200, 720)

        self._build_menu_bar()

        # Central layout: sidebar | content
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        self._sidebar = Sidebar()
        self._sidebar.nav_selected.connect(self._on_nav_selected)
        splitter.addWidget(self._sidebar)

        self._stack = QStackedWidget()
        splitter.addWidget(self._stack)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 980])
        self.setCentralWidget(splitter)

        # Register screens lazily — created on first navigation.
        self._screens: dict[str, QWidget] = {}
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
        for key, name, desc, section in APPS:
            act = QAction(name, self)
            act.setStatusTip(desc)
            act.triggered.connect(lambda checked=False, k=key: self._on_nav_selected(k))
            app_menu.addAction(act)
        app_menu.addSeparator()
        act_home = QAction("Home", self)
        act_home.setShortcut(QKeySequence("Ctrl+H"))
        act_home.triggered.connect(lambda: self._on_nav_selected("__home__"))
        app_menu.addAction(act_home)
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
        for app_key, label in (
            ("mask",      "Mask demo…"),
            ("measure",   "Measure demo…"),
            ("crop",      "Crop demo…"),
            ("classify",  "Classify demo…"),
            ("timelapse", "Timelapse demo…"),
        ):
            act = QAction(label, self)
            act.setStatusTip(
                f"Generate a synthetic {app_key} dataset and open it "
                "in the matching app.")
            act.triggered.connect(
                lambda checked=False, k=app_key: self._on_load_demo(k))
            demo_menu.addAction(act)
        demo_menu.addSeparator()
        act_e2e = QAction("End-to-end (Mask → Measure → Annotate) real dataset…", self)
        act_e2e.setStatusTip(
            "Download the toxo_mito HF demo dataset + settings pack, "
            "then chain Mask → Measure → Annotate on it.")
        act_e2e.triggered.connect(self._on_e2e_demo)
        demo_menu.addAction(act_e2e)

        help_menu = mb.addMenu("&Help")
        act_tutorial = QAction("Tutorial (web)", self)
        act_tutorial.triggered.connect(
            lambda: self._open_url("https://einarolafsson.github.io/spacr/tutorial/"))
        help_menu.addAction(act_tutorial)
        act_docs = QAction("Documentation (web)", self)
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
        "timelapse": ("mask",       "generate_timelapse_demo"),
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

    def _check_for_updates(self):
        """Query PyPI/GitHub in a background thread, prompt to upgrade.

        The GUI thread stays responsive: the network call runs on a
        Qt worker thread, and the ``UpdateInfo`` result gets marshalled
        back via a signal.
        """
        from PySide6.QtCore import QThread, Signal
        try:
            from spacr.updater import check_for_updates, run_pip_upgrade
        except Exception as e:
            QMessageBox.warning(self, "Updates",
                                f"Update check unavailable: {e}")
            return

        class _Worker(QThread):
            done = Signal(object)
            def run(self_):
                self_.done.emit(check_for_updates())

        self.statusBar().showMessage("Checking for updates…", 4000)
        worker = _Worker(self)
        def _on_done(info):
            if info.error and not info.latest_release:
                QMessageBox.warning(self, "Updates",
                    f"Couldn't reach update server:\n{info.error}")
                return
            if info.upgrade_available:
                msg = (f"A new version is available.\n\n"
                       f"Installed: {info.installed_version}\n"
                       f"Latest:    {info.latest_release}\n\n"
                       f"Run pip install --upgrade spacr now?")
                if QMessageBox.question(self, "Update available", msg
                        ) == QMessageBox.Yes:
                    rc = run_pip_upgrade()
                    if rc == 0:
                        QMessageBox.information(self, "Updates",
                            "Upgrade finished. Restart spaCR to use it.")
                    else:
                        QMessageBox.warning(self, "Updates",
                            f"pip returned exit code {rc}. "
                            "Check the terminal for details.")
            else:
                QMessageBox.information(self, "Updates",
                    f"You're on {info.installed_version}. No updates.")
        worker.done.connect(_on_done)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        self._update_worker = worker

    # -- shutdown ----------------------------------------------------------
    def closeEvent(self, event):
        """Cancel every active AI stream + wait for its QThread to exit
        BEFORE Qt starts destroying widgets. Prevents the
        'QThread: Destroyed while thread is still running / Aborted'
        crash on quit."""
        from .widgets.console_panel import ConsolePanel
        for panel in self.findChildren(ConsolePanel):
            try:
                panel.shutdown()
            except Exception:
                pass
        super().closeEvent(event)

    # -- navigation -------------------------------------------------------
    def _install_startup_page(self):
        """Instantiate the Home startup page and add it to the stack."""
        from .screens.startup import StartupPage
        self._startup = StartupPage(APPS, _icon_for_app)
        self._startup.tile_clicked.connect(self._on_nav_selected)
        self._stack.addWidget(self._startup)

    def _on_nav_selected(self, key: str):
        """Navigate to app ``key``, lazily instantiating its screen on first use."""
        if key == "__home__":
            self._stack.setCurrentWidget(self._startup)
            self._status_app_label.setText("Home")
            self.statusBar().showMessage("Home", 2000)
            return
        if key not in self._screens:
            self._screens[key] = self._build_screen(key)
            self._stack.addWidget(self._screens[key])
        self._stack.setCurrentWidget(self._screens[key])
        # Find nice display name
        name = next((n for k, n, _d, _s in APPS if k == key), key)
        self._status_app_label.setText(name)
        self.statusBar().showMessage(f"Opened {name}", 2000)

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
        # Fall back to the last non-queue AppScreen the user visited
        for key, scr in reversed(list(self._screens.items())):
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
                pass

    @staticmethod
    def _apply_seed_value(w: QWidget, value) -> None:
        from PySide6.QtWidgets import (
            QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox,
        )
        if isinstance(w, QCheckBox):
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

    # Bundle Open Sans (Regular + Light + SemiBold) so the app renders
    # the same on every OS regardless of what fonts the user has
    # installed. Registered before applying the stylesheet so any
    # `font-family: "Open Sans"` rule resolves.
    _load_bundled_fonts()

    # Apply user preferences (theme + font scale) — falls back to the
    # dark defaults on the first launch when nothing is stored yet.
    from .preferences import apply_preferences_to_app
    apply_preferences_to_app(app)

    # Every launch drops a "spaCR started" line into
    # ~/.spacr/logs/spacr-YYYYMMDD.log so a subsequent bug report has
    # a clear timeline start. See spacr.qt.verbose_logger.current_log_file
    # for the path.
    import logging as _lg
    import sys as _sys
    from .verbose_logger import current_log_file
    _lg.getLogger("spacr").info(
        "spaCR launched (python=%s.%s.%s, log=%s)",
        _sys.version_info.major, _sys.version_info.minor,
        _sys.version_info.micro, current_log_file())

    # Real Python logging → rotating file + Qt signal so ConsolePanel
    # can render records inline. Must be set up before MainWindow so
    # child widgets can log at construct time.
    from .logging_util import setup_logging
    setup_logging()

    win = MainWindow(initial_app=initial_app)
    win.show()

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
                pass
        # Also kill any subprocess still tracked by a provider
        try:
            from . import ai as _ai
            for p in _ai.list_providers():
                p.cancel_stream()
        except Exception:
            pass
    app.aboutToQuit.connect(_drain_ai)

    return app.exec()
