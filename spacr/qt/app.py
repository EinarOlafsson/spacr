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

from .theme import PALETTE, SPACING, apply_qpalette, stylesheet


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
]


def _icon_for_app(key: str) -> Optional[QIcon]:
    """Return a QIcon for an app key. Uses the bundled spacr PNG icon if
    present; falls back to a qtawesome glyph."""
    # First: local PNG icon shipped with spacr
    here = os.path.dirname(os.path.abspath(__file__))
    resources_dir = os.path.normpath(os.path.join(here, "..", "resources", "icons"))
    for candidate in (f"{key}.png", f"{key.replace('_', ' ')}.png"):
        p = os.path.join(resources_dir, candidate)
        if os.path.exists(p):
            return QIcon(p)
    # Fallback to qtawesome
    try:
        import qtawesome as qta
    except Exception:
        return None
    QTA_MAP = {
        "mask":           "fa5s.mask",
        "measure":        "fa5s.ruler",
        "annotate":       "fa5s.tag",
        "make_masks":     "fa5s.paint-brush",
        "classify":       "fa5s.layer-group",
        "umap":           "fa5s.project-diagram",
        "ml_analyze":     "fa5s.chart-line",
        "regression":     "fa5s.wave-square",
        "recruitment":    "fa5s.crosshairs",
        "activation":     "fa5s.bolt",
        "analyze_plaques": "fa5s.microscope",
        "train_cellpose": "fa5s.dumbbell",
        "cellpose_masks": "fa5s.shapes",
        "cellpose_all":   "fa5s.th",
        "map_barcodes":   "fa5s.barcode",
    }
    name = QTA_MAP.get(key, "fa5s.puzzle-piece")
    try:
        return qta.icon(name, color=PALETTE["fg_muted"])
    except Exception:
        return None


class Sidebar(QWidget):
    """Left navigation column. Emits `nav_selected(str key)` when a tile
    is clicked. `Home` reverts to the startup page."""

    from PySide6.QtCore import Signal
    nav_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title = QLabel("SpaCR")
        title.setObjectName("SidebarTitle")
        layout.addWidget(title)

        home = QPushButton("  Home")
        home.setObjectName("SidebarItem")
        home.setCursor(Qt.PointingHandCursor)
        try:
            import qtawesome as qta
            home.setIcon(qta.icon("fa5s.home", color=PALETTE["fg_muted"]))
        except Exception:
            pass
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
            icon = _icon_for_app(key)
            if icon is not None:
                btn.setIcon(icon)
                btn.setIconSize(QSize(16, 16))
            btn.clicked.connect(lambda checked=False, k=key: self.nav_selected.emit(k))
            layout.addWidget(btn)

        layout.addStretch(1)


class MainWindow(QMainWindow):
    def __init__(self, initial_app: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("SpaCR")
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

        # Status bar
        status = QStatusBar()
        status.showMessage("Ready")
        self.setStatusBar(status)

        if initial_app:
            self._on_nav_selected(initial_app)

    # -- menu -------------------------------------------------------------
    def _build_menu_bar(self):
        mb = self.menuBar()

        app_menu = mb.addMenu("&SpaCR")
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
        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        app_menu.addAction(act_quit)

        help_menu = mb.addMenu("&Help")
        act_tutorial = QAction("Tutorial (web)", self)
        act_tutorial.triggered.connect(
            lambda: self._open_url("https://einarolafsson.github.io/spacr/tutorial/"))
        help_menu.addAction(act_tutorial)
        act_docs = QAction("Documentation (web)", self)
        act_docs.triggered.connect(
            lambda: self._open_url("https://einarolafsson.github.io/spacr/index.html"))
        help_menu.addAction(act_docs)
        act_about = QAction("About SpaCR", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _open_url(self, url: str):
        import webbrowser
        try:
            webbrowser.open(url)
        except Exception as e:
            self.statusBar().showMessage(f"Failed to open {url}: {e}", 5000)

    def _show_about(self):
        try:
            import spacr
            version = spacr.__version__
        except Exception:
            version = "unknown"
        QMessageBox.about(self, "About SpaCR",
                          f"<h3>SpaCR</h3>"
                          f"<p>Spatial single-cell analysis for microscopy data.</p>"
                          f"<p><b>Version:</b> {version}</p>"
                          f"<p>© Olafsson Lab</p>")

    # -- navigation -------------------------------------------------------
    def _install_startup_page(self):
        from .screens.startup import StartupPage
        self._startup = StartupPage(APPS, _icon_for_app)
        self._startup.tile_clicked.connect(self._on_nav_selected)
        self._stack.addWidget(self._startup)

    def _on_nav_selected(self, key: str):
        if key == "__home__":
            self._stack.setCurrentWidget(self._startup)
            self.statusBar().showMessage("Home", 2000)
            return
        if key not in self._screens:
            self._screens[key] = self._build_screen(key)
            self._stack.addWidget(self._screens[key])
        self._stack.setCurrentWidget(self._screens[key])
        # Find nice display name
        name = next((n for k, n, _d, _s in APPS if k == key), key)
        self.statusBar().showMessage(f"{name}", 2000)

    def _build_screen(self, key: str) -> QWidget:
        if key == "annotate":
            from .screens.annotate import AnnotateScreen
            return AnnotateScreen()
        if key == "make_masks":
            from .screens.make_masks import MakeMasksScreen
            return MakeMasksScreen()
        from .screens.app_screen import AppScreen
        return AppScreen(app_key=key)


def launch(argv=None) -> int:
    """Bootstrap QApplication and show the main window."""
    if argv is None:
        argv = sys.argv[1:]

    # Support `spacr-qt <app>` to open directly into an app.
    initial_app = argv[0] if argv else None

    # Enable high-DPI early.
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QApplication(sys.argv[:1])
    app.setApplicationName("SpaCR")
    app.setOrganizationName("Olafsson Lab")
    apply_qpalette(app)
    app.setStyleSheet(stylesheet())

    win = MainWindow(initial_app=initial_app)
    win.show()
    return app.exec()
