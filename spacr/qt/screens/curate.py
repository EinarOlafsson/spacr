"""``B12`` ``C7`` — the screen where a mask and its tracks get corrected.

A layer viewer with two panels beside it: the brush, and the track surgery.
They are on one screen because they are one job. A track breaks *because* the
mask lost the cell for a frame, so the person who joins the track is the
person who wants to look at the mask at that frame, and making them two apps
would mean two file dialogs and two mental models for one correction.

Everything on this screen is recorded. See :mod:`spacr.curation` for why that
is a rule rather than a feature: a hand-edited mask that looks exactly like a
segmented one is a reproducibility hole, and the ledger written beside each
artefact is what lets a curated dataset be told from a raw one six months
later by someone who was not there.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QSplitter, QTabWidget,
                               QVBoxLayout, QWidget)

from ...curation import is_curated
from ..curation_tool import BrushPanel, TrackCurationPanel
from ..layer_viewer import LayerViewer
from ..theme import SPACING, active_palette

LOG = logging.getLogger(__name__)

__all__ = [
    "CurateScreen",
    "register",
    "APP_KEY",
]

APP_KEY = "curate"


class CurateScreen(QWidget):
    """Correct a mask by hand and curate its tracks, on the record."""

    #: A mask was opened. Carries the path.
    mask_opened = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CurateScreen")
        self._mask_path = ""
        self.brush: Optional[BrushPanel] = None
        self._build()

    # -- construction --------------------------------------------------------
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["sm"])

        title = QLabel("Curate", self)
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)
        intro = QLabel(
            "Paint over a mask to correct it, and join, split or delete "
            "tracks by hand. Every correction is written to a ledger beside "
            "the file, so a curated dataset can be told from a raw one.", self)
        intro.setObjectName("Muted")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        source = QHBoxLayout()
        source.addWidget(QLabel("Mask", self))
        self._mask_edit = QLineEdit(self)
        self._mask_edit.setPlaceholderText("…/masks/plate1_A01_1.tif")
        self._mask_edit.returnPressed.connect(self.open_mask)
        source.addWidget(self._mask_edit, 1)
        self._browse_mask = QPushButton("Browse…", self)
        self._browse_mask.clicked.connect(self._choose_mask)
        source.addWidget(self._browse_mask)
        self._open_mask = QPushButton("Open", self)
        self._open_mask.setObjectName("PrimaryButton")
        self._open_mask.clicked.connect(self.open_mask)
        source.addWidget(self._open_mask)
        outer.addLayout(source)

        tracks_row = QHBoxLayout()
        tracks_row.addWidget(QLabel("Tracks", self))
        self._tracks_edit = QLineEdit(self)
        self._tracks_edit.setPlaceholderText(
            "…/tracks/btrack_tracks_cell_plate1_A01_1.csv")
        self._tracks_edit.returnPressed.connect(self.open_tracks)
        tracks_row.addWidget(self._tracks_edit, 1)
        self._browse_tracks = QPushButton("Browse…", self)
        self._browse_tracks.clicked.connect(self._choose_tracks)
        tracks_row.addWidget(self._browse_tracks)
        self._open_tracks = QPushButton("Open", self)
        self._open_tracks.clicked.connect(self.open_tracks)
        tracks_row.addWidget(self._open_tracks)
        outer.addLayout(tracks_row)

        split = QSplitter(Qt.Horizontal, self)
        self.viewer = LayerViewer(parent=self)
        split.addWidget(self.viewer)

        self.tabs = QTabWidget(self)
        self.brush_host = QWidget(self.tabs)
        self._brush_layout = QVBoxLayout(self.brush_host)
        self._brush_layout.setContentsMargins(0, 0, 0, 0)
        self._brush_hint = QLabel(
            "Open a mask to paint on it.", self.brush_host)
        self._brush_hint.setWordWrap(True)
        self._brush_layout.addWidget(self._brush_hint)
        self._brush_layout.addStretch(1)
        self.tabs.addTab(self.brush_host, "Brush")

        self.tracks = TrackCurationPanel(self.tabs)
        self.tabs.addTab(self.tracks, "Tracks")
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([680, 320])
        outer.addWidget(split, 1)

        self.status = QLabel("", self)
        self.status.setObjectName("Muted")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

    # -- the mask ------------------------------------------------------------
    def _choose_mask(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a label mask", self._mask_edit.text().strip(),
            "Masks (*.tif *.tiff *.png *.npy);;All files (*)")
        if path:
            self._mask_edit.setText(path)
            self.open_mask()

    def open_mask(self) -> Optional[BrushPanel]:
        """Load the mask and put a brush on it. Returns the panel."""
        path = self._mask_edit.text().strip()
        if not path or not os.path.isfile(path):
            self.status.setText("Choose a mask file first.")
            return None
        layer = self.viewer.add_labels_file(path)
        if layer is None:
            self.status.setText(f"Could not load {path} as a label mask.")
            return None
        self._mask_path = path
        panel = self.attach_brush(layer, artifact=path)
        self.mask_opened.emit(path)
        self._say_whether_curated(path)
        return panel

    def attach_brush(self, layer, *, artifact: str = "") -> BrushPanel:
        """Put a brush panel over ``layer``. The seam a test goes through."""
        if self.brush is not None:
            self.brush.stop_painting()
            self.brush.setParent(None)
            self.brush.deleteLater()
        self._brush_hint.setVisible(False)
        self.brush = BrushPanel(self.viewer.canvas, self.brush_host,
                                layer=layer, artifact=artifact)
        self._brush_layout.insertWidget(0, self.brush)
        return self.brush

    # -- the tracks ----------------------------------------------------------
    def _choose_tracks(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a tracks table", self._tracks_edit.text().strip(),
            "CSV (*.csv);;All files (*)")
        if path:
            self._tracks_edit.setText(path)
            self.open_tracks()

    def open_tracks(self) -> bool:
        """Load a tracks CSV into the track panel."""
        path = self._tracks_edit.text().strip()
        if not path or not os.path.isfile(path):
            self.status.setText("Choose a tracks CSV first.")
            return False
        opened = self.tracks.load(path) is not None
        if opened:
            self.tabs.setCurrentWidget(self.tracks)
            self._say_whether_curated(path)
        return opened

    def _say_whether_curated(self, path: str) -> None:
        """The one question the ledger exists to answer, on screen.

        Said on *open*, not on save: the person who needs to know whether a
        file has been edited by hand is the one about to analyse it, and they
        find out by opening it.
        """
        if is_curated(path):
            self.status.setText(
                f"{os.path.basename(path)} has been curated by hand — see "
                f"{os.path.basename(path)}.curation.json for what changed.")
            self.status.setStyleSheet(
                f"color: {active_palette()['warning']};")
        else:
            self.status.setText(
                f"{os.path.basename(path)} is as the pipeline produced it.")
            self.status.setStyleSheet("")

    def closeEvent(self, event) -> None:
        if self.brush is not None:
            self.brush.stop_painting()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

APP_NAME = "Curate"
APP_DESCRIPTION = "Paint a mask right, and fix tracks by hand — on the record"
APP_INTRO = (
    "Cellpose merges two touching cells; btrack breaks a track when a cell "
    "leaves focus for a frame. Both are obvious to look at and, until now, "
    "impossible to correct. Paint over the mask with a world-space brush, "
    "join / split / delete tracks by hand, and every correction is written "
    "to a ledger beside the file — so a curated dataset can be told from a "
    "raw one, by someone who was not there when it was edited.")
APP_CLI_NOTE = (
    "Curate is hand correction — the brush and the track surgery are the "
    "whole feature; run it in the GUI (spacr-qt). Headless, "
    "spacr.curation.TrackCuration does the same edits to a tracks table with "
    "the same ledger.")


def make_curate_screen(**_kwargs) -> CurateScreen:
    """Build the screen. The ``factory=`` for :func:`spacr.qt.app.register_app`."""
    return CurateScreen()


def register(*, section: Optional[str] = None, stage: Optional[str] = None,
             key: str = APP_KEY):
    """Put Curate in the app registry. Idempotent.

    :returns: the registry row, or ``None`` when the key was already there.
    """
    from ..app import APPS, SECTION_CORE, STAGE_ALPHA, register_app
    if any(row[0] == key for row in APPS):
        return None
    return register_app(
        key, APP_NAME, APP_DESCRIPTION, section or SECTION_CORE,
        factory=make_curate_screen,
        stage=STAGE_ALPHA if stage is None else stage,
        intro=APP_INTRO, cli_note=APP_CLI_NOTE,
        api_module="qt/screens/curate",
        translations=("Kurera", "Kuratieren", "Curar", "校正", "Curar",
                      "क्यूरेट", "큐레이트", "Grisja", "Curer"))
