"""Make Masks' example-set check.

Thin Qt over :mod:`spacr.channel_sorting`, which holds every decision:

* :class:`ExampleSetsDialog` shows three detected sets side by side, every
  channel with its mask, and asks whether they are right.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
    QWidget,
)

from ... import channel_sorting as cs
from ..bridge import emit_safely
from ..i18n import tr

#: Thumbnail side in the image list, in pixels.
THUMB = 64

def _grey_icon(array: Optional[np.ndarray], mask: Optional[np.ndarray] = None,
               size: int = THUMB) -> QPixmap:
    """A pixmap of a grey thumbnail, tinted red where ``mask`` has objects.

    :param array: 2-D ``uint8`` pixels, or None for a blank tile.
    :param mask: 2-D ``uint8`` mask preview of the same size, or None.
    :param size: the tile's side when ``array`` is None.
    """
    if array is None:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.gray)
        return pixmap
    rgb = np.repeat(array[:, :, None], 3, axis=2).copy()
    if mask is not None and mask.shape == array.shape:
        on = mask > 0
        rgb[on, 0] = np.maximum(rgb[on, 0], 200)
        rgb[on, 1] = (rgb[on, 1] * 0.5).astype(np.uint8)
        rgb[on, 2] = (rgb[on, 2] * 0.5).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)
    height, width = rgb.shape[:2]
    image = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class _ThumbWorker(QThread):
    """Reads thumbnails off the GUI thread, one signal per image."""

    ready = Signal(str, object, object)

    def __init__(self, folder: str, names: List[str],
                 masks_dir: Optional[str] = None, parent=None, *, size=None):
        """Remember what to read.

        :param folder: the image folder.
        :param names: the images, in list order.
        :param masks_dir: an explicit masks folder.
        :param parent: the Qt parent.
        :param size: source sampling limit; defaults to the persisted quality.
        """
        super().__init__(parent)
        from ..mask_thumbnail_quality import QUALITY_SIZES, get_quality

        self._size = int(size or QUALITY_SIZES[get_quality()])
        self._folder = folder
        self._names = list(names)
        self._masks_dir = masks_dir
        self._stop = False

    def stop(self) -> None:
        """Ask the loop to end after the image it is reading."""
        self._stop = True

    def run(self) -> None:
        """Emit ``(name, thumbnail, mask thumbnail)`` for each image."""
        for name in self._names:
            if self._stop:
                return
            try:
                image = cs.thumbnail(os.path.join(self._folder, name), self._size)
                mask_path = cs.mask_for(self._folder, name, self._masks_dir)
                mask = (cs.mask_thumbnail(mask_path, self._size)
                        if mask_path else None)
            except Exception:
                image, mask = None, None
            if not emit_safely(self.ready, name, image, mask):
                return


class ExampleSetsDialog(QDialog):
    """Three sets side by side -- every channel and its mask -- and a question."""

    def __init__(self, folder: str, sets: Dict[cs.SetKey, Dict[int, str]],
                 masks_dir: Optional[str] = None, parent=None):
        """Lay out up to three sets: the first, the middle and the last.

        :param folder: the image folder.
        :param sets: ``{set key: {channel: name}}``.
        :param masks_dir: an explicit masks folder.
        :param parent: the Qt parent.
        """
        super().__init__(parent)
        self.setWindowTitle(tr("Are these sets right?"))
        layout = QVBoxLayout(self)
        intro = QLabel(tr(
            "spaCR paired each image with the most similar image of the same "
            "size in every other channel. Here are three of the {n} sets it "
            "found, each channel with its mask in red. Are they right?",
            n=len(sets)))
        intro.setWordWrap(True)
        layout.addWidget(intro)
        keys = list(sets)
        picks = sorted({0, len(keys) // 2, len(keys) - 1}) if keys else []
        self.shown = [keys[i] for i in picks]
        self._preview_tiles = []
        grid = QGridLayout()
        for row, key in enumerate(self.shown):
            for column, (channel, name) in enumerate(sorted(sets[key].items())):
                path = os.path.join(folder, name)
                mask_path = cs.mask_for(folder, name, masks_dir)
                tile = QLabel()
                tile.setFixedSize(160, 160)
                tile.setAlignment(Qt.AlignCenter)
                self._preview_tiles.append((tile, path, mask_path))
                caption = QLabel(tr("Channel {channel}: {name}",
                                    channel=channel, name=name))
                caption.setWordWrap(True)
                cell = QVBoxLayout()
                cell.addWidget(tile)
                cell.addWidget(caption)
                holder = QWidget()
                holder.setLayout(cell)
                grid.addWidget(holder, row, column)
        layout.addLayout(grid)
        from ..mask_thumbnail_quality import changes

        changes.changed.connect(self._refresh_previews)
        self._refresh_previews()
        buttons = QHBoxLayout()
        self.yes_button = QPushButton(tr("Yes, these are right"))
        self.no_button = QPushButton(tr("No"))
        self.yes_button.clicked.connect(self.accept)
        self.no_button.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(self.no_button)
        buttons.addWidget(self.yes_button)
        layout.addLayout(buttons)

    def _refresh_previews(self, _quality=None):
        """Refresh aligned source/mask previews without changing tile geometry.

        :param _quality: optional notification of a shared quality change.
        """
        from ..hidpi import scaled_for
        from ..mask_thumbnail_quality import QUALITY_SIZES, get_quality

        size = max(160, QUALITY_SIZES[get_quality()])
        for tile, path, mask_path in self._preview_tiles:
            picture = _grey_icon(cs.thumbnail(path, size),
                                cs.mask_thumbnail(mask_path, size) if mask_path else None)
            tile.setPixmap(scaled_for(picture, tile, 160))
