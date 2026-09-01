"""Choose which pieces of the published screen to download.

The screen is 33 GB across four plates, and almost nobody wants all of it: the
Regression measurement and cell functions read the DATABASES, and the crop
folders are only needed when something has to display an image. One download
would make trying one function cost 33 GB, so each piece is fetched on its own
and this is where they are chosen.

EVERY ROW STATES ITS SIZE, and the total updates as rows are ticked. A picker
that lists eight items and then downloads an unstated number of gigabytes is
the thing this exists to avoid.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QPushButton,
                               QVBoxLayout)

from ...screen_data import (SCREEN_ASSETS, ScreenAsset, human_size,
                            total_size)

LOG = logging.getLogger("spacr.qt.screen_data_picker")

__all__ = ["ScreenDataPicker", "choose_screen_data"]


class ScreenDataPicker(QDialog):
    """A checkable list of screen pieces, each with its size."""

    def __init__(self, folder=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Download screen data")
        self.setMinimumWidth(520)
        self._folder = folder
        outer = QVBoxLayout(self)

        outer.addWidget(QLabel(
            "Choose what to download. Each piece is fetched on its own, so "
            "nothing is transferred that is not ticked.\n\n"
            "The measurement and cell functions read the DATABASES; the crop "
            "folders are only needed to display images."))

        self._list = QListWidget(self)
        # A tick and a highlight say the same thing, which is what makes the
        # selection legible at a glance rather than only on close inspection.
        self._list.setSelectionMode(QListWidget.MultiSelection)
        for asset in SCREEN_ASSETS:
            item = QListWidgetItem(self._text_for(asset))
            item.setData(Qt.UserRole, asset)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            here = self._is_present(asset)
            item.setCheckState(Qt.Unchecked)
            if here:
                # ALREADY ON DISK. Left selectable rather than disabled: a
                # re-download is how a truncated or edited copy gets repaired,
                # and a row that cannot be ticked gives no way to do that.
                item.setToolTip("Already downloaded. Tick it to fetch it "
                                "again, which replaces the copy on disk.")
            self._list.addItem(item)
        self._list.itemChanged.connect(self._refresh_total)
        self._list.itemSelectionChanged.connect(self._follow_selection)
        outer.addWidget(self._list, 1)

        buttons_row = QHBoxLayout()
        for label, kind in (("All databases", "measurements"),
                            ("All crops", "crops"),
                            ("Clear", None)):
            quick = QPushButton(label, self)
            quick.clicked.connect(
                lambda _checked=False, _k=kind: self._tick_kind(_k))
            buttons_row.addWidget(quick)
        buttons_row.addStretch(1)
        outer.addLayout(buttons_row)

        self._total = QLabel("", self)
        outer.addWidget(self._total)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self._download = QPushButton("Download", self)
        self._download.setDefault(True)
        self._download.clicked.connect(self.accept)
        buttons.addButton(self._download, QDialogButtonBox.AcceptRole)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._refresh_total()

    # -- rows ---------------------------------------------------------------

    def _text_for(self, asset: ScreenAsset) -> str:
        here = " — already downloaded" if self._is_present(asset) else ""
        return f"{asset.label}    {human_size(asset.bytes)}{here}"

    def _is_present(self, asset: ScreenAsset) -> bool:
        if self._folder is None:
            return False
        try:
            return asset.is_present(self._folder)
        except Exception:                                    # noqa: BLE001
            return False

    def _items(self):
        return [self._list.item(i) for i in range(self._list.count())]

    def _follow_selection(self) -> None:
        """Clicking a row ticks it, so one gesture does one thing.

        Without this a row can be highlighted and unticked at once, and the
        highlight -- which is what the eye reads -- would be lying about what
        will be downloaded.
        """
        blocked = self._list.blockSignals(True)
        try:
            for item in self._items():
                item.setCheckState(
                    Qt.Checked if item.isSelected() else Qt.Unchecked)
        finally:
            self._list.blockSignals(blocked)
        self._refresh_total()

    def _tick_kind(self, kind: Optional[str]) -> None:
        """Select every piece of one kind, or none at all."""
        for item in self._items():
            asset = item.data(Qt.UserRole)
            item.setSelected(bool(kind) and asset.kind == kind)
        self._follow_selection()

    def _refresh_total(self, *_args) -> None:
        chosen = self.chosen()
        if not chosen:
            self._total.setText("Nothing selected.")
            self._download.setEnabled(False)
            return
        self._total.setText(
            f"{len(chosen)} item(s), {human_size(total_size(chosen))} to "
            f"download.")
        self._download.setEnabled(True)

    def chosen(self) -> List[ScreenAsset]:
        """The pieces that will be downloaded."""
        return [item.data(Qt.UserRole) for item in self._items()
                if item.checkState() == Qt.Checked]


def choose_screen_data(parent=None, folder=None) -> List[ScreenAsset]:
    """Ask which pieces to download.

    :returns: the chosen pieces, empty when the dialog was cancelled.
    """
    dialog = ScreenDataPicker(folder=folder, parent=parent)
    if dialog.exec() != QDialog.Accepted:
        return []
    return dialog.chosen()
