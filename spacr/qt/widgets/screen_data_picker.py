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

    #: What each kind costs the user, said before the list rather than after.
    ADVICE = {
        "measurements": (
            "These are the measurement databases — what the measurement and "
            "cell functions actually read.\n\n"
            "About 0.5 GB per plate. Take only the plates you mean to work "
            "on; they are not needed together."),
        "crops": (
            "These are the object crops, and they are LARGE: about 8 GB per "
            "plate, roughly 30 GB for the whole screen.\n\n"
            "They are only needed to DISPLAY images. Every measurement and "
            "cell function reads the database instead, so if you are testing "
            "those, take the Feature download and none of this."),
        None: (
            "Choose what to download. Each piece is fetched on its own, so "
            "nothing is transferred that is not ticked.\n\n"
            "The measurement and cell functions read the DATABASES; the crop "
            "folders are only needed to display images."),
    }

    def __init__(self, folder=None, parent=None, kind=None):
        """Ask which pieces of a published screen to fetch.

        :param folder: where the download will be written.
        :param parent: parent widget.
        :param kind: which screen's data to offer. ``None`` means it could
            not be told, which is deliberately NOT the same as "nothing is
            published" -- the dialog says so rather than showing an empty
            list.
        """
        super().__init__(parent)
        self.setWindowTitle("Download screen data")
        self.setMinimumWidth(520)
        self._folder = folder
        self._kind = kind
        # One lookup for the whole dialog. None means "could not tell", which
        # is deliberately not the same as "nothing is published".
        from ...screen_data import published_archives

        self._published = published_archives()
        outer = QVBoxLayout(self)

        advice = QLabel(self.ADVICE.get(kind, self.ADVICE[None]))
        advice.setWordWrap(True)
        outer.addWidget(advice)

        self._list = QListWidget(self)
        # A tick and a highlight say the same thing, which is what makes the
        # selection legible at a glance rather than only on close inspection.
        self._list.setSelectionMode(QListWidget.MultiSelection)
        for asset in SCREEN_ASSETS:
            if kind is not None and asset.kind != kind:
                # Filtered rather than greyed: a Feature download that listed
                # eight rows and refused four of them would be four chances to
                # start a 30 GB transfer by mistake.
                continue
            item = QListWidgetItem(self._text_for(asset))
            item.setData(Qt.UserRole, asset)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            here = self._is_present(asset)
            item.setCheckState(Qt.Unchecked)
            if self._is_missing_upstream(asset):
                # Disabled, because ticking it could only fail. Left visible
                # so the set still reads as eight pieces with one not ready,
                # rather than as a set that never had it.
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                item.setToolTip("Not published yet — nothing to download.")
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
        quick_picks = ((("All", kind), ("Clear", None)) if kind is not None
                       else (("All databases", "measurements"),
                             ("All crops", "crops"), ("Clear", None)))
        for label, which in quick_picks:
            quick = QPushButton(label, self)
            quick.clicked.connect(
                lambda _checked=False, _k=which: self._tick_kind(_k))
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
        """Compose one asset's row.

        The note says whether it is already downloaded or not published yet --
        both are reasons the size beside it does not mean what it looks like.

        :param asset: the asset to describe.
        :returns: the row text.
        """
        if self._is_missing_upstream(asset):
            note = " — not published yet"
        elif self._is_present(asset):
            note = " — already downloaded"
        else:
            note = ""
        return f"{asset.label}    {human_size(asset.bytes)}{note}"

    def _is_missing_upstream(self, asset: ScreenAsset) -> bool:
        """Whether the hub does not have this piece.

        Checked ONCE when the dialog opens rather than at download time. A
        piece named in the manifest but absent upstream -- a publish that has
        not finished, or one that failed -- would otherwise be ticked, start,
        and fail after the user had committed to it.

        A lookup that could not be made is not an absence: `published_archives`
        returns None when it could not tell, and every row stays offered.
        """
        return (self._published is not None
                and asset.archive not in self._published)

    def _is_present(self, asset: ScreenAsset) -> bool:
        """Report whether an asset is already in the chosen folder.

        :param asset: the asset to check.
        :returns: ``False`` with no folder chosen, and ``False`` rather than
            raising when the check itself fails -- an unreadable folder means
            "download it", not "lose the dialog".
        """
        if self._folder is None:
            return False
        try:
            return asset.is_present(self._folder)
        except Exception:                                    # noqa: BLE001
            return False

    def _items(self):
        """Return every row of the asset list, in order."""
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
            item.setSelected(bool(kind) and asset.kind == kind
                             and bool(item.flags() & Qt.ItemIsEnabled))
        self._follow_selection()

    def _refresh_total(self, *_args) -> None:
        """Restate how much the current selection would download.

        Nothing selected disables the button rather than offering a zero-byte
        download.

        :param _args: whatever the emitting control passes; ignored.
        """
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
        """The pieces that will be downloaded.

        A row disabled for being unpublished is never returned even if some
        other path ticked it: what cannot be fetched must not be promised.
        """
        return [item.data(Qt.UserRole) for item in self._items()
                if item.checkState() == Qt.Checked
                and item.flags() & Qt.ItemIsEnabled]


def choose_screen_data(parent=None, folder=None,
                       kind: Optional[str] = None) -> List[ScreenAsset]:
    """Ask which pieces to download.

    :param kind: show only this kind, so a Feature download cannot start a
        30 GB crop transfer by a mis-click.
    :returns: the chosen pieces, empty when the dialog was cancelled.
    """
    dialog = ScreenDataPicker(folder=folder, parent=parent, kind=kind)
    if dialog.exec() != QDialog.Accepted:
        return []
    return dialog.chosen()
