"""Make Masks' "Sort into channels…" dialog, its regex window and its set check.

Thin Qt over :mod:`spacr.channel_sorting`, which holds every decision:

* :class:`ChannelSortDialog` lists the open folder's images with thumbnails
  and whether each has a mask. Images are selected (click, Shift, Ctrl,
  Select all) and assigned to a channel -- the SELECTION strategy -- and a
  regex can assign every image instead -- the REGEX strategy. Apply builds a
  :class:`spacr.channel_sorting.SortPlan`; when the names cannot be turned
  into sets, the user is asked to change the regex.
* :class:`RegexWindow` explains the named groups, tests a regex live against
  every file name, proposes one ("Auto regex") or pairs images by similarity
  and size, ignoring names ("Detect sets").
* :class:`ExampleSetsDialog` shows three detected sets side by side, every
  channel with its mask, and asks whether they are right.

Nothing here moves a file. The dialog hands back a plan; the screen applies
it off the GUI thread.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QTableWidget, QVBoxLayout, QWidget,
)

from ... import channel_sorting as cs
from ..bridge import emit_safely
from ..i18n import tr
from .sortable_table import install_sorting, table_item

#: Thumbnail side in the image list, in pixels.
THUMB = 64

#: The mask-role choices, with "none" for masks that are moved but not merged.
ROLE_CHOICES = ("none",) + cs.MASK_ROLES


def _headless() -> bool:
    """Whether no one can answer a modal box (offscreen/minimal platform)."""
    from PySide6.QtGui import QGuiApplication

    return QGuiApplication.platformName() in ("offscreen", "minimal")


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
                 masks_dir: Optional[str] = None, parent=None):
        """Remember what to read.

        :param folder: the image folder.
        :param names: the images, in list order.
        :param masks_dir: an explicit masks folder.
        :param parent: the Qt parent.
        """
        super().__init__(parent)
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
                image = cs.thumbnail(os.path.join(self._folder, name), THUMB)
                mask_path = cs.mask_for(self._folder, name, self._masks_dir)
                mask = (cs.mask_thumbnail(mask_path, THUMB)
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
        grid = QGridLayout()
        for row, key in enumerate(self.shown):
            for column, (channel, name) in enumerate(sorted(sets[key].items())):
                path = os.path.join(folder, name)
                mask_path = cs.mask_for(folder, name, masks_dir)
                tile = QLabel()
                tile.setPixmap(_grey_icon(
                    cs.thumbnail(path, 160),
                    cs.mask_thumbnail(mask_path, 160) if mask_path else None))
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
        buttons = QHBoxLayout()
        self.yes_button = QPushButton(tr("Yes, these are right"))
        self.no_button = QPushButton(tr("No"))
        self.yes_button.clicked.connect(self.accept)
        self.no_button.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(self.no_button)
        buttons.addWidget(self.yes_button)
        layout.addLayout(buttons)


class RegexWindow(QDialog):
    """Write, test, infer or bypass the regex that sorts images into sets.

    :ivar regex: the regex when the window is accepted with "Use this regex".
    :ivar detected: the sets "Detect sets" found and the user confirmed, or
        None.
    """

    def __init__(self, folder: str, names: List[str],
                 channels: Optional[Dict[str, int]] = None, regex: str = "",
                 masks_dir: Optional[str] = None, parent=None):
        """Build the window and run the test once.

        :param folder: the image folder.
        :param names: the image names to test against.
        :param channels: ``{name: channel}`` from the selection.
        :param regex: the regex to start from.
        :param masks_dir: an explicit masks folder.
        :param parent: the Qt parent.
        """
        super().__init__(parent)
        self.setWindowTitle(tr("Regex for channels and sets"))
        self.resize(900, 640)
        self._folder = folder
        self._names = list(names)
        self._channels = dict(channels or {})
        self._masks_dir = masks_dir
        self.regex = regex
        self.detected: Optional[Dict[cs.SetKey, Dict[int, str]]] = None
        #: Answers the example-sets question; replaced in tests.
        self.confirm_examples = self._ask_examples

        layout = QVBoxLayout(self)
        guide = QLabel(tr(
            "A regex with named groups sorts every image into a channel and "
            "a set (one field across all channels). chanID is the channel; "
            "plateID, wellID, fieldID and timeID say where a set goes; any "
            "other named group only tells sets apart. Images assigned to a "
            "channel by selection keep that channel.\n"
            "Examples:\n"
            "spaCR's Yokogawa default: {cellvoyager}\n"
            "well, field and channel: {typical}\n"
            "names made by Consolidate folders: {consolidated}",
            cellvoyager=cs.CELLVOYAGER_EXAMPLE,
            typical=r"(?P<wellID>[A-P]\d{2})_(?P<fieldID>\d+)_(?P<chanID>\d+)\.tif",
            consolidated=r"exp_(?P<chanID>[A-Za-z]+)(?:_(?P<fieldID>\d+))?\.tif"))
        guide.setWordWrap(True)
        guide.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(guide)

        self.regex_edit = QLineEdit(regex)
        self.regex_edit.setPlaceholderText(tr("Regex with named groups"))
        layout.addWidget(self.regex_edit)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.run_test)
        self.regex_edit.textChanged.connect(lambda _text: self._timer.start())

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            tr("File"), tr("Matched"), tr("Channel"), tr("Set"), tr("Groups")])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        install_sorting(self.table)
        layout.addWidget(self.table, 1)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        buttons = QHBoxLayout()
        self.auto_button = QPushButton(tr("Auto regex"))
        self.auto_button.setToolTip(tr(
            "Find a regex that puts every image in exactly one complete set: "
            "the token that changes with the channel, and the tokens that "
            "name the field."))
        self.detect_button = QPushButton(tr("Detect sets"))
        self.detect_button.setToolTip(tr(
            "Ignore the names' structure: pair each image with the most "
            "similar image of the same size in every other channel, then "
            "show three sets to check."))
        self.use_button = QPushButton(tr("Use this regex"))
        self.cancel_button = QPushButton(tr("Cancel"))
        self.auto_button.clicked.connect(self.auto_regex)
        self.detect_button.clicked.connect(self.detect)
        self.use_button.clicked.connect(self._use)
        self.cancel_button.clicked.connect(self.reject)
        for button in (self.auto_button, self.detect_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.use_button)
        layout.addLayout(buttons)
        self.run_test()

    def run_test(self) -> cs.SetReport:
        """Parse every name with the regex and fill the table and summary.

        :returns: the set report.
        """
        pattern = self.regex_edit.text()
        _compiled, error = cs.compile_regex(pattern) if pattern else (None, None)
        parsed = cs.parse_names(self._names, pattern if not error else "",
                                self._channels)
        report = cs.check_sets(parsed)
        self.table.setRowCount(len(parsed))
        for row, item in enumerate(parsed):
            values = [
                item.name,
                tr("yes") if item.matched else tr("no"),
                "" if item.channel is None else str(item.channel),
                cs.set_label(item.set_key),
                " ".join(f"{k}={v}" for k, v in item.groups.items()),
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column, table_item(value))
        if error:
            self.summary.setText(tr("The regex is not valid: {error}",
                                    error=error))
        else:
            self.summary.setText(report.summary())
        self.use_button.setEnabled(bool(pattern) and not error)
        return report

    def auto_regex(self) -> Optional[str]:
        """Propose a regex, put it in the field and test it.

        :returns: the regex, or None when none was found.
        """
        pattern = cs.infer_regex(self._names, self._channels)
        if not pattern:
            self.summary.setText(tr(
                "spaCR found no regex that puts every image in exactly one "
                "complete set. Assign channels by selection and try again, or "
                "use Detect sets."))
            return None
        self.regex_edit.setText(pattern)
        self._timer.stop()
        self.run_test()
        return pattern

    def _channels_for_detection(self) -> Dict[str, int]:
        """Each image's channel: the selection's, else the regex's."""
        parsed = cs.parse_names(self._names, self.regex_edit.text(),
                                self._channels)
        return {p.name: p.channel for p in parsed if p.channel is not None}

    def detect(self) -> Optional[Dict[cs.SetKey, Dict[int, str]]]:
        """Pair images into sets, show three, and keep them when confirmed.

        :returns: the sets when the user said they are right, else None.
        """
        channels = self._channels_for_detection()
        if len(set(channels.values())) < 2:
            self.summary.setText(tr(
                "Detect sets needs at least two channels: assign images to "
                "channels first, by selection or with a chanID group."))
            return None
        found = cs.detect_sets(self._folder, channels, self._masks_dir)
        if not found.sets:
            self.summary.setText("\n".join(found.problems))
            return None
        note = "\n".join(found.problems)
        self.summary.setText(tr("{n} set(s) detected.", n=len(found.sets))
                             + ("\n" + note if note else ""))
        if not self.confirm_examples(found.sets):
            return None
        self.detected = found.sets
        self.accept()
        return found.sets

    def _ask_examples(self, sets) -> bool:
        """Show three sets and ask; False when no one can answer.

        :param sets: the detected sets.
        """
        if _headless():
            return False
        dialog = ExampleSetsDialog(self._folder, sets, self._masks_dir, self)
        return dialog.exec() == QDialog.Accepted

    def _use(self) -> None:
        """Keep the regex in the field and close."""
        self.regex = self.regex_edit.text()
        self.detected = None
        self.accept()


class ChannelSortDialog(QDialog):
    """Assign the folder's images to channels, check the sets, plan the move.

    :ivar plan: the confirmed :class:`spacr.channel_sorting.SortPlan` once the
        dialog is accepted, else None.
    """

    def __init__(self, folder: str, masks_dir: Optional[str] = None,
                 parent=None):
        """List the folder and propose a regex.

        :param folder: the folder Make Masks has open.
        :param masks_dir: its masks folder, when not ``<folder>/masks``.
        :param parent: the Qt parent.
        """
        super().__init__(parent)
        self.setWindowTitle(tr("Sort into channels"))
        self.resize(820, 680)
        self.folder = folder
        self.masks_dir = masks_dir
        self.names = cs.list_folder_images(folder)
        self.channels: Dict[str, int] = {}
        self.regex = cs.infer_regex(self.names) or ""
        self.detected: Optional[Dict[cs.SetKey, Dict[int, str]]] = None
        self.plan: Optional[cs.SortPlan] = None
        self._channel_count = 2
        self._role_boxes: Dict[int, QComboBox] = {}
        #: Opens the regex window; replaced in tests.
        self.open_regex_window = self._open_regex_window

        layout = QVBoxLayout(self)
        intro = QLabel(tr(
            "Select images and assign them to a channel, or let a regex "
            "assign every image. On Apply the images and their masks are "
            "MOVED into one folder per channel under {dest}, renamed in "
            "Yokogawa format, and merged into merged/. Every move is written "
            "to a manifest.", dest=cs.DEFAULT_DEST_NAME))
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setIconSize(QPixmap(THUMB, THUMB).size())
        self._items: Dict[str, QListWidgetItem] = {}
        for name in self.names:
            item = QListWidgetItem(QIcon(_grey_icon(None)), "")
            item.setData(Qt.UserRole, name)
            self.list.addItem(item)
            self._items[name] = item
        layout.addWidget(self.list, 1)

        select_row = QHBoxLayout()
        self.select_all_button = QPushButton(tr("Select all"))
        self.select_all_button.clicked.connect(self.list.selectAll)
        self.clear_button = QPushButton(tr("Clear channel"))
        self.clear_button.setToolTip(tr(
            "Take the selected images out of their channel, so the regex "
            "decides theirs."))
        self.clear_button.clicked.connect(lambda: self.assign_selected(None))
        select_row.addWidget(self.select_all_button)
        select_row.addWidget(self.clear_button)
        self._channel_row = QHBoxLayout()
        select_row.addLayout(self._channel_row)
        self.add_channel_button = QPushButton(tr("Add channel"))
        self.add_channel_button.clicked.connect(self.add_channel)
        select_row.addWidget(self.add_channel_button)
        select_row.addStretch(1)
        layout.addLayout(select_row)
        self._channel_buttons: List[QPushButton] = []
        for channel in range(1, self._channel_count + 1):
            self._add_channel_button(channel)

        regex_row = QHBoxLayout()
        regex_row.addWidget(QLabel(tr("Regex")))
        self.regex_label = QLineEdit(self.regex)
        self.regex_label.setReadOnly(True)
        regex_row.addWidget(self.regex_label, 1)
        self.regex_button = QPushButton(tr("Edit regex…"))
        self.regex_button.clicked.connect(lambda: self.open_regex_window())
        regex_row.addWidget(self.regex_button)
        layout.addLayout(regex_row)

        self._roles_row = QHBoxLayout()
        layout.addLayout(self._roles_row)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QHBoxLayout()
        self.cancel_button = QPushButton(tr("Cancel"))
        self.apply_button = QPushButton(tr("Apply"))
        self.apply_button.setObjectName("PrimaryButton")
        self.cancel_button.clicked.connect(self.reject)
        self.apply_button.clicked.connect(self._on_apply)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.apply_button)
        layout.addLayout(buttons)

        self.refresh()
        self._thumbs = _ThumbWorker(folder, self.names, masks_dir, self)
        self._thumbs.ready.connect(self._take_thumbnail)
        self._thumbs.start()
        self.finished.connect(lambda _code: self._stop_thumbs())

    def _stop_thumbs(self) -> None:
        """End the thumbnail thread and wait for it."""
        if self._thumbs.isRunning():
            self._thumbs.stop()
            self._thumbs.wait(5000)

    def _take_thumbnail(self, name: str, image, mask) -> None:
        """Put one thumbnail, mask in red, on its list item.

        :param name: the image name.
        :param image: its thumbnail, or None.
        :param mask: its mask thumbnail, or None.
        """
        item = self._items.get(name)
        if item is not None and image is not None:
            item.setIcon(QIcon(_grey_icon(image, mask)))

    def _add_channel_button(self, channel: int) -> None:
        """Add the "Channel N" button that assigns the selection to N.

        :param channel: the 1-based channel.
        """
        button = QPushButton(tr("Channel {n}", n=channel))
        button.setToolTip(tr("Assign the selected images to channel {n}.",
                             n=channel))
        button.clicked.connect(lambda _checked=False, c=channel:
                               self.assign_selected(c))
        self._channel_row.addWidget(button)
        self._channel_buttons.append(button)

    def add_channel(self) -> int:
        """Offer one more channel.

        :returns: the new channel's number.
        """
        self._channel_count += 1
        self._add_channel_button(self._channel_count)
        return self._channel_count

    def selected_names(self) -> List[str]:
        """The selected images' names, in list order."""
        return [self.list.item(i).data(Qt.UserRole)
                for i in range(self.list.count())
                if self.list.item(i).isSelected()]

    def assign_selected(self, channel: Optional[int]) -> None:
        """Assign the selected images to ``channel``; None clears them.

        Assigning by hand replaces any sets "Detect sets" found.

        :param channel: the 1-based channel, or None.
        """
        for name in self.selected_names():
            if channel is None:
                self.channels.pop(name, None)
            else:
                self.channels[name] = int(channel)
        self.detected = None
        self.refresh()

    def set_regex(self, regex: str) -> None:
        """Use ``regex`` for channels and sets from now on.

        :param regex: the pattern.
        """
        self.regex = regex or ""
        self.detected = None
        self.regex_label.setText(self.regex)
        self.refresh()

    def parsed(self) -> List[cs.ParsedName]:
        """Every image read through the regex, with the selection's channels."""
        return cs.parse_names(self.names, self.regex, self.channels)

    def current_sets(self):
        """The sets Apply would use, and what is wrong when there are none.

        :returns: ``(sets or None, report text)``.
        """
        if self.detected is not None:
            return self.detected, tr("{n} set(s) detected by pairing.",
                                     n=len(self.detected))
        report = cs.check_sets(self.parsed())
        return (report.complete if report.ok else None), report.summary()

    def refresh(self) -> None:
        """Redraw the list texts, the mask-role choices and the status line."""
        channel_of = {p.name: p.channel for p in self.parsed()}
        if self.detected is not None:
            channel_of = {name: channel for members in self.detected.values()
                          for channel, name in members.items()}
        for name, item in self._items.items():
            has_mask = cs.mask_for(self.folder, name, self.masks_dir) is not None
            channel = channel_of.get(name)
            item.setText(tr(
                "{name}\n{mask} · channel {channel}", name=name,
                mask=tr("mask") if has_mask else tr("no mask"),
                channel="?" if channel is None else channel))
        masked = sorted({channel for name, channel in channel_of.items()
                         if channel is not None
                         and cs.mask_for(self.folder, name, self.masks_dir)})
        self._rebuild_roles(masked, channel_of)
        _sets, text = self.current_sets()
        self.status.setText(text)

    def _rebuild_roles(self, masked: List[int], channel_of) -> None:
        """One mask-role choice per channel that has masks.

        :param masked: channels with at least one mask.
        :param channel_of: ``{name: channel}``.
        """
        if sorted(self._role_boxes) == masked:
            return
        while self._roles_row.count():
            widget = self._roles_row.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        self._role_boxes = {}
        members: Dict[int, List[str]] = {}
        for name, channel in channel_of.items():
            if channel is not None:
                members.setdefault(channel, []).append(name)
        guess = cs.default_mask_roles(members, masked)
        for channel in masked:
            self._roles_row.addWidget(QLabel(tr("Channel {n} masks are",
                                                n=channel)))
            box = QComboBox()
            captions = {"none": tr("not merged"), "cell": tr("cell"),
                        "nucleus": tr("nucleus"), "pathogen": tr("pathogen")}
            for role in ROLE_CHOICES:
                box.addItem(captions[role], role)
            box.setCurrentIndex(ROLE_CHOICES.index(guess.get(channel, "none")))
            self._roles_row.addWidget(box)
            self._role_boxes[channel] = box
        self._roles_row.addStretch(1)

    def mask_roles(self) -> Dict[int, str]:
        """``{channel: role}`` as chosen, leaving out ``none``."""
        return {channel: box.currentData()
                for channel, box in self._role_boxes.items()
                if box.currentData() in cs.MASK_ROLES}

    def prepare_plan(self, convert: bool = False) -> Optional[cs.SortPlan]:
        """Build the plan from the current sets, or None when there are none.

        :param convert: convert RGB images and z-stacks to one plane.
        :returns: the plan, which may still carry problems.
        """
        sets, _text = self.current_sets()
        if not sets:
            return None
        return cs.build_plan(self.folder, sets, masks_dir=self.masks_dir,
                             mask_roles=self.mask_roles(), convert=convert)

    def _ask_to_convert(self, plan: cs.SortPlan) -> bool:
        """Ask whether RGB images and z-stacks should be converted.

        :param plan: a plan whose ``convertible`` list is not empty.
        :returns: whether the user said yes (always False headless).
        """
        shown = "\n".join(plan.convertible[:8])
        more = len(plan.convertible) - 8
        if more > 0:
            shown += "\n" + tr("…and {n} more", n=more)
        question = tr(
            "{n} image(s) are not a single grey plane:\n{names}\n\nConvert "
            "them? RGB images become grey (the mean of their colours) and "
            "z-stacks become their maximum projection. The originals are kept "
            "in originals/ of the sorted folder.",
            n=len(plan.convertible), names=shown)
        return (not _headless() and QMessageBox.question(
            self, tr("Convert RGB images and z-stacks?"),
            question) == QMessageBox.Yes)

    def _open_regex_window(self) -> bool:
        """Open the regex window and take what it returns.

        :returns: whether the window was accepted.
        """
        window = RegexWindow(self.folder, self.names, self.channels,
                             self.regex, self.masks_dir, self)
        if _headless() or window.exec() != QDialog.Accepted:
            return False
        if window.detected is not None:
            self.regex = window.regex_edit.text()
            self.regex_label.setText(self.regex)
            self.detected = window.detected
            self.refresh()
        else:
            self.set_regex(window.regex)
        return True

    def _on_apply(self) -> None:
        """Plan, confirm and accept -- or send the user to the regex window."""
        plan = self.prepare_plan()
        if plan is None:
            _sets, text = self.current_sets()
            question = tr(
                "The names cannot be turned into Yokogawa names with the "
                "current regex: not every image maps to a unique set of "
                "plate, well and field plus a channel.\n\n{why}\n\nChange the "
                "regex?", why=text)
            if _headless() or QMessageBox.question(
                    self, tr("Change the regex?"), question) != QMessageBox.Yes:
                return
            if self.open_regex_window():
                self._on_apply()
            return
        if plan.convertible and self._ask_to_convert(plan):
            plan = self.prepare_plan(convert=True)
        if not plan.ok:
            if not _headless():
                QMessageBox.warning(self, tr("Cannot sort"), plan.summary())
            self.status.setText(plan.summary())
            return
        if not _headless() and QMessageBox.question(
                self, tr("Move and merge?"), plan.summary()) != QMessageBox.Yes:
            return
        self.plan = plan
        self.accept()
