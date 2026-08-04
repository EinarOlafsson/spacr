"""``A18`` — the Napari Bridge screen: correct a mask over there, keep it here.

A thin surface over :mod:`spacr.napari_bridge`, which owns every decision
that matters: what is handed over, what is refused on the way back, how the
corrected mask is written, and how the correction is recorded. This file
collects two paths and presses three buttons.

Three things about it are deliberate rather than incidental.

**It never imports napari at module scope, and neither does anything it
imports.** ``spacr.qt`` walks its own package in the perf guard and every
settings panel in spaCR is built from modules in it; a second Qt stack
arriving because someone opened an unrelated screen would be paid for by
every user, installed or not. The import lives inside the button handler,
behind :func:`spacr.napari_bridge.require_napari`, and a missing install
produces the ``pip install "spacr[napari]"`` paragraph in the status pane
rather than a traceback.

**It does not run napari's event loop.** spaCR is already a running
``QApplication``; starting a second loop nests them. So the viewer is opened
and left open, and the user presses *Take the mask back* when they are done —
which is also the friendlier interaction, because it lets them look, edit,
take it back, and keep going. :func:`spacr.napari_bridge.run_event_loop` is
for scripts, and its docstring says so.

**A correction made here is recorded exactly as one made in Curate.** Same
append-only ledger, same sidecar, same answer from
:func:`spacr.curation.is_curated`. The bridge is for people who prefer
napari's brush; it is not a way to edit data off the record.

:func:`register` is **not** called at import; read its docstring.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from ..theme import SPACING

LOG = logging.getLogger("spacr.qt.screens.napari_bridge")

__all__ = ["NapariBridgeScreen", "make_napari_bridge_screen", "register",
           "APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO",
           "APP_CLI_NOTE", "APP_NAME_TRANSLATIONS"]

#: The registry key. Chosen once and never renamed.
APP_KEY = "napari_bridge"

_MASK_FILTER = "Masks (*.tif *.tiff *.npy *.png);;All files (*)"
_IMAGE_FILTER = "Images (*.tif *.tiff *.npy *.png *.jpg);;All files (*)"


class NapariBridgeScreen(QWidget):
    """Pick a field, open it in napari, take the corrected mask back."""

    #: A field was opened in napari. Carries the mask path.
    opened = Signal(str)
    #: A correction came back and was written. Carries the mask path.
    corrected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("NapariBridge")
        self._viewer: Any = None
        self._handoff: Any = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["sm"])

        title = QLabel("Napari Bridge", self)
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)
        intro = QLabel(
            "Send a field's image and mask to napari, correct the mask with "
            "the tools you already know, and bring it back. The corrected "
            "mask is written the way spaCR writes masks, and the correction "
            "is recorded in the same ledger the Curate screen writes — so a "
            "curated dataset still says so.", self)
        intro.setObjectName("Muted")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        self._mask_edit = QLineEdit(self)
        self._mask_edit.setPlaceholderText("Label mask (.tif, .npy)")
        outer.addLayout(self._path_row("Mask", self._mask_edit,
                                       self._choose_mask))
        self._image_edit = QLineEdit(self)
        self._image_edit.setPlaceholderText(
            "Image to show underneath (optional)")
        outer.addLayout(self._path_row("Image", self._image_edit,
                                       self._choose_image))

        buttons = QHBoxLayout()
        buttons.setSpacing(SPACING["sm"])
        self.open_button = QPushButton("Open in napari", self)
        self.open_button.setToolTip(
            "Open the field in a napari window. spaCR stays running.")
        self.open_button.clicked.connect(self.open_in_napari)
        buttons.addWidget(self.open_button)
        self.take_button = QPushButton("Take the mask back", self)
        self.take_button.setToolTip(
            "Read the corrected labels out of napari, write them back and "
            "record the correction")
        self.take_button.setEnabled(False)
        self.take_button.clicked.connect(self.take_mask_back)
        buttons.addWidget(self.take_button)
        self.close_button = QPushButton("Close viewer", self)
        self.close_button.setEnabled(False)
        self.close_button.clicked.connect(self.close_viewer)
        buttons.addWidget(self.close_button)
        buttons.addStretch(1)
        outer.addLayout(buttons)

        self.status = QPlainTextEdit(self)
        self.status.setObjectName("NapariBridgeStatus")
        self.status.setReadOnly(True)
        self.status.setPlaceholderText(
            "Choose a mask and press Open in napari.")
        outer.addWidget(self.status, 1)
        # Drop anywhere on this screen: the path is resolved through spaCR's
        # project layout, so the plate folder finds what this screen reads.
        from ..dnd import install_for
        install_for(self, "napari_bridge")

    # -- the form -----------------------------------------------------------
    def _path_row(self, label: str, edit: QLineEdit, chooser) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(SPACING["sm"])
        caption = QLabel(label, self)
        caption.setMinimumWidth(56)
        row.addWidget(caption)
        row.addWidget(edit, 1)
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(chooser)
        row.addWidget(browse)
        return row

    def _choose_mask(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a label mask", self._mask_edit.text().strip(),
            _MASK_FILTER)
        if path:
            self._mask_edit.setText(path)
            self.describe_mask(path)

    def _choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open the image underneath",
            self._image_edit.text().strip(), _IMAGE_FILTER)
        if path:
            self._image_edit.setText(path)

    def set_paths(self, mask: str = "", image: str = "") -> None:
        """Fill the form. The seam a host or a test drives the screen through."""
        if mask:
            self._mask_edit.setText(str(mask))
        if image:
            self._image_edit.setText(str(image))

    def mask_path(self) -> str:
        """The mask path in the form."""
        return self._mask_edit.text().strip()

    def image_path(self) -> str:
        """The image path in the form, or ``""``."""
        return self._image_edit.text().strip()

    # -- saying things ------------------------------------------------------
    def say(self, text: str, *, append: bool = False) -> str:
        """Put ``text`` in the status pane. Returns what the pane now holds.

        A pane rather than a one-line label, because the two things this
        screen most often has to say are a multi-line install instruction and
        a multi-line refusal, and both are written to be read.
        """
        text = str(text)
        if append and self.status.toPlainText():
            self.status.setPlainText(
                f"{self.status.toPlainText()}\n\n{text}")
        else:
            self.status.setPlainText(text)
        return self.status.toPlainText()

    def describe_mask(self, path: str = "") -> str:
        """Say what is in a mask file, and whether it has been edited before."""
        path = path or self.mask_path()
        if not path or not os.path.isfile(path):
            return self.say("Choose a mask file first.")
        try:
            from ...napari_bridge import load_handoff
            handoff = load_handoff(path, self.image_path())
        except Exception as exc:
            return self.say(f"Could not read {os.path.basename(path)}: {exc}")
        return self.say(handoff.describe())

    # -- the bridge ---------------------------------------------------------
    def open_in_napari(self) -> Any:
        """Hand the field to napari. Returns the viewer, or None.

        The napari import happens here and nowhere earlier — see the module
        docstring. A missing install is answered with the install
        instruction, not a traceback.
        """
        path = self.mask_path()
        if not path or not os.path.isfile(path):
            self.say("Choose a mask file first.")
            return None
        try:
            from ...napari_bridge import (NapariExtraMissing, load_handoff,
                                          open_in_napari)
        except ImportError as exc:            # pragma: no cover - broken tree
            self.say(f"Could not load the napari bridge: {exc}")
            return None
        try:
            handoff = load_handoff(path, self.image_path())
        except Exception as exc:
            self.say(f"Could not read {os.path.basename(path)}: {exc}")
            return None
        try:
            viewer = open_in_napari(handoff)
        except NapariExtraMissing as exc:
            # The one refusal that is an instruction rather than an error.
            self.say(str(exc))
            return None
        except Exception as exc:
            LOG.exception("could not open napari")
            self.say(f"napari could not open this field: {exc}")
            return None
        self._viewer = viewer
        self._handoff = handoff
        self.take_button.setEnabled(True)
        self.close_button.setEnabled(True)
        self.say(f"{handoff.describe()}\n\nOpen in napari. Correct the mask "
                 f"there, then come back and press Take the mask back. "
                 f"spaCR is still running — nothing is written until you do.")
        self.opened.emit(handoff.mask_path)
        return viewer

    def take_mask_back(self):
        """Read the corrected labels back, write them, record the correction.

        Returns the :class:`spacr.napari_bridge.CorrectionResult`, or None
        when there was nothing to take.
        """
        if self._viewer is None or self._handoff is None:
            self.say("Open a field in napari first.")
            return None
        from ...napari_bridge import labels_from_viewer, write_back

        try:
            corrected = labels_from_viewer(self._viewer,
                                           name=self._handoff.name)
        except Exception as exc:
            # Every refusal `to_spacr_mask` raises is written to be read by
            # the person who has to act on it, so it is shown verbatim rather
            # than replaced with a house apology.
            self.say(str(exc))
            return None
        try:
            result = write_back(self._handoff.mask_path, corrected,
                                original=self._handoff.mask)
        except Exception as exc:
            self.say(str(exc))
            return None
        self.say(result.describe(), append=False)
        if result.written:
            # The handoff now holds what is on disk, so pressing the button
            # twice reports "unchanged" rather than recording the same edit
            # a second time.
            self._handoff = self._reloaded(result)
            self.corrected.emit(result.mask_path)
        return result

    def _reloaded(self, result) -> Any:
        """The handoff, with the mask that was just written."""
        import dataclasses

        return dataclasses.replace(self._handoff, mask=result.mask)

    def close_viewer(self) -> None:
        """Close the napari window, if one is open."""
        viewer = self._viewer
        self._viewer = None
        self._handoff = None
        self.take_button.setEnabled(False)
        self.close_button.setEnabled(False)
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                LOG.debug("napari viewer would not close", exc_info=True)

    def viewer(self) -> Any:
        """The open napari viewer, or None. For a host and for tests."""
        return self._viewer

    def closeEvent(self, event):  # noqa: N802 - Qt name
        self.close_viewer()
        super().closeEvent(event)


def make_napari_bridge_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return NapariBridgeScreen()


APP_NAME = "Napari Bridge"
APP_DESCRIPTION = "Correct a mask in napari and bring the corrected labels back"
APP_INTRO = (
    "Hand a field's image and its label mask to napari, fix the mask with "
    "the brush, the fill and the keybindings you already have in your "
    "fingers, and bring it back. Label values survive exactly — object 41 "
    "comes back as object 41, on the same pixels — and the corrected mask is "
    "written as the uint16 label image the rest of spaCR reads, in the "
    "orientation it was handed over in. The correction is appended to the "
    "same curation ledger the Curate screen writes, so a hand-edited mask "
    "can still be told from a segmented one months later. spaCR's own brush "
    "is not going anywhere; this is for people who would rather not learn a "
    "second one. Needs `pip install \"spacr[napari]\"`.")
APP_CLI_NOTE = (
    "The Napari Bridge is two windows talking to each other, so run it in "
    "the GUI (spacr-qt). Headless, spacr.napari_bridge.correct_mask(mask, "
    "image) does the whole round trip from a script or a notebook — it opens "
    "napari, waits for you to close it, writes the corrected mask back and "
    "records the correction.")
#: The display name in the nine non-English UI languages, in
#: `spacr.qt.i18n.LANGUAGES` order (sv, de, es, zh_CN, pt, hi, ko, is, fr).
APP_NAME_TRANSLATIONS = (
    "Napari-brygga", "Napari-Brücke", "Puente con napari",
    "napari 桥接", "Ponte para o napari", "नैपारी ब्रिज", "나파리 브리지",
    "Napari-brú", "Passerelle napari")


def register() -> bool:
    """Put the Napari Bridge in the app registry. Idempotent.

    Called from :data:`spacr.qt.SELF_REGISTERING_MODULES`, which
    :func:`spacr.qt.run` runs after ``spacr.qt.app`` is fully executed and
    before ``MainWindow.__init__`` reads the registry.

    Filed under Segmentation models, beside Make Masks: this is where a mask
    that came out wrong gets fixed.

    Everything after the section is a table this key would otherwise need a
    hand-edit in: the screen header and blurb, the "no headless run"
    sentence, the API doc link and the nine translations of the display name.
    :func:`spacr.qt.app.register_app` distributes them from this one call.

    :returns: ``True`` if this call is what registered it. Safe to call
        again: a module imported twice, or a test that re-imports it, must
        not raise on the duplicate key.
    """
    from ..app import APPS, SECTION_MODELS, STAGE_ALPHA, register_app
    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_MODELS,
                 factory=make_napari_bridge_screen, stage=STAGE_ALPHA,
                 intro=APP_INTRO, cli_note=APP_CLI_NOTE,
                 api_module="napari_bridge",
                 translations=APP_NAME_TRANSLATIONS)
    return True
