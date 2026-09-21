"""Import's "Load test data…": every vendor format and naming, one press away.

A press of "Test Zeiss CZI import" downloads Zeiss CZI files named by ZEN's
split-tiles convention, fills the Import screen with their paths on disk and
previews them, so the user can import them and see each file land on the
well, field and channel it came from. Every other format and filename
convention in :data:`spacr.import_examples.IMPORT_VARIANTS` works the same way.

WHAT THE BUTTON DOES. It opens :class:`ImportTestDataChooser`, one button
per variant in :data:`spacr.import_examples.IMPORT_VARIANTS`, each described
on hover. The chosen one is fetched -- the whole set is one archive, so the
first press downloads every variant and every later press opens from the
cache -- and then:

* an ``import`` variant fills the Import screen itself: the image folder,
  the three mask folders, the measurement table, the naming convention and,
  for Custom, its pattern. A fresh destination is chosen, so a second demo
  never lands in the first one's project, and Preview is pressed. Nothing
  is written until the user presses Import.
* a ``convert`` variant -- the public ND2 and LIF samples, which have no
  masks -- opens the Format Converter page and fills and previews that.

SAME MACHINERY AS EVERY OTHER EXAMPLE SET: :class:`_TarExampleWorker`
streams the archive on a worker thread behind the shared progress dialog and
unpacks it through the tar data filter.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

from .. import import_examples as ix
from .hf_download import (DownloadResult, _TarExampleWorker,
                          download_toxo_mito_demo, explain_download_failure,
                          example_plate_folder)
from .i18n import tr
from .widgets.test_data_chooser import TestDataChooser

LOG = logging.getLogger("spacr.qt.import_demo")

__all__ = [
    "ImportTestDataChooser",
    "apply_variant",
    "choose_import_test_data",
    "download_import_example",
    "load_import_test_data",
]

#: What the worker reports when the user pressed Cancel.
CANCELLED = "Cancelled by user."


class ImportTestDataChooser(TestDataChooser):
    """Every Import test variant, described on hover, in a grid.

    :param parent: parent widget.
    """

    COLUMNS = 3
    DIALOG_WIDTH = 720
    RESTING_TEXT = ("Hover a button to see which microscope format and file "
                    "naming it tests. The first press downloads every variant "
                    "at once, about 285 MB; later presses open from the "
                    "cache. Nothing is written into a project until you "
                    "press Import.")

    def __init__(self, parent=None):
        """Build the chooser from the variant registry."""
        self.ROUTES = tuple((variant.key, variant.label, variant.description)
                            for variant in ix.IMPORT_VARIANTS)
        super().__init__(parent)
        self.setWindowTitle(tr("Load Import test data"))


class _ImportTarWorker(_TarExampleWorker):
    """Fetches Import's test data: every variant, masks and measurements."""

    repo = ix.IMPORT_EXAMPLE_REPO
    archive = ix.IMPORT_EXAMPLE_ARCHIVE

    def dataset_root(self, dest) -> Path:
        """The unpacked ``import_example`` folder inside the plate."""
        return Path(dest) / ix.IMPORT_EXAMPLE_FOLDER


def download_import_example(parent, dest, on_done: Callable[
        [Optional[DownloadResult], str], None]) -> None:
    """Fetch Import's test data behind the shared progress dialog.

    :param parent: the widget the progress dialog belongs to.
    :param dest: the example plate folder the archive unpacks into.
    :param on_done: called on the GUI thread as ``on_done(result, error)``.
    """
    download_toxo_mito_demo(
        parent, Path(dest), on_done,
        worker_factory=_ImportTarWorker,
        title=tr("Downloading the Import test data"))


def choose_import_test_data(screen, *, chooser=None, ask=None,
                            plate=None) -> bool:
    """Show the chooser, then load whatever was picked.

    :param screen: the Import screen.
    :param chooser: replaces :class:`ImportTestDataChooser`, for tests -- a
        headless run cannot show a modal dialog.
    :param ask: passed to :func:`load_import_test_data`.
    :param plate: passed to :func:`load_import_test_data`.
    :returns: whether a variant was opened by this call.
    """
    dialog = chooser if chooser is not None else ImportTestDataChooser(screen)
    if not dialog.exec() or not getattr(dialog, "chosen", ""):
        return False
    return load_import_test_data(screen, dialog.chosen, ask=ask, plate=plate)


def load_import_test_data(screen, key: str, *, ask=None, plate=None) -> bool:
    """Open one variant, downloading the set first when it is not cached.

    :param screen: the Import screen.
    :param key: the variant.
    :param ask: replaces :func:`download_import_example`, for tests. Called
        as ``ask(screen, plate, on_done)``.
    :param plate: replaces the example plate folder, for tests.
    :returns: whether the variant was opened by this call. ``False`` while a
        download is still running, and after a failure.
    """
    plate = Path(plate) if plate is not None else example_plate_folder()
    root = ix.import_example_folder(plate)
    if ix.is_present(root, key):
        return apply_variant(screen, root, key)

    button = getattr(screen, "_btn_test_data", None)
    if button is not None:
        button.setEnabled(False)
        button.setText(tr("Fetching test data…"))
    screen._set_status(tr("Downloading the Import test data"))

    def _done(result, error):
        """Put the button back, then open the variant or say why not."""
        if button is not None:
            button.setEnabled(True)
            button.setText(tr("Load test data…"))
        if result is None:
            if error == CANCELLED:
                screen._set_status(tr("The test data download was "
                                      "cancelled."))
                return
            LOG.info("Import test data not downloaded: %s", error)
            screen._set_status(tr("The test data could not be downloaded: "
                                  "{detail}", detail=error), error=True)
            return
        apply_variant(screen, root, key)

    download = ask if ask is not None else download_import_example
    try:
        download(screen, plate, _done)
    except Exception as exc:                                 # noqa: BLE001
        LOG.warning("the Import test data download did not start",
                    exc_info=True)
        _done(None, explain_download_failure(exc))
    return False


def _fresh_destination(images: str, suffix: str = "_spacr") -> str:
    """A project folder beside the variant that no earlier demo wrote."""
    base = os.path.normpath(images) + suffix
    candidate, number = base, 2
    while os.path.exists(candidate):
        candidate = f"{base}_{number}"
        number += 1
    return candidate


def apply_variant(screen, root, key: str) -> bool:
    """Fill the screen for one unpacked variant and press Preview.

    :param screen: the Import screen.
    :param root: the unpacked ``import_example`` folder.
    :param key: the variant.
    :returns: whether the variant was complete and the preview started.
    """
    root = Path(root)
    if not ix.is_present(root, key):
        screen._set_status(tr("The downloaded test data is incomplete: "
                              "{path}", path=str(root)), error=True)
        return False
    inputs = ix.variant_inputs(root, key)
    if inputs["route"] == "convert":
        return _apply_to_converter(screen, inputs)
    screen.clear_mask_folders()
    screen.set_images(inputs["images"])
    for role, folder in dict(inputs["masks"]).items():
        screen.add_mask_folder(role, folder)
    screen.set_measurements(inputs["measurements"])
    screen.set_metadata_type(inputs["metadata_type"])
    screen.set_custom_regex(inputs["custom_regex"])
    screen.set_destination(_fresh_destination(inputs["images"]))
    return bool(screen.preview())


def _converter_for(screen):
    """The Format Converter page, opened from Import's fold strip.

    Falls back to a converter window of its own when the screen has no fold
    strip -- a screen built outside the main window.
    """
    from .screens.convert import ConvertScreen

    for opener in getattr(screen, "_fold_openers", None) or ():
        if getattr(opener, "key", "") != "convert":
            continue
        shown = opener.open()
        if isinstance(shown, ConvertScreen):
            return shown
        if shown is not None:
            found = shown.findChild(ConvertScreen)
            if found is not None:
                return found
    converter = getattr(screen, "_test_converter", None)
    if converter is None:
        from .screens.map_barcodes import show_as_window

        converter = ConvertScreen()
        show_as_window(converter, screen, tr("Format Converter"))
        screen._test_converter = converter
    return converter


def _apply_to_converter(screen, inputs) -> bool:
    """Fill and preview the Format Converter for an images-only variant."""
    converter = _converter_for(screen)
    if converter is None:
        screen._set_status(tr("The Format Converter could not be opened."),
                           error=True)
        return False
    images = str(inputs["images"])
    converter.set_source(images)
    converter.set_destination(_fresh_destination(images, "_yokogawa"))
    screen._set_status(tr("Opened the Format Converter on {path}",
                          path=images))
    return bool(converter.preview())
