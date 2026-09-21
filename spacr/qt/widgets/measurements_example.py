"""One "Load test data…" button for every screen that reads a measurements DB.

The Annotate example (``einarolafsson/spacr-example-annotate``) is a real
plate: ``measurements/measurements.db`` with its cell, cytoplasm, nucleus and
pathogen tables, and the 2,341 crops under ``data/`` it indexes. Annotate and
Classify already fetch it into the shared example plate folder
(:func:`spacr.example_archives.example_plate_folder`). Every other screen
that opens a measurements database can be shown working on the same plate,
so this module fetches it once and hands it to whichever screen asked.

ONE HELPER, NOT ONE COPY PER SCREEN. The cached-or-download decision, the
button's busy state and the failure message are the same for Plate Viewer,
Tabulate, Graph Builder and the rest; what differs is only which field is
filled and which method opens it. So a screen passes that one difference as
a callback and nothing else.

THE DOWNLOAD IS THE EXISTING ONE. Nothing new is published: the archive,
the worker and the progress dialog are
:func:`spacr.qt.hf_download.download_annotate_example`, the same call
Classify's button makes, and a plate already unpacked by any of them is
reused without touching the network.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..i18n import tr

LOG = logging.getLogger(__name__)

__all__ = [
    "BUTTON_OBJECT_NAME",
    "EXAMPLE_MEASUREMENT",
    "EXAMPLE_TABLE",
    "example_measurements_folder",
    "example_measurements_db",
    "install_test_data_button",
    "load_test_data",
]

BUTTON_OBJECT_NAME = "LoadTestDataButton"

#: The table a single-table screen opens from the example: one row per cell,
#: 2,341 of them across four wells, which is what Tabulate, Graph Builder and
#: Gate Editor are for. The nucleus and pathogen tables stay one pick away.
EXAMPLE_TABLE = "cell"

#: The measurement Plate Viewer draws first: a column every cell table has,
#: that differs between wells and that nobody needs explained.
EXAMPLE_MEASUREMENT = "cell_area"

_DB_RELATIVE = Path("measurements") / "measurements.db"


def example_measurements_folder() -> Path:
    """The shared example plate folder the Annotate example unpacks into."""
    from ...example_archives import example_plate_folder

    return Path(example_plate_folder())


def example_measurements_db() -> Path:
    """The example plate's ``measurements/measurements.db``, present or not."""
    return example_measurements_folder() / _DB_RELATIVE


def install_test_data_button(screen, layout, apply: Callable[[Path, Path], Any],
                             *, say: Optional[Callable[[str], Any]] = None,
                             index: Optional[int] = None):
    """Add the "Load test data…" button to ``layout`` and wire it to ``apply``.

    :param screen: the screen the button belongs to. The button, the callback
        and the reporter are kept on it, so :func:`load_test_data` can be
        called with the screen alone -- which is what a test does.
    :param layout: the box layout the button goes into, normally the row that
        holds the screen's own source field.
    :param apply: called as ``apply(folder, database)`` once the example is on
        disk: the plate folder and its ``measurements/measurements.db``. It
        fills the screen's source field and opens it.
    :param say: where a failed download is reported, usually the screen's
        status label's ``setText``. Logged when omitted.
    :param index: position in ``layout``; appended when omitted.
    :returns: the button.
    """
    from PySide6.QtWidgets import QPushButton

    button = QPushButton(tr("Load test data…"), screen)
    button.setObjectName(BUTTON_OBJECT_NAME)
    button.setToolTip(tr(
        "Download about 280 MB of example data: 2,341 single-cell crops "
        "with a measurements database, of which 88 are already labelled. "
        "Settings are filled in with it, so the module can be run "
        "straight away. Cached afterwards."))
    screen._test_data_button = button
    screen._test_data_apply = apply
    screen._test_data_say = say
    button.clicked.connect(lambda _checked=False: load_test_data(screen))
    if layout is not None:
        if index is None:
            layout.addWidget(button)
        else:
            layout.insertWidget(index, button)
    return button


def load_test_data(screen, *, ask=None) -> Dict[str, str]:
    """Reuse or fetch the example plate, then hand it to the screen.

    :param screen: a screen :func:`install_test_data_button` was called on.
    :param ask: replaces :func:`spacr.qt.hf_download.download_annotate_example`,
        with the same ``(parent, destination, on_done)`` signature. For tests.
    :returns: ``{"folder": ..., "db": ...}`` when the plate was handed over
        before returning -- always, when it was already cached -- and an
        empty mapping while a download is still running or after it failed.
    """
    folder = example_measurements_folder()
    database = folder / _DB_RELATIVE
    if database.is_file():
        return _hand_over(screen, folder, database)

    folder.mkdir(parents=True, exist_ok=True)
    button = getattr(screen, "_test_data_button", None)
    if button is not None:
        button.setEnabled(False)
        button.setText(tr("Fetching test data…"))
    handed: Dict[str, str] = {}

    def _done(result, error) -> None:
        """Restore the button, then hand over the plate or say why not."""
        if button is not None:
            button.setEnabled(True)
            button.setText(tr("Load test data…"))
        if result is None or not database.is_file():
            _report(screen, tr("The test data could not be downloaded: "
                               "{detail}",
                               detail=error or tr("unknown error")))
            return
        handed.update(_hand_over(screen, folder, database))

    download = ask
    if download is None:
        from ..hf_download import download_annotate_example as download
    download(screen, folder, _done)
    return handed


def _hand_over(screen, folder: Path, database: Path) -> Dict[str, str]:
    """Call the screen's callback with the plate; report what it raised."""
    apply = getattr(screen, "_test_data_apply", None)
    if apply is not None:
        try:
            apply(folder, database)
        except Exception as exc:
            LOG.exception("the screen could not open the example plate")
            _report(screen, str(exc) or exc.__class__.__name__)
            return {}
    return {"folder": str(folder), "db": str(database)}


def _report(screen, message: str) -> None:
    """Say ``message`` where the screen asked, or in the log."""
    say = getattr(screen, "_test_data_say", None)
    if say is not None:
        try:
            say(message)
            return
        except Exception:
            LOG.debug("the screen's reporter failed", exc_info=True)
    LOG.warning("%s", message)
