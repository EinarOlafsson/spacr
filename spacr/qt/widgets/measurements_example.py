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

THE DOWNLOAD IS THE EXISTING ONE. The archive, the worker and the progress
dialog are :func:`spacr.qt.hf_download.download_annotate_example`, the same
call Classify's button makes, and a plate already unpacked by any of them is
reused without touching the network.

Two public sets have buttons of their own, fetched through the example-set
registry (:data:`spacr.example_archives.EXAMPLE_SETS`): ``dose`` (LINCS Cell
Painting, ``einarolafsson/spacr-example-dose``) for Dose-Response, Prediction
Profiler, Run Compare, Run History and Training Runs, and ``control_chart``
(CPJUMP1, ``einarolafsson/spacr-example-control-chart``) for Control Chart.
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


_DOSE_KEY = "dose"
_DOSE_FOLDER = "dose_response_lincs"
_DOSE_PLATE = "dose_plate.csv"
_DOSE_RESPONSE = "Cells_Number_Object_Number"
_DOSE_PROFILER_RUN = "regression_dmso_normalised"
_DOSE_JOURNAL_MARK = ".journalled_runs"


def _dose_example_folder() -> Path:
    """The dose example's folder, beside the shared example plate.

    Four replicate plates of the LINCS Cell Painting set (cpg0004, CC0 1.0):
    56 compounds at six doses with DMSO wells, well-level profiles cut to
    eleven features, plus two regression runs and two training runs made
    from them.
    """
    return example_measurements_folder().parent / _DOSE_FOLDER


def _fetch_example_set(key: str, folder: Path) -> None:
    """Download the registered example set ``key`` and unpack it in ``folder``.

    The set's one archive comes from its dataset repository through
    :func:`spacr.example_archives.download_archive`, is unpacked with the
    path filter of :func:`spacr.example_archives.extract_example_archive`,
    and is then deleted.

    :param key: an :data:`spacr.example_archives.EXAMPLE_SETS` key.
    :param folder: where the set is unpacked.
    """
    from ...example_archives import (download_archive, example_set,
                                     extract_example_archive)

    entry = example_set(key)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    archive = download_archive(entry.repo, entry.archive, folder,
                               chunk_size=1 << 20)
    try:
        extract_example_archive(archive, folder)
    finally:
        Path(archive).unlink(missing_ok=True)


def _install_dose_test_data_button(screen, layout, apply: Callable[[Path], Any],
                                   *, say: Optional[Callable[[str], Any]] = None,
                                   index: Optional[int] = None):
    """Add a "Load test data…" button that hands the dose example to ``apply``.

    :param screen: the screen the button, callback and reporter are kept on.
    :param layout: the box layout the button goes into.
    :param apply: called as ``apply(folder)`` with the dose example folder.
    :param say: where a failure is reported; logged when omitted.
    :param index: position in ``layout``; appended when omitted.
    :returns: the button. The caller names it.
    """
    from PySide6.QtWidgets import QPushButton

    button = QPushButton(tr("Load test data…"), screen)
    button.setToolTip(tr(
        "Load a public dose plate: four replicate A549 plates of the LINCS "
        "Cell Painting set (CC0), 56 compounds at six doses with DMSO wells, "
        "as well-level profiles, with regression and training runs made "
        "from them. Under 1 MB, cached afterwards."))
    screen._dose_test_data_apply = apply
    screen._test_data_say = say
    button.clicked.connect(lambda _checked=False: _load_dose_test_data(screen))
    if layout is not None:
        if index is None:
            layout.addWidget(button)
        else:
            layout.insertWidget(index, button)
    return button


def _fetch_dose_example(folder: Path) -> None:
    """Download the dose example (``einarolafsson/spacr-example-dose``)."""
    _fetch_example_set(_DOSE_KEY, folder)


def _load_dose_test_data(screen, *, ask=None) -> bool:
    """Reuse or fetch the dose example, then hand it to the screen.

    :param screen: a screen :func:`_install_dose_test_data_button` was
        called on.
    :param ask: replaces the download, called with the folder. For tests.
    :returns: whether the screen received the example.
    """
    folder = _dose_example_folder()
    if not (folder / _DOSE_PLATE).is_file():
        try:
            (ask or _fetch_dose_example)(folder)
        except Exception as exc:
            _report(screen, _download_failure(exc))
            return False
    if not (folder / _DOSE_PLATE).is_file():
        _report(screen, tr("The test data could not be downloaded: {detail}",
                           detail=str(folder / _DOSE_PLATE)))
        return False
    apply = getattr(screen, "_dose_test_data_apply", None)
    try:
        if apply is not None:
            apply(folder)
    except Exception as exc:
        LOG.exception("the screen could not open the dose example")
        _report(screen, str(exc) or exc.__class__.__name__)
        return False
    return True


def _dose_runs(folder: Path):
    """``[(run name, results.csv, settings)]`` for the example's recorded runs."""
    import json

    runs = []
    for run in sorted((Path(folder) / "runs").glob("*/results.csv")):
        settings_file = run.parent / "settings.json"
        settings = (json.loads(settings_file.read_text(encoding="utf-8"))
                    if settings_file.is_file() else {})
        settings["src"] = str(Path(folder) / _DOSE_PLATE)
        settings["dst"] = str(run.parent)
        runs.append((run.parent.name, run, settings))
    return runs


def _register_dose_runs(folder: Path) -> int:
    """Put the example's regression runs in its project's artifact registry.

    Registering the same file again updates its row, so this is safe to
    repeat.

    :returns: how many runs were registered.
    """
    from ... import ports
    from ...artifacts import Registry

    registry = Registry(project=str(folder))
    runs = _dose_runs(folder)
    for name, results, settings in runs:
        registry.register(module="regression", kind=ports.REGRESSION_RESULTS,
                          path=str(results), settings=settings,
                          run_id=f"dose-example-{name}")
    return len(runs)


def _journal_dose_runs(folder: Path) -> int:
    """Write the example's regression runs into this computer's run journal.

    Done once: the journal folders written are remembered beside the example
    and nothing is written while they still exist.

    :returns: how many runs were journalled now.
    """
    from ...run_journal import open_run, runs_root

    mark = Path(folder) / _DOSE_JOURNAL_MARK
    if mark.is_file():
        names = [line for line in mark.read_text(encoding="utf-8").split()
                 if line]
        if names and all((runs_root() / name).is_dir() for name in names):
            return 0
    written = []
    for _name, results, settings in _dose_runs(folder):
        with open_run("regression", settings) as run:
            run.record_input(settings["src"], setting_key="src")
            run.record_output(results, setting_key="dst")
            run.set_status("success")
        written.append(Path(run.dir).name)
    mark.write_text("\n".join(written) + "\n", encoding="utf-8")
    return len(written)


def _download_failure(exc: BaseException) -> str:
    """The message a failed example download is reported with."""
    from ...example_archives import explain_download_failure

    try:
        detail = explain_download_failure(exc)
    except Exception:
        detail = str(exc) or exc.__class__.__name__
    return tr("The test data could not be downloaded: {detail}", detail=detail)


_CONTROL_CHART_KEY = "control_chart"
_CONTROL_CHART_FOLDER = "control_chart_cpjump1"
_CONTROL_CHART_TABLE = "control_chart_wells.csv"
_CONTROL_CHART_VALUE = "Cells_Number_Object_Number"
_CONTROL_CHART_NEGATIVE = "negcon"
_CONTROL_CHART_POSITIVE = "poscon_cp"


def _control_chart_example_folder() -> Path:
    """The Control Chart example's folder, beside the shared example plate.

    Every well of the 24 CPJUMP1 compound plates (Cell Painting Gallery
    cpg0000-jump-pilot, CC0 1.0), cut to plate, well and control metadata and
    eleven CellProfiler features.
    """
    return example_measurements_folder().parent / _CONTROL_CHART_FOLDER


def _install_control_chart_test_data_button(
        screen, layout, apply: Callable[[Path], Any], *,
        say: Optional[Callable[[str], Any]] = None,
        index: Optional[int] = None):
    """Add a "Load test data…" button that hands the CPJUMP1 table to ``apply``.

    :param screen: the screen the button, callback and reporter are kept on.
    :param layout: the box layout the button goes into.
    :param apply: called as ``apply(table)`` with ``control_chart_wells.csv``.
    :param say: where a failure is reported; logged when omitted.
    :param index: position in ``layout``; appended when omitted.
    :returns: the button. The caller names it.
    """
    from PySide6.QtWidgets import QPushButton

    button = QPushButton(tr("Load test data…"), screen)
    button.setToolTip(tr(
        "Load a public plate series: every well of the 24 CPJUMP1 compound "
        "plates of the Cell Painting Gallery (CC0), with DMSO and "
        "positive-control wells marked, and chart the DMSO wells plate by "
        "plate. About 2 MB, cached afterwards."))
    screen._control_chart_test_data_apply = apply
    screen._test_data_say = say
    button.clicked.connect(
        lambda _checked=False: _load_control_chart_test_data(screen))
    if layout is not None:
        if index is None:
            layout.addWidget(button)
        else:
            layout.insertWidget(index, button)
    return button


def _fetch_control_chart_example(folder: Path) -> None:
    """Download the CPJUMP1 table (``einarolafsson/spacr-example-control-chart``)."""
    _fetch_example_set(_CONTROL_CHART_KEY, folder)


def _load_control_chart_test_data(screen, *, ask=None) -> bool:
    """Reuse or fetch the CPJUMP1 table, then hand it to the screen.

    :param screen: a screen :func:`_install_control_chart_test_data_button`
        was called on.
    :param ask: replaces the download, called with the folder. For tests.
    :returns: whether the screen received the table.
    """
    folder = _control_chart_example_folder()
    table = folder / _CONTROL_CHART_TABLE
    if not table.is_file():
        try:
            (ask or _fetch_control_chart_example)(folder)
        except Exception as exc:
            _report(screen, _download_failure(exc))
            return False
    if not table.is_file():
        _report(screen, tr("The test data could not be downloaded: {detail}",
                           detail=str(table)))
        return False
    apply = getattr(screen, "_control_chart_test_data_apply", None)
    try:
        if apply is not None:
            apply(table)
    except Exception as exc:
        LOG.exception("the screen could not open the control chart example")
        _report(screen, str(exc) or exc.__class__.__name__)
        return False
    return True
