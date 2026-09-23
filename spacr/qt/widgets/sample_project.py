"""Start a sample project of the kind you have, from the Home screen.

GitHub #130, 2026-09-22: "I would like to open a sample project that is
similar to a project I may have and see the platform in action." Home shows
every module and says nothing about which one to press first; this asks what
kind of experiment the person has and starts it -- the first module of that
pathway, opened with its example data already downloaded and its settings
filled in.

THE PATHWAYS COME FROM THE SHARED MAP when it is there:
``spacr/resources/module_workflows.json``, the one source of truth item 472
builds for the tutorials, the walkthroughs and the API pages alike, so what
this dialog offers and what the tutorials teach cannot drift apart. Until that
file exists the three pathways below are used, which are the ones the
maintainer named. The selected route remains available from a walkthrough
button in the status bar after its example opens.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QListWidget, QListWidgetItem,
    QToolButton, QVBoxLayout,
)

from ..i18n import tr

__all__ = ["MAP_FILE", "pathways", "SampleProjectDialog", "start_example"]

LOG = logging.getLogger(__name__)

#: The shared map of what feeds what (item 472, part 0).
MAP_FILE = (Path(__file__).resolve().parent.parent.parent / "resources"
            / "module_workflows.json")

#: Used while :data:`MAP_FILE` does not exist. ``modules`` is the pathway in
#: order; the first is what the sample project opens.
FALLBACK = (
    {"id": "spacr_screen",
     "title": "A pooled CRISPR screen (a spaCR screen)",
     "summary": "Segment the cells, measure them, label a few by hand, train "
                "a classifier, read the barcodes, and regress the phenotype "
                "on the guides to get a ranked hit list.",
     "modules": ["mask", "measure", "annotate", "classify_merged",
                 "map_barcodes", "regression"]},
    {"id": "high_content",
     "title": "A high-content imaging experiment (no sequencing)",
     "summary": "Segment and measure every cell, then explore: classify "
                "phenotypes, map the cells, gate populations, plot the "
                "result.",
     "modules": ["mask", "measure", "classify_merged", "umap"]},
    {"id": "train_model",
     "title": "Train a segmentation model of my own",
     "summary": "Correct masks by hand on a few fields, then train a Cellpose "
                "model on them and use it for the rest of the plate.",
     "modules": ["make_masks", "mask"]},
)


def pathways(map_file: Optional[Path] = None) -> List[Dict]:
    """The pathways to offer, from the shared map or the fallback.

    :param map_file: the map; :data:`MAP_FILE` when None.
    :returns: dicts with ``id``, ``title``, ``summary`` and ``modules``.
    """
    path = Path(map_file) if map_file is not None else MAP_FILE
    try:
        written = json.loads(path.read_text(encoding="utf-8"))
        found = written.get("pathways") if isinstance(written, dict) else written
        if isinstance(found, dict):
            modules = written["modules"]
            return [
                {"id": key, "title": route["title"],
                 "summary": route["steps"][0]["action"],
                 "modules": [step["module"] for step in route["steps"]],
                 "module_names": [modules[step["module"]]["name"]
                                  for step in route["steps"]],
                 "home_app": route["home_app"], "walkthrough": True}
                for key, route in found.items()
                if route.get("steps") and route.get("title")
            ]
        out = [dict(entry) for entry in found or []
               if entry.get("modules") and entry.get("title")]
        if out:
            return out
    except (OSError, ValueError, AttributeError, TypeError, KeyError):
        LOG.debug("no usable workflow map at %s", path, exc_info=True)
    return [dict(entry) for entry in FALLBACK]


def start_example(screen) -> str:
    """Load the example data of the module ``screen`` shows.

    Each module already knows how to fetch its own example -- Annotate asks
    which half it needs and OPS has its sequencing-cycle sample -- so this
    calls what the module has rather than a second downloader.

    :param screen: the module screen just opened.
    :returns: what was started: ``"chooser"``, ``"test data"`` or ``""``.
    """
    if getattr(screen, "app_key", None) == "ops":
        screen.load_the_ops_example()
        return "test data"
    chooser = getattr(screen, "_choose_the_test_data", None)
    if callable(chooser):
        chooser()
        return "chooser"
    if getattr(screen, "_test_data_apply", None) is not None:
        from .measurements_example import load_test_data

        load_test_data(screen)
        return "test data"
    return ""


class SampleProjectDialog(QDialog):
    """Which kind of experiment, and start it.

    :param parent: the window it belongs to.
    :param entries: the pathways; :func:`pathways` when None.
    """

    chosen = Signal(str)

    def __init__(self, parent=None, entries: Optional[List[Dict]] = None):
        """Show one row per pathway, the first selected."""
        super().__init__(parent)
        self.setObjectName("SampleProjectDialog")
        self.setWindowTitle(tr("Start a sample project"))
        self.resize(640, 520)
        self._entries = list(entries if entries is not None else pathways())
        layout = QVBoxLayout(self)
        intro = QLabel(tr(
            "Pick the kind of experiment you have. spaCR opens the first "
            "module of that pathway with example data, so you can see it "
            "work before pointing it at your own images."), self)
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.list = QListWidget(self)
        self.list.setObjectName("SampleProjectList")
        self.list.setWordWrap(True)
        self.list.setTextElideMode(Qt.ElideNone)
        for entry in self._entries:
            item = QListWidgetItem(tr(entry["title"]), self.list)
            item.setData(Qt.UserRole, entry)
        self.list.setCurrentRow(0)
        self.list.itemDoubleClicked.connect(lambda _item: self.accept())
        layout.addWidget(self.list, 1)
        self.summary = QLabel("", self)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.steps = QLabel("", self)
        self.steps.setWordWrap(True)
        layout.addWidget(self.steps)
        self.list.currentRowChanged.connect(self._say_the_steps)
        self._say_the_steps(0)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                                   self)
        buttons.button(QDialogButtonBox.Ok).setText(tr("Start"))
        buttons.button(QDialogButtonBox.Cancel).setText(tr("Cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _say_the_steps(self, row: int) -> None:
        """Name the modules the selected pathway goes through, in order."""
        entry = self.selected()
        self.summary.setText(tr(entry.get("summary", "")) if entry else "")
        names = ([tr(name) for name in entry["module_names"]]
                 if entry and entry.get("module_names")
                 else self._module_names(entry.get("modules", ()) if entry else ()))
        self.steps.setText(
            tr("Modules, in order: {names}", names=" → ".join(names))
            if names else "")

    @staticmethod
    def _module_names(keys) -> List[str]:
        """The titles Home shows for these module keys."""
        try:
            from ..screens.app_screen import APP_TITLES
        except Exception:                                     # noqa: BLE001
            APP_TITLES = {}
        return [tr(APP_TITLES.get(key, key)) for key in keys]

    def selected(self) -> Optional[Dict]:
        """The pathway highlighted now, or None."""
        item = self.list.currentItem()
        return item.data(Qt.UserRole) if item is not None else None

    def accept(self) -> None:
        """Announce the chosen pathway's first module and close."""
        entry = self.selected()
        super().accept()
        if entry and entry.get("modules"):
            self.chosen.emit(str(entry["modules"][0]))


def _offer_pathway_walkthrough(window, entry) -> None:
    """Keep the chosen route one click away without starting any analysis.

    Fit the status bar again after Qt propagates the inherited font to the
    new button. Its initial size hint can still use the previous font size.
    """
    if window is None or not entry.get("walkthrough"):
        return
    from ..walkthrough import show_walkthrough

    status = window.statusBar()
    button = getattr(window, "_sample_pathway_button", None)
    if button is None:
        button = QToolButton(status)
        button.setObjectName("SamplePathwayWalkthrough")
        button.setAutoRaise(True)
        button.clicked.connect(lambda _checked=False: show_walkthrough(
            window, "pathway:" + button.property("workflowPathway")))
        status.addPermanentWidget(button)
        window._sample_pathway_button = button
    button.setProperty("workflowPathway", entry["id"])
    button.setText(tr("Walkthrough"))
    button.setToolTip(tr(entry["title"]))
    button.ensurePolished()
    button.show()
    status.setFixedHeight(max(status.height(), status.sizeHint().height()))
    QTimer.singleShot(0, status, lambda: status.setFixedHeight(
        max(status.height(), status.sizeHint().height())))


def offer_a_sample_project(window, opener: Callable[[str], object],
                           entries: Optional[List[Dict]] = None) -> str:
    """Ask which pathway, open its first module, and load its example.

    :param window: the main window, the dialog's parent.
    :param opener: ``fn(module key) -> the screen``, normally the window's
        own navigation.
    :param entries: the pathways, for tests.
    :returns: the module key opened, or ``""`` when nothing was chosen.
    """
    dialog = SampleProjectDialog(window, entries)
    if dialog.exec() != QDialog.Accepted:
        return ""
    entry = dialog.selected() or {}
    keys = list(entry.get("modules") or ())
    if not keys:
        return ""
    key = keys[0]
    screen = opener(key)
    if screen is not None:
        _offer_pathway_walkthrough(window, entry)
        start_example(screen)
    return key
