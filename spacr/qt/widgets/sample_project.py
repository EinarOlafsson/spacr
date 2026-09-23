"""Explore pipeline flowcharts and optionally start an example from Home.

The dialog opens the first module in the selected pathway with example data
and settings supplied. A status-bar walkthrough button keeps the route
available after the example opens.

Pathways come from ``spacr/resources/module_workflows.json``, shared with
tutorials, walkthroughs and API pages. Three fallback pathways are available
if the bundled map is missing.
"""
from __future__ import annotations

import json
from copy import deepcopy
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QListWidget, QListWidgetItem,
    QAbstractItemView, QStyledItemDelegate, QStyle, QToolButton, QVBoxLayout, QWidget,
)

from ..i18n import tr
from ..theme import active_palette, font_px
from .workflow_diagram import DiagramDialog, WorkflowView, diagram_splitter, workflow_map
from .pipeline_details import PipelineDetails

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
                 "description": route.get("description", route["steps"][0]["action"]),
                 "inputs": route.get("inputs", []),
                 "modules": [step["module"] for step in route["steps"]],
                 "module_names": [modules[step["module"]]["name"]
                                  for step in route["steps"]],
                 "home_app": route["home_app"], "walkthrough": True,
                 "steps": route["steps"], "note": route.get("note", ""),
                 "graph_data": written}
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


def pathway_graph(entry, data, steps):
    """Add external inputs to a display copy without changing runnable steps.

    :param entry: pathway metadata with optional external input records.
    :param data: shared module/artifact map, left unchanged.
    :param steps: runnable module dependencies, left unchanged.
    :returns: display map, node keys and dependencies including input nodes.
    """
    display = dict(data, modules=dict(data["modules"]),
                   connections=list(data.get("connections", [])))
    steps = deepcopy(steps)
    keys = list(entry["modules"])
    for source in entry.get("inputs", []):
        key = "input:" + source["id"]
        keys.append(key)
        display["modules"][key] = {"name": source["title"], "inputs": [],
                                   "outputs": source["artifacts"],
                                   "guidance": source["description"]}
        steps.append({"module": key, "after": []})
        for target in source["targets"]:
            if target not in entry["modules"]:
                continue
            next(step for step in steps if step["module"] == target).setdefault("after", []).append(key)
            display["connections"].append({"from": key, "to": target,
                                           "artifacts": source["artifacts"],
                                           "handoff": source["description"]})
    return display, keys, steps


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


class _PipelineRowDelegate(QStyledItemDelegate):
    """Draw a restrained selection rim behind each flowchart row."""

    def paint(self, painter, option, index):
        """Leave text to the row widget instead of painting a duplicate title."""
        if option.state & QStyle.State_Selected:
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(QColor(active_palette()['accent']), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(option.rect.adjusted(1, 1, -1, -1), 12, 12)
            painter.restore()


class SampleProjectDialog(DiagramDialog):
    """Explore pathway flowcharts and optionally start an example project.

    :param parent: the window it belongs to.
    :param entries: the pathways; :func:`pathways` when None.
    """

    chosen = Signal(str)

    def __init__(self, parent=None, entries: Optional[List[Dict]] = None):
        """Show a module/input/output flowchart for each pathway row."""
        super().__init__(parent)
        self.setObjectName("SampleProjectDialog")
        self.setWindowTitle(tr("Pipeline overviews"))
        self.resize(1280, 1000)
        self._entries = list(entries if entries is not None else pathways())
        layout = QVBoxLayout(self)
        intro = QLabel(tr(
            "Explore the modules, inputs and outputs in each pipeline. "
            "Hover a module or arrow for details. Select a pipeline and "
            "choose Start example to open its first module with sample data. "
            "Ctrl+wheel zooms a diagram; drag to pan."), self)
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.list = QListWidget(self)
        self.list.setObjectName("SampleProjectList")
        self.list.setWordWrap(True)
        self.list.setTextElideMode(Qt.ElideNone)
        self.list.setItemDelegate(_PipelineRowDelegate(self.list))
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.list.setStyleSheet("QListWidget#SampleProjectList { background: transparent; border: none; }")
        self.diagrams = []
        for index, entry in enumerate(self._entries):
            item = QListWidgetItem(tr(entry["title"]), self.list)
            item.setData(Qt.UserRole, entry)
            row = QWidget()
            row_layout = QVBoxLayout(row)
            title = QLabel(tr(entry["title"]))
            title.setObjectName("CardTitle")
            row_layout.addWidget(title)
            data = entry.get("graph_data")
            if data is None:
                try:
                    data = workflow_map()
                except (OSError, ValueError):
                    data = {"modules": {}, "artifacts": {}, "connections": []}
                for key in entry["modules"]:
                    data["modules"].setdefault(key, {"name": key, "inputs": [], "outputs": []})
            steps = entry.get("steps")
            if steps is None:
                steps = [{"module": key, "after": entry["modules"][i-1:i] if i else []}
                         for i, key in enumerate(entry["modules"])]
            data, keys, steps = pathway_graph(entry, data, steps)
            diagram = WorkflowView(data, keys, steps, row, compact=True)
            diagram.activated.connect(lambda i=index: self.list.setCurrentRow(i))
            diagram.selection_changed.connect(self._describe_graph_item)
            row_layout.addWidget(diagram, 1)
            self.diagrams.append(diagram)
            item.setSizeHint(QSize(900, 360))
            self.list.setItemWidget(item, row)
        self.list.setCurrentRow(0)
        self.splitter = diagram_splitter(self)
        self.splitter.addWidget(self.list)
        layout.addWidget(self.splitter, 1)
        self.summary = QLabel("", self)
        self.summary.setWordWrap(True)
        self.summary.setFixedHeight(font_px("body") * 3)
        self.summary.hide()
        self.steps = QLabel("", self)
        self.steps.setWordWrap(True)
        self.steps.hide()
        self.details = PipelineDetails(self)
        self.splitter.addWidget(self.details)
        self.splitter.setSizes([380, 520])
        self.list.currentRowChanged.connect(self._say_the_steps)
        self._say_the_steps(0)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                                   self)
        buttons.button(QDialogButtonBox.Ok).setText(tr("Start example"))
        buttons.button(QDialogButtonBox.Ok).setEnabled(bool(self._entries))
        buttons.button(QDialogButtonBox.Cancel).setText(tr("Cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _describe_graph_item(self, identifier):
        """Outline the matching persistent card without changing row geometry."""
        self.details.select_element(identifier)

    def _say_the_steps(self, row: int) -> None:
        """Name the modules the selected pathway goes through, in order."""
        entry = self.selected()
        if entry and 0 <= row < len(self.diagrams):
            self.details.set_pipeline(entry, self.diagrams[row])
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
