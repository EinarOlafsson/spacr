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
from .. import _after
from copy import deepcopy
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import QEvent, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QTextDocument
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QListWidget, QListWidgetItem,
    QAbstractItemView, QStyledItemDelegate, QStyle, QToolButton, QVBoxLayout, QWidget,
)

from ..i18n import tr
from ..theme import active_palette, font_px
from .workflow_diagram import DiagramDialog, WorkflowView, diagram_splitter, workflow_map
from .pipeline_details import PipelineDetails

__all__ = ["MAP_FILE", "pathways", "pathway_graph", "SampleProjectDialog", "start_example"]

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
                 "map_barcodes", "regression"],
     "steps": [{"module": "mask", "after": []},
               {"module": "measure", "after": ["mask"]},
               {"module": "annotate", "after": ["measure"]},
               {"module": "classify_merged", "after": ["annotate"]},
               {"module": "map_barcodes", "after": []},
               {"module": "regression", "after": ["classify_merged", "map_barcodes"]}]},
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
     "modules": ["make_masks", "train_cellpose", "mask"]},
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


def fallback_workflow_map(keys):
    """Describe fallback modules using their runtime input/output declarations.

    :param keys: modules needed by a fallback pathway.
    :returns: display map with port descriptions, locations and API references.
        GUI-only editing modules use their documented source/mask/database
        contracts. This performs no filesystem scan of user data.
    """
    from ...ports import ModulePorts, Port, UnknownModule, module_ports

    titles = dict(zip(keys, SampleProjectDialog._module_names(keys)))
    api = {"mask": "spacr.core", "measure": "spacr.measure",
           "annotate": "spacr.qt.screens.annotate", "classify_merged": "spacr.classify",
           "map_barcodes": "spacr.sequencing", "regression": "spacr.ml",
           "umap": "spacr.core", "make_masks": "spacr.qt.screens.make_masks",
           "train_cellpose": "spacr.submodules"}
    editors = {
        "annotate": ModulePorts("annotate",
            consumes=(Port("measurements-db", "db", "measurements/measurements.db",
                           description="Object measurements and crop paths"),
                      Port("crops", "crops", "data", description="Single-object image crops")),
            produces=(Port("labels", "labels", "measurements/measurements.db",
                           description="Reviewed object labels in png_list"),),
            summary="Label representative objects before training a phenotype classifier."),
        "make_masks": ModulePorts("make_masks",
            consumes=(Port("raw-images", "images", description="Microscopy images"),
                      Port("masks", "masks", "masks", required=False,
                           description="Existing masks to correct, when available")),
            produces=(Port("curated-pairs", "pairs", description="Reviewed image and integer-mask pairs"),),
            summary="Draw or correct object masks, then train a segmentation model in Cellpose Workbench."),
        "train_cellpose": ModulePorts("train_cellpose",
            consumes=(Port("curated-pairs", "pairs", description="Reviewed image and integer-mask pairs"),),
            produces=(Port("segmentation-model", "model", "models",
                           description="Trained Cellpose segmentation checkpoint"),),
            summary="Train in Cellpose Workbench inside Make Masks, then evaluate on separate fields before applying the checkpoint."),
    }
    data = {"modules": {}, "artifacts": {}, "connections": []}
    for key in keys:
        try:
            contract = editors.get(key) or module_ports(key)
        except UnknownModule:
            contract = ModulePorts(key)
        module = dict(name=titles[key], guidance=contract.summary,
                      api_module=api.get(key, "spacr.ports"), inputs=[], outputs=[])
        for role, ports in (("inputs", contract.consumes), ("outputs", contract.produces)):
            for port in ports:
                artifact = f"{key}:{role}:{port.role}"
                module[role].append(artifact)
                data["artifacts"][artifact] = dict(
                    title=port.description or port.kind, location=port.relative())
        data["modules"][key] = module
    return data


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
                    data = fallback_workflow_map(entry["modules"])
                for key in entry["modules"]:
                    data["modules"].setdefault(key, {"name": key, "inputs": [], "outputs": []})
            steps = entry.get("steps")
            if steps is None:
                steps = [{"module": key, "after": entry["modules"][i-1:i] if i else []}
                         for i, key in enumerate(entry["modules"])]
            data, keys, steps = pathway_graph(entry, data, steps)
            diagram = WorkflowView(data, keys, steps, row)
            diagram.activated.connect(lambda i=index: self.list.setCurrentRow(i))
            diagram.selection_changed.connect(self._describe_graph_item)
            row_layout.addWidget(diagram, 1)
            self.diagrams.append(diagram)
            item.setSizeHint(QSize(1100, 460))
            self.list.setItemWidget(item, row)
            for child in [row, *row.findChildren(QWidget)]:
                child.setProperty('pipelineRow', index)
                child.installEventFilter(self)
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
        self.splitter.setSizes([460, 440])
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

    def eventFilter(self, watched, event):
        """Select a pipeline from any part of its row, preserving pan and clicks.

        :param watched: widget whose optional pipelineRow property identifies a row.
        :param event: Qt event; only a left-button press changes the selection.
        :returns: the base event filter's result, preserving normal event handling.
        """
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            row = watched.property('pipelineRow')
            if row is not None:
                self.list.setCurrentRow(int(row))
        return super().eventFilter(watched, event)

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
    _after(0, status, lambda: status.setFixedHeight(
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
    try:
        if dialog.exec() != QDialog.Accepted:
            return ""
        entry = dialog.selected() or {}
    finally:
        dialog.deleteLater()
    keys = list(entry.get("modules") or ())
    if not keys:
        return ""
    key = keys[0]
    screen = opener(key)
    if screen is not None:
        _offer_pathway_walkthrough(window, entry)
        start_example(screen)
    return key


class _ProtocolDocument(QTextDocument):
    def loadResource(self, kind, url):
        return None


class _ProtocolBrowserDialog(QDialog):
    def __init__(self, parent=None, source_path='', context=None, *, data=None):
        from PySide6.QtWidgets import QListWidget, QPushButton, QTextBrowser

        super().__init__(parent)
        self.setObjectName('ProtocolBrowser')
        self.setWindowTitle(tr('Protocols'))
        self.setModal(False)
        self.resize(1080, 820)
        self.setMinimumSize(640, 480)
        self._data = data if data is not None else workflow_map()
        self._protocols = self._data.get('protocols', {})
        self._tree = self._data.get('protocol_tree', {})
        self._source_path = ''
        self._context = {}
        self._diagram = None
        self._selected_protocol = None
        self._active_route = {}
        self._trail = []
        self._choices = {}
        layout = QVBoxLayout(self)
        self._source_label = QLabel(self)
        self._source_label.setWordWrap(True)
        self._source_label.setTextFormat(Qt.PlainText)
        layout.addWidget(self._source_label)
        self._heading = QLabel(self)
        self._heading.setTextFormat(Qt.PlainText)
        self._heading.setWordWrap(True)
        layout.addWidget(self._heading)
        self._choice_list = QListWidget(self)
        self._choice_list.setObjectName('ProtocolChoices')
        self._choice_list.currentRowChanged.connect(self._choice_changed)
        layout.addWidget(self._choice_list, 1)
        self._splitter = diagram_splitter(self)
        self._diagram_host = QWidget(self)
        self._diagram_layout = QVBoxLayout(self._diagram_host)
        self._diagram_layout.setContentsMargins(0, 0, 0, 0)
        self._diagram_host.setMinimumHeight(130)
        self._splitter.addWidget(self._diagram_host)
        self._text = QTextBrowser(self)
        self._text.setObjectName('ProtocolInstructions')
        self._text.setDocument(_ProtocolDocument(self._text))
        self._text.setOpenLinks(False)
        self._text.setOpenExternalLinks(False)
        self._text.anchorClicked.connect(self._activate_link)
        self._text.setMinimumHeight(170)
        self._splitter.addWidget(self._text)
        self._splitter.setSizes([270, 350])
        layout.addWidget(self._splitter, 2)
        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        self._previous = QPushButton(tr('Previous'), self)
        self._previous.setObjectName('ProtocolPrevious')
        self._previous.clicked.connect(self._previous_level)
        buttons.addButton(self._previous, QDialogButtonBox.ActionRole)
        self._next = QPushButton(tr('Next'), self)
        self._next.setObjectName('ProtocolNext')
        self._next.clicked.connect(self._next_level)
        buttons.addButton(self._next, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        self._set_input(source_path, context)

    def _set_input(self, source_path='', context=None):
        self._source_path = str(source_path or '')
        self._context = deepcopy(context or {})
        self._source_label.setText(tr('Input: {path}', path=self._source_path))
        self._source_label.setVisible(bool(self._source_path))
        metadata = self._context.get('context', self._context)
        artifacts = {item.get('kind') for item in metadata.get('generated_outputs', {}).get('artifacts', ())
                     if isinstance(item, dict)}
        analyzed = any(item.get('objects') and item.get('numeric')
                       for item in metadata.get('database_schemas', ()) if isinstance(item, dict))
        analysis = metadata.get('analysis_inputs', {})
        fastq = analysis.get('map_barcodes', {}).get('fastq')
        counts = any(item.get('count_data') for item in analysis.values() if isinstance(item, dict))
        tables = any(item.get('table') for item in analysis.values() if isinstance(item, dict))
        source = self._source_path.lower()
        kind = metadata.get('protocol_data') or (
            'images' if source.endswith('.pdf') else
            'fastq' if fastq or source.endswith(('.fastq', '.fq', '.fastq.gz', '.fq.gz')) else
            'barcode_counts' if counts else
            'regression' if 'regression' in artifacts else
            'classifier' if 'classifier' in artifacts else
            'features' if 'measurements' in artifacts or analyzed or tables else
            'crops' if 'crops' in artifacts else
            'images_masks' if artifacts.intersection(('merged', 'masks')) or metadata.get('paired_layout') else
            'images' if metadata.get('image_folders') or metadata.get('image_dimensions') else
            'run_records' if metadata.get('run_evidence', {}).get('manifests') else '')
        self._choices = {}
        for child in self._tree.get('children', ()):
            if child.get('data', child.get('id')) == kind:
                self._choices[self._tree.get('id', '')] = child['id']
                break
        self._trail = [self._tree]
        self._show_level()

    def _show_level(self):
        node = self._trail[-1]
        self._heading.setText(tr(node.get('title', 'Protocols')))
        self._previous.setEnabled(len(self._trail) > 1)
        children = node.get('children', ())
        self._choice_list.setVisible(bool(children))
        blocked = self._choice_list.blockSignals(True)
        self._choice_list.clear()
        for child in children:
            self._choice_list.addItem(tr(child['title']))
        wanted = self._choices.get(node.get('id', ''))
        selected = next((i for i, child in enumerate(children) if child['id'] == wanted), -1)
        self._choice_list.setCurrentRow(selected)
        self._choice_list.blockSignals(blocked)
        self._active_route = node
        self._display_protocol(node.get('protocol') if not children else None)
        self._next.setVisible(bool(children))
        self._next.setEnabled(selected >= 0)
        if children:
            self._choice_changed(selected)
        elif self._selected_protocol is None:
            self._text.setPlainText(tr(node.get('description', '')))

    def _choice_changed(self, row):
        node = self._trail[-1]
        children = node.get('children', ())
        self._next.setEnabled(0 <= row < len(children))
        selected = children[row] if 0 <= row < len(children) else None
        if selected is not None:
            self._choices[node.get('id', '')] = selected['id']
        self._text.setPlainText(tr((selected or node).get('description', '')))
        self._text.verticalScrollBar().setValue(0)

    def _next_level(self):
        children = self._trail[-1].get('children', ())
        row = self._choice_list.currentRow()
        if 0 <= row < len(children):
            self._trail.append(children[row])
            self._show_level()

    def _previous_level(self):
        if len(self._trail) > 1:
            self._trail.pop()
            self._show_level()

    def _display_protocol(self, key):
        while self._diagram_layout.count():
            item = self._diagram_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        self._diagram = None
        self._selected_protocol = key if key in self._protocols else None
        self._diagram_host.setVisible(self._selected_protocol is not None)
        if self._selected_protocol is None:
            self._text.setPlainText(tr('No matching protocol.'))
            return
        protocol = self._protocols[key]
        steps = [step for step in protocol.get('steps', ()) if step.get('module') in self._data['modules']]
        keys = list(dict.fromkeys(step['module'] for step in steps))
        inputs = []
        for artifact in protocol.get('inputs', ()):
            info = self._data.get('artifacts', {}).get(artifact, {})
            inputs.append(dict(id=artifact, title=info.get('title', artifact), artifacts=[artifact],
                               description=info.get('location', ''), targets=[step['module'] for step in steps
                               if artifact in self._data['modules'][step['module']].get('inputs', ())]))
        entry = dict(modules=keys, inputs=inputs)
        display, keys, graph_steps = pathway_graph(entry, self._data, steps)
        self._diagram = WorkflowView(display, keys, graph_steps, self._diagram_host, compact=True)
        self._diagram.selection_changed.connect(self._select_step)
        self._diagram_layout.addWidget(self._diagram)
        self._text.setHtml(self._protocol_html(protocol, steps))
        self._text.verticalScrollBar().setValue(0)
        start = self._active_route.get('start_module')
        if start in self._diagram.nodes:
            self._diagram.describe_node(start)

    def _protocol_html(self, protocol, steps):
        from html import escape
        from ..ai.prompts import _navigation_modules
        from ..help_search import api_url
        from ..tutorials import tutorial_url

        available = {row[0] for row in _navigation_modules()}
        parts = ['<h2>' + escape(tr(protocol['title'])) + '</h2>',
                 '<p>' + escape(tr(protocol.get('summary', ''))) + '</p>']
        if protocol.get('inputs'):
            parts.append('<p><b>' + escape(tr('Inputs')) + ':</b> ' +
                         self._artifacts_html(protocol['inputs']) + '</p>')
        parts.append('<ol>')
        for step in steps:
            key = step['module']
            module = self._data['modules'][key]
            title = escape(tr(module.get('name', key)))
            if key in available:
                title = '<a href="spacr://module/' + escape(key, quote=True) + '">' + title + '</a>'
            parts.append('<li><a name="step-' + escape(key, quote=True) + '"></a><b>' + title + '</b><p>' +
                         escape(tr(step.get('action', ''))) + '</p>')
            if key not in available:
                parts.append('<p>' + escape(tr('This module is hidden by the current application preferences.')) + '</p>')
            for field, label in (('inputs', tr('Inputs')), ('outputs', tr('Outputs'))):
                if module.get(field):
                    parts.append('<p><b>' + escape(label) + ':</b> ' + self._artifacts_html(module[field]) + '</p>')
            links = []
            if module.get('api_module'):
                links.append('<a href="' + escape(api_url(module['api_module']), quote=True) + '">' + escape(tr('API')) + '</a>')
            tutorial = tutorial_url(key)
            if tutorial:
                links.append('<a href="' + escape(tutorial, quote=True) + '">' + escape(tr('Tutorial')) + '</a>')
            parts.append('<p>' + ' · '.join(links) + '</p></li>')
        parts.append('</ol>')
        if protocol.get('checkpoint'):
            parts.append('<p><b>' + escape(tr('Checkpoint')) + ':</b> ' + escape(tr(protocol['checkpoint'])) + '</p>')
        return ''.join(parts)

    def _artifacts_html(self, keys):
        from html import escape

        records = []
        for key in keys:
            artifact = self._data.get('artifacts', {}).get(key, {})
            text = escape(tr(artifact.get('title', key)))
            if artifact.get('location'):
                text += ' (' + escape(tr(artifact['location'])) + ')'
            records.append(text)
        return '; '.join(records)

    def _select_step(self, key):
        self._text.scrollToAnchor('step-' + key)

    def _activate_link(self, url):
        from .ai_chat_panel import _activate_ai_link, _ai_module_key

        key = _ai_module_key(url)
        if key is not None and self._source_path:
            owner = self.parentWidget()
            while owner is not None:
                opener = getattr(owner, '_open_easy_input', None)
                if callable(opener) and key != 'overview':
                    return bool(opener(key, self._source_path,
                                       self._context.get('context', self._context)))
                opener = getattr(owner, '_open_easy_suggestion', None)
                if callable(opener):
                    opener(key, self._source_path)
                    return True
                owner = owner.parentWidget()
        return _activate_ai_link(self, url)


def _show_protocols(parent, source_path='', context=None):
    dialog = getattr(parent, '_protocol_browser_dialog', None)
    try:
        reusable = dialog is not None and dialog.objectName() == 'ProtocolBrowser'
    except RuntimeError:
        reusable = False
    if not reusable:
        dialog = _ProtocolBrowserDialog(parent, source_path, context)
        if parent is not None:
            parent._protocol_browser_dialog = dialog
    elif str(source_path or '') != dialog._source_path or dict(context or {}) != dialog._context:
        dialog._set_input(source_path, context)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
