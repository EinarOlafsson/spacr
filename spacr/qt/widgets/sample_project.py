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


def _protocol_options(family='', format='', choices=None, *, data=None):
    """Return family/format records, or the flat dependent detail fields.

    :param family: empty for input families, otherwise a canonical family key.
    :param format: empty for formats, otherwise a canonical format key.
    :param choices: retained selections; no caller values are modified.
    :param data: optional bundled workflow map for an isolated caller.
    :returns: immutable tuples of fresh choice or detail-field dictionaries.
    """
    from ...regex_infer import _metadata_convention_menu
    from ..organisms import ORGANISMS

    data = workflow_map() if data is None else data
    wizard = data['protocol_wizard']
    choices = dict(choices or {})
    records = wizard['families'] if not family else wizard['families'][family]['formats']
    if not format:
        return tuple(dict(key=key, label=tr(row['label']), explanation=tr(row['explanation']))
                     for key, row in records.items())
    if format not in records:
        raise ValueError(tr('Choose a supported data format.'))
    fields = []
    if family in ('small_images', 'plate_images') and format != 'pdf':
        if format.startswith('other_'):
            fields.append(dict(name='image_format', label=tr('Image format'), required=True,
                               options=tuple(dict(key=key, label=tr(row['label']),
                                                  explanation=tr(row['explanation']))
                                             for key, row in wizard['image_formats'].items())))
        if format.endswith(('other', 'metadata')):
            options = tuple(dict(key=key, label=tr(label), explanation=tr(status))
                            for vendor, rows in _metadata_convention_menu()
                            for key, label, status in rows)
            fields.append(dict(name='metadata_type', label=tr('Filename metadata'),
                               required=True, options=options))
    elif family == 'sequencing' and format == 'fastq_gz':
        for name, label, source in (('sequencer', tr('Sequencer'), 'sequencers'),
                                    ('read_mode', tr('Read pairing'), 'read_modes')):
            fields.append(dict(name=name, label=label, required=True,
                               options=tuple(dict(key=k, label=tr(v['label']),
                                                  explanation=tr(v['explanation']))
                                             for k, v in wizard[source].items())))
        if choices.get('read_mode') == 'single':
            fields.append(dict(name='single_direction', label=tr('Read orientation'), required=True,
                               options=tuple(dict(key=mate, label=mate,
                                                  explanation=tr('Match the barcode pattern and reference to this mate orientation.'))
                                             for mate in ('R1', 'R2'))))
    elif family == 'features' and format in ('database', 'database_crops'):
        fields.append(dict(name='experiment', label=tr('Experiment'), required=True,
                           options=tuple(dict(key=k, label=tr(v['label']),
                                              explanation=tr(v['explanation']))
                                         for k, v in wizard['experiments'].items())))
        if choices.get('experiment') == 'screen':
            fields.append(dict(name='barcodes', label=tr('Barcode mapping'), required=True,
                               options=tuple(dict(key=k, label=tr(v['label']),
                                                  explanation=tr(v['explanation']))
                                             for k, v in wizard['barcodes'].items())))
        if choices.get('experiment') == 'other':
            fields.append(dict(name='organism', label=tr('Organism'), required=True,
                               options=tuple(dict(key=k, label=tr(v['name']),
                                                  explanation=tr(v['description']))
                                             for k, v in ORGANISMS.items())))
    if family in ('small_images', 'plate_images') and format != 'pdf':
        catalog = _protocol_catalog(data)
        candidates = [(key, row) for key, row in catalog.items()
                      if row['available'] and set(row['inputs']).intersection(('raw_images', 'cellpose_tiffs'))]
        fields.append(dict(name='goal', label=tr('Analysis route (optional)'), required=False,
                           options=(dict(key='recommended', label=tr('Recommended starting modules'),
                                         explanation=tr('Choose a starting module after checking its prerequisites.')),)
                           + tuple(dict(key=key, label=tr(row['name']), explanation=tr(row['guidance']))
                                   for key, row in candidates)))
    return tuple(fields)


def _protocol_catalog(data=None):
    """Join every editorial module and registered navigation/organism route.

    :param data: optional workflow map.
    :returns: fresh records with actual I/O, prerequisites and visibility.
    """
    from ..ai.prompts import _navigation_modules
    from ..organisms import ORGANISMS

    data = workflow_map() if data is None else data
    available = {row[0]: row for row in _navigation_modules()}
    catalog = deepcopy(data['modules'])
    for key, organism in ORGANISMS.items():
        catalog.setdefault(key, dict(name=organism['name'], guidance=organism['description'],
                                     inputs=['organism_reference'], outputs=['organism_navigation'],
                                     api_module='spacr.qt.organisms', parent=None))
    for key, row in available.items():
        if key not in catalog:
            inputs = ['raw_images'] if key in ('overview', '3d') else ['organism_reference']
            contract = data['protocol_wizard'].get('navigation_contracts', {}).get(key, {})
            catalog[key] = dict(name=row[1], guidance=contract.get('guidance', row[2]),
                                inputs=contract.get('inputs', inputs),
                                outputs=contract.get('outputs', ['organism_navigation']),
                                api_module='', parent=None)
    for key, module in catalog.items():
        module['available'] = key in available
        module['recipes'] = [name for name, recipe in data.get('protocols', {}).items()
                             if any(step['module'] == key for step in recipe['steps'])]
        module['organisms'] = [name for name, organism in ORGANISMS.items()
                               if key == name or any(route[0] == key
                                                     for route in organism.get('workflows', {}).values())]
        module['preconditions'] = [data['artifacts'][artifact]['location']
                                   for artifact in module['inputs']]
    return catalog


def _protocol_result(family, format, choices=None, source_path='', context=None, *, data=None):
    """Resolve a fully chosen current-capability route without running analysis.

    :param family: one of the four canonical input families.
    :param format: an actual format in that family.
    :param choices: retained required detail selections.
    :param source_path: the original user-selected input, never synthesized.
    :param context: copied schema/provenance/settings inspection context.
    :param data: optional workflow map.
    :returns: cards, compatible per-module settings and an editable recipe model.
    :raises ValueError: if any current required selection is absent or invalid.
    """
    from ..organisms import ORGANISMS

    data = workflow_map() if data is None else data
    choices = deepcopy(choices or {})
    wizard = data['protocol_wizard']
    if family not in wizard['families'] or format not in wizard['families'][family]['formats']:
        raise ValueError(tr('Choose a supported input family and format.'))
    for field in _protocol_options(family, format, choices, data=data):
        if field.get('required') or choices.get(field['name']):
            valid = choices.get(field['name']) in {row['key'] for row in field['options']}
        else:
            valid = True
        if not valid:
            raise ValueError(tr('Complete the required protocol choices.'))
    record = wizard['families'][family]['formats'][format]
    catalog = _protocol_catalog(data)
    starts = list(record['starts'])
    if family in ('small_images', 'plate_images') and choices.get('goal', 'recommended') != 'recommended':
        starts = [choices['goal']]
    requirements = []
    if choices.get('metadata_type') == 'custom' and not (context or {}).get('custom_regex'):
        requirements.append(tr('Custom filename metadata requires a tested named-group expression before running. Configure custom_regex in the selected module.'))
    if family == 'sequencing':
        if format == 'other' or choices.get('read_mode') == 'mixed':
            starts = []
            requirements.append(tr(wizard['requirements']['sequencing']))
    elif family == 'features':
        if choices.get('experiment') == 'drug':
            starts = ['dose_response', 'graph_builder', 'gate_editor', 'qc_dashboard', 'db_browser']
            requirements.append(tr(wizard['requirements']['drug']))
        elif choices.get('experiment') == 'screen':
            starts = (['map_barcodes'] if choices.get('barcodes') == 'unmapped' else
                      ['regression', 'graph_builder', 'gate_editor', 'qc_dashboard', 'db_browser'])
            if format == 'database_crops':
                starts.extend(['annotate', 'classify_merged', 'umap'])
        elif choices.get('organism') in ORGANISMS:
            organism = ORGANISMS[choices['organism']]
            starts = [choices['organism'], *dict.fromkeys(route[0]
                       for route in organism.get('workflows', {}).values()),
                       'foreign', 'graph_builder', 'gate_editor', 'db_browser', 'qc_dashboard']
            if format == 'database_crops':
                starts.extend(['annotate', 'classify_merged', 'umap'])
    starts = list(dict.fromkeys(k for k in starts if k in catalog and catalog[k]['available']))
    source_path = str(source_path or '')
    selection = dict(family=family, format=format, choices=choices)
    carried = deepcopy(context or {})
    carried['protocol_selection'] = deepcopy(selection)
    overrides = {}
    for key in starts:
        values = {}
        if key in ('mask', 'import_images', 'foreign', '3d', 'timelapse') and family in ('small_images', 'plate_images'):
            values['metadata_type'] = ('cellvoyager' if format.endswith('yokogawa') else choices.get('metadata_type', 'cellvoyager'))
            if values['metadata_type'] == 'custom' and carried.get('custom_regex'):
                values['custom_regex'] = carried['custom_regex']
        if key == 'map_barcodes' and family == 'sequencing' and format == 'fastq_gz':
            values['mode'] = choices['read_mode']
            if choices['read_mode'] == 'single':
                values['single_direction'] = choices['single_direction']
        overrides[key] = values
    steps = []
    for key in starts:
        module = catalog[key]
        steps.append(dict(module=key, action=module['guidance'], after=[], optional=True))
    reachable = set(starts)
    for edge in data['connections']:
        if edge['from'] in starts and edge['to'] not in reachable:
            reachable.add(edge['to'])
            steps.append(dict(module=edge['to'], action=edge['handoff'], after=[edge['from']], optional=True))
    selected_options = []
    for field in _protocol_options(family, format, choices, data=data):
        selected = next((row for row in field['options'] if row['key'] == choices.get(field['name'])), None)
        if selected:
            selected_options.append(dict(label=field['label'], value=selected['label']))
    protocol = dict(title=tr('Protocol: {family}', family=tr(wizard['families'][family]['label'])),
                    summary=tr(record['explanation']), inputs=sorted({a for k in starts for a in catalog[k]['inputs']}),
                    steps=steps, catalog=catalog, selection=selection, selected_options=selected_options, requirements=requirements)
    artifacts = ({'raw_images', 'cellpose_tiffs'} if family in ('small_images', 'plate_images') else
                 {'fastq'} if family == 'sequencing' else
                 {'measurements', 'foreign_tables', 'crops'} if format == 'database_crops' else
                 {'measurements', 'foreign_tables'})
    native_sources = carried.get('protocol_native_sources', {})
    action_sources = {key: str(native_sources.get(key) or source_path)
                      if set(catalog[key]['inputs']).intersection(artifacts) else ''
                      for key in starts}
    result = dict(actions=tuple((k, tr(catalog[k]['name']), tr(catalog[k]['guidance']), action_sources[k])
                                for k in starts), action_sources=action_sources, action_settings=overrides,
                  settings_overrides=deepcopy(overrides), message=tr(record['explanation']),
                  protocol=protocol, path=source_path, source_path=source_path, context=carried,
                  requirements=tuple(requirements))
    protocol['markdown'] = _protocol_markdown(result)
    return result


def _protocol_handoff(result, key):
    """Copy one available explicit card's context and compatible settings.

    :param result: a resolved protocol result.
    :param key: an available starting card's module key.
    :returns: a detached payload; this function neither opens nor runs a module.
    :raises ValueError: if the requested module is not one of the resolved cards.
    """
    from ..ai.prompts import _navigation_modules

    if key not in {row[0] for row in result['actions']} or key not in {row[0] for row in _navigation_modules()}:
        raise ValueError(tr('This module is not available for the selected protocol.'))
    from ..screens.settings_model import resolve_default_settings

    context = deepcopy(result['context'])
    defaults = resolve_default_settings(key)
    settings = {name: deepcopy(value) for name, value in result['action_settings'].get(key, {}).items()
                if name in defaults}
    path = result['action_sources'][key]
    if path and 'src' in defaults:
        settings['src'] = path
    if settings:
        inferred = context.setdefault('submodule_settings', {}).setdefault(key, {})
        inferred.setdefault('settings', {}).update(deepcopy(settings))
    return dict(module=key, source_path=path, settings=settings, context=context)


def _protocol_setting_help(key):
    """Read current settings/default explanations only for an explicitly opened step.

    :param key: registered module key.
    :returns: setting-name, default and authored-help tuples.
    """
    from ..screens.settings_model import resolve_default_settings
    from ... import settings

    defaults = resolve_default_settings(key)
    tips = getattr(settings, 'tooltips', {})
    return tuple((name, value, tips.get(name, '')) for name, value in defaults.items())


def _protocol_markdown(result, checked=(), *, settings_help=None):
    """Produce a passive Markdown recipe with nested expandable explanations.

    :param result: resolved protocol result with actual module/artifact contracts.
    :param checked: completed step keys, copied by the caller.
    :param settings_help: lazily resolved authored settings, indexed by module key.
    :returns: Markdown containing explicit links and nested details, never runnable code.
    """
    from ..help_search import api_url
    from ..tutorials import tutorial_url
    from ..screens.settings_model import DOCS_SITE_BASE

    protocol = result['protocol']
    lines = ['# ' + protocol['title'], '', protocol['summary'], '',
             tr('Starting modules are alternatives. Open a module explicitly; no analysis starts from this recipe.'), '']
    lines.extend(protocol['requirements'])
    lines.extend(['', tr('Selected options'), *[row['label'] + ': ' + row['value'] for row in protocol['selected_options']]])
    for step in protocol['steps']:
        key = step['module']
        module = protocol['catalog'][key]
        lines.extend(['', '- [' + ('x' if key in checked else ' ') + '] [' + tr(module['name']) +
                      '](spacr://module/' + key + ')', '', tr(step['action']), '',
                      '<details><summary>' + tr('Inputs, outputs and settings') + '</summary>', '',
                      tr('Inputs') + ': ' + ', '.join(module['inputs']), '',
                      tr('Outputs') + ': ' + ', '.join(module['outputs']), '',
                      *[tr(text) for text in module['preconditions']], '',
                      '<details><summary>' + tr('Setting explanations') + '</summary>', ''])
        for name, default, help_text in (settings_help or {}).get(key, ()):
            configured = dict(result['action_settings'].get(key, {}))
            if result['action_sources'].get(key):
                configured['src'] = result['action_sources'][key]
            label = tr('Selected: {value}', value=repr(configured[name])) if name in configured else tr('Factory default: {value}', value=repr(default))
            lines.extend(['### ' + name, label, tr(help_text), ''])
        if key not in (settings_help or {}):
            lines.append(tr('Open this step to inspect its current settings and authored help.'))
        lines.extend(['', '</details>', '', '</details>'])
        links = []
        if module.get('api_module'):
            links.append('[' + tr('API') + '](' + api_url(module['api_module']) + ')')
        links.append('[' + tr('Documentation') + '](' + DOCS_SITE_BASE + '/workflows.html)')
        tutorial = tutorial_url(key)
        if tutorial:
            links.append('[' + tr('Tutorial') + '](' + tutorial + ')')
        lines.extend(['', ' · '.join(links)])
    lines.extend(['', '<details><summary>' + tr('All registered routes and prerequisites') + '</summary>', ''])
    for key, module in protocol['catalog'].items():
        lines.extend(['<details><summary>' + tr(module['name']) + '</summary>',
                      tr(module['guidance']), tr('Inputs') + ': ' + ', '.join(module['inputs']),
                      tr('Outputs') + ': ' + ', '.join(module['outputs']),
                      *[tr(text) for text in module['preconditions']], '</details>', ''])
    lines.append('</details>')
    return '\n'.join(lines)


class _ProtocolDocument(QTextDocument):
    """Keep protocol instructions passive by refusing document resource loads."""
    def loadResource(self, kind, url):
        """Refuse resource loading while the protocol instructions are rendered.

        :param kind: requested document resource category.
        :param url: requested resource URL.
        :returns: ``None`` instead of resource data.
        """
        return None


class _ProtocolBrowserDialog(QDialog):
    """Browse source-aware protocol choices, workflow diagrams and linked instructions."""
    def __init__(self, parent=None, source_path='', context=None, *, data=None, result=None):
        """Initialize the browser choices and views from the supplied input context.

        :param parent: owner widget, or ``None``.
        :param source_path: input path retained for explicit module links.
        :param context: input metadata copied to choose the initial protocol branch.
        :param data: optional workflow map; the shared map is used when omitted.
        :param result: resolved three-step recipe, or None for legacy navigation.
        """
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
        self._recipe_result = None
        self._expanded = set()
        self._checked = set()
        self._settings_help = {}
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
        self._save_pdf = QPushButton(tr('Save PDF…'), self)
        self._save_pdf.setObjectName('ProtocolSavePdf')
        self._save_pdf.clicked.connect(self._save_recipe_pdf)
        buttons.addButton(self._save_pdf, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        self._save_pdf.hide()
        if result is None:
            self._set_input(source_path, context)
        else:
            self._set_result(result)

    def _set_result(self, result):
        """Mount a detached wizard recipe without changing legacy navigation data.

        :param result: complete current-capability result from the wizard resolver.
        """
        self._recipe_result = deepcopy(result)
        self._source_path = result['source_path']
        self._context = deepcopy(result['context'])
        self._checked.clear()
        self._expanded.clear()
        self._settings_help.clear()
        self._source_label.setText(tr('Input: {path}', path=self._source_path))
        self._source_label.setVisible(bool(self._source_path))
        self._heading.setText(result['protocol']['title'])
        self._choice_list.hide()
        self._diagram_host.hide()
        self._previous.hide()
        self._next.hide()
        self._save_pdf.show()
        self._render_recipe()

    def _recipe_html(self, expand_all=False):
        """Render nested recipe sections with real check states and authored settings.

        :param expand_all: include every explanation for a self-contained PDF.
        :returns: escaped rich text; resource loading remains disabled.
        """
        from html import escape
        from ..help_search import api_url
        from ..tutorials import tutorial_url
        from ..screens.settings_model import DOCS_SITE_BASE

        result = self._recipe_result
        protocol = result['protocol']
        parts = ['<h1>' + escape(protocol['title']) + '</h1>',
                 '<p>' + escape(protocol['summary']) + '</p>',
                 '<p>' + escape(tr('Starting modules are alternatives. Open a module explicitly; no analysis starts from this recipe.')) + '</p>']
        if self._source_path:
            parts.append('<p>' + escape(tr('Input: {path}', path=self._source_path)) + '</p>')
        for requirement in protocol['requirements']:
            parts.append('<p><b>' + escape(requirement) + '</b></p>')
        if protocol['selected_options']:
            parts.append('<h3>' + escape(tr('Selected options')) + '</h3><ul>')
            parts.extend('<li>' + escape(row['label'] + ': ' + row['value']) + '</li>' for row in protocol['selected_options'])
            parts.append('</ul>')
        for step in protocol['steps']:
            key = step['module']
            module = protocol['catalog'][key]
            mark = '[x]' if key in self._checked else '[ ]'
            check = mark if expand_all else '<a href="spacr://protocol/check/' + key + '">' + mark + '</a>'
            parts.append('<h2>' + check + ' ' + escape(tr(module['name'])) + '</h2><p>' + escape(tr(step['action'])) + '</p>')
            links = []
            if module['available']:
                if expand_all:
                    links.append(escape(tr('Open {module} in spaCR', module=tr(module['name']))))
                else:
                    links.append('<a href="spacr://module/' + key + '">' + escape(tr('Open module')) + '</a>')
            if module.get('api_module'):
                links.append('<a href="' + escape(api_url(module['api_module']), quote=True) + '">' + tr('API') + '</a>')
            links.append('<a href="' + DOCS_SITE_BASE + '/workflows.html">' + escape(tr('Documentation')) + '</a>')
            tutorial = tutorial_url(key)
            if tutorial:
                links.append('<a href="' + escape(tutorial, quote=True) + '">' + escape(tr('Tutorial')) + '</a>')
            parts.append('<p>' + ' | '.join(links) + '</p>')
            title = escape(tr('Inputs, outputs and settings'))
            parts.append('<h3>' + title + '</h3>' if expand_all else '<p><a href="spacr://protocol/expand/' + key + '">' + title + '</a></p>')
            if expand_all or key in self._expanded:
                for name, label in (('inputs', tr('Inputs')), ('outputs', tr('Outputs'))):
                    parts.append('<h3>' + escape(label) + '</h3><ul>')
                    for artifact in module[name]:
                        record = self._data['artifacts'][artifact]
                        parts.append('<li>' + escape(tr(record['title'])) + ': ' + escape(tr(record['location'])) + '</li>')
                    parts.append('</ul>')
                inner = key + ':settings'
                title = escape(tr('Setting explanations'))
                parts.append('<h3>' + title + '</h3>' if expand_all else '<p><a href="spacr://protocol/expand/' + inner + '">' + title + '</a></p>')
                if expand_all or inner in self._expanded:
                    if key not in self._settings_help:
                        self._settings_help[key] = _protocol_setting_help(key)
                    for name, default, help_text in self._settings_help[key]:
                        configured = dict(result['action_settings'].get(key, {}))
                        if result['action_sources'].get(key):
                            configured['src'] = result['action_sources'][key]
                        label = tr('Selected: {value}', value=repr(configured[name])) if name in configured else tr('Factory default: {value}', value=repr(default))
                        parts.append('<h4>' + escape(name) + '</h4><p>' +
                                     escape(label) + '</p><p>' +
                                     escape(tr(help_text)) + '</p>')
        if not expand_all or 'catalog' in self._expanded:
            title = escape(tr('All registered routes and prerequisites'))
            parts.append('<h2>' + title + '</h2>' if expand_all else '<h2><a href="spacr://protocol/expand/catalog">' + title + '</a></h2>')
        if 'catalog' in self._expanded:
            for key, module in protocol['catalog'].items():
                parts.append('<h3>' + escape(tr(module['name'])) + '</h3><p>' + escape(tr(module['guidance'])) + '</p>')
                if not module['available']:
                    parts.append('<p>' + escape(tr('This module is hidden by the current application preferences or is an API-only entry.')) + '</p>')
                parts.append('<p>' + escape(tr('Inputs')) + ': ' + escape(', '.join(tr(self._data['artifacts'][item]['title']) for item in module['inputs'])) + '</p>')
                parts.extend('<p>' + escape(tr(text)) + '</p>' for text in module['preconditions'])
                parts.append('<p>' + escape(tr('Outputs')) + ': ' + escape(', '.join(tr(self._data['artifacts'][item]['title']) for item in module['outputs'])) + '</p>')
        return ''.join(parts)

    def _render_recipe(self):
        """Refresh the current recipe while retaining its scroll position."""
        position = self._text.verticalScrollBar().value()
        self._text.setHtml(self._recipe_html())
        self._recipe_result['protocol']['markdown'] = _protocol_markdown(self._recipe_result, self._checked, settings_help=self._settings_help)
        self._text.verticalScrollBar().setValue(position)

    def _export_recipe_pdf(self, path):
        """Write a readable, expanded recipe PDF with current checkbox states.

        :param path: user-chosen destination; existing files follow Qt's save dialog policy.
        :returns: the destination path after the PDF writer completes.
        """
        from PySide6.QtCore import QMarginsF
        from PySide6.QtGui import QFont, QPageLayout, QPageSize, QPdfWriter

        writer = QPdfWriter(str(path))
        writer.setPageSize(QPageSize(QPageSize.A4))
        writer.setPageMargins(QMarginsF(16, 18, 16, 18), QPageLayout.Millimeter)
        writer.setResolution(144)
        writer.setTitle(self._recipe_result['protocol']['title'])
        document = _ProtocolDocument()
        document.setDefaultFont(QFont('DejaVu Sans', 10))
        document.setDefaultStyleSheet('h1 {font-size: 18pt;} h2 {font-size: 13pt; margin-top: 16px;} h3 {font-size: 11pt;} h4 {font-size: 10pt;} p {line-height: 120%;} a {color: #176a98;}')
        document.setHtml(self._recipe_html(expand_all=True))
        self._recipe_result['protocol']['markdown'] = _protocol_markdown(self._recipe_result, self._checked, settings_help=self._settings_help)
        document.print_(writer)
        return str(path)

    def _save_recipe_pdf(self):
        """Ask for a PDF destination and export the expanded current recipe."""
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(self, tr('Save protocol PDF'), '', tr('PDF files (*.pdf)'))
        if path:
            self._export_recipe_pdf(path if path.lower().endswith('.pdf') else path + '.pdf')

    def _set_input(self, source_path='', context=None):
        """Choose the initial protocol branch from the source path and copied metadata.

        :param source_path: input path retained for explicit module links.
        :param context: input metadata copied before selecting a branch.
        """
        self._recipe_result = None
        self._save_pdf.hide()
        self._previous.show()
        self._next.show()
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
        """Render the current choice level and its selected protocol, if any."""
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
        """Remember the chosen child and show its description.

        :param row: choice-list row; an invalid row shows the current level's description.
        """
        node = self._trail[-1]
        children = node.get('children', ())
        self._next.setEnabled(0 <= row < len(children))
        selected = children[row] if 0 <= row < len(children) else None
        if selected is not None:
            self._choices[node.get('id', '')] = selected['id']
        self._text.setPlainText(tr((selected or node).get('description', '')))
        self._text.verticalScrollBar().setValue(0)

    def _next_level(self):
        """Enter the currently selected child when its row is valid."""
        children = self._trail[-1].get('children', ())
        row = self._choice_list.currentRow()
        if 0 <= row < len(children):
            self._trail.append(children[row])
            self._show_level()

    def _previous_level(self):
        """Return to the parent choice level without leaving the root."""
        if len(self._trail) > 1:
            self._trail.pop()
            self._show_level()

    def _display_protocol(self, key):
        """Show the selected protocol workflow and its linked step instructions.

        :param key: protocol identifier; an unknown key displays the no-match message.
        """
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
        """Render escaped protocol steps, artifacts and explicit module, API and tutorial links.

        :param protocol: protocol record containing its title, summary and checkpoint.
        :param steps: ordered steps whose modules are present in the workflow map.
        :returns: the protocol instructions as HTML.
        """
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
        """Render escaped artifact titles and locations for the protocol instructions.

        :param keys: artifact identifiers in display order.
        :returns: semicolon-separated HTML text for their titles and locations.
        """
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
        """Scroll the rendered instructions to the selected module step.

        :param key: module key used by the step's HTML anchor.
        """
        self._text.scrollToAnchor('step-' + key)

    def _activate_link(self, url):
        """Open explicitly selected links, passing the retained input to available module routes.

        :param url: explicitly activated module or external link.
        :returns: whether the module or external link was handled.
        """
        from .ai_chat_panel import _activate_ai_link, _ai_module_key

        if self._recipe_result is not None and url.scheme() == 'spacr' and url.host() == 'protocol':
            kind, _, key = url.path().lstrip('/').partition('/')
            selected = self._checked if kind == 'check' else self._expanded
            if key in selected:
                selected.remove(key)
            else:
                selected.add(key)
            self._render_recipe()
            return True
        key = _ai_module_key(url)
        if key is not None and self._recipe_result is not None and key not in {row[0] for row in self._recipe_result['actions']}:
            return _activate_ai_link(self, url)
        if key is not None and self._recipe_result is not None:
            owner = self.parentWidget()
            while owner is not None:
                opener = getattr(owner, '_open_protocol_module', None)
                if callable(opener):
                    return bool(opener(self._recipe_result, key))
                owner = owner.parentWidget()
            handoff = _protocol_handoff(self._recipe_result, key)
            self._context = handoff['context']
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


def _show_protocols(parent, source_path='', context=None, *, result=None):
    """Show or reuse a protocol browser for the supplied source context.

    :param parent: owner widget, or ``None``.
    :param source_path: input path retained for explicit module links.
    :param context: input metadata used to choose protocol branches.
    :param result: optional resolved wizard recipe; legacy navigation is otherwise retained.
    :returns: the reused or newly created protocol browser dialog.
    """
    dialog = getattr(parent, '_protocol_browser_dialog', None)
    try:
        reusable = dialog is not None and dialog.objectName() == 'ProtocolBrowser'
    except RuntimeError:
        reusable = False
    if not reusable:
        dialog = _ProtocolBrowserDialog(parent, source_path, context, result=result)
        if parent is not None:
            parent._protocol_browser_dialog = dialog
    elif result is not None:
        dialog._set_result(result)
    elif dialog._recipe_result is not None or str(source_path or '') != dialog._source_path or dict(context or {}) != dialog._context:
        dialog._set_input(source_path, context)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
