"""The Voyager3 control in OPS's settings: a button, a progress bar and a status line (item 973, alpha).

The button runs only the ``voyager3`` phase of :func:`spacr.ops_engine.run_ops`
on the settings the panel holds, through the same worker thread a Run uses
(:func:`spacr.qt.bridge.make_thread`), so the run appears on Home and in the
console like any other. The bar follows the run's ``Progress: n/total`` lines.
The AI that answers is the one chosen under Preferences > AI, or
``ops_voyager3_provider``; Voyager3 never uses a vendor API key.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from .i18n import tr

LOG = logging.getLogger(__name__)

#: The setting whose section gets the control.
ANCHOR_KEY = "ops_voyager3_enabled"


def _run_voyager3_phase(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Run only the ``voyager3`` phase of the OPS module on ``settings``.

    :param settings: the OPS settings the panel holds.
    :returns: :func:`spacr.ops_engine.run_ops`' result.
    """
    from ..ops_engine import run_ops

    return run_ops(settings, phases=("voyager3",))


class Voyager3Controls(QWidget):
    """The button, bar and status line that start and follow a Voyager3 run.

    :param screen: the OPS module screen; its settings model gives the
        settings and its console, when it has one, shows the run's lines.
    :param parent: the Qt parent.
    """

    def __init__(self, screen: Any, parent: Optional[QWidget] = None) -> None:
        """Build the control, idle."""
        super().__init__(parent)
        self._screen = screen
        self._thread: Optional[QObject] = None
        self._worker: Optional[QObject] = None
        self.button = QPushButton(tr("Annotate clusters (Voyager3)"), self)
        self.button.setToolTip(tr(
            "Fetch every cluster gene's UniProt facts and ask the AI chosen under Preferences to name "
            "each cluster's biological role. Runs only this step, on the cluster table in the settings "
            "above, and writes results/ops_voyager3."))
        self.button.clicked.connect(self._start)
        self.bar = QProgressBar(self)
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setVisible(False)
        self.status = QLabel(tr("Not run yet."), self)
        self.status.setWordWrap(True)
        row = QHBoxLayout()
        row.addWidget(self.button)
        row.addWidget(self.bar, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self.status)

    def _refresh_visibility(self) -> bool:
        """Show the control only while Show alpha features shows Voyager3's settings.

        :returns: whether it is shown.
        """
        try:
            from .preferences import _is_alpha_visible

            shown = bool(_is_alpha_visible("settings", ANCHOR_KEY))
        except Exception:                                      # noqa: BLE001
            shown = True
        self.setVisible(shown)
        return shown

    def _is_running(self) -> bool:
        """Whether a Voyager3 run started here has not finished."""
        return self._thread is not None

    def _settings(self) -> Dict[str, Any]:
        """The settings the screen's panel holds now."""
        model = getattr(self._screen, "_settings_model", None)
        return dict(model.collect()) if model is not None else {}

    def _start(self, _checked: bool = False) -> bool:
        """Start a Voyager3 run on the current settings.

        :param _checked: swallows the bool a ``clicked`` signal passes.
        :returns: whether a run was started.
        """
        if self._is_running():
            return False
        try:
            settings = self._settings()
        except Exception as exc:                               # noqa: BLE001
            self.status.setText(tr("The settings could not be read: {error}", error=str(exc)))
            return False
        from .bridge import make_thread

        self._thread, self._worker = make_thread(_run_voyager3_phase, settings, app_key="ops")
        self._worker.line_ready.connect(self._on_line)
        console = getattr(self._screen, "_console", None)
        if console is not None and hasattr(console, "_display_stdout"):
            self._worker.line_ready.connect(console._display_stdout)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._thread.finished.connect(self._release)
        self.button.setEnabled(False)
        self.bar.setValue(0)
        self.bar.setVisible(True)
        self.status.setText(tr("Voyager3 is fetching UniProt facts and asking the AI..."))
        self._thread.start()
        return True

    def _on_line(self, line: str) -> None:
        """Move the bar on a ``Progress:`` line.

        :param line: one line the run printed.
        """
        from .run_progress import parse_progress

        samples = parse_progress(str(line))
        if samples and samples[-1].total:
            last = samples[-1]
            self.bar.setValue(int(round(100 * last.done / last.total)))
            found = re.search(r"(\d+) replayed, (\d+) asked", str(line))
            replayed, asked = (found.group(1), found.group(2)) if found else ("0", str(last.done))
            self.status.setText(tr("Voyager3: {done} of {total} clusters annotated ({replayed} replayed, "
                                   "{asked} asked)", done=last.done, total=last.total,
                                   replayed=replayed, asked=asked))

    def _on_error(self, message: str) -> None:
        """Show a failed run's message.

        :param message: the error text the worker sent.
        """
        self.status.setText(tr("Voyager3 stopped: {error}", error=str(message).strip()[:400]))

    def _on_finished(self, ok: bool) -> None:
        """Say where the results are, or leave the error showing.

        :param ok: whether the run finished without an error.
        """
        self.button.setEnabled(True)
        if ok:
            self.bar.setValue(100)
            self.status.setText(tr("Voyager3 finished. The table, report and provenance are in "
                                   "results/ops_voyager3."))

    def _release(self) -> None:
        """Drop the finished thread and worker."""
        self._thread = None
        self._worker = None


def install_voyager3_controls(screen: Any, section: Any) -> Optional[Voyager3Controls]:
    """Add the Voyager3 control at the top of ``section`` once.

    :param screen: the OPS module screen.
    :param section: the settings section that holds :data:`ANCHOR_KEY`.
    :returns: the control, or None when the section already has one or
        cannot hold widgets.
    """
    if getattr(section, "_voyager3_controls", None) is not None:
        return None
    add = getattr(section, "add_prose", None)
    if not callable(add):
        return None
    controls = Voyager3Controls(screen, section if isinstance(section, QWidget) else None)
    add(controls, at_top=True)
    section._voyager3_controls = controls
    controls._refresh_visibility()
    return controls


_VOYAGER3_TOGGLE_TEXT = "Voyager3"

_VOYAGER3_PAGE_KEY = "voyager3"

_CONFIDENCE_COLOURS = {"high": "#0A77B8", "medium": "#0A9878", "low": "#283868",
                       "none": "#8C8C8C"}

_GENE_COLOURS = {"supporting": "#0A77B8", "candidate": "#C878A8", "misfit": "#8C8C8C",
                 "member": "#C8C8C8"}


def _voyager3_tooltip() -> str:
    """What the Voyager3 switch says on hover, in the interface language."""
    return tr("Show or hide the Voyager3 page: name the biological role of each OPS gene "
              "cluster, with an AI or with an enrichment test and no AI, and browse the "
              "map of clusters, genes and terms it makes. Alpha.")


def _cluster_label(cluster: str, role: str) -> str:
    """A cluster name on the map: its number, and its role when one was named."""
    if not role or role.lower().startswith(("no enriched term", "unknown", "none")):
        return cluster
    return f"{cluster}: {role}"


def _knowledge_map(provenance: Dict[str, Any], *, genes: bool = True, terms: bool = True,
                   shuffled: bool = False, confidences: Optional[set] = None
                   ) -> Dict[str, Any]:
    """The knowledge map of one Voyager3 run: its clusters, their genes and their terms.

    Built from the run's provenance record alone (``voyager3_provenance.json``),
    through :func:`spacr.voyager3._tables`, the code that writes the run's
    tables, so the map says what the tables say. Clusters sit on a circle,
    each with its genes around it and its enriched terms outside it.

    :param provenance: the run's provenance record.
    :param genes: put each cluster's genes on the map.
    :param terms: put each cluster's best terms on the map (enrichment runs).
    :param shuffled: put the shuffled-gene control clusters on the map too.
    :param confidences: the confidence words to keep; every one when None.
    :returns: ``{"nodes": [...], "edges": [(a, b), ...]}``; each node has
        ``id``, ``kind`` (cluster, gene or term), ``label``, ``x``, ``y``,
        ``colour``, ``size`` and ``info``.
    """
    import math

    from ..voyager3 import _tables

    clusters, _candidates, _facts, parsed = _tables(provenance)
    facts = provenance.get("facts") or {}
    records = list(provenance.get("clusters") or [])
    kept = []
    for index, record in enumerate(records):
        row = clusters.iloc[index].to_dict() if index < len(clusters) else {}
        kind = str(record.get("kind") or "real")
        if kind != "real" and not shuffled:
            continue
        confidence = str(row.get("confidence_rule") or "none").lower()
        if confidences is not None and confidence not in confidences:
            continue
        kept.append((record, row, parsed[index] if index < len(parsed) else {}))
    nodes: List[Dict[str, Any]] = []
    edges: List[tuple] = []
    radius = max(8.0, 3.2 * len(kept))
    for slot, (record, row, answer) in enumerate(kept):
        angle = 2.0 * math.pi * slot / max(1, len(kept))
        cx, cy = radius * math.cos(angle), radius * math.sin(angle)
        name = f"c:{record.get('kind', 'real')}:{record['cluster']}"
        confidence = str(row.get("confidence_rule") or "none").lower()
        members = list(record.get("genes") or [])
        nodes.append({"id": name, "kind": "cluster", "x": cx, "y": cy,
                      "label": _cluster_label(str(record["cluster"]),
                                              str(row.get("biological_role") or "")),
                      "colour": _CONFIDENCE_COLOURS.get(confidence, "#8C8C8C"),
                      "size": 14 + 2 * min(len(members), 20),
                      "info": {"cluster": str(record["cluster"]),
                               "kind": str(record.get("kind") or "real"),
                               "role": str(row.get("biological_role") or ""),
                               "confidence": confidence,
                               "support": str(row.get("support_fraction") or ""),
                               "supporting": str(row.get("supporting_genes") or ""),
                               "candidates": str(row.get("candidates") or ""),
                               "summary": str(row.get("summary") or ""),
                               "genes": members}})
        if genes and members:
            supporting = {e["gene"] for e in answer.get("supporting") or []}
            candidates = {c["gene"] for c in answer.get("candidates") or []}
            misfit = set(answer.get("misfit") or [])
            ring = 2.2 + 0.2 * len(members)
            for j, gene in enumerate(members):
                theta = 2.0 * math.pi * j / len(members)
                role = ("supporting" if gene in supporting else "candidate"
                        if gene in candidates else "misfit" if gene in misfit else "member")
                fact = dict(facts.get(gene) or {})
                node = f"g:{record.get('kind', 'real')}:{record['cluster']}:{gene}"
                nodes.append({"id": node, "kind": "gene", "label": str(gene),
                              "x": cx + ring * math.cos(theta), "y": cy + ring * math.sin(theta),
                              "colour": _GENE_COLOURS[role], "size": 8,
                              "info": {"gene": str(gene), "cluster": str(record["cluster"]),
                                       "role": role, **{k: fact.get(k) for k in (
                                           "protein_name", "function", "location",
                                           "accession", "annotation_score")}}})
                edges.append((name, node))
        if terms:
            for k, term in enumerate((record.get("terms") or [])[:3]):
                distance = 2.2 + 0.2 * len(members) + 3.0 + 1.5 * k
                node = f"t:{record.get('kind', 'real')}:{record['cluster']}:{term.get('term_id')}"
                nodes.append({"id": node, "kind": "term", "label": str(term.get("term_name") or ""),
                              "x": cx + distance * math.cos(angle),
                              "y": cy + distance * math.sin(angle),
                              "colour": "#F0A830", "size": 10,
                              "info": {"term": str(term.get("term_name") or ""),
                                       "source": str(term.get("source") or ""),
                                       "term_id": str(term.get("term_id") or ""),
                                       "fdr": term.get("fdr"), "fold": term.get("fold"),
                                       "genes": list(term.get("genes") or []),
                                       "cluster": str(record["cluster"])}})
                edges.append((name, node))
    return {"nodes": nodes, "edges": edges}


def _node_tile_html(node: Dict[str, Any]) -> str:
    """The information tile of one map node, as HTML."""
    import html as html_text

    info = node.get("info") or {}
    kind = node.get("kind")
    rows: List[tuple] = []
    if kind == "cluster":
        head = tr("Cluster {cluster}", cluster=info.get("cluster", ""))
        rows = [(tr("Role"), info.get("role")), (tr("Confidence"), info.get("confidence")),
                (tr("Genes cited as support"), info.get("support")),
                (tr("Supporting genes"), info.get("supporting")),
                (tr("Candidates"), info.get("candidates")),
                (tr("Genes"), ", ".join(info.get("genes") or [])),
                (tr("Summary"), info.get("summary"))]
        if info.get("kind") != "real":
            rows.insert(0, (tr("Control"), tr("shuffled genes")))
    elif kind == "gene":
        head = str(info.get("gene") or "")
        rows = [(tr("Cluster"), info.get("cluster")), (tr("Place in the cluster"), info.get("role")),
                (tr("Protein"), info.get("protein_name")), (tr("Function"), info.get("function")),
                (tr("Location"), info.get("location")), (tr("UniProt entry"), info.get("accession")),
                (tr("Annotation score"), info.get("annotation_score"))]
    else:
        head = str(info.get("term") or "")
        rows = [(tr("Source"), info.get("source")), (tr("Term"), info.get("term_id")),
                (tr("Cluster"), info.get("cluster")), (tr("FDR"), info.get("fdr")),
                (tr("Fold"), info.get("fold")), (tr("Genes"), ", ".join(info.get("genes") or []))]
    body = "".join(f"<tr><td><b>{html_text.escape(str(k))}</b></td>"
                   f"<td>{html_text.escape(str(v))}</td></tr>"
                   for k, v in rows if v not in (None, "", []))
    link = ""
    if kind == "gene" and info.get("accession"):
        link = (f'<p><a href="https://www.uniprot.org/uniprotkb/'
                f'{html_text.escape(str(info["accession"]))}">UniProt</a></p>')
    return f"<h3>{html_text.escape(head)}</h3><table cellspacing='4'>{body}</table>{link}"


class _Voyager3Run(QObject):
    """Run Voyager3 off the GUI thread and hand back its result.

    Calls :func:`spacr.voyager3.run_voyager3`, the code the OPS ``voyager3``
    phase and the OPS live Voyager3 tab call.

    :param settings: the settings of the run.
    :param out_dir: the output folder, or None for the run's own.
    """

    line = Signal(str)
    done = Signal(object, str)

    def __init__(self, settings: Dict[str, Any], out_dir: Optional[str] = None) -> None:
        """Hold what the run needs."""
        super().__init__()
        self._settings, self._out_dir = dict(settings), out_dir

    def run(self) -> None:
        """Run Voyager3; every failure comes back as text."""
        from ..voyager3 import run_voyager3

        try:
            result = run_voyager3(self._settings, report=self.line.emit, out_dir=self._out_dir)
        except Exception as error:                                  # noqa: BLE001
            LOG.info("Voyager3 failed", exc_info=True)
            self.done.emit(None, f"{type(error).__name__}: {error}")
            return
        self.done.emit(result, "")


class _Voyager3Page(QWidget):
    """The Voyager3 page: pick an OPS run, run Voyager3 with or without AI, browse its map.

    The map shows each cluster coloured by how well its genes bear out the
    named role, its genes around it and its enriched terms outside it. The
    wheel zooms and a drag pans; the search box finds a cluster, gene or
    term and centres the map on it; clicking a node fills the information
    tile; the filters choose what the map shows.

    :param screen: the screen the page opens from; its OPS settings, when it
        has them, are the base of a run.
    :param parent: parent widget.
    """

    def __init__(self, screen: Any = None, parent: Optional[QWidget] = None) -> None:
        """Lay out the run row, the map and the side panel."""
        super().__init__(parent)
        from PySide6.QtWidgets import (QCheckBox, QComboBox, QLineEdit, QSplitter,
                                       QTextBrowser)

        from .widgets.fast_plots import HAVE_PYQTGRAPH, pg

        self.setObjectName("Voyager3Page")
        self._screen = screen
        self._thread = None
        self._worker: Optional[_Voyager3Run] = None
        self.provenance: Optional[Dict[str, Any]] = None
        self.map: Dict[str, Any] = {"nodes": [], "edges": []}
        self.found: List[str] = []
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("OPS run or cluster table"), self))
        self.source = QLineEdit(self)
        self.source.setObjectName("Voyager3Source")
        self.source.setToolTip(tr(
            "The OPS run folder (its results/ops_clusters/consensus_partition.csv is read) or a "
            "cluster table with gene and cluster columns."))
        self.source.setText(self._default_source())
        row.addWidget(self.source, 1)
        self.browse = QPushButton(tr("Browse…"), self)
        self.browse.clicked.connect(self._browse)
        row.addWidget(self.browse)
        self.method = QComboBox(self)
        self.method.setObjectName("Voyager3Method")
        self.method.addItem(tr("Without AI (enrichment test)"), "enrichment")
        self.method.addItem(tr("With AI (LLM)"), "ai")
        self.method.setToolTip(tr(
            "Without AI, each cluster is named by its most enriched UniProt, CORUM or Reactome "
            "term. With AI, the AI chosen under Preferences > AI names it from the genes' UniProt "
            "facts."))
        row.addWidget(self.method)
        self.run_btn = QPushButton(tr("Run Voyager3"), self)
        self.run_btn.setObjectName("Voyager3Run")
        self.run_btn.clicked.connect(self.run)
        row.addWidget(self.run_btn)
        self.open_btn = QPushButton(tr("Open results…"), self)
        self.open_btn.setToolTip(tr("Show the map of a Voyager3 run made before."))
        self.open_btn.clicked.connect(self._open_results)
        row.addWidget(self.open_btn)
        layout.addLayout(row)
        self.status = QLabel(tr("Choose an OPS run and press Run Voyager3, or open results."), self)
        self.status.setObjectName("Voyager3Status")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.bar = QProgressBar(self)
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.hide()
        layout.addWidget(self.bar)
        split = QSplitter(self)
        self.plot = pg.PlotWidget(split) if HAVE_PYQTGRAPH else QLabel(
            tr("Install pyqtgraph to see the map."), split)
        if HAVE_PYQTGRAPH:
            from ..figures.style import resolve_style

            self._spacr_style_spec = resolve_style("screen", "schematic")
            self.plot.setObjectName("Voyager3Map")
            try:
                from .theme import active_palette

                self.plot.setBackground(active_palette().get("surface", "#161719"))
            except Exception:                                   # noqa: BLE001
                self.plot.setBackground("#161719")
            self.plot.hideAxis("bottom")
            self.plot.hideAxis("left")
            self.plot.getViewBox().setAspectLocked(True)
        side = QWidget(split)
        column = QVBoxLayout(side)
        self.search = QLineEdit(side)
        self.search.setObjectName("Voyager3Search")
        self.search.setPlaceholderText(tr("Find a cluster, gene or term"))
        self.search.textChanged.connect(self.find)
        column.addWidget(self.search)
        self.filters: Dict[str, QCheckBox] = {}
        for key, text in (("high", tr("High confidence")), ("medium", tr("Medium confidence")),
                          ("low", tr("Low confidence")), ("none", tr("No role named")),
                          ("genes", tr("Show genes")), ("terms", tr("Show terms")),
                          ("shuffled", tr("Show the shuffled-gene control"))):
            box = QCheckBox(text, side)
            box.setChecked(key != "shuffled")
            box.toggled.connect(lambda _on: self.redraw())
            self.filters[key] = box
            column.addWidget(box)
        self.tile = QTextBrowser(side)
        self.tile.setObjectName("Voyager3Tile")
        self.tile.setOpenLinks(False)
        self.tile.anchorClicked.connect(self._open_link)
        self.tile.setPlainText(tr("Click a cluster, gene or term on the map to see what it is."))
        column.addWidget(self.tile, 1)
        split.addWidget(self.plot)
        split.addWidget(side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)
        self.scatter = None
        self.shown_node: Optional[Dict[str, Any]] = None

    def _default_source(self) -> str:
        """The OPS form's dst_root, when the screen has an OPS page open."""
        try:
            from .screens.mask import ops_page

            page = ops_page(self._screen) if self._screen is not None else None
            built = getattr(page, "page", None)
            model = getattr(built, "_settings_model", None)
            return str((model.collect() or {}).get("dst_root") or "") if model else ""
        except Exception:                                           # noqa: BLE001
            return ""

    def _base_settings(self) -> Dict[str, Any]:
        """The OPS settings a run starts from: the OPS page's form when open, else the defaults."""
        from ..ops_settings import ops_defaults

        settings: Dict[str, Any] = {}
        try:
            from .screens.mask import ops_page

            page = ops_page(self._screen) if self._screen is not None else None
            model = getattr(getattr(page, "page", None), "_settings_model", None)
            if model is not None:
                settings = dict(model.collect() or {})
        except Exception:                                           # noqa: BLE001
            LOG.debug("no OPS form to start Voyager3 from", exc_info=True)
        return ops_defaults(settings)

    def run_settings(self) -> tuple:
        """The settings and output folder a run uses, from the page's choices.

        :returns: ``(settings, out_dir)``.
        """
        import os

        settings = self._base_settings()
        source = self.source.text().strip()
        settings["ops_voyager3_method"] = self.method.currentData()
        if os.path.isfile(source):
            settings["ops_voyager3_clusters"] = source
            return settings, os.path.join(os.path.dirname(source), "ops_voyager3")
        settings["dst_root"] = source
        settings["ops_voyager3_clusters"] = None
        return settings, None

    def run(self, _checked: bool = False) -> bool:
        """Start a Voyager3 run off the GUI thread.

        :returns: whether a run started.
        """
        from PySide6.QtCore import QThread

        if self._thread is not None:
            return False
        if not self.source.text().strip():
            self.status.setText(tr("Choose an OPS run folder or a cluster table first."))
            return False
        settings, out_dir = self.run_settings()
        self._thread = QThread(self)
        self._worker = _Voyager3Run(settings, out_dir)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.line.connect(self._on_line)
        self._worker.done.connect(self._on_done)
        self.run_btn.setEnabled(False)
        self.bar.setValue(0)
        self.bar.show()
        self.status.setText(tr("Voyager3 is fetching UniProt facts…"))
        self._thread.start()
        return True

    def is_running(self) -> bool:
        """Whether a run started here has not finished."""
        return self._thread is not None

    def _on_line(self, line: str) -> None:
        """Move the bar on a ``Progress:`` line."""
        found = re.search(r"Progress: (\d+)/(\d+)", str(line))
        if found and int(found.group(2)):
            self.bar.setValue(int(100 * int(found.group(1)) / int(found.group(2))))
            self.status.setText(tr("Voyager3: {done} of {total} clusters named.",
                                   done=found.group(1), total=found.group(2)))

    def _on_done(self, result, error: str) -> None:
        """Show the finished run's map, or say why it stopped."""
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.quit()
            thread.wait()
        self._worker = None
        self.run_btn.setEnabled(True)
        self.bar.hide()
        if error or result is None:
            self.status.setText(tr("Voyager3 stopped: {error}", error=str(error)[:400]))
            return
        self.load(result["provenance"])

    def _browse(self) -> None:
        """Choose the OPS run folder."""
        from PySide6.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(self, tr("Choose an OPS run folder"),
                                                  self.source.text())
        if folder:
            self.source.setText(folder)

    def _open_results(self) -> None:
        """Choose a Voyager3 output folder and show its map."""
        import os

        from PySide6.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(self, tr("Choose a Voyager3 results folder"))
        if folder:
            self.load(os.path.join(folder, "voyager3_provenance.json"))

    def load(self, provenance: Any) -> bool:
        """Show the map of one Voyager3 run.

        :param provenance: its provenance record, or the path of
            ``voyager3_provenance.json``.
        :returns: whether a map is shown.
        """
        import json

        if not isinstance(provenance, dict):
            try:
                with open(str(provenance), encoding="utf-8") as handle:
                    provenance = json.load(handle)
            except (OSError, ValueError) as error:
                self.status.setText(tr("No Voyager3 results there: {error}", error=str(error)))
                return False
        self.provenance = provenance
        method = str(provenance.get("method") or "ai")
        named = tr("the enrichment test") if method == "enrichment" else str(
            provenance.get("provider_label") or provenance.get("provider") or tr("the AI"))
        self.status.setText(tr("{n} clusters named by {named}.", n=sum(
            1 for c in provenance.get("clusters") or [] if c.get("kind", "real") == "real"),
            named=named))
        self.redraw()
        return True

    def redraw(self) -> None:
        """Draw the map with the filters as they are now."""
        if self.provenance is None or not hasattr(self.plot, "getViewBox"):
            return
        from ..figures.style import resolve_style
        from .widgets.fast_plots import pg

        self._spacr_style_spec = resolve_style("screen", "schematic")
        keep = {k for k in ("high", "medium", "low", "none") if self.filters[k].isChecked()}
        self.map = _knowledge_map(self.provenance, genes=self.filters["genes"].isChecked(),
                                  terms=self.filters["terms"].isChecked(),
                                  shuffled=self.filters["shuffled"].isChecked(),
                                  confidences=keep)
        self.plot.clear()
        where = {n["id"]: (n["x"], n["y"]) for n in self.map["nodes"]}
        if self.map["edges"]:
            xs, ys = [], []
            for a, b in self.map["edges"]:
                xs += [where[a][0], where[b][0]]
                ys += [where[a][1], where[b][1]]
            self.plot.plot(xs, ys, connect="pairs", pen=pg.mkPen("#8C8C8C", width=1))
        spots = [{"pos": (n["x"], n["y"]), "size": n["size"], "data": n["id"],
                  "brush": pg.mkBrush(n["colour"]), "pen": pg.mkPen(None)}
                 for n in self.map["nodes"]]
        self.scatter = pg.ScatterPlotItem(spots)
        self.scatter.setOpacity(self._spacr_style_spec.alpha)
        self.scatter.sigClicked.connect(self._on_clicked)
        self.plot.addItem(self.scatter)
        for node in self.map["nodes"]:
            if node["kind"] == "cluster":
                text = pg.TextItem(node["label"][:40], anchor=(0.5, -0.4))
                text.setPos(node["x"], node["y"])
                self.plot.addItem(text)
        self.plot.getViewBox().autoRange()
        self.find(self.search.text())

    def node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """The map node with this id, or None."""
        return next((n for n in self.map["nodes"] if n["id"] == node_id), None)

    def _on_clicked(self, _item, points, *_event) -> None:
        """Fill the tile for the clicked node."""
        if len(points):
            self.show_node(str(points[0].data()))

    def show_node(self, node_id: str) -> bool:
        """Fill the information tile for one node.

        :returns: whether the node exists.
        """
        node = self.node(node_id)
        if node is None:
            return False
        self.shown_node = node
        self.tile.setHtml(_node_tile_html(node))
        return True

    def find(self, text: str) -> List[str]:
        """Mark the nodes whose name or role holds ``text`` and centre the map on the first.

        :returns: the ids found.
        """
        from .widgets.fast_plots import pg

        words = str(text or "").strip().lower()
        self.found = []
        if words:
            for node in self.map["nodes"]:
                info = node.get("info") or {}
                blob = " ".join(str(v) for v in (node["label"], info.get("role"),
                                                 info.get("gene"), info.get("term"))).lower()
                if words in blob:
                    self.found.append(node["id"])
        if self.scatter is not None:
            hit = set(self.found)
            for point in self.scatter.points():
                point.setPen(pg.mkPen("#000000", width=3) if point.data() in hit
                             else pg.mkPen(None))
        if self.found:
            first = self.node(self.found[0])
            xs = [n["x"] for n in self.map["nodes"]]
            ys = [n["y"] for n in self.map["nodes"]]
            span = max(2.0, 0.2 * max(max(xs) - min(xs), max(ys) - min(ys)))
            self.plot.getViewBox().setRange(xRange=(first["x"] - span, first["x"] + span),
                                            yRange=(first["y"] - span, first["y"] + span))
            self.show_node(self.found[0])
        return list(self.found)

    @staticmethod
    def _open_link(url) -> None:
        """Open a link of the tile in the web browser."""
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(url)


def _install_voyager3_switch(screen: Any, switch: Any):
    """Wire Mask Generation's actions-row Voyager3 switch to its page beside OPS.

    The switch is alpha, like the OPS switch beside it.

    :param screen: the Mask Generation screen.
    :param switch: the toggle.
    :returns: the page fold, or None when this is not the host screen.
    """
    from .screens.map_barcodes import _gate_alpha, install_page_fold

    switch.setObjectName("MaskVoyager3Switch")
    fold = install_page_fold(screen, switch, "mask", _VOYAGER3_PAGE_KEY,
                             build=lambda _host: _Voyager3Page(screen))
    _gate_alpha(switch)
    return fold
