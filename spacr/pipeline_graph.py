"""The DAG of what produced what, with staleness marked.

:mod:`spacr.artifacts` already knows both halves of this picture and neither
half is visible anywhere: ``upstream_of`` / ``downstream_of`` walk the
provenance edges one artifact at a time, and ``is_stale`` answers "is this
still what it was made from" for one artifact at a time. A user does not have
one artifact. They have a project that has been run, partly re-run, and
re-run again with a different mask diameter, and the question they actually
ask is *which of these files can I still believe*.

This module turns those per-artifact answers into one whole-project graph:

* every registered artifact is a :class:`Node`, carrying its module, kind,
  path, run id, settings hash, spaCR version and — the point — its
  :attr:`Node.state`, one of :data:`STATE_CURRENT`, :data:`STATE_STALE` or
  :data:`STATE_MISSING`, together with the machine ``causes`` and the
  human ``reasons`` that :class:`spacr.artifacts.Staleness` gives;
* every ``inputs`` entry is an :class:`Edge`, so the shape a user recognises
  ("mask fed measure fed classify") is drawn from what actually happened
  rather than from what the pipeline is supposed to do;
* :attr:`PipelineGraph.layers` puts each node in a column by its longest
  distance from a root, which is what makes the drawing readable and is also
  what a text renderer needs.

Two things are deliberately kept apart:

**Stale is not missing.** An artifact whose inputs moved on is stale — the
number in it no longer follows from the files it names. An artifact that was
deleted is missing — an availability problem. :mod:`spacr.artifacts` already
refuses to conflate them and so does this: a node can be both, and the
:attr:`Node.state` reports missing first because a file that is not there
cannot be re-read to check anything else.

**The registry is not the only source of truth.** A project can have run
nothing at all, and a graph of zero nodes tells that user nothing. So
:func:`module_graph` draws the *static* DAG from :data:`spacr.ports.PORTS` —
what feeds what, by declaration — and :func:`build_graph` records which
modules of it have actually produced something. The screen shows the module
DAG behind the artifact DAG for exactly that reason: an empty project still
gets a picture of the pipeline it is about to run.

Everything here is read-only and headless. Nothing in this module writes to
the registry, imports Qt, or raises for a project that has no registry at
all — a missing ``artifacts.db`` is an empty graph with a note, because "you
have not run anything yet" is an answer, not an error.

Public API::

    from spacr.pipeline_graph import build_graph, format_graph, module_graph

    graph = build_graph("/data/plate7")
    print(format_graph(graph))
    stale = graph.stale_nodes()
    print(module_graph().layers)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import (Any, Dict, List, Mapping, Optional, Sequence, Tuple,
                    Union)

from . import ports
from .artifacts import Artifact, Registry, Staleness, registry_path

__all__ = [
    "Edge",
    "ModuleGraph",
    "Node",
    "PipelineGraph",
    "STATE_CURRENT",
    "STATE_MISSING",
    "STATE_ORDER",
    "STATE_STALE",
    "build_graph",
    "format_graph",
    "module_graph",
    "stale_summary",
    "to_dot",
]

#: The artifact still follows from everything it was made from.
STATE_CURRENT = "current"
#: An input moved on, or a material setting changed, after this was written.
STATE_STALE = "stale"
#: The path this artifact was registered at is no longer on disk.
STATE_MISSING = "missing"

#: Worst first. Used for sorting a summary and for choosing one colour when a
#: node is both missing and stale — a file that is not there cannot be
#: re-read, so its absence is the finding to show.
STATE_ORDER: Tuple[str, ...] = (STATE_MISSING, STATE_STALE, STATE_CURRENT)


def _utcnow() -> str:
    """Current UTC time, ISO-8601 to the second."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Node:
    """One registered artifact, with the verdict on whether to believe it.

    :param artifact_id: the registry id; the node's identity in the graph.
    :param project: absolute project root.
    :param kind: a :mod:`spacr.ports` kind, e.g. ``"measurements-db"``.
    :param role: the producing module's port role.
    :param module: producing module key.
    :param path: absolute path of the file or folder.
    :param run_id: the run that produced it, when one was recorded.
    :param settings_hash: digest of the material settings.
    :param spacr_version: the version that produced it.
    :param created_utc: registration time.
    :param created_ns: the same instant as ``time.time_ns()``, for ordering.
    :param size_bytes: bytes on disk at registration.
    :param n_files: files covered.
    :param status: the artifact's own ``complete`` / ``partial`` / ``failed``.
    :param exists: whether the path is still on disk.
    :param state: :data:`STATE_CURRENT`, :data:`STATE_STALE` or
        :data:`STATE_MISSING`.
    :param reasons: sentences from :class:`spacr.artifacts.Staleness`.
    :param causes: the matching machine codes, e.g. ``"upstream-newer"``.
    :param depth: longest distance from a root, i.e. the column to draw it in.
    :param inputs: artifact ids this was derived from, as recorded.
    """

    artifact_id: str
    project: str
    kind: str
    role: str
    module: str
    path: str
    run_id: str = ""
    settings_hash: str = ""
    spacr_version: str = ""
    created_utc: str = ""
    created_ns: int = 0
    size_bytes: int = 0
    n_files: int = 0
    status: str = ""
    exists: bool = True
    state: str = STATE_CURRENT
    reasons: Tuple[str, ...] = ()
    causes: Tuple[str, ...] = ()
    depth: int = 0
    inputs: Tuple[str, ...] = ()

    @property
    def stale(self) -> bool:
        """True when an input or a setting moved on after this was written."""
        return self.state == STATE_STALE or bool(self.causes)

    @property
    def label(self) -> str:
        """Short two-line-able label: the module and what it produced."""
        name = os.path.basename(self.path.rstrip(os.sep)) or self.path
        return f"{self.module}: {name}"

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serializable copy of the node."""
        return {
            "artifact_id": self.artifact_id, "project": self.project,
            "kind": self.kind, "role": self.role, "module": self.module,
            "path": self.path, "run_id": self.run_id,
            "settings_hash": self.settings_hash,
            "spacr_version": self.spacr_version,
            "created_utc": self.created_utc, "created_ns": self.created_ns,
            "size_bytes": self.size_bytes, "n_files": self.n_files,
            "status": self.status, "exists": self.exists, "state": self.state,
            "reasons": list(self.reasons), "causes": list(self.causes),
            "depth": self.depth, "inputs": list(self.inputs),
        }

    def __str__(self) -> str:
        """One line: state, module, kind and path."""
        return f"[{self.state}] {self.module} -> {self.kind} at {self.path}"


@dataclass(frozen=True)
class Edge:
    """One provenance edge: ``source`` was an input to ``target``.

    :param source: artifact id that was consumed.
    :param target: artifact id that was produced from it.
    :param kind: the source's kind, so an edge can be labelled without a
        second lookup.
    :param dangling: the source id is recorded on the target but is no longer
        in the registry. The edge is kept — the fact that something was
        consumed and then forgotten is exactly what makes the target stale,
        and dropping the edge would hide it.
    """

    source: str
    target: str
    kind: str = ""
    dangling: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serializable copy of the edge."""
        return {"source": self.source, "target": self.target,
                "kind": self.kind, "dangling": self.dangling}


@dataclass(frozen=True)
class ModuleGraph:
    """The static module DAG declared by :data:`spacr.ports.PORTS`.

    What feeds what by declaration, independent of whether anything has run.

    :param modules: every module key in the graph, sorted.
    :param edges: ``(producer, consumer)`` pairs.
    :param layers: modules grouped by longest distance from a source.
    :param ran: the subset of ``modules`` that has produced a registered
        artifact in the project this was built alongside; empty for a bare
        :func:`module_graph` call.
    """

    modules: Tuple[str, ...] = ()
    edges: Tuple[Tuple[str, str], ...] = ()
    layers: Tuple[Tuple[str, ...], ...] = ()
    ran: Tuple[str, ...] = ()

    def next_of(self, module: str) -> Tuple[str, ...]:
        """Modules this one can feed, from the declared edges."""
        return tuple(sorted({b for a, b in self.edges if a == module}))

    def previous_of(self, module: str) -> Tuple[str, ...]:
        """Modules that can feed this one, from the declared edges."""
        return tuple(sorted({a for a, b in self.edges if b == module}))

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serializable copy of the module graph."""
        return {"modules": list(self.modules),
                "edges": [list(e) for e in self.edges],
                "layers": [list(row) for row in self.layers],
                "ran": list(self.ran)}


@dataclass(frozen=True)
class PipelineGraph:
    """Every artifact of one project and the edges between them.

    :param project: the project root the graph covers; ``""`` for a registry
        holding several.
    :param nodes: every artifact, in draw order (by depth, then time).
    :param edges: every provenance edge.
    :param layers: node ids grouped by :attr:`Node.depth`.
    :param modules: the static module DAG, with ``ran`` filled in.
    :param generated_utc: when the graph was built.
    :param notes: anything a caller should know — no registry, an empty
        project, a provenance cycle.
    :param registry_file: the registry the graph was read from.
    """

    project: str = ""
    nodes: Tuple[Node, ...] = ()
    edges: Tuple[Edge, ...] = ()
    layers: Tuple[Tuple[str, ...], ...] = ()
    modules: ModuleGraph = field(default_factory=ModuleGraph)
    generated_utc: str = field(default_factory=_utcnow)
    notes: Tuple[str, ...] = ()
    registry_file: str = ""

    def __len__(self) -> int:
        """How many artifacts the graph holds."""
        return len(self.nodes)

    def __bool__(self) -> bool:
        """True when the graph has at least one artifact.

        Spelled out because ``__len__`` alone would make an empty-but-valid
        graph falsy in a way that reads as "the call failed", and the two are
        different: an empty graph is the correct answer for a project that
        has not been run.
        """
        return True

    def node(self, artifact_id: str) -> Optional[Node]:
        """The node with this id, or ``None``."""
        for node in self.nodes:
            if node.artifact_id == artifact_id:
                return node
        return None

    def roots(self) -> Tuple[Node, ...]:
        """Nodes with no registered input — where the project starts."""
        with_inputs = {e.target for e in self.edges}
        return tuple(n for n in self.nodes
                     if n.artifact_id not in with_inputs)

    def leaves(self) -> Tuple[Node, ...]:
        """Nodes nothing was derived from — the current end of the pipeline."""
        consumed = {e.source for e in self.edges}
        return tuple(n for n in self.nodes if n.artifact_id not in consumed)

    def stale_nodes(self) -> Tuple[Node, ...]:
        """Every node that is stale or missing, worst first."""
        flagged = [n for n in self.nodes if n.state != STATE_CURRENT]
        return tuple(sorted(
            flagged, key=lambda n: (STATE_ORDER.index(n.state), -n.created_ns,
                                    n.artifact_id)))

    def downstream(self, artifact_id: str) -> Tuple[Node, ...]:
        """Every node reachable *from* this one, following the edges down.

        The "what does re-running this invalidate?" question, answered from
        the graph already in memory rather than by another registry walk.
        """
        return self._reach(artifact_id, forward=True)

    def upstream(self, artifact_id: str) -> Tuple[Node, ...]:
        """Every node this one was derived from, transitively."""
        return self._reach(artifact_id, forward=False)

    def _reach(self, start: str, *, forward: bool) -> Tuple[Node, ...]:
        """Breadth-first reachability in one direction, cycle-guarded."""
        adjacency: Dict[str, List[str]] = {}
        for edge in self.edges:
            key, value = ((edge.source, edge.target) if forward
                          else (edge.target, edge.source))
            adjacency.setdefault(key, []).append(value)
        seen = {start}
        frontier = list(adjacency.get(start, ()))
        found: List[Node] = []
        while frontier:
            current = frontier.pop(0)
            if current in seen:
                continue
            seen.add(current)
            node = self.node(current)
            if node is not None:
                found.append(node)
            frontier.extend(adjacency.get(current, ()))
        return tuple(sorted(found, key=lambda n: (n.depth, -n.created_ns,
                                                  n.artifact_id)))

    def by_module(self) -> Dict[str, Tuple[Node, ...]]:
        """Nodes grouped by the module that produced them."""
        grouped: Dict[str, List[Node]] = {}
        for node in self.nodes:
            grouped.setdefault(node.module, []).append(node)
        return {module: tuple(rows) for module, rows in sorted(grouped.items())}

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serializable copy of the whole graph."""
        return {
            "project": self.project,
            "generated_utc": self.generated_utc,
            "registry_file": self.registry_file,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "layers": [list(row) for row in self.layers],
            "modules": self.modules.to_dict(),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# The static module DAG
# ---------------------------------------------------------------------------

def module_graph(modules: Optional[Sequence[str]] = None, *,
                 ran: Sequence[str] = ()) -> ModuleGraph:
    """Build the declared module DAG from :data:`spacr.ports.PORTS`.

    An edge exists from ``a`` to ``b`` when ``b`` REQUIRES a kind that ``a``
    produces — the same rule :func:`spacr.ports.next_modules` applies, used
    here for every module at once. Optional consumers are left out on
    purpose: an optional input is a module that *can* read something, and
    drawing those edges turns the picture into a mesh in which the pipeline
    is no longer visible.

    :param modules: restrict to these module keys; default is every key in
        :data:`spacr.ports.PORTS`.
    :param ran: module keys that have actually produced something, recorded
        on the result so a drawing can dim the ones that have not.
    :returns: a :class:`ModuleGraph`.
    """
    keys = tuple(sorted(modules)) if modules else ports.known_modules()
    known = set(keys)
    edges = sorted(
        (module, consumer)
        for module in keys
        for consumer in ports.next_modules(module)
        if consumer in known
    )
    layers = _layer(keys, edges)
    return ModuleGraph(modules=keys, edges=tuple(edges), layers=layers,
                       ran=tuple(sorted(set(ran) & known)))


def _layer(nodes: Sequence[str],
           edges: Sequence[Tuple[str, str]]) -> Tuple[Tuple[str, ...], ...]:
    """Group ``nodes`` into columns by longest distance from a source.

    Longest path rather than shortest: an artifact that a late module reads
    directly from an early one must still be drawn to the right of everything
    between them, or its edges point backwards.

    A cycle cannot be layered — its members have no finite longest distance —
    so anything not settled by the sweep is placed after everything that was.
    That keeps the function total, which matters because the caller is a
    drawing routine and a raised exception there is a blank screen.
    """
    incoming: Dict[str, List[str]] = {n: [] for n in nodes}
    for source, target in edges:
        if target in incoming and source in incoming:
            incoming[target].append(source)

    depth: Dict[str, int] = {}
    remaining = set(nodes)
    while remaining:
        settled = {
            node for node in remaining
            if all(parent in depth for parent in incoming[node])
        }
        if not settled:
            # Everything left is in (or downstream of) a cycle. Park it one
            # column past the deepest thing that resolved.
            base = max(depth.values(), default=-1) + 1
            for node in sorted(remaining):
                depth[node] = base
            break
        for node in settled:
            parents = incoming[node]
            depth[node] = 1 + max((depth[p] for p in parents), default=-1)
        remaining -= settled

    width = max(depth.values(), default=-1) + 1
    return tuple(
        tuple(sorted(n for n in nodes if depth.get(n) == column))
        for column in range(width)
    )


# ---------------------------------------------------------------------------
# The artifact DAG
# ---------------------------------------------------------------------------

def build_graph(project: Union[str, os.PathLike, None] = None, *,
                registry: Optional[Registry] = None,
                settings: Optional[Mapping[str, Any]] = None,
                limit: Optional[int] = None,
                all_projects: bool = False) -> PipelineGraph:
    """Build the provenance graph of one project, with staleness marked.

    :param project: the project root. Ignored when ``registry`` is given and
        ``all_projects`` is True.
    :param registry: an open :class:`spacr.artifacts.Registry`; one is opened
        read-only for ``project`` when this is omitted. A project with no
        registry yields an empty graph carrying a note, never an exception —
        "nothing has been run here" is a legitimate answer.
    :param settings: the settings a caller is about to run with. When given,
        every node is additionally checked against them, so a graph can show
        "this would be stale if you ran now" before anything is overwritten.
        Note that this compares EVERY node's recorded settings against the one
        dict, which is what "would my current settings reproduce this?" means.
    :param limit: cap on how many artifacts to read, newest first.
    :param all_projects: read every project in the registry file rather than
        one. For the shared registry :data:`spacr.artifacts.ARTIFACTS_DB_ENV`
        points at.
    :returns: a :class:`PipelineGraph`.
    """
    notes: List[str] = []
    root = ""
    if project is not None:
        root = os.path.abspath(os.path.expanduser(os.fspath(project)))

    if registry is None:
        target = registry_path(root or None)
        if not os.path.isfile(target):
            return PipelineGraph(
                project=root, registry_file=target,
                modules=module_graph(),
                notes=("No artifact registry at "
                       f"{target} — nothing in this project has registered an "
                       "output yet.",))
        try:
            registry = Registry(path=target, project=root or None, create=False)
        except (FileNotFoundError, OSError) as exc:  # pragma: no cover - rare
            return PipelineGraph(
                project=root, registry_file=target, modules=module_graph(),
                notes=(f"Could not open the artifact registry: {exc}",))

    records = _read_records(registry, root, limit, all_projects)
    if not records:
        notes.append("The registry has no artifacts for this project yet.")

    known = {record.artifact_id: record for record in records}
    edges = _build_edges(records, known)
    depth = _node_depth(records, edges)
    staleness = _staleness(registry, records, settings)

    nodes = tuple(sorted(
        (_node(record, staleness.get(record.artifact_id), depth)
         for record in records),
        key=lambda n: (n.depth, -n.created_ns, n.artifact_id)))

    width = max((n.depth for n in nodes), default=-1) + 1
    layers = tuple(
        tuple(n.artifact_id for n in nodes if n.depth == column)
        for column in range(width))

    dangling = sum(1 for e in edges if e.dangling)
    if dangling:
        notes.append(
            f"{dangling} recorded input(s) are no longer in the registry; the "
            f"artifacts that named them are stale on that account.")

    return PipelineGraph(
        project=root or registry.project,
        nodes=nodes, edges=tuple(edges), layers=layers,
        modules=module_graph(ran=sorted({n.module for n in nodes})),
        notes=tuple(notes), registry_file=registry.path)


def _read_records(registry: Registry, root: str, limit: Optional[int],
                  all_projects: bool) -> List[Artifact]:
    """Every artifact the graph should cover, newest first."""
    if all_projects:
        return list(registry.all(limit=limit))
    return list(registry.by_project(root or None, limit=limit))


def _build_edges(records: Sequence[Artifact],
                 known: Mapping[str, Artifact]) -> List[Edge]:
    """One :class:`Edge` per recorded input, dangling ones included."""
    edges: List[Edge] = []
    for record in records:
        for input_id in record.inputs:
            upstream = known.get(input_id)
            edges.append(Edge(
                source=input_id, target=record.artifact_id,
                kind=upstream.kind if upstream is not None else "",
                dangling=upstream is None))
    return edges


def _node_depth(records: Sequence[Artifact],
                edges: Sequence[Edge]) -> Dict[str, int]:
    """Longest distance from a root, per artifact id.

    Dangling edges are skipped: a node whose only input has been forgotten
    still starts the picture, and hanging it off an id with no node would
    give it a parent that cannot be drawn.
    """
    ids = [record.artifact_id for record in records]
    present = set(ids)
    pairs = [(e.source, e.target) for e in edges
             if not e.dangling and e.source in present and e.target in present]
    layers = _layer(ids, pairs)
    return {node_id: column
            for column, row in enumerate(layers) for node_id in row}


def _staleness(registry: Registry, records: Sequence[Artifact],
               settings: Optional[Mapping[str, Any]]
               ) -> Dict[str, Staleness]:
    """Ask the registry about every artifact once.

    One call per artifact is what :meth:`spacr.artifacts.Registry.is_stale`
    offers, and it is what this does. A failure on one artifact is recorded
    as "unknown" for that artifact rather than losing the whole graph: the
    registry is shared with running jobs, and a row can go while this walks.
    """
    verdicts: Dict[str, Staleness] = {}
    for record in records:
        try:
            verdicts[record.artifact_id] = registry.is_stale(
                record.artifact_id, settings=settings)
        except Exception:  # pragma: no cover - registry raced us
            verdicts[record.artifact_id] = Staleness(
                record.artifact_id, False,
                ("Could not check this artifact's provenance.",), ())
    return verdicts


def _node(record: Artifact, verdict: Optional[Staleness],
          depth: Mapping[str, int]) -> Node:
    """Turn one registry row plus its verdict into a :class:`Node`."""
    exists = record.exists
    reasons = tuple(verdict.reasons) if verdict is not None else ()
    causes = tuple(verdict.causes) if verdict is not None else ()
    stale = bool(verdict) if verdict is not None else False
    if not exists:
        state = STATE_MISSING
    elif stale:
        state = STATE_STALE
    else:
        state = STATE_CURRENT
    return Node(
        artifact_id=record.artifact_id, project=record.project,
        kind=record.kind, role=record.role, module=record.module,
        path=record.path, run_id=record.run_id,
        settings_hash=record.settings_hash,
        spacr_version=record.spacr_version, created_utc=record.created_utc,
        created_ns=record.created_ns, size_bytes=record.size_bytes,
        n_files=record.n_files, status=record.status, exists=exists,
        state=state, reasons=reasons, causes=causes,
        depth=int(depth.get(record.artifact_id, 0)),
        inputs=tuple(record.inputs))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def stale_summary(graph: PipelineGraph) -> Dict[str, Any]:
    """Counts and cause tallies for a graph, for a one-line verdict.

    :param graph: the graph to summarise.
    :returns: ``n_nodes``, ``n_edges``, ``n_current``, ``n_stale``,
        ``n_missing``, ``causes`` (``{code: count}``), ``modules`` and
        ``verdict`` — one sentence fit for a status bar.
    """
    counts = {state: 0 for state in STATE_ORDER}
    causes: Dict[str, int] = {}
    for node in graph.nodes:
        counts[node.state] = counts.get(node.state, 0) + 1
        for cause in node.causes:
            causes[cause] = causes.get(cause, 0) + 1
    n_stale = counts[STATE_STALE]
    n_missing = counts[STATE_MISSING]
    if not graph.nodes:
        verdict = "Nothing registered yet."
    elif n_missing and n_stale:
        verdict = (f"{n_missing} artifact(s) missing and {n_stale} stale of "
                   f"{len(graph.nodes)}.")
    elif n_missing:
        verdict = f"{n_missing} of {len(graph.nodes)} artifact(s) missing."
    elif n_stale:
        verdict = f"{n_stale} of {len(graph.nodes)} artifact(s) stale."
    else:
        verdict = f"All {len(graph.nodes)} artifact(s) current."
    return {
        "n_nodes": len(graph.nodes), "n_edges": len(graph.edges),
        "n_current": counts[STATE_CURRENT], "n_stale": n_stale,
        "n_missing": n_missing, "causes": dict(sorted(causes.items())),
        "modules": sorted({n.module for n in graph.nodes}),
        "verdict": verdict,
    }


def format_graph(graph: PipelineGraph, *, width: int = 100) -> str:
    """Render a graph as text: one block per column, edges named.

    The headless counterpart of the screen, and what the tests read. No
    trailing newline.

    :param graph: the graph to render.
    :param width: soft wrap for a path; longer ones are elided in the middle.
    """
    summary = stale_summary(graph)
    lines = [f"Pipeline graph — {graph.project or '(all projects)'}",
             f"  {summary['verdict']}"]
    for note in graph.notes:
        lines.append(f"  note: {note}")
    if not graph.nodes:
        declared = " -> ".join(
            "/".join(row) for row in graph.modules.layers) or "(none)"
        lines.append("")
        lines.append(f"  Declared module order: {declared}")
        return "\n".join(lines)

    incoming: Dict[str, List[Edge]] = {}
    for edge in graph.edges:
        incoming.setdefault(edge.target, []).append(edge)

    for column, row in enumerate(graph.layers):
        lines.append("")
        lines.append(f"  Step {column + 1}")
        for artifact_id in row:
            node = graph.node(artifact_id)
            if node is None:  # pragma: no cover - layers come from nodes
                continue
            mark = {STATE_CURRENT: "ok", STATE_STALE: "STALE",
                    STATE_MISSING: "MISSING"}[node.state]
            lines.append(
                f"    [{mark}] {node.module} -> {node.kind} "
                f"({node.role or 'output'})")
            lines.append(f"        {_elide(node.path, width)}")
            if node.run_id:
                lines.append(f"        run {node.run_id}"
                             f"  spaCR {node.spacr_version or '?'}")
            for edge in incoming.get(artifact_id, ()):
                source = graph.node(edge.source)
                where = (f"{source.module} {source.kind}" if source is not None
                         else f"{edge.source} (forgotten)")
                lines.append(f"        <- {where}")
            for reason in node.reasons:
                lines.append(f"        ! {reason}")
    return "\n".join(lines)


def _elide(text: str, width: int) -> str:
    """Shorten a path from the middle so both ends stay readable."""
    text = str(text)
    if width <= 8 or len(text) <= width:
        return text
    keep = (width - 3) // 2
    return f"{text[:keep]}...{text[-keep:]}"


def to_dot(graph: PipelineGraph) -> str:
    """Render a graph as Graphviz DOT, coloured by state.

    Not used by the GUI — it draws itself — but it is what a user pastes into
    a methods figure, and it is the cheapest way to eyeball a graph while
    developing. No trailing newline.
    """
    lines = ["digraph spacr {", "  rankdir=LR;",
             '  node [shape=box, style="rounded,filled", fontname="Helvetica"];']
    colours = {STATE_CURRENT: "#d9ead3", STATE_STALE: "#fce5cd",
               STATE_MISSING: "#f4cccc"}
    for node in graph.nodes:
        label = (f"{node.module}\\n{node.kind}\\n"
                 f"{os.path.basename(node.path.rstrip(os.sep)) or node.path}")
        lines.append(
            f'  "{node.artifact_id}" [label="{label}", '
            f'fillcolor="{colours.get(node.state, "#ffffff")}"];')
    for edge in graph.edges:
        style = ' [style=dashed, color="#cc0000"]' if edge.dangling else ""
        lines.append(f'  "{edge.source}" -> "{edge.target}"{style};')
    lines.append("}")
    return "\n".join(lines)
