"""``N4`` — every project on disk in one place: stage, size, last run, staleness.

Until now the only way to see the shape of your work was to navigate by
folder. Which plates have been measured? Which finished last week and have
since had their masks re-run underneath them? Which one is the 400 GB that
filled the disk? Each of those questions had an answer somewhere in spaCR and
none of them had a *list*.

This module is that list, and it is deliberately thin: everything it reports
already existed, and re-deriving any of it would be how the browser and the
rest of spaCR start disagreeing.

======================  =====================================================
What the browser shows  Where the answer comes from
======================  =====================================================
stage reached           :func:`spacr.ports.declared_outputs` — which of a
                        module's *declared* outputs are on disk. Not a
                        hand-written ladder of stage names: the order is
                        topologically sorted out of the port graph, so a
                        module registered by a plugin takes its place in it
                        without this file being edited.
size, and what of it    :func:`spacr.data_manager.scan_project`. ONE walk of
is unaccounted for      the tree, per-kind attribution, and the registry
                        reconciled against the filesystem. There is no second
                        walker here, and there must not be one.
last run                the registry's newest ``created_ns`` for the project;
                        failing that, the mtime of the outputs themselves.
                        Which of the two answered is reported, because
                        "spaCR recorded this run" and "something wrote into
                        this folder" are different claims.
what is stale           :meth:`spacr.artifacts.Registry.is_stale`, whose
                        :class:`~spacr.artifacts.Staleness` already carries
                        the reasons, the machine cause codes and the
                        separate ``missing`` flag.
what to run next        :func:`spacr.chaining.next_steps`, so an offer here
                        is the same offer, with the same readiness check,
                        that the module's own screen makes.
======================  =====================================================

The project the registry has never seen
---------------------------------------

This is the case the browser exists for and the one that is easy to get
wrong. A user copies a colleague's plate folder onto the machine and opens
spaCR: a browser that lists only what spaCR itself recorded shows them
nothing, which is worse than useless because it looks like an answer.

So a project is found on disk (:func:`discover`, :func:`looks_like_project`)
rather than read out of a registry, and an unrecorded one appears with
everything the filesystem can answer — stage, size, files, last write — and
with :attr:`ProjectSummary.known` False. What it must NOT do is report
"0 stale", which reads as *clean*. With no provenance there is nothing to
compare against, so staleness is **unknown**, and
:attr:`ProjectSummary.staleness_known` is the flag that says so.
:meth:`ProjectSummary.staleness_note` puts it in words for a table cell.

Nothing here imports Qt, so the same summary is available from a notebook and
from a test, and the browser screen is a renderer of it.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import data_manager as _dm
from . import ports as _ports
from .artifacts import Artifact, Registry, Staleness
from .data_manager import ProjectUsage, human_bytes

__all__ = [
    "MTIME_SAMPLE",
    "ModuleState",
    "ProjectSummary",
    "SOURCE_FILESYSTEM",
    "SOURCE_REGISTRY",
    "STATE_ABSENT",
    "STATE_DONE",
    "STATE_PARTIAL",
    "StaleArtifact",
    "browse",
    "discover",
    "evidence_ports",
    "format_project",
    "format_projects",
    "looks_like_project",
    "module_states",
    "pipeline_order",
    "producing_modules",
    "scan",
]

LOG = logging.getLogger("spacr.projects")

#: The module produced everything it declares as required.
STATE_DONE = "done"
#: Some declared outputs are there and some are not — an interrupted run, or
#: one whose intermediates were cleaned up afterwards.
STATE_PARTIAL = "partial"
#: Nothing this module declares is on disk. It has not run here.
STATE_ABSENT = "absent"

#: :attr:`ProjectSummary.last_run_source` — spaCR recorded the run.
SOURCE_REGISTRY = "registry"
#: :attr:`ProjectSummary.last_run_source` — nobody recorded anything; the
#: timestamp is the mtime of the outputs themselves.
SOURCE_FILESYSTEM = "filesystem"

#: How many matched paths a patterned port contributes to the "last written"
#: answer. The port's own target (the ``merged/`` folder, the database file)
#: is always stated; a folder's mtime moves when a file is added to it but
#: not when one is rewritten in place, so a sample of its contents is taken
#: as well. Capped because a merged folder holds one file per field and
#: stat-ing forty thousand of them to put a date in a table cell is not a
#: trade anyone would make.
MTIME_SAMPLE = 32

#: How deep :func:`discover` descends looking for projects. Two is the shape
#: people actually have: a folder of experiments, each holding plates.
DEFAULT_DEPTH = 2

#: Folder names never descended into. Not a guess about the user's layout —
#: each is somewhere spaCR itself writes, and a project's ``merged/`` is part
#: of that project rather than a project of its own.
SKIP_DIRS = frozenset({
    "merged", "masks", "stack", "orig", "consolidated", "measurements",
    "settings", "data", "datasets", "models", "figure", "results",
    "__pycache__", ".git", ".ipynb_checkpoints",
})


def _now() -> str:
    """UTC, ISO-8601, seconds — the stamp :mod:`spacr.data_manager` uses."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp(nanoseconds: int) -> str:
    """Render ``time_ns``-style nanoseconds as UTC ISO-8601, or ``""``."""
    if not nanoseconds:
        return ""
    return datetime.fromtimestamp(nanoseconds / 1e9,
                                  tz=timezone.utc).isoformat(
                                      timespec="seconds")


# ---------------------------------------------------------------------------
# The pipeline order, read off the port graph
# ---------------------------------------------------------------------------

def producing_modules() -> Tuple[str, ...]:
    """Every declared module that writes something, sorted.

    A module that only *reads* — the analysis apps that open
    ``measurements.db`` — cannot be a stage a project has reached, because it
    leaves nothing behind for anyone to find.
    """
    return tuple(sorted(key for key, spec in _ports.PORTS.items()
                        if spec.produces))


def pipeline_order() -> Tuple[str, ...]:
    """Producing modules, upstream first.

    A topological sort of :data:`spacr.ports.PORTS` using
    :func:`spacr.ports.upstream_modules`, with an alphabetical tie-break so
    the answer is stable between runs. **Derived, not written down**: a module
    that registers its ports through
    :func:`spacr.ports.register_module_ports` — a plugin, or a module written
    after this one — takes its place in the ladder without this file changing.

    A cycle cannot happen with the declared graph and is not an error worth
    refusing over, so one is broken alphabetically and logged: a browser that
    raised rather than showing a slightly odd ordering would be a worse tool.
    """
    nodes = set(producing_modules())
    pending: Dict[str, set] = {
        node: {up for up in _ports.upstream_modules(node) if up in nodes}
        for node in nodes
    }
    order: List[str] = []
    while pending:
        ready = sorted(node for node, deps in pending.items()
                       if not (deps & set(pending)))
        if not ready:
            ready = [sorted(pending)[0]]
            LOG.debug("port graph has a cycle; breaking it at %s", ready[0])
        for node in ready:
            order.append(node)
            pending.pop(node)
    return tuple(order)


# ---------------------------------------------------------------------------
# Stage reached
# ---------------------------------------------------------------------------

def _newest_mtime_ns(resolved: "_ports.ResolvedPort") -> int:
    """When this port was last written, in ``time_ns`` units. 0 if absent.

    The target itself plus at most :data:`MTIME_SAMPLE` of the paths that
    matched it — see that constant for why the sample is capped.
    """
    newest = 0.0
    candidates: List[str] = []
    if resolved.target:
        candidates.append(resolved.target)
    candidates.extend(sorted(resolved.paths)[-MTIME_SAMPLE:])
    for path in candidates:
        try:
            newest = max(newest, os.stat(path).st_mtime)
        except OSError:
            continue
    return int(newest * 1_000_000_000)


def _location(port: "_ports.Port") -> str:
    """The file or folder a port sits in. ``"."`` for the project root."""
    return port.path or "."


def evidence_ports(module: str) -> Tuple["_ports.Port", ...]:
    """The produced ports whose existence PROVES ``module`` ran here.

    The naive reading of "stage reached" — every declared output that is on
    disk — is wrong, and wrong in a way a user would notice immediately.
    ``ml_analyze`` declares one output, ``measurements/measurements.db``,
    which ``mask`` created and ``measure`` filled. On that reading, a project
    that has only been segmented reports classical ML as complete.

    So a produced port counts as evidence only when nothing EARLIER in
    :func:`pipeline_order` already explains the same location with a
    different kind. An earlier writer's file being there says the earlier
    module ran, and nothing more. This is the same distinction
    :mod:`spacr.artifacts` draws when it keeps ``object-counts`` and
    ``measurements-db`` apart in one file — "mask creating the file does not
    make the measurements in it current".

    Two consequences worth stating rather than discovering:

    * ``measure`` is left with only its optional ``data/`` crops, because
      mask wrote the database first. A measured project whose crops were
      never saved therefore reports its stage as ``mask`` **from the
      filesystem alone**. The registry answers it exactly when it has a
      record — see :func:`module_states`.
    * A module with no evidence ports at all (``ml_analyze``) has no on-disk
      signature whatsoever. :attr:`ModuleState.detectable` is False for it,
      so the browser can say "no way to tell without a run record" instead of
      the flatly wrong "not run".

    Modules that declare an identical set of outputs are interchangeable
    (``mask`` and ``timelapse`` share theirs literally — ``ports.py`` passes
    the same tuple to both). Only the earlier of such a pair carries
    filesystem evidence, or one segmented project would report that both had
    run.

    :param module: module key or alias.
    :returns: the subset of ``module_ports(module).produces`` that is
        discriminating, in declaration order.
    """
    spec = _ports.module_ports(module)
    order = pipeline_order()
    position = order.index(spec.key) if spec.key in order else len(order)
    earlier = order[:position]
    for other in earlier:
        if _ports.PORTS[other].produces == spec.produces:
            return ()
    claimed: Dict[str, set] = {}
    for other in earlier:
        for port in _ports.PORTS[other].produces:
            claimed.setdefault(_location(port), set()).add(port.kind)
    return tuple(port for port in spec.produces
                 if not (claimed.get(_location(port), set()) - {port.kind}))


@dataclass(frozen=True)
class ModuleState:
    """Whether one module has run in a project, judged by what it declares.

    :param module: the module key.
    :param state: :data:`STATE_DONE`, :data:`STATE_PARTIAL` or
        :data:`STATE_ABSENT`.
    :param found: roles whose declared output is on disk — or, when the
        answer came from the registry, the roles it recorded.
    :param missing: *required* roles whose declared output is not there. An
        optional output that was cleaned up (``masks/``) is not missing —
        the declaration already says it may legitimately be absent, and
        reporting it would make every tidied project look broken.
    :param optional_missing: optional roles that are absent, kept separate so
        the difference stays visible without being alarming.
    :param newest_ns: when the newest of the found outputs was written.
    :param evidence: what answered — :data:`SOURCE_FILESYSTEM`,
        :data:`SOURCE_REGISTRY`, or ``""`` when nothing did.
    :param detectable: whether this module has any on-disk signature at all.
        See :func:`evidence_ports`. False plus :data:`STATE_ABSENT` means
        "unknown", not "no".
    """

    module: str
    state: str
    found: Tuple[str, ...] = ()
    missing: Tuple[str, ...] = ()
    optional_missing: Tuple[str, ...] = ()
    newest_ns: int = 0
    evidence: str = ""
    detectable: bool = True

    @property
    def ran(self) -> bool:
        """True when this module left anything at all behind."""
        return self.state != STATE_ABSENT

    def describe(self) -> str:
        """One line: the module, its state and what is short."""
        if self.state == STATE_ABSENT:
            if not self.detectable:
                return (f"{self.module}: no record, and it writes nothing "
                        f"only it could have written")
            return f"{self.module}: not run here"
        when = _stamp(self.newest_ns)
        tail = f", last written {when}" if when else ""
        via = (" (from the run record)" if self.evidence == SOURCE_REGISTRY
               else "")
        if self.state == STATE_DONE:
            return (f"{self.module}: complete "
                    f"({', '.join(self.found)}){tail}{via}")
        return (f"{self.module}: partial — has {', '.join(self.found)}, "
                f"missing {', '.join(self.missing)}{tail}{via}")


def module_states(root: Any, *, modules: Sequence[str] = (),
                  records: Sequence[Artifact] = ()
                  ) -> Tuple[ModuleState, ...]:
    """Judge every producing module against one project folder.

    Two sources, in order of authority. A run record in ``records`` settles
    the question outright — the registry saw the run happen. Without one,
    the answer comes from the discriminating outputs on disk (see
    :func:`evidence_ports`), which is what makes a project spaCR has never
    seen readable at all.

    :param root: the project root.
    :param modules: which modules to judge, in the order to report them.
        Defaults to :func:`pipeline_order`.
    :param records: artifacts the registry holds for this project.
    :returns: one :class:`ModuleState` per module, in that order.
    """
    project = os.path.abspath(os.path.expanduser(os.fspath(root)))
    recorded: Dict[str, List[Artifact]] = {}
    for record in records:
        recorded.setdefault(record.module, []).append(record)

    states: List[ModuleState] = []
    for module in (tuple(modules) or pipeline_order()):
        try:
            wanted = evidence_ports(module)
        except _ports.UnknownModule:
            continue
        resolved_root = project
        found: List[str] = []
        missing: List[str] = []
        optional_missing: List[str] = []
        newest = 0
        for port in wanted:
            resolved = _ports.resolve_port(port, resolved_root)
            if resolved.exists:
                found.append(port.role)
                newest = max(newest, _newest_mtime_ns(resolved))
            elif port.required:
                missing.append(port.role)
            else:
                optional_missing.append(port.role)
        if not found:
            state, evidence = STATE_ABSENT, ""
        elif missing:
            state, evidence = STATE_PARTIAL, SOURCE_FILESYSTEM
        else:
            state, evidence = STATE_DONE, SOURCE_FILESYSTEM

        rows = recorded.get(module) or []
        if rows and state == STATE_ABSENT:
            # The registry watched this run finish. Whether its outputs are
            # still on disk is a different question, and the one PARTIAL is
            # for.
            alive = [row for row in rows if row.exists]
            state = STATE_DONE if len(alive) == len(rows) else STATE_PARTIAL
            evidence = SOURCE_REGISTRY
            found = sorted({row.role for row in alive})
            missing = sorted({row.role for row in rows if not row.exists})
            newest = max(row.created_ns for row in rows)

        states.append(ModuleState(
            module=module, state=state, found=tuple(found),
            missing=tuple(missing), optional_missing=tuple(optional_missing),
            newest_ns=newest, evidence=evidence, detectable=bool(wanted)))
    return tuple(states)


# ---------------------------------------------------------------------------
# Finding projects
# ---------------------------------------------------------------------------

def looks_like_project(root: Any) -> bool:
    """Whether a folder is a spaCR project, judged without the registry.

    True when a registry file sits in it, or when any declared *output* of any
    producing module is on disk, or when the mask pipeline's declared *input*
    finds raw images there. That last clause is what makes a plate folder
    somebody just copied in a project: nothing has been run on it yet, and it
    is exactly the folder a user wants the browser to list.

    Reusing the port declarations rather than testing for ``merged/`` by name
    means a plugin's module makes its own outputs count as evidence.
    """
    project = os.path.abspath(os.path.expanduser(os.fspath(root)))
    if not os.path.isdir(project):
        return False
    if os.path.isfile(os.path.join(project, _dm.ARCHIVE_MANIFEST_NAME)):
        return True
    from .artifacts import ARTIFACTS_DB_NAME
    if os.path.isfile(os.path.join(project, ARTIFACTS_DB_NAME)):
        return True
    for state in module_states(project):
        if state.ran:
            return True
    # No output anywhere: is there raw data waiting to have something run on
    # it? The mask pipeline's own input declaration answers that.
    try:
        inputs = _ports.declared_inputs("mask", root=project)
    except _ports.UnknownModule:            # pragma: no cover - always declared
        return False
    return any(resolved.exists for resolved in inputs)


def discover(roots: Iterable[Any], *, depth: int = DEFAULT_DEPTH,
             limit: int = 500) -> Tuple[str, ...]:
    """Find project folders under ``roots``.

    Descent **stops at a project**: a project's ``merged/`` holds ``.npy``
    files and would otherwise be reported as a project of its own, which is
    both wrong and the kind of wrong that fills a table with noise.

    :param roots: folders to search. A root that is itself a project is
        returned as one.
    :param depth: how many levels below each root to look. 0 means "test the
        roots themselves and nothing under them".
    :param limit: stop after this many projects. A browser pointed at a home
        directory must return *something* rather than walk a filesystem.
    :returns: absolute paths, sorted, de-duplicated.
    """
    found: List[str] = []
    seen: set = set()

    def _walk(path: str, level: int) -> None:
        if len(found) >= limit:
            return
        real = os.path.realpath(path)
        if real in seen:
            return
        seen.add(real)
        if looks_like_project(path):
            found.append(os.path.abspath(path))
            return
        if level >= depth:
            return
        try:
            entries = sorted(os.scandir(path), key=lambda e: e.name)
        except OSError as exc:
            LOG.debug("cannot list %s: %s", path, exc)
            return
        for entry in entries:
            if len(found) >= limit:
                return
            name = entry.name
            if name.startswith(".") or name in SKIP_DIRS:
                continue
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:                  # pragma: no cover - race
                continue
            _walk(entry.path, level + 1)

    for root in roots:
        if not root:
            continue
        _walk(os.path.abspath(os.path.expanduser(os.fspath(root))), 0)
    return tuple(sorted(set(found)))


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StaleArtifact:
    """One registered result that no longer matches what made it.

    A thin projection of :class:`spacr.artifacts.Staleness` onto the artifact
    it describes, so a table row has the kind and the producing module beside
    the reason without the caller re-querying the registry.

    :param artifact_id: the registry id.
    :param kind: a :mod:`spacr.ports` kind.
    :param module: the module that produced it.
    :param role: its port role.
    :param path: where it is, or was.
    :param reasons: the registry's own sentences.
    :param causes: the machine cause codes, e.g. ``"upstream-newer"``.
    :param missing: the file is gone. An availability problem, not a
        provenance one — :class:`spacr.artifacts.Staleness` keeps the two
        apart and so does this.
    """

    artifact_id: str
    kind: str
    module: str
    role: str
    path: str
    reasons: Tuple[str, ...] = ()
    causes: Tuple[str, ...] = ()
    missing: bool = False

    def explain(self) -> str:
        """The causes as one readable clause, via :mod:`spacr.chaining`."""
        from .chaining import explain_causes
        return explain_causes(self.causes)

    def describe(self) -> str:
        """One line, fit for a list under a project."""
        head = f"{self.kind} from {self.module}"
        if self.missing:
            return f"{head}: gone from {self.path}"
        why = "; ".join(self.reasons) or self.explain() or "out of date"
        return f"{head}: {why}"


@dataclass(frozen=True)
class ProjectSummary:
    """One project, as the browser lists it.

    :param root: absolute project root.
    :param name: its folder name — what the table's first column shows.
    :param exists: the folder is there. A registry can outlive the data.
    :param known: **the registry holds at least one artifact for it.** False
        for the folder a user just copied in, and the reason
        :attr:`staleness_known` exists.
    :param has_registry: a registry *file* sits in the project. True with
        ``known`` False when the file exists but records another project —
        the shared-registry case (:data:`spacr.artifacts.ARTIFACTS_DB_ENV`).
    :param stage: the furthest module in :func:`pipeline_order` that left
        anything behind, or ``""`` for a project nothing has been run on.
    :param modules: every producing module's :class:`ModuleState`, in
        pipeline order.
    :param size_bytes: every byte under the root, from
        :func:`spacr.data_manager.scan_project`.
    :param n_files: how many files that is.
    :param unregistered_bytes: of those, how many nothing claims.
    :param unregistered_files: and how many files.
    :param n_artifacts: registry rows for this project.
    :param last_run_ns: when it last produced something.
    :param last_run_utc: the same instant, ISO-8601, or ``""``.
    :param last_run_source: :data:`SOURCE_REGISTRY` or
        :data:`SOURCE_FILESYSTEM` — *which* question was answered. They are
        not the same claim and a browser that blurs them is lying by omission.
    :param stale: results that no longer match their inputs.
    :param missing: registered results whose file is gone.
    :param next_steps: modules that could run next, ready ones first, as
        ``(module, blocked_reason)``; the reason is ``""`` when it can run.
    :param errors: paths that could not be read.
    :param scanned_utc: when this summary was taken.
    :param usage: the full :class:`spacr.data_manager.ProjectUsage`, for a
        detail pane that wants the per-kind breakdown.
    """

    root: str
    name: str = ""
    exists: bool = True
    known: bool = False
    has_registry: bool = False
    stage: str = ""
    modules: Tuple[ModuleState, ...] = ()
    size_bytes: int = 0
    n_files: int = 0
    unregistered_bytes: int = 0
    unregistered_files: int = 0
    n_artifacts: int = 0
    last_run_ns: int = 0
    last_run_utc: str = ""
    last_run_source: str = ""
    stale: Tuple[StaleArtifact, ...] = ()
    missing: Tuple[StaleArtifact, ...] = ()
    next_steps: Tuple[Tuple[str, str], ...] = ()
    errors: Tuple[str, ...] = ()
    scanned_utc: str = ""
    usage: Optional[ProjectUsage] = field(default=None, repr=False,
                                          compare=False)

    # -- stage --------------------------------------------------------------
    @property
    def stage_label(self) -> str:
        """The stage as a table cell: the module, and whether it finished."""
        if not self.stage:
            return "nothing run"
        for state in self.modules:
            if state.module == self.stage:
                return (self.stage if state.state == STATE_DONE
                        else f"{self.stage} (partial)")
        return self.stage                    # pragma: no cover - unreachable

    @property
    def ran(self) -> Tuple[str, ...]:
        """Every module that left something behind, in pipeline order."""
        return tuple(s.module for s in self.modules if s.ran)

    # -- staleness ----------------------------------------------------------
    @property
    def n_stale(self) -> int:
        """How many recorded results are out of date."""
        return len(self.stale)

    @property
    def staleness_known(self) -> bool:
        """Whether "is anything stale?" has an answer at all.

        False for a project the registry has never seen. Reporting such a
        project as having zero stale artifacts would read as *clean*, and it
        is not clean — it is unexamined.
        """
        return self.known

    def staleness_note(self) -> str:
        """What the stale column should say, in words."""
        if not self.staleness_known:
            return "unknown — nothing recorded"
        if self.missing and self.stale:
            return (f"{len(self.stale)} stale, "
                    f"{len(self.missing)} gone")
        if self.missing:
            return f"{len(self.missing)} gone"
        if self.stale:
            return f"{len(self.stale)} stale"
        return "current"

    def note(self) -> str:
        """The one thing about this project worth saying in a table.

        Ordered by what a user has to act on: a folder that is gone, then a
        project spaCR has no record of, then results that no longer match,
        then a large pile of bytes nobody claims.
        """
        if not self.exists:
            return "folder is gone"
        if not self.known:
            return ("not in the registry — found on disk"
                    if self.stage else
                    "not in the registry — nothing run yet")
        if self.missing:
            return f"{len(self.missing)} recorded result(s) no longer on disk"
        if self.stale:
            return f"{len(self.stale)} result(s) out of date"
        if self.unregistered_bytes > self.size_bytes // 2 and self.size_bytes:
            return (f"{human_bytes(self.unregistered_bytes)} unaccounted for")
        return ""

    def describe(self) -> str:
        """The full report; see :func:`format_project`."""
        return format_project(self)

    def __str__(self) -> str:
        return format_project(self)


def _registry_for(project: str,
                  registry: Optional[Registry]) -> Optional[Registry]:
    """The registry to read this project through, or None.

    Delegates to :mod:`spacr.data_manager` so the browser's answer to "does
    spaCR have a record of this?" is the *same* answer
    :func:`spacr.data_manager.scan_project` used to decide what counts as
    unregistered. A second, independently-derived registry lookup here is
    exactly how the two would come to disagree about one project.
    """
    return _dm._open_if_present(registry, project)


def _stale_of(registry: Registry, records: Sequence[Artifact]
              ) -> Tuple[Tuple[StaleArtifact, ...], Tuple[StaleArtifact, ...]]:
    """Split ``records`` into (stale, missing) projections."""
    stale: List[StaleArtifact] = []
    gone: List[StaleArtifact] = []
    for record in records:
        try:
            verdict: Staleness = registry.is_stale(record)
        except Exception as exc:             # pragma: no cover - corrupt row
            LOG.debug("staleness check failed for %s: %s",
                      record.artifact_id, exc)
            continue
        projection = StaleArtifact(
            artifact_id=record.artifact_id, kind=record.kind,
            module=record.module, role=record.role, path=record.path,
            reasons=tuple(verdict.reasons), causes=tuple(verdict.causes),
            missing=bool(verdict.missing))
        if verdict.missing:
            gone.append(projection)
        if verdict.stale:
            stale.append(projection)
    return tuple(stale), tuple(gone)


def _next_steps(stage: str, project: str,
                registry: Optional[Registry]) -> Tuple[Tuple[str, str], ...]:
    """What can run next after ``stage``, as ``(module, blocked_reason)``.

    Asked of :func:`spacr.chaining.next_steps` rather than answered here, so
    the browser offers what the module's own screen offers and refuses for
    the same reasons.
    """
    if not stage:
        return ()
    try:
        from .chaining import next_steps
        steps = next_steps(stage, root=project, registry=registry)
    except Exception as exc:
        LOG.debug("next steps for %s in %s failed: %s", stage, project, exc)
        return ()
    return tuple((step.module, step.blocked) for step in steps)


def scan(root: Any, *, registry: Optional[Registry] = None,
         usage: Optional[ProjectUsage] = None,
         with_next_steps: bool = True) -> ProjectSummary:
    """Summarise one project. The whole of :mod:`spacr.projects` in one call.

    :param root: the project root. It does not have to be a project, and it
        does not have to be in the registry — that is the point.
    :param registry: an open registry to read through. Omit and the project's
        own is opened when a file exists; a project with none is summarised
        anyway, with :attr:`ProjectSummary.known` False.
    :param usage: a :class:`spacr.data_manager.ProjectUsage` already taken for
        this root, to be reused rather than re-walked. The browser has none;
        the Data Manager screen, which has just walked the same project, does.
    :param with_next_steps: compute what could run next. Off skips one
        :func:`spacr.ports.check_ready` per successor, which is the only part
        of this that globs a second time.
    :returns: a :class:`ProjectSummary`. Never raises for a folder that is
        missing, unreadable or not a project: "there is nothing here" is an
        answer a browser has to be able to show.
    """
    project = os.path.abspath(os.path.expanduser(os.fspath(root)))
    name = os.path.basename(project.rstrip(os.sep)) or project
    scanned = _now()
    if not os.path.isdir(project):
        return ProjectSummary(root=project, name=name, exists=False,
                              scanned_utc=scanned)

    if usage is None:
        try:
            usage = _dm.scan_project(project, registry=registry)
        except _dm.DataManagerError as exc:  # pragma: no cover - raced delete
            LOG.debug("cannot measure %s: %s", project, exc)
            return ProjectSummary(root=project, name=name, exists=False,
                                  scanned_utc=scanned)

    # The registry first: it is the authority on what ran, and
    # `module_states` takes it into account rather than guessing from files
    # alone wherever it has an answer.
    store = _registry_for(project, registry)
    records: List[Artifact] = []
    if store is not None:
        try:
            records = list(store.by_project(project))
        except Exception as exc:             # pragma: no cover - locked db
            LOG.debug("cannot read the registry for %s: %s", project, exc)

    states = module_states(project, records=records)
    stage = ""
    for state in states:
        if state.ran:
            stage = state.module

    stale: Tuple[StaleArtifact, ...] = ()
    gone: Tuple[StaleArtifact, ...] = ()
    last_ns = 0
    source = ""
    if records:
        stale, gone = _stale_of(store, records)
        last_ns = max(record.created_ns for record in records)
        source = SOURCE_REGISTRY
    else:
        # Nothing recorded. The outputs themselves still carry a date, and it
        # is a weaker claim reported as a weaker claim rather than dressed up
        # as a run record.
        last_ns = max((state.newest_ns for state in states), default=0)
        if last_ns:
            source = SOURCE_FILESYSTEM

    return ProjectSummary(
        root=project, name=name, exists=True,
        known=bool(records),
        has_registry=store is not None,
        stage=stage, modules=states,
        size_bytes=usage.total_bytes, n_files=usage.total_files,
        unregistered_bytes=usage.unregistered_bytes,
        unregistered_files=usage.unregistered_files,
        n_artifacts=len(records),
        last_run_ns=last_ns, last_run_utc=_stamp(last_ns),
        last_run_source=source,
        stale=stale, missing=gone,
        next_steps=(_next_steps(stage, project, store)
                    if with_next_steps else ()),
        errors=tuple(usage.errors), scanned_utc=scanned, usage=usage)


def browse(roots: Iterable[Any], *, depth: int = DEFAULT_DEPTH,
           registry: Optional[Registry] = None,
           limit: int = 500,
           with_next_steps: bool = True,
           on_progress: Optional[Any] = None) -> Tuple[ProjectSummary, ...]:
    """Find every project under ``roots`` and summarise each. The browser.

    :param roots: folders to search, or the projects themselves.
    :param depth: how deep to look; see :func:`discover`.
    :param registry: an open registry to read every project through — the
        shared-registry case. Omit and each project's own is used.
    :param limit: stop after this many projects.
    :param with_next_steps: passed to :func:`scan`.
    :param on_progress: optional ``fn(done, total, root)``, called on the
        calling thread after each project. A GUI passes something that only
        touches its own counters — this runs on a worker thread.
    :returns: summaries, most recently run first, then by name. A project
        nobody has run sorts last, which is where a browser wants it.
    """
    projects = discover(roots, depth=depth, limit=limit)
    summaries: List[ProjectSummary] = []
    for index, project in enumerate(projects, start=1):
        summaries.append(scan(project, registry=registry,
                              with_next_steps=with_next_steps))
        if on_progress is not None:
            try:
                on_progress(index, len(projects), project)
            except Exception:                # pragma: no cover - caller's bug
                LOG.debug("progress callback raised", exc_info=True)
    summaries.sort(key=lambda s: (-s.last_run_ns, s.name.lower()))
    return tuple(summaries)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def format_project(summary: ProjectSummary, *, limit: int = 6) -> str:
    """Render one :class:`ProjectSummary` as a block of text.

    :param summary: what :func:`scan` returned.
    :param limit: how many stale entries to name.
    """
    lines = [summary.root]
    if not summary.exists:
        lines.append("  the folder is gone")
        return "\n".join(lines)
    lines.append(f"  stage: {summary.stage_label}")
    lines.append(f"  size: {human_bytes(summary.size_bytes)} in "
                 f"{summary.n_files:,} files")
    if summary.unregistered_bytes:
        lines.append(f"  unaccounted for: "
                     f"{human_bytes(summary.unregistered_bytes)} in "
                     f"{summary.unregistered_files:,} files")
    if summary.last_run_utc:
        lines.append(f"  last run: {summary.last_run_utc} "
                     f"(from the {summary.last_run_source})")
    else:
        lines.append("  last run: never")
    if not summary.known:
        lines.append("  the registry has no record of this project, so "
                     "nothing here can be checked against what produced it")
    else:
        lines.append(f"  registry: {summary.n_artifacts} artifact(s), "
                     f"{summary.staleness_note()}")
    for entry in summary.stale[:limit]:
        lines.append(f"    stale  {entry.describe()}")
    for entry in summary.missing[:limit]:
        lines.append(f"    gone   {entry.describe()}")
    ready = [module for module, blocked in summary.next_steps if not blocked]
    if ready:
        lines.append(f"  next: {', '.join(ready)}")
    for problem in summary.errors[:limit]:
        lines.append(f"  could not read {problem}")
    return "\n".join(lines)


def format_projects(summaries: Sequence[ProjectSummary]) -> str:
    """Render a whole browse as a table, one row per project."""
    if not summaries:
        return "No projects found."
    rows = [("Project", "Stage", "Size", "Last run", "State", "Note")]
    for summary in summaries:
        rows.append((
            summary.name,
            summary.stage_label,
            human_bytes(summary.size_bytes),
            (summary.last_run_utc or "never").replace("+00:00", ""),
            summary.staleness_note(),
            summary.note(),
        ))
    widths = [max(len(row[column]) for row in rows)
              for column in range(len(rows[0]))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
        for row in rows)
