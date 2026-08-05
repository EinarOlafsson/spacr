"""What a project costs in disk, and how to get it back without losing data.

An imaging screen is mostly intermediates.  ``stack/`` is the raw files
re-shaped, ``merged/`` is ``stack/`` with the masks concatenated on, ``data/``
is one PNG per object, and a plate that arrives as 40 GB of TIFFs leaves 400
GB behind.  A multi-terabyte project has no in-app answer to "where did the
space go?", and no way to clear derived data that does not also risk the one
thing on the disk nobody can make again — the images off the microscope.

This module answers both, and it answers the second one *conservatively*.

Disk usage
----------
:func:`scan_project` walks the project once and reports bytes per artifact
kind — raw images, channel stacks, merged arrays, masks, crops, databases,
model weights — measured from the filesystem, not from what the registry
happens to have recorded.  Every file is reconciled against
:class:`spacr.artifacts.Registry`:

* a file under a **registered** artifact's path is attributed to that
  artifact's kind;
* a file under none of them is **unregistered** — its bytes are reported (and
  labelled by the project layout, as a guess, so the number is readable) but
  its provenance is unknown;
* a registered artifact whose path is **no longer on disk** is reported as
  missing, because a registry that claims files that are gone is a registry
  nobody should prune from.

Pruning
-------
The safety property is the whole feature.  An artifact is prunable only when
it is *regenerable*, and "regenerable" is not a guess:

1. the registry knows it — anything unregistered is never a candidate;
2. its kind is not an **original** (:data:`ORIGINAL_KINDS`): nothing in
   :mod:`spacr.ports` declares that it produces raw images or sequencing
   reads, so nothing can make them again;
3. a declared **producing module** exists for its kind, and the module the
   registry recorded is one of them;
4. that module's **inputs are complete** — every recorded input artifact is
   still registered and still on disk, and :func:`spacr.ports.check_ready`
   says the module could run in this project right now;
5. it was written by a run that **finished** (``status == "complete"``);
6. what is on disk still **fingerprints to what was registered**, so a folder
   somebody has since dropped a file into is not prunable;
7. it lives **inside the project**, reached without following a symlink;
8. **every** artifact sharing its path passes all of the above — three kinds
   live in ``measurements.db``, and deleting the file for one of them
   destroys the other two;
9. no other registered artifact sits inside it or around it, because the
   bytes belong to the innermost one and a plan that under-reports what it
   deletes is the failure this module exists to prevent.

Anything that fails any of those is kept, and :attr:`PrunePlan.kept` says
which rule kept it.  Unknown provenance always means keep.

Rule 3 is doing more work than it looks.  ``ports.producers_of`` is empty for
``raw-images`` and ``sequencing-reads``, which is *why* originals are safe —
and it is empty today for ``channel-stacks`` too, so ``stack/`` is reported
as unregistered bytes and never offered, even though it is one of the largest
intermediates.  That is the correct answer while nothing declares it produces
them: the fix is a port declaration, not an exception here.

Nothing is deleted without a plan.  :func:`plan_prune` returns exactly what
would go and how much it frees; :func:`prune` refuses to run unless it is
handed :attr:`PrunePlan.token`, a digest over that exact path set and byte
total, so a confirmation cannot authorise a deletion other than the one that
was shown.

Deleting is **count first, delete second, verify**.  The tree is
re-fingerprinted against the plan before anything is removed; the registry
write is a count and a write on one predicate, checked for equality and
rolled back on any difference.  See :func:`_verified_write` for why that is
the only property worth asserting here.

Archiving
---------
:func:`plan_archive` and :func:`archive` move a project, or a subset of it,
somewhere else and leave a record: a manifest at the destination, a ledger at
the origin, and rows in the destination's registry carrying the provenance
the artifacts arrived with.  The registry still knows where everything went.

Public API
----------
``scan_project``, ``ProjectUsage``, ``KindUsage``, ``format_usage``
    Where the space went.
``plan_prune``, ``PrunePlan``, ``PruneCandidate``, ``PruneSkip``,
``format_prune_plan``, ``prune``, ``PruneResult``
    What can safely go, and the deletion that is gated on it.
``plan_archive``, ``ArchivePlan``, ``archive``, ``ArchiveResult``
    Moving it, with a record.
``is_prunable``
    The predicate, on its own, for anything that wants to ask.
"""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple)

from . import ports
from .artifacts import (Artifact, Registry, STATUS_COMPLETE, content_fingerprint,
                        open_registry, registry_path)
from .database_concurrency import connect, transaction
from .version import get_version

__all__ = [
    "ARCHIVE_LEDGER_NAME",
    "ARCHIVE_MANIFEST_NAME",
    "ArchiveError",
    "ArchiveItem",
    "ArchivePlan",
    "ArchiveResult",
    "ConfirmationRequired",
    "DEFAULT_PRUNABLE_KINDS",
    "DataManagerError",
    "KIND_LABELS",
    "KindUsage",
    "MAX_RECORDED_FILES",
    "ORIGINAL_KINDS",
    "OTHER_KIND",
    "PROTECTED_KINDS",
    "PruneAborted",
    "PruneCandidate",
    "PruneIncomplete",
    "PrunePlan",
    "PruneResult",
    "PruneSkip",
    "ProjectUsage",
    "archive",
    "format_prune_plan",
    "format_usage",
    "human_bytes",
    "is_prunable",
    "plan_archive",
    "plan_prune",
    "prune",
    "scan_project",
]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: Bytes whose kind could not be determined. Never prunable — the whole point
#: of a separate bucket is that nobody knows what it is.
OTHER_KIND = "other"

#: Kinds nothing produces, so nothing can reproduce them. Not a policy knob:
#: there is no argument and no keyword that makes these prunable, because the
#: file that came off the microscope is the experiment.
#:
#: The rule is *derived* rather than declared —
#: ``ports.producers_of("raw-images")`` is empty — but it is also written down
#: here, because a future module that declares itself a producer of
#: ``raw-images`` (a converter, a synthetic-data generator) must not thereby
#: make somebody's plate deletable.
ORIGINAL_KINDS: Tuple[str, ...] = (ports.RAW_IMAGES, ports.SEQUENCING_READS)

#: Regenerable in principle, kept by default. A caller may name one of these
#: in ``kinds=`` and it then has to pass every other rule like anything else.
#:
#: * ``model-weights`` — re-training is stochastic, so the model you get back
#:   is not the model you deleted, and it costs GPU-days.
#: * ``measurements-db`` — every analysis reads it, re-measuring a plate is a
#:   day of compute, and two other kinds live in the same file.
#: * ``predictions`` / ``object-counts`` — those two other kinds. Selecting
#:   either would delete the database.
#: * ``settings-csv`` — kilobytes, and the record of what produced everything
#:   else. Deleting provenance to save disk is a bad trade at any size.
PROTECTED_KINDS: Tuple[str, ...] = (
    ports.MODEL_WEIGHTS, ports.MEASUREMENTS_DB, ports.PREDICTIONS,
    ports.OBJECT_COUNTS, ports.SETTINGS_CSV,
)

#: What :func:`plan_prune` considers when the caller names no kinds: the big
#: derived intermediates, and nothing else. These are the folders that make a
#: plate ten times its own size.
DEFAULT_PRUNABLE_KINDS: Tuple[str, ...] = (
    ports.CHANNEL_STACKS, ports.MERGED_ARRAYS, ports.MASKS, ports.CROPS,
    ports.EMBEDDING, ports.BARCODE_MAP, ports.REGRESSION_RESULTS,
)

#: Display names, in the order a usage report lists them.
KIND_LABELS: Dict[str, str] = {
    ports.RAW_IMAGES: "raw images",
    ports.CHANNEL_STACKS: "channel stacks",
    ports.MERGED_ARRAYS: "merged arrays",
    ports.MASKS: "masks",
    ports.CROPS: "crops",
    ports.MEASUREMENTS_DB: "measurement databases",
    ports.OBJECT_COUNTS: "object counts",
    ports.PREDICTIONS: "predictions",
    ports.MODEL_WEIGHTS: "model weights",
    ports.EMBEDDING: "embeddings",
    ports.SEQUENCING_READS: "sequencing reads",
    ports.BARCODE_MAP: "barcode maps",
    ports.REGRESSION_RESULTS: "regression results",
    ports.SETTINGS_CSV: "provenance records",
    OTHER_KIND: "unclassified",
}

#: Which kind labels a path when several artifacts share it. ``measurements.db``
#: carries ``measurements-db``, ``object-counts`` and ``predictions``; its bytes
#: are counted once, under the first of those. Every kind at the path is still
#: listed on the entry, so nothing is hidden by the choice.
_KIND_RANK: Dict[str, int] = {
    kind: index for index, kind in enumerate((
        ports.MEASUREMENTS_DB, ports.MODEL_WEIGHTS, ports.RAW_IMAGES,
        ports.SEQUENCING_READS, ports.MERGED_ARRAYS, ports.CHANNEL_STACKS,
        ports.MASKS, ports.CROPS, ports.EMBEDDING, ports.REGRESSION_RESULTS,
        ports.BARCODE_MAP, ports.SETTINGS_CSV, ports.OBJECT_COUNTS,
        ports.PREDICTIONS, OTHER_KIND,
    ))
}

#: Layout rules used to *label* unregistered bytes, so a usage report reads as
#: something other than one enormous "unclassified" row. Matched against the
#: path relative to the project root, first hit wins.
#:
#: These guesses never make anything prunable. A file matched here is still
#: unregistered, and :func:`is_prunable` starts from the registry, not from a
#: filename.
_LAYOUT_RULES: Tuple[Tuple[str, str], ...] = (
    # The bookkeeping, first: `artifacts.db` matches `*.db` further down and
    # would otherwise be reported as somebody's measurements.
    ("artifacts.db", ports.SETTINGS_CSV),
    ("spacr_archive*.json", ports.SETTINGS_CSV),
    ("stack/*", ports.CHANNEL_STACKS),
    ("*/stack/*", ports.CHANNEL_STACKS),
    ("merged/*", ports.MERGED_ARRAYS),
    ("*/merged/*", ports.MERGED_ARRAYS),
    ("masks/*", ports.MASKS),
    ("*/masks/*", ports.MASKS),
    ("norm_channel_stack/*", ports.CHANNEL_STACKS),
    ("data/*", ports.CROPS),
    ("*/data/*", ports.CROPS),
    ("orig/*", ports.RAW_IMAGES),
    ("consolidated/*", ports.RAW_IMAGES),
    ("settings/*", ports.SETTINGS_CSV),
    ("model/*", ports.MODEL_WEIGHTS),
    ("*.pth", ports.MODEL_WEIGHTS),
    ("*.fastq.gz", ports.SEQUENCING_READS),
    ("*.fq.gz", ports.SEQUENCING_READS),
    ("*annotated_reads.h5", ports.BARCODE_MAP),
    ("*unique_combinations.csv", ports.BARCODE_MAP),
    ("results/results*.csv", ports.REGRESSION_RESULTS),
    ("*/results/results*.csv", ports.REGRESSION_RESULTS),
    ("results/*", ports.EMBEDDING),
    ("*/results/*", ports.EMBEDDING),
    ("measurements/*.db", ports.MEASUREMENTS_DB),
    ("*.db", ports.MEASUREMENTS_DB),
    ("*.sqlite", ports.MEASUREMENTS_DB),
)

#: Image suffixes that mark a file at the project root as a raw acquisition.
_RAW_SUFFIXES = tuple(
    s if s.startswith(".") else f".{s}" for s in ports.IMAGE_EXTENSIONS)

#: The manifest written into the archive destination.
ARCHIVE_MANIFEST_NAME = "spacr_archive.json"

#: The ledger left at the origin, appended to on every archive, so a folder
#: somebody finds empty next year still says where its contents went.
ARCHIVE_LEDGER_NAME = "spacr_archive_log.json"

#: Above this many files, a plan and a result stop carrying the full path
#: list and set ``files_truncated``. A project with five million crops must
#: not cost half a gigabyte of strings to plan a deletion in.
MAX_RECORDED_FILES = 100_000

#: Bookkeeping files a prune must never touch, whatever the registry says.
_NEVER_DELETE = ("artifacts.db", ARCHIVE_LEDGER_NAME, ARCHIVE_MANIFEST_NAME)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class DataManagerError(Exception):
    """Anything this module refuses to do."""


class ConfirmationRequired(DataManagerError):
    """A destructive call arrived without the plan's confirmation token."""


class PruneAborted(DataManagerError):
    """A prune stopped before removing anything, and nothing was removed.

    Raised when the tree no longer matches the plan, or when a registry write
    changed a different number of rows than the count that gated it. The
    invariant this type carries is in its name: nothing on disk was deleted.
    """


class PruneIncomplete(DataManagerError):
    """A prune deleted some of the plan and could not finish it.

    Distinct from :class:`PruneAborted` on purpose: that type promises
    nothing was removed, and a type whose promise is sometimes true is worse
    than no type at all.
    """


class ArchiveError(DataManagerError):
    """An archive could not be carried out, or could not be verified."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def human_bytes(size: float) -> str:
    """Render a byte count the way a disk report should read.

    :param size: bytes.
    :returns: e.g. ``"1.4 GB"``. Powers of 1000, because that is what the
        disk vendor, ``df`` and the user's storage quota all use.
    """
    value = float(size)
    for unit in ("B", "kB", "MB", "GB", "TB", "PB"):
        if abs(value) < 1000.0 or unit == "PB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1000.0
    return f"{value:.1f} PB"          # pragma: no cover - loop always returns


def _absolute(path: Any) -> str:
    """Return ``path`` as an absolute, user-expanded string."""
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


def _real(path: str) -> str:
    """Return ``path`` with symlinks resolved, for containment checks."""
    try:
        return os.path.realpath(path)
    except OSError:                    # pragma: no cover - exotic filesystems
        return path


def _contained(path: str, root: str) -> bool:
    """True when ``path`` is ``root`` or lies under it, symlinks resolved."""
    real_path, real_root = _real(path), _real(root)
    return real_path == real_root or real_path.startswith(real_root + os.sep)


def _now() -> str:
    """The current instant, ISO-8601 UTC."""
    return datetime.now(tz=timezone.utc).isoformat()


def _digest(parts: Iterable[str]) -> str:
    """A short stable digest over an ordered list of strings."""
    import hashlib
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _open_if_present(registry: Optional[Registry],
                     project: str) -> Optional[Registry]:
    """Return ``registry``, or one opened for ``project`` if a file exists.

    Never creates: a read-only consumer that conjures an empty registry has
    turned "this project has no provenance" — the answer that keeps
    everything in it — into "this project's provenance is empty", which
    reads the same and is not.
    """
    if registry is not None:
        return registry
    candidate = registry_path(project)
    if os.path.isfile(candidate):
        return open_registry(project, path=candidate, create=False)
    return None


def _classify_by_layout(relative: str) -> str:
    """Guess a kind for an unregistered file from where it sits.

    :param relative: path relative to the project root, with ``/`` separators.
    :returns: a :mod:`spacr.ports` kind, or :data:`OTHER_KIND`.
    """
    lowered = relative.lower()
    for pattern, kind in _LAYOUT_RULES:
        if fnmatch.fnmatch(lowered, pattern):
            return kind
    if "/" not in lowered and lowered.endswith(_RAW_SUFFIXES):
        return ports.RAW_IMAGES
    return OTHER_KIND


# ---------------------------------------------------------------------------
# The usage report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KindUsage:
    """What one artifact kind costs in this project.

    :param kind: a :mod:`spacr.ports` kind, or :data:`OTHER_KIND`.
    :param label: the display name.
    :param size_bytes: bytes on disk, measured by walking the project.
    :param n_files: files counted.
    :param n_paths: distinct artifact paths (a folder counts once).
    :param n_artifacts: registry rows of this kind.
    :param registered_bytes: of ``size_bytes``, how much sits under a
        registered artifact.
    :param unregistered_bytes: the rest — bytes nobody claims.
    :param recorded_bytes: what the registry *says* those artifacts weigh.
        A gap against ``size_bytes`` means the folder changed since it was
        registered, which is exactly when a prune must decline.
    :param shared_paths: artifacts of this kind sitting at a path whose bytes
        are counted under a different kind — the three kinds inside
        ``measurements.db``. Reported so a kind showing zero bytes reads as
        "counted next door" rather than "gone".
    """

    kind: str
    label: str
    size_bytes: int = 0
    n_files: int = 0
    n_paths: int = 0
    n_artifacts: int = 0
    registered_bytes: int = 0
    unregistered_bytes: int = 0
    recorded_bytes: int = 0
    shared_paths: int = 0

    @property
    def drifted(self) -> bool:
        """True when disk and registry disagree about this kind's size."""
        return bool(self.n_artifacts) and self.recorded_bytes != self.registered_bytes


@dataclass(frozen=True)
class ArtifactUsage:
    """One registered artifact, measured on disk.

    :param path: the file or folder.
    :param kinds: every kind registered at this path, ranked.
    :param artifacts: the registry rows at this path, newest first.
    :param size_bytes: bytes actually there now.
    :param n_files: files actually there now.
    :param exists: whether anything is there at all.
    """

    path: str
    kinds: Tuple[str, ...]
    artifacts: Tuple[Artifact, ...]
    size_bytes: int = 0
    n_files: int = 0
    exists: bool = True

    @property
    def kind(self) -> str:
        """The kind this path is reported under."""
        return self.kinds[0] if self.kinds else OTHER_KIND


@dataclass(frozen=True)
class ProjectUsage:
    """Where a project's disk went, reconciled against the registry.

    :param root: the project root.
    :param total_bytes: every byte under ``root``, symlinks excluded.
    :param total_files: every file under ``root``, symlinks excluded.
    :param kinds: per-kind breakdown, largest first.
    :param artifacts: one entry per registered artifact path.
    :param unregistered: ``(path, size_bytes)`` for the largest unregistered
        top-level entries, largest first — the bytes no artifact claims.
    :param unregistered_bytes: their total.
    :param unregistered_files: how many files that is.
    :param missing: artifacts the registry has whose path is gone.
    :param outside: registered artifacts whose path is not under ``root``.
    :param symlinks: links found and not followed. Never counted, never
        deleted: a link into somebody else's storage is the one shape where
        a recursive delete leaves the project entirely.
    :param errors: paths that could not be read, as strings.
    :param scanned_utc: when the walk ran.
    """

    root: str
    total_bytes: int = 0
    total_files: int = 0
    kinds: Tuple[KindUsage, ...] = ()
    artifacts: Tuple[ArtifactUsage, ...] = ()
    unregistered: Tuple[Tuple[str, int], ...] = ()
    unregistered_bytes: int = 0
    unregistered_files: int = 0
    missing: Tuple[Artifact, ...] = ()
    outside: Tuple[Artifact, ...] = ()
    symlinks: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()
    scanned_utc: str = ""

    def kind(self, kind: str) -> KindUsage:
        """Return the row for one kind, zeroed when it is absent."""
        for row in self.kinds:
            if row.kind == kind:
                return row
        return KindUsage(kind, KIND_LABELS.get(kind, kind))

    @property
    def registered_bytes(self) -> int:
        """Bytes sitting under an artifact the registry knows about."""
        return sum(row.registered_bytes for row in self.kinds)

    def artifact_at(self, path: str) -> Optional[ArtifactUsage]:
        """Return the entry for one registered path, or None."""
        target = _absolute(path)
        for entry in self.artifacts:
            if entry.path == target:
                return entry
        return None

    def __str__(self) -> str:
        """The full report; see :func:`format_usage`."""
        return format_usage(self)


def _walk_project(root: str) -> Tuple[Dict[str, int], List[str], List[str]]:
    """Walk ``root`` once and return ``(files, symlinks, errors)``.

    Symlinks — files and directories both — are recorded and never followed.
    A project whose ``merged/`` is a link into shared storage must not have
    that storage walked into the size report, and must certainly not have it
    walked into a delete.
    """
    files: Dict[str, int] = {}
    symlinks: List[str] = []
    errors: List[str] = []

    def _note(exc: OSError) -> None:
        errors.append(f"{getattr(exc, 'filename', '')}: {exc}")

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False,
                                                onerror=_note):
        kept: List[str] = []
        for name in sorted(dirnames):
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                symlinks.append(full)
            else:
                kept.append(name)
        dirnames[:] = kept
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                symlinks.append(full)
                continue
            try:
                files[full] = os.stat(full).st_size
            except OSError as exc:
                errors.append(f"{full}: {exc}")
    return files, symlinks, errors


def _owner_of(path: str, index: Mapping[str, Any], root: str) -> Optional[str]:
    """Return the longest registered path that contains ``path``.

    Walking up from the file rather than testing every registered prefix: it
    is the same answer, it costs the depth of the tree instead of the number
    of artifacts, and longest-prefix falls out of the order. That matters
    because ``data/`` (the crops folder) and ``data/plate1/well_A01_png``
    can both be registered, and the bytes must be counted once.
    """
    current = path
    while True:
        if current in index:
            return current
        parent = os.path.dirname(current)
        if parent == current or len(parent) < len(root):
            return None
        current = parent


def scan_project(root: Any, *,
                 registry: Optional[Registry] = None) -> ProjectUsage:
    """Measure a project and reconcile it against the artifact registry.

    One walk of the tree, then every file is attributed to the registered
    artifact whose path contains it — longest path wins — or to nobody. The
    per-kind numbers come from the filesystem; the registry's own
    ``size_bytes`` is reported alongside so drift between the two is visible
    rather than assumed away.

    :param root: the project root.
    :param registry: an open :class:`spacr.artifacts.Registry`. Omit and one
        is opened for ``root`` when a registry file exists — including the
        shared one :data:`spacr.artifacts.ARTIFACTS_DB_ENV` points at. A
        project with no registry is scanned anyway, and reports every byte as
        unregistered, which is the correct answer and the reason nothing in
        it is prunable.
    :returns: a :class:`ProjectUsage`.
    :raises DataManagerError: when ``root`` is not a directory.
    """
    project = _absolute(root)
    if not os.path.isdir(project):
        raise DataManagerError(f"{project} is not a folder, so there is "
                               f"nothing to measure")

    registered: List[Artifact] = []
    target = _open_if_present(registry, project)
    if target is not None:
        registered = list(target.by_project(project))

    files, symlinks, errors = _walk_project(project)

    # Registered paths, grouped. Several artifacts may name one path.
    by_path: Dict[str, List[Artifact]] = {}
    outside: List[Artifact] = []
    for artifact in registered:
        path = _absolute(artifact.path)
        if not _contained(path, project):
            outside.append(artifact)
            continue
        by_path.setdefault(path, []).append(artifact)
    for group in by_path.values():
        group.sort(key=lambda a: (-a.created_ns, a.artifact_id))

    # Attribute every walked file to at most one registered path.
    owned_bytes: Dict[str, int] = {path: 0 for path in by_path}
    owned_files: Dict[str, int] = {path: 0 for path in by_path}
    unowned: Dict[str, int] = {}
    for path, size in files.items():
        owner = _owner_of(path, by_path, project)
        if owner is None:
            unowned[path] = size
            continue
        owned_bytes[owner] += size
        owned_files[owner] += 1

    entries: List[ArtifactUsage] = []
    missing: List[Artifact] = []
    for path, group in sorted(by_path.items()):
        kinds = tuple(sorted({a.kind for a in group},
                             key=lambda k: _KIND_RANK.get(k, len(_KIND_RANK))))
        exists = os.path.exists(path)
        entries.append(ArtifactUsage(
            path=path, kinds=kinds, artifacts=tuple(group),
            size_bytes=owned_bytes[path], n_files=owned_files[path],
            exists=exists))
        if not exists:
            missing.extend(group)

    # Per-kind totals: registered bytes attributed to the path's ranked kind,
    # unregistered bytes to whatever the layout suggests.
    stats: Dict[str, Dict[str, int]] = {}

    def _bucket(kind: str) -> Dict[str, int]:
        return stats.setdefault(kind, {
            "size": 0, "files": 0, "paths": 0, "artifacts": 0,
            "registered": 0, "unregistered": 0, "recorded": 0, "shared": 0})

    for entry in entries:
        bucket = _bucket(entry.kind)
        bucket["size"] += entry.size_bytes
        bucket["files"] += entry.n_files
        bucket["paths"] += 1
        bucket["registered"] += entry.size_bytes
        for artifact in entry.artifacts:
            other = _bucket(artifact.kind)
            other["artifacts"] += 1
            if artifact.kind != entry.kind:
                other["shared"] += 1
        bucket["recorded"] += max(a.size_bytes for a in entry.artifacts)

    for path, size in unowned.items():
        relative = os.path.relpath(path, project).replace(os.sep, "/")
        bucket = _bucket(_classify_by_layout(relative))
        bucket["size"] += size
        bucket["files"] += 1
        bucket["unregistered"] += size

    rows = [
        KindUsage(kind=kind, label=KIND_LABELS.get(kind, kind),
                  size_bytes=values["size"], n_files=values["files"],
                  n_paths=values["paths"], n_artifacts=values["artifacts"],
                  registered_bytes=values["registered"],
                  unregistered_bytes=values["unregistered"],
                  recorded_bytes=values["recorded"],
                  shared_paths=values["shared"])
        for kind, values in stats.items()
    ]
    rows.sort(key=lambda r: (-r.size_bytes, r.label))

    return ProjectUsage(
        root=project,
        total_bytes=sum(files.values()),
        total_files=len(files),
        kinds=tuple(rows),
        artifacts=tuple(entries),
        unregistered=tuple(sorted(unowned.items(), key=lambda kv: -kv[1])[:50]),
        unregistered_bytes=sum(unowned.values()),
        unregistered_files=len(unowned),
        missing=tuple(missing),
        outside=tuple(outside),
        symlinks=tuple(sorted(symlinks)),
        errors=tuple(errors),
        scanned_utc=_now(),
    )


def format_usage(usage: ProjectUsage, *, limit: int = 8) -> str:
    """Render a :class:`ProjectUsage` as a block of text.

    :param usage: the result of :func:`scan_project`.
    :param limit: how many unregistered paths to list.
    """
    lines = [f"{usage.root}",
             f"  {human_bytes(usage.total_bytes)} in "
             f"{usage.total_files:,} files"]
    for row in usage.kinds:
        if not row.size_bytes and not row.n_artifacts:
            continue
        note = ""
        if row.unregistered_bytes:
            note = f" ({human_bytes(row.unregistered_bytes)} unregistered)"
        elif not row.size_bytes and row.shared_paths:
            note = (f" (lives in {row.shared_paths} file(s) counted under "
                    f"another kind)")
        lines.append(f"    {row.label:<22} {human_bytes(row.size_bytes):>10}  "
                     f"{row.n_files:>8,} files{note}")
    if usage.unregistered_bytes:
        lines.append(f"  unregistered: {human_bytes(usage.unregistered_bytes)} "
                     f"in {usage.unregistered_files:,} files — never pruned, "
                     f"because nothing records what made them")
        for path, size in usage.unregistered[:limit]:
            lines.append(f"    {human_bytes(size):>10}  "
                         f"{os.path.relpath(path, usage.root)}")
    if usage.missing:
        lines.append(f"  {len(usage.missing)} registered artifact(s) are no "
                     f"longer on disk")
    if usage.symlinks:
        lines.append(f"  {len(usage.symlinks)} symlink(s), not followed")
    for problem in usage.errors:
        lines.append(f"  could not read {problem}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The prunable predicate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PruneSkip:
    """One thing that was considered and kept, and why.

    :param path: the file or folder.
    :param kind: its kind, as far as anything knows.
    :param size_bytes: what keeping it costs.
    :param reason: the rule that kept it, in a sentence a user can act on.
    :param artifact_id: the registry row, when there was one.
    """

    path: str
    kind: str
    size_bytes: int
    reason: str
    artifact_id: str = ""


def _regenerable_reason(artifact: Artifact, root: str,
                        registry: Optional[Registry]) -> str:
    """Return ``""`` when ``artifact`` is regenerable, else why it is not.

    The rules are in :mod:`spacr.data_manager`'s module docstring; this is
    them, in the order that makes the cheapest check fail first. Only the
    ones about *this* artifact — the shared-path rule needs the whole group
    and lives in :func:`is_prunable`.
    """
    kind, module = artifact.kind, artifact.module
    if kind in ORIGINAL_KINDS:
        return (f"{KIND_LABELS.get(kind, kind)} are originals — nothing "
                f"produces them, so nothing could make them again")
    path = _absolute(artifact.path)
    if os.path.basename(path) in _NEVER_DELETE:
        return "this is the provenance record itself"
    if not _contained(path, root):
        return f"it is outside {root}, so this project does not own it"
    if os.path.islink(path):
        return "it is a symlink; deleting through one leaves the project"
    if artifact.status != STATUS_COMPLETE:
        return (f"the run that wrote it ended '{artifact.status}', so "
                f"re-running would not reproduce it")
    producers = ports.producers_of(kind)
    if not producers:
        return (f"no module declares that it produces "
                f"{KIND_LABELS.get(kind, kind)}, so nothing can make it again")
    if module not in producers:
        return (f"the registry says {module} wrote it, but {module} is not a "
                f"declared producer of {KIND_LABELS.get(kind, kind)}")
    if not artifact.fingerprint:
        return "nothing was on disk when it was registered"

    if registry is not None:
        for input_id in artifact.inputs:
            upstream = registry.get(input_id)
            if upstream is None:
                return (f"the input it was made from ({input_id}) is no "
                        f"longer in the registry, so it cannot be remade")
            if not upstream.exists:
                return (f"its input {KIND_LABELS.get(upstream.kind, upstream.kind)} "
                        f"at {upstream.path} is gone, so it cannot be remade")

    try:
        readiness = ports.check_ready(module, root=root, sample=1)
    except ports.UnknownModule:
        return f"no ports are declared for {module}, so it cannot be re-run"
    if not readiness.ok:
        return (f"{module} could not run here now: "
                f"{readiness.errors[0].message}")

    current = content_fingerprint(path)
    if not current.digest:
        return "it is already gone from disk"
    if current.digest != artifact.fingerprint:
        return ("what is on disk is not what was registered — something "
                "changed it, so its provenance no longer describes it")
    return ""


def is_prunable(artifact: Artifact, *, root: Any,
                registry: Optional[Registry] = None,
                group: Sequence[Artifact] = ()) -> str:
    """Return ``""`` when ``artifact`` may be pruned, else the reason to keep.

    An empty string is the *only* value that means "delete this". Every
    failure mode — including one this function did not anticipate — produces
    a sentence, which is the direction a delete predicate must fail in.

    :param artifact: a registered artifact. Something the registry has never
        heard of cannot be passed here at all, which is the point.
    :param root: the project root it must live inside.
    :param registry: the registry, for checking that its inputs survive.
    :param group: every artifact registered at the same path. A path is
        prunable only when all of them are: ``measurements.db`` carries three
        kinds, and deleting the file for one destroys the other two.
    :returns: ``""`` or a reason.
    """
    project = _absolute(root)
    reason = _regenerable_reason(artifact, project, registry)
    if reason:
        return reason
    for other in group:
        if other.artifact_id == artifact.artifact_id:
            continue
        shared = _regenerable_reason(other, project, registry)
        if shared:
            return (f"the same path also holds "
                    f"{KIND_LABELS.get(other.kind, other.kind)} from "
                    f"{other.module}, and {shared}")
    return ""


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PruneCandidate:
    """One thing a prune would delete.

    :param path: the file or folder that would go.
    :param kind: its kind.
    :param module: the module that would make it again.
    :param artifact_ids: every registry row at this path.
    :param size_bytes: what deleting it frees, measured now.
    :param n_files: files it holds.
    :param inventory_digest: the content fingerprint the plan was made
        against. :func:`prune` re-computes it and refuses on any difference,
        which is how "the tree changed under us" stops a delete.
    :param sample_files: up to twenty paths, for showing a user what this is.
    :param downstream: artifact ids derived from this one — results that
        would no longer be reproducible from what is left.
    :param regenerate_with: the sentence telling a user how to get it back.
    """

    path: str
    kind: str
    module: str
    artifact_ids: Tuple[str, ...]
    size_bytes: int
    n_files: int
    inventory_digest: str
    sample_files: Tuple[str, ...] = ()
    downstream: Tuple[str, ...] = ()
    regenerate_with: str = ""

    @property
    def label(self) -> str:
        """The kind's display name."""
        return KIND_LABELS.get(self.kind, self.kind)


@dataclass(frozen=True)
class PrunePlan:
    """Exactly what a prune would delete, and what it would leave.

    Produced by :func:`plan_prune` and consumed by :func:`prune`. Nothing
    else may be deleted: :func:`prune` takes a plan, not a folder.

    :param root: the project.
    :param candidates: what would go, largest first.
    :param kept: what was considered and kept, with the rule that kept it.
    :param total_bytes: what the prune frees.
    :param total_files: how many files that is.
    :param kinds: the kinds this plan was asked for.
    :param token: the confirmation. :func:`prune` refuses without it, and it
        is a digest over the exact path set and byte total, so a token taken
        from one plan cannot authorise a different one.
    :param unregistered_bytes: bytes in the project that no artifact claims —
        reported here because they are the reason a prune frees less than a
        user expected, and they are never candidates.
    :param created_utc: when the plan was made.
    """

    root: str
    candidates: Tuple[PruneCandidate, ...] = ()
    kept: Tuple[PruneSkip, ...] = ()
    total_bytes: int = 0
    total_files: int = 0
    kinds: Tuple[str, ...] = ()
    token: str = ""
    unregistered_bytes: int = 0
    created_utc: str = ""

    def __bool__(self) -> bool:
        """True when there is something to delete."""
        return bool(self.candidates)

    @property
    def paths(self) -> Tuple[str, ...]:
        """Every path this plan would delete, in plan order."""
        return tuple(c.path for c in self.candidates)

    def file_list(self) -> Tuple[Tuple[str, ...], bool]:
        """Enumerate every file this plan would delete, right now.

        Walked fresh rather than stored: a plan for a project with millions
        of crops must not carry millions of strings, and the list a user is
        shown should be the list that is on disk when they look at it.

        :returns: ``(paths, truncated)``. ``truncated`` is True when the plan
            holds more than :data:`MAX_RECORDED_FILES` files and the list was
            cut short.
        """
        found: List[str] = []
        for candidate in self.candidates:
            for path in _enumerate(candidate.path):
                if len(found) >= MAX_RECORDED_FILES:
                    return tuple(found), True
                found.append(path)
        return tuple(found), False

    def __str__(self) -> str:
        """The full report; see :func:`format_prune_plan`."""
        return format_prune_plan(self)


def _enumerate(path: str) -> List[str]:
    """Every file at ``path``: itself, or everything under it, sorted."""
    if os.path.isfile(path) and not os.path.islink(path):
        return [path]
    if not os.path.isdir(path) or os.path.islink(path):
        return []
    found: List[str] = []
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames
                             if not os.path.islink(os.path.join(dirpath, d)))
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            if not os.path.islink(full):
                found.append(full)
    return sorted(found)


def plan_prune(root: Any, *,
               registry: Optional[Registry] = None,
               kinds: Optional[Sequence[str]] = None,
               usage: Optional[ProjectUsage] = None,
               paths: Optional[Sequence[str]] = None) -> PrunePlan:
    """Work out what could be deleted, and prove it before deleting anything.

    Every registered artifact in the project is put through
    :func:`is_prunable`. What passes becomes a candidate; what does not
    becomes a :class:`PruneSkip` carrying the rule that kept it, so a user
    who expected to free 300 GB and was offered 12 GB can read why.

    Unregistered bytes are never candidates. They are counted and reported
    (:attr:`PrunePlan.unregistered_bytes`), because "spaCR does not know what
    made this" is information, but the answer for them is always keep.

    :param root: the project root.
    :param registry: an open registry; opened from ``root`` when omitted.
        A project with no registry produces an empty plan, which is correct:
        nothing there has known provenance.
    :param kinds: kinds to consider. Defaults to
        :data:`DEFAULT_PRUNABLE_KINDS`. Naming a :data:`PROTECTED_KINDS`
        member opts it in — it still has to pass every safety rule. Naming an
        :data:`ORIGINAL_KINDS` member does nothing: there is no path through
        this module that deletes an original.
    :param usage: a :class:`ProjectUsage` from :func:`scan_project`, to save
        a second walk of a large project.
    :param paths: restrict to these artifact paths.
    :returns: a :class:`PrunePlan`.
    """
    project = _absolute(root)
    target = _open_if_present(registry, project)
    report = usage if usage is not None else scan_project(
        project, registry=target)

    selected = tuple(kinds) if kinds is not None else DEFAULT_PRUNABLE_KINDS
    wanted = {str(k) for k in selected} - set(ORIGINAL_KINDS)
    restrict = {_absolute(p) for p in paths} if paths is not None else None

    candidates: List[PruneCandidate] = []
    kept: List[PruneSkip] = []
    registered_paths = {entry.path for entry in report.artifacts}

    for entry in report.artifacts:
        if restrict is not None and entry.path not in restrict:
            continue
        nested = _nested_registration(entry.path, registered_paths)
        if nested:
            # Two registered artifacts, one inside the other. The bytes were
            # attributed to the inner one (longest prefix wins), so deleting
            # the outer would free more than the plan says and take an
            # artifact nobody judged with it. Nothing declares such a pair
            # today; the guard is here because a plan that under-reports what
            # it deletes is the failure this whole module is about.
            kept.append(PruneSkip(
                entry.path, entry.kind, entry.size_bytes,
                f"another registered artifact is at {nested}, inside or "
                f"around it; deleting one would take the other",
                entry.artifacts[0].artifact_id))
            continue
        if not entry.exists:
            kept.append(PruneSkip(
                entry.path, entry.kind, 0,
                "the registry has it but it is not on disk",
                entry.artifacts[0].artifact_id))
            continue
        if entry.kind not in wanted:
            kept.append(PruneSkip(
                entry.path, entry.kind, entry.size_bytes,
                _not_selected_reason(entry.kind, wanted),
                entry.artifacts[0].artifact_id))
            continue

        # The row this candidate is *reported* as: the newest of the kind the
        # path is filed under. It is only the label and the "how do I get it
        # back" module — every artifact at the path still has to pass.
        newest = next((a for a in entry.artifacts if a.kind == entry.kind),
                      entry.artifacts[0])
        reason = is_prunable(newest, root=project, registry=target,
                             group=entry.artifacts)
        if reason:
            kept.append(PruneSkip(entry.path, entry.kind, entry.size_bytes,
                                  reason, newest.artifact_id))
            continue

        files = _enumerate(entry.path)
        downstream: Tuple[str, ...] = ()
        if target is not None:
            downstream = tuple(
                a.artifact_id for a in target.downstream_of(newest.artifact_id))
        candidates.append(PruneCandidate(
            path=entry.path, kind=entry.kind, module=newest.module,
            artifact_ids=tuple(a.artifact_id for a in entry.artifacts),
            size_bytes=entry.size_bytes, n_files=entry.n_files,
            inventory_digest=newest.fingerprint,
            sample_files=tuple(files[:20]),
            downstream=downstream,
            regenerate_with=f"re-run {newest.module} on {project}"))

    candidates.sort(key=lambda c: (-c.size_bytes, c.path))
    kept.sort(key=lambda s: (-s.size_bytes, s.path))
    total_bytes = sum(c.size_bytes for c in candidates)
    token = _digest([project, str(total_bytes)]
                    + [f"{c.path}:{c.size_bytes}:{c.inventory_digest}"
                       for c in candidates])
    return PrunePlan(
        root=project, candidates=tuple(candidates), kept=tuple(kept),
        total_bytes=total_bytes,
        total_files=sum(c.n_files for c in candidates),
        kinds=tuple(sorted(wanted)), token=token,
        unregistered_bytes=report.unregistered_bytes,
        created_utc=_now())


def _nested_registration(path: str, registered: Iterable[str]) -> str:
    """Return another registered path that contains ``path`` or sits in it."""
    for other in registered:
        if other == path:
            continue
        if other.startswith(path + os.sep) or path.startswith(other + os.sep):
            return other
    return ""


def _not_selected_reason(kind: str, wanted: Iterable[str]) -> str:
    """Say why a kind was not considered, distinguishing policy from safety."""
    label = KIND_LABELS.get(kind, kind)
    if kind in ORIGINAL_KINDS:
        return (f"{label} are originals and can never be pruned, whatever is "
                f"asked for")
    if kind in PROTECTED_KINDS:
        return (f"{label} are kept by default; name the kind explicitly to "
                f"consider them")
    if kind == OTHER_KIND:
        return "nothing records what produced this"
    return f"{label} were not in the kinds this plan asked for"


def format_prune_plan(plan: PrunePlan, *, limit: int = 20) -> str:
    """Render a :class:`PrunePlan` as the text shown before a deletion.

    :param plan: the plan.
    :param limit: how many kept entries to explain.
    """
    if not plan.candidates:
        lines = [f"Nothing in {plan.root} can be pruned safely."]
    else:
        lines = [f"Pruning {plan.root} would delete "
                 f"{len(plan.candidates)} item(s), "
                 f"{plan.total_files:,} files, and free "
                 f"{human_bytes(plan.total_bytes)}:"]
        for candidate in plan.candidates:
            lines.append(f"  {human_bytes(candidate.size_bytes):>10}  "
                         f"{candidate.label:<16} "
                         f"{os.path.relpath(candidate.path, plan.root)}"
                         f"  ({candidate.n_files:,} files)")
            lines.append(f"              get it back: "
                         f"{candidate.regenerate_with}")
            if candidate.downstream:
                lines.append(f"              {len(candidate.downstream)} "
                             f"downstream result(s) were derived from it")
    if plan.kept:
        lines.append("Kept:")
        for skip in plan.kept[:limit]:
            lines.append(f"  {human_bytes(skip.size_bytes):>10}  "
                         f"{os.path.relpath(skip.path, plan.root)} — "
                         f"{skip.reason}")
        if len(plan.kept) > limit:
            lines.append(f"  … and {len(plan.kept) - limit} more")
    if plan.unregistered_bytes:
        lines.append(f"{human_bytes(plan.unregistered_bytes)} in this project "
                     f"is not in the registry and is never pruned.")
    if plan.candidates:
        lines.append(f"This cannot be undone. Confirm with token "
                     f"{plan.token}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deleting, gated on a count
# ---------------------------------------------------------------------------

def _count_matching(connection, table: str, predicate: str,
                    params: Sequence[Any]) -> int:
    """Count the rows a predicate selects.

    Factored out so a test can make it lie, and prove that the equality below
    is what stops the write rather than an accident of the two statements
    happening to agree.
    """
    return int(connection.execute(
        f'SELECT COUNT(*) FROM "{table}" WHERE {predicate}',
        tuple(params)).fetchone()[0])


def _verified_write(connection, table: str, predicate: str,
                    params: Sequence[Any], statement: str, *,
                    what: str) -> int:
    """Count with a predicate, write with the *same* predicate, verify.

    This project has been destroyed twice by a delete written against a row
    identity that was not one. ``DELETE ... WHERE rowid IN (...)`` removed a
    whole table, because every spaCR object table declares a column called
    ``rowID`` and SQLite identifiers are case-insensitive, so ``rowid`` read
    the plate row. The obvious repair — delete by the declared key — was
    equally destructive, because two rows for one object can share all five
    key columns.

    So no row identity is named here at all, by anybody. The caller supplies
    one predicate; this counts with it, writes with it, and treats any
    difference between the two numbers as a failure rather than a result. The
    predicate string is interpolated once, into both statements, so the two
    cannot drift apart in a later edit.

    :param connection: an open connection, inside a transaction.
    :param table: the table to write.
    :param predicate: the WHERE clause, without ``WHERE``.
    :param params: its parameters, bound to both statements.
    :param statement: ``DELETE FROM "t"`` or ``UPDATE "t" SET ...``, without
        its ``WHERE``.
    :param what: what this write is, for the error message.
    :returns: rows written, which equals the rows counted.
    :raises PruneAborted: on any difference. The caller's transaction rolls
        back, because :func:`spacr.database_concurrency.transaction` rolls
        back on any exception out of its body.
    """
    counted = _count_matching(connection, table, predicate, params)
    changed = int(connection.execute(
        f"{statement} WHERE {predicate}", tuple(params)).rowcount or 0)
    if changed != counted:
        raise PruneAborted(
            f"Refusing to {what}: the write changed {changed} row(s) in "
            f"{table} where the count that gated it said {counted}. The "
            f"statement did not act on the rows that were checked, so "
            f"nothing about the result can be trusted. The transaction was "
            f"rolled back and nothing was written.")
    return changed


def _mark_artifacts(registry: Registry, artifact_ids: Sequence[str],
                    note: Mapping[str, Any], *, forget: bool = False) -> int:
    """Record what happened to some artifacts, or forget them.

    Marking rather than deleting is the default on purpose: the registry row
    is *how* a pruned artifact gets made again. Throwing away the settings
    that produced something, to save the kilobyte the row costs, would make
    the deletion irreversible in the one sense that matters.

    :param registry: the project registry.
    :param artifact_ids: rows to touch.
    :param note: JSON-safe keys merged into each row's ``extra``.
    :param forget: delete the rows instead of marking them.
    :returns: rows written.
    :raises PruneAborted: when a count and its write disagree.
    """
    ids = [str(a) for a in dict.fromkeys(artifact_ids)]
    if not ids:
        return 0
    placeholders = ", ".join("?" * len(ids))
    predicate = f"artifact_id IN ({placeholders})"
    connection = connect(registry.path, timeout=registry.timeout)
    try:
        with transaction(connection, mode="IMMEDIATE", attempts=6,
                         busy_timeout=registry.timeout):
            if forget:
                # Edges first: `artifacts` is the parent of a foreign key, and
                # the two are one write either way. Both go through the same
                # count-write-compare, on the same predicate.
                _verified_write(connection, "artifact_inputs",
                                predicate, ids,
                                'DELETE FROM "artifact_inputs"',
                                what="forget the pruned artifacts' inputs")
                return _verified_write(connection, "artifacts",
                                       predicate, ids,
                                       'DELETE FROM "artifacts"',
                                       what="forget the pruned artifacts")
            # The merged JSON is computed per row and staged in a temp table,
            # so the UPDATE below can be ONE statement over ONE predicate --
            # the same predicate the count is taken with. A loop of
            # `WHERE artifact_id = ?` updates would be a write on a predicate
            # nothing counted, which is the whole failure mode being avoided.
            connection.execute(
                "CREATE TEMP TABLE IF NOT EXISTS _spacr_mark ("
                "artifact_id TEXT PRIMARY KEY, extra_json TEXT)")
            connection.execute("DELETE FROM _spacr_mark")
            staged = []
            for artifact_id, extra_json in connection.execute(
                    f"SELECT artifact_id, extra_json FROM artifacts "
                    f"WHERE {predicate}", ids).fetchall():
                merged = dict(json.loads(extra_json) if extra_json else {})
                merged.update(note)
                staged.append((str(artifact_id),
                               json.dumps(merged, sort_keys=True,
                                          separators=(",", ":"))))
            connection.executemany(
                "INSERT INTO _spacr_mark (artifact_id, extra_json) "
                "VALUES (?, ?)", staged)
            return _verified_write(
                connection, "artifacts", predicate, ids,
                'UPDATE "artifacts" SET extra_json = COALESCE('
                '(SELECT m.extra_json FROM _spacr_mark m '
                'WHERE m.artifact_id = "artifacts".artifact_id), '
                'extra_json)',
                what="record what happened to the pruned artifacts")
    finally:
        connection.close()


@dataclass(frozen=True)
class PruneResult:
    """What a prune actually did.

    :param root: the project.
    :param removed_paths: the artifact paths deleted, in plan order.
    :param removed_files: every file removed. Empty with ``files_truncated``
        when the plan held more than :data:`MAX_RECORDED_FILES`.
    :param files_truncated: the list was too long to keep.
    :param freed_bytes: bytes freed, as counted by the plan.
    :param n_files: files removed.
    :param registry_rows: registry rows marked or forgotten.
    :param forgotten: whether those rows were deleted rather than marked.
    :param finished_utc: when it finished.
    """

    root: str
    removed_paths: Tuple[str, ...] = ()
    removed_files: Tuple[str, ...] = ()
    files_truncated: bool = False
    freed_bytes: int = 0
    n_files: int = 0
    registry_rows: int = 0
    forgotten: bool = False
    finished_utc: str = ""


def prune(plan: PrunePlan, *, confirm: str,
          registry: Optional[Registry] = None,
          forget_rows: bool = False) -> PruneResult:
    """Carry out a plan. Irreversible, and gated on the plan being unchanged.

    The order is the safety story:

    1. the confirmation must equal :attr:`PrunePlan.token`, a digest over the
       plan's exact paths and byte total, so a token cannot authorise a
       deletion other than the one it was shown for;
    2. every candidate is re-fingerprinted and must still match the plan.
       Any difference — a file added, a file changed, the folder gone —
       aborts the whole call before a single delete;
    3. the registry write happens next, inside one transaction, counted and
       verified (see :func:`_verified_write`). A mismatch rolls it back and
       raises with **nothing on disk deleted**;
    4. only then are the files removed;
    5. every path is checked to be gone afterwards.

    Step 3 commits before step 4 deliberately. A crash between them leaves
    the registry saying an artifact was pruned while its files are still
    there — recoverable by running the prune again. The other order leaves
    files deleted that the registry still describes as present, which is the
    state nobody can recover from.

    :param plan: from :func:`plan_prune`.
    :param confirm: :attr:`PrunePlan.token`.
    :param registry: the project registry; opened from the plan's root when
        omitted.
    :param forget_rows: delete the registry rows instead of marking them.
        Off by default: the row is the recipe for making the data again.
    :returns: a :class:`PruneResult`.
    :raises ConfirmationRequired: without the right token.
    :raises PruneAborted: when the tree changed, or a count and its write
        disagreed. Nothing was deleted in either case.
    :raises PruneIncomplete: when a path survived its own deletion — a
        permission, a busy file. Some of the plan did go, which is why this
        is a different type from :class:`PruneAborted`.
    """
    if str(confirm) != plan.token:
        raise ConfirmationRequired(
            f"This prune would delete {human_bytes(plan.total_bytes)} in "
            f"{plan.total_files:,} files and cannot be undone. Pass "
            f"confirm=plan.token to carry it out. The token is a digest over "
            f"the exact paths and sizes above, so a plan that has changed "
            f"since you read it will not accept the token you were given.")
    if not plan.candidates:
        return PruneResult(root=plan.root, finished_utc=_now())

    # 2. Nothing is deleted until the whole plan still describes the disk.
    paths = [c.path for c in plan.candidates]
    for candidate in plan.candidates:
        if not _contained(candidate.path, plan.root):
            raise PruneAborted(
                f"Refusing to prune {candidate.path}: it is not inside "
                f"{plan.root}. Nothing was deleted.")
        nested = _nested_registration(candidate.path, paths)
        if nested:
            raise PruneAborted(
                f"Refusing to prune {plan.root}: {candidate.path} and "
                f"{nested} are both in the plan and one is inside the other, "
                f"so the plan's byte total is not what deleting both would "
                f"free. Nothing was deleted.")
        current = content_fingerprint(candidate.path)
        if current.digest != candidate.inventory_digest:
            raise PruneAborted(
                f"Refusing to prune {plan.root}: "
                f"{os.path.relpath(candidate.path, plan.root)} is not what "
                f"the plan measured — it now holds {current.n_files} file(s) "
                f"and {human_bytes(current.size_bytes)} against the plan's "
                f"{candidate.n_files} and "
                f"{human_bytes(candidate.size_bytes)}. Something wrote there "
                f"after the plan was made. Nothing was deleted; make a new "
                f"plan and read it.")

    target = _open_if_present(registry, plan.root)

    # 3. The registry write, counted and verified, before any file goes.
    rows = 0
    if target is not None:
        try:
            rows = _mark_artifacts(
                target,
                [aid for c in plan.candidates for aid in c.artifact_ids],
                {"pruned_utc": _now(), "pruned_by_spacr": get_version(),
                 "pruned_freed_bytes": plan.total_bytes},
                forget=forget_rows)
        except PruneAborted as exc:
            raise PruneAborted(
                f"{exc} Nothing on disk was deleted: the registry write "
                f"happens before any file is removed, exactly so that this "
                f"failure costs nothing.") from exc

    # 4. Now delete.
    removed_files: List[str] = []
    truncated = False
    for candidate in plan.candidates:
        listing = _enumerate(candidate.path)
        if len(removed_files) + len(listing) > MAX_RECORDED_FILES:
            truncated = True
        else:
            removed_files.extend(listing)
        if os.path.isdir(candidate.path) and not os.path.islink(candidate.path):
            shutil.rmtree(candidate.path)
        elif os.path.exists(candidate.path):
            os.remove(candidate.path)

    # 5. And check.
    left = [c.path for c in plan.candidates if os.path.exists(c.path)]
    if left:
        raise PruneIncomplete(
            f"The prune of {plan.root} removed what it could, but "
            f"{len(left)} path(s) are still there: {', '.join(left[:3])}. "
            f"The registry has already recorded them as pruned; run the "
            f"prune again once the reason is fixed.")

    return PruneResult(
        root=plan.root,
        removed_paths=plan.paths,
        removed_files=() if truncated else tuple(removed_files),
        files_truncated=truncated,
        freed_bytes=plan.total_bytes,
        n_files=plan.total_files,
        registry_rows=rows,
        forgotten=bool(forget_rows),
        finished_utc=_now())


# ---------------------------------------------------------------------------
# Archiving
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArchiveItem:
    """One top-level entry an archive would move.

    :param source: where it is now.
    :param destination: where it would go.
    :param size_bytes: its size.
    :param n_files: files it holds.
    :param kind: its kind, when an artifact claims it.
    :param artifact_ids: the registry rows it carries.
    """

    source: str
    destination: str
    size_bytes: int = 0
    n_files: int = 0
    kind: str = OTHER_KIND
    artifact_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ArchivePlan:
    """What an archive would move, and where to.

    :param root: the project being archived from.
    :param destination: the folder it would move into.
    :param items: the top-level entries that would move.
    :param total_bytes: how much moves.
    :param total_files: how many files.
    :param whole_project: True when every top-level entry is moving.
    :param token: the confirmation; see :attr:`PrunePlan.token`.
    :param created_utc: when the plan was made.
    """

    root: str
    destination: str
    items: Tuple[ArchiveItem, ...] = ()
    total_bytes: int = 0
    total_files: int = 0
    whole_project: bool = False
    token: str = ""
    created_utc: str = ""

    def __bool__(self) -> bool:
        """True when there is something to move."""
        return bool(self.items)


@dataclass(frozen=True)
class ArchiveResult:
    """What an archive actually did.

    :param root: the origin.
    :param destination: where it went.
    :param moved: ``(source, destination)`` per entry, in the order moved.
    :param total_bytes: bytes moved.
    :param manifest_path: the manifest written at the destination.
    :param ledger_path: the record left at the origin.
    :param registered: artifacts registered in the destination's registry.
    :param finished_utc: when it finished.
    """

    root: str
    destination: str
    moved: Tuple[Tuple[str, str], ...] = ()
    total_bytes: int = 0
    manifest_path: str = ""
    ledger_path: str = ""
    registered: int = 0
    finished_utc: str = ""


def _read_ledger(path: str) -> List[Any]:
    """Return the archive records already at ``path``, or an empty list.

    A ledger nobody can parse is not a reason to lose the record about to be
    written, so an unreadable file starts a new list rather than raising.
    """
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
    except (OSError, ValueError):
        return []
    return list(existing) if isinstance(existing, list) else [existing]


def plan_archive(root: Any, destination: Any, *,
                 registry: Optional[Registry] = None,
                 paths: Optional[Sequence[str]] = None,
                 usage: Optional[ProjectUsage] = None) -> ArchivePlan:
    """Work out what moving a project — or part of one — would move.

    Entries are top-level: a whole project archive moves every child of the
    root, and a subset archive moves the paths named. Moving a *file out of*
    a registered folder is not offered, because half a ``merged/`` in two
    places is worse than either place having all of it.

    :param root: the project root.
    :param destination: the folder to move into. It must not already hold an
        entry with the same name; an archive never overwrites.
    :param registry: an open registry, for the provenance carried along.
    :param paths: entries to move. Defaults to everything under ``root``.
    :param usage: a :class:`ProjectUsage`, to save a second walk.
    :returns: an :class:`ArchivePlan`.
    :raises DataManagerError: when ``root`` is not a folder, when a named
        path is not inside it, or when the destination is inside the root.
    """
    project = _absolute(root)
    target_dir = _absolute(destination)
    if not os.path.isdir(project):
        raise DataManagerError(f"{project} is not a folder")
    if _contained(target_dir, project):
        raise DataManagerError(
            f"the archive destination {target_dir} is inside the project "
            f"being archived; moving a folder into itself loses it")

    report = usage if usage is not None else scan_project(project,
                                                          registry=registry)
    by_path = {entry.path: entry for entry in report.artifacts}

    if paths is None:
        chosen = [os.path.join(project, name)
                  for name in sorted(os.listdir(project))]
        whole = True
    else:
        chosen = []
        for path in paths:
            absolute = _absolute(path)
            if not _contained(absolute, project) or absolute == project:
                raise DataManagerError(
                    f"{absolute} is not inside {project}, so this project "
                    f"cannot archive it")
            chosen.append(absolute)
        whole = set(chosen) == {os.path.join(project, n)
                                for n in os.listdir(project)}

    items: List[ArchiveItem] = []
    for source in chosen:
        if not os.path.exists(source):
            continue
        # Every artifact this entry carries, not only one registered at
        # exactly this path: a whole-project archive moves `measurements/`,
        # and the database's provenance is inside it. Missing that is how the
        # destination ends up describing four of a project's seven artifacts.
        inside = [entry for path, entry in by_path.items()
                  if path == source or path.startswith(source + os.sep)]
        listing = _enumerate(source)
        size = sum(os.path.getsize(p) for p in listing)
        files = len(listing)
        ids = tuple(dict.fromkeys(
            a.artifact_id for entry in inside for a in entry.artifacts))
        kinds = sorted({entry.kind for entry in inside},
                       key=lambda k: _KIND_RANK.get(k, len(_KIND_RANK)))
        kind = kinds[0] if kinds else _classify_by_layout(
            os.path.relpath(source, project).replace(os.sep, "/"))
        items.append(ArchiveItem(
            source=source,
            destination=os.path.join(target_dir,
                                     os.path.relpath(source, project)),
            size_bytes=size, n_files=files, kind=kind, artifact_ids=ids))

    items.sort(key=lambda i: i.source)
    total = sum(i.size_bytes for i in items)
    return ArchivePlan(
        root=project, destination=target_dir, items=tuple(items),
        total_bytes=total, total_files=sum(i.n_files for i in items),
        whole_project=whole,
        token=_digest([project, target_dir, str(total)]
                      + [f"{i.source}:{i.size_bytes}" for i in items]),
        created_utc=_now())


def archive(plan: ArchivePlan, *, confirm: str,
            registry: Optional[Registry] = None) -> ArchiveResult:
    """Move a project somewhere else and leave a record of where it went.

    Three records are left, because one of them may move with the data:

    * a **manifest** at the destination naming the origin, the time, the
      spaCR version and every artifact that arrived, with its provenance;
    * a **ledger** at the origin — appended to, never overwritten — so a
      folder somebody finds nearly empty still says where its contents are;
    * **registry rows** at the destination, one per artifact, carrying the
      module, settings hash and inputs the artifact arrived with plus
      ``extra['archived_from']``. The destination is self-describing, and
      :func:`spacr.artifacts.by_project` on it answers.

    The origin's own rows are marked ``archived_to`` — through the same
    counted, verified write a prune uses — unless the registry file is itself
    moving, in which case the marks would travel with it and say the wrong
    thing.

    Nothing is overwritten: a destination entry that already exists stops the
    call before anything moves.

    :param plan: from :func:`plan_archive`.
    :param confirm: :attr:`ArchivePlan.token`.
    :param registry: the origin's registry; opened from the root when
        omitted.
    :returns: an :class:`ArchiveResult`.
    :raises ConfirmationRequired: without the right token.
    :raises ArchiveError: when a destination entry exists, or when a move
        cannot be verified afterwards.
    """
    if str(confirm) != plan.token:
        raise ConfirmationRequired(
            f"Archiving {plan.root} moves {human_bytes(plan.total_bytes)} to "
            f"{plan.destination}. Pass confirm=plan.token to carry it out.")
    if not plan.items:
        return ArchiveResult(root=plan.root, destination=plan.destination,
                             finished_utc=_now())

    for item in plan.items:
        if os.path.exists(item.destination):
            raise ArchiveError(
                f"{item.destination} already exists. An archive never "
                f"overwrites; move it aside or choose another destination. "
                f"Nothing was moved.")

    source_registry = _open_if_present(registry, plan.root)

    moving_registry = source_registry is not None and any(
        _contained(source_registry.path, item.source)
        or source_registry.path == item.source
        for item in plan.items)

    # Everything the artifacts know, read before the registry can move.
    provenance: List[Artifact] = []
    if source_registry is not None:
        known = {a.artifact_id: a for a in source_registry.by_project(plan.root)}
        for item in plan.items:
            provenance.extend(known[i] for i in item.artifact_ids if i in known)

    if source_registry is not None and not moving_registry:
        _mark_artifacts(
            source_registry,
            [i for item in plan.items for i in item.artifact_ids],
            {"archived_utc": _now(), "archived_to": plan.destination})

    # Read before the move: a whole-project archive moves the ledger too, and
    # the origin's earlier archives must not be forgotten because the file
    # that recorded them went with the data.
    ledger_path = os.path.join(plan.root, ARCHIVE_LEDGER_NAME)
    ledger = _read_ledger(ledger_path)

    os.makedirs(plan.destination, exist_ok=True)
    moved: List[Tuple[str, str]] = []
    for item in plan.items:
        os.makedirs(os.path.dirname(item.destination) or plan.destination,
                    exist_ok=True)
        shutil.move(item.source, item.destination)
        if not os.path.exists(item.destination):
            raise ArchiveError(
                f"{item.source} was moved to {item.destination}, which is "
                f"not there afterwards. {len(moved)} earlier entr(ies) did "
                f"move; the manifest was not written. Check the destination "
                f"filesystem before doing anything else.")
        moved.append((item.source, item.destination))

    # The destination is made self-describing: same module, kind, role,
    # settings hash and inputs, at the new path.
    registered = 0
    if provenance:
        destination_registry = open_registry(plan.destination)
        for artifact in provenance:
            relative = os.path.relpath(artifact.path, plan.root)
            new_path = os.path.join(plan.destination, relative)
            if not os.path.exists(new_path):
                continue
            destination_registry.register(
                module=artifact.module, kind=artifact.kind, role=artifact.role,
                path=new_path, project=plan.destination,
                settings=artifact.settings,
                settings_digest=artifact.settings_hash,
                inputs=artifact.inputs, run_id=artifact.run_id,
                status=artifact.status,
                extra={**artifact.extra,
                       "archived_from": artifact.path,
                       "archived_from_project": artifact.project,
                       "archived_from_artifact_id": artifact.artifact_id,
                       "archived_utc": _now()})
            registered += 1

    record = {
        "spacr_version": get_version(),
        "archived_utc": _now(),
        "origin": plan.root,
        "destination": plan.destination,
        "whole_project": plan.whole_project,
        "total_bytes": plan.total_bytes,
        "total_files": plan.total_files,
        "entries": [
            {"source": item.source, "destination": item.destination,
             "kind": item.kind, "size_bytes": item.size_bytes,
             "n_files": item.n_files,
             "artifact_ids": list(item.artifact_ids)}
            for item in plan.items
        ],
        "artifacts": [a.to_dict() for a in provenance],
    }
    manifest_path = os.path.join(plan.destination, ARCHIVE_MANIFEST_NAME)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)

    os.makedirs(plan.root, exist_ok=True)
    ledger.append({k: record[k] for k in
                   ("spacr_version", "archived_utc", "origin", "destination",
                    "whole_project", "total_bytes", "total_files", "entries")})
    with open(ledger_path, "w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2, sort_keys=True)

    return ArchiveResult(
        root=plan.root, destination=plan.destination, moved=tuple(moved),
        total_bytes=plan.total_bytes, manifest_path=manifest_path,
        ledger_path=ledger_path, registered=registered,
        finished_utc=_now())
