"""Two runs, side by side: what changed, how many fewer, which hits moved.

A screen is re-run with one setting nudged and the numbers come out
different. Nothing in spaCR has ever answered *why*: the settings that
produced a result were not recorded next to it, the object counts were
never compared, and the hit list was a CSV you diffed by eye. So the
honest answer to "is this better than last week's run?" was to open two
folders and squint.

This module answers it, in three parts, from
:mod:`spacr.artifacts` — the registry every output now registers with —
rather than from a filesystem scan:

**Settings diff**
    Which parameters moved, grouped by the same headings the settings
    panel groups them by. :mod:`spacr.qt.settings_diff` does the work.

**Count diff**
    Objects, wells and fields, per plate and overall. A run that produced
    12% fewer cells is the single most useful early signal that something
    changed for the worse, and it is visible long before anyone looks at
    a p-value.

**Hit-list diff**
    For runs that produced regression results: which hits appeared, which
    vanished, and which merely moved. Rank churn among a stable hit set
    matters as much as set membership — a screen whose top ten reshuffle
    every run is not a screen anyone should publish from.

Not every pair of runs *can* be compared. Different plates, a module one
run never ran, a different spaCR version: :func:`comparability` says so
in words, and :func:`compare_runs` refuses to produce the three tables
rather than presenting a misleading one. A version difference is
deliberately a warning and not a blocker — it does not stop the
comparison, but it has to be on screen, because a Cellpose upgrade
between two runs can account for a count change on its own.

Public API
----------
``runs_in``
    Group a project's registered artifacts into :class:`RunRef` runs.
``comparability``
    Whether two runs may be compared, and what to say if not.
``compare_runs``
    All three diffs, or the reason there are none.
``count_database``, ``diff_counts``
    The count half on its own, for a caller holding two database paths.
``read_hits``, ``diff_hits``
    The hit-list half on its own, for two result CSVs.
"""
from __future__ import annotations

import csv
import os
import sqlite3
from dataclasses import dataclass, field
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple, Union)

from .qt.settings_diff import SettingsDiff, diff_settings_grouped

__all__ = [
    "COUNT_METRICS",
    "OBJECT_TABLES",
    "CountDiff",
    "CountRow",
    "Comparability",
    "Finding",
    "Hit",
    "HitChange",
    "HitDiff",
    "HitList",
    "RunComparison",
    "RunCounts",
    "RunRef",
    "comparability",
    "compare_runs",
    "count_database",
    "diff_counts",
    "diff_hits",
    "read_hits",
    "runs_in",
]


#: The per-object measurement tables Measure writes, in the order a count
#: table lists them. ``png_list`` is last because it is a crop index rather
#: than an object table — it is counted because a run that measured the same
#: cells but exported half the crops is a real and confusing failure.
OBJECT_TABLES: Tuple[str, ...] = (
    "cell", "nucleus", "pathogen", "cytoplasm", "png_list",
)

#: Acquisition-level metrics, derived from whichever object table is
#: richest. Wells and fields are properties of the plate, not of the object
#: type, so counting them once per table would print the same number five
#: times and imply five independent measurements.
COUNT_METRICS: Tuple[str, ...] = ("plates", "wells", "fields")

#: Columns that identify where an object came from. Every object table
#: carries them (``spacr.utils`` writes them on the way in).
_PLATE_COLUMN = "plateID"
_WELL_COLUMNS = ("rowID", "columnID")
_FIELD_COLUMNS = ("rowID", "columnID", "fieldID")

#: Column names a hit list may use for the thing being ranked, best first.
#: ``feature`` and ``variable`` are what :mod:`spacr.ml` writes; the rest
#: cover gene- and gRNA-level result files and hand-made CSVs.
_HIT_KEY_COLUMNS: Tuple[str, ...] = (
    "gene", "grna", "feature", "variable", "n_gene", "name", "id", "target",
)

#: Column names a hit list may use for effect size, best first.
_HIT_SCORE_COLUMNS: Tuple[str, ...] = (
    "coefficient", "coef", "effect", "effect_size", "beta", "score",
    "log2fc", "fold_change",
)

#: Settings keys that name the folders a run read. Used to work out which
#: plates a run touched when its outputs are not on disk to be counted.
_SOURCE_KEYS: Tuple[str, ...] = ("src", "source", "dst")


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

#: A finding that stops the comparison.
BLOCKING = "blocking"
#: A finding the user must see but that does not stop anything.
WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    """One reason two runs are hard, or impossible, to compare.

    :param code: machine-readable, e.g. ``"version-changed"``.
    :param severity: :data:`BLOCKING` or :data:`WARNING`.
    :param message: one sentence, written for the user.
    """

    code: str
    severity: str
    message: str

    @property
    def blocking(self) -> bool:
        """True when this finding stops the comparison."""
        return self.severity == BLOCKING

    def __str__(self) -> str:
        """The message."""
        return self.message


@dataclass(frozen=True)
class Comparability:
    """Whether two runs may be put side by side, and what to say about it.

    :param comparable: no blocking finding was raised.
    :param findings: everything found, blockers first.
    :param shared_modules: modules both runs ran.
    :param shared_kinds: output kinds both runs produced.
    :param shared_plates: plates both runs touched. Empty when neither run
        names any, which is not the same as "no overlap".
    """

    comparable: bool
    findings: Tuple[Finding, ...] = ()
    shared_modules: Tuple[str, ...] = ()
    shared_kinds: Tuple[str, ...] = ()
    shared_plates: Tuple[str, ...] = ()

    @property
    def blockers(self) -> Tuple[Finding, ...]:
        """The findings that stop the comparison."""
        return tuple(f for f in self.findings if f.blocking)

    @property
    def warnings(self) -> Tuple[Finding, ...]:
        """The findings that do not stop it but have to be on screen."""
        return tuple(f for f in self.findings if not f.blocking)

    @property
    def version_changed(self) -> bool:
        """True when the two runs came out of different spaCR versions.

        Surfaced on its own because it explains a count change without
        any setting having moved.
        """
        return any(f.code == "version-changed" for f in self.findings)

    def __bool__(self) -> bool:
        """True when the runs may be compared."""
        return self.comparable

    def summary(self) -> str:
        """One line: the verdict and the reasons behind it."""
        if not self.findings:
            return "These runs are comparable."
        lead = ("These runs are comparable." if self.comparable
                else "These runs are not comparable.")
        return lead + " " + " ".join(f.message for f in self.findings)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunRef:
    """One run, assembled from the artifacts it registered.

    A run is what a :attr:`spacr.artifacts.Artifact.run_id` groups. An
    artifact registered without one is its own single-output run rather
    than being dropped: an output nobody stamped is still an output
    somebody may want to compare against.

    :param run_id: the run id, or ``"artifact:<id>"`` for an unstamped one.
    :param project: the project root the outputs belong to.
    :param modules: producing modules, sorted.
    :param kinds: output kinds, sorted.
    :param plates: plates the run touched, sorted. Derived from its
        settings; the count diff finds the real ones in the database.
    :param spacr_version: the version that produced the outputs. Empty
        when unrecorded; ``"mixed"`` when the run's artifacts disagree.
    :param settings: the material settings, from the newest artifact that
        carries any.
    :param settings_hash: that run's settings digest.
    :param created_ns: the newest artifact's registration time.
    :param created_utc: the same instant, ISO-8601.
    :param status: the worst status any of its artifacts recorded.
    :param artifacts: the artifacts themselves, newest first.
    """

    run_id: str
    project: str = ""
    modules: Tuple[str, ...] = ()
    kinds: Tuple[str, ...] = ()
    plates: Tuple[str, ...] = ()
    spacr_version: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)
    settings_hash: str = ""
    created_ns: int = 0
    created_utc: str = ""
    status: str = "complete"
    artifacts: Tuple[Any, ...] = ()

    @property
    def label(self) -> str:
        """A one-line name for a dropdown: when, what and which version."""
        stamp = (self.created_utc or "").replace("T", " ")[:19]
        modules = "+".join(self.modules) or "?"
        version = f" · spaCR {self.spacr_version}" if self.spacr_version else ""
        return f"{stamp} · {modules}{version}"

    def artifact_of(self, kind: str) -> Optional[Any]:
        """The newest artifact of ``kind`` this run produced, or ``None``.

        :param kind: a :mod:`spacr.ports` kind, e.g. ``"measurements-db"``.
        """
        for artifact in self.artifacts:
            if getattr(artifact, "kind", "") == kind:
                return artifact
        return None


def runs_in(registry: Any, project: Union[str, os.PathLike, None] = None, *,
            limit: Optional[int] = None) -> List[RunRef]:
    """Group everything a project registered into runs, newest first.

    This is how the comparison screen fills its two dropdowns. It reads
    :meth:`spacr.artifacts.Registry.by_project` rather than walking the
    filesystem, so a run whose outputs have since been deleted still
    appears — with the settings that produced them — instead of silently
    vanishing from the list of things you can compare against.

    :param registry: a :class:`spacr.artifacts.Registry`.
    :param project: the project root; ``None`` means the registry's own,
        ``""`` means every project in a shared registry file.
    :param limit: keep only the newest ``limit`` runs.
    :returns: :class:`RunRef` objects, newest first.
    """
    artifacts = registry.by_project(project)
    grouped: Dict[str, List[Any]] = {}
    for artifact in artifacts:
        run_id = (getattr(artifact, "run_id", "") or "").strip()
        key = run_id or f"artifact:{getattr(artifact, 'artifact_id', '')}"
        grouped.setdefault(key, []).append(artifact)

    runs = [_run_from_artifacts(run_id, items)
            for run_id, items in grouped.items()]
    runs.sort(key=lambda r: (-r.created_ns, r.run_id))
    return runs[:limit] if limit is not None else runs


def _run_from_artifacts(run_id: str, artifacts: Sequence[Any]) -> RunRef:
    """Fold one run's artifacts into a :class:`RunRef`."""
    ordered = sorted(artifacts,
                     key=lambda a: (-int(getattr(a, "created_ns", 0) or 0),
                                    str(getattr(a, "artifact_id", ""))))
    newest = ordered[0]
    versions = {str(getattr(a, "spacr_version", "") or "") for a in ordered}
    versions.discard("")
    if len(versions) == 1:
        version = versions.pop()
    elif versions:
        # Two modules of one run produced by different spaCR versions is
        # itself a finding — reporting the newest would hide it.
        version = "mixed"
    else:
        version = ""

    settings: Dict[str, Any] = {}
    settings_hash = ""
    for artifact in ordered:
        candidate = getattr(artifact, "settings", None)
        if candidate:
            settings = dict(candidate)
            settings_hash = str(getattr(artifact, "settings_hash", "") or "")
            break

    statuses = {str(getattr(a, "status", "") or "") for a in ordered}
    for worst in ("failed", "partial"):
        if worst in statuses:
            status = worst
            break
    else:
        status = "complete"

    return RunRef(
        run_id=run_id,
        project=str(getattr(newest, "project", "") or ""),
        modules=tuple(sorted({str(getattr(a, "module", "") or "")
                              for a in ordered} - {""})),
        kinds=tuple(sorted({str(getattr(a, "kind", "") or "")
                            for a in ordered} - {""})),
        plates=_recorded_plates(ordered) or plates_of(settings),
        spacr_version=version,
        settings=settings,
        settings_hash=settings_hash,
        created_ns=int(getattr(newest, "created_ns", 0) or 0),
        created_utc=str(getattr(newest, "created_utc", "") or ""),
        status=status,
        artifacts=tuple(ordered),
    )


def _recorded_plates(artifacts: Sequence[Any]) -> Tuple[str, ...]:
    """Plates a producer recorded in an artifact's ``extra``, if any.

    Preferred over the settings, because the settings do not have them:
    ``src`` is on :data:`spacr.resume.COSMETIC_SETTINGS` (where a run read
    from cannot change its numbers), so
    :func:`spacr.artifacts.material_settings` drops it before the registry
    stores it. Which plate a number is *about* is not cosmetic, so a
    producer that knows says so here, and
    :func:`compare_runs` falls back to counting the database.
    """
    found: set = set()
    for artifact in artifacts:
        extra = getattr(artifact, "extra", None) or {}
        value = extra.get("plates") if isinstance(extra, Mapping) else None
        if value is None:
            continue
        for item in (value if isinstance(value, (list, tuple, set)) else [value]):
            text = str(item).strip()
            if text:
                found.add(text)
    return tuple(sorted(found))


def plates_of(settings: Optional[Mapping[str, Any]]) -> Tuple[str, ...]:
    """Return the plate folders a run's settings name, sorted.

    A spaCR run is pointed at one or more plate folders through ``src``,
    and the folder's basename is the plate. Used to notice that two runs
    were pointed at *different experiments*, which is the one difference
    that makes a count comparison meaningless rather than merely
    interesting.

    Note that a run read back out of the artifact registry usually has
    *no* ``src``: it is cosmetic as far as the settings hash is concerned
    and is not stored. This is for the callers that hold the real
    settings — a settings CSV, a run journal entry, a queued job.

    :param settings: a settings dict, or None.
    :returns: plate names, sorted and de-duplicated. Empty when the
        settings name no source at all.
    """
    if not settings:
        return ()
    found: set = set()
    for key in _SOURCE_KEYS:
        value = settings.get(key)
        if value is None:
            continue
        for item in (value if isinstance(value, (list, tuple, set)) else [value]):
            text = str(item).strip().rstrip("/\\")
            if text:
                found.add(os.path.basename(text) or text)
    return tuple(sorted(found))


# ---------------------------------------------------------------------------
# Comparability
# ---------------------------------------------------------------------------

def comparability(a: RunRef, b: RunRef, *,
                  a_plates: Optional[Sequence[str]] = None,
                  b_plates: Optional[Sequence[str]] = None) -> Comparability:
    """Decide whether two runs may be compared, and say why not.

    Blocking, because diffing past them produces a table that is worse
    than no table:

    * **no shared module** — the runs did different things, so there is
      no output of the same kind to line up.
    * **no shared output kind** — the modules overlap but what they
      actually produced does not.
    * **different plates** — both runs name plates and the sets are
      disjoint. Counting objects across two different experiments and
      calling the difference a regression is exactly the misleading
      table this check exists to prevent.

    Warnings, because they change what the numbers mean but the
    comparison is still worth seeing:

    * **version-changed** — different spaCR versions. Called out on its
      own: a segmentation change between versions moves object counts
      with no setting having moved, so a count delta must never be read
      without it.
    * **different project**, **partial plate overlap**, **modules only
      one run ran**, **a run that did not finish**.

    :param a: the baseline run.
    :param b: the run being compared to it.
    :param a_plates: the plates the baseline really touched, when the
        caller knows better than :attr:`RunRef.plates` does — which it
        does whenever it has counted the database.
    :param b_plates: the same for the compared run.
    :returns: a :class:`Comparability`; ``bool()`` of it is the verdict.
    """
    findings: List[Finding] = []

    shared_modules = tuple(sorted(set(a.modules) & set(b.modules)))
    shared_kinds = tuple(sorted(set(a.kinds) & set(b.kinds)))
    a_names = tuple(a.plates if a_plates is None else a_plates)
    b_names = tuple(b.plates if b_plates is None else b_plates)
    a_plates, b_plates = set(a_names), set(b_names)
    shared_plates = tuple(sorted(a_plates & b_plates))

    if a.run_id and a.run_id == b.run_id:
        findings.append(Finding(
            "same-run", WARNING,
            "Both sides are the same run, so everything will match."))

    if not shared_modules:
        findings.append(Finding(
            "no-shared-module", BLOCKING,
            f"These runs ran different modules "
            f"({_join(a.modules)} vs {_join(b.modules)}), so there is "
            f"nothing of the same kind to line up."))
    elif not shared_kinds:
        findings.append(Finding(
            "no-shared-kind", BLOCKING,
            f"The runs share a module but no output: one produced "
            f"{_join(a.kinds)}, the other {_join(b.kinds)}."))

    if a_plates and b_plates and not shared_plates:
        findings.append(Finding(
            "different-plates", BLOCKING,
            f"These runs are of different plates ({_join(a_names)} vs "
            f"{_join(b_names)}); comparing their counts would compare "
            f"two experiments."))
    elif shared_plates and (a_plates != b_plates):
        findings.append(Finding(
            "partial-plate-overlap", WARNING,
            f"Only {_join(shared_plates)} is in both runs; counts over "
            f"the others are not comparable."))

    if a.spacr_version and b.spacr_version and a.spacr_version != b.spacr_version:
        findings.append(Finding(
            "version-changed", WARNING,
            f"These runs came out of different spaCR versions "
            f"({a.spacr_version} vs {b.spacr_version}), which can change "
            f"object counts on its own."))

    if a.project and b.project and a.project != b.project:
        findings.append(Finding(
            "different-project", WARNING,
            "These runs are in different projects; their plates and "
            "metadata may not mean the same thing."))

    only_a = tuple(sorted(set(a.modules) - set(b.modules)))
    only_b = tuple(sorted(set(b.modules) - set(a.modules)))
    if shared_modules and (only_a or only_b):
        extra = []
        if only_a:
            extra.append(f"only the first ran {_join(only_a)}")
        if only_b:
            extra.append(f"only the second ran {_join(only_b)}")
        findings.append(Finding(
            "partial-module-overlap", WARNING,
            f"The runs do not cover the same modules: {', '.join(extra)}."))

    for run, side in ((a, "first"), (b, "second")):
        if run.status in ("partial", "failed"):
            findings.append(Finding(
                f"{run.status}-run", WARNING,
                f"The {side} run is marked {run.status}, so its numbers "
                f"are not a whole run's worth."))

    findings.sort(key=lambda f: 0 if f.blocking else 1)
    return Comparability(
        comparable=not any(f.blocking for f in findings),
        findings=tuple(findings),
        shared_modules=shared_modules,
        shared_kinds=shared_kinds,
        shared_plates=shared_plates,
    )


def _join(items: Iterable[str]) -> str:
    """Render a list of names for a sentence; ``"nothing"`` when empty."""
    items = [str(i) for i in items if str(i)]
    if not items:
        return "nothing"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunCounts:
    """What one run actually produced, counted out of its database.

    :param path: the database it was counted from.
    :param available: whether it could be read.
    :param note: why not, when it could not.
    :param overall: metric → count over the whole run.
    :param per_plate: plate → metric → count.
    """

    path: str = ""
    available: bool = False
    note: str = ""
    overall: Dict[str, int] = field(default_factory=dict)
    per_plate: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def plates(self) -> Tuple[str, ...]:
        """The plates found in the database, sorted."""
        return tuple(sorted(self.per_plate))

    def __bool__(self) -> bool:
        """True when there is something to compare."""
        return self.available and bool(self.overall)


@dataclass(frozen=True)
class CountRow:
    """One count, on both sides.

    :param scope: ``"overall"`` or a plate id.
    :param metric: ``"cell"``, ``"wells"``, …
    :param a: the baseline count, or ``None`` when that side has no such
        metric at all — which is not the same as zero.
    :param b: the comparison count, or ``None``.
    """

    scope: str
    metric: str
    a: Optional[int]
    b: Optional[int]

    @property
    def delta(self) -> Optional[int]:
        """``b - a``, or ``None`` when one side is missing."""
        if self.a is None or self.b is None:
            return None
        return self.b - self.a

    @property
    def pct(self) -> Optional[float]:
        """Percentage change from ``a`` to ``b``.

        ``None`` when a side is missing, and when the baseline is zero —
        "up from nothing" has no percentage, and printing ``inf`` in a
        table that a user scans for the biggest number is worse than
        printing nothing.
        """
        if self.a is None or self.b is None or self.a == 0:
            return None
        return 100.0 * (self.b - self.a) / float(self.a)

    @property
    def changed(self) -> bool:
        """True when the two sides do not hold the same number."""
        return self.a != self.b


@dataclass(frozen=True)
class CountDiff:
    """The count comparison: overall first, then one block per plate.

    :param rows: every metric on both sides, unchanged ones included —
        a count that did *not* move is evidence, unlike a setting that
        did not move.
    :param a: the baseline counts.
    :param b: the comparison counts.
    """

    rows: Tuple[CountRow, ...] = ()
    a: RunCounts = field(default_factory=RunCounts)
    b: RunCounts = field(default_factory=RunCounts)

    @property
    def available(self) -> bool:
        """True when both sides could be counted."""
        return self.a.available and self.b.available

    @property
    def note(self) -> str:
        """Why there is nothing to show, when there is nothing."""
        return "; ".join(n for n in (self.a.note, self.b.note) if n)

    @property
    def changed(self) -> Tuple[CountRow, ...]:
        """Only the rows that moved."""
        return tuple(r for r in self.rows if r.changed)

    def overall(self) -> Tuple[CountRow, ...]:
        """The whole-run rows."""
        return tuple(r for r in self.rows if r.scope == "overall")

    def for_plate(self, plate: str) -> Tuple[CountRow, ...]:
        """The rows for one plate."""
        return tuple(r for r in self.rows if r.scope == plate)

    @property
    def plates(self) -> Tuple[str, ...]:
        """Every plate either side reported, sorted."""
        return tuple(sorted({r.scope for r in self.rows} - {"overall"}))

    def worst(self) -> Optional[CountRow]:
        """The overall row that dropped the most, in percent.

        The one number worth putting in a headline: a run that produced
        12% fewer cells has a problem, and the user should not have to
        find that by reading a table.
        """
        losses = [r for r in self.overall()
                  if r.pct is not None and r.pct < 0]
        if not losses:
            return None
        return min(losses, key=lambda r: r.pct)

    def headline(self) -> str:
        """One sentence about the biggest drop, or that nothing moved."""
        if not self.available:
            return self.note or "Counts are not available for these runs."
        row = self.worst()
        if row is None:
            if not self.changed:
                return "Every count matched."
            return "No count dropped."
        return (f"{row.metric}: {row.a:,} → {row.b:,} "
                f"({row.pct:+.1f}%) over the whole run.")


def count_database(path: Union[str, os.PathLike, None]) -> RunCounts:
    """Count objects, wells and fields in one measurements database.

    Deliberately tolerant: a database missing ``pathogen`` is a run that
    segmented no pathogens, not an error, and a run whose database has
    been deleted must report that rather than raise — the comparison
    screen has to say *why* half the table is empty.

    :param path: the ``measurements.db`` to count.
    :returns: a :class:`RunCounts`; ``available`` is False with a ``note``
        when there was nothing to count.
    """
    target = str(path or "")
    if not target:
        return RunCounts(note="no measurements database was registered")
    if not os.path.isfile(target):
        return RunCounts(path=target,
                         note=f"{os.path.basename(target)} is no longer on disk")

    overall: Dict[str, int] = {}
    per_plate: Dict[str, Dict[str, int]] = {}
    try:
        connection = sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=30)
    except sqlite3.Error as exc:
        return RunCounts(path=target, note=f"could not open the database: {exc}")
    try:
        # ``connect`` on a file that is not SQLite succeeds; the failure
        # arrives on the first statement. So the whole read is guarded,
        # not just the open — a truncated or half-written database is a
        # thing the comparison has to *report*, and an exception here
        # would take the screen down instead.
        present = _tables(connection)
        richest = ""
        for table in OBJECT_TABLES:
            if table not in present:
                continue
            columns = _columns(connection, table)
            total = _scalar(connection, f'SELECT COUNT(*) FROM "{table}"')
            if total is None:
                continue
            overall[table] = total
            if not richest and _PLATE_COLUMN in columns:
                richest = table
            if _PLATE_COLUMN not in columns:
                continue
            for plate, count in _grouped(connection, table, _PLATE_COLUMN):
                per_plate.setdefault(plate, {})[table] = count

        if richest:
            columns = _columns(connection, richest)
            overall.update(_acquisition_counts(connection, richest, columns))
            for plate in list(per_plate):
                per_plate[plate].update(_acquisition_counts(
                    connection, richest, columns, plate=plate))
    except sqlite3.Error as exc:
        return RunCounts(path=target, note=f"could not read the database: {exc}")
    finally:
        connection.close()

    if not overall:
        return RunCounts(path=target, available=True,
                         note=(f"{os.path.basename(target)} holds none of the "
                               f"measurement tables"),
                         overall={}, per_plate={})
    return RunCounts(path=target, available=True, overall=overall,
                     per_plate=per_plate)


def _acquisition_counts(connection: sqlite3.Connection, table: str,
                        columns: Sequence[str], *,
                        plate: str = "") -> Dict[str, int]:
    """Plate / well / field counts from one object table."""
    out: Dict[str, int] = {}
    where, params = ("", ())
    if plate:
        where, params = f' WHERE "{_PLATE_COLUMN}" = ?', (plate,)
    for metric, needed in (("plates", (_PLATE_COLUMN,)),
                           ("wells", _WELL_COLUMNS),
                           ("fields", _FIELD_COLUMNS)):
        if plate and metric == "plates":
            continue
        if any(column not in columns for column in needed):
            continue
        expression = " || '_' || ".join(f'"{c}"' for c in needed)
        value = _scalar(
            connection,
            f'SELECT COUNT(DISTINCT {expression}) FROM "{table}"{where}',
            params)
        if value is not None:
            out[metric] = value
    return out


def _tables(connection: sqlite3.Connection) -> set:
    """Every table and view name in the database.

    Views count. A project that keeps ``cell`` as a view over partitioned
    storage is still a project whose cells can be counted, and excluding
    views would report zero cells for it rather than saying it could not
    look — which is the failure mode this whole module exists to avoid.
    A view whose definition no longer resolves fails its ``COUNT`` and is
    skipped, like any other table that will not answer.
    """
    return {str(row[0]) for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}


def _columns(connection: sqlite3.Connection, table: str) -> Tuple[str, ...]:
    """Column names of one table, or ``()`` if SQLite will not say.

    ``PRAGMA table_info`` on a view whose definition no longer resolves
    raises rather than returning nothing, so this is guarded like every
    other read here: one unreadable table costs that table's row, not the
    whole comparison.
    """
    try:
        return tuple(
            str(row[1])
            for row in connection.execute(f'PRAGMA table_info("{table}")'))
    except sqlite3.Error:
        return ()


def _scalar(connection: sqlite3.Connection, sql: str,
            params: Sequence[Any] = ()) -> Optional[int]:
    """Run a one-value query, returning ``None`` if SQLite refuses it."""
    try:
        row = connection.execute(sql, tuple(params)).fetchone()
    except sqlite3.Error:
        return None
    return int(row[0]) if row and row[0] is not None else None


def _grouped(connection: sqlite3.Connection, table: str,
             column: str) -> List[Tuple[str, int]]:
    """``(value, count)`` for one grouping column."""
    try:
        rows = connection.execute(
            f'SELECT "{column}", COUNT(*) FROM "{table}" GROUP BY "{column}"'
        ).fetchall()
    except sqlite3.Error:
        return []
    return [(str(row[0]), int(row[1])) for row in rows if row[0] is not None]


def diff_counts(a: RunCounts, b: RunCounts) -> CountDiff:
    """Line two count tables up, overall first and then per plate.

    :param a: the baseline counts.
    :param b: the comparison counts.
    :returns: a :class:`CountDiff`. Rows are emitted for every metric
        either side reported, so a table that vanished between runs shows
        as ``None`` on one side rather than being absent.
    """
    rows: List[CountRow] = []
    for metric in _metric_order(set(a.overall) | set(b.overall)):
        rows.append(CountRow("overall", metric,
                             a.overall.get(metric), b.overall.get(metric)))
    for plate in sorted(set(a.per_plate) | set(b.per_plate)):
        a_plate = a.per_plate.get(plate, {})
        b_plate = b.per_plate.get(plate, {})
        for metric in _metric_order(set(a_plate) | set(b_plate)):
            rows.append(CountRow(plate, metric,
                                 a_plate.get(metric), b_plate.get(metric)))
    return CountDiff(rows=tuple(rows), a=a, b=b)


def _metric_order(present: Iterable[str]) -> List[str]:
    """Object tables in declaration order, then the acquisition metrics."""
    present = set(present)
    ordered = [m for m in OBJECT_TABLES if m in present]
    ordered += [m for m in COUNT_METRICS if m in present]
    return ordered + sorted(present - set(ordered))


# ---------------------------------------------------------------------------
# Hit lists
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Hit:
    """One row of a hit list.

    :param key: the thing being called — a gene, a gRNA, a feature.
    :param rank: 1-based position; see :func:`read_hits` for the ordering.
    :param score: the effect size it was ranked on, when there was one.
    """

    key: str
    rank: int
    score: Optional[float] = None


@dataclass(frozen=True)
class HitList:
    """One run's hits, ranked.

    :param path: the CSV they were read from.
    :param available: whether it could be read.
    :param note: why not, when it could not.
    :param hits: the rows, best rank first.
    :param key_column: the column the keys came from.
    :param score_column: the column the ranking used, or ``""`` when the
        file's own order was kept.
    """

    path: str = ""
    available: bool = False
    note: str = ""
    hits: Tuple[Hit, ...] = ()
    key_column: str = ""
    score_column: str = ""

    @property
    def keys(self) -> Tuple[str, ...]:
        """Every key, in rank order."""
        return tuple(h.key for h in self.hits)

    def by_key(self) -> Dict[str, Hit]:
        """key → :class:`Hit`."""
        return {h.key: h for h in self.hits}

    def __len__(self) -> int:
        """Number of hits."""
        return len(self.hits)

    def __bool__(self) -> bool:
        """True when there is a list to compare."""
        return self.available and bool(self.hits)


@dataclass(frozen=True)
class HitChange:
    """What happened to one key between two hit lists.

    :param key: the gene / gRNA / feature.
    :param status: ``"appeared"``, ``"vanished"``, ``"moved"`` or ``"held"``.
    :param a_rank: rank in the baseline, or ``None``.
    :param b_rank: rank in the comparison, or ``None``.
    :param a_score: effect size in the baseline, when there was one.
    :param b_score: effect size in the comparison.
    """

    key: str
    status: str
    a_rank: Optional[int] = None
    b_rank: Optional[int] = None
    a_score: Optional[float] = None
    b_score: Optional[float] = None

    @property
    def rank_delta(self) -> Optional[int]:
        """``b_rank - a_rank``; negative means it climbed."""
        if self.a_rank is None or self.b_rank is None:
            return None
        return self.b_rank - self.a_rank


@dataclass(frozen=True)
class HitDiff:
    """Which hits appeared, which vanished, and which merely moved.

    :param changes: every key from either list, in a stable order:
        appeared, vanished, then the shared ones by how far they moved.
    :param a: the baseline list.
    :param b: the comparison list.
    """

    changes: Tuple[HitChange, ...] = ()
    a: HitList = field(default_factory=HitList)
    b: HitList = field(default_factory=HitList)

    @property
    def available(self) -> bool:
        """True when both lists could be read."""
        return self.a.available and self.b.available

    @property
    def note(self) -> str:
        """Why there is nothing to show, when there is nothing."""
        return "; ".join(n for n in (self.a.note, self.b.note) if n)

    @property
    def appeared(self) -> Tuple[HitChange, ...]:
        """Keys only the second run called."""
        return tuple(c for c in self.changes if c.status == "appeared")

    @property
    def vanished(self) -> Tuple[HitChange, ...]:
        """Keys only the first run called."""
        return tuple(c for c in self.changes if c.status == "vanished")

    @property
    def moved(self) -> Tuple[HitChange, ...]:
        """Keys in both lists whose rank changed, biggest move first."""
        return tuple(c for c in self.changes if c.status == "moved")

    @property
    def held(self) -> Tuple[HitChange, ...]:
        """Keys in both lists at the same rank."""
        return tuple(c for c in self.changes if c.status == "held")

    @property
    def n_shared(self) -> int:
        """Keys both runs called."""
        return len(self.moved) + len(self.held)

    @property
    def churn(self) -> float:
        """Fraction of the shared hits whose rank moved, 0.0–1.0.

        The number that says a screen is unstable even when its hit set
        is not: identical membership with a reshuffled top ten scores 1.0
        here and 0 appeared / 0 vanished up there.
        """
        if not self.n_shared:
            return 0.0
        return len(self.moved) / float(self.n_shared)

    @property
    def identical(self) -> bool:
        """True when both membership and order match exactly."""
        return not (self.appeared or self.vanished or self.moved)

    def headline(self) -> str:
        """One sentence: membership first, then churn."""
        if not self.available:
            return self.note or "Neither run produced a hit list."
        if self.identical:
            return f"The same {self.n_shared} hits, in the same order."
        parts = []
        if self.appeared:
            parts.append(f"{len(self.appeared)} appeared")
        if self.vanished:
            parts.append(f"{len(self.vanished)} vanished")
        if self.moved:
            parts.append(f"{len(self.moved)} of {self.n_shared} shared hits "
                         f"changed rank ({self.churn:.0%} churn)")
        return "; ".join(parts) + "."


def read_hits(path: Union[str, os.PathLike, None], *,
              key_column: str = "",
              score_column: str = "",
              limit: Optional[int] = None) -> HitList:
    """Read one hit list off a results CSV.

    The rank is defined here, once, so both sides of a diff are ranked
    the same way: rows are sorted by **descending absolute score**, ties
    broken by key. Absolute, because a screen's strongest hit is its
    largest effect in either direction and ranking a strong protective
    hit below a weak sensitising one would report rank churn that is an
    artefact of the sort. When no score column can be found the file's
    own row order is kept and :attr:`HitList.score_column` is empty, so a
    caller can tell a real ranking from a preserved one.

    :param path: a ``results*.csv`` from the regression module, or any
        CSV with a name column.
    :param key_column: force the key column instead of detecting it.
    :param score_column: force the score column. Pass ``"-"`` to rank by
        file order even when a score column exists.
    :param limit: keep only the top ``limit`` rows after ranking.
    :returns: a :class:`HitList`; ``available`` is False with a ``note``
        when there was nothing to read.
    """
    target = str(path or "")
    if not target:
        return HitList(note="no regression results were registered")
    if not os.path.isfile(target):
        return HitList(path=target,
                       note=f"{os.path.basename(target)} is no longer on disk")
    try:
        with open(target, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [str(name) for name in (reader.fieldnames or [])]
            rows = list(reader)
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        return HitList(path=target, note=f"could not read the hit list: {exc}")

    key_name = key_column or _pick(fieldnames, _HIT_KEY_COLUMNS)
    if not key_name:
        return HitList(path=target,
                       note=(f"{os.path.basename(target)} has no column "
                             f"naming what was hit"))
    if score_column == "-":
        score_name = ""
    else:
        score_name = score_column or _pick(fieldnames, _HIT_SCORE_COLUMNS)

    entries: List[Tuple[str, Optional[float]]] = []
    seen: set = set()
    for row in rows:
        key = str(row.get(key_name, "") or "").strip()
        if not key or key in seen:
            # A duplicated key would be ranked twice and then reported as
            # having "moved" against itself.
            continue
        seen.add(key)
        entries.append((key, _as_float(row.get(score_name)) if score_name
                        else None))

    if score_name:
        entries.sort(key=lambda item: (-abs(item[1]) if item[1] is not None
                                       else float("inf"), item[0]))
    if limit is not None:
        entries = entries[:limit]
    hits = tuple(Hit(key, index + 1, score)
                 for index, (key, score) in enumerate(entries))
    return HitList(path=target, available=True, hits=hits,
                   key_column=key_name, score_column=score_name)


def _pick(fieldnames: Sequence[str], candidates: Sequence[str]) -> str:
    """First candidate present in ``fieldnames``, matched case-insensitively."""
    lowered = {str(name).strip().lower(): str(name) for name in fieldnames}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return ""


def _as_float(value: Any) -> Optional[float]:
    """Parse a CSV cell as a float, or ``None``."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def diff_hits(a: HitList, b: HitList) -> HitDiff:
    """Compare two ranked hit lists: membership, then rank churn.

    :param a: the baseline list.
    :param b: the comparison list.
    :returns: a :class:`HitDiff`.
    """
    a_by_key, b_by_key = a.by_key(), b.by_key()
    changes: List[HitChange] = []

    for key in sorted(set(b_by_key) - set(a_by_key),
                      key=lambda k: b_by_key[k].rank):
        hit = b_by_key[key]
        changes.append(HitChange(key, "appeared", None, hit.rank,
                                 None, hit.score))
    for key in sorted(set(a_by_key) - set(b_by_key),
                      key=lambda k: a_by_key[k].rank):
        hit = a_by_key[key]
        changes.append(HitChange(key, "vanished", hit.rank, None,
                                 hit.score, None))

    shared: List[HitChange] = []
    for key in set(a_by_key) & set(b_by_key):
        left, right = a_by_key[key], b_by_key[key]
        status = "held" if left.rank == right.rank else "moved"
        shared.append(HitChange(key, status, left.rank, right.rank,
                                left.score, right.score))
    shared.sort(key=lambda c: (-abs(c.rank_delta or 0), c.b_rank or 0, c.key))
    changes.extend(shared)
    return HitDiff(changes=tuple(changes), a=a, b=b)


# ---------------------------------------------------------------------------
# The whole comparison
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunComparison:
    """Two runs and the three ways they differ.

    :param a: the baseline run.
    :param b: the run being compared to it.
    :param comparability: the verdict and its reasons.
    :param settings: the settings diff, or ``None`` when the runs are not
        comparable and the caller did not force it.
    :param counts: the count diff, same rule.
    :param hits: the hit-list diff, same rule.
    :param forced: the caller asked for the diffs despite a blocker.
    """

    a: RunRef
    b: RunRef
    comparability: Comparability
    settings: Optional[SettingsDiff] = None
    counts: Optional[CountDiff] = None
    hits: Optional[HitDiff] = None
    forced: bool = False

    @property
    def comparable(self) -> bool:
        """Whether the three diffs were produced."""
        return self.settings is not None

    def headline(self) -> str:
        """The one line to put above the tables."""
        if not self.comparable:
            return self.comparability.summary()
        lead = []
        if self.comparability.warnings:
            lead.append(" ".join(f.message
                                 for f in self.comparability.warnings))
        # `comparable` IS `settings is not None`, so re-testing it here
        # would be a branch that cannot go the other way.
        lead.append(self.settings.summary())
        if self.counts is not None and self.counts.available:
            lead.append(self.counts.headline())
        if self.hits is not None and self.hits.available:
            lead.append(self.hits.headline())
        return " ".join(lead)


def compare_runs(a: RunRef, b: RunRef, *,
                 include_same_settings: bool = False,
                 force: bool = False) -> RunComparison:
    """Compare two runs: settings, counts and hit list.

    Refuses by default when :func:`comparability` raises a blocker. That
    is the point of the check — two runs of different plates *can* be
    subtracted from one another, and the resulting table looks exactly
    like a regression report, which is why it must not be drawn without
    the user having said they know what they are doing.

    :param a: the baseline run, from :func:`runs_in`.
    :param b: the run being compared to it.
    :param include_same_settings: carry the settings both runs agree on,
        for the "show everything" view.
    :param force: produce the diffs even when the runs are not comparable.
    :returns: a :class:`RunComparison`. When it refused,
        :attr:`RunComparison.comparable` is False and the diffs are
        ``None``; ``comparability.blockers`` says why.
    """
    # Counted first, and deliberately: the plate identity that decides
    # comparability is in the database, not in the settings — ``src`` is
    # cosmetic as far as the settings hash goes and the registry does not
    # keep it. Counting is a handful of read-only COUNT queries, so
    # paying for it before the verdict costs nothing and is the only way
    # the "different plates" blocker fires on a registry-loaded run.
    a_counts = count_database(_database_of(a))
    b_counts = count_database(_database_of(b))
    verdict = comparability(a, b,
                            a_plates=a.plates or a_counts.plates,
                            b_plates=b.plates or b_counts.plates)
    if not verdict.comparable and not force:
        return RunComparison(a=a, b=b, comparability=verdict)

    settings = diff_settings_grouped(a.settings, b.settings,
                                     include_same=include_same_settings)
    hits = diff_hits(read_hits(_hitlist_of(a)), read_hits(_hitlist_of(b)))
    return RunComparison(a=a, b=b, comparability=verdict, settings=settings,
                         counts=diff_counts(a_counts, b_counts), hits=hits,
                         forced=bool(force))


def _database_of(run: RunRef) -> str:
    """The measurements database a run wrote, or ``""``.

    ``measurements-db`` first and ``object-counts`` second: they are the
    same file, but the first means Measure ran and the second only means
    Mask did, and counting a Mask-only run's cells out of a stale
    measurement table would be a fabricated comparison.
    """
    from . import ports
    for kind in (ports.MEASUREMENTS_DB, ports.OBJECT_COUNTS):
        artifact = run.artifact_of(kind)
        if artifact is not None:
            return str(getattr(artifact, "path", "") or "")
    return ""


def _hitlist_of(run: RunRef) -> str:
    """The regression results a run wrote, or ``""``.

    The registered artifact may be the results folder rather than one
    CSV, in which case the significant-hits file is preferred over the
    full coefficient table: a hit-list diff is about what was *called*,
    and every feature the model touched is not that.
    """
    from . import ports
    artifact = run.artifact_of(ports.REGRESSION_RESULTS)
    if artifact is None:
        return ""
    path = str(getattr(artifact, "path", "") or "")
    if not path or os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        return path
    for name in ("results_significant.csv", "results_gene.csv",
                 "results_grna.csv", "results.csv"):
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate):
            return candidate
    found = sorted(name for name in os.listdir(path)
                   if name.startswith("results") and name.endswith(".csv"))
    return os.path.join(path, found[0]) if found else path
