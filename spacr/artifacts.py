"""The artifact registry: what produced every file in a spaCR project.

A spaCR project is a folder of derived data.  ``merged/`` came from raw
images, ``measurements.db`` came from ``merged/``, the model came from the
crops the database indexes, and the hit list came from the model.  Nothing on
disk records any of that, so "is this result still current?" has never had an
answer — and re-running Mask with a different diameter leaves every
downstream number quietly wrong.

This module records it.  Every output registers with:

* the **producing module** (``"mask"``, ``"measure"``, … — the keys
  :mod:`spacr.ports` and :mod:`spacr.validate` use);
* a **settings hash** over the settings that could change the numbers
  (:data:`spacr.resume.COSMETIC_SETTINGS` decides which cannot);
* the **spaCR version** that produced it;
* the **input artifact ids** it was derived from, making the project a DAG;
* a **timestamp**, a **path**, and a **content fingerprint**.

Storage is SQLite — ``artifacts.db`` in the project root, one row per
artifact plus one edge row per input — because two Measure workers, a GUI and
a batch runner can all be touching a project at once, and a pickle cannot
survive that.  Writes go through :func:`spacr.database_concurrency.transaction`
with an explicit lock budget, which is also why nothing here does its own
retry arithmetic.

Public API
----------
``Registry``
    The registry for one project: :meth:`~Registry.register`,
    :meth:`~Registry.get`, :meth:`~Registry.by_kind`,
    :meth:`~Registry.by_project`, :meth:`~Registry.latest`,
    :meth:`~Registry.downstream_of`, :meth:`~Registry.upstream_of`,
    :meth:`~Registry.is_stale`, :meth:`~Registry.forget`.
``open_registry``, and module-level ``register`` / ``by_kind`` /
``by_project`` / ``latest`` / ``downstream_of`` / ``is_stale``
    The same, resolved from a project path.
``register_run_outputs``
    The one call a finished run makes; walks
    :func:`spacr.ports.declared_outputs` and registers what is there.
``Artifact``, ``Staleness``, ``Fingerprint``
    The records that come back.
``settings_hash``, ``material_settings``, ``content_fingerprint``
    The provenance primitives, usable on their own.
"""
from __future__ import annotations

import contextlib
import json
import hashlib
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (Any, Dict, Iterable, Iterator, List, Mapping, Optional,
                    Sequence, Tuple, Union)

from . import ports
from .checkpoint import fingerprint as _json_fingerprint, json_safe
from .database_concurrency import (NETWORK_FILESYSTEMS, DatabaseConfigurationError,
                                   connect, filesystem_type, transaction)
from .resume import COSMETIC_SETTINGS, ENV_SETTINGS
from .version import get_version

__all__ = [
    "ARTIFACTS_DB_ENV",
    "ARTIFACTS_DB_NAME",
    "Artifact",
    "Fingerprint",
    "FULL_HASH_LIMIT_BYTES",
    "Registry",
    "SCHEMA_VERSION",
    "STATUS_COMPLETE",
    "STATUS_FAILED",
    "STATUS_PARTIAL",
    "Staleness",
    "by_kind",
    "by_project",
    "content_fingerprint",
    "downstream_of",
    "is_stale",
    "latest",
    "material_settings",
    "open_registry",
    "register",
    "register_run_outputs",
    "registry_path",
    "settings_hash",
]


#: Bumped when the on-disk table layout changes incompatibly.
SCHEMA_VERSION = 1

#: The registry file, in the project root.
ARTIFACTS_DB_NAME = "artifacts.db"

#: Environment override pointing every project at one shared registry — for a
#: campaign spanning many plates, or for a test that wants a scratch file.
ARTIFACTS_DB_ENV = "SPACR_ARTIFACTS_DB"

#: A file larger than this is fingerprinted from its size plus its first and
#: last megabyte rather than end to end. A finished measurements.db can be
#: tens of gigabytes, and a run must not pay minutes of I/O to record that it
#: happened.
FULL_HASH_LIMIT_BYTES = 256 * 1024 * 1024

#: The run wrote everything it declared.
STATUS_COMPLETE = "complete"
#: Some fields failed; the artifact exists but is not the whole run.
STATUS_PARTIAL = "partial"
#: The run failed; the artifact is whatever was on disk when it did.
STATUS_FAILED = "failed"

_ID_LENGTH = 16

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id        TEXT PRIMARY KEY,
        project            TEXT NOT NULL,
        kind               TEXT NOT NULL,
        role               TEXT NOT NULL DEFAULT '',
        path               TEXT NOT NULL,
        module             TEXT NOT NULL,
        run_id             TEXT NOT NULL DEFAULT '',
        settings_hash      TEXT NOT NULL DEFAULT '',
        spacr_version      TEXT NOT NULL DEFAULT '',
        created_ns         INTEGER NOT NULL,
        created_utc        TEXT NOT NULL,
        fingerprint        TEXT NOT NULL DEFAULT '',
        fingerprint_method TEXT NOT NULL DEFAULT '',
        size_bytes         INTEGER NOT NULL DEFAULT 0,
        n_files            INTEGER NOT NULL DEFAULT 0,
        status             TEXT NOT NULL DEFAULT 'complete',
        settings_json      TEXT NOT NULL DEFAULT '{}',
        extra_json         TEXT NOT NULL DEFAULT '{}',
        schema_version     INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS artifact_inputs (
        artifact_id TEXT NOT NULL
            REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
        input_id    TEXT NOT NULL,
        position    INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (artifact_id, input_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_artifacts_kind "
    "ON artifacts(project, kind, created_ns)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_path "
    "ON artifacts(path, created_ns)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_module "
    "ON artifacts(module, created_ns)",
    "CREATE INDEX IF NOT EXISTS idx_artifact_inputs_input "
    "ON artifact_inputs(input_id)",
)

_COLUMNS = (
    "artifact_id", "project", "kind", "role", "path", "module", "run_id",
    "settings_hash", "spacr_version", "created_ns", "created_utc",
    "fingerprint", "fingerprint_method", "size_bytes", "n_files", "status",
    "settings_json", "extra_json", "schema_version",
)

# Staleness cause codes.
CAUSE_UNKNOWN = "unknown-artifact"
CAUSE_UPSTREAM_MISSING = "upstream-missing"
CAUSE_UPSTREAM_NEWER = "upstream-newer"
CAUSE_UPSTREAM_SUPERSEDED = "upstream-superseded"
CAUSE_UPSTREAM_STALE = "upstream-stale"
CAUSE_SETTINGS_CHANGED = "settings-changed"
CAUSE_CYCLE = "cycle"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fingerprint:
    """A content fingerprint plus how it was arrived at.

    :param digest: hexadecimal SHA-256, or ``""`` when nothing was there.
    :param method: ``"sha256"`` (a file read end to end), ``"sampled"`` (a
        large file: size plus its first and last megabyte), ``"tree"`` (a
        folder: every file's relative path, size and mtime) or ``"missing"``.
    :param size_bytes: total bytes covered.
    :param n_files: number of files covered; 1 for a single file.
    """

    digest: str
    method: str
    size_bytes: int = 0
    n_files: int = 0

    def __bool__(self) -> bool:
        """True when something was actually fingerprinted."""
        return bool(self.digest)


@dataclass(frozen=True)
class Artifact:
    """One registered output and everything known about where it came from.

    :param artifact_id: 16 hexadecimal characters, derived from the identity
        below. Registering byte-identical content from the same module,
        project, role and settings yields the same id, so a repeated
        registration updates one row rather than growing the table.
    :param project: absolute project root.
    :param kind: a :mod:`spacr.ports` kind, e.g. ``"measurements-db"``.
    :param role: the producing module's port role, e.g. ``"merged"``.
    :param path: absolute path of the file or folder.
    :param module: producing module key, e.g. ``"mask"``.
    :param run_id: the run this came out of, when the caller knows it.
    :param settings_hash: digest of the material settings; see
        :func:`settings_hash`.
    :param spacr_version: the version that produced it.
    :param created_ns: registration time, ``time.time_ns()``.
    :param created_utc: the same instant, ISO-8601.
    :param fingerprint: content digest; see :func:`content_fingerprint`.
    :param fingerprint_method: how that digest was computed.
    :param size_bytes: bytes on disk at registration.
    :param n_files: files covered.
    :param status: :data:`STATUS_COMPLETE`, :data:`STATUS_PARTIAL` or
        :data:`STATUS_FAILED`.
    :param settings: the material settings, kept so a later run can be
        diffed against this one rather than only compared by hash.
    :param extra: anything else the producer wanted recorded.
    :param inputs: artifact ids this was derived from.
    :param schema_version: the registry layout that wrote the row.
    """

    artifact_id: str
    project: str
    kind: str
    role: str
    path: str
    module: str
    run_id: str
    settings_hash: str
    spacr_version: str
    created_ns: int
    created_utc: str
    fingerprint: str
    fingerprint_method: str
    size_bytes: int
    n_files: int
    status: str
    settings: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    inputs: Tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    @property
    def exists(self) -> bool:
        """True when the artifact is still on disk where it was registered."""
        return os.path.exists(self.path)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable copy of the record."""
        return {
            "artifact_id": self.artifact_id, "project": self.project,
            "kind": self.kind, "role": self.role, "path": self.path,
            "module": self.module, "run_id": self.run_id,
            "settings_hash": self.settings_hash,
            "spacr_version": self.spacr_version,
            "created_ns": self.created_ns, "created_utc": self.created_utc,
            "fingerprint": self.fingerprint,
            "fingerprint_method": self.fingerprint_method,
            "size_bytes": self.size_bytes, "n_files": self.n_files,
            "status": self.status, "settings": json_safe(self.settings),
            "extra": json_safe(self.extra), "inputs": list(self.inputs),
            "schema_version": self.schema_version,
        }

    def __str__(self) -> str:
        """One line: id, kind, module and path."""
        return (f"{self.artifact_id} {self.kind} from {self.module} "
                f"at {self.path}")


@dataclass(frozen=True)
class Staleness:
    """Whether an artifact still matches what it was made from.

    Stale means an upstream artifact or a material setting changed *after*
    this was produced. A file that has simply been deleted is reported by
    :attr:`missing` instead — that is an availability problem, not a
    provenance one, and conflating the two hides both.

    :param artifact_id: the artifact asked about.
    :param stale: the answer. ``bool(staleness)`` is the same value.
    :param reasons: human-readable sentences, one per finding.
    :param causes: machine codes for the same findings, e.g.
        ``"upstream-newer"``.
    :param missing: the artifact's own path is no longer on disk.
    """

    artifact_id: str
    stale: bool
    reasons: Tuple[str, ...] = ()
    causes: Tuple[str, ...] = ()
    missing: bool = False

    def __bool__(self) -> bool:
        """True when the artifact is stale."""
        return self.stale

    def __str__(self) -> str:
        """A one-line verdict with its reasons."""
        verdict = "stale" if self.stale else "current"
        if not self.reasons:
            return f"{self.artifact_id}: {verdict}"
        return f"{self.artifact_id}: {verdict} — {'; '.join(self.reasons)}"


# ---------------------------------------------------------------------------
# Provenance primitives
# ---------------------------------------------------------------------------

def material_settings(settings: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return only the settings that can change the numbers.

    Verbosity, worker counts, plot cosmetics and the environment snapshot are
    dropped, using the same deny-list
    (:data:`spacr.resume.COSMETIC_SETTINGS`, and
    :data:`spacr.resume.ENV_SETTINGS`) that decides whether a resume is
    allowed. Any key nobody has classified counts as material, so a new knob
    is conservatively assumed to matter.

    :param settings: a settings dict, or None.
    :returns: a new dict with the inconsequential keys removed.
    """
    if not settings:
        return {}
    return {
        str(key): value for key, value in settings.items()
        if str(key) not in COSMETIC_SETTINGS and str(key) not in ENV_SETTINGS
    }


def settings_hash(settings: Optional[Mapping[str, Any]]) -> str:
    """Return a digest over the material settings of a run.

    Two runs with the same digest cannot differ in anything that changes
    their output, so a downstream artifact whose recorded digest still
    matches the current settings is not stale on their account.

    :param settings: a settings dict, or None.
    :returns: a lowercase SHA-256 hex digest (of ``{}`` when there are none).
    """
    return _json_fingerprint(material_settings(settings))


def content_fingerprint(
        path: Union[str, os.PathLike],
        *, full_hash_limit: int = FULL_HASH_LIMIT_BYTES) -> Fingerprint:
    """Fingerprint whatever is at ``path``: a file, a folder, or nothing.

    A regular file is hashed end to end while it is small enough to be worth
    it, and above ``full_hash_limit`` from its size plus its first and last
    megabyte — enough to notice a rewritten database without spending minutes
    of I/O at the end of every run. A folder is fingerprinted from its file
    inventory: every file's relative path, size and modification time, in
    sorted order.

    :param path: file or folder.
    :param full_hash_limit: byte size above which a file is sampled instead
        of read end to end.
    :returns: a :class:`Fingerprint`; ``method="missing"`` when nothing is
        there.
    """
    target = os.fspath(path)
    if os.path.isfile(target):
        size = os.path.getsize(target)
        digest = hashlib.sha256()
        if size <= full_hash_limit:
            with open(target, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            return Fingerprint(digest.hexdigest(), "sha256", size, 1)
        window = 1 << 20
        digest.update(str(size).encode("utf-8"))
        with open(target, "rb") as handle:
            digest.update(handle.read(window))
            handle.seek(max(0, size - window))
            digest.update(handle.read(window))
        return Fingerprint(digest.hexdigest(), "sampled", size, 1)
    if os.path.isdir(target):
        records: List[Tuple[str, int, int]] = []
        total = 0
        for root, dirnames, filenames in os.walk(target, followlinks=False):
            dirnames[:] = sorted(
                name for name in dirnames
                if not os.path.islink(os.path.join(root, name)))
            for name in sorted(filenames):
                candidate = os.path.join(root, name)
                if os.path.islink(candidate):
                    continue
                stat = os.stat(candidate)
                records.append((os.path.relpath(candidate, target),
                                stat.st_size, stat.st_mtime_ns))
                total += stat.st_size
        payload = json.dumps(records, separators=(",", ":")).encode("utf-8")
        return Fingerprint(hashlib.sha256(payload).hexdigest(), "tree",
                           total, len(records))
    return Fingerprint("", "missing", 0, 0)


def registry_path(project: Union[str, os.PathLike, None] = None) -> str:
    """Return the registry file for ``project``.

    :param project: the project root. Ignored when :data:`ARTIFACTS_DB_ENV`
        is set, which points every project at one shared registry.
    :returns: an absolute path. The file need not exist yet.
    :raises ValueError: when no project is given and no override is set.
    """
    override = os.environ.get(ARTIFACTS_DB_ENV, "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    if not project:
        raise ValueError(
            f"no project root given and {ARTIFACTS_DB_ENV} is not set, so "
            f"there is nowhere to keep the artifact registry")
    root = os.path.abspath(os.path.expanduser(os.fspath(project)))
    return os.path.join(root, ARTIFACTS_DB_NAME)


def _artifact_id(project: str, module: str, kind: str, role: str, path: str,
                 settings_digest: str, content_digest: str) -> str:
    """Return the deterministic id for one artifact identity."""
    payload = "\x1f".join((project, module, kind, role, path,
                           settings_digest, content_digest))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_ID_LENGTH]


def _identifier(artifact: Union[str, Artifact]) -> str:
    """Accept an :class:`Artifact` or a bare id, and return the id."""
    return artifact.artifact_id if isinstance(artifact, Artifact) else str(artifact)


def _loads(text: str) -> Dict[str, Any]:
    """Parse a JSON object column, tolerating an empty cell."""
    return json.loads(text) if text else {}


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

class Registry:
    """The artifact registry for one project.

    One SQLite file, opened per operation and closed again: Measure workers,
    the GUI and a batch runner all register into the same project, and a
    connection held open across a whole run is a lock held across a whole
    run. Writes go through
    :func:`spacr.database_concurrency.transaction` with an explicit lock
    budget, so a second registration arriving mid-write waits rather than
    failing.

    :param path: the registry file. Defaults to
        :func:`registry_path` for ``project``.
    :param project: the project root recorded on artifacts that do not name
        one of their own.
    :param timeout: seconds a write may wait on the lock, in total.
    :param create: create the file and its tables when missing. Pass False
        for a read-only consumer that must not conjure an empty registry.
    :raises FileNotFoundError: when ``create`` is False and there is no
        registry.
    :raises ValueError: when neither ``path`` nor ``project`` is given.
    """

    def __init__(self,
                 path: Union[str, os.PathLike, None] = None,
                 *,
                 project: Union[str, os.PathLike, None] = None,
                 timeout: float = 30.0,
                 create: bool = True) -> None:
        self.project = (os.path.abspath(os.path.expanduser(os.fspath(project)))
                        if project else "")
        self.path = (os.path.abspath(os.path.expanduser(os.fspath(path)))
                     if path else registry_path(self.project))
        self.timeout = float(timeout)
        if not os.path.isfile(self.path):
            if not create:
                raise FileNotFoundError(
                    f"no artifact registry at {self.path}")
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._ensure_schema()

    # -- plumbing ---------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open one connection, in WAL where the filesystem supports it."""
        fs_type = filesystem_type(self.path)
        network = bool(fs_type and fs_type.casefold() in NETWORK_FILESYSTEMS)
        if not network:
            try:
                return connect(self.path, timeout=self.timeout,
                               journal_mode="WAL")
            except DatabaseConfigurationError:
                # SQLite kept the old journal mode -- an older file, or a
                # filesystem that will not do shared memory. DELETE mode is
                # slower under contention but always correct.
                pass
        return connect(self.path, timeout=self.timeout)

    @contextlib.contextmanager
    def _open(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection owned by this call and close it afterwards."""
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _write(self, connection: sqlite3.Connection):
        """Return a write transaction with an explicit whole-operation budget.

        The budget is handed to
        :func:`spacr.database_concurrency.transaction` rather than divided
        here: splitting a lock budget across attempts is subtle enough to
        have been got wrong once already, and there is no reason for a second
        implementation of it to exist.
        """
        return transaction(connection, mode="IMMEDIATE", attempts=6,
                           busy_timeout=self.timeout)

    def _ensure_schema(self) -> None:
        """Create the tables and indexes when they are not already there."""
        with self._open() as connection:
            with self._write(connection):
                for statement in _SCHEMA:
                    connection.execute(statement)

    # -- writing ----------------------------------------------------------

    def register(self,
                 *,
                 module: str,
                 kind: str,
                 path: Union[str, os.PathLike],
                 role: str = "",
                 project: Union[str, os.PathLike, None] = None,
                 settings: Optional[Mapping[str, Any]] = None,
                 settings_digest: str = "",
                 inputs: Sequence[Union[str, "Artifact"]] = (),
                 run_id: str = "",
                 status: str = STATUS_COMPLETE,
                 extra: Optional[Mapping[str, Any]] = None,
                 fingerprint: Optional[Fingerprint] = None,
                 ) -> Artifact:
        """Record one output and what it was made from.

        The id is derived from the identity — project, module, kind, role,
        path, settings hash, content fingerprint — so registering the same
        content twice updates the existing row (refreshing its timestamp, run
        id and inputs) instead of adding a duplicate. Registering *different*
        content, or the same content under different settings, creates a new
        row, which is exactly what makes the older downstream artifacts stale.

        :param module: producing module key, e.g. ``"mask"``.
        :param kind: a :mod:`spacr.ports` kind.
        :param path: the file or folder produced.
        :param role: the module's port role for this output.
        :param project: project root; defaults to the registry's own.
        :param settings: the run's settings. Only the material ones are
            hashed and stored.
        :param settings_digest: use this digest instead of hashing
            ``settings`` — for a caller that already computed one.
        :param inputs: artifact ids (or :class:`Artifact` objects) this was
            derived from.
        :param run_id: the run this came out of.
        :param status: :data:`STATUS_COMPLETE`, :data:`STATUS_PARTIAL` or
            :data:`STATUS_FAILED`.
        :param extra: any additional JSON-safe provenance.
        :param fingerprint: a precomputed :class:`Fingerprint`; omit to
            compute one from ``path``.
        :returns: the stored :class:`Artifact`.
        :raises ValueError: when ``module``, ``kind`` or ``path`` is empty.
        """
        if not str(module).strip():
            raise ValueError("an artifact needs the module that produced it")
        if not str(kind).strip():
            raise ValueError("an artifact needs a kind")
        if not str(path).strip():
            raise ValueError("an artifact needs a path")

        absolute = os.path.abspath(os.path.expanduser(os.fspath(path)))
        root = (os.path.abspath(os.path.expanduser(os.fspath(project)))
                if project else self.project)
        digest = settings_digest or settings_hash(settings)
        content = fingerprint if fingerprint is not None else content_fingerprint(absolute)
        now_ns = time.time_ns()
        artifact = Artifact(
            artifact_id=_artifact_id(root, str(module), str(kind), str(role),
                                     absolute, digest, content.digest),
            project=root,
            kind=str(kind),
            role=str(role),
            path=absolute,
            module=str(module),
            run_id=str(run_id),
            settings_hash=digest,
            spacr_version=get_version(),
            created_ns=now_ns,
            created_utc=datetime.fromtimestamp(
                now_ns / 1e9, tz=timezone.utc).isoformat(),
            fingerprint=content.digest,
            fingerprint_method=content.method,
            size_bytes=content.size_bytes,
            n_files=content.n_files,
            status=str(status),
            settings=material_settings(settings),
            extra=dict(extra or {}),
            inputs=tuple(dict.fromkeys(_identifier(i) for i in inputs)),
            schema_version=SCHEMA_VERSION,
        )
        with self._open() as connection:
            with self._write(connection):
                connection.execute(
                    f"INSERT INTO artifacts ({', '.join(_COLUMNS)}) "
                    f"VALUES ({', '.join('?' * len(_COLUMNS))}) "
                    f"ON CONFLICT(artifact_id) DO UPDATE SET "
                    f"created_ns=excluded.created_ns, "
                    f"created_utc=excluded.created_utc, "
                    f"run_id=excluded.run_id, status=excluded.status, "
                    f"spacr_version=excluded.spacr_version, "
                    f"extra_json=excluded.extra_json",
                    (
                        artifact.artifact_id, artifact.project, artifact.kind,
                        artifact.role, artifact.path, artifact.module,
                        artifact.run_id, artifact.settings_hash,
                        artifact.spacr_version, artifact.created_ns,
                        artifact.created_utc, artifact.fingerprint,
                        artifact.fingerprint_method, artifact.size_bytes,
                        artifact.n_files, artifact.status,
                        json.dumps(json_safe(artifact.settings),
                                   sort_keys=True, separators=(",", ":")),
                        json.dumps(json_safe(artifact.extra),
                                   sort_keys=True, separators=(",", ":")),
                        artifact.schema_version,
                    ))
                connection.execute(
                    "DELETE FROM artifact_inputs WHERE artifact_id = ?",
                    (artifact.artifact_id,))
                connection.executemany(
                    "INSERT INTO artifact_inputs (artifact_id, input_id, "
                    "position) VALUES (?, ?, ?)",
                    [(artifact.artifact_id, input_id, position)
                     for position, input_id in enumerate(artifact.inputs)])
        return artifact

    def forget(self, artifact: Union[str, Artifact]) -> int:
        """Delete one artifact row and the edges pointing out of it.

        Edges pointing *at* it are left alone on purpose: a downstream
        artifact that names a vanished input must keep saying so, which is
        what makes it report as stale rather than as current.

        :param artifact: id or :class:`Artifact`.
        :returns: number of artifact rows deleted — 0 or 1.
        """
        artifact_id = _identifier(artifact)
        with self._open() as connection:
            with self._write(connection):
                cursor = connection.execute(
                    "DELETE FROM artifacts WHERE artifact_id = ?",
                    (artifact_id,))
                return int(cursor.rowcount)

    # -- reading ----------------------------------------------------------

    def _row_to_artifact(self, connection: sqlite3.Connection,
                         row: Sequence[Any]) -> Artifact:
        """Build an :class:`Artifact` from one row plus its input edges."""
        values = dict(zip(_COLUMNS, row))
        inputs = tuple(
            str(item[0]) for item in connection.execute(
                "SELECT input_id FROM artifact_inputs WHERE artifact_id = ? "
                "ORDER BY position", (values["artifact_id"],)))
        return Artifact(
            artifact_id=str(values["artifact_id"]),
            project=str(values["project"]), kind=str(values["kind"]),
            role=str(values["role"]), path=str(values["path"]),
            module=str(values["module"]), run_id=str(values["run_id"]),
            settings_hash=str(values["settings_hash"]),
            spacr_version=str(values["spacr_version"]),
            created_ns=int(values["created_ns"]),
            created_utc=str(values["created_utc"]),
            fingerprint=str(values["fingerprint"]),
            fingerprint_method=str(values["fingerprint_method"]),
            size_bytes=int(values["size_bytes"]),
            n_files=int(values["n_files"]), status=str(values["status"]),
            settings=_loads(str(values["settings_json"])),
            extra=_loads(str(values["extra_json"])),
            inputs=inputs,
            schema_version=int(values["schema_version"]),
        )

    def _select(self, connection: sqlite3.Connection, where: str,
                params: Sequence[Any], limit: Optional[int]) -> List[Artifact]:
        """Run one newest-first SELECT and inflate its rows."""
        sql = f"SELECT {', '.join(_COLUMNS)} FROM artifacts"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY created_ns DESC, artifact_id"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = connection.execute(sql, tuple(params)).fetchall()
        return [self._row_to_artifact(connection, row) for row in rows]

    @staticmethod
    def _filters(project: Union[str, None], kind: Optional[str],
                 module: Optional[str], role: Optional[str],
                 path: Optional[str]) -> Tuple[str, List[Any]]:
        """Build the WHERE clause shared by every query."""
        clauses: List[str] = []
        params: List[Any] = []
        for column, value in (("project", project), ("kind", kind),
                              ("module", module), ("role", role)):
            if value:
                clauses.append(f"{column} = ?")
                params.append(str(value))
        if path:
            clauses.append("path = ?")
            params.append(os.path.abspath(os.path.expanduser(str(path))))
        return " AND ".join(clauses), params

    def all(self, *, limit: Optional[int] = None) -> List[Artifact]:
        """Return every artifact in this registry, newest first.

        :param limit: cap the number of rows returned.
        """
        with self._open() as connection:
            return self._select(connection, "", (), limit)

    def get(self, artifact: Union[str, Artifact]) -> Optional[Artifact]:
        """Return one artifact by id, or None when it is not registered.

        :param artifact: id or :class:`Artifact`.
        """
        with self._open() as connection:
            return self._get(connection, _identifier(artifact))

    def _get(self, connection: sqlite3.Connection,
             artifact_id: str) -> Optional[Artifact]:
        """Fetch one artifact on an open connection."""
        found = self._select(connection, "artifact_id = ?", (artifact_id,), 1)
        return found[0] if found else None

    def by_kind(self, kind: str, *,
                project: Union[str, None] = None,
                module: Optional[str] = None,
                limit: Optional[int] = None) -> List[Artifact]:
        """Return every artifact of ``kind``, newest first.

        :param kind: a :mod:`spacr.ports` kind, e.g. ``"merged-arrays"``.
        :param project: restrict to one project root.
        :param module: restrict to one producing module.
        :param limit: cap the number of rows returned.
        """
        where, params = self._filters(project, kind, module, None, None)
        with self._open() as connection:
            return self._select(connection, where, params, limit)

    def by_project(self, project: Union[str, None] = None, *,
                   kind: Optional[str] = None,
                   module: Optional[str] = None,
                   limit: Optional[int] = None) -> List[Artifact]:
        """Return every artifact belonging to one project, newest first.

        :param project: the project root; defaults to this registry's own.
            Pass ``""`` explicitly for "every project in this file", which is
            what a shared registry (see :data:`ARTIFACTS_DB_ENV`) holds.
        :param kind: restrict to one kind.
        :param module: restrict to one producing module.
        :param limit: cap the number of rows returned.
        """
        root = self.project if project is None else project
        if root:
            root = os.path.abspath(os.path.expanduser(str(root)))
        where, params = self._filters(root, kind, module, None, None)
        with self._open() as connection:
            return self._select(connection, where, params, limit)

    def latest(self, kind: str, *,
               project: Union[str, None] = None,
               module: Optional[str] = None,
               role: Optional[str] = None,
               path: Optional[str] = None) -> Optional[Artifact]:
        """Return the most recent artifact matching the filters, or None.

        The call auto-chaining makes: "what is the current
        ``measurements-db`` for this project?".

        :param kind: a :mod:`spacr.ports` kind.
        :param project: restrict to one project root.
        :param module: restrict to one producing module.
        :param role: restrict to one port role.
        :param path: restrict to one exact path.
        """
        where, params = self._filters(project, kind, module, role, path)
        with self._open() as connection:
            found = self._select(connection, where, params, 1)
        return found[0] if found else None

    def upstream_of(self, artifact: Union[str, Artifact], *,
                    transitive: bool = False) -> List[Artifact]:
        """Return the artifacts ``artifact`` was derived from.

        :param artifact: id or :class:`Artifact`.
        :param transitive: follow inputs of inputs, to the roots of the DAG.
        :returns: registered ancestors, newest first. Input ids that are no
            longer registered are simply absent — :meth:`is_stale` is what
            reports them.
        """
        with self._open() as connection:
            found = self._walk(connection, _identifier(artifact), transitive,
                               self._input_ids)
        return sorted(found.values(), key=lambda a: (-a.created_ns,
                                                     a.artifact_id))

    def downstream_of(self, artifact: Union[str, Artifact], *,
                      transitive: bool = True) -> List[Artifact]:
        """Return the artifacts derived from ``artifact``.

        Transitive by default: the question a user asks — "what does this
        invalidate?" — is about everything downstream, not only the immediate
        children.

        :param artifact: id or :class:`Artifact`.
        :param transitive: follow the edges all the way down.
        :returns: registered descendants, newest first.
        """
        with self._open() as connection:
            found = self._walk(connection, _identifier(artifact), transitive,
                               self._consumer_ids)
        return sorted(found.values(), key=lambda a: (-a.created_ns,
                                                     a.artifact_id))

    @staticmethod
    def _input_ids(connection: sqlite3.Connection,
                   artifact_id: str) -> List[str]:
        """Ids one artifact declares as its inputs."""
        return [str(row[0]) for row in connection.execute(
            "SELECT input_id FROM artifact_inputs WHERE artifact_id = ? "
            "ORDER BY position", (artifact_id,))]

    @staticmethod
    def _consumer_ids(connection: sqlite3.Connection,
                      artifact_id: str) -> List[str]:
        """Ids of the artifacts that declare this one as an input."""
        return [str(row[0]) for row in connection.execute(
            "SELECT artifact_id FROM artifact_inputs WHERE input_id = ?",
            (artifact_id,))]

    def _walk(self, connection: sqlite3.Connection, start: str,
              transitive: bool, edges) -> Dict[str, Artifact]:
        """Breadth-first walk of the provenance DAG from ``start``."""
        found: Dict[str, Artifact] = {}
        seen = {start}
        frontier = list(edges(connection, start))
        while frontier:
            artifact_id = frontier.pop(0)
            if artifact_id in seen:
                continue
            seen.add(artifact_id)
            record = self._get(connection, artifact_id)
            if record is not None:
                found[artifact_id] = record
            if transitive:
                frontier.extend(edges(connection, artifact_id))
        return found

    # -- staleness --------------------------------------------------------

    def is_stale(self, artifact: Union[str, Artifact], *,
                 settings: Optional[Mapping[str, Any]] = None) -> Staleness:
        """Answer whether an upstream artifact or setting changed since this.

        An artifact is stale when any of these hold:

        * one of its recorded inputs is no longer in the registry;
        * an input was registered again *after* this artifact was;
        * a newer artifact of the same kind now sits at an input's path —
          the shape "Mask was re-run with a different diameter" takes;
        * an input is itself stale, transitively;
        * ``settings`` is supplied and its material hash differs from the one
          recorded here.

        :param artifact: id or :class:`Artifact`.
        :param settings: the settings a caller is about to use, compared
            against the ones that produced the artifact.
        :returns: a :class:`Staleness`; ``bool(result)`` is the answer and
            :attr:`Staleness.reasons` is what to show a user.
        """
        with self._open() as connection:
            return self._staleness(connection, _identifier(artifact),
                                   settings, set())

    def _staleness(self, connection: sqlite3.Connection, artifact_id: str,
                   settings: Optional[Mapping[str, Any]],
                   visiting: set) -> Staleness:
        """Recursive staleness, guarding against a self-referencing edge."""
        if artifact_id in visiting:
            return Staleness(
                artifact_id, False,
                (f"provenance cycle at {artifact_id}; not followed further",),
                (CAUSE_CYCLE,))
        record = self._get(connection, artifact_id)
        if record is None:
            return Staleness(
                artifact_id, True,
                (f"{artifact_id} is not in the registry",),
                (CAUSE_UNKNOWN,), missing=True)

        visiting = visiting | {artifact_id}
        reasons: List[str] = []
        causes: List[str] = []

        if settings is not None and settings_hash(settings) != record.settings_hash:
            reasons.append(
                f"the settings differ from the ones that produced "
                f"{record.kind} at {record.path}")
            causes.append(CAUSE_SETTINGS_CHANGED)

        for input_id in record.inputs:
            upstream = self._get(connection, input_id)
            if upstream is None:
                reasons.append(
                    f"input {input_id} is no longer in the registry")
                causes.append(CAUSE_UPSTREAM_MISSING)
                continue
            if upstream.created_ns > record.created_ns:
                reasons.append(
                    f"{upstream.kind} at {upstream.path} was produced again "
                    f"after this")
                causes.append(CAUSE_UPSTREAM_NEWER)
                continue
            newer = self._select(
                connection,
                "kind = ? AND path = ? AND created_ns > ?",
                (upstream.kind, upstream.path, record.created_ns), 1)
            if newer:
                reasons.append(
                    f"{newer[0].kind} at {newer[0].path} was re-produced by "
                    f"{newer[0].module} after this")
                causes.append(CAUSE_UPSTREAM_SUPERSEDED)
                continue
            inherited = self._staleness(connection, input_id, None, visiting)
            if inherited.stale:
                reasons.append(
                    f"input {upstream.kind} at {upstream.path} is itself "
                    f"stale ({'; '.join(inherited.reasons)})")
                causes.append(CAUSE_UPSTREAM_STALE)

        return Staleness(artifact_id, bool(reasons), tuple(reasons),
                         tuple(causes), missing=not record.exists)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def open_registry(project: Union[str, os.PathLike, None] = None, *,
                  path: Union[str, os.PathLike, None] = None,
                  create: bool = True) -> Registry:
    """Return the :class:`Registry` for one project.

    :param project: the project root.
    :param path: an explicit registry file, overriding ``project``'s default.
    :param create: create the file and tables when missing.
    """
    return Registry(path, project=project, create=create)


def _resolve(registry: Optional[Registry],
             project: Union[str, os.PathLike, None]) -> Registry:
    """Return the registry to use for a module-level call."""
    return registry if registry is not None else open_registry(project)


def register(*, registry: Optional[Registry] = None,
             project: Union[str, os.PathLike, None] = None,
             **kwargs: Any) -> Artifact:
    """Register one artifact; see :meth:`Registry.register`.

    :param registry: an open registry to use instead of opening one.
    :param project: the project root, used to find the registry and recorded
        on the artifact.
    :param kwargs: passed through to :meth:`Registry.register`.
    """
    target = _resolve(registry, project)
    kwargs.setdefault("project", project)
    return target.register(**kwargs)


def by_kind(kind: str, *, project: Union[str, os.PathLike, None] = None,
            registry: Optional[Registry] = None,
            **kwargs: Any) -> List[Artifact]:
    """Every artifact of ``kind``; see :meth:`Registry.by_kind`.

    :param kind: a :mod:`spacr.ports` kind.
    :param project: the project root.
    :param registry: an open registry to use instead of opening one.
    :param kwargs: passed through to :meth:`Registry.by_kind`.
    """
    return _resolve(registry, project).by_kind(kind, **kwargs)


def by_project(project: Union[str, os.PathLike, None] = None, *,
               registry: Optional[Registry] = None,
               **kwargs: Any) -> List[Artifact]:
    """Every artifact in a project; see :meth:`Registry.by_project`.

    :param project: the project root.
    :param registry: an open registry to use instead of opening one.
    :param kwargs: passed through to :meth:`Registry.by_project`.
    """
    return _resolve(registry, project).by_project(**kwargs)


def latest(kind: str, *, project: Union[str, os.PathLike, None] = None,
           registry: Optional[Registry] = None,
           **kwargs: Any) -> Optional[Artifact]:
    """The newest artifact of ``kind``; see :meth:`Registry.latest`.

    :param kind: a :mod:`spacr.ports` kind.
    :param project: the project root.
    :param registry: an open registry to use instead of opening one.
    :param kwargs: passed through to :meth:`Registry.latest`.
    """
    return _resolve(registry, project).latest(kind, **kwargs)


def downstream_of(artifact: Union[str, Artifact], *,
                  project: Union[str, os.PathLike, None] = None,
                  registry: Optional[Registry] = None,
                  **kwargs: Any) -> List[Artifact]:
    """What an artifact invalidates; see :meth:`Registry.downstream_of`.

    :param artifact: id or :class:`Artifact`.
    :param project: the project root.
    :param registry: an open registry to use instead of opening one.
    :param kwargs: passed through to :meth:`Registry.downstream_of`.
    """
    return _resolve(registry, project).downstream_of(artifact, **kwargs)


def is_stale(artifact: Union[str, Artifact], *,
             project: Union[str, os.PathLike, None] = None,
             registry: Optional[Registry] = None,
             **kwargs: Any) -> Staleness:
    """Whether an artifact is out of date; see :meth:`Registry.is_stale`.

    :param artifact: id or :class:`Artifact`.
    :param project: the project root.
    :param registry: an open registry to use instead of opening one.
    :param kwargs: passed through to :meth:`Registry.is_stale`.
    """
    return _resolve(registry, project).is_stale(artifact, **kwargs)


# ---------------------------------------------------------------------------
# The run-completion hook
# ---------------------------------------------------------------------------

def register_run_outputs(module: str,
                         settings: Optional[Mapping[str, Any]] = None,
                         *,
                         roots: Optional[Iterable[Any]] = None,
                         run_id: str = "",
                         status: str = STATUS_COMPLETE,
                         inputs: Optional[Sequence[Union[str, Artifact]]] = None,
                         registry: Optional[Registry] = None,
                         strict: bool = True) -> Tuple[Artifact, ...]:
    """Register everything a finished run declared it would write.

    The one call a pipeline entry point makes on completion. For each project
    root it walks :func:`spacr.ports.declared_outputs`, registers every
    produced port that is actually on disk, and links each one to the
    artifacts currently sitting at the module's *input* ports — which is what
    turns a folder of files into a provenance DAG.

    :param module: producing module key, e.g. ``"mask"``.
    :param settings: the settings the run used; hashed into every artifact.
    :param roots: project roots to record. Defaults to the one
        :func:`spacr.ports.project_root` derives from ``settings``. A
        multi-plate run passes its whole ``src`` list.
    :param run_id: the run this came out of.
    :param status: :data:`STATUS_COMPLETE`, :data:`STATUS_PARTIAL` or
        :data:`STATUS_FAILED` — a run that lost fields should say so.
    :param inputs: explicit input artifact ids, overriding the lookup.
    :param registry: an open registry to use for every root, instead of one
        per project.
    :param strict: raise on failure. Pipelines pass False: a registry that
        cannot be written is worth one printed line, never a lost run.
    :returns: the artifacts registered, in declaration order.
    """
    try:
        return _register_run_outputs(module, settings, roots, run_id, status,
                                     inputs, registry)
    except Exception as exc:                       # noqa: BLE001 - see strict
        if strict:
            raise
        print(f"spacr.artifacts: could not record {module} outputs: "
              f"{type(exc).__name__}: {exc}")
        return ()


def _register_run_outputs(module: str,
                          settings: Optional[Mapping[str, Any]],
                          roots: Optional[Iterable[Any]],
                          run_id: str,
                          status: str,
                          inputs: Optional[Sequence[Union[str, Artifact]]],
                          registry: Optional[Registry],
                          ) -> Tuple[Artifact, ...]:
    """The body of :func:`register_run_outputs`, without the guard."""
    spec = ports.module_ports(module)
    if roots is None:
        candidates = [ports.project_root(settings, spec.key)]
    else:
        candidates = [ports.project_root(root, spec.key) for root in roots]
    digest = settings_hash(settings)
    registered: List[Artifact] = []
    for root in candidates:
        if not root or not os.path.isdir(root):
            continue
        target = registry if registry is not None else open_registry(root)
        upstream = (list(inputs) if inputs is not None
                    else _current_inputs(target, spec, root))
        for resolved in ports.declared_outputs(spec.key, root=root):
            if not resolved.exists:
                continue
            registered.append(target.register(
                module=spec.key, kind=resolved.kind, role=resolved.role,
                path=resolved.location, project=root, settings=settings,
                settings_digest=digest, inputs=upstream, run_id=run_id,
                status=status))
    return tuple(registered)


def _current_inputs(registry: Registry, spec: "ports.ModulePorts",
                    root: str) -> List[str]:
    """Artifact ids currently sitting at ``spec``'s input ports under ``root``."""
    found: List[str] = []
    for resolved in ports.declared_inputs(spec.key, root=root):
        artifact = registry.latest(resolved.kind, path=resolved.location)
        if artifact is not None:
            found.append(artifact.artifact_id)
    return found
