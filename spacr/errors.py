"""Fail-loud error accounting for spaCR pipelines.

spaCR processes batches — 384 wells, thousands of fields, dozens of
image files. A single unreadable image must not abort the plate, so
almost every batch loop in the codebase wraps its body in
``try/except Exception`` and carries on. The historical problem was
not the surviving; it was that the survival left no trace: forty wells
would fail to segment, forty lines would scroll past in a log nobody
was watching, ``measurements.db`` would be written anyway, and every
downstream regression would silently run on 344 wells while reporting
as if it had 384.

This module supplies the missing half — **survive, but account for
it, and make the accounting impossible to miss**:

* :class:`RunLedger` records every per-item success and failure,
  logs each failure at ``ERROR`` with the item id and traceback, and
  prints one loud, grouped block at the end of the run.
* :meth:`RunLedger.stamp` writes that verdict *into the artifact* —
  a ``run_status`` table inside a SQLite database, or a sibling
  ``<name>.run_status.json`` next to any other output — so a later
  reader can tell that the result is partial.
* :func:`read_run_status` / :func:`run_is_complete` /
  :func:`assert_run_complete` let downstream code check before it
  trusts a file.
* :class:`ConfigurationError` marks the failures that must *not* be
  survived. A wrong ``src`` path or a missing metadata column is not
  a per-item problem — continuing past it only produces garbage.
  :meth:`RunLedger.item` deliberately re-raises it.

Typical adoption inside a batch loop::

    from .errors import RunLedger

    ledger = RunLedger('convert_to_yokogawa')
    for file in files:
        with ledger.item(file, stage='convert'):
            convert(file)
    ledger.finalize(artifact=csv_path)

and downstream, before trusting the output::

    from spacr.errors import run_is_complete
    if not run_is_complete(db_path):
        ...            # the numbers in here are computed on a subset

The module is deliberately **stdlib-only**. It is imported by
``io``/``core``/``measure``/``deep_spacr``/``plot`` at module scope, so
it must never drag in torch, cellpose, pandas or numpy.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time
import traceback as _traceback
import uuid
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

__all__ = [
    'SpacrError',
    'ConfigurationError',
    'DataIntegrityError',
    'PartialRunError',
    'RunStatusUnreadable',
    'Failure',
    'RunLedger',
    'read_run_status',
    'run_is_complete',
    'assert_run_complete',
    'strict_errors',
    'raise_if_strict',
    'RUN_STATUS_TABLE',
    'RUN_STATUS_SUFFIX',
    'RUN_STATUS_READ_TIMEOUT',
    'STATUS_COMPLETE',
    'STATUS_PARTIAL',
    'STATUS_EMPTY',
    'STRICT_ENV_VAR',
]

LOG = logging.getLogger('spacr.errors')

#: Table written into a SQLite artifact by :meth:`RunLedger.stamp`.
RUN_STATUS_TABLE = 'run_status'

#: Sidecar filename suffix used for non-database artifacts.
RUN_STATUS_SUFFIX = '.run_status.json'

#: Path suffixes treated as SQLite databases by :meth:`RunLedger.stamp`.
DB_SUFFIXES = ('.db', '.sqlite', '.sqlite3')

#: Seconds :func:`read_run_status` waits for a locked database before it
#: gives up and says so. SQLite's own default is 5 s; the point of naming
#: it is that "how long do we wait" and "what do we conclude if we never
#: got in" are two different decisions, and only the second one was wrong.
RUN_STATUS_READ_TIMEOUT = 5.0

STATUS_COMPLETE = 'complete'
STATUS_PARTIAL = 'partial'
STATUS_EMPTY = 'empty'

#: Environment variable that turns recoverable configuration problems
#: into immediate :class:`ConfigurationError` raises.
STRICT_ENV_VAR = 'SPACR_STRICT_ERRORS'

_TRUTHY = frozenset({'1', 'true', 'yes', 'on', 'y', 't'})

_RULE = '=' * 78


# ---------------------------------------------------------------------------
# Exception types
# ---------------------------------------------------------------------------

class SpacrError(Exception):
    """Base class for every spaCR-raised error.

    Catch this to catch anything spaCR raises deliberately, as opposed
    to an incidental ``ValueError`` from numpy or pandas.
    """


class ConfigurationError(SpacrError):
    """The run was set up wrongly and cannot produce a valid result.

    A missing ``src`` folder, an unparseable regex, a metadata column
    that does not exist. These are *not* per-item failures: continuing
    past one produces garbage for every item, so
    :meth:`RunLedger.item` re-raises this instead of recording it.
    """


class DataIntegrityError(SpacrError):
    """The data produced cannot be trusted.

    Raised when an artifact is internally inconsistent, or when a run
    failed on so many items that its output is not meaningful.
    """


class PartialRunError(DataIntegrityError):
    """Raised by :meth:`RunLedger.raise_if_worse_than` past the threshold.

    A subclass of :class:`DataIntegrityError` so callers that only care
    about "the answer is wrong" can catch the parent.
    """


class RunStatusUnreadable(DataIntegrityError):
    """The stamp could not be read, so the run's verdict is unknown.

    Distinct from "this artifact was never stamped", which is a perfectly
    ordinary state and reads as ``[]`` / complete. This one means the
    reader was *stopped*: the database is locked by a writer that still
    holds it, or the file is truncated or otherwise not a database.

    Both of those are what an interrupted run leaves behind, and both used
    to be swallowed by one ``except sqlite3.Error: return []`` and reported
    as "no stamps, therefore complete" — so a run killed mid-write, whose
    process still held the lock, read as finished. Measured on a real
    measurements.db stamped ``partial``: ``run_is_complete`` said ``False``
    when nothing held the file, and ``True`` — with ``assert_run_complete``
    passing — while a second connection held ``BEGIN EXCLUSIVE``. The
    database said the run failed a field either way; the lock is what
    stopped anyone hearing it.
    """


# ---------------------------------------------------------------------------
# Failure record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Failure:
    """One recorded per-item failure.

    :param item: identifier of the thing that failed — a filename, a
        well id, a fold number. This is what makes the ledger
        actionable, so it is always stringified and never empty.
    :param stage: pipeline stage the failure happened in.
    :param exc_type: exception class name, used as the grouping key.
    :param message: ``str(exc)``.
    :param traceback_str: formatted traceback, kept so the ``ERROR``
        log record carries the whole story.
    :param timestamp: Unix time the failure was recorded.
    """

    item: str
    stage: str
    exc_type: str
    message: str
    traceback_str: str = ''
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable copy of this record."""
        return {
            'item': self.item,
            'stage': self.stage,
            'exc_type': self.exc_type,
            'message': self.message,
            'traceback_str': self.traceback_str,
            'timestamp': self.timestamp,
        }

    def short(self) -> str:
        """Return a one-line ``item: message`` rendering for the summary."""
        return f'{self.item}: {self.message}'


def _utcnow() -> str:
    """Return the current UTC time as an ISO-8601 string (seconds resolution)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

class RunLedger:
    """Accounting for one batch run: what was attempted, what failed, and why.

    A ledger is cheap — create one per pipeline invocation (or per
    source folder), wrap each loop body in :meth:`item`, and call
    :meth:`finalize` before returning.

    :param name: run name, shown in the summary block and stored in
        the artifact stamp. Use the pipeline stage, e.g.
        ``'measure_crop'``.
    :param logger: logger to emit failure records on. Defaults to the
        module logger, which funnels into ``~/.spacr/logs/spacr.log``
        once :func:`spacr.logging_util.setup_logging` has run.

    Example:
        .. code-block:: python

            ledger = RunLedger('measure_crop')
            for well in wells:
                with ledger.item(well, stage='measure'):
                    measure(well)
            ledger.finalize(artifact='measurements.db')
    """

    def __init__(self, name: str = 'run', logger: Optional[logging.Logger] = None):
        self.name = str(name)
        self.run_id = uuid.uuid4().hex[:12]
        self.started_utc = _utcnow()
        self._log = logger if logger is not None else LOG
        self._failures: List[Failure] = []
        self._n_succeeded = 0
        self._success_by_stage: Counter = Counter()

    # -- counters ---------------------------------------------------------

    @property
    def failures(self) -> List[Failure]:
        """Every recorded :class:`Failure`, in the order they happened."""
        return list(self._failures)

    @property
    def n_failed(self) -> int:
        """Number of items that failed."""
        return len(self._failures)

    @property
    def n_succeeded(self) -> int:
        """Number of items that completed cleanly."""
        return self._n_succeeded

    @property
    def n_attempted(self) -> int:
        """Number of items attempted — successes plus failures."""
        return self._n_succeeded + len(self._failures)

    @property
    def failure_rate(self) -> float:
        """Fraction of attempted items that failed; ``0.0`` when nothing ran."""
        attempted = self.n_attempted
        if attempted == 0:
            return 0.0
        return len(self._failures) / attempted

    @property
    def status(self) -> str:
        """``'complete'``, ``'partial'`` or ``'empty'``."""
        if self.n_attempted == 0:
            return STATUS_EMPTY
        if self._failures:
            return STATUS_PARTIAL
        return STATUS_COMPLETE

    @property
    def is_complete(self) -> bool:
        """True when nothing failed. An empty run counts as complete."""
        return not self._failures

    def __repr__(self) -> str:
        return (f'<RunLedger {self.name!r} status={self.status} '
                f'attempted={self.n_attempted} failed={self.n_failed}>')

    # -- recording --------------------------------------------------------

    def record_success(self, item: Any, stage: Optional[str] = None) -> 'RunLedger':
        """Record that ``item`` completed cleanly.

        :param item: identifier of the processed item.
        :param stage: pipeline stage; defaults to the ledger name.
        :returns: ``self``, so calls can be chained.
        """
        stage_name = str(stage) if stage is not None else self.name
        self._n_succeeded += 1
        self._success_by_stage[stage_name] += 1
        self._log.debug('[%s] %s succeeded at stage %r', self.name, item, stage_name)
        return self

    def record_failure(self, item: Any, stage: Optional[str] = None,
                       exc: Any = None) -> Failure:
        """Record that ``item`` failed, logging it loudly at ``ERROR``.

        :param item: identifier of the item that failed — the thing a
            human needs in order to go and look at it.
        :param stage: pipeline stage; defaults to the ledger name.
        :param exc: the caught exception. A plain string is accepted
            for failures that were detected rather than raised.
        :returns: the stored :class:`Failure`.
        """
        stage_name = str(stage) if stage is not None else self.name
        if isinstance(exc, BaseException):
            exc_type = type(exc).__name__
            message = str(exc) or exc_type
            tb = ''.join(_traceback.format_exception(
                type(exc), exc, exc.__traceback__))
        elif exc is None:
            exc_type = 'Failure'
            message = 'unspecified failure'
            tb = ''
        else:
            exc_type = 'Failure'
            message = str(exc)
            tb = ''

        failure = Failure(item=str(item), stage=stage_name, exc_type=exc_type,
                          message=message, traceback_str=tb)
        self._failures.append(failure)
        self._log.error('[%s] FAILED %s (stage %s): %s: %s',
                        self.name, failure.item, stage_name, exc_type, message)
        if tb:
            # DEBUG, not ERROR: forty failures would otherwise dump forty
            # tracebacks over the console. The full text is kept on the
            # Failure and persisted by stamp(), which is the durable record.
            self._log.debug('[%s] traceback for %s:\n%s',
                            self.name, failure.item, tb)
        return failure

    @contextmanager
    def item(self, name: Any, stage: Optional[str] = None,
             echo: Optional[str] = None) -> Iterator['RunLedger']:
        """Run one loop body: swallow and record its failure, keep the batch alive.

        On a clean exit the item is counted as a success. On an
        ordinary exception the item is recorded as a failure and the
        loop carries on.

        Two things are deliberately **re-raised** rather than recorded:

        * :class:`ConfigurationError` — a wrong ``src`` path is not a
          per-item failure, and pretending it is would turn one
          mistake into N recorded "data" errors.
        * ``KeyboardInterrupt`` / ``SystemExit`` — Ctrl-C must abort,
          not be filed as a corrupt image.

        :param name: identifier of this item, recorded verbatim.
        :param stage: pipeline stage; defaults to the ledger name.
        :param echo: when set, a failure additionally prints
            ``f"{echo}: {exc}"`` to stdout. Adoption sites use this to
            keep the exact console message users already rely on.

        Example:
            .. code-block:: python

                for path in paths:
                    with ledger.item(path, stage='load'):
                        arrays.append(np.load(path))
        """
        try:
            yield self
        except ConfigurationError:
            # Setup is wrong for every item — surviving is not an option.
            raise
        except (KeyboardInterrupt, SystemExit):
            # Operator intent, not a data problem.
            raise
        except Exception as exc:
            self.record_failure(name, stage, exc)
            if echo is not None:
                print(f'{echo}: {exc}')
        else:
            self.record_success(name, stage)

    # -- reporting --------------------------------------------------------

    def grouped_failures(self) -> "OrderedDict[str, List[Failure]]":
        """Group failures by exception type, in first-seen order.

        Forty identical ``FileNotFoundError``\\ s are one problem, not
        forty, and this is what makes the summary readable.
        """
        groups: "OrderedDict[str, List[Failure]]" = OrderedDict()
        for failure in self._failures:
            groups.setdefault(failure.exc_type, []).append(failure)
        return groups

    def summary(self, max_groups: int = 10, max_examples: int = 3) -> str:
        """Render the loud end-of-run block.

        :param max_groups: at most this many exception types are shown.
        :param max_examples: at most this many *distinct* messages are
            shown per exception type.
        :returns: a multi-line string, ready to print.
        """
        lines: List[str] = [_RULE]
        if self._failures:
            lines.append(f' spaCR RUN INCOMPLETE — {self.name}')
        elif self.n_attempted == 0:
            lines.append(f' spaCR run processed nothing — {self.name}')
        else:
            lines.append(f' spaCR run complete — {self.name}')
        lines.append(_RULE)
        lines.append(f' attempted : {self.n_attempted}')
        lines.append(f' succeeded : {self.n_succeeded}')
        lines.append(f' failed    : {self.n_failed}  ({self.failure_rate:.1%})')

        groups = self.grouped_failures()
        if groups:
            lines.append('')
            lines.append(' Failures grouped by exception type:')
            for gi, (exc_type, failures) in enumerate(groups.items()):
                if gi >= max_groups:
                    lines.append(f'   ... and {len(groups) - max_groups} '
                                 f'more exception type(s)')
                    break
                lines.append(f'   {exc_type} x{len(failures)}')
                by_message: "OrderedDict[str, List[Failure]]" = OrderedDict()
                for failure in failures:
                    by_message.setdefault(failure.message, []).append(failure)
                for mi, (message, same) in enumerate(by_message.items()):
                    if mi >= max_examples:
                        lines.append(f'       ... and {len(by_message) - max_examples} '
                                     f'more distinct message(s)')
                        break
                    suffix = f' (x{len(same)})' if len(same) > 1 else ''
                    lines.append(f'       {same[0].item}: {message}{suffix}')

            lines.append('')
            lines.append(f' ARTIFACTS FROM THIS RUN ARE INCOMPLETE — they cover '
                         f'{self.n_succeeded} of {self.n_attempted} items.')
            lines.append(' Treat any downstream result computed from them as suspect.')
        lines.append(_RULE)
        return '\n'.join(lines)

    def raise_if_worse_than(self, threshold: float,
                            message: Optional[str] = None) -> 'RunLedger':
        """Abort when the failure rate is *strictly above* ``threshold``.

        Use where a partial result is not merely incomplete but
        meaningless — a 5-fold cross-validation in which 3 folds died
        does not have a spread worth reporting.

        :param threshold: fraction in ``[0, 1]``. ``0.5`` aborts when
            more than half the items failed; a rate exactly equal to
            the threshold does *not* abort.
        :param message: override for the error text.
        :raises PartialRunError: when the rate exceeds ``threshold``.
        :returns: ``self`` when the run is acceptable.
        """
        if self.n_attempted == 0:
            return self
        if self.failure_rate <= threshold:
            return self
        text = message or (
            f'{self.name}: {self.n_failed} of {self.n_attempted} items failed '
            f'({self.failure_rate:.1%}), above the {threshold:.1%} threshold — '
            f'the result is not meaningful.')
        self._log.error(text)
        raise PartialRunError(f'{text}\n{self.summary()}')

    def finalize(self, artifact: Optional[Union[str, os.PathLike]] = None,
                 threshold: Optional[float] = None,
                 quiet_when_clean: bool = True) -> 'RunLedger':
        """Emit the summary, stamp the artifact, then optionally abort.

        Call this as the *last* thing a pipeline function does, so the
        verdict is the last thing on screen rather than 400 lines up.

        :param artifact: path to the file this run produced. Stamped
            via :meth:`stamp` so the artifact itself records that it is
            partial.
        :param threshold: when given, :meth:`raise_if_worse_than` is
            applied *after* the artifact has been stamped, so the
            evidence survives the abort.
        :param quiet_when_clean: when True (default) a run with no
            failures prints nothing and only logs at ``INFO``. A run
            with failures always prints the loud block.
        :returns: ``self``.
        """
        text = self.summary()
        if self._failures:
            # One-line log record, full block on stdout: logging the whole
            # block too rendered the summary twice in a plain terminal
            # session (logging's last-resort handler writes to stderr).
            self._log.error('[%s] RUN INCOMPLETE — %d of %d items failed '
                            '(%.1f%%); artifacts are partial',
                            self.name, self.n_failed, self.n_attempted,
                            self.failure_rate * 100)
            print(text, file=sys.stdout)
        else:
            self._log.info('[%s] %d/%d items succeeded',
                           self.name, self.n_succeeded, self.n_attempted)
            if not quiet_when_clean:
                print(text, file=sys.stdout)
        if artifact is not None:
            self.stamp(artifact)
        if threshold is not None:
            self.raise_if_worse_than(threshold)
        return self

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the whole ledger as a JSON-serialisable dict."""
        return {
            'run_id': self.run_id,
            'name': self.name,
            'status': self.status,
            'n_attempted': self.n_attempted,
            'n_succeeded': self.n_succeeded,
            'n_failed': self.n_failed,
            'failure_rate': self.failure_rate,
            'started_utc': self.started_utc,
            'stamped_utc': _utcnow(),
            'success_by_stage': dict(self._success_by_stage),
            'failures': [f.to_dict() for f in self._failures],
            'summary': self.summary(),
        }

    def to_json(self, path: Union[str, os.PathLike]) -> Path:
        """Write :meth:`to_dict` to ``path`` as JSON, creating parent dirs.

        :param path: destination file.
        :returns: the written :class:`~pathlib.Path`.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding='utf-8')
        return target

    def stamp(self, artifact: Union[str, os.PathLike]) -> Path:
        """Record this run's verdict *into* the artifact it produced.

        For a SQLite path (``.db`` / ``.sqlite`` / ``.sqlite3``) a row
        is appended to the :data:`RUN_STATUS_TABLE` table. For anything
        else a sibling ``<stem>.run_status.json`` is written next to
        the file. Either way a later reader — a person or
        :func:`read_run_status` — can tell the result is partial.

        Stamps accumulate, so a database written by several stages ends
        up with one row per stage.

        :param artifact: path of the file this run produced.
        :returns: the path actually written (the db, or the sidecar).
        """
        target = Path(artifact)
        if target.suffix.lower() in DB_SUFFIXES:
            return self._stamp_db(target)
        return self._stamp_sidecar(target)

    def _stamp_db(self, db_path: Path) -> Path:
        """Append one ``run_status`` row to a SQLite database."""
        payload = self.to_dict()
        conn = sqlite3.connect(str(db_path), timeout=30)
        try:
            conn.execute(
                f'CREATE TABLE IF NOT EXISTS {RUN_STATUS_TABLE} ('
                'run_id TEXT, name TEXT, status TEXT, '
                'n_attempted INTEGER, n_succeeded INTEGER, n_failed INTEGER, '
                'failure_rate REAL, started_utc TEXT, stamped_utc TEXT, '
                'failures_json TEXT, summary TEXT)')
            conn.execute(
                f'INSERT INTO {RUN_STATUS_TABLE} VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                (payload['run_id'], payload['name'], payload['status'],
                 payload['n_attempted'], payload['n_succeeded'],
                 payload['n_failed'], payload['failure_rate'],
                 payload['started_utc'], payload['stamped_utc'],
                 json.dumps(payload['failures']), payload['summary']))
            conn.commit()
        finally:
            conn.close()
        return db_path

    def _stamp_sidecar(self, artifact: Path) -> Path:
        """Append this run to ``<stem>.run_status.json`` beside ``artifact``."""
        sidecar = _sidecar_path(artifact)
        records: List[Dict[str, Any]] = []
        if sidecar.is_file():
            existing = json.loads(sidecar.read_text(encoding='utf-8'))
            if isinstance(existing, list):
                records = existing
            else:
                records = [existing]
        records.append(self.to_dict())
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(records, indent=2), encoding='utf-8')
        return sidecar


# ---------------------------------------------------------------------------
# Reading a stamp back
# ---------------------------------------------------------------------------

def _sidecar_path(artifact: Union[str, os.PathLike]) -> Path:
    """Return the ``<stem>.run_status.json`` path for ``artifact``."""
    target = Path(artifact)
    if target.name.endswith(RUN_STATUS_SUFFIX):
        return target
    return target.parent / (target.stem + RUN_STATUS_SUFFIX)


_STATUS_COLUMNS = ('run_id', 'name', 'status', 'n_attempted', 'n_succeeded',
                   'n_failed', 'failure_rate', 'started_utc', 'stamped_utc',
                   'failures_json', 'summary')


def _has_run_status_table(conn: sqlite3.Connection) -> bool:
    """True when this database declares a :data:`RUN_STATUS_TABLE`.

    Asked separately from the ``SELECT`` so that "there is no such table"
    (an ordinary, informative answer) stops being indistinguishable from
    "the database would not let me look" — which is what a locked or
    truncated file gives, and which used to be reported as *complete*.
    Both questions go through the same connection, so a database that
    cannot be opened at all fails on this one and never reaches the read.
    """
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        (RUN_STATUS_TABLE,)).fetchone()
    return row is not None


def read_run_status(artifact: Union[str, os.PathLike],
                    timeout: float = RUN_STATUS_READ_TIMEOUT
                    ) -> List[Dict[str, Any]]:
    """Read back every :meth:`RunLedger.stamp` recorded for ``artifact``.

    Works for both stamp flavours: a SQLite path is read from its
    :data:`RUN_STATUS_TABLE`, anything else from its
    ``<stem>.run_status.json`` sidecar.

    Three outcomes, deliberately kept apart:

    * **stamps exist** — they are returned, oldest first;
    * **the artifact exists and holds no stamp** — ``[]``, meaning "no
      information". Stamping is opt-in, so this covers every output
      written before a ledger reached that code path;
    * **the artifact could not be read** — :class:`RunStatusUnreadable`.
      A database still locked by its writer, or truncated by a ``kill``
      mid-write, is exactly what an *interrupted* run leaves, and folding
      it into the second case is how an interrupted run came back
      "complete".

    :param artifact: path of a spaCR output — e.g. ``measurements.db``.
    :param timeout: seconds to wait for a locked database before giving
        up. Default :data:`RUN_STATUS_READ_TIMEOUT`.
    :returns: one dict per recorded run, oldest first. Each has
        ``status`` / ``n_attempted`` / ``n_succeeded`` / ``n_failed`` /
        ``failure_rate`` / ``summary`` and a ``failures`` list. An
        artifact that was never stamped returns ``[]``.
    :raises RunStatusUnreadable: when the artifact exists but cannot be
        read — locked, truncated, corrupt, or malformed JSON.

    Example:
        .. code-block:: python

            from spacr.errors import read_run_status
            for run in read_run_status('/data/plate1/measurements/measurements.db'):
                if run['n_failed']:
                    print(run['summary'])
    """
    target = Path(artifact)
    if target.suffix.lower() in DB_SUFFIXES:
        if not target.is_file():
            return []
        conn = None
        try:
            conn = sqlite3.connect(str(target), timeout=timeout)
            if not _has_run_status_table(conn):
                # Never stamped. The artifact predates stamping, or was
                # written by a code path that does not stamp yet.
                return []
            rows = conn.execute(
                f'SELECT {", ".join(_STATUS_COLUMNS)} FROM {RUN_STATUS_TABLE} '
                'ORDER BY rowid').fetchall()
        except sqlite3.Error as exc:
            raise RunStatusUnreadable(
                f'{target} exists but its run status cannot be read: {exc}. '
                f'A database still held by the process that was writing it, '
                f'or one truncated by a crash, fails here — so this is "the '
                f'run may not have finished", not "the run finished". Wait '
                f'for the writer to exit, or check the file.') from exc
        finally:
            if conn is not None:
                conn.close()
        records = []
        for row in rows:
            record = dict(zip(_STATUS_COLUMNS, row))
            record['failures'] = json.loads(record.pop('failures_json') or '[]')
            records.append(record)
        return records

    sidecar = _sidecar_path(target)
    if not sidecar.is_file():
        return []
    try:
        payload = json.loads(sidecar.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        # A sidecar half-written by an interrupted run is the same
        # species of evidence as a locked database, and gets the same
        # answer: unknown, never "complete".
        raise RunStatusUnreadable(
            f'{sidecar} exists but cannot be read: {exc}. A run status '
            f'sidecar truncated mid-write means the run that was writing '
            f'it did not finish.') from exc
    if isinstance(payload, list):
        return payload
    return [payload]


def run_is_complete(artifact: Union[str, os.PathLike],
                    timeout: float = RUN_STATUS_READ_TIMEOUT) -> bool:
    """True when no stamp on ``artifact`` recorded a failure.

    An artifact that was never stamped reads as complete — stamping is
    opt-in and predates neither the older outputs on disk nor the code
    paths that have not adopted a ledger yet. Use
    :func:`read_run_status` when you need to distinguish "verified
    clean" from "no information".

    An artifact whose status *cannot be read* reads as **not** complete.
    That is the one case where the two answers differ in consequence: an
    unstamped file is silent, whereas a locked or truncated one is
    positive evidence that something was interrupted, and answering
    "complete" there is how a killed run passed for a finished one.

    :param artifact: path of a spaCR output.
    :param timeout: seconds to wait for a locked database.
    """
    try:
        records = read_run_status(artifact, timeout=timeout)
    except RunStatusUnreadable:
        return False
    return all(int(record.get('n_failed', 0) or 0) == 0
               for record in records)


def assert_run_complete(artifact: Union[str, os.PathLike],
                        timeout: float = RUN_STATUS_READ_TIMEOUT) -> None:
    """Raise :class:`DataIntegrityError` if ``artifact`` is stamped partial.

    The one-liner for downstream code that must not silently analyse a
    subset. An artifact whose status cannot be read raises
    :class:`RunStatusUnreadable`, which is a
    :class:`DataIntegrityError` too — so ``except DataIntegrityError``
    catches both "this run failed items" and "I cannot tell whether it
    did", which are the two cases a caller must not proceed past.

    :param artifact: path of a spaCR output.
    :param timeout: seconds to wait for a locked database.
    :raises DataIntegrityError: when any stamp recorded a failure.
    :raises RunStatusUnreadable: when the status cannot be read at all.
    """
    records = read_run_status(artifact, timeout=timeout)
    bad = [r for r in records if int(r.get('n_failed', 0) or 0) > 0]
    if not bad:
        return
    detail = '\n'.join(str(r.get('summary', '')) for r in bad)
    raise DataIntegrityError(
        f'{artifact} was produced by a run that did not complete:\n{detail}')


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------

def strict_errors(settings: Any = None) -> bool:
    """True when recoverable setup errors should be raised instead of printed.

    Resolution order: an explicit ``strict_errors`` key in ``settings``
    wins; otherwise the :data:`STRICT_ENV_VAR` environment variable is
    consulted. Off by default, so adopting this never changes an
    existing pipeline's behaviour.

    :param settings: a spaCR settings dict, or None.
    """
    if isinstance(settings, dict) and settings.get('strict_errors') is not None:
        return bool(settings['strict_errors'])
    return os.environ.get(STRICT_ENV_VAR, '').strip().lower() in _TRUTHY


def raise_if_strict(message: str, exc: Optional[BaseException] = None,
                    settings: Any = None,
                    error_type: type = ConfigurationError) -> bool:
    """Raise ``error_type(message)`` in strict mode; otherwise log it at ``ERROR``.

    Used at the category-B sites that historically printed and carried
    on. The default path keeps the legacy behaviour but stops the
    problem from being invisible to the log; setting
    ``SPACR_STRICT_ERRORS=1`` turns the same site into a hard stop.

    :param message: what went wrong, and why the result is untrustworthy.
    :param exc: the caught exception, chained onto the raise.
    :param settings: settings dict consulted for ``strict_errors``.
    :param error_type: exception class to raise.
    :returns: False when not strict, so callers can branch on it.
    """
    if strict_errors(settings):
        error = error_type(message)
        if exc is not None:
            raise error from exc
        raise error
    LOG.error(message)
    return False
