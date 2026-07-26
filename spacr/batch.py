"""``spacr.batch`` — stack arbitrary module+settings jobs into one queue and
run them unattended.

The Plate Queue (:mod:`spacr.qt.plate_queue`) chains *plates* through *one*
pipeline with *one* settings dict. This module is the other axis: a queue of
arbitrary ``(module, settings)`` jobs in any order —
``Mask → Measure → Classify (CV) → Classify (ML)``, then the same four again
with a different diameter, then a fifth plate's Mask — which is what a night
of plate-scale work actually looks like.

A queued job **is** a ``spacr-run`` invocation. Nothing here re-implements
module dispatch or settings loading: :mod:`spacr.cli` owns the module
registry, the settings loader, the ``--set`` coercion and the exit-code
contract (0 ok, 1 the module raised, 2 bad arguments or settings), and this
module drives it. Likewise a queue is a **ledger of ledgers**: the queue's own
verdict is a :class:`spacr.errors.RunLedger`, stamped next to the queue file,
and each job's :func:`spacr.errors.read_run_status` stamp is read back so a
job that exited 0 having silently skipped 40 fields is reported as *partial*
rather than as a success.

What makes a queue survive a night nobody is watching:

* **Every job is validated when it is added, not when it runs.** Discovering
  at 3 a.m. that job 9's ``src`` is misspelled wastes the whole night.
  :func:`validate_queue` reports *all* problems at once and
  :func:`run_queue` refuses to start while any of them is an error.
* **State is persisted after every transition, atomically.** A machine that
  reboots mid-queue is resumed with :func:`resume_queue`, not restarted. The
  file is written to a temp name and ``os.replace``\\ d, because a queue file
  truncated by a crash is worse than no queue file at all.
* **A job whose dependency failed is SKIPPED, not run.** Measure after a
  failed Mask produces a database that looks like a real result.
  ``skipped`` (deliberately not run), ``failed`` and ``not_run`` (the queue
  halted before reaching it) are three different things and stay that way.
* **continue-on-error stops hiding a systematic failure.** If the first three
  jobs died the same way the remaining nine will too;
  ``max_consecutive_failures`` halts the queue and says so.
* **Per-job logs go to their own file** — one interleaved log from twelve
  overnight jobs is unreadable — and the path is on the job record.

Concurrency: jobs run **strictly one at a time**. They compete for one GPU,
and two Cellpose jobs sharing a card is how an overnight run turns into an
overnight CUDA OOM. There is deliberately no ``max_workers``; run two queues
in two processes if you really have two cards.

Import cost: this module imports only the standard library plus
:mod:`spacr.cli`, :mod:`spacr.errors` and :mod:`spacr.validate`, all of which
are torch-free. A queue file can be read, validated and planned on a login
node without a GPU stack — the pipeline itself is only imported inside the
subprocess that runs a job.

Typical use::

    from spacr.batch import Job, Queue, run_queue, save_queue

    q = Queue(name='overnight')
    q.add(Job(module='mask',    settings='/data/p1/settings/mask.csv'))
    q.add(Job(module='measure', settings='/data/p1/settings/measure.csv',
              depends_on=['mask-1']))
    save_queue(q, '/data/overnight.queue.json')
    result = run_queue(q, path='/data/overnight.queue.json')
    print(result.summary())
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .cli import (
    EXIT_OK,
    EXIT_USAGE,
    INTERACTIVE_ONLY,
    SettingsError,
    apply_overrides,
    load_settings_file,
    module_defaults,
    resolve_module,
)
from .errors import DB_SUFFIXES, RUN_STATUS_SUFFIX, RunLedger, SpacrError, read_run_status
from .validate import ERROR, WARNING, validate_settings

__all__ = [
    'QUEUE_FORMAT',
    'STATUS_PENDING',
    'STATUS_RUNNING',
    'STATUS_SUCCESS',
    'STATUS_FAILED',
    'STATUS_SKIPPED',
    'STATUS_NOT_RUN',
    'ALL_STATUSES',
    'RESUMABLE_STATUSES',
    'ON_ERROR_CHOICES',
    'QueueError',
    'fmt_duration',
    'classify_failure',
    'Problem',
    'format_problems',
    'Job',
    'Queue',
    'Progress',
    'QueueResult',
    'load_queue',
    'save_queue',
    'validate_job',
    'validate_queue',
    'plan',
    'resolve_job_settings',
    'job_command',
    'subprocess_runner',
    'inprocess_runner',
    'run_queue',
    'resume_queue',
]

LOG = logging.getLogger('spacr.batch')

#: On-disk format version of the queue file. Bumped only for a breaking
#: change; :func:`load_queue` refuses a newer one rather than guessing.
QUEUE_FORMAT = 1

# -- job lifecycle ----------------------------------------------------------

#: Never attempted; still runnable.
STATUS_PENDING = 'pending'
#: Currently executing. Persisted, so a crash leaves evidence of where.
STATUS_RUNNING = 'running'
#: Exited 0. May still be *partial* — see :attr:`Job.run_status`.
STATUS_SUCCESS = 'success'
#: Ran and failed.
STATUS_FAILED = 'failed'
#: Deliberately not run because something it depends on did not succeed.
STATUS_SKIPPED = 'skipped'
#: The queue halted before reaching it. Distinct from ``skipped``: nothing is
#: wrong with this job, it simply never got its turn.
STATUS_NOT_RUN = 'not_run'

ALL_STATUSES: Tuple[str, ...] = (
    STATUS_PENDING, STATUS_RUNNING, STATUS_SUCCESS,
    STATUS_FAILED, STATUS_SKIPPED, STATUS_NOT_RUN,
)

#: Statuses :func:`resume_queue` will pick up and run.
RESUMABLE_STATUSES: Tuple[str, ...] = (STATUS_PENDING, STATUS_NOT_RUN)

ON_ERROR_CHOICES: Tuple[str, ...] = ('continue', 'stop')

_RULE = '=' * 78


class QueueError(SpacrError):
    """The queue itself is wrong — a bad job, a cycle, a refused start.

    A subclass of :class:`spacr.errors.SpacrError` so ``except SpacrError``
    catches it alongside everything else spaCR raises deliberately.
    """


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string, microseconds kept.

    Seconds resolution (what :mod:`spacr.errors` uses for its stamps) would
    round a 0.4 s job to a 0 s job, and per-job durations are part of the
    deliverable here.
    """
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(text: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp written by :func:`_now_iso`, or None."""
    if not isinstance(text, str) or not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def fmt_duration(seconds: Optional[float]) -> str:
    """Render a duration the way an overnight summary should read.

    :param seconds: elapsed seconds, or None when the job never finished.
    :returns: ``'—'``, ``'42.1s'``, ``'7m 12s'`` or ``'7h 41m'``.
    """
    if seconds is None:
        return '—'
    seconds = float(seconds)
    if seconds < 60:
        return f'{seconds:.1f}s'
    if seconds < 3600:
        return f'{int(seconds // 60)}m {int(seconds % 60):02d}s'
    return f'{int(seconds // 3600)}h {int((seconds % 3600) // 60):02d}m'


# ---------------------------------------------------------------------------
# problems
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """One thing wrong with a job or with the queue.

    Deliberately the same shape as :class:`spacr.validate.Problem` — severity,
    message, fix — plus the ``job_id`` needed to say *which* of twelve jobs is
    at fault. Problems produced by :func:`spacr.validate.validate_settings`
    are wrapped into this, not re-invented.

    :param job_id: id of the offending job; ``''`` for a queue-level problem.
    :param severity: ``'error'`` (the queue must not start) or ``'warning'``.
    :param message: what is wrong, in the user's terms.
    :param fix: what to actually do about it.
    :param setting: the settings key at fault, when there is one.
    """

    job_id: str
    severity: str
    message: str
    fix: str
    setting: str = ''

    @property
    def is_error(self) -> bool:
        """True when this problem must stop the queue from starting."""
        return self.severity == ERROR

    def __str__(self) -> str:
        where = f'{self.job_id}: ' if self.job_id else ''
        key = f'[{self.setting}] ' if self.setting else ''
        return f'{where}{key}{self.message}\n    fix: {self.fix}'


def format_problems(problems: Sequence[Problem], title: str = 'queue check') -> str:
    """Render every problem at once, errors first.

    Reporting the first error and stopping is what makes a twelve-job queue
    take twelve rounds of fixing; this prints all of them.

    :param problems: what :func:`validate_queue` returned.
    :param title: heading for the block.
    :returns: the report as one string, no trailing newline.
    """
    errors = [p for p in problems if p.is_error]
    warnings = [p for p in problems if not p.is_error]
    lines = [f'{title}: {len(errors)} error(s), {len(warnings)} warning(s)']
    if not problems:
        return f'{title}: no problems found — every job is runnable.'
    if errors:
        lines.append('')
        lines.append('ERRORS — the queue will not start until these are fixed:')
        for problem in errors:
            lines.append(f'  {problem}')
    if warnings:
        lines.append('')
        lines.append('WARNINGS — the queue will run, but check these:')
        for problem in warnings:
            lines.append(f'  {problem}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# jobs and queues
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """One ``spacr-run`` invocation, with its place in the queue's history.

    :param module: module key or alias understood by
        :func:`spacr.cli.resolve_module` — ``'mask'``, ``'measure'``,
        ``'ml_analyze'``, ...
    :param settings: path to a settings CSV/JSON (what ``spacr-run
        --settings`` takes), or an inline settings dict for a job that has no
        file of its own.
    :param id: unique, human-typable identifier. Left empty,
        :meth:`Queue.add` assigns one like ``'mask-1'``.
    :param label: what to show a human; defaults to ``module @ src``.
    :param overrides: ``key=value`` strings applied on top of ``settings``,
        exactly like ``spacr-run --set``. This is how "the same four jobs
        again with a different diameter" is written without copying four
        settings files. A mapping is accepted and normalised.
    :param depends_on: ids of jobs that must succeed first. A job whose
        dependency failed is *skipped*, never run.
    :param status: one of :data:`ALL_STATUSES`.
    :param started: ISO-8601 UTC start time, or ``''``.
    :param finished: ISO-8601 UTC end time, or ``''``.
    :param exit_code: the process exit code — 0 ok, 1 the module raised,
        2 bad settings (:mod:`spacr.cli`'s contract).
    :param error: one-line explanation of a failure or a skip.
    :param log_path: this job's own log file.
    :param run_status: the job's :func:`spacr.errors.read_run_status` verdict,
        summarised — ``None`` when the job stamped nothing.
    """

    module: str
    settings: Union[str, Dict[str, Any]] = ''
    id: str = ''
    label: str = ''
    overrides: Union[List[str], Dict[str, Any]] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    status: str = STATUS_PENDING
    started: str = ''
    finished: str = ''
    exit_code: Optional[int] = None
    error: str = ''
    log_path: str = ''
    run_status: Optional[Dict[str, Any]] = None

    # -- derived ----------------------------------------------------------

    @property
    def override_args(self) -> List[str]:
        """``overrides`` normalised to the ``key=value`` strings ``--set`` takes."""
        if isinstance(self.overrides, dict):
            return [f'{k}={v}' for k, v in self.overrides.items()]
        return [str(item) for item in (self.overrides or [])]

    @property
    def settings_path(self) -> str:
        """The settings file path, or ``''`` when the job carries a dict."""
        return self.settings if isinstance(self.settings, str) else ''

    @property
    def duration_s(self) -> Optional[float]:
        """Wall-clock seconds the job took, or None if it never finished."""
        start, end = _parse_iso(self.started), _parse_iso(self.finished)
        if start is None or end is None:
            return None
        return (end - start).total_seconds()

    @property
    def elapsed_s(self) -> Optional[float]:
        """Seconds the job has taken, counting up while it is still running.

        :returns: :attr:`duration_s` once it has finished, the time since it
            started while it is running, else None.
        """
        if self.duration_s is not None:
            return self.duration_s
        start = _parse_iso(self.started)
        if self.status == STATUS_RUNNING and start is not None:
            now = datetime.now(timezone.utc)
            return max(0.0, (now - start).total_seconds())
        return None

    @property
    def is_partial(self) -> bool:
        """True when the job exited 0 but its own ledger says items failed.

        This is the case the queue exists to catch: a measure run that
        processed 344 of 384 wells exits 0 and looks like a success.
        """
        if not isinstance(self.run_status, dict):
            return False
        return int(self.run_status.get('n_failed', 0) or 0) > 0

    def default_label(self) -> str:
        """A readable label derived from the module and its ``src``."""
        target = ''
        if isinstance(self.settings, dict):
            src = self.settings.get('src')
            target = str(src) if src else ''
        elif self.settings:
            target = os.path.basename(str(self.settings))
        return f'{self.module} {target}'.strip()

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the job as a JSON-serialisable dict, in a stable key order."""
        return OrderedDict((
            ('id', self.id),
            ('module', self.module),
            ('label', self.label),
            ('settings', self.settings),
            ('overrides', self.override_args),
            ('depends_on', list(self.depends_on)),
            ('status', self.status),
            ('started', self.started),
            ('finished', self.finished),
            ('exit_code', self.exit_code),
            ('error', self.error),
            ('log_path', self.log_path),
            ('run_status', self.run_status),
        ))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'Job':
        """Rebuild a job from :meth:`to_dict`, tolerating a hand-edited file.

        Every field has a default and unknown keys are ignored, so a user can
        delete the bookkeeping (``status``, ``started``, ...) from a queue
        file and still have it load — which is the whole point of a
        hand-editable format.

        :param data: one entry of the queue file's ``jobs`` list.
        :raises QueueError: when ``data`` is not a mapping with a ``module``.
        """
        if not isinstance(data, Mapping):
            raise QueueError(
                f'a job entry is a {type(data).__name__}, not an object.\n'
                f'  Each entry of "jobs" must be {{"module": ..., "settings": ...}}.')
        module = data.get('module')
        if not isinstance(module, str) or not module.strip():
            raise QueueError(
                f'a job entry has no "module": {dict(data)!r}\n'
                f'  Every job needs a module key, e.g. "module": "mask".')
        status = data.get('status') or STATUS_PENDING
        if status not in ALL_STATUSES:
            raise QueueError(
                f'job {data.get("id", "?")!r} has status {status!r}, which is not one of '
                f'{", ".join(ALL_STATUSES)}.')
        depends = data.get('depends_on') or []
        if isinstance(depends, str):
            depends = [depends]
        overrides = data.get('overrides') or []
        return cls(
            module=module.strip(),
            settings=data.get('settings') or '',
            id=str(data.get('id') or ''),
            label=str(data.get('label') or ''),
            overrides=overrides if isinstance(overrides, (list, dict)) else [str(overrides)],
            depends_on=[str(d) for d in depends],
            status=status,
            started=str(data.get('started') or ''),
            finished=str(data.get('finished') or ''),
            exit_code=data.get('exit_code'),
            error=str(data.get('error') or ''),
            log_path=str(data.get('log_path') or ''),
            run_status=data.get('run_status') if isinstance(data.get('run_status'), dict) else None,
        )

    def reset(self) -> 'Job':
        """Clear the bookkeeping so the job can run again. Returns ``self``."""
        self.status = STATUS_PENDING
        self.started = ''
        self.finished = ''
        self.exit_code = None
        self.error = ''
        self.run_status = None
        return self

    def copy(self, **changes: Any) -> 'Job':
        """Return a fresh, never-run copy of this job with ``changes`` applied.

        The GUI's "Duplicate" button: the common way to build a queue is one
        job, then eleven variations of it.
        """
        data = dict(self.to_dict())
        data.update(changes)
        clone = Job.from_dict(data)
        clone.reset()
        if 'id' not in changes:
            clone.id = ''
        return clone


@dataclass
class Queue:
    """An ordered list of :class:`Job`\\ s, run one at a time, top to bottom.

    :param jobs: the jobs, in the order they will run.
    :param created: ISO-8601 UTC creation time.
    :param name: what to call this queue in the summary and log folder.
    """

    jobs: List[Job] = field(default_factory=list)
    created: str = field(default_factory=_now_iso)
    name: str = 'queue'

    # -- container ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.jobs)

    def __iter__(self):
        return iter(self.jobs)

    @property
    def ids(self) -> List[str]:
        """Every job id, in run order."""
        return [job.id for job in self.jobs]

    def find(self, job_id: str) -> Optional[Job]:
        """Return the job with ``job_id``, or None."""
        for job in self.jobs:
            if job.id == job_id:
                return job
        return None

    def index(self, job_id: str) -> int:
        """Position of ``job_id`` in run order, or ``-1``."""
        for i, job in enumerate(self.jobs):
            if job.id == job_id:
                return i
        return -1

    # -- editing -----------------------------------------------------------

    def mint_id(self, module: str) -> str:
        """Return an unused, human-typable id for a job of ``module``."""
        base = re.sub(r'[^a-z0-9_]+', '-', str(module).strip().lower()) or 'job'
        n = 1
        taken = set(self.ids)
        while f'{base}-{n}' in taken:
            n += 1
        return f'{base}-{n}'

    def add(self, job: Job, validate: bool = True) -> Job:
        """Append ``job``, validating it *now* rather than at 3 a.m.

        :param job: the job to add; its ``id`` and ``label`` are filled in
            when empty.
        :param validate: set False only when deliberately building an invalid
            queue (loading a hand-edited file, for instance, which reports its
            problems through :func:`validate_queue` instead of raising).
        :returns: the job, now owned by this queue.
        :raises QueueError: when the job cannot run — an unknown or GUI-only
            module, an unreadable settings file, a bad override, a duplicate
            id, or a dependency that is not already in the queue.
        """
        if not job.id:
            job.id = self.mint_id(job.module)
        elif self.find(job.id) is not None:
            raise QueueError(
                f'a job with id {job.id!r} is already in this queue.\n'
                f'  Job ids must be unique — they are how depends_on refers to a job.')
        if not job.label:
            job.label = job.default_label()
        if validate:
            problems = validate_job(job, self)
            errors = [p for p in problems if p.is_error]
            if errors:
                raise QueueError(
                    f'job {job.id!r} cannot be added:\n' +
                    '\n'.join(f'  {p}' for p in errors))
        self.jobs.append(job)
        return job

    def remove(self, job_id: str) -> bool:
        """Remove ``job_id`` and drop it from every other job's ``depends_on``.

        Leaving a dangling dependency behind would silently skip the jobs that
        referred to it, so the reference is cleaned up here.

        :returns: True when a job was removed.
        """
        job = self.find(job_id)
        if job is None:
            return False
        self.jobs.remove(job)
        for other in self.jobs:
            if job_id in other.depends_on:
                other.depends_on = [d for d in other.depends_on if d != job_id]
        return True

    def move(self, job_id: str, offset: int) -> int:
        """Move ``job_id`` ``offset`` places (negative is earlier).

        :returns: the job's new index, or ``-1`` when it is not in the queue.
        """
        i = self.index(job_id)
        if i < 0:
            return -1
        j = max(0, min(len(self.jobs) - 1, i + int(offset)))
        if i != j:
            self.jobs.insert(j, self.jobs.pop(i))
        return j

    def reset(self) -> 'Queue':
        """Clear every job's bookkeeping so the whole queue runs again."""
        for job in self.jobs:
            job.reset()
        return self

    def counts(self) -> "OrderedDict[str, int]":
        """Jobs per status, in :data:`ALL_STATUSES` order."""
        out: "OrderedDict[str, int]" = OrderedDict((s, 0) for s in ALL_STATUSES)
        for job in self.jobs:
            out[job.status] = out.get(job.status, 0) + 1
        return out

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the whole queue as a JSON-serialisable dict."""
        return OrderedDict((
            ('spacr_queue', QUEUE_FORMAT),
            ('name', self.name),
            ('created', self.created),
            ('jobs', [job.to_dict() for job in self.jobs]),
        ))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'Queue':
        """Rebuild a queue from :meth:`to_dict` or from a hand-written file.

        :raises QueueError: when the document is not a queue, is a format from
            the future, or holds a job entry that cannot be read.
        """
        if not isinstance(data, Mapping):
            raise QueueError(
                f'a queue file must hold an object, not a {type(data).__name__}.')
        version = data.get('spacr_queue', QUEUE_FORMAT)
        try:
            version = int(version)
        except (TypeError, ValueError):
            raise QueueError(f'"spacr_queue" must be a version number, got {version!r}.')
        if version > QUEUE_FORMAT:
            raise QueueError(
                f'this queue file is format {version}, but this spaCR understands '
                f'format {QUEUE_FORMAT}.\n  Upgrade spaCR, or write the queue again '
                f'from this version.')
        raw_jobs = data.get('jobs')
        if raw_jobs is None:
            raw_jobs = []
        if not isinstance(raw_jobs, (list, tuple)):
            raise QueueError('"jobs" must be a list of job objects.')
        queue = cls(jobs=[], created=str(data.get('created') or _now_iso()),
                    name=str(data.get('name') or 'queue'))
        for entry in raw_jobs:
            job = Job.from_dict(entry)
            if not job.id:
                job.id = queue.mint_id(job.module)
            if not job.label:
                job.label = job.default_label()
            queue.jobs.append(job)
        return queue


# ---------------------------------------------------------------------------
# the queue file
# ---------------------------------------------------------------------------


def _atomic_write(path: Union[str, os.PathLike], text: str) -> Path:
    """Write ``text`` to ``path`` so a crash can never truncate the target.

    The bytes go to a sibling temp file, are flushed and ``fsync``\\ ed, and
    only then replace the target with :func:`os.replace`, which is atomic on
    every platform spaCR runs on. A machine that dies at any point leaves the
    *previous* queue file intact — a half-written one would lose the record of
    which of twelve jobs had already run, which is worse than losing nothing.

    :param path: destination file.
    :returns: the written path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f'{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}')
    try:
        with open(tmp, 'w', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return target


def save_queue(queue: Queue, path: Union[str, os.PathLike]) -> Path:
    """Write ``queue`` to ``path`` atomically, as indented JSON.

    JSON rather than a bespoke format because it is stdlib (no PyYAML on a
    compute node), because :mod:`spacr.errors` already persists its stamps
    this way, and because indented JSON is genuinely hand-editable: a user
    fixing job 9's ``src`` at 3 a.m. opens this file in an editor, not a GUI.

    :param queue: the queue to persist.
    :param path: destination file.
    :returns: the written path.
    """
    return _atomic_write(path, json.dumps(queue.to_dict(), indent=2) + '\n')


def load_queue(path: Union[str, os.PathLike]) -> Queue:
    """Read a queue file written by :func:`save_queue`, or hand-written.

    :param path: the queue file.
    :returns: the :class:`Queue`.
    :raises QueueError: when the file is missing or is not a queue. Never a
        traceback: an unattended runner should fail with a sentence.
    """
    target = Path(path)
    if not target.exists():
        raise QueueError(f'queue file not found: {target}')
    if target.is_dir():
        raise QueueError(f'{target} is a folder, not a queue file.')
    try:
        text = target.read_text(encoding='utf-8')
    except OSError as exc:
        raise QueueError(f'could not read {target}: {exc}') from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise QueueError(
            f'{target} is not valid JSON: {exc}\n'
            f'  A queue file looks like {{"spacr_queue": 1, "jobs": [...]}}.') from exc
    return Queue.from_dict(data)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _src_values(settings: Mapping[str, Any]) -> List[str]:
    """``src`` normalised to a list of strings.

    Mirrors :func:`spacr.validate._src_values` (itself mirroring
    ``spacr.utils.normalize_src_path``); copied rather than imported so this
    module depends only on validate's public API.
    """
    src = settings.get('src')
    if isinstance(src, (list, tuple)):
        return [str(v) for v in src if isinstance(v, (str, os.PathLike))]
    if isinstance(src, str) and src.strip():
        return [src]
    return []


def _within(path: str, root: str) -> bool:
    """True when ``path`` is ``root`` or lives underneath it."""
    try:
        p = os.path.normpath(os.path.abspath(str(path)))
        r = os.path.normpath(os.path.abspath(str(root)))
    except (TypeError, ValueError):
        return False
    return p == r or p.startswith(r + os.sep)


def _looks_like_path(value: Any) -> bool:
    """True when a settings value is plausibly a filesystem path.

    Keeps ``cell_mask_dim='4'`` (a real type error) from being mistaken for a
    path that some upstream job might create.
    """
    if not isinstance(value, str) or not value.strip():
        return False
    return (os.sep in value or value.startswith('~')
            or bool(os.path.splitext(value)[1]))


def resolve_job_settings(job: Job) -> Dict[str, Any]:
    """Build the settings dict this job's module will actually receive.

    Layered exactly as ``spacr-run`` layers them — module defaults, then the
    settings file (or the inline dict), then the ``--set`` overrides — using
    :mod:`spacr.cli`'s own loader and coercion so a value the CLI accepts is a
    value the queue accepts.

    :param job: the job.
    :returns: the resolved settings.
    :raises SettingsError: on an unknown module, an unreadable settings file,
        an unknown override key or an uncoercible override value.
    """
    module = resolve_module(job.module)
    if module is None:
        raise SettingsError(f'unknown module {job.module!r}.')
    resolved = module_defaults(module)
    if isinstance(job.settings, Mapping):
        resolved.update(dict(job.settings))
    elif isinstance(job.settings, str) and job.settings.strip():
        resolved.update(load_settings_file(job.settings))
    apply_overrides(resolved, job.override_args, module)
    return resolved


def _upstream_roots(job: Job, queue: Optional[Queue]) -> List[str]:
    """Folders an upstream job of ``job`` writes into.

    Used to tell "this input does not exist *yet*" from "this path is
    misspelled". Transitive, because Measure→Classify(CV)→Classify(ML) chains
    three deep and the folder is created by the job at the top.
    """
    if queue is None or not job.depends_on:
        return []
    roots: List[str] = []
    seen = set()
    frontier = list(job.depends_on)
    while frontier:
        dep_id = frontier.pop()
        if dep_id in seen:
            continue
        seen.add(dep_id)
        dep = queue.find(dep_id)
        if dep is None:
            continue
        frontier.extend(dep.depends_on)
        try:
            dep_settings = resolve_job_settings(dep)
        except SettingsError:
            continue
        roots.extend(_src_values(dep_settings))
    return roots


def _deferrable(problem: Any, settings: Mapping[str, Any],
                upstream_roots: Sequence[str]) -> bool:
    """True when a settings error is only "not there *yet*".

    The rule, and it is deliberately the only rule: a data error is deferred
    when the path it is about lies inside a folder an upstream job in this
    queue writes into. So a Measure job chained behind a Mask job on the same
    plate is addable before ``merged/`` exists, while job 9's misspelled
    ``/data/plaet9`` — which no upstream job produces — stays an error and is
    caught tonight rather than at 3 a.m.

    Type, name and range errors are never deferred: ``cell_mask_dim='4'`` is
    wrong now and will still be wrong in six hours.
    """
    if not upstream_roots:
        return False
    setting = getattr(problem, 'setting', '')
    if setting in ('', 'src'):
        targets = _src_values(settings)
    else:
        value = settings.get(setting)
        targets = [value] if _looks_like_path(value) else []
    if not targets:
        return False
    return all(any(_within(t, root) for root in upstream_roots) for t in targets)


def validate_job(job: Job, queue: Optional[Queue] = None) -> List[Problem]:
    """Check one job the way ``spacr-run --dry-run`` would, plus queue rules.

    Every check that can be made without running anything is made here, so it
    is made when the job is *added*:

    * the module resolves, and is not one of the GUI-only apps
      (:data:`spacr.cli.INTERACTIVE_ONLY`) that has no headless callable;
    * the settings file exists, parses, and the overrides name real settings
      with coercible values;
    * :func:`spacr.validate.validate_settings` agrees the settings are
      runnable against the data on disk — with data that an upstream job in
      this queue has not produced yet deferred to a warning (see
      :func:`_deferrable`);
    * dependencies exist and come earlier in the queue.

    :param job: the job to check.
    :param queue: the queue it belongs to, needed for the dependency rules.
    :returns: every problem found, errors and warnings mixed. Never raises.
    """
    problems: List[Problem] = []
    jid = job.id or job.default_label()

    module = resolve_module(job.module)
    if module is None:
        key = str(job.module).strip().lower().replace('-', '_')
        if key in INTERACTIVE_ONLY:
            problems.append(Problem(
                jid, ERROR,
                f"'{job.module}' is a GUI-only module and cannot run in a queue: "
                f'{INTERACTIVE_ONLY[key]}',
                'Remove this job, or replace it with a module that runs headless '
                "(spacr-run --list)."))
        else:
            problems.append(Problem(
                jid, ERROR, f'unknown module {job.module!r}.',
                "Use a module key from 'spacr-run --list'."))
        return problems + _dependency_problems(job, queue)

    if not isinstance(job.settings, (str, Mapping)):
        problems.append(Problem(
            jid, ERROR,
            f'settings is a {type(job.settings).__name__}; a job takes a settings '
            f'file path or an inline settings object.',
            'Point settings at the CSV the GUI wrote into <src>/settings/.'))
        return problems + _dependency_problems(job, queue)

    if isinstance(job.settings, str) and not job.settings.strip():
        problems.append(Problem(
            jid, WARNING,
            f"no settings file — job '{jid}' would run {module.key} on its "
            f'defaults alone.',
            'Give it the settings CSV for this plate, or an inline settings object.'))

    try:
        settings = resolve_job_settings(job)
    except SettingsError as exc:
        problems.append(Problem(
            jid, ERROR, str(exc).splitlines()[0],
            'Fix the settings file or the override, then add the job again.'))
        return problems + _dependency_problems(job, queue)

    roots = _upstream_roots(job, queue)
    for found in validate_settings(dict(settings), module.validate_key):
        deferred = found.is_error and _deferrable(found, settings, roots)
        severity = WARNING if deferred else found.severity
        message = str(found.message)
        if deferred:
            message = (f'{message} (deferred — an earlier job in this queue writes '
                       f'there; re-checked when this job starts)')
        problems.append(Problem(jid, severity, message, found.fix,
                                setting=getattr(found, 'setting', '')))

    return problems + _dependency_problems(job, queue)


def _dependency_problems(job: Job, queue: Optional[Queue]) -> List[Problem]:
    """Dependency rules: known, earlier, not itself."""
    if queue is None:
        return []
    problems: List[Problem] = []
    jid = job.id or job.default_label()
    here = queue.index(job.id)
    for dep_id in job.depends_on:
        if dep_id == job.id:
            problems.append(Problem(
                jid, ERROR, f'job {jid!r} depends on itself.',
                'Remove the self-reference from depends_on.'))
            continue
        dep = queue.find(dep_id)
        if dep is None:
            problems.append(Problem(
                jid, ERROR,
                f'depends_on names {dep_id!r}, which is not a job in this queue.',
                f'Use one of: {", ".join(queue.ids) or "(no other jobs)"}.'))
            continue
        there = queue.index(dep_id)
        if here >= 0 and there >= 0 and there > here:
            problems.append(Problem(
                jid, ERROR,
                f'depends_on names {dep_id!r}, which comes *later* in the queue — '
                f'this job would always be skipped.',
                f'Move {dep_id!r} above {jid!r}, or drop the dependency.'))
    return problems


def _cycle_problems(queue: Queue) -> List[Problem]:
    """Report dependency cycles, which no run order can satisfy."""
    problems: List[Problem] = []
    by_id = {job.id: job for job in queue.jobs}
    state: Dict[str, int] = {}

    def walk(job_id: str, trail: List[str]) -> None:
        if state.get(job_id) == 2:
            return
        if state.get(job_id) == 1:
            start = trail.index(job_id) if job_id in trail else 0
            cycle = trail[start:] + [job_id]
            problems.append(Problem(
                job_id, ERROR,
                'dependency cycle: ' + ' -> '.join(cycle),
                'Break the cycle — a queue runs top to bottom and cannot satisfy it.'))
            return
        state[job_id] = 1
        for dep in (by_id[job_id].depends_on if job_id in by_id else []):
            if dep in by_id:
                walk(dep, trail + [job_id])
        state[job_id] = 2

    for job in queue.jobs:
        walk(job.id, [])
    return problems


def validate_queue(queue: Queue) -> List[Problem]:
    """Validate every job, and the queue's own structure, all at once.

    Reporting the first problem and stopping is how a twelve-job queue takes
    twelve rounds of fixing, so this returns *everything*: duplicate ids,
    cycles, unknown dependencies, and each job's own settings problems.

    :param queue: the queue to check.
    :returns: every problem found. ``[p for p in problems if p.is_error]``
        being empty is exactly the condition :func:`run_queue` requires.
    """
    problems: List[Problem] = []
    if not queue.jobs:
        return [Problem('', WARNING, 'the queue is empty — nothing would run.',
                        'Add a job before running the queue.')]

    seen: Dict[str, int] = {}
    for job in queue.jobs:
        if not job.id:
            problems.append(Problem(
                '', ERROR, f'a {job.module} job has no id.',
                'Give every job a unique id; depends_on refers to jobs by id.'))
            continue
        seen[job.id] = seen.get(job.id, 0) + 1
    for job_id, n in seen.items():
        if n > 1:
            problems.append(Problem(
                job_id, ERROR, f'{n} jobs share the id {job_id!r}.',
                'Job ids must be unique — rename all but one.'))

    problems.extend(_cycle_problems(queue))
    for job in queue.jobs:
        problems.extend(validate_job(job, queue))
    return problems


def plan(queue: Queue, detail: bool = False) -> str:
    """Describe what the queue would do, without doing any of it.

    :param queue: the queue.
    :param detail: also render :func:`spacr.validate.describe_plan` for every
        job — the full "here is what would actually happen" per job. Off by
        default because it lists directories, which is slow over NFS.
    :returns: the plan as one string.
    """
    lines = [_RULE, f' spaCR batch queue: {queue.name} — {len(queue.jobs)} job(s)', _RULE,
             ' Jobs run one at a time, in this order. They compete for one GPU;',
             ' nothing here ever runs two of them at once.', '']
    width = max((len(job.id) for job in queue.jobs), default=2)
    for i, job in enumerate(queue.jobs, start=1):
        module = resolve_module(job.module)
        entry = (f'{module.module_name}.{module.func_name}()' if module
                 else f'?? unknown module {job.module!r}')
        deps = f'  after {", ".join(job.depends_on)}' if job.depends_on else ''
        lines.append(f' {i:>3}. {job.id.ljust(width)}  {job.label}{deps}')
        lines.append(f'      {entry}')
        src = ''
        try:
            src = ', '.join(_src_values(resolve_job_settings(job)))
        except SettingsError as exc:
            src = f'(settings unreadable: {str(exc).splitlines()[0]})'
        if src:
            lines.append(f'      src {src}')
        if job.override_args:
            lines.append(f'      --set {" --set ".join(job.override_args)}')
        if job.status != STATUS_PENDING:
            lines.append(f'      status {job.status}')
        if detail and module is not None:
            try:
                from .validate import describe_plan
                text = describe_plan(dict(resolve_job_settings(job)), module.validate_key)
                lines.extend('      ' + line for line in text.splitlines())
            except (SettingsError, OSError) as exc:
                lines.append(f'      (plan unavailable: {exc})')
        lines.append('')

    problems = validate_queue(queue)
    lines.append(format_problems(problems, title=' pre-flight'))
    lines.append(_RULE)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Progress:
    """One incremental progress report, handed to ``on_progress``.

    A queue runs for hours; a GUI that only learns the outcome at the end is
    not showing progress. Every transition emits one of these.

    :param event: ``'queue_started'``, ``'job_started'``, ``'job_finished'``,
        ``'job_skipped'``, ``'queue_stopped'`` or ``'queue_finished'``.
    :param job_id: the job this is about, or ``''`` for queue-level events.
    :param index: 1-based position of the job in the queue, ``0`` when N/A.
    :param total: number of jobs in the queue.
    :param status: the job's status at the moment of the event.
    :param message: one line fit to put in a status bar.
    """

    event: str
    job_id: str = ''
    index: int = 0
    total: int = 0
    status: str = ''
    message: str = ''


@dataclass
class QueueResult:
    """What happened, and the summary that is the actual deliverable.

    :param queue: the queue, with every job's final status on it.
    :param log_dir: folder holding the per-job logs.
    :param started: ISO-8601 UTC start of the whole queue.
    :param finished: ISO-8601 UTC end of the whole queue.
    :param stopped_reason: why the queue halted early, or ``''``.
    :param ledger: the queue's own :class:`spacr.errors.RunLedger` — one
        recorded item per job, which is what groups identical failures and
        what gets stamped next to the queue file.
    :param path: the queue file that was kept up to date, or ``''``.
    """

    queue: Queue
    log_dir: str = ''
    started: str = ''
    finished: str = ''
    stopped_reason: str = ''
    ledger: RunLedger = field(default_factory=lambda: RunLedger('queue'))
    path: str = ''

    # -- accessors ---------------------------------------------------------

    def jobs_with(self, status: str) -> List[Job]:
        """Every job that ended in ``status``."""
        return [job for job in self.queue.jobs if job.status == status]

    @property
    def succeeded(self) -> List[Job]:
        """Jobs that exited 0 — including the ones that are only partial."""
        return self.jobs_with(STATUS_SUCCESS)

    @property
    def failed(self) -> List[Job]:
        """Jobs that ran and failed."""
        return self.jobs_with(STATUS_FAILED)

    @property
    def skipped(self) -> List[Job]:
        """Jobs deliberately not run because an upstream job did not succeed."""
        return self.jobs_with(STATUS_SKIPPED)

    @property
    def not_run(self) -> List[Job]:
        """Jobs the queue halted before reaching."""
        return self.jobs_with(STATUS_NOT_RUN) + self.jobs_with(STATUS_PENDING)

    @property
    def partial(self) -> List[Job]:
        """Jobs that exited 0 but whose own ledger recorded failed items."""
        return [job for job in self.queue.jobs if job.is_partial]

    @property
    def ok(self) -> bool:
        """True when everything ran, nothing failed and nothing is partial."""
        return not (self.failed or self.skipped or self.not_run or self.partial)

    @property
    def duration_s(self) -> Optional[float]:
        """Wall-clock seconds the queue took."""
        start, end = _parse_iso(self.started), _parse_iso(self.finished)
        if start is None or end is None:
            return None
        return (end - start).total_seconds()

    # -- the deliverable ---------------------------------------------------

    def summary(self) -> str:
        """Render the end-of-queue report.

        What ran, what failed and why (identical failures grouped), what was
        skipped and because of which upstream job, what is only partial, and
        how long each took. This is the thing a user reads over coffee
        instead of scrolling four thousand lines of interleaved log.
        """
        counts = self.queue.counts()
        head = (f'{counts[STATUS_SUCCESS]} ok, {counts[STATUS_FAILED]} failed, '
                f'{counts[STATUS_SKIPPED]} skipped, '
                f'{counts[STATUS_NOT_RUN] + counts[STATUS_PENDING]} not run')
        lines = [_RULE,
                 f' spaCR batch queue — {self.queue.name}: {len(self.queue.jobs)} job(s), {head}',
                 _RULE,
                 f' started   {self.started or "—"}',
                 f' finished  {self.finished or "—"}   ({fmt_duration(self.duration_s)})']
        if self.log_dir:
            lines.append(f' logs      {self.log_dir}')
        if self.path:
            lines.append(f' queue     {self.path}')
        lines.append('')

        width = max((len(job.id) for job in self.queue.jobs), default=2)
        for i, job in enumerate(self.queue.jobs, start=1):
            mark = job.status
            if job.status == STATUS_SUCCESS and job.is_partial:
                mark = 'success (PARTIAL)'
            lines.append(f' {i:>3}. {job.id.ljust(width)}  {mark.ljust(18)} '
                         f'{fmt_duration(job.duration_s).rjust(8)}  {job.label}')
        lines.append('')

        failures = self.ledger.grouped_failures()
        if failures:
            lines.append(' Failures, grouped — identical failures are one problem, not many:')
            for exc_type, group in failures.items():
                lines.append(f'   {exc_type} x{len(group)}')
                by_message: "OrderedDict[str, List[Any]]" = OrderedDict()
                for failure in group:
                    by_message.setdefault(failure.message, []).append(failure)
                for message, same in by_message.items():
                    suffix = f'  (x{len(same)}: {", ".join(f.item for f in same)})' \
                        if len(same) > 1 else f'  ({same[0].item})'
                    lines.append(f'     {message}{suffix}')
            lines.append('')
            for job in self.failed:
                if job.log_path:
                    lines.append(f'   {job.id}: full log {job.log_path}')
            lines.append('')

        if self.skipped:
            lines.append(' Skipped — NOT a success and NOT a failure; these never ran:')
            for job in self.skipped:
                lines.append(f'   {job.id} ({job.label}) — {job.error}')
            lines.append('')

        if self.partial:
            lines.append(' PARTIAL results — these exited 0 but their own run_status '
                         'says items failed:')
            for job in self.partial:
                status = job.run_status or {}
                lines.append(
                    f'   {job.id} ({job.label}) — {status.get("n_succeeded", "?")} of '
                    f'{status.get("n_attempted", "?")} items; '
                    f'{status.get("n_failed", "?")} failed. Artifacts: '
                    f'{", ".join(status.get("artifacts") or []) or "see log"}')
            lines.append(' Anything computed downstream from those artifacts covers a subset.')
            lines.append('')

        not_run = self.not_run
        if not_run:
            lines.append(f' Not run — the queue halted before reaching {len(not_run)} job(s): '
                         f'{", ".join(job.id for job in not_run)}')
            lines.append('')

        if self.stopped_reason:
            lines.append(f' STOPPED: {self.stopped_reason}')
            lines.append('')

        if self.ok:
            lines.append(' Every job completed and every artifact is stamped complete.')
        lines.append(_RULE)
        return '\n'.join(lines)


# -- the runners ------------------------------------------------------------


def job_command(job: Job, settings_path: str,
                python: Optional[str] = None) -> List[str]:
    """Return the exact ``spacr-run`` command line for ``job``.

    Spelled as ``<this python> -m spacr.cli ...`` rather than the ``spacr-run``
    console script so the job runs in the interpreter the queue is running in
    — the venv the user actually installed spaCR into — even when ``PATH``
    says otherwise.

    :param job: the job.
    :param settings_path: settings file the job will be given.
    :param python: interpreter to use; ``sys.executable`` by default.
    :returns: the argv list.
    """
    module = resolve_module(job.module)
    key = module.key if module is not None else str(job.module)
    cmd = [python or sys.executable, '-m', 'spacr.cli', key]
    if settings_path:
        cmd += ['--settings', settings_path]
    for item in job.override_args:
        cmd += ['--set', item]
    return cmd


def subprocess_runner(job: Job, settings_path: str, log_path: str) -> int:
    """Run one job as its own ``spacr-run`` process. **The default runner.**

    A separate process is the point: cellpose segfaulting or the CUDA driver
    wedging kills that job, not the other eleven and not the GUI. The exit
    code comes straight from :mod:`spacr.cli` — 0 ok, 1 the module raised,
    2 bad settings — and stdout+stderr go to this job's own log file.

    :param job: the job to run.
    :param settings_path: settings file to pass to ``--settings``.
    :param log_path: file to write this job's output to.
    :returns: the process exit code.
    """
    cmd = job_command(job, settings_path)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8', errors='replace') as handle:
        handle.write(f'# spaCR queue job {job.id} ({job.label})\n')
        handle.write(f'# started {_now_iso()}\n')
        handle.write(f'# {" ".join(cmd)}\n\n')
        handle.flush()
        try:
            completed = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                       check=False)
            code = int(completed.returncode)
        except OSError as exc:
            handle.write(f'\n# could not start the job process: {exc}\n')
            return EXIT_USAGE
        handle.write(f'\n# finished {_now_iso()} with exit code {code}\n')
    return code


def inprocess_runner(job: Job, settings_path: str, log_path: str) -> int:
    """Run one job in this interpreter, with its output tee'd to its log.

    Same argv, same exit-code contract as :func:`subprocess_runner` — it calls
    :func:`spacr.cli.main` directly — but with none of the isolation: a
    segfault here takes the queue with it. Use it only where spawning is not
    possible (a frozen single-file build), and never from the GUI.

    :returns: the exit code :func:`spacr.cli.main` returned.
    """
    import contextlib

    from .cli import main as cli_main

    argv = job_command(job, settings_path)[3:]  # drop python -m spacr.cli
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8', errors='replace') as handle:
        handle.write(f'# spaCR queue job {job.id} ({job.label}) — in-process\n')
        handle.write(f'# spacr-run {" ".join(argv)}\n\n')
        handle.flush()
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            try:
                code = int(cli_main(argv))
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
    return code


# -- run_status collection --------------------------------------------------


def _status_artifacts(settings: Mapping[str, Any]) -> List[Path]:
    """Paths under ``src`` that a spaCR run may have stamped.

    Bounded on purpose: the top level of ``src`` and its ``measurements``
    folder. Walking a plate folder with 100 000 PNGs to find a sidecar would
    cost more than the job.
    """
    out: List[Path] = []
    for src in _src_values(settings):
        root = Path(src)
        for folder in (root, root / 'measurements'):
            if not folder.is_dir():
                continue
            for entry in sorted(folder.iterdir()):
                if entry.suffix.lower() in DB_SUFFIXES:
                    out.append(entry)
                elif entry.name.endswith(RUN_STATUS_SUFFIX):
                    out.append(entry)
    return out


def _status_snapshot(settings: Mapping[str, Any]) -> Dict[str, int]:
    """How many stamps each artifact already had, before this job ran.

    A ``measurements.db`` accumulates one ``run_status`` row per stage, so
    "what did *this* job record" is the difference, not the total.
    """
    snapshot: Dict[str, int] = {}
    for artifact in _status_artifacts(settings):
        try:
            snapshot[str(artifact)] = len(read_run_status(artifact))
        except Exception:  # a locked or corrupt artifact must not stop the queue
            snapshot[str(artifact)] = 0
    return snapshot


def _collect_run_status(settings: Mapping[str, Any],
                        before: Mapping[str, int]) -> Optional[Dict[str, Any]]:
    """Read back the stamps this job added, and summarise them.

    :param settings: the job's resolved settings, for locating its artifacts.
    :param before: :func:`_status_snapshot` taken before the job ran.
    :returns: ``{'status', 'n_attempted', 'n_succeeded', 'n_failed',
        'artifacts', 'records'}``, or None when the job stamped nothing (which
        means "no information", not "clean" — see
        :func:`spacr.errors.run_is_complete`).
    """
    attempted = succeeded = failed = 0
    artifacts: List[str] = []
    records = 0
    for artifact in _status_artifacts(settings):
        key = str(artifact)
        try:
            stamps = read_run_status(artifact)
        except Exception:
            continue
        fresh = stamps[int(before.get(key, 0)):]
        if not fresh:
            continue
        artifacts.append(key)
        for stamp in fresh:
            records += 1
            attempted += int(stamp.get('n_attempted', 0) or 0)
            succeeded += int(stamp.get('n_succeeded', 0) or 0)
            failed += int(stamp.get('n_failed', 0) or 0)
    if not records:
        return None
    if failed:
        status = 'partial'
    elif attempted:
        status = 'complete'
    else:
        status = 'empty'
    return {
        'status': status,
        'n_attempted': attempted,
        'n_succeeded': succeeded,
        'n_failed': failed,
        'records': records,
        'artifacts': artifacts,
    }


# -- failure classification -------------------------------------------------

_EXC_LINE = re.compile(r'^(?P<type>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Interrupt))'
                       r'(?::\s*(?P<msg>.*))?$')

_FAILURE_TYPES: Dict[str, type] = {}


def _failure_class(name: str) -> type:
    """Return (and cache) an exception class called ``name``.

    :class:`spacr.errors.RunLedger` groups failures by ``type(exc).__name__``.
    A queue's failures arrive as an exit code and a log tail rather than as a
    live exception, so the kind is reconstituted into a real class here and
    the ledger's grouping — and its whole summary shape — is reused as is.
    """
    key = re.sub(r'[^A-Za-z0-9_]', '', name) or 'JobFailure'
    if key not in _FAILURE_TYPES:
        _FAILURE_TYPES[key] = type(key, (SpacrError,), {})
    return _FAILURE_TYPES[key]


def _log_tail(log_path: str, max_lines: int = 40) -> List[str]:
    """Last few non-empty lines of a job log, for the failure message."""
    try:
        text = Path(log_path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return []
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-max_lines:]


def classify_failure(exit_code: int, log_path: str) -> Tuple[str, str]:
    """Turn an exit code plus a log into ``(kind, one-line message)``.

    The kind is what groups identical failures: twelve jobs that all died on
    the same missing share are one problem worth one line, not twelve.

    :param exit_code: the runner's exit code.
    :param log_path: the job's log file.
    :returns: ``(kind, message)``.
    """
    tail = _log_tail(log_path)
    for line in reversed(tail):
        match = _EXC_LINE.match(line.strip())
        if match:
            kind = match.group('type').rsplit('.', 1)[-1]
            return kind, (match.group('msg') or kind).strip()
        stripped = line.strip()
        if stripped.startswith('error: '):
            return ('ConfigurationError' if exit_code == EXIT_USAGE else 'JobFailure',
                    stripped[len('error: '):])
    if exit_code == EXIT_USAGE:
        return 'ConfigurationError', 'bad settings or arguments (exit code 2)'
    last = tail[-1] if tail else ''
    return 'JobFailure', (last or f'exited with code {exit_code}')


# -- the loop ---------------------------------------------------------------


def _default_log_dir(queue: Queue, path: Optional[Union[str, os.PathLike]]) -> Path:
    """Where per-job logs go when the caller does not say.

    Next to the queue file, so the record and the evidence travel together.
    With no queue file there is nowhere obvious, so a temp folder is used and
    its path is reported in the summary.
    """
    if path:
        target = Path(path)
        return target.parent / f'{target.name.split(".")[0]}_logs'
    return Path(tempfile.mkdtemp(prefix='spacr_queue_'))


def _safe(fn: Optional[Callable[..., Any]], *args: Any) -> None:
    """Call a user callback without letting it take the queue down with it."""
    if fn is None:
        return
    try:
        fn(*args)
    except Exception:  # a GUI callback must never kill an overnight run
        LOG.exception('batch: progress callback raised; the queue continues')


def _materialize_settings(job: Job, settings: Mapping[str, Any],
                          log_dir: Path) -> str:
    """Return a settings file path for ``job``, writing one when it has none.

    An inline settings dict is written next to the job's log as
    ``<id>.settings.json`` — which doubles as provenance: exactly what this
    job was given, beside exactly what it printed.
    """
    if isinstance(job.settings, str) and job.settings.strip():
        return job.settings
    path = log_dir / f'{job.id}.settings.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(settings), indent=2, default=str), encoding='utf-8')
    return str(path)


def run_queue(queue: Queue,
              path: Optional[Union[str, os.PathLike]] = None,
              on_error: str = 'continue',
              log_dir: Optional[Union[str, os.PathLike]] = None,
              max_consecutive_failures: Optional[int] = 3,
              on_progress: Optional[Callable[[Progress], None]] = None,
              runner: Optional[Callable[[Job, str, str], int]] = None,
              stop_flag: Optional[Callable[[], bool]] = None,
              force: bool = False,
              echo: bool = True) -> QueueResult:
    """Run the queue, one job at a time, and report on all of it.

    Jobs run **sequentially and only sequentially** — they compete for one
    GPU. The queue is validated in full before the first job starts, its state
    is written to ``path`` after every transition, and a job whose dependency
    did not succeed is skipped rather than run against a half-written input.

    :param queue: the queue to run. Mutated in place: each job's status,
        timestamps, exit code, log path and run_status are filled in.
    :param path: queue file kept up to date after every transition. Without
        it the run cannot be resumed, which is a real loss on a long night.
    :param on_error: ``'continue'`` (default) runs the jobs that do not depend
        on the failed one; ``'stop'`` halts the queue immediately.
    :param log_dir: folder for per-job logs; defaults to ``<queue file>_logs``
        or a temp folder.
    :param max_consecutive_failures: halt after this many failures in a row.
        Continue-on-error is meant to save a night from one bad plate, not to
        spend it repeating one systematic mistake twelve times. ``None`` or
        ``0`` disables the check.
    :param on_progress: called with a :class:`Progress` on every transition,
        so a GUI can show the queue moving. Exceptions from it are logged and
        swallowed.
    :param runner: ``(job, settings_path, log_path) -> exit_code``;
        :func:`subprocess_runner` by default. Injected by the tests, and by
        anyone who wants to submit jobs to a scheduler instead.
    :param stop_flag: polled between jobs; return True to stop the queue after
        the running job finishes (the GUI's Stop button).
    :param force: run even though validation found errors. The escape hatch
        for a check that is wrong about your data — it is not the default for
        a reason.
    :param echo: print the summary at the end. Always logged either way.
    :returns: the :class:`QueueResult`.
    :raises QueueError: when validation found errors and ``force`` is False —
        with *every* problem in the message, not just the first.
    """
    if on_error not in ON_ERROR_CHOICES:
        raise ValueError(
            f'on_error must be one of {ON_ERROR_CHOICES}, got {on_error!r}.')

    problems = validate_queue(queue)
    errors = [p for p in problems if p.is_error]
    if errors and not force:
        raise QueueError(
            f'refusing to start: {len(errors)} job(s) in this queue cannot run.\n'
            f'Everything wrong with the queue is listed here so it can be fixed in '
            f'one pass, rather than discovered one job at a time overnight.\n\n'
            + format_problems(problems, title='queue check'))
    if errors:
        LOG.warning('batch: starting with %d validation error(s) — force=True', len(errors))

    log_root = Path(log_dir) if log_dir else _default_log_dir(queue, path)
    log_root.mkdir(parents=True, exist_ok=True)

    ledger = RunLedger(f'queue:{queue.name}', logger=LOG)
    result = QueueResult(queue=queue, log_dir=str(log_root), started=_now_iso(),
                         ledger=ledger, path=str(path) if path else '')

    def persist() -> None:
        """Write the queue after a transition. Never fatal."""
        if not path:
            return
        try:
            save_queue(queue, path)
        except Exception as exc:  # a persistence problem must not end the night
            LOG.error('batch: could not persist queue state to %s: %s — the run '
                      'continues but could not be resumed from here', path, exc)

    total = len(queue.jobs)
    runner = runner or subprocess_runner
    consecutive = 0
    stopped = ''

    _safe(on_progress, Progress('queue_started', '', 0, total, '',
                                f'{queue.name}: {total} job(s), one at a time'))
    LOG.info('batch: queue %r starting — %d job(s), logs in %s',
             queue.name, total, log_root)
    persist()

    for i, job in enumerate(queue.jobs, start=1):
        if stopped:
            if job.status in RESUMABLE_STATUSES:
                job.status = STATUS_NOT_RUN
            continue
        if job.status in (STATUS_SUCCESS, STATUS_FAILED, STATUS_SKIPPED):
            continue  # a resumed queue: already settled

        blocker = _blocking_dependency(job, queue)
        if blocker is not None:
            job.status = STATUS_SKIPPED
            job.error = (f'skipped: it depends on {blocker.id} ({blocker.label}), '
                         f'which is {blocker.status}. Running it anyway would '
                         f'produce a result computed from a missing or partial input.')
            job.finished = _now_iso()
            LOG.error('batch: SKIPPED %s — %s', job.id, job.error)
            persist()
            _safe(on_progress, Progress('job_skipped', job.id, i, total,
                                        job.status, job.error))
            continue

        try:
            settings = resolve_job_settings(job)
        except SettingsError as exc:
            job.status = STATUS_FAILED
            job.exit_code = EXIT_USAGE
            job.error = str(exc).splitlines()[0]
            job.started = job.finished = _now_iso()
            ledger.record_failure(job.id, stage=job.module,
                                  exc=_failure_class('ConfigurationError')(job.error))
            consecutive += 1
            persist()
            _safe(on_progress, Progress('job_finished', job.id, i, total,
                                        job.status, job.error))
            if on_error == 'stop':
                stopped = (f'job {job.id} failed and on_error="stop", so the '
                           f'remaining jobs were not run: {job.error}')
            elif _too_many(consecutive, max_consecutive_failures):
                stopped = _systematic_message(consecutive, queue)
            continue

        settings_path = _materialize_settings(job, settings, log_root)
        log_path = str(log_root / f'{i:02d}_{job.id}.log')
        job.log_path = log_path
        job.status = STATUS_RUNNING
        job.started = _now_iso()
        job.finished = ''
        job.exit_code = None
        job.error = ''
        job.run_status = None
        persist()
        LOG.info('batch: [%d/%d] %s — %s', i, total, job.id, job.label)
        _safe(on_progress, Progress('job_started', job.id, i, total, job.status,
                                    f'[{i}/{total}] {job.label}'))

        before = _status_snapshot(settings)
        interrupted = False
        try:
            code = int(runner(job, settings_path, log_path))
        except KeyboardInterrupt:
            code = 1
            interrupted = True
        except Exception as exc:  # the runner itself broke; that is a job failure
            code = 1
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'a', encoding='utf-8') as handle:
                handle.write(f'\n# the queue runner raised: {type(exc).__name__}: {exc}\n')

        job.exit_code = code
        job.finished = _now_iso()
        job.run_status = _collect_run_status(settings, before)

        if code == EXIT_OK:
            job.status = STATUS_SUCCESS
            ledger.record_success(job.id, stage=job.module)
            consecutive = 0
            if job.is_partial:
                status = job.run_status or {}
                job.error = (f'exited 0 but its run_status says {status.get("n_failed")} '
                             f'of {status.get("n_attempted")} items failed — the '
                             f'artifacts cover a subset')
                LOG.error('batch: %s is PARTIAL — %s', job.id, job.error)
            LOG.info('batch: %s finished in %s', job.id, fmt_duration(job.duration_s))
        else:
            kind, message = classify_failure(code, log_path)
            if interrupted:
                kind, message = 'KeyboardInterrupt', 'interrupted by the user'
            job.status = STATUS_FAILED
            job.error = f'{message} (exit code {code}; log {log_path})'
            ledger.record_failure(job.id, stage=job.module,
                                  exc=_failure_class(kind)(message))
            consecutive += 1

        persist()
        _safe(on_progress, Progress('job_finished', job.id, i, total, job.status,
                                    job.error or f'{job.label} finished'))

        if interrupted:
            stopped = 'interrupted by the user (Ctrl-C) — the remaining jobs were not run.'
            continue
        if job.status == STATUS_FAILED and on_error == 'stop':
            stopped = (f'job {job.id} failed and on_error="stop", so the remaining '
                       f'jobs were not run: {job.error}')
            continue
        if job.status == STATUS_FAILED and _too_many(consecutive, max_consecutive_failures):
            stopped = _systematic_message(consecutive, queue)
            continue
        if stop_flag is not None and stop_flag():
            stopped = f'stopped by request after job {job.id}.'
            continue

    for job in queue.jobs:
        if job.status in (STATUS_PENDING, STATUS_RUNNING) and stopped:
            job.status = STATUS_NOT_RUN

    result.finished = _now_iso()
    result.stopped_reason = stopped
    persist()

    if stopped:
        _safe(on_progress, Progress('queue_stopped', '', total, total, '', stopped))
        LOG.error('batch: queue stopped — %s', stopped)
    text = result.summary()
    if path:
        try:
            ledger.stamp(path)
        except Exception as exc:
            LOG.error('batch: could not stamp the queue file: %s', exc)
    if echo:
        print(text)
    LOG.info('batch: queue %r finished — %s ok, %s failed, %s skipped, %s not run',
             queue.name, len(result.succeeded), len(result.failed),
             len(result.skipped), len(result.not_run))
    _safe(on_progress, Progress('queue_finished', '', total, total, '',
                                f'{len(result.succeeded)} ok, {len(result.failed)} failed, '
                                f'{len(result.skipped)} skipped'))
    return result


def _too_many(consecutive: int, threshold: Optional[int]) -> bool:
    """True when the consecutive-failure threshold has been reached."""
    if not threshold or int(threshold) <= 0:
        return False
    return consecutive >= int(threshold)


def _systematic_message(consecutive: int, queue: Queue) -> str:
    """Explain a stop-on-systematic-failure, which is a kindness, not a giving-up."""
    return (f'{consecutive} jobs failed in a row. That is a systematic problem — a '
            f'missing share, a broken environment, a wrong path root — not {consecutive} '
            f'unrelated accidents, and the remaining jobs would fail the same way. '
            f'The queue stopped so the night is spent fixing it rather than repeating '
            f'it. Raise max_consecutive_failures (or set it to 0) to run on regardless.')


def _blocking_dependency(job: Job, queue: Queue) -> Optional[Job]:
    """Return the first dependency that did not succeed, or None.

    A dependency that is missing, failed, was itself skipped, or never ran all
    block: in each case the input this job needs was not produced.
    """
    for dep_id in job.depends_on:
        dep = queue.find(dep_id)
        if dep is None:
            continue  # validate_queue already reported this as an error
        if dep.status != STATUS_SUCCESS:
            return dep
    return None


def resume_queue(path: Union[str, os.PathLike],
                 retry_failed: bool = False,
                 **kwargs: Any) -> QueueResult:
    """Pick a queue up where a crash, a reboot or a Stop left it.

    Jobs that already succeeded are left alone. Jobs that were ``not_run``
    (the queue halted before reaching them) or still ``pending`` are run. A
    job that was ``running`` when the machine went down is reset and run
    again: its artifacts are half-written, and half a mask run is not a
    result. Failed and skipped jobs stay as they are unless ``retry_failed``
    is set.

    :param path: the queue file :func:`run_queue` was persisting to.
    :param retry_failed: also re-run jobs that failed, and un-skip the jobs
        that were skipped because of them.
    :param kwargs: forwarded to :func:`run_queue`.
    :returns: the :class:`QueueResult` for the resumed run.
    """
    queue = load_queue(path)
    for job in queue.jobs:
        if job.status == STATUS_RUNNING:
            LOG.warning('batch: %s was still running when the queue stopped; its '
                        'output is half-written, so it will run again', job.id)
            job.reset()
        elif job.status == STATUS_NOT_RUN:
            job.status = STATUS_PENDING
        elif retry_failed and job.status in (STATUS_FAILED, STATUS_SKIPPED):
            job.reset()
    kwargs.setdefault('path', path)
    return run_queue(queue, **kwargs)
