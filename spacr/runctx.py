"""The run context: one id, one seed, one error policy, for a whole run.

Three things every spaCR pipeline needs and none of them had:

**One run id, on every log line and every output.** A run used to be
untraceable. The log said a field failed; the registry said an artifact
existed; nothing connected them, so "show me everything from the run that
produced this file" had no answer. :func:`run_context` mints one id, stamps
it onto every :class:`logging.LogRecord` created anywhere in the process,
writes a per-run JSONL log that :func:`read_run_log` reads back, and hands
the *same* id to :func:`spacr.artifacts.register_run_outputs`. A log line
and an output can therefore be joined on ``run_id``, which is the whole
point.

**One seed that reaches everything.** ``random_seed`` used to be read by
:mod:`spacr.deep_spacr` and :mod:`spacr.sim` and nowhere else, so a
"reproducible" run still shuffled its fields differently, split its folds
differently and initialised Cellpose differently every time.
:func:`seed_everything` seeds Python, NumPy (legacy global *and* the
:class:`~numpy.random.Generator` stream :func:`spacr_rng` hands out), Torch
on CPU and every CUDA device, and — through those two — Cellpose. sklearn
has no global seed at all; :func:`random_state` is what estimator
construction sites pass. What cannot be made deterministic is listed in
:attr:`SeedReport.caveats` and in :func:`seed_everything`'s docstring
rather than papered over.

**One error policy, honoured at every batch boundary.** ``on_error`` is a
tri-state, default ``"stop"``:

``stop``
    the first failed unit aborts the run. The default, because a pipeline
    that quietly drops a third of its plates produces a number that looks
    exactly like a good one.
``skip``
    the unit is recorded — on the :class:`~spacr.errors.RunLedger` *and* as
    a :class:`SkipRecord` naming the unit, the stage and why — and the run
    carries on. Never a silent drop: :attr:`ErrorPolicy.skips` is what was
    lost.
``retry``
    the unit is attempted :attr:`ErrorPolicy.attempts` times with an
    exponential backoff, and if the budget runs out it behaves exactly like
    ``stop``.

Usage
-----
.. code-block:: python

    from spacr.runctx import run_context

    with run_context("mask", settings) as run:
        for plate in plates:
            for attempt in run.policy.attempts_for(plate, stage="plate"):
                with attempt:
                    process(plate)
        run.register_outputs(roots=plates)

Public API
----------
``run_context``, ``RunContext``, ``current_run_context``, ``current_run_id``
    The run itself, and the ambient lookup a worker or a library call uses.
``new_run_id``, ``install_run_id_logging``, ``uninstall_run_id_logging``, ``RunIdFilter``, ``runs_log_dir``, ``run_log_path``, ``read_run_log``
    The S7 machinery: minting, stamping and querying by run id.
``seed_everything``, ``SeedReport``, ``resolve_seed``, ``random_state``, ``spacr_rng``, ``torch_generator``, ``seed_worker``, ``DEFAULT_SEED``
    The S5 machinery: one call that seeds them all, plus the per-library
    handles for the places a global seed cannot reach.
``ErrorPolicy``, ``resolve_error_policy``, ``SkipRecord``, ``SKIPPED``, ``ON_ERROR_STOP``, ``ON_ERROR_SKIP``, ``ON_ERROR_RETRY``, ``ON_ERROR_MODES``
    The S9 machinery.
``apply_defaults``, ``RUN_SETTING_KEYS``
    The settings seam: the keys this module owns, applied to any settings
    dict.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import logging
import os
import random as _random
import sys
import threading
import time
import traceback as _traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (Any, Callable, Dict, Iterator, List, Mapping, Optional,
                    Tuple, Union)

import numpy as np

from .cancellation import PipelineCancelled
from .errors import ConfigurationError, RunLedger

__all__ = [
    "DEFAULT_ON_ERROR",
    "DEFAULT_RETRIES",
    "DEFAULT_BACKOFF",
    "DEFAULT_SEED",
    "ErrorPolicy",
    "ON_ERROR_MODES",
    "ON_ERROR_RETRY",
    "ON_ERROR_SKIP",
    "ON_ERROR_STOP",
    "RUN_ID_ENV",
    "RUN_SETTING_KEYS",
    "RunContext",
    "RunIdFilter",
    "SKIPPED",
    "SEED_ENV",
    "SeedReport",
    "SkipRecord",
    "apply_defaults",
    "current_run_context",
    "current_run_id",
    "install_run_id_logging",
    "new_run_id",
    "random_state",
    "read_run_log",
    "resolve_error_policy",
    "resolve_seed",
    "run_context",
    "run_log_path",
    "runs_log_dir",
    "seed_everything",
    "seed_worker",
    "spacr_rng",
    "torch_generator",
    "uninstall_run_id_logging",
]

LOG = logging.getLogger("spacr.runctx")

#: Environment variable carrying the active run id into child processes.
#: :mod:`multiprocessing` workers started with ``spawn`` or ``forkserver``
#: get a fresh interpreter and therefore a fresh (empty) context variable,
#: so the id travels in the environment as well — otherwise every Measure
#: worker would log under a different id than the run that started it.
RUN_ID_ENV = "SPACR_RUN_ID"

#: Environment override for the run seed, read when settings carry none.
SEED_ENV = "SPACR_SEED"

#: The seed a run uses when its settings do not name one. Matches the
#: value :mod:`spacr.deep_spacr` has always defaulted ``random_seed`` to,
#: so turning seeding on globally does not move that module's numbers.
DEFAULT_SEED = 42

ON_ERROR_STOP = "stop"
ON_ERROR_SKIP = "skip"
ON_ERROR_RETRY = "retry"

#: The tri-state, in the order the GUI should offer it.
ON_ERROR_MODES: Tuple[str, ...] = (ON_ERROR_STOP, ON_ERROR_SKIP,
                                   ON_ERROR_RETRY)

#: Stop, not skip. A run that drops work is a run whose numbers are wrong
#: in a way nothing downstream can detect, so the tolerant modes are the
#: ones you have to ask for.
DEFAULT_ON_ERROR = ON_ERROR_STOP

#: Attempts ``retry`` makes in total, including the first one.
DEFAULT_RETRIES = 3

#: Seconds before the second attempt; doubled before each one after that.
DEFAULT_BACKOFF = 1.0

#: Ceiling on the backoff, so attempt 10 does not sleep for eight minutes.
MAX_BACKOFF = 60.0

#: The settings keys this module owns.
RUN_SETTING_KEYS: Tuple[str, ...] = ("random_seed", "on_error",
                                     "on_error_attempts", "on_error_backoff")

#: Exceptions that are never skipped and never retried, whatever the policy
#: says. A wrong ``src`` is wrong for every unit, so "skip and continue"
#: would turn one mistake into N recorded data errors; Ctrl-C and a
#: cancelled pipeline are operator intent, not a flaky field. This is the
#: same list :meth:`spacr.errors.RunLedger.item` re-raises.
FATAL_EXCEPTIONS: Tuple[type, ...] = (ConfigurationError, PipelineCancelled,
                                      KeyboardInterrupt, SystemExit)

#: "the caller did not pass this argument", distinct from an explicit
#: ``None`` — which for a seed means "do not seed at all".
_UNSET = object()

_ACTIVE: "contextvars.ContextVar[Optional[RunContext]]" = contextvars.ContextVar(
    "spacr_run_context", default=None)

_LOGGING_LOCK = threading.RLock()
_BASE_RECORD_FACTORY: Optional[Callable[..., logging.LogRecord]] = None
_OUR_RECORD_FACTORY: Optional[Callable[..., logging.LogRecord]] = None


def _utcnow() -> str:
    """Return the current UTC instant as ISO-8601, seconds resolution."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# S7 — the run id, and getting it onto every log line
# ---------------------------------------------------------------------------

def new_run_id() -> str:
    """Mint a fresh run id.

    Twelve hex characters — deliberately the same shape as
    :attr:`spacr.errors.RunLedger.run_id`, so ledger stamps, ``run_status``
    rows and :class:`spacr.artifacts.Artifact` rows all join on one column
    of one format.
    """
    return uuid.uuid4().hex[:12]


def current_run_context() -> Optional["RunContext"]:
    """Return the :class:`RunContext` of the innermost active run, or None.

    Context-local, so two runs on two threads do not see each other's.
    """
    return _ACTIVE.get()


def current_run_id() -> str:
    """Return the active run id, or ``""`` when no run is open.

    Falls back to :data:`RUN_ID_ENV`, which is how a ``spawn``-ed Measure
    worker — a fresh interpreter with an empty context variable — still
    logs under the run that started it.
    """
    context = _ACTIVE.get()
    if context is not None:
        return context.run_id
    return os.environ.get(RUN_ID_ENV, "").strip()


class RunIdFilter(logging.Filter):
    """Give every record a ``run_id`` attribute, so a formatter can use it.

    Belt and braces for the record factory installed by
    :func:`install_run_id_logging`: a handler whose format string contains
    ``%(run_id)s`` must never raise on a record that came from somewhere
    the factory did not reach (a record unpickled from a worker, say).

    :param run_id: stamp this id instead of the ambient one. Used by the
        per-run log so a nested run's records are not misattributed.
    """

    def __init__(self, run_id: Optional[str] = None) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Set ``record.run_id`` when it is missing. Never drops a record."""
        if not getattr(record, "run_id", ""):
            record.run_id = self.run_id or current_run_id() or "-"
        return True


def install_run_id_logging() -> None:
    """Stamp ``run_id`` onto every :class:`logging.LogRecord` in this process.

    Done with :func:`logging.setLogRecordFactory` rather than a filter on
    the root logger, because a filter attached to a *logger* is consulted
    only for records logged through that logger — records propagating up
    from ``spacr.measure`` never see it, which is every record that
    matters. The factory sees them all, whichever handler is attached and
    whenever it was attached.

    Idempotent, and chains onto whatever factory is already installed
    instead of replacing it.
    """
    global _BASE_RECORD_FACTORY, _OUR_RECORD_FACTORY
    with _LOGGING_LOCK:
        current = logging.getLogRecordFactory()
        if current is _OUR_RECORD_FACTORY:
            return
        base = current
        _BASE_RECORD_FACTORY = base

        def _factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            """Create a record and stamp the ambient run id onto it."""
            record = base(*args, **kwargs)
            if not getattr(record, "run_id", ""):
                record.run_id = current_run_id() or "-"
            return record

        _OUR_RECORD_FACTORY = _factory
        logging.setLogRecordFactory(_factory)
        for handler in logging.getLogger().handlers:
            _ensure_filter(handler)


def _ensure_filter(handler: logging.Handler) -> None:
    """Attach one :class:`RunIdFilter` to ``handler``, at most once."""
    if not any(isinstance(f, RunIdFilter) for f in handler.filters):
        handler.addFilter(RunIdFilter())


def uninstall_run_id_logging() -> None:
    """Restore the record factory that was installed before us.

    Only undoes our own installation: if something else replaced the
    factory in the meantime, ripping ours out would discard theirs too, so
    this leaves the chain alone.
    """
    global _BASE_RECORD_FACTORY, _OUR_RECORD_FACTORY
    with _LOGGING_LOCK:
        if _OUR_RECORD_FACTORY is None:
            return
        if logging.getLogRecordFactory() is _OUR_RECORD_FACTORY:
            logging.setLogRecordFactory(
                _BASE_RECORD_FACTORY or logging.LogRecord)
        _OUR_RECORD_FACTORY = None
        _BASE_RECORD_FACTORY = None


def runs_log_dir() -> str:
    """Return the folder holding per-run logs, creating it when needed.

    ``<log dir>/runs``, where the log dir honours ``SPACR_LOG_DIR`` — see
    :func:`spacr.logging_util.log_dir`. Pointing that variable at a scratch
    folder is how a test gets a private set of run logs.
    """
    from .logging_util import log_dir
    root = os.path.join(str(log_dir()), "runs")
    os.makedirs(root, exist_ok=True)
    return root


def run_log_path(run_id: str) -> str:
    """Return the JSONL log path for ``run_id``. It need not exist yet."""
    return os.path.join(runs_log_dir(), f"{str(run_id).strip()}.jsonl")


def read_run_log(run_id: str,
                 *,
                 level: Optional[Union[int, str]] = None,
                 logger: Optional[str] = None,
                 contains: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return every log line this run emitted — "show me everything from run X".

    The query side of S7. Each record is a dict with ``run_id``, ``utc``,
    ``level``, ``logger``, ``message``, ``file``, ``line``, ``process`` and
    ``thread``; a record carrying an exception also has ``traceback``.

    :param run_id: the run to read.
    :param level: minimum level, as a number or a name (``"WARNING"``).
    :param logger: only records from this logger or its children.
    :param contains: only records whose message contains this substring.
    :returns: the matching records in the order they were written. An empty
        list when the run wrote no log — never an exception, because
        "nothing was logged" is an ordinary answer.
    """
    path = run_log_path(run_id)
    if not os.path.isfile(path):
        return []
    threshold = _level_number(level)
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                # A run killed mid-write leaves a half line. Everything
                # before it is still perfectly good evidence.
                continue
            if threshold is not None and int(record.get("levelno", 0)) < threshold:
                continue
            if logger and not str(record.get("logger", "")).startswith(logger):
                continue
            if contains and contains not in str(record.get("message", "")):
                continue
            records.append(record)
    return records


def _level_number(level: Optional[Union[int, str]]) -> Optional[int]:
    """Coerce a level name or number to a number; None stays None."""
    if level is None:
        return None
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(str(level).upper())
    return resolved if isinstance(resolved, int) else None


def _open_the_spacr_level() -> Optional[int]:
    """Let INFO through on ``spacr.*`` for the life of a run.

    A handler only sees a record its *logger* already let through, and a
    bare library import leaves the root logger at WARNING — so without
    this the per-run log would hold the warnings and none of the INFO
    lines that say what the run did, which is most of what makes it worth
    reading. Only the ``spacr`` logger is touched, not the root: a host
    application's own loggers keep whatever level it chose, and every
    handler still applies its own level on top, so this cannot make a
    quiet console noisy on its own.

    :returns: the level to restore, or None when nothing needed changing.
    """
    logger = logging.getLogger("spacr")
    previous = logger.level
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
        return previous
    return None


class _RunLogHandler(logging.Handler):
    """Write this run's records to ``<runs>/<run_id>.jsonl``, one per line.

    Attached to the root logger for the life of the run, so it sees every
    record from every spaCR module regardless of what else is configured.
    Records belonging to a *different* run — a nested run on another
    thread — are not written here: an id that identifies everything
    identifies nothing.
    """

    def __init__(self, run_id: str, path: str, level: int = logging.NOTSET):
        super().__init__(level)
        self.run_id = str(run_id)
        self.path = path
        self._stream = None
        self._lock_out = threading.Lock()

    def _open(self):
        """Open the JSONL file lazily, appending."""
        if self._stream is None:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self._stream = open(self.path, "a", encoding="utf-8")
        return self._stream

    def emit(self, record: logging.LogRecord) -> None:
        """Serialise one record, unless it belongs to another run."""
        stamped = getattr(record, "run_id", "") or current_run_id()
        if stamped and stamped != self.run_id:
            return
        try:
            payload = {
                "run_id": self.run_id,
                "utc": datetime.fromtimestamp(
                    record.created, tz=timezone.utc).isoformat(),
                "created": record.created,
                "level": record.levelname,
                "levelno": int(record.levelno),
                "logger": record.name,
                "message": record.getMessage(),
                "file": record.pathname,
                "line": int(record.lineno),
                "func": record.funcName,
                "process": int(record.process or 0),
                "thread": record.threadName,
            }
            if record.exc_info:
                payload["traceback"] = "".join(
                    _traceback.format_exception(*record.exc_info))
            line = json.dumps(payload, default=str)
        except Exception:                              # noqa: BLE001
            self.handleError(record)
            return
        try:
            with self._lock_out:
                stream = self._open()
                stream.write(line + "\n")
                stream.flush()
        except Exception:                              # noqa: BLE001
            # A full disk must not take the run down with it: the run log
            # is evidence, not the result.
            self.handleError(record)

    def close(self) -> None:
        """Flush and close the file, then detach."""
        with self._lock_out:
            if self._stream is not None:
                try:
                    self._stream.flush()
                    self._stream.close()
                finally:
                    self._stream = None
        super().close()


# ---------------------------------------------------------------------------
# S5 — one seed, and an honest account of where it does not reach
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeedReport:
    """What :func:`seed_everything` actually managed to seed.

    :param seed: the seed applied.
    :param seeded: library handles that were seeded, e.g. ``"python"``,
        ``"numpy"``, ``"torch"``, ``"torch.cuda"``.
    :param unavailable: handles that could not be seeded because the
        library is not installed. Not an error — a headless analysis box
        without Torch is a supported install.
    :param caveats: the honest part. Every place this seed does **not**
        buy determinism, in plain sentences, so a caller can quote them
        rather than assume a guarantee that does not exist.
    :param deterministic: whether the deterministic-kernel switches were
        requested as well.
    """

    seed: int
    seeded: Tuple[str, ...] = ()
    unavailable: Tuple[str, ...] = ()
    caveats: Tuple[str, ...] = ()
    deterministic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable copy of the report."""
        return {"seed": self.seed, "seeded": list(self.seeded),
                "unavailable": list(self.unavailable),
                "caveats": list(self.caveats),
                "deterministic": self.deterministic}

    def __str__(self) -> str:
        """One line: the seed and what it reached."""
        return (f"seed {self.seed} → {', '.join(self.seeded) or 'nothing'}"
                + (f" (no {', '.join(self.unavailable)})"
                   if self.unavailable else ""))


#: Named limits of a global seed. Kept as data rather than prose so
#: :class:`SeedReport` can carry them to a GUI, a log line and a test.
SEED_CAVEATS: Dict[str, str] = {
    "hashseed": (
        "PYTHONHASHSEED is set for child processes only — this "
        "interpreter fixed its string hash seed before any of this ran, so "
        "set-iteration order in *this* process is whatever it already was."),
    "cellpose": (
        "Cellpose has no seed of its own; it draws from NumPy and Torch, so "
        "it follows this call on CPU. Its CUDA path is not bit-reproducible "
        "unless deterministic=True, and its resize/interpolation kernels may "
        "not be even then."),
    "sklearn": (
        "scikit-learn has no global seed. Estimators built with "
        "random_state=None do draw from the NumPy global stream this call "
        "seeds, but one built with an explicit random_state ignores it — "
        "pass spacr.runctx.random_state() at the construction site."),
    "cuda": (
        "CUDA kernels that accumulate atomically (scatter-add, some "
        "pooling and interpolation backward passes) are non-deterministic "
        "regardless of the seed; deterministic=True asks Torch to use "
        "deterministic kernels where it has them and warn where it does "
        "not."),
    "cublas": (
        "cuBLAS reductions are only deterministic when "
        "CUBLAS_WORKSPACE_CONFIG is set before the CUDA context is created. "
        "deterministic=True sets it, but if CUDA is already initialised the "
        "setting arrives too late for this process."),
    "workers": (
        "A forked worker inherits this seeded state, so every worker draws "
        "the *same* stream; a spawned worker inherits none of it. Pass "
        "spacr.runctx.seed_worker as a DataLoader worker_init_fn, and "
        "spacr.runctx.spacr_rng(stream=...) for a per-worker stream."),
    "threads": (
        "Thread and process scheduling still decide the order results come "
        "back in. A reduction over a set whose order varies can differ in "
        "the last bits even with every RNG pinned."),
}


def resolve_seed(settings: Optional[Mapping[str, Any]] = None,
                 default: Any = DEFAULT_SEED) -> Optional[int]:
    """Return the seed a run should use.

    Reads ``random_seed`` from ``settings``, then :data:`SEED_ENV`, then
    ``default``. An explicit ``random_seed=None`` means "do not seed" and
    is honoured — a caller who deliberately wants a free-running RNG (a
    simulation sweep that must not produce the same draw twice) gets one.

    :param settings: a settings dict, or None.
    :param default: what to use when nothing names a seed. Pass ``None``
        for "leave the RNGs alone unless asked".
    :returns: the seed, or None for "do not seed".
    """
    if settings is not None and "random_seed" in settings:
        raw = settings["random_seed"]
    elif settings is not None and "seed" in settings:
        raw = settings["seed"]
    elif os.environ.get(SEED_ENV, "").strip():
        raw = os.environ[SEED_ENV].strip()
    else:
        raw = default
    if raw is None or raw is False:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text or text.lower() in {"none", "null", "off", "false"}:
            return None
        try:
            return int(text, 0)
        except ValueError:
            # A word rather than a number: hash it, so `random_seed:
            # "plate3-rerun"` is a usable, reproducible seed rather than a
            # crash or a silent fall back to 42.
            return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def seed_everything(seed: Optional[int] = None,
                    *,
                    deterministic: bool = False,
                    quiet: bool = True) -> SeedReport:
    """Seed every RNG spaCR can reach, and report what that does not cover.

    Seeds, in order: :mod:`random`, NumPy's legacy global (``np.random``),
    the :class:`~numpy.random.Generator` stream :func:`spacr_rng` derives,
    Torch on CPU, and Torch on every visible CUDA device. Cellpose is
    seeded transitively — it has no seed API and draws from NumPy and
    Torch. ``PYTHONHASHSEED`` is exported for child processes.

    **What this does not buy you.** A seed makes the *draws* reproducible.
    It does not make CUDA reductions associative, it does not stop a
    forked worker pool from sharing one stream, and it cannot re-seed this
    interpreter's string hashing. Every such limit is named in
    :attr:`SeedReport.caveats` and in :data:`SEED_CAVEATS`; do not promise
    a user more than that list allows. In particular, calling this and
    then reporting "the run is deterministic" is wrong on a GPU unless
    ``deterministic=True`` *and* the model avoids the kernels Torch has no
    deterministic implementation for.

    :param seed: the seed. ``None`` uses :data:`DEFAULT_SEED`.
    :param deterministic: also ask for deterministic kernels —
        ``cudnn.deterministic``, ``cudnn.benchmark=False``,
        ``torch.use_deterministic_algorithms(warn_only=True)`` and
        ``CUBLAS_WORKSPACE_CONFIG``. Slower, sometimes much slower, and
        still not a guarantee; see the caveats.
    :param quiet: when False, log the report at INFO.
    :returns: a :class:`SeedReport`.
    """
    value = DEFAULT_SEED if seed is None else int(seed)
    # Python's random and NumPy's legacy seeder want different ranges;
    # 2**32 is the narrower of the two, so normalise once.
    narrow = int(value) % (2 ** 32)
    seeded: List[str] = []
    unavailable: List[str] = []
    caveats: List[str] = [SEED_CAVEATS["hashseed"], SEED_CAVEATS["sklearn"],
                          SEED_CAVEATS["cellpose"], SEED_CAVEATS["workers"],
                          SEED_CAVEATS["threads"]]

    os.environ["PYTHONHASHSEED"] = str(narrow)
    _random.seed(value)
    seeded.append("python")

    np.random.seed(narrow)
    seeded.append("numpy")
    global _ROOT_SEED_SEQUENCE
    _ROOT_SEED_SEQUENCE = np.random.SeedSequence(value)
    seeded.append("numpy.Generator")

    torch = sys.modules.get("torch")
    if torch is None:
        try:
            import torch                                # type: ignore
        except Exception:                               # noqa: BLE001
            torch = None
    if torch is None:
        unavailable.append("torch")
    else:
        torch.manual_seed(value)
        seeded.append("torch")
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(value)
                seeded.append("torch.cuda")
                caveats.append(SEED_CAVEATS["cuda"])
        except Exception as exc:                        # noqa: BLE001
            # A driver mismatch must not stop a CPU run from being seeded.
            LOG.debug("could not seed CUDA: %s", exc)
            unavailable.append("torch.cuda")
        if deterministic:
            caveats.append(SEED_CAVEATS["cublas"])
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            with contextlib.suppress(Exception):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                seeded.append("torch.cudnn")
            with contextlib.suppress(Exception):
                torch.use_deterministic_algorithms(True, warn_only=True)
                seeded.append("torch.deterministic-algorithms")

    if "cellpose" in sys.modules:
        # Nothing to call — recorded so the report does not read as though
        # Cellpose was overlooked.
        seeded.append("cellpose(via numpy+torch)")

    report = SeedReport(seed=value, seeded=tuple(seeded),
                        unavailable=tuple(unavailable),
                        caveats=tuple(dict.fromkeys(caveats)),
                        deterministic=bool(deterministic))
    if not quiet:
        LOG.info("%s", report)
    return report


#: Root entropy for :func:`spacr_rng`; replaced by every seeding call.
_ROOT_SEED_SEQUENCE: Optional["np.random.SeedSequence"] = None


def random_state(default: Optional[int] = None) -> Optional[int]:
    """Return the seed to hand an estimator's ``random_state=``.

    sklearn, XGBoost, LightGBM and CatBoost all take a ``random_state``
    (or ``seed``) and all ignore the NumPy global stream once one is
    given, so a construction site that hard-codes ``random_state=42``
    silently overrides the run's seed. Call this instead::

        RandomForestClassifier(random_state=random_state(42))

    :param default: what to return when no run is open and nothing has
        been seeded.
    :returns: the active run's seed, or ``default``.
    """
    context = _ACTIVE.get()
    if context is not None and context.seed is not None:
        return int(context.seed)
    from_env = resolve_seed(None, default=None)
    return from_env if from_env is not None else default


def spacr_rng(stream: str = "",
              seed: Optional[int] = None) -> "np.random.Generator":
    """Return an independent :class:`numpy.random.Generator` for one stream.

    Derived from the run seed by :class:`~numpy.random.SeedSequence`
    spawning rather than by re-seeding from ``seed + 1``: adjacent seeds
    produce correlated streams in some bit generators, and two workers
    drawing correlated "random" subsamples is a bug that looks like data.

    :param stream: a name for this stream — a worker id, a stage, a fold.
        Different names give independent streams; the same name gives the
        same stream every run.
    :param seed: override the run seed.
    :returns: a fresh Generator.
    """
    if seed is not None:
        root = np.random.SeedSequence(int(seed))
    elif _ROOT_SEED_SEQUENCE is not None:
        root = _ROOT_SEED_SEQUENCE
    else:
        resolved = random_state()
        root = np.random.SeedSequence(
            DEFAULT_SEED if resolved is None else int(resolved))
    if not stream:
        return np.random.default_rng(root)
    key = int(hashlib.sha256(str(stream).encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(
        np.random.SeedSequence(entropy=root.entropy, spawn_key=(key,)))


def torch_generator(device: str = "cpu", stream: str = ""):
    """Return a seeded :class:`torch.Generator` for a DataLoader or sampler.

    :param device: the device the generator belongs to.
    :param stream: a stream name, as for :func:`spacr_rng`.
    :returns: a ``torch.Generator`` seeded from the run seed.
    :raises RuntimeError: when Torch is not installed. Deliberate: a
        caller asking for a Torch generator cannot proceed without one,
        and handing back None would fail further away.
    """
    try:
        import torch                                    # type: ignore
    except Exception as exc:                            # noqa: BLE001
        raise RuntimeError(
            "torch_generator() needs PyTorch, which is not installed") from exc
    base = random_state(DEFAULT_SEED) or DEFAULT_SEED
    if stream:
        base ^= int(hashlib.sha256(
            str(stream).encode("utf-8")).hexdigest()[:8], 16)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(base) % (2 ** 63))
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed one DataLoader worker. Pass as ``worker_init_fn=seed_worker``.

    A spawned worker inherits none of the parent's RNG state and a forked
    one inherits *all* of it — so every worker augments identically, which
    is the classic silent bug where a batch of eight "random" crops is
    eight copies of the same transform. This derives a per-worker stream
    from Torch's initial seed, which the DataLoader has already varied per
    worker and per epoch.

    :param worker_id: the worker's index, supplied by the DataLoader.
    """
    try:
        import torch                                    # type: ignore
        base = torch.initial_seed()
    except Exception:                                   # noqa: BLE001
        base = random_state(DEFAULT_SEED) or DEFAULT_SEED
    worker_seed = (int(base) + int(worker_id)) % (2 ** 32)
    np.random.seed(worker_seed)
    _random.seed(worker_seed)


# ---------------------------------------------------------------------------
# S9 — on_error: stop | skip | retry
# ---------------------------------------------------------------------------

class _SkippedType:
    """Sentinel returned by :meth:`ErrorPolicy.run` for a skipped unit."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        """False, so ``if result:`` treats a skip as no result."""
        return False

    def __repr__(self) -> str:
        return "SKIPPED"


#: Returned by :meth:`ErrorPolicy.run` when the unit was skipped. Falsy, and
#: identity-comparable: ``if result is SKIPPED``.
SKIPPED = _SkippedType()


@dataclass(frozen=True)
class SkipRecord:
    """One unit of work that ``on_error='skip'`` dropped, and why.

    The whole point of ``skip`` over a bare ``except: pass``: what was
    lost is named, counted and persisted, so a run that covered 97 of 100
    plates cannot be read as one that covered 100.

    :param unit: the unit skipped — a plate folder, a well, a field file.
    :param stage: the pipeline stage it was skipped at.
    :param reason: why, in a sentence.
    :param exc_type: the exception class name.
    :param message: ``str(exc)``.
    :param attempts: how many times it was tried before being given up on.
    :param run_id: the run that skipped it.
    :param utc: when.
    :param traceback_str: the full traceback, kept for the ledger.
    """

    unit: str
    stage: str
    reason: str
    exc_type: str
    message: str
    attempts: int = 1
    run_id: str = ""
    utc: str = field(default_factory=_utcnow)
    traceback_str: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable copy of the record."""
        return {"unit": self.unit, "stage": self.stage, "reason": self.reason,
                "exc_type": self.exc_type, "message": self.message,
                "attempts": self.attempts, "run_id": self.run_id,
                "utc": self.utc}

    def __str__(self) -> str:
        """``unit (stage): reason``."""
        where = f" ({self.stage})" if self.stage else ""
        return f"{self.unit}{where}: {self.reason}"


class _Attempt:
    """One try at one unit. Use as ``with attempt:`` inside the loop.

    Swallows an ordinary exception and hands it back to
    :meth:`ErrorPolicy.attempts_for`, which decides whether to yield
    another attempt, record a skip, or re-raise. Fatal exceptions
    (:data:`FATAL_EXCEPTIONS`) are never swallowed.
    """

    __slots__ = ("policy", "unit", "stage", "number", "of", "exc", "ok")

    def __init__(self, policy: "ErrorPolicy", unit: str, stage: str,
                 number: int, of: int) -> None:
        self.policy = policy
        self.unit = unit
        self.stage = stage
        self.number = number
        self.of = of
        self.exc: Optional[BaseException] = None
        self.ok = False

    @property
    def last(self) -> bool:
        """True when no further attempt will be made after this one."""
        return self.number >= self.of

    def __enter__(self) -> "_Attempt":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Record the outcome; suppress an ordinary failure, propagate a fatal."""
        if exc is None:
            self.ok = True
            return False
        if isinstance(exc, FATAL_EXCEPTIONS):
            return False
        self.exc = exc
        return True

    def __repr__(self) -> str:
        return (f"<attempt {self.number}/{self.of} on {self.unit!r} "
                f"stage={self.stage!r}>")


class ErrorPolicy:
    """What a run does when one unit of work fails: stop, skip, or retry.

    Applied at a *batch boundary* — per plate, per well, per field —
    wherever the pipeline has a unit it can name. Every mode records the
    failure on the :class:`~spacr.errors.RunLedger`, so the artifact stamp
    tells the truth in all three cases; the mode only decides whether the
    run survives it.

    :param mode: :data:`ON_ERROR_STOP` (default), :data:`ON_ERROR_SKIP` or
        :data:`ON_ERROR_RETRY`.
    :param attempts: total tries per unit in ``retry`` mode, including the
        first. Bounded on purpose: an unbounded retry against a dead NAS
        is an infinite loop with a progress bar.
    :param backoff: seconds before the second attempt. Doubled before each
        subsequent one, capped at :data:`MAX_BACKOFF`.
    :param ledger: the ledger to record on. One is created when omitted.
    :param logger: where to log; defaults to ``spacr.runctx``.
    :param run_id: stamped onto every :class:`SkipRecord`.
    :param record: write successes and failures to ``ledger``. Pass False
        at a boundary whose call site already records them — the Measure
        pool does, through its own job and error callbacks — so the ledger
        counts each field once rather than twice.
    :param sleep: the sleep function, injectable so a test can assert the
        backoff schedule without waiting for it.
    :raises ValueError: on an unknown mode, or a non-positive attempt
        count — a "retry" that never retries is a silent stop.
    """

    def __init__(self,
                 mode: str = DEFAULT_ON_ERROR,
                 *,
                 attempts: int = DEFAULT_RETRIES,
                 backoff: float = DEFAULT_BACKOFF,
                 ledger: Optional[RunLedger] = None,
                 logger: Optional[logging.Logger] = None,
                 run_id: str = "",
                 record: bool = True,
                 sleep: Optional[Callable[[float], None]] = None) -> None:
        normalized = str(mode or DEFAULT_ON_ERROR).strip().lower()
        if normalized not in ON_ERROR_MODES:
            raise ValueError(
                f"on_error must be one of {', '.join(ON_ERROR_MODES)}, "
                f"not {mode!r}")
        self.mode = normalized
        self.attempts = int(attempts)
        if self.attempts < 1:
            raise ValueError(
                f"on_error_attempts must be at least 1, not {attempts!r}; "
                "a retry budget of zero is a stop wearing a different name")
        self.backoff = max(0.0, float(backoff))
        self.ledger = ledger if ledger is not None else RunLedger("run")
        self.log = logger if logger is not None else LOG
        self.run_id = str(run_id)
        self.record = bool(record)
        self._sleep = sleep if sleep is not None else time.sleep
        self._skips: List[SkipRecord] = []
        self._retried: List[Tuple[str, int]] = []

    def bind(self, ledger: Optional[RunLedger] = None,
             record: Optional[bool] = None) -> "ErrorPolicy":
        """Point this policy at another ledger, and return it.

        A run with several ledgers — Measure keeps one per source folder —
        needs the failures recorded against the right one, while the skip
        list stays with the run. Mutates and returns ``self`` rather than
        copying, so :attr:`skips` remains the single account of what the
        run did not cover.

        :param ledger: the ledger to record on from now on.
        :param record: whether to record at all; see the constructor.
        """
        if ledger is not None:
            self.ledger = ledger
        if record is not None:
            self.record = bool(record)
        return self

    # -- what happened ----------------------------------------------------

    @property
    def skips(self) -> List[SkipRecord]:
        """Every unit ``skip`` dropped, in the order it dropped them."""
        return list(self._skips)

    @property
    def skipped_units(self) -> List[str]:
        """Just the names — the answer to "what did this run not cover?"."""
        return [record.unit for record in self._skips]

    @property
    def n_skipped(self) -> int:
        """How many units were skipped."""
        return len(self._skips)

    @property
    def retries(self) -> List[Tuple[str, int]]:
        """``(unit, attempts_made)`` for every unit that had to be retried."""
        return list(self._retried)

    def __repr__(self) -> str:
        extra = (f" attempts={self.attempts} backoff={self.backoff}"
                 if self.mode == ON_ERROR_RETRY else "")
        return f"<ErrorPolicy {self.mode}{extra} skipped={self.n_skipped}>"

    # -- the loop ---------------------------------------------------------

    def attempts_for(self, unit: Any,
                     stage: Optional[str] = None) -> Iterator[_Attempt]:
        """Yield attempts at one unit, honouring the mode. The core of S9.

        Drive it with an inner ``with``::

            for attempt in policy.attempts_for(plate, stage='plate'):
                with attempt:
                    process(plate)

        In ``stop`` and ``skip`` mode exactly one attempt is yielded; in
        ``retry`` mode up to :attr:`attempts`, with a sleep in between. The
        generator raises the last exception after the final attempt unless
        the mode is ``skip``, so the ``for`` statement is where a ``stop``
        run aborts.

        A body that omits the inner ``with`` is a bug this cannot detect —
        the failure would propagate straight out of the ``for``, which is
        ``stop`` behaviour whatever the mode says.

        :param unit: the thing being processed; stringified for the record.
        :param stage: pipeline stage, recorded on the ledger and the skip.
        :raises Exception: the unit's own exception, in ``stop`` mode and
            in ``retry`` mode once the budget is spent.
        """
        name = str(unit)
        stage_name = str(stage) if stage else self.ledger.name
        total = self.attempts if self.mode == ON_ERROR_RETRY else 1
        last: Optional[BaseException] = None

        for number in range(1, total + 1):
            attempt = _Attempt(self, name, stage_name, number, total)
            yield attempt
            if attempt.ok:
                if self.record:
                    self.ledger.record_success(name, stage_name)
                if number > 1:
                    self._retried.append((name, number))
                    self.log.info(
                        "[%s] %s succeeded on attempt %d of %d",
                        stage_name, name, number, total)
                return
            if attempt.exc is None:
                # The body never ran, or `break`/`continue` skipped the
                # `with`. Nothing to judge; leave the loop alone.
                return
            last = attempt.exc
            if number < total:
                delay = self._delay_for(number)
                self.log.warning(
                    "[%s] %s failed on attempt %d of %d (%s: %s); retrying "
                    "in %.1fs", stage_name, name, number, total,
                    type(last).__name__, last, delay)
                if delay:
                    self._sleep(delay)

        self._give_up(name, stage_name, last, total)

    def _delay_for(self, attempt_number: int) -> float:
        """Return the backoff before the attempt after ``attempt_number``."""
        return min(MAX_BACKOFF, self.backoff * (2 ** (attempt_number - 1)))

    def _give_up(self, unit: str, stage: str, exc: Optional[BaseException],
                 tried: int) -> None:
        """Record the failure, then skip or re-raise according to the mode."""
        if self.record:
            self.ledger.record_failure(unit, stage, exc)
        if exc is None:                                 # pragma: no cover
            return
        if self.mode == ON_ERROR_SKIP:
            reason = (f"{type(exc).__name__}: {exc}" if str(exc)
                      else type(exc).__name__)
            record = SkipRecord(
                unit=unit, stage=stage, reason=reason,
                exc_type=type(exc).__name__, message=str(exc),
                attempts=tried, run_id=self.run_id or current_run_id(),
                traceback_str="".join(_traceback.format_exception(
                    type(exc), exc, exc.__traceback__)))
            self._skips.append(record)
            self.log.warning("on_error=skip: skipping %s — %s", unit, reason)
            return
        if self.mode == ON_ERROR_RETRY:
            self._retried.append((unit, tried))
            self.log.error(
                "on_error=retry: %s failed all %d attempts; stopping the run",
                unit, tried)
        else:
            self.log.error("on_error=stop: %s failed; stopping the run", unit)
        raise exc

    def run(self, unit: Any, fn: Callable[..., Any], *args: Any,
            stage: Optional[str] = None, **kwargs: Any) -> Any:
        """Call ``fn(*args, **kwargs)`` for one unit under this policy.

        The ergonomic form of :meth:`attempts_for`, for a boundary whose
        body is already a callable.

        :param unit: the thing being processed.
        :param fn: the work.
        :param stage: pipeline stage.
        :returns: whatever ``fn`` returned, or :data:`SKIPPED` when the
            unit was skipped.
        :raises Exception: as :meth:`attempts_for`.
        """
        result: Any = SKIPPED
        for attempt in self.attempts_for(unit, stage=stage):
            with attempt:
                result = fn(*args, **kwargs)
        return result

    def summary(self) -> str:
        """A one-block account of what the policy did. Empty when nothing did."""
        if not self._skips and not self._retried:
            return ""
        lines = [f"on_error={self.mode}"]
        if self._retried:
            lines.append(f"  retried  : {len(self._retried)} unit(s)")
            for unit, tried in self._retried[:10]:
                lines.append(f"    {unit} — {tried} attempt(s)")
        if self._skips:
            lines.append(f"  SKIPPED  : {len(self._skips)} unit(s) — these "
                         f"are NOT in the output")
            for record in self._skips[:20]:
                lines.append(f"    {record}")
            if len(self._skips) > 20:
                lines.append(f"    ... and {len(self._skips) - 20} more")
        return "\n".join(lines)


def resolve_error_policy(settings: Optional[Mapping[str, Any]] = None,
                         *,
                         ledger: Optional[RunLedger] = None,
                         logger: Optional[logging.Logger] = None,
                         run_id: str = "",
                         sleep: Optional[Callable[[float], None]] = None,
                         default: str = DEFAULT_ON_ERROR) -> ErrorPolicy:
    """Build the :class:`ErrorPolicy` a settings dict asks for.

    Reads ``on_error``, ``on_error_attempts`` and ``on_error_backoff``.
    An unset ``on_error`` means :data:`DEFAULT_ON_ERROR`.

    :param settings: a settings dict, or None.
    :param ledger: the ledger failures are recorded on.
    :param logger: where the policy logs.
    :param run_id: stamped onto skip records.
    :param sleep: injectable sleep, for tests.
    :param default: mode to use when the settings name none.
    :returns: an :class:`ErrorPolicy`.
    :raises ValueError: when ``on_error`` is not one of the three modes.
        Loud on purpose: a typo like ``on_error='continue'`` silently
        falling back to ``stop`` is how a user believes they asked for
        tolerance and did not get it.
    """
    values = dict(settings or {})
    mode = values.get("on_error", default)
    attempts = values.get("on_error_attempts", DEFAULT_RETRIES)
    backoff = values.get("on_error_backoff", DEFAULT_BACKOFF)
    try:
        attempts = int(attempts)
    except (TypeError, ValueError):
        attempts = DEFAULT_RETRIES
    try:
        backoff = float(backoff)
    except (TypeError, ValueError):
        backoff = DEFAULT_BACKOFF
    return ErrorPolicy(mode if mode is not None else default,
                       attempts=attempts, backoff=backoff, ledger=ledger,
                       logger=logger, run_id=run_id, sleep=sleep)


# ---------------------------------------------------------------------------
# The run context itself
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    """One pipeline run: its id, its seed, its error policy, its ledger.

    Created by :func:`run_context`; read anywhere by
    :func:`current_run_context`. The id on this object is the id on every
    log line the run emits (:func:`read_run_log`), on every
    :class:`~spacr.errors.RunLedger` it hands out, and on every
    :class:`spacr.artifacts.Artifact` it registers — which is what lets a
    log line and an output be joined.

    :param run_id: the id.
    :param module: the producing module key — ``"mask"``, ``"measure"``,
        the same keys :mod:`spacr.ports` uses.
    :param seed: the seed applied, or None when the run is unseeded.
    :param policy: the :class:`ErrorPolicy` in force.
    :param ledger: the run's :class:`~spacr.errors.RunLedger`, whose
        ``run_id`` has been set to this run's.
    :param settings: the settings the run was started with.
    :param seed_report: what :func:`seed_everything` managed to seed.
    :param started_utc: when the run opened.
    :param log_path: the run's JSONL log, or ``""`` when logging is off.
    """

    run_id: str
    module: str = ""
    seed: Optional[int] = None
    policy: ErrorPolicy = field(default_factory=ErrorPolicy)
    ledger: Optional[RunLedger] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    seed_report: Optional[SeedReport] = None
    started_utc: str = field(default_factory=_utcnow)
    log_path: str = ""

    @property
    def log(self) -> logging.Logger:
        """The run's logger. Every record it makes carries :attr:`run_id`."""
        return logging.getLogger(f"spacr.{self.module}" if self.module
                                 else "spacr.run")

    @property
    def skips(self) -> List[SkipRecord]:
        """Units the policy skipped — shorthand for ``policy.skips``."""
        return self.policy.skips

    def new_ledger(self, name: str) -> RunLedger:
        """Return a :class:`~spacr.errors.RunLedger` stamped with this run id.

        A ledger mints its own uuid, which would put a *second* id on the
        run's ``run_status`` rows and break the join to the artifact
        registry. This overwrites it, so every stamp the run leaves —
        ledger row, log line, artifact — carries one id.

        :param name: the ledger name, i.e. the pipeline stage.
        """
        ledger = RunLedger(name)
        ledger.run_id = self.run_id
        return ledger

    def adopt(self, ledger: RunLedger) -> RunLedger:
        """Re-stamp an existing ledger with this run's id, and return it.

        For a call site that already builds its own ledger and should not
        have to change how.
        """
        ledger.run_id = self.run_id
        return ledger

    def rng(self, stream: str = "") -> "np.random.Generator":
        """An independent seeded Generator; see :func:`spacr_rng`."""
        return spacr_rng(stream, seed=self.seed)

    def random_state(self, default: Optional[int] = None) -> Optional[int]:
        """This run's seed, for an estimator's ``random_state=``."""
        return int(self.seed) if self.seed is not None else default

    def register_outputs(self, module: Optional[str] = None,
                         settings: Optional[Mapping[str, Any]] = None,
                         **kwargs: Any) -> Tuple[Any, ...]:
        """Register this run's outputs, stamped with this run id.

        Thin wrapper over :func:`spacr.artifacts.register_run_outputs` that
        supplies ``run_id`` and defaults ``strict`` to False, so a registry
        that cannot be written costs one printed line and never the run.

        :param module: override the module key.
        :param settings: override the settings hashed into the artifacts.
        :param kwargs: passed through, e.g. ``roots=[...]``, ``status=``.
        :returns: the registered artifacts.
        """
        from .artifacts import register_run_outputs
        kwargs.setdefault("strict", False)
        return register_run_outputs(
            module or self.module,
            self.settings if settings is None else settings,
            run_id=self.run_id, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable account of the run, for a manifest or a log."""
        return {
            "run_id": self.run_id, "module": self.module, "seed": self.seed,
            "on_error": self.policy.mode,
            "on_error_attempts": self.policy.attempts,
            "started_utc": self.started_utc, "log_path": self.log_path,
            "skipped": [record.to_dict() for record in self.policy.skips],
            "seed_report": (self.seed_report.to_dict()
                            if self.seed_report else None),
        }

    def __str__(self) -> str:
        """``run <id> (<module>) seed=… on_error=…``."""
        return (f"run {self.run_id} ({self.module or 'spacr'}) "
                f"seed={self.seed} on_error={self.policy.mode}")


@contextlib.contextmanager
def run_context(module: str = "",
                settings: Optional[Mapping[str, Any]] = None,
                *,
                run_id: Optional[str] = None,
                seed: Any = _UNSET,
                on_error: Optional[str] = None,
                deterministic: Optional[bool] = None,
                ledger: Optional[RunLedger] = None,
                log: bool = True,
                sleep: Optional[Callable[[float], None]] = None,
                ) -> Iterator[RunContext]:
    """Open a run: mint an id, seed the world, arm the error policy.

    The one call a pipeline entry point makes. Inside the block,
    :func:`current_run_id` answers everywhere in this process (and, via
    :data:`RUN_ID_ENV`, in its children), every log record carries the id,
    and :func:`read_run_log` can pull the run's own log back out
    afterwards.

    :param module: the producing module key — ``"mask"``, ``"measure"``,
        … — used for the artifact registry and the logger name.
    :param settings: the run's settings. ``random_seed``, ``on_error``,
        ``on_error_attempts`` and ``on_error_backoff`` are read from here.
    :param run_id: use this id instead of minting one — for a resumed run,
        or a distributed worker continuing its parent's run.
    :param seed: override the seed. An explicit ``None`` means "do not
        seed at all"; omit the argument to read ``settings``.
    :param on_error: override the mode.
    :param deterministic: also request deterministic kernels; see
        :func:`seed_everything`. Defaults to the ``deterministic`` setting.
    :param ledger: use this ledger rather than making one.
    :param log: write the per-run JSONL log. False for a caller that only
        wants the id and the policy.
    :param sleep: injectable sleep for the retry backoff, for tests.
    :yields: the :class:`RunContext`.
    """
    identifier = str(run_id).strip() if run_id else new_run_id()
    values = dict(settings or {})

    resolved_seed = (resolve_seed(values) if seed is _UNSET
                     else resolve_seed({"random_seed": seed}, default=None))
    want_deterministic = (bool(values.get("deterministic", False))
                          if deterministic is None else bool(deterministic))
    report = (seed_everything(resolved_seed, deterministic=want_deterministic)
              if resolved_seed is not None else None)

    run_ledger = ledger if ledger is not None else RunLedger(module or "run")
    run_ledger.run_id = identifier
    policy = resolve_error_policy(
        values if on_error is None else {**values, "on_error": on_error},
        ledger=run_ledger, run_id=identifier, sleep=sleep)

    context = RunContext(run_id=identifier, module=str(module),
                         seed=resolved_seed, policy=policy, ledger=run_ledger,
                         settings=values, seed_report=report)

    install_run_id_logging()
    handler: Optional[_RunLogHandler] = None
    restore_level: Optional[int] = None
    if log:
        try:
            context.log_path = run_log_path(identifier)
            handler = _RunLogHandler(identifier, context.log_path)
            handler.addFilter(RunIdFilter(identifier))
            logging.getLogger().addHandler(handler)
            restore_level = _open_the_spacr_level()
        except Exception as exc:                        # noqa: BLE001
            LOG.warning("could not open the run log for %s: %s",
                        identifier, exc)
            handler = None
            context.log_path = ""

    token = _ACTIVE.set(context)
    previous_env = os.environ.get(RUN_ID_ENV)
    os.environ[RUN_ID_ENV] = identifier
    started = time.time()
    context.log.info("run %s started — module=%s seed=%s on_error=%s",
                     identifier, module or "spacr", resolved_seed, policy.mode)
    if report is not None:
        context.log.debug("run %s seeding: %s", identifier, report)
    try:
        yield context
    except BaseException as exc:
        context.log.error("run %s failed after %.1fs — %s: %s", identifier,
                          time.time() - started, type(exc).__name__, exc)
        raise
    else:
        summary = policy.summary()
        if summary:
            context.log.warning("run %s: %s", identifier, summary)
        context.log.info("run %s finished in %.1fs — %d skipped",
                         identifier, time.time() - started, policy.n_skipped)
    finally:
        _ACTIVE.reset(token)
        if previous_env is None:
            os.environ.pop(RUN_ID_ENV, None)
        else:
            os.environ[RUN_ID_ENV] = previous_env
        if restore_level is not None:
            logging.getLogger("spacr").setLevel(restore_level)
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()


# ---------------------------------------------------------------------------
# The settings seam
# ---------------------------------------------------------------------------

def _defaults(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Return the run-control defaults, filled into ``settings``."""
    values = dict(settings or {})
    values.setdefault("random_seed", DEFAULT_SEED)
    values.setdefault("on_error", DEFAULT_ON_ERROR)
    values.setdefault("on_error_attempts", DEFAULT_RETRIES)
    values.setdefault("on_error_backoff", DEFAULT_BACKOFF)
    return values


def apply_defaults(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fill this module's keys into ``settings``, in place when given a dict.

    For a caller that wants an explicit, complete settings dict — a batch
    runner writing a settings CSV, or a test. The pipelines do not need
    it: :func:`resolve_seed` and :func:`resolve_error_policy` default
    every key, so a settings dict that names none of them still gets
    ``on_error='stop'`` and :data:`DEFAULT_SEED`.

    :param settings: the dict to fill; a new one is made when None.
    :returns: the same dict, with :data:`RUN_SETTING_KEYS` present.
    """
    if settings is None:
        return _defaults()
    for key, value in _defaults().items():
        settings.setdefault(key, value)
    return settings


def _register_settings() -> None:
    """Declare this module's keys with :func:`spacr.settings.register_defaults`.

    Types and tooltips come from here rather than from another 20 lines
    appended to the 4000-line settings module, which is what the defaults
    seam is for. That is enough for :func:`spacr.settings.check_settings`
    to accept and coerce ``on_error`` out of a settings CSV, and for
    ``spacr-run --set on_error=skip`` to type it.

    Two deliberate omissions:

    * ``random_seed`` is not re-declared. It already exists in
      :mod:`spacr.settings` as an ``int`` with help text of its own, and
      the registry rightly refuses to let one module rewrite another's.
    * no ``categories`` contribution, and no key is injected into the
      ``set_default_*`` factories. The Qt settings panel buckets a
      module's keys through the per-app layouts in
      ``spacr.qt.screens.settings_model``, and a key that reaches a
      module's defaults without appearing in that module's layout renders
      in a catch-all section — which the layout tests correctly reject.
      Giving ``on_error`` a settings-panel row therefore means naming it
      in those per-app layouts, in that file, which is not this module's
      to edit. Until then the knob is reachable from a settings dict, a
      settings CSV and the CLI, and every pipeline defaults it to
      :data:`DEFAULT_ON_ERROR`.
    """
    try:
        from .settings import register_defaults, has_registered_defaults
    except Exception:                                   # noqa: BLE001
        return
    if has_registered_defaults("runctx"):
        return
    try:
        register_defaults(
            "runctx", _defaults,
            expected_types={"on_error": str, "on_error_attempts": int,
                            "on_error_backoff": float, "random_seed": int},
            tooltips={
                "on_error": (
                    "(str) - What a failed unit of work does to the run, "
                    "checked at every batch boundary (field, well, plate). "
                    "'stop' aborts on the first failure - the default, "
                    "because a run that quietly drops a third of its plates "
                    "produces a number that looks exactly like a good one. "
                    "'skip' records the unit, the stage and the reason, then "
                    "carries on; what was skipped is listed at the end and "
                    "stamped into the run ledger. 'retry' re-attempts the "
                    "unit on_error_attempts times with a doubling backoff "
                    "and then behaves like 'stop'. Default 'stop'."),
                "on_error_attempts": (
                    "(int) - Total attempts per unit when on_error='retry', "
                    "including the first. Bounded on purpose: an unbounded "
                    "retry against a dead network share is an infinite loop "
                    "with a progress bar. Default 3."),
                "on_error_backoff": (
                    "(float) - Seconds to wait before the second attempt "
                    "when on_error='retry', doubled before each attempt "
                    "after that and capped at 60s. Default 1.0."),
            })
    except ValueError as exc:
        # Another module already declared one of these. Say so once rather
        # than take the import of spacr.runctx down with it.
        LOG.debug("run-control settings not registered: %s", exc)


_register_settings()
