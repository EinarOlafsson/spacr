"""The macro recorder: every run also writes the script that would repeat it.

A GUI run used to leave settings behind and nothing else. The settings are
the *inputs*, not the method — reading ``settings.json`` tells you what the
knobs were, never which function consumed them, in what order, or how the
second module found what the first one made. So the record of an analysis
was a folder of numbers plus whatever the analyst remembered.

This module closes that. Every run that opens a journal
(:func:`spacr.run_journal.open_run` — the one seam the Qt GUI, the Tk GUI
and the CLI all launch through) also emits ``macro.py``: real imports, a
real settings dict, a real call. Run it and the same thing happens again.

It is three deliverables in one file, and the third is why the emitted
script is *also* a data structure:

**A reproducibility record.** The script is the method section. Its header
carries the spaCR version, the run id and the settings hash, and those are
not decoration: the run id is the id :mod:`spacr.runctx` stamped on every
log line the run emitted and :mod:`spacr.artifacts` stamped on every output
it registered, so the script, the log and the files all join on one column.
The settings hash is :func:`spacr.artifacts.settings_hash`, the same digest
the artifact rows carry.

**The on-ramp from clicking to the API.** A user who has outgrown the GUI
opens ``macro.py`` and finds the code they would have written — not a
wrapper, not a replay harness, the actual two lines.

**Most of the input to the methods-and-results exporter.** Which is why
the script also carries :data:`MACRO`, a plain dict literal holding every
step's module, entry point, run id, settings, which of those settings were
merely defaults, what it produced and how long it took.
:func:`read_macro` reads it back *without executing the script* — it is
parsed, not imported — so the exporter can consume a macro it did not
generate and does not trust.

Four things the emitted script is careful about
-----------------------------------------------

*Defaults are written out.* A settings dict that omits ``cell_diameter``
runs with whatever spaCR's default is **today**. Pin the version and that
is reproducible; anything less is not. So every key of the module's
defaults is emitted with its value, and :data:`MACRO` records which keys
the user actually set (``user_set``) and which were filled in
(``defaulted``) — the exporter needs that distinction and the script must
not lose it.

*A chain is one script.* Mask, then Measure, then Classify on the same
project is one file with three steps in dependency order, not three files.
The edge is confirmed against :func:`spacr.ports.next_modules`, so the
order in the script is the order the pipeline contract declares.

*Intermediate paths are threaded, not repeated.* The project each step ran
on becomes a named constant, and every path underneath it is rebuilt from
that constant with :func:`os.path.join`. Repointing the whole chain at
another plate is one edit on one line.

*Nothing here may fail a run.* Recording is bracketed by
:func:`begin_recording` / :func:`finish_recording`, both of which swallow
everything. A macro that could not be written costs one log line.

Usage
-----
.. code-block:: python

    from spacr.macro import current_macro, read_macro

    macro = current_macro()          # the chain recorded in this process
    print(macro.source())            # the script
    meta = read_macro(macro.path)    # the same thing, as data

Public API
----------
``begin_recording``, ``finish_recording``
    The hook :func:`spacr.run_journal.open_run` calls. Everything else is
    downstream of these two.
``Macro``, ``MacroStep``, ``current_macro``, ``macros``, ``reset``
    The recorded chains, in this process.
``render``, ``read_macro``, ``macro_path``, ``macros_dir``
    Rendering the script, and reading one back as data.
``entry_for``, ``module_defaults``, ``explicit_settings``
    The resolution steps, usable on their own — what function a module key
    runs, what its defaults are, and the fully-explicit settings dict.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
import textwrap
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

LOG = logging.getLogger("spacr.macro")

__all__ = [
    "MACRO_FILENAME",
    "MACRO_SCHEMA",
    "META_NAME",
    "Macro",
    "MacroError",
    "MacroStep",
    "Recording",
    "begin_recording",
    "current_macro",
    "entry_for",
    "explicit_settings",
    "finish_recording",
    "macro_path",
    "macros",
    "macros_dir",
    "module_defaults",
    "read_macro",
    "render",
    "reset",
    "summarise",
    "to_json",
]

#: Bumped when the shape of :data:`MACRO` changes incompatibly. A consumer
#: that finds a schema it does not know should say so rather than guess.
MACRO_SCHEMA = 1

#: The script written into every run journal folder.
MACRO_FILENAME = "macro.py"

#: The name of the machine-readable dict inside the emitted script.
META_NAME = "MACRO"

#: Environment override for :func:`macros_dir`, so a test — or a lab that
#: keeps its scripts on a share — can put them somewhere else.
MACRO_DIR_ENV = "SPACR_MACRO_DIR"

#: Most steps one chain will hold before the next run starts a fresh one.
#: A GUI process lives for as long as the window is open, and a script of
#: two hundred steps is not a method section — it is a log with a `.py`
#: extension. The bound is generous enough that no real pipeline reaches
#: it and small enough that nothing runs away.
MAX_CHAIN_STEPS = 25

#: Chains kept in memory. Older ones are already written to disk; dropping
#: the reference only means a much later run starts a new chain rather than
#: reopening an ancient one.
MAX_RETAINED_MACROS = 50

#: A run starting more than this long after the previous one finished is a
#: new piece of work, not the next step of the last one — even on the same
#: plate. Six hours: long enough to survive lunch and a slow segmentation,
#: short enough that yesterday's analysis does not get welded onto today's.
CHAIN_IDLE_SECONDS = 6 * 3600.0

#: Every entry point in :data:`spacr.validate.APP_FUNCTIONS` takes the
#: settings dict as its first positional parameter, which is also exactly
#: how ``spacr.qt.bridge.PipelineWorker`` calls it (``self._fn(settings)``).
#: The emitted call is therefore ``func(SETTINGS)`` for every module, and
#: ``tests/test_macro.py`` asserts that contract against the real
#: signatures rather than trusting this comment.
CALL_TEMPLATE = "{func}({settings})"

_LOCK = threading.RLock()
_MACROS: "List[Macro]" = []


def _utcnow() -> str:
    """Return the current UTC instant as ISO-8601, seconds resolution."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Where a macro lives
# ---------------------------------------------------------------------------

def macros_dir() -> str:
    """Return the folder holding one script per recorded chain.

    ``~/.spacr/macros``, honouring :data:`MACRO_DIR_ENV`. Created on first
    use. This is the *stable* copy: a chain's file is rewritten in place as
    each step joins it, so the path does not change while the chain grows.
    Every run also gets its own copy next to its manifest — see
    :func:`macro_path`.
    """
    override = os.environ.get(MACRO_DIR_ENV, "").strip()
    root = (os.path.abspath(os.path.expanduser(override)) if override
            else os.path.join(os.path.expanduser("~"), ".spacr", "macros"))
    os.makedirs(root, exist_ok=True)
    return root


def macro_path(run_dir: Any) -> str:
    """Return the macro script path inside a run journal folder.

    Deliberately beside ``manifest.json`` and ``settings.json``: the
    journal folder is already what ``spacr-repro`` and
    :func:`spacr.notebook_export.export_run` are pointed at, so the script
    is found by anything that can already find the run.
    """
    return os.path.join(str(run_dir), MACRO_FILENAME)


# ---------------------------------------------------------------------------
# What a module key actually runs
# ---------------------------------------------------------------------------

def entry_for(module: str) -> Tuple[str, str]:
    """Return ``(import_path, function_name)`` for a module key.

    Three sources, in the order :func:`spacr.qt.bridge.resolve_pipeline_entry`
    consults them, so the script calls what the Run button called:

    1. :data:`spacr.validate.APP_FUNCTIONS`, the shipped table;
    2. ``register_app(..., entry="mod:func")``, the registration seam;
    3. a plugin's ``entrypoint``.

    Resolved *textually*. Rendering a script must not import Cellpose,
    Torch and pandas to find out what a name is, and a recorder that
    imported the pipeline it was describing would turn a cheap write into
    a several-second stall at the end of every run.

    :param module: the module / app key.
    :returns: the pair, or ``("", "")`` when the key names an
        interactive-only app (Annotate, Make Masks) or nothing at all.
    """
    key = str(module or "").strip()
    if not key:
        return "", ""
    try:
        from .validate import APP_FUNCTIONS
        dotted = APP_FUNCTIONS.get(key, "")
    except Exception:                                   # noqa: BLE001
        dotted = ""
    if not dotted:
        dotted = _registered_entry_text(key) or _plugin_entry_text(key)
    return _split_entry(dotted)


def _split_entry(dotted: str) -> Tuple[str, str]:
    """Split ``"pkg.mod.func"`` or ``"pkg.mod:func"`` into its two halves."""
    text = str(dotted or "").strip()
    if not text:
        return "", ""
    if ":" in text:
        module_path, _, func = text.partition(":")
    else:
        module_path, _, func = text.rpartition(".")
    module_path = module_path.strip()
    func = func.strip()
    if not module_path or not func.isidentifier():
        return "", ""
    return module_path, func


def _registered_entry_text(key: str) -> str:
    """Return the ``entry=`` a module declared via ``register_app``, or ""."""
    try:
        from .qt.app import APP_META
    except Exception:                                   # noqa: BLE001
        # No Qt in this process. A headless recorder is a supported install,
        # and the shipped table above already answered for every built-in.
        return ""
    try:
        return str((APP_META.get(key) or {}).get("entry") or "")
    except Exception:                                   # noqa: BLE001
        return ""


def _plugin_entry_text(key: str) -> str:
    """Return a plugin app's ``entrypoint``, or ""."""
    try:
        from .plugins import get_app
        contribution = get_app(key)
    except Exception:                                   # noqa: BLE001
        return ""
    return str(getattr(contribution, "entrypoint", "") or "")


# ---------------------------------------------------------------------------
# Defaults, made explicit
# ---------------------------------------------------------------------------

def module_defaults(module: str) -> Tuple[Dict[str, Any], str]:
    """Return ``(defaults, source)`` for a module key.

    The sources, in order, are the same ones a settings panel consults —
    reused rather than re-derived, so the script cannot disagree with the
    screen it came from:

    ``"registered"``
        :func:`spacr.settings.defaults_for`, the ``register_defaults`` seam.
    ``"plugin"``
        a plugin app's ``defaults`` factory.
    ``"settings_model"``
        :func:`spacr.qt.screens.settings_model.resolve_default_settings`,
        the built-in dispatch. Guarded: it imports Qt, which a headless
        run does not have, and a missing Qt must cost the script its
        default-filling, never the script.
    ``"none"``
        nothing answered. The settings the caller passed are still emitted
        in full; only the unset keys are missing, and :data:`MACRO` says so
        through ``defaults_source``.

    :param module: the module / app key.
    :returns: a fresh dict, and the name of the source that produced it.
    """
    key = str(module or "").strip()
    if not key:
        return {}, "none"
    try:
        from .settings import defaults_for, has_registered_defaults
        if has_registered_defaults(key):
            return dict(defaults_for(key, {})), "registered"
    except Exception as exc:                            # noqa: BLE001
        LOG.debug("registered defaults for %s unavailable: %s", key, exc)
    plugin = _plugin_defaults(key)
    if plugin is not None:
        return plugin, "plugin"
    try:
        from .qt.screens.settings_model import resolve_default_settings
        return dict(resolve_default_settings(key)), "settings_model"
    except Exception as exc:                            # noqa: BLE001
        LOG.debug("built-in defaults for %s unavailable: %s", key, exc)
    return {}, "none"


def _plugin_defaults(key: str) -> Optional[Dict[str, Any]]:
    """Return a plugin app's defaults dict, or None when there is no plugin."""
    try:
        from .plugins import get_app, load_object
        contribution = get_app(key)
        if contribution is None:
            return None
        factory = load_object(contribution.defaults)
        result = factory({}) if _takes_an_argument(factory) else factory()
        return dict(result) if isinstance(result, dict) else None
    except Exception as exc:                            # noqa: BLE001
        LOG.debug("plugin defaults for %s unavailable: %s", key, exc)
        return None


def _takes_an_argument(fn: Any) -> bool:
    """Whether ``fn`` accepts a positional settings dict.

    The same test :func:`spacr.settings.defaults_for` makes, for the same
    reason: calling and retrying on TypeError cannot tell a wrong call from
    a TypeError raised *inside* a factory that was called correctly.
    """
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    return any(param.kind in (inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              inspect.Parameter.VAR_POSITIONAL)
               for param in params.values())


def explicit_settings(module: str,
                      settings: Optional[Mapping[str, Any]],
                      ) -> Tuple[Dict[str, Any], Tuple[str, ...], str]:
    """Return the settings a reproduction needs, with nothing left implicit.

    The module's defaults, then the run-control defaults
    (:func:`spacr.runctx.apply_defaults` — ``random_seed`` above all, since
    an unpinned seed is the one "default" that changes the numbers on its
    own), then the caller's values on top. Keys are ordered defaults-first
    so a diff between two macros lines up.

    :param module: the module / app key.
    :param settings: what the run was launched with.
    :returns: ``(settings, defaulted_keys, defaults_source)`` where
        ``defaulted_keys`` are the keys the caller did **not** set and the
        script is therefore pinning on their behalf.
    """
    given = dict(settings or {})
    defaults, source = module_defaults(module)
    try:
        from .runctx import apply_defaults
        apply_defaults(defaults)
    except Exception as exc:                            # noqa: BLE001
        LOG.debug("run-control defaults unavailable: %s", exc)
    resolved: Dict[str, Any] = dict(defaults)
    resolved.update(given)
    defaulted = tuple(key for key in defaults if key not in given)
    return resolved, defaulted, source


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

@dataclass
class MacroStep:
    """One recorded run: what ran, on what, under which id.

    :param module: the module / app key — ``"mask"``, ``"measure"``.
    :param entry_module: the import path of the entry point.
    :param entry_func: the function name.
    :param settings: the fully explicit settings dict.
    :param defaulted: keys filled in from the module defaults.
    :param user_set: keys the caller actually supplied.
    :param defaults_source: which source answered; see
        :func:`module_defaults`.
    :param run_id: the id the run stamped on its log lines and outputs.
    :param run_ids: every run id observed during the run, in order. More
        than one means the pipeline opened nested runs.
    :param run_id_source: ``"runctx"`` when the id came from the run's own
        log records, ``"journal"`` when it was taken from the journal
        folder because no run context opened.
    :param settings_hash: :func:`spacr.artifacts.settings_hash` over the
        explicit settings — the digest the artifact rows carry.
    :param project_root: the project the step ran on.
    :param run_dir: the journal folder.
    :param status: ``"success"``, ``"failed"``, ``"cancelled"``.
    :param started_utc: when the step opened.
    :param finished_utc: when it closed.
    :param elapsed_s: how long it took.
    :param outputs: declared output locations that exist on disk.
    :param link: how this step follows the previous one — ``"ports"`` when
        :func:`spacr.ports.next_modules` declares the edge, ``"project"``
        when they merely share a project, ``""`` for the first step.
    :param coerced: settings keys whose value is not a Python literal and
        was rendered as its string form.
    :param spacr_version: the version that ran it.
    """

    module: str
    entry_module: str = ""
    entry_func: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)
    defaulted: Tuple[str, ...] = ()
    user_set: Tuple[str, ...] = ()
    defaults_source: str = "none"
    run_id: str = ""
    run_ids: Tuple[str, ...] = ()
    run_id_source: str = ""
    settings_hash: str = ""
    project_root: str = ""
    run_dir: str = ""
    status: str = ""
    started_utc: str = ""
    finished_utc: str = ""
    elapsed_s: float = 0.0
    outputs: Tuple[str, ...] = ()
    link: str = ""
    coerced: Tuple[str, ...] = ()
    spacr_version: str = ""

    @property
    def entry(self) -> str:
        """``"spacr.core.preprocess_generate_masks"``, or ``""``."""
        if not self.entry_module or not self.entry_func:
            return ""
        return f"{self.entry_module}.{self.entry_func}"

    @property
    def runnable(self) -> bool:
        """Whether this step has an entry point the script can call."""
        return bool(self.entry_module and self.entry_func)

    def to_dict(self, index: int = 0, variable: str = "") -> Dict[str, Any]:
        """Return the step as the dict :data:`MACRO` carries.

        :param index: the 1-based position in the chain.
        :param variable: the name of the settings constant in the script,
            so a consumer can map a metadata entry back to the code.
        """
        return {
            "index": int(index),
            "module": self.module,
            "entry": self.entry,
            "variable": variable,
            "run_id": self.run_id,
            "run_ids": list(self.run_ids),
            "run_id_source": self.run_id_source,
            "settings_hash": self.settings_hash,
            "defaults_source": self.defaults_source,
            "defaulted": list(self.defaulted),
            "user_set": list(self.user_set),
            "coerced": list(self.coerced),
            "project_root": self.project_root,
            "run_dir": self.run_dir,
            "status": self.status,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "elapsed_s": round(float(self.elapsed_s), 3),
            "outputs": list(self.outputs),
            "link": self.link,
            "spacr_version": self.spacr_version,
        }


@dataclass
class Macro:
    """One chain of runs, and the script that repeats it.

    A chain grows while consecutive runs stay connected — the next module
    consumes what the previous one produced, on the same project. A run
    that connects to nothing starts a new :class:`Macro`, because a script
    that welds two unrelated plates together is not a reproduction of
    either.

    :param macro_id: twelve hex characters, the same shape as a run id.
    :param steps: the recorded steps, in the order they ran.
    :param created_utc: when the chain started.
    """

    macro_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    steps: List[MacroStep] = field(default_factory=list)
    created_utc: str = field(default_factory=_utcnow)
    #: Monotonic-ish wall clock of the last step, for :data:`CHAIN_IDLE_SECONDS`.
    touched: float = field(default_factory=time.time)

    @property
    def modules(self) -> Tuple[str, ...]:
        """The module keys, in order."""
        return tuple(step.module for step in self.steps)

    @property
    def path(self) -> str:
        """The stable script path for this chain, under :func:`macros_dir`."""
        return os.path.join(macros_dir(), f"{self.macro_id}.py")

    def source(self) -> str:
        """Render the chain as a runnable Python script."""
        return render(self)

    def metadata(self) -> Dict[str, Any]:
        """Return the machine-readable record the script carries."""
        names = _variable_names(self.steps)
        return {
            "schema": MACRO_SCHEMA,
            "macro_id": self.macro_id,
            "spacr_version": _version(),
            "generated_utc": _utcnow(),
            "created_utc": self.created_utc,
            "modules": list(self.modules),
            "steps": [step.to_dict(index + 1, names[index])
                      for index, step in enumerate(self.steps)],
        }

    def write(self, path: Any) -> str:
        """Write the script to ``path`` and return the path.

        Written to a neighbouring temporary file and renamed, so a reader
        that opens it while a later step is being appended never sees half
        a script.
        """
        target = str(path)
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        temporary = f"{target}.{os.getpid()}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(self.source())
        os.replace(temporary, target)
        return target

    def __len__(self) -> int:
        return len(self.steps)

    def __str__(self) -> str:
        return (f"macro {self.macro_id}: "
                f"{' -> '.join(self.modules) or '(empty)'}")


def macros() -> Tuple[Macro, ...]:
    """Every chain recorded in this process, oldest first."""
    with _LOCK:
        return tuple(_MACROS)


def current_macro() -> Optional[Macro]:
    """The chain the next run would join, or None before anything has run."""
    with _LOCK:
        return _MACROS[-1] if _MACROS else None


def reset() -> None:
    """Forget every recorded chain.

    The next run starts a fresh one. For a test, and for a caller that
    wants a chain boundary it chose rather than one inferred from the
    project layout.
    """
    with _LOCK:
        _MACROS.clear()


# ---------------------------------------------------------------------------
# Recording — the two calls run_journal.open_run makes
# ---------------------------------------------------------------------------

class _RunIdCapture(logging.Handler):
    """Read the run's id off its own log records.

    :func:`spacr.runctx.run_context` mints the id *inside* the pipeline
    function, well after the journal — and therefore this recorder —
    opened, and it hands that id to nobody outside the ``with`` block. But
    :func:`spacr.runctx.install_run_id_logging` stamps it onto every
    :class:`logging.LogRecord` created in the process, which is the same
    join :func:`spacr.runctx.read_run_log` reads back off disk. So the id
    is observed here rather than guessed: attach for the life of the run,
    keep the ids seen, detach.

    Ids are kept per thread and the recording thread's are preferred,
    because a second run on another thread is stamping its own id onto its
    own records at the same time and the two must not be confused.
    """

    def __init__(self, thread_id: int) -> None:
        super().__init__(level=logging.NOTSET)
        self.thread_id = int(thread_id)
        self.mine: List[str] = []
        self.other: List[str] = []

    def createLock(self) -> None:
        """No lock. ``handle`` does not use one — see :meth:`handle`."""
        self.lock = None

    def handle(self, record: logging.LogRecord) -> bool:
        """Note the record's run id. Never filters, never formats, never raises.

        Overridden rather than implemented as :meth:`emit` so no lock is
        acquired per record: this sits on the root logger for the whole
        run, and a per-record mutex on a pipeline that logs a line per
        field is a cost with no benefit — list ``append`` under the GIL is
        already atomic and the worst a race can do is record an id twice.
        """
        try:
            identifier = getattr(record, "run_id", "")
            if not identifier or identifier == "-":
                return True
            bucket = (self.mine if threading.get_ident() == self.thread_id
                      else self.other)
            if identifier not in bucket:
                bucket.append(str(identifier))
        except Exception:                               # noqa: BLE001
            pass
        return True

    def emit(self, record: logging.LogRecord) -> None:
        """Never called — :meth:`handle` does the work and does not delegate."""

    @property
    def observed(self) -> Tuple[str, ...]:
        """Ids seen, this thread's first."""
        return tuple(self.mine) + tuple(
            identifier for identifier in self.other if identifier not in self.mine)


@dataclass
class Recording:
    """One run being recorded. Created by :func:`begin_recording`."""

    module: str
    settings: Dict[str, Any]
    run_dir: str = ""
    started: float = 0.0
    started_utc: str = ""
    capture: Optional[_RunIdCapture] = None


def begin_recording(module: str,
                    settings: Optional[Mapping[str, Any]] = None,
                    *,
                    run_dir: Any = "") -> Optional[Recording]:
    """Start recording a run. Half of the hook; never raises.

    :param module: the module / app key the run was launched for.
    :param settings: the settings it was launched with. Copied, because
        several pipelines mutate the dict they are given and the script
        must show what was *asked for*, not what the run left behind.
    :param run_dir: the journal folder, when there is one.
    :returns: the :class:`Recording` to hand :func:`finish_recording`, or
        None when recording could not start — in which case finishing is a
        no-op and the run is entirely unaffected.
    """
    try:
        capture = _RunIdCapture(threading.get_ident())
        logging.getLogger().addHandler(capture)
        return Recording(module=str(module or ""),
                         settings=dict(settings or {}),
                         run_dir=str(run_dir or ""),
                         started=time.time(),
                         started_utc=_utcnow(),
                         capture=capture)
    except Exception:                                   # noqa: BLE001
        LOG.debug("macro recording could not start for %s", module,
                  exc_info=True)
        return None


def finish_recording(recording: Optional[Recording],
                     *,
                     status: str = "",
                     settings: Optional[Mapping[str, Any]] = None,
                     ) -> Optional[MacroStep]:
    """Finish a recording, append its step, write the script. Never raises.

    :param recording: what :func:`begin_recording` returned. ``None`` is a
        no-op, which is how a recorder that failed to start stays harmless.
    :param status: the run's outcome, as the journal recorded it.
    :param settings: override the settings to record.
    :returns: the appended :class:`MacroStep`, or None when nothing was
        recorded.
    """
    if recording is None:
        return None
    try:
        if recording.capture is not None:
            logging.getLogger().removeHandler(recording.capture)
            recording.capture.close()
    except Exception:                                   # noqa: BLE001
        pass
    try:
        step = _build_step(recording, status=status, settings=settings)
        macro = _append(step)
        _write_everywhere(macro, step)
        return step
    except Exception:                                   # noqa: BLE001
        # The script is a record of the run, not the run. Losing it is
        # worth a log line and nothing else.
        LOG.exception("could not record the macro for %s", recording.module)
        return None


def _build_step(recording: Recording,
                *,
                status: str = "",
                settings: Optional[Mapping[str, Any]] = None) -> MacroStep:
    """Turn a finished :class:`Recording` into a :class:`MacroStep`."""
    given = dict(settings if settings is not None else recording.settings)
    resolved, defaulted, source = explicit_settings(recording.module, given)
    entry_module, entry_func = entry_for(recording.module)
    observed = recording.capture.observed if recording.capture else ()
    run_id, id_source = _resolve_run_id(observed, recording.run_dir)
    return MacroStep(
        module=recording.module,
        entry_module=entry_module,
        entry_func=entry_func,
        settings=resolved,
        defaulted=defaulted,
        user_set=tuple(given),
        defaults_source=source,
        run_id=run_id,
        run_ids=tuple(observed),
        run_id_source=id_source,
        settings_hash=_settings_hash(resolved),
        project_root=_project_root(recording.module, resolved),
        run_dir=recording.run_dir,
        status=str(status or ""),
        started_utc=recording.started_utc,
        finished_utc=_utcnow(),
        elapsed_s=max(0.0, time.time() - recording.started),
        outputs=_outputs(recording.module, resolved),
        spacr_version=_version(),
    )


def _resolve_run_id(observed: Sequence[str], run_dir: str) -> Tuple[str, str]:
    """Return ``(run_id, source)`` for a finished run.

    The first id the run stamped on its own records wins — the outermost
    :func:`spacr.runctx.run_context`, which is the run the artifacts are
    registered under. A pipeline that opened no run context at all leaves
    nothing to observe, and the journal folder's own tag is used instead;
    ``run_id_source`` says which happened, so a consumer joining on
    ``run_id`` knows whether the join can succeed.
    """
    for identifier in observed:
        return str(identifier), "runctx"
    name = os.path.basename(str(run_dir).rstrip(os.sep))
    tag, _, _ = name.partition("__")
    _, _, short = tag.rpartition("_")
    return (short, "journal") if short else ("", "")


def _settings_hash(settings: Mapping[str, Any]) -> str:
    """Return the artifact registry's digest over these settings."""
    try:
        from .artifacts import settings_hash
        return str(settings_hash(settings))
    except Exception as exc:                            # noqa: BLE001
        LOG.debug("settings hash unavailable: %s", exc)
        return ""


def _project_root(module: str, settings: Mapping[str, Any]) -> str:
    """Return the project this step ran on, as :mod:`spacr.ports` resolves it."""
    try:
        from .ports import project_root
        return str(project_root(settings, module) or "")
    except Exception:                                   # noqa: BLE001
        value = settings.get("src") if isinstance(settings, Mapping) else None
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        return str(value or "")


def _outputs(module: str, settings: Mapping[str, Any]) -> Tuple[str, ...]:
    """Return the locations ``module`` declares it writes, that exist.

    The declarations come from :mod:`spacr.ports`, so this module owns no
    second, drifting table of where each pipeline puts things. What it
    does *not* do is call :func:`spacr.ports.declared_outputs`: that
    resolves every port's glob, and a mask run's ``merged/*.npy`` port
    matches ten thousand files. :func:`spacr.artifacts.register_run_outputs`
    already pays that at the end of every run, and paying it twice — on a
    network share, for a record that only wants the folder name — is a
    cost with no answer to show for it. The one line borrowed from
    :func:`spacr.ports.resolve_port` is its ``target``, which is exactly
    :attr:`spacr.ports.ResolvedPort.location`.
    """
    try:
        from .ports import module_ports, project_root
        root = project_root(settings, module)
        if not root:
            return ()
        found: List[str] = []
        for port in module_ports(module).produces:
            target = os.path.join(root, port.path) if port.path else root
            if target not in found and os.path.exists(target):
                found.append(target)
        return tuple(found)
    except Exception as exc:                            # noqa: BLE001
        LOG.debug("declared outputs for %s unavailable: %s", module, exc)
        return ()


def _version() -> str:
    """Return the running spaCR version."""
    try:
        from .version import get_version
        return str(get_version())
    except Exception:                                   # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# Chaining — when two runs belong in one script
# ---------------------------------------------------------------------------

def _link(previous: MacroStep, step: MacroStep) -> str:
    """Return how ``step`` follows ``previous``, or ``""`` when it does not.

    ``"ports"``
        :func:`spacr.ports.next_modules` declares the edge and the two
        share a project. This is mask → measure → classify.
    ``"project"``
        the same project, but no declared edge — two runs of the same
        module, or a module whose ports are not declared. Still one chain:
        they ran one after the other on the same data, which is exactly
        what a method section describes.
    ``""``
        different projects. A new chain starts.
    """
    if not previous.project_root or not step.project_root:
        return ""
    before = os.path.normpath(previous.project_root)
    after = os.path.normpath(step.project_root)
    if after != before and not after.startswith(before + os.sep):
        return ""
    try:
        from .ports import next_modules
        if step.module in next_modules(previous.module):
            return "ports"
    except Exception as exc:                            # noqa: BLE001
        LOG.debug("next_modules(%s) unavailable: %s", previous.module, exc)
    return "project"


def _continues(macro: Macro, step: MacroStep) -> str:
    """Return how ``step`` continues ``macro``, or ``""`` when it starts anew.

    Three ways a chain ends, and all three are the same judgement: does
    this script still describe one piece of work?

    * the last step and this one are not connected (:func:`_link`);
    * the chain is already :data:`MAX_CHAIN_STEPS` long;
    * more than :data:`CHAIN_IDLE_SECONDS` passed since the last step. A
      GUI process outlives the analysis running in it, and welding
      yesterday's plate onto today's would produce a script that
      reproduces neither.
    """
    if not macro.steps:
        return ""
    if len(macro.steps) >= MAX_CHAIN_STEPS:
        return ""
    if time.time() - macro.touched > CHAIN_IDLE_SECONDS:
        return ""
    return _link(macro.steps[-1], step)


def _append(step: MacroStep) -> Macro:
    """Add ``step`` to the chain it continues, or start a new one."""
    with _LOCK:
        macro = _MACROS[-1] if _MACROS else None
        if macro is not None:
            link = _continues(macro, step)
            if link:
                step.link = link
                macro.steps.append(step)
                macro.touched = time.time()
                return macro
        macro = Macro()
        step.link = ""
        macro.steps.append(step)
        _MACROS.append(macro)
        del _MACROS[:-MAX_RETAINED_MACROS]
        return macro


def _write_everywhere(macro: Macro, step: MacroStep) -> None:
    """Write the chain's script to its stable path and this run's folder.

    Two copies of one thing on purpose. The run folder's copy sits beside
    the manifest, so anything pointed at a run finds the script that
    reproduces it; the copy under :func:`macros_dir` has a path that does
    not change while the chain grows, so a chain of three is one file
    rather than three folders to reassemble.
    """
    written: List[str] = []
    try:
        written.append(macro.write(macro.path))
    except Exception:                                   # noqa: BLE001
        LOG.exception("could not write the macro for chain %s", macro.macro_id)
    if step.run_dir and os.path.isdir(step.run_dir):
        try:
            written.append(macro.write(macro_path(step.run_dir)))
        except Exception:                               # noqa: BLE001
            LOG.exception("could not write the macro into %s", step.run_dir)
    if written:
        LOG.info("macro %s (%s) → %s", macro.macro_id,
                 " -> ".join(macro.modules), written[-1])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _identifier(text: str) -> str:
    """Return ``text`` as a legal upper-case Python identifier fragment."""
    cleaned = re.sub(r"\W+", "_", str(text or "")).strip("_").upper()
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"STEP_{cleaned}" if cleaned else "STEP"
    return cleaned


def _variable_names(steps: Sequence[MacroStep]) -> List[str]:
    """Return one settings-constant name per step, all distinct."""
    names: List[str] = []
    used: Dict[str, int] = {}
    for index, step in enumerate(steps, start=1):
        base = f"{_identifier(step.module)}_SETTINGS"
        count = used.get(base, 0) + 1
        used[base] = count
        names.append(base if count == 1 else f"{base}_{count}")
    return names


def _project_names(steps: Sequence[MacroStep]) -> "List[Tuple[str, str]]":
    """Return ``(constant_name, path)`` for each distinct project, in order."""
    seen: Dict[str, str] = {}
    ordered: List[Tuple[str, str]] = []
    for step in steps:
        if not step.project_root:
            continue
        key = os.path.normpath(step.project_root)
        if key in seen:
            continue
        name = f"PROJECT_{len(ordered) + 1}"
        seen[key] = name
        ordered.append((name, step.project_root))
    return ordered


class _Threader:
    """Rewrites recorded paths as expressions over the project constants.

    The point of the whole exercise: ``MEASURE_SETTINGS['src']`` is not the
    literal ``/data/plate7`` again, it is ``PROJECT_1`` — the same name
    Mask's ``src`` used. Repointing the chain at another plate is then one
    edit, and the script says out loud that step two reads what step one
    wrote rather than leaving a reader to compare two long strings.
    """

    def __init__(self, projects: Sequence[Tuple[str, str]]) -> None:
        # Longest first, so a nested project does not lose to its parent.
        self.roots = sorted(
            ((os.path.normpath(path), name) for name, path in projects),
            key=lambda pair: len(pair[0]), reverse=True)
        self.used = False

    def express(self, value: str) -> str:
        """Return the expression for a path, or ``""`` when it is not one."""
        if not value or not isinstance(value, str):
            return ""
        candidate = os.path.normpath(value)
        for root, name in self.roots:
            if candidate == root:
                return name
            if candidate.startswith(root + os.sep):
                relative = os.path.relpath(candidate, root)
                parts = [part for part in relative.split(os.sep) if part]
                self.used = True
                joined = ", ".join(repr(part) for part in parts)
                return f"os.path.join({name}, {joined})"
        return ""


_LITERAL_TYPES = (str, bytes, bool, int, float, type(None))


def _render_value(value: Any, threader: _Threader, indent: str,
                  coerced: Optional[List[str]] = None, key: str = "") -> str:
    """Render one settings value as Python source.

    Literals are rendered with :func:`repr`, which round-trips through
    :func:`ast.literal_eval`; a path recognised by ``threader`` becomes an
    expression over a project constant instead. Anything that is not a
    literal — a numpy array or a callable that found its way into a
    settings dict — is rendered as its string form and named in
    ``coerced``, so :data:`MACRO` admits the lossy conversion rather than
    letting a reader assume the script is exact.
    """
    if isinstance(value, str):
        expression = threader.express(value)
        return expression or repr(value)
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, (int, float, bytes)):
        return repr(value)
    if isinstance(value, Mapping):
        if not value:
            return "{}"
        inner = indent + "    "
        items = [
            f"{inner}{_render_value(name, threader, inner, coerced, key)}: "
            f"{_render_value(item, threader, inner, coerced, key)},"
            for name, item in value.items()
        ]
        return "{\n" + "\n".join(items) + f"\n{indent}}}"
    if isinstance(value, (list, tuple, set, frozenset)):
        rendered = [_render_value(item, threader, indent, coerced, key)
                    for item in value]
        if isinstance(value, tuple):
            body = ", ".join(rendered)
            return f"({body},)" if len(rendered) == 1 else f"({body})"
        if isinstance(value, (set, frozenset)):
            return "set()" if not rendered else "{" + ", ".join(rendered) + "}"
        return "[" + ", ".join(rendered) + "]"
    if coerced is not None and key and key not in coerced:
        coerced.append(key)
    return repr(str(value))


def _render_settings(step: MacroStep, name: str, threader: _Threader) -> str:
    """Render one step's settings dict as an assignment."""
    coerced: List[str] = []
    lines = [f"{name} = {{"]
    for key, value in step.settings.items():
        rendered = _render_value(value, threader, "    ", coerced, str(key))
        note = "" if key in step.user_set else "  # spaCR default"
        lines.append(f"    {key!r}: {rendered},{note}")
    lines.append("}")
    step.coerced = tuple(coerced)
    return "\n".join(lines)


def _wrap(text: str, width: int = 74) -> List[str]:
    """Wrap a sentence to comment width."""
    return textwrap.wrap(str(text), width=width) or [""]


def _header(macro: Macro, steps: Sequence[MacroStep],
            names: Sequence[str]) -> List[str]:
    """Return the header comment: version, run ids, settings hashes."""
    rule = "# " + "-" * 74
    lines = [
        "#!/usr/bin/env python3",
        "# -*- coding: utf-8 -*-",
        rule,
        "# spaCR macro — the script this run is the record of.",
        "#",
        f"#   spacr version  : {_version()}",
        f"#   macro id       : {macro.macro_id}",
        f"#   recorded (UTC) : {_utcnow()}",
        f"#   steps          : {' -> '.join(macro.modules) or '(none)'}",
        "#",
    ]
    for index, (step, name) in enumerate(zip(steps, names), start=1):
        lines.append(f"#   step {index}  {step.module}")
        lines.append(f"#       run id       : {step.run_id or '(none)'}"
                     + (f"  [{step.run_id_source}]" if step.run_id_source
                        else ""))
        lines.append(f"#       settings hash: {step.settings_hash or '(none)'}")
        lines.append(f"#       entry        : {step.entry or '(interactive)'}")
        lines.append(f"#       settings     : {name}")
        if step.status:
            lines.append(f"#       status       : {step.status}")
    lines.append("#")
    for sentence in (
            "The run ids above are the ids those runs stamped onto every log "
            "line (spacr.runctx.read_run_log) and onto every output they "
            "registered (spacr.artifacts), so this script, that log and those "
            "files all join on one column.",
            "Every setting is written out explicitly, including the ones that "
            "were spaCR defaults on the day this ran — they are marked. A "
            "script that leans on a future default is not a reproduction.",
            "The machine-readable copy of all of this is the MACRO dict "
            "below; spacr.macro.read_macro() parses it back out without "
            "executing anything."):
        for line in _wrap(sentence):
            lines.append(f"# {line}")
        lines.append("#")
    lines.append(rule)
    return lines


def _docstring(macro: Macro) -> List[str]:
    """Return the module docstring — what this script does, and how to run it."""
    chain = " → ".join(macro.modules) or "nothing"
    return [
        '"""spaCR macro: ' + chain + ".",
        "",
        "Generated by spaCR. Running this file repeats the run it was",
        "generated from::",
        "",
        "    python " + MACRO_FILENAME,
        "",
        "Edit the settings below to run the same analysis somewhere else;",
        "the project constants are what the paths are built from.",
        '"""',
    ]


def _imports(steps: Sequence[MacroStep], threader: _Threader) -> List[str]:
    """Return the import block — one line per distinct entry point."""
    lines: List[str] = []
    if threader.used:
        lines.append("import os")
        lines.append("")
    grouped: Dict[str, List[str]] = {}
    for step in steps:
        if not step.runnable:
            continue
        names = grouped.setdefault(step.entry_module, [])
        if step.entry_func not in names:
            names.append(step.entry_func)
    for module_path in sorted(grouped):
        lines.append(f"from {module_path} import "
                     + ", ".join(sorted(grouped[module_path])))
    return lines


def _main_block(steps: Sequence[MacroStep], names: Sequence[str]) -> List[str]:
    """Return ``main()`` — every step called in order, and nothing else."""
    lines = ["def main():",
             '    """Run every step, in the order the recorded run ran them."""']
    body = False
    for index, (step, name) in enumerate(zip(steps, names), start=1):
        lines.append(f"    # step {index} — {step.module}")
        if step.runnable:
            lines.append("    " + CALL_TEMPLATE.format(func=step.entry_func,
                                                       settings=name))
            body = True
        else:
            lines.append(f"    #   {step.module} has no API entry point: it is "
                         "an interactive")
            lines.append("    #   module, so its settings are recorded but "
                         "cannot be replayed.")
    if not body:
        lines.append("    return None")
    lines.extend(["", "", 'if __name__ == "__main__":', "    main()"])
    return lines


def render(macro: Macro) -> str:
    """Render a :class:`Macro` as a standalone, runnable Python script.

    :param macro: the chain to render.
    :returns: the source. Always parses: ``tests/test_macro.py`` compiles
        every script this function produces, because a recorder that emits
        plausible-looking code that does not run is worse than no recorder.
    """
    steps = list(macro.steps)
    names = _variable_names(steps)
    projects = _project_names(steps)
    threader = _Threader(projects)

    # Everything that *uses* the project constants is rendered before the
    # import block is built, though both are emitted later: rendering is
    # what decides whether os.path.join appears, and therefore whether the
    # script needs `import os`. Get that order wrong and a macro whose only
    # joined path is in the MACRO record raises NameError on line one — the
    # exact failure mode this recorder exists to avoid.
    blocks = [_render_settings(step, name, threader)
              for step, name in zip(steps, names)]
    metadata = _render_metadata(macro, steps, names, threader)

    lines: List[str] = []
    lines.extend(_header(macro, steps, names))
    lines.append("")
    lines.extend(_docstring(macro))
    lines.append("")
    import_lines = _imports(steps, threader)
    if import_lines:
        lines.extend(import_lines)
        lines.append("")
    if projects:
        lines.append("# The project(s) this chain ran on. Every path below is "
                     "built from these,")
        lines.append("# so pointing the whole chain at another plate is one "
                     "edit per project.")
        for name, path in projects:
            lines.append(f"{name} = {path!r}")
        lines.append("")
    for index, (step, name, block) in enumerate(
            zip(steps, names, blocks), start=1):
        lines.append("")
        lines.append("# " + "-" * 74)
        lines.append(f"# Step {index} — {step.module}"
                     + (f"  ({step.entry})" if step.entry else ""))
        if step.link == "ports":
            previous = steps[index - 2].module
            lines.append(f"# Reads what step {index - 1} ({previous}) produced "
                         "— spacr.ports declares the edge.")
        elif step.link == "project":
            lines.append(f"# Ran on the same project as step {index - 1}.")
        lines.append("# " + "-" * 74)
        lines.append(block)
        lines.append("")
    lines.append("")
    lines.append("#: Everything above, as data. Read it with "
                 "spacr.macro.read_macro(path),")
    lines.append("#: which parses this file rather than importing it.")
    lines.append(metadata)
    lines.append("")
    lines.append("")
    lines.extend(_main_block(steps, names))
    lines.append("")
    return "\n".join(lines)


def _render_metadata(macro: Macro, steps: Sequence[MacroStep],
                     names: Sequence[str], threader: _Threader) -> str:
    """Render the :data:`MACRO` dict, referring to the settings constants.

    The settings are not repeated here: each step's ``"settings"`` is the
    *name* of the constant above, so the file has one copy of every value
    and a reader cannot be shown two that disagree. :func:`read_macro`
    resolves the names.
    """
    payload = macro.metadata()
    lines = [f"{META_NAME} = {{"]
    for key, value in payload.items():
        if key != "steps":
            lines.append(f"    {key!r}: "
                         f"{_render_value(value, threader, '    ')},")
    lines.append("    'steps': [")
    for step, name, entry in zip(steps, names, payload["steps"]):
        lines.append("        {")
        for key, value in entry.items():
            lines.append(f"            {key!r}: "
                         f"{_render_value(value, threader, '            ')},")
        lines.append(f"            'settings': {name},")
        lines.append("        },")
    lines.append("    ],")
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reading one back — the seam the methods exporter uses
# ---------------------------------------------------------------------------

class MacroError(ValueError):
    """A file is not a spaCR macro, or carries a schema this build cannot read."""


def read_macro(path: Any) -> Dict[str, Any]:
    """Return the :data:`MACRO` record of an emitted script.

    Parsed, never executed. The exporter's input is a file spaCR may not
    have written — someone else's macro, an edited one, one from a newer
    build — and importing it would run whatever is in it. So this walks the
    AST, evaluates the top-level literal assignments (which is how the
    ``PROJECT_1`` and ``*_SETTINGS`` names the metadata refers to are
    resolved) and returns the record.

    :param path: the script.
    :returns: the record, with each step's ``"settings"`` resolved to the
        real dict.
    :raises MacroError: when the file has no :data:`MACRO`, or its schema
        is newer than this build understands.
    :raises OSError: when the file cannot be read.
    :raises SyntaxError: when it is not Python at all.
    """
    with open(str(path), "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=str(path))
    namespace: Dict[str, Any] = {}
    record: Optional[Dict[str, Any]] = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                value = _evaluate(node.value, namespace)
            except MacroError:
                continue
            namespace[target.id] = value
            if target.id == META_NAME and isinstance(value, dict):
                record = value
    if record is None:
        raise MacroError(f"{path} carries no {META_NAME} record; it is not a "
                         "spaCR macro")
    schema = record.get("schema")
    if isinstance(schema, int) and schema > MACRO_SCHEMA:
        raise MacroError(
            f"{path} was written with macro schema {schema}; this spaCR "
            f"reads up to {MACRO_SCHEMA}. Upgrade spaCR to read it.")
    return record


def _evaluate(node: ast.AST, namespace: Mapping[str, Any]) -> Any:
    """Evaluate one expression node against already-bound names.

    Deliberately tiny: literals, the containers built from them, names
    already bound in this file, and ``os.path.join`` — which is the one
    call :func:`render` emits. Everything else raises, because the point
    of not importing the file is not to run anything in it.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in namespace:
            return namespace[node.id]
        raise MacroError(f"macro refers to an unbound name {node.id!r}")
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate(item, namespace) for item in node.elts)
    if isinstance(node, ast.List):
        return [_evaluate(item, namespace) for item in node.elts]
    if isinstance(node, ast.Set):
        return {_evaluate(item, namespace) for item in node.elts}
    if isinstance(node, ast.Dict):
        return {_evaluate(key, namespace): _evaluate(value, namespace)
                for key, value in zip(node.keys, node.values)
                if key is not None}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd,
                                                              ast.USub)):
        operand = _evaluate(node.operand, namespace)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.Call) and _is_path_join(node.func):
        parts = [str(_evaluate(argument, namespace)) for argument in node.args]
        return os.path.join(*parts) if parts else ""
    raise MacroError("macro contains an expression this reader will not "
                     f"evaluate: {ast.dump(node)[:80]}")


def _is_path_join(node: ast.AST) -> bool:
    """True for the ``os.path.join`` attribute chain :func:`render` emits."""
    return (isinstance(node, ast.Attribute) and node.attr == "join"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "path"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os")


def summarise(record: Mapping[str, Any]) -> str:
    """Return a one-block human summary of a record from :func:`read_macro`.

    What a methods section starts from: the version, the chain, and per
    step the entry point, the run id and how many settings were the user's
    rather than defaults.
    """
    steps = list(record.get("steps") or [])
    lines = [f"spaCR {record.get('spacr_version', '?')} — "
             f"{' -> '.join(str(s.get('module', '?')) for s in steps) or '(none)'}",
             f"macro {record.get('macro_id', '?')}, "
             f"recorded {record.get('generated_utc', '?')}"]
    for step in steps:
        settings = step.get("settings") or {}
        chosen = len(step.get("user_set") or [])
        lines.append(
            f"  {step.get('index', '?')}. {step.get('module', '?')} — "
            f"{step.get('entry') or '(interactive)'}")
        lines.append(
            f"     run {step.get('run_id') or '(none)'}, "
            f"settings {step.get('settings_hash', '')[:12] or '(none)'}, "
            f"{chosen} of {len(settings)} settings chosen, "
            f"status {step.get('status') or '?'}")
        outputs = step.get("outputs") or []
        if outputs:
            lines.append(f"     produced {len(outputs)}: "
                         + ", ".join(str(path) for path in outputs[:3])
                         + (" …" if len(outputs) > 3 else ""))
    return "\n".join(lines)


def to_json(record: Mapping[str, Any], **kwargs: Any) -> str:
    """Return a record from :func:`read_macro` as JSON.

    For a consumer that would rather have JSON than a Python dict — the
    methods exporter prompt, a web view, a diff. ``default=str`` so a value
    that survived as a coerced string does not take the dump down.
    """
    kwargs.setdefault("indent", 2)
    kwargs.setdefault("sort_keys", False)
    return json.dumps(dict(record), default=str, **kwargs)
